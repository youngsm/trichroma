"""CUDA-backend side of the bitwise legacy mode (imports PyCUDA).

Installed by :mod:`chroma.sim_cuda` when ``CHROMA_TRITON_TAPE`` is set:

* ``record:<dir>`` -- record a native RNG tape (see
  :mod:`chroma.triton.legacy.tape`) of every batch simulated through
  ``chroma.sim.Simulation``.
* ``canonical`` -- sort each launch's surviving-photon queue before the next
  launch. The original appends survivors with warp-aggregated atomics in
  arbitrary warp order; ascending order is one legal execution of the
  unmodified kernels and makes multi-launch runs repeatable.
  ``CHROMA_TRITON_TAPE_SORT=1`` applies the same sort while recording.

How the draws are obtained without touching the kernels
-------------------------------------------------------
Every random number the propagate and DAQ kernels consume is one
``curand_uniform`` call on the slot's ``curandStateXORWOW``; each call adds
362437 to the Weyl word ``d``. Around every kernel chunk launch the recorder
copies the slot states before and after the (unmodified) kernel. The number
of draws of slot ``i`` is ``(d_after - d_before) * 362437^-1 mod 2^32`` and
the draws are regenerated from the entry state with the same inline
``curand_uniform`` compiled with the same ``cuda_options``. Regenerating must
reproduce the complete exit state (``d`` and ``v[0..4]``) of every slot, or
recording fails. The kernels, their arguments and the photon/RNG buffers are
never written by the recorder, so a recorded run executes exactly the same
instructions on the same data as an unrecorded run with the same queue order.
"""

import hashlib
import os
import platform
import sys
import time

import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray as ga
from pycuda import characterize
import pycuda.compiler

from chroma import gpu
from chroma.gpu.tools import cuda_options
from chroma.triton.legacy import tape as tapefmt

WEYL = 362437
WEYL_INVERSE = pow(WEYL, -1, 2 ** 32)
STATE_BYTES = 48  # sizeof(curandStateXORWOW) on 64-bit Linux
MAX_DRAWS_PER_SLOT = 1 << 26

_KERNEL_SOURCE = r"""
#include <curand_kernel.h>
extern "C" {
__global__ void tape_counts(int n, const curandStateXORWOW *before,
                            const curandStateXORWOW *after,
                            unsigned int inverse, unsigned int *counts)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= n) return;
    counts[id] = (after[id].d - before[id].d) * inverse;
}

__global__ void peek_words(int n, const unsigned long long *pointers,
                           const unsigned int *index, unsigned int *out)
{
    // Read the word the propagate kernel reads at fp[wavelength_n] (one past
    // a table) when a wavelength lands on the grid's top point.
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= n) return;
    const unsigned int *table = (const unsigned int *) pointers[id];
    out[id] = table[index[id]];
}

__global__ void tape_draws(int n, const curandStateXORWOW *before,
                           const curandStateXORWOW *after,
                           const unsigned int *counts,
                           const unsigned long long *offsets,
                           unsigned int *draws, unsigned int *mismatch,
                           unsigned int max_draws)
{
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    if (id >= n) return;
    curandStateXORWOW s = before[id];
    unsigned int c = counts[id];
    if (c > max_draws) { atomicAdd(mismatch, 1u); return; }
    unsigned long long o = offsets[id];
    for (unsigned int i = 0; i < c; i++)
        draws[o + i] = __float_as_uint(curand_uniform(&s));
    const curandStateXORWOW a = after[id];
    if (s.d != a.d || s.v[0] != a.v[0] || s.v[1] != a.v[1] || s.v[2] != a.v[2]
        || s.v[3] != a.v[3] || s.v[4] != a.v[4])
        atomicAdd(mismatch, 1u);
}
}
"""

_module = None


def _kernels():
    global _module
    if _module is None:
        _module = pycuda.compiler.SourceModule(_KERNEL_SOURCE, no_extern_c=True, options=list(cuda_options))
    return _module


def _peek(pointers, index):
    """Words at ``pointers[i][index[i]]`` read by a kernel (past-the-end reads)."""
    pointers = np.asarray(pointers, np.uint64)
    if len(pointers) == 0:
        return np.zeros(0, np.uint32)
    out = ga.empty(len(pointers), np.uint32)
    _kernels().get_function("peek_words")(np.int32(len(pointers)), ga.to_gpu(pointers),
                                          ga.to_gpu(np.asarray(index, np.uint32)), out,
                                          block=(64, 1, 1), grid=((len(pointers) + 63) // 64, 1))
    return out.get()


def _read(pointer, count, dtype):
    out = np.empty(count, dtype)
    if out.nbytes:
        cuda.memcpy_dtoh(out, int(pointer))
    return out


def _source_hashes():
    import chroma

    root = os.path.dirname(os.path.abspath(chroma.__file__))
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "_build_ext")]
        for name in sorted(filenames):
            if name.endswith((".py", ".h", ".cu")):
                path = os.path.join(dirpath, name)
                with open(path, "rb") as f:
                    out[os.path.relpath(path, os.path.dirname(root))] = hashlib.sha256(f.read()).hexdigest()
    return root, out


def _environment():
    import pycuda

    try:
        from pycuda.compiler import get_nvcc_version
        nvcc = get_nvcc_version("nvcc").strip().splitlines()[-1]
    except Exception as error:  # pragma: no cover - informational only
        nvcc = "unknown (%s)" % error
    device = cuda.Context.get_device()
    return dict(python=platform.python_version(), numpy=np.__version__, pycuda=pycuda.VERSION_TEXT,
                nvcc=nvcc, device=device.name(), compute_capability=list(device.compute_capability()),
                driver=cuda.get_driver_version(), cuda_options=list(cuda_options),
                environ={k: os.environ[k] for k in ("CHROMA_FORCE_SCATTER_AT_PASS", "CHROMA_DEVICE_PROFILE")
                         if k in os.environ})


# ----------------------------------------------------------------- scene words


def export_device_scene(gpu_geometry):
    """Read back the words the kernel dereferences (same keys as scene.scene_words)."""
    from chroma.geometry import standard_wavelengths

    geometry = gpu_geometry.geometry
    raw = _read(gpu_geometry.gpudata, 96, np.uint8)
    ptr = raw[:72].view(np.uint64)
    world_origin = raw[72:84].view(np.float32).copy()
    world_scale = raw[84:88].view(np.float32).copy()
    nprimary = int(raw[88:92].view(np.int32)[0])
    nwire = int(raw[92:96].view(np.int32)[0])
    nv = len(geometry.mesh.vertices)
    nt = len(geometry.mesh.triangles)
    nn = len(geometry.bvh.nodes)
    out = {}
    out["wavelength_grid"] = np.asarray(standard_wavelengths, np.float32)
    out["vertices"] = _read(ptr[0], nv * 3, np.float32).reshape(nv, 3)
    out["triangles"] = _read(ptr[1], nt * 3, np.uint32).reshape(nt, 3)
    out["material_codes"] = _read(ptr[2], nt, np.uint32)
    out["colors"] = _read(ptr[3], nt, np.uint32)
    primary = _read(ptr[4], min(nprimary, nn) * 4, np.uint32).reshape(-1, 4)
    extra = _read(ptr[5], max(0, nn - nprimary) * 4, np.uint32).reshape(-1, 4)
    out["nodes"] = np.concatenate([primary, extra])
    out["world_origin"] = world_origin
    out["world_scale"] = world_scale
    out["solid_id_map"] = gpu_geometry.solid_id_map.get().astype(np.uint32)
    nmat = len(gpu_geometry.material_ptrs)
    mptrs = _read(ptr[6], nmat, np.uint64)
    headers, tables, comp_offsets = [], {k: [] for k in range(3)}, [0]
    comps = {k: [] for k in range(3, 7)}
    pad_ptrs = {k: [] for k in range(7)}
    for p in mptrs:
        words = _read(p, 7 * 2 + 7, np.uint32)
        pointers = words[:14].view(np.uint64)
        header = words[14:21]
        headers.append(header)
        ncomp, nw, ntimes = int(header[0]), int(header[1]), int(header[4])
        for k in range(3):
            tables[k].append(_read(pointers[k], nw, np.float32))
            pad_ptrs[k].append((int(pointers[k]), nw))
        for k in range(3, 7):
            if ncomp:
                cps = _read(pointers[k], ncomp, np.uint64)
                width = ntimes if k == 5 else nw
                comps[k].extend(_read(c, width, np.float32) for c in cps)
                pad_ptrs[k].extend((int(c), width) for c in cps)
        comp_offsets.append(comp_offsets[-1] + ncomp)
    nw = int(headers[0][1]) if headers else len(standard_wavelengths)
    ntimes = int(headers[0][4]) if headers else 0
    for k, name in enumerate(("refractive_index", "absorption_length", "scattering_length")):
        out["material_" + name] = np.asarray(tables[k], np.float32).reshape(nmat, nw)
        out["pad_material_" + name] = _peek([q for q, _ in pad_ptrs[k]], [w for _, w in pad_ptrs[k]])
    for k, name in zip(range(3, 7), ("comp_reemission_prob", "comp_reemission_wvl_cdf",
                                    "comp_reemission_time_cdf", "comp_absorption_length")):
        out["pad_" + name] = _peek([q for q, _ in pad_ptrs[k]], [w for _, w in pad_ptrs[k]])
    out["material_header"] = np.asarray(headers, np.uint32).reshape(nmat, 7)
    out["material_comp_offsets"] = np.asarray(comp_offsets, np.int64)
    for k, name in zip(range(3, 7), ("comp_reemission_prob", "comp_reemission_wvl_cdf",
                                    "comp_reemission_time_cdf", "comp_absorption_length")):
        width = ntimes if k == 5 else nw
        out[name] = np.asarray(comps[k], np.float32).reshape(len(comps[k]), width)
    nsurf = len(gpu_geometry.surface_ptrs)
    sptrs = _read(ptr[7], nsurf, np.uint64)
    fields = ("detect", "absorb", "reemit", "reflect_diffuse", "reflect_specular", "eta", "k", "reemission_cdf")
    st = {f: [] for f in fields}
    spad = {f: [] for f in fields}
    sheader, present = [], []
    doff, dang, dref, dtr = [0], [], [], []
    dpad_r, dpad_t = [], []
    aoff = [0]
    ang = {k: [] for k in ("angles", "transmit", "reflect_specular", "reflect_diffuse")}
    for p in sptrs:
        if int(p) == 0:
            present.append(0)
            for f in fields:
                st[f].append(np.zeros(nw, np.float32))
            sheader.append(np.zeros(6, np.uint32))
            doff.append(doff[-1])
            aoff.append(aoff[-1])
            continue
        present.append(1)
        words = _read(p, 20 + 6, np.uint32)
        pointers = words[:20].view(np.uint64)
        header = words[20:26]
        sheader.append(header)
        snw = int(header[1])
        for i, f in enumerate(fields):
            st[f].append(_read(pointers[i], snw, np.float32))
            spad[f].append((int(pointers[i]), snw))
        if int(pointers[8]):
            dwords = _read(pointers[8], 7, np.uint32)
            dp = dwords[:6].view(np.uint64)
            nang = int(dwords[6])
            dang.append(_read(dp[0], nang, np.float32))
            rps = _read(dp[1], nang, np.uint64)
            tps = _read(dp[2], nang, np.uint64)
            for i in range(nang):
                dref.append(_read(rps[i], snw, np.float32))
                dtr.append(_read(tps[i], snw, np.float32))
                dpad_r.append((int(rps[i]), snw))
                dpad_t.append((int(tps[i]), snw))
            doff.append(doff[-1] + nang)
        else:
            doff.append(doff[-1])
        if int(pointers[9]):
            awords = _read(pointers[9], 9, np.uint32)
            ap = awords[:8].view(np.uint64)
            nang = int(awords[8])
            for i, k in enumerate(("angles", "transmit", "reflect_specular", "reflect_diffuse")):
                ang[k].append(_read(ap[i], nang, np.float32))
            aoff.append(aoff[-1] + nang)
        else:
            aoff.append(aoff[-1])
    for f in fields:
        out["surface_" + f] = np.asarray(st[f], np.float32).reshape(nsurf, nw)
        present_ptrs = [x for x in spad[f]]
        pads = _peek([q for q, _ in present_ptrs], [w for _, w in present_ptrs])
        full = np.zeros(nsurf, np.uint32)
        full[np.flatnonzero(np.asarray(present, bool))] = pads
        out["pad_surface_" + f] = full
    out["pad_dichroic_reflect"] = _peek([q for q, _ in dpad_r], [w for _, w in dpad_r])
    out["pad_dichroic_transmit"] = _peek([q for q, _ in dpad_t], [w for _, w in dpad_t])
    out["surface_present"] = np.asarray(present, np.uint8)
    out["surface_header"] = np.asarray(sheader, np.uint32).reshape(nsurf, 6)
    out["dichroic_offsets"] = np.asarray(doff, np.int64)
    out["dichroic_angles"] = np.concatenate(dang).astype(np.float32) if dang else np.zeros(0, np.float32)
    out["dichroic_reflect"] = np.asarray(dref, np.float32).reshape(-1, nw)
    out["dichroic_transmit"] = np.asarray(dtr, np.float32).reshape(-1, nw)
    out["angular_offsets"] = np.asarray(aoff, np.int64)
    for k, v in ang.items():
        out["angular_" + k] = np.concatenate(v).astype(np.float32) if v else np.zeros(0, np.float32)
    if nwire:
        wptrs = _read(ptr[8], nwire, np.uint64)
        out["wireplanes"] = np.stack([_read(p, 31, np.uint32) for p in wptrs])
    else:
        out["wireplanes"] = np.zeros((0, 31), np.uint32)
    if hasattr(gpu_geometry, "detector_gpu"):
        words = _read(gpu_geometry.detector_gpu, 14, np.uint32)
        dptr = words[:10].view(np.uint64)
        nch, ntc, nqc = (int(x) for x in words[10:13].view(np.int32))
        out["solid_id_to_channel_index"] = gpu_geometry.solid_id_to_channel_index_gpu.get().astype(np.int32)
        # The DAQ reads time/charge_cdf_len entries of both x and y; Chroma's
        # _pdf_to_cdf builds y one entry short (an out-of-bounds read in the
        # original). Export what was allocated; the replay rejects the mismatch.
        out["time_cdf_x"] = _read(dptr[1], min(ntc, gpu_geometry.time_cdf_x_gpu.size), np.float32)
        out["time_cdf_y"] = _read(dptr[2], min(ntc, gpu_geometry.time_cdf_y_gpu.size), np.float32)
        out["charge_cdf_x"] = _read(dptr[3], min(nqc, gpu_geometry.charge_cdf_x_gpu.size), np.float32)
        out["charge_cdf_y"] = _read(dptr[4], min(nqc, gpu_geometry.charge_cdf_y_gpu.size), np.float32)
        out["detector_header"] = words[10:14].copy()
    return out


# --------------------------------------------------------------- slot tracing


class _SlotTracer(object):
    """Derives per-slot draw counts and values around one kernel chunk."""

    def __init__(self):
        self.capacity = 0
        self.before = None
        self.after = None
        self.mismatch = ga.zeros(1, np.uint32)

    def reserve(self, count):
        if count > self.capacity:
            self.capacity = max(count, 1024)
            self.before = cuda.mem_alloc(self.capacity * STATE_BYTES)
            self.after = cuda.mem_alloc(self.capacity * STATE_BYTES)

    def snapshot(self, target, rng_states, count):
        if count:
            cuda.memcpy_dtod(target, rng_states, count * STATE_BYTES)

    def derive(self, count):
        """(counts u32[count], draws u32[sum]) for the last before/after pair."""
        if count == 0:
            return np.zeros(0, np.uint32), np.zeros(0, np.uint32)
        mod = _kernels()
        counts_gpu = ga.empty(count, np.uint32)
        block = 256
        grid = (count + block - 1) // block
        mod.get_function("tape_counts")(np.int32(count), self.before, self.after, np.uint32(WEYL_INVERSE),
                                        counts_gpu, block=(block, 1, 1), grid=(grid, 1))
        counts = counts_gpu.get()
        if np.any(counts > MAX_DRAWS_PER_SLOT):
            raise tapefmt.TapeError("a slot consumed an implausible number of draws; the RNG state "
                                    "was not advanced by curand_uniform only")
        offsets = np.zeros(count, np.uint64)
        np.cumsum(counts[:-1], out=offsets[1:])
        total = int(counts.astype(np.int64).sum())
        draws_gpu = ga.empty(max(total, 1), np.uint32)
        self.mismatch.fill(0)
        mod.get_function("tape_draws")(np.int32(count), self.before, self.after, counts_gpu,
                                       ga.to_gpu(offsets), draws_gpu, self.mismatch,
                                       np.uint32(MAX_DRAWS_PER_SLOT), block=(block, 1, 1), grid=(grid, 1))
        if int(self.mismatch.get()[0]):
            raise tapefmt.TapeError("regenerated XORWOW draws do not reproduce the kernel's exit state")
        return counts, draws_gpu.get()[:total]


def _csr_by_photon(nphotons, seg_photon, seg_count, seg_draws):
    """Per-photon CSR from chronological (photon, count, draws) segments."""
    if len(seg_photon) == 0:
        return np.zeros(0, np.uint32), np.zeros(nphotons + 1, np.int64)
    photon = np.concatenate(seg_photon).astype(np.int64)
    count = np.concatenate(seg_count).astype(np.int64)
    flat = np.concatenate(seg_draws).astype(np.uint32) if seg_draws else np.zeros(0, np.uint32)
    src = np.zeros(len(count), np.int64)
    np.cumsum(count[:-1], out=src[1:])
    per_photon = np.bincount(photon, weights=count, minlength=nphotons).astype(np.int64)
    offsets = np.zeros(nphotons + 1, np.int64)
    np.cumsum(per_photon, out=offsets[1:])
    order = np.argsort(photon, kind="stable")
    cnt = count[order]
    dst = np.zeros(len(order), np.int64)
    np.cumsum(cnt[:-1], out=dst[1:])
    total = int(cnt.sum())
    index = np.arange(total, dtype=np.int64) + np.repeat(src[order] - dst, cnt)
    return flat[index], offsets


# ------------------------------------------------------------------ recording


class _Batch(object):
    """Recording state of one ``_simulate_batch`` call."""

    def __init__(self, session, batch_events, params):
        self.session = session
        self.params = params
        counts = []
        for ev in batch_events:
            src = ev.photons_beg
            try:
                counts.append(len(src))
            except TypeError:
                counts.append(int(getattr(src, "true_nphotons")))
        self.event_bounds = np.cumsum(np.concatenate([[0], counts])).astype(np.int64)
        self.nphotons = int(self.event_bounds[-1])
        self.inputs = None
        self.gpu_photons = None
        self.launches = []          # dict(start, nsteps, nphotons, chunks)
        self.chunks = []            # (launch, first, count, blocks)
        self.queue = []
        self.queue_draws = []
        self.seg_photon, self.seg_count, self.seg_draws = [], [], []
        self.daq_seg_photon, self.daq_seg_count, self.daq_seg_draws = [], [], []
        self.daq_event_ran = []
        self._daq_pending = False
        self.final = None
        self.tracking = None
        self.events = []
        self.step = 0
        self.last_nsteps = 0

    # ---------------------------------------------------------- propagation

    def attach(self, gpu_photons):
        if self.gpu_photons is not None:
            return
        self.gpu_photons = gpu_photons
        self.inputs = _download_photons(gpu_photons)
        gpu_photons.gpu_funcs = _FuncObserver(gpu_photons.gpu_funcs, self)
        for name in ("propagate", "propagate_packed"):
            original = getattr(gpu_photons, name)
            setattr(gpu_photons, name, _wrap_propagate(original, self, name))

    def kernel(self, function, name, args, kwargs):
        first, count = int(args[0]), int(args[1])
        queue = args[2]
        rng_states = args[4]
        nsteps = int(args[14] if name == "propagate" else args[11])
        session = self.session
        if first == 0:
            if self.launches:
                self.step += self.last_nsteps
            nphotons = int(_read(int(queue.gpudata) - 4, 1, np.uint32)[0]) - 1 if self.launches else None
            if self.launches and session.sort_queues and nphotons > 1:
                view = queue[0:nphotons]
                ordered = np.sort(view.get())
                view.set(ordered)
            self.launches.append(dict(start=self.step, nsteps=nsteps, nphotons=0, kernel=name,
                                      use_weights=int(args[15] if name == "propagate" else args[12])))
            self.last_nsteps = nsteps
        launch = len(self.launches) - 1
        self.launches[launch]["nphotons"] += count
        blocks = int(kwargs.get("grid", (0, 1))[0])
        self.chunks.append((launch, first, count, blocks))
        if not session.record:
            return function(*args, **kwargs)
        tracer = session.tracer
        tracer.reserve(count)
        tracer.snapshot(tracer.before, rng_states, count)
        result = function(*args, **kwargs)
        tracer.snapshot(tracer.after, rng_states, count)
        ids = queue[first:first + count].get() if count else np.zeros(0, np.uint32)
        counts, draws = tracer.derive(count)
        self.queue.append(ids.astype(np.uint32))
        self.queue_draws.append(counts)
        self.seg_photon.append(ids)
        self.seg_count.append(counts)
        self.seg_draws.append(draws)
        return result

    def propagated(self, name, gpu_photons, tracking):
        if self.session.record:
            self.final = _download_final(gpu_photons, packed=(name == "propagate_packed"))
            if tracking is not None:
                ids, steps = tracking
                self.tracking = (list(ids), [_host_photons(p) for p in steps])

    # ------------------------------------------------------------------ DAQ

    def daq_kernel(self, function, args, kwargs):
        if not self.session.record:
            return function(*args, **kwargs)
        rng_states = args[0]
        first, count = int(args[2]), int(args[3])
        tracer = self.session.tracer
        tracer.reserve(count)
        tracer.snapshot(tracer.before, rng_states, count)
        result = function(*args, **kwargs)
        tracer.snapshot(tracer.after, rng_states, count)
        counts, draws = tracer.derive(count)
        self.daq_seg_photon.append(np.arange(first, first + count, dtype=np.int64))
        self.daq_seg_count.append(counts)
        self.daq_seg_draws.append(draws)
        self._daq_pending = True
        return result

    def observe_event(self, ev):
        self.events.append(dict(
            photons_end=getattr(ev, "photons_end", None),
            flat_hits=getattr(ev, "flat_hits", None),
            channels=getattr(ev, "channels", None),
            daq=self._daq_pending))
        self._daq_pending = False

    # -------------------------------------------------------------- output

    def arrays(self):
        a = {}
        for name, value in self.inputs.items():
            a["in_" + name] = value
        a["event_bounds"] = self.event_bounds
        a["draws"], a["offsets"] = _csr_by_photon(self.nphotons, self.seg_photon, self.seg_count, self.seg_draws)
        a["launch_starts"] = np.asarray([l["start"] for l in self.launches], np.int32)
        a["launch_nsteps"] = np.asarray([l["nsteps"] for l in self.launches], np.int32)
        a["launch_nphotons"] = np.asarray([l["nphotons"] for l in self.launches], np.int32)
        chunks = np.asarray(self.chunks, np.int64).reshape(-1, 4)
        for i, key in enumerate(("chunk_launch", "chunk_first", "chunk_count", "chunk_blocks")):
            a[key] = chunks[:, i].astype(np.int32)
        a["queue"] = np.concatenate(self.queue).astype(np.uint32) if self.queue else np.zeros(0, np.uint32)
        a["queue_draws"] = (np.concatenate(self.queue_draws).astype(np.uint32) if self.queue_draws
                            else np.zeros(0, np.uint32))
        a["queue_offsets"] = np.concatenate([[0], np.cumsum(a["launch_nphotons"])]).astype(np.int64)
        a["daq_draws"], a["daq_offsets"] = _csr_by_photon(self.nphotons, self.daq_seg_photon,
                                                          self.daq_seg_count, self.daq_seg_draws)
        a["daq_event_ran"] = np.asarray([e["daq"] for e in self.events], np.uint8)
        if self.final is not None:
            for name, value in self.final.items():
                a["final_" + name] = value
            geometry = self.session.simulation.detector
            if hasattr(geometry, "num_channels"):
                hits = tapefmt.photon_order_hits(self.final, geometry.solid_id,
                                                 geometry.solid_id_to_channel_index, self.event_bounds)
                for name, value in hits.items():
                    a["hits_expected_" + name] = value
        ends = [e["photons_end"] for e in self.events]
        if ends and all(p is not None for p in ends):
            for name in tapefmt.PHOTON_FIELD_NAMES:
                a["end_" + name] = np.concatenate([getattr(p, name) for p in ends])
        hits = [(i, e["flat_hits"]) for i, e in enumerate(self.events) if e["flat_hits"] is not None]
        if hits:
            for name in tapefmt.PHOTON_FIELD_NAMES + ("channel",):
                a["hits_" + name] = np.concatenate([getattr(h, name) for _, h in hits])
            a["hits_event"] = np.concatenate([np.full(len(h), i, np.int32) for i, h in hits])
        chans = [(i, e["channels"]) for i, e in enumerate(self.events) if e["channels"] is not None]
        if chans:
            a["channels_event"] = np.asarray([i for i, _ in chans], np.int32)
            a["channels_t"] = np.stack([np.asarray(c.t, np.float32) for _, c in chans])
            a["channels_q"] = np.stack([np.asarray(c.q, np.float32) for _, c in chans])
            a["channels_flags"] = np.stack([np.asarray(c.flags, np.uint32) for _, c in chans])
            a["channels_hit"] = np.stack([np.asarray(c.hit, np.uint8) for _, c in chans])
        if self.tracking is not None:
            ids, steps = self.tracking
            a["track_step_offsets"] = np.concatenate([[0], np.cumsum([len(x) for x in ids])]).astype(np.int64)
            a["track_ids"] = (np.concatenate(ids).astype(np.uint32) if ids else np.zeros(0, np.uint32))
            for name in tapefmt.PHOTON_FIELD_NAMES:
                a["track_" + name] = np.concatenate([s[name] for s in steps]) if steps else np.zeros(0)
        return a

    def entry(self):
        return dict(nphotons=self.nphotons, nevents=len(self.event_bounds) - 1, params=self.params,
                    launches=len(self.launches), queue_sorted=bool(self.session.sort_queues),
                    complete=True)


def _download_photons(p):
    n = len(p.pos)
    return dict(
        pos=p.pos.get().view(np.float32).reshape(n, 3).copy(),
        dir=p.dir.get().view(np.float32).reshape(n, 3).copy(),
        pol=p.pol.get().view(np.float32).reshape(n, 3).copy(),
        wavelengths=p.wavelengths.get().astype(np.float32),
        t=p.t.get().astype(np.float32),
        last_hit_triangles=p.last_hit_triangles.get().astype(np.int32),
        flags=p.flags.get().astype(np.uint32),
        weights=p.weights.get().astype(np.float32),
        evidx=p.evidx.get().astype(np.uint32),
    )


def _download_final(p, packed):
    out = _download_photons(p)
    if packed:
        n = len(p.pos)
        pw = p.pos_wl.get().view(np.float32).reshape(n, 4)
        dt = p.dir_t.get().view(np.float32).reshape(n, 4)
        plw = p.pol_w.get().view(np.float32).reshape(n, 4)
        out.update(pos=pw[:, :3].copy(), wavelengths=pw[:, 3].copy(), dir=dt[:, :3].copy(), t=dt[:, 3].copy(),
                   pol=plw[:, :3].copy(), weights=plw[:, 3].copy())
    return out


def _host_photons(ph):
    return {name: np.asarray(getattr(ph, name)) for name in tapefmt.PHOTON_FIELD_NAMES}


class _FuncObserver(object):
    """Stands in for ``GPUPhotons.gpu_funcs``; intercepts the propagate kernels."""

    def __init__(self, funcs, batch):
        self._funcs = funcs
        self._batch = batch

    def __getattr__(self, name):
        function = getattr(self._funcs, name)
        if name in ("propagate", "propagate_packed"):
            batch = self._batch

            def observed(*args, **kwargs):
                return batch.kernel(function, name, args, kwargs)
            return observed
        return function


class _DaqObserver(object):
    def __init__(self, funcs, session):
        self._funcs = funcs
        self._session = session

    def __getattr__(self, name):
        function = getattr(self._funcs, name)
        if name == "run_daq":
            session = self._session

            def observed(*args, **kwargs):
                batch = session.batch
                if batch is None:
                    return function(*args, **kwargs)
                return batch.daq_kernel(function, args, kwargs)
            return observed
        if name == "run_daq_many" and self._session.record:
            raise NotImplementedError("the tape recorder supports ndaq == 1 only (run_daq_many uses curand_normal)")
        return function


def _wrap_propagate(original, batch, name):
    def propagate(*args, **kwargs):
        result = original(*args, **kwargs)
        batch.propagated(name, batch.gpu_photons, result)
        return result
    return propagate


class TapeSession(object):
    """Per-Simulation recorder/sorter."""

    writer = None

    def __init__(self, simulation, mode):
        self.simulation = simulation
        self.mode = mode
        self.record = mode.mode == "record"
        self.sort_queues = mode.mode == "canonical" or (
            self.record and os.environ.get("CHROMA_TRITON_TAPE_SORT", "").strip().lower() in ("1", "true", "yes", "on"))
        self.batch = None
        self.tracer = _SlotTracer() if self.record else None
        if hasattr(simulation, "gpu_daq"):
            simulation.gpu_daq.gpu_funcs = _DaqObserver(simulation.gpu_daq.gpu_funcs, self)
        self.index = None
        if self.record:
            writer = _writer_for(mode.directory)
            nslots = simulation.nthreads_per_block * simulation.max_blocks
            initial = _read(simulation.rng_states, nslots * STATE_BYTES // 4, np.uint32).reshape(nslots, -1)
            entry = dict(seed=int(simulation.seed), nthreads_per_block=int(simulation.nthreads_per_block),
                         max_blocks=int(simulation.max_blocks), photon_tracking=bool(simulation.photon_tracking),
                         use_packed=bool(simulation.use_packed), rng_slots=nslots,
                         rng_initial_sha256=tapefmt.digest(initial[:, :6]),
                         detector_class=type(simulation.detector).__name__,
                         has_channels=hasattr(simulation.detector, "num_channels"))
            scene = export_device_scene(simulation.gpu_geometry)
            self.index = writer.add_simulation(entry, scene)
            self.writer = writer

    def begin_batch(self, batch_events, params):
        self.batch = _Batch(self, batch_events, params)
        return self.batch

    def end_batch(self, complete=True):
        batch, self.batch = self.batch, None
        if batch is None or not self.record:
            return
        if batch.inputs is None:
            return
        entry = batch.entry()
        entry["complete"] = bool(complete)
        nslots = self.simulation.nthreads_per_block * self.simulation.max_blocks
        final_rng = _read(self.simulation.rng_states, nslots * STATE_BYTES // 4, np.uint32).reshape(nslots, -1)
        entry["rng_final_sha256"] = tapefmt.digest(final_rng[:, :6])
        self.writer.add_batch(self.index, batch.arrays(), entry)


_WRITERS = {}


def _writer_for(directory):
    directory = os.path.abspath(directory)
    writer = _WRITERS.get(directory)
    if writer is None:
        root, hashes = _source_hashes()
        writer = tapefmt.TapeWriter(directory, dict(
            mode="record", chroma_root=root, source_sha256=hashes, environment=_environment(),
            argv=list(sys.argv), queue_sorted=os.environ.get("CHROMA_TRITON_TAPE_SORT", "") not in ("", "0")))
        _WRITERS[directory] = writer
    return writer


_ACTIVE = []


_OriginalGPUPhotons = gpu.GPUPhotons


class _TapedGPUPhotons(_OriginalGPUPhotons):
    """GPUPhotons that attaches the active batch recorder on construction."""

    def __init__(self, *args, **kwargs):
        _OriginalGPUPhotons.__init__(self, *args, **kwargs)
        if _ACTIVE:
            session = _ACTIVE[-1]
            if session.batch is not None:
                session.batch.attach(self)


def install(simulation_class, mode):
    """Return a Simulation subclass implementing ``mode`` (a TapeMode)."""
    if mode.mode == "replay":
        raise ValueError("CHROMA_TRITON_TAPE=replay:<dir> applies to the Triton backend; "
                         "record with CHROMA_BACKEND=cuda and CHROMA_TRITON_TAPE=record:<dir>")
    if mode.mode not in ("record", "canonical"):
        raise ValueError("unsupported tape mode %r" % (mode.mode,))
    if os.environ.get("CHROMA_DEVICE_PROFILE", "").strip().lower() in ("1", "true", "yes", "on"):
        raise ValueError("CHROMA_DEVICE_PROFILE changes the compiled kernels; disable it for tape runs")

    class TapeSimulation(simulation_class):
        __doc__ = simulation_class.__doc__

        def __init__(self, *args, **kwargs):
            simulation_class.__init__(self, *args, **kwargs)
            self._tape_session = TapeSession(self, mode)

        def _simulate_batch(self, batch_events, keep_photons_beg=False, keep_photons_end=False,
                            keep_hits=True, keep_flat_hits=True, run_daq=False, max_steps=100,
                            use_weights=False, verbose=False):
            session = self._tape_session
            params = dict(keep_photons_beg=bool(keep_photons_beg), keep_photons_end=bool(keep_photons_end),
                          keep_hits=bool(keep_hits), keep_flat_hits=bool(keep_flat_hits),
                          run_daq=bool(run_daq), max_steps=int(max_steps), use_weights=bool(use_weights))
            session.begin_batch(batch_events, params)
            _ACTIVE.append(session)
            complete = False
            try:
                for ev in simulation_class._simulate_batch(
                        self, batch_events, keep_photons_beg=keep_photons_beg,
                        keep_photons_end=keep_photons_end, keep_hits=keep_hits,
                        keep_flat_hits=keep_flat_hits, run_daq=run_daq, max_steps=max_steps,
                        use_weights=use_weights, verbose=verbose):
                    if session.batch is not None:
                        session.batch.observe_event(ev)
                    yield ev
                complete = True
            finally:
                if _ACTIVE and _ACTIVE[-1] is session:
                    _ACTIVE.pop()
                session.end_batch(complete=complete)

    TapeSimulation.__name__ = simulation_class.__name__
    TapeSimulation.__qualname__ = simulation_class.__qualname__
    gpu.GPUPhotons = _TapedGPUPhotons
    return TapeSimulation
