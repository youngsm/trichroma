"""Production transport engine: scene upload, wavefront scheduler and DAQ.

One wavefront round runs the nearest-boundary query and one transport step
for every queued photon; survivors are appended to the next queue on the
device. The host reads one counter per round.
"""

import os
import threading

import numpy as np
import torch
import triton
import triton.language as tl

from trichroma.engine.api import DaqChannels, DevicePhotons, TERMINAL
from trichroma.engine import physics as P
from trichroma.engine.scene import compile_scene
from trichroma.engine.traverse import nearest_hit_kernel, wire_kernel

BLOCK = 32  # one ray/photon per thread: launch with num_warps=1
DAQ_COUNTER = tl.constexpr(0x7FFFFF00)  # Philox draw counters reserved for the DAQ
SIGN = tl.constexpr(-2147483648)


def _warm_up_triton():
    """Before its first kernel cache lookup, a process running Triton hashes
    the Triton installation (``triton_key``, ~0.4 s, mostly libtriton.so) and
    the kernel's source tree. Doing both on a worker thread while the scene
    compiles takes them off the first propagation (the hashing releases the
    GIL). Best effort: any failure just leaves the work to the first launch."""
    try:
        from triton.compiler.compiler import triton_key

        triton_key()
        from trichroma.engine.fused import fused_kernel

        fused_kernel.cache_key
    except Exception:
        pass


def _to_device(array, device, dtype=None):
    array = np.ascontiguousarray(array if dtype is None else np.asarray(array, dtype=dtype))
    if not array.flags.writeable:  # torch.from_numpy warns on read-only arrays
        array = array.copy()
    if array.size == 0:
        array = np.zeros(max(1, array.shape[-1] if array.ndim > 1 else 1), dtype=array.dtype)
    return torch.from_numpy(array).to(device)


@triton.jit
def _interp_cdf(x_ptr, y_ptr, n, u, mask):
    """Chroma interp(u, n, cdf_y, cdf_x) on a non-uniform CDF."""
    lo = tl.zeros(u.shape, tl.int32)
    hi = tl.zeros(u.shape, tl.int32) + n - 1
    while tl.sum((mask & (lo < hi - 1)).to(tl.int32), axis=0) > 0:
        half = (lo + hi) // 2
        yv = tl.load(y_ptr + half, mask=mask, other=1.)
        width = mask & (lo < hi - 1)
        hi = tl.where(width & (u < yv), half, hi)
        lo = tl.where(width & (u >= yv), half, lo)
    y0 = tl.load(y_ptr + lo, mask=mask, other=0.)
    y1 = tl.load(y_ptr + hi, mask=mask, other=1.)
    x0 = tl.load(x_ptr + lo, mask=mask, other=0.)
    x1 = tl.load(x_ptr + hi, mask=mask, other=0.)
    return x0 + (u - y0) * (x1 - x0) / tl.where(y1 != y0, y1 - y0, 1.)


@triton.jit(do_not_specialize=["start", "count", "seed"])
def _daq_kernel(t_ptr, flags_ptr, last_ptr, w_ptr, ids_ptr, start, count,
                solid_offsets, nsolids, channel_of_solid,
                tcdf_x, tcdf_y, n_t, qcdf_x, qcdf_y, n_q, charge_unit,
                out_time_bits, out_q, out_hist, seed, BLOCK: tl.constexpr):
    seed = seed.to(tl.uint32, bitcast=True)  # passed as its int32 bit pattern (ProductionEngine.seed_arg)
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < count
    row = start + lane
    tri = tl.load(last_ptr + row, mask=valid, other=-1)
    hist = tl.load(flags_ptr + row, mask=valid, other=0)
    ok = valid & (tri > -1) & ((hist & 4) != 0)
    # Solid of the triangle: binary search in the cumulative offsets.
    lo = tl.zeros(tri.shape, tl.int32)
    hi = tl.zeros(tri.shape, tl.int32) + nsolids
    while tl.sum((ok & (lo < hi - 1)).to(tl.int32), axis=0) > 0:
        mid = (lo + hi) // 2
        off = tl.load(solid_offsets + mid, mask=ok, other=0).to(tl.int32)
        width = ok & (lo < hi - 1)
        lo = tl.where(width & (off <= tri), mid, lo)
        hi = tl.where(width & (off > tri), mid, hi)
    channel = tl.load(channel_of_solid + lo, mask=ok, other=-1)
    ok = ok & (channel >= 0)
    ids = tl.load(ids_ptr + row, mask=valid, other=0)
    weight = tl.load(w_ptr + row, mask=valid, other=1.)
    ok = ok & (P.philox_uniform(ids, seed, tl.zeros(ids.shape, tl.int32) + (DAQ_COUNTER + 0)) < weight)
    time = tl.load(t_ptr + row, mask=valid, other=0.) + _interp_cdf(tcdf_x, tcdf_y, n_t, P.philox_uniform(ids, seed, tl.zeros(ids.shape, tl.int32) + (DAQ_COUNTER + 1)), ok)
    charge = _interp_cdf(qcdf_x, qcdf_y, n_q, P.philox_uniform(ids, seed, tl.zeros(ids.shape, tl.int32) + (DAQ_COUNTER + 2)), ok)
    q_int = (charge / charge_unit + 0.5).to(tl.int32)
    ch = tl.maximum(channel, 0)
    # Chroma compares the raw float bits as unsigned integers; flipping the
    # sign bit maps that order onto signed int32 atomics.
    tl.atomic_min(out_time_bits + ch, time.to(tl.int32, bitcast=True) ^ SIGN, mask=ok)
    tl.atomic_add(out_q + ch, q_int, mask=ok)
    tl.atomic_or(out_hist + ch, hist, mask=ok)


class ProductionEngine(object):
    """General Triton transport engine for any Chroma Geometry/Detector."""

    #: exact (bitwise legacy) mode: trichroma.engine.exact_mode.ExactMode, or None
    exact = None

    def __init__(self, detector, *, seed, device, leaf_size=4, tape=None, simulation=None):
        self.device = torch.device(device)
        self.seed = int(seed) & 0xFFFFFFFF
        # Kernels take the seed as its int32 bit pattern and never specialize on
        # it: Triton types a Python int by value (i32 or i64) and specializes on
        # divisibility by 16, so every new seed could otherwise recompile them.
        self.seed_arg = self.seed - (1 << 32) if self.seed >= (1 << 31) else self.seed
        if tape is not None and tape.enabled:
            # CHROMA_TRITON_TAPE=replay:<dir>: the same round scheduler with W's arithmetic,
            # BVH and recorded draws (engine/exact_mode.py); ``simulation`` holds the
            # Simulation parameters the tape must match.
            from trichroma.engine.exact_mode import ExactMode

            self._workspace = None  # device queues of _buffers(), used by the exact rounds
            self.exact = ExactMode(self, detector, tape, seed=seed, **(simulation or {}))
            return
        threading.Thread(target=_warm_up_triton, name="chroma-triton-warm-up", daemon=True).start()
        self.scene = scene = compile_scene(detector, leaf_size=leaf_size)
        self.leaf_size = leaf_size
        dev = self.device
        self.nodes = _to_device(scene.nodes, dev)
        self.tlas_nodes = int(scene.tlas_node_count)  # nodes per octant copy of the top-level tree
        self.instances = _to_device(scene.instances, dev)
        self.tri_data = _to_device(scene.tri_data, dev)
        self.tri_local = _to_device(scene.tri_local, dev)
        self.code_m1 = _to_device(scene.code_m1, dev)
        self.code_m2 = _to_device(scene.code_m2, dev)
        self.code_s = _to_device(scene.code_surface, dev)
        self.wires = _to_device(scene.wires if len(scene.wires) else np.zeros((1, 24), np.float32), dev)
        self.n_wires = int(len(scene.wires))
        self.boxes = _to_device(scene.boxes if len(scene.boxes) else np.zeros((1, 20), np.float32), dev)
        self.box_tris = _to_device(scene.box_tris if len(scene.box_tris) else np.zeros((1, 16), np.float32), dev)
        self.n_boxes = int(len(scene.boxes))
        face_counts = scene.boxes[:, 7:18:2].view(np.int32) if len(scene.boxes) else np.zeros(1, np.int32)
        self.face_tris = int(min(16, max(1, int(face_counts.max()))))
        o = scene.optics
        m, s = o.materials, o.surfaces
        self.wl_start = float(o.wavelength_grid.start)
        self.wl_step = float(o.wavelength_grid.step)
        self.nw = int(o.wavelength_grid.count)
        self.time_start = float(o.time_grid.start)
        self.time_step = float(o.time_grid.step)
        self.nt = int(o.time_grid.count)
        f32 = np.float32
        self.rindex = _to_device(m.refractive_index, dev, f32)
        self.absorption = _to_device(m.absorption_length, dev, f32)
        self.scattering = _to_device(m.scattering_length, dev, f32)
        self.comp_offsets = _to_device(m.component_offsets, dev, np.int32)
        self.comp_prob = _to_device(m.component_reemission_prob.reshape(-1, self.nw) if m.component_count else np.zeros((1, self.nw), f32), dev, f32)
        self.comp_wcdf = _to_device(m.component_reemission_wavelength_cdf.reshape(-1, self.nw) if m.component_count else np.zeros((1, self.nw), f32), dev, f32)
        self.comp_tcdf = _to_device(m.component_reemission_time_cdf.reshape(-1, self.nt) if m.component_count else np.zeros((1, self.nt), f32), dev, f32)
        self.comp_abs = _to_device(m.component_absorption_length.reshape(-1, self.nw) if m.component_count else np.ones((1, self.nw), f32), dev, f32)
        self.max_comp = max(1, int(np.max(np.diff(m.component_offsets))) if m.count else 1)
        models = np.asarray(s.model)[np.asarray(s.present, bool)]
        unsupported = sorted(set(int(v) for v in models) - {0, 2})
        if unsupported:
            raise NotImplementedError("surface models %s are not implemented yet in the Triton production engine" % unsupported)
        nsurf = max(1, s.count)

        def surf_table(values):
            values = np.asarray(values, f32)
            return _to_device(values if s.count else np.zeros((1, self.nw), f32), dev, f32)

        self.s_present = _to_device(np.asarray(s.present, np.int32) if s.count else np.zeros(1, np.int32), dev)
        self.s_model = _to_device(np.asarray(s.model, np.int32) if s.count else np.zeros(1, np.int32), dev)
        self.s_detect = surf_table(s.detect)
        self.s_absorb = surf_table(s.absorb)
        self.s_reemit = surf_table(s.reemit)
        self.s_diffuse = surf_table(s.reflect_diffuse)
        self.s_specular = surf_table(s.reflect_specular)
        self.s_cdf = surf_table(s.reemission_cdf)
        del nsurf
        self.solid_offsets = _to_device(scene.solid_tri_offset, dev, np.int64)
        self.solid_id_to_channel_index = _to_device(scene.solid_id_to_channel_index, dev, np.int32)
        if hasattr(detector, "time_cdf"):
            self.tcdf_x = _to_device(detector.time_cdf[0], dev, f32)
            self.tcdf_y = _to_device(detector.time_cdf[1], dev, f32)
            self.qcdf_x = _to_device(detector.charge_cdf[0], dev, f32)
            self.qcdf_y = _to_device(detector.charge_cdf[1], dev, f32)
            self.charge_unit = float(np.float32(detector.charge_cdf[0][-1] / 2**16))
            self.nchannels = int(detector.num_channels())
        self._workspace = None
        self.two_phase = True
        self.traversal_steps = 1
        # CHROMA_TRITON_FIXES=0 keeps the installed Chroma's behaviour where the
        # production engine fixes it (specular polarization, Fresnel NaNs,
        # 16-bit history, FP32 wire intersection) while keeping the Philox
        # RNG: a statistical like-for-like comparison with CUDA Chroma.
        # CHROMA_TRITON_LEGACY_WIRES=0/1 overrides the wire algorithm alone.
        self.fixes = os.environ.get("CHROMA_TRITON_FIXES", "1") not in ("", "0")
        wires = os.environ.get("CHROMA_TRITON_LEGACY_WIRES", "")
        self.legacy_wires = (not self.fixes) if wires == "" else wires != "0"
        # CHROMA_TRITON_ROULETTE=<w> (opt-in, weighted mode only): Russian
        # roulette below weight w. Unbiased for every tally, but not Chroma's
        # weighted-mode semantics (hit weights are w or more instead of down to
        # 1e-4); off by default.
        self.roulette = float(os.environ.get("CHROMA_TRITON_ROULETTE", "0") or 0)
        # Rounds with at most this many live photons are replayed from a CUDA graph.
        self.tail_capacity = 32768
        self.tail_graphs = True
        # Fused transport (photon state in registers; see engine/fused.py).
        self.fused = os.environ.get("CHROMA_TRITON_FUSED", "1") not in ("", "0")
        self.fused_warps_per_sm = 24
        self.fused_park = 8
        self.fused_maxnreg = 80
        self.sm_count = torch.cuda.get_device_properties(self.device).multi_processor_count
        self._dummy_f32 = torch.zeros(1, dtype=torch.float32, device=dev)
        self._dummy_i64 = torch.zeros(1, dtype=torch.int64, device=dev)
        self._dummy_i32 = torch.zeros(1, dtype=torch.int32, device=dev)
        # The certified empty-space grid serves only the wavefront scheduler's
        # bulk shortcut; the fused kernel never reads it.
        self.grid = None
        if not self.fused and os.environ.get("CHROMA_TRITON_GRID", "1") != "0":
            self.enable_grid(detector)

    # ------------------------------------------------------------ helpers

    def triangle_solid(self, triangles):
        """Solid id of every global triangle id in ``triangles`` (int tensor)."""
        return torch.searchsorted(self.solid_offsets, triangles.to(torch.int64), right=True) - 1

    def _buffers(self, capacity):
        ws = self._workspace
        if ws is None or ws["capacity"] < capacity:
            dev = self.device
            ws = dict(
                capacity=capacity,
                bulk=[torch.empty(capacity + 1, dtype=torch.int32, device=dev) for _ in range(2)],
                bulk_count=[torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(2)],
                boundary=torch.empty(capacity, dtype=torch.int32, device=dev),
                boundary_count=torch.zeros(1, dtype=torch.int32, device=dev),
                hit_t=torch.empty(capacity, dtype=torch.float32, device=dev),
                hit_tri=torch.empty(capacity, dtype=torch.int32, device=dev),
                hit_n=torch.empty((capacity, 3), dtype=torch.float32, device=dev),
                hit_codes=torch.empty((capacity, 3), dtype=torch.int32, device=dev),
                wire_slots=torch.empty(capacity, dtype=torch.int32, device=dev),
                wire_count=torch.zeros(1, dtype=torch.int32, device=dev),
                blas_slots=torch.empty(capacity, dtype=torch.int32, device=dev),
                blas_count=torch.zeros(1, dtype=torch.int32, device=dev),
                head=torch.zeros(1, dtype=torch.int32, device=dev),
            )
            self._workspace = ws
        return ws

    def enable_grid(self, geometry, **options):
        """Build and upload the certified empty-space grid (bulk shortcut)."""
        from trichroma.engine.grid import build_grid

        grid = build_grid(self, geometry, **options)
        self.grid = grid
        self.grid_material = _to_device(grid.material, self.device, np.int32)
        self.grid_boxes = _to_device(grid.boxes, self.device, np.float32)
        return grid

    def query(self, origins, directions, last_hit=None):
        """Nearest boundary for arbitrary rays (tensors [N,3]); diagnostics/certification.

        Returns (distance, triangle, unit normal [N,3], codes [N,3] =
        inner material, outer material, surface).
        """
        dev = self.device
        origins = torch.as_tensor(origins, dtype=torch.float32, device=dev).contiguous()
        directions = torch.as_tensor(directions, dtype=torch.float32, device=dev).contiguous()
        n = origins.shape[0]
        rows = torch.arange(n, dtype=torch.int32, device=dev)
        count = torch.tensor([n], dtype=torch.int32, device=dev)
        last = torch.full((n,), -1, dtype=torch.int32, device=dev) if last_hit is None else \
            torch.as_tensor(last_hit, dtype=torch.int32, device=dev)
        out_t = torch.empty(n, dtype=torch.float32, device=dev)
        out_tri = torch.empty(n, dtype=torch.int32, device=dev)
        out_n = torch.empty((n, 3), dtype=torch.float32, device=dev)
        out_codes = torch.empty((n, 3), dtype=torch.int32, device=dev)
        if n:
            nearest_hit_kernel[(triton.cdiv(n, BLOCK),)](
                rows, count, origins, directions, last,
                self.nodes, self.instances, self.tri_data, self.tri_local,
                self.code_m1, self.code_m2, self.code_s, self.wires, self.n_wires,
                self.boxes, self.n_boxes, self.box_tris,
                out_t, out_tri, out_n, out_codes, n, self._dummy_i32, self._dummy_i32,
                tlas_nodes=self.tlas_nodes, LEAF=self.leaf_size, WIRE_MODE=0, BLOCK=BLOCK, FACE_TRIS=self.face_tris,
                LEGACY_WIRES=self.legacy_wires, num_warps=1)
        return out_t, out_tri, out_n, out_codes

    # ---------------------------------------------------------- transport

    def _step_args(self, photons, steps, cursor, norm, renorm):
        return (photons.pos, photons.dir, photons.pol, photons.wavelengths, photons.t,
                photons.last_hit_triangles, photons.flags, photons.weights, photons.ids,
                steps, cursor, norm, self._dummy_f32, self._dummy_i64, renorm, int(renorm.numel()))

    def _material_args(self):
        return (self.rindex, self.absorption, self.scattering,
                self.comp_offsets, self.comp_prob, self.comp_wcdf, self.comp_tcdf, self.comp_abs)

    def nearest_hits(self, rows, cnt, count, n, pos, dirs, last, out_t, out_tri, out_n, out_codes):
        """Nearest boundary of the queued rows (device count ``cnt``, host upper bound ``count``).

        Pass 1 runs on every ray: analytic boxes and the top-level tree, queuing
        the rays that reach an instance (and, conservatively, those that may
        reach a wire plane). Pass 2 descends into instance meshes for the
        compacted rays only, so that cheap rays do not wait in a warp for the
        few that traverse detailed meshes. Then analytic wires are merged for
        their candidates. The result equals one complete query (up to the
        rounding of a ray that hits an edge shared by two triangles).
        """
        ws = self._workspace
        grid = (triton.cdiv(count, BLOCK),)
        ws["wire_count"].zero_()
        ws["blas_count"].zero_()
        common = (self.nodes, self.instances, self.tri_data, self.tri_local,
                  self.code_m1, self.code_m2, self.code_s, self.wires, self.n_wires,
                  self.boxes, self.n_boxes, self.box_tris, out_t, out_tri, out_n, out_codes, n,
                  ws["wire_slots"], ws["wire_count"])
        if self.two_phase:
            nearest_hit_kernel[grid](
                rows, cnt, pos, dirs, last, *common,
                tlas_nodes=self.tlas_nodes, LEAF=self.leaf_size, WIRE_MODE=1, BLOCK=BLOCK, FACE_TRIS=self.face_tris, STEPS=self.traversal_steps,
                blas_slots=ws["blas_slots"], blas_count=ws["blas_count"], PHASE=1, num_warps=1)
            nearest_hit_kernel[grid](
                rows, ws["blas_count"], pos, dirs, last, *common,
                tlas_nodes=self.tlas_nodes, LEAF=self.leaf_size, WIRE_MODE=1, BLOCK=BLOCK, FACE_TRIS=self.face_tris, STEPS=self.traversal_steps,
                blas_slots=ws["blas_slots"], blas_count=ws["blas_count"], PHASE=2, num_warps=1)
        else:
            nearest_hit_kernel[grid](
                rows, cnt, pos, dirs, last, *common,
                tlas_nodes=self.tlas_nodes, LEAF=self.leaf_size, WIRE_MODE=1, BLOCK=BLOCK, FACE_TRIS=self.face_tris, STEPS=self.traversal_steps,
                num_warps=1)
        if self.n_wires:
            # Analytic wires only for the compacted rays that can reach a slab.
            wire_kernel[grid](ws["wire_slots"], ws["wire_count"], n, rows, pos, dirs,
                              out_t, out_tri, out_n, out_codes, self.wires, self.n_wires, BLOCK=BLOCK,
                              LEGACY_WIRES=self.legacy_wires, num_warps=1)

    def _boundary_round(self, rows, cnt, count, n, photons, steps, cursor, norm, renorm, out_rows, out_count,
                        max_steps, use_weights):
        ws = self._workspace
        grid = (triton.cdiv(count, BLOCK),)
        P.renorm_kernel[grid](rows, cnt, n, photons.dir, photons.pol, steps, norm, renorm, int(renorm.numel()),
                              BLOCK=BLOCK, num_warps=1)
        self.nearest_hits(rows, cnt, count, n, photons.pos, photons.dir, photons.last_hit_triangles,
                          ws["hit_t"], ws["hit_tri"], ws["hit_n"], ws["hit_codes"])
        P.step_kernel[grid](
            rows, cnt, n, *self._step_args(photons, steps, cursor, norm, renorm),
            ws["hit_t"], ws["hit_tri"], ws["hit_n"], ws["hit_codes"],
            *self._material_args(),
            self.s_present, self.s_model, self.s_detect, self.s_absorb, self.s_reemit,
            self.s_diffuse, self.s_specular, self.s_cdf,
            out_rows, out_count,
            self.seed_arg, max_steps, self.wl_start, self.wl_step, self.time_start, self.time_step,
            NW=self.nw, NT=self.nt, MAX_COMP=self.max_comp,
            USE_WEIGHTS=bool(use_weights), TAPE=False, FIXES=self.fixes, BLOCK=BLOCK,
            ROULETTE=bool(use_weights) and self.roulette > 0, w_rr=self.roulette, num_warps=1)

    def _bulk_epoch(self, photons, steps, cursor, norm, renorm, cur, nxt, grid_count, max_steps, use_weights,
                    history):
        """One bulk launch: queue ``cur`` -> queue ``nxt`` (collisions) or the boundary queue."""
        ws = self._workspace
        g = self.grid
        ws["bulk_count"][nxt].zero_()
        P.bulk_kernel[(triton.cdiv(grid_count, BLOCK),)](
            ws["bulk"][cur], ws["bulk_count"][cur], len(photons),
            *self._step_args(photons, steps, cursor, norm, renorm),
            self.grid_material, self.grid_boxes,
            float(g.lower[0]), float(g.lower[1]), float(g.lower[2]),
            float(g.cell[0]), float(g.cell[1]), float(g.cell[2]),
            g.shape[0], g.shape[1], g.shape[2],
            *self._material_args(),
            ws["bulk"][nxt], ws["bulk_count"][nxt], ws["boundary"], ws["boundary_count"],
            self.seed_arg, max_steps, self.wl_start, self.wl_step, self.time_start, self.time_step,
            NW=self.nw, NT=self.nt, MAX_COMP=self.max_comp, USE_WEIGHTS=bool(use_weights),
            TAPE=False, FIXES=self.fixes, HISTORY=history, BLOCK=BLOCK,
            ROULETTE=bool(use_weights) and self.roulette > 0, w_rr=self.roulette, num_warps=1)

    def propagate(self, photons, *, max_steps, use_weights=False, track=False, history=16, epochs_per_poll=4):
        if self.exact is not None:
            return self.exact.propagate(photons, max_steps=max_steps, use_weights=use_weights, track=track)
        n = len(photons)
        if n == 0:
            return [] if track else None
        dev = self.device
        steps = torch.zeros(n, dtype=torch.int32, device=dev)
        cursor = torch.zeros(n, dtype=torch.int32, device=dev)
        norm = torch.full((n,), -1, dtype=torch.int32, device=dev)
        # Production mode normalizes direction/polarization once on entry,
        # like a single original launch.
        renorm = torch.zeros(1, dtype=torch.int32, device=dev)
        ws = self._buffers(n)
        self.last_steps = steps  # per-photon step counts of the last call (diagnostics)
        args = (photons, steps, cursor, norm, renorm)
        if self.fused and not track:
            return self._propagate_fused(args, max_steps, use_weights)
        live = torch.nonzero((photons.flags & TERMINAL) == 0).flatten().to(torch.int32)
        count = live.numel()
        if track or self.grid is None:
            return self._propagate_stepwise(photons, live, steps, cursor, norm, renorm, max_steps, use_weights, track)
        cur = 0
        ws["bulk"][cur][:count] = live
        ws["bulk_count"][cur].fill_(count)
        bulk_count = count
        while True:
            ws["boundary_count"].zero_()
            # Bulk epochs: several launches between host polls.
            while bulk_count > 0:
                for _ in range(epochs_per_poll):
                    self._bulk_epoch(*args, cur, 1 - cur, bulk_count, max_steps, use_weights, history)
                    cur = 1 - cur
                bulk_count = int(ws["bulk_count"][cur].item())
            boundary_count = int(ws["boundary_count"].item())
            if boundary_count == 0:
                break
            ws["bulk_count"][cur].zero_()
            self._boundary_round(ws["boundary"], ws["boundary_count"], boundary_count, n, *args,
                                 ws["bulk"][cur], ws["bulk_count"][cur], max_steps, use_weights)
            bulk_count = int(ws["bulk_count"][cur].item())
            if 0 < bulk_count <= self.tail_capacity and self.tail_graphs:
                self._finish_tail(args, cur, max_steps, use_weights, history)
                break
        return None

    def _propagate_fused(self, args, max_steps, use_weights):
        """All photons in one persistent fused launch (see engine/fused.py).

        Nothing here waits for the GPU: the work list (the rows without a
        TERMINAL bit, in order) and its length are built on the device, where
        the kernel reads them.
        """
        from trichroma.engine.fused import fused_kernel

        photons, steps, cursor, norm, renorm = args
        ws = self._workspace
        n = len(photons)
        live = (photons.flags & TERMINAL) == 0
        slot = torch.cumsum(live, 0, dtype=torch.int32)
        rows = torch.arange(n, dtype=torch.int32, device=self.device)
        ws["bulk"][0].scatter_(0, torch.where(live, slot - 1, n).long(), rows)  # dead rows land in slot n
        ws["bulk_count"][0].copy_(slot[-1:])
        ws["head"].zero_()
        programs = max(1, min(self.sm_count * self.fused_warps_per_sm, triton.cdiv(n, BLOCK)))
        fused_kernel[(programs,)](
            ws["bulk"][0], ws["bulk_count"][0], ws["head"],
            photons.pos, photons.dir, photons.pol, photons.wavelengths, photons.t, photons.last_hit_triangles,
            photons.flags, photons.weights, photons.ids, steps, norm, renorm, int(renorm.numel()),
            self.nodes, self.tlas_nodes, self.instances, self.tri_data, self.tri_local, self.code_m1, self.code_m2,
            self.code_s, self.boxes, self.n_boxes, self.box_tris, self.wires, self.n_wires,
            *self._material_args(),
            self.s_present, self.s_model, self.s_detect, self.s_absorb, self.s_reemit,
            self.s_diffuse, self.s_specular, self.s_cdf,
            self.seed_arg, max_steps, self.wl_start, self.wl_step, self.time_start, self.time_step,
            NW=self.nw, NT=self.nt, MAX_COMP=self.max_comp, USE_WEIGHTS=bool(use_weights), FIXES=self.fixes,
            LEGACY_WIRES=self.legacy_wires, LEAF=self.leaf_size, FACE_TRIS=self.face_tris,
            STEPS=self.traversal_steps, BLOCK=BLOCK, PARK=self.fused_park,
            ROULETTE=bool(use_weights) and self.roulette > 0, w_rr=self.roulette,
            **({"maxnreg": self.fused_maxnreg} if self.fused_maxnreg else {}), num_warps=1)
        return None

    def _finish_tail(self, args, cur, max_steps, use_weights, history, epochs=2, rounds_per_graph=4, replays=8):
        """Finish the last photons with CUDA-graph replays of whole rounds.

        Once few photons remain, a round costs far more in host launch
        overhead than on the GPU. Every kernel reads its queue length from
        device memory, so a fixed-size round (``epochs`` bulk launches, which
        return the bulk queue to buffer ``cur``, then a boundary round that
        appends to it) is captured once and replayed; photons left in the bulk
        queue after the epochs simply continue in the next round. The host
        checks for completion every ``rounds_per_graph * replays`` rounds.
        """
        ws = self._workspace
        n = len(args[0])
        capacity = self.tail_capacity

        def one_round():
            ws["boundary_count"].zero_()
            c = cur
            for _ in range(epochs):
                self._bulk_epoch(*args, c, 1 - c, capacity, max_steps, use_weights, history)
                c = 1 - c
            self._boundary_round(ws["boundary"], ws["boundary_count"], capacity, n, *args,
                                 ws["bulk"][cur], ws["bulk_count"][cur], max_steps, use_weights)

        assert epochs % 2 == 0
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(rounds_per_graph):
                one_round()
        torch.cuda.current_stream(self.device).wait_stream(stream)
        while True:
            for _ in range(replays):
                graph.replay()
            if int(ws["bulk_count"][cur].item()) == 0:
                break

    def _propagate_stepwise(self, photons, live, steps, cursor, norm, renorm, max_steps, use_weights, track):
        """One full step per round (no bulk shortcut); used for tracking."""
        ws = self._workspace
        n = len(photons)
        count = live.numel()
        cur = 0
        ws["bulk"][cur][:count] = live
        ws["bulk_count"][cur].fill_(count)
        tracks = [] if track else None
        if track:
            tracks.append((live.to(torch.int64), photons.select(live.long()).clone()))
        rounds = 0
        while count > 0 and rounds < max_steps:
            nxt = 1 - cur
            ws["bulk_count"][nxt].zero_()
            self._boundary_round(ws["bulk"][cur], ws["bulk_count"][cur], count, n, photons, steps, cursor, norm,
                                 renorm, ws["bulk"][nxt], ws["bulk_count"][nxt], max_steps, use_weights)
            if track:
                processed = ws["bulk"][cur][:count].to(torch.int64).clone()
                tracks.append((processed, photons.select(processed).clone()))
            cur = nxt
            count = int(ws["bulk_count"][cur].item())
            rounds += 1
        return tracks

    # ---------------------------------------------------------------- DAQ

    def acquire(self, photons, start, count):
        if self.exact is not None:
            return self.exact.acquire(photons, start, count)
        words = self.acquire_batch(photons, [start, start + count])
        return self.daq_channels(*(w[0].cpu().numpy() for w in words))

    def acquire_batch(self, photons, bounds):
        """DAQ words of every event ``[bounds[i], bounds[i+1])`` of a batch.

        Returns device int32 tensors (time bits, charge counts, history) of
        shape [events, padded channels], for :meth:`daq_channels`. Rows are
        padded to 16 bytes so that every row pointer has the same alignment
        (one compiled kernel).
        """
        nch = self.nchannels
        nev = len(bounds) - 1
        dev = self.device
        width = (nch + 3) // 4 * 4
        time_bits = torch.full((nev, width), int(np.float32(1e9).view(np.int32)) ^ -2147483648, dtype=torch.int32,
                               device=dev)
        q = torch.zeros((nev, width), dtype=torch.int32, device=dev)
        hist = torch.zeros((nev, width), dtype=torch.int32, device=dev)
        for i in range(nev):
            start, count = int(bounds[i]), int(bounds[i + 1] - bounds[i])
            if count > 0:
                _daq_kernel[(triton.cdiv(count, 256),)](
                    photons.t, photons.flags, photons.last_hit_triangles, photons.weights, photons.ids,
                    start, count, self.solid_offsets, int(self.solid_offsets.numel() - 1),
                    self.solid_id_to_channel_index,
                    self.tcdf_x, self.tcdf_y, int(self.tcdf_x.numel()),
                    self.qcdf_x, self.qcdf_y, int(self.qcdf_x.numel()), self.charge_unit,
                    time_bits[i], q[i], hist[i], self.seed_arg, BLOCK=256)
        return time_bits, q, hist

    def daq_channels(self, time_bits, q, hist):
        """DaqChannels of one event from its host (numpy int32) DAQ words."""
        nch = self.nchannels
        t = (time_bits[:nch] ^ np.int32(-2147483648)).view(np.float32).copy()
        charge = (q[:nch].view(np.uint32).astype(np.float32) * np.float32(self.charge_unit)).astype(np.float32)
        return DaqChannels(t=t, q=charge, flags=hist[:nch].view(np.uint32).copy())
