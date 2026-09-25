"""Production transport engine: scene upload, wavefront scheduler and DAQ.

One wavefront round runs the nearest-boundary query and one transport step
for every queued photon; survivors are appended to the next queue on the
device. The host reads one counter per round.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from chroma.triton.engine.api import DaqChannels, DevicePhotons, TERMINAL
from chroma.triton.engine import physics as P
from chroma.triton.engine.scene import compile_scene
from chroma.triton.engine.traverse import nearest_hit_kernel, wire_kernel

BLOCK = 32  # one ray/photon per thread: launch with num_warps=1
DAQ_COUNTER = tl.constexpr(0x7FFFFF00)  # Philox draw counters reserved for the DAQ
SIGN = tl.constexpr(-2147483648)


def _to_device(array, device, dtype=None):
    array = np.ascontiguousarray(array if dtype is None else np.asarray(array, dtype=dtype))
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


@triton.jit
def _daq_kernel(t_ptr, flags_ptr, last_ptr, w_ptr, ids_ptr, start, count,
                solid_offsets, nsolids, channel_of_solid,
                tcdf_x, tcdf_y, n_t, qcdf_x, qcdf_y, n_q, charge_unit,
                out_time_bits, out_q, out_hist, seed, BLOCK: tl.constexpr):
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

    def __init__(self, detector, *, seed, device, leaf_size=4):
        self.device = torch.device(device)
        self.seed = int(seed) & 0xFFFFFFFF
        self.scene = scene = compile_scene(detector, leaf_size=leaf_size)
        self.leaf_size = leaf_size
        dev = self.device
        self.nodes = _to_device(scene.nodes, dev)
        self.instances = _to_device(scene.instances, dev)
        self.tri_data = _to_device(scene.tri_data, dev)
        self.tri_local = _to_device(scene.tri_local, dev)
        self.code_m1 = _to_device(scene.code_m1, dev)
        self.code_m2 = _to_device(scene.code_m2, dev)
        self.code_s = _to_device(scene.code_surface, dev)
        self.wires = _to_device(scene.wires if len(scene.wires) else np.zeros((1, 24), np.float32), dev)
        self.n_wires = int(len(scene.wires))
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
        self._dummy_f32 = torch.zeros(1, dtype=torch.float32, device=dev)
        self._dummy_i64 = torch.zeros(1, dtype=torch.int64, device=dev)
        self._dummy_i32 = torch.zeros(1, dtype=torch.int32, device=dev)
        self.grid = None
        import os
        if os.environ.get("CHROMA_TRITON_GRID", "1") != "0":
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
                bulk=[torch.empty(capacity, dtype=torch.int32, device=dev) for _ in range(2)],
                bulk_count=[torch.zeros(1, dtype=torch.int32, device=dev) for _ in range(2)],
                boundary=torch.empty(capacity, dtype=torch.int32, device=dev),
                boundary_count=torch.zeros(1, dtype=torch.int32, device=dev),
                hit_t=torch.empty(capacity, dtype=torch.float32, device=dev),
                hit_tri=torch.empty(capacity, dtype=torch.int32, device=dev),
                hit_n=torch.empty((capacity, 3), dtype=torch.float32, device=dev),
                hit_codes=torch.empty((capacity, 3), dtype=torch.int32, device=dev),
                wire_slots=torch.empty(capacity, dtype=torch.int32, device=dev),
                wire_count=torch.zeros(1, dtype=torch.int32, device=dev),
            )
            self._workspace = ws
        return ws

    def enable_grid(self, geometry, **options):
        """Build and upload the certified empty-space grid (bulk shortcut)."""
        from chroma.triton.engine.grid import build_grid

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
                out_t, out_tri, out_n, out_codes, n, self._dummy_i32, self._dummy_i32,
                LEAF=self.leaf_size, WIRE_MODE=0, BLOCK=BLOCK, num_warps=1)
        return out_t, out_tri, out_n, out_codes

    # ---------------------------------------------------------- transport

    def _step_args(self, photons, steps, cursor, norm, renorm):
        return (photons.pos, photons.dir, photons.pol, photons.wavelengths, photons.t,
                photons.last_hit_triangles, photons.flags, photons.weights, photons.ids,
                steps, cursor, norm, self._dummy_f32, self._dummy_i64, renorm, int(renorm.numel()))

    def _material_args(self):
        return (self.rindex, self.absorption, self.scattering,
                self.comp_offsets, self.comp_prob, self.comp_wcdf, self.comp_tcdf, self.comp_abs)

    def _boundary_round(self, rows, cnt, count, n, photons, steps, cursor, norm, renorm, out_rows, out_count,
                        max_steps, use_weights):
        ws = self._workspace
        grid = (triton.cdiv(count, BLOCK),)
        split = self.n_wires > 0
        if split:
            ws["wire_count"].zero_()
        nearest_hit_kernel[grid](
            rows, cnt, photons.pos, photons.dir, photons.last_hit_triangles,
            self.nodes, self.instances, self.tri_data, self.tri_local,
            self.code_m1, self.code_m2, self.code_s, self.wires, self.n_wires,
            ws["hit_t"], ws["hit_tri"], ws["hit_n"], ws["hit_codes"], n,
            ws["wire_slots"], ws["wire_count"],
            LEAF=self.leaf_size, WIRE_MODE=1 if split else 0, BLOCK=BLOCK, num_warps=1)
        if split:
            # Analytic wires only for the compacted rays that can reach a slab.
            wire_kernel[grid](ws["wire_slots"], ws["wire_count"], n, rows, photons.pos, photons.dir,
                              ws["hit_t"], ws["hit_tri"], ws["hit_n"], ws["hit_codes"],
                              self.wires, self.n_wires, BLOCK=BLOCK, num_warps=1)
        P.step_kernel[grid](
            rows, cnt, n, *self._step_args(photons, steps, cursor, norm, renorm),
            ws["hit_t"], ws["hit_tri"], ws["hit_n"], ws["hit_codes"],
            *self._material_args(),
            self.s_present, self.s_model, self.s_detect, self.s_absorb, self.s_reemit,
            self.s_diffuse, self.s_specular, self.s_cdf,
            out_rows, out_count,
            self.seed, max_steps, self.wl_start, self.wl_step, self.time_start, self.time_step,
            NW=self.nw, NT=self.nt, MAX_COMP=self.max_comp,
            USE_WEIGHTS=bool(use_weights), TAPE=False, FIXES=True, BLOCK=BLOCK, num_warps=1)

    def propagate(self, photons, *, max_steps, use_weights=False, track=False, history=16, epochs_per_poll=4):
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
        live = torch.nonzero((photons.flags & TERMINAL) == 0).flatten().to(torch.int32)
        ws = self._buffers(n)
        count = live.numel()
        if track or getattr(self, "grid", None) is None:
            return self._propagate_stepwise(photons, live, steps, cursor, norm, renorm, max_steps, use_weights, track)
        g = self.grid
        cur = 0
        ws["bulk"][cur][:count] = live
        ws["bulk_count"][cur].fill_(count)
        bulk_count = count
        while True:
            ws["boundary_count"].zero_()
            # Bulk epochs: several launches between host polls.
            while bulk_count > 0:
                for _ in range(epochs_per_poll):
                    nxt = 1 - cur
                    ws["bulk_count"][nxt].zero_()
                    P.bulk_kernel[(triton.cdiv(bulk_count, BLOCK),)](
                        ws["bulk"][cur], ws["bulk_count"][cur], n,
                        *self._step_args(photons, steps, cursor, norm, renorm),
                        self.grid_material, self.grid_boxes,
                        float(g.lower[0]), float(g.lower[1]), float(g.lower[2]),
                        float(g.cell[0]), float(g.cell[1]), float(g.cell[2]),
                        g.shape[0], g.shape[1], g.shape[2],
                        *self._material_args(),
                        ws["bulk"][nxt], ws["bulk_count"][nxt], ws["boundary"], ws["boundary_count"],
                        self.seed, max_steps, self.wl_start, self.wl_step, self.time_start, self.time_step,
                        NW=self.nw, NT=self.nt, MAX_COMP=self.max_comp, USE_WEIGHTS=bool(use_weights),
                        TAPE=False, FIXES=True, HISTORY=history, BLOCK=BLOCK, num_warps=1)
                    cur = nxt
                bulk_count = int(ws["bulk_count"][cur].item())
            boundary_count = int(ws["boundary_count"].item())
            if boundary_count == 0:
                break
            ws["bulk_count"][cur].zero_()
            self._boundary_round(ws["boundary"], ws["boundary_count"], boundary_count, n, photons, steps, cursor,
                                 norm, renorm, ws["bulk"][cur], ws["bulk_count"][cur], max_steps, use_weights)
            bulk_count = int(ws["bulk_count"][cur].item())
        return None

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
        nch = self.nchannels
        dev = self.device
        time_bits = torch.full((nch,), int(np.float32(1e9).view(np.int32)) ^ -2147483648, dtype=torch.int32, device=dev)
        q = torch.zeros(nch, dtype=torch.int32, device=dev)
        hist = torch.zeros(nch, dtype=torch.int32, device=dev)
        if count > 0:
            grid = (triton.cdiv(count, 256),)
            _daq_kernel[grid](photons.t, photons.flags, photons.last_hit_triangles, photons.weights, photons.ids,
                              start, count, self.solid_offsets, int(self.solid_offsets.numel() - 1),
                              self.solid_id_to_channel_index,
                              self.tcdf_x, self.tcdf_y, int(self.tcdf_x.numel()),
                              self.qcdf_x, self.qcdf_y, int(self.qcdf_x.numel()), self.charge_unit,
                              time_bits, q, hist, self.seed, BLOCK=256)
        t = (time_bits.cpu().numpy() ^ np.int32(-2147483648)).view(np.float32).copy()
        charge = (q.cpu().numpy().view(np.uint32).astype(np.float32) * np.float32(self.charge_unit)).astype(np.float32)
        return DaqChannels(t=t, q=charge, flags=hist.cpu().numpy().view(np.uint32).copy())
