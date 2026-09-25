"""Exact mode of the production engine (bitwise legacy mode).

``CHROMA_TRITON_TAPE=replay:<dir>`` makes the compatibility layer build
``ProductionEngine(detector, ..., tape=TapeMode)``. The engine keeps its
device-queue scheduler -- one boundary round per transport step, survivors
appended to the next queue on the device (:func:`physics.append_rows`), one
host read of the queue length per round -- and swaps in exact kernels that
reproduce the installed CUDA Chroma (W) bit for bit:

* scene: the words the CUDA backend uploaded, read from the tape (after
  checking them against the detector; material/surface labels are matched by
  content), including W's BVH and the word W reads after every table;
* renorm: W's launch-entry ``dir/pol /= norm()`` at the recorded launch
  starts (per photon: step count listed and not yet normalized at that step);
* geometry: ``fill_state``'s boundary part (flattened BVH + FP32 wires);
* step: one iteration of W's ``propagate`` loop body with every uniform taken
  from the photon's recorded draws (``tape.draws[offsets[i]:offsets[i+1]]``);
* ``acquire``: ``run_daq`` with the recorded DAQ draws.

There is no empty-space grid, bulk shortcut or CUDA-graph tail in this mode.
The arithmetic lives in :mod:`chroma.triton.engine.exact`; the tape format in
:mod:`chroma.triton.legacy.tape`. Every case W leaves undefined or the tape
does not determine fails closed (:class:`chroma.triton.legacy.tape.TapeError`
or ``NotImplementedError``): different Simulation parameters, inputs,
detector or batch boundaries; draws exhausted or left over; a launch schedule
that the replay does not reproduce; traversal stack overflow; lookups past
the end of dichroic/angular tables; DAQ tables W reads out of bounds.
"""

import time

import numpy as np
import torch
import triton
import triton.language as tl

from chroma.triton.engine import exact as X
from chroma.triton.engine import physics as P
from chroma.triton.engine.api import DaqChannels

BLOCK = 32  # one photon per thread, one warp per program (as the production rounds)
NUM_WARPS = 1  # must equal BLOCK // 32: the traversal stack is in global memory
# Integer arguments that change between rounds, batches or scenes are not
# specialized (do_not_specialize): Triton would otherwise compile the large
# step kernel again for queue lengths divisible by 16 or equal to 1.
TERMINAL_16 = 0x800F  # W's device history (16 bit): NO_HIT|BULK_ABSORB|SURFACE_DETECT|SURFACE_ABSORB|NAN
SIGN = tl.constexpr(-2147483648)


# ------------------------------------------------------------------- scene

def bvh_stack_bound(nodes):
    """Upper bound of pending ranges in mesh.h's traversal (all boxes hit)."""
    w = np.asarray(nodes, np.uint32).reshape(-1, 4)[:, 3]
    child = (w & 0x0FFFFFFF).astype(np.int64)
    nchild = (w >> 28).astype(np.int64)
    peak = np.zeros(len(w), np.int64)
    for i in range(len(w) - 1, -1, -1):
        if nchild[i] == 0:
            continue
        pos = 0
        best = 0
        for j in range(child[i], child[i] + nchild[i]):
            if nchild[j] != 0:
                best = max(best, pos + peak[j])
                pos += 1
        peak[i] = max(pos, best)
    return int(peak[0]) if len(w) else 0


class ExactScene(object):
    """The recorded scene words (:mod:`chroma.triton.legacy.scene` keys) on the
    device, per-wavelength tables padded with the word W reads one past their
    end (``pad_*``)."""

    def __init__(self, words, device="cuda"):
        self.words = words
        self.device = device

        def t(x, dtype=None):
            x = np.ascontiguousarray(x)
            if x.dtype == np.uint32:
                x = x.view(np.int32)
            return torch.from_numpy(x.copy()).to(device)

        def padded(table, pad):
            table = np.asarray(table, np.float32)
            rows = table.shape[0]
            out = np.zeros((rows, table.shape[1] + 1), np.float32)
            out[:, :-1] = table
            if pad is not None and len(pad) == rows:
                out[:, -1] = np.asarray(pad, np.uint32).view(np.float32)
            return t(out.reshape(-1))

        self.nodes = t(words["nodes"].reshape(-1))
        self.vertices = t(np.asarray(words["vertices"], np.float32).reshape(-1))
        self.triangles = t(np.asarray(words["triangles"], np.uint32).reshape(-1))
        self.material_codes = t(np.asarray(words["material_codes"], np.uint32))
        self.planes = t(np.asarray(words["wireplanes"], np.uint32).reshape(-1))
        if self.planes.numel() == 0:
            self.planes = torch.zeros(31, dtype=torch.int32, device=device)
        self.nplanes = int(len(words["wireplanes"]))
        self.world = [float(x) for x in np.asarray(words["world_origin"], np.float32)]
        self.scale = float(np.asarray(words["world_scale"], np.float32).reshape(-1)[0])
        mh = np.asarray(words["material_header"], np.uint32)
        if len(mh) and not np.all(mh[:, 1:] == mh[0, 1:]):
            raise NotImplementedError("materials with different wavelength/time grids")
        self.nw = int(mh[0, 1])
        self.wl_step = float(mh[0, 2:3].view(np.float32)[0])
        self.wl_start = float(mh[0, 3:4].view(np.float32)[0])
        self.nt = int(mh[0, 4])
        self.t_step = float(mh[0, 5:6].view(np.float32)[0])
        self.t_start = float(mh[0, 6:7].view(np.float32)[0])
        self.stride = self.nw + 1
        self.rindex = padded(words["material_refractive_index"], words.get("pad_material_refractive_index"))
        self.absorption = padded(words["material_absorption_length"], words.get("pad_material_absorption_length"))
        self.scattering = padded(words["material_scattering_length"], words.get("pad_material_scattering_length"))
        offs = np.asarray(words["material_comp_offsets"], np.int64)
        self.comp_first = t(offs[:-1].astype(np.int32))
        self.num_comp = t(np.diff(offs).astype(np.int32))
        ncomp = int(offs[-1])
        z = np.zeros((1, self.nw), np.float32)
        self.comp_absorption = padded(words["comp_absorption_length"] if ncomp else z,
                                      words.get("pad_comp_absorption_length"))
        self.comp_prob = padded(words["comp_reemission_prob"] if ncomp else z, words.get("pad_comp_reemission_prob"))
        self.comp_wvl_cdf = t(np.asarray(words["comp_reemission_wvl_cdf"] if ncomp else z, np.float32).reshape(-1))
        zt = np.zeros((1, max(self.nt, 1)), np.float32)
        self.comp_time_cdf = t(np.asarray(words["comp_reemission_time_cdf"] if ncomp else zt, np.float32).reshape(-1))
        sh = np.asarray(words["surface_header"], np.uint32)
        present = np.asarray(words["surface_present"], np.uint8)
        self.nsurf = len(sh)
        if self.nsurf:
            live = sh[present.astype(bool)]
            if len(live) and not np.all(live[:, [1, 3, 4]] == live[0, [1, 3, 4]]):
                raise NotImplementedError("surfaces with different wavelength grids")
            if len(live) and (int(live[0, 1]) != self.nw or not np.array_equal(live[0, 3:5], mh[0, 2:4])):
                raise NotImplementedError("surface and material wavelength grids differ")
        self.surface_model = t(np.where(present.astype(bool), sh[:, 0], 99).astype(np.int32)
                               if self.nsurf else np.zeros(1, np.int32))
        self.surface_transmissive = t(sh[:, 2].astype(np.int32) if self.nsurf else np.zeros(1, np.int32))
        self.surface_thickness = t(sh[:, 5].view(np.float32) if self.nsurf else np.zeros(1, np.float32))
        sfields = ("detect", "absorb", "reemit", "reflect_diffuse", "reflect_specular", "eta", "k")
        self.surface = {}
        for f in sfields:
            table = words["surface_" + f] if self.nsurf else np.zeros((1, self.nw), np.float32)
            self.surface[f] = padded(table, words.get("pad_surface_" + f))
        self.surface_reemission_cdf = t(np.asarray(words["surface_reemission_cdf"] if self.nsurf
                                                   else np.zeros((1, self.nw)), np.float32).reshape(-1))
        doff = np.asarray(words["dichroic_offsets"], np.int64)
        self.dichroic_first = t(doff[:-1].astype(np.int32) if len(doff) > 1 else np.zeros(1, np.int32))
        self.dichroic_count = t(np.diff(doff).astype(np.int32) if len(doff) > 1 else np.zeros(1, np.int32))
        self.dichroic_angles = t(np.concatenate([np.asarray(words["dichroic_angles"], np.float32), [0.0]]).astype(np.float32))
        nd = len(words["dichroic_reflect"])
        self.dichroic_reflect = padded(words["dichroic_reflect"] if nd else z, words.get("pad_dichroic_reflect"))
        self.dichroic_transmit = padded(words["dichroic_transmit"] if nd else z, words.get("pad_dichroic_transmit"))
        aoff = np.asarray(words["angular_offsets"], np.int64)
        self.angular_first = t(aoff[:-1].astype(np.int32) if len(aoff) > 1 else np.zeros(1, np.int32))
        self.angular_count = t(np.diff(aoff).astype(np.int32) if len(aoff) > 1 else np.zeros(1, np.int32))
        for k in ("angles", "transmit", "reflect_specular", "reflect_diffuse"):
            setattr(self, "angular_" + k, t(np.concatenate([np.asarray(words["angular_" + k], np.float32), [0.0]])
                                          .astype(np.float32)))
        self.stack = max(1, bvh_stack_bound(words["nodes"]))
        if self.stack > 1000:
            raise NotImplementedError("the BVH needs %d stack entries; the original overflows at 1000" % self.stack)
        self.solid_id_map = np.asarray(words["solid_id_map"], np.uint32)
        if "solid_id_to_channel_index" in words:
            self.channel_of_solid = np.asarray(words["solid_id_to_channel_index"], np.int32)
            self.time_cdf = (np.asarray(words["time_cdf_x"], np.float32), np.asarray(words["time_cdf_y"], np.float32))
            self.charge_cdf = (np.asarray(words["charge_cdf_x"], np.float32),
                               np.asarray(words["charge_cdf_y"], np.float32))
            header = np.asarray(words["detector_header"], np.uint32)
            self.nchannels = int(header[0:1].view(np.int32)[0])
            self.charge_unit = float(header[3:4].view(np.float32)[0])
            self.detector_header = header


# ----------------------------------------------------------------- kernels


@triton.jit(do_not_specialize=["capacity", "n_mask"])
def exact_renorm_kernel(rows, count_ptr, capacity, dirs, pols, steps, norm, start_mask, n_mask,
                        BLOCK: tl.constexpr):
    """W's launch-entry normalization (``dir /= norm(dir); pol /= norm(pol)``).

    Applied to queued photons whose step count is a recorded launch start
    (``start_mask[step] != 0``) and that were not yet normalized at that step
    (``norm[row] != step``); ``norm`` remembers the step.
    """
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(count_ptr)
    if tl.program_id(0) * BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    row = tl.load(rows + lane, mask=valid, other=0).to(tl.int64)
    step = tl.load(steps + row, mask=valid, other=0)
    norm_step = tl.load(norm + row, mask=valid, other=-1)
    listed = tl.load(start_mask + step, mask=valid & (step >= 0) & (step < n_mask), other=0) != 0
    due = valid & listed & (norm_step != step)
    x = tl.load(dirs + row * 3 + 0, mask=due, other=1.0)
    y = tl.load(dirs + row * 3 + 1, mask=due, other=1.0)
    z = tl.load(dirs + row * 3 + 2, mask=due, other=1.0)
    x, y, z = X.normalize3(x, y, z)
    tl.store(dirs + row * 3 + 0, x, mask=due)
    tl.store(dirs + row * 3 + 1, y, mask=due)
    tl.store(dirs + row * 3 + 2, z, mask=due)
    x = tl.load(pols + row * 3 + 0, mask=due, other=1.0)
    y = tl.load(pols + row * 3 + 1, mask=due, other=1.0)
    z = tl.load(pols + row * 3 + 2, mask=due, other=1.0)
    x, y, z = X.normalize3(x, y, z)
    tl.store(pols + row * 3 + 0, x, mask=due)
    tl.store(pols + row * 3 + 1, y, mask=due)
    tl.store(pols + row * 3 + 2, z, mask=due)
    tl.store(norm + row, step, mask=due)


@triton.jit(do_not_specialize=["capacity", "nplanes"])
def exact_geometry_kernel(rows, count_ptr, capacity, pos, dirs, lht, nodes, vertices, triangles, material_codes,
                          planes, nplanes, stack, out_tri, out_dist, out_surface, out_m1, out_m2, out_normal, out_flag,
                          wx, wy, wz, scale, STACK: tl.constexpr, BLOCK: tl.constexpr):
    """fill_state's boundary part (W's flattened BVH, FP32 wires) for the queued photons.

    Results are per queue position (lane): triangle (-1 none, -2 wire), distance,
    surface index, material1/2, normal facing the photon, and a flag word (1 NaN
    position/direction, 2 traversal stack overflow).
    """
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(count_ptr)
    if tl.program_id(0) * BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    row = tl.load(rows + lane, mask=valid, other=0).to(tl.int64)
    px = tl.load(pos + row * 3 + 0, mask=valid, other=0.0)
    py = tl.load(pos + row * 3 + 1, mask=valid, other=0.0)
    pz = tl.load(pos + row * 3 + 2, mask=valid, other=0.0)
    dx = tl.load(dirs + row * 3 + 0, mask=valid, other=1.0)
    dy = tl.load(dirs + row * 3 + 1, mask=valid, other=0.0)
    dz = tl.load(dirs + row * 3 + 2, mask=valid, other=0.0)
    last = tl.load(lht + row, mask=valid, other=-1)
    product = X.fmul(X.fmul(X.fmul(X.fmul(X.fmul(dx, dy), dz), px), py), pz)
    nan = valid & X.fisnan(product)
    active = valid & ~nan
    tri, dist, surf, m1, m2, nx, ny, nz, overflow = X.fill_state_geometry(
        nodes, vertices, triangles, material_codes, planes, nplanes, px, py, pz, dx, dy, dz, last, active,
        stack, lane.to(tl.int64) * STACK, wx, wy, wz, scale, STACK)
    tl.store(out_tri + lane, tri, mask=valid)
    tl.store(out_dist + lane, dist, mask=valid)
    tl.store(out_surface + lane, surf, mask=valid)
    tl.store(out_m1 + lane, m1, mask=valid)
    tl.store(out_m2 + lane, m2, mask=valid)
    tl.store(out_normal + lane * 3 + 0, nx, mask=valid)
    tl.store(out_normal + lane * 3 + 1, ny, mask=valid)
    tl.store(out_normal + lane * 3 + 2, nz, mask=valid)
    tl.store(out_flag + lane, nan.to(tl.int32) + 2 * overflow.to(tl.int32), mask=valid)


@triton.jit
def _take(draws, base, cursor, length, mask, err):
    """Next draw of each masked lane from its tape segment."""
    ok = mask & (cursor < length)
    u = tl.load(draws + base + cursor, mask=ok, other=0.5)
    err = err | (mask & ~ok)
    return u, cursor + mask.to(tl.int64), err


@triton.jit(do_not_specialize=["capacity", "stride", "nw", "nt", "use_weights", "max_steps"])
def exact_step_kernel(rows, count_ptr, capacity, pos, dirs, pols, wls, times, lht, flags, weights, steps,
                      g_tri, g_dist, g_surface, g_m1, g_m2, g_normal, g_flag,
                      draws, offsets, cursor, errors,
                      rindex, absorption, scattering, stride, wl_start, wl_step, nw,
                      comp_first, num_comp, comp_absorption, comp_prob, comp_wvl_cdf, comp_time_cdf,
                      t_start, t_step, nt,
                      s_model, s_transmissive, s_thickness, s_detect, s_absorb, s_reemit, s_diffuse, s_specular,
                      s_eta, s_k, s_reemission_cdf,
                      d_first, d_count, d_angles, d_reflect, d_transmit,
                      a_first, a_count, a_angles, a_transmit, a_specular, a_diffuse,
                      use_weights, next_rows, next_count, max_steps, BLOCK: tl.constexpr):
    """One iteration of W's propagate loop body for every queued photon.

    Every uniform comes from the photon's tape segment (``draws[offsets[i]:
    offsets[i+1]]``, ``cursor`` = draws used so far) in W's call order. The
    history is stored as W's 16-bit device word. Survivors (not terminal and
    fewer than ``max_steps`` steps) are appended to ``next_rows``.
    """
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(count_ptr)
    if tl.program_id(0) * BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    row32 = tl.load(rows + lane, mask=valid, other=0)
    row = row32.to(tl.int64)
    px = tl.load(pos + row * 3 + 0, mask=valid, other=0.0)
    py = tl.load(pos + row * 3 + 1, mask=valid, other=0.0)
    pz = tl.load(pos + row * 3 + 2, mask=valid, other=0.0)
    dx = tl.load(dirs + row * 3 + 0, mask=valid, other=1.0)
    dy = tl.load(dirs + row * 3 + 1, mask=valid, other=0.0)
    dz = tl.load(dirs + row * 3 + 2, mask=valid, other=0.0)
    qx = tl.load(pols + row * 3 + 0, mask=valid, other=0.0)
    qy = tl.load(pols + row * 3 + 1, mask=valid, other=1.0)
    qz = tl.load(pols + row * 3 + 2, mask=valid, other=0.0)
    wl = tl.load(wls + row, mask=valid, other=400.0)
    t = tl.load(times + row, mask=valid, other=0.0)
    last = tl.load(lht + row, mask=valid, other=-1)
    hist = tl.load(flags + row, mask=valid, other=0)
    weight = tl.load(weights + row, mask=valid, other=1.0)
    base = tl.load(offsets + row, mask=valid, other=0)
    length = tl.load(offsets + row + 1, mask=valid, other=0) - base
    cur = tl.load(cursor + row, mask=valid, other=0)
    err = tl.zeros((BLOCK,), tl.int1)
    gflag = tl.load(g_flag + lane, mask=valid, other=0)
    nan = valid & ((gflag & 1) != 0)
    err = err | (valid & ((gflag & 2) != 0))
    tri = tl.load(g_tri + lane, mask=valid, other=-1)
    hist = tl.where(nan, hist | 32769, hist)
    live = valid & ~nan
    nohit = live & (tri == -1)
    hist = tl.where(nohit, hist | X.NO_HIT, hist)
    last = tl.where(live, tri, last)
    live = live & ~nohit
    dist = tl.load(g_dist + lane, mask=live, other=1.0)
    surf = tl.load(g_surface + lane, mask=live, other=-1)
    m1 = tl.load(g_m1 + lane, mask=live, other=0)
    m2 = tl.load(g_m2 + lane, mask=live, other=0)
    nx = tl.load(g_normal + lane * 3 + 0, mask=live, other=0.0)
    ny = tl.load(g_normal + lane * 3 + 1, mask=live, other=0.0)
    nz = tl.load(g_normal + lane * 3 + 2, mask=live, other=1.0)
    n1 = X.interp_property(rindex + m1.to(tl.int64) * stride, wl, wl_start, wl_step, nw, live)
    n2 = X.interp_property(rindex + m2.to(tl.int64) * stride, wl, wl_start, wl_step, nw, live)
    alen = X.interp_property(absorption + m1.to(tl.int64) * stride, wl, wl_start, wl_step, nw, live)
    slen = X.interp_property(scattering + m1.to(tl.int64) * stride, wl, wl_start, wl_step, nw, live)
    # ---- propagate_to_boundary
    u0, cur, err = _take(draws, base, cur, length, live, err)
    u1, cur, err = _take(draws, base, cur, length, live, err)
    da, ds, wact = X.bulk_distances(alen, slen, u0, u1, weight, use_weights)
    outcome = X.bulk_outcome(da, ds, dist)
    absorbed = live & (outcome == 0)
    scattered = live & (outcome == 1)
    passing = live & (outcome == 2)
    travel = tl.where(absorbed, da, tl.where(scattered, ds, dist))
    weight = tl.where((scattered | passing) & wact, X.attenuate(weight, travel, alen), weight)
    apx, apy, apz, at = X.advance(px, py, pz, dx, dy, dz, t, travel, n1)
    px = tl.where(live, apx, px)
    py = tl.where(live, apy, py)
    pz = tl.where(live, apz, pz)
    t = tl.where(live, at, t)
    # bulk absorption / re-emission
    ncomp = tl.load(num_comp + m1, mask=absorbed, other=0)
    plain_absorb = absorbed & (ncomp == 0)
    has_comp = absorbed & (ncomp > 0)
    last = tl.where(absorbed | scattered, -1, last)
    if tl.max(has_comp.to(tl.int32), 0) > 0:
        uc, cur, err = _take(draws, base, cur, length, has_comp, err)
        first = tl.load(comp_first + m1, mask=has_comp, other=0)
        comp = X.select_component(comp_absorption, first, ncomp, wl, alen, wl_start, wl_step, nw, uc,
                                     stride, has_comp)
        ur, cur, err = _take(draws, base, cur, length, has_comp, err)
        crow = (first + comp).to(tl.int64)
        rprob = X.interp_property(comp_prob + crow * stride, wl, wl_start, wl_step, nw, has_comp)
        reemit = has_comp & X.flt(ur, rprob)
        uw, cur, err = _take(draws, base, cur, length, reemit, err)
        new_wl = X.sample_cdf_uniform(comp_wvl_cdf + crow * nw, nw, wl_start, wl_step, uw, reemit)
        ut, cur, err = _take(draws, base, cur, length, reemit, err)
        dt = X.sample_cdf_uniform(comp_time_cdf + crow * nt, nt, t_start, t_step, ut, reemit)
        ua, cur, err = _take(draws, base, cur, length, reemit, err)
        ub, cur, err = _take(draws, base, cur, length, reemit, err)
        rx, ry, rz = X.uniform_sphere(ua, ub)
        ua, cur, err = _take(draws, base, cur, length, reemit, err)
        ub, cur, err = _take(draws, base, cur, length, reemit, err)
        rpx, rpy, rpz = X.random_polarization(ua, ub, rx, ry, rz)
        wl = tl.where(reemit, new_wl, wl)
        t = tl.where(reemit, X.fadd(t, dt), t)
        dx = tl.where(reemit, rx, dx)
        dy = tl.where(reemit, ry, dy)
        dz = tl.where(reemit, rz, dz)
        qx = tl.where(reemit, rpx, qx)
        qy = tl.where(reemit, rpy, qy)
        qz = tl.where(reemit, rpz, qz)
        hist = tl.where(reemit, hist | X.BULK_REEMIT, hist)
        plain_absorb = plain_absorb | (has_comp & ~reemit)
    hist = tl.where(plain_absorb, hist | X.BULK_ABSORB, hist)
    # Rayleigh scattering
    ua, cur, err = _take(draws, base, cur, length, scattered, err)
    ub, cur, err = _take(draws, base, cur, length, scattered, err)
    if tl.max(scattered.to(tl.int32), 0) > 0:
        sdx, sdy, sdz, sqx, sqy, sqz = X.rayleigh_scatter(dx, dy, dz, qx, qy, qz, ua, ub)
        dx = tl.where(scattered, sdx, dx)
        dy = tl.where(scattered, sdy, dy)
        dz = tl.where(scattered, sdz, dz)
        qx = tl.where(scattered, sqx, qx)
        qy = tl.where(scattered, sqy, qy)
        qz = tl.where(scattered, sqz, qz)
    hist = tl.where(scattered, hist | X.RAYLEIGH_SCATTER, hist)
    # ---- propagate_at_surface
    at_surface = passing & (surf != -1)
    model = tl.load(s_model + surf, mask=at_surface, other=0)
    diffuse = tl.zeros((BLOCK,), tl.int1)
    specular = tl.zeros((BLOCK,), tl.int1)
    fresnel_needed = passing & (surf == -1)
    sw_start = wl_start
    sw_step = wl_step
    srow = surf.to(tl.int64) * stride
    # default model
    m_def = at_surface & (model == 0)
    if tl.max(m_def.to(tl.int32), 0) > 0:
        det = X.interp_property(s_detect + srow, wl, sw_start, sw_step, nw, m_def)
        ab = X.interp_property(s_absorb + srow, wl, sw_start, sw_step, nw, m_def)
        dif = X.interp_property(s_diffuse + srow, wl, sw_start, sw_step, nw, m_def)
        spe = X.interp_property(s_specular + srow, wl, sw_start, sw_step, nw, m_def)
        us, cur, err = _take(draws, base, cur, length, m_def, err)
        action, w2 = X.surface_default(det, ab, dif, spe, us, weight, use_weights)
        weight = tl.where(m_def, w2, weight)
        hist = tl.where(m_def & (action == X.ACT_ABSORB), hist | X.SURFACE_ABSORB, hist)
        hist = tl.where(m_def & (action == X.ACT_DETECT), hist | X.SURFACE_DETECT, hist)
        diffuse = diffuse | (m_def & (action == X.ACT_DIFFUSE))
        specular = specular | (m_def & (action == X.ACT_SPECULAR))
        fresnel_needed = fresnel_needed | (m_def & (action == X.ACT_PASS))
    # WLS model
    m_wls = at_surface & (model == 2)
    if tl.max(m_wls.to(tl.int32), 0) > 0:
        ab = X.interp_property(s_absorb + srow, wl, sw_start, sw_step, nw, m_wls)
        spe = X.interp_property(s_specular + srow, wl, sw_start, sw_step, nw, m_wls)
        dif = X.interp_property(s_diffuse + srow, wl, sw_start, sw_step, nw, m_wls)
        ree = X.interp_property(s_reemit + srow, wl, sw_start, sw_step, nw, m_wls)
        us, cur, err = _take(draws, base, cur, length, m_wls, err)
        stage, w2, spe2, dif2 = X.surface_wls(ab, spe, dif, ree, us, weight, use_weights)
        weight = tl.where(m_wls, w2, weight)
        st_abs = m_wls & (stage == 0)
        st_ref = m_wls & (stage == 1)
        st_tr = m_wls & (stage == 2)
        u2, cur, err = _take(draws, base, cur, length, st_abs | st_ref, err)
        reem = st_abs & X.wls_reemits(u2, ree)
        hist = tl.where(st_abs & ~reem, hist | X.SURFACE_ABSORB, hist)
        hist = tl.where(reem, hist | X.SURFACE_REEMIT, hist)
        uw, cur, err = _take(draws, base, cur, length, reem, err)
        new_wl = X.sample_cdf_uniform(s_reemission_cdf + surf.to(tl.int64) * nw, nw, sw_start, sw_step, uw, reem)
        ua, cur, err = _take(draws, base, cur, length, reem, err)
        ub, cur, err = _take(draws, base, cur, length, reem, err)
        rx, ry, rz = X.uniform_sphere(ua, ub)
        ua, cur, err = _take(draws, base, cur, length, reem, err)
        ub, cur, err = _take(draws, base, cur, length, reem, err)
        rpx, rpy, rpz = X.random_polarization(ua, ub, rx, ry, rz)
        wl = tl.where(reem, new_wl, wl)
        dx = tl.where(reem, rx, dx)
        dy = tl.where(reem, ry, dy)
        dz = tl.where(reem, rz, dz)
        qx = tl.where(reem, rpx, qx)
        qy = tl.where(reem, rpy, qy)
        qz = tl.where(reem, rpz, qz)
        is_spec = X.wls_reflect(u2, spe2, dif2)
        specular = specular | (st_ref & is_spec)
        diffuse = diffuse | (st_ref & ~is_spec)
        hist = tl.where(st_tr, hist | X.SURFACE_TRANSMIT, hist)
        fresnel_needed = fresnel_needed | st_tr
    # dichroic model
    m_dic = at_surface & (model == 3)
    if tl.max(m_dic.to(tl.int32), 0) > 0:
        dfirst = tl.load(d_first + surf, mask=m_dic, other=0)
        dcount = tl.load(d_count + surf, mask=m_dic, other=2)
        us, cur, err = _take(draws, base, cur, length, m_dic, err)
        drow = dfirst.to(tl.int64) * stride
        action, bad = X.surface_dichroic(nx, ny, nz, dx, dy, dz, d_angles + dfirst, dcount, d_reflect + drow,
                                            d_transmit + drow, stride, wl, sw_start, sw_step, nw, us, m_dic)
        err = err | bad
        refl = m_dic & (action == X.ACT_SPECULAR)
        trans = m_dic & (action == X.ACT_TRANSMIT)
        specular = specular | refl
        hist = tl.where(trans, hist | X.SURFACE_TRANSMIT, hist)
        hist = tl.where(m_dic & (action == X.ACT_ABSORB), hist | X.SURFACE_ABSORB, hist)
        fresnel_needed = fresnel_needed | trans
    # angular model
    m_ang = at_surface & (model == 4)
    if tl.max(m_ang.to(tl.int32), 0) > 0:
        afirst = tl.load(a_first + surf, mask=m_ang, other=0)
        acount = tl.load(a_count + surf, mask=m_ang, other=2)
        us, cur, err = _take(draws, base, cur, length, m_ang, err)
        action, w2, bad = X.surface_angular(nx, ny, nz, dx, dy, dz, a_angles + afirst, a_transmit + afirst,
                                              a_specular + afirst, a_diffuse + afirst, acount, us, weight,
                                              use_weights, m_ang)
        err = err | bad
        weight = tl.where(m_ang, w2, weight)
        hist = tl.where(m_ang & (action == X.ACT_ABSORB), hist | X.SURFACE_ABSORB, hist)
        hist = tl.where(m_ang & (action == X.ACT_TRANSMIT), hist | X.SURFACE_TRANSMIT, hist)
        fresnel_needed = fresnel_needed | (m_ang & (action == X.ACT_TRANSMIT))
        specular = specular | (m_ang & (action == X.ACT_SPECULAR))
        diffuse = diffuse | (m_ang & (action == X.ACT_DIFFUSE))
    # complex (thin film) model
    m_cpx = at_surface & (model == 1)
    refract = tl.zeros((BLOCK,), tl.int1)
    if tl.max(m_cpx.to(tl.int32), 0) > 0:
        det = X.interp_property(s_detect + srow, wl, sw_start, sw_step, nw, m_cpx)
        rdif = X.interp_property(s_diffuse + srow, wl, sw_start, sw_step, nw, m_cpx)
        eta = X.interp_property(s_eta + srow, wl, sw_start, sw_step, nw, m_cpx)
        kk = X.interp_property(s_k + srow, wl, sw_start, sw_step, nw, m_cpx)
        thick = tl.load(s_thickness + surf, mask=m_cpx, other=0.0)
        trans = tl.load(s_transmissive + surf, mask=m_cpx, other=0)
        refl, absb, ax_, ay_, az_ = X.complex_probabilities(dx, dy, dz, nx, ny, nz, qx, qy, qz, n1, n2, eta, kk,
                                                               wl, thick, trans)
        forced, det2, refl2, absb2, w2 = X.surface_complex(det, refl, absb, weight, use_weights)
        weight = tl.where(m_cpx, w2, weight)
        hist = tl.where(m_cpx & forced, hist | X.SURFACE_DETECT, hist)
        drawing = m_cpx & ~forced
        us, cur, err = _take(draws, base, cur, length, drawing, err)
        stage = X.complex_stage(us, absb2, refl2, trans)
        c_abs = drawing & (stage == 0)
        c_ref = drawing & (stage == 1)
        u2, cur, err = _take(draws, base, cur, length, c_abs | c_ref, err)
        c_det = c_abs & X.flt(u2, det2)
        hist = tl.where(c_det, hist | X.SURFACE_DETECT, hist)
        hist = tl.where(c_abs & ~c_det, hist | X.SURFACE_ABSORB, hist)
        c_dif = c_ref & X.flt(u2, rdif)
        diffuse = diffuse | c_dif
        specular = specular | (c_ref & ~c_dif)
        refract = drawing & (stage == 2)
        rdx, rdy, rdz, rqx, rqy, rqz = X.complex_refract(dx, dy, dz, nx, ny, nz, n1, n2, ax_, ay_, az_)
        dx = tl.where(refract, rdx, dx)
        dy = tl.where(refract, rdy, dy)
        dz = tl.where(refract, rdz, dz)
        qx = tl.where(refract, rqx, qx)
        qy = tl.where(refract, rqy, qy)
        qz = tl.where(refract, rqz, qz)
        hist = tl.where(refract, hist | X.SURFACE_TRANSMIT, hist)
    err = err | (at_surface & (model > 4))
    # diffuse reflector (rejection loop) and its polarization
    pending = diffuse
    cdx = dx
    cdy = dy
    cdz = dz
    while tl.max(pending.to(tl.int32), 0) > 0:
        ua, cur, err = _take(draws, base, cur, length, pending, err)
        ub, cur, err = _take(draws, base, cur, length, pending, err)
        xx, yy, zz, ndotv = X.diffuse_candidate(nx, ny, nz, ua, ub)
        cdx = tl.where(pending, xx, cdx)
        cdy = tl.where(pending, yy, cdy)
        cdz = tl.where(pending, zz, cdz)
        uacc, cur, err = _take(draws, base, cur, length, pending, err)
        pending = pending & ~X.diffuse_accept(uacc, ndotv) & (cur < length)
    ua, cur, err = _take(draws, base, cur, length, diffuse, err)
    ub, cur, err = _take(draws, base, cur, length, diffuse, err)
    dpx, dpy, dpz = X.random_polarization(ua, ub, cdx, cdy, cdz)
    dx = tl.where(diffuse, cdx, dx)
    dy = tl.where(diffuse, cdy, dy)
    dz = tl.where(diffuse, cdz, dz)
    qx = tl.where(diffuse, dpx, qx)
    qy = tl.where(diffuse, dpy, qy)
    qz = tl.where(diffuse, dpz, qz)
    hist = tl.where(diffuse, hist | X.REFLECT_DIFFUSE, hist)
    # specular reflector
    if tl.max(specular.to(tl.int32), 0) > 0:
        sx, sy, sz = X.specular_reflect(dx, dy, dz, nx, ny, nz)
        dx = tl.where(specular, sx, dx)
        dy = tl.where(specular, sy, dy)
        dz = tl.where(specular, sz, dz)
    hist = tl.where(specular, hist | X.REFLECT_SPECULAR, hist)
    # dielectric boundary
    uf, cur, err = _take(draws, base, cur, length, fresnel_needed, err)
    ug, cur, err = _take(draws, base, cur, length, fresnel_needed, err)
    if tl.max(fresnel_needed.to(tl.int32), 0) > 0:
        fx, fy, fz, fqx, fqy, fqz, refl = X.fresnel(dx, dy, dz, qx, qy, qz, nx, ny, nz, n1, n2, uf, ug)
        dx = tl.where(fresnel_needed, fx, dx)
        dy = tl.where(fresnel_needed, fy, dy)
        dz = tl.where(fresnel_needed, fz, dz)
        qx = tl.where(fresnel_needed, fqx, qx)
        qy = tl.where(fresnel_needed, fqy, qy)
        qz = tl.where(fresnel_needed, fqz, qz)
        hist = tl.where(fresnel_needed & refl, hist | X.REFLECT_SPECULAR, hist)
    tl.store(pos + row * 3 + 0, px, mask=valid)
    tl.store(pos + row * 3 + 1, py, mask=valid)
    tl.store(pos + row * 3 + 2, pz, mask=valid)
    tl.store(dirs + row * 3 + 0, dx, mask=valid)
    tl.store(dirs + row * 3 + 1, dy, mask=valid)
    tl.store(dirs + row * 3 + 2, dz, mask=valid)
    tl.store(pols + row * 3 + 0, qx, mask=valid)
    tl.store(pols + row * 3 + 1, qy, mask=valid)
    tl.store(pols + row * 3 + 2, qz, mask=valid)
    tl.store(wls + row, wl, mask=valid)
    tl.store(times + row, t, mask=valid)
    tl.store(lht + row, last, mask=valid)
    tl.store(flags + row, hist & 0xFFFF, mask=valid)
    tl.store(weights + row, weight, mask=valid)
    tl.store(cursor + row, cur, mask=valid)
    tl.store(errors + row, err.to(tl.int32), mask=valid & err)
    step = tl.load(steps + row, mask=valid, other=0) + 1
    tl.store(steps + row, step, mask=valid)
    alive = valid & ((hist & 0x800F) == 0)
    P.append_rows(row32, alive & (step < max_steps), next_rows, next_count)


@triton.jit(do_not_specialize=["start", "count", "ntc", "nqc"])
def exact_daq_kernel(start, count, t_ptr, w_ptr, flags_ptr, lht_ptr, solid_map, channel_of_solid,
                     daq_draws, daq_offsets, tx, ty, ntc, qx, qy, nqc, charge_unit, global_weight,
                     out_time, out_q, out_hist, errors, BLOCK: tl.constexpr):
    """W's ``run_daq`` for rows ``[start, start+count)`` with the recorded draws.

    A photon whose last triangle belongs to a channel and whose history has
    SURFACE_DETECT draws the weight test, then (if it passes) the time and
    charge CDF samples. Channel words are combined with integer atomics as in
    W: minimum of the time bits compared as unsigned (sign bit flipped for the
    signed atomic), sum of the quantized charge, OR of the histories.
    ``errors`` counts photons whose recorded number of draws differs.
    """
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < count
    row = (start + lane).to(tl.int64)
    tri = tl.load(lht_ptr + row, mask=valid, other=-1)
    hist = tl.load(flags_ptr + row, mask=valid, other=0)
    has_tri = valid & (tri > -1)
    solid = tl.load(solid_map + tri, mask=has_tri, other=0)
    channel = tl.load(channel_of_solid + solid, mask=has_tri, other=-1)
    detected = has_tri & (channel >= 0) & ((hist & 4) != 0)
    base = tl.load(daq_offsets + row, mask=valid, other=0)
    length = tl.load(daq_offsets + row + 1, mask=valid, other=0) - base
    u0 = tl.load(daq_draws + base, mask=detected & (length > 0), other=1.0)
    weight = tl.load(w_ptr + row, mask=detected, other=0.0)
    passed = detected & (length > 0) & X.daq_passes(u0, weight, global_weight)
    u1 = tl.load(daq_draws + base + 1, mask=passed & (length > 2), other=0.5)
    u2 = tl.load(daq_draws + base + 2, mask=passed & (length > 2), other=0.5)
    tt = tl.load(t_ptr + row, mask=passed, other=0.0)
    time = X.fadd(tt, X.interp_nonuniform(u1, ty, tx, ntc, passed))
    charge = X.interp_nonuniform(u2, qy, qx, nqc, passed)
    q_int = X.daq_charge_int(charge, charge_unit)
    expected = tl.where(detected, tl.where(passed, 3, 1), 0)
    bad = valid & (expected != length)
    tl.atomic_add(errors, tl.sum(bad.to(tl.int32), axis=0))
    ch = tl.maximum(channel, 0)
    tl.atomic_min(out_time + ch, time.to(tl.int32, bitcast=True) ^ SIGN, mask=passed)
    tl.atomic_add(out_q + ch, q_int, mask=passed)
    tl.atomic_or(out_hist + ch, hist, mask=passed)


@triton.jit(do_not_specialize=["n"])
def charge_float_kernel(q, unit, out, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(out + i, X.daq_charge_float(tl.load(q + i, mask=m, other=0), unit), mask=m)


# --------------------------------------------------------------- scheduler


class _Schedule(object):
    """The recorded launch schedule of one batch and the checks the replay must pass.

    W launches ``propagate`` over the queue of live photons; launch ``k``
    covers global steps ``[starts[k], starts[k] + nsteps[k])`` and the queue
    of launch ``k+1`` holds the photons alive after launch ``k``. One engine
    round is one global step, so at round ``starts[k]`` (k >= 1) the engine's
    queue must hold exactly ``nphotons[k]`` photons, and no photon may be
    alive after the last recorded launch.
    """

    def __init__(self, batch, n, max_steps):
        from chroma.triton.legacy.tape import TapeError

        starts = np.asarray(batch["launch_starts"], np.int64)
        nsteps = np.asarray(batch["launch_nsteps"], np.int64)
        counts = np.asarray(batch["launch_nphotons"], np.int64)
        if len(starts):
            if starts[0] != 0 or np.any(starts[1:] != starts[:-1] + nsteps[:-1]) or np.any(nsteps < 1):
                raise TapeError("batch %d: the recorded launch schedule is not contiguous" % batch.index)
            if int(starts[-1] + nsteps[-1]) > int(max_steps):
                raise TapeError("batch %d: the recorded launches exceed max_steps=%d" % (batch.index, max_steps))
            if int(counts[0]) != n:
                raise TapeError("batch %d: the first recorded launch had %d photons, the batch has %d"
                                % (batch.index, counts[0], n))
        elif n and max_steps > 0:
            raise TapeError("batch %d: the tape has no launch" % batch.index)
        self.index = batch.index
        self.starts = [int(s) for s in starts]
        self.end = int(starts[-1] + nsteps[-1]) if len(starts) else 0
        self.expected = {int(s): int(c) for s, c in zip(starts[1:], counts[1:])}
        self.checked = set()

    def check_round(self, step, count):
        from chroma.triton.legacy.tape import TapeError

        if step >= self.end:
            raise TapeError("batch %d: %d photons are alive at step %d, after the last recorded launch"
                            % (self.index, count, step))
        want = self.expected.get(step)
        if want is not None:
            if want != count:
                raise TapeError("batch %d: the launch at step %d had %d photons, the replay has %d"
                                % (self.index, step, want, count))
            self.checked.add(step)

    def check_end(self):
        from chroma.triton.legacy.tape import TapeError

        missing = sorted(set(self.expected) - self.checked)
        if missing:
            raise TapeError("batch %d: the replay ran out of photons before the recorded launch at step %d"
                            % (self.index, missing[0]))

    def start_mask(self, device):
        mask = np.zeros(max(1, self.end + 1), np.int32)
        mask[self.starts] = 1
        return torch.from_numpy(mask).to(device)


class ExactMode(object):
    """Replay of a native tape by :class:`chroma.triton.engine.core.ProductionEngine`.

    Built by the engine constructor for ``tape=TapeMode('replay', dir)``. It
    owns the recorded scene, the tape reader and the exact kernels; the
    engine's ``propagate()`` and ``acquire()`` dispatch here.
    """

    def __init__(self, engine, detector, tape, *, seed=None, nthreads_per_block=None, max_blocks=None,
                 photon_tracking=None, use_packed=None):
        from chroma.triton.legacy import tape as tapefmt
        from chroma.triton.legacy.scene import check_legacy_limits, compare_scene_relabeled, scene_words

        if tape.mode != "replay":
            raise NotImplementedError(
                "CHROMA_TRITON_TAPE=%s: the Triton backend replays native tapes only; record with "
                "CHROMA_BACKEND=cuda CHROMA_TRITON_TAPE=record:<dir>, replay with replay:<dir>" % tape.mode)
        self.engine = engine
        self.device = engine.device
        self.replay = tapefmt.TapeReplay(tape.directory)
        self.replay.check_simulation(seed=None if seed is None else int(seed),
                                     nthreads_per_block=None if nthreads_per_block is None else int(nthreads_per_block),
                                     max_blocks=None if max_blocks is None else int(max_blocks),
                                     photon_tracking=None if photon_tracking is None else bool(photon_tracking),
                                     use_packed=None if use_packed is None else bool(use_packed))
        recorded = self.replay.tape.scene(self.replay.simulation)
        if detector is not None:
            # Without geometry.bvh the recorded BVH is used; every other word must match.
            problems = compare_scene_relabeled(scene_words(detector, require_bvh=False), recorded)
            if problems:
                raise tapefmt.TapeError("the detector differs from the recorded one: %s" % problems[:5])
        check_legacy_limits(recorded)
        self.scene = ExactScene(recorded, self.device)
        self._set_solids(recorded)
        self.batch = None
        self._ws = None
        self._daq = None
        self._daq_checked = False
        self.timing = None  # seconds of the last propagate(): read_tape, prepare, rounds, verify

    # ------------------------------------------------------------ helpers

    def _set_solids(self, words):
        """Solid and channel lookups for the compatibility layer's hit extraction.

        W's ``flatten`` numbers triangles solid by solid, so the recorded
        ``solid_id_map`` is non-decreasing and the engine's offset search
        (``triangle_solid``) returns exactly ``solid_id_map[triangle]``.
        """
        solid_map = np.asarray(words["solid_id_map"], np.int64)
        if len(solid_map) and np.any(np.diff(solid_map) < 0):
            raise NotImplementedError("solid ids are not in triangle order; the hit lookup assumes W's flatten()")
        if "solid_id_to_channel_index" in words:
            channels = np.asarray(words["solid_id_to_channel_index"], np.int32)
            nsolids = len(channels)
            self.engine.solid_id_to_channel_index = torch.from_numpy(channels.copy()).to(self.device)
        else:
            nsolids = int(solid_map.max()) + 1 if len(solid_map) else 0
            self.engine.solid_id_to_channel_index = None
        offsets = np.searchsorted(solid_map, np.arange(nsolids + 1), side="left").astype(np.int64)
        self.engine.solid_offsets = torch.from_numpy(offsets).to(self.device)

    def _workspace(self, capacity):
        ws = self._ws
        if ws is None or ws["capacity"] < capacity:
            dev = self.device
            i32 = dict(dtype=torch.int32, device=dev)
            ws = dict(capacity=capacity,
                      tri=torch.empty(capacity, **i32), surface=torch.empty(capacity, **i32),
                      m1=torch.empty(capacity, **i32), m2=torch.empty(capacity, **i32),
                      flag=torch.empty(capacity, **i32),
                      dist=torch.empty(capacity, dtype=torch.float32, device=dev),
                      normal=torch.empty(capacity * 3, dtype=torch.float32, device=dev),
                      stack=torch.empty(capacity * self.scene.stack, **i32))
            self._ws = ws
        return ws

    def _check_inputs(self, batch, photons):
        from chroma.triton.legacy.tape import PHOTON_FIELD_NAMES, TapeError

        n = len(photons)
        if n != batch.nphotons:
            raise TapeError("batch %d: the replay has %d photons, the tape has %d" % (batch.index, n, batch.nphotons))
        if n == 0:
            return
        expected = batch.inputs()
        for name in PHOTON_FIELD_NAMES:
            ref = torch.from_numpy(np.ascontiguousarray(expected[name]).view(np.int32).reshape(n, -1).copy())
            mine = getattr(photons, name).reshape(n, -1).contiguous().view(torch.int32)
            bad = (ref.to(self.device) != mine).any(dim=1)
            if bool(bad.any()):
                raise TapeError("batch %d: input field %s differs from the tape first at photon %d"
                                % (batch.index, name, int(torch.nonzero(bad)[0])))

    # ---------------------------------------------------------- transport

    def propagate(self, photons, *, max_steps, use_weights=False, track=False):
        from chroma.triton.legacy.tape import TapeError

        clock = [time.perf_counter()]
        batch = self.replay.next_batch(None, max_steps=int(max_steps), use_weights=bool(use_weights))
        batch.arrays  # read (and hash-check) the batch file
        clock.append(time.perf_counter())
        self._check_inputs(batch, photons)
        self.batch = batch
        self._daq = None
        self._next_event = 0
        n = len(photons)
        schedule = _Schedule(batch, n, int(max_steps))
        dev = self.device
        tracks = None
        if track:
            everyone = torch.arange(n, dtype=torch.int64, device=dev)
            tracks = [(everyone, photons.clone())]
        if n == 0:
            return tracks
        draws = batch.draws(dev)
        offsets = batch.offsets(dev)
        start_mask = schedule.start_mask(dev)
        steps = torch.zeros(n, dtype=torch.int32, device=dev)
        norm = torch.full((n,), -1, dtype=torch.int32, device=dev)
        cursor = torch.zeros(n, dtype=torch.int64, device=dev)
        errors = torch.zeros(n, dtype=torch.int32, device=dev)
        # W skips photons whose 16-bit history is terminal on entry (and stores nothing).
        live = torch.nonzero((photons.flags & TERMINAL_16) == 0).flatten().to(torch.int32)
        count = int(live.numel())
        ws = self.engine._buffers(n)
        self._workspace(n)
        torch.cuda.synchronize(dev)
        clock.append(time.perf_counter())
        cur = 0
        ws["bulk"][cur][:count] = live
        ws["bulk_count"][cur].fill_(count)
        step = 0
        if count == 0 and schedule.starts and track:
            tracks.append((everyone, photons.clone()))  # W still launches (and records) its first queue
        while count > 0 and step < max_steps:
            schedule.check_round(step, count)
            nxt = 1 - cur
            ws["bulk_count"][nxt].zero_()
            self.boundary_round(ws["bulk"][cur], ws["bulk_count"][cur], count, photons, steps, norm, cursor,
                                errors, draws, offsets, start_mask, ws["bulk"][nxt], ws["bulk_count"][nxt],
                                max_steps, use_weights, renorm=step in schedule.starts)
            if track:
                if step == 0:  # W's first launch queue holds every photon
                    tracks.append((everyone, photons.clone()))
                else:
                    processed = ws["bulk"][cur][:count].to(torch.int64)
                    tracks.append((processed, photons.select(processed)))
            cur = nxt
            count = int(ws["bulk_count"][cur].item())
            step += 1
        clock.append(time.perf_counter())
        schedule.check_end()
        used = cursor
        wanted = offsets[1:] - offsets[:-1]
        bad = (used != wanted) | (errors != 0)
        if bool(bad.any()):
            first = int(torch.nonzero(bad)[0])
            raise TapeError("batch %d: the replay diverged from the tape (%d photons; first photon %d used %d of "
                            "%d draws, harness error %d)" % (batch.index, int(bad.sum()), first, int(used[first]),
                                                             int(wanted[first]), int(errors[first])))
        clock.append(time.perf_counter())
        self.timing = dict(read_tape=clock[1] - clock[0], prepare=clock[2] - clock[1], rounds=clock[3] - clock[2],
                           verify=clock[4] - clock[3], steps=step, photons=n)
        return tracks

    def boundary_round(self, rows, cnt, count, photons, steps, norm, cursor, errors, draws, offsets, start_mask,
                       out_rows, out_count, max_steps, use_weights, renorm=True):
        """One engine round (one global step) for the queued photons: launch-entry
        normalization, the boundary query, then the step with tape draws."""
        s = self.scene
        ws = self._ws
        grid = (triton.cdiv(count, BLOCK),)
        if renorm:
            exact_renorm_kernel[grid](rows, cnt, count, photons.dir, photons.pol, steps, norm, start_mask,
                                      int(start_mask.numel()), BLOCK=BLOCK, num_warps=NUM_WARPS)
        exact_geometry_kernel[grid](rows, cnt, count, photons.pos, photons.dir, photons.last_hit_triangles,
                                    s.nodes, s.vertices, s.triangles, s.material_codes, s.planes, s.nplanes,
                                    ws["stack"], ws["tri"], ws["dist"], ws["surface"], ws["m1"], ws["m2"],
                                    ws["normal"], ws["flag"], s.world[0], s.world[1], s.world[2], s.scale,
                                    STACK=s.stack, BLOCK=BLOCK, num_warps=NUM_WARPS)
        exact_step_kernel[grid](rows, cnt, count, photons.pos, photons.dir, photons.pol, photons.wavelengths,
                                photons.t, photons.last_hit_triangles, photons.flags, photons.weights, steps,
                                ws["tri"], ws["dist"], ws["surface"], ws["m1"], ws["m2"], ws["normal"], ws["flag"],
                                draws, offsets, cursor, errors,
                                s.rindex, s.absorption, s.scattering, s.stride, s.wl_start, s.wl_step, s.nw,
                                s.comp_first, s.num_comp, s.comp_absorption, s.comp_prob, s.comp_wvl_cdf,
                                s.comp_time_cdf, s.t_start, s.t_step, s.nt,
                                s.surface_model, s.surface_transmissive, s.surface_thickness,
                                s.surface["detect"], s.surface["absorb"], s.surface["reemit"],
                                s.surface["reflect_diffuse"], s.surface["reflect_specular"], s.surface["eta"],
                                s.surface["k"], s.surface_reemission_cdf,
                                s.dichroic_first, s.dichroic_count, s.dichroic_angles, s.dichroic_reflect,
                                s.dichroic_transmit,
                                s.angular_first, s.angular_count, s.angular_angles, s.angular_transmit,
                                s.angular_reflect_specular, s.angular_reflect_diffuse,
                                int(bool(use_weights)), out_rows, out_count, int(max_steps),
                                BLOCK=BLOCK, num_warps=NUM_WARPS)

    # ---------------------------------------------------------------- DAQ

    def acquire(self, photons, start, count):
        from chroma.triton.legacy.scene import check_daq_limits
        from chroma.triton.legacy.tape import TapeError

        batch = self.batch
        if batch is None:
            raise TapeError("acquire() before propagate()")
        s = self.scene
        if not hasattr(s, "nchannels"):
            raise TapeError("the recorded geometry has no channels")
        start, count = int(start), int(count)
        bounds = np.asarray(batch.event_bounds, np.int64)
        event = next((e for e in range(self._next_event, len(bounds) - 1)
                      if bounds[e] == start and bounds[e + 1] - bounds[e] == count), None)
        if event is None:
            raise TapeError("batch %d: acquire(%d, %d) is not the next recorded event" % (batch.index, start, count))
        self._next_event = event + 1
        dev = self.device
        nch = s.nchannels
        time_bits = torch.full((nch,), int(np.float32(1e9).view(np.int32)) ^ -2147483648, dtype=torch.int32,
                               device=dev)
        q_int = torch.zeros(nch, dtype=torch.int32, device=dev)
        hist = torch.zeros(nch, dtype=torch.int32, device=dev)
        if count > 0:
            ran = batch["daq_event_ran"] if "daq_event_ran" in batch else np.zeros(len(bounds) - 1, np.uint8)
            if not ran[event]:
                raise TapeError("batch %d: the tape has no DAQ record for event %d" % (batch.index, event))
            if not self._daq_checked:
                check_daq_limits(s.words)
                self._daq_checked = True
            if self._daq is None:
                self._daq = dict(draws=batch.daq_draws(dev), offsets=batch.daq_offsets(dev),
                                 solid_map=torch.from_numpy(np.asarray(s.words["solid_id_map"], np.uint32)
                                                            .view(np.int32).copy()).to(dev),
                                 channel=torch.from_numpy(np.asarray(s.channel_of_solid, np.int32).copy()).to(dev),
                                 tx=torch.from_numpy(s.time_cdf[0].copy()).to(dev),
                                 ty=torch.from_numpy(s.time_cdf[1].copy()).to(dev),
                                 qx=torch.from_numpy(s.charge_cdf[0].copy()).to(dev),
                                 qy=torch.from_numpy(s.charge_cdf[1].copy()).to(dev),
                                 errors=torch.zeros(1, dtype=torch.int32, device=dev))
            d = self._daq
            exact_daq_kernel[(triton.cdiv(count, 128),)](
                start, count, photons.t, photons.weights, photons.flags, photons.last_hit_triangles,
                d["solid_map"], d["channel"], d["draws"], d["offsets"],
                d["tx"], d["ty"], len(s.time_cdf[0]), d["qx"], d["qy"], len(s.charge_cdf[0]),
                s.charge_unit, 1.0, time_bits, q_int, hist, d["errors"], BLOCK=128)
            if int(d["errors"].item()):
                raise TapeError("batch %d, event %d: the DAQ replay used a different number of draws than recorded"
                                % (batch.index, event))
        q = torch.empty(nch, dtype=torch.float32, device=dev)
        charge_float_kernel[(triton.cdiv(nch, 128),)](q_int, s.charge_unit, q, nch, BLOCK=128)
        t = (time_bits.cpu().numpy() ^ np.int32(-2147483648)).view(np.float32).copy()
        return DaqChannels(t=t, q=q.cpu().numpy(), flags=hist.cpu().numpy().view(np.uint32).copy())
