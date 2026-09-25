"""Minimal reference replay of a native RNG tape through ``engine.exact``.

This is a *test harness*, not an engine: one global step per kernel pair
(geometry, then physics) over the active photons, with every draw taken from
the photon's tape segment in CUDA's order. It exists to validate tapes and
the exact-math library, and documents how an engine sequences the draws.

For every batch it checks, byte for byte, the final photon words (all
fields), the hits in photon order, the DAQ channels, and that every photon
consumed exactly its recorded number of draws.
"""

import time

import numpy as np
import torch
import triton
import triton.language as tl

from chroma.triton.engine import exact as X
from chroma.triton.legacy import tape as tapefmt

BLOCK = 64


def _torch():
    return torch


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


class DeviceScene(object):
    """Scene words (see ``scene.py``) on the device, tables padded with the
    word the original reads one past their end."""

    def __init__(self, words, device="cuda"):
        torch = _torch()
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


# ------------------------------------------------------------------ kernels


if True:  # kernels (module level: Triton resolves names from module globals)

    @triton.jit
    def normalize_kernel(rows, n, dirs, pols, BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = lane < n
        row = tl.load(rows + lane, mask=valid, other=0).to(tl.int64)
        x = tl.load(dirs + row * 3 + 0, mask=valid, other=1.0)
        y = tl.load(dirs + row * 3 + 1, mask=valid, other=1.0)
        z = tl.load(dirs + row * 3 + 2, mask=valid, other=1.0)
        x, y, z = X.normalize3(x, y, z)
        tl.store(dirs + row * 3 + 0, x, mask=valid)
        tl.store(dirs + row * 3 + 1, y, mask=valid)
        tl.store(dirs + row * 3 + 2, z, mask=valid)
        x = tl.load(pols + row * 3 + 0, mask=valid, other=1.0)
        y = tl.load(pols + row * 3 + 1, mask=valid, other=1.0)
        z = tl.load(pols + row * 3 + 2, mask=valid, other=1.0)
        x, y, z = X.normalize3(x, y, z)
        tl.store(pols + row * 3 + 0, x, mask=valid)
        tl.store(pols + row * 3 + 1, y, mask=valid)
        tl.store(pols + row * 3 + 2, z, mask=valid)

    @triton.jit
    def geometry_kernel(rows, n, pos, dirs, lht, nodes, vertices, triangles, material_codes, planes, nplanes,
                        stack, out_tri, out_dist, out_surface, out_m1, out_m2, out_normal, out_flag,
                        wx, wy, wz, scale, STACK: tl.constexpr, BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = lane < n
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

    @triton.jit
    def physics_kernel(rows, n, pos, dirs, pols, wls, times, lht, flags, weights,
                       g_tri, g_dist, g_surface, g_m1, g_m2, g_normal, g_flag,
                       draws, offsets, cursor, errors,
                       rindex, absorption, scattering, stride, wl_start, wl_step, nw,
                       comp_first, num_comp, comp_absorption, comp_prob, comp_wvl_cdf, comp_time_cdf,
                       t_start, t_step, nt,
                       s_model, s_transmissive, s_thickness, s_detect, s_absorb, s_reemit, s_diffuse, s_specular,
                       s_eta, s_k, s_reemission_cdf,
                       d_first, d_count, d_angles, d_reflect, d_transmit,
                       a_first, a_count, a_angles, a_transmit, a_specular, a_diffuse,
                       use_weights, BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = lane < n
        row = tl.load(rows + lane, mask=valid, other=0).to(tl.int64)
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
            theta = X.get_theta_neg(nx, ny, nz, dx, dy, dz)
            dfirst = tl.load(d_first + surf, mask=m_dic, other=0)
            dcount = tl.load(d_count + surf, mask=m_dic, other=2)
            idx, iidx, top = X.interp_idx(theta, d_angles + dfirst, dcount, m_dic)
            err = err | (m_dic & top)
            ok = m_dic & ~top
            r0 = X.interp_property(d_reflect + (dfirst + iidx).to(tl.int64) * stride, wl, sw_start, sw_step, nw, ok)
            r1 = X.interp_property(d_reflect + (dfirst + iidx + 1).to(tl.int64) * stride, wl, sw_start, sw_step, nw, ok)
            t0 = X.interp_property(d_transmit + (dfirst + iidx).to(tl.int64) * stride, wl, sw_start, sw_step, nw, ok)
            t1 = X.interp_property(d_transmit + (dfirst + iidx + 1).to(tl.int64) * stride, wl, sw_start, sw_step, nw, ok)
            frac = X.fsub(idx, X.cvt_f32_u32(iidx))
            rp = X.ffma(frac, X.fsub(r1, r0), r0)
            tp = X.ffma(frac, X.fsub(t1, t0), t0)
            us, cur, err = _take(draws, base, cur, length, m_dic, err)
            refl = m_dic & X.flt(us, rp)
            trans = m_dic & ~refl & X.flt(us, X.fadd(tp, rp))
            specular = specular | refl
            hist = tl.where(trans, hist | X.SURFACE_TRANSMIT, hist)
            hist = tl.where(m_dic & ~refl & ~trans, hist | X.SURFACE_ABSORB, hist)
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
        # complex model is not in the reference harness yet
        err = err | (at_surface & (model == 1)) | (at_surface & (model > 4))
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

    @triton.jit
    def daq_kernel(n, t, w, u_w, u_t, u_q, tx, ty, ntc, qx, qy, nqc, unit, out_pass, out_time, out_q,
                   BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tt = tl.load(t + i, mask=m, other=0.0)
        ww = tl.load(w + i, mask=m, other=1.0)
        a = tl.load(u_w + i, mask=m, other=1.0)
        b = tl.load(u_t + i, mask=m, other=0.5)
        c = tl.load(u_q + i, mask=m, other=0.5)
        ok = m & X.daq_passes(a, ww, tl.full((BLOCK,), 1.0, tl.float32))
        time = X.fadd(tt, X.interp_nonuniform(b, ty, tx, ntc, ok))
        charge = X.interp_nonuniform(c, qy, qx, nqc, ok)
        qi = X.daq_charge_int(charge, unit)
        tl.store(out_pass + i, ok.to(tl.int32), mask=m)
        tl.store(out_time + i, time, mask=m)
        tl.store(out_q + i, qi, mask=m)

    @triton.jit
    def charge_float_kernel(q, unit, out, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(out + i, X.daq_charge_float(tl.load(q + i, mask=m, other=0), unit), mask=m)


def kernels():
    return normalize_kernel, geometry_kernel, physics_kernel


# ------------------------------------------------------------------ replay


def propagate(scene, fields, batch, device="cuda"):
    """Replay the recorded propagation of one batch. Returns (fields, report)."""
    torch = _torch()
    normalize_kernel, geometry_kernel, physics_kernel = kernels()
    n = batch.nphotons
    as_t = lambda x, dt=None: torch.from_numpy(np.ascontiguousarray(x).copy()).to(device)
    pos = as_t(np.asarray(fields["pos"], np.float32).reshape(-1))
    dirs = as_t(np.asarray(fields["dir"], np.float32).reshape(-1))
    pols = as_t(np.asarray(fields["pol"], np.float32).reshape(-1))
    wls = as_t(np.asarray(fields["wavelengths"], np.float32))
    times = as_t(np.asarray(fields["t"], np.float32))
    lht = as_t(np.asarray(fields["last_hit_triangles"], np.int32))
    flags_np = np.asarray(fields["flags"], np.uint32)
    live0 = ((flags_np & 0xFFFF) & 0x800F) == 0
    flags = as_t(np.where(live0, flags_np & 0xFFFF, flags_np).view(np.int32))
    weights = as_t(np.asarray(fields["weights"], np.float32))
    draws = batch.draws(device)
    offsets = batch.offsets(device)
    cursor = torch.zeros(n, dtype=torch.int64, device=device)
    errors = torch.zeros(n, dtype=torch.int32, device=device)
    live = torch.from_numpy(live0).to(device)
    starts = [int(x) for x in batch["launch_starts"]]
    nsteps = [int(x) for x in batch["launch_nsteps"]]
    use_weights = int(bool(batch.params.get("use_weights", False)))
    cap = max(1, n)
    g_tri = torch.empty(cap, dtype=torch.int32, device=device)
    g_dist = torch.empty(cap, dtype=torch.float32, device=device)
    g_surface = torch.empty(cap, dtype=torch.int32, device=device)
    g_m1 = torch.empty(cap, dtype=torch.int32, device=device)
    g_m2 = torch.empty(cap, dtype=torch.int32, device=device)
    g_normal = torch.empty(cap * 3, dtype=torch.float32, device=device)
    g_flag = torch.empty(cap, dtype=torch.int32, device=device)
    stack = torch.empty(cap * scene.stack, dtype=torch.int32, device=device)
    steps_done = 0
    s = scene
    for start, count in zip(starts, nsteps):
        if start != steps_done:
            raise tapefmt.TapeError("launch schedule is not contiguous")
        rows = torch.nonzero(live).flatten().to(torch.int32)
        if len(rows) == 0:
            break
        grid = ((len(rows) + BLOCK - 1) // BLOCK,)
        normalize_kernel[grid](rows, len(rows), dirs, pols, BLOCK=BLOCK)
        for _ in range(count):
            if len(rows) == 0:
                break
            grid = ((len(rows) + BLOCK - 1) // BLOCK,)
            geometry_kernel[grid](rows, len(rows), pos, dirs, lht, s.nodes, s.vertices, s.triangles,
                                  s.material_codes, s.planes, s.nplanes, stack, g_tri, g_dist, g_surface,
                                  g_m1, g_m2, g_normal, g_flag, s.world[0], s.world[1], s.world[2], s.scale,
                                  STACK=s.stack, BLOCK=BLOCK)
            physics_kernel[grid](rows, len(rows), pos, dirs, pols, wls, times, lht, flags, weights,
                                 g_tri, g_dist, g_surface, g_m1, g_m2, g_normal, g_flag,
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
                                 use_weights, BLOCK=BLOCK)
            steps_done += 1
            alive = (flags.index_select(0, rows.long()) & 0x800F) == 0
            rows = rows[alive]
        live = torch.zeros(n, dtype=torch.bool, device=device)
        if len(rows):
            live[rows.long()] = True
        steps_done = start + count
    torch.cuda.synchronize()
    out = dict(pos=pos.view(n, 3).cpu().numpy(), dir=dirs.view(n, 3).cpu().numpy(),
               pol=pols.view(n, 3).cpu().numpy(), wavelengths=wls.cpu().numpy(), t=times.cpu().numpy(),
               last_hit_triangles=lht.cpu().numpy(), flags=flags.cpu().numpy().view(np.uint32),
               weights=weights.cpu().numpy(), evidx=np.asarray(fields["evidx"], np.uint32))
    used = cursor.cpu().numpy()
    expected = np.diff(batch["offsets"])
    report = dict(photons=int(n), launches=len(starts), steps=int(steps_done),
                  draws=int(expected.sum()), draw_count_mismatch=int(np.count_nonzero(used != expected)),
                  harness_errors=int(np.count_nonzero(errors.cpu().numpy())))
    if report["draw_count_mismatch"]:
        report["first_draw_count_mismatch"] = int(np.flatnonzero(used != expected)[0])
    return out, report


def daq(scene, fields, batch):
    """Replay run_daq for every event that ran it (host arithmetic via Triton
    kernels would add nothing here; the per-photon draws are few)."""
    if not hasattr(scene, "nchannels"):
        return {}, 0
    from chroma.triton.legacy.scene import check_daq_limits
    check_daq_limits(scene.words)

    bounds = batch.event_bounds
    ran = batch["daq_event_ran"] if "daq_event_ran" in batch else np.zeros(len(bounds) - 1, np.uint8)
    doffs = batch["daq_offsets"]
    ddraws = batch["daq_draws"].view(np.float32)
    tri = np.asarray(fields["last_hit_triangles"], np.int64)
    flags = np.asarray(fields["flags"], np.uint32)
    channel = np.full(len(tri), -1, np.int64)
    ok = tri > -1
    channel[ok] = scene.channel_of_solid[scene.solid_id_map[tri[ok]]]
    detected = ok & (channel >= 0) & ((flags & 4) != 0)
    results = {}
    mism = 0
    dev = "cuda"
    tx = torch.from_numpy(scene.time_cdf[0].copy()).to(dev)
    ty = torch.from_numpy(scene.time_cdf[1].copy()).to(dev)
    qx = torch.from_numpy(scene.charge_cdf[0].copy()).to(dev)
    qy = torch.from_numpy(scene.charge_cdf[1].copy()).to(dev)
    for e in range(len(bounds) - 1):
        if not ran[e]:
            continue
        lo, hi = int(bounds[e]), int(bounds[e + 1])
        idx = np.arange(lo, hi)
        det = idx[detected[lo:hi]]
        counts = np.diff(doffs)[lo:hi]
        expect_counts = np.zeros(hi - lo, np.int64)
        u = np.ones((len(det), 3), np.float32)
        for k, i in enumerate(det):
            seg = ddraws[doffs[i]:doffs[i + 1]]
            u[k, :len(seg)] = seg[:3]
        tt = torch.from_numpy(np.asarray(fields["t"], np.float32)[det].copy()).to(dev)
        ww = torch.from_numpy(np.asarray(fields["weights"], np.float32)[det].copy()).to(dev)
        uu = torch.from_numpy(u.copy()).to(dev)
        npass = torch.empty(len(det), dtype=torch.int32, device=dev)
        ntime = torch.empty(len(det), dtype=torch.float32, device=dev)
        nq = torch.empty(len(det), dtype=torch.int32, device=dev)
        if len(det):
            daq_kernel[((len(det) + 127) // 128,)](len(det), tt, ww, uu[:, 0].contiguous(), uu[:, 1].contiguous(),
                                                   uu[:, 2].contiguous(), tx, ty, len(scene.time_cdf[0]), qx, qy,
                                                   len(scene.charge_cdf[0]), scene.charge_unit, npass, ntime, nq,
                                                   BLOCK=128)
        passed = npass.cpu().numpy().astype(bool)
        times = ntime.cpu().numpy()
        charges = nq.cpu().numpy().view(np.uint32)
        expect_counts[det - lo] = np.where(passed, 3, 1)
        mism += int(np.count_nonzero(expect_counts != counts))
        nch = scene.nchannels
        tint = np.full(nch, np.float32(1e9).view(np.uint32), np.uint32)
        qint = np.zeros(nch, np.uint32)
        hist = np.zeros(nch, np.uint32)
        ch = channel[det][passed]
        np.minimum.at(tint, ch, times[passed].view(np.uint32))
        np.add.at(qint, ch, charges[passed])
        np.bitwise_or.at(hist, ch, flags[det][passed])
        tq = torch.from_numpy(qint.view(np.int32).copy()).to(dev)
        qout = torch.empty(nch, dtype=torch.float32, device=dev)
        charge_float_kernel[((nch + 127) // 128,)](tq, scene.charge_unit, qout, nch, BLOCK=128)
        results[e] = dict(t=tint.view(np.float32), q=qout.cpu().numpy(), flags=hist,
                          hit=(tint.view(np.float32) < 1e8))
    return results, mism


def _compare_words(a, b):
    wa = tapefmt.photon_words(a)
    wb = tapefmt.photon_words(b)
    diff = np.flatnonzero(np.any(wa != wb, axis=1))
    if not len(diff):
        return None
    r = int(diff[0])
    c = int(np.flatnonzero(wa[r] != wb[r])[0])
    return dict(photon=r, word=tapefmt.WORD_NAMES[c], expected=hex(int(wa[r, c])), actual=hex(int(wb[r, c])),
                differing_photons=int(len(diff)))


def replay_tape(directory, device="cuda"):
    """Replay every batch of a tape; compare with its recorded outputs."""
    tape = tapefmt.Tape(directory)
    report = dict(tape=directory, batches=[], equal=True)
    scenes = {}
    for batch in tape:
        t0 = time.time()
        if batch.simulation not in scenes:
            scenes[batch.simulation] = DeviceScene(tape.scene(batch.simulation), device)
        scene = scenes[batch.simulation]
        info = tape.simulations[batch.simulation]
        final, rep = propagate(scene, batch.inputs(), batch, device)
        rep["seconds"] = time.time() - t0
        rep["final_equal"] = None
        expected_final = batch.final()
        if expected_final is not None:
            d = _compare_words(expected_final, final)
            rep["final_equal"] = d is None
            rep["final_first_difference"] = d
        packed = bool(info.get("use_packed", False))
        visible = dict(final)
        if packed:
            inputs = batch.inputs()
            for key in ("pos", "dir", "pol", "wavelengths", "t", "weights"):
                visible[key] = inputs[key]
        if "end_pos" in batch:
            d = _compare_words(batch.fields("end"), visible)
            rep["photons_end_equal"] = d is None
            rep["photons_end_first_difference"] = d
        if "hits_expected_pos" in batch and hasattr(scene, "channel_of_solid"):
            hits = tapefmt.photon_order_hits(final, scene.solid_id_map, scene.channel_of_solid, batch.event_bounds)
            exp = {k[len("hits_expected_"):]: batch[k] for k in batch.arrays if k.startswith("hits_expected_")}
            same = len(hits["photon"]) == len(exp["photon"]) and np.array_equal(hits["photon"], exp["photon"])
            d = _compare_words(exp, hits) if same else "different hit photons"
            same = same and d is None and np.array_equal(hits["channel"], exp["channel"])
            rep["hits_equal"] = bool(same)
            rep["hits"] = int(len(exp["photon"]))
            if not same:
                rep["hits_first_difference"] = d
        if "channels_t" in batch:
            params = batch.params
            daq_fields = dict(final)
            if packed and not (params.get("keep_hits") or params.get("keep_flat_hits")):
                inputs = batch.inputs()
                daq_fields["t"] = inputs["t"]
                daq_fields["weights"] = inputs["weights"]
            channels, dmism = daq(scene, daq_fields, batch)
            ok = dmism == 0
            for row, e in enumerate(batch["channels_event"]):
                got = channels.get(int(e))
                for key in ("t", "q", "flags", "hit"):
                    exp = batch["channels_" + key][row]
                    val = got[key] if got is not None else None
                    if val is None or np.asarray(exp).astype(np.asarray(val).dtype).tobytes() != np.asarray(val).tobytes():
                        ok = False
            rep["channels_equal"] = bool(ok)
            rep["daq_draw_count_mismatch"] = int(dmism)
        rep["equal"] = bool(all(rep.get(k, True) in (True, None) for k in
                                ("final_equal", "photons_end_equal", "hits_equal", "channels_equal"))
                            and rep["draw_count_mismatch"] == 0 and rep["harness_errors"] == 0)
        report["batches"].append(rep)
        report["equal"] = report["equal"] and rep["equal"]
    return report
