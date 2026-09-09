"""Triton execution for spectral.py; imported only when GPU execution is requested."""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.random import philox

from .bvh_kernels import nearest_hit_local
from .boundary_kernels import offset_boundary_point
from .physics_kernels import fresnel_step, rayleigh_scatter


@triton.jit
def random_uniform(ids, seed, stream):
    low = ids.to(tl.uint64).to(tl.uint32)
    high = (ids.to(tl.uint64) >> 32).to(tl.uint32)
    stream_u = (tl.full(ids.shape, 0, tl.uint32) + stream).to(tl.uint32)
    x, _, _, _ = philox(seed, low, high, stream_u, tl.full(ids.shape, 0, tl.uint32))
    return ((x >> 9).to(tl.float32) + 0.5) * 1.1920928955078125e-7


@triton.jit
def interpolate(table, row, wavelength, start, step, count: tl.constexpr, mask):
    f = tl.minimum(tl.maximum((wavelength-start)/step, 0.), count-1.)
    lo = tl.minimum(f.to(tl.int32), count-2)
    alpha = f-lo
    left = tl.load(table+row*count+lo, mask=mask, other=0.)
    right = tl.load(table+row*count+lo+1, mask=mask, other=0.)
    value = tl.where(left == right, left, (1-alpha)*left + alpha*right)
    return tl.where(alpha <= 0, left, tl.where(alpha >= 1, right, value))


@triton.jit
def sample_regular_cdf(table, row, u, start, step, count: tl.constexpr, mask):
    lo = tl.full(u.shape, 0, tl.int32)
    hi = tl.full(u.shape, count-1, tl.int32)
    while tl.sum((mask & (hi-lo > 1)).to(tl.int32), axis=0) > 0:
        mid = (lo+hi)//2
        value = tl.load(table+row*count+mid, mask=mask, other=1.)
        right = value <= u
        lo = tl.where(right & (hi-lo > 1), mid, lo)
        hi = tl.where(~right & (hi-lo > 1), mid, hi)
    left = tl.load(table+row*count+lo, mask=mask, other=0.)
    right = tl.load(table+row*count+hi, mask=mask, other=1.)
    return start + (lo + (u-left)/tl.maximum(right-left, 1.e-30)) * step


@triton.jit
def sample_time_cdf(offsets, x, cdf, pdf, row, u, mask):
    begin = tl.load(offsets+row, mask=mask, other=0)
    end = tl.load(offsets+row+1, mask=mask, other=2)
    lo, hi = begin, end-1
    while tl.sum((mask & (hi-lo > 1)).to(tl.int32), axis=0) > 0:
        mid = (lo+hi)//2
        value = tl.load(cdf+mid, mask=mask, other=1.)
        right = value <= u
        old_width = hi-lo
        lo = tl.where(right & (old_width > 1), mid, lo)
        hi = tl.where(~right & (old_width > 1), mid, hi)
    a = tl.load(cdf+lo, mask=mask, other=0.)
    b = tl.load(cdf+hi, mask=mask, other=1.)
    xa = tl.load(x+lo, mask=mask, other=0.)
    xb = tl.load(x+hi, mask=mask, other=0.)
    fraction = (u-a)/tl.maximum(b-a, 1.e-30)
    p0 = tl.load(pdf+lo, mask=mask, other=-1.)
    p1 = tl.load(pdf+hi, mask=mask, other=-1.)
    root = tl.sqrt(tl.maximum(0., p0*p0+fraction*(p1*p1-p0*p0)))
    fraction = tl.where(p0 >= 0, fraction*(p0+p1)/tl.maximum(p0+root, 1.e-30), fraction)
    return xa+fraction*(xb-xa)


@triton.jit
def tangent_frame(nx, ny, nz):
    pole = tl.abs(nz) > .9
    tx = tl.where(pole, nz, -ny)
    ty = tl.where(pole, 0., nx)
    tz = tl.where(pole, -nx, 0.)
    inv = tl.rsqrt(tl.maximum(tx*tx+ty*ty+tz*tz, 1.e-30))
    tx, ty, tz = tx*inv, ty*inv, tz*inv
    qx, qy, qz = ny*tz-nz*ty, nz*tx-nx*tz, nx*ty-ny*tx
    return tx, ty, tz, qx, qy, qz


@triton.jit
def hemisphere(nx, ny, nz, u, phi_u, lambert: tl.constexpr):
    tx, ty, tz, qx, qy, qz = tangent_frame(nx, ny, nz)
    cosine = tl.sqrt(u) if lambert else u
    sine = tl.sqrt(tl.maximum(0., 1.-cosine*cosine))
    phi = 6.283185307179586*phi_u
    a, b = sine*tl.cos(phi), sine*tl.sin(phi)
    return cosine*nx+a*tx+b*qx, cosine*ny+a*ty+b*qy, cosine*nz+a*tz+b*qz


@triton.jit
def polarization(dx, dy, dz, u):
    tx, ty, tz, qx, qy, qz = tangent_frame(dx, dy, dz)
    a, b = tl.cos(6.283185307179586*u), tl.sin(6.283185307179586*u)
    return a*tx+b*qx, a*ty+b*qy, a*tz+b*qz


@triton.jit
def interaction_kernel(
    positions, directions, polarizations, wavelengths, times, flags, last_hit, channels, photon_ids,
    active_rows, hit_triangles, hit_distances, triangle_vertices, normals, material1, material2, surface_ids, channel_map,
    rindex, absorption, scattering, velocities, surface_present, surface_model,
    surface_detect, surface_absorb, surface_diffuse, surface_specular, surface_reemit, surface_cdf,
    time_offsets, time_x, time_cdf, time_pdf, reemit_side,
    nactive, seed, iteration, wl_start, wl_step, NW: tl.constexpr, BLOCK: tl.constexpr,
):
    lane = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = lane < nactive
    row = tl.load(active_rows+lane, mask=valid, other=0)
    tri = tl.load(hit_triangles+lane, mask=valid, other=-1)
    distance = tl.load(hit_distances+lane, mask=valid, other=float("inf"))
    hit = valid & (tri >= 0)
    tri = tl.maximum(tri, 0)
    old_flags = tl.load(flags+row, mask=valid, other=0).to(tl.uint32)
    out_flags = old_flags | tl.where(valid & ~hit, 1, 0).to(tl.uint32)
    ids = tl.load(photon_ids+row, mask=valid, other=0)
    wl = tl.load(wavelengths+row, mask=valid, other=wl_start)
    t = tl.load(times+row, mask=valid, other=0.)
    dx = tl.load(directions+row*3, mask=valid, other=0.)
    dy = tl.load(directions+row*3+1, mask=valid, other=0.)
    dz = tl.load(directions+row*3+2, mask=valid, other=1.)
    px = tl.load(polarizations+row*3, mask=valid, other=1.)
    py = tl.load(polarizations+row*3+1, mask=valid, other=0.)
    pz = tl.load(polarizations+row*3+2, mask=valid, other=0.)
    nx = tl.load(normals+tri*3, mask=hit, other=0.)
    ny = tl.load(normals+tri*3+1, mask=hit, other=0.)
    nz = tl.load(normals+tri*3+2, mask=hit, other=1.)
    outward = dx*nx+dy*ny+dz*nz > 0
    m1 = tl.load(material1+tri, mask=hit, other=0)
    m2 = tl.load(material2+tri, mask=hit, other=0)
    incident, other = tl.where(outward, m1, m2), tl.where(outward, m2, m1)
    sign = tl.where(outward, -1., 1.)
    inx, iny, inz = sign*nx, sign*ny, sign*nz
    alen = interpolate(absorption, incident, wl, wl_start, wl_step, NW, hit)
    slen = interpolate(scattering, incident, wl, wl_start, wl_step, NW, hit)
    da = -alen*tl.log(random_uniform(ids, seed, iteration*32))
    ds = -slen*tl.log(random_uniform(ids, seed, iteration*32+1))
    absorbed = hit & (da <= ds) & (da <= distance)
    scattered = hit & (ds < da) & (ds <= distance)
    boundary = hit & ~(absorbed | scattered)
    travel = tl.where(hit, tl.minimum(distance, tl.minimum(da, ds)), 0.)
    velocity = interpolate(velocities, incident, wl, wl_start, wl_step, NW, hit)
    t += tl.where(hit, travel/tl.maximum(velocity, 1.e-30), 0.)
    x = tl.load(positions+row*3, mask=valid, other=0.)+travel*dx
    y = tl.load(positions+row*3+1, mask=valid, other=0.)+travel*dy
    z = tl.load(positions+row*3+2, mask=valid, other=0.)+travel*dz
    out_flags |= tl.where(absorbed, 2, 0).to(tl.uint32)
    old_last = tl.load(last_hit+row, mask=valid, other=-1)
    new_last = tl.where(hit, tl.where(boundary, tri, -1), old_last)
    sdx, sdy, sdz, spx, spy, spz = rayleigh_scatter(dx, dy, dz, px, py, pz,
        random_uniform(ids, seed, iteration*32+2), random_uniform(ids, seed, iteration*32+3),
        random_uniform(ids, seed, iteration*32+4), random_uniform(ids, seed, iteration*32+5))
    dx, dy, dz = tl.where(scattered, sdx, dx), tl.where(scattered, sdy, dy), tl.where(scattered, sdz, dz)
    px, py, pz = tl.where(scattered, spx, px), tl.where(scattered, spy, py), tl.where(scattered, spz, pz)
    out_flags |= tl.where(scattered, 16, 0).to(tl.uint32)
    sid = tl.load(surface_ids+tri, mask=hit, other=-1)
    safe_sid = tl.maximum(sid, 0)
    present = tl.load(surface_present+safe_sid, mask=boundary & (sid >= 0), other=0) != 0
    surface = boundary & (sid >= 0) & present
    model = tl.load(surface_model+safe_sid, mask=surface, other=0)
    absorb = interpolate(surface_absorb, safe_sid, wl, wl_start, wl_step, NW, surface)
    diffuse = interpolate(surface_diffuse, safe_sid, wl, wl_start, wl_step, NW, surface)
    specular = interpolate(surface_specular, safe_sid, wl, wl_start, wl_step, NW, surface)
    detect = interpolate(surface_detect, safe_sid, wl, wl_start, wl_step, NW, surface & (model == 0))
    u = random_uniform(ids, seed, iteration*32+6)
    killed = surface & (u < absorb)
    detected = surface & (u >= absorb) & (u < absorb+detect)
    diff = surface & (u >= absorb+detect) & (u < absorb+detect+diffuse)
    spec = surface & (u >= absorb+detect+diffuse) & (u < absorb+detect+diffuse+specular)
    passed = boundary & ~(killed | detected | diff | spec)
    reemit_prob = interpolate(surface_reemit, safe_sid, wl, wl_start, wl_step, NW, surface & (model == 2))
    reemit = killed & (model == 2) & (random_uniform(ids, seed, iteration*32+7) < reemit_prob)
    new_wl = sample_regular_cdf(surface_cdf, safe_sid, random_uniform(ids, seed, iteration*32+8),
                               wl_start, wl_step, NW, reemit)
    dt = sample_time_cdf(time_offsets, time_x, time_cdf, time_pdf, safe_sid,
                         random_uniform(ids, seed, iteration*32+9), reemit)
    wl = tl.where(reemit, new_wl, wl)
    t += tl.where(reemit, dt, 0.)
    side_prob = tl.load(reemit_side+safe_sid, mask=reemit, other=.5)
    side = tl.where(random_uniform(ids, seed, iteration*32+10) < side_prob, -1., 1.)
    rdx, rdy, rdz = hemisphere(side*nx, side*ny, side*nz, random_uniform(ids, seed, iteration*32+11),
                              random_uniform(ids, seed, iteration*32+12), False)
    rpx, rpy, rpz = polarization(rdx, rdy, rdz, random_uniform(ids, seed, iteration*32+13))
    dx, dy, dz = tl.where(reemit, rdx, dx), tl.where(reemit, rdy, dy), tl.where(reemit, rdz, dz)
    px, py, pz = tl.where(reemit, rpx, px), tl.where(reemit, rpy, py), tl.where(reemit, rpz, pz)
    out_flags |= tl.where(reemit, 128, 0).to(tl.uint32)
    out_flags |= tl.where(killed & ~reemit, 8, 0).to(tl.uint32)
    channel = tl.load(channel_map+tri, mask=detected, other=-1)
    out_flags |= tl.where(detected, 4, 0).to(tl.uint32)
    tl.store(channels+row, channel, mask=detected)
    ddx, ddy, ddz = hemisphere(inx, iny, inz, random_uniform(ids, seed, iteration*32+14),
                              random_uniform(ids, seed, iteration*32+15), True)
    dpx, dpy, dpz = polarization(ddx, ddy, ddz, random_uniform(ids, seed, iteration*32+16))
    dx, dy, dz = tl.where(diff, ddx, dx), tl.where(diff, ddy, dy), tl.where(diff, ddz, dz)
    px, py, pz = tl.where(diff, dpx, px), tl.where(diff, dpy, py), tl.where(diff, dpz, pz)
    out_flags |= tl.where(diff, 32, 0).to(tl.uint32)
    dn, pn = dx*inx+dy*iny+dz*inz, px*inx+py*iny+pz*inz
    dx = tl.where(spec, dx-2*dn*inx, dx)
    dy = tl.where(spec, dy-2*dn*iny, dy)
    dz = tl.where(spec, dz-2*dn*inz, dz)
    px = tl.where(spec, px-2*pn*inx, px)
    py = tl.where(spec, py-2*pn*iny, py)
    pz = tl.where(spec, pz-2*pn*inz, pz)
    out_flags |= tl.where(spec, 64, 0).to(tl.uint32)
    n1 = interpolate(rindex, incident, wl, wl_start, wl_step, NW, passed)
    n2 = interpolate(rindex, other, wl, wl_start, wl_step, NW, passed)
    n1, n2 = tl.where(passed, n1, 1.), tl.where(passed, n2, 1.)
    fdx, fdy, fdz, fpx, fpy, fpz, reflected, _, _, _ = fresnel_step(
        dx, dy, dz, px, py, pz, inx, iny, inz, n1, n2,
        random_uniform(ids, seed, iteration*32+17), random_uniform(ids, seed, iteration*32+18))
    dx, dy, dz = tl.where(passed, fdx, dx), tl.where(passed, fdy, dy), tl.where(passed, fdz, dz)
    px, py, pz = tl.where(passed, fpx, px), tl.where(passed, fpy, py), tl.where(passed, fpz, pz)
    out_flags |= tl.where(passed, tl.where(reflected, 64, 256), 0).to(tl.uint32)
    continuing = boundary & ((out_flags & 15) == 0)
    x, y, z = offset_boundary_point(triangle_vertices, tri, x, y, z, dx, dy, dz, continuing)
    tl.store(positions+row*3, x, mask=valid)
    tl.store(positions+row*3+1, y, mask=valid)
    tl.store(positions+row*3+2, z, mask=valid)
    tl.store(directions+row*3, dx, mask=valid)
    tl.store(directions+row*3+1, dy, mask=valid)
    tl.store(directions+row*3+2, dz, mask=valid)
    tl.store(polarizations+row*3, px, mask=valid)
    tl.store(polarizations+row*3+1, py, mask=valid)
    tl.store(polarizations+row*3+2, pz, mask=valid)
    tl.store(wavelengths+row, wl, mask=valid)
    tl.store(times+row, t, mask=valid)
    tl.store(flags+row, out_flags, mask=valid)
    tl.store(last_hit+row, new_last, mask=valid)


class DeviceSpectralScene:
    def __init__(self, scene, *, device=None):
        if not torch.cuda.is_available():
            raise RuntimeError("Triton spectral transport requires a CUDA GPU; use backend='reference' for CPU validation")
        self.scene = scene
        self.device = torch.device("cuda" if device is None else device)
        self.bvh = scene.bvh.to_triton(self.device)
        self.arrays = []
        host = scene.host
        m, s = host.optics.materials, host.optics.surfaces
        arrays = (scene.normals, host.material1_index, host.material2_index, host.surface_index,
                  host.triangle_channel_index, m.refractive_index, m.absorption_length, m.scattering_length,
                  scene.velocities, s.present, s.model, s.detect, s.absorb, s.reflect_diffuse,
                  s.reflect_specular, s.reemit, s.reemission_cdf, scene.time_offsets,
                  scene.time_x, scene.time_cdf, scene.time_pdf, scene.reemit_to_material1)
        for a in arrays:
            value = np.array(a, copy=True)
            if not value.size:
                value = np.zeros(2, dtype=value.dtype)
            self.arrays.append(torch.from_numpy(value).to(self.device))

    def propagate(self, batch, *, seed, max_steps, timings=None):
        from .spectral import _state, _finish, STEP_LIMIT, TERMINAL
        if timings is not None:
            import time
            torch.cuda.synchronize(self.device)
            stage_start = time.perf_counter()
        state = {k: torch.from_numpy(v.astype(np.int64) if v.dtype == np.uint32 else v).to(self.device)
                 for k, v in _state(batch).items()}
        workspace = self.bvh.allocate_workspace(batch.photon_count)
        if timings is not None:
            torch.cuda.synchronize(self.device)
            uploaded = time.perf_counter()
        steps = 0
        grid = self.scene.host.optics.wavelength_grid
        for step in range(max_steps):
            rows = torch.nonzero((state["flags"] & TERMINAL) == 0).flatten().contiguous()
            if not rows.numel():
                break
            steps = step+1
            nearest = nearest_hit_local(self.bvh, state["pos"][rows], state["direction"][rows], high_precision=True,
                                         last_hit=state["last_hit"][rows], workspace=workspace,
                                         check_overflow=True)
            interaction_kernel[(triton.cdiv(rows.numel(), 128),)](
                state["pos"], state["direction"], state["polarization"], state["wavelengths"], state["times"],
                state["flags"], state["last_hit"], state["channels"], state["photon_ids"],
                rows, nearest.triangle_ids, nearest.distances, self.bvh.triangle_vertices, *self.arrays,
                rows.numel(), int(seed), step, float(grid.start), float(grid.step),
                NW=grid.count, BLOCK=128, enable_fp_fusion=False,
            )
        live = (state["flags"] & TERMINAL) == 0
        state["flags"][live] |= int(STEP_LIMIT)
        if timings is not None:
            torch.cuda.synchronize(self.device)
            propagated = time.perf_counter()
        host_state = {k: v.cpu().numpy() for k, v in state.items()}
        host_state["flags"] = host_state["flags"].astype(np.uint32)
        result = _finish(host_state, steps, self.scene.fingerprint)
        if timings is not None:
            timings.append({"photons":batch.photon_count,"steps":steps,
                            "upload_allocation_seconds":uploaded-stage_start,
                            "transport_seconds":propagated-uploaded,
                            "download_result_seconds":time.perf_counter()-propagated})
        return result
