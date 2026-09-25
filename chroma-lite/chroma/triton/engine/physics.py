"""One transport step for every queued photon (production physics, Triton).

The step follows the installed Chroma ``propagate`` loop body
(``chroma/cuda/photon.h``): sample absorption and scattering distances in the
incident material; absorb (with multi-component bulk re-emission), Rayleigh
scatter, or move to the boundary; at a boundary apply the surface model or,
on PASS, the Fresnel step. Weights follow ``use_weights``. Deliberate
production fixes (documented in ``docs/triton_dropin_design.md``): angle-free
Fresnel without NaNs at normal/critical incidence, specular reflection that
also reflects the polarization, and a few-ULP push off the surface after a
boundary interaction (Chroma relies only on skipping the last triangle).

Random numbers are Philox counters keyed by the global photon id and the
photon's step count, so the result does not depend on queue order or batch
size. Slots per step (see ``SLOT_*``) never overlap.
"""

import triton
import triton.language as tl
from triton.language.random import philox

from chroma.triton.physics_kernels import fresnel_step, rayleigh_scatter

SPEED_OF_LIGHT = tl.constexpr(299.792458)  # mm/ns, as in chroma/cuda/physics.h
WEIGHT_LOWER_THRESHOLD = tl.constexpr(0.0001)

# Photon history bits.
NO_HIT = tl.constexpr(1)
BULK_ABSORB = tl.constexpr(2)
SURFACE_DETECT = tl.constexpr(4)
SURFACE_ABSORB = tl.constexpr(8)
RAYLEIGH_SCATTER = tl.constexpr(16)
REFLECT_DIFFUSE = tl.constexpr(32)
REFLECT_SPECULAR = tl.constexpr(64)
SURFACE_REEMIT = tl.constexpr(128)
SURFACE_TRANSMIT = tl.constexpr(256)
BULK_REEMIT = tl.constexpr(512)
TERMINAL = tl.constexpr(1 | 2 | 4 | 8 | (1 << 31))

# Random-stream slots within one step (32 per step).
SLOT_ABSORB = tl.constexpr(0)
SLOT_SCATTER = tl.constexpr(1)
SLOT_RAYLEIGH = tl.constexpr(2)  # 4 slots
SLOT_SURFACE = tl.constexpr(6)
SLOT_SURFACE2 = tl.constexpr(7)
SLOT_WLS_WAVELENGTH = tl.constexpr(8)
SLOT_WLS_DIRECTION = tl.constexpr(9)  # 2 slots
SLOT_WLS_POLARIZATION = tl.constexpr(11)
SLOT_DIFFUSE = tl.constexpr(12)  # 2 slots
SLOT_DIFFUSE_POLARIZATION = tl.constexpr(14)
SLOT_FRESNEL = tl.constexpr(15)  # 2 slots
SLOT_BULK_COMPONENT = tl.constexpr(17)
SLOT_BULK_REEMIT = tl.constexpr(18)
SLOT_BULK_WAVELENGTH = tl.constexpr(19)
SLOT_BULK_TIME = tl.constexpr(20)
SLOT_BULK_DIRECTION = tl.constexpr(21)  # 2 slots
SLOT_BULK_POLARIZATION = tl.constexpr(23)
# 24-31 are reserved for further surface models.


@triton.jit
def rng(ids, seed, step, slot):
    """Uniform in (0,1) from Philox(seed; id_lo, id_hi, step*32+slot, 0)."""
    low = ids.to(tl.uint64).to(tl.uint32)
    high = (ids.to(tl.uint64) >> 32).to(tl.uint32)
    stream = (tl.zeros(ids.shape, tl.int32) + step * 32 + slot).to(tl.uint32)
    x, _, _, _ = philox(seed, low, high, stream, tl.zeros(ids.shape, tl.uint32))
    return ((x >> 9).to(tl.float32) + 0.5) * 1.1920928955078125e-7


@triton.jit
def interp_uniform(table, row, x, start, step, count: tl.constexpr, mask):
    """Chroma interp_property on a uniform grid (clamped at both ends)."""
    f = (x - start) / step
    jl = tl.minimum(tl.maximum(f.to(tl.int32), 0), count - 2)
    left = tl.load(table + row * count + jl, mask=mask, other=0.)
    right = tl.load(table + row * count + jl + 1, mask=mask, other=0.)
    value = left + (x - (start + jl.to(tl.float32) * step)) * (right - left) / step
    value = tl.where(f <= 0., tl.load(table + row * count, mask=mask, other=0.), value)
    last = tl.load(table + row * count + count - 1, mask=mask, other=0.)
    return tl.where(f >= count - 1., last, value)


@triton.jit
def sample_uniform_cdf(table, row, u, start, step, count: tl.constexpr, mask):
    """Chroma sample_cdf on a uniform grid: invert a tabulated CDF."""
    lo = tl.zeros(u.shape, tl.int32)
    hi = tl.full(u.shape, count - 1, tl.int32)
    while tl.sum((mask & (lo < hi - 1)).to(tl.int32), axis=0) > 0:
        half = (lo + hi) // 2
        value = tl.load(table + row * count + half, mask=mask, other=1.)
        go_low = u < value
        width = mask & (lo < hi - 1)
        hi = tl.where(width & go_low, half, hi)
        lo = tl.where(width & ~go_low, half, lo)
    left = tl.load(table + row * count + lo, mask=mask, other=0.)
    right = tl.load(table + row * count + hi, mask=mask, other=1.)
    delta = right - left
    return start + step * lo.to(tl.float32) + step * (u - left) / tl.where(delta != 0., delta, 1.)


@triton.jit
def uniform_sphere(u0, u1):
    z = 2.0 * u0 - 1.0
    r = tl.sqrt(tl.maximum(0., 1.0 - z * z))
    phi = 6.283185307179586 * u1
    return r * tl.cos(phi), r * tl.sin(phi), z


@triton.jit
def random_polarization(dx, dy, dz, u):
    """Uniform polarization transverse to ``d`` (unit)."""
    pole = tl.abs(dz) > 0.9
    tx = tl.where(pole, dz, -dy)
    ty = tl.where(pole, 0., dx)
    tz = tl.where(pole, -dx, 0.)
    inv = tl.rsqrt(tl.maximum(tx * tx + ty * ty + tz * tz, 1e-30))
    tx, ty, tz = tx * inv, ty * inv, tz * inv
    qx, qy, qz = dy * tz - dz * ty, dz * tx - dx * tz, dx * ty - dy * tx
    a = 6.283185307179586 * u
    ca, sa = tl.cos(a), tl.sin(a)
    return ca * tx + sa * qx, ca * ty + sa * qy, ca * tz + sa * qz


@triton.jit
def lambert(nx, ny, nz, u0, u1):
    """Cosine-weighted direction about unit ``n``."""
    pole = tl.abs(nz) > 0.9
    tx = tl.where(pole, nz, -ny)
    ty = tl.where(pole, 0., nx)
    tz = tl.where(pole, -nx, 0.)
    inv = tl.rsqrt(tl.maximum(tx * tx + ty * ty + tz * tz, 1e-30))
    tx, ty, tz = tx * inv, ty * inv, tz * inv
    qx, qy, qz = ny * tz - nz * ty, nz * tx - nx * tz, nx * ty - ny * tx
    c = tl.sqrt(u0)
    s = tl.sqrt(tl.maximum(0., 1.0 - u0))
    phi = 6.283185307179586 * u1
    a, b = s * tl.cos(phi), s * tl.sin(phi)
    return c * nx + a * tx + b * qx, c * ny + a * ty + b * qy, c * nz + a * tz + b * qz


@triton.jit
def append_rows(rows, mask, output, output_count):
    selected = mask.to(tl.int32)
    total = tl.sum(selected, axis=0)
    offset = tl.cumsum(selected, axis=0) - selected
    start = tl.atomic_add(output_count, total)
    tl.store(output + start + offset, rows, mask=mask)


@triton.jit
def nudge(x, n):
    """Move ``x`` a few float32 ULPs along the sign of ``n``."""
    return x + n * (tl.abs(x) + 1.0) * 4.8e-7


@triton.jit
def step_kernel(
    rows_ptr, count_ptr, capacity,
    pos_ptr, dir_ptr, pol_ptr, wl_ptr, t_ptr, last_ptr, flags_ptr, w_ptr, ids_ptr, steps_ptr,
    hit_t, hit_tri, hit_n, hit_codes,
    rindex, absorption, scattering,
    comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs,
    s_present, s_model, s_detect, s_absorb, s_reemit, s_diffuse, s_specular, s_cdf,
    next_rows, next_count,
    seed, max_steps, wl_start, wl_step, time_start, time_step,
    NW: tl.constexpr, NT: tl.constexpr, MAX_COMP: tl.constexpr,
    USE_WEIGHTS: tl.constexpr, BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(count_ptr)
    if tl.program_id(0) * BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    row = tl.load(rows_ptr + lane, mask=valid, other=0)
    ids = tl.load(ids_ptr + row, mask=valid, other=0)
    step = tl.load(steps_ptr + row, mask=valid, other=0)
    flags = tl.load(flags_ptr + row, mask=valid, other=0)
    x = tl.load(pos_ptr + row * 3 + 0, mask=valid, other=0.)
    y = tl.load(pos_ptr + row * 3 + 1, mask=valid, other=0.)
    z = tl.load(pos_ptr + row * 3 + 2, mask=valid, other=0.)
    dx = tl.load(dir_ptr + row * 3 + 0, mask=valid, other=1.)
    dy = tl.load(dir_ptr + row * 3 + 1, mask=valid, other=0.)
    dz = tl.load(dir_ptr + row * 3 + 2, mask=valid, other=0.)
    px = tl.load(pol_ptr + row * 3 + 0, mask=valid, other=0.)
    py = tl.load(pol_ptr + row * 3 + 1, mask=valid, other=1.)
    pz = tl.load(pol_ptr + row * 3 + 2, mask=valid, other=0.)
    wl = tl.load(wl_ptr + row, mask=valid, other=wl_start)
    t = tl.load(t_ptr + row, mask=valid, other=0.)
    last = tl.load(last_ptr + row, mask=valid, other=-1)
    weight = tl.load(w_ptr + row, mask=valid, other=1.)

    dist = tl.load(hit_t + lane, mask=valid, other=0.)
    tri = tl.load(hit_tri + lane, mask=valid, other=-1)
    nx = tl.load(hit_n + lane * 3 + 0, mask=valid, other=0.)
    ny = tl.load(hit_n + lane * 3 + 1, mask=valid, other=0.)
    nz = tl.load(hit_n + lane * 3 + 2, mask=valid, other=1.)
    m_in = tl.load(hit_codes + lane * 3 + 0, mask=valid, other=0)
    m_out = tl.load(hit_codes + lane * 3 + 1, mask=valid, other=0)
    sidx = tl.load(hit_codes + lane * 3 + 2, mask=valid, other=-1)

    hit = valid & (tri != -1)
    flags = tl.where(valid & ~hit, flags | NO_HIT, flags)
    last = tl.where(valid & ~hit, -1, last)

    # Orient: the normal faces the incoming photon; material1 is incident.
    outward = (dx * nx + dy * ny + dz * nz) > 0.
    inc = tl.where(outward, m_in, m_out)
    oth = tl.where(outward, m_out, m_in)
    sg = tl.where(outward, -1.0, 1.0)
    nx, ny, nz = sg * nx, sg * ny, sg * nz

    n1 = interp_uniform(rindex, inc, wl, wl_start, wl_step, NW, hit)
    n2 = interp_uniform(rindex, oth, wl, wl_start, wl_step, NW, hit)
    alen = interp_uniform(absorption, inc, wl, wl_start, wl_step, NW, hit)
    slen = interp_uniform(scattering, inc, wl, wl_start, wl_step, NW, hit)

    da = -alen * tl.log(rng(ids, seed, step, SLOT_ABSORB))
    ds = -slen * tl.log(rng(ids, seed, step, SLOT_SCATTER))
    if USE_WEIGHTS:
        weighted = hit & (weight > WEIGHT_LOWER_THRESHOLD)
    else:
        weighted = hit & False
    da = tl.where(weighted, 1e30, da)

    absorbed = hit & (da <= ds) & (da <= dist)
    scattered = hit & (ds < da) & (ds <= dist)
    boundary = hit & ~absorbed & ~scattered
    travel = tl.where(absorbed, da, tl.where(scattered, ds, tl.where(hit, dist, 0.)))
    t += travel * n1 / SPEED_OF_LIGHT
    x += travel * dx
    y += travel * dy
    z += travel * dz
    if USE_WEIGHTS:
        weight = tl.where(weighted & scattered, weight * tl.exp(-ds / alen), weight)
        weight = tl.where(weighted & boundary, weight * tl.exp(-dist / alen), weight)

    # ---- bulk absorption with multi-component re-emission ---------------
    c_begin = tl.load(comp_offsets + inc, mask=absorbed, other=0)
    c_end = tl.load(comp_offsets + inc + 1, mask=absorbed, other=0)
    ncomp = c_end - c_begin
    has_comp = absorbed & (ncomp > 0)
    u_comp = rng(ids, seed, step, SLOT_BULK_COMPONENT)
    chosen = c_begin
    found = tl.zeros(ids.shape, tl.int1)
    prob = tl.zeros(ids.shape, tl.float32)
    for c in tl.static_range(MAX_COMP):
        in_range = has_comp & (c < ncomp)
        cabs = interp_uniform(comp_abs, c_begin + c, wl, wl_start, wl_step, NW, in_range)
        prob += tl.where(in_range, alen / cabs, 0.)
        pick = in_range & ~found & ((u_comp < prob) | (c + 1 == ncomp))
        chosen = tl.where(pick, c_begin + c, chosen)
        found = found | pick
    reemit_p = interp_uniform(comp_prob, chosen, wl, wl_start, wl_step, NW, has_comp)
    bulk_reemit = has_comp & (rng(ids, seed, step, SLOT_BULK_REEMIT) < reemit_p)
    new_wl = sample_uniform_cdf(comp_wcdf, chosen, rng(ids, seed, step, SLOT_BULK_WAVELENGTH),
                                wl_start, wl_step, NW, bulk_reemit)
    dt = sample_uniform_cdf(comp_tcdf, chosen, rng(ids, seed, step, SLOT_BULK_TIME),
                            time_start, time_step, NT, bulk_reemit)
    rdx, rdy, rdz = uniform_sphere(rng(ids, seed, step, SLOT_BULK_DIRECTION),
                                   rng(ids, seed, step, SLOT_BULK_DIRECTION + 1))
    rpx, rpy, rpz = random_polarization(rdx, rdy, rdz, rng(ids, seed, step, SLOT_BULK_POLARIZATION))
    wl = tl.where(bulk_reemit, new_wl, wl)
    t = tl.where(bulk_reemit, t + dt, t)
    dx, dy, dz = tl.where(bulk_reemit, rdx, dx), tl.where(bulk_reemit, rdy, dy), tl.where(bulk_reemit, rdz, dz)
    px, py, pz = tl.where(bulk_reemit, rpx, px), tl.where(bulk_reemit, rpy, py), tl.where(bulk_reemit, rpz, pz)
    flags = tl.where(bulk_reemit, flags | BULK_REEMIT, flags)
    flags = tl.where(absorbed & ~bulk_reemit, flags | BULK_ABSORB, flags)

    # ---- Rayleigh scattering --------------------------------------------
    sdx, sdy, sdz, spx, spy, spz = rayleigh_scatter(
        dx, dy, dz, px, py, pz,
        rng(ids, seed, step, SLOT_RAYLEIGH), rng(ids, seed, step, SLOT_RAYLEIGH + 1),
        rng(ids, seed, step, SLOT_RAYLEIGH + 2), rng(ids, seed, step, SLOT_RAYLEIGH + 3))
    dx, dy, dz = tl.where(scattered, sdx, dx), tl.where(scattered, sdy, dy), tl.where(scattered, sdz, dz)
    px, py, pz = tl.where(scattered, spx, px), tl.where(scattered, spy, py), tl.where(scattered, spz, pz)
    flags = tl.where(scattered, flags | RAYLEIGH_SCATTER, flags)
    last = tl.where(absorbed | scattered, -1, last)

    # ---- boundary -----------------------------------------------------------
    last = tl.where(boundary, tri, last)
    safe_s = tl.maximum(sidx, 0)
    present = tl.load(s_present + safe_s, mask=boundary & (sidx >= 0), other=0) != 0
    surf = boundary & (sidx >= 0) & present
    model = tl.load(s_model + safe_s, mask=surf, other=0)
    absorb = interp_uniform(s_absorb, safe_s, wl, wl_start, wl_step, NW, surf)
    detect = interp_uniform(s_detect, safe_s, wl, wl_start, wl_step, NW, surf & (model == 0))
    diffuse = interp_uniform(s_diffuse, safe_s, wl, wl_start, wl_step, NW, surf)
    specular = interp_uniform(s_specular, safe_s, wl, wl_start, wl_step, NW, surf)
    reemit = interp_uniform(s_reemit, safe_s, wl, wl_start, wl_step, NW, surf & (model == 2))
    u = rng(ids, seed, step, SLOT_SURFACE)
    if USE_WEIGHTS:
        reweight = surf & (weight > WEIGHT_LOWER_THRESHOLD) & (absorb < 1.0 - WEIGHT_LOWER_THRESHOLD)
        survive = tl.where(reweight, 1.0 - absorb, 1.0)
        weight = weight * survive
        detect = detect / survive
        diffuse = diffuse / survive
        specular = specular / survive
        absorb = tl.where(reweight, 0., absorb)
    default = surf & (model == 0)
    wls = surf & (model == 2)

    # Default model: absorb, detect, diffuse, specular, else PASS.
    if USE_WEIGHTS:
        forced_detect = default & (detect > 0.)
    else:
        forced_detect = default & False
    d_absorb = default & ~forced_detect & (u < absorb)
    d_detect = default & (forced_detect | (~d_absorb & (u < absorb + detect)))
    d_diffuse = default & ~d_absorb & ~d_detect & (u < absorb + detect + diffuse)
    d_specular = default & ~d_absorb & ~d_detect & ~d_diffuse & (u < absorb + detect + diffuse + specular)
    weight = tl.where(forced_detect, weight * detect, weight)

    # WLS model: absorb (re-emit or absorb), reflect (specular vs diffuse), else transmit.
    w_absorbed = wls & (u < absorb)
    w_reemit = w_absorbed & (rng(ids, seed, step, SLOT_SURFACE2) < reemit)
    w_reflect = wls & ~w_absorbed & (u < absorb + specular + diffuse)
    choose = rng(ids, seed, step, SLOT_SURFACE2) * (specular + diffuse)
    w_specular = w_reflect & (choose < specular)
    w_diffuse = w_reflect & ~w_specular
    w_transmit = wls & ~w_absorbed & ~w_reflect

    flags = tl.where(d_absorb | (w_absorbed & ~w_reemit), flags | SURFACE_ABSORB, flags)
    flags = tl.where(d_detect, flags | SURFACE_DETECT, flags)
    flags = tl.where(w_transmit, flags | SURFACE_TRANSMIT, flags)

    # WLS re-emission: new wavelength, isotropic direction, random polarization.
    new_wl = sample_uniform_cdf(s_cdf, safe_s, rng(ids, seed, step, SLOT_WLS_WAVELENGTH),
                                wl_start, wl_step, NW, w_reemit)
    edx, edy, edz = uniform_sphere(rng(ids, seed, step, SLOT_WLS_DIRECTION),
                                   rng(ids, seed, step, SLOT_WLS_DIRECTION + 1))
    epx, epy, epz = random_polarization(edx, edy, edz, rng(ids, seed, step, SLOT_WLS_POLARIZATION))
    wl = tl.where(w_reemit, new_wl, wl)
    dx, dy, dz = tl.where(w_reemit, edx, dx), tl.where(w_reemit, edy, dy), tl.where(w_reemit, edz, dz)
    px, py, pz = tl.where(w_reemit, epx, px), tl.where(w_reemit, epy, py), tl.where(w_reemit, epz, pz)
    flags = tl.where(w_reemit, flags | SURFACE_REEMIT, flags)

    # Diffuse reflection (Lambertian about the incident-side normal).
    diff = d_diffuse | w_diffuse
    ldx, ldy, ldz = lambert(nx, ny, nz, rng(ids, seed, step, SLOT_DIFFUSE), rng(ids, seed, step, SLOT_DIFFUSE + 1))
    lpx, lpy, lpz = random_polarization(ldx, ldy, ldz, rng(ids, seed, step, SLOT_DIFFUSE_POLARIZATION))
    dx, dy, dz = tl.where(diff, ldx, dx), tl.where(diff, ldy, dy), tl.where(diff, ldz, dz)
    px, py, pz = tl.where(diff, lpx, px), tl.where(diff, lpy, py), tl.where(diff, lpz, pz)
    flags = tl.where(diff, flags | REFLECT_DIFFUSE, flags)

    # Specular reflection (direction and polarization reflected).
    spec = d_specular | w_specular
    dn = dx * nx + dy * ny + dz * nz
    pn = px * nx + py * ny + pz * nz
    dx = tl.where(spec, dx - 2.0 * dn * nx, dx)
    dy = tl.where(spec, dy - 2.0 * dn * ny, dy)
    dz = tl.where(spec, dz - 2.0 * dn * nz, dz)
    px = tl.where(spec, px - 2.0 * pn * nx, px)
    py = tl.where(spec, py - 2.0 * pn * ny, py)
    pz = tl.where(spec, pz - 2.0 * pn * nz, pz)
    flags = tl.where(spec, flags | REFLECT_SPECULAR, flags)

    # PASS: no surface, default fall-through, or WLS transmission -> Fresnel.
    passed = boundary & (~surf | (default & ~d_absorb & ~d_detect & ~d_diffuse & ~d_specular) | w_transmit)
    fdx, fdy, fdz, fpx, fpy, fpz, reflected, _, _, _ = fresnel_step(
        dx, dy, dz, px, py, pz, nx, ny, nz, tl.where(passed, n1, 1.), tl.where(passed, n2, 1.),
        rng(ids, seed, step, SLOT_FRESNEL), rng(ids, seed, step, SLOT_FRESNEL + 1))
    dx, dy, dz = tl.where(passed, fdx, dx), tl.where(passed, fdy, dy), tl.where(passed, fdz, dz)
    px, py, pz = tl.where(passed, fpx, px), tl.where(passed, fpy, py), tl.where(passed, fpz, pz)
    flags = tl.where(passed & reflected, flags | REFLECT_SPECULAR, flags)

    # Push a surviving photon off the surface towards its outgoing side
    # (mesh boundaries only; wires rely on Chroma's 1e-4 mm minimum root).
    alive = valid & ((flags & TERMINAL) == 0)
    side = tl.where((dx * nx + dy * ny + dz * nz) >= 0., 1.0, -1.0)
    push = alive & boundary & (tri >= 0)
    x = tl.where(push, nudge(x, side * nx), x)
    y = tl.where(push, nudge(y, side * ny), y)
    z = tl.where(push, nudge(z, side * nz), z)

    step = step + valid.to(tl.int32)
    tl.store(pos_ptr + row * 3 + 0, x, mask=valid)
    tl.store(pos_ptr + row * 3 + 1, y, mask=valid)
    tl.store(pos_ptr + row * 3 + 2, z, mask=valid)
    tl.store(dir_ptr + row * 3 + 0, dx, mask=valid)
    tl.store(dir_ptr + row * 3 + 1, dy, mask=valid)
    tl.store(dir_ptr + row * 3 + 2, dz, mask=valid)
    tl.store(pol_ptr + row * 3 + 0, px, mask=valid)
    tl.store(pol_ptr + row * 3 + 1, py, mask=valid)
    tl.store(pol_ptr + row * 3 + 2, pz, mask=valid)
    tl.store(wl_ptr + row, wl, mask=valid)
    tl.store(t_ptr + row, t, mask=valid)
    tl.store(last_ptr + row, last, mask=valid)
    tl.store(flags_ptr + row, flags, mask=valid)
    tl.store(w_ptr + row, weight, mask=valid)
    tl.store(steps_ptr + row, step, mask=valid)
    append_rows(row, alive & (step < max_steps), next_rows, next_count)
