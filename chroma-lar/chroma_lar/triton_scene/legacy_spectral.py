"""Diagnostic spectral replay of native Chroma draws in Triton.

This deliberately retains historical numerical behavior. It is separate from
the corrected production engine and fails closed on unsupported optical models.
The capture archive supplies geometry and inputs, never expected output states
to the execution path. Raw output and draw ledgers are compared by the harness.
"""

from __future__ import annotations

import hashlib
import numpy as np
import torch
import triton
import triton.language as tl

from chroma.triton.physics_kernels import (
    advance_position_chroma_fast,
    advance_time_chroma_fast,
    exponential_distance_chroma_fast,
    fresnel_step_chroma,
    reflect_specular_chroma,
)
from chroma_lar import triton_backend as _legacy
from .chroma_global_traversal import ChromaGlobalBVHDevice, nearest_chroma_global_hit

# Reuse the already certified scalar legacy operations. These will move into
# the shared math module once the extended spectral certificate is established.
_legacy._load_boundary_kernel()
legacy_rayleigh = _legacy.boundary_legacy_rayleigh
legacy_sphere = _legacy.boundary_legacy_uniform_sphere
legacy_orient = _legacy.boundary_legacy_orient_diffuse
legacy_polarization = _legacy.boundary_legacy_diffuse_polarization


@triton.jit
def _float(words, address, mask, other: tl.constexpr = 0.0):
    return tl.load(words.to(tl.pointer_type(tl.float32)) + address, mask, other=other)


@triton.jit
def _store_float(words, address, value, mask):
    tl.store(words.to(tl.pointer_type(tl.float32)) + address, value, mask)


@triton.jit
def _normalize_vector(x, y, z):
    return tl.inline_asm_elementwise(
        """{
        .reg .f32 norm2, length;
        mul.ftz.f32 norm2, $4, $4;
        fma.rn.ftz.f32 norm2, $3, $3, norm2;
        fma.rn.ftz.f32 norm2, $5, $5, norm2;
        sqrt.approx.ftz.f32 length, norm2;
        div.approx.ftz.f32 $0, $3, length;
        div.approx.ftz.f32 $1, $4, length;
        div.approx.ftz.f32 $2, $5, length;
        }""",
        constraints="=f,=f,=f,f,f,f",
        args=[x, y, z],
        dtype=(tl.float32, tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _initialize(words, count, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = row < count
    flags = tl.load(words + row * 15 + 11, valid, other=0)
    # The original returns before committing initially terminal photon states.
    live = valid & ((flags & 32783) == 0)
    for begin in tl.static_range(3, 9, 3):
        x, y, z = (
            _float(words, row * 15 + begin, live),
            _float(words, row * 15 + begin + 1, live),
            _float(words, row * 15 + begin + 2, live),
        )
        x, y, z = _normalize_vector(x, y, z)
        _store_float(words, row * 15 + begin, x, live)
        _store_float(words, row * 15 + begin + 1, y, live)
        _store_float(words, row * 15 + begin + 2, z, live)
    tl.store(words + row * 15 + 11, flags & 65535, live)


@triton.jit
def _draw(tape, row, iteration, cursor, mask, overflow, STEPS: tl.constexpr, DRAWS: tl.constexpr):
    if tape.dtype.element_ty == tl.int32:
        d = tl.load(tape + row * 6, mask, other=0).to(tl.uint32)
        a = tl.load(tape + row * 6 + 1, mask, other=0).to(tl.uint32)
        b = tl.load(tape + row * 6 + 2, mask, other=0).to(tl.uint32)
        c = tl.load(tape + row * 6 + 3, mask, other=0).to(tl.uint32)
        e = tl.load(tape + row * 6 + 4, mask, other=0).to(tl.uint32)
        f = tl.load(tape + row * 6 + 5, mask, other=0).to(tl.uint32)
        t = a ^ (a >> 2)
        fnew = (f ^ (f << 4)) ^ (t ^ (t << 1))
        d += 362437
        tl.store(tape + row * 6, d, mask)
        tl.store(tape + row * 6 + 1, b, mask)
        tl.store(tape + row * 6 + 2, c, mask)
        tl.store(tape + row * 6 + 3, e, mask)
        tl.store(tape + row * 6 + 4, f, mask)
        tl.store(tape + row * 6 + 5, fnew, mask)
        bits = (fnew + d).to(tl.uint32)
        value = tl.inline_asm_elementwise(
            "{ .reg .f32 x; cvt.rn.f32.u32 x, $1; fma.rn.ftz.f32 $0, x, 0f2F800000, 0f2F000000; }",
            constraints="=f,r",
            args=[bits],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        return value, cursor + mask.to(tl.int32)
    else:
        available = (cursor < DRAWS) & (iteration < STEPS)
        value = tl.load(
            tape + (row.to(tl.int64) * STEPS + iteration) * DRAWS + cursor,
            mask=mask & available,
            other=float("nan"),
        )
        failed = mask & (~available | (value != value))
        tl.store(overflow + row, 1, mask=failed)
        return value, cursor + mask.to(tl.int32)


@triton.jit
def _property(table, row, wavelength, start, step, NW: tl.constexpr, mask):
    # Literal interpolation order from geometry.h, with opaque operations to
    # prevent reassociation. Endpoint padding is not treated as valid physics.
    fraction = tl.div_rn(wavelength - start, step)
    lower = fraction.to(tl.int32)
    below = wavelength < start
    above = wavelength > start + (NW - 1) * step
    lower = tl.minimum(tl.maximum(lower, 0), NW - 2)
    left = tl.load(table + row * NW + lower, mask=mask, other=0.0)
    right = tl.load(table + row * NW + lower + 1, mask=mask, other=0.0)
    value = tl.inline_asm_elementwise(
        """{
        .reg .f32 xlo, offset, delta, product, quotient;
        fma.rn.ftz.f32 xlo, $3, $5, $4;
        sub.ftz.f32 offset, $2, xlo;
        sub.ftz.f32 delta, $6, $1;
        mul.ftz.f32 product, offset, delta;
        div.approx.ftz.f32 quotient, product, $5;
        add.ftz.f32 $0, $1, quotient;
        }""",
        constraints="=f,f,f,f,f,f,f",
        args=[left, wavelength, lower.to(tl.float32), start, step, right],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    first = tl.load(table + row * NW, mask=mask & below, other=0.0)
    last = tl.load(table + row * NW + NW - 1, mask=mask & above, other=0.0)
    return tl.where(below, first, tl.where(above, last, value))


@triton.jit
def _cdf(table, row, u, start, step, NW: tl.constexpr, mask):
    lo = tl.full(u.shape, 0, tl.int32)
    hi = tl.full(u.shape, NW - 1, tl.int32)
    while tl.sum((mask & (lo < hi - 1)).to(tl.int32), 0) > 0:
        active = mask & (lo < hi - 1)
        mid = (lo + hi) // 2
        y = tl.load(table + row * NW + mid, mask=active, other=1.0)
        hi = tl.where(active & (u < y), mid, hi)
        lo = tl.where(active & ~(u < y), mid, lo)
    a = tl.load(table + row * NW + lo, mask=mask, other=0.0)
    b = tl.load(table + row * NW + hi, mask=mask, other=1.0)
    return tl.inline_asm_elementwise(
        """{
        .reg .f32 origin, dy, du, product, quotient;
        fma.rn.ftz.f32 origin, $4, $5, $6;
        sub.ftz.f32 dy, $3, $2;
        sub.ftz.f32 du, $1, $2;
        mul.ftz.f32 product, $5, du;
        div.approx.ftz.f32 quotient, product, dy;
        add.ftz.f32 $0, origin, quotient;
        }""",
        constraints="=f,f,f,f,f,f,f",
        args=[u, a, b, lo.to(tl.float32), step, start],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _step(
    words,
    rows,
    nactive,
    triangles,
    distances,
    normals,
    mat_from,
    mat_to,
    surfaces,
    rindex,
    absorption,
    scattering,
    models,
    detect_table,
    absorb_table,
    diffuse_table,
    specular_table,
    reemit_table,
    cdf_table,
    component_offsets,
    component_absorption,
    component_prob,
    component_wavelength_cdf,
    component_time_cdf,
    tape,
    ledger,
    draw_ledger,
    interactions,
    overflow,
    iteration,
    start,
    step,
    time_start,
    time_step,
    NW: tl.constexpr,
    NT: tl.constexpr,
    HAS_COMPONENTS: tl.constexpr,
    RECORD_HISTORY: tl.constexpr,
    STEPS: tl.constexpr,
    DRAWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < nactive
    row = tl.load(rows + lane, valid, other=0)
    offset = row * 15
    x, y, z = (
        _float(words, offset, valid),
        _float(words, offset + 1, valid),
        _float(words, offset + 2, valid),
    )
    dx, dy, dz = (
        _float(words, offset + 3, valid),
        _float(words, offset + 4, valid),
        _float(words, offset + 5, valid),
    )
    px, py, pz = (
        _float(words, offset + 6, valid),
        _float(words, offset + 7, valid),
        _float(words, offset + 8, valid),
    )
    wl, time = _float(words, offset + 9, valid), _float(words, offset + 10, valid)
    history = tl.load(words + offset + 11, valid, other=0)
    product = dx * dy * dz * x * y * z
    nan_abort = valid & (product != product)
    tri = tl.load(triangles + lane, valid, other=-1)
    hit = valid & (tri != -1) & ~nan_abort
    old_last = tl.load(words + offset + 12, valid, other=-1)
    last = tl.where(nan_abort, old_last, tri)
    history |= tl.where(nan_abort, 32769, tl.where(valid & ~hit, 1, 0))
    distance = tl.load(distances + lane, hit, other=0.0)
    nx = tl.load(normals + lane * 3, hit, other=0.0)
    ny = tl.load(normals + lane * 3 + 1, hit, other=0.0)
    nz = tl.load(normals + lane * 3 + 2, hit, other=1.0)
    m1, m2 = tl.load(mat_from + lane, hit, other=0), tl.load(mat_to + lane, hit, other=0)
    n1 = _property(rindex, m1, wl, start, step, NW, hit)
    n2 = _property(rindex, m2, wl, start, step, NW, hit)
    alen = _property(absorption, m1, wl, start, step, NW, hit)
    slen = _property(scattering, m1, wl, start, step, NW, hit)
    cursor = tl.full((BLOCK,), 0, tl.int32)
    ua, cursor = _draw(tape, row, iteration, cursor, hit, overflow, STEPS, DRAWS)
    us, cursor = _draw(tape, row, iteration, cursor, hit, overflow, STEPS, DRAWS)
    da = exponential_distance_chroma_fast(alen, ua)
    ds = exponential_distance_chroma_fast(slen, us)
    absorbed = hit & (da <= ds) & (da <= distance)
    scattered = hit & (ds < da) & (ds <= distance)
    boundary = hit & ~(absorbed | scattered)
    travel = tl.where(absorbed, da, tl.where(scattered, ds, distance))
    x = tl.where(hit, advance_position_chroma_fast(x, travel, dx), x)
    y = tl.where(hit, advance_position_chroma_fast(y, travel, dy), y)
    z = tl.where(hit, advance_position_chroma_fast(z, travel, dz), z)
    time = tl.where(hit, advance_time_chroma_fast(time, travel, n1), time)
    last = tl.where(absorbed | scattered, -1, last)
    bulk_reemit = tl.full((BLOCK,), False, tl.int1)
    if HAS_COMPONENTS:
        first = tl.load(component_offsets + m1, absorbed, other=0)
        end = tl.load(component_offsets + m1 + 1, absorbed, other=0)
        component = first
        has_component = absorbed & (end > first)
        ucomp, cursor = _draw(tape, row, iteration, cursor, has_component, overflow, STEPS, DRAWS)
        probability = tl.full((BLOCK,), 0.0, tl.float32)
        choosing = has_component
        while tl.sum(choosing.to(tl.int32), 0) > 0:
            length = _property(component_absorption, component, wl, start, step, NW, choosing)
            probability += tl.where(choosing, alen / length, 0.0)
            choosing &= ~(ucomp < probability) & (component + 1 < end)
            component += choosing.to(tl.int32)
        ur, cursor = _draw(tape, row, iteration, cursor, has_component, overflow, STEPS, DRAWS)
        reemit_prob = _property(component_prob, component, wl, start, step, NW, has_component)
        bulk_reemit = has_component & (ur < reemit_prob)
        uw, cursor = _draw(tape, row, iteration, cursor, bulk_reemit, overflow, STEPS, DRAWS)
        wl = tl.where(
            bulk_reemit,
            _cdf(component_wavelength_cdf, component, uw, start, step, NW, bulk_reemit),
            wl,
        )
        ut, cursor = _draw(tape, row, iteration, cursor, bulk_reemit, overflow, STEPS, DRAWS)
        dt = _cdf(component_time_cdf, component, ut, time_start, time_step, NT, bulk_reemit)
        time = tl.where(bulk_reemit, time + dt, time)
        u0, cursor = _draw(tape, row, iteration, cursor, bulk_reemit, overflow, STEPS, DRAWS)
        u1, cursor = _draw(tape, row, iteration, cursor, bulk_reemit, overflow, STEPS, DRAWS)
        bx, by, bz = legacy_sphere(u0, u1)
        u0, cursor = _draw(tape, row, iteration, cursor, bulk_reemit, overflow, STEPS, DRAWS)
        u1, cursor = _draw(tape, row, iteration, cursor, bulk_reemit, overflow, STEPS, DRAWS)
        bpx, bpy, bpz = legacy_sphere(u0, u1)
        bpx, bpy, bpz = legacy_polarization(bpx, bpy, bpz, bx, by, bz)
        dx, dy, dz = (
            tl.where(bulk_reemit, bx, dx),
            tl.where(bulk_reemit, by, dy),
            tl.where(bulk_reemit, bz, dz),
        )
        px, py, pz = (
            tl.where(bulk_reemit, bpx, px),
            tl.where(bulk_reemit, bpy, py),
            tl.where(bulk_reemit, bpz, pz),
        )
    history |= tl.where(bulk_reemit, 512, tl.where(absorbed, 2, tl.where(scattered, 16, 0)))
    u0, cursor = _draw(tape, row, iteration, cursor, scattered, overflow, STEPS, DRAWS)
    u1, cursor = _draw(tape, row, iteration, cursor, scattered, overflow, STEPS, DRAWS)
    sx, sy, sz, spx, spy, spz = legacy_rayleigh(px, py, pz, u0, u1)
    dx, dy, dz = (
        tl.where(scattered, sx, dx),
        tl.where(scattered, sy, dy),
        tl.where(scattered, sz, dz),
    )
    px, py, pz = (
        tl.where(scattered, spx, px),
        tl.where(scattered, spy, py),
        tl.where(scattered, spz, pz),
    )
    sid = tl.load(surfaces + lane, boundary, other=-1)
    surface = boundary & (sid >= 0)
    model = tl.load(models + sid, surface, other=-1)
    default = surface & (model == 0)
    wls = surface & (model == 2)
    absorb = _property(absorb_table, sid, wl, start, step, NW, surface)
    detect = _property(detect_table, sid, wl, start, step, NW, default)
    diffuse = _property(diffuse_table, sid, wl, start, step, NW, surface)
    specular = _property(specular_table, sid, wl, start, step, NW, surface)
    reemit_prob = _property(reemit_table, sid, wl, start, step, NW, wls)
    u, cursor = _draw(tape, row, iteration, cursor, surface, overflow, STEPS, DRAWS)
    killed = surface & (u < absorb)
    detected = default & ~killed & (u < absorb + detect)
    diff = default & ~killed & ~detected & (u < absorb + detect + diffuse)
    spec = default & ~killed & ~detected & ~diff & (u < absorb + detect + diffuse + specular)
    wls_reflect = wls & ~killed & (u < absorb + specular + diffuse)
    ur, cursor = _draw(tape, row, iteration, cursor, wls_reflect, overflow, STEPS, DRAWS)
    spec |= wls_reflect & (ur * (specular + diffuse) < specular)
    diff |= wls_reflect & ~(ur * (specular + diffuse) < specular)
    ur, cursor = _draw(tape, row, iteration, cursor, wls & killed, overflow, STEPS, DRAWS)
    reemit = wls & killed & (ur < reemit_prob)
    uc, cursor = _draw(tape, row, iteration, cursor, reemit, overflow, STEPS, DRAWS)
    wl = tl.where(reemit, _cdf(cdf_table, sid, uc, start, step, NW, reemit), wl)
    u0, cursor = _draw(tape, row, iteration, cursor, reemit, overflow, STEPS, DRAWS)
    u1, cursor = _draw(tape, row, iteration, cursor, reemit, overflow, STEPS, DRAWS)
    rx, ry, rz = legacy_sphere(u0, u1)
    u0, cursor = _draw(tape, row, iteration, cursor, reemit, overflow, STEPS, DRAWS)
    u1, cursor = _draw(tape, row, iteration, cursor, reemit, overflow, STEPS, DRAWS)
    rpx, rpy, rpz = legacy_sphere(u0, u1)
    rpx, rpy, rpz = legacy_polarization(rpx, rpy, rpz, rx, ry, rz)
    dx, dy, dz = tl.where(reemit, rx, dx), tl.where(reemit, ry, dy), tl.where(reemit, rz, dz)
    px, py, pz = tl.where(reemit, rpx, px), tl.where(reemit, rpy, py), tl.where(reemit, rpz, pz)
    history |= tl.where(reemit, 128, tl.where(killed, 8, tl.where(detected, 4, 0)))
    pending = diff
    while tl.sum(pending.to(tl.int32), 0) > 0:
        u0, cursor = _draw(tape, row, iteration, cursor, pending, overflow, STEPS, DRAWS)
        u1, cursor = _draw(tape, row, iteration, cursor, pending, overflow, STEPS, DRAWS)
        cx, cy, cz = legacy_sphere(u0, u1)
        cx, cy, cz, cosine = legacy_orient(cx, cy, cz, nx, ny, nz)
        accept, cursor = _draw(tape, row, iteration, cursor, pending, overflow, STEPS, DRAWS)
        dx, dy, dz = tl.where(pending, cx, dx), tl.where(pending, cy, dy), tl.where(pending, cz, dz)
        pending &= ~(accept < cosine) & (cursor < DRAWS)
    u0, cursor = _draw(tape, row, iteration, cursor, diff, overflow, STEPS, DRAWS)
    u1, cursor = _draw(tape, row, iteration, cursor, diff, overflow, STEPS, DRAWS)
    dpx, dpy, dpz = legacy_sphere(u0, u1)
    dpx, dpy, dpz = legacy_polarization(dpx, dpy, dpz, dx, dy, dz)
    px, py, pz = tl.where(diff, dpx, px), tl.where(diff, dpy, py), tl.where(diff, dpz, pz)
    sx, sy, sz = reflect_specular_chroma(dx, dy, dz, nx, ny, nz)
    dx, dy, dz = tl.where(spec, sx, dx), tl.where(spec, sy, dy), tl.where(spec, sz, dz)
    history |= tl.where(diff, 32, tl.where(spec, 64, 0))
    passed = boundary & ~(killed | detected | diff | spec)
    history |= tl.where(wls & passed, 256, 0)
    u0, cursor = _draw(tape, row, iteration, cursor, passed, overflow, STEPS, DRAWS)
    u1, cursor = _draw(tape, row, iteration, cursor, passed, overflow, STEPS, DRAWS)
    fx, fy, fz, fpx, fpy, fpz, reflected, _, _, _ = fresnel_step_chroma(
        dx, dy, dz, px, py, pz, nx, ny, nz, n1, n2, u0, u1
    )
    dx, dy, dz = tl.where(passed, fx, dx), tl.where(passed, fy, dy), tl.where(passed, fz, dz)
    px, py, pz = tl.where(passed, fpx, px), tl.where(passed, fpy, py), tl.where(passed, fpz, pz)
    history |= tl.where(passed & reflected, 64, 0)
    _store_float(words, offset, x, valid)
    _store_float(words, offset + 1, y, valid)
    _store_float(words, offset + 2, z, valid)
    _store_float(words, offset + 3, dx, valid)
    _store_float(words, offset + 4, dy, valid)
    _store_float(words, offset + 5, dz, valid)
    _store_float(words, offset + 6, px, valid)
    _store_float(words, offset + 7, py, valid)
    _store_float(words, offset + 8, pz, valid)
    _store_float(words, offset + 9, wl, valid)
    _store_float(words, offset + 10, time, valid)
    tl.store(words + offset + 11, history, valid)
    tl.store(words + offset + 12, last, valid)
    if RECORD_HISTORY:
        record = (row.to(tl.int64) * STEPS + iteration) * 15
        for field in tl.static_range(15):
            value = tl.load(words + offset + field, valid, other=0)
            tl.store(ledger + record + field, value, valid)
        tl.store(draw_ledger + row.to(tl.int64) * STEPS + iteration, cursor, valid)
    else:
        used = tl.load(draw_ledger + row, valid, other=0)
        tl.store(draw_ledger + row, used + cursor, valid)
    tl.store(interactions + row, iteration + 1, valid)


def propagate_legacy(
    data,
    source_words,
    *,
    max_steps,
    seed=None,
    rng_words=None,
    random_tape=None,
    normalize=True,
    record_history=True,
):
    """Propagate supplied photon words using historical Chroma numerics.

    ``data`` contains immutable scene arrays; it needs no reference output.
    Exactly one random source is required: an independently initialized seed,
    explicit six-word XORWOW worker states, or an audited float32 random tape.
    Callers using an original compacted queue must retain its worker assignment.
    Every committed interaction is returned for a raw-word audit. This strict
    compatibility executor is separate from the corrected production engine.
    """
    if sum(value is not None for value in (seed, rng_words, random_tape)) != 1:
        raise ValueError("provide exactly one of seed, rng_words, or random_tape")
    initial = np.asarray(source_words)
    if initial.dtype != np.uint32 or initial.ndim != 2 or initial.shape[1] != 15:
        raise ValueError("source_words must be a uint32 [N,15] photon array")
    count, steps = len(initial), int(max_steps)
    if count < 1 or steps < 1 or steps != max_steps:
        raise ValueError("positive photon count and integer max_steps required")
    if np.any(~np.isin(data["surface_models"], [-1, 0, 2])):
        raise NotImplementedError("legacy replay currently supports default/WLS surfaces")
    if np.any(initial[:, 13] != np.float32(1).view(np.uint32)):
        raise NotImplementedError("weighted legacy replay is not yet implemented")
    grid = data["wavelength_grid"]
    if not np.all(np.diff(grid) == grid[1] - grid[0]):
        raise ValueError("the original optical grid must be uniform")

    def upload(x):
        x = np.array(x, copy=True)
        if x.dtype == np.uint32:
            x = x.view(np.int32)
        return torch.from_numpy(x).cuda()

    nodes, vertices, triangles = [upload(data[k]) for k in ("nodes", "vertices", "triangles")]
    fingerprint = hashlib.sha256(np.ascontiguousarray(data["nodes"]).tobytes()).hexdigest()
    accelerator = ChromaGlobalBVHDevice(
        nodes,
        vertices,
        triangles,
        upload(data["solid_ids"]),
        upload(data["channel_ids"]),
        upload(data["surface_ids"]),
        upload(data["material_inner"]),
        upload(data["material_outer"]),
        torch.zeros(len(triangles), dtype=torch.int32, device="cuda"),
        tuple(float(x) for x in data["world_origin"]),
        float(data["world_scale"]),
        "capture",
        fingerprint,
        fingerprint,
        len(triangles),
        1000,
    )
    tables = [
        upload(data[k])
        for k in (
            "rindex",
            "absorption",
            "scattering",
            "surface_models",
            "surface_detect",
            "surface_absorb",
            "surface_diffuse",
            "surface_specular",
            "surface_reemit",
            "surface_reemission_cdf",
        )
    ]
    has_components = bool(np.any(data["material_components"]))
    if has_components:
        component_tables = [
            upload(data[key])
            for key in (
                "component_offsets",
                "component_absorption",
                "component_prob",
                "component_wavelength_cdf",
                "component_time_cdf",
            )
        ]
        header = data["time_grid_spec"]
        nt, dt, t0 = (
            int(header[0]),
            float(header[1:].view(np.float32)[0]),
            float(header[1:].view(np.float32)[1]),
        )
    else:
        component_tables = [torch.zeros(1, device="cuda", dtype=torch.int32)] + [
            torch.zeros(1, device="cuda") for _ in range(4)
        ]
        nt, dt, t0 = 2, 1.0, 0.0
    words = upload(initial)
    if normalize:
        _initialize[(triton.cdiv(count, 128),)](words, count, BLOCK=128)
    if random_tape is not None:
        if (
            random_tape.dtype != np.float32
            or random_tape.ndim != 3
            or random_tape.shape[:2] != (count, steps)
        ):
            raise ValueError("random tape must be float32 [N,max_steps,draw_capacity]")
        draws = random_tape.shape[2]
        if draws < 1:
            raise ValueError("random tape needs a positive draw capacity")
        tape = upload(random_tape)
    else:
        if seed is not None:
            from chroma.triton.xorwow import initialize_xorwow

            rng_words = initialize_xorwow(seed, np.arange(count, dtype=np.uint64))
        else:
            rng_words = np.asarray(rng_words)
            if rng_words.dtype != np.uint32 or rng_words.shape != (count, 6):
                raise ValueError("XORWOW states must be uint32 [N,6]")
        tape = upload(rng_words)
        draws = 2147483647  # native streams have no finite recorded-tape capacity
    ledger = torch.full(
        (count, steps, 15) if record_history else (1,), -1, dtype=torch.int32, device="cuda"
    )
    draw_ledger = torch.full(
        (count, steps) if record_history else (count,),
        -1 if record_history else 0,
        dtype=torch.int32,
        device="cuda",
    )
    interactions = torch.zeros(count, dtype=torch.int32, device="cuda")
    overflow = torch.zeros_like(interactions)
    wire_words = data.get("wire_words", np.empty((0, 31), np.uint32))
    wires = upload(wire_words) if len(wire_words) else None
    if wires is not None:
        from .legacy_wires import merge_original_wires
    for iteration in range(steps):
        rows = torch.nonzero((words[:, 11] & 32783) == 0).reshape(-1).to(torch.int32)
        if not len(rows):
            break
        active = words.index_select(0, rows).view(torch.float32)
        positions, directions = active[:, :3].contiguous(), active[:, 3:6].contiguous()
        hit = nearest_chroma_global_hit(
            accelerator,
            positions,
            directions,
            last_triangle=words.index_select(0, rows)[:, 12],
        )
        if wires is not None:
            merge_original_wires[(triton.cdiv(len(rows), 128),)](
                positions,
                directions,
                wires,
                hit.distances,
                hit.surface_normals,
                hit.material_from_indices,
                hit.material_to_indices,
                hit.surface_indices,
                hit.triangle_ids,
                len(rows),
                NPLANES=len(wires),
                BLOCK=128,
                enable_fp_fusion=True,
            )
        _step[(triton.cdiv(len(rows), 128),)](
            words,
            rows,
            len(rows),
            hit.triangle_ids,
            hit.distances,
            hit.surface_normals,
            hit.material_from_indices,
            hit.material_to_indices,
            hit.surface_indices,
            *tables,
            *component_tables,
            tape,
            ledger,
            draw_ledger,
            interactions,
            overflow,
            iteration,
            float(grid[0]),
            float(grid[1] - grid[0]),
            t0,
            dt,
            NW=len(grid),
            NT=nt,
            HAS_COMPONENTS=has_components,
            RECORD_HISTORY=record_history,
            STEPS=steps,
            DRAWS=draws,
            BLOCK=128,
            enable_fp_fusion=True,
        )
    torch.cuda.synchronize()
    result = dict(
        final_words=words.cpu().numpy().view(np.uint32),
        draw_counts=draw_ledger.cpu().numpy(),
        interaction_counts=interactions.cpu().numpy(),
        overflow=overflow.cpu().numpy(),
    )
    if record_history:
        result["state_words"] = ledger.cpu().numpy().view(np.uint32)
    if random_tape is None:
        result["native_rng_words"] = tape.cpu().numpy().view(np.uint32)
    return result


def replay_capture(data, *, native_seed=None):
    """Audit adapter: extract inputs without forwarding expected output states."""
    count, steps, _ = data["random_tape"].shape
    if native_seed is not None and count >= 16384:
        raise ValueError("native seed replay requires the audited single-launch population")
    source = data.get("source_words", data["initial_words"])
    return propagate_legacy(
        data,
        source,
        max_steps=steps,
        seed=native_seed,
        random_tape=data["random_tape"] if native_seed is None else None,
        normalize="source_words" in data,
    )
