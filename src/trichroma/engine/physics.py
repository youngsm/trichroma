"""One Chroma transport-loop iteration for every queued photon (Triton).

The kernel mirrors one iteration of ``propagate()`` in the installed Chroma
(``chroma/cuda/propagate.cu`` and ``photon.h``): NaN check, ``fill_state``
(from the boundary query), ``propagate_to_boundary`` (absorption with
multi-component re-emission, Rayleigh scattering, or motion to the boundary),
the surface model, and ``propagate_at_boundary``. Random numbers are drawn
**sequentially per photon in Chroma's call order**, from one of two sources:

* production (``TAPE=False``): Philox(seed; photon id, draw counter), so the
  result does not depend on batch size, launch shape or queue order;
* exact (``TAPE=True``): the photon's recorded native RNG tape.

``FIXES`` selects the documented numerical fixes (angle-free Fresnel without
normal/critical-incidence NaNs, specular reflection of the polarization,
32-bit history). With ``FIXES=False`` the original behaviour is kept,
including the 16-bit device history (NaN abort = ``NO_HIT | 1<<15``).
"""

import triton
import triton.language as tl
from triton.language.random import philox
from triton.language.extra import libdevice

from trichroma.engine.physics_kernels import fresnel_step, fresnel_step_chroma
from trichroma.engine.warp import _warp_any

SPEED_OF_LIGHT = tl.constexpr(299.792458)  # mm/ns (chroma/cuda/physical_constants.h)
WEIGHT_LOWER_THRESHOLD = tl.constexpr(0.0001)
PI = tl.constexpr(3.141592653589793)

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
NAN_ABORT_32 = tl.constexpr(-2147483648)  # 1 << 31 as int32
NAN_ABORT_16 = tl.constexpr(32768)  # original device history bit
ROULETTE_KILL = tl.constexpr(536870912)  # 1 << 29: ended by Russian roulette (opt-in, weighted mode)
TERMINAL_32 = tl.constexpr(1 | 2 | 4 | 8 | 536870912 | -2147483648)
TERMINAL_16 = tl.constexpr(1 | 2 | 4 | 8 | 536870912 | 32768)


# ---------------------------------------------------------------- randoms

@triton.jit
def _philox(seed, c0, c1, c2, c3):
    """``triton.language.random.philox`` (32-bit, 10 rounds) with each round's
    two 32x32-bit products formed by one widening multiply: umulhi(B, c2) and
    B * c2 are the high and low words of the same 64-bit product, so the
    counters are the same integers (one IMAD.WIDE.U32 instead of IMAD.HI plus
    IMAD per product)."""
    seed = seed.to(tl.uint64)
    k0 = (seed & 0xffffffff).to(tl.uint32)
    k1 = ((seed >> 32) & 0xffffffff).to(tl.uint32)
    for _ in tl.static_range(10):
        pb = c2.to(tl.uint64) * 0xCD9E8D57
        pa = c0.to(tl.uint64) * 0xD2511F53
        c0 = (pb >> 32).to(tl.uint32) ^ c1 ^ k0
        c2 = (pa >> 32).to(tl.uint32) ^ c3 ^ k1
        c1 = pb.to(tl.uint32)
        c3 = pa.to(tl.uint32)
        k0 = k0 + 0x9E3779B9
        k1 = k1 + 0xBB67AE85
    return c0, c1, c2, c3


@triton.jit
def philox_uniform(ids, seed, counter):
    """Uniform in (0,1) from Philox(seed; id_lo, id_hi, counter, 0)."""
    low = ids.to(tl.uint64).to(tl.uint32)
    high = (ids.to(tl.uint64) >> 32).to(tl.uint32)
    x, _, _, _ = _philox(seed, low, high, counter.to(tl.uint32), tl.zeros(ids.shape, tl.uint32))
    return _unit(x)


# Production draws: every event of a photon (one loop iteration of Chroma's
# propagate) takes its uniforms from Philox blocks keyed by (photon id, number
# of steps done before the event, block); one Philox call yields four. The
# draws therefore do not depend on batch composition, launch shape, queue order
# or on which kernel (bulk shortcut, wavefront step, fused) takes the event.
B_TRANSPORT = tl.constexpr(0)  # absorption, scattering, Rayleigh cos, Rayleigh phi
B_SURFACE = tl.constexpr(1)  # surface choice, second surface choice, Fresnel polarization, Fresnel reflection
B_WLS = tl.constexpr(2)  # WLS reflection kind, re-emission wavelength
B_WLS_DIR = tl.constexpr(3)  # WLS re-emission direction and polarization spheres
B_BULK_REEMIT = tl.constexpr(4)  # component, re-emission, wavelength, time
B_BULK_DIR = tl.constexpr(5)  # bulk re-emission direction and polarization spheres
B_DIFFUSE_POL = tl.constexpr(6)  # diffuse polarization sphere
B_ROULETTE = tl.constexpr(7)  # Russian roulette
B_DIFFUSE = tl.constexpr(8)  # diffuse rejection loop, one block per trial




@triton.jit
def _unit(x):
    """(x >> 9 + 0.5) * 2**-23 as one fma: with n = x >> 9 < 2**23 both forms
    are exact ((2n + 1) * 2**-24 has at most 24 significant bits)."""
    return tl.fma((x >> 9).to(tl.float32), 1.1920928955078125e-7, 5.9604644775390625e-08)


@triton.jit
def uniforms4(ids, seed, key, block):
    """Four uniforms in (0,1) from Philox(seed; id_lo, id_hi, key, block)."""
    low = ids.to(tl.uint64).to(tl.uint32)
    high = (ids.to(tl.uint64) >> 32).to(tl.uint32)
    c3 = (tl.zeros(ids.shape, tl.int32) + block).to(tl.uint32)
    a, b, c, d = _philox(seed, low, high, key.to(tl.uint32), c3)
    return _unit(a), _unit(b), _unit(c), _unit(d)


@triton.jit
def draw(cursor, mask, ids, seed, tape_ptr, tape_off, TAPE: tl.constexpr):
    """Next uniform for lanes in ``mask``; returns (u, advanced cursor)."""
    if TAPE:
        u = tl.load(tape_ptr + tape_off + cursor, mask=mask, other=0.5)
    else:
        u = philox_uniform(ids, seed, cursor)
    return u, cursor + mask.to(tl.int32)


# ------------------------------------------------------------ table helpers

@triton.jit
def interp_uniform(table, row, x, start, step, count: tl.constexpr, mask):
    """Chroma ``interp_property`` on a uniform grid (clamped at both ends).

    The clamped end values are the interval's own ``left``/``right`` loads:
    ``x < start`` gives ``(x - start) / step <= 0``, hence ``jl == 0`` and
    ``left == table[0]``; above the last grid point ``jl`` is set to
    ``count - 2``, so ``right == table[count - 1]`` (the interpolated value
    is not used there). Every other lane computes exactly as before.

    Lanes outside ``mask`` read row 0 instead of nothing (the loads need no
    predicate, zero fill and select); their value is never used."""
    f = (x - start) / step
    jl = tl.minimum(tl.maximum(f.to(tl.int32), 0), count - 2)
    above = x > start + (count - 1) * step
    jl = tl.where(above, count - 2, jl)
    at = table + tl.where(mask, row, 0) * count + jl
    left = tl.load(at)
    right = tl.load(at + 1)
    value = left + (x - (start + jl.to(tl.float32) * step)) * (right - left) / step
    value = tl.where(x < start, left, value)
    return tl.where(above, right, value)


@triton.jit
def sample_uniform_cdf(table, row, u, start, step, count: tl.constexpr, mask):
    """Chroma ``sample_cdf`` on a uniform grid (binary search, linear inverse).

    The search runs a fixed ITERS = ceil(log2(count - 1)) iterations (the
    interval ``upper - lower`` shrinks from ``count - 1`` to at most
    ``ceil(s / 2)`` per iteration, so every lane has converged by then; a
    converged or masked lane is left unchanged), with no warp vote per
    iteration. The probes and the result are those of the open-ended loop."""
    SPAN: tl.constexpr = count - 1
    ITERS: tl.constexpr = ((SPAN > 1) + (SPAN > 2) + (SPAN > 4) + (SPAN > 8) + (SPAN > 16) + (SPAN > 32)
                           + (SPAN > 64) + (SPAN > 128) + (SPAN > 256) + (SPAN > 512) + (SPAN > 1024)
                           + (SPAN > 2048) + (SPAN > 4096) + (SPAN > 8192) + (SPAN > 16384) + (SPAN > 32768)
                           + (SPAN > 65536) + (SPAN > 131072) + (SPAN > 262144) + (SPAN > 524288)
                           + (SPAN > 1048576) + (SPAN > 2097152) + (SPAN > 4194304) + (SPAN > 8388608)
                           + (SPAN > 16777216) + (SPAN > 33554432) + (SPAN > 67108864) + (SPAN > 134217728)
                           + (SPAN > 268435456) + (SPAN > 536870912) + (SPAN > 1073741824))
    lower = tl.zeros(u.shape, tl.int32)
    upper = tl.full(u.shape, count - 1, tl.int32)
    for _ in range(ITERS):
        half = (lower + upper) >> 1  # lower + upper >= 0: the same as // 2
        value = tl.load(table + row * count + half, mask=mask, other=1.)
        width = mask & (lower < upper - 1)
        upper = tl.where(width & (u < value), half, upper)
        lower = tl.where(width & (u >= value), half, lower)
    # (masked loads: unpredicated ones let LLVM sink the expression below into
    # another block, where it contracts differently: +-1 ulp on some lanes)
    left = tl.load(table + row * count + lower, mask=mask, other=0.)
    right = tl.load(table + row * count + upper, mask=mask, other=1.)
    return start + step * lower.to(tl.float32) + step * (u - left) / (right - left)


# ------------------------------------------------------------- geometry math

@triton.jit
def sincos(a):
    """``(tl.sin(a), tl.cos(a))`` bit for bit, for |a| < 105615 (and NaN).

    The instructions of libdevice's ``__nv_sinf``/``__nv_cosf`` (FTZ build,
    as Triton links them) on their fast path: the three-part Cody-Waite
    reduction by pi/2, which the two share, and the sine and cosine
    polynomials of the reduced argument (quadrant q selects which one each
    returns, and the sign). Libdevice leaves that path only for
    |a| >= 105615 (Payne-Hanek reduction, or NaN for infinities); every
    caller here passes an angle within [-2 pi, 2 pi], where the result is
    identical (checked exhaustively against tl.sin/tl.cos on every float32
    in that range) without the reduction's slow path, its local-memory
    array and the per-function branches."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 j, qf, t, x2, us, ps, rs, pc, rc, s, c, ns, nc;
            .reg .b32 q, q1, b;
            .reg .pred odd, negs, negc;
            mul.rn.f32 j, $2, 0f3F22F983;
            cvt.rni.ftz.s32.f32 q, j;
            cvt.rn.f32.s32 qf, q;
            fma.rn.ftz.f32 t, qf, 0fBFC90FDA, $2;
            fma.rn.ftz.f32 t, qf, 0fB3A22168, t;
            fma.rn.ftz.f32 t, qf, 0fA7C234C5, t;
            mul.rn.ftz.f32 x2, t, t;
            fma.rn.ftz.f32 us, x2, t, 0f00000000;
            fma.rn.ftz.f32 ps, 0fB94D4153, x2, 0f3C0885E4;
            fma.rn.ftz.f32 ps, ps, x2, 0fBE2AAAA8;
            fma.rn.ftz.f32 rs, ps, us, t;
            fma.rn.ftz.f32 pc, 0f37CBAC00, x2, 0fBAB607ED;
            fma.rn.ftz.f32 pc, pc, x2, 0f3D2AAABB;
            fma.rn.ftz.f32 pc, pc, x2, 0fBEFFFFFF;
            fma.rn.ftz.f32 rc, pc, x2, 0f3F800000;
            and.b32 b, q, 1;
            setp.ne.b32 odd, b, 0;
            selp.f32 s, rc, rs, odd;
            selp.f32 c, rs, rc, odd;
            and.b32 b, q, 2;
            setp.ne.b32 negs, b, 0;
            add.s32 q1, q, 1;
            and.b32 b, q1, 2;
            setp.ne.b32 negc, b, 0;
            fma.rn.ftz.f32 ns, s, 0fBF800000, 0f00000000;
            fma.rn.ftz.f32 nc, c, 0fBF800000, 0f00000000;
            selp.f32 $0, ns, s, negs;
            selp.f32 $1, nc, c, negc;
        }
        """,
        constraints="=f,=f,f",
        args=[a],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def log_unit(u):
    """``tl.log(u)`` bit for bit for positive normal finite ``u`` (every
    Philox uniform: ``u`` in [2**-24, 1 - 2**-24]).

    The instructions of libdevice's ``__nv_logf`` (FTZ build) without its
    subnormal pre-scaling and its zero/negative/infinity/NaN fix-ups, which
    such an argument never takes (checked exhaustively against tl.log on
    every float32 in [2**-126, 1])."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .f32 m, ef, i, f, p, r;
            .reg .b32 ix, e, mi;
            mov.b32 ix, $1;
            add.s32 e, ix, -1059760811;
            and.b32 e, e, -8388608;
            sub.s32 mi, ix, e;
            mov.b32 m, mi;
            cvt.rn.f32.s32 ef, e;
            fma.rn.ftz.f32 i, ef, 0f34000000, 0f00000000;
            add.rn.f32 f, m, 0fBF800000;
            fma.rn.ftz.f32 p, 0fBE055027, f, 0f3E1039F6;
            fma.rn.ftz.f32 p, p, f, 0fBDF8CDCC;
            fma.rn.ftz.f32 p, p, f, 0f3E0F2955;
            fma.rn.ftz.f32 p, p, f, 0fBE2AD8B9;
            fma.rn.ftz.f32 p, p, f, 0f3E4CED0B;
            fma.rn.ftz.f32 p, p, f, 0fBE7FFF22;
            fma.rn.ftz.f32 p, p, f, 0f3EAAAA78;
            fma.rn.ftz.f32 p, p, f, 0fBF000000;
            mul.rn.f32 p, f, p;
            fma.rn.ftz.f32 r, p, f, f;
            fma.rn.ftz.f32 $0, i, 0f3F317218, r;
        }
        """,
        constraints="=f,f",
        args=[u],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def uniform_sphere(u_theta, u_z):
    """Chroma ``uniform_sphere``: theta from the first draw, z from the second."""
    theta = 2.0 * PI * u_theta
    u = -1.0 + 2.0 * u_z
    c = tl.sqrt(1.0 - u * u)
    # (libdevice sin/cos here: with :func:`sincos` the compiler contracts the
    # callers' dot/cross products and norms in a different order, which moves
    # a few results by an ulp; see REPORT.md)
    return c * tl.cos(theta), c * tl.sin(theta), u


@triton.jit
def normalize(x, y, z):
    inv = 1.0 / tl.sqrt(x * x + y * y + z * z)
    return x * inv, y * inv, z * inv


@triton.jit
def pick_new_direction(ax, ay, az, theta, phi):
    """Chroma ``pick_new_direction`` (SNOMAN rayscatter)."""
    st, ct = sincos(theta)
    sp, cp = sincos(phi)
    return _new_direction(ax, ay, az, st, ct, sp, cp)


@triton.jit
def _new_direction(ax, ay, az, st, ct, sp, cp):
    """:func:`pick_new_direction` given the sines and cosines of theta and phi."""
    sin_axis_theta = tl.sqrt(1.0 - az * az)
    degenerate = (sin_axis_theta != sin_axis_theta) | (sin_axis_theta < 0.00001)
    cap = tl.where(degenerate, 1.0, ax / sin_axis_theta)
    sap = tl.where(degenerate, 0.0, ay / sin_axis_theta)
    return (ct * ax + st * (az * cp * cap - sp * sap),
            ct * ay + st * (cp * az * sap + sp * cap),
            ct * az - st * cp * sin_axis_theta)


@triton.jit
def rayleigh(dx, dy, dz, px, py, pz, u_cos, u_phi):
    """Chroma ``rayleigh_scatter`` (two draws, polarization-axis frame)."""
    _, c3 = sincos((libdevice.acos(1.0 - 2.0 * u_cos) - 2.0 * PI) / 3.0)  # argument in [-2 pi/3, -pi/3]
    cos_theta = 2.0 * c3
    cos_theta = tl.minimum(tl.maximum(cos_theta, -1.0), 1.0)
    theta = libdevice.acos(cos_theta)
    phi = 2.0 * PI * u_phi
    st, ct = sincos(theta)  # theta in [0, pi]
    sp, cp = sincos(phi)  # phi in (0, 2 pi)
    ndx, ndy, ndz = _new_direction(px, py, pz, st, ct, sp, cp)
    edge = 1.0 - tl.abs(cos_theta) < 1e-6
    # theta = pi/2: the compile-time sin/cos constants, as before
    ex, ey, ez = _new_direction(px, py, pz, tl.sin(PI / 2.0), tl.cos(PI / 2.0), sp, cp)
    npx = tl.where(edge, ex, px - cos_theta * ndx)
    npy = tl.where(edge, ey, py - cos_theta * ndy)
    npz = tl.where(edge, ez, pz - cos_theta * ndz)
    ndx, ndy, ndz = normalize(ndx, ndy, ndz)
    npx, npy, npz = normalize(npx, npy, npz)
    return ndx, ndy, ndz, npx, npy, npz



@triton.jit
def roulette(alive, weight, flags, ids, seed, key, w_rr):
    """Russian roulette for weighted photons below ``w_rr`` (opt-in): survive
    with probability ``weight / w_rr`` at weight ``w_rr``, else end with
    ROULETTE_KILL. The expected weight of every future tally is unchanged."""
    cand = alive & (weight < w_rr) & (weight > 0.)
    n_cand = _warp_any(cand)
    if n_cand != 0:
        u, _, _, _ = uniforms4(ids, seed, key, B_ROULETTE)
        survive = cand & (u * w_rr < weight)
        weight = tl.where(survive, w_rr, weight)
        flags = tl.where(cand & ~survive, flags | ROULETTE_KILL, flags)
    return weight, flags


@triton.jit
def bulk_process(absorbed, scattered, inc, alen, wl, t, dx, dy, dz, px, py, pz, flags, last,
                 u_cos, u_phi, ids, seed, key,
                 comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs,
                 wl_start, wl_step, time_start, time_step,
                 NW: tl.constexpr, NT: tl.constexpr, MAX_COMP: tl.constexpr, HAS_REEMIT: tl.constexpr = True):
    """Chroma's absorption (with multi-component re-emission) and Rayleigh
    branches of ``propagate_to_boundary`` after the photon has moved.

    Branches that no lane of the warp takes are skipped. (Conditions are
    computed before each ``if``: a call in a nested loop test makes Triton
    lower the ``if`` to unstructured branches.)
    """
    n_absorbed = _warp_any(absorbed)
    if not HAS_REEMIT:
        # No material has re-emission components: absorption ends the photon.
        flags = tl.where(absorbed, flags | BULK_ABSORB, flags)
    elif n_absorbed != 0:
        inc_a = tl.where(absorbed, inc, 0)  # (unused outside absorbed)
        c_begin = tl.load(comp_offsets + inc_a)
        ncomp = tl.load(comp_offsets + inc_a + 1) - c_begin
        has_comp = absorbed & (ncomp > 0)
        reemit_i = tl.zeros(ids.shape, tl.int32)
        n_comp = _warp_any(has_comp)
        if n_comp != 0:
            u_comp, u_re, u_w, u_t = uniforms4(ids, seed, key, B_BULK_REEMIT)
            chosen = c_begin
            found = tl.zeros(ids.shape, tl.int1)
            prob = tl.zeros(ids.shape, tl.float32)
            for c in tl.static_range(MAX_COMP):
                in_range = has_comp & (c < ncomp) & ~found
                cabs = interp_uniform(comp_abs, c_begin + c, wl, wl_start, wl_step, NW, in_range)
                prob = tl.where(in_range, prob + alen / cabs, prob)
                pick = in_range & ((u_comp < prob) | (c + 1 == ncomp))
                chosen = tl.where(pick, c_begin + c, chosen)
                found = found | pick
            reemit_p = interp_uniform(comp_prob, chosen, wl, wl_start, wl_step, NW, has_comp)
            bulk_reemit = has_comp & (u_re < reemit_p)
            reemit_i = bulk_reemit.to(tl.int32)
            n_reemit = _warp_any(bulk_reemit)
            if n_reemit != 0:
                new_wl = sample_uniform_cdf(comp_wcdf, chosen, u_w, wl_start, wl_step, NW, bulk_reemit)
                dt = sample_uniform_cdf(comp_tcdf, chosen, u_t, time_start, time_step, NT, bulk_reemit)
                s0, s1, s2, s3 = uniforms4(ids, seed, key, B_BULK_DIR)
                rdx, rdy, rdz = uniform_sphere(s0, s1)
                sx, sy, sz = uniform_sphere(s2, s3)
                rpx, rpy, rpz = normalize(sy * rdz - sz * rdy, sz * rdx - sx * rdz, sx * rdy - sy * rdx)
                wl = tl.where(bulk_reemit, new_wl, wl)
                t = tl.where(bulk_reemit, t + dt, t)
                dx, dy, dz = tl.where(bulk_reemit, rdx, dx), tl.where(bulk_reemit, rdy, dy), tl.where(bulk_reemit, rdz, dz)
                px, py, pz = tl.where(bulk_reemit, rpx, px), tl.where(bulk_reemit, rpy, py), tl.where(bulk_reemit, rpz, pz)
                flags = tl.where(bulk_reemit, flags | BULK_REEMIT, flags)
        flags = tl.where(absorbed & (reemit_i == 0), flags | BULK_ABSORB, flags)
    n_scattered = _warp_any(scattered)
    if n_scattered != 0:
        sdx, sdy, sdz, spx, spy, spz = rayleigh(dx, dy, dz, px, py, pz, u_cos, u_phi)
        dx, dy, dz = tl.where(scattered, sdx, dx), tl.where(scattered, sdy, dy), tl.where(scattered, sdz, dz)
        px, py, pz = tl.where(scattered, spx, px), tl.where(scattered, spy, py), tl.where(scattered, spz, pz)
        flags = tl.where(scattered, flags | RAYLEIGH_SCATTER, flags)
    last = tl.where(absorbed | scattered, -1, last)
    return wl, t, dx, dy, dz, px, py, pz, flags, last


@triton.jit
def launch_normalize(valid, step, norm_step, dx, dy, dz, px, py, pz, renorm_ptr, n_renorm):
    """Original launches normalize direction and polarization on entry.

    ``renorm_ptr`` lists the step numbers at which launches start; each photon
    is normalized at most once per listed step (``norm_step`` remembers it).
    """
    due = tl.zeros(step.shape, tl.int1)
    for r in range(n_renorm):
        due = due | (step == tl.load(renorm_ptr + r))
    due = valid & due & (norm_step != step)
    n_due = _warp_any(due)
    if n_due != 0:
        ndx, ndy, ndz = normalize(dx, dy, dz)
        npx, npy, npz = normalize(px, py, pz)
        dx, dy, dz = tl.where(due, ndx, dx), tl.where(due, ndy, dy), tl.where(due, ndz, dz)
        px, py, pz = tl.where(due, npx, px), tl.where(due, npy, py), tl.where(due, npz, pz)
        norm_step = tl.where(due, step, norm_step)
    return dx, dy, dz, px, py, pz, norm_step

@triton.jit
def renorm_kernel(rows_ptr, count_ptr, capacity, dir_ptr, pol_ptr, steps_ptr, norm_ptr, renorm_ptr, n_renorm,
                  BLOCK: tl.constexpr):
    """Launch-entry normalization for the queued photons, before their boundary
    query: Chroma normalizes direction and polarization when a launch starts,
    so ``fill_state`` already sees the normalized direction."""
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(count_ptr)
    if tl.program_id(0) * BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    row = tl.load(rows_ptr + lane, mask=valid, other=0)
    step = tl.load(steps_ptr + row, mask=valid, other=0)
    norm_step = tl.load(norm_ptr + row, mask=valid, other=-1)
    dx = tl.load(dir_ptr + row * 3 + 0, mask=valid, other=1.)
    dy = tl.load(dir_ptr + row * 3 + 1, mask=valid, other=0.)
    dz = tl.load(dir_ptr + row * 3 + 2, mask=valid, other=0.)
    px = tl.load(pol_ptr + row * 3 + 0, mask=valid, other=0.)
    py = tl.load(pol_ptr + row * 3 + 1, mask=valid, other=1.)
    pz = tl.load(pol_ptr + row * 3 + 2, mask=valid, other=0.)
    ndx, ndy, ndz, npx, npy, npz, new_norm = launch_normalize(valid, step, norm_step, dx, dy, dz, px, py, pz,
                                                              renorm_ptr, n_renorm)
    due = valid & (new_norm != norm_step)
    tl.store(dir_ptr + row * 3 + 0, ndx, mask=due)
    tl.store(dir_ptr + row * 3 + 1, ndy, mask=due)
    tl.store(dir_ptr + row * 3 + 2, ndz, mask=due)
    tl.store(pol_ptr + row * 3 + 0, npx, mask=due)
    tl.store(pol_ptr + row * 3 + 1, npy, mask=due)
    tl.store(pol_ptr + row * 3 + 2, npz, mask=due)
    tl.store(norm_ptr + row, new_norm, mask=due)


# ----------------------------------------------------------------- the step

@triton.jit
def boundary_step(live, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, ids, key,
                  dist, tri, nx, ny, nz, m_inner, m_outer, sidx,
                  rindex, absorption, scattering,
                  comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs,
                  s_present, s_model, s_detect, s_absorb, s_reemit, s_diffuse, s_specular, s_cdf,
                  seed, wl_start, wl_step, time_start, time_step,
                  NW: tl.constexpr, NT: tl.constexpr, MAX_COMP: tl.constexpr,
                  USE_WEIGHTS: tl.constexpr, FIXES: tl.constexpr, ROULETTE: tl.constexpr, w_rr,
                  HAS_WLS: tl.constexpr = True, HAS_REEMIT: tl.constexpr = True,
                  DRAWS: tl.constexpr = False, u_abs=None, u_sca=None, u_cos=None, u_phi=None):
    """One Chroma loop iteration after the NaN check, for lanes in ``live``,
    given their nearest boundary (``dist``, triangle ``tri`` (-1 none, -2
    wire), unit normal, inner/outer material and surface index). ``key`` is
    the photon's step count before this event (the draw key)."""
    # ---- fill_state -----------------------------------------------------
    missed = live & (tri == -1)
    flags = tl.where(missed, flags | NO_HIT, flags)
    last = tl.where(live, tri, last)
    hit = live & (tri != -1)
    # Chroma: facing = dot(n, -d) > 0 -> material1 = outer, normal kept;
    # otherwise material1 = inner and the normal is flipped.
    facing = (nx * -dx + ny * -dy + nz * -dz) > 0.
    inc = tl.where(facing, m_outer, m_inner)
    oth = tl.where(facing, m_inner, m_outer)
    nx = tl.where(facing, nx, -nx)
    ny = tl.where(facing, ny, -ny)
    nz = tl.where(facing, nz, -nz)
    n1 = interp_uniform(rindex, inc, wl, wl_start, wl_step, NW, hit)
    n2 = interp_uniform(rindex, oth, wl, wl_start, wl_step, NW, hit)
    alen = interp_uniform(absorption, inc, wl, wl_start, wl_step, NW, hit)
    slen = interp_uniform(scattering, inc, wl, wl_start, wl_step, NW, hit)

    # ---- propagate_to_boundary -----------------------------------------------
    if not DRAWS:  # else the caller drew uniforms4(ids, seed, key, B_TRANSPORT)
        u_abs, u_sca, u_cos, u_phi = uniforms4(ids, seed, key, B_TRANSPORT)
    da = -alen * log_unit(u_abs)
    ds = -slen * log_unit(u_sca)
    if USE_WEIGHTS:
        weighted = hit & (weight > WEIGHT_LOWER_THRESHOLD)
        da = tl.where(weighted, 1e30, da)
    else:
        weighted = hit & False
    absorbed = hit & (da <= ds) & (da <= dist)
    scattered = hit & ~(da <= ds) & (ds <= dist)
    boundary = hit & ~absorbed & ~scattered
    if USE_WEIGHTS:
        weight = tl.where(weighted & scattered, weight * tl.exp(-ds / alen), weight)
        weight = tl.where(weighted & boundary, weight * tl.exp(-dist / alen), weight)
    travel = tl.where(absorbed, da, tl.where(scattered, ds, dist))
    moved = hit
    # Explicit fma: the bulk shortcut must round the move exactly like this.
    t = tl.where(moved, t + travel / (SPEED_OF_LIGHT / n1), t)
    x = tl.where(moved, tl.fma(travel, dx, x), x)
    y = tl.where(moved, tl.fma(travel, dy, y), y)
    z = tl.where(moved, tl.fma(travel, dz, z), z)

    wl, t, dx, dy, dz, px, py, pz, flags, last = bulk_process(
        absorbed, scattered, inc, alen, wl, t, dx, dy, dz, px, py, pz, flags, last, u_cos, u_phi, ids, seed, key,
        comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs,
        wl_start, wl_step, time_start, time_step, NW, NT, MAX_COMP, HAS_REEMIT)

    # ---- propagate_at_surface ----------------------------------------------------
    # Branches that no lane of the warp takes are skipped; masks leave the
    # branches as int32.
    u_s, u_s2, u_pol, u_refl = uniforms4(ids, seed, key, B_SURFACE)
    safe_s = tl.maximum(sidx, 0)
    surf = boundary & (sidx != -1)
    surf_pass = tl.zeros(hit.shape, tl.int32)
    n_surf = _warp_any(surf)
    if n_surf != 0:
        model = tl.load(s_model + tl.where(surf, safe_s, 0))  # (unused outside surf)
        absorb = interp_uniform(s_absorb, safe_s, wl, wl_start, wl_step, NW, surf)
        detect = interp_uniform(s_detect, safe_s, wl, wl_start, wl_step, NW, surf)
        diffuse = interp_uniform(s_diffuse, safe_s, wl, wl_start, wl_step, NW, surf)
        specular = interp_uniform(s_specular, safe_s, wl, wl_start, wl_step, NW, surf)
        default = surf & (model == 0)
        wls = surf & (model == 2)
        if USE_WEIGHTS:
            reweight = (default | wls) & (weight > WEIGHT_LOWER_THRESHOLD) & (absorb < 1.0 - WEIGHT_LOWER_THRESHOLD)
            survive = tl.where(reweight, 1.0 - absorb, 1.0)
            weight = tl.where(reweight, weight * survive, weight)
            detect = tl.where(reweight & default, detect / survive, detect)
            diffuse = tl.where(reweight, diffuse / survive, diffuse)
            specular = tl.where(reweight, specular / survive, specular)
            absorb = tl.where(reweight, 0., absorb)
            forced = default & (detect > 0.)
        else:
            forced = default & False
        d_absorb = default & ~forced & (u_s < absorb)
        d_detect = default & (forced | (~d_absorb & (u_s < absorb + detect)))
        d_diffuse = default & ~d_absorb & ~d_detect & (u_s < absorb + detect + diffuse)
        d_specular = default & ~d_absorb & ~d_detect & ~d_diffuse & (u_s < absorb + detect + diffuse + specular)
        weight = tl.where(forced, weight * detect, weight)
        flags = tl.where(d_absorb, flags | SURFACE_ABSORB, flags)
        flags = tl.where(d_detect, flags | SURFACE_DETECT, flags)

        w_specular_i = tl.zeros(hit.shape, tl.int32)
        w_diffuse_i = tl.zeros(hit.shape, tl.int32)
        w_transmit_i = tl.zeros(hit.shape, tl.int32)
        n_wls = _warp_any(wls) * HAS_WLS
        if n_wls != 0:
            reemit = interp_uniform(s_reemit, safe_s, wl, wl_start, wl_step, NW, wls)
            u3, u_w, _, _ = uniforms4(ids, seed, key, B_WLS)
            w_absorbed = wls & (u_s < absorb)
            w_reemit = w_absorbed & (u_s2 < reemit)
            w_reflect = wls & ~w_absorbed & (u_s < absorb + specular + diffuse)
            w_specular = w_reflect & (u3 * (specular + diffuse) < specular)
            w_diffuse = w_reflect & ~w_specular
            w_transmit = wls & ~w_absorbed & ~w_reflect
            flags = tl.where(w_absorbed & ~w_reemit, flags | SURFACE_ABSORB, flags)
            flags = tl.where(w_transmit, flags | SURFACE_TRANSMIT, flags)
            w_specular_i = w_specular.to(tl.int32)
            w_diffuse_i = w_diffuse.to(tl.int32)
            w_transmit_i = w_transmit.to(tl.int32)
            # WLS re-emission: new wavelength, isotropic direction, random polarization.
            n_reemit = _warp_any(w_reemit)
            if n_reemit != 0:
                new_wl = sample_uniform_cdf(s_cdf, safe_s, u_w, wl_start, wl_step, NW, w_reemit)
                e0, e1, e2, e3 = uniforms4(ids, seed, key, B_WLS_DIR)
                edx, edy, edz = uniform_sphere(e0, e1)
                sx, sy, sz = uniform_sphere(e2, e3)
                epx, epy, epz = normalize(sy * edz - sz * edy, sz * edx - sx * edz, sx * edy - sy * edx)
                wl = tl.where(w_reemit, new_wl, wl)
                dx, dy, dz = tl.where(w_reemit, edx, dx), tl.where(w_reemit, edy, dy), tl.where(w_reemit, edz, dz)
                px, py, pz = tl.where(w_reemit, epx, px), tl.where(w_reemit, epy, py), tl.where(w_reemit, epz, pz)
                flags = tl.where(w_reemit, flags | SURFACE_REEMIT, flags)
        w_specular = w_specular_i != 0
        w_diffuse = w_diffuse_i != 0
        w_transmit = w_transmit_i != 0

        # Diffuse reflector: rejection sampling about the incident-side normal.
        diff = d_diffuse | w_diffuse
        n_diff = _warp_any(diff)
        if n_diff != 0:
            pending = diff
            trial = n_diff * 0
            n_pending = n_diff
            while n_pending > 0:
                a0, a1, aa, _ = uniforms4(ids, seed, key, B_DIFFUSE + trial)
                cx, cy, cz = uniform_sphere(a0, a1)
                ndotv = cx * nx + cy * ny + cz * nz
                flip = ndotv < 0.
                cx, cy, cz = tl.where(flip, -cx, cx), tl.where(flip, -cy, cy), tl.where(flip, -cz, cz)
                ndotv = tl.where(flip, -ndotv, ndotv)
                dx, dy, dz = tl.where(pending, cx, dx), tl.where(pending, cy, dy), tl.where(pending, cz, dz)
                pending = pending & ~(aa < ndotv)
                n_pending = _warp_any(pending)
                trial += 1
            q0, q1, _, _ = uniforms4(ids, seed, key, B_DIFFUSE_POL)
            sx, sy, sz = uniform_sphere(q0, q1)
            lpx, lpy, lpz = normalize(sy * dz - sz * dy, sz * dx - sx * dz, sx * dy - sy * dx)
            px, py, pz = tl.where(diff, lpx, px), tl.where(diff, lpy, py), tl.where(diff, lpz, pz)
            flags = tl.where(diff, flags | REFLECT_DIFFUSE, flags)

        # Specular reflector.
        spec = d_specular | w_specular
        dn = dx * nx + dy * ny + dz * nz
        rdx, rdy, rdz = dx - 2.0 * dn * nx, dy - 2.0 * dn * ny, dz - 2.0 * dn * nz
        if FIXES:
            pn = px * nx + py * ny + pz * nz
            px = tl.where(spec, px - 2.0 * pn * nx, px)
            py = tl.where(spec, py - 2.0 * pn * ny, py)
            pz = tl.where(spec, pz - 2.0 * pn * nz, pz)
        dx, dy, dz = tl.where(spec, rdx, dx), tl.where(spec, rdy, dy), tl.where(spec, rdz, dz)
        flags = tl.where(spec, flags | REFLECT_SPECULAR, flags)
        surf_pass = ((default & ~d_absorb & ~d_detect & ~d_diffuse & ~d_specular) | w_transmit).to(tl.int32)

    # ---- propagate_at_boundary (PASS) ------------------------------------------------
    passed = boundary & (~surf | (surf_pass != 0))
    n_passed = _warp_any(passed)
    if n_passed != 0:
        if FIXES:
            fdx, fdy, fdz, fpx, fpy, fpz, reflected, _, _, _ = fresnel_step(
                dx, dy, dz, px, py, pz, nx, ny, nz, tl.where(passed, n1, 1.), tl.where(passed, n2, 1.), u_pol, u_refl)
        else:
            # Literal photon.h propagate_at_boundary (NaN at exactly normal incidence).
            fdx, fdy, fdz, fpx, fpy, fpz, reflected, _, _, _ = fresnel_step_chroma(
                dx, dy, dz, px, py, pz, nx, ny, nz, tl.where(passed, n1, 1.), tl.where(passed, n2, 1.), u_pol, u_refl)
        dx, dy, dz = tl.where(passed, fdx, dx), tl.where(passed, fdy, dy), tl.where(passed, fdz, dz)
        px, py, pz = tl.where(passed, fpx, px), tl.where(passed, fpy, py), tl.where(passed, fpz, pz)
        flags = tl.where(passed & reflected, flags | REFLECT_SPECULAR, flags)

    if ROULETTE:
        if FIXES:
            terminal = TERMINAL_32
        else:
            terminal = TERMINAL_16
        weight, flags = roulette(live & ((flags & terminal) == 0), weight, flags, ids, seed, key, w_rr)
    return x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight


@triton.jit(do_not_specialize=["capacity", "seed", "max_steps"])
def step_kernel(
    rows_ptr, count_ptr, capacity,
    pos_ptr, dir_ptr, pol_ptr, wl_ptr, t_ptr, last_ptr, flags_ptr, w_ptr, ids_ptr,
    steps_ptr, cursor_ptr, norm_ptr, tape_ptr, tape_off_ptr, renorm_ptr, n_renorm,
    hit_t, hit_tri, hit_n, hit_codes, last_wire_ptr, hit_wire_ptr,
    rindex, absorption, scattering,
    comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs,
    s_present, s_model, s_detect, s_absorb, s_reemit, s_diffuse, s_specular, s_cdf,
    next_rows, next_count,
    seed, max_steps, wl_start, wl_step, time_start, time_step,
    NW: tl.constexpr, NT: tl.constexpr, MAX_COMP: tl.constexpr,
    USE_WEIGHTS: tl.constexpr, TAPE: tl.constexpr, FIXES: tl.constexpr, BLOCK: tl.constexpr,
    ROULETTE: tl.constexpr = False, w_rr=0.0, LEGACY_WIRES: tl.constexpr = False,
):
    seed = seed.to(tl.uint32, bitcast=True)  # passed as its int32 bit pattern (ProductionEngine.seed_arg)
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(count_ptr)
    if tl.program_id(0) * BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    row = tl.load(rows_ptr + lane, mask=valid, other=0)
    ids = tl.load(ids_ptr + row, mask=valid, other=0)
    step = tl.load(steps_ptr + row, mask=valid, other=0)
    cursor = tl.load(cursor_ptr + row, mask=valid, other=0)
    if TAPE:
        tape_off = tl.load(tape_off_ptr + row, mask=valid, other=0)
    else:
        tape_off = tl.zeros(ids.shape, tl.int64)
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

    norm_step = tl.load(norm_ptr + row, mask=valid, other=-1)
    dx, dy, dz, px, py, pz, norm_step = launch_normalize(valid, step, norm_step, dx, dy, dz, px, py, pz,
                                                         renorm_ptr, n_renorm)

    step = step + valid.to(tl.int32)  # steps++ at the top of the loop body
    if FIXES:
        nan_abort = NAN_ABORT_32
    else:
        nan_abort = NAN_ABORT_16
    product = dx * dy * dz * x * y * z
    is_nan = valid & (product != product)
    flags = tl.where(is_nan, flags | NO_HIT | nan_abort, flags)
    live = valid & ~is_nan

    # ---- fill_state -----------------------------------------------------
    dist = tl.load(hit_t + lane, mask=live, other=0.)
    tri = tl.load(hit_tri + lane, mask=live, other=-1)
    nx = tl.load(hit_n + lane * 3 + 0, mask=live, other=0.)
    ny = tl.load(hit_n + lane * 3 + 1, mask=live, other=0.)
    nz = tl.load(hit_n + lane * 3 + 2, mask=live, other=1.)
    m_inner = tl.load(hit_codes + lane * 3 + 0, mask=live, other=0)
    m_outer = tl.load(hit_codes + lane * 3 + 1, mask=live, other=0)
    sidx = tl.load(hit_codes + lane * 3 + 2, mask=live, other=-1)
    x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight = boundary_step(
        live, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, ids, step - 1,
        dist, tri, nx, ny, nz, m_inner, m_outer, sidx,
        rindex, absorption, scattering,
        comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs,
        s_present, s_model, s_detect, s_absorb, s_reemit, s_diffuse, s_specular, s_cdf,
        seed, wl_start, wl_step, time_start, time_step, NW, NT, MAX_COMP, USE_WEIGHTS, FIXES, ROULETTE, w_rr)
    if not LEGACY_WIRES:
        # A photon that reached a wire (last == -2: no bulk event first) and
        # leaves it outward is outside that convex wire: the next query skips it.
        hit_wire = tl.load(hit_wire_ptr + lane, mask=live & (last == -2), other=-1)
        outward = (dx * nx + dy * ny + dz * nz) > 0.
        tl.store(last_wire_ptr + row, tl.where((last == -2) & outward, hit_wire, -1), mask=live)

    if FIXES:
        terminal = TERMINAL_32
    else:
        terminal = TERMINAL_16
    alive = valid & ((flags & terminal) == 0)
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
    tl.store(cursor_ptr + row, cursor, mask=valid)
    tl.store(norm_ptr + row, norm_step, mask=valid)
    append_rows(row, alive & (step < max_steps), next_rows, next_count)


@triton.jit
def append_rows(rows, mask, output, output_count):
    selected = mask.to(tl.int32)
    total = tl.sum(selected, axis=0)
    offset = tl.cumsum(selected, axis=0) - selected
    start = tl.atomic_add(output_count, total)
    tl.store(output + start + offset, rows, mask=mask)


@triton.jit
def bulk_attempt(active, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, ids, step,
                 lx, ly, lz, ux, uy, uz, inc, n1, alen, slen,
                 comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs, rindex, absorption, scattering,
                 seed, max_steps, wl_start, wl_step, time_start, time_step,
                 NW: tl.constexpr, NT: tl.constexpr, MAX_COMP: tl.constexpr, USE_WEIGHTS: tl.constexpr,
                 FIXES: tl.constexpr, ROULETTE: tl.constexpr, w_rr, HAS_REEMIT: tl.constexpr = True):
    """One bulk collision attempt inside the certified box ``[l, u]`` of
    material ``inc`` for lanes in ``active`` (direction and position finite).

    The event's distance draws (keyed by ``step``) decide a collision; it is
    committed only before the shrunk box exit, where Chroma's ``fill_state``
    would report this material and a farther boundary, so the outcome equals a
    full step with the same draws. Returns the updated state, whether each lane
    committed a collision, and whether it remains in the bulk (not terminal,
    below ``max_steps``).
    """
    if FIXES:
        terminal = TERMINAL_32
    else:
        terminal = TERMINAL_16
    u_abs, u_sca, u_cos, u_phi = uniforms4(ids, seed, step, B_TRANSPORT)
    da = -alen * log_unit(u_abs)
    ds = -slen * log_unit(u_sca)
    if USE_WEIGHTS:
        weighted = active & (weight > WEIGHT_LOWER_THRESHOLD)
        da = tl.where(weighted, 1e30, da)
    else:
        weighted = active & False
    tx = tl.where(dx > 0., (ux - x) / dx, tl.where(dx < 0., (lx - x) / dx, float("inf")))
    ty = tl.where(dy > 0., (uy - y) / dy, tl.where(dy < 0., (ly - y) / dy, float("inf")))
    tz = tl.where(dz > 0., (uz - z) / dz, tl.where(dz < 0., (lz - z) / dz, float("inf")))
    exit_d = tl.minimum(tl.minimum(tx, ty), tz)
    safe = exit_d - tl.maximum(0.01, 2e-6 * exit_d)
    absorbed = active & (da <= ds) & (da < safe)
    scattered = active & ~(da <= ds) & (ds < safe)
    commit = absorbed | scattered
    key = step
    step = step + commit.to(tl.int32)
    if USE_WEIGHTS:
        weight = tl.where(weighted & scattered, weight * tl.exp(-ds / alen), weight)
    travel = tl.where(absorbed, da, ds)
    t = tl.where(commit, t + travel / (SPEED_OF_LIGHT / n1), t)
    x = tl.where(commit, tl.fma(travel, dx, x), x)
    y = tl.where(commit, tl.fma(travel, dy, y), y)
    z = tl.where(commit, tl.fma(travel, dz, z), z)
    wl_before = wl
    wl, t, dx, dy, dz, px, py, pz, flags, last = bulk_process(
        absorbed, scattered, inc, alen, wl, t, dx, dy, dz, px, py, pz, flags, last, u_cos, u_phi, ids, seed, key,
        comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs,
        wl_start, wl_step, time_start, time_step, NW, NT, MAX_COMP, HAS_REEMIT)
    if ROULETTE:
        weight, flags = roulette(commit & ((flags & terminal) == 0), weight, flags, ids, seed, key, w_rr)
    active = commit & ((flags & terminal) == 0) & (step < max_steps)
    changed = active & (wl != wl_before)
    n_changed = _warp_any(changed)
    if n_changed != 0:
        n1 = tl.where(changed, interp_uniform(rindex, inc, wl, wl_start, wl_step, NW, changed), n1)
        alen = tl.where(changed, interp_uniform(absorption, inc, wl, wl_start, wl_step, NW, changed), alen)
        slen = tl.where(changed, interp_uniform(scattering, inc, wl, wl_start, wl_step, NW, changed), slen)
    return (x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, step, n1, alen, slen,
            commit, active)


@triton.jit(do_not_specialize=["capacity", "seed", "max_steps"])
def bulk_kernel(
    rows_ptr, count_ptr, capacity,
    pos_ptr, dir_ptr, pol_ptr, wl_ptr, t_ptr, last_ptr, flags_ptr, w_ptr, ids_ptr,
    steps_ptr, cursor_ptr, norm_ptr, tape_ptr, tape_off_ptr, renorm_ptr, n_renorm,
    grid_material, grid_boxes, gx0, gy0, gz0, gcx, gcy, gcz, NX, NY, NZ,
    rindex, absorption, scattering,
    comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs,
    bulk_rows, bulk_count, boundary_rows, boundary_count,
    seed, max_steps, wl_start, wl_step, time_start, time_step,
    NW: tl.constexpr, NT: tl.constexpr, MAX_COMP: tl.constexpr, USE_WEIGHTS: tl.constexpr,
    TAPE: tl.constexpr, FIXES: tl.constexpr, HISTORY: tl.constexpr, BLOCK: tl.constexpr,
    ROULETTE: tl.constexpr = False, w_rr=0.0,
):
    """Up to HISTORY bulk collisions per photon inside its certified safe box.

    A collision is committed only if it happens before the (shrunk) box exit,
    where Chroma's ``fill_state`` would report the cell's material and a
    farther boundary; the decision and all updates are then identical to a
    full step. Otherwise the photon is handed to the boundary queue without
    consuming its draws.
    """
    seed = seed.to(tl.uint32, bitcast=True)  # passed as its int32 bit pattern (ProductionEngine.seed_arg)
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(count_ptr)
    if tl.program_id(0) * BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    row = tl.load(rows_ptr + lane, mask=valid, other=0)
    ids = tl.load(ids_ptr + row, mask=valid, other=0)
    step = tl.load(steps_ptr + row, mask=valid, other=0)
    cursor = tl.load(cursor_ptr + row, mask=valid, other=0)
    norm_step = tl.load(norm_ptr + row, mask=valid, other=-1)
    if TAPE:
        tape_off = tl.load(tape_off_ptr + row, mask=valid, other=0)
    else:
        tape_off = tl.zeros(ids.shape, tl.int64)
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

    # Certified cell and its safe box.
    ix = tl.floor((x - gx0) / gcx).to(tl.int32)
    iy = tl.floor((y - gy0) / gcy).to(tl.int32)
    iz = tl.floor((z - gz0) / gcz).to(tl.int32)
    in_grid = valid & (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY) & (iz >= 0) & (iz < NZ)
    cell = (ix * NY + iy) * NZ + iz
    mat = tl.load(grid_material + cell, mask=in_grid, other=-1)
    boxed = in_grid & (mat >= 0)
    lx = tl.load(grid_boxes + cell * 6 + 0, mask=boxed, other=0.)
    ly = tl.load(grid_boxes + cell * 6 + 1, mask=boxed, other=0.)
    lz = tl.load(grid_boxes + cell * 6 + 2, mask=boxed, other=0.)
    ux = tl.load(grid_boxes + cell * 6 + 3, mask=boxed, other=0.)
    uy = tl.load(grid_boxes + cell * 6 + 4, mask=boxed, other=0.)
    uz = tl.load(grid_boxes + cell * 6 + 5, mask=boxed, other=0.)
    boxed = boxed & (x > lx) & (x < ux) & (y > ly) & (y < uy) & (z > lz) & (z < uz)
    if FIXES:
        terminal = TERMINAL_32
    else:
        terminal = TERMINAL_16
    active = boxed & ((flags & terminal) == 0) & (step < max_steps)
    handoff = valid & ~boxed & ((flags & terminal) == 0) & (step < max_steps)
    inc = tl.maximum(mat, 0)
    # The material is fixed inside the box; its properties change only with
    # the wavelength (re-emission).
    n1 = interp_uniform(rindex, inc, wl, wl_start, wl_step, NW, active)
    alen = interp_uniform(absorption, inc, wl, wl_start, wl_step, NW, active)
    slen = interp_uniform(scattering, inc, wl, wl_start, wl_step, NW, active)
    go = active
    h = 0
    while tl.sum(go.to(tl.int32), axis=0) > 0:
        active = go
        dx, dy, dz, px, py, pz, norm_step = launch_normalize(active, step, norm_step, dx, dy, dz, px, py, pz,
                                                             renorm_ptr, n_renorm)
        product = dx * dy * dz * x * y * z
        odd = active & (product != product)
        active = active & ~odd
        attempted = active
        (x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, step, n1, alen, slen,
         commit, active) = bulk_attempt(
            active, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, ids, step,
            lx, ly, lz, ux, uy, uz, inc, n1, alen, slen,
            comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs, rindex, absorption, scattering,
            seed, max_steps, wl_start, wl_step, time_start, time_step, NW, NT, MAX_COMP, USE_WEIGHTS, FIXES,
            ROULETTE, w_rr)
        handoff = handoff | odd | (attempted & ~commit)
        h += 1
        go = active & (h < HISTORY)

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
    tl.store(cursor_ptr + row, cursor, mask=valid)
    tl.store(norm_ptr + row, norm_step, mask=valid)
    append_rows(row, active, bulk_rows, bulk_count)
    append_rows(row, handoff & ~active, boundary_rows, boundary_count)
