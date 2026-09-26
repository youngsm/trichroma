"""Bitwise replicas of the installed CUDA Chroma arithmetic for Triton kernels.

Reference: the installed working tree W (``chroma/cuda/photon.h`` and
friends) compiled by PyCUDA with CUDA 12.4 ``nvcc --use_fast_math
-Xptxas=-dlcm=ca`` for sm_80. Every function below reproduces the *integrated*
``propagate`` kernel's final machine arithmetic, including its known defects
(16-bit history, NaN-producing Fresnel at normal incidence and at the
critical angle, unnormalized paths, out-of-table interpolation reads, FP32
wires). See ``docs/triton_dropin_design.md`` ("Bitwise legacy mode") for the
tape contract and the draw order.

How exactness is guaranteed
---------------------------
nvcc emits some ``fma.rn`` itself, but ptxas additionally contracts a
``mul.f32``/``div.approx.f32`` into a following ``add``/``sub`` (and folds
negations) depending on the data flow of the whole kernel. The decisions of
the integrated original kernel were read back from its SASS (``nvdisasm
-gp`` maps every SASS instruction to its PTX line) and re-emitted as explicit
PTX; recompiling that explicit PTX reproduces the original SASS instruction
stream. Every floating-point operation here is written with the same explicit
form: ``mul/add/sub`` carry ``.rn`` (ptxas never contracts them) and every
contraction the original performed is an explicit ``fma.rn``. The functions
are therefore independent of the surrounding kernel: calling them from any
Triton kernel yields the original bits.

Conventions
-----------
* Arguments are fp32/int32 tensors of one block shape (scalars broadcast).
* ``u*`` arguments are pre-drawn uniforms (``curand_uniform`` values from the
  tape) in the documented order. Functions whose draw count depends on the
  outcome return how many they consumed.
* Masks are int1 tensors. History words are uint32 in int32 tensors; only
  the low 16 bits exist inside the original kernel.
"""

import triton
import triton.language as tl

# ----------------------------------------------------------------- constants

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
NAN_ABORT_16 = tl.constexpr(32768)
#: history bits that end propagation inside the original kernel (16-bit word)
TERMINAL_16 = tl.constexpr(0x800F)

#: propagate_to_boundary / propagate_at_surface return values
BREAK = tl.constexpr(0)
CONTINUE = tl.constexpr(1)
PASS = tl.constexpr(2)

#: surface models (geometry_types.h)
SURFACE_DEFAULT = tl.constexpr(0)
SURFACE_COMPLEX = tl.constexpr(1)
SURFACE_WLS = tl.constexpr(2)
SURFACE_DICHROIC = tl.constexpr(3)
SURFACE_ANGULAR = tl.constexpr(4)

WIRE_TRIANGLE = tl.constexpr(-2)

# ======================================================================
# 1. Explicit PTX operations
# ======================================================================


@triton.jit
def fmul(a, b):
    """mul.rn.ftz.f32 (never contracted)."""
    return tl.inline_asm_elementwise("mul.rn.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fadd(a, b):
    """add.rn.ftz.f32 (never contracted)."""
    return tl.inline_asm_elementwise("add.rn.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fsub(a, b):
    """sub.rn.ftz.f32 (never contracted)."""
    return tl.inline_asm_elementwise("sub.rn.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def ffma(a, b, c):
    """fma.rn.ftz.f32: a*b + c with one rounding."""
    return tl.inline_asm_elementwise("fma.rn.ftz.f32 $0, $1, $2, $3;", "=f,f,f,f", [a, b, c],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fnms(a, b, c):
    """c - a*b as the original contracts it: fma.rn(-a, b, c)."""
    return tl.inline_asm_elementwise(
        "{ .reg .f32 t; neg.ftz.f32 t, $1; fma.rn.ftz.f32 $0, t, $2, $3; }",
        "=f,f,f,f", [a, b, c], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fms(a, b, c):
    """a*b - c as the original contracts it: fma.rn(a, b, -c)."""
    return tl.inline_asm_elementwise(
        "{ .reg .f32 t; neg.ftz.f32 t, $3; fma.rn.ftz.f32 $0, $1, $2, t; }",
        "=f,f,f,f", [a, b, c], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fnmsn(a, b, c):
    """-(a*b) - c contracted: fma.rn(-a, b, -c)."""
    return tl.inline_asm_elementwise(
        "{ .reg .f32 t; .reg .f32 u; neg.ftz.f32 t, $1; neg.ftz.f32 u, $3; fma.rn.ftz.f32 $0, t, $2, u; }",
        "=f,f,f,f", [a, b, c], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fneg(a):
    return tl.inline_asm_elementwise("neg.ftz.f32 $0, $1;", "=f,f", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fabs(a):
    return tl.inline_asm_elementwise("abs.ftz.f32 $0, $1;", "=f,f", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fmin(a, b):
    """min.ftz.f32 (NaN operand -> the other operand)."""
    return tl.inline_asm_elementwise("min.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fmax(a, b):
    """max.ftz.f32 (NaN operand -> the other operand)."""
    return tl.inline_asm_elementwise("max.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def frcp(a):
    """rcp.approx.ftz.f32 (fast-math ``1.0f/x``)."""
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fdiv(a, b):
    """Fast-math ``a/b`` (div.approx.ftz.f32 = a * rcp(b)), not contracted."""
    return tl.inline_asm_elementwise(
        "{ .reg .f32 r; rcp.approx.ftz.f32 r, $2; mul.rn.ftz.f32 $0, $1, r; }",
        "=f,f,f", [a, b], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fdiv_add(a, b, c):
    """``c + a/b`` as contracted by ptxas: fma.rn(a, rcp(b), c)."""
    return tl.inline_asm_elementwise(
        "{ .reg .f32 r; rcp.approx.ftz.f32 r, $2; fma.rn.ftz.f32 $0, $1, r, $3; }",
        "=f,f,f,f", [a, b, c], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fdiv_full(a, b):
    """div.full.ftz.f32 (used by the libdevice atan2f)."""
    return tl.inline_asm_elementwise("div.full.ftz.f32 $0, $1, $2;", "=f,f,f", [a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fsqrt(a):
    return tl.inline_asm_elementwise("sqrt.approx.ftz.f32 $0, $1;", "=f,f", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def frsqrt(a):
    return tl.inline_asm_elementwise("rsqrt.approx.ftz.f32 $0, $1;", "=f,f", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fsin(a):
    return tl.inline_asm_elementwise("sin.approx.ftz.f32 $0, $1;", "=f,f", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fcos(a):
    return tl.inline_asm_elementwise("cos.approx.ftz.f32 $0, $1;", "=f,f", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def flg2(a):
    return tl.inline_asm_elementwise("lg2.approx.ftz.f32 $0, $1;", "=f,f", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def fex2(a):
    return tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;", "=f,f", [a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def flt(a, b):
    """setp.lt.ftz.f32"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.lt.ftz.f32 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,f,f", [a, b], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def fle(a, b):
    """setp.le.ftz.f32"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.le.ftz.f32 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,f,f", [a, b], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def fgt(a, b):
    """setp.gt.ftz.f32"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.gt.ftz.f32 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,f,f", [a, b], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def fge(a, b):
    """setp.ge.ftz.f32"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ge.ftz.f32 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,f,f", [a, b], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def feq(a, b):
    """setp.eq.ftz.f32"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.eq.ftz.f32 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,f,f", [a, b], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def fltu(a, b):
    """setp.ltu.ftz.f32"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ltu.ftz.f32 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,f,f", [a, b], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def fleu(a, b):
    """setp.leu.ftz.f32"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.leu.ftz.f32 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,f,f", [a, b], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def fgtu(a, b):
    """setp.gtu.ftz.f32"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.gtu.ftz.f32 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,f,f", [a, b], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def fgeu(a, b):
    """setp.geu.ftz.f32"""
    return tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.geu.ftz.f32 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,f,f", [a, b], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def fisnan(a):
    """isnan via abs.ftz > inf (unordered), as nvcc lowers __builtin_isnan."""
    return fgtu(fabs(a), tl.full(a.shape, float("inf"), tl.float32))


@triton.jit
def fisfinite(a):
    """isfinite: !(abs.ftz(a) >= inf, unordered)."""
    return ~fgeu(fabs(a), tl.full(a.shape, float("inf"), tl.float32))


@triton.jit
def fbits(a):
    return a.to(tl.int32, bitcast=True)


@triton.jit
def ffrom_bits(a):
    return a.to(tl.float32, bitcast=True)


@triton.jit
def _zero_index(like):
    """int32 zeros of ``like``'s shape (turns a scalar pointer into a block)."""
    return tl.zeros(like.shape, tl.int32)


@triton.jit
def _const(like, bits: tl.constexpr):
    return tl.full(like.shape, bits, tl.int32).to(tl.float32, bitcast=True)


@triton.jit
def cvt_f32_u32(x):
    """cvt.rn.f32.u32 (x holds uint32 bits in int32)."""
    return tl.inline_asm_elementwise("cvt.rn.f32.u32 $0, $1;", "=f,r", [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def cvt_f32_s32(x):
    return tl.inline_asm_elementwise("cvt.rn.f32.s32 $0, $1;", "=f,r", [x],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def cvt_rzi_s32(x):
    """cvt.rzi.ftz.s32.f32 (C float->int truncation; NaN -> 0, saturating)."""
    return tl.inline_asm_elementwise("cvt.rzi.ftz.s32.f32 $0, $1;", "=r,f", [x],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def floor_to_s32(x):
    """(int)floorf(x): cvt.rmi.ftz.f32.f32 then cvt.rzi.ftz.s32.f32."""
    return tl.inline_asm_elementwise(
        "{ .reg .f32 t; cvt.rmi.ftz.f32.f32 t, $1; cvt.rzi.ftz.s32.f32 $0, t; }",
        "=r,f", [x], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def ceil_to_s32(x):
    """(int)ceilf(x): cvt.rpi.ftz.f32.f32 then cvt.rzi.ftz.s32.f32."""
    return tl.inline_asm_elementwise(
        "{ .reg .f32 t; cvt.rpi.ftz.f32.f32 t, $1; cvt.rzi.ftz.s32.f32 $0, t; }",
        "=r,f", [x], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def d_lt_neg_eps(x):
    """(double)x < -1e-6 with cvt.ftz.f64.f32 (intersect.h double tests)."""
    return tl.inline_asm_elementwise(
        "{ .reg .f64 d; .reg .pred p; cvt.ftz.f64.f32 d, $1; setp.lt.f64 p, d, 0dBEB0C6F7A0B5ED8D; selp.u32 $0, 1, 0, p; }",
        "=r,f", [x], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def d_gt_one_eps(x):
    """(double)x > 1.0 + 1e-6 with cvt.ftz.f64.f32 (intersect.h double tests)."""
    return tl.inline_asm_elementwise(
        "{ .reg .f64 d; .reg .pred p; cvt.ftz.f64.f32 d, $1; setp.gt.f64 p, d, 0d3FF000010C6F7A0B; selp.u32 $0, 1, 0, p; }",
        "=r,f", [x], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def d_gt_eps(x):
    """(double)x > 1e-6 with cvt.ftz.f64.f32 (intersect.h double tests)."""
    return tl.inline_asm_elementwise(
        "{ .reg .f64 d; .reg .pred p; cvt.ftz.f64.f32 d, $1; setp.gt.f64 p, d, 0d3EB0C6F7A0B5ED8D; selp.u32 $0, 1, 0, p; }",
        "=r,f", [x], dtype=tl.int32, is_pure=True, pack=1) != 0


@triton.jit
def interp_idx_value(x, lo_value, hi_value, lower):
    """lower + 1.0*(x-xp[lower])/dx evaluated as the original (double path)."""
    return tl.inline_asm_elementwise(
        """{
        .reg .f32 a; .reg .f32 b; .reg .f64 da; .reg .f64 db; .reg .f64 q; .reg .f64 l; .reg .f64 s;
        sub.rn.ftz.f32 a, $1, $2;
        cvt.ftz.f64.f32 da, a;
        sub.rn.ftz.f32 b, $3, $2;
        cvt.ftz.f64.f32 db, b;
        div.rn.f64 q, da, db;
        cvt.rn.f64.s32 l, $4;
        add.f64 s, q, l;
        cvt.rn.ftz.f32.f64 $0, s;
        }""",
        "=f,f,f,f,r", [x, lo_value, hi_value, lower], dtype=tl.float32, is_pure=True, pack=1)


# ======================================================================
# 2. Fast-math library functions as lowered in the original kernel
# ======================================================================


@triton.jit
def logf_(x):
    """logf under --use_fast_math: lg2.approx(x) * ln2 (not contracted)."""
    return fmul(flg2(x), _const(x, 0x3F317218))


@triton.jit
def expf_(x):
    """expf under --use_fast_math: ex2.approx(x * log2e)."""
    return fex2(fmul(x, _const(x, 0x3FB8AA3B)))


@triton.jit
def acosf_(x):
    """libdevice acosf with ftz/approx sqrt, transcribed from the kernel PTX."""
    half = _const(x, 0x3F000000)
    ax = fabs(x)
    t = ffma(half, fneg(ax), half)
    r = frsqrt(t)
    s = fmul(t, r)
    h = fmul(r, half)
    e = ffma(fneg(s), h, half)
    root = ffma(s, e, s)
    root = tl.where(feq(ax, _const(x, 0x3F800000)), 0.0, root)
    large = fgt(ax, _const(x, 0x3F0F5C29))
    arg = tl.where(large, root, ax)
    signed = ffrom_bits((fbits(x) & -2147483648) | fbits(arg))
    sq = fmul(signed, signed)
    p = ffma(_const(x, 0x3D10ECEF), sq, _const(x, 0x3C8B1ABB))
    p = ffma(p, sq, _const(x, 0x3CFC028C))
    p = ffma(p, sq, _const(x, 0x3D372139))
    p = ffma(p, sq, _const(x, 0x3D9993DB))
    p = ffma(p, sq, _const(x, 0x3E2AAAC6))
    p2 = fmul(p, sq)
    approx = ffma(p2, signed, signed)
    neg_approx = tl.where(large, approx, fneg(approx))
    adj = ffma(_const(x, 0x3F6EE581), _const(x, 0x3FD774EB), neg_approx)
    sel = tl.where(fgt(x, _const(x, 0x3F0F5C29)), approx, adj)
    dbl = fadd(sel, sel)
    return tl.where(large, dbl, sel)


@triton.jit
def asinf_(x):
    """libdevice asinf with ftz/approx sqrt, transcribed from the kernel PTX."""
    half = _const(x, 0x3F000000)
    ax = fabs(x)
    t = ffma(half, fneg(ax), half)
    r = frsqrt(t)
    s = fmul(t, r)
    h = fmul(r, half)
    e = ffma(fneg(s), h, half)
    root = ffma(s, e, s)
    root = tl.where(feq(ax, _const(x, 0x3F800000)), 0.0, root)
    large = fgt(ax, _const(x, 0x3F0F5C29))
    a = tl.where(large, root, ax)
    a2 = fmul(a, a)
    p = ffma(_const(x, 0x3D4DD2F7), a2, _const(x, 0x3C99CA97))
    p = ffma(p, a2, _const(x, 0x3D3F90E8))
    p = ffma(p, a2, _const(x, 0x3D993CCF))
    p = ffma(p, a2, _const(x, 0x3E2AAC04))
    p2 = fmul(a2, p)
    approx = ffma(p2, a, a)
    big = ffma(_const(x, 0x3F6EE581), _const(x, 0x3FD774EB), fmul(approx, _const(x, 0xC0000000)))
    mag = tl.where(large, big, approx)
    signed = ffrom_bits((fbits(x) & -2147483648) | fbits(mag))
    return tl.where(fle(mag, _const(x, 0x7F800000)), signed, mag)


@triton.jit
def atan2f_(y, x):
    """libdevice atan2f(y, x) with ftz, transcribed from the kernel PTX."""
    ay = fabs(y)
    ax = fabs(x)
    both_zero = feq(ax, 0.0) & feq(ay, 0.0)
    both_inf = feq(ax, _const(x, 0x7F800000)) & feq(ay, _const(x, 0x7F800000))
    # zero/zero: copysign(x < 0 ? pi : 0, y)
    xneg = fbits(x) < 0
    zero_case = ffrom_bits(((fbits(x) >> 31) & 1078530011) | (fbits(y) & -2147483648))
    inf_case = ffrom_bits(tl.where(xneg, 1075235812, 1061752795) | (fbits(y) & -2147483648))
    hi = fmax(ay, ax)
    lo = fmin(ay, ax)
    q = fdiv_full(lo, hi)
    q2 = fmul(q, q)
    p = ffma(q2, _const(x, 0xBF52C7EA), _const(x, 0xC0B59883))
    p = ffma(p, q2, _const(x, 0xC0D21907))
    p = fmul(q2, p)
    p = fmul(q, p)
    d = fadd(q2, _const(x, 0x41355DC0))
    d = ffma(d, q2, _const(x, 0x41E6BD60))
    d = ffma(d, q2, _const(x, 0x419D92C8))
    r = ffma(p, frcp(d), q)
    r = tl.where(fgt(ay, ax), fsub(_const(x, 0x3FC90FDB), r), r)
    r = tl.where(xneg, fsub(_const(x, 0x40490FDB), r), r)
    signed = ffrom_bits((fbits(y) & -2147483648) | fbits(r))
    s = fadd(ax, ay)
    general = tl.where(fle(s, _const(x, 0x7F800000)), signed, s)
    return tl.where(both_zero, zero_case, tl.where(both_inf, inf_case, general))


# ======================================================================
# 3. Vector helpers with the original contraction pattern
# ======================================================================


@triton.jit
def normalize3(x, y, z):
    """``v /= norm(v)`` (entry normalization, reemission, Rayleigh, ...).

    NVVM (not ptxas) chooses which product of ``dot(v, v)`` stays a plain
    multiply. Every normalization site of the original ``propagate`` kernel
    (and its noinline helpers) keeps ``y*y``: fma(z, z, fma(x, x, y*y)). A
    small kernel compiled in isolation may instead keep ``x*x`` (seen for the
    Rayleigh and diffuse polarizations); :func:`normalize3_xfirst` is that
    form, used only by the unit tests against isolated probes.
    """
    n = fsqrt(ffma(z, z, ffma(x, x, fmul(y, y))))
    return fdiv(x, n), fdiv(y, n), fdiv(z, n)


@triton.jit
def normalize3_xfirst(x, y, z):
    """``v /= norm(v)`` with dot(v, v) = fma(z, z, fma(y, y, x*x)) (see normalize3)."""
    n = fsqrt(ffma(z, z, ffma(y, y, fmul(x, x))))
    return fdiv(x, n), fdiv(y, n), fdiv(z, n)


@triton.jit
def dot3(ax, ay, az, bx, by, bz):
    """dot(a, b) = fma(a.z, b.z, fma(a.x, b.x, a.y*b.y))."""
    return ffma(az, bz, ffma(ax, bx, fmul(ay, by)))


@triton.jit
def dot3_neg(ax, ay, az, bx, by, bz):
    """dot(a, -b) as contracted: fma(-a.z, b.z, fma(-a.y, b.y, -(a.x*b.x)))."""
    return fnms(az, bz, fnmsn(ay, by, fmul(ax, bx)))


@triton.jit
def cross3(ax, ay, az, bx, by, bz):
    """cross(a, b) with the first product of each component contracted."""
    return (fms(ay, bz, fmul(az, by)),
            fms(az, bx, fmul(ax, bz)),
            fms(ax, by, fmul(ay, bx)))


@triton.jit
def uniform_sphere(u_theta, u_z):
    """random.h uniform_sphere: draws (theta, z) in that order."""
    theta = ffma(u_theta, _const(u_theta, 0x40C90FDB), 0.0)
    z = ffma(u_z, _const(u_z, 0x40000000), _const(u_z, 0xBF800000))
    c = fsqrt(fnms(z, z, _const(u_z, 0x3F800000)))
    return fmul(c, fcos(theta)), fmul(c, fsin(theta)), z


@triton.jit
def random_polarization(u_theta, u_z, dx, dy, dz):
    """``cross(uniform_sphere(), dir)`` normalized (bulk/WLS re-emission, diffuse)."""
    sx, sy, sz = uniform_sphere(u_theta, u_z)
    px, py, pz = cross3(sx, sy, sz, dx, dy, dz)
    return normalize3(px, py, pz)


@triton.jit
def rotate(ax, ay, az, phi_cos, phi_sin, nx, ny, nz):
    """rotate.h rotate(a, phi, n) given cos(phi), sin(phi)."""
    d = dot3(nx, ny, nz, ax, ay, az)
    tx = fmul(nx, d)
    ty = fmul(ny, d)
    tz = fmul(nz, d)
    omc = fsub(_const(d, 0x3F800000), phi_cos)
    rx = ffma(phi_cos, ax, fmul(omc, tx))
    ry = ffma(phi_cos, ay, fmul(omc, ty))
    rz = ffma(phi_cos, az, fmul(omc, tz))
    cx = fms(nz, ay, fmul(ny, az))
    cy = fms(nx, az, fmul(nz, ax))
    cz = fms(ny, ax, fmul(nx, ay))
    return ffma(phi_sin, cx, rx), ffma(phi_sin, cy, ry), ffma(phi_sin, cz, rz)


@triton.jit
def get_theta_neg(nx, ny, nz, dx, dy, dz):
    """get_theta(normal, -direction) = acosf(clamp(dot(n, -d)))."""
    c = dot3_neg(dx, dy, dz, nx, ny, nz)
    c = fmax(_const(c, 0xBF800000), fmin(_const(c, 0x3F800000), c))
    return acosf_(c)


# ======================================================================
# 4. Tables
# ======================================================================


@triton.jit
def interp_property(table, x, start, step, n, mask):
    """geometry.h interp_property(m, x, fp) on a uniform grid.

    ``table`` points to fp[0]; the original reads fp[jl+1] with jl = n-1
    when x lands on (or rounds to) the last grid point, one element past the
    table. Tables are uploaded with that padding word (see
    :mod:`chroma.triton.legacy.scene`), so the read is reproduced.
    """
    table = table + _zero_index(x)
    below = flt(x, start)
    top = ffma(step, cvt_f32_u32(n - 1 + _zero_index(x)), start)
    above = fgt(x, top)
    jl = cvt_rzi_s32(fdiv(fsub(x, start), step))
    inner = mask & ~below & ~above
    fj = tl.load(table + jl, mask=inner, other=0.0)
    fj1 = tl.load(table + jl + 1, mask=inner, other=0.0)
    x0 = ffma(step, cvt_f32_s32(jl), start)
    value = fdiv_add(fmul(fsub(x, x0), fsub(fj1, fj)), step, fj)
    first = tl.load(table, mask=mask & below, other=0.0)
    last = tl.load(table + (n - 1), mask=mask & ~below & above, other=0.0)
    return tl.where(below, first, tl.where(above, last, value))


@triton.jit
def sample_cdf_uniform(cdf, n, x0, delta, u, mask):
    """random.h sample_cdf(rng, ncdf, x0, delta, cdf_y) with the drawn ``u``."""
    cdf = cdf + _zero_index(u)
    lower = tl.zeros(u.shape, tl.int32)
    upper = tl.zeros(u.shape, tl.int32) + (n - 1)
    searching = mask & (lower < upper - 1)
    while tl.max(searching.to(tl.int32), 0) > 0:
        half = (lower + upper) >> 1
        y = tl.load(cdf + half, mask=searching, other=0.0)
        go_low = flt(u, y)
        upper = tl.where(searching & go_low, half, upper)
        lower = tl.where(searching & ~go_low, half, lower)
        searching = mask & (lower < upper - 1)
    ylo = tl.load(cdf + lower, mask=mask, other=0.0)
    yhi = tl.load(cdf + upper, mask=mask, other=1.0)
    base = ffma(delta, cvt_f32_s32(lower), x0)
    return fdiv_add(fmul(delta, fsub(u, ylo)), fsub(yhi, ylo), base)


@triton.jit
def interp_nonuniform(x, xp, fp, n, mask):
    """interpolate.h interp(x, n, xp, fp) (DAQ sample_cdf with explicit x)."""
    xp = xp + _zero_index(x)
    fp = fp + _zero_index(x)
    x_first = tl.load(xp, mask=mask, other=0.0)
    x_last = tl.load(xp + (n - 1), mask=mask, other=0.0)
    at_low = fle(x, x_first)
    at_high = ~at_low & fge(x, x_last)
    lower = tl.zeros(x.shape, tl.int32)
    upper = tl.zeros(x.shape, tl.int32) + (n - 1)
    searching = mask & ~at_low & ~at_high & (lower < upper - 1)
    while tl.max(searching.to(tl.int32), 0) > 0:
        half = (lower + upper) >> 1
        y = tl.load(xp + half, mask=searching, other=0.0)
        go_low = flt(x, y)
        upper = tl.where(searching & go_low, half, upper)
        lower = tl.where(searching & ~go_low, half, lower)
        searching = mask & ~at_low & ~at_high & (lower < upper - 1)
    inner = mask & ~at_low & ~at_high
    flo = tl.load(fp + lower, mask=inner, other=0.0)
    fhi = tl.load(fp + upper, mask=inner, other=0.0)
    xlo = tl.load(xp + lower, mask=inner, other=0.0)
    xhi = tl.load(xp + upper, mask=inner, other=1.0)
    value = fdiv_add(fmul(fsub(fhi, flo), fsub(x, xlo)), fsub(xhi, xlo), flo)
    f_first = tl.load(fp, mask=mask & at_low, other=0.0)
    f_last = tl.load(fp + (n - 1), mask=mask & at_high, other=0.0)
    return tl.where(at_low, f_first, tl.where(at_high, f_last, value))


@triton.jit
def interp_idx(x, xp, n, mask):
    """interpolate.h interp_idx(x, n, xp) -> (idx, iidx, at_top).

    ``at_top`` marks idx == n-1: the callers then read table row ``iidx+1``
    past the end (an original out-of-bounds read that cannot be replayed).
    """
    xp = xp + _zero_index(x)
    x_first = tl.load(xp, mask=mask, other=0.0)
    x_last = tl.load(xp + (n - 1), mask=mask, other=0.0)
    at_low = fge(x_first, x)
    at_high = ~at_low & ~fgtu(x_last, x)
    lower = tl.zeros(x.shape, tl.int32)
    upper = tl.zeros(x.shape, tl.int32) + (n - 1)
    searching = mask & ~at_low & ~at_high & (lower < upper - 1)
    while tl.max(searching.to(tl.int32), 0) > 0:
        half = (lower + upper) >> 1
        y = tl.load(xp + half, mask=searching, other=0.0)
        go_low = fgt(y, x)
        upper = tl.where(searching & go_low, half, upper)
        lower = tl.where(searching & ~go_low, half, lower)
        searching = mask & ~at_low & ~at_high & (lower < upper - 1)
    inner = mask & ~at_low & ~at_high
    xlo = tl.load(xp + lower, mask=inner, other=0.0)
    xhi = tl.load(xp + upper, mask=inner, other=1.0)
    value = interp_idx_value(x, xlo, xhi, lower)
    idx = tl.where(at_low, 0.0, tl.where(at_high, cvt_f32_s32(tl.zeros(x.shape, tl.int32) + (n - 1)), value))
    iidx = cvt_rzi_s32(idx)
    return idx, iidx, at_high


# ======================================================================
# 5. Geometry: flattened-BVH nearest hit, FP32 wires, fill_state
# ======================================================================


@triton.jit
def _axis_slab(packed, o, d, world, scale, tmin, tmax):
    """intersect_box for one axis of a packed node (skipped if 1/d is not finite)."""
    lo = ffma(scale, cvt_f32_u32(packed & 0xFFFF), world)
    hi = ffma(scale, cvt_f32_u32((packed >> 16) & 0xFFFF), world)
    inv = fdiv(_const(d, 0x3F800000), d)
    finite = fisfinite(inv)
    noid = fdiv(fneg(o), d)
    t0 = ffma(inv, lo, noid)
    t1 = ffma(inv, hi, noid)
    new_min = fmax(tmin, fmin(t0, t1))
    new_max = fmin(tmax, fmax(t0, t1))
    return tl.where(finite, new_min, tmin), tl.where(finite, new_max, tmax)


@triton.jit
def node_box(px, py, pz, ox, oy, oz, dx, dy, dz, wx, wy, wz, scale):
    """(tmin, tmax) of Chroma's slab test for one packed node."""
    tmin = tl.zeros(ox.shape, tl.float32)
    tmax = tl.full(ox.shape, float("inf"), tl.float32)
    tmin, tmax = _axis_slab(px, ox, dx, wx, scale, tmin, tmax)
    tmin, tmax = _axis_slab(py, oy, dy, wy, scale, tmin, tmax)
    tmin, tmax = _axis_slab(pz, oz, dz, wz, scale, tmin, tmax)
    return tmin, tmax


@triton.jit
def intersect_triangle(v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, ox, oy, oz, dx, dy, dz):
    """intersect.h intersect_triangle -> (hit, t) (Moller-Trumbore, double tests)."""
    e1x = fsub(v1x, v0x)
    e1y = fsub(v1y, v0y)
    e1z = fsub(v1z, v0z)
    e2x = fsub(v2x, v0x)
    e2y = fsub(v2y, v0y)
    e2z = fsub(v2z, v0z)
    hx = fms(dy, e2z, fmul(e2y, dz))
    hy = fms(e2x, dz, fmul(e2z, dx))
    hz = fms(e2y, dx, fmul(e2x, dy))
    a = ffma(e1z, hz, ffma(e1x, hx, fmul(e1y, hy)))
    parallel = fgt(a, _const(a, 0xB4000000)) & flt(a, _const(a, 0x34000000))
    f = frcp(a)
    sx = fsub(ox, v0x)
    sy = fsub(oy, v0y)
    sz = fsub(oz, v0z)
    u = fmul(f, ffma(hz, sz, ffma(hx, sx, fmul(hy, sy))))
    bad_u = d_lt_neg_eps(u) | d_gt_one_eps(u)
    qx = fms(e1z, sy, fmul(e1y, sz))
    qy = fms(e1x, sz, fmul(e1z, sx))
    qz = fms(e1y, sx, fmul(e1x, sy))
    v = fmul(f, ffma(qz, dz, ffma(qx, dx, fmul(qy, dy))))
    bad_v = d_lt_neg_eps(v) | d_gt_one_eps(fadd(u, v))
    t = fmul(f, ffma(e2z, qz, ffma(e2x, qx, fmul(e2y, qy))))
    good_t = d_gt_eps(t) & flt(t, _const(t, 0x7F800000))
    return ~parallel & ~bad_u & ~bad_v & good_t, t


@triton.jit
def intersect_mesh(nodes, vertices, triangles, ox, oy, oz, dx, dy, dz, last_triangle, active,
                   stack, stack_base, wx, wy, wz, scale, STACK: tl.constexpr):
    """mesh.h intersect_mesh -> (triangle or -1, distance, overflow).

    Same traversal order, pruning and strict nearest-hit update as the
    original (ties keep the first triangle found). ``stack`` is an int32
    buffer with ``STACK`` entries per lane starting at ``stack_base``;
    entries pack ``child | nchild << 28``. ``overflow`` flags lanes that
    would exceed the original 1000-entry stack (undefined in the original).
    """
    z0 = _zero_index(ox)
    rx = tl.load(nodes + z0 + 0, mask=active, other=0)
    ry = tl.load(nodes + z0 + 1, mask=active, other=0)
    rz = tl.load(nodes + z0 + 2, mask=active, other=0)
    rw = tl.load(nodes + z0 + 3, mask=active, other=0)
    tmin, tmax = node_box(rx, ry, rz, ox, oy, oz, dx, dy, dz, wx, wy, wz, scale)
    root_hit = active & ~fgt(tmin, tmax)
    best = tl.full(ox.shape, -1, tl.int32)
    best_d = tl.full(ox.shape, -1.0, tl.float32)
    overflow = tl.zeros(ox.shape, tl.int1)
    # current range (node index, remaining) and stack pointer (entries in use)
    node = rw & 0x0FFFFFFF
    remaining = (rw >> 28) & 0xF
    sp = tl.zeros(ox.shape, tl.int32)
    walking = root_hit & (remaining > 0)
    while tl.max(walking.to(tl.int32), 0) > 0:
        nx = tl.load(nodes + node * 4 + 0, mask=walking, other=0)
        ny = tl.load(nodes + node * 4 + 1, mask=walking, other=0)
        nz = tl.load(nodes + node * 4 + 2, mask=walking, other=0)
        nw = tl.load(nodes + node * 4 + 3, mask=walking, other=0)
        tmin, tmax = node_box(nx, ny, nz, ox, oy, oz, dx, dy, dz, wx, wy, wz, scale)
        pruned = (fgeu(best_d, 0.0) & fgt(tmin, best_d)) | fgt(tmin, tmax)
        hit = walking & ~pruned
        child = nw & 0x0FFFFFFF
        nchild = (nw >> 28) & 0xF
        leaf = hit & (nchild == 0) & (child != last_triangle)
        inner = hit & (nchild != 0)
        i0 = tl.load(triangles + child * 3 + 0, mask=leaf, other=0)
        i1 = tl.load(triangles + child * 3 + 1, mask=leaf, other=0)
        i2 = tl.load(triangles + child * 3 + 2, mask=leaf, other=0)
        tri_hit, t = intersect_triangle(
            tl.load(vertices + i0 * 3 + 0, mask=leaf, other=0.0),
            tl.load(vertices + i0 * 3 + 1, mask=leaf, other=0.0),
            tl.load(vertices + i0 * 3 + 2, mask=leaf, other=0.0),
            tl.load(vertices + i1 * 3 + 0, mask=leaf, other=0.0),
            tl.load(vertices + i1 * 3 + 1, mask=leaf, other=0.0),
            tl.load(vertices + i1 * 3 + 2, mask=leaf, other=0.0),
            tl.load(vertices + i2 * 3 + 0, mask=leaf, other=0.0),
            tl.load(vertices + i2 * 3 + 1, mask=leaf, other=0.0),
            tl.load(vertices + i2 * 3 + 2, mask=leaf, other=0.0),
            ox, oy, oz, dx, dy, dz)
        better = leaf & tri_hit & ((best == -1) | flt(t, best_d))
        best = tl.where(better, child, best)
        best_d = tl.where(better, t, best_d)
        push = inner & (sp < STACK)
        overflow |= inner & (sp >= STACK)
        tl.store(stack + stack_base + sp, nw, mask=push)
        sp = sp + push.to(tl.int32)
        # the original writes entry curr == STACK_SIZE (out of bounds) and then
        # breaks out of the child loop: undefined, reported as overflow
        overflow |= walking & (sp > 1000)
        remaining = remaining - 1
        node = node + 1
        stay = walking & (remaining > 0) & ~overflow
        pop = walking & ~stay & (sp > 0) & ~overflow
        top = tl.load(stack + stack_base + sp - 1, mask=pop, other=0)
        sp = sp - pop.to(tl.int32)
        node = tl.where(pop, top & 0x0FFFFFFF, node)
        remaining = tl.where(pop, (top >> 28) & 0xF, remaining)
        walking = stay | pop
    return best, best_d, overflow


@triton.jit
def _plane_f(planes, ip, word: tl.constexpr):
    return tl.load(planes + ip * 31 + word).to(tl.float32, bitcast=True)


@triton.jit
def wire_hit(planes, nplanes, ox, oy, oz, dx, dy, dz, mesh_triangle, mesh_distance, active):
    """FP32 analytic wire planes of fill_state (photon.h lines 96-270).

    ``planes`` points to int32 [nplanes, 31] WirePlane words (124-byte
    records). Returns (distance, surface, material_inner, material_outer,
    normal_x, normal_y, normal_z, dot_raw) of the nearest accepted wire, with
    surface = -1 when none. ``mesh_distance`` is ignored where
    ``mesh_triangle == -1`` (the original uses 1e30f then).
    """
    big = _const(ox, 0x7149F2CA)
    best_mesh = tl.where(mesh_triangle == -1, big, mesh_distance)
    a_dist = big
    a_surf = tl.full(ox.shape, -1, tl.int32)
    a_in = tl.full(ox.shape, -1, tl.int32)
    a_out = tl.full(ox.shape, -1, tl.int32)
    a_nx = tl.zeros(ox.shape, tl.float32)
    a_ny = tl.zeros(ox.shape, tl.float32)
    a_nz = tl.zeros(ox.shape, tl.float32)
    a_dot = tl.zeros(ox.shape, tl.float32)
    for ip in range(nplanes):
        wox = _plane_f(planes, ip, 0)
        woy = _plane_f(planes, ip, 1)
        woz = _plane_f(planes, ip, 2)
        pitch = _plane_f(planes, ip, 9)
        radius = _plane_f(planes, ip, 10)
        umin = _plane_f(planes, ip, 11)
        umax = _plane_f(planes, ip, 12)
        v0 = _plane_f(planes, ip, 15)
        sidx = tl.load(planes + ip * 31 + 16)
        m_out = tl.load(planes + ip * 31 + 17)
        m_in = tl.load(planes + ip * 31 + 18)
        ux = _plane_f(planes, ip, 20)
        uy = _plane_f(planes, ip, 21)
        uz = _plane_f(planes, ip, 22)
        vx = _plane_f(planes, ip, 23)
        vy = _plane_f(planes, ip, 24)
        vz = _plane_f(planes, ip, 25)
        nx = _plane_f(planes, ip, 26)
        ny = _plane_f(planes, ip, 27)
        nz = _plane_f(planes, ip, 28)
        kmin = tl.load(planes + ip * 31 + 29)
        kmax = tl.load(planes + ip * 31 + 30)
        wx = fsub(ox, wox)
        wy = fsub(oy, woy)
        wz = fsub(oz, woz)
        dn = dot3(nx, ny, nz, dx, dy, dz)
        wn0 = dot3(nx, ny, nz, wx, wy, wz)
        plane_dist = fabs(wn0)
        far = ~fleu(plane_dist, fadd(radius, _const(ox, 0x3C23D70A)))
        moving_away = fgt(fmul(wn0, dn), 0.0)
        t_plane = fdiv(fneg(wn0), dn)
        too_far = fgt(t_plane, fadd(best_mesh, radius))
        ok = active & ~(far & (moving_away | too_far))
        du = dot3(ux, uy, uz, dx, dy, dz)
        dv = dot3(vx, vy, vz, dx, dy, dz)
        wu = dot3(ux, uy, uz, wx, wy, wz)
        wv0 = fsub(dot3(vx, vy, vz, wx, wy, wz), v0)
        flat_u = flt(fabs(du), _const(ox, 0x33D6BF95))
        t1 = fdiv(fsub(umin, wu), du)
        t2 = fdiv(fsub(umax, wu), du)
        swap = fgt(t1, t2)
        t1s = tl.where(swap, t2, t1)
        t2s = tl.where(swap, t1, t2)
        t_in = tl.where(flat_u, _const(ox, 0xF149F2CA), fmax(t1s, _const(ox, 0xF149F2CA)))
        t_out = tl.where(flat_u, big, fmin(t2s, big))
        ok &= tl.where(flat_u, ~(flt(wu, umin) | fgt(wu, umax)), ~fgt(t_in, t_out))
        inv_pitch = tl.where(feq(pitch, 0.0), 0.0, frcp(pitch))
        thick = fadd(radius, radius)
        pad = ffma(thick, _const(ox, 0x3F000000), _const(ox, 0x3727C5AC))
        A = ffma(dn, dn, fmul(dv, dv))
        # k-range narrowing (only when kmin <= kmax)
        t_lo = fmax(t_in, _const(ox, 0x38D1B717))
        t_hi = tl.where(flt(best_mesh, t_out), best_mesh, t_out)
        steep_n = fgt(fabs(dn), _const(ox, 0x33D6BF95))
        tn1 = fdiv(fsub(fneg(pad), wn0), dn)
        tn2 = fdiv(fsub(pad, wn0), dn)
        swap_n = fgt(tn1, tn2)
        tn1s = tl.where(swap_n, tn2, tn1)
        tn2s = tl.where(swap_n, tn1, tn2)
        t_lo = tl.where(steep_n, fmax(t_lo, tn1s), t_lo)
        t_hi = tl.where(steep_n, fmin(t_hi, tn2s), t_hi)
        narrow = kmin <= kmax
        ok &= ~(narrow & ~steep_n & fgt(plane_dist, pad))
        ok &= ~(narrow & flt(t_hi, t_lo))
        span = ~fgtu(fabs(dn), _const(ox, 0x33D6BF95)) & ~fleu(fabs(dv), _const(ox, 0x33D6BF95))
        t_hi = tl.where(span, fmin(t_hi, fdiv_add(fadd(pitch, fadd(radius, radius)), fabs(dv), t_lo)), t_hi)
        v_entry = ffma(dv, t_lo, wv0)
        v_exit = ffma(dv, t_hi, wv0)
        v_lo = fsub(fmin(v_entry, v_exit), pad)
        v_hi = fadd(pad, fmax(v_entry, v_exit))
        c_lo = fsub(wv0, pad)
        v_lo = tl.where(flt(c_lo, v_lo), c_lo, v_lo)
        c_hi = fadd(wv0, pad)
        v_hi = tl.where(fgt(c_hi, v_hi), c_hi, v_hi)
        k_lo = tl.maximum(floor_to_s32(fmul(inv_pitch, v_lo)), kmin)
        k_hi = tl.minimum(ceil_to_s32(fmul(inv_pitch, v_hi)), kmax)
        ok &= ~(narrow & (k_lo > k_hi))
        k = tl.where(narrow, k_lo, kmin)
        k_stop = tl.where(narrow, k_hi, kmax)
        r2 = fmul(radius, radius)
        b0 = fmul(wn0, dn)
        c0 = fmul(wn0, wn0)
        eps_arg = fmul(r2, _const(ox, 0x358637BD))
        looping = ok & (k <= k_stop)
        while tl.max(looping.to(tl.int32), 0) > 0:
            fk = cvt_f32_s32(k)
            wv = fnms(pitch, fk, wv0)
            B = ffma(dv, wv, b0)
            r2_0 = ffma(wv, wv, c0)
            C = fsub(r2_0, r2)
            disc = fms(B, B, fmul(A, C))
            cand = looping & ~flt(disc, 0.0)
            sq = fsqrt(disc)
            t_small = fdiv(fsub(fneg(B), sq), A)
            t_large = fdiv(fsub(sq, B), A)
            eps0 = fmax(_const(ox, 0x2B8CBCCC), eps_arg)
            outside = fgt(r2_0, fadd(r2, eps0))
            inside = ~outside & ~fgeu(r2_0, fsub(r2, eps0))
            t_min = _const(ox, 0x38D1B717)
            t = tl.where(outside, t_small, tl.where(inside, t_large, t_min))
            cand &= ~(outside & fle(t_small, t_min))
            cand &= ~(inside & ~fgtu(t_large, t_min))
            uc = ffma(du, t, wu)
            cand &= ~flt(uc, umin) & ~fge(t, a_dist) & ~fgt(uc, umax)
            cand &= ~flt(t, t_in) & ~fgt(t, t_out)
            vn_hit = ffma(dv, t, wv)
            nn_hit = ffma(dn, t, wn0)
            ln = fsqrt(ffma(vn_hit, vn_hit, fmul(nn_hit, nn_hit)))
            cand &= ~fle(ln, 0.0)
            inv_len = frcp(ln)
            a_v = fmul(vn_hit, inv_len)
            a_n = fmul(nn_hit, inv_len)
            hx = ffma(vx, a_v, fmul(nx, a_n))
            hy = ffma(vy, a_v, fmul(ny, a_n))
            hz = ffma(vz, a_v, fmul(nz, a_n))
            hdot = dot3_neg(hx, hy, hz, dx, dy, dz)
            a_dist = tl.where(cand, t, a_dist)
            a_surf = tl.where(cand, sidx, a_surf)
            a_in = tl.where(cand, m_in, a_in)
            a_out = tl.where(cand, m_out, a_out)
            a_nx = tl.where(cand, hx, a_nx)
            a_ny = tl.where(cand, hy, a_ny)
            a_nz = tl.where(cand, hz, a_nz)
            a_dot = tl.where(cand, hdot, a_dot)
            k = k + 1
            looping = looping & (k <= k_stop)
    return a_dist, a_surf, a_in, a_out, a_nx, a_ny, a_nz, a_dot


@triton.jit
def _convert8(c):
    """photon.h convert(): sign-extend an 8-bit index."""
    return tl.where((c & 0x80) != 0, c | -256, c)


@triton.jit
def fill_state_geometry(nodes, vertices, triangles, material_codes, planes, nplanes,
                        px, py, pz, dx, dy, dz, last_triangle, active, stack, stack_base,
                        wx, wy, wz, scale, STACK: tl.constexpr):
    """Boundary part of photon.h fill_state (no material tables).

    Returns ``(triangle, distance, surface, material1, material2,
    normal_x, normal_y, normal_z, overflow)``: ``triangle`` is the new
    ``last_hit_triangle`` (mesh triangle, -2 for a wire, -1 for no hit, in
    which case the caller sets NO_HIT); ``normal`` is oriented against the
    photon (s.surface_normal); ``material1`` is the incident side.
    """
    tri, dist, overflow = intersect_mesh(nodes, vertices, triangles, px, py, pz, dx, dy, dz,
                                         last_triangle, active, stack, stack_base, wx, wy, wz,
                                         scale, STACK)
    a_dist, a_surf, a_in, a_out, a_nx, a_ny, a_nz, a_dot = wire_hit(
        planes, nplanes, px, py, pz, dx, dy, dz, tri, dist, active)
    best = tl.where(tri == -1, _const(px, 0x7149F2CA), dist)
    use_wire = (a_surf > -1) & flt(fadd(a_dist, _const(px, 0x358637BD)), best)
    mesh = ~use_wire & (tri != -1)
    safe = tl.maximum(tri, 0)
    i0 = tl.load(triangles + safe * 3 + 0, mask=active & mesh, other=0)
    i1 = tl.load(triangles + safe * 3 + 1, mask=active & mesh, other=0)
    i2 = tl.load(triangles + safe * 3 + 2, mask=active & mesh, other=0)
    v0x = tl.load(vertices + i0 * 3 + 0, mask=active & mesh, other=0.0)
    v0y = tl.load(vertices + i0 * 3 + 1, mask=active & mesh, other=0.0)
    v0z = tl.load(vertices + i0 * 3 + 2, mask=active & mesh, other=0.0)
    v1x = tl.load(vertices + i1 * 3 + 0, mask=active & mesh, other=0.0)
    v1y = tl.load(vertices + i1 * 3 + 1, mask=active & mesh, other=0.0)
    v1z = tl.load(vertices + i1 * 3 + 2, mask=active & mesh, other=0.0)
    v2x = tl.load(vertices + i2 * 3 + 0, mask=active & mesh, other=0.0)
    v2y = tl.load(vertices + i2 * 3 + 1, mask=active & mesh, other=0.0)
    v2z = tl.load(vertices + i2 * 3 + 2, mask=active & mesh, other=0.0)
    code = tl.load(material_codes + safe, mask=active & mesh, other=0)
    m_inner = _convert8((code >> 24) & 0xFF)
    m_outer = _convert8((code >> 16) & 0xFF)
    m_surface = _convert8((code >> 8) & 0xFF)
    ax_ = fsub(v1x, v0x)
    ay_ = fsub(v1y, v0y)
    az_ = fsub(v1z, v0z)
    bx_ = fsub(v2x, v1x)
    by_ = fsub(v2y, v1y)
    bz_ = fsub(v2z, v1z)
    cx, cy, cz = cross3(ax_, ay_, az_, bx_, by_, bz_)
    mnx, mny, mnz = normalize3(cx, cy, cz)
    raw_x = tl.where(use_wire, a_nx, mnx)
    raw_y = tl.where(use_wire, a_ny, mny)
    raw_z = tl.where(use_wire, a_nz, mnz)
    incident = tl.where(use_wire, a_dot, dot3_neg(dx, dy, dz, mnx, mny, mnz))
    outside = fgt(incident, 0.0)
    inner = tl.where(use_wire, a_in, m_inner)
    outer = tl.where(use_wire, a_out, m_outer)
    material1 = tl.where(outside, outer, inner)
    material2 = tl.where(outside, inner, outer)
    nx = tl.where(outside, raw_x, fneg(raw_x))
    ny = tl.where(outside, raw_y, fneg(raw_y))
    nz = tl.where(outside, raw_z, fneg(raw_z))
    triangle = tl.where(use_wire, -2, tl.where(mesh, tri, -1))
    distance = tl.where(use_wire, a_dist, dist)
    surface = tl.where(use_wire, a_surf, m_surface)
    return triangle, distance, surface, material1, material2, nx, ny, nz, overflow


# ======================================================================
# 6. propagate_to_boundary
# ======================================================================

SPEED_OF_LIGHT_BITS = tl.constexpr(0x4395E56F)  # 299.792458f


@triton.jit
def bulk_distances(absorption_length, scattering_length, u_absorb, u_scatter, weight, use_weights):
    """Draws 0/1 of a step: (absorption_distance, scattering_distance, weights_active).

    With ``use_weights`` and weight > 1e-4 the absorption distance is 1e30f
    and ``weights_active`` stays true (the step's weight decays instead).
    """
    da = fneg(fmul(absorption_length, logf_(u_absorb)))
    ds = fneg(fmul(scattering_length, logf_(u_scatter)))
    active = (use_weights != 0) & fgt(weight, _const(weight, 0x38D1B717))
    da = tl.where(active, _const(da, 0x7149F2CA), da)
    return da, ds, active


@triton.jit
def bulk_outcome(da, ds, distance):
    """0 = absorbed at da, 1 = scattered at ds, 2 = reaches the boundary (PASS)."""
    absorbed = ~fgtu(da, ds) & ~fgtu(da, distance)
    scattered = fgtu(da, ds) & ~fgtu(ds, distance)
    return tl.where(absorbed, 0, tl.where(scattered, 1, 2))


@triton.jit
def advance(px, py, pz, dx, dy, dz, t, distance, n1):
    """``time += d/(c/n1); position += d*direction`` (two approx divisions, one fused)."""
    v = fdiv(_const(n1, SPEED_OF_LIGHT_BITS), n1)
    return (ffma(distance, dx, px), ffma(distance, dy, py), ffma(distance, dz, pz),
            fdiv_add(distance, v, t))


@triton.jit
def attenuate(weight, distance, absorption_length):
    """``weight *= expf(-distance/absorption_length)`` (use_weights paths)."""
    return fmul(expf_(fdiv(fneg(distance), absorption_length)), weight)


@triton.jit
def rayleigh_scatter_raw(dx, dy, dz, px, py, pz, u_theta, u_phi):
    """rayleigh_scatter before its two ``/= norm()``: (dir, pol) unnormalized."""
    one = _const(u_theta, 0x3F800000)
    x = fsub(one, fadd(u_theta, u_theta))
    ang = fdiv(fadd(acosf_(x), _const(x, 0xC0C90FDB)), _const(x, 0x40400000))
    c = fcos(ang)
    cos_theta = fadd(c, c)
    cos_theta = tl.where(fgt(cos_theta, one), one,
                         tl.where(fgeu(cos_theta, _const(x, 0xBF800000)), cos_theta, _const(x, 0xBF800000)))
    theta = acosf_(cos_theta)
    phi = ffma(u_phi, _const(u_phi, 0x40C90FDB), 0.0)
    sin_phi = fsin(phi)
    cos_phi = fcos(phi)
    sat = fsqrt(fnms(pz, pz, one))
    degenerate = fisnan(sat) | flt(sat, _const(x, 0x3727C5AC))
    cap = tl.where(degenerate, one, fdiv(px, sat))
    sap = tl.where(degenerate, 0.0, fdiv(py, sat))
    st = fsin(theta)
    ct = fcos(theta)
    t = fmul(pz, cos_phi)
    ndx = ffma(px, ct, fmul(st, fms(t, cap, fmul(sin_phi, sap))))
    ndy = ffma(py, ct, fmul(st, ffma(sin_phi, cap, fmul(t, sap))))
    ndz = fms(pz, ct, fmul(fmul(st, cos_phi), sat))
    special = flt(fsub(one, fabs(cos_theta)), _const(x, 0x358637BD))
    # pick_new_direction(pol, PI/2, phi): ptxas folds sinf(PI/2f) = 1 and
    # cosf(PI/2f) = -4.3711388e-08f at compile time.
    c90 = _const(x, 0xB33BBD2E)
    spx = ffma(px, c90, fms(t, cap, fmul(sin_phi, sap)))
    spy = ffma(py, c90, ffma(sin_phi, cap, fmul(t, sap)))
    spz = fms(pz, c90, fmul(cos_phi, sat))
    npx = tl.where(special, spx, fnms(cos_theta, ndx, px))
    npy = tl.where(special, spy, fnms(cos_theta, ndy, py))
    npz = tl.where(special, spz, fnms(cos_theta, ndz, pz))
    return ndx, ndy, ndz, npx, npy, npz


@triton.jit
def rayleigh_scatter(dx, dy, dz, px, py, pz, u_theta, u_phi):
    """photon.h rayleigh_scatter: draws (cos-theta, phi); returns (dir, pol) normalized."""
    ndx, ndy, ndz, npx, npy, npz = rayleigh_scatter_raw(dx, dy, dz, px, py, pz, u_theta, u_phi)
    ndx, ndy, ndz = normalize3(ndx, ndy, ndz)
    npx, npy, npz = normalize3(npx, npy, npz)
    return ndx, ndy, ndz, npx, npy, npz


@triton.jit
def select_component(comp_absorption, comp_first, num_comp, wavelength, absorption_length,
                     start, step, n, u_comp, stride, mask):
    """Bulk re-emission component loop (photon.h 512-517) -> component index.

    ``comp_absorption`` rows are ``stride`` floats apart; the material's
    components occupy rows ``comp_first .. comp_first+num_comp-1``.
    """
    comp = tl.zeros(u_comp.shape, tl.int32)
    prob = tl.zeros(u_comp.shape, tl.float32)
    looping = mask & (num_comp > 0)
    while tl.max(looping.to(tl.int32), 0) > 0:
        row = comp_absorption + (comp_first + comp).to(tl.int64) * stride
        comp_abs = interp_property(row, wavelength, start, step, n, looping)
        prob = tl.where(looping, fdiv_add(absorption_length, comp_abs, prob), prob)
        done = flt(u_comp, prob) | (comp + 1 == num_comp)
        comp = tl.where(looping & ~done, comp + 1, comp)
        looping = looping & ~done
    return comp


# ======================================================================
# 7. Reflectors and the dielectric boundary
# ======================================================================


@triton.jit
def specular_reflect(dx, dy, dz, nx, ny, nz):
    """propagate_at_specular_reflector: rotate(normal, incident angle, axis)."""
    theta = get_theta_neg(nx, ny, nz, dx, dy, dz)
    ax, ay, az = cross3(dx, dy, dz, nx, ny, nz)
    ax, ay, az = normalize3(ax, ay, az)
    return rotate(nx, ny, nz, fcos(theta), fsin(theta), ax, ay, az)


@triton.jit
def diffuse_candidate(nx, ny, nz, u_theta, u_z):
    """One rejection iteration of propagate_at_diffuse_reflector.

    Draws (theta, z); returns the candidate direction flipped into the
    normal's hemisphere and ``ndotv``. The caller then draws ``u_accept``
    and repeats while ``not (u_accept < ndotv)`` (use :func:`diffuse_accept`).
    """
    x, y, z = uniform_sphere(u_theta, u_z)
    ndotv = ffma(z, nz, ffma(x, nx, fmul(y, ny)))
    flip = ~fgeu(ndotv, 0.0)
    x = tl.where(flip, fneg(x), x)
    y = tl.where(flip, fneg(y), y)
    z = tl.where(flip, fneg(z), z)
    ndotv = tl.where(flip, fneg(ndotv), ndotv)
    return x, y, z, ndotv


@triton.jit
def diffuse_accept(u_accept, ndotv):
    return flt(u_accept, ndotv)


@triton.jit
def fresnel(dx, dy, dz, px, py, pz, nx, ny, nz, n1, n2, u_polarization, u_reflect):
    """propagate_at_boundary (bug compatible). Draws (polarization, reflect).

    Returns (dir, pol, reflected). At normal incidence the axis is the
    polarization; at and beyond the critical angle refracted_angle is NaN and
    the photon is reflected; the NaN/0 cases are reproduced, not repaired.
    """
    theta_i = get_theta_neg(nx, ny, nz, dx, dy, dz)
    s_i = fsin(theta_i)
    theta_r = asinf_(fdiv(fmul(s_i, n1), n2))
    cx, cy, cz = cross3(dx, dy, dz, nx, ny, nz)
    ln = fsqrt(ffma(cz, cz, ffma(cx, cx, fmul(cy, cy))))
    tiny = flt(ln, _const(ln, 0x358637BD))
    ax = tl.where(tiny, px, fdiv(cx, ln))
    ay = tl.where(tiny, py, fdiv(cy, ln))
    az = tl.where(tiny, pz, fdiv(cz, ln))
    nc = ffma(az, pz, ffma(ax, px, fmul(ay, py)))
    s_pol = flt(u_polarization, fmul(nc, nc))
    diff = fsub(theta_i, theta_r)
    sd = fsin(diff)
    ssum = fadd(theta_i, theta_r)
    rc_s = fdiv(fneg(sd), fsin(ssum))
    rc_p = fdiv(fdiv(sd, fcos(diff)), fdiv(fsin(ssum), fcos(ssum)))
    rc = tl.where(s_pol, rc_s, rc_p)
    reflected = flt(u_reflect, fmul(rc, rc)) | fisnan(theta_r)
    phi = tl.where(reflected, theta_i, fsub(_const(theta_r, 0x40490FDB), theta_r))
    sin_phi = tl.where(reflected, s_i, fsin(phi))
    ox, oy, oz = rotate(nx, ny, nz, fcos(phi), sin_phi, ax, ay, az)
    qx, qy, qz = cross3(ax, ay, az, ox, oy, oz)
    qx, qy, qz = normalize3(qx, qy, qz)
    return (ox, oy, oz,
            tl.where(s_pol, ax, qx), tl.where(s_pol, ay, qy), tl.where(s_pol, az, qz),
            reflected)


# ======================================================================
# 8. Surface models (decision parts; the caller performs the actions)
# ======================================================================

#: surface actions returned by the surface_* functions
ACT_ABSORB = tl.constexpr(0)      # history |= SURFACE_ABSORB, BREAK
ACT_DETECT = tl.constexpr(1)      # history |= SURFACE_DETECT, BREAK
ACT_DIFFUSE = tl.constexpr(2)     # propagate_at_diffuse_reflector, CONTINUE
ACT_SPECULAR = tl.constexpr(3)    # propagate_at_specular_reflector, CONTINUE
ACT_PASS = tl.constexpr(4)        # dielectric boundary (fresnel) follows
ACT_TRANSMIT = tl.constexpr(5)    # history |= SURFACE_TRANSMIT, PASS (fresnel follows)
ACT_REEMIT = tl.constexpr(6)      # WLS re-emission (history |= SURFACE_REEMIT), CONTINUE
ACT_REFRACT = tl.constexpr(7)     # complex model refraction (history |= SURFACE_TRANSMIT), CONTINUE


@triton.jit
def _reweight(absorb, weight, use_weights):
    """Shared use_weights prologue: (apply, survive, new_weight)."""
    apply = (use_weights != 0) & fgt(weight, _const(weight, 0x38D1B717)) & flt(absorb, _const(absorb, 0x3F7FF972))
    survive = fsub(_const(absorb, 0x3F800000), absorb)
    return apply, survive, tl.where(apply, fmul(survive, weight), weight)


@triton.jit
def surface_default(detect, absorb, diffuse, specular, u, weight, use_weights):
    """Default model (photon.h 970-1043): draw ``u`` -> (action, weight).

    With ``use_weights`` a detecting surface detects without drawing
    further (weight *= detect), after ``u`` was drawn.
    """
    apply, survive, weight = _reweight(absorb, weight, use_weights)
    detect = tl.where(apply, fdiv(detect, survive), detect)
    diffuse = tl.where(apply, fdiv(diffuse, survive), diffuse)
    specular = tl.where(apply, fdiv(specular, survive), specular)
    absorb = tl.where(apply, 0.0, absorb)
    forced = (use_weights != 0) & fgt(detect, 0.0)
    s1 = fadd(detect, absorb)
    s2 = fadd(s1, diffuse)
    s3 = fadd(s2, specular)
    action = tl.where(flt(u, absorb), ACT_ABSORB,
                      tl.where(flt(u, s1), ACT_DETECT,
                               tl.where(flt(u, s2), ACT_DIFFUSE,
                                        tl.where(flt(u, s3), ACT_SPECULAR, ACT_PASS))))
    action = tl.where(forced, ACT_DETECT, action)
    weight = tl.where(forced, fmul(detect, weight), weight)
    return action, weight


@triton.jit
def surface_wls(absorb, specular, diffuse, reemit, u, weight, use_weights):
    """WLS model first stage: draw ``u`` -> (stage, weight, specular, diffuse).

    stage 0: absorbed -> caller draws ``u_reemit`` (:func:`wls_reemits`);
    stage 1: reflected -> caller draws ``u_reflect`` (:func:`wls_reflect`);
    stage 2: transmitted (SURFACE_TRANSMIT, PASS).
    """
    apply, survive, weight = _reweight(absorb, weight, use_weights)
    diffuse = tl.where(apply, fdiv(diffuse, survive), diffuse)
    specular = tl.where(apply, fdiv(specular, survive), specular)
    absorb = tl.where(apply, 0.0, absorb)
    stage = tl.where(flt(u, absorb), 0, tl.where(flt(u, fadd(fadd(absorb, specular), diffuse)), 1, 2))
    return stage, weight, specular, diffuse


@triton.jit
def wls_reemits(u_reemit, reemit):
    return flt(u_reemit, reemit)


@triton.jit
def wls_reflect(u_reflect, specular, diffuse):
    """True -> specular, False -> diffuse."""
    return flt(fmul(fadd(specular, diffuse), u_reflect), specular)


@triton.jit
def surface_angular(nx, ny, nz, dx, dy, dz, angles, transmit, reflect_specular, reflect_diffuse,
                    nangles, u, weight, use_weights, mask):
    """Angular model (photon.h 913-951): draw ``u`` -> (action, weight, out_of_table).

    ``out_of_table`` marks the original's read one entry past the property
    arrays (incident angle >= last tabulated angle): not reproducible.
    """
    theta = get_theta_neg(nx, ny, nz, dx, dy, dz)
    idx, iidx, top = interp_idx(theta, angles, nangles, mask)
    t = fsub(idx, cvt_f32_u32(iidx))
    ok = mask & ~top
    t0 = tl.load(transmit + iidx, mask=ok, other=0.0)
    t1 = tl.load(transmit + iidx + 1, mask=ok, other=0.0)
    s0 = tl.load(reflect_specular + iidx, mask=ok, other=0.0)
    s1 = tl.load(reflect_specular + iidx + 1, mask=ok, other=0.0)
    d0 = tl.load(reflect_diffuse + iidx, mask=ok, other=0.0)
    d1 = tl.load(reflect_diffuse + iidx + 1, mask=ok, other=0.0)
    tr = ffma(t, fsub(t1, t0), t0)
    sp = ffma(t, fsub(s1, s0), s0)
    df = ffma(t, fsub(d1, d0), d0)
    absorb = fsub(fsub(fsub(_const(t, 0x3F800000), tr), sp), df)
    apply, survive, weight = _reweight(absorb, weight, use_weights)
    tr = tl.where(apply, fdiv(tr, survive), tr)
    sp = tl.where(apply, fdiv(sp, survive), sp)
    absorb = tl.where(apply, 0.0, absorb)
    a1 = fadd(tr, absorb)
    a2 = fadd(sp, a1)
    action = tl.where(flt(u, absorb), ACT_ABSORB,
                      tl.where(flt(u, a1), ACT_TRANSMIT,
                               tl.where(flt(u, a2), ACT_SPECULAR, ACT_DIFFUSE)))
    return action, weight, mask & top


@triton.jit
def surface_dichroic(nx, ny, nz, dx, dy, dz, angles, nangles, reflect, transmit, stride,
                     wavelength, start, step, n, u, mask):
    """Dichroic model (photon.h 881-910, noinline): draw ``u`` -> (action, out_of_table).

    ``angles`` points at this surface's ``nangles`` incidence angles; row
    ``r`` of ``reflect``/``transmit`` (rows ``stride`` floats apart, uniform
    wavelength grid ``start``/``step``/``n``) is the table for angle ``r``.
    Actions: ACT_SPECULAR, ACT_TRANSMIT (SURFACE_TRANSMIT, PASS) or
    ACT_ABSORB. ``use_weights`` is ignored like in the original.
    ``out_of_table`` marks angle index n-1 (the original reads row n).
    """
    theta = get_theta_neg(nx, ny, nz, dx, dy, dz)
    idx, iidx, top = interp_idx(theta, angles, nangles, mask)
    ok = mask & ~top
    row = iidx.to(tl.int64) * stride
    r0 = interp_property(reflect + row, wavelength, start, step, n, ok)
    r1 = interp_property(reflect + row + stride, wavelength, start, step, n, ok)
    t0 = interp_property(transmit + row, wavelength, start, step, n, ok)
    t1 = interp_property(transmit + row + stride, wavelength, start, step, n, ok)
    frac = fsub(idx, cvt_f32_u32(iidx))
    rp = ffma(frac, fsub(r1, r0), r0)
    tp = ffma(frac, fsub(t1, t0), t0)
    action = tl.where(flt(u, rp), ACT_SPECULAR, tl.where(flt(u, fadd(tp, rp)), ACT_TRANSMIT, ACT_ABSORB))
    return action, mask & top


# ======================================================================
# 9. DAQ (daq.cu run_daq, convert_*)
# ======================================================================


@triton.jit
def daq_passes(u_weight, weight, global_weight):
    """First DAQ draw: ``curand_uniform < weights[i] * global_weight``."""
    return flt(u_weight, fmul(weight, global_weight))


@triton.jit
def daq_charge_int(charge, charge_unit):
    """``(unsigned) roundf(charge / charge_unit)`` as lowered by nvcc."""
    q = fdiv(charge, charge_unit)
    return tl.inline_asm_elementwise(
        """{
        .reg .b32 s; .reg .f32 h; .reg .f32 t; .reg .f32 r;
        mov.b32 s, $1;
        and.b32 s, s, -2147483648;
        or.b32 s, s, 1056964608;
        mov.b32 h, s;
        add.rz.ftz.f32 t, $1, h;
        cvt.rzi.f32.f32 r, t;
        cvt.rzi.ftz.u32.f32 $0, r;
        }""",
        "=r,f", [q], dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def daq_charge_float(q_int, charge_unit):
    """convert_charge_int_to_float: charge_unit * (float)(unsigned) q_int."""
    return fmul(charge_unit, cvt_f32_u32(q_int))


# ======================================================================
# 10. Thin-film (SURFACE_COMPLEX) model
# ======================================================================

import hashlib as _hashlib

from chroma.triton.engine._complex_ptx import ASM as _COMPLEX_ASM_TEXT, CONSTRAINTS as _COMPLEX_CONS_TEXT

_COMPLEX_ASM = tl.constexpr(_COMPLEX_ASM_TEXT)
_COMPLEX_CONS = tl.constexpr(_COMPLEX_CONS_TEXT)
#: Triton's kernel cache keys on the *source* of jit functions, not on the
#: values of the global constexprs they use. The digest below is part of
#: complex_probabilities' source, so a regenerated block cannot silently reuse
#: kernels compiled from the old one; the import-time check keeps it honest.
_COMPLEX_ASM_SHA256 = "32e9f2dbe801ea9cbda2e52f8e03709eb2d8cc2cbec44f20c035a5b1d8969646"
if _hashlib.sha256(_COMPLEX_ASM_TEXT.encode()).hexdigest() != _COMPLEX_ASM_SHA256:
    raise ImportError("chroma.triton.engine._complex_ptx changed: update _COMPLEX_ASM_SHA256 and the digest "
                      "in complex_probabilities' docstring together")


@triton.jit
def complex_probabilities(dx, dy, dz, nx, ny, nz, px, py, pz, n1, n2, eta, k, wavelength, thickness,
                          transmissive):
    """propagate_complex lines 686-783 -> (reflect, absorb, axis_x, axis_y, axis_z).

    ``axis`` is the normalized incident-plane normal (the polarization at
    normal incidence) that the refraction branch uses.
    asm sha256 32e9f2dbe801ea9cbda2e52f8e03709eb2d8cc2cbec44f20c035a5b1d8969646
    """
    return tl.inline_asm_elementwise(
        _COMPLEX_ASM, _COMPLEX_CONS,
        [dy, ny, dx, nx, dz, nz, thickness, wavelength, n2, n1, k, eta, px, py, pz, px, py, pz, transmissive],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32, tl.float32), is_pure=True, pack=1)


@triton.jit
def surface_complex(detect, reflect, absorb, weight, use_weights):
    """use_weights prologue of propagate_complex (before any draw).

    Returns (forced_detect, detect, reflect, absorb, weight). A forced
    detection (use_weights and detect > 0) consumes no draw.
    """
    apply, survive, weight = _reweight(absorb, weight, use_weights)
    detect = tl.where(apply, fdiv(detect, survive), detect)
    reflect = tl.where(apply, fdiv(reflect, survive), reflect)
    absorb = tl.where(apply, 0.0, absorb)
    forced = (use_weights != 0) & fgt(detect, 0.0)
    weight = tl.where(forced, fmul(detect, weight), weight)
    return forced, detect, reflect, absorb, weight


@triton.jit
def complex_stage(u, absorb, reflect, transmissive):
    """After drawing ``u``: 0 absorb (then draw ``u_detect < detect``),
    1 reflect (then draw ``u_reflect < reflect_diffuse`` -> diffuse),
    2 refract (no draw)."""
    return tl.where(flt(u, absorb), 0,
                    tl.where(flt(u, fadd(reflect, absorb)) | (transmissive == 0), 1, 2))


@triton.jit
def complex_refract(dx, dy, dz, nx, ny, nz, n1, n2, ax, ay, az):
    """Refraction branch: rotate(normal, PI - refracted_angle, axis); pol = cross(axis, dir)."""
    theta_i = get_theta_neg(nx, ny, nz, dx, dy, dz)
    theta_r = asinf_(fdiv(fmul(fsin(theta_i), n1), n2))
    phi = fsub(_const(theta_r, 0x40490FDB), theta_r)
    ox, oy, oz = rotate(nx, ny, nz, fcos(phi), fsin(phi), ax, ay, az)
    qx, qy, qz = cross3(ax, ay, az, ox, oy, oz)
    qx, qy, qz = normalize3(qx, qy, qz)
    return ox, oy, oz, qx, qy, qz


if _COMPLEX_ASM_SHA256 not in (complex_probabilities.fn.__doc__ or ""):
    raise ImportError("complex_probabilities' docstring must carry the asm digest (Triton cache key)")
