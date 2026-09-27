"""Nearest-boundary queries for the production engine (Triton).

One lane per ray. The top-level threaded BVH over instances and each
instance's bottom-level threaded BVH are traversed in one stackless loop: a
lane in top-level mode that reaches an instance leaf transforms its ray to the
instance frame and continues in that instance's BLAS; when the BLAS ends it
resumes the top level at the saved escape node. Analytic wire planes follow
the installed Chroma FP32 algorithm and are merged with the mesh hit exactly
as Chroma does (a wire wins when ``t_wire + 1e-6 < t_mesh``).

Outputs per query slot: distance, global triangle id (``-2`` wire, ``-1``
none), unit normal of the hit surface (triangle winding normal, or the
outward wire normal), and the surface's inside/outside materials and surface
index (Chroma ``material1``/``material2``/``surface``).
"""

import triton
import triton.language as tl

from . import scene as _scene
from .warp import _lane_rank, _ring, _warp_any, _warp_count  # noqa: F401

NODE_WIDTH = tl.constexpr(_scene.NODE_WIDTH)
INSTANCE_WIDTH = tl.constexpr(_scene.INSTANCE_WIDTH)
TRI_WIDTH = tl.constexpr(_scene.TRI_WIDTH)
WIRE_WIDTH = tl.constexpr(_scene.WIRE_WIDTH)
BOX_WIDTH = tl.constexpr(_scene.BOX_WIDTH)
BOX_TRI_WIDTH = tl.constexpr(_scene.BOX_TRI_WIDTH)
HAS_CORE = tl.constexpr(_scene.HAS_CORE)
PIN_LEAF_MIN = tl.constexpr(7)  # see nearest_hit_kernel's instance entry


# Conservative slab test: every ray/box distance carries at most a few float32
# roundings (Ize, "Robust BVH Ray Traversal", JCGT 2013), so widening the
# interval by 1e-6 of its ends never drops a box that the ray meets.
SLAB_WIDEN = tl.constexpr(1e-6)


@triton.jit
def _mul_rn(a, b):
    """a*b rounded once; ptxas never contracts it into an fma."""
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=f,f,f", [a, b], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _fma_rn(a, b, c):
    return tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f", [a, b, c], dtype=tl.float32,
                                     is_pure=True, pack=1)


@triton.jit
def _shear(dx, dy, dz, ox, oy, oz):
    """Per-ray constants of the watertight triangle test (Woop, Benthin & Wald,
    "Watertight Ray/Triangle Intersection", JCGT 2013): the dominant axis kz
    of the direction (kx = kz+1, ky = kz+2 mod 3), the shear -d_kx/d_kz,
    -d_ky/d_kz, 1/d_kz that maps the ray onto +z, and the origin in (kx, ky,
    kz) order."""
    ax = tl.abs(dx)
    ay = tl.abs(dy)
    az = tl.abs(dz)
    kz = tl.where((ax >= ay) & (ax >= az), 0, tl.where(ay >= az, 1, 2))
    z0 = kz == 0
    z1 = kz == 1
    dkx = tl.where(z0, dy, tl.where(z1, dz, dx))
    dky = tl.where(z0, dz, tl.where(z1, dx, dy))
    dkz = tl.where(z0, dx, tl.where(z1, dy, dz))
    okx = tl.where(z0, oy, tl.where(z1, oz, ox))
    oky = tl.where(z0, oz, tl.where(z1, ox, oy))
    okz = tl.where(z0, ox, tl.where(z1, oy, oz))
    return kz, -(dkx / dkz), -(dky / dkz), 1.0 / dkz, okx, oky, okz


@triton.jit
def _bits(x):
    return x.to(tl.int32, bitcast=True)


@triton.jit
def _ld4(addr):
    """16 bytes of read-only data at ``addr`` (int64, 16-byte aligned): one
    vector load per lane instead of four scalar ones."""
    return tl.inline_asm_elementwise("ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l", [addr],
                                     dtype=(tl.float32, tl.float32, tl.float32, tl.float32), is_pure=True, pack=1)


@triton.jit
def _load_node(nodes_ptr, node):
    """Node record (NODE_WIDTH = 8 words) of ``node`` (node 0 where negative)."""
    a = (nodes_ptr + tl.maximum(node, 0) * NODE_WIDTH).to(tl.int64, bitcast=True)
    lx, ly, lz, ux = _ld4(a)
    uy, uz, escape, leaf = _ld4(a + 16)
    return lx, ly, lz, ux, uy, uz, _bits(escape), _bits(leaf)


@triton.jit
def _ld_tri(tri_ptr, slot):
    """Record of triangle ``slot`` (a valid slot in every lane) in three
    16-byte loads: vertices v0, v1, v2 (xyz) and the local triangle id."""
    a = (tri_ptr + slot * TRI_WIDTH).to(tl.int64, bitcast=True)
    v0x, v0y, v0z, v1x = _ld4(a)
    v1y, v1z, v2x, v2y = _ld4(a + 16)
    v2z, local_id, _pad0, _pad1 = _ld4(a + 32)
    return v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, _bits(local_id)


@triton.jit
def _ld_inst(inst_ptr, inst):
    """Words 0-15 of the record of instance ``inst`` (a valid instance in
    every lane) in four 16-byte loads: the world->local matrix M (row-major),
    the translation d, the placement's det sign, and (as ints) the BLAS root,
    global triangle offset and code offset."""
    a = (inst_ptr + inst * INSTANCE_WIDTH).to(tl.int64, bitcast=True)
    m00, m01, m02, m10 = _ld4(a)
    m11, m12, m20, m21 = _ld4(a + 16)
    m22, tdx, tdy, tdz = _ld4(a + 32)
    sgn, root, tri, code = _ld4(a + 48)
    return m00, m01, m02, m10, m11, m12, m20, m21, m22, tdx, tdy, tdz, sgn, _bits(root), _bits(tri), _bits(code)


@triton.jit
def _perm(x, y, z, z0, z1):
    """(x, y, z) in the ray's (kx, ky, kz) order, where ``z0``/``z1`` say
    kz == 0 / kz == 1: (y, z, x), (z, x, y) or (x, y, z)."""
    return (tl.where(z0, y, tl.where(z1, z, x)), tl.where(z0, z, tl.where(z1, x, y)),
            tl.where(z0, x, tl.where(z1, y, z)))


@triton.jit
def _watertight(tri_ptr, slot, m, kz, nsx, nsy, sz, okx, oky, okz):
    """Watertight test of the triangle in ``slot`` (every lane must hold a
    valid slot; lanes outside ``m`` report no hit): returns (hit, t,
    unnormalized winding normal e1 x e2, local triangle id).

    Every vertex is translated and sheared by the same instructions, and the
    edge functions are formed without contraction, so the edge function of an
    edge shared by two triangles has exactly opposite values in the two and a
    ray that meets their common edge meets at least one of them; edges and
    vertices count as inside. The record comes in three vector loads; the
    ray's permutation of the axes is applied by selects."""
    z0 = kz == 0
    z1 = kz == 1
    p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, local_id = _ld_tri(tri_ptr, slot)
    v0x, v0y, v0z = _perm(p0x, p0y, p0z, z0, z1)
    v1x, v1y, v1z = _perm(p1x, p1y, p1z, z0, z1)
    v2x, v2y, v2z = _perm(p2x, p2y, p2z, z0, z1)
    az = _sub_rn(v0z, okz)
    bz = _sub_rn(v1z, okz)
    cz = _sub_rn(v2z, okz)
    ax = _fma_rn(nsx, az, _sub_rn(v0x, okx))
    ay = _fma_rn(nsy, az, _sub_rn(v0y, oky))
    bx = _fma_rn(nsx, bz, _sub_rn(v1x, okx))
    by = _fma_rn(nsy, bz, _sub_rn(v1y, oky))
    cx = _fma_rn(nsx, cz, _sub_rn(v2x, okx))
    cy = _fma_rn(nsy, cz, _sub_rn(v2y, oky))
    u = _sub_rn(_mul_rn(cx, by), _mul_rn(cy, bx))
    v = _sub_rn(_mul_rn(ax, cy), _mul_rn(ay, cx))
    w = _sub_rn(_mul_rn(bx, ay), _mul_rn(by, ax))
    inside = ((u >= 0.) & (v >= 0.) & (w >= 0.)) | ((u <= 0.) & (v <= 0.) & (w <= 0.))
    det = u + v + w
    # t = (u sz az + v sz bz + w sz cz) / det, rounded as fixed here.
    t = _fma_rn(_mul_rn(sz, cz), w, _fma_rn(_mul_rn(sz, az), u, _mul_rn(_mul_rn(sz, bz), v))) / det
    # Winding normal in (kx, ky, kz) order (a cyclic permutation commutes with
    # the cross product), returned in (x, y, z) order.
    e1x = v1x - v0x
    e1y = v1y - v0y
    e1z = v1z - v0z
    e2x = v2x - v0x
    e2y = v2y - v0y
    e2z = v2z - v0z
    px = e1y * e2z - e1z * e2y
    py = e1z * e2x - e1x * e2z
    pz = e1x * e2y - e1y * e2x
    nx = tl.where(z0, pz, tl.where(z1, py, px))
    ny = tl.where(z0, px, tl.where(z1, pz, py))
    nz = tl.where(z0, py, tl.where(z1, px, pz))
    return m & inside & (det != 0.), t, nx, ny, nz, local_id


@triton.jit
def _shfl(x, src):
    """``x`` (int32) of lane ``src`` (0..31), per lane: ``shfl.sync.idx`` over
    the full warp (the one-warp programs; all lanes converged)."""
    return tl.inline_asm_elementwise("shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;", "=r,r,r", [x, src],
                                     dtype=tl.int32, is_pure=True, pack=1)


@triton.jit
def _shfl_f(x, src):
    """``x`` (float32) of lane ``src``: the same bits (shuffled as int32)."""
    return _shfl(x.to(tl.int32, bitcast=True), src).to(tl.float32, bitcast=True)


@triton.jit
def _pair_prefix(c):
    """(Sum of ``c`` over the lower lanes, sum over the warp) per lane, for
    ``c`` in 0..15: one ballot per bit of ``c``, popc of each (masked by
    %lanemask_lt for the prefix) -- the integers ``tl.cumsum(c) - c`` and
    ``tl.sum(c)`` (one warp, element i in lane i, all lanes converged)."""
    return tl.inline_asm_elementwise(
        "{ .reg .pred %pfp; .reg .b32 %pfl, %pfm, %pfx, %pfe, %pft; mov.u32 %pfl, %lanemask_lt; "
        "and.b32 %pfx, $2, 1; setp.ne.b32 %pfp, %pfx, 0; vote.sync.ballot.b32 %pfm, %pfp, 0xffffffff; "
        "popc.b32 %pft, %pfm; and.b32 %pfm, %pfm, %pfl; popc.b32 %pfe, %pfm; "
        "and.b32 %pfx, $2, 2; setp.ne.b32 %pfp, %pfx, 0; vote.sync.ballot.b32 %pfm, %pfp, 0xffffffff; "
        "popc.b32 %pfx, %pfm; shl.b32 %pfx, %pfx, 1; add.s32 %pft, %pft, %pfx; "
        "and.b32 %pfm, %pfm, %pfl; popc.b32 %pfx, %pfm; shl.b32 %pfx, %pfx, 1; add.s32 %pfe, %pfe, %pfx; "
        "and.b32 %pfx, $2, 4; setp.ne.b32 %pfp, %pfx, 0; vote.sync.ballot.b32 %pfm, %pfp, 0xffffffff; "
        "popc.b32 %pfx, %pfm; shl.b32 %pfx, %pfx, 2; add.s32 %pft, %pft, %pfx; "
        "and.b32 %pfm, %pfm, %pfl; popc.b32 %pfx, %pfm; shl.b32 %pfx, %pfx, 2; add.s32 %pfe, %pfe, %pfx; "
        "and.b32 %pfx, $2, 8; setp.ne.b32 %pfp, %pfx, 0; vote.sync.ballot.b32 %pfm, %pfp, 0xffffffff; "
        "popc.b32 %pfx, %pfm; shl.b32 %pfx, %pfx, 3; add.s32 %pft, %pft, %pfx; "
        "and.b32 %pfm, %pfm, %pfl; popc.b32 %pfx, %pfm; shl.b32 %pfx, %pfx, 3; add.s32 %pfe, %pfe, %pfx; "
        "mov.u32 $0, %pfe; mov.u32 $1, %pft; }",
        "=r,=r,r", [c], dtype=(tl.int32, tl.int32), is_pure=True, pack=1)


@triton.jit
def _seg_min_step(te, w, lane, seg_end, d):
    """One step of the segmented argmin over (t, lane) pairs (``w`` the
    candidate's lane, -1 for none): lane i takes lane i+d's pair when i+d is
    in its segment and holds a candidate, and lane i holds none or a strictly
    larger t (so a tie keeps the lower lane)."""
    src = tl.minimum(lane + d, 31)
    t2 = _shfl_f(te, src)
    w2 = _shfl(w, src)
    take = (lane + d < seg_end) & (w2 >= 0) & ((w < 0) | (t2 < te))
    return tl.where(take, t2, te), tl.where(take, w2, w)


@triton.jit
def coop_leaf_round(tri_ptr, pend, kz, nsx, nsy, sz, okx, oky, okz, tri_off, inst, last,
                    best_t, best_tri, best_inst, best_slot, box_hit, tie_ok, BLOCK: tl.constexpr):
    """Watertight tests of every lane's recorded leaf, the warp's lanes
    sharing the triangles: returns (best_t, best_tri, best_inst, best_slot,
    box_hit, pend, acc), ``pend`` cleared (-1) where it was set and ``acc``
    where a hit was accepted.

    Per lane (one warp, element i in lane i; call with all lanes converged,
    i.e. from warp-uniform control flow): ``pend`` is the recorded leaf
    (first triangle slot * 16 + count, count 1..15) or -1; ``kz``, ``nsx``,
    ``nsy``, ``sz``, ``okx``, ``oky``, ``okz`` its ray's watertight constants
    (``_shear``), ``tri_off``/``inst`` its instance's triangle offset and
    index, ``last`` the triangle to ignore, the current best hit (``best_t``
    ... ``box_hit``) and ``tie_ok``: a hit at exactly ``best_t`` is accepted
    too, until the lane's first acceptance. The result is bit for bit that of
    each lane testing its own triangles k = 0 .. count-1 in order:

        ok, t, local_id = _watertight(slot = first + k); gid = tri_off + local_id
        if ok & (t > 1e-6) & ((t < best_t) | (tie & (t == best_t))) & (gid != last):
            best_t, best_tri, best_inst, best_slot, box_hit = t, gid, inst, first + k, False
            acc = True; tie = False            (tie starts as tie_ok)

    Its closed form: with T the least t among the valid candidates (ok, t >
    1e-6, gid != last) and k* the first k that has it, the lane ends with
    (T, k*) if T < best_t or (tie_ok and T == best_t), else unchanged. (An
    update before k* has t > T; k* is then accepted, strictly nearer; after
    it no t is smaller, and an equal t is not accepted: the tie is off after
    an acceptance, and before k* none has t = T.) The same holds for any run
    of consecutive k from any state.

    Here the (lane, k) pairs are spread over the warp, 32 per pass: pair q =
    (sum of the counts of the lower lanes) + k is tested by lane q mod 32 of
    pass q // 32, which reads its owner's constants by shuffles (exact copies
    of the bits) and runs the same ``_watertight`` on the same triangle. An
    owner's pairs of a pass are adjacent lanes; a segmented argmin over them
    gives the least t of the valid candidates and, among equal t, the lowest
    lane (= lowest k); the owner applies the closed form to it. Passes go in
    increasing k, so the owner applies it to consecutive runs of k in order.
    Invalid candidates never enter the argmin (NaN t, t <= 1e-6, det == 0 and
    ``last`` fail the same tests as in the loop; a valid t may be +inf), and
    pair lanes past the last pair or outside an owner's segment are never
    read by it."""
    tl.static_assert(BLOCK == 32, "one warp: element i in lane i")
    lane = tl.arange(0, BLOCK)
    fl = pend >= 0
    cnt = tl.where(fl, pend & 15, 0)
    excl, total = _pair_prefix(cnt)
    incl = excl + cnt
    acc = fl & False
    tie = tie_ok & fl
    # Segments of more than 4 pairs (leaves of more than 4 triangles) need two more argmin steps.
    long_seg = _warp_any(cnt > 4)
    b = total * 0
    more = _warp_any(total > b)
    while more != 0:
        # ---- lane = pair q = b + lane of owner o (excl_o <= q < incl_o; o = the
        # number of lanes with incl <= q: incl never decreases), k = q - excl_o
        q = b + lane
        pos = lane * 0
        for s in tl.static_range(5):
            v = _shfl(incl, pos + ((16 >> s) - 1))
            pos = tl.where(v <= q, pos + (16 >> s), pos)
        valid = q < total
        o = tl.minimum(pos, 31)
        o_ex = _shfl(excl, o)
        o_pend = _shfl(pend, o)
        slot = tl.where(valid, (o_pend >> 4) + (q - o_ex), 0)
        ok, t, _nx, _ny, _nz, local_id = _watertight(tri_ptr, slot, valid, _shfl(kz, o), _shfl_f(nsx, o),
                                                     _shfl_f(nsy, o), _shfl_f(sz, o), _shfl_f(okx, o),
                                                     _shfl_f(oky, o), _shfl_f(okz, o))
        gid = _shfl(tri_off, o) + local_id
        cand = ok & (t > 1e-6) & (gid != _shfl(last, o))
        te = t
        w = tl.where(cand, lane, -1)
        # ---- segmented argmin over each owner's adjacent pair lanes of this pass
        seg_end = tl.minimum(o_ex + (o_pend & 15) - b, 32)
        te, w = _seg_min_step(te, w, lane, seg_end, 1)
        te, w = _seg_min_step(te, w, lane, seg_end, 2)
        if long_seg != 0:
            te, w = _seg_min_step(te, w, lane, seg_end, 4)
            te, w = _seg_min_step(te, w, lane, seg_end, 8)
        # ---- each owner with pairs in this pass applies the closed form to their best
        start = tl.maximum(excl, b)
        mine = start < tl.minimum(incl, b + 32)
        src = tl.minimum(start - b, 31)
        tm = _shfl_f(te, src)
        wm = _shfl(w, src)
        gm = _shfl(gid, tl.maximum(wm, 0))
        upd = mine & (wm >= 0) & ((tm < best_t) | (tie & (tm == best_t)))
        best_t = tl.where(upd, tm, best_t)
        best_tri = tl.where(upd, gm, best_tri)
        best_inst = tl.where(upd, inst, best_inst)
        best_slot = tl.where(upd, (pend >> 4) + (b + wm - excl), best_slot)
        box_hit = box_hit & ~upd
        acc = acc | upd
        tie = tie & ~upd
        b += 32
        more = _warp_any(total > b)
    return best_t, best_tri, best_inst, best_slot, box_hit, tl.where(fl, -1, pend), acc


@triton.jit
def _octant(dx, dy, dz):
    """Ray-direction octant: bit a set when component a is negative (selects
    the near-child-first copy of every tree)."""
    return (dx < 0.).to(tl.int32) + 2 * (dy < 0.).to(tl.int32) + 4 * (dz < 0.).to(tl.int32)


@triton.jit
def _rcp(x):
    """Approximate 1/x (rcp.approx: within 1 ulp) for padded interval bounds."""
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True,
                                     pack=1)


@triton.jit
def _inv_exact(d):
    """:func:`_inv` for any input: ``1.0 / x`` (``div.full.f32``), which also
    gives the subnormal reciprocal of a component above 2**126 in magnitude
    (``rcp.approx.ftz`` flushes it to 0). Used where directions come from the
    caller unnormalized (``nearest_hit_kernel``: engine.query and the
    wavefront path)."""
    return 1.0 / tl.where(tl.abs(d) > 1e-30, d, tl.where(d < 0., -1e-30, 1e-30))


@triton.jit
def _inv(d):
    """1/d, finite: components below 1e-30 in magnitude count as +-1e-30, so
    the slab distances of a ray parallel to a slab are huge but never NaN
    (and :func:`_slab` needs no special case).

    Computed as ``rcp.approx.ftz`` of the clamped value: Triton's ``1.0 / x``
    is ``div.full.f32``, which for 2**-126 <= |x| <= 2**126 (and x = +-inf)
    is exactly 1.0 * MUFU.RCP(x) = ``rcp.approx.ftz.f32(x)`` (checked
    bitwise over all float32 x on sm_75 and sm_80, tools/verify_rcp.py); the
    clamped value is never below 1e-30 in magnitude, and a direction
    component never exceeds 2**126, so the result is bit for bit the same
    without div.full's operand-scaling range checks."""
    return _rcp(tl.where(tl.abs(d) > 1e-30, d, tl.where(d < 0., -1e-30, 1e-30)))


@triton.jit
def _slab(nx, ny, nz, fx, fy, fz, ox, oy, oz, ix, iy, iz):
    """Ray/AABB interval (tnear, tfar) of a node box stored as its near and far
    corners (``scene._pack_octants``) for a ray of the node's octant, given
    the finite inverse direction ``i`` (:func:`_inv`), widened to cover
    rounding. The sign of ``i`` on each axis is the octant's (``_octant`` and
    ``_inv`` agree on zeros), and rounding is monotone, so (n - o) * i <=
    (f - o) * i: these are exactly the smaller and larger slab distances that
    a per-axis min/max of the lower/upper distances would select."""
    tnear = tl.maximum(tl.maximum((nx - ox) * ix, (ny - oy) * iy), (nz - oz) * iz)
    tfar = tl.minimum(tl.minimum((fx - ox) * ix, (fy - oy) * iy), (fz - oz) * iz)
    return tnear - tl.abs(tnear) * SLAB_WIDEN, tfar + tl.abs(tfar) * SLAB_WIDEN


@triton.jit
def _slab_hit(nx, ny, nz, fx, fy, fz, ox, oy, oz, ix, iy, iz, best_t):
    """The ray meets the node box within [0, best_t] (conservatively)."""
    tnear, tfar = _slab(nx, ny, nz, fx, fy, fz, ox, oy, oz, ix, iy, iz)
    return tl.maximum(tnear, 0.) <= tl.minimum(tfar, best_t)


@triton.jit
def first_leaf(valid, ox, oy, oz, dx, dy, dz, ix, iy, iz, best_t, nodes_ptr, tlas_nodes, STEPS: tl.constexpr,
               TRIPS: tl.constexpr):
    """The top-level walk to the first instance leaf reached within [0,
    best_t] (its absolute row in the ray's octant copy, -1 for none): the
    leaf ``batch_traversal`` resumes from. ``i`` = :func:`_inv` of ``d``."""
    stop = tl.full(ox.shape, -1, tl.int32)
    octant = _octant(dx, dy, dz)
    # Start at the root's first child: a box inside a box the ray misses is
    # missed too (the slab test is monotone in the bounds), so the root's own
    # test decides nothing. (``tlas_nodes`` may be specialized to the constant
    # 1: compare it as a tensor.)
    node = tl.where(valid, octant * tlas_nodes + ((octant * 0 + tlas_nodes) > 1).to(tl.int32), -1)
    trips = 0
    # (The loop test is a variable: a call in a while test inside the caller's
    # dynamic if would make Triton lower that if to unstructured branches.)
    more = _warp_any(node >= 0)
    while (more != 0) & ((TRIPS == 0) | (trips < TRIPS)):
        for _step in range(STEPS):
            active = node >= 0
            lx, ly, lz, ux, uy, uz, escape, leaf = _load_node(nodes_ptr, node)
            hit = active & _slab_hit(lx, ly, lz, ux, uy, uz, ox, oy, oz, ix, iy, iz, best_t)
            is_leaf = leaf >= 0
            reach = hit & is_leaf
            stop = tl.where(reach, node, stop)
            nxt = tl.where(hit & ~is_leaf, node + 1, escape)
            node = tl.where(active, tl.where(reach, -1, nxt), node)
        trips += STEPS
        more = _warp_any(node >= 0)
    if TRIPS > 0:
        stop = tl.where(node >= 0, node, stop)
    return stop


@triton.jit
def top_level_query(valid, ox, oy, oz, dx, dy, dz, last, nodes_ptr, tlas_nodes, boxes_ptr, n_boxes, box_tris_ptr,
                    FACE_TRIS: tl.constexpr, STEPS: tl.constexpr, BOX_SNAP: tl.constexpr = False,
                    TRIPS: tl.constexpr = 0):
    """Analytic boxes, then the top-level tree without entering instances.

    Returns the nearest box hit (distance, triangle or -1, unnormalized
    normal, inner/outer material, surface), whether the ray may reach an
    instance before it (the instance's mesh may then be nearer) and the
    top-level node where the walk stopped (``instance_traversal`` resumes
    there: no earlier node of the same octant order reaches an instance): the
    first instance leaf the ray reaches or, with ``TRIPS > 0``, the node the
    walk has come to after TRIPS nodes.
    """
    best_t = tl.full(ox.shape, float("inf"), tl.float32)
    best_tri = tl.full(ox.shape, -1, tl.int32)
    bnx = tl.zeros(ox.shape, tl.float32)
    bny = tl.zeros(ox.shape, tl.float32)
    bnz = tl.zeros(ox.shape, tl.float32)
    box_hit = tl.zeros(ox.shape, tl.int1)
    bm1 = tl.zeros(ox.shape, tl.int32)
    bm2 = tl.zeros(ox.shape, tl.int32)
    bsf = tl.full(ox.shape, -1, tl.int32)
    ix = _inv(dx)
    iy = _inv(dy)
    iz = _inv(dz)
    # A ray inside a box (or entering it through the face it starts on) cannot
    # meet a box that contains that box before that box's exit face.
    inside = tl.zeros(ox.shape, tl.int32)
    for box_index in range(n_boxes):
        contains = _bits(tl.load(boxes_ptr + box_index * BOX_WIDTH + 20))
        act = valid & ((inside & contains) == 0)
        if _warp_any(act) != 0:
            best_t, best_tri, bnx, bny, bnz, box_hit, bm1, bm2, bsf, inside = _analytic_box(
                boxes_ptr, box_tris_ptr, box_index, act, ox, oy, oz, dx, dy, dz, ix, iy, iz, last,
                best_t, best_tri, bnx, bny, bnz, box_hit, bm1, bm2, bsf, inside, BOX_SNAP)
    stop = first_leaf(valid, ox, oy, oz, dx, dy, dz, ix, iy, iz, best_t, nodes_ptr, tlas_nodes, STEPS, TRIPS)
    needs = stop >= 0
    return best_t, best_tri, bnx, bny, bnz, bm1, bm2, bsf, needs, stop


@triton.jit
def instance_traversal(valid, ox, oy, oz, dx, dy, dz, last, best_t, best_tri, best_code, bnx, bny, bnz, box_hit,
                       start, nodes_ptr, tlas_nodes, inst_ptr, tri_ptr, tri_local_ptr, LEAF: tl.constexpr,
                       STEPS: tl.constexpr):
    """Top-level tree and instance meshes for the lanes in ``valid`` from node
    ``start`` of their octant's top-level copy (the root, or the leaf where
    ``top_level_query`` stopped), starting from the nearest hit found so far
    (the analytic boxes'). Mesh hits clear ``box_hit`` and set ``best_code``
    (the triangle's material/surface code)."""
    rox, roy, roz = ox, oy, oz
    rdx, rdy, rdz = dx, dy, dz
    wix = _inv(dx)
    wiy = _inv(dy)
    wiz = _inv(dz)
    rix, riy, riz = wix, wiy, wiz
    zf = tl.zeros(ox.shape, tl.float32)
    m00 = zf
    m01 = zf
    m02 = zf
    m10 = zf
    m11 = zf
    m12 = zf
    m20 = zf
    m21 = zf
    m22 = zf
    sgn = zf
    tri_off = tl.zeros(ox.shape, tl.int32)
    code_off = tl.zeros(ox.shape, tl.int32)
    mode = tl.zeros(ox.shape, tl.int32)  # 0: top level, 1: instance
    tlas_return = tl.full(ox.shape, -1, tl.int32)
    zi = tl.zeros(ox.shape, tl.int32)
    # Watertight-test constants of the local ray (set on instance entry).
    kz = zi
    nsx = zf
    nsy = zf
    sz = zf
    okx = zf
    oky = zf
    okz = zf
    node = tl.where(valid, start, -1)
    while tl.sum((node >= 0).to(tl.int32), axis=0) > 0:
        for _step in range(STEPS):
            active = node >= 0
            lx, ly, lz, ux, uy, uz, escape, leaf = _load_node(nodes_ptr, node)
            hit = active & _slab_hit(lx, ly, lz, ux, uy, uz, rox, roy, roz, rix, riy, riz, best_t)
            is_leaf = leaf >= 0
            first = leaf >> 4
            cnt = leaf & 15
            # Top-level leaf: enter the instance (rare: only when some lane does).
            enter = hit & is_leaf & (mode == 0)
            eroot = zi
            n_enter = tl.sum(enter.to(tl.int32), axis=0)
            if n_enter > 0:
                ib = first * INSTANCE_WIDTH
                e00 = tl.load(inst_ptr + ib + 0, mask=enter, other=0.)
                e01 = tl.load(inst_ptr + ib + 1, mask=enter, other=0.)
                e02 = tl.load(inst_ptr + ib + 2, mask=enter, other=0.)
                e10 = tl.load(inst_ptr + ib + 3, mask=enter, other=0.)
                e11 = tl.load(inst_ptr + ib + 4, mask=enter, other=0.)
                e12 = tl.load(inst_ptr + ib + 5, mask=enter, other=0.)
                e20 = tl.load(inst_ptr + ib + 6, mask=enter, other=0.)
                e21 = tl.load(inst_ptr + ib + 7, mask=enter, other=0.)
                e22 = tl.load(inst_ptr + ib + 8, mask=enter, other=0.)
                tdx = tl.load(inst_ptr + ib + 9, mask=enter, other=0.)
                tdy = tl.load(inst_ptr + ib + 10, mask=enter, other=0.)
                tdz = tl.load(inst_ptr + ib + 11, mask=enter, other=0.)
                esg = tl.load(inst_ptr + ib + 12, mask=enter, other=1.)
                eroot = _bits(tl.load(inst_ptr + ib + 13, mask=enter, other=0.))
                etri = _bits(tl.load(inst_ptr + ib + 14, mask=enter, other=0.))
                ecode = _bits(tl.load(inst_ptr + ib + 15, mask=enter, other=0.))
                wx, wy, wz = ox - tdx, oy - tdy, oz - tdz
                rox = tl.where(enter, e00 * wx + e01 * wy + e02 * wz, rox)
                roy = tl.where(enter, e10 * wx + e11 * wy + e12 * wz, roy)
                roz = tl.where(enter, e20 * wx + e21 * wy + e22 * wz, roz)
                rdx = tl.where(enter, e00 * dx + e01 * dy + e02 * dz, rdx)
                rdy = tl.where(enter, e10 * dx + e11 * dy + e12 * dz, rdy)
                rdz = tl.where(enter, e20 * dx + e21 * dy + e22 * dz, rdz)
                estride = _bits(tl.load(inst_ptr + ib + 18, mask=enter, other=0.))
                eroot = eroot + _octant(rdx, rdy, rdz) * estride
                rix = tl.where(enter, _inv(rdx), rix)
                riy = tl.where(enter, _inv(rdy), riy)
                riz = tl.where(enter, _inv(rdz), riz)
                ekz, ensx, ensy, esz, eokx, eoky, eokz = _shear(rdx, rdy, rdz, rox, roy, roz)
                kz = tl.where(enter, ekz, kz)
                nsx = tl.where(enter, ensx, nsx)
                nsy = tl.where(enter, ensy, nsy)
                sz = tl.where(enter, esz, sz)
                okx = tl.where(enter, eokx, okx)
                oky = tl.where(enter, eoky, oky)
                okz = tl.where(enter, eokz, okz)
                m00 = tl.where(enter, e00, m00)
                m01 = tl.where(enter, e01, m01)
                m02 = tl.where(enter, e02, m02)
                m10 = tl.where(enter, e10, m10)
                m11 = tl.where(enter, e11, m11)
                m12 = tl.where(enter, e12, m12)
                m20 = tl.where(enter, e20, m20)
                m21 = tl.where(enter, e21, m21)
                m22 = tl.where(enter, e22, m22)
                sgn = tl.where(enter, esg, sgn)
                tri_off = tl.where(enter, etri, tri_off)
                code_off = tl.where(enter, ecode, code_off)
            # Instance leaf: watertight test of up to LEAF triangles (local frame).
            tri_leaf = hit & is_leaf & (mode == 1)
            n_leaf = tl.sum(tri_leaf.to(tl.int32), axis=0)
            for k in range(LEAF * (n_leaf > 0).to(tl.int32)):
                m = tri_leaf & (k < cnt)
                ok, t, cx, cy, cz, local_id = _watertight(tri_ptr, tl.where(m, first + k, 0), m, kz, nsx, nsy, sz,
                                                          okx, oky, okz)
                gid = tri_off + local_id
                ok = ok & (t > 1e-6) & (t < best_t) & (gid != last)
                # World normal of the winding normal: sign(det R) * M^T (e1 x e2).
                best_t = tl.where(ok, t, best_t)
                best_tri = tl.where(ok, gid, best_tri)
                best_code = tl.where(ok, code_off + local_id, best_code)
                box_hit = box_hit & ~ok
                bnx = tl.where(ok, sgn * (m00 * cx + m10 * cy + m20 * cz), bnx)
                bny = tl.where(ok, sgn * (m01 * cx + m11 * cy + m21 * cz), bny)
                bnz = tl.where(ok, sgn * (m02 * cx + m12 * cy + m22 * cz), bnz)
            nxt = tl.where(hit & ~is_leaf, node + 1, escape)
            nxt = tl.where(enter, eroot, nxt)
            tlas_return = tl.where(enter, escape, tlas_return)
            mode = tl.where(enter, 1, mode)
            back = active & ~enter & (mode == 1) & (nxt < 0)
            nxt = tl.where(back, tlas_return, nxt)
            mode = tl.where(back, 0, mode)
            rox = tl.where(back, ox, rox)
            roy = tl.where(back, oy, roy)
            roz = tl.where(back, oz, roz)
            rdx = tl.where(back, dx, rdx)
            rdy = tl.where(back, dy, rdy)
            rdz = tl.where(back, dz, rdz)
            rix = tl.where(back, wix, rix)
            riy = tl.where(back, wiy, riy)
            riz = tl.where(back, wiz, riz)
            node = tl.where(active, nxt, node)
    return best_t, best_tri, best_code, bnx, bny, bnz, box_hit


@triton.jit
def merge_hit(valid, mesh_t, mesh_tri, bnx, bny, bnz, m1, m2, sidx,
              wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz):
    """Chroma's mesh/wire merge (a wire wins when ``t_wire + 1e-6 < t_mesh``)
    and the unit normal of the chosen surface."""
    use_wire = valid & (wire_s >= 0) & (wire_t + 1e-6 < mesh_t)
    tri = tl.where(use_wire, -2, mesh_tri)
    dist = tl.where(use_wire, wire_t, mesh_t)
    nx = tl.where(use_wire, wnx, bnx)
    ny = tl.where(use_wire, wny, bny)
    nz = tl.where(use_wire, wnz, bnz)
    norm = tl.sqrt(nx * nx + ny * ny + nz * nz)
    inv_norm = tl.where(norm > 0., 1.0 / norm, 0.)
    m1 = tl.where(use_wire, wire_in, m1)
    m2 = tl.where(use_wire, wire_out, m2)
    sidx = tl.where(use_wire, wire_s, sidx)
    return dist, tri, nx * inv_norm, ny * inv_norm, nz * inv_norm, m1, m2, sidx


@triton.jit
def nearest_hit_kernel(
    rows_ptr, count_ptr,
    pos_ptr, dir_ptr, last_ptr,
    nodes_ptr, inst_ptr, tri_ptr, tri_local_ptr,
    code_m1_ptr, code_m2_ptr, code_s_ptr,
    wires_ptr, n_wires, boxes_ptr, n_boxes, box_tris_ptr,
    out_t, out_tri, out_n, out_codes,
    capacity, wire_slots, wire_count,
    LEAF: tl.constexpr, WIRE_MODE: tl.constexpr, BLOCK: tl.constexpr, FACE_TRIS: tl.constexpr = 8,
    STEPS: tl.constexpr = 4, STATS: tl.constexpr = False,
    stats_ptr=None, blas_slots=None, blas_count=None, PHASE: tl.constexpr = 0,
    LEGACY_WIRES: tl.constexpr = False, tlas_nodes=0, BOX_SNAP: tl.constexpr = False,
):
    """PHASE 0: complete query. PHASE 1: analytic boxes and the top level only;
    rays that reach an instance are queued in ``blas_slots`` (wire candidates
    use the box distance as a conservative cap). PHASE 2: complete traversal
    for the queued slots (``count_ptr``/``blas_slots``), starting from the
    phase-1 result stored in the outputs; only mesh improvements are stored."""
    tl.static_assert(PHASE != 1 or WIRE_MODE == 1, "phase 1 defers wires to wire_kernel")
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(count_ptr)
    if tl.program_id(0) * BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    if PHASE == 2:
        slot = tl.load(blas_slots + lane, mask=valid, other=0)
    else:
        slot = lane
    row = tl.load(rows_ptr + slot, mask=valid, other=0)
    ox = tl.load(pos_ptr + row * 3 + 0, mask=valid, other=0.)
    oy = tl.load(pos_ptr + row * 3 + 1, mask=valid, other=0.)
    oz = tl.load(pos_ptr + row * 3 + 2, mask=valid, other=0.)
    dx = tl.load(dir_ptr + row * 3 + 0, mask=valid, other=1.)
    dy = tl.load(dir_ptr + row * 3 + 1, mask=valid, other=0.)
    dz = tl.load(dir_ptr + row * 3 + 2, mask=valid, other=0.)
    last = tl.load(last_ptr + row, mask=valid, other=-1)

    # Current ray (world in top-level mode, local inside an instance).
    rox, roy, roz = ox, oy, oz
    rdx, rdy, rdz = dx, dy, dz
    best_t = tl.full((BLOCK,), float("inf"), tl.float32)
    best_tri = tl.full((BLOCK,), -1, tl.int32)
    best_code = tl.full((BLOCK,), 0, tl.int32)
    bnx = tl.zeros((BLOCK,), tl.float32)
    bny = tl.zeros((BLOCK,), tl.float32)
    bnz = tl.zeros((BLOCK,), tl.float32)
    # Instance registers.
    m00 = tl.zeros((BLOCK,), tl.float32)
    m01 = tl.zeros((BLOCK,), tl.float32)
    m02 = tl.zeros((BLOCK,), tl.float32)
    m10 = tl.zeros((BLOCK,), tl.float32)
    m11 = tl.zeros((BLOCK,), tl.float32)
    m12 = tl.zeros((BLOCK,), tl.float32)
    m20 = tl.zeros((BLOCK,), tl.float32)
    m21 = tl.zeros((BLOCK,), tl.float32)
    m22 = tl.zeros((BLOCK,), tl.float32)
    sgn = tl.zeros((BLOCK,), tl.float32)
    tri_off = tl.zeros((BLOCK,), tl.int32)
    code_off = tl.zeros((BLOCK,), tl.int32)
    mode = tl.zeros((BLOCK,), tl.int32)  # 0: top level, 1: instance
    tlas_return = tl.full((BLOCK,), -1, tl.int32)
    node = tl.where(valid, _octant(dx, dy, dz) * tlas_nodes, -1)
    # Watertight-test constants of the local ray (set on instance entry).
    kz = tl.zeros((BLOCK,), tl.int32)
    nsx = tl.zeros((BLOCK,), tl.float32)
    nsy = tl.zeros((BLOCK,), tl.float32)
    sz = tl.zeros((BLOCK,), tl.float32)
    okx = tl.zeros((BLOCK,), tl.float32)
    oky = tl.zeros((BLOCK,), tl.float32)
    okz = tl.zeros((BLOCK,), tl.float32)
    # Box codes are carried directly (not through the variant tables).
    box_hit = tl.zeros((BLOCK,), tl.int1)
    bm1 = tl.zeros((BLOCK,), tl.int32)
    bm2 = tl.zeros((BLOCK,), tl.int32)
    bsf = tl.full((BLOCK,), -1, tl.int32)

    if PHASE == 2:
        # Continue from the phase-1 (analytic box) result.
        best_t = tl.load(out_t + slot, mask=valid, other=float("inf"))
        best_tri = tl.load(out_tri + slot, mask=valid, other=-1)
        bnx = tl.load(out_n + slot * 3 + 0, mask=valid, other=0.)
        bny = tl.load(out_n + slot * 3 + 1, mask=valid, other=0.)
        bnz = tl.load(out_n + slot * 3 + 2, mask=valid, other=0.)
        bm1 = tl.load(out_codes + slot * 3 + 0, mask=valid, other=0)
        bm2 = tl.load(out_codes + slot * 3 + 1, mask=valid, other=0)
        bsf = tl.load(out_codes + slot * 3 + 2, mask=valid, other=-1)
        best_t = tl.where(best_tri == -1, float("inf"), best_t)
        box_hit = valid & (best_tri != -1)
    needs_blas = tl.zeros((BLOCK,), tl.int1)

    # ---- analytic boxes first: their distance prunes the top level ----------
    inside = tl.zeros((BLOCK,), tl.int32)
    for box_index in range(n_boxes if PHASE != 2 else 0):
        best_t, best_tri, bnx, bny, bnz, box_hit, bm1, bm2, bsf, inside = _analytic_box(
            boxes_ptr, box_tris_ptr, box_index, valid, ox, oy, oz, dx, dy, dz, _inv_exact(dx), _inv_exact(dy),
            _inv_exact(dz), last,
            best_t, best_tri, bnx, bny, bnz, box_hit, bm1, bm2, bsf, inside, BOX_SNAP)

    visits = tl.zeros((BLOCK,), tl.int32)
    tri_tests = tl.zeros((BLOCK,), tl.int32)
    blas_visits = tl.zeros((BLOCK,), tl.int32)
    while tl.sum((node >= 0).to(tl.int32), axis=0) > 0:
        for _step in range(STEPS):
            active = node >= 0
            if STATS:
                visits += active.to(tl.int32)
                blas_visits += (active & (mode == 1)).to(tl.int32)
            lx, ly, lz, ux, uy, uz, escape, leaf = _load_node(nodes_ptr, node)
            # Slab test; zero direction components never limit the interval.
            hit = active & _slab_hit(lx, ly, lz, ux, uy, uz, rox, roy, roz, _inv_exact(rdx), _inv_exact(rdy),
                                     _inv_exact(rdz), best_t)
            is_leaf = leaf >= 0
            first = leaf >> 4
            cnt = leaf & 15

            if PHASE == 1:
                # Top level only: a ray that reaches an instance is queued for phase 2.
                reach = hit & is_leaf
                needs_blas = needs_blas | reach
                nxt = tl.where(hit & ~is_leaf, node + 1, escape)
                node = tl.where(active, tl.where(reach, -1, nxt), node)
            else:
                # Top-level leaf: enter the instance.
                enter = hit & is_leaf & (mode == 0)
                # (Lanes not entering read instance 0; every use below is selected by ``enter``.)
                ie = tl.where(enter, first, 0)
                e00, e01, e02, e10, e11, e12, e20, e21, e22, tdx, tdy, tdz, esg, eroot, etri, ecode = _ld_inst(
                    inst_ptr, ie)
                wx, wy, wz = ox - tdx, oy - tdy, oz - tdz
                if (PHASE == 0) and (LEAF >= PIN_LEAF_MIN):
                    # The complete query with leaves of PIN_LEAF_MIN+ triangles: the transform rounded exactly as
                    # LLVM contracted it in these specializations of the parent tree (it rounds the e_i0 product
                    # of rows 0 and 2 of the origin and row 0 of the direction), pinned so that engine.query
                    # stays bitwise the same; elsewhere the parent's contraction is the one below's.
                    rox = tl.where(enter, _fma_rn(e02, wz, _fma_rn(e01, wy, _mul_rn(e00, wx))), rox)
                    roy = tl.where(enter, _fma_rn(e12, wz, _fma_rn(e10, wx, _mul_rn(e11, wy))), roy)
                    roz = tl.where(enter, _fma_rn(e22, wz, _fma_rn(e21, wy, _mul_rn(e20, wx))), roz)
                    rdx = tl.where(enter, _fma_rn(e02, dz, _fma_rn(e01, dy, _mul_rn(e00, dx))), rdx)
                    rdy = tl.where(enter, _fma_rn(e12, dz, _fma_rn(e10, dx, _mul_rn(e11, dy))), rdy)
                    rdz = tl.where(enter, _fma_rn(e22, dz, _fma_rn(e20, dx, _mul_rn(e21, dy))), rdz)
                else:
                    rox = tl.where(enter, e00 * wx + e01 * wy + e02 * wz, rox)
                    roy = tl.where(enter, e10 * wx + e11 * wy + e12 * wz, roy)
                    roz = tl.where(enter, e20 * wx + e21 * wy + e22 * wz, roz)
                    rdx = tl.where(enter, e00 * dx + e01 * dy + e02 * dz, rdx)
                    rdy = tl.where(enter, e10 * dx + e11 * dy + e12 * dz, rdy)
                    rdz = tl.where(enter, e20 * dx + e21 * dy + e22 * dz, rdz)
                eroot = eroot + _octant(rdx, rdy, rdz) * _bits(tl.load(inst_ptr + ie * INSTANCE_WIDTH + 18))
                ekz, ensx, ensy, esz, eokx, eoky, eokz = _shear(rdx, rdy, rdz, rox, roy, roz)
                kz = tl.where(enter, ekz, kz)
                nsx = tl.where(enter, ensx, nsx)
                nsy = tl.where(enter, ensy, nsy)
                sz = tl.where(enter, esz, sz)
                okx = tl.where(enter, eokx, okx)
                oky = tl.where(enter, eoky, oky)
                okz = tl.where(enter, eokz, okz)
                m00 = tl.where(enter, e00, m00)
                m01 = tl.where(enter, e01, m01)
                m02 = tl.where(enter, e02, m02)
                m10 = tl.where(enter, e10, m10)
                m11 = tl.where(enter, e11, m11)
                m12 = tl.where(enter, e12, m12)
                m20 = tl.where(enter, e20, m20)
                m21 = tl.where(enter, e21, m21)
                m22 = tl.where(enter, e22, m22)
                sgn = tl.where(enter, esg, sgn)
                tri_off = tl.where(enter, etri, tri_off)
                code_off = tl.where(enter, ecode, code_off)

                # Instance leaf: watertight test of up to LEAF triangles (local frame).
                tri_leaf = hit & is_leaf & (mode == 1)
                for k in tl.static_range(LEAF):
                    m = tri_leaf & (k < cnt)
                    ok, t, cx, cy, cz, local_id = _watertight(tri_ptr, tl.where(m, first + k, 0), m, kz, nsx, nsy,
                                                              sz, okx, oky, okz)
                    gid = tri_off + local_id
                    if STATS:
                        tri_tests += m.to(tl.int32)
                    ok = ok & (t > 1e-6) & (t < best_t) & (gid != last)
                    # World normal of the winding normal: sign(det R) * M^T (e1 x e2).
                    wnx = sgn * (m00 * cx + m10 * cy + m20 * cz)
                    wny = sgn * (m01 * cx + m11 * cy + m21 * cz)
                    wnz = sgn * (m02 * cx + m12 * cy + m22 * cz)
                    best_t = tl.where(ok, t, best_t)
                    best_tri = tl.where(ok, gid, best_tri)
                    best_code = tl.where(ok, code_off + local_id, best_code)
                    box_hit = box_hit & ~ok
                    bnx = tl.where(ok, wnx, bnx)
                    bny = tl.where(ok, wny, bny)
                    bnz = tl.where(ok, wnz, bnz)

                nxt = tl.where(hit & ~is_leaf, node + 1, escape)
                nxt = tl.where(enter, eroot, nxt)
                tlas_return = tl.where(enter, escape, tlas_return)
                mode = tl.where(enter, 1, mode)
                back = active & ~enter & (mode == 1) & (nxt < 0)
                nxt = tl.where(back, tlas_return, nxt)
                mode = tl.where(back, 0, mode)
                rox = tl.where(back, ox, rox)
                roy = tl.where(back, oy, roy)
                roz = tl.where(back, oz, roz)
                rdx = tl.where(back, dx, rdx)
                rdy = tl.where(back, dy, rdy)
                rdz = tl.where(back, dz, rdz)
                node = tl.where(active, nxt, node)

    if STATS:
        tl.store(stats_ptr + lane * 3 + 0, visits, mask=valid)
        tl.store(stats_ptr + lane * 3 + 1, blas_visits, mask=valid)
        tl.store(stats_ptr + lane * 3 + 2, tri_tests, mask=valid)
    mesh_t = best_t
    from_mesh = valid & (best_tri >= 0) & ~box_hit
    m1 = tl.where(box_hit, bm1, tl.load(code_m1_ptr + best_code, mask=from_mesh, other=0))
    m2 = tl.where(box_hit, bm2, tl.load(code_m2_ptr + best_code, mask=from_mesh, other=0))
    sidx = tl.where(box_hit, bsf, tl.load(code_s_ptr + best_code, mask=from_mesh, other=-1))

    if PHASE == 1:
        queued = valid & needs_blas
        sel = queued.to(tl.int32)
        tot = tl.sum(sel, axis=0)
        off = tl.cumsum(sel, axis=0) - sel
        base_q = tl.atomic_add(blas_count, tot)
        tl.store(blas_slots + base_q + off, lane, mask=queued)
    cap = tl.where(best_tri >= 0, best_t, 1e30)
    if PHASE == 2:
        wire_t = tl.full((BLOCK,), 1e30, tl.float32)
        wire_s = tl.full((BLOCK,), -1, tl.int32)
        wire_in = tl.zeros((BLOCK,), tl.int32)
        wire_out = tl.zeros((BLOCK,), tl.int32)
        wnx = tl.zeros((BLOCK,), tl.float32)
        wny = tl.zeros((BLOCK,), tl.float32)
        wnz = tl.zeros((BLOCK,), tl.float32)
    elif WIRE_MODE == 0:
        wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz, _wid = _all_wires(
            valid, ox, oy, oz, dx, dy, dz, cap, wires_ptr, n_wires, tl.zeros(ox.shape, tl.int32) - 1, LEGACY_WIRES)
    else:
        # Mesh-only pass: queue the slots whose rays may reach a wire slab.
        candidate = _wire_candidate(valid, ox, oy, oz, dx, dy, dz, cap, wires_ptr, n_wires)
        selected = candidate.to(tl.int32)
        total = tl.sum(selected, axis=0)
        offset = tl.cumsum(selected, axis=0) - selected
        start = tl.atomic_add(wire_count, total)
        tl.store(wire_slots + start + offset, lane, mask=candidate)
        wire_t = tl.full((BLOCK,), 1e30, tl.float32)
        wire_s = tl.full((BLOCK,), -1, tl.int32)
        wire_in = tl.zeros((BLOCK,), tl.int32)
        wire_out = tl.zeros((BLOCK,), tl.int32)
        wnx = tl.zeros((BLOCK,), tl.float32)
        wny = tl.zeros((BLOCK,), tl.float32)
        wnz = tl.zeros((BLOCK,), tl.float32)
    if PHASE == 2:
        store = from_mesh
    else:
        store = valid
    dist, tri, nx, ny, nz, m1, m2, sidx = merge_hit(valid, mesh_t, best_tri, bnx, bny, bnz, m1, m2, sidx,
                                                   wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz)
    tl.store(out_t + slot, dist, mask=store)
    tl.store(out_tri + slot, tri, mask=store)
    tl.store(out_n + slot * 3 + 0, nx, mask=store)
    tl.store(out_n + slot * 3 + 1, ny, mask=store)
    tl.store(out_n + slot * 3 + 2, nz, mask=store)
    tl.store(out_codes + slot * 3 + 0, m1, mask=store)
    tl.store(out_codes + slot * 3 + 1, m2, mask=store)
    tl.store(out_codes + slot * 3 + 2, sidx, mask=store)


NO_WIRE = tl.constexpr(-2147483647)  # a wire index no wire has


@triton.jit
def _wire_plane(go, wx, wy, wz, dx, dy, dz, dn, wn0, uux, uuy, uuz, vvx, vvy, vvz, nnx, nny, nnz,
                pitch, radius, umin, umax, v0, kmin, kmax, psurf, pin, pout, cap,
                wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz, skip_k, hit_k, LEGACY: tl.constexpr,
                inv_pitch_c=0., pad_c=0., r2_c=0., eps0_c=0., span_c=0.):
    """Wire intersection for one plane (lanes in ``go``).

    ``LEGACY`` reproduces the installed Chroma FP32 algorithm. It tests every
    wire between the ray origin's v coordinate and the slab exit, and forms
    the discriminant as ``B*B - A*C``: for a wire at distance ``t`` both terms
    are ~``t*t`` while their difference is ~``r*r``, so beyond a few hundred mm
    FP32 rounding (~1e-7 * t*t) exceeds ``r*r`` and far hits are decided by
    noise. The production form tests only the wires the padded slab can reach
    (one extra wire each side covers rounding), uses the identical
    ``A*r*r - (wv*dn - wn0*dv)**2`` (no cancellation), and takes the hit
    normal from the same decomposition.

    The production form also ignores wire ``skip_k`` of this plane: the wire
    a photon has just left outward (a ray leaving a convex cylinder cannot
    meet it again, but at metre-scale coordinates the FP32 surface point can
    fall just inside, where the exit would be taken for a hit from inside).
    ``hit_k`` returns the index of the wire hit (unchanged where none is).
    """
    du = dx * uux + dy * uuy + dz * uuz
    dv = dx * vvx + dy * vvy + dz * vvz
    wu = wx * uux + wy * uuy + wz * uuz
    wv0 = wx * vvx + wy * vvy + wz * vvz - v0
    flat_u = tl.abs(du) < 1e-7
    go = go & ~(flat_u & ((wu < umin) | (wu > umax)))
    A = dv * dv + dn * dn
    flat_n = tl.abs(dn) <= 1e-7
    if LEGACY:
        t1 = (umin - wu) / du
        t2 = (umax - wu) / du
        inv_pitch = tl.where(pitch != 0., 1.0 / pitch, 0.)
        pad = radius + 1e-5
        tn1 = (-pad - wn0) / dn
        tn2 = (pad - wn0) / dn
        r2 = radius * radius
        eps0 = tl.maximum(1e-12, 1e-6 * r2)
    else:
        # The slab bounds below are padded (pad, err), so approximate
        # reciprocals do; the plane's constants come precomputed.
        idu = _rcp(du)
        t1 = (umin - wu) * idu
        t2 = (umax - wu) * idu
        inv_pitch = inv_pitch_c
        pad = pad_c
        idn = _rcp(dn)
        tn1 = (-pad - wn0) * idn
        tn2 = (pad - wn0) * idn
        r2 = r2_c
        eps0 = eps0_c
    t_in = tl.where(flat_u, -1.0e30, tl.minimum(t1, t2))
    t_out = tl.where(flat_u, 1.0e30, tl.maximum(t1, t2))
    go = go & (t_in <= t_out)
    t_lo = tl.maximum(t_in, 1.0e-4)
    t_hi = tl.minimum(t_out, cap)
    t_lo = tl.where(flat_n, t_lo, tl.maximum(t_lo, tl.minimum(tn1, tn2)))
    t_hi = tl.where(flat_n, t_hi, tl.minimum(t_hi, tl.maximum(tn1, tn2)))
    go = go & ~(flat_n & (tl.abs(wn0) > pad))
    go = go & (t_hi >= t_lo)
    span = flat_n & (tl.abs(dv) > 1e-7)
    if LEGACY:
        t_hi = tl.where(span, tl.minimum(t_hi, t_lo + (pitch + 2.0 * radius) / tl.abs(dv)), t_hi)
    else:
        t_hi = tl.where(span, tl.minimum(t_hi, t_lo + span_c * _rcp(tl.abs(dv))), t_hi)
    v_entry = wv0 + dv * t_lo
    v_exit = wv0 + dv * t_hi
    v_lo = tl.minimum(v_entry, v_exit) - pad
    v_hi = tl.maximum(v_entry, v_exit) + pad
    if LEGACY:
        v_lo = tl.minimum(v_lo, wv0 - pad)
        v_hi = tl.maximum(v_hi, wv0 + pad)
        k_lo = tl.maximum(tl.floor(v_lo * inv_pitch).to(tl.int32), kmin)
        k_hi = tl.minimum(tl.ceil(v_hi * inv_pitch).to(tl.int32), kmax)
        go = go & (kmin <= kmax) & (k_lo <= k_hi)
        k = k_lo
        n_iter = tl.max(tl.where(go, k_hi - k_lo + 1, 0), axis=0)
        for _it in range(n_iter):
            live = go & (k <= k_hi)
            wv = wv0 - k.to(tl.float32) * pitch
            B = wv * dv + wn0 * dn
            C = wv * wv + wn0 * wn0 - r2
            disc = B * B - A * C
            live2 = live & (disc >= 0.)
            sq = tl.sqrt(tl.maximum(disc, 0.))
            t_small = (-B - sq) / A
            t_large = (-B + sq) / A
            r20 = wv * wv + wn0 * wn0
            outside = r20 > r2 + eps0
            inside = r20 < r2 - eps0
            t = tl.where(outside, t_small, tl.where(inside, t_large, 1.0e-4))
            live2 = live2 & ~(outside & (t_small <= 1.0e-4)) & ~(inside & (t_large <= 1.0e-4))
            uc = wu + du * t
            live2 = live2 & (uc >= umin) & (uc <= umax) & (t < wire_t) & (t >= t_in) & (t <= t_out)
            vn = wv + dv * t
            nn = wn0 + dn * t
            length = tl.sqrt(vn * vn + nn * nn)
            live2 = live2 & (length > 0.)
            il = 1.0 / length
            hx = (vn * il) * vvx + (nn * il) * nnx
            hy = (vn * il) * vvy + (nn * il) * nny
            hz = (vn * il) * vvz + (nn * il) * nnz
            wire_t = tl.where(live2, t, wire_t)
            hit_k = tl.where(live2, k, hit_k)
            wire_s = tl.where(live2, psurf, wire_s)
            wire_in = tl.where(live2, pin, wire_in)
            wire_out = tl.where(live2, pout, wire_out)
            wnx = tl.where(live2, hx, wnx)
            wny = tl.where(live2, hy, wny)
            wnz = tl.where(live2, hz, wnz)
            k += 1
    else:
        # Wires whose axis lies within the padded slab's v extent, widened by
        # a bound on its FP32 error (a few 1e-7 of the coordinates involved).
        err = 1e-3 + 4e-7 * (tl.abs(v_entry) + tl.abs(v_exit) + tl.minimum(tl.abs(t_hi), 1e6))
        k_lo = tl.maximum(tl.ceil((v_lo - err) * inv_pitch).to(tl.int32), kmin)
        k_hi = tl.minimum(tl.floor((v_hi + err) * inv_pitch).to(tl.int32), kmax)
        go = go & (kmin <= kmax) & (k_lo <= k_hi)
        k = k_lo
        inv_a = 1.0 / A
        best_k = k_lo
        found = go & False
        n_iter = tl.max(tl.where(go, k_hi - k_lo + 1, 0), axis=0)
        for _it in range(n_iter):
            live = go & (k <= k_hi)
            wv = wv0 - k.to(tl.float32) * pitch
            B = wv * dv + wn0 * dn
            cross = wv * dn - wn0 * dv
            disc = A * r2 - cross * cross
            sq = tl.sqrt(tl.maximum(disc, 0.))
            t_small = (-B - sq) * inv_a
            t_large = (-B + sq) * inv_a
            r20 = wv * wv + wn0 * wn0
            outside = r20 > r2 + eps0
            inside = r20 < r2 - eps0
            t = tl.where(outside, t_small, tl.where(inside, t_large, 1.0e-4))
            live2 = live & (disc >= 0.) & ~(outside & (t_small <= 1.0e-4)) & ~(inside & (t_large <= 1.0e-4))
            live2 = live2 & (k != skip_k)
            uc = wu + du * t
            live2 = live2 & (uc >= umin) & (uc <= umax) & (t < wire_t) & (t >= t_in) & (t <= t_out)
            wire_t = tl.where(live2, t, wire_t)
            best_k = tl.where(live2, k, best_k)
            found = found | live2
            k += 1
        # Normal of the chosen wire: the hit point relative to the wire axis
        # (times A) is the closest approach (dn, -dv) * cross, then -/+
        # sqrt(disc) along the ray; at the start boundary, the ray origin.
        n_found = _warp_any(found)
        if n_found != 0:
            wv = wv0 - best_k.to(tl.float32) * pitch
            cross = wv * dn - wn0 * dv
            sq = tl.sqrt(tl.maximum(A * r2 - cross * cross, 0.))
            r20 = wv * wv + wn0 * wn0
            outside = r20 > r2 + eps0
            inside = r20 < r2 - eps0
            s_sq = tl.where(outside, -sq, sq)
            vn = tl.where(outside | inside, dn * cross + dv * s_sq, wv + dv * wire_t)
            nn = tl.where(outside | inside, -dv * cross + dn * s_sq, wn0 + dn * wire_t)
            length = tl.sqrt(vn * vn + nn * nn)
            found = found & (length > 0.)
            il = 1.0 / length
            hit_k = tl.where(found, best_k, hit_k)
            wire_s = tl.where(found, psurf, wire_s)
            wire_in = tl.where(found, pin, wire_in)
            wire_out = tl.where(found, pout, wire_out)
            wnx = tl.where(found, (vn * il) * vvx + (nn * il) * nnx, wnx)
            wny = tl.where(found, (vn * il) * vvy + (nn * il) * nny, wny)
            wnz = tl.where(found, (vn * il) * vvz + (nn * il) * nnz, wnz)

    return wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz, hit_k


@triton.jit
def _load_plane(wires_ptr, ip):
    wb = ip * WIRE_WIDTH
    return (tl.load(wires_ptr + wb + 0), tl.load(wires_ptr + wb + 1), tl.load(wires_ptr + wb + 2),
            tl.load(wires_ptr + wb + 9), tl.load(wires_ptr + wb + 10), tl.load(wires_ptr + wb + 11),
            tl.load(wires_ptr + wb + 13))


@triton.jit
def _wire_candidate(valid, ox, oy, oz, dx, dy, dz, cap, wires_ptr, n_wires):
    """True where the ray may reach some wire plane's slab before ``cap``: the
    production per-plane cull of :func:`_all_wires`, which keeps every ray
    Chroma's cull keeps (so it also selects for the bug-compatible wires)."""
    any_plane = tl.zeros(ox.shape, tl.int1)
    for ip in range(n_wires):
        pox, poy, poz, nnx, nny, nnz, radius = _load_plane(wires_ptr, ip)
        dn = dx * nnx + dy * nny + dz * nnz
        wn0 = (ox - pox) * nnx + (oy - poy) * nny + (oz - poz) * nnz
        far_plane = tl.abs(wn0) > radius + 0.01
        away = far_plane & (dn * wn0 > 0.)
        too_far = far_plane & ~away & (tl.abs(wn0) - (radius + 0.01) > cap * tl.abs(dn))
        any_plane = any_plane | (valid & ~away & ~too_far)
    return any_plane


@triton.jit
def _all_wires(valid, ox, oy, oz, dx, dy, dz, cap, wires_ptr, n_wires, last_wire, LEGACY_WIRES: tl.constexpr):
    """Nearest analytic wire within ``cap``.

    ``last_wire`` (-1 for none) is the id of the wire the photon has just
    left outward, which the production form skips; the returned ``wire_id``
    identifies the wire hit (plane * 2**20 + wire index + 2**19).
    """
    wire_id = tl.full(ox.shape, -1, tl.int32)
    wire_t = tl.full(ox.shape, 1e30, tl.float32)
    wire_s = tl.full(ox.shape, -1, tl.int32)
    wire_in = tl.zeros(ox.shape, tl.int32)
    wire_out = tl.zeros(ox.shape, tl.int32)
    wnx = tl.zeros(ox.shape, tl.float32)
    wny = tl.zeros(ox.shape, tl.float32)
    wnz = tl.zeros(ox.shape, tl.float32)
    zi = tl.zeros(ox.shape, tl.int32)
    for ip in range(n_wires):
        # The plane record, 16 bytes per load (the same address in every lane).
        a = (wires_ptr + ip * WIRE_WIDTH + zi).to(tl.int64, bitcast=True)
        pox, poy, poz, uux = _ld4(a)
        uuy, uuz, vvx, vvy = _ld4(a + 16)
        vvz, nnx, nny, nnz = _ld4(a + 32)
        pitch, radius, umin, umax = _ld4(a + 48)
        v0, f17, f18, f19 = _ld4(a + 64)
        f20, f21, inv_pitch_c, pad_c = _ld4(a + 80)
        r2_c, eps0_c, span_c, _f27 = _ld4(a + 96)
        kmin = _bits(f17)
        kmax = _bits(f18)
        psurf = _bits(f19)
        pin = _bits(f20)
        pout = _bits(f21)

        wx = ox - pox
        wy = oy - poy
        wz = oz - poz
        dn = dx * nnx + dy * nny + dz * nnz
        wn0 = wx * nnx + wy * nny + wz * nnz
        far_plane = tl.abs(wn0) > radius + 0.01
        away = far_plane & (dn * wn0 > 0.)
        if LEGACY_WIRES:
            # Chroma's cull (photon.h): the distance to the axis plane against cap + radius.
            too_far = far_plane & ~away & (-wn0 / dn > cap + radius)
        else:
            # The ray enters the padded slab at (|wn0| - radius - 0.01) / |dn|. Chroma's
            # test above culls oblique rays that end inside the slab (on a wall the
            # wires pass through), which then reach wall points inside a wire.
            too_far = far_plane & ~away & (tl.abs(wn0) - (radius + 0.01) > cap * tl.abs(dn))
        go = valid & ~away & ~too_far
        if _warp_any(go) != 0:
            skip_k = tl.where((last_wire >= 0) & ((last_wire >> 20) == ip), (last_wire & 1048575) - 524288, NO_WIRE)
            wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz, hit_k = _wire_plane(
                go, wx, wy, wz, dx, dy, dz, dn, wn0, uux, uuy, uuz, vvx, vvy, vvz, nnx, nny, nnz,
                pitch, radius, umin, umax, v0, kmin, kmax, psurf, pin, pout, cap,
                wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz, skip_k, tl.zeros(ox.shape, tl.int32) + NO_WIRE,
                LEGACY_WIRES, inv_pitch_c, pad_c, r2_c, eps0_c, span_c)
            wire_id = tl.where(hit_k != NO_WIRE, ip * 1048576 + hit_k + 524288, wire_id)
    return wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz, wire_id


@triton.jit
def wire_kernel(slots_ptr, slot_count_ptr, capacity, rows_ptr, pos_ptr, dir_ptr, last_ptr, last_wire_ptr,
                out_t, out_tri, out_n, out_codes, hit_wire_ptr, wires_ptr, n_wires, BLOCK: tl.constexpr,
                LEGACY_WIRES: tl.constexpr = False):
    """Merge analytic wires into the mesh results of the compacted candidate slots.

    ``last_wire_ptr`` holds, per photon, the wire it has just left outward
    (skipped while its last event is that wire boundary); the id of the wire
    hit goes to ``hit_wire_ptr`` (per slot) for the step kernel."""
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(slot_count_ptr)
    if tl.program_id(0) * BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    slot = tl.load(slots_ptr + lane, mask=valid, other=0)
    row = tl.load(rows_ptr + slot, mask=valid, other=0)
    ox = tl.load(pos_ptr + row * 3 + 0, mask=valid, other=0.)
    oy = tl.load(pos_ptr + row * 3 + 1, mask=valid, other=0.)
    oz = tl.load(pos_ptr + row * 3 + 2, mask=valid, other=0.)
    dx = tl.load(dir_ptr + row * 3 + 0, mask=valid, other=1.)
    dy = tl.load(dir_ptr + row * 3 + 1, mask=valid, other=0.)
    dz = tl.load(dir_ptr + row * 3 + 2, mask=valid, other=0.)
    mesh_t = tl.load(out_t + slot, mask=valid, other=float("inf"))
    mesh_tri = tl.load(out_tri + slot, mask=valid, other=-1)
    cap = tl.where(mesh_tri >= 0, mesh_t, 1e30)
    last = tl.load(last_ptr + row, mask=valid, other=-1)
    last_wire = tl.load(last_wire_ptr + row, mask=valid, other=-1)
    wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz, wire_id = _all_wires(
        valid, ox, oy, oz, dx, dy, dz, cap, wires_ptr, n_wires, tl.where(last == -2, last_wire, -1), LEGACY_WIRES)
    use_wire = valid & (wire_s >= 0) & (wire_t + 1e-6 < mesh_t)
    tl.store(hit_wire_ptr + slot, wire_id, mask=use_wire)
    norm = tl.sqrt(wnx * wnx + wny * wny + wnz * wnz)
    inv_norm = tl.where(norm > 0., 1.0 / norm, 0.)
    tl.store(out_t + slot, wire_t, mask=use_wire)
    tl.store(out_tri + slot, tl.full(slot.shape, -2, tl.int32), mask=use_wire)
    tl.store(out_n + slot * 3 + 0, wnx * inv_norm, mask=use_wire)
    tl.store(out_n + slot * 3 + 1, wny * inv_norm, mask=use_wire)
    tl.store(out_n + slot * 3 + 2, wnz * inv_norm, mask=use_wire)
    tl.store(out_codes + slot * 3 + 0, wire_in, mask=use_wire)
    tl.store(out_codes + slot * 3 + 1, wire_out, mask=use_wire)
    tl.store(out_codes + slot * 3 + 2, wire_s, mask=use_wire)


@triton.jit
def _analytic_box(boxes_ptr, box_tris_ptr, ib, valid, ox, oy, oz, dx, dy, dz, ix, iy, iz, last,
                  best_t, best_tri, bnx, bny, bnz, box_hit, bm1, bm2, bsf, inside, BOX_SNAP: tl.constexpr = False):
    """Crossing of an axis-aligned box whose faces are exact axis planes: the
    face by the slab method (``ix``..``iz``: inverse direction), the distance
    to its plane, and the face triangle containing the crossing point by its
    2D edge functions (exact Chroma triangle ids and codes).

    The face is the entry face, or the exit face when the ray starts inside
    the box or on its surface (its last hit is one of the box's triangles). A
    ray leaving the box surface it starts on cannot hit the box again. Every
    crossing gets a triangle: the first of the face that contains the point
    (edge functions of shared edges are exact negatives, see
    ``scene._box_face_triangle``), or, for a point rounded just outside the
    face, the least violated one.

    ``BOX_SNAP`` (production corrections): a ray that starts on the plane of
    its exit face (within rounding) without having just met that face meets
    it at distance 0. Coincident surfaces put photons there: one that leaves
    another solid through a face lying in the box's face (chroma-lar's PMT
    backs lie in the TPC walls) would otherwise pass through the box surface
    whenever the rounding of its position gives t <= 1e-6, as it does with
    Chroma's triangle test.
    """
    rb = ib * BOX_WIDTH
    lx = tl.load(boxes_ptr + rb + 0)
    ly = tl.load(boxes_ptr + rb + 1)
    lz = tl.load(boxes_ptr + rb + 2)
    ux = tl.load(boxes_ptr + rb + 3)
    uy = tl.load(boxes_ptr + rb + 4)
    uz = tl.load(boxes_ptr + rb + 5)
    g_first = _bits(tl.load(boxes_ptr + rb + 18))
    g_end = _bits(tl.load(boxes_ptr + rb + 19))
    t_near = tl.full(ox.shape, -float("inf"), tl.float32)
    t_far = tl.full(ox.shape, float("inf"), tl.float32)
    f_near = tl.full(ox.shape, -1, tl.int32)
    f_far = tl.full(ox.shape, -1, tl.int32)
    miss = ~valid
    for axis in tl.static_range(3):
        if axis == 0:
            lo, hi, o_a, d_a, i_a = lx, ux, ox, dx, ix
        elif axis == 1:
            lo, hi, o_a, d_a, i_a = ly, uy, oy, dy, iy
        else:
            lo, hi, o_a, d_a, i_a = lz, uz, oz, dz, iz
        t0 = (lo - o_a) * i_a
        t1 = (hi - o_a) * i_a
        pos = d_a > 0.
        flat = d_a == 0.
        near = tl.where(flat, -float("inf"), tl.where(pos, t0, t1))
        far = tl.where(flat, float("inf"), tl.where(pos, t1, t0))
        miss = miss | (flat & ((o_a < lo) | (o_a > hi)))
        upd = near > t_near
        t_near = tl.where(upd, near, t_near)
        f_near = tl.where(upd, tl.where(pos, 2 * axis, 2 * axis + 1), f_near)
        upd = far < t_far
        t_far = tl.where(upd, far, t_far)
        f_far = tl.where(upd, tl.where(pos, 2 * axis + 1, 2 * axis), f_far)
    miss = miss | (t_near > t_far)
    # Face of the triangle the ray starts on, if it is one of this box's.
    on_box = (last >= g_first) & (last < g_end)
    first_slot = _bits(tl.load(boxes_ptr + rb + 6))
    last_face = _bits(tl.load(box_tris_ptr + (first_slot + last - g_first) * BOX_TRI_WIDTH + 13, mask=on_box,
                              other=0.))
    last_face = tl.where(on_box, last_face, -1)
    # Entered through the face it starts on: the crossing is the exit face;
    # exiting through the face it starts on: it is leaving (no crossing).
    use_far = (t_near <= 0.) | (f_near == last_face)
    face = tl.where(use_far, f_far, f_near)
    leaving = use_far & (f_far == last_face)
    # Distance to the face plane (one rounded division).
    axis = tl.maximum(face, 0) // 2
    lower_side = face % 2 == 0
    a0 = axis == 0
    a1 = axis == 1
    plane = tl.where(lower_side, tl.where(a0, lx, tl.where(a1, ly, lz)), tl.where(a0, ux, tl.where(a1, uy, uz)))
    o_f = tl.where(a0, ox, tl.where(a1, oy, oz))
    d_f = tl.where(a0, dx, tl.where(a1, dy, dz))
    t = (plane - o_f) / d_f
    if BOX_SNAP:
        # "On the plane" means within the float32 rounding of that face's
        # coordinate (~8 ulp): a photon farther out, for example one that
        # scattered in the medium a micron outside a steel box, stays outside.
        # A snapped ray may start up to that far past the plane (t >= -1).
        d_face = tl.abs(d_f)
        tol_face = 1e-6 + 1e-6 * tl.abs(plane)
        snap = ~miss & use_far & ~leaving & (tl.abs(t) * d_face <= tol_face)
        # An entry face counts for any t > 0: rejecting one at t <= 1e-6 dropped
        # the box, and the ray passed through the solid. (Fix; the legacy path
        # below, CHROMA_TRITON=legacy, keeps the previous behaviour bit for bit.)
        above = tl.where(snap, (t * tl.maximum(d_face, 1e-6) > -tol_face) & (t > -1.0),
                         t > tl.where(use_far, 1e-6, 0.))
    else:
        above = t > 1e-6
    ok = valid & ~miss & ~leaving & (face >= 0) & above & (t < best_t)
    # Inside: the origin lies in all three slabs (t_near <= 0 < t_far) or enters through its face.
    inside = inside | ((valid & ~miss & use_far & ~leaving & (t_far > 0.)).to(tl.int32) << ib)
    if BOX_SNAP:
        t = tl.maximum(t, 0.)  # a snapped ray meets its face where it is
    # The face triangle that contains the crossing point (b, c): in-plane axes.
    pb = tl.fma(t, tl.where(a0, dy, tl.where(a1, dz, dx)), tl.where(a0, oy, tl.where(a1, oz, ox)))
    pc = tl.fma(t, tl.where(a0, dz, tl.where(a1, dx, dy)), tl.where(a0, oz, tl.where(a1, ox, oy)))
    first = _bits(tl.load(boxes_ptr + rb + 6 + 2 * tl.maximum(face, 0), mask=ok, other=0.))
    cnt = _bits(tl.load(boxes_ptr + rb + 7 + 2 * tl.maximum(face, 0), mask=ok, other=0.))
    cnt = tl.where(ok, cnt, 0)
    chosen = tl.where(ok, first, first_slot)
    best_e = tl.full(ox.shape, -float("inf"), tl.float32)
    for k in range(tl.max(cnt, axis=0)):
        m = ok & (k < cnt)
        s = tl.where(m, first + k, first_slot)
        a = (box_tris_ptr + s * BOX_TRI_WIDTH).to(tl.int64, bitcast=True)
        ea0, eb0, ec0, ea1 = _ld4(a)
        eb1, ec1, ea2, eb2 = _ld4(a + 16)
        e0 = tl.fma(ea0, pb, tl.fma(eb0, pc, ec0))
        e1 = tl.fma(ea1, pb, tl.fma(eb1, pc, ec1))
        e2 = tl.fma(ea2, pb, tl.fma(eb2, pc, tl.load(box_tris_ptr + s * BOX_TRI_WIDTH + 8, mask=m, other=0.)))
        e_min = tl.minimum(tl.minimum(e0, e1), e2)
        take = m & (best_e < 0.) & (e_min > best_e)
        chosen = tl.where(take, s, chosen)
        best_e = tl.where(take, e_min, best_e)
    a = (box_tris_ptr + chosen * BOX_TRI_WIDTH).to(tl.int64, bitcast=True)
    _c2, gid, m1, m2 = _ld4(a + 32)
    sf, _face, nsign, _pad = _ld4(a + 48)
    best_t = tl.where(ok, t, best_t)
    best_tri = tl.where(ok, _bits(gid), best_tri)
    bnx = tl.where(ok, tl.where(a0, nsign, 0.), bnx)
    bny = tl.where(ok, tl.where(a1, nsign, 0.), bny)
    bnz = tl.where(ok, tl.where(~a0 & ~a1, nsign, 0.), bnz)
    bm1 = tl.where(ok, _bits(m1), bm1)
    bm2 = tl.where(ok, _bits(m2), bm2)
    bsf = tl.where(ok, _bits(sf), bsf)
    box_hit = box_hit | ok
    return best_t, best_tri, bnx, bny, bnz, box_hit, bm1, bm2, bsf, inside


# Fields of a parked photon in the fused kernel's per-warp ring (engine/fused.py),
# stored field-major (field f of slot s at f * stride + s): the photon state,
# then the partial query (analytic boxes) and the top-level node where the walk
# stopped. batch_traversal replaces the query fields by the complete result.
F_X = tl.constexpr(0)
F_Y = tl.constexpr(1)
F_Z = tl.constexpr(2)
F_DX = tl.constexpr(3)
F_DY = tl.constexpr(4)
F_DZ = tl.constexpr(5)
F_PX = tl.constexpr(6)
F_PY = tl.constexpr(7)
F_PZ = tl.constexpr(8)
F_WL = tl.constexpr(9)
F_T = tl.constexpr(10)
F_LAST = tl.constexpr(11)
F_FLAGS = tl.constexpr(12)
F_W = tl.constexpr(13)
F_STEP = tl.constexpr(14)
F_IDLO = tl.constexpr(15)  # photon id, low and high 32 bits
F_ROW = tl.constexpr(16)
F_LWIRE = tl.constexpr(17)
F_BT = tl.constexpr(18)
F_BTRI = tl.constexpr(19)
F_NX = tl.constexpr(20)
F_NY = tl.constexpr(21)
F_NZ = tl.constexpr(22)
F_M1 = tl.constexpr(23)
F_M2 = tl.constexpr(24)
F_SF = tl.constexpr(25)
F_NODE = tl.constexpr(26)
F_IDHI = tl.constexpr(27)
RING_FIELDS = tl.constexpr(28)


@triton.jit
def _world_normal(inst_ptr, tri_ptr, inst, slot, m):
    """World winding normal sign(det R) * M^T (e1 x e2) of the local triangle in
    ``slot`` of instance ``inst`` (as the leaf test computes it) and the
    triangle's material/surface code (instance code offset + local id), in
    the lanes of ``m`` (the others read instance 0 and slot 0)."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22, _tdx, _tdy, _tdz, sgn, _root, _tri, code = _ld_inst(
        inst_ptr, tl.where(m, inst, 0))
    v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z, local_id = _ld_tri(tri_ptr, tl.where(m, slot, 0))
    e1x = v1x - v0x
    e1y = v1y - v0y
    e1z = v1z - v0z
    e2x = v2x - v0x
    e2y = v2y - v0y
    e2z = v2z - v0z
    cx = e1y * e2z - e1z * e2y
    cy = e1z * e2x - e1x * e2z
    cz = e1x * e2y - e1y * e2x
    return (sgn * (m00 * cx + m10 * cy + m20 * cz), sgn * (m01 * cx + m11 * cy + m21 * cz),
            sgn * (m02 * cx + m12 * cy + m22 * cz), code + local_id)


@triton.jit
def _in_core(px, py, pz, mask, cx, cy, cz, r2, hx, hy, hz):
    """The point lies in the mesh's empty core (scene._empty_core): its
    squared distance from c over the round axes (bits of ``mask``) is at
    most r2, and |p - c| is at most the half-extent on every axis. False for
    NaN or infinite points."""
    ux = px - cx
    uy = py - cy
    uz = pz - cz
    d2 = tl.where((mask & 1) != 0, ux * ux, 0.) + tl.where((mask & 2) != 0, uy * uy, 0.) + \
        tl.where((mask & 4) != 0, uz * uz, 0.)
    return (d2 <= r2) & (tl.abs(ux) <= hx) & (tl.abs(uy) <= hy) & (tl.abs(uz) <= hz)


@triton.jit
def batch_traversal(ring_f, ring_i, STRIDE: tl.constexpr, CAP: tl.constexpr, first, count,
                    nodes_ptr, inst_ptr, tri_ptr, tri_local_ptr, code_m1_ptr, code_m2_ptr, code_s_ptr,
                    LEAF: tl.constexpr, BLOCK: tl.constexpr, REFILL: tl.constexpr, PEND: tl.constexpr = 12,
                    INNER: tl.constexpr = 8, DEFER: tl.constexpr = True):
    """Complete the queries of ring entries ``first .. first+count-1`` (slot =
    entry mod CAP): the top-level walk from the node where ``top_level_query``
    stopped and the instance meshes, starting from the analytic-box result.
    BLOCK lanes work at once; a lane whose query is done stores its result
    (normal and material/surface codes of a mesh hit are formed then, from
    the hit's instance and triangle slot) and, once REFILL lanes are idle,
    takes the next entry, so the lanes stay busy until the last entries (Aila
    & Laine's persistent while-while traversal). Each query is exactly
    ``instance_traversal``'s.

    Every round stores finished queries, refills idle lanes, then runs INNER
    node steps in which a lane only follows its threaded tree: inner nodes
    and misses, recording its first mesh leaf, and returning from an instance
    mesh to the top level. A lane stops at a top-level leaf (an instance
    entry) or at a mesh leaf while it holds one. After the node steps the warp
    tests the recorded leaves when a lane waits (stopped holding one, or done
    holding one) or PEND lanes hold one, then enters instances: a lane stopped
    at a top-level leaf re-tests that leaf's box with its distance after the
    leaf tests and enters or moves past it.

    Leaf tests are postponed (Aila & Laine's speculative traversal): the leaf
    test needs only the constants set on instance entry; a lane tests its
    leaves in the order it reaches them and re-tests the node it waited at
    with the updated distance; a leaf box that passes a distance test passes
    it for all its ancestors, so the lane tests exactly the leaves, in the
    same order, as without postponement.

    An enclosing instance (instance word 19, ``scene.compile_scene``) is
    deferred: when the walk reaches its leaf the lane notes the leaf and walks
    on, and enters it after the rest of the walk, when the nearer instances'
    hits prune its mesh walk. The result stays the full walk's first hit in
    walk order among the nearest (an exact tie between the deferred mesh and
    another hit goes to the one the walk reaches first: the deferred mesh's
    hit wins a tie only against a hit accepted after its leaf's place in the
    walk). An instance whose mesh has an empty core (``scene._empty_core``)
    is not entered when the ray segment [0, best_t] (local frame) lies in the
    core: no triangle of the mesh can then be at or before best_t."""
    zf = tl.zeros((BLOCK,), tl.float32)
    zi = tl.zeros((BLOCK,), tl.int32)
    end = first + count
    cursor = first
    has = zi != 0
    slot = zi
    node = zi - 1
    mode = zi
    ox = zf
    oy = zf
    oz = zf
    dx = zf + 1.
    dy = zf
    dz = zf
    wix = zf
    wiy = zf
    wiz = zf
    rox = zf
    roy = zf
    roz = zf
    rix = zf
    riy = zf
    riz = zf
    kz = zi
    nsx = zf
    nsy = zf
    sz = zf
    okx = zf
    oky = zf
    okz = zf
    inst = zi
    tri_off = zi
    tlas_return = zi - 1
    last = zi - 1
    best_t = zf
    best_tri = zi - 1
    best_inst = zi
    best_slot = zi
    box_hit = has
    pend = zi - 1  # postponed leaf record (first * 16 + count), -1 none
    dnode = zi - 1  # deferred enclosing instance: its top-level leaf, -1 none
    dflag = zi  # bit 0: walking the deferred instance; bit 1: a hit was accepted after its leaf
    go_on = count
    while go_on > 0:
        # A lane whose walk is done (its recorded leaf tested) enters its deferred instance.
        resume = (node < 0) & (dnode >= 0)
        node = tl.where(resume, dnode, node)
        dnode = tl.where(resume, -1, dnode)
        dflag = tl.where(resume, dflag | 1, dflag)
        idle = node < 0
        # Finished queries: store the result.
        fin = has & idle
        any_fin = _warp_any(fin)
        if any_fin != 0:
            from_mesh = fin & (best_tri >= 0) & ~box_hit
            tl.store(ring_f + F_BT * STRIDE + slot, best_t, mask=fin, cache_modifier=".cg")
            tl.store(ring_i + F_BTRI * STRIDE + slot, best_tri, mask=fin, cache_modifier=".cg")
            any_mesh = _warp_any(from_mesh)
            if any_mesh != 0:
                nx, ny, nz, code = _world_normal(inst_ptr, tri_ptr, best_inst, best_slot, from_mesh)
                # The code tables are read before the ring is written (they cannot
                # alias, but the compiler does not know that), the other lanes at code 0.
                code = tl.where(from_mesh, code, 0)
                m1 = tl.load(code_m1_ptr + code)
                m2 = tl.load(code_m2_ptr + code)
                sf = tl.load(code_s_ptr + code)
                tl.store(ring_f + F_NX * STRIDE + slot, nx, mask=from_mesh, cache_modifier=".cg")
                tl.store(ring_f + F_NY * STRIDE + slot, ny, mask=from_mesh, cache_modifier=".cg")
                tl.store(ring_f + F_NZ * STRIDE + slot, nz, mask=from_mesh, cache_modifier=".cg")
                tl.store(ring_i + F_M1 * STRIDE + slot, m1, mask=from_mesh)
                tl.store(ring_i + F_M2 * STRIDE + slot, m2, mask=from_mesh)
                tl.store(ring_i + F_SF * STRIDE + slot, sf, mask=from_mesh)
            has = has & ~fin
        # Idle lanes take the next entries.
        n_idle = _warp_count(idle)
        left = end - cursor
        if (left > 0) & ((n_idle >= REFILL) | (n_idle == BLOCK)):
            rank = _lane_rank(idle)
            take = idle & (rank < left)
            s = _ring(cursor + rank, CAP)
            slot = tl.where(take, s, slot)
            nox = tl.load(ring_f + F_X * STRIDE + s, mask=take, other=0., cache_modifier=".cg")
            noy = tl.load(ring_f + F_Y * STRIDE + s, mask=take, other=0., cache_modifier=".cg")
            noz = tl.load(ring_f + F_Z * STRIDE + s, mask=take, other=0., cache_modifier=".cg")
            ndx = tl.load(ring_f + F_DX * STRIDE + s, mask=take, other=1., cache_modifier=".cg")
            ndy = tl.load(ring_f + F_DY * STRIDE + s, mask=take, other=0., cache_modifier=".cg")
            ndz = tl.load(ring_f + F_DZ * STRIDE + s, mask=take, other=0., cache_modifier=".cg")
            nbt = tl.load(ring_f + F_BT * STRIDE + s, mask=take, other=0., cache_modifier=".cg")
            nbtri = tl.load(ring_i + F_BTRI * STRIDE + s, mask=take, other=-1, cache_modifier=".cg")
            ox = tl.where(take, nox, ox)
            oy = tl.where(take, noy, oy)
            oz = tl.where(take, noz, oz)
            dx = tl.where(take, ndx, dx)
            dy = tl.where(take, ndy, dy)
            dz = tl.where(take, ndz, dz)
            nix = _inv(ndx)
            niy = _inv(ndy)
            niz = _inv(ndz)
            wix = tl.where(take, nix, wix)
            wiy = tl.where(take, niy, wiy)
            wiz = tl.where(take, niz, wiz)
            rox = tl.where(take, nox, rox)
            roy = tl.where(take, noy, roy)
            roz = tl.where(take, noz, roz)
            rix = tl.where(take, nix, rix)
            riy = tl.where(take, niy, riy)
            riz = tl.where(take, niz, riz)
            last = tl.where(take, tl.load(ring_i + F_LAST * STRIDE + s, mask=take, other=-1, cache_modifier=".cg"), last)
            best_t = tl.where(take, nbt, best_t)
            best_tri = tl.where(take, nbtri, best_tri)
            box_hit = tl.where(take, nbtri >= 0, box_hit)
            mode = tl.where(take, 0, mode)
            tlas_return = tl.where(take, -1, tlas_return)
            pend = tl.where(take, -1, pend)
            dnode = tl.where(take, -1, dnode)
            dflag = tl.where(take, 0, dflag)
            node = tl.where(take, tl.load(ring_i + F_NODE * STRIDE + s, mask=take, other=-1, cache_modifier=".cg"), node)
            has = has | take
            cursor += tl.minimum(n_idle, left)
        go_on = _warp_any(node >= 0)  # some lane active
        if go_on != 0:
            # ---- node steps: a lane stops at a top-level leaf, or at a mesh leaf while it holds one
            stopped = zi != 0
            for _i in range(INNER):
                active = (node >= 0) & ~stopped
                lx, ly, lz, ux, uy, uz, escape, leaf = _load_node(nodes_ptr, node)
                hit = active & _slab_hit(lx, ly, lz, ux, uy, uz, rox, roy, roz, rix, riy, riz, best_t)
                is_leaf = leaf >= 0
                at_leaf = hit & is_leaf
                rec = at_leaf & (mode == 1) & (pend < 0)
                pend = tl.where(rec, leaf, pend)
                stop = at_leaf & ~rec
                nxt = tl.where(hit & ~is_leaf, node + 1, escape)
                back = active & ~stop & (mode == 1) & (nxt < 0)
                nxt = tl.where(back, tlas_return, nxt)
                mode = tl.where(back, 0, mode)
                rox = tl.where(back, ox, rox)
                roy = tl.where(back, oy, roy)
                roz = tl.where(back, oz, roz)
                rix = tl.where(back, wix, rix)
                riy = tl.where(back, wiy, riy)
                riz = tl.where(back, wiz, riz)
                node = tl.where(active & ~stop, nxt, node)
                stopped = stopped | stop
            # ---- recorded leaves: tested once a lane waits (stopped holding one, or
            # done holding one) or PEND lanes hold one
            holds = pend >= 0
            at_top = stopped & (mode == 0)
            n_wait = _warp_any(holds & (stopped | (node < 0)))
            n_pend = _warp_count(holds)
            if (n_pend >= PEND) | (n_wait != 0):
                # The warp tests them together (coop_leaf_round), each lane's hits
                # accepted as in the loop over its triangles in order. A tie goes to
                # the hit the full walk reaches first: in the deferred instance's walk
                # (bit 0), the first of its hits, if the best hit was accepted after
                # the deferred leaf's place in the walk (bit 1). (The update of dflag
                # on an acceptance is idempotent: applied once after the round.)
                best_t, best_tri, best_inst, best_slot, box_hit, pend, acc = coop_leaf_round(
                    tri_ptr, pend, kz, nsx, nsy, sz, okx, oky, okz, tri_off, inst, last,
                    best_t, best_tri, best_inst, best_slot, box_hit, dflag == 3, BLOCK)
                dflag = tl.where(acc, tl.where((dflag & 1) != 0, 1, tl.where(dnode >= 0, 2, dflag)), dflag)
            # ---- instance entries: a lane at a top-level leaf re-tests it with its
            # current distance (it may have changed in the leaf tests) and enters
            n_top = _warp_any(at_top)
            if n_top != 0:
                lx, ly, lz, ux, uy, uz, escape, leaf = _load_node(nodes_ptr, tl.where(at_top, node, 0))
                enter = at_top & _slab_hit(lx, ly, lz, ux, uy, uz, ox, oy, oz, wix, wiy, wiz, best_t)
                # (Lanes not entering read instance 0; every use below is selected by ``enter``.)
                ie = tl.where(enter, leaf >> 4, 0)
                e00, e01, e02, e10, e11, e12, e20, e21, e22, tdx, tdy, tdz, _esg, eroot, etri, _ecode = _ld_inst(
                    inst_ptr, ie)
                a = (inst_ptr + ie * INSTANCE_WIDTH).to(tl.int64, bitcast=True)
                _fs, _f17, estride, fencl = _ld4(a + 64)
                encl = _bits(fencl)  # -1, or ENCLOSING (+ HAS_CORE + the core's round-axis mask)
                # The walk of the deferred instance (bit 0) ends after its mesh.
                phase = (dflag & 1) != 0
                ret = tl.where(phase, -1, escape)
                if DEFER:
                    defer = enter & ~phase & (dnode < 0) & (encl >= 0)
                else:
                    # CHROMA_TRITON=strict: enclosing instances in the walk's own order,
                    # so every box test sees the distance of the plain traversal.
                    defer = enter & False
                dnode = tl.where(defer, node, dnode)
                dflag = tl.where(defer, 0, dflag)
                enter = enter & ~defer
                wx, wy, wz = ox - tdx, oy - tdy, oz - tdz
                lox = e00 * wx + e01 * wy + e02 * wz
                loy = e10 * wx + e11 * wy + e12 * wz
                loz = e20 * wx + e21 * wy + e22 * wz
                ldx = e00 * dx + e01 * dy + e02 * dz
                ldy = e10 * dx + e11 * dy + e12 * dz
                ldz = e20 * dx + e21 * dy + e22 * dz
                # A mesh with an empty core cannot hold a nearer hit when the ray
                # segment [0, best_t] lies in the core (scene._empty_core).
                cored = enter & (encl >= 0) & ((encl & HAS_CORE) != 0)
                if _warp_any(cored) != 0:
                    ccx, ccy, ccz, cr2 = _ld4(a + 80)
                    chx, chy, chz, _f27 = _ld4(a + 96)
                    cmask = encl & 7
                    inside = cored & _in_core(lox, loy, loz, cmask, ccx, ccy, ccz, cr2, chx, chy, chz)
                    inside = inside & _in_core(lox + best_t * ldx, loy + best_t * ldy, loz + best_t * ldz,
                                               cmask, ccx, ccy, ccz, cr2, chx, chy, chz)
                    enter = enter & ~inside
                rox = tl.where(enter, lox, rox)
                roy = tl.where(enter, loy, roy)
                roz = tl.where(enter, loz, roz)
                eroot = eroot + _octant(ldx, ldy, ldz) * _bits(estride)
                rix = tl.where(enter, _inv(ldx), rix)
                riy = tl.where(enter, _inv(ldy), riy)
                riz = tl.where(enter, _inv(ldz), riz)
                ekz, ensx, ensy, esz, eokx, eoky, eokz = _shear(ldx, ldy, ldz, lox, loy, loz)
                kz = tl.where(enter, ekz, kz)
                nsx = tl.where(enter, ensx, nsx)
                nsy = tl.where(enter, ensy, nsy)
                sz = tl.where(enter, esz, sz)
                okx = tl.where(enter, eokx, okx)
                oky = tl.where(enter, eoky, oky)
                okz = tl.where(enter, eokz, okz)
                inst = tl.where(enter, ie, inst)
                tri_off = tl.where(enter, etri, tri_off)
                tlas_return = tl.where(enter, ret, tlas_return)
                mode = tl.where(enter, 1, mode)
                node = tl.where(at_top, tl.where(enter, eroot, ret), node)
