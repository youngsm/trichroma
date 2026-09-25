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

NODE_WIDTH = tl.constexpr(_scene.NODE_WIDTH)
INSTANCE_WIDTH = tl.constexpr(_scene.INSTANCE_WIDTH)
TRI_WIDTH = tl.constexpr(_scene.TRI_WIDTH)
WIRE_WIDTH = tl.constexpr(_scene.WIRE_WIDTH)
BOX_WIDTH = tl.constexpr(_scene.BOX_WIDTH)
BOX_TRI_WIDTH = tl.constexpr(_scene.BOX_TRI_WIDTH)


@triton.jit
def _bits(x):
    return x.to(tl.int32, bitcast=True)


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
    LEGACY_WIRES: tl.constexpr = False,
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
    node = tl.where(valid, 0, -1)
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
    for box_index in range(n_boxes if PHASE != 2 else 0):
        best_t, best_tri, bnx, bny, bnz, box_hit, bm1, bm2, bsf = _analytic_box(
            boxes_ptr, box_tris_ptr, box_index, valid, ox, oy, oz, dx, dy, dz, last,
            best_t, best_tri, bnx, bny, bnz, box_hit, bm1, bm2, bsf, FACE_TRIS)

    visits = tl.zeros((BLOCK,), tl.int32)
    tri_tests = tl.zeros((BLOCK,), tl.int32)
    blas_visits = tl.zeros((BLOCK,), tl.int32)
    while tl.sum((node >= 0).to(tl.int32), axis=0) > 0:
        for _step in range(STEPS):
            active = node >= 0
            if STATS:
                visits += active.to(tl.int32)
                blas_visits += (active & (mode == 1)).to(tl.int32)
            base = node * NODE_WIDTH
            lx = tl.load(nodes_ptr + base + 0, mask=active, other=0.)
            ly = tl.load(nodes_ptr + base + 1, mask=active, other=0.)
            lz = tl.load(nodes_ptr + base + 2, mask=active, other=0.)
            ux = tl.load(nodes_ptr + base + 3, mask=active, other=0.)
            uy = tl.load(nodes_ptr + base + 4, mask=active, other=0.)
            uz = tl.load(nodes_ptr + base + 5, mask=active, other=0.)
            escape = _bits(tl.load(nodes_ptr + base + 6, mask=active, other=0.))
            leaf = _bits(tl.load(nodes_ptr + base + 7, mask=active, other=0.))
            # Slab test; zero direction components never limit the interval.
            ix = 1.0 / rdx
            iy = 1.0 / rdy
            iz = 1.0 / rdz
            tx0 = (lx - rox) * ix
            tx1 = (ux - rox) * ix
            ty0 = (ly - roy) * iy
            ty1 = (uy - roy) * iy
            tz0 = (lz - roz) * iz
            tz1 = (uz - roz) * iz
            inx = (rox >= lx) & (rox <= ux)
            iny = (roy >= ly) & (roy <= uy)
            inz = (roz >= lz) & (roz <= uz)
            near_x = tl.where(rdx != 0., tl.minimum(tx0, tx1), tl.where(inx, -float("inf"), float("inf")))
            far_x = tl.where(rdx != 0., tl.maximum(tx0, tx1), tl.where(inx, float("inf"), -float("inf")))
            near_y = tl.where(rdy != 0., tl.minimum(ty0, ty1), tl.where(iny, -float("inf"), float("inf")))
            far_y = tl.where(rdy != 0., tl.maximum(ty0, ty1), tl.where(iny, float("inf"), -float("inf")))
            near_z = tl.where(rdz != 0., tl.minimum(tz0, tz1), tl.where(inz, -float("inf"), float("inf")))
            far_z = tl.where(rdz != 0., tl.maximum(tz0, tz1), tl.where(inz, float("inf"), -float("inf")))
            tnear = tl.maximum(tl.maximum(near_x, near_y), near_z)
            tfar = tl.minimum(tl.minimum(far_x, far_y), far_z)
            hit = active & (tnear <= tfar) & (tfar >= 0.) & (tnear <= best_t)
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

                # Instance leaf: Moller-Trumbore on up to LEAF triangles (local frame).
                tri_leaf = hit & is_leaf & (mode == 1)
                for k in tl.static_range(LEAF):
                    m = tri_leaf & (k < cnt)
                    s = (first + k) * TRI_WIDTH
                    v0x = tl.load(tri_ptr + s + 0, mask=m, other=0.)
                    v0y = tl.load(tri_ptr + s + 1, mask=m, other=0.)
                    v0z = tl.load(tri_ptr + s + 2, mask=m, other=0.)
                    e1x = tl.load(tri_ptr + s + 3, mask=m, other=0.)
                    e1y = tl.load(tri_ptr + s + 4, mask=m, other=0.)
                    e1z = tl.load(tri_ptr + s + 5, mask=m, other=0.)
                    e2x = tl.load(tri_ptr + s + 6, mask=m, other=0.)
                    e2y = tl.load(tri_ptr + s + 7, mask=m, other=0.)
                    e2z = tl.load(tri_ptr + s + 8, mask=m, other=0.)
                    local_id = tl.load(tri_local_ptr + first + k, mask=m, other=0)
                    gid = tri_off + local_id
                    hx = rdy * e2z - rdz * e2y
                    hy = rdz * e2x - rdx * e2z
                    hz = rdx * e2y - rdy * e2x
                    det = e1x * hx + e1y * hy + e1z * hz
                    inv = 1.0 / det
                    sx, sy, sz = rox - v0x, roy - v0y, roz - v0z
                    u = inv * (sx * hx + sy * hy + sz * hz)
                    qx = sy * e1z - sz * e1y
                    qy = sz * e1x - sx * e1z
                    qz = sx * e1y - sy * e1x
                    v = inv * (rdx * qx + rdy * qy + rdz * qz)
                    t = inv * (e2x * qx + e2y * qy + e2z * qz)
                    if STATS:
                        tri_tests += m.to(tl.int32)
                    ok = m & (det != 0.) & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
                    ok = ok & (t > 1e-6) & (t < best_t) & (gid != last)
                    # World normal of the winding normal: sign(det R) * M^T (e1 x e2).
                    cx = e1y * e2z - e1z * e2y
                    cy = e1z * e2x - e1x * e2z
                    cz = e1x * e2y - e1y * e2x
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
        wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz = _all_wires(valid, ox, oy, oz, dx, dy, dz, cap,
                                                                       wires_ptr, n_wires, LEGACY_WIRES)
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
    use_wire = valid & (wire_s >= 0) & (wire_t + 1e-6 < mesh_t)
    if PHASE == 2:
        store = from_mesh
    else:
        store = valid
    tri = tl.where(use_wire, -2, best_tri)
    dist = tl.where(use_wire, wire_t, mesh_t)
    nx = tl.where(use_wire, wnx, bnx)
    ny = tl.where(use_wire, wny, bny)
    nz = tl.where(use_wire, wnz, bnz)
    norm = tl.sqrt(nx * nx + ny * ny + nz * nz)
    inv_norm = tl.where(norm > 0., 1.0 / norm, 0.)
    m1 = tl.where(use_wire, wire_in, m1)
    m2 = tl.where(use_wire, wire_out, m2)
    sidx = tl.where(use_wire, wire_s, sidx)
    tl.store(out_t + slot, dist, mask=store)
    tl.store(out_tri + slot, tri, mask=store)
    tl.store(out_n + slot * 3 + 0, nx * inv_norm, mask=store)
    tl.store(out_n + slot * 3 + 1, ny * inv_norm, mask=store)
    tl.store(out_n + slot * 3 + 2, nz * inv_norm, mask=store)
    tl.store(out_codes + slot * 3 + 0, m1, mask=store)
    tl.store(out_codes + slot * 3 + 1, m2, mask=store)
    tl.store(out_codes + slot * 3 + 2, sidx, mask=store)


@triton.jit
def _wire_plane(go, wx, wy, wz, dx, dy, dz, dn, wn0, uux, uuy, uuz, vvx, vvy, vvz, nnx, nny, nnz,
                pitch, radius, umin, umax, v0, kmin, kmax, psurf, pin, pout, cap,
                wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz, LEGACY: tl.constexpr):
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
    """
    du = dx * uux + dy * uuy + dz * uuz
    dv = dx * vvx + dy * vvy + dz * vvz
    wu = wx * uux + wy * uuy + wz * uuz
    wv0 = wx * vvx + wy * vvy + wz * vvz - v0
    flat_u = tl.abs(du) < 1e-7
    go = go & ~(flat_u & ((wu < umin) | (wu > umax)))
    t1 = (umin - wu) / du
    t2 = (umax - wu) / du
    t_in = tl.where(flat_u, -1.0e30, tl.minimum(t1, t2))
    t_out = tl.where(flat_u, 1.0e30, tl.maximum(t1, t2))
    go = go & (t_in <= t_out)
    inv_pitch = tl.where(pitch != 0., 1.0 / pitch, 0.)
    pad = radius + 1e-5
    A = dv * dv + dn * dn
    t_lo = tl.maximum(t_in, 1.0e-4)
    t_hi = tl.minimum(t_out, cap)
    flat_n = tl.abs(dn) <= 1e-7
    tn1 = (-pad - wn0) / dn
    tn2 = (pad - wn0) / dn
    t_lo = tl.where(flat_n, t_lo, tl.maximum(t_lo, tl.minimum(tn1, tn2)))
    t_hi = tl.where(flat_n, t_hi, tl.minimum(t_hi, tl.maximum(tn1, tn2)))
    go = go & ~(flat_n & (tl.abs(wn0) > pad))
    go = go & (t_hi >= t_lo)
    span = flat_n & (tl.abs(dv) > 1e-7)
    t_hi = tl.where(span, tl.minimum(t_hi, t_lo + (pitch + 2.0 * radius) / tl.abs(dv)), t_hi)
    v_entry = wv0 + dv * t_lo
    v_exit = wv0 + dv * t_hi
    v_lo = tl.minimum(v_entry, v_exit) - pad
    v_hi = tl.maximum(v_entry, v_exit) + pad
    if LEGACY:
        v_lo = tl.minimum(v_lo, wv0 - pad)
        v_hi = tl.maximum(v_hi, wv0 + pad)
        k_lo = tl.maximum(tl.floor(v_lo * inv_pitch).to(tl.int32), kmin)
        k_hi = tl.minimum(tl.ceil(v_hi * inv_pitch).to(tl.int32), kmax)
    else:
        k_lo = tl.maximum(tl.floor(v_lo * inv_pitch).to(tl.int32) - 1, kmin)
        k_hi = tl.minimum(tl.ceil(v_hi * inv_pitch).to(tl.int32) + 1, kmax)
    go = go & (kmin <= kmax) & (k_lo <= k_hi)
    k = k_lo
    r2 = radius * radius
    eps0 = tl.maximum(1e-12, 1e-6 * r2)
    n_iter = tl.max(tl.where(go, k_hi - k_lo + 1, 0), axis=0)
    for _it in range(n_iter):
        live = go & (k <= k_hi)
        wv = wv0 - k.to(tl.float32) * pitch
        B = wv * dv + wn0 * dn
        if LEGACY:
            C = wv * wv + wn0 * wn0 - r2
            disc = B * B - A * C
        else:
            cross = wv * dn - wn0 * dv
            disc = A * r2 - cross * cross
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
        if LEGACY:
            vn = wv + dv * t
            nn = wn0 + dn * t
        else:
            # Hit point relative to the wire axis, times A: closest approach
            # (dn, -dv) * cross, then -/+ sqrt(disc) along the ray.
            s_sq = tl.where(outside, -sq, sq)
            vn = tl.where(outside | inside, dn * cross + dv * s_sq, wv + dv * t)
            nn = tl.where(outside | inside, -dv * cross + dn * s_sq, wn0 + dn * t)
        length = tl.sqrt(vn * vn + nn * nn)
        live2 = live2 & (length > 0.)
        il = 1.0 / length
        hx = (vn * il) * vvx + (nn * il) * nnx
        hy = (vn * il) * vvy + (nn * il) * nny
        hz = (vn * il) * vvz + (nn * il) * nnz
        wire_t = tl.where(live2, t, wire_t)
        wire_s = tl.where(live2, psurf, wire_s)
        wire_in = tl.where(live2, pin, wire_in)
        wire_out = tl.where(live2, pout, wire_out)
        wnx = tl.where(live2, hx, wnx)
        wny = tl.where(live2, hy, wny)
        wnz = tl.where(live2, hz, wnz)
        k += 1

    return wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz


@triton.jit
def _load_plane(wires_ptr, ip):
    wb = ip * WIRE_WIDTH
    return (tl.load(wires_ptr + wb + 0), tl.load(wires_ptr + wb + 1), tl.load(wires_ptr + wb + 2),
            tl.load(wires_ptr + wb + 9), tl.load(wires_ptr + wb + 10), tl.load(wires_ptr + wb + 11),
            tl.load(wires_ptr + wb + 13))


@triton.jit
def _wire_candidate(valid, ox, oy, oz, dx, dy, dz, cap, wires_ptr, n_wires):
    """True where some wire plane survives Chroma's per-plane early cull."""
    any_plane = tl.zeros(ox.shape, tl.int1)
    for ip in range(n_wires):
        pox, poy, poz, nnx, nny, nnz, radius = _load_plane(wires_ptr, ip)
        dn = dx * nnx + dy * nny + dz * nnz
        wn0 = (ox - pox) * nnx + (oy - poy) * nny + (oz - poz) * nnz
        far_plane = tl.abs(wn0) > radius + 0.01
        away = far_plane & (dn * wn0 > 0.)
        too_far = far_plane & ~away & (-wn0 / dn > cap + radius)
        any_plane = any_plane | (valid & ~away & ~too_far)
    return any_plane


@triton.jit
def _all_wires(valid, ox, oy, oz, dx, dy, dz, cap, wires_ptr, n_wires, LEGACY_WIRES: tl.constexpr):
    """Nearest analytic wire (installed Chroma FP32 algorithm) within ``cap``."""
    wire_t = tl.full(ox.shape, 1e30, tl.float32)
    wire_s = tl.full(ox.shape, -1, tl.int32)
    wire_in = tl.zeros(ox.shape, tl.int32)
    wire_out = tl.zeros(ox.shape, tl.int32)
    wnx = tl.zeros(ox.shape, tl.float32)
    wny = tl.zeros(ox.shape, tl.float32)
    wnz = tl.zeros(ox.shape, tl.float32)
    for ip in range(n_wires):
        wb = ip * WIRE_WIDTH
        pox = tl.load(wires_ptr + wb + 0)
        poy = tl.load(wires_ptr + wb + 1)
        poz = tl.load(wires_ptr + wb + 2)
        uux = tl.load(wires_ptr + wb + 3)
        uuy = tl.load(wires_ptr + wb + 4)
        uuz = tl.load(wires_ptr + wb + 5)
        vvx = tl.load(wires_ptr + wb + 6)
        vvy = tl.load(wires_ptr + wb + 7)
        vvz = tl.load(wires_ptr + wb + 8)
        nnx = tl.load(wires_ptr + wb + 9)
        nny = tl.load(wires_ptr + wb + 10)
        nnz = tl.load(wires_ptr + wb + 11)
        pitch = tl.load(wires_ptr + wb + 12)
        radius = tl.load(wires_ptr + wb + 13)
        umin = tl.load(wires_ptr + wb + 14)
        umax = tl.load(wires_ptr + wb + 15)
        v0 = tl.load(wires_ptr + wb + 16)
        kmin = _bits(tl.load(wires_ptr + wb + 17))
        kmax = _bits(tl.load(wires_ptr + wb + 18))
        psurf = _bits(tl.load(wires_ptr + wb + 19))
        pin = _bits(tl.load(wires_ptr + wb + 20))
        pout = _bits(tl.load(wires_ptr + wb + 21))

        wx = ox - pox
        wy = oy - poy
        wz = oz - poz
        dn = dx * nnx + dy * nny + dz * nnz
        wn0 = wx * nnx + wy * nny + wz * nnz
        far_plane = tl.abs(wn0) > radius + 0.01
        away = far_plane & (dn * wn0 > 0.)
        too_far = far_plane & ~away & (-wn0 / dn > cap + radius)
        go = valid & ~away & ~too_far
        if tl.sum(go.to(tl.int32), axis=0) > 0:
            wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz = _wire_plane(
                go, wx, wy, wz, dx, dy, dz, dn, wn0, uux, uuy, uuz, vvx, vvy, vvz, nnx, nny, nnz,
                pitch, radius, umin, umax, v0, kmin, kmax, psurf, pin, pout, cap,
                wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz, LEGACY_WIRES)
    return wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz


@triton.jit
def wire_kernel(slots_ptr, slot_count_ptr, capacity, rows_ptr, pos_ptr, dir_ptr,
                out_t, out_tri, out_n, out_codes, wires_ptr, n_wires, BLOCK: tl.constexpr,
                LEGACY_WIRES: tl.constexpr = False):
    """Merge analytic wires into the mesh results of the compacted candidate slots."""
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
    wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz = _all_wires(valid, ox, oy, oz, dx, dy, dz, cap,
                                                                   wires_ptr, n_wires, LEGACY_WIRES)
    use_wire = valid & (wire_s >= 0) & (wire_t + 1e-6 < mesh_t)
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
def _box_triangle(box_tris_ptr, slot, m, ox, oy, oz, dx, dy, dz, last, best_t):
    """Moller-Trumbore on one world-space box triangle; returns (ok, t, gid, n, codes)."""
    b = slot * BOX_TRI_WIDTH
    v0x = tl.load(box_tris_ptr + b + 0, mask=m, other=0.)
    v0y = tl.load(box_tris_ptr + b + 1, mask=m, other=0.)
    v0z = tl.load(box_tris_ptr + b + 2, mask=m, other=0.)
    e1x = tl.load(box_tris_ptr + b + 3, mask=m, other=0.)
    e1y = tl.load(box_tris_ptr + b + 4, mask=m, other=0.)
    e1z = tl.load(box_tris_ptr + b + 5, mask=m, other=0.)
    e2x = tl.load(box_tris_ptr + b + 6, mask=m, other=0.)
    e2y = tl.load(box_tris_ptr + b + 7, mask=m, other=0.)
    e2z = tl.load(box_tris_ptr + b + 8, mask=m, other=0.)
    gid = _bits(tl.load(box_tris_ptr + b + 9, mask=m, other=0.))
    hx = dy * e2z - dz * e2y
    hy = dz * e2x - dx * e2z
    hz = dx * e2y - dy * e2x
    det = e1x * hx + e1y * hy + e1z * hz
    inv = 1.0 / det
    sx, sy, sz = ox - v0x, oy - v0y, oz - v0z
    u = inv * (sx * hx + sy * hy + sz * hz)
    qx = sy * e1z - sz * e1y
    qy = sz * e1x - sx * e1z
    qz = sx * e1y - sy * e1x
    v = inv * (dx * qx + dy * qy + dz * qz)
    t = inv * (e2x * qx + e2y * qy + e2z * qz)
    ok = m & (det != 0.) & (u >= -1e-6) & (v >= -1e-6) & (u + v <= 1.0 + 1e-6)
    ok = ok & (t > 1e-6) & (t < best_t) & (gid != last)
    nx = e1y * e2z - e1z * e2y
    ny = e1z * e2x - e1x * e2z
    nz = e1x * e2y - e1y * e2x
    return ok, t, gid, nx, ny, nz, slot


@triton.jit
def _analytic_box(boxes_ptr, box_tris_ptr, ib, valid, ox, oy, oz, dx, dy, dz, last,
                  best_t, best_tri, bnx, bny, bnz, box_hit, bm1, bm2, bsf, FACE_TRIS: tl.constexpr):
    """Nearest face plane of an axis-aligned box, then Moller-Trumbore on that
    face's triangles (exact Chroma triangle ids); all triangles if that fails."""
    rb = ib * BOX_WIDTH
    lx = tl.load(boxes_ptr + rb + 0)
    ly = tl.load(boxes_ptr + rb + 1)
    lz = tl.load(boxes_ptr + rb + 2)
    ux = tl.load(boxes_ptr + rb + 3)
    uy = tl.load(boxes_ptr + rb + 4)
    uz = tl.load(boxes_ptr + rb + 5)
    tol = 1e-3 + 1e-6 * tl.maximum(tl.maximum(tl.abs(lx), tl.abs(ux)), tl.maximum(tl.maximum(tl.abs(ly), tl.abs(uy)),
                                                                                 tl.maximum(tl.abs(lz), tl.abs(uz))))
    face_t = tl.full(ox.shape, float("inf"), tl.float32)
    face = tl.full(ox.shape, -1, tl.int32)
    for f in tl.static_range(6):
        axis = f // 2
        if axis == 0:
            plane = lx if f % 2 == 0 else ux
            o_a, d_a = ox, dx
        elif axis == 1:
            plane = ly if f % 2 == 0 else uy
            o_a, d_a = oy, dy
        else:
            plane = lz if f % 2 == 0 else uz
            o_a, d_a = oz, dz
        t = (plane - o_a) / d_a
        px = ox + t * dx
        py = oy + t * dy
        pz = oz + t * dz
        inside = tl.full(ox.shape, True, tl.int1)
        if axis != 0:
            inside = inside & (px >= lx - tol) & (px <= ux + tol)
        if axis != 1:
            inside = inside & (py >= ly - tol) & (py <= uy + tol)
        if axis != 2:
            inside = inside & (pz >= lz - tol) & (pz <= uz + tol)
        better = valid & (d_a != 0.) & (t > 0.) & inside & (t < face_t)
        face_t = tl.where(better, t, face_t)
        face = tl.where(better, f, face)
    candidate = (face >= 0) & (face_t < best_t + tol)
    first = _bits(tl.load(boxes_ptr + rb + 6 + 2 * tl.maximum(face, 0), mask=candidate, other=0.))
    cnt = _bits(tl.load(boxes_ptr + rb + 7 + 2 * tl.maximum(face, 0), mask=candidate, other=0.))
    found = tl.zeros(ox.shape, tl.int1)
    for k in tl.static_range(FACE_TRIS):
        m = candidate & (k < cnt)
        ok, t, gid, nx, ny, nz, slot = _box_triangle(box_tris_ptr, first + k, m, ox, oy, oz, dx, dy, dz, last, best_t)
        best_t = tl.where(ok, t, best_t)
        best_tri = tl.where(ok, gid, best_tri)
        bnx, bny, bnz = tl.where(ok, nx, bnx), tl.where(ok, ny, bny), tl.where(ok, nz, bnz)
        bm1 = tl.where(ok, _bits(tl.load(box_tris_ptr + slot * BOX_TRI_WIDTH + 10, mask=ok, other=0.)), bm1)
        bm2 = tl.where(ok, _bits(tl.load(box_tris_ptr + slot * BOX_TRI_WIDTH + 11, mask=ok, other=0.)), bm2)
        bsf = tl.where(ok, _bits(tl.load(box_tris_ptr + slot * BOX_TRI_WIDTH + 12, mask=ok, other=0.)), bsf)
        box_hit = box_hit | ok
        found = found | ok
    # Fallback (edges/corners, excluded last hit): every triangle of the box.
    fallback = candidate & ~found
    if tl.sum(fallback.to(tl.int32), axis=0) > 0:
        all_first = _bits(tl.load(boxes_ptr + rb + 6))
        total = _bits(tl.load(boxes_ptr + rb + 17)) + _bits(tl.load(boxes_ptr + rb + 16)) - all_first
        for kk in range(total):
            ok, t, gid, nx, ny, nz, slot = _box_triangle(box_tris_ptr, tl.zeros(ox.shape, tl.int32) + (all_first + kk),
                                                          fallback, ox, oy, oz, dx, dy, dz,
                                                          last, best_t)
            best_t = tl.where(ok, t, best_t)
            best_tri = tl.where(ok, gid, best_tri)
            bnx, bny, bnz = tl.where(ok, nx, bnx), tl.where(ok, ny, bny), tl.where(ok, nz, bnz)
            bm1 = tl.where(ok, _bits(tl.load(box_tris_ptr + slot * BOX_TRI_WIDTH + 10, mask=ok, other=0.)), bm1)
            bm2 = tl.where(ok, _bits(tl.load(box_tris_ptr + slot * BOX_TRI_WIDTH + 11, mask=ok, other=0.)), bm2)
            bsf = tl.where(ok, _bits(tl.load(box_tris_ptr + slot * BOX_TRI_WIDTH + 12, mask=ok, other=0.)), bsf)
            box_hit = box_hit | ok
    return best_t, best_tri, bnx, bny, bnz, box_hit, bm1, bm2, bsf
