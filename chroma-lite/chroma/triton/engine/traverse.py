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


@triton.jit
def _bits(x):
    return x.to(tl.int32, bitcast=True)


@triton.jit
def nearest_hit_kernel(
    rows_ptr, count_ptr,
    pos_ptr, dir_ptr, last_ptr,
    nodes_ptr, inst_ptr, tri_ptr, tri_local_ptr,
    code_m1_ptr, code_m2_ptr, code_s_ptr,
    wires_ptr, n_wires,
    out_t, out_tri, out_n, out_codes,
    capacity,
    LEAF: tl.constexpr, BLOCK: tl.constexpr,
):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(count_ptr)
    if tl.program_id(0) * BLOCK >= count:
        return
    valid = lane < tl.minimum(count, capacity)
    row = tl.load(rows_ptr + lane, mask=valid, other=0)
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

    while tl.sum((node >= 0).to(tl.int32), axis=0) > 0:
        active = node >= 0
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

    mesh_t = best_t
    m1 = tl.load(code_m1_ptr + best_code, mask=valid & (best_tri >= 0), other=0)
    m2 = tl.load(code_m2_ptr + best_code, mask=valid & (best_tri >= 0), other=0)
    sidx = tl.load(code_s_ptr + best_code, mask=valid & (best_tri >= 0), other=-1)

    # ------------------------------------------------------------ wires (FP32)
    wire_t = tl.full((BLOCK,), 1e30, tl.float32)
    wire_s = tl.full((BLOCK,), -1, tl.int32)
    wire_in = tl.zeros((BLOCK,), tl.int32)
    wire_out = tl.zeros((BLOCK,), tl.int32)
    wnx = tl.zeros((BLOCK,), tl.float32)
    wny = tl.zeros((BLOCK,), tl.float32)
    wnz = tl.zeros((BLOCK,), tl.float32)
    cap = tl.where(best_tri >= 0, best_t, 1e30)
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
        v_lo = tl.minimum(tl.minimum(v_entry, v_exit) - pad, wv0 - pad)
        v_hi = tl.maximum(tl.maximum(v_entry, v_exit) + pad, wv0 + pad)
        k_lo = tl.maximum(tl.floor(v_lo * inv_pitch).to(tl.int32), kmin)
        k_hi = tl.minimum(tl.ceil(v_hi * inv_pitch).to(tl.int32), kmax)
        go = go & (kmin <= kmax) & (k_lo <= k_hi)
        k = k_lo
        r2 = radius * radius
        eps0 = tl.maximum(1e-12, 1e-6 * r2)
        while tl.sum((go & (k <= k_hi)).to(tl.int32), axis=0) > 0:
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
            wire_s = tl.where(live2, psurf, wire_s)
            wire_in = tl.where(live2, pin, wire_in)
            wire_out = tl.where(live2, pout, wire_out)
            wnx = tl.where(live2, hx, wnx)
            wny = tl.where(live2, hy, wny)
            wnz = tl.where(live2, hz, wnz)
            k += 1

    use_wire = valid & (wire_s >= 0) & (wire_t + 1e-6 < mesh_t)
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
    tl.store(out_t + lane, dist, mask=valid)
    tl.store(out_tri + lane, tri, mask=valid)
    tl.store(out_n + lane * 3 + 0, nx * inv_norm, mask=valid)
    tl.store(out_n + lane * 3 + 1, ny * inv_norm, mask=valid)
    tl.store(out_n + lane * 3 + 2, nz * inv_norm, mask=valid)
    tl.store(out_codes + lane * 3 + 0, m1, mask=valid)
    tl.store(out_codes + lane * 3 + 1, m2, mask=valid)
    tl.store(out_codes + lane * 3 + 2, sidx, mask=valid)
