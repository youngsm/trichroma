"""Fused transport kernel: one lane per photon, state in registers.

The wavefront scheduler (``core.ProductionEngine._boundary_round`` and the
bulk kernel) writes every photon's state to global memory between steps and
reads it back through queue indices. For long histories (weighted photons
bounce ~100 times) that traffic dominates. Here each persistent one-warp
program keeps 32 photons in registers and runs them to completion; a lane
whose photon finishes takes the next one from the work list.

Every iteration takes one Chroma step for every live lane, so the lanes of a
warp execute the same code: analytic boxes and the top-level tree
(``traverse.top_level_query``), instance meshes only if some lane's ray
reaches an instance (``traverse.instance_traversal``), analytic wires, and
``physics.boundary_step``. Rare branches inside the step (re-emission, WLS,
diffuse reflection, Fresnel) run only when some lane of the warp takes them.
The helpers and the per-event Philox keys are those of the wavefront
kernels, so both schedules give the same photons.
"""

import triton
import triton.language as tl

from chroma.triton.engine import physics as P
from chroma.triton.engine.traverse import _all_wires, instance_traversal, merge_hit, top_level_query



@triton.jit
def _store_state(mask, row, pos_ptr, dir_ptr, pol_ptr, wl_ptr, t_ptr, last_ptr, flags_ptr, w_ptr, steps_ptr,
                 norm_ptr, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, step, norm_step):
    tl.store(pos_ptr + row * 3 + 0, x, mask=mask)
    tl.store(pos_ptr + row * 3 + 1, y, mask=mask)
    tl.store(pos_ptr + row * 3 + 2, z, mask=mask)
    tl.store(dir_ptr + row * 3 + 0, dx, mask=mask)
    tl.store(dir_ptr + row * 3 + 1, dy, mask=mask)
    tl.store(dir_ptr + row * 3 + 2, dz, mask=mask)
    tl.store(pol_ptr + row * 3 + 0, px, mask=mask)
    tl.store(pol_ptr + row * 3 + 1, py, mask=mask)
    tl.store(pol_ptr + row * 3 + 2, pz, mask=mask)
    tl.store(wl_ptr + row, wl, mask=mask)
    tl.store(t_ptr + row, t, mask=mask)
    tl.store(last_ptr + row, last, mask=mask)
    tl.store(flags_ptr + row, flags, mask=mask)
    tl.store(w_ptr + row, weight, mask=mask)
    tl.store(steps_ptr + row, step, mask=mask)
    tl.store(norm_ptr + row, norm_step, mask=mask)


@triton.jit
def fused_kernel(
    work_ptr, work_count_ptr, head_ptr,
    pos_ptr, dir_ptr, pol_ptr, wl_ptr, t_ptr, last_ptr, flags_ptr, w_ptr, ids_ptr,
    steps_ptr, norm_ptr, renorm_ptr, n_renorm,
    nodes_ptr, tlas_nodes, inst_ptr, tri_ptr, tri_local_ptr, code_m1_ptr, code_m2_ptr, code_s_ptr,
    boxes_ptr, n_boxes, box_tris_ptr, wires_ptr, n_wires,
    rindex, absorption, scattering,
    comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs,
    s_present, s_model, s_detect, s_absorb, s_reemit, s_diffuse, s_specular, s_cdf,
    seed, max_steps, wl_start, wl_step, time_start, time_step,
    NW: tl.constexpr, NT: tl.constexpr, MAX_COMP: tl.constexpr, USE_WEIGHTS: tl.constexpr,
    FIXES: tl.constexpr, LEGACY_WIRES: tl.constexpr, LEAF: tl.constexpr, FACE_TRIS: tl.constexpr,
    STEPS: tl.constexpr, BLOCK: tl.constexpr, PARK: tl.constexpr = 8,
    ROULETTE: tl.constexpr = False, w_rr=0.0,
):
    """Propagate the photons listed in ``work_ptr[:work_count]`` (rows) until
    they are terminal or reach ``max_steps``. Persistent: launch a few programs
    per SM; ``head_ptr`` (zeroed) hands out the work."""
    lane = tl.arange(0, BLOCK)
    n_work = tl.load(work_count_ptr)
    if FIXES:
        terminal = P.TERMINAL_32
        nan_abort = P.NAN_ABORT_32
    else:
        terminal = P.TERMINAL_16
        nan_abort = P.NAN_ABORT_16
    zf = tl.zeros((BLOCK,), tl.float32)
    zi = tl.zeros((BLOCK,), tl.int32)
    alive = zi != 0
    row = zi
    x = zf
    y = zf
    z = zf
    dx = zf + 1.
    dy = zf
    dz = zf
    px = zf
    py = zf + 1.
    pz = zf
    wl = zf + wl_start
    t = zf
    last = zi - 1
    flags = zi
    weight = zf + 1.
    ids = tl.zeros((BLOCK,), tl.int64)
    step = zi
    norm_step = zi - 1
    # Lanes whose ray reached an instance wait ("parked") with their top-level
    # result until PARK lanes of the warp (or all its live lanes) need an
    # instance descent; the warp then descends for all of them together.
    parked = alive
    p_t = zf
    p_tri = zi
    p_nx = zf
    p_ny = zf
    p_nz = zf
    p_m1 = zi
    p_m2 = zi
    p_sf = zi
    p_node = zi
    exhausted = n_work * 0
    more = n_work * 0 + 1
    while more > 0:
        # ---- lanes without a photon take the next rows of the work list
        free = ~alive
        n_free = tl.sum(free.to(tl.int32), axis=0)
        if (n_free > 0) & (exhausted == 0):
            base = tl.atomic_add(head_ptr, n_free)
            k = base + tl.cumsum(free.to(tl.int32), axis=0) - 1
            got = free & (k < n_work)
            exhausted = (base + n_free >= n_work).to(tl.int32)
            r = tl.load(work_ptr + k, mask=got, other=0)
            row = tl.where(got, r, row)
            x = tl.where(got, tl.load(pos_ptr + r * 3 + 0, mask=got, other=0.), x)
            y = tl.where(got, tl.load(pos_ptr + r * 3 + 1, mask=got, other=0.), y)
            z = tl.where(got, tl.load(pos_ptr + r * 3 + 2, mask=got, other=0.), z)
            dx = tl.where(got, tl.load(dir_ptr + r * 3 + 0, mask=got, other=1.), dx)
            dy = tl.where(got, tl.load(dir_ptr + r * 3 + 1, mask=got, other=0.), dy)
            dz = tl.where(got, tl.load(dir_ptr + r * 3 + 2, mask=got, other=0.), dz)
            px = tl.where(got, tl.load(pol_ptr + r * 3 + 0, mask=got, other=0.), px)
            py = tl.where(got, tl.load(pol_ptr + r * 3 + 1, mask=got, other=1.), py)
            pz = tl.where(got, tl.load(pol_ptr + r * 3 + 2, mask=got, other=0.), pz)
            wl = tl.where(got, tl.load(wl_ptr + r, mask=got, other=wl_start), wl)
            t = tl.where(got, tl.load(t_ptr + r, mask=got, other=0.), t)
            last = tl.where(got, tl.load(last_ptr + r, mask=got, other=-1), last)
            flags = tl.where(got, tl.load(flags_ptr + r, mask=got, other=0), flags)
            weight = tl.where(got, tl.load(w_ptr + r, mask=got, other=1.), weight)
            ids = tl.where(got, tl.load(ids_ptr + r, mask=got, other=0), ids)
            step = tl.where(got, tl.load(steps_ptr + r, mask=got, other=0), step)
            norm_step = tl.where(got, tl.load(norm_ptr + r, mask=got, other=-1), norm_step)
            alive = alive | got
            parked = parked & ~got
            dx, dy, dz, px, py, pz, norm_step = P.launch_normalize(got, step, norm_step, dx, dy, dz, px, py, pz,
                                                                   renorm_ptr, n_renorm)

        # ---- nearest boundary: boxes and the top level, instances when reached, wires
        query = alive & ~parked
        best_t, best_tri, bnx, bny, bnz, bm1, bm2, bsf, needs, stop = top_level_query(
            query, x, y, z, dx, dy, dz, last, nodes_ptr, tlas_nodes, boxes_ptr, n_boxes, box_tris_ptr, FACE_TRIS,
            STEPS)
        needs = query & needs
        go = query & ~needs
        p_t = tl.where(needs, best_t, p_t)
        p_tri = tl.where(needs, best_tri, p_tri)
        p_nx = tl.where(needs, bnx, p_nx)
        p_ny = tl.where(needs, bny, p_ny)
        p_nz = tl.where(needs, bnz, p_nz)
        p_m1 = tl.where(needs, bm1, p_m1)
        p_m2 = tl.where(needs, bm2, p_m2)
        p_sf = tl.where(needs, bsf, p_sf)
        p_node = tl.where(needs, stop, p_node)
        parked = parked | needs
        box_hit = best_tri >= 0
        best_code = zi
        n_parked = tl.sum(parked.to(tl.int32), axis=0)
        n_alive = tl.sum(alive.to(tl.int32), axis=0)
        if (n_parked >= PARK) | ((n_parked > 0) & (n_parked == n_alive)):
            d_t, d_tri, d_code, d_nx, d_ny, d_nz, d_box = instance_traversal(
                parked, x, y, z, dx, dy, dz, last, p_t, p_tri, zi, p_nx, p_ny, p_nz, p_tri >= 0, p_node,
                nodes_ptr, tlas_nodes, inst_ptr, tri_ptr, tri_local_ptr, LEAF, STEPS)
            best_t = tl.where(parked, d_t, best_t)
            best_tri = tl.where(parked, d_tri, best_tri)
            best_code = tl.where(parked, d_code, best_code)
            bnx = tl.where(parked, d_nx, bnx)
            bny = tl.where(parked, d_ny, bny)
            bnz = tl.where(parked, d_nz, bnz)
            bm1 = tl.where(parked, p_m1, bm1)
            bm2 = tl.where(parked, p_m2, bm2)
            bsf = tl.where(parked, p_sf, bsf)
            box_hit = tl.where(parked, d_box, box_hit)
            go = go | parked
            parked = parked & False
        from_mesh = go & (best_tri >= 0) & ~box_hit
        bm1 = tl.where(from_mesh, tl.load(code_m1_ptr + best_code, mask=from_mesh, other=0), bm1)
        bm2 = tl.where(from_mesh, tl.load(code_m2_ptr + best_code, mask=from_mesh, other=0), bm2)
        bsf = tl.where(from_mesh, tl.load(code_s_ptr + best_code, mask=from_mesh, other=-1), bsf)
        cap = tl.where(best_tri >= 0, best_t, 1e30)
        wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz = _all_wires(go, x, y, z, dx, dy, dz, cap, wires_ptr,
                                                                       n_wires, LEGACY_WIRES)
        dist, tri, nx, ny, nz, m_inner, m_outer, sidx = merge_hit(go, best_t, best_tri, bnx, bny, bnz, bm1, bm2,
                                                                  bsf, wire_t, wire_s, wire_in, wire_out,
                                                                  wnx, wny, wnz)

        # ---- one Chroma step
        key = step
        step = step + go.to(tl.int32)
        product = dx * dy * dz * x * y * z
        is_nan = go & (product != product)
        flags = tl.where(is_nan, flags | P.NO_HIT | nan_abort, flags)
        live = go & ~is_nan
        dist = tl.where(live, dist, 0.)
        tri = tl.where(live, tri, -1)
        nz = tl.where(live, nz, 1.)
        sidx = tl.where(live, sidx, -1)
        x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight = P.boundary_step(
            live, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, ids, key,
            dist, tri, nx, ny, nz, m_inner, m_outer, sidx,
            rindex, absorption, scattering,
            comp_offsets, comp_prob, comp_wcdf, comp_tcdf, comp_abs,
            s_present, s_model, s_detect, s_absorb, s_reemit, s_diffuse, s_specular, s_cdf,
            seed, wl_start, wl_step, time_start, time_step, NW, NT, MAX_COMP, USE_WEIGHTS, FIXES, ROULETTE, w_rr)

        # ---- finished photons are written back; their lanes take new work
        fin = alive & (((flags & terminal) != 0) | (step >= max_steps))
        n_fin = tl.sum(fin.to(tl.int32), axis=0)
        if n_fin > 0:
            _store_state(fin, row, pos_ptr, dir_ptr, pol_ptr, wl_ptr, t_ptr, last_ptr, flags_ptr, w_ptr,
                         steps_ptr, norm_ptr, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, step,
                         norm_step)
            alive = alive & ~fin
        more = ((tl.sum(alive.to(tl.int32), axis=0) > 0) | (exhausted == 0)).to(tl.int32)
