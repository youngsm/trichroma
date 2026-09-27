"""Fused transport kernel: persistent programs of one warp with per-warp queues.

The wavefront scheduler (``core.ProductionEngine._boundary_round`` and the
bulk kernel) writes every photon's state to global memory between steps and
reads it back through queue indices, with a kernel launch per phase. Here
each persistent one-warp program keeps its photons in three small queues of
its own (field-major rings in ``ring_f``/``ring_i``, L2-resident) and in a
*resident batch* of up to 32 photons held in registers, and schedules them
itself; every iteration processes up to 32 photons *of one kind*, so that
the lanes of the warp run the same code:

* FULL: photons whose next step needs the full query: analytic boxes and the
  top-level tree (``traverse.top_level_query``), analytic wires and
  ``physics.boundary_step``. They are the resident batch: a FULL iteration
  steps the resident photons (its empty lanes first take photons from the
  FULL queue), and the photons whose next step is FULL again stay in their
  lanes for the next FULL iteration, so the common case (a photon that
  moves on to its next FULL step) costs no queue traffic. Before any other
  kind of iteration the resident photons go to the FULL queue. A photon
  whose ray reaches an instance (a mesh placed in the top-level tree) moves
  to the descent queue with its box result instead of taking its step.
* DESCENT: once BATCH photons wait there, all 32 lanes complete their queries
  (``traverse.batch_traversal``: each lane takes the next waiting photon when
  it is done); the photons then take their step in READY iterations (and
  those whose next step is FULL become the resident batch).
* BULK (only with the empty-space grid, ``grid.py``, off by default):
  photons whose next step is a collision inside a certified empty box of the
  scene take that step with the box's shrunk exit as their boundary distance
  and the box's material, without a geometry query. This equals the full
  step only if the full query would report that material; see ``grid.py``
  for the cases where it does not.
* REFILL: new photons from the work list (32 at a time): those of FULL kind
  take the empty lanes of the resident batch (lane i takes work entry i),
  the others go to the queues.

After its step a photon is classified for its next step (BULK when it lies
in a certified box and that step's own absorption/scattering draws fall
before the box exit, FULL otherwise) and stays resident or is queued, or is
written back when it is done. The per-event Philox keys are those of the
wavefront kernels and every query is complete, so every schedule gives the
same photons (bit for bit).
"""

import triton
import triton.language as tl

from trichroma.engine import physics as P
from trichroma.engine import traverse as TR
from trichroma.engine.traverse import _all_wires, batch_traversal, merge_hit, top_level_query
from trichroma.engine.warp import _lane_rank, _ring, _warp_any, _warp_count

# Step-queue entry fields: the photon fields are TR.F_X .. TR.F_LWIRE (0-17,
# with TR.F_IDLO); then the bulk step's boundary distance and material, and
# the id's high bits.
Q_BT = tl.constexpr(18)
Q_BM = tl.constexpr(19)
Q_IDHI = tl.constexpr(20)
Q_FIELDS = tl.constexpr(21)

MODE_DESC = tl.constexpr(0)
MODE_READY = tl.constexpr(1)
MODE_BULK = tl.constexpr(2)
MODE_FULL = tl.constexpr(3)
MODE_REFILL = tl.constexpr(4)
MODE_DONE = tl.constexpr(5)


@triton.jit
def _store_state(mask, row, pos_ptr, dir_ptr, pol_ptr, wl_ptr, t_ptr, last_ptr, flags_ptr, w_ptr, steps_ptr,
                 x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, step):
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


@triton.jit
def _put(rf, ri, stride, s, m, idhi_field, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, step, row,
         last_wire, ids):
    """Photon into slot ``s`` of the ring at rf/ri (field-major, ``stride`` slots)."""
    tl.store(rf + TR.F_X * stride + s, x, mask=m, cache_modifier=".cg")
    tl.store(rf + TR.F_Y * stride + s, y, mask=m, cache_modifier=".cg")
    tl.store(rf + TR.F_Z * stride + s, z, mask=m, cache_modifier=".cg")
    tl.store(rf + TR.F_DX * stride + s, dx, mask=m, cache_modifier=".cg")
    tl.store(rf + TR.F_DY * stride + s, dy, mask=m, cache_modifier=".cg")
    tl.store(rf + TR.F_DZ * stride + s, dz, mask=m, cache_modifier=".cg")
    tl.store(rf + TR.F_PX * stride + s, px, mask=m, cache_modifier=".cg")
    tl.store(rf + TR.F_PY * stride + s, py, mask=m, cache_modifier=".cg")
    tl.store(rf + TR.F_PZ * stride + s, pz, mask=m, cache_modifier=".cg")
    tl.store(rf + TR.F_WL * stride + s, wl, mask=m, cache_modifier=".cg")
    tl.store(rf + TR.F_T * stride + s, t, mask=m, cache_modifier=".cg")
    tl.store(ri + TR.F_LAST * stride + s, last, mask=m, cache_modifier=".cg")
    tl.store(ri + TR.F_FLAGS * stride + s, flags, mask=m, cache_modifier=".cg")
    tl.store(rf + TR.F_W * stride + s, weight, mask=m, cache_modifier=".cg")
    tl.store(ri + TR.F_STEP * stride + s, step, mask=m, cache_modifier=".cg")
    tl.store(ri + TR.F_ROW * stride + s, row, mask=m, cache_modifier=".cg")
    tl.store(ri + TR.F_LWIRE * stride + s, last_wire, mask=m, cache_modifier=".cg")
    tl.store(ri + TR.F_IDLO * stride + s, (ids & 0xFFFFFFFF).to(tl.int32), mask=m, cache_modifier=".cg")
    tl.store(ri + idhi_field * stride + s, (ids >> 32).to(tl.int32), mask=m, cache_modifier=".cg")


@triton.jit
def _get(rf, ri, stride, s, m, idhi_field):
    """Photon of slot ``s`` of the ring at rf/ri (lanes in ``m``)."""
    x = tl.load(rf + TR.F_X * stride + s, mask=m, other=0., cache_modifier=".cg")
    y = tl.load(rf + TR.F_Y * stride + s, mask=m, other=0., cache_modifier=".cg")
    z = tl.load(rf + TR.F_Z * stride + s, mask=m, other=0., cache_modifier=".cg")
    dx = tl.load(rf + TR.F_DX * stride + s, mask=m, other=1., cache_modifier=".cg")
    dy = tl.load(rf + TR.F_DY * stride + s, mask=m, other=0., cache_modifier=".cg")
    dz = tl.load(rf + TR.F_DZ * stride + s, mask=m, other=0., cache_modifier=".cg")
    px = tl.load(rf + TR.F_PX * stride + s, mask=m, other=0., cache_modifier=".cg")
    py = tl.load(rf + TR.F_PY * stride + s, mask=m, other=1., cache_modifier=".cg")
    pz = tl.load(rf + TR.F_PZ * stride + s, mask=m, other=0., cache_modifier=".cg")
    wl = tl.load(rf + TR.F_WL * stride + s, mask=m, other=0., cache_modifier=".cg")
    t = tl.load(rf + TR.F_T * stride + s, mask=m, other=0., cache_modifier=".cg")
    last = tl.load(ri + TR.F_LAST * stride + s, mask=m, other=-1, cache_modifier=".cg")
    flags = tl.load(ri + TR.F_FLAGS * stride + s, mask=m, other=0, cache_modifier=".cg")
    weight = tl.load(rf + TR.F_W * stride + s, mask=m, other=1., cache_modifier=".cg")
    step = tl.load(ri + TR.F_STEP * stride + s, mask=m, other=0, cache_modifier=".cg")
    row = tl.load(ri + TR.F_ROW * stride + s, mask=m, other=0, cache_modifier=".cg")
    last_wire = tl.load(ri + TR.F_LWIRE * stride + s, mask=m, other=-1, cache_modifier=".cg")
    lo = tl.load(ri + TR.F_IDLO * stride + s, mask=m, other=0, cache_modifier=".cg")
    hi = tl.load(ri + idhi_field * stride + s, mask=m, other=0, cache_modifier=".cg")
    ids = (hi.to(tl.int64) << 32) | (lo.to(tl.int64) & 0xFFFFFFFF)
    return x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, step, row, last_wire, ids


@triton.jit
def _flush(res, n_res, qff, qfi, qf_wr, CAP: tl.constexpr, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags,
           weight, step, row, last_wire, ids):
    """The resident photons (lanes in ``res``) to the FULL queue; returns its new write index."""
    sf = _ring(qf_wr + _lane_rank(res), CAP)
    _put(qff, qfi, CAP, sf, res, Q_IDHI, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, step, row,
         last_wire, ids)
    return qf_wr + n_res


@triton.jit
def _classify(m, x, y, z, dx, dy, dz, wl, last, flags, weight, step, ids, seed, max_steps,
              grid_material, grid_boxes, gx0, gy0, gz0, gcx, gcy, gcz, GNX, GNY, GNZ,
              absorption, scattering, wl_start, wl_step,
              NW: tl.constexpr, USE_WEIGHTS: tl.constexpr, terminal, GRID: tl.constexpr):
    """Whether the photon's next step is a collision inside a certified empty
    box (it then needs no boundary query), with that box's shrunk exit
    distance and material. Decided with the step's own transport draws."""
    bulk = m & False
    bulk_t = tl.zeros(x.shape, tl.float32)
    bulk_m = tl.zeros(x.shape, tl.int32)
    if GRID:
        finite = (dx * dy * dz * x * y * z) == (dx * dy * dz * x * y * z)
        # A photon on a surface (last != -1) lies in an occupied cell.
        cand = m & finite & (last == -1) & ((flags & terminal) == 0) & (step < max_steps)
        gi = tl.floor((x - gx0) / gcx).to(tl.int32)
        gj = tl.floor((y - gy0) / gcy).to(tl.int32)
        gk = tl.floor((z - gz0) / gcz).to(tl.int32)
        in_grid = cand & (gi >= 0) & (gi < GNX) & (gj >= 0) & (gj < GNY) & (gk >= 0) & (gk < GNZ)
        cell = (gi * GNY + gj) * GNZ + gk
        mat = tl.load(grid_material + cell, mask=in_grid, other=-1)
        boxed = in_grid & (mat >= 0)
        blx = tl.load(grid_boxes + cell * 6 + 0, mask=boxed, other=0.)
        bly = tl.load(grid_boxes + cell * 6 + 1, mask=boxed, other=0.)
        blz = tl.load(grid_boxes + cell * 6 + 2, mask=boxed, other=0.)
        bux = tl.load(grid_boxes + cell * 6 + 3, mask=boxed, other=0.)
        buy = tl.load(grid_boxes + cell * 6 + 4, mask=boxed, other=0.)
        buz = tl.load(grid_boxes + cell * 6 + 5, mask=boxed, other=0.)
        boxed = boxed & (x > blx) & (x < bux) & (y > bly) & (y < buy) & (z > blz) & (z < buz)
        if _warp_any(boxed) != 0:
            inc = tl.maximum(mat, 0)
            alen = P.interp_uniform(absorption, inc, wl, wl_start, wl_step, NW, boxed)
            slen = P.interp_uniform(scattering, inc, wl, wl_start, wl_step, NW, boxed)
            u_abs, u_sca, _u_cos, _u_phi = P.uniforms4(ids, seed, step, P.B_TRANSPORT)
            d_abs = -alen * P.log_unit(u_abs)
            d_sca = -slen * P.log_unit(u_sca)
            if USE_WEIGHTS:
                d_abs = tl.where(weight > P.WEIGHT_LOWER_THRESHOLD, 1e30, d_abs)
            tx = tl.where(dx > 0., (bux - x) / dx, tl.where(dx < 0., (blx - x) / dx, float("inf")))
            ty = tl.where(dy > 0., (buy - y) / dy, tl.where(dy < 0., (bly - y) / dy, float("inf")))
            tz = tl.where(dz > 0., (buz - z) / dz, tl.where(dz < 0., (blz - z) / dz, float("inf")))
            exit_d = tl.minimum(tl.minimum(tx, ty), tz)
            safe = exit_d - tl.maximum(0.01, 2e-6 * exit_d)
            bulk = boxed & (((d_abs <= d_sca) & (d_abs < safe)) | (~(d_abs <= d_sca) & (d_sca < safe)))
            bulk_t = safe
            bulk_m = inc
    return bulk, bulk_t, bulk_m


@triton.jit
def _take_work(n_take, lane, head_ptr, work_ptr, n_work, pos_ptr, dir_ptr, pol_ptr, wl_ptr, t_ptr, last_ptr, flags_ptr,
               w_ptr, ids_ptr, steps_ptr, norm_ptr, renorm_ptr, n_renorm, wl_start):
    """Up to ``n_take`` new photons from the work list (lane i: entry head + i),
    normalized as a launch does on entry. Returns (got, exhausted, row, photon)."""
    base = tl.atomic_add(head_ptr, n_take)
    k = base + lane
    got = (lane < n_take) & (k < n_work)
    exhausted = (base + n_take >= n_work).to(tl.int32)
    r = tl.load(work_ptr + k, mask=got, other=0)
    x = tl.load(pos_ptr + r * 3 + 0, mask=got, other=0.)
    y = tl.load(pos_ptr + r * 3 + 1, mask=got, other=0.)
    z = tl.load(pos_ptr + r * 3 + 2, mask=got, other=0.)
    dx = tl.load(dir_ptr + r * 3 + 0, mask=got, other=1.)
    dy = tl.load(dir_ptr + r * 3 + 1, mask=got, other=0.)
    dz = tl.load(dir_ptr + r * 3 + 2, mask=got, other=0.)
    px = tl.load(pol_ptr + r * 3 + 0, mask=got, other=0.)
    py = tl.load(pol_ptr + r * 3 + 1, mask=got, other=1.)
    pz = tl.load(pol_ptr + r * 3 + 2, mask=got, other=0.)
    wl = tl.load(wl_ptr + r, mask=got, other=wl_start)
    t = tl.load(t_ptr + r, mask=got, other=0.)
    last = tl.load(last_ptr + r, mask=got, other=-1)
    flags = tl.load(flags_ptr + r, mask=got, other=0)
    weight = tl.load(w_ptr + r, mask=got, other=1.)
    ids = tl.load(ids_ptr + r, mask=got, other=0)
    step = tl.load(steps_ptr + r, mask=got, other=0)
    norm_step = tl.load(norm_ptr + r, mask=got, other=-1)
    # The launch-entry normalization's step marker changes only here.
    dx, dy, dz, px, py, pz, norm_step = P.launch_normalize(got, step, norm_step, dx, dy, dz, px, py, pz,
                                                           renorm_ptr, n_renorm)
    tl.store(norm_ptr + r, norm_step, mask=got)
    return got, exhausted, r, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, step, ids


@triton.jit(do_not_specialize=["seed", "max_steps"])
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
    ring_f, ring_i,
    grid_material, grid_boxes, gx0, gy0, gz0, gcx, gcy, gcz, GNX, GNY, GNZ,
    NW: tl.constexpr, NT: tl.constexpr, MAX_COMP: tl.constexpr, USE_WEIGHTS: tl.constexpr,
    FIXES: tl.constexpr, LEGACY_WIRES: tl.constexpr, LEAF: tl.constexpr, FACE_TRIS: tl.constexpr,
    STEPS: tl.constexpr, BLOCK: tl.constexpr,
    ROULETTE: tl.constexpr = False, w_rr=0.0,
    CAP: tl.constexpr = 128, BATCH: tl.constexpr = 32, REFILL: tl.constexpr = 8,
    GRID: tl.constexpr = False, HAS_WLS: tl.constexpr = True, HAS_REEMIT: tl.constexpr = True,
    DESC_INNER: tl.constexpr = 8, DEFER: tl.constexpr = True,
):
    """Propagate the photons listed in ``work_ptr[:work_count]`` (rows) until
    they are terminal or reach ``max_steps``. Persistent: launch a few programs
    per SM; ``head_ptr`` (zeroed) hands out the work.

    Each program owns ``WORDS = (2 * Q_FIELDS + RING_FIELDS) * CAP`` words of
    ``ring_f``/``ring_i`` (float32/int32 views of one buffer): the BULK and
    FULL step queues and the descent ring (see the module docstring), CAP
    slots each; at most CAP photons are in flight per program (queued or
    resident), so no queue overflows.
    """
    seed = seed.to(tl.uint32, bitcast=True)  # passed as its int32 bit pattern (ProductionEngine.seed_arg)
    lane = tl.arange(0, BLOCK)
    # The warp-vote helpers (TR._lane_rank) take element i to be lane i: one
    # warp (num_warps=1) of one element per lane.
    tl.static_assert(BLOCK == 32, "fused_kernel runs one element per lane of one warp")
    n_work = tl.load(work_count_ptr)
    if FIXES:
        terminal = P.TERMINAL_32
        nan_abort = P.NAN_ABORT_32
    else:
        terminal = P.TERMINAL_16
        nan_abort = P.NAN_ABORT_16
    zf = tl.zeros((BLOCK,), tl.float32)
    zi = tl.zeros((BLOCK,), tl.int32)
    WORDS: tl.constexpr = (2 * Q_FIELDS + TR.RING_FIELDS) * CAP
    base_f = ring_f + tl.program_id(0) * WORDS
    base_i = ring_i + tl.program_id(0) * WORDS
    qbf = base_f
    qbi = base_i
    qff = base_f + Q_FIELDS * CAP
    qfi = base_i + Q_FIELDS * CAP
    rf = base_f + 2 * Q_FIELDS * CAP
    ri = base_i + 2 * Q_FIELDS * CAP
    zero = n_work * 0
    exhausted = zero
    qb_rd = zero  # BULK queue [qb_rd, qb_wr)
    qb_wr = zero
    qf_rd = zero  # FULL queue [qf_rd, qf_wr)
    qf_wr = zero
    rd = zero  # descent ring: [rd, ds) done (READY), [ds, wr) waiting
    ds = zero
    wr = zero
    mode = zero
    # The resident batch: lanes in ``res`` hold photons whose next step needs
    # the full query. They stay in registers from one FULL iteration to the
    # next (a FULL iteration first fills its empty lanes from the FULL queue;
    # REFILL puts new photons into empty lanes) and go to the FULL queue only
    # when the warp turns to DESC, READY or BULK work, which then run with
    # these registers free.
    res = lane < 0
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
    step = zi
    row = zi
    last_wire = zi - 1
    ids = zi.to(tl.int64)
    while mode != MODE_DONE:
        n_res = _warp_count(res)
        n_b = qb_wr - qb_rd
        n_f = qf_wr - qf_rd
        n_ready = ds - rd
        n_wait = wr - ds
        n_fq = n_res + n_f
        total = n_fq + n_b + (wr - rd)
        big = tl.maximum(tl.maximum(n_b, n_fq), n_ready)
        room = (exhausted == 0) & (total <= CAP - BLOCK)
        # Full batches first, the ring queues before FULL (the resident batch,
        # counted with the FULL queue, can wait in its registers; READY and
        # BULK leave their FULL-kind photons resident), then new work while
        # there is room, then (no room: work exhausted or CAP photons in
        # flight) the fullest queue.
        mode = tl.where(n_wait >= BATCH, MODE_DESC,
               tl.where(n_ready >= BLOCK, MODE_READY,
               tl.where(n_b >= BLOCK, MODE_BULK,
               tl.where(n_fq >= BLOCK, MODE_FULL,
               tl.where(room, MODE_REFILL,
               tl.where((n_ready == big) & (n_ready > 0), MODE_READY,
               tl.where((n_b == big) & (n_b > 0), MODE_BULK,
               tl.where(n_fq > 0, MODE_FULL,
               tl.where(n_wait > 0, MODE_DESC, MODE_DONE)))))))))
        if mode == MODE_DESC:
            if n_res > 0:
                qf_wr = _flush(res, n_res, qff, qfi, qf_wr, CAP, x, y, z, dx, dy, dz, px, py, pz, wl, t, last,
                               flags, weight, step, row, last_wire, ids)
            # The resident values are dead here (fewer live registers in the batch).
            res = lane < 0
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
            step = zi
            row = zi
            last_wire = zi - 1
            ids = zi.to(tl.int64)
            tl.debug_barrier()
            batch_traversal(rf, ri, CAP, CAP, ds, n_wait, nodes_ptr, inst_ptr, tri_ptr, tri_local_ptr,
                            code_m1_ptr, code_m2_ptr, code_s_ptr, LEAF, BLOCK, REFILL, INNER=DESC_INNER,
                            DEFER=DEFER)
            tl.debug_barrier()
            ds = wr
        elif mode == MODE_REFILL:
            # ---- new photons, classified for their first step: those of FULL kind
            # take the empty lanes of the resident batch (lane i: work entry i), the
            # others go to the queues
            (got, exhausted, a_row, a_x, a_y, a_z, a_dx, a_dy, a_dz, a_px, a_py, a_pz, a_wl, a_t, a_last, a_flags,
             a_weight, a_step, a_ids) = _take_work(tl.minimum(CAP - total, BLOCK), lane, head_ptr, work_ptr, n_work,
                                                   pos_ptr, dir_ptr, pol_ptr, wl_ptr, t_ptr, last_ptr, flags_ptr,
                                                   w_ptr, ids_ptr, steps_ptr, norm_ptr, renorm_ptr, n_renorm,
                                                   wl_start)
            a_lwire = zi - 1
            bulk, bulk_t, bulk_m = _classify(got, a_x, a_y, a_z, a_dx, a_dy, a_dz, a_wl, a_last, a_flags, a_weight,
                                             a_step, a_ids, seed, max_steps, grid_material, grid_boxes, gx0, gy0,
                                             gz0, gcx, gcy, gcz, GNX, GNY, GNZ, absorption, scattering, wl_start,
                                             wl_step, NW, USE_WEIGHTS, terminal, GRID)
            to_b = got & bulk
            if GRID:
                sb = _ring(qb_wr + _lane_rank(to_b), CAP)
                _put(qbf, qbi, CAP, sb, to_b, Q_IDHI, a_x, a_y, a_z, a_dx, a_dy, a_dz, a_px, a_py, a_pz, a_wl, a_t,
                     a_last, a_flags, a_weight, a_step, a_row, a_lwire, a_ids)
                tl.store(qbf + Q_BT * CAP + sb, bulk_t, mask=to_b, cache_modifier=".cg")
                tl.store(qbi + Q_BM * CAP + sb, bulk_m, mask=to_b, cache_modifier=".cg")
                qb_wr += _warp_count(to_b)
            to_f = got & ~to_b & res
            n_to_f = _warp_count(to_f)
            if n_to_f > 0:
                sf = _ring(qf_wr + _lane_rank(to_f), CAP)
                _put(qff, qfi, CAP, sf, to_f, Q_IDHI, a_x, a_y, a_z, a_dx, a_dy, a_dz, a_px, a_py, a_pz, a_wl, a_t,
                     a_last, a_flags, a_weight, a_step, a_row, a_lwire, a_ids)
                qf_wr += n_to_f
            direct = got & ~to_b & ~res
            x = tl.where(direct, a_x, x)
            y = tl.where(direct, a_y, y)
            z = tl.where(direct, a_z, z)
            dx = tl.where(direct, a_dx, dx)
            dy = tl.where(direct, a_dy, dy)
            dz = tl.where(direct, a_dz, dz)
            px = tl.where(direct, a_px, px)
            py = tl.where(direct, a_py, py)
            pz = tl.where(direct, a_pz, pz)
            wl = tl.where(direct, a_wl, wl)
            t = tl.where(direct, a_t, t)
            last = tl.where(direct, a_last, last)
            flags = tl.where(direct, a_flags, flags)
            weight = tl.where(direct, a_weight, weight)
            step = tl.where(direct, a_step, step)
            row = tl.where(direct, a_row, row)
            last_wire = tl.where(direct, a_lwire, last_wire)
            ids = tl.where(direct, a_ids, ids)
            res = res | direct
        elif mode != MODE_DONE:
            # ---- one step for up to 32 photons of one kind
            is_ready = mode == MODE_READY
            is_bulk = mode == MODE_BULK
            s = zi
            if mode == MODE_FULL:
                # Empty lanes take photons from the FULL queue.
                hole = ~res
                rank = _lane_rank(hole)
                take_q = tl.minimum(BLOCK - n_res, n_f)
                fromq = hole & (rank < take_q)
                if take_q > 0:
                    sq = _ring(qf_rd + rank, CAP)
                    (qx, qy, qz, qdx, qdy, qdz, qpx, qpy, qpz, qwl, qt, qlast, qflags, qweight, qstep, qrow,
                     qlast_wire, qids) = _get(qff, qfi, CAP, sq, fromq, Q_IDHI)
                    x = tl.where(fromq, qx, x)
                    y = tl.where(fromq, qy, y)
                    z = tl.where(fromq, qz, z)
                    dx = tl.where(fromq, qdx, dx)
                    dy = tl.where(fromq, qdy, dy)
                    dz = tl.where(fromq, qdz, dz)
                    px = tl.where(fromq, qpx, px)
                    py = tl.where(fromq, qpy, py)
                    pz = tl.where(fromq, qpz, pz)
                    wl = tl.where(fromq, qwl, wl)
                    t = tl.where(fromq, qt, t)
                    last = tl.where(fromq, qlast, last)
                    flags = tl.where(fromq, qflags, flags)
                    weight = tl.where(fromq, qweight, weight)
                    step = tl.where(fromq, qstep, step)
                    row = tl.where(fromq, qrow, row)
                    last_wire = tl.where(fromq, qlast_wire, last_wire)
                    ids = tl.where(fromq, qids, ids)
                    qf_rd += take_q
                    res = res | fromq
                act = res
            else:
                # READY / BULK: entries of a ring (the resident batch goes to the FULL queue first).
                if n_res > 0:
                    qf_wr = _flush(res, n_res, qff, qfi, qf_wr, CAP, x, y, z, dx, dy, dz, px, py, pz, wl, t, last,
                                   flags, weight, step, row, last_wire, ids)
                n_have = tl.where(is_ready, n_ready, n_b)
                n_take = tl.minimum(n_have, BLOCK)
                act = lane < n_take
                q_rd = tl.where(is_ready, rd, qb_rd)
                s = _ring(q_rd + lane, CAP)
                q_off = tl.where(is_ready, 2 * Q_FIELDS * CAP, 0)
                idhi = tl.where(is_ready, TR.F_IDHI, Q_IDHI)
                x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, step, row, last_wire, ids = _get(
                    base_f + q_off, base_i + q_off, CAP, s, act, idhi)
                rd = tl.where(is_ready, rd + n_take, rd)
                qb_rd = tl.where(is_bulk, qb_rd + n_take, qb_rd)
            finite = (dx * dy * dz * x * y * z) == (dx * dy * dz * x * y * z)
            best_t = zf + float("inf")
            best_tri = zi - 1
            bnx = zf
            bny = zf
            bnz = zf
            bm1 = zi
            bm2 = zi
            bsf = zi - 1
            go = act
            if mode == MODE_FULL:
                # Analytic boxes and the top level; a photon whose ray reaches an
                # instance waits in the descent ring. (A NaN photon needs no
                # query: its step flags it.)
                query = act & finite
                best_t, best_tri, bnx, bny, bnz, bm1, bm2, bsf, needs, stop = top_level_query(
                    query, x, y, z, dx, dy, dz, last, nodes_ptr, tlas_nodes, boxes_ptr, n_boxes, box_tris_ptr,
                    FACE_TRIS, STEPS, FIXES)
                needs = query & needs
                n_needs = _warp_count(needs)
                if n_needs > 0:
                    sp = _ring(wr + _lane_rank(needs), CAP)
                    _put(rf, ri, CAP, sp, needs, TR.F_IDHI, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags,
                         weight, step, row, last_wire, ids)
                    tl.store(rf + TR.F_BT * CAP + sp, best_t, mask=needs, cache_modifier=".cg")
                    tl.store(ri + TR.F_BTRI * CAP + sp, best_tri, mask=needs, cache_modifier=".cg")
                    tl.store(rf + TR.F_NX * CAP + sp, bnx, mask=needs, cache_modifier=".cg")
                    tl.store(rf + TR.F_NY * CAP + sp, bny, mask=needs, cache_modifier=".cg")
                    tl.store(rf + TR.F_NZ * CAP + sp, bnz, mask=needs, cache_modifier=".cg")
                    tl.store(ri + TR.F_M1 * CAP + sp, bm1, mask=needs, cache_modifier=".cg")
                    tl.store(ri + TR.F_M2 * CAP + sp, bm2, mask=needs, cache_modifier=".cg")
                    tl.store(ri + TR.F_SF * CAP + sp, bsf, mask=needs, cache_modifier=".cg")
                    tl.store(ri + TR.F_NODE * CAP + sp, stop, mask=needs, cache_modifier=".cg")
                    wr += n_needs
                go = act & ~needs
            elif mode == MODE_READY:
                # Their queries are complete (codes resolved).
                best_t = tl.load(rf + TR.F_BT * CAP + s, mask=act, other=0., cache_modifier=".cg")
                best_tri = tl.load(ri + TR.F_BTRI * CAP + s, mask=act, other=-1, cache_modifier=".cg")
                bnx = tl.load(rf + TR.F_NX * CAP + s, mask=act, other=0., cache_modifier=".cg")
                bny = tl.load(rf + TR.F_NY * CAP + s, mask=act, other=0., cache_modifier=".cg")
                bnz = tl.load(rf + TR.F_NZ * CAP + s, mask=act, other=0., cache_modifier=".cg")
                bm1 = tl.load(ri + TR.F_M1 * CAP + s, mask=act, other=0, cache_modifier=".cg")
                bm2 = tl.load(ri + TR.F_M2 * CAP + s, mask=act, other=0, cache_modifier=".cg")
                bsf = tl.load(ri + TR.F_SF * CAP + s, mask=act, other=-1, cache_modifier=".cg")
            else:
                # BULK: the boundary is beyond the certified box's shrunk exit.
                best_t = tl.load(qbf + Q_BT * CAP + s, mask=act, other=0., cache_modifier=".cg")
                best_tri = zi
                bnx = -dx
                bny = -dy
                bnz = -dz
                bm1 = tl.load(qbi + Q_BM * CAP + s, mask=act, other=0, cache_modifier=".cg")
                bm2 = bm1
                bsf = zi - 1
            wire_t = zf + 1e30
            wire_s = zi - 1
            wire_in = zi
            wire_out = zi
            wnx = zf
            wny = zf
            wnz = zf
            wire_id = zi - 1
            if mode != MODE_BULK:
                cap = tl.where(best_tri >= 0, best_t, 1e30)
                wire_t, wire_s, wire_in, wire_out, wnx, wny, wnz, wire_id = _all_wires(
                    go & finite, x, y, z, dx, dy, dz, cap, wires_ptr, n_wires, tl.where(last == -2, last_wire, -1),
                    LEGACY_WIRES)
            dist, tri, nx, ny, nz, m_inner, m_outer, sidx = merge_hit(go, best_t, best_tri, bnx, bny, bnz, bm1, bm2,
                                                                      bsf, wire_t, wire_s, wire_in, wire_out,
                                                                      wnx, wny, wnz)
            # ---- one Chroma step
            key = step
            step = step + go.to(tl.int32)
            is_nan = go & ~finite
            flags = tl.where(is_nan, flags | P.NO_HIT | nan_abort, flags)
            live = go & finite
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
                seed, wl_start, wl_step, time_start, time_step, NW, NT, MAX_COMP, USE_WEIGHTS, FIXES, ROULETTE, w_rr,
                HAS_WLS, HAS_REEMIT)
            if not LEGACY_WIRES:
                # A photon that reached a wire (last == -2: no bulk event first)
                # and leaves it outward is outside that convex wire: skip it next.
                outward = (dx * nx + dy * ny + dz * nz) > 0.
                last_wire = tl.where(live, tl.where((last == -2) & outward, wire_id, -1), last_wire)
            # ---- finished photons are written back; the others stay resident
            # (next step FULL) or go to the BULK queue
            fin = go & (((flags & terminal) != 0) | (step >= max_steps))
            cont = go & ~fin
            n_fin = _warp_any(fin)
            if n_fin != 0:
                _store_state(fin, row, pos_ptr, dir_ptr, pol_ptr, wl_ptr, t_ptr, last_ptr, flags_ptr, w_ptr,
                             steps_ptr, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags, weight, step)
            bulk, bulk_t, bulk_m = _classify(cont, x, y, z, dx, dy, dz, wl, last, flags, weight, step, ids, seed,
                                             max_steps, grid_material, grid_boxes, gx0, gy0, gz0, gcx, gcy, gcz,
                                             GNX, GNY, GNZ, absorption, scattering, wl_start, wl_step, NW,
                                             USE_WEIGHTS, terminal, GRID)
            to_b = cont & bulk
            if GRID:
                n_to_b = _warp_count(to_b)
                if n_to_b > 0:
                    sb = _ring(qb_wr + _lane_rank(to_b), CAP)
                    _put(qbf, qbi, CAP, sb, to_b, Q_IDHI, x, y, z, dx, dy, dz, px, py, pz, wl, t, last, flags,
                         weight, step, row, last_wire, ids)
                    tl.store(qbf + Q_BT * CAP + sb, bulk_t, mask=to_b, cache_modifier=".cg")
                    tl.store(qbi + Q_BM * CAP + sb, bulk_m, mask=to_b, cache_modifier=".cg")
                    qb_wr += n_to_b
            res = cont & ~to_b
