"""Collision-first optical transport and GPU history/event queues.

This module contains the detector-independent hot loop used by the specialised
Triton backend.  It deliberately does *not* guess where the next surface is.
Instead, callers provide an axis-aligned box which is certified to contain only
one homogeneous material.  A photon may be advanced without a geometry query
only while it remains strictly inside that box.  At the certified exit it is
placed on the boundary-event queue for an exact geometry query.

The construction is exact in distribution for homogeneous, non-reemitting
media.  Absorption and Rayleigh scattering are sampled as one competing-hazard
process.  If the artificial box exit wins, the photon is advanced to a point
strictly before that exit.  The exponential residual can then be resampled by
memorylessness in the boundary stage.  Merely discarding the conditioned draw
without advancing would be biased; the implementation below never does that.

The opt-in alignment tape follows a different handoff rule.  It samples the
two independent Chroma exponentials and leaves both the photon and its tape
cursors unchanged when the artificial exit wins.  This lets a tape-aware exact
boundary stage reuse the same semantic draws.  The ordinary fast path and its
production behavior are unchanged.

Torch and Triton are optional Chroma dependencies, so their imports are lazy.
The NumPy reference routines remain usable in a CPU-only Chroma installation.
"""

from __future__ import division

from collections import namedtuple

import numpy as np

from .physics import rayleigh_scatter as _physics_rayleigh_scatter_reference


# Queue/event states.  Values are intentionally small so status may eventually
# be stored as uint8; int32 is currently used because it is friendlier to GPU
# atomics and debugging tools.
CONTINUE = 0
BOUNDARY = 1
ABSORBED = 2
INVALID = 3

# Chroma history flags (chroma/cuda/photon.h).
NO_HIT = 1 << 0
BULK_ABSORB = 1 << 1
RAYLEIGH_SCATTER = 1 << 4
NAN_ABORT = 1 << 15

SPEED_OF_LIGHT_MM_PER_NS = np.float32(299.792458)


EpochResult = namedtuple(
    "EpochResult", ["status", "scatter_count", "rng_counter"]
)


def _as_vec3(value, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError("%s must have shape (3,), got %r" % (name, value.shape))
    return value


def certified_aabb_exit_distance(position, direction, lower, upper,
                                 absolute_guard=1.0e-2,
                                 relative_guard=2.0e-6):
    """Return a conservative distance to the exit of an empty AABB.

    ``lower`` and ``upper`` must describe a box which has independently been
    proved to contain no boundary.  A downward guard is applied to cover the
    float32 arithmetic used by the GPU implementation.  Points outside the
    closed box return zero and therefore receive no geometry-free advance.

    Parameters are broadcast over ``position[..., 3]`` and
    ``direction[..., 3]``.  The returned array has shape ``position.shape[:-1]``.
    """

    position = np.asarray(position, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    lower = _as_vec3(lower, "lower")
    upper = _as_vec3(upper, "upper")
    if position.shape != direction.shape or position.shape[-1:] != (3,):
        raise ValueError("position and direction must have matching (..., 3) shapes")
    if np.any(lower >= upper):
        raise ValueError("lower must be strictly smaller than upper")

    inside = np.all((position >= lower) & (position <= upper), axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = np.where(
            direction > 0.0,
            (upper - position) / direction,
            np.where(direction < 0.0, (lower - position) / direction, np.inf),
        )
    exit_distance = np.min(distance, axis=-1)
    guard = absolute_guard + relative_guard * np.abs(exit_distance)
    safe = np.maximum(exit_distance - guard, 0.0)
    return np.where(inside & np.isfinite(safe), safe, 0.0)


def _rayleigh_scatter_reference(direction, polarization, uniforms):
    """Distribution-exact, angle-free Rayleigh update for CPU validation.

    The median of three uniforms is Beta(2, 2).  Mapping it to ``[-1, 1]``
    gives density ``3/4 (1-cos(theta)**2)``, exactly the polarized Rayleigh
    law used by Chroma.  The fourth uniform selects azimuth.
    """

    u = np.asarray(uniforms, dtype=np.float64)
    new_direction, new_polarization = _physics_rayleigh_scatter_reference(
        np.asarray(direction, dtype=np.float64),
        np.asarray(polarization, dtype=np.float64),
        u[0], u[1], u[2], u[3],
    )
    return new_direction.astype(np.float32), new_polarization.astype(np.float32)


def collision_first_epoch_reference(
        positions,
        directions,
        polarizations,
        times,
        histories,
        rng,
        lower,
        upper,
        absorption_length,
        scattering_length,
        refractive_index,
        max_scatter=4,
        absolute_guard=1.0e-2,
        relative_guard=2.0e-6):
    """NumPy reference for one collision-history epoch.

    Arrays are updated in-place, matching the GPU API.  This path is intended
    for invariants and statistical tests, not production.  The target detector
    has no bulk re-emission at 450 nm; finite re-emission components and weighted
    transport must stay on the general Chroma path.
    """

    positions = np.asarray(positions)
    directions = np.asarray(directions)
    polarizations = np.asarray(polarizations)
    times = np.asarray(times)
    histories = np.asarray(histories)
    nphotons = positions.shape[0]
    if directions.shape != (nphotons, 3) or polarizations.shape != (nphotons, 3):
        raise ValueError("photon vectors must have shape (N, 3)")
    if times.shape != (nphotons,) or histories.shape != (nphotons,):
        raise ValueError("times and histories must have shape (N,)")
    if max_scatter < 1:
        raise ValueError("max_scatter must be positive")

    inv_abs = 0.0 if np.isinf(absorption_length) else 1.0 / float(absorption_length)
    inv_scat = 0.0 if np.isinf(scattering_length) else 1.0 / float(scattering_length)
    rate = inv_abs + inv_scat
    if rate <= 0.0:
        absorption_probability = 0.0
    else:
        absorption_probability = inv_abs / rate

    status = np.full(nphotons, CONTINUE, dtype=np.int32)
    scatter_count = np.zeros(nphotons, dtype=np.int32)
    rng_counter = np.zeros(nphotons, dtype=np.int64)
    speed = float(SPEED_OF_LIGHT_MM_PER_NS) / float(refractive_index)

    for photon_id in range(nphotons):
        for _ in range(max_scatter):
            values = rng.random(6)
            rng_counter[photon_id] += 2  # two Philox4 calls on the GPU
            safe_distance = float(certified_aabb_exit_distance(
                positions[photon_id:photon_id + 1],
                directions[photon_id:photon_id + 1],
                lower,
                upper,
                absolute_guard,
                relative_guard,
            )[0])
            if rate <= 0.0:
                collision_distance = np.inf
            else:
                collision_distance = -np.log(max(values[0], 2.3283064e-10)) / rate

            if collision_distance < safe_distance:
                positions[photon_id] += collision_distance * directions[photon_id]
                times[photon_id] += collision_distance / speed
                if values[1] < absorption_probability:
                    histories[photon_id] |= BULK_ABSORB
                    status[photon_id] = ABSORBED
                    break
                directions[photon_id], polarizations[photon_id] = (
                    _rayleigh_scatter_reference(
                        directions[photon_id], polarizations[photon_id], values[2:6]
                    )
                )
                histories[photon_id] |= RAYLEIGH_SCATTER
                scatter_count[photon_id] += 1
            else:
                # Advancing is essential: conditioned exponential residuals may
                # only be discarded after reaching this certified point.
                positions[photon_id] += safe_distance * directions[photon_id]
                times[photon_id] += safe_distance / speed
                status[photon_id] = BOUNDARY
                break

    return EpochResult(status, scatter_count, rng_counter)


def _load_gpu_kernels():
    """Create and memoize Triton kernels without imposing a hard dependency."""

    cached = getattr(_load_gpu_kernels, "_cached", None)
    if cached is not None:
        return cached

    try:
        import triton
        import triton.language as tl
        from triton.language.extra import libdevice
        from .physics_kernels import (
            acos_chroma_fast,
            advance_position_chroma_fast,
            advance_time_chroma_fast,
            exponential_distance_chroma_fast,
            rayleigh_scatter as physics_rayleigh_scatter,
        )
        from .rng_alignment import random_tape_uniform_at
    except ImportError as exc:  # pragma: no cover - exercised on CPU installations
        raise RuntimeError("Triton transport requires the optional triton package") from exc

    # Triton's AST frontend resolves annotations and language intrinsics from
    # the defining module's globals rather than the Python closure.  Install the
    # optional modules only after a GPU caller has requested them.
    globals()["triton"] = triton
    globals()["tl"] = tl
    globals()["libdevice"] = libdevice
    globals()["acos_chroma_fast"] = acos_chroma_fast
    globals()["advance_position_chroma_fast"] = advance_position_chroma_fast
    globals()["advance_time_chroma_fast"] = advance_time_chroma_fast
    globals()["exponential_distance_chroma_fast"] = (
        exponential_distance_chroma_fast
    )
    globals()["physics_rayleigh_scatter"] = physics_rayleigh_scatter
    globals()["random_tape_uniform_at"] = random_tape_uniform_at

    @triton.jit
    def legacy_pick_new_direction(axis_x, axis_y, axis_z, theta, phi):
        """Literal Triton translation of cuda/photon.h pick_new_direction."""

        cos_theta = libdevice.fast_cosf(theta)
        sin_theta = libdevice.fast_sinf(theta)
        cos_phi = libdevice.fast_cosf(phi)
        sin_phi = libdevice.fast_sinf(phi)
        sin_axis_theta = tl.sqrt(1.0 - axis_z * axis_z)
        ordinary_axis = (sin_axis_theta == sin_axis_theta) & (
            sin_axis_theta >= 0.00001
        )
        cos_axis_phi = tl.where(ordinary_axis, axis_x / sin_axis_theta, 1.0)
        sin_axis_phi = tl.where(ordinary_axis, axis_y / sin_axis_theta, 0.0)
        out_x = cos_theta * axis_x + sin_theta * (
            axis_z * cos_phi * cos_axis_phi - sin_phi * sin_axis_phi
        )
        out_y = cos_theta * axis_y + sin_theta * (
            cos_phi * axis_z * sin_axis_phi + sin_phi * cos_axis_phi
        )
        out_z = cos_theta * axis_z - sin_theta * cos_phi * sin_axis_theta
        return out_x, out_y, out_z

    globals()["legacy_pick_new_direction"] = legacy_pick_new_direction

    @triton.jit
    def legacy_rayleigh_scatter(qx, qy, qz, uniform_theta, uniform_phi):
        """Literal two-draw Rayleigh sampler from cuda/photon.h."""

        pi: tl.constexpr = 3.14159265358979323846
        cos_theta = 2.0 * libdevice.fast_cosf(
            (acos_chroma_fast(1.0 - 2.0 * uniform_theta) - 2.0 * pi) / 3.0
        )
        cos_theta = tl.maximum(-1.0, tl.minimum(1.0, cos_theta))
        theta = acos_chroma_fast(cos_theta)
        phi = uniform_phi * (2.0 * pi)
        dx, dy, dz = legacy_pick_new_direction(
            qx, qy, qz, theta, phi
        )
        special = 1.0 - tl.abs(cos_theta) < 1.0e-6
        special_qx, special_qy, special_qz = legacy_pick_new_direction(
            qx, qy, qz, pi / 2.0, phi
        )
        ordinary_qx = qx - cos_theta * dx
        ordinary_qy = qy - cos_theta * dy
        ordinary_qz = qz - cos_theta * dz
        new_qx = tl.where(special, special_qx, ordinary_qx)
        new_qy = tl.where(special, special_qy, ordinary_qy)
        new_qz = tl.where(special, special_qz, ordinary_qz)
        direction_norm = tl.sqrt(dx * dx + dy * dy + dz * dz)
        polarization_norm = tl.sqrt(
            new_qx * new_qx + new_qy * new_qy + new_qz * new_qz
        )
        return (
            dx / direction_norm,
            dy / direction_norm,
            dz / direction_norm,
            new_qx / polarization_norm,
            new_qy / polarization_norm,
            new_qz / polarization_norm,
        )

    globals()["legacy_rayleigh_scatter"] = legacy_rayleigh_scatter

    @triton.jit(do_not_specialize=[18, 29, 42, 43])
    def collision_first_kernel(
            positions,
            directions,
            polarizations,
            times,
            histories,
            rng_counters,
            step_counts,
            last_instances,
            last_triangles,
            input_queue,
            global_photon_ids,
            tape_values,
            tape_global_ids,
            tape_row_indices,
            tape_interaction_cursor,
            tape_draw_cursor,
            tape_overflow,
            tape_interaction_certificate,
            tape_photon_count,
            statuses,
            scatter_counts,
            continuing_queue,
            continuing_count,
            boundary_queue,
            boundary_count,
            absorbed_queue,
            absorbed_count,
            invalid_queue,
            invalid_count,
            nitems,
            state_certificate_words,
            lower_x: tl.constexpr,
            lower_y: tl.constexpr,
            lower_z: tl.constexpr,
            upper_x: tl.constexpr,
            upper_y: tl.constexpr,
            upper_z: tl.constexpr,
            inv_absorption: tl.constexpr,
            inv_scattering: tl.constexpr,
            absorption_length: tl.constexpr,
            scattering_length: tl.constexpr,
            refractive_index: tl.constexpr,
            seed,
            photon_id_base,
            MAX_SCATTER: tl.constexpr,
            MAX_STEPS: tl.constexpr,
            TRACK_STEPS: tl.constexpr,
            TRACK_LAST_HIT: tl.constexpr,
            TRACK_GLOBAL_IDS: tl.constexpr,
            USE_RANDOM_TAPE: tl.constexpr,
            USE_TAPE_ROWS: tl.constexpr,
            CERTIFY_RANDOM_TAPE: tl.constexpr,
            MAX_TAPE_INTERACTIONS: tl.constexpr,
            TAPE_DRAWS_PER_INTERACTION: tl.constexpr,
            CERTIFY_STATE: tl.constexpr,
            STATE_WORDS_PER_INTERACTION: tl.constexpr,
            STATE_WAVELENGTH_WORD: tl.constexpr,
            STATE_WEIGHT_WORD: tl.constexpr,
            STATE_EVIDX_WORD: tl.constexpr,
            STORE_STATUS: tl.constexpr,
            EMIT_ACTIVE: tl.constexpr,
            EMIT_TERMINAL: tl.constexpr,
            NITEMS_IS_POINTER: tl.constexpr,
            BLOCK: tl.constexpr):
        program_start = tl.program_id(0) * BLOCK
        if NITEMS_IS_POINTER:
            live_items = tl.load(nitems).to(tl.int32)
            # Device-queue epochs deliberately launch a capacity-sized grid
            # so several producer/consumer kernels can be enqueued without a
            # host count read.  A lane mask alone does not make Triton's
            # expensive static collision history conditional: empty suffix
            # CTAs would still execute it.  Retire each wholly empty program
            # after the single device-counter load instead.
            if program_start >= live_items:
                return
        else:
            live_items = nitems
        lane = program_start + tl.arange(0, BLOCK)
        mask = lane < live_items
        queued_photon_id = tl.load(input_queue + lane, mask=mask, other=0)
        photon_id = queued_photon_id.to(tl.int64)
        if TRACK_GLOBAL_IDS:
            global_photon_id = tl.load(
                global_photon_ids + photon_id, mask=mask, other=0
            ).to(tl.int64)
        else:
            global_photon_id = photon_id + photon_id_base.to(tl.int64)
        base = photon_id * 3

        px = tl.load(positions + base, mask=mask, other=0.0)
        py = tl.load(positions + base + 1, mask=mask, other=0.0)
        pz = tl.load(positions + base + 2, mask=mask, other=0.0)
        dx = tl.load(directions + base, mask=mask, other=1.0)
        dy = tl.load(directions + base + 1, mask=mask, other=0.0)
        dz = tl.load(directions + base + 2, mask=mask, other=0.0)
        qx = tl.load(polarizations + base, mask=mask, other=0.0)
        qy = tl.load(polarizations + base + 1, mask=mask, other=1.0)
        qz = tl.load(polarizations + base + 2, mask=mask, other=0.0)
        photon_time = tl.load(times + photon_id, mask=mask, other=0.0)
        history = tl.load(histories + photon_id, mask=mask, other=0).to(tl.int32)
        rng_counter = tl.load(rng_counters + photon_id, mask=mask, other=0).to(tl.int64)
        if TRACK_STEPS:
            step_count = tl.load(
                step_counts + photon_id, mask=mask, other=MAX_STEPS
            ).to(tl.int32)
        else:
            step_count = tl.zeros((BLOCK,), tl.int32)

        if USE_RANDOM_TAPE:
            if USE_TAPE_ROWS:
                tape_row = tl.load(
                    tape_row_indices + photon_id, mask=mask, other=-1
                ).to(tl.int64)
            else:
                tape_row = photon_id
            tape_row_valid = mask & (tape_row >= 0) & (
                tape_row < tape_photon_count
            )
            stored_tape_global_id = tl.load(
                tape_global_ids + tape_row,
                mask=tape_row_valid,
                other=-2,
            ).to(tl.int64)
            tape_mapping_ok = tape_row_valid & (
                stored_tape_global_id == global_photon_id
            )
            tape_interaction = tl.load(
                tape_interaction_cursor + tape_row,
                mask=tape_row_valid,
                other=0,
            ).to(tl.int32)
            tape_draw = tl.load(
                tape_draw_cursor + tape_row,
                mask=tape_row_valid,
                other=0,
            ).to(tl.int32)
            tape_flags = tl.load(
                tape_overflow + tape_row,
                mask=tape_row_valid,
                other=0,
            ).to(tl.int32)
            tape_flags |= tl.where(mask & ~tape_row_valid, 8, 0)
            tape_flags |= tl.where(
                tape_row_valid & ~tape_mapping_ok, 4, 0
            )
            tape_interaction_ok = (
                (tape_interaction >= 0)
                & (tape_interaction < MAX_TAPE_INTERACTIONS)
            )
            tape_flags |= tl.where(
                tape_mapping_ok & ~tape_interaction_ok, 2, 0
            )
        else:
            tape_row = photon_id
            tape_row_valid = mask
            tape_mapping_ok = mask
            tape_interaction = tl.zeros((BLOCK,), tl.int32)
            tape_draw = tl.zeros((BLOCK,), tl.int32)
            tape_flags = tl.zeros((BLOCK,), tl.int32)
            tape_interaction_ok = mask

        # Literal values mirror the public constants above.  Triton 3.1 does
        # not allow ordinary Python globals inside a JIT function.
        status = tl.full((BLOCK,), 0, tl.int32)  # CONTINUE
        nscatter = tl.zeros((BLOCK,), tl.int32)
        had_bulk_collision = tl.zeros((BLOCK,), tl.int1)
        active = mask & (step_count < MAX_STEPS)
        rate: tl.constexpr = inv_absorption + inv_scattering
        if rate > 0.0:
            absorb_probability: tl.constexpr = inv_absorption / rate
        else:
            absorb_probability: tl.constexpr = 0.0
        velocity: tl.constexpr = 299.792458 / refractive_index
        huge: tl.constexpr = 1.0e30

        for _ in tl.static_range(MAX_SCATTER):
            # NaN is the only IEEE value unequal to itself; this works on
            # Triton releases predating tl.isnan/tl.libdevice exposure.
            finite = (
                (px == px) & (py == py) & (pz == pz) &
                (dx == dx) & (dy == dy) & (dz == dz)
            )
            bad = active & ~finite
            status = tl.where(bad, 3, status)  # INVALID
            history = tl.where(bad, history | (1 << 0) | (1 << 15), history)
            active = active & finite

            inside = active & (
                (px >= lower_x) & (px <= upper_x) &
                (py >= lower_y) & (py <= upper_y) &
                (pz >= lower_z) & (pz <= upper_z)
            )
            status = tl.where(active & ~inside, 1, status)  # BOUNDARY
            active = inside

            tx = tl.where(
                dx > 0.0, (upper_x - px) / dx,
                tl.where(dx < 0.0, (lower_x - px) / dx, huge),
            )
            ty = tl.where(
                dy > 0.0, (upper_y - py) / dy,
                tl.where(dy < 0.0, (lower_y - py) / dy, huge),
            )
            tz = tl.where(
                dz > 0.0, (upper_z - pz) / dz,
                tl.where(dz < 0.0, (lower_z - pz) / dz, huge),
            )
            exit_distance = tl.minimum(tx, tl.minimum(ty, tz))
            # At detector-scale coordinates one float32 ulp is larger than
            # Chroma's 1e-4-mm ray epsilon.  A 1e-4 guard can therefore round
            # the advanced point back onto the artificial box face; an
            # outward post-Rayleigh direction then sees no positive-distance
            # surface and is incorrectly declared NO_HIT.  The exponential
            # memoryless split is exact at *any* interior point, so retain a
            # comfortably representable 10-micron guard.
            guard = 1.0e-2 + 2.0e-6 * tl.abs(exit_distance)
            safe_distance = tl.maximum(exit_distance - guard, 0.0)

            if USE_RANDOM_TAPE:
                # A tape interaction corresponds to one real Chroma transport
                # interaction, never to an artificial certified-box split.  We
                # therefore peek at the distance draws here and leave both the
                # position and cursors unchanged when neither bulk process wins
                # before the safe exit.  A future exact-boundary tape consumer
                # can reuse the same semantic draw slots.
                prior_tape_failure = active & (tape_flags != 0)
                status = tl.where(prior_tape_failure, 3, status)  # INVALID
                history = tl.where(
                    prior_tape_failure,
                    history | (1 << 0) | (1 << 15),
                    history,
                )
                active &= tape_flags == 0
                distance_draws_ok = (
                    active
                    & tape_mapping_ok
                    & tape_interaction_ok
                    & (tape_draw >= 0)
                    & (tape_draw + 1 < TAPE_DRAWS_PER_INTERACTION)
                )
                uniform_absorption = random_tape_uniform_at(
                    tape_values,
                    tape_row,
                    tape_interaction,
                    tape_draw,
                    distance_draws_ok,
                    MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                    DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
                )
                uniform_scattering = random_tape_uniform_at(
                    tape_values,
                    tape_row,
                    tape_interaction,
                    tape_draw + 1,
                    distance_draws_ok,
                    MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                    DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
                )
                if inv_absorption == 0.0:
                    absorption_distance = tl.full(
                        (BLOCK,), huge, tl.float32
                    )
                else:
                    absorption_distance = exponential_distance_chroma_fast(
                        absorption_length, uniform_absorption
                    )
                if inv_scattering == 0.0:
                    scattering_distance = tl.full(
                        (BLOCK,), huge, tl.float32
                    )
                else:
                    scattering_distance = exponential_distance_chroma_fast(
                        scattering_length, uniform_scattering
                    )
                absorb_candidate = (
                    distance_draws_ok
                    & (absorption_distance <= scattering_distance)
                    & (absorption_distance <= safe_distance)
                )
                scatter_candidate = (
                    distance_draws_ok
                    & (scattering_distance < absorption_distance)
                    & (scattering_distance <= safe_distance)
                )
                rayleigh_draws_ok = (
                    scatter_candidate
                    & (tape_draw + 3 < TAPE_DRAWS_PER_INTERACTION)
                )
                uniform_rayleigh_theta = random_tape_uniform_at(
                    tape_values,
                    tape_row,
                    tape_interaction,
                    tape_draw + 2,
                    rayleigh_draws_ok,
                    MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                    DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
                )
                uniform_rayleigh_phi = random_tape_uniform_at(
                    tape_values,
                    tape_row,
                    tape_interaction,
                    tape_draw + 3,
                    rayleigh_draws_ok,
                    MAX_INTERACTIONS=MAX_TAPE_INTERACTIONS,
                    DRAWS_PER_INTERACTION=TAPE_DRAWS_PER_INTERACTION,
                )
                draw_failure = active & (
                    ~distance_draws_ok
                    | (scatter_candidate & ~rayleigh_draws_ok)
                )
                tape_flags |= tl.where(draw_failure, 1, 0)
                attempted_draws = tl.where(scatter_candidate, 4, 2)
                saturated_draw = tl.minimum(
                    TAPE_DRAWS_PER_INTERACTION,
                    tl.maximum(tape_draw, 0) + attempted_draws,
                )
                tape_draw = tl.where(
                    draw_failure, saturated_draw, tape_draw
                )
                status = tl.where(draw_failure, 3, status)  # INVALID
                history = tl.where(
                    draw_failure,
                    history | (1 << 0) | (1 << 15),
                    history,
                )
                absorb = absorb_candidate & ~draw_failure
                scatter = scatter_candidate & ~draw_failure
                collide = absorb | scatter
                collision_distance = tl.where(
                    absorb, absorption_distance, scattering_distance
                )
                new_dx, new_dy, new_dz, new_qx, new_qy, new_qz = (
                    legacy_rayleigh_scatter(
                        qx,
                        qy,
                        qz,
                        uniform_rayleigh_theta,
                        uniform_rayleigh_phi,
                    )
                )
                # Preserve the tape coordinate used by this physical
                # interaction.  ``tape_interaction`` is advanced below, while
                # both the process and state certificates are indexed by the
                # pre-increment value.
                completed_interaction = collide
                completed_interaction_index = tape_interaction
                consumed_draws = tl.where(scatter, 4, 2)
                committed_draw_count = tape_draw + consumed_draws
                if CERTIFY_RANDOM_TAPE:
                    certificate_process = tl.where(absorb, 1, 2).to(
                        tl.uint32
                    )
                    certificate_word = (
                        (certificate_process << 28)
                        | committed_draw_count.to(tl.uint32)
                    )
                    certificate_offset = (
                        tape_row * MAX_TAPE_INTERACTIONS + tape_interaction
                    )
                    tl.store(
                        tape_interaction_certificate + certificate_offset,
                        certificate_word,
                        mask=(
                            completed_interaction
                            & tape_row_valid
                            & tape_mapping_ok
                            & tape_interaction_ok
                        ),
                    )
                tape_draw = tl.where(
                    completed_interaction, committed_draw_count, tape_draw
                )
                tape_interaction += completed_interaction.to(tl.int32)
                tape_draw = tl.where(completed_interaction, 0, tape_draw)
                tape_flags |= tl.where(
                    completed_interaction
                    & (tape_interaction >= MAX_TAPE_INTERACTIONS),
                    2,
                    0,
                )
                # Do not advance to the artificial exit in lockstep mode.
                advance = tl.where(collide, collision_distance, 0.0)
                valid_bulk_work = active & ~draw_failure
                needs_boundary = valid_bulk_work & ~collide
            else:
                # The high 32 bits select a photon stream and the low bits count
                # Philox4 invocations.  This remains stable under queue reordering.
                offset = global_photon_id * 4294967296 + rng_counter
                u0, u1, u2, u3 = tl.rand4x(seed, offset)
                u4, u5, _, _ = tl.rand4x(seed, offset + 1)
                rng_counter += active.to(tl.int64) * 2
                if rate > 0.0:
                    collision_distance = (
                        -tl.log(tl.maximum(u0, 2.3283064e-10)) / rate
                    )
                else:
                    collision_distance = tl.full(
                        (BLOCK,), huge, tl.float32
                    )
                collide = active & (collision_distance < safe_distance)
                advance = tl.where(collide, collision_distance, safe_distance)
                absorb = collide & (u1 < absorb_probability)
                scatter = collide & ~absorb
                # Reuse the standalone, CPU-parity-tested fast primitive.  Its
                # bounded four-draw sampler is intentionally retained outside
                # tape alignment mode.
                new_dx, new_dy, new_dz, new_qx, new_qy, new_qz = (
                    physics_rayleigh_scatter(
                        dx, dy, dz, qx, qy, qz, u2, u3, u4, u5
                    )
                )
                needs_boundary = active & ~collide

            had_bulk_collision |= collide
            if USE_RANDOM_TAPE:
                px = tl.where(
                    active,
                    advance_position_chroma_fast(px, advance, dx),
                    px,
                )
                py = tl.where(
                    active,
                    advance_position_chroma_fast(py, advance, dy),
                    py,
                )
                pz = tl.where(
                    active,
                    advance_position_chroma_fast(pz, advance, dz),
                    pz,
                )
                photon_time = tl.where(
                    active,
                    advance_time_chroma_fast(
                        photon_time, advance, refractive_index
                    ),
                    photon_time,
                )
            else:
                px += tl.where(active, advance * dx, 0.0)
                py += tl.where(active, advance * dy, 0.0)
                pz += tl.where(active, advance * dz, 0.0)
                photon_time += tl.where(active, advance / velocity, 0.0)
            history = tl.where(absorb, history | (1 << 1), history)
            status = tl.where(absorb, 2, status)  # ABSORBED
            history = tl.where(scatter, history | (1 << 4), history)
            nscatter += scatter.to(tl.int32)
            # One collision is one Chroma propagation step.  The artificial
            # certified-box exit is not: its still-pending geometry step is
            # completed by the boundary queue.
            step_count += collide.to(tl.int32)
            if TRACK_STEPS:
                status = tl.where(
                    scatter & (step_count >= MAX_STEPS), 4, status
                )  # step-budget exhausted, alive but no longer scheduled
            dx = tl.where(scatter, new_dx, dx)
            dy = tl.where(scatter, new_dy, dy)
            dz = tl.where(scatter, new_dz, dz)
            qx = tl.where(scatter, new_qx, qx)
            qy = tl.where(scatter, new_qy, qy)
            qz = tl.where(scatter, new_qz, qz)
            if CERTIFY_STATE:
                # This debug-only transcript is deliberately adjacent to the
                # completed physical mutation.  It records neither an
                # artificial certified-box exit nor a failed tape attempt.
                # The logical layout is [tape row, interaction, Photon word]
                # and all floating fields are reinterpreted, never converted.
                state_mask = (
                    completed_interaction
                    & tape_row_valid
                    & tape_mapping_ok
                    & tape_interaction_ok
                )
                state_offset = (
                    (
                        tape_row * MAX_TAPE_INTERACTIONS
                        + completed_interaction_index
                    )
                    * STATE_WORDS_PER_INTERACTION
                )
                tl.store(
                    state_certificate_words + state_offset + 0,
                    px.to(tl.uint32, bitcast=True), mask=state_mask,
                )
                tl.store(
                    state_certificate_words + state_offset + 1,
                    py.to(tl.uint32, bitcast=True), mask=state_mask,
                )
                tl.store(
                    state_certificate_words + state_offset + 2,
                    pz.to(tl.uint32, bitcast=True), mask=state_mask,
                )
                tl.store(
                    state_certificate_words + state_offset + 3,
                    dx.to(tl.uint32, bitcast=True), mask=state_mask,
                )
                tl.store(
                    state_certificate_words + state_offset + 4,
                    dy.to(tl.uint32, bitcast=True), mask=state_mask,
                )
                tl.store(
                    state_certificate_words + state_offset + 5,
                    dz.to(tl.uint32, bitcast=True), mask=state_mask,
                )
                tl.store(
                    state_certificate_words + state_offset + 6,
                    qx.to(tl.uint32, bitcast=True), mask=state_mask,
                )
                tl.store(
                    state_certificate_words + state_offset + 7,
                    qy.to(tl.uint32, bitcast=True), mask=state_mask,
                )
                tl.store(
                    state_certificate_words + state_offset + 8,
                    qz.to(tl.uint32, bitcast=True), mask=state_mask,
                )
                state_wavelength_word = tl.full(
                    (BLOCK,), STATE_WAVELENGTH_WORD, tl.uint32
                )
                tl.store(
                    state_certificate_words + state_offset + 9,
                    state_wavelength_word,
                    mask=state_mask,
                )
                tl.store(
                    state_certificate_words + state_offset + 10,
                    photon_time.to(tl.uint32, bitcast=True), mask=state_mask,
                )
                tl.store(
                    state_certificate_words + state_offset + 11,
                    history.to(tl.uint32, bitcast=True), mask=state_mask,
                )
                # Chroma clears last_hit_triangle after every bulk
                # interaction.  The resident array is updated once below for
                # the whole register epoch, but the post-interaction value is
                # already known here.
                state_last_triangle = tl.full((BLOCK,), -1, tl.int32)
                tl.store(
                    state_certificate_words + state_offset + 12,
                    state_last_triangle.to(tl.uint32, bitcast=True),
                    mask=state_mask,
                )
                state_weight_word = tl.full(
                    (BLOCK,), STATE_WEIGHT_WORD, tl.uint32
                )
                tl.store(
                    state_certificate_words + state_offset + 13,
                    state_weight_word, mask=state_mask,
                )
                state_evidx_word = tl.full(
                    (BLOCK,), STATE_EVIDX_WORD, tl.uint32
                )
                tl.store(
                    state_certificate_words + state_offset + 14,
                    state_evidx_word, mask=state_mask,
                )
            status = tl.where(needs_boundary, 1, status)  # BOUNDARY
            active = scatter & (step_count < MAX_STEPS)

        # Lanes still active exhausted their short register-resident history
        # loop and are returned to the continuation queue.
        status = tl.where(active, 0, status)  # CONTINUE
        tl.store(positions + base, px, mask=mask)
        tl.store(positions + base + 1, py, mask=mask)
        tl.store(positions + base + 2, pz, mask=mask)
        tl.store(directions + base, dx, mask=mask)
        tl.store(directions + base + 1, dy, mask=mask)
        tl.store(directions + base + 2, dz, mask=mask)
        tl.store(polarizations + base, qx, mask=mask)
        tl.store(polarizations + base + 1, qy, mask=mask)
        tl.store(polarizations + base + 2, qz, mask=mask)
        tl.store(times + photon_id, photon_time, mask=mask)
        tl.store(histories + photon_id, history, mask=mask)
        tl.store(rng_counters + photon_id, rng_counter, mask=mask)
        if USE_RANDOM_TAPE:
            tl.store(
                tape_interaction_cursor + tape_row,
                tape_interaction,
                mask=tape_row_valid,
            )
            tl.store(
                tape_draw_cursor + tape_row,
                tape_draw,
                mask=tape_row_valid,
            )
            tl.store(
                tape_overflow + tape_row,
                tape_flags,
                mask=tape_row_valid,
            )
        if TRACK_STEPS:
            tl.store(step_counts + photon_id, step_count, mask=mask)
        if TRACK_LAST_HIT:
            # Chroma clears last_hit_triangle after every bulk interaction.
            # Without this, a boundary identity can survive Rayleigh scatters
            # performed inside the certified region and suppress a later,
            # physically distinct return to that boundary.
            tl.store(
                last_instances + photon_id, -1,
                mask=mask & had_bulk_collision,
            )
            tl.store(
                last_triangles + photon_id, -1,
                mask=mask & had_bulk_collision,
            )
        if STORE_STATUS:
            tl.store(statuses + lane, status, mask=mask)
        tl.store(scatter_counts + lane, nscatter, mask=mask)

        # Produce live queues in the same launch which computed the terminal
        # status.  This is the same block-local stable partition used by
        # compact_status_kernel, but it avoids writing and rereading an
        # intermediate status array and removes one compaction launch per
        # emitted queue.  Atomic block reservation intentionally leaves the
        # ordering between blocks unspecified, just like the compatibility
        # compactor below.
        if EMIT_ACTIVE:
            selected = mask & (status == 0)  # CONTINUE
            flag = selected.to(tl.int32)
            local_offset = tl.cumsum(flag, axis=0) - flag
            count = tl.sum(flag, axis=0)
            output_base = tl.atomic_add(continuing_count, count)
            tl.store(
                continuing_queue + output_base + local_offset,
                queued_photon_id,
                mask=selected,
            )

            selected = mask & (status == 1)  # BOUNDARY
            flag = selected.to(tl.int32)
            local_offset = tl.cumsum(flag, axis=0) - flag
            count = tl.sum(flag, axis=0)
            output_base = tl.atomic_add(boundary_count, count)
            tl.store(
                boundary_queue + output_base + local_offset,
                queued_photon_id,
                mask=selected,
            )

        if EMIT_TERMINAL:
            selected = mask & (status == 2)  # ABSORBED
            flag = selected.to(tl.int32)
            local_offset = tl.cumsum(flag, axis=0) - flag
            count = tl.sum(flag, axis=0)
            output_base = tl.atomic_add(absorbed_count, count)
            tl.store(
                absorbed_queue + output_base + local_offset,
                queued_photon_id,
                mask=selected,
            )

            selected = mask & (status == 3)  # INVALID
            flag = selected.to(tl.int32)
            local_offset = tl.cumsum(flag, axis=0) - flag
            count = tl.sum(flag, axis=0)
            output_base = tl.atomic_add(invalid_count, count)
            tl.store(
                invalid_queue + output_base + local_offset,
                queued_photon_id,
                mask=selected,
            )

    @triton.jit(do_not_specialize=[4])
    def compact_status_kernel(input_queue, statuses, output_queue, counter,
                              nitems, target: tl.constexpr,
                              BLOCK: tl.constexpr):
        lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = lane < nitems
        selected = mask & (tl.load(statuses + lane, mask=mask, other=-1) == target)
        flag = selected.to(tl.int32)
        local_offset = tl.cumsum(flag, axis=0) - flag
        count = tl.sum(flag, axis=0)
        output_base = tl.atomic_add(counter, count)
        photon_id = tl.load(input_queue + lane, mask=mask, other=0)
        tl.store(output_queue + output_base + local_offset, photon_id, mask=selected)

    cached = (triton, collision_first_kernel, compact_status_kernel)
    _load_gpu_kernels._cached = cached
    return cached


def _tensor_storage_overlaps(left, right):
    """Return whether two contiguous tensor views share any storage bytes."""
    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False
    left_begin = int(left.data_ptr())
    right_begin = int(right.data_ptr())
    left_end = left_begin + left.numel() * left.element_size()
    right_end = right_begin + right.numel() * right.element_size()
    return max(left_begin, right_begin) < min(left_end, right_end)


class DeviceQueue(object):
    """A device-resident queue buffer and its one-element device count.

    Producers treat ``count`` as the length of the live buffer prefix.  Callers
    must maintain ``0 <= count <= launch capacity``; checking that invariant on
    the host would defeat device-resident scheduling.
    """

    __slots__ = ("buffer", "count")

    def __init__(self, buffer, count):
        self.buffer = buffer
        self.count = count

    @classmethod
    def allocate(cls, capacity, *, device="cuda", dtype=None):
        """Allocate an empty queue while retaining lazy Torch imports."""
        import torch

        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("queue capacity cannot be negative")
        if dtype is None:
            dtype = torch.int32
        if dtype not in (torch.int32, torch.int64):
            raise TypeError("queue indices must use int32 or int64")
        buffer = torch.empty(capacity, dtype=dtype, device=device)
        count = torch.zeros(1, dtype=torch.int32, device=device)
        return cls(buffer, count)

    @property
    def capacity(self):
        return int(self.buffer.numel())

    def reset(self):
        """Asynchronously clear the device count without touching storage."""
        self.count.zero_()
        return self

    def size(self):
        """Synchronize and return the queue length as a Python integer."""
        return int(self.count.item())

    def tensor(self):
        """Synchronize and return the live prefix of the queue."""
        return self.buffer[:self.size()]


class CollisionQueueWorkspace(object):
    """Reusable storage for queues emitted by a collision producer.

    The first two rows hold the continuing and boundary queues.  Workspaces
    constructed with ``include_terminal=True`` add absorbed and invalid rows.
    Counts share one allocation so resetting every queue takes one device fill.
    Returned queues and status/scatter scratch are views of this workspace and
    must be consumed before it is reused by another epoch.  For a capacity
    launch from :class:`DeviceQueue`, only the device-count prefix of each
    available scratch array is valid.
    """

    __slots__ = (
        "storage", "counts", "include_terminal",
        "continuing", "boundary", "absorbed", "invalid",
        "status_scratch", "scatter_count_scratch",
    )

    def __init__(self, storage, counts, include_terminal=False,
                 status_scratch=None, scatter_count_scratch=None):
        import torch

        expected_rows = 4 if include_terminal else 2
        if not isinstance(storage, torch.Tensor) or not storage.is_cuda:
            raise ValueError("queue storage must be a CUDA tensor")
        if storage.ndim != 2 or storage.shape[0] != expected_rows:
            raise ValueError(
                "queue storage must have shape (%d, capacity)" % expected_rows
            )
        if storage.dtype not in (torch.int32, torch.int64):
            raise TypeError("queue storage must use int32 or int64 indices")
        if not storage.is_contiguous():
            raise ValueError("queue storage must be contiguous")
        if (
            not isinstance(counts, torch.Tensor)
            or not counts.is_cuda
            or counts.device != storage.device
            or counts.dtype != torch.int32
            or counts.shape != (expected_rows,)
            or not counts.is_contiguous()
        ):
            raise ValueError(
                "queue counts must be contiguous CUDA int32 with shape (%d,)"
                % expected_rows
            )
        self.storage = storage
        self.counts = counts
        self.include_terminal = bool(include_terminal)
        if status_scratch is None and include_terminal:
            status_scratch = torch.empty(
                storage.shape[1], dtype=torch.int32, device=storage.device
            )
        if scatter_count_scratch is None:
            scatter_count_scratch = torch.empty(
                storage.shape[1], dtype=torch.int32, device=storage.device
            )
        scratch_values = [(scatter_count_scratch, "scatter_count_scratch")]
        if status_scratch is not None:
            scratch_values.append((status_scratch, "status_scratch"))
        for scratch, name in scratch_values:
            if (
                not isinstance(scratch, torch.Tensor)
                or not scratch.is_cuda
                or scratch.device != storage.device
                or scratch.dtype != torch.int32
                or scratch.shape != (storage.shape[1],)
                or not scratch.is_contiguous()
            ):
                raise ValueError(
                    "%s must be contiguous CUDA int32 with shape (capacity,)"
                    % name
                )
        self.status_scratch = status_scratch
        self.scatter_count_scratch = scatter_count_scratch
        self.continuing = DeviceQueue(storage[0], counts[0:1])
        self.boundary = DeviceQueue(storage[1], counts[1:2])
        if include_terminal:
            self.absorbed = DeviceQueue(storage[2], counts[2:3])
            self.invalid = DeviceQueue(storage[3], counts[3:4])
        else:
            self.absorbed = None
            self.invalid = None

    @classmethod
    def allocate(cls, capacity, *, device="cuda", dtype=None,
                 include_terminal=False):
        """Allocate a workspace without importing Torch at module import time."""
        import torch

        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("queue capacity cannot be negative")
        if dtype is None:
            dtype = torch.int32
        rows = 4 if include_terminal else 2
        storage = torch.empty(
            (rows, capacity), dtype=dtype, device=device
        )
        counts = torch.zeros(rows, dtype=torch.int32, device=device)
        status_scratch = (
            torch.empty(capacity, dtype=torch.int32, device=device)
            if include_terminal else None
        )
        scatter_count_scratch = torch.empty(
            capacity, dtype=torch.int32, device=device
        )
        return cls(
            storage,
            counts,
            include_terminal=include_terminal,
            status_scratch=status_scratch,
            scatter_count_scratch=scatter_count_scratch,
        )

    @property
    def capacity(self):
        return int(self.storage.shape[1])

    def reset(self):
        """Asynchronously clear all queue counts for the next producer launch."""
        self.counts.zero_()
        return self

    def validate_for(self, input_buffer, capacity=None, include_terminal=False):
        """Validate capacity, device, index dtype, and requested partitions."""
        if capacity is None:
            capacity = input_buffer.numel()
        if self.storage.device != input_buffer.device:
            raise ValueError("queue workspace and input queue must share a device")
        if self.storage.dtype != input_buffer.dtype:
            raise TypeError("queue workspace index dtype must match input_queue")
        if self.capacity < int(capacity):
            raise ValueError("queue workspace capacity is smaller than input queue")
        if include_terminal and not self.include_terminal:
            raise ValueError("terminal partitioning requires a four-queue workspace")
        return self


DeviceEpochResult = namedtuple(
    "DeviceEpochResult",
    ["status", "scatter_count", "continuing", "boundary", "absorbed", "invalid"],
)


def compact_status(input_queue, statuses, target, output=None, count=None,
                   block_size=256):
    """Compact queue IDs matching ``target`` with one atomic per GPU block.

    The returned count remains on device.  Keeping counts device-resident lets a
    future persistent scheduler consume them without a CPU round trip; ``size``
    and ``tensor`` are explicit synchronization points for convenient callers.
    """

    import torch

    triton, _, kernel = _load_gpu_kernels()
    if not input_queue.is_cuda or not statuses.is_cuda:
        raise ValueError("queue compaction requires CUDA tensors")
    nitems = input_queue.numel()
    if output is None:
        output = torch.empty_like(input_queue)
    if count is None:
        count = torch.zeros(1, dtype=torch.int32, device=input_queue.device)
    else:
        count.zero_()
    grid = (triton.cdiv(nitems, block_size),)
    kernel[grid](input_queue, statuses, output, count, nitems,
                 target=target, BLOCK=block_size, num_warps=4)
    return DeviceQueue(output, count)


def collision_first_epoch(
        positions,
        directions,
        polarizations,
        times,
        histories,
        rng_counters,
        input_queue,
        lower,
        upper,
        absorption_length,
        scattering_length,
        refractive_index,
        seed=1,
        photon_id_base=0,
        max_scatter=4,
        block_size=256,
        partition=True,
        step_counts=None,
        max_steps=None,
        last_instances=None,
        last_triangles=None,
        global_photon_ids=None,
        queue_workspace=None,
        input_capacity=None,
        boundary_accumulator=None,
        append_boundary=False,
        random_tape=None,
        tape_audit=None,
        tape_row_indices=None,
        tape_certificate=None,
        state_certificate=None,
        state_wavelength=None,
        state_weight=None,
        state_evidx=None):
    """Launch a collision-history epoch on a CUDA photon queue.

    Photon arrays are updated in-place and use array-of-vec3 layout ``(N, 3)``.
    ``input_queue`` contains local state indices.  By default, adding
    ``photon_id_base`` gives each one a stable Philox identity independent of
    event-queue ordering.  When ``global_photon_ids`` is supplied, RNG identity
    is instead loaded from that array and ``photon_id_base`` is ignored; output
    queues still contain the local state indices.

    ``partition=False`` returns per-lane status without queues.  ``partition``
    equal to ``True`` retains that status and directly emits all four public
    queues.  The production ``partition="active"`` mode omits the intermediate
    status allocation and emits only continuing and boundary queues.  Supplying
    a reusable :class:`CollisionQueueWorkspace` avoids queue allocation between
    epochs.  The producer resets its counters before launch.  Queue, status,
    and scatter-count results backed by that workspace are invalidated when the
    workspace is reused.

    ``input_queue`` may also be a :class:`DeviceQueue`.  In that form the
    producer launches over ``input_capacity`` while reading the live item count
    on device, which permits several ping-pong epochs without a host poll.  An
    external ``boundary_accumulator`` can collect boundary IDs across those
    epochs; set ``append_boundary=True`` after resetting it once to preserve its
    existing device count.  Its capacity must cover the cumulative appended
    population, which the device producer does not resize or poll.

    ``random_tape`` and ``tape_audit`` opt into the Chroma draw-order alignment
    path.  ``tape_row_indices`` maps each local state slot to the tape row for
    its stable global photon ID; when omitted, state slots and tape rows must be
    identical.  This debug path mutates the row-indexed audit buffers and never
    falls back to Philox after overflow.  It consumes distance slots 0 and 1,
    followed by Rayleigh angle slots 2 and 3 only when scattering wins.  Draw,
    interaction, and global-ID failures remain sticky in ``tape_audit.overflow``
    and fail the affected lane closed as ``INVALID``.  Bulk reemission,
    weighted transport, and forced-scatter modes are not represented by this
    homogeneous collision API and therefore are not silently approximated.
    ``tape_certificate`` optionally records one packed process/draw word for
    every committed bulk interaction; artificial certified-box exits do not
    consume an interaction and leave the ledger untouched.

    ``state_certificate`` optionally records the post-interaction raw state at
    that same tape coordinate.  It may be a contiguous CUDA int32/uint32
    tensor, or a wrapper exposing one as ``.words``, with logical shape
    ``[tape_row, max_interactions, 15]`` (a flat tensor of the same size is also
    accepted).  The words are position xyz, direction xyz, polarization xyz,
    wavelength, time, history, last triangle, weight, and evidx.  The fixed
    ``state_wavelength``, ``state_weight``, and ``state_evidx`` must be supplied
    explicitly because this homogeneous specialization does not otherwise
    carry those invariant per-photon arrays.  State certification requires
    random-tape mode and is a constexpr-disabled, zero-store production path
    when omitted.

    Supplying the optional paired ``last_instances``/``last_triangles`` arrays
    makes every bulk collision clear the previous geometry identity, matching
    Chroma's ``last_hit_triangle = -1`` behavior after Rayleigh scattering or
    absorption.
    """

    import torch

    triton, kernel, _ = _load_gpu_kernels()
    input_count_is_pointer = isinstance(input_queue, DeviceQueue)
    if input_count_is_pointer:
        input_buffer = input_queue.buffer
        input_count = input_queue.count
        if (
            not isinstance(input_buffer, torch.Tensor)
            or not input_buffer.is_cuda
            or input_buffer.ndim != 1
            or not input_buffer.is_contiguous()
            or input_buffer.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(
                "DeviceQueue buffer must be contiguous one-dimensional CUDA indices"
            )
        if (
            not isinstance(input_count, torch.Tensor)
            or not input_count.is_cuda
            or input_count.device != input_buffer.device
            or input_count.dtype != torch.int32
            or input_count.shape != (1,)
            or not input_count.is_contiguous()
        ):
            raise ValueError(
                "DeviceQueue count must be contiguous CUDA int32 with shape (1,)"
            )
        launch_capacity = (
            input_buffer.numel() if input_capacity is None
            else int(input_capacity)
        )
        if launch_capacity < 0 or launch_capacity > input_buffer.numel():
            raise ValueError("input_capacity must fit the DeviceQueue buffer")
        nitems_argument = input_count
    else:
        input_buffer = input_queue
        if not isinstance(input_buffer, torch.Tensor):
            raise TypeError("input_queue must be a CUDA tensor or DeviceQueue")
        if input_capacity is not None and int(input_capacity) != input_buffer.numel():
            raise ValueError(
                "input_capacity is only variable for DeviceQueue input"
            )
        launch_capacity = input_buffer.numel()
        nitems_argument = launch_capacity
    lower = _as_vec3(lower, "lower")
    upper = _as_vec3(upper, "upper")
    if np.any(lower >= upper):
        raise ValueError("lower must be strictly smaller than upper")
    tensors = (positions, directions, polarizations, times, histories,
               rng_counters, input_buffer)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("collision_first_epoch requires CUDA tensors")
    if positions.dtype != torch.float32 or positions.ndim != 2 or positions.shape[1] != 3:
        raise TypeError("positions must be contiguous CUDA float32 with shape (N, 3)")
    if directions.shape != positions.shape or polarizations.shape != positions.shape:
        raise ValueError("directions and polarizations must match positions")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all collision-first tensors must be contiguous")
    if histories.dtype not in (torch.int32, torch.uint32):
        raise TypeError("histories must be int32 or uint32")
    if rng_counters.dtype != torch.int64:
        raise TypeError("rng_counters must be int64")
    if input_buffer.dtype not in (torch.int32, torch.int64):
        raise TypeError("input_queue must be int32 or int64")
    if partition is False:
        partition_mode = "status"
    elif partition is True:
        partition_mode = "all"
    elif partition == "active":
        partition_mode = "active"
    else:
        raise ValueError("partition must be False, True, or 'active'")
    track_steps = step_counts is not None
    if track_steps:
        if not step_counts.is_cuda or not step_counts.is_contiguous():
            raise ValueError("step_counts must be a contiguous CUDA tensor")
        if step_counts.dtype != torch.int32 or step_counts.shape != histories.shape:
            raise TypeError("step_counts must be int32 and match histories")
        if max_steps is None or int(max_steps) < 1:
            raise ValueError("max_steps must be positive when step_counts is supplied")
    elif max_steps is not None:
        raise ValueError("max_steps requires step_counts")
    track_last_hit = last_instances is not None or last_triangles is not None
    if (last_instances is None) != (last_triangles is None):
        raise ValueError(
            "last_instances and last_triangles must be supplied together"
        )
    if track_last_hit:
        for value, name in (
            (last_instances, "last_instances"),
            (last_triangles, "last_triangles"),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_cuda
                or not value.is_contiguous()
            ):
                raise ValueError(f"{name} must be a contiguous CUDA tensor")
            if value.dtype != torch.int32 or value.shape != histories.shape:
                raise TypeError(f"{name} must be int32 and match histories")
    track_global_ids = global_photon_ids is not None
    if track_global_ids:
        if (
            not isinstance(global_photon_ids, torch.Tensor)
            or not global_photon_ids.is_cuda
            or not global_photon_ids.is_contiguous()
            or global_photon_ids.device != input_buffer.device
        ):
            raise ValueError(
                "global_photon_ids must be a contiguous CUDA tensor on the queue device"
            )
        if (
            global_photon_ids.dtype != torch.int64
            or global_photon_ids.shape != histories.shape
        ):
            raise TypeError("global_photon_ids must be int64 and match histories")
    use_random_tape = random_tape is not None or tape_audit is not None
    if (random_tape is None) != (tape_audit is None):
        raise ValueError("random_tape and tape_audit must be supplied together")
    certify_random_tape = tape_certificate is not None
    if certify_random_tape and not use_random_tape:
        raise ValueError("tape_certificate requires random_tape and tape_audit")
    certify_state = state_certificate is not None
    if certify_state and not use_random_tape:
        raise ValueError("state_certificate requires random_tape and tape_audit")
    if certify_state and not certify_random_tape:
        raise ValueError("state_certificate requires tape_certificate")
    if certify_state and (
        state_wavelength is None
        or state_weight is None
        or state_evidx is None
    ):
        raise ValueError(
            "state_certificate requires explicit state_wavelength, state_weight, "
            "and state_evidx"
        )
    use_tape_rows = tape_row_indices is not None
    if use_tape_rows and not use_random_tape:
        raise ValueError("tape_row_indices requires random_tape and tape_audit")
    if use_random_tape:
        from .rng_alignment import (
            CERTIFICATE_DRAW_MASK,
            STATE_CERTIFICATE_FIELD_COUNT,
            TorchInteractionCertificate,
            TorchRandomTape,
            TorchTapeAudit,
        )

        if not isinstance(random_tape, TorchRandomTape):
            raise TypeError("random_tape must be a TorchRandomTape")
        if not isinstance(tape_audit, TorchTapeAudit):
            raise TypeError("tape_audit must be a TorchTapeAudit")
        tape_values = random_tape.values
        tape_global_id_values = random_tape.global_photon_ids
        if (
            not isinstance(tape_values, torch.Tensor)
            or not tape_values.is_cuda
            or tape_values.device != input_buffer.device
            or tape_values.dtype != torch.float32
            or tape_values.ndim != 3
            or not tape_values.is_contiguous()
        ):
            raise ValueError(
                "random tape values must be contiguous CUDA float32 [row, interaction, draw]"
            )
        if (
            not isinstance(tape_global_id_values, torch.Tensor)
            or not tape_global_id_values.is_cuda
            or tape_global_id_values.device != input_buffer.device
            or tape_global_id_values.dtype != torch.int64
            or tape_global_id_values.shape != (tape_values.shape[0],)
            or not tape_global_id_values.is_contiguous()
        ):
            raise ValueError(
                "random tape global IDs must be contiguous CUDA int64 [row]"
            )
        if (
            tape_values.shape[1] != int(random_tape.spec.max_interactions)
            or tape_values.shape[2]
            != int(random_tape.spec.draws_per_interaction)
        ):
            raise ValueError("random tape tensor shape does not match its spec")
        tape_photon_count = int(tape_values.shape[0])
        for value, name in (
            (tape_audit.interaction_cursor, "interaction_cursor"),
            (tape_audit.draw_cursor, "draw_cursor"),
            (tape_audit.overflow, "overflow"),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or not value.is_cuda
                or value.device != input_buffer.device
                or value.dtype != torch.int32
                or value.shape != (tape_photon_count,)
                or not value.is_contiguous()
            ):
                raise ValueError(
                    "tape audit %s must be contiguous CUDA int32 [row]" % name
                )
        if use_tape_rows:
            if (
                not isinstance(tape_row_indices, torch.Tensor)
                or not tape_row_indices.is_cuda
                or tape_row_indices.device != input_buffer.device
                or tape_row_indices.dtype != torch.int32
                or tape_row_indices.shape != histories.shape
                or not tape_row_indices.is_contiguous()
            ):
                raise ValueError(
                    "tape_row_indices must be contiguous CUDA int32 and match histories"
                )
        elif tape_photon_count != histories.numel():
            raise ValueError(
                "implicit tape rows require one tape row per local state slot"
            )
        max_tape_interactions = int(random_tape.spec.max_interactions)
        tape_draws_per_interaction = int(
            random_tape.spec.draws_per_interaction
        )
        if certify_random_tape:
            if not isinstance(tape_certificate, TorchInteractionCertificate):
                raise TypeError(
                    "tape_certificate must be a TorchInteractionCertificate"
                )
            certificate_words = tape_certificate.words
            if (
                not isinstance(certificate_words, torch.Tensor)
                or not certificate_words.is_cuda
                or certificate_words.device != input_buffer.device
                or certificate_words.dtype != torch.int32
                or certificate_words.shape != (
                    tape_photon_count, max_tape_interactions
                )
                or not certificate_words.is_contiguous()
            ):
                raise ValueError(
                    "tape certificate must be contiguous CUDA int32 "
                    "[row, interaction]"
                )
            if tape_draws_per_interaction > int(CERTIFICATE_DRAW_MASK):
                raise ValueError("certificate draw counts must fit 28 bits")
        else:
            certificate_words = tape_audit.overflow
        if certify_state:
            state_certificate_words = getattr(
                state_certificate, "words", state_certificate
            )
            expected_state_shape = (
                tape_photon_count,
                max_tape_interactions,
                int(STATE_CERTIFICATE_FIELD_COUNT),
            )
            expected_state_words = int(np.prod(expected_state_shape))
            if (
                not isinstance(state_certificate_words, torch.Tensor)
                or not state_certificate_words.is_cuda
                or state_certificate_words.device != input_buffer.device
                or state_certificate_words.dtype
                not in (torch.int32, torch.uint32)
                or not state_certificate_words.is_contiguous()
                or (
                    tuple(state_certificate_words.shape) != expected_state_shape
                    and not (
                        state_certificate_words.ndim == 1
                        and state_certificate_words.numel()
                        == expected_state_words
                    )
                )
            ):
                raise ValueError(
                    "state certificate must be contiguous same-device int32/uint32 "
                    "[row, interaction, 15] or its flat storage"
                )
            try:
                state_wavelength_word = int(
                    np.asarray([state_wavelength], dtype=np.float32)
                    .view(np.uint32)[0]
                )
                state_weight_word = int(
                    np.asarray([state_weight], dtype=np.float32)
                    .view(np.uint32)[0]
                )
                state_evidx_value = int(
                    np.asarray(state_evidx).reshape(()).item()
                )
                if not 0 <= state_evidx_value <= 0xFFFFFFFF:
                    raise ValueError("state_evidx must fit uint32")
                state_evidx_word = int(
                    np.asarray([state_evidx_value], dtype=np.uint32)[0]
                )
            except (OverflowError, TypeError, ValueError) as exc:
                raise TypeError(
                    "state_wavelength/state_weight must be scalar float32 values "
                    "and state_evidx must be a scalar uint32 value"
                ) from exc
        else:
            state_certificate_words = histories
            state_wavelength_word = 0
            state_weight_word = 0
            state_evidx_word = 0
    else:
        tape_values = positions
        tape_global_id_values = rng_counters
        tape_row_indices = input_buffer
        tape_interaction_pointer = histories
        tape_draw_pointer = histories
        tape_overflow_pointer = histories
        certificate_words = histories
        tape_photon_count = 0
        max_tape_interactions = 1
        tape_draws_per_interaction = 1
        state_field_count = 15
        state_certificate_words = histories
        state_wavelength_word = 0
        state_weight_word = 0
        state_evidx_word = 0
    if use_random_tape:
        state_field_count = int(STATE_CERTIFICATE_FIELD_COUNT)
        tape_interaction_pointer = tape_audit.interaction_cursor
        tape_draw_pointer = tape_audit.draw_cursor
        tape_overflow_pointer = tape_audit.overflow
    if absorption_length <= 0.0 or scattering_length <= 0.0:
        raise ValueError("optical lengths must be positive")
    if refractive_index <= 0.0 or max_scatter < 1:
        raise ValueError("refractive_index and max_scatter must be positive")

    store_status = partition_mode != "active"
    emit_active = partition_mode in ("active", "all")
    emit_terminal = partition_mode == "all"
    if emit_active:
        if queue_workspace is None:
            queue_workspace = CollisionQueueWorkspace.allocate(
                launch_capacity,
                device=input_buffer.device,
                dtype=input_buffer.dtype,
                include_terminal=emit_terminal,
            )
        elif not isinstance(queue_workspace, CollisionQueueWorkspace):
            raise TypeError("queue_workspace must be a CollisionQueueWorkspace")
        queue_workspace.validate_for(
            input_buffer,
            capacity=launch_capacity,
            include_terminal=emit_terminal,
        )
        continuing = queue_workspace.continuing
        absorbed = queue_workspace.absorbed if emit_terminal else None
        invalid = queue_workspace.invalid if emit_terminal else None
        if boundary_accumulator is None:
            boundary = queue_workspace.boundary
        else:
            if not isinstance(boundary_accumulator, DeviceQueue):
                raise TypeError("boundary_accumulator must be a DeviceQueue")
            boundary = boundary_accumulator
            if (
                not isinstance(boundary.buffer, torch.Tensor)
                or not boundary.buffer.is_cuda
                or boundary.buffer.device != input_buffer.device
                or boundary.buffer.dtype != input_buffer.dtype
                or boundary.buffer.ndim != 1
                or not boundary.buffer.is_contiguous()
                or boundary.capacity < launch_capacity
            ):
                raise ValueError(
                    "boundary accumulator must be a same-device queue with "
                    "matching index dtype and sufficient capacity"
                )
            if (
                not isinstance(boundary.count, torch.Tensor)
                or not boundary.count.is_cuda
                or boundary.count.device != input_buffer.device
                or boundary.count.dtype != torch.int32
                or boundary.count.shape != (1,)
                or not boundary.count.is_contiguous()
            ):
                raise ValueError(
                    "boundary accumulator count must be contiguous CUDA int32 "
                    "with shape (1,)"
                )
        output_queues = [continuing, boundary]
        if emit_terminal:
            output_queues.extend((absorbed, invalid))
        if launch_capacity:
            for output in output_queues:
                if _tensor_storage_overlaps(output.buffer, input_buffer):
                    raise ValueError("input and output queue buffers must not alias")
            for left_index, left in enumerate(output_queues):
                for right in output_queues[left_index + 1:]:
                    if _tensor_storage_overlaps(left.buffer, right.buffer):
                        raise ValueError("output queue buffers must not alias")
        for left_index, left in enumerate(output_queues):
            if (
                input_count_is_pointer
                and _tensor_storage_overlaps(left.count, input_count)
            ):
                raise ValueError("input and output queue counters must not alias")
            for right in output_queues[left_index + 1:]:
                if _tensor_storage_overlaps(left.count, right.count):
                    raise ValueError("output queue counters must not alias")

        # A shared workspace can clear all counters in one device operation.
        # Persistent boundary accumulation instead resets only the queues whose
        # contents are replaced by this epoch.
        if boundary_accumulator is None and not append_boundary:
            queue_workspace.reset()
        else:
            continuing.reset()
            if not append_boundary:
                boundary.reset()
            if emit_terminal:
                absorbed.reset()
                invalid.reset()
    else:
        if queue_workspace is not None or boundary_accumulator is not None:
            raise ValueError(
                "queue_workspace and boundary_accumulator require partitioned output"
            )
        if append_boundary:
            raise ValueError("append_boundary requires partitioned output")
        continuing = boundary = absorbed = invalid = None

    if emit_active:
        scatter_counts = queue_workspace.scatter_count_scratch[:launch_capacity]
        statuses = (
            queue_workspace.status_scratch[:launch_capacity]
            if store_status else None
        )
    else:
        statuses = (
            torch.empty(
                launch_capacity, dtype=torch.int32, device=input_buffer.device
            )
            if store_status else None
        )
        scatter_counts = torch.empty(
            launch_capacity, dtype=torch.int32, device=input_buffer.device
        )

    # Dead pointers are never dereferenced in the corresponding constexpr
    # specialization.  Passing existing allocations keeps the launch signature
    # uniform without allocating dummy tensors.
    status_pointer = statuses if store_status else scatter_counts
    if emit_active:
        continuing_pointer = continuing.buffer
        continuing_counter = continuing.count
        boundary_pointer = boundary.buffer
        boundary_counter = boundary.count
    else:
        continuing_pointer = boundary_pointer = input_buffer
        continuing_counter = boundary_counter = scatter_counts
    if emit_terminal:
        absorbed_pointer = absorbed.buffer
        absorbed_counter = absorbed.count
        invalid_pointer = invalid.buffer
        invalid_counter = invalid.count
    else:
        absorbed_pointer = invalid_pointer = input_buffer
        absorbed_counter = invalid_counter = scatter_counts

    grid = (triton.cdiv(launch_capacity, block_size),)
    if launch_capacity:
        kernel[grid](
            positions, directions, polarizations, times, histories, rng_counters,
            step_counts if track_steps else histories,
            last_instances if track_last_hit else histories,
            last_triangles if track_last_hit else histories,
            input_buffer,
            global_photon_ids if track_global_ids else rng_counters,
            tape_values,
            tape_global_id_values,
            tape_row_indices,
            tape_interaction_pointer,
            tape_draw_pointer,
            tape_overflow_pointer,
            certificate_words,
            tape_photon_count,
            status_pointer,
            scatter_counts,
            continuing_pointer,
            continuing_counter,
            boundary_pointer,
            boundary_counter,
            absorbed_pointer,
            absorbed_counter,
            invalid_pointer,
            invalid_counter,
            nitems_argument,
            state_certificate_words,
            lower_x=float(lower[0]), lower_y=float(lower[1]), lower_z=float(lower[2]),
            upper_x=float(upper[0]), upper_y=float(upper[1]), upper_z=float(upper[2]),
            inv_absorption=(
                0.0 if np.isinf(absorption_length)
                else 1.0 / float(absorption_length)
            ),
            inv_scattering=(
                0.0 if np.isinf(scattering_length)
                else 1.0 / float(scattering_length)
            ),
            absorption_length=float(absorption_length),
            scattering_length=float(scattering_length),
            refractive_index=float(refractive_index), seed=int(seed),
            photon_id_base=int(photon_id_base),
            MAX_SCATTER=int(max_scatter),
            MAX_STEPS=int(max_steps) if track_steps else 2_147_483_647,
            TRACK_STEPS=track_steps,
            TRACK_LAST_HIT=track_last_hit,
            TRACK_GLOBAL_IDS=track_global_ids,
            USE_RANDOM_TAPE=use_random_tape,
            USE_TAPE_ROWS=use_tape_rows,
            CERTIFY_RANDOM_TAPE=certify_random_tape,
            MAX_TAPE_INTERACTIONS=max_tape_interactions,
            TAPE_DRAWS_PER_INTERACTION=tape_draws_per_interaction,
            CERTIFY_STATE=certify_state,
            STATE_WORDS_PER_INTERACTION=state_field_count,
            STATE_WAVELENGTH_WORD=state_wavelength_word,
            STATE_WEIGHT_WORD=state_weight_word,
            STATE_EVIDX_WORD=state_evidx_word,
            STORE_STATUS=store_status,
            EMIT_ACTIVE=emit_active,
            EMIT_TERMINAL=emit_terminal,
            NITEMS_IS_POINTER=input_count_is_pointer,
            BLOCK=int(block_size),
            # Do not ask Triton to distribute a 64-lane reduction across four
            # warps.  Besides wasting lanes, Triton 3.1 can double-reserve an
            # atomic queue slot at multi-million-element scale for that shape.
            # Two warps exactly cover BLOCK=64; larger production blocks keep
            # the previously validated four-warp lowering.
            num_warps=min(4, max(1, int(block_size) // 32)),
        )

    return DeviceEpochResult(
        statuses, scatter_counts, continuing, boundary, absorbed, invalid
    )


def benchmark_collision_first(nphotons=1_048_576, max_scatter=4,
                              warmup=20, rep=100):
    """Benchmark the register-resident bulk loop on the active CUDA device."""

    import torch

    triton, _, _ = _load_gpu_kernels()
    device = torch.device("cuda")
    positions = torch.zeros((nphotons, 3), dtype=torch.float32, device=device)
    directions = torch.zeros_like(positions)
    directions[:, 0] = 1.0
    polarizations = torch.zeros_like(positions)
    polarizations[:, 1] = 1.0
    times = torch.zeros(nphotons, dtype=torch.float32, device=device)
    histories = torch.zeros(nphotons, dtype=torch.int32, device=device)
    counters = torch.zeros(nphotons, dtype=torch.int64, device=device)
    queue = torch.arange(nphotons, dtype=torch.int32, device=device)

    def launch():
        collision_first_epoch(
            positions, directions, polarizations, times, histories, counters,
            queue, [-1.0e9]*3, [1.0e9]*3, np.inf, 950.0, 1.3784,
            seed=9127, max_scatter=max_scatter, partition=False,
        )

    milliseconds = triton.testing.do_bench(launch, warmup=warmup, rep=rep)
    return {
        "photons": int(nphotons),
        "collisions_per_photon": int(max_scatter),
        "milliseconds": float(milliseconds),
        "million_photons_per_second": nphotons/(milliseconds*1.0e3),
        "million_collisions_per_second": (
            nphotons*max_scatter/(milliseconds*1.0e3)
        ),
    }


__all__ = [
    "ABSORBED", "BOUNDARY", "BULK_ABSORB", "CONTINUE",
    "CollisionQueueWorkspace", "DeviceEpochResult", "DeviceQueue", "EpochResult",
    "INVALID", "NAN_ABORT", "NO_HIT",
    "RAYLEIGH_SCATTER", "SPEED_OF_LIGHT_MM_PER_NS",
    "certified_aabb_exit_distance", "collision_first_epoch",
    "collision_first_epoch_reference", "compact_status",
    "benchmark_collision_first",
]
