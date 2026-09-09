"""Conservative portal classification for the certified empty LAr box.

The collision-first transport kernel stops a short, representable distance
inside an obstacle-free orthotope.  Five of that orthotope's six exit faces
coincide exactly with known detector surfaces: the active-volume Y/Z walls
and the source-facing cathode face.  A ray reaching one of those faces cannot
encounter a wire or PMT first, so resolving it through the general scene is
pure overhead.  The remaining X face is an internal portal into the wire/PMT
region and always falls back to the ordinary exact query.

This module only classifies and materializes already-known hit records.  It
does not perform boundary physics, which keeps RNG ownership in the existing
production boundary kernels and makes the optimization easy to disable for
strict Chroma replay.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple, Optional, Sequence

import numpy as np


# Zero retains the uncapped capacity grid selected by isolated A100 timing.
# The classifier uses four warps at the production BLOCK=128 setting, but an
# eight-program/SM persistent cohort was 1.8% slower for a dense 1M queue and
# 5.6% slower for an 8K-live/1M-capacity tail.  Positive values remain exposed
# for future-device sweeps without a source edit; the kernel then grid-strides
# while preserving one atomic reservation per logical block.
DEVICE_PORTAL_PROGRAMS_PER_SM = 0


def _persistent_portal_program_count(
    input_capacity: int,
    block_size: int,
    multiprocessor_count: int,
    programs_per_sm: Optional[int] = None,
) -> int:
    """Return a tunable persistent grid; zero programs/SM means uncapped."""

    input_capacity = int(input_capacity)
    block_size = int(block_size)
    multiprocessor_count = int(multiprocessor_count)
    if programs_per_sm is None:
        programs_per_sm = DEVICE_PORTAL_PROGRAMS_PER_SM
    programs_per_sm = int(programs_per_sm)
    if input_capacity < 0:
        raise ValueError("portal input capacity cannot be negative")
    if block_size <= 0:
        raise ValueError("portal block size must be positive")
    if multiprocessor_count <= 0:
        raise ValueError("portal multiprocessor count must be positive")
    if programs_per_sm < 0:
        raise ValueError("portal programs per SM cannot be negative")
    capacity_programs = (
        input_capacity + block_size - 1
    ) // block_size
    if programs_per_sm == 0:
        return capacity_programs
    return min(
        capacity_programs,
        multiprocessor_count * programs_per_sm,
    )


class PortalPartition(NamedTuple):
    """Device queues and hit arrays emitted by one portal partition."""

    direct: Any
    fallback: Any
    hit: tuple[Any, ...]


@dataclass(frozen=True)
class PortalDescriptor:
    """Host-certified metadata for the five direct detector faces."""

    lower: np.ndarray
    upper: np.ndarray
    lar_material: int
    active_outside_material: int
    cathode_inside_material: int
    active_surface: int
    cathode_surface: int
    active_box_index: int
    cathode_box_index: int


@dataclass
class PortalWorkspace:
    """Reusable queue and SoA hit storage for portal classification."""

    queues: Any
    counts: Any
    distance: Any
    normal: Any
    material_from: Any
    material_to: Any
    surface: Any
    instance: Any
    triangle: Any
    channel: Any

    @classmethod
    def allocate(cls, capacity: int, device: Any = "cuda") -> "PortalWorkspace":
        import torch

        capacity = int(capacity)
        if capacity < 0:
            raise ValueError("portal workspace capacity must be non-negative")
        return cls(
            queues=torch.empty((2, capacity), dtype=torch.int32, device=device),
            counts=torch.zeros(2, dtype=torch.int32, device=device),
            distance=torch.empty(capacity, dtype=torch.float32, device=device),
            normal=torch.empty((capacity, 3), dtype=torch.float32, device=device),
            material_from=torch.empty(capacity, dtype=torch.int32, device=device),
            material_to=torch.empty(capacity, dtype=torch.int32, device=device),
            surface=torch.empty(capacity, dtype=torch.int32, device=device),
            instance=torch.empty(capacity, dtype=torch.int32, device=device),
            triangle=torch.empty(capacity, dtype=torch.int32, device=device),
            channel=torch.empty(capacity, dtype=torch.int32, device=device),
        )

    @property
    def capacity(self) -> int:
        return int(self.queues.shape[1])

    def hit_outputs(self) -> tuple[Any, ...]:
        """Return full-capacity views matching ``_BoundaryMergeWorkspace``."""

        return (
            self.distance,
            self.normal,
            self.material_from,
            self.material_to,
            self.surface,
            self.instance,
            self.triangle,
            self.channel,
        )


def _tensor_storage_overlaps(left: Any, right: Any) -> bool:
    """Return whether two contiguous tensor views share storage bytes."""

    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False
    left_begin = int(left.data_ptr())
    right_begin = int(right.data_ptr())
    left_end = left_begin + left.numel() * left.element_size()
    right_end = right_begin + right.numel() * right.element_size()
    return max(left_begin, right_begin) < min(left_end, right_end)


def certify_reflect3wires_portals(
    scene: Any,
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    padded_pmt_bounds_max: Any,
) -> PortalDescriptor:
    """Prove the target's certified box terminates on known opaque faces.

    The proof is repeated when a simulator is constructed.  It deliberately
    consumes the PMT accelerator's outward-padded bounds, not merely nominal
    instance bounds, so a later broadphase-policy change cannot silently make
    a direct segment overlap a PMT candidate.
    """

    scene.validate()
    lo = np.asarray(lower, dtype=np.float32)
    hi = np.asarray(upper, dtype=np.float32)
    pmt_hi = np.asarray(padded_pmt_bounds_max, dtype=np.float32)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(lo >= hi):
        raise ValueError("portal lower/upper must be ordered three-vectors")
    if pmt_hi.ndim != 2 or pmt_hi.shape[1] != 3 or pmt_hi.shape[0] == 0:
        raise ValueError("padded PMT maxima must have shape [N,3]")
    if scene.reachability.source_x_sign != -1:
        raise ValueError("direct portal certificate currently requires source_x_sign=-1")
    try:
        active = scene.boxes.kinds.index("active")
        cathode = scene.boxes.kinds.index("cathode")
        lar = scene.tables.material_names.index("liquid_argon")
    except ValueError as exc:
        raise ValueError("scene is missing required active/cathode/LAr metadata") from exc

    # These equalities are bitwise: the portal kernel receives exactly the
    # float32 words used by the analytic box kernel.
    if not np.array_equal(lo[1:], scene.boxes.bounds_min[active, 1:]):
        raise ValueError("certified lower Y/Z do not equal the active box")
    if not np.array_equal(hi[1:], scene.boxes.bounds_max[active, 1:]):
        raise ValueError("certified upper Y/Z do not equal the active box")
    if hi[0] != scene.boxes.bounds_min[cathode, 0]:
        raise ValueError("certified upper X does not equal the cathode face")
    if not np.all(scene.boxes.reachable_face_mask[active, [2, 3, 4, 5]]):
        raise ValueError("active Y/Z portal faces are not all reachable")
    if not bool(scene.boxes.reachable_face_mask[cathode, 0]):
        raise ValueError("cathode source face is not reachable")

    active_from = int(scene.boxes.material_inside_index[active])
    cathode_from = int(scene.boxes.material_outside_index[cathode])
    if active_from != lar or cathode_from != lar:
        raise ValueError("portal incident material is not liquid argon")
    active_surface = int(scene.boxes.surface_index[active])
    cathode_surface = int(scene.boxes.surface_index[cathode])
    if active_surface < 0 or cathode_surface < 0:
        raise ValueError("direct portal faces require explicit opaque surfaces")
    for surface in (active_surface, cathode_surface):
        probability_sum = float(
            scene.tables.surface_absorb[surface]
            + scene.tables.surface_detect[surface]
            + scene.tables.surface_reflect_diffuse[surface]
            + scene.tables.surface_reflect_specular[surface]
        )
        if probability_sum < 1.0:
            raise ValueError("direct portal surface is transmissive")

    if not bool(np.max(pmt_hi[:, 0]) < lo[0]):
        raise ValueError("padded PMT bounds overlap the certified portal box")
    if not (
        np.all(np.abs(scene.wires.n[:, 0]) == 1.0)
        and np.all(scene.wires.n[:, 1:] == 0.0)
    ):
        raise ValueError("wire clearance proof requires X-normal wire planes")
    wire_x_max = np.max(scene.wires.origin[:, 0] + scene.wires.radius)
    if not bool(wire_x_max < lo[0]):
        raise ValueError("wire cylinders overlap the certified portal box")

    return PortalDescriptor(
        lower=lo.copy(),
        upper=hi.copy(),
        lar_material=lar,
        active_outside_material=int(scene.boxes.material_outside_index[active]),
        cathode_inside_material=int(scene.boxes.material_inside_index[cathode]),
        active_surface=active_surface,
        cathode_surface=cathode_surface,
        active_box_index=active,
        cathode_box_index=cathode,
    )

def classify_certified_box_portals_numpy(
    positions: Any,
    directions: Any,
    lower: Sequence[float],
    upper: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reference classifier returning ``(direct, distance, face)``.

    Faces use the analytic-box convention ``x-/x+/y-/y+/z-/z+ == 0..5``.
    Face 0 is never direct: the certified box's lower-X plane is an internal
    portal rather than detector geometry.  Invalid, outside, non-forward, and
    lower-X-first rays conservatively fall back.
    """

    position = np.ascontiguousarray(positions, dtype=np.float32)
    direction = np.ascontiguousarray(directions, dtype=np.float32)
    lo = np.asarray(lower, dtype=np.float32)
    hi = np.asarray(upper, dtype=np.float32)
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError("positions must have shape [N,3]")
    if direction.shape != position.shape:
        raise ValueError("directions must match positions")
    if lo.shape != (3,) or hi.shape != (3,) or np.any(lo >= hi):
        raise ValueError("lower/upper must be ordered three-vectors")

    n = position.shape[0]
    direct = np.zeros(n, dtype=np.bool_)
    distance = np.full(n, np.float32(np.inf), dtype=np.float32)
    face = np.full(n, -1, dtype=np.int8)
    finite = np.isfinite(position).all(axis=1) & np.isfinite(direction).all(axis=1)
    inside = finite & np.all(position >= lo, axis=1) & np.all(position <= hi, axis=1)
    huge = np.float32(1.0e30)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        tx = np.where(
            direction[:, 0] > 0.0,
            (hi[0] - position[:, 0]) / direction[:, 0],
            np.where(
                direction[:, 0] < 0.0,
                (lo[0] - position[:, 0]) / direction[:, 0],
                huge,
            ),
        ).astype(np.float32)
        ty = np.where(
            direction[:, 1] > 0.0,
            (hi[1] - position[:, 1]) / direction[:, 1],
            np.where(
                direction[:, 1] < 0.0,
                (lo[1] - position[:, 1]) / direction[:, 1],
                huge,
            ),
        ).astype(np.float32)
        tz = np.where(
            direction[:, 2] > 0.0,
            (hi[2] - position[:, 2]) / direction[:, 2],
            np.where(
                direction[:, 2] < 0.0,
                (lo[2] - position[:, 2]) / direction[:, 2],
                huge,
            ),
        ).astype(np.float32)

    # Analytic boxes visit Y faces before Z, and the cathode is visited after
    # the active box.  Non-strict comparisons reproduce that tie priority.
    choose_y = (ty <= tz) & (ty <= tx)
    choose_z = (~choose_y) & (tz <= tx)
    choose_x_high = (~choose_y) & (~choose_z) & (direction[:, 0] > 0.0)
    candidate = np.where(choose_y, ty, np.where(choose_z, tz, tx)).astype(
        np.float32
    )
    # If the artificial lower-X portal is tied with a real wall after
    # float32 rounding, fall back.  This deliberately sacrifices a vanishing
    # amount of work rather than extrapolating across the certified region.
    lower_x_tie = (direction[:, 0] < 0.0) & (tx <= candidate)
    admissible = (
        inside
        & (candidate > np.float32(1.0e-6))
        & np.isfinite(candidate)
        & ~lower_x_tie
    )
    direct = admissible & (choose_y | choose_z | choose_x_high)
    distance[direct] = candidate[direct]
    face[direct & choose_y & (direction[:, 1] < 0.0)] = 2
    face[direct & choose_y & (direction[:, 1] > 0.0)] = 3
    face[direct & choose_z & (direction[:, 2] < 0.0)] = 4
    face[direct & choose_z & (direction[:, 2] > 0.0)] = 5
    face[direct & choose_x_high] = 0  # Cathode's source-facing local face.
    direct &= face >= 0
    distance[~direct] = np.float32(np.inf)
    return direct, distance, face


def _load_portal_kernel():
    cached = getattr(_load_portal_kernel, "_cached", None)
    if cached is not None:
        return cached
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:  # pragma: no cover - optional installation
        raise RuntimeError("portal classification requires Triton") from exc

    # Triton 3.1 resolves JIT globals from the defining module.
    globals().update(triton=triton, tl=tl)

    @triton.jit(do_not_specialize=[16, 17, 18, 19, 20, 21, 22, 23, 30])
    def portal_partition_kernel(
        positions,
        directions,
        input_queue,
        last_instances,
        last_triangles,
        direct_queue,
        fallback_queue,
        counts,
        out_distance,
        out_normal,
        out_material_from,
        out_material_to,
        out_surface,
        out_instance,
        out_triangle,
        out_channel,
        nitems,
        lower_x,
        lower_y,
        lower_z,
        upper_x,
        upper_y,
        upper_z,
        lar_material,
        active_outside_material,
        cathode_inside_material,
        active_surface,
        cathode_surface,
        active_instance,
        cathode_instance,
        input_capacity,
        NITEMS_IS_POINTER: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        program_start = tl.program_id(0) * BLOCK
        if NITEMS_IS_POINTER:
            live_items = tl.load(nitems).to(tl.int32)
            # Clamp only for memory safety.  The asynchronous scheduler owns
            # the stronger 0 <= count <= input_capacity invariant.  Retire a
            # wholly inactive CTA before it loads a stale queue suffix or
            # executes the portal arithmetic.
            live_items = tl.maximum(0, tl.minimum(live_items, input_capacity))
            if program_start >= live_items:
                return
        else:
            live_items = nitems
        lane = program_start + tl.arange(0, BLOCK)
        valid = lane < live_items
        photon_id = tl.load(input_queue + lane, mask=valid, other=0).to(tl.int64)
        base = photon_id * 3
        px = tl.load(positions + base, mask=valid, other=0.0).to(tl.float32)
        py = tl.load(positions + base + 1, mask=valid, other=0.0).to(tl.float32)
        pz = tl.load(positions + base + 2, mask=valid, other=0.0).to(tl.float32)
        dx = tl.load(directions + base, mask=valid, other=0.0).to(tl.float32)
        dy = tl.load(directions + base + 1, mask=valid, other=0.0).to(tl.float32)
        dz = tl.load(directions + base + 2, mask=valid, other=0.0).to(tl.float32)

        finite = (
            (px == px) & (py == py) & (pz == pz)
            & (dx == dx) & (dy == dy) & (dz == dz)
        )
        inside = (
            valid & finite
            & (px >= lower_x) & (px <= upper_x)
            & (py >= lower_y) & (py <= upper_y)
            & (pz >= lower_z) & (pz <= upper_z)
        )
        huge: tl.constexpr = 1.0e30
        tx = tl.where(
            dx > 0.0,
            (upper_x - px) / dx,
            tl.where(dx < 0.0, (lower_x - px) / dx, huge),
        )
        ty = tl.where(
            dy > 0.0,
            (upper_y - py) / dy,
            tl.where(dy < 0.0, (lower_y - py) / dy, huge),
        )
        tz = tl.where(
            dz > 0.0,
            (upper_z - pz) / dz,
            tl.where(dz < 0.0, (lower_z - pz) / dz, huge),
        )
        choose_y = (ty <= tz) & (ty <= tx)
        choose_z = (~choose_y) & (tz <= tx)
        choose_x_high = (~choose_y) & (~choose_z) & (dx > 0.0)
        distance = tl.where(choose_y, ty, tl.where(choose_z, tz, tx))
        lower_x_tie = (dx < 0.0) & (tx <= distance)
        face = tl.where(
            choose_y,
            tl.where(dy < 0.0, 2, 3),
            tl.where(choose_z, tl.where(dz < 0.0, 4, 5), 0),
        ).to(tl.int32)
        instance = tl.where(
            choose_x_high, cathode_instance, active_instance
        ).to(tl.int32)
        previous_instance = tl.load(
            last_instances + photon_id, mask=valid, other=-1
        ).to(tl.int32)
        previous_triangle = tl.load(
            last_triangles + photon_id, mask=valid, other=-1
        ).to(tl.int32)
        suppress_previous = (
            (previous_instance == instance) & (previous_triangle == face)
        )
        direct = (
            inside & (distance > 1.0e-6) & (distance < huge)
            & (choose_y | choose_z | choose_x_high) & ~lower_x_tie
            & ~suppress_previous
        )

        direct_flag = direct.to(tl.int32)
        direct_local = tl.cumsum(direct_flag, axis=0) - direct_flag
        direct_n = tl.sum(direct_flag, axis=0)
        direct_base = tl.atomic_add(counts, direct_n)
        direct_slot = direct_base + direct_local
        tl.store(direct_queue + direct_slot, photon_id.to(tl.int32), mask=direct)

        fallback = valid & ~direct
        fallback_flag = fallback.to(tl.int32)
        fallback_local = tl.cumsum(fallback_flag, axis=0) - fallback_flag
        fallback_n = tl.sum(fallback_flag, axis=0)
        fallback_base = tl.atomic_add(counts + 1, fallback_n)
        tl.store(
            fallback_queue + fallback_base + fallback_local,
            photon_id.to(tl.int32),
            mask=fallback,
        )

        y_low = choose_y & (dy < 0.0)
        z_low = choose_z & (dz < 0.0)
        nx = tl.where(choose_x_high, -1.0, 0.0)
        ny = tl.where(choose_y, tl.where(y_low, 1.0, -1.0), 0.0)
        nz = tl.where(choose_z, tl.where(z_low, 1.0, -1.0), 0.0)
        material_to = tl.where(
            choose_x_high, cathode_inside_material, active_outside_material
        ).to(tl.int32)
        surface = tl.where(
            choose_x_high, cathode_surface, active_surface
        ).to(tl.int32)
        output_base = direct_slot * 3
        tl.store(out_distance + direct_slot, distance, mask=direct)
        tl.store(out_normal + output_base, nx, mask=direct)
        tl.store(out_normal + output_base + 1, ny, mask=direct)
        tl.store(out_normal + output_base + 2, nz, mask=direct)
        tl.store(out_material_from + direct_slot, lar_material, mask=direct)
        tl.store(out_material_to + direct_slot, material_to, mask=direct)
        tl.store(out_surface + direct_slot, surface, mask=direct)
        tl.store(out_instance + direct_slot, instance, mask=direct)
        tl.store(out_triangle + direct_slot, face, mask=direct)
        tl.store(out_channel + direct_slot, -1, mask=direct)

    @triton.jit(do_not_specialize=[16, 17, 18, 19, 20, 21, 22, 23, 30])
    def portal_partition_persistent_kernel(
        positions,
        directions,
        input_queue,
        last_instances,
        last_triangles,
        direct_queue,
        fallback_queue,
        counts,
        out_distance,
        out_normal,
        out_material_from,
        out_material_to,
        out_surface,
        out_instance,
        out_triangle,
        out_channel,
        nitems,
        lower_x,
        lower_y,
        lower_z,
        upper_x,
        upper_y,
        upper_z,
        lar_material,
        active_outside_material,
        cathode_inside_material,
        active_surface,
        cathode_surface,
        active_instance,
        cathode_instance,
        input_capacity,
        NITEMS_IS_POINTER: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        live_items = tl.load(nitems).to(tl.int32)
        live_items = tl.maximum(0, tl.minimum(live_items, input_capacity))
        program_start = tl.program_id(0) * BLOCK
        if program_start >= live_items:
            return
        # The host launches only an occupancy-sized cohort.  Each program
        # keeps its BLOCK lanes and processes later logical blocks in grid
        # strides.  Queue compaction remains one atomic reservation per
        # logical block, exactly as in the capacity-grid implementation.
        program_stride = tl.num_programs(0) * BLOCK
        huge: tl.constexpr = 1.0e30
        while program_start < live_items:
            lane = program_start + tl.arange(0, BLOCK)
            valid = lane < live_items
            photon_id = tl.load(
                input_queue + lane, mask=valid, other=0
            ).to(tl.int64)
            base = photon_id * 3
            px = tl.load(
                positions + base, mask=valid, other=0.0
            ).to(tl.float32)
            py = tl.load(
                positions + base + 1, mask=valid, other=0.0
            ).to(tl.float32)
            pz = tl.load(
                positions + base + 2, mask=valid, other=0.0
            ).to(tl.float32)
            dx = tl.load(
                directions + base, mask=valid, other=0.0
            ).to(tl.float32)
            dy = tl.load(
                directions + base + 1, mask=valid, other=0.0
            ).to(tl.float32)
            dz = tl.load(
                directions + base + 2, mask=valid, other=0.0
            ).to(tl.float32)

            finite = (
                (px == px) & (py == py) & (pz == pz)
                & (dx == dx) & (dy == dy) & (dz == dz)
            )
            inside = (
                valid & finite
                & (px >= lower_x) & (px <= upper_x)
                & (py >= lower_y) & (py <= upper_y)
                & (pz >= lower_z) & (pz <= upper_z)
            )
            tx = tl.where(
                dx > 0.0,
                (upper_x - px) / dx,
                tl.where(dx < 0.0, (lower_x - px) / dx, huge),
            )
            ty = tl.where(
                dy > 0.0,
                (upper_y - py) / dy,
                tl.where(dy < 0.0, (lower_y - py) / dy, huge),
            )
            tz = tl.where(
                dz > 0.0,
                (upper_z - pz) / dz,
                tl.where(dz < 0.0, (lower_z - pz) / dz, huge),
            )
            choose_y = (ty <= tz) & (ty <= tx)
            choose_z = (~choose_y) & (tz <= tx)
            choose_x_high = (~choose_y) & (~choose_z) & (dx > 0.0)
            distance = tl.where(choose_y, ty, tl.where(choose_z, tz, tx))
            lower_x_tie = (dx < 0.0) & (tx <= distance)
            face = tl.where(
                choose_y,
                tl.where(dy < 0.0, 2, 3),
                tl.where(choose_z, tl.where(dz < 0.0, 4, 5), 0),
            ).to(tl.int32)
            instance = tl.where(
                choose_x_high, cathode_instance, active_instance
            ).to(tl.int32)
            previous_instance = tl.load(
                last_instances + photon_id, mask=valid, other=-1
            ).to(tl.int32)
            previous_triangle = tl.load(
                last_triangles + photon_id, mask=valid, other=-1
            ).to(tl.int32)
            suppress_previous = (
                (previous_instance == instance) & (previous_triangle == face)
            )
            direct = (
                inside & (distance > 1.0e-6) & (distance < huge)
                & (choose_y | choose_z | choose_x_high) & ~lower_x_tie
                & ~suppress_previous
            )

            direct_flag = direct.to(tl.int32)
            direct_local = tl.cumsum(direct_flag, axis=0) - direct_flag
            direct_n = tl.sum(direct_flag, axis=0)
            direct_base = tl.atomic_add(counts, direct_n)
            direct_slot = direct_base + direct_local
            tl.store(
                direct_queue + direct_slot,
                photon_id.to(tl.int32),
                mask=direct,
            )

            fallback = valid & ~direct
            fallback_flag = fallback.to(tl.int32)
            fallback_local = tl.cumsum(fallback_flag, axis=0) - fallback_flag
            fallback_n = tl.sum(fallback_flag, axis=0)
            fallback_base = tl.atomic_add(counts + 1, fallback_n)
            tl.store(
                fallback_queue + fallback_base + fallback_local,
                photon_id.to(tl.int32),
                mask=fallback,
            )

            y_low = choose_y & (dy < 0.0)
            z_low = choose_z & (dz < 0.0)
            nx = tl.where(choose_x_high, -1.0, 0.0)
            ny = tl.where(
                choose_y, tl.where(y_low, 1.0, -1.0), 0.0
            )
            nz = tl.where(
                choose_z, tl.where(z_low, 1.0, -1.0), 0.0
            )
            material_to = tl.where(
                choose_x_high,
                cathode_inside_material,
                active_outside_material,
            ).to(tl.int32)
            surface = tl.where(
                choose_x_high, cathode_surface, active_surface
            ).to(tl.int32)
            output_base = direct_slot * 3
            tl.store(out_distance + direct_slot, distance, mask=direct)
            tl.store(out_normal + output_base, nx, mask=direct)
            tl.store(out_normal + output_base + 1, ny, mask=direct)
            tl.store(out_normal + output_base + 2, nz, mask=direct)
            tl.store(
                out_material_from + direct_slot,
                lar_material,
                mask=direct,
            )
            tl.store(
                out_material_to + direct_slot, material_to, mask=direct
            )
            tl.store(out_surface + direct_slot, surface, mask=direct)
            tl.store(out_instance + direct_slot, instance, mask=direct)
            tl.store(out_triangle + direct_slot, face, mask=direct)
            tl.store(out_channel + direct_slot, -1, mask=direct)
            program_start += program_stride

    _load_portal_kernel._cached = (
        triton,
        portal_partition_kernel,
        portal_partition_persistent_kernel,
    )
    return _load_portal_kernel._cached


def partition_certified_box_portals(
    positions: Any,
    directions: Any,
    input_queue: Any,
    last_instances: Any,
    last_triangles: Any,
    descriptor: PortalDescriptor,
    *,
    workspace: Optional[PortalWorkspace] = None,
    block_size: int = 256,
    input_capacity: Optional[int] = None,
    launch_capacity: Optional[int] = None,
    programs_per_sm: Optional[int] = None,
) -> PortalPartition:
    """Partition boundary IDs into exact known portals and scene fallbacks.

    A tensor input retains the original synchronous-shape API: its full length
    is launched and both capacity keywords must be omitted.  A
    :class:`~chroma.triton.transport.DeviceQueue` instead requires explicit
    ``input_capacity`` and ``launch_capacity`` host bounds while its live
    length remains exclusively in ``input_queue.count``.  The required
    scheduler invariant is::

        0 <= input_queue.count <= input_capacity <= launch_capacity

    The DeviceQueue route never materializes that count on the host.
    ``programs_per_sm`` can select an occupancy-bounded persistent grid whose
    programs walk logical blocks in grid strides; zero, the measured A100
    default, retains the uncapped capacity grid.  Tensor input retains its
    original one-program-per-block grid and rejects this device-only tuning
    keyword.  Returned direct/fallback queues plus compact direct-hit rows are
    views into ``workspace`` valid until it is reused.
    """

    import torch
    from chroma.triton.transport import DeviceQueue

    count_is_pointer = isinstance(input_queue, DeviceQueue)
    if count_is_pointer and (
        input_capacity is None or launch_capacity is None
    ):
        raise ValueError(
            "DeviceQueue input requires explicit input_capacity and "
            "launch_capacity"
        )
    if not count_is_pointer and (
        input_capacity is not None or launch_capacity is not None
    ):
        raise ValueError(
            "input_capacity and launch_capacity are only valid for DeviceQueue input"
        )
    if not count_is_pointer and programs_per_sm is not None:
        raise ValueError("programs_per_sm is only valid for DeviceQueue input")

    if (
        not isinstance(positions, torch.Tensor)
        or not isinstance(directions, torch.Tensor)
        or positions.shape != directions.shape
        or positions.ndim != 2
        or positions.shape[1] != 3
        or positions.dtype != torch.float32
        or directions.dtype != torch.float32
        or not positions.is_cuda
        or not directions.is_cuda
        or not positions.is_contiguous()
        or not directions.is_contiguous()
    ):
        raise ValueError("positions/directions must be contiguous CUDA float32 [N,3]")
    if count_is_pointer:
        input_buffer = input_queue.buffer
        input_count = input_queue.count
        if (
            not isinstance(input_buffer, torch.Tensor)
            or not input_buffer.is_cuda
            or input_buffer.ndim != 1
            or input_buffer.dtype != torch.int32
            or not input_buffer.is_contiguous()
            or input_buffer.device != positions.device
        ):
            raise ValueError(
                "DeviceQueue buffer must be contiguous same-device CUDA int32 [M]"
            )
        if (
            not isinstance(input_count, torch.Tensor)
            or not input_count.is_cuda
            or input_count.device != positions.device
            or input_count.dtype != torch.int32
            or input_count.shape != (1,)
            or not input_count.is_contiguous()
        ):
            raise ValueError(
                "DeviceQueue count must be contiguous same-device CUDA int32 [1]"
            )
        input_capacity = int(input_capacity)
        launch_capacity = int(launch_capacity)
        if input_capacity < 0 or input_capacity > input_buffer.numel():
            raise ValueError("input_capacity must fit the DeviceQueue buffer")
        if launch_capacity < input_capacity:
            raise ValueError(
                "launch_capacity must be at least input_capacity to avoid truncation"
            )
        nitems_argument = input_count
    else:
        input_buffer = input_queue
        if (
            not isinstance(input_buffer, torch.Tensor)
            or not input_buffer.is_cuda
            or input_buffer.ndim != 1
            or input_buffer.dtype != torch.int32
            or not input_buffer.is_contiguous()
            or input_buffer.device != positions.device
        ):
            raise ValueError(
                "input_queue must be contiguous same-device CUDA int32 [M]"
            )
        input_capacity = int(input_buffer.numel())
        launch_capacity = input_capacity
        nitems_argument = input_capacity
    for value, name in (
        (last_instances, "last_instances"),
        (last_triangles, "last_triangles"),
    ):
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_cuda
            or value.device != positions.device
            or value.dtype != torch.int32
            or value.shape != (positions.shape[0],)
            or not value.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous same-device CUDA int32 [N]")
    if not isinstance(descriptor, PortalDescriptor):
        raise TypeError("descriptor must be a PortalDescriptor")
    lo = np.asarray(descriptor.lower, dtype=np.float32)
    hi = np.asarray(descriptor.upper, dtype=np.float32)
    if lo.shape != (3,) or hi.shape != (3,) or np.any(lo >= hi):
        raise ValueError("portal descriptor has invalid bounds")
    if block_size not in (64, 128, 256, 512):
        raise ValueError("block_size must be 64, 128, 256, or 512")
    output_capacity = launch_capacity
    if workspace is None:
        workspace = PortalWorkspace.allocate(output_capacity, positions.device)
    if (
        not isinstance(workspace, PortalWorkspace)
        or workspace.capacity < output_capacity
        or workspace.queues.device != positions.device
        or workspace.queues.dtype != torch.int32
        or workspace.queues.shape != (2, workspace.capacity)
        or not workspace.queues.is_contiguous()
        or workspace.counts.dtype != torch.int32
        or workspace.counts.shape != (2,)
        or workspace.counts.device != positions.device
        or not workspace.counts.is_contiguous()
    ):
        raise ValueError(
            "portal workspace must be same-device int32 with capacity >= launch_capacity"
        )
    specifications = {
        "distance": ((workspace.capacity,), torch.float32),
        "normal": ((workspace.capacity, 3), torch.float32),
        "material_from": ((workspace.capacity,), torch.int32),
        "material_to": ((workspace.capacity,), torch.int32),
        "surface": ((workspace.capacity,), torch.int32),
        "instance": ((workspace.capacity,), torch.int32),
        "triangle": ((workspace.capacity,), torch.int32),
        "channel": ((workspace.capacity,), torch.int32),
    }
    for name, (shape, dtype) in specifications.items():
        value = getattr(workspace, name)
        if (
            not isinstance(value, torch.Tensor)
            or value.device != positions.device
            or value.dtype != dtype
            or value.shape != shape
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"portal workspace {name} must be contiguous {dtype} {shape}"
            )
    if count_is_pointer:
        if _tensor_storage_overlaps(input_buffer, workspace.queues):
            raise ValueError(
                "DeviceQueue input and portal output queue buffers must not alias"
            )
        if _tensor_storage_overlaps(input_count, workspace.counts):
            raise ValueError(
                "DeviceQueue input and portal output queue counts must not alias"
            )
    workspace.counts.zero_()
    triton, tensor_kernel, persistent_kernel = _load_portal_kernel()
    if count_is_pointer:
        selected_programs_per_sm = (
            DEVICE_PORTAL_PROGRAMS_PER_SM
            if programs_per_sm is None else int(programs_per_sm)
        )
        if selected_programs_per_sm < 0:
            raise ValueError("portal programs per SM cannot be negative")
        if selected_programs_per_sm:
            multiprocessor_count = torch.cuda.get_device_properties(
                positions.device
            ).multi_processor_count
            program_count = _persistent_portal_program_count(
                input_capacity,
                block_size,
                multiprocessor_count,
                programs_per_sm=selected_programs_per_sm,
            )
            kernel = persistent_kernel
        else:
            # The measured A100 default is the original capacity-grid kernel.
            program_count = triton.cdiv(launch_capacity, block_size)
            kernel = tensor_kernel
    else:
        # Preserve the synchronized tensor route's original launch grid.
        program_count = triton.cdiv(launch_capacity, block_size)
        kernel = tensor_kernel
    if program_count:
        kernel[(program_count,)](
            positions,
            directions,
            input_buffer,
            last_instances,
            last_triangles,
            workspace.queues[0],
            workspace.queues[1],
            workspace.counts,
            *workspace.hit_outputs(),
            nitems_argument,
            float(lo[0]),
            float(lo[1]),
            float(lo[2]),
            float(hi[0]),
            float(hi[1]),
            float(hi[2]),
            int(descriptor.lar_material),
            int(descriptor.active_outside_material),
            int(descriptor.cathode_inside_material),
            int(descriptor.active_surface),
            int(descriptor.cathode_surface),
            -int(descriptor.active_box_index) - 2,
            -int(descriptor.cathode_box_index) - 2,
            input_capacity,
            NITEMS_IS_POINTER=count_is_pointer,
            BLOCK=int(block_size),
            num_warps=min(4, max(1, block_size // 32)),
        )
    if count_is_pointer:
        direct_buffer = workspace.queues[0, :launch_capacity]
        fallback_buffer = workspace.queues[1, :launch_capacity]
        hit_outputs = tuple(
            value[:launch_capacity] for value in workspace.hit_outputs()
        )
    else:
        # Keep the original tensor-input result shapes, including the case in
        # which a caller supplied a workspace larger than the input tensor.
        direct_buffer = workspace.queues[0]
        fallback_buffer = workspace.queues[1]
        hit_outputs = workspace.hit_outputs()
    return PortalPartition(
        direct=DeviceQueue(direct_buffer, workspace.counts[0:1]),
        fallback=DeviceQueue(fallback_buffer, workspace.counts[1:2]),
        hit=hit_outputs,
    )


__all__ = [
    "DEVICE_PORTAL_PROGRAMS_PER_SM",
    "PortalPartition",
    "PortalDescriptor",
    "PortalWorkspace",
    "_persistent_portal_program_count",
    "certify_reflect3wires_portals",
    "classify_certified_box_portals_numpy",
    "partition_certified_box_portals",
]
