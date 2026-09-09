"""Experimental exact fused traversal for the reflect3wires PMT lattice.

This module is intentionally isolated from :mod:`instances` so the fused
algorithm can be benchmarked before it becomes a production dependency.  It
removes the legacy count/prefix/materialize/traverse/reduce sequence.  One
Triton program owns each ray and does all of the following in registers:

* derive a conservative row/column rectangle from the validated PMT lattice,
* test the exact outward-padded instance boxes in row-major order,
* transform the ray and traverse the shared canonical BLAS immediately, and
* retain the nearest hit with the same strict-distance tie rule.

The lattice locator is only an acceleration proof.  A ray whose interval is
not finite/certifiable, or whose rectangle is wider than an optional tuning
limit, scans every instance box in ascending order inside the same kernel.
That per-ray fallback is exact and needs neither a host rendezvous nor a
second output merge.  Rays missing the conservative union box are immediate
misses.

Integration contract
--------------------
``nearest_pmt_hit_fused_grid_device_count`` is a drop-in experimental peer of
``nearest_pmt_hit_tlas_device_count``.  It accepts a device-resident live
count, leaves the inactive output suffix untouched, uses the existing
``PMTInstanceWorkspace``, and records topology overflow in its sticky flag.
It is production-arithmetic only: strict flattened-world Chroma replay must
continue to use the established traversal.  ``nearest_pmt_hit_fused_grid`` is
the synchronized convenience/audit entry point and independently falls back
to the legacy exact implementation if a corrupted stack ever overflows.

The private jitted helpers imported from :mod:`instances` are deliberate.
They make triangle arithmetic, edge tolerance, last-triangle exclusion, and
the detector-scale progress guard literally identical rather than merely
equivalent source code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from . import instances as _instances


# The fallback in this implementation scans every instance rather than a
# separate TLAS.  Consequently every certified rectangle (at most 9*9 boxes)
# is weakly cheaper than fallback; 81 is both the exact dominance bound and
# the target detector's default.  Smaller values remain useful for ablations.
FUSED_GRID_MAX_CANDIDATES = 81
ROUTING_CERTIFIED = 0
ROUTING_FALLBACK = 1
ROUTING_INSTANCE_VISITS = 2
ROUTING_COUNTER_COUNT = 3


try:  # Preserve NumPy-only scene tooling imports.
    import torch
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised by minimal artifact installs.
    torch = None
    triton = None
    tl = None


@dataclass(frozen=True)
class FusedPMTCandidatePlan:
    """CPU audit of the instance sequence selected for each ray.

    ``instance_ids`` are conservative candidates *before* the exact instance
    AABB test.  A certified ray contains its row-major lattice rectangle; a
    fallback ray contains every instance in ascending order; a union miss is
    empty.  This object is diagnostic only and is not materialized by the GPU
    implementation.
    """

    instance_ids: tuple[np.ndarray, ...]
    fallback: np.ndarray
    union_hit: np.ndarray


@dataclass
class FusedPMTRoutingCounters:
    """Opt-in device counters for fused-routing diagnostics.

    The three int64 words accumulate certified rays, exact-fallback rays, and
    instance-box visits.  The kernel contributes at most three atomics per
    program tile, never one atomic per ray.  When this object is not supplied,
    ``COUNT_ROUTING=False`` removes all reductions and atomics at compile time.
    """

    values: Any

    @classmethod
    def allocate(cls, device: Any = None) -> "FusedPMTRoutingCounters":
        _instances._require_backend()
        selected = torch.device("cuda" if device is None else device)
        if selected.type != "cuda":
            raise ValueError("fused PMT routing counters require a CUDA device")
        return cls(
            torch.zeros(
                ROUTING_COUNTER_COUNT,
                dtype=torch.int64,
                device=selected,
            )
        )

    def reset(self) -> "FusedPMTRoutingCounters":
        self.values.zero_()
        return self

    def snapshot(self) -> dict[str, int]:
        certified, fallback, visits = map(
            int, self.values.detach().cpu().tolist()
        )
        return {
            "certified_rays": certified,
            "fallback_rays": fallback,
            "instance_box_visits": visits,
        }


def _ray_box_interval_cpu(
    origin: np.ndarray,
    direction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tmax: float,
) -> tuple[bool, float, float]:
    """Mirror the GPU slab predicate while avoiding divide-by-zero warnings."""

    near = -np.inf
    far = np.inf
    for axis in range(3):
        component = float(direction[axis])
        coordinate = float(origin[axis])
        lo = float(lower[axis])
        hi = float(upper[axis])
        if component == 0.0:
            if coordinate < lo or coordinate > hi:
                return False, np.inf, -np.inf
            continue
        first = (lo - coordinate) / component
        second = (hi - coordinate) / component
        near = max(near, min(first, second))
        far = min(far, max(first, second))
    enter = max(near, 0.0)
    leave = min(far, float(tmax))
    return bool(leave >= enter), float(enter), float(leave)


def plan_fused_pmt_candidates(
    locator: _instances.PMTGridLocator,
    bounds_min: Any,
    bounds_max: Any,
    union_bounds_min: Any,
    union_bounds_max: Any,
    origins: Any,
    directions: Any,
    *,
    tmax: Optional[Any] = None,
    maximum_grid_candidates: int = FUSED_GRID_MAX_CANDIDATES,
) -> FusedPMTCandidatePlan:
    """Build a CPU proof/audit plan matching the fused-kernel routing.

    This function deliberately returns the conservative rectangle rather than
    only boxes that intersect.  Tests and artifact audits can therefore check
    the important invariant directly: every exact AABB hit is contained in
    the selected sequence, while an uncertified ray receives all instances.
    """

    lower = np.asarray(bounds_min, dtype=np.float32)
    upper = np.asarray(bounds_max, dtype=np.float32)
    union_lower = np.asarray(union_bounds_min, dtype=np.float32)
    union_upper = np.asarray(union_bounds_max, dtype=np.float32)
    origin_array = np.asarray(origins, dtype=np.float32)
    direction_array = np.asarray(directions, dtype=np.float32)
    if lower.ndim != 2 or lower.shape[1:] != (3,) or upper.shape != lower.shape:
        raise ValueError("bounds must have matching shape (N, 3)")
    expected_instances = int(locator.rows * locator.columns)
    if len(lower) != expected_instances:
        raise ValueError("bounds do not match the locator instance count")
    if union_lower.shape != (3,) or union_upper.shape != (3,):
        raise ValueError("union bounds must have shape (3,)")
    if origin_array.ndim != 2 or origin_array.shape[1:] != (3,):
        raise ValueError("origins must have shape (N, 3)")
    if direction_array.shape != origin_array.shape:
        raise ValueError("directions must have the same shape as origins")
    count = len(origin_array)
    if tmax is None:
        tmax_array = np.full(count, np.inf, dtype=np.float32)
    elif np.ndim(tmax) == 0:
        tmax_array = np.full(count, tmax, dtype=np.float32)
    else:
        tmax_array = np.asarray(tmax, dtype=np.float32)
        if tmax_array.shape != (count,):
            raise ValueError("tmax must be scalar or have shape (N,)")
    maximum_grid_candidates = int(maximum_grid_candidates)
    if maximum_grid_candidates <= 0:
        raise ValueError("maximum_grid_candidates must be positive")

    plans: list[np.ndarray] = []
    fallback = np.zeros(count, dtype=np.bool_)
    union_hit = np.zeros(count, dtype=np.bool_)
    all_instances = np.arange(expected_instances, dtype=np.int32)
    for ray in range(count):
        hit, enter, leave = _ray_box_interval_cpu(
            origin_array[ray],
            direction_array[ray],
            union_lower,
            union_upper,
            float(tmax_array[ray]),
        )
        union_hit[ray] = hit
        if not hit:
            plans.append(np.empty(0, dtype=np.int32))
            continue

        origin = origin_array[ray].astype(np.float64)
        direction = direction_array[ray].astype(np.float64)
        norm2 = float(np.dot(direction, direction))
        finite = bool(
            np.isfinite(enter)
            and np.isfinite(leave)
            and np.all(np.abs(origin) <= locator.origin_limit)
            and 0.5 <= norm2 <= 2.0
        )
        if finite:
            y_enter = origin[1] + enter * direction[1]
            y_leave = origin[1] + leave * direction[1]
            z_enter = origin[2] + enter * direction[2]
            z_leave = origin[2] + leave * direction[2]
            y_min = (
                min(y_enter, y_leave)
                - locator.half_y
                - locator.coordinate_guard
            )
            y_max = (
                max(y_enter, y_leave)
                + locator.half_y
                + locator.coordinate_guard
            )
            z_min = (
                min(z_enter, z_leave)
                - locator.half_z
                - locator.coordinate_guard
            )
            z_max = (
                max(z_enter, z_leave)
                + locator.half_z
                + locator.coordinate_guard
            )
            row_first = int(
                np.ceil(
                    (y_min - locator.row_zero_max_y) / locator.row_pitch
                )
            )
            row_stop = int(
                np.floor(
                    (y_max - locator.row_zero_min_y) / locator.row_pitch
                )
                + 1
            )
            column_first = int(
                np.ceil(
                    (z_min - locator.column_zero_z) / locator.column_pitch
                )
            )
            column_stop = int(
                np.floor(
                    (z_max - locator.column_zero_z) / locator.column_pitch
                )
                + 1
            )
            row_first = min(locator.rows, max(0, row_first))
            row_stop = min(locator.rows, max(0, row_stop))
            column_first = min(locator.columns, max(0, column_first))
            column_stop = min(locator.columns, max(0, column_stop))
            rectangle_size = max(0, row_stop - row_first) * max(
                0, column_stop - column_first
            )
        else:
            rectangle_size = expected_instances + 1

        if not finite or rectangle_size > maximum_grid_candidates:
            fallback[ray] = True
            plans.append(all_instances.copy())
            continue
        plans.append(
            np.asarray(
                [
                    row * locator.columns + column
                    for row in range(row_first, row_stop)
                    for column in range(column_first, column_stop)
                ],
                dtype=np.int32,
            )
        )
    if len(plans) != count:  # Structural guard for future routing edits.
        raise AssertionError("candidate planner must emit exactly one row per ray")
    return FusedPMTCandidatePlan(tuple(plans), fallback, union_hit)


if triton is not None and torch is not None:

    # Bind these as direct JIT globals.  Triton's dependency resolver can then
    # hash/in-line the exact helpers without having to interpret a Python
    # module attribute lookup from inside kernel source.
    _instance_aabb_hit = _instances._instance_aabb_hit
    _grid_candidate_range = _instances._grid_candidate_range
    _canonical_blas_hit = _instances._canonical_blas_hit

    # Live count and capacity vary on every scheduler round.  Keeping both as
    # runtime scalars prevents a new kernel compilation for each queue size;
    # the earlier [25,26] indices accidentally covered locator floats instead.
    @triton.jit(do_not_specialize=[21, 22])
    def _nearest_pmt_fused_grid_kernel(
        bounds_min,
        bounds_max,
        union_bounds_min,
        union_bounds_max,
        world_to_object_rotation,
        world_to_object_translation,
        blas_nodes,
        triangle_vertices,
        origins,
        directions,
        tmax_values,
        last_instances,
        last_triangles,
        blas_stack,
        out_triangles,
        out_distances,
        out_instances,
        out_overflow,
        out_candidate_counts,
        sticky_overflow,
        routing_counts,
        n_rays,
        ray_stride,
        ray_index_ptr,
        unused_compatibility_pointer,
        row_pitch,
        column_pitch,
        row_zero_min_y,
        row_zero_max_y,
        column_zero_z,
        half_y,
        half_z,
        coordinate_guard,
        origin_limit,
        world_x,
        world_y,
        world_z,
        world_scale,
        N_INSTANCES: tl.constexpr,
        GRID_ROWS: tl.constexpr,
        GRID_COLUMNS: tl.constexpr,
        MAX_GRID_CANDIDATES: tl.constexpr,
        N_RAYS_IS_POINTER: tl.constexpr,
        INDIRECT: tl.constexpr,
        COUNT_ROUTING: tl.constexpr,
        BLAS_STACK_CAPACITY: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Traverse a certified lattice rectangle or exact ascending fallback."""

        program_start = tl.program_id(0) * BLOCK_SIZE
        if N_RAYS_IS_POINTER:
            active_count = tl.load(n_rays).to(tl.int32)
            active_count = tl.maximum(0, tl.minimum(active_count, ray_stride))
            if program_start >= active_count:
                return
        else:
            active_count = n_rays
        program_stride = tl.num_programs(0) * BLOCK_SIZE
        while program_start < active_count:
            work_slot = program_start + tl.arange(0, BLOCK_SIZE)
            valid = work_slot < active_count
            if INDIRECT:
                ray = tl.load(
                    ray_index_ptr + work_slot, mask=valid, other=0
                ).to(tl.int32)
            else:
                ray = work_slot
            ox = tl.load(origins + ray * 3 + 0, mask=valid, other=0.0)
            oy = tl.load(origins + ray * 3 + 1, mask=valid, other=0.0)
            oz = tl.load(origins + ray * 3 + 2, mask=valid, other=0.0)
            dx = tl.load(directions + ray * 3 + 0, mask=valid, other=1.0)
            dy = tl.load(directions + ray * 3 + 1, mask=valid, other=1.0)
            dz = tl.load(directions + ray * 3 + 2, mask=valid, other=1.0)
            ray_tmax = tl.load(tmax_values + ray, mask=valid, other=-1.0)
            previous_instance = tl.load(
                last_instances + ray, mask=valid, other=-1
            )
            previous_triangle = tl.load(
                last_triangles + ray, mask=valid, other=-1
            )
            union_hit = valid & _instance_aabb_hit(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                tl.load(union_bounds_min + 0),
                tl.load(union_bounds_min + 1),
                tl.load(union_bounds_min + 2),
                tl.load(union_bounds_max + 0),
                tl.load(union_bounds_max + 1),
                tl.load(union_bounds_max + 2),
                ray_tmax,
            )
            (
                row_first,
                row_count,
                column_first,
                column_count,
                interval_finite,
            ) = _grid_candidate_range(
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                ray_tmax,
                union_bounds_min,
                union_bounds_max,
                row_pitch,
                column_pitch,
                row_zero_min_y,
                row_zero_max_y,
                column_zero_z,
                half_y,
                half_z,
                coordinate_guard,
                origin_limit,
                GRID_ROWS=GRID_ROWS,
                GRID_COLUMNS=GRID_COLUMNS,
            )
            rectangle_count = row_count * column_count
            certified = (
                union_hit
                & interval_finite
                & (rectangle_count <= MAX_GRID_CANDIDATES)
            )
            visit_count = tl.where(
                certified,
                rectangle_count,
                tl.where(union_hit, N_INSTANCES, 0),
            )
            safe_column_count = tl.maximum(column_count, 1)

            best_distance = ray_tmax
            best_triangle = tl.full((BLOCK_SIZE,), -1, tl.int32)
            best_instance = tl.full((BLOCK_SIZE,), -1, tl.int32)
            candidate_count = tl.zeros((BLOCK_SIZE,), tl.int32)
            overflow = tl.zeros((BLOCK_SIZE,), tl.int1)
            candidate_slot = tl.zeros((BLOCK_SIZE,), tl.int32)
            visiting = valid & (candidate_slot < visit_count)
            while tl.sum(visiting.to(tl.int32), axis=0) != 0:
                grid_row = row_first + candidate_slot // safe_column_count
                grid_column = (
                    column_first + candidate_slot % safe_column_count
                )
                grid_instance = grid_row * GRID_COLUMNS + grid_column
                instance = tl.where(certified, grid_instance, candidate_slot)
                safe_instance = tl.where(visiting, instance, 0)
                lo_x = tl.load(
                    bounds_min + safe_instance * 3 + 0,
                    mask=visiting,
                    other=0.0,
                )
                lo_y = tl.load(
                    bounds_min + safe_instance * 3 + 1,
                    mask=visiting,
                    other=0.0,
                )
                lo_z = tl.load(
                    bounds_min + safe_instance * 3 + 2,
                    mask=visiting,
                    other=0.0,
                )
                hi_x = tl.load(
                    bounds_max + safe_instance * 3 + 0,
                    mask=visiting,
                    other=0.0,
                )
                hi_y = tl.load(
                    bounds_max + safe_instance * 3 + 1,
                    mask=visiting,
                    other=0.0,
                )
                hi_z = tl.load(
                    bounds_max + safe_instance * 3 + 2,
                    mask=visiting,
                    other=0.0,
                )
                box_hit = visiting & _instance_aabb_hit(
                    ox,
                    oy,
                    oz,
                    dx,
                    dy,
                    dz,
                    lo_x,
                    lo_y,
                    lo_z,
                    hi_x,
                    hi_y,
                    hi_z,
                    ray_tmax,
                )
                candidate_count += box_hit.to(tl.int32)
                could_improve = box_hit & _instance_aabb_hit(
                    ox,
                    oy,
                    oz,
                    dx,
                    dy,
                    dz,
                    lo_x,
                    lo_y,
                    lo_z,
                    hi_x,
                    hi_y,
                    hi_z,
                    best_distance,
                )

                rotation = safe_instance * 9
                r00 = tl.load(
                    world_to_object_rotation + rotation + 0,
                    mask=could_improve,
                    other=0.0,
                )
                r01 = tl.load(
                    world_to_object_rotation + rotation + 1,
                    mask=could_improve,
                    other=0.0,
                )
                r02 = tl.load(
                    world_to_object_rotation + rotation + 2,
                    mask=could_improve,
                    other=0.0,
                )
                r10 = tl.load(
                    world_to_object_rotation + rotation + 3,
                    mask=could_improve,
                    other=0.0,
                )
                r11 = tl.load(
                    world_to_object_rotation + rotation + 4,
                    mask=could_improve,
                    other=0.0,
                )
                r12 = tl.load(
                    world_to_object_rotation + rotation + 5,
                    mask=could_improve,
                    other=0.0,
                )
                r20 = tl.load(
                    world_to_object_rotation + rotation + 6,
                    mask=could_improve,
                    other=0.0,
                )
                r21 = tl.load(
                    world_to_object_rotation + rotation + 7,
                    mask=could_improve,
                    other=0.0,
                )
                r22 = tl.load(
                    world_to_object_rotation + rotation + 8,
                    mask=could_improve,
                    other=0.0,
                )
                tx = tl.load(
                    world_to_object_translation + safe_instance * 3 + 0,
                    mask=could_improve,
                    other=0.0,
                )
                ty = tl.load(
                    world_to_object_translation + safe_instance * 3 + 1,
                    mask=could_improve,
                    other=0.0,
                )
                tz = tl.load(
                    world_to_object_translation + safe_instance * 3 + 2,
                    mask=could_improve,
                    other=0.0,
                )
                local_ox = r00 * ox + r01 * oy + r02 * oz + tx
                local_oy = r10 * ox + r11 * oy + r12 * oz + ty
                local_oz = r20 * ox + r21 * oy + r22 * oz + tz
                local_dx = r00 * dx + r01 * dy + r02 * dz
                local_dy = r10 * dx + r11 * dy + r12 * dz
                local_dz = r20 * dx + r21 * dy + r22 * dz
                excluded = tl.where(
                    previous_instance == safe_instance,
                    previous_triangle,
                    -1,
                )
                local_triangle, local_distance, blas_overflow = (
                    _canonical_blas_hit(
                        blas_nodes,
                        triangle_vertices,
                        triangle_vertices,
                        unused_compatibility_pointer,
                        safe_instance,
                        local_ox,
                        local_oy,
                        local_oz,
                        local_dx,
                        local_dy,
                        local_dz,
                        ox,
                        oy,
                        oz,
                        dx,
                        dy,
                        dz,
                        best_distance,
                        excluded,
                        could_improve,
                        blas_stack,
                        ray_stride,
                        work_slot,
                        world_x,
                        world_y,
                        world_z,
                        world_scale,
                        N_WORLD_VERTICES=1,
                        CHROMA_WORLD_GEOMETRY=False,
                        STACK_CAPACITY=BLAS_STACK_CAPACITY,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                )
                improved = could_improve & (local_triangle >= 0)
                best_distance = tl.where(
                    improved, local_distance, best_distance
                )
                best_triangle = tl.where(
                    improved, local_triangle, best_triangle
                )
                best_instance = tl.where(
                    improved, safe_instance, best_instance
                )
                overflow |= blas_overflow
                candidate_slot += 1
                visiting = valid & (candidate_slot < visit_count)

            tl.store(out_triangles + ray, best_triangle, mask=valid)
            tl.store(
                out_distances + ray,
                tl.where(best_triangle >= 0, best_distance, float("inf")),
                mask=valid,
            )
            tl.store(out_instances + ray, best_instance, mask=valid)
            tl.store(out_overflow + ray, overflow.to(tl.uint8), mask=valid)
            tl.store(
                out_candidate_counts + ray, candidate_count, mask=valid
            )
            tl.atomic_or(
                sticky_overflow + tl.zeros((BLOCK_SIZE,), tl.int32),
                tl.full((BLOCK_SIZE,), 1, tl.int32),
                mask=valid & overflow,
            )
            if COUNT_ROUTING:
                certified_total = tl.sum(
                    (valid & certified).to(tl.int32), axis=0
                ).to(tl.int64)
                fallback_total = tl.sum(
                    (valid & union_hit & ~certified).to(tl.int32), axis=0
                ).to(tl.int64)
                visit_total = tl.sum(
                    tl.where(valid, visit_count, 0), axis=0
                ).to(tl.int64)
                tl.atomic_add(
                    routing_counts + 0, certified_total
                )
                tl.atomic_add(
                    routing_counts + 1, fallback_total
                )
                tl.atomic_add(
                    routing_counts + 2, visit_total
                )
            program_start += program_stride


def _per_ray_prefix(
    values: Any,
    *,
    capacity: int,
    default: Any,
    dtype: Any,
    name: str,
    device: Any,
) -> Any:
    """Validate a scalar or capacity-sized input without a device count read."""

    if values is None:
        return torch.full((capacity,), default, dtype=dtype, device=device)
    if not isinstance(values, torch.Tensor):
        if np.ndim(values) == 0:
            return torch.full((capacity,), values, dtype=dtype, device=device)
        result = torch.as_tensor(values, dtype=dtype, device=device)
    else:
        result = values.to(device=device, dtype=dtype)
    if result.ndim == 0:
        return torch.full(
            (capacity,), result.item(), dtype=dtype, device=device
        )
    if result.ndim != 1 or result.shape[0] < capacity:
        raise ValueError(
            f"{name} must be scalar or have at least launch_capacity entries"
        )
    return result[:capacity].contiguous()


def _routing_counter_storage(
    counters: Optional[Any], device: Any, dummy: Any
) -> tuple[Any, bool]:
    """Return validated counter storage and its compile-time enable flag."""

    if counters is None:
        return dummy, False
    values = counters.values if isinstance(
        counters, FusedPMTRoutingCounters
    ) else counters
    if (
        not isinstance(values, torch.Tensor)
        or values.shape != (ROUTING_COUNTER_COUNT,)
        or values.dtype != torch.int64
        or values.device != device
        or not values.is_contiguous()
    ):
        raise ValueError(
            "routing_counters must be a FusedPMTRoutingCounters or contiguous "
            "CUDA int64 tensor with shape (3,) on the accelerator device"
        )
    return values, True


def nearest_pmt_hit_fused_grid_device_count(
    accelerator: _instances.PMTInstanceAccelerator,
    origins: Any,
    directions: Any,
    active_count: Any,
    *,
    launch_capacity: Optional[int] = None,
    tmax: Optional[Any] = None,
    last_instance: Optional[Any] = None,
    last_triangle: Optional[Any] = None,
    workspace: Optional[_instances.PMTInstanceWorkspace] = None,
    out: Optional[_instances.PMTInstanceResult] = None,
    maximum_grid_candidates: int = FUSED_GRID_MAX_CANDIDATES,
    routing_counters: Optional[Any] = None,
    compact_union: bool = True,
    _fixed_count: bool = False,
) -> _instances.PMTInstanceResult:
    """Run the fused lattice/BLAS traversal from a device-resident count.

    The returned tensors have ``launch_capacity`` storage.  Only rows below
    ``active_count`` are written; the inactive suffix remains untouched.
    With ``compact_union=False`` the fused kernel consumes that live prefix
    directly and writes all geometric hit/miss records; one small exact-normal
    finalizer follows.  The old initialize/union-compact/indirect route remains
    available for A/B tests.  Inlining normal reconstruction into the already
    complex nested traversal currently crashes Triton's TTIR compiler.

    A100 full-simulation evidence keeps compaction as the default: at five
    million photons the compact device route sustained 17.41--18.24 M/s,
    while the warmed direct route reached 16.93 M/s.  Packing the roughly 40%
    PMT-union population into full warps repays the surrounding launches.
    """

    _instances._require_backend()
    if (last_instance is None) != (last_triangle is None):
        raise ValueError(
            "last_instance and last_triangle must be supplied together"
        )
    locator = accelerator.grid_locator
    if locator is None:
        raise ValueError("accelerator has no certified regular PMT lattice")
    if accelerator.instance_count != locator.rows * locator.columns:
        raise ValueError("grid locator and accelerator instance count disagree")
    maximum_grid_candidates = int(maximum_grid_candidates)
    compact_union = bool(compact_union)
    if maximum_grid_candidates <= 0:
        raise ValueError("maximum_grid_candidates must be positive")

    device = accelerator.device
    origin_storage = _instances._torch_rays(origins, "origins", device)
    direction_storage = _instances._torch_rays(
        directions, "directions", device
    )
    if direction_storage.shape != origin_storage.shape:
        raise ValueError("directions must have the same shape as origins")
    storage_capacity = int(origin_storage.shape[0])
    capacity = (
        storage_capacity if launch_capacity is None else int(launch_capacity)
    )
    if capacity < 0 or capacity > storage_capacity:
        raise ValueError(
            "launch_capacity must be between zero and ray storage capacity"
        )
    fixed_count = bool(_fixed_count)
    if fixed_count:
        active_argument = int(active_count)
        if active_argument < 0 or active_argument > capacity:
            raise ValueError("fixed active count must fit launch_capacity")
        if compact_union:
            raise ValueError("fixed-count launch is available only without compaction")
    else:
        _instances._validate_device_count_tensor(active_count, device)
        active_argument = active_count
    origin_tensor = origin_storage[:capacity]
    direction_tensor = direction_storage[:capacity]
    tmax_tensor = _per_ray_prefix(
        tmax,
        capacity=capacity,
        default=float("inf"),
        dtype=torch.float32,
        name="tmax",
        device=device,
    )
    last_instance_tensor = _per_ray_prefix(
        last_instance,
        capacity=capacity,
        default=-1,
        dtype=torch.int32,
        name="last_instance",
        device=device,
    )
    last_triangle_tensor = _per_ray_prefix(
        last_triangle,
        capacity=capacity,
        default=-1,
        dtype=torch.int32,
        name="last_triangle",
        device=device,
    )

    if workspace is None:
        workspace = accelerator.allocate_workspace(
            capacity, result_capacity=capacity
        )
    elif workspace.accelerator is not accelerator:
        raise ValueError("workspace belongs to a different PMT accelerator")
    else:
        workspace.ensure_ray_capacity(capacity)
        workspace.ensure_result_capacity(capacity)
    if out is None:
        out = workspace.outputs(capacity)
    result = _instances._validate_instance_result(out, capacity, device)
    routing_storage, count_routing = _routing_counter_storage(
        routing_counters, device, workspace.bvh_dummy
    )
    if capacity == 0:
        return result

    if compact_union:
        # Historical A/B route: initialize union misses, atomically compact
        # union candidates, then scatter their fused geometric records.
        _instances._initialize_instance_results_kernel[
            (triton.cdiv(capacity, 256),)
        ](
            result.distances,
            result.triangle_ids,
            result.instance_ids,
            result.channel_ids,
            result.world_normals,
            result.overflow,
            result.candidate_counts,
            active_argument,
            capacity,
            N_RAYS_IS_POINTER=True,
            BLOCK_SIZE=256,
            num_warps=8,
        )
        workspace.device_candidate_count.zero_()
        compact_block = 256
        _instances._compact_union_candidates_device_count_kernel[
            (triton.cdiv(capacity, compact_block),)
        ](
            origin_tensor,
            direction_tensor,
            tmax_tensor,
            accelerator.union_bounds_min,
            accelerator.union_bounds_max,
            active_argument,
            workspace.active_ray_ids,
            workspace.device_candidate_count,
            capacity,
            BLOCK_SIZE=compact_block,
            num_warps=8,
        )
        traversal_count = workspace.device_candidate_count
        traversal_indices = workspace.active_ray_ids
    else:
        traversal_count = active_argument
        # Compile-time dead when INDIRECT=False; retain a valid same-device
        # pointer so the launch signature is uniform.
        traversal_indices = workspace.active_ray_ids

    bvh = accelerator.device_bvh
    block_size = 32
    multiprocessor_count = torch.cuda.get_device_properties(
        device
    ).multi_processor_count
    traversal_programs = _instances._persistent_tlas_program_count(
        capacity, block_size, multiprocessor_count
    )
    _nearest_pmt_fused_grid_kernel[(traversal_programs,)](
        accelerator.bounds_min,
        accelerator.bounds_max,
        accelerator.union_bounds_min,
        accelerator.union_bounds_max,
        accelerator.world_to_object_rotation,
        accelerator.world_to_object_translation,
        bvh.nodes,
        bvh.triangle_vertices,
        origin_tensor,
        direction_tensor,
        tmax_tensor,
        last_instance_tensor,
        last_triangle_tensor,
        workspace.fused_traversal.stack,
        result.triangle_ids,
        result.distances,
        result.instance_ids,
        result.overflow,
        result.candidate_counts,
        workspace.sticky_overflow,
        routing_storage,
        traversal_count,
        capacity,
        traversal_indices,
        workspace.bvh_dummy,
        locator.row_pitch,
        locator.column_pitch,
        locator.row_zero_min_y,
        locator.row_zero_max_y,
        locator.column_zero_z,
        locator.half_y,
        locator.half_z,
        locator.coordinate_guard,
        locator.origin_limit,
        bvh.world_origin[0],
        bvh.world_origin[1],
        bvh.world_origin[2],
        bvh.world_scale,
        N_INSTANCES=accelerator.instance_count,
        GRID_ROWS=locator.rows,
        GRID_COLUMNS=locator.columns,
        MAX_GRID_CANDIDATES=maximum_grid_candidates,
        N_RAYS_IS_POINTER=not fixed_count,
        INDIRECT=compact_union,
        COUNT_ROUTING=count_routing,
        BLAS_STACK_CAPACITY=bvh.stack_capacity,
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
    finalize_count = (
        workspace.device_candidate_count if compact_union else active_argument
    )
    _instances._finalize_instance_hits_kernel[
        (triton.cdiv(capacity, 256),)
    ](
        bvh.triangle_vertices,
        accelerator.object_to_world_rotation,
        accelerator.channel_ids,
        result.triangle_ids,
        result.instance_ids,
        result.channel_ids,
        result.world_normals,
        finalize_count,
        capacity,
        workspace.active_ray_ids,
        N_RAYS_IS_POINTER=(True if compact_union else not fixed_count),
        INDIRECT=compact_union,
        BLOCK_SIZE=256,
        num_warps=8,
    )
    return result


def nearest_pmt_hit_fused_grid(
    accelerator: _instances.PMTInstanceAccelerator,
    origins: Any,
    directions: Any,
    *,
    tmax: Optional[Any] = None,
    last_instance: Optional[Any] = None,
    last_triangle: Optional[Any] = None,
    workspace: Optional[_instances.PMTInstanceWorkspace] = None,
    out: Optional[_instances.PMTInstanceResult] = None,
    maximum_grid_candidates: int = FUSED_GRID_MAX_CANDIDATES,
    routing_counters: Optional[Any] = None,
    compact_union: bool = True,
    check_overflow: bool = True,
) -> _instances.PMTInstanceResult:
    """Synchronized convenience entry point for correctness and A/B tests."""

    _instances._require_backend()
    device = accelerator.device
    origin_tensor = _instances._torch_rays(origins, "origins", device)
    direction_tensor = _instances._torch_rays(
        directions, "directions", device
    )
    if direction_tensor.shape != origin_tensor.shape:
        raise ValueError("directions must have the same shape as origins")
    count = int(origin_tensor.shape[0])
    if workspace is None:
        workspace = accelerator.allocate_workspace(
            count, result_capacity=count
        )
    elif workspace.accelerator is not accelerator:
        raise ValueError("workspace belongs to a different PMT accelerator")
    else:
        workspace.ensure_ray_capacity(count)
        workspace.ensure_result_capacity(count)
    if out is None:
        out = workspace.outputs(count)
    if compact_union:
        # The A/B route's compactor consumes a pointer.  Reuse prefix scratch
        # rather than allocating a CUDA scalar on every boundary round.
        count_argument = workspace.counts[:1]
        count_argument.fill_(count)
    else:
        # The all-live specialization embeds the already-known host count and
        # avoids a count-scalar fill before traversal and finalization.
        count_argument = count
    result = nearest_pmt_hit_fused_grid_device_count(
        accelerator,
        origin_tensor,
        direction_tensor,
        count_argument,
        launch_capacity=count,
        tmax=tmax,
        last_instance=last_instance,
        last_triangle=last_triangle,
        workspace=workspace,
        out=out,
        maximum_grid_candidates=maximum_grid_candidates,
        routing_counters=routing_counters,
        compact_union=compact_union,
        _fixed_count=not compact_union,
    )
    if check_overflow and count and bool(torch.any(result.overflow).item()):
        # This is only reachable for externally corrupted topology/workspace
        # metadata.  Use the independent exact path rather than accepting a
        # partial result.
        return _instances.nearest_pmt_hit(
            accelerator,
            origin_tensor,
            direction_tensor,
            tmax=tmax,
            last_instance=last_instance,
            last_triangle=last_triangle,
            workspace=workspace,
            out=result,
            check_overflow=True,
            use_tlas=False,
            use_grid=False,
        )
    return result


__all__ = [
    "FUSED_GRID_MAX_CANDIDATES",
    "FusedPMTCandidatePlan",
    "FusedPMTRoutingCounters",
    "ROUTING_CERTIFIED",
    "ROUTING_COUNTER_COUNT",
    "ROUTING_FALLBACK",
    "ROUTING_INSTANCE_VISITS",
    "nearest_pmt_hit_fused_grid",
    "nearest_pmt_hit_fused_grid_device_count",
    "plan_fused_pmt_candidates",
]
