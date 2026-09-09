"""Exact analytic intersections for the specialized reflect3wires scene.

The target geometry contains two classes of boundaries which do not need a
triangle BVH:

* enabled faces of the active enclosure, cathode, and terminal outer cavity;
  and
* three finite, periodic arrays of cylindrical wires.

The wire implementation intentionally follows ``chroma/cuda/photon.h``.  In
particular it preserves the FP64 local frame, the conservative integer-wire
cull, the 0.1 micron self-hit step, and the outside/on/inside root choice.
The optional Triton path performs the same operations on a boundary-event
queue; importing this module does not require Torch or Triton.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional, TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - imports used only by type checkers
    from .compiler import AnalyticBoxes, AnalyticWires, CompiledReflect3WiresScene


CHROMA_EPSILON = np.float32(1.0e-6)
WIRE_T_MIN = 1.0e-4
WIRE_PARALLEL_U = 1.0e-15
WIRE_PARALLEL_PLANE = 1.0e-12


class GeometryKind(IntEnum):
    """Stable tags shared with the target-specific transport backend."""

    MISS = 0
    BOX = 1
    WIRE = 2


@dataclass(frozen=True)
class BoundaryIntersections:
    """Nearest analytic boundary for each input ray.

    ``outward_normal`` is the primitive's geometric normal.
    ``surface_normal`` is oriented against the incident direction exactly as
    Chroma's ``fill_state`` expects.  ``primitive_index`` is a box face in
    ``compiler.FACE_ORDER`` for boxes and the periodic lattice integer ``k``
    for wires.  The arrays may be NumPy arrays or CUDA Torch tensors.
    """

    distance: Any
    kind: Any
    index: Any
    primitive_index: Any
    outward_normal: Any
    surface_normal: Any
    surface_index: Any
    material_inner_index: Any
    material_outer_index: Any
    material_from_index: Any
    material_to_index: Any
    inside_to_outside: Any

    @property
    def hit(self):
        return self.kind != int(GeometryKind.MISS)

    @property
    def primitive(self):
        """Short alias useful in queue code."""

        return self.primitive_index


def _rays_numpy(
    origins: Any, directions: Any, tmax: Union[float, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origins = np.ascontiguousarray(origins, dtype=np.float32)
    directions = np.ascontiguousarray(directions, dtype=np.float32)
    if origins.ndim != 2 or origins.shape[1] != 3:
        raise ValueError("origins must have shape [N,3]")
    if directions.shape != origins.shape:
        raise ValueError("directions must have the same [N,3] shape as origins")
    if not np.all(np.isfinite(origins)) or not np.all(np.isfinite(directions)):
        raise ValueError("origins and directions must be finite")
    if np.any(np.linalg.norm(directions.astype(np.float64), axis=1) == 0.0):
        raise ValueError("ray directions must be non-zero")

    cap = np.asarray(tmax, dtype=np.float32)
    if cap.ndim == 0:
        cap = np.full(origins.shape[0], cap, dtype=np.float32)
    else:
        try:
            cap = np.broadcast_to(cap, (origins.shape[0],)).copy()
        except ValueError as exc:
            raise ValueError("tmax must be scalar or broadcastable to [N]") from exc
    if np.any(np.isnan(cap)) or np.any(cap < 0.0):
        raise ValueError("tmax must be non-negative and not NaN")
    return origins, directions, cap


def _empty_numpy(cap: np.ndarray) -> BoundaryIntersections:
    n = cap.size
    minus_one = np.full(n, -1, dtype=np.int32)
    normals = np.zeros((n, 3), dtype=np.float32)
    return BoundaryIntersections(
        distance=cap.copy(),
        kind=np.zeros(n, dtype=np.int8),
        index=minus_one.copy(),
        primitive_index=minus_one.copy(),
        outward_normal=normals.copy(),
        surface_normal=normals.copy(),
        surface_index=minus_one.copy(),
        material_inner_index=minus_one.copy(),
        material_outer_index=minus_one.copy(),
        material_from_index=minus_one.copy(),
        material_to_index=minus_one.copy(),
        inside_to_outside=np.zeros(n, dtype=np.bool_),
    )


def _orient_numpy(
    result: BoundaryIntersections, directions: np.ndarray
) -> BoundaryIntersections:
    hit = result.kind != int(GeometryKind.MISS)
    # Raw normals are float32 in fill_state, so retain float32 dot semantics.
    dot_raw = np.sum(
        result.outward_normal * np.negative(directions, dtype=np.float32),
        axis=1,
        dtype=np.float32,
    )
    outside_now = hit & (dot_raw > np.float32(0.0))
    inside_to_outside = hit & ~outside_now
    surface_normal = np.where(
        outside_now[:, None], result.outward_normal, -result.outward_normal
    ).astype(np.float32, copy=False)
    surface_normal[~hit] = 0.0
    material_from = np.where(
        outside_now, result.material_outer_index, result.material_inner_index
    ).astype(np.int32, copy=False)
    material_to = np.where(
        outside_now, result.material_inner_index, result.material_outer_index
    ).astype(np.int32, copy=False)
    material_from[~hit] = -1
    material_to[~hit] = -1
    return BoundaryIntersections(
        distance=result.distance,
        kind=result.kind,
        index=result.index,
        primitive_index=result.primitive_index,
        outward_normal=result.outward_normal,
        surface_normal=surface_normal,
        surface_index=result.surface_index,
        material_inner_index=result.material_inner_index,
        material_outer_index=result.material_outer_index,
        material_from_index=material_from,
        material_to_index=material_to,
        inside_to_outside=inside_to_outside,
    )


def intersect_boxes_numpy(
    boxes: "AnalyticBoxes",
    origins: Any,
    directions: Any,
    tmax: Union[float, np.ndarray] = np.inf,
) -> BoundaryIntersections:
    """Intersect enabled macro-box faces using an exclusive ``tmax``.

    A hit must be farther than Chroma's triangle epsilon (``1e-6 mm``).  Face
    edges are closed; ties retain the first face in ``FACE_ORDER`` and the
    first box in compiler order.
    """

    origins, directions, cap = _rays_numpy(origins, directions, tmax)
    result = _empty_numpy(cap)

    for box_index in range(boxes.count):
        if not bool(boxes.collision_enabled[box_index]):
            continue
        lo = boxes.bounds_min[box_index]
        hi = boxes.bounds_max[box_index]
        for face in range(6):
            if not bool(boxes.reachable_face_mask[box_index, face]):
                continue
            axis = face // 2
            high_face = (face & 1) != 0
            coordinate = hi[axis] if high_face else lo[axis]
            denominator = directions[:, axis]
            nonparallel = denominator != np.float32(0.0)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                distance = np.asarray(
                    (np.float32(coordinate) - origins[:, axis]) / denominator,
                    dtype=np.float32,
                )

            with np.errstate(invalid="ignore", over="ignore"):
                hit_point = origins + distance[:, None] * directions
            other0 = (axis + 1) % 3
            other1 = (axis + 2) % 3
            within = (
                (hit_point[:, other0] >= lo[other0])
                & (hit_point[:, other0] <= hi[other0])
                & (hit_point[:, other1] >= lo[other1])
                & (hit_point[:, other1] <= hi[other1])
            )
            take = (
                nonparallel
                & within
                & (distance > CHROMA_EPSILON)
                & (distance < result.distance)
            )
            if not np.any(take):
                continue

            result.distance[take] = distance[take]
            result.kind[take] = int(GeometryKind.BOX)
            result.index[take] = box_index
            result.primitive_index[take] = face
            result.outward_normal[take] = 0.0
            result.outward_normal[take, axis] = 1.0 if high_face else -1.0
            result.surface_index[take] = boxes.surface_index[box_index]
            result.material_inner_index[take] = boxes.material_inside_index[box_index]
            result.material_outer_index[take] = boxes.material_outside_index[box_index]

    return _orient_numpy(result, directions)


def _wire_intersections_numpy(
    wires: "AnalyticWires",
    origins: np.ndarray,
    directions: np.ndarray,
    cap: np.ndarray,
    *,
    conservative_cull: bool,
) -> BoundaryIntersections:
    result = _empty_numpy(cap)

    for ray_index in range(origins.shape[0]):
        # The CUDA reference forms float3 w first, then promotes components.
        origin32 = origins[ray_index]
        direction32 = directions[ray_index]
        analytic_distance = np.float32(np.inf)
        chosen = None

        for plane_index in range(wires.count):
            wire_origin32 = wires.origin[plane_index].astype(np.float32)
            w = np.asarray(origin32 - wire_origin32, dtype=np.float32).astype(np.float64)
            direction = direction32.astype(np.float64)
            u = wires.u[plane_index]
            v = wires.v[plane_index]
            normal = wires.n[plane_index]

            du = float(np.dot(direction, u))
            dv = float(np.dot(direction, v))
            dn = float(np.dot(direction, normal))
            wu = float(np.dot(w, u))
            wv0 = float(np.dot(w, v) - wires.v0[plane_index])
            wn0 = float(np.dot(w, normal))

            t_in = -1.0e300
            t_out = 1.0e300
            if abs(du) < WIRE_PARALLEL_U:
                if wu < wires.umin[plane_index] or wu > wires.umax[plane_index]:
                    continue
            else:
                t1 = (wires.umin[plane_index] - wu) / du
                t2 = (wires.umax[plane_index] - wu) / du
                if t1 > t2:
                    t1, t2 = t2, t1
                t_in = max(t_in, t1)
                t_out = min(t_out, t2)
                if t_in > t_out:
                    continue

            k_start = int(wires.kmin[plane_index])
            k_stop = int(wires.kmax[plane_index])
            if conservative_cull:
                t_lo = max(t_in, WIRE_T_MIN)
                t_hi = min(t_out, float(cap[ray_index]))
                if abs(dn) > WIRE_PARALLEL_PLANE:
                    tn1 = (-wires.pad_n[plane_index] - wn0) / dn
                    tn2 = (wires.pad_n[plane_index] - wn0) / dn
                    if tn1 > tn2:
                        tn1, tn2 = tn2, tn1
                    t_lo = max(t_lo, tn1)
                    t_hi = min(t_hi, tn2)
                elif abs(wn0) > wires.pad_n[plane_index]:
                    continue
                if t_hi < t_lo:
                    continue
                if abs(dn) <= WIRE_PARALLEL_PLANE and abs(dv) > WIRE_PARALLEL_PLANE:
                    t_span = (
                        wires.pitch[plane_index] + wires.diameter[plane_index]
                    ) / abs(dv)
                    t_hi = min(t_hi, t_lo + t_span)

                v_entry = wv0 + dv * t_lo
                v_exit = wv0 + dv * t_hi
                v_lo = min(v_entry, v_exit) - wires.pad_v[plane_index]
                v_hi = max(v_entry, v_exit) + wires.pad_v[plane_index]
                v_lo = min(v_lo, wv0 - wires.pad_v[plane_index])
                v_hi = max(v_hi, wv0 + wires.pad_v[plane_index])
                k_start = max(
                    k_start,
                    int(np.floor(v_lo * wires.inv_pitch[plane_index])),
                )
                k_stop = min(
                    k_stop,
                    int(np.ceil(v_hi * wires.inv_pitch[plane_index])),
                )
                if k_start > k_stop:
                    continue

            a = dv * dv + dn * dn
            if a == 0.0:
                # CUDA's off-axis lattice candidates obtain non-finite roots
                # and poison the analytic incumbent after any on-cylinder
                # t=1e-4 candidate.  The final fill_state mesh comparison then
                # rejects that NaN analytic distance, so the observable result
                # is no wire hit.
                continue
            radius2 = wires.radius2[plane_index]
            eps0 = max(1.0e-18, 1.0e-12 * radius2)
            for lattice_index in range(k_start, k_stop + 1):
                wv = wv0 - lattice_index * wires.pitch[plane_index]
                b = wv * dv + wn0 * dn
                c = wv * wv + wn0 * wn0 - radius2
                discriminant = b * b - a * c
                if discriminant < 0.0:
                    continue
                square_root = np.sqrt(discriminant)
                t_small = (-b - square_root) / a
                t_large = (-b + square_root) / a
                radius2_at_origin = wv * wv + wn0 * wn0
                if radius2_at_origin > radius2 + eps0:
                    if t_small <= WIRE_T_MIN:
                        continue
                    distance64 = t_small
                elif radius2_at_origin < radius2 - eps0:
                    if t_large <= WIRE_T_MIN:
                        continue
                    distance64 = t_large
                else:
                    distance64 = WIRE_T_MIN

                uc = wu + du * distance64
                if uc < wires.umin[plane_index] or uc > wires.umax[plane_index]:
                    continue
                if distance64 < t_in or distance64 > t_out:
                    continue
                distance32 = np.float32(distance64)
                if not distance32 < cap[ray_index]:
                    continue
                if distance32 >= analytic_distance:
                    continue

                vn_hit = wv + dv * distance64
                nn_hit = wn0 + dn * distance64
                length = np.sqrt(vn_hit * vn_hit + nn_hit * nn_hit)
                if length <= 0.0:
                    continue
                outward = np.asarray(
                    (vn_hit / length) * v + (nn_hit / length) * normal,
                    dtype=np.float32,
                )
                analytic_distance = distance32
                chosen = (plane_index, lattice_index, outward)

        if chosen is None:
            continue
        plane_index, lattice_index, outward = chosen
        # fill_state only selects an analytic hit when its surface is valid.
        if wires.surface_index[plane_index] < 0:
            continue
        result.distance[ray_index] = analytic_distance
        result.kind[ray_index] = int(GeometryKind.WIRE)
        result.index[ray_index] = plane_index
        result.primitive_index[ray_index] = lattice_index
        result.outward_normal[ray_index] = outward
        result.surface_index[ray_index] = wires.surface_index[plane_index]
        result.material_inner_index[ray_index] = wires.material_inner_index[plane_index]
        result.material_outer_index[ray_index] = wires.material_outer_index[plane_index]

    return _orient_numpy(result, directions)


def intersect_wires_numpy(
    wires: "AnalyticWires",
    origins: Any,
    directions: Any,
    tmax: Union[float, np.ndarray] = np.inf,
) -> BoundaryIntersections:
    """Intersect finite periodic cylinders with CUDA's conservative k cull."""

    origins, directions, cap = _rays_numpy(origins, directions, tmax)
    return _wire_intersections_numpy(
        wires, origins, directions, cap, conservative_cull=True
    )


def intersect_wires_bruteforce_numpy(
    wires: "AnalyticWires",
    origins: Any,
    directions: Any,
    tmax: Union[float, np.ndarray] = np.inf,
) -> BoundaryIntersections:
    """Slow validation oracle which enumerates every finite wire cylinder."""

    origins, directions, cap = _rays_numpy(origins, directions, tmax)
    return _wire_intersections_numpy(
        wires, origins, directions, cap, conservative_cull=False
    )


def intersect_scene_numpy(
    scene: "CompiledReflect3WiresScene",
    origins: Any,
    directions: Any,
    tmax: Union[float, np.ndarray] = np.inf,
) -> BoundaryIntersections:
    """Return the nearest enabled box or wire, strictly before ``tmax``."""

    origins, directions, cap = _rays_numpy(origins, directions, tmax)
    boxes = intersect_boxes_numpy(scene.boxes, origins, directions, cap)
    # Passing the box incumbent both accelerates the wire lattice cull and
    # implements Chroma's strict analytic-vs-mesh tie rule.
    wires = intersect_wires_numpy(
        scene.wires, origins, directions, tmax=boxes.distance
    )
    take = wires.hit
    if not np.any(take):
        return boxes

    fields = {}
    for name in BoundaryIntersections.__dataclass_fields__:
        box_value = getattr(boxes, name)
        wire_value = getattr(wires, name)
        selector = take[:, None] if getattr(box_value, "ndim", 1) == 2 else take
        fields[name] = np.where(selector, wire_value, box_value)
    return BoundaryIntersections(**fields)


@dataclass(frozen=True)
class PreparedAnalyticScene:
    """CUDA-resident scene arrays reusable across boundary queue launches."""

    tensors: dict[str, Any]
    device: Any
    box_count: int
    wire_count: int
    wire_normals_are_x_aligned: bool


@dataclass(frozen=True)
class SplitIntersectionWorkspace:
    """Reusable scratch and optional outputs for the split analytic query.

    :meth:`outputs` returns views owned by this workspace.  Those views are
    deliberately opt-in at the query API: a later launch into the same views
    overwrites an earlier result.  Callers which need independent result
    lifetimes can continue to omit ``out`` and retain the allocating behavior.
    """

    candidate_ids: Any
    candidate_count: Any
    distance: Any
    kind: Any
    index: Any
    primitive_index: Any
    outward_normal: Any
    surface_normal: Any
    surface_index: Any
    material_inner_index: Any
    material_outer_index: Any
    material_from_index: Any
    material_to_index: Any
    inside_to_outside: Any

    @property
    def capacity(self) -> int:
        return int(self.candidate_ids.numel())

    def outputs(self, count: int) -> BoundaryIntersections:
        """Return a live prefix of the preallocated result buffers."""

        count = int(count)
        if count < 0 or count > self.capacity:
            raise ValueError(
                "analytic output count must be between zero and workspace capacity"
            )
        return BoundaryIntersections(
            distance=self.distance[:count],
            kind=self.kind[:count],
            index=self.index[:count],
            primitive_index=self.primitive_index[:count],
            outward_normal=self.outward_normal[:count],
            surface_normal=self.surface_normal[:count],
            surface_index=self.surface_index[:count],
            material_inner_index=self.material_inner_index[:count],
            material_outer_index=self.material_outer_index[:count],
            material_from_index=self.material_from_index[:count],
            material_to_index=self.material_to_index[:count],
            inside_to_outside=self.inside_to_outside[:count],
        )


try:  # optional dependency; CPU scene compilation must remain lightweight
    import triton  # type: ignore
    import triton.language as tl  # type: ignore
except ImportError:  # pragma: no cover - exercised in Chroma-only environments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _analytic_boundary_kernel(
        origin_ptr,
        direction_ptr,
        tmax_ptr,
        ray_index_ptr,
        n_rays_ptr,
        previous_instance_ptr,
        previous_triangle_ptr,
        box_lo_ptr,
        box_hi_ptr,
        box_triangle_ptr,
        box_triangle_face_ptr,
        box_face_mask_ptr,
        box_collision_ptr,
        box_surface_ptr,
        box_mat_inner_ptr,
        box_mat_outer_ptr,
        wire_origin_ptr,
        wire_raw_u_ptr,
        wire_raw_v_ptr,
        wire_u_ptr,
        wire_v_ptr,
        wire_n_ptr,
        wire_pitch_ptr,
        wire_inv_pitch_ptr,
        wire_radius2_ptr,
        wire_diameter_ptr,
        wire_pad_v_ptr,
        wire_pad_n_ptr,
        wire_umin_ptr,
        wire_umax_ptr,
        wire_v0_ptr,
        wire_kmin_ptr,
        wire_kmax_ptr,
        wire_surface_ptr,
        wire_mat_inner_ptr,
        wire_mat_outer_ptr,
        out_distance_ptr,
        out_kind_ptr,
        out_index_ptr,
        out_primitive_ptr,
        out_outward_ptr,
        out_surface_normal_ptr,
        out_surface_ptr,
        out_mat_inner_ptr,
        out_mat_outer_ptr,
        out_mat_from_ptr,
        out_mat_to_ptr,
        out_inside_to_outside_ptr,
        n_rays,
        scalar_tmax,
        TMAX_IS_POINTER: tl.constexpr,
        INDIRECT: tl.constexpr,
        LOAD_INCUMBENT: tl.constexpr,
        N_RAYS_IS_POINTER: tl.constexpr,
        SUPPRESS_PREVIOUS_BOX: tl.constexpr,
        CHROMA_MESH_BOXES: tl.constexpr,
        CHROMA_WIRE_FRAME: tl.constexpr,
        CHROMA_WIRE_FULL_SCAN: tl.constexpr,
        NBOX: tl.constexpr,
        NWIRE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_start = tl.program_id(0) * BLOCK_SIZE
        if N_RAYS_IS_POINTER:
            active_count = tl.load(n_rays_ptr).to(tl.int32)
            active_count = tl.maximum(0, tl.minimum(active_count, n_rays))
            # Device-scheduled queries deliberately launch the allocation
            # capacity.  Retire a wholly inactive CTA before the statically
            # unrolled box/wire work; a lane mask alone would still execute
            # that expensive instruction stream for the empty suffix.
            if program_start >= active_count:
                return
        else:
            active_count = n_rays
        lane = program_start + tl.arange(0, BLOCK_SIZE)
        mask = lane < active_count
        if INDIRECT:
            ray_index = tl.load(ray_index_ptr + lane, mask=mask, other=0).to(
                tl.int64
            )
        else:
            ray_index = lane.to(tl.int64)
        base = ray_index * 3
        ox = tl.load(origin_ptr + base, mask=mask, other=0.0).to(tl.float32)
        oy = tl.load(origin_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        oz = tl.load(origin_ptr + base + 2, mask=mask, other=0.0).to(tl.float32)
        dx = tl.load(direction_ptr + base, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(direction_ptr + base + 1, mask=mask, other=0.0).to(tl.float32)
        dz = tl.load(direction_ptr + base + 2, mask=mask, other=0.0).to(tl.float32)
        if SUPPRESS_PREVIOUS_BOX:
            previous_instance = tl.load(
                previous_instance_ptr + ray_index, mask=mask, other=-1
            ).to(tl.int32)
            previous_triangle = tl.load(
                previous_triangle_ptr + ray_index, mask=mask, other=-1
            ).to(tl.int32)
            # PMTs retain non-negative instance IDs.  A macro box is encoded
            # as -(box_index + 2), leaving -1 as the ordinary no-previous-hit
            # sentinel and preserving PMT candidate suppression unchanged.
            previous_box = -previous_instance - 2
        else:
            previous_box = tl.full((BLOCK_SIZE,), -1, tl.int32)
            previous_triangle = tl.full((BLOCK_SIZE,), -1, tl.int32)
        if LOAD_INCUMBENT:
            best = tl.load(
                out_distance_ptr + ray_index, mask=mask, other=float("inf")
            ).to(tl.float32)
        elif TMAX_IS_POINTER:
            best = tl.load(
                tmax_ptr + ray_index, mask=mask, other=float("inf")
            ).to(tl.float32)
        else:
            best = tl.full((BLOCK_SIZE,), scalar_tmax, tl.float32)

        if LOAD_INCUMBENT:
            kind = tl.load(out_kind_ptr + ray_index, mask=mask, other=0).to(tl.int32)
            geometry_index = tl.load(
                out_index_ptr + ray_index, mask=mask, other=-1
            ).to(tl.int32)
            primitive_index = tl.load(
                out_primitive_ptr + ray_index, mask=mask, other=-1
            ).to(tl.int32)
            outward_x = tl.load(out_outward_ptr + base, mask=mask, other=0.0).to(
                tl.float32
            )
            outward_y = tl.load(
                out_outward_ptr + base + 1, mask=mask, other=0.0
            ).to(tl.float32)
            outward_z = tl.load(
                out_outward_ptr + base + 2, mask=mask, other=0.0
            ).to(tl.float32)
            surface = tl.load(
                out_surface_ptr + ray_index, mask=mask, other=-1
            ).to(tl.int32)
            mat_inner = tl.load(
                out_mat_inner_ptr + ray_index, mask=mask, other=-1
            ).to(tl.int32)
            mat_outer = tl.load(
                out_mat_outer_ptr + ray_index, mask=mask, other=-1
            ).to(tl.int32)
        else:
            kind = tl.zeros((BLOCK_SIZE,), tl.int32)
            geometry_index = tl.full((BLOCK_SIZE,), -1, tl.int32)
            primitive_index = tl.full((BLOCK_SIZE,), -1, tl.int32)
            outward_x = tl.zeros((BLOCK_SIZE,), tl.float32)
            outward_y = tl.zeros((BLOCK_SIZE,), tl.float32)
            outward_z = tl.zeros((BLOCK_SIZE,), tl.float32)
            surface = tl.full((BLOCK_SIZE,), -1, tl.int32)
            mat_inner = tl.full((BLOCK_SIZE,), -1, tl.int32)
            mat_outer = tl.full((BLOCK_SIZE,), -1, tl.int32)

        # Preserve the incoming incumbent so compatibility replay can replace
        # the production slab result with Chroma's original triangle math.
        incumbent_best = best
        incumbent_kind = kind
        incumbent_geometry_index = geometry_index
        incumbent_primitive_index = primitive_index
        incumbent_outward_x = outward_x
        incumbent_outward_y = outward_y
        incumbent_outward_z = outward_z
        incumbent_surface = surface
        incumbent_mat_inner = mat_inner
        incumbent_mat_outer = mat_outer

        # Macro faces.  Production uses one slab division per face.
        for box_index in tl.static_range(0, NBOX):
            box_enabled = tl.load(box_collision_ptr + box_index).to(tl.int1)
            box_surface = tl.load(box_surface_ptr + box_index).to(tl.int32)
            box_inner = tl.load(box_mat_inner_ptr + box_index).to(tl.int32)
            box_outer = tl.load(box_mat_outer_ptr + box_index).to(tl.int32)
            for face in tl.static_range(0, 6):
                face_enabled = tl.load(
                    box_face_mask_ptr + box_index * 6 + face
                ).to(tl.int1)
                if face == 0 or face == 1:
                    denominator = dx
                    origin_axis = ox
                    coordinate = tl.load(
                        (box_hi_ptr if face == 1 else box_lo_ptr)
                        + box_index * 3
                    ).to(tl.float32)
                    point_other0_a = oy
                    point_other0_d = dy
                    point_other1_a = oz
                    point_other1_d = dz
                    lo_other0 = tl.load(box_lo_ptr + box_index * 3 + 1)
                    hi_other0 = tl.load(box_hi_ptr + box_index * 3 + 1)
                    lo_other1 = tl.load(box_lo_ptr + box_index * 3 + 2)
                    hi_other1 = tl.load(box_hi_ptr + box_index * 3 + 2)
                    normal_x = -1.0 if face == 0 else 1.0
                    normal_y = 0.0
                    normal_z = 0.0
                elif face == 2 or face == 3:
                    denominator = dy
                    origin_axis = oy
                    coordinate = tl.load(
                        (box_hi_ptr if face == 3 else box_lo_ptr)
                        + box_index * 3
                        + 1
                    ).to(tl.float32)
                    point_other0_a = oz
                    point_other0_d = dz
                    point_other1_a = ox
                    point_other1_d = dx
                    lo_other0 = tl.load(box_lo_ptr + box_index * 3 + 2)
                    hi_other0 = tl.load(box_hi_ptr + box_index * 3 + 2)
                    lo_other1 = tl.load(box_lo_ptr + box_index * 3)
                    hi_other1 = tl.load(box_hi_ptr + box_index * 3)
                    normal_x = 0.0
                    normal_y = -1.0 if face == 2 else 1.0
                    normal_z = 0.0
                else:
                    denominator = dz
                    origin_axis = oz
                    coordinate = tl.load(
                        (box_hi_ptr if face == 5 else box_lo_ptr)
                        + box_index * 3
                        + 2
                    ).to(tl.float32)
                    point_other0_a = ox
                    point_other0_d = dx
                    point_other1_a = oy
                    point_other1_d = dy
                    lo_other0 = tl.load(box_lo_ptr + box_index * 3)
                    hi_other0 = tl.load(box_hi_ptr + box_index * 3)
                    lo_other1 = tl.load(box_lo_ptr + box_index * 3 + 1)
                    hi_other1 = tl.load(box_hi_ptr + box_index * 3 + 1)
                    normal_x = 0.0
                    normal_y = 0.0
                    normal_z = -1.0 if face == 4 else 1.0

                box_candidate = (coordinate - origin_axis) / denominator
                other0 = point_other0_a + box_candidate * point_other0_d
                other1 = point_other1_a + box_candidate * point_other1_d
                take = (
                    mask
                    & box_enabled
                    & face_enabled
                    & ((previous_box != box_index) | (previous_triangle != face))
                    & (denominator != 0.0)
                    & (box_candidate > 1.0e-6)
                    & (box_candidate < best)
                    & (other0 >= lo_other0)
                    & (other0 <= hi_other0)
                    & (other1 >= lo_other1)
                    & (other1 <= hi_other1)
                )
                best = tl.where(take, box_candidate, best)
                kind = tl.where(take, 1, kind)
                geometry_index = tl.where(take, box_index, geometry_index)
                primitive_index = tl.where(take, face, primitive_index)
                outward_x = tl.where(take, normal_x, outward_x)
                outward_y = tl.where(take, normal_y, outward_y)
                outward_z = tl.where(take, normal_z, outward_z)
                surface = tl.where(take, box_surface, surface)
                mat_inner = tl.where(take, box_inner, mat_inner)
                mat_outer = tl.where(take, box_outer, mat_outer)

        if CHROMA_MESH_BOXES:
            # Lockstep replay deliberately discards the slab candidates and
            # intersects the exact sixteen float32 triangles that Geometry
            # flattening supplied to Chroma.  This preserves the otherwise
            # observable rounding of Moller-Trumbore at large coordinates.
            best = incumbent_best
            kind = incumbent_kind
            geometry_index = incumbent_geometry_index
            primitive_index = incumbent_primitive_index
            outward_x = incumbent_outward_x
            outward_y = incumbent_outward_y
            outward_z = incumbent_outward_z
            surface = incumbent_surface
            mat_inner = incumbent_mat_inner
            mat_outer = incumbent_mat_outer
            for box_index in tl.static_range(0, NBOX):
                box_enabled = tl.load(
                    box_collision_ptr + box_index
                ).to(tl.int1)
                box_surface = tl.load(
                    box_surface_ptr + box_index
                ).to(tl.int32)
                box_inner = tl.load(
                    box_mat_inner_ptr + box_index
                ).to(tl.int32)
                box_outer = tl.load(
                    box_mat_outer_ptr + box_index
                ).to(tl.int32)
                for local_triangle in tl.static_range(0, 16):
                    face = tl.load(
                        box_triangle_face_ptr + box_index * 16 + local_triangle
                    ).to(tl.int32)
                    face_enabled = tl.load(
                        box_face_mask_ptr + box_index * 6 + face
                    ).to(tl.int1)
                    triangle_base = (box_index * 16 + local_triangle) * 9
                    v0x = tl.load(box_triangle_ptr + triangle_base)
                    v0y = tl.load(box_triangle_ptr + triangle_base + 1)
                    v0z = tl.load(box_triangle_ptr + triangle_base + 2)
                    v1x = tl.load(box_triangle_ptr + triangle_base + 3)
                    v1y = tl.load(box_triangle_ptr + triangle_base + 4)
                    v1z = tl.load(box_triangle_ptr + triangle_base + 5)
                    v2x = tl.load(box_triangle_ptr + triangle_base + 6)
                    v2y = tl.load(box_triangle_ptr + triangle_base + 7)
                    v2z = tl.load(box_triangle_ptr + triangle_base + 8)

                    e1x = v1x - v0x
                    e1y = v1y - v0y
                    e1z = v1z - v0z
                    e2x = v2x - v0x
                    e2y = v2y - v0y
                    e2z = v2z - v0z
                    # CUDA's cross helper emits two rounded multiplies and a
                    # subtraction; contracting either product changes the
                    # boundary distance at detector-scale coordinates.
                    hx = dy * e2z - dz * e2y
                    hy = dz * e2x - dx * e2z
                    hz = dx * e2y - dy * e2x
                    determinant = tl.fma(
                        e1z, hz, tl.fma(e1x, hx, e1y * hy)
                    )
                    determinant_ok = (
                        (determinant < -1.1920928955078125e-7)
                        | (determinant > 1.1920928955078125e-7)
                    )
                    reciprocal = 1.0 / determinant
                    sx = ox - v0x
                    sy = oy - v0y
                    sz = oz - v0z
                    dot_sh = tl.fma(
                        sz, hz, tl.fma(sx, hx, sy * hy)
                    )
                    barycentric_u = reciprocal * dot_sh
                    qx_triangle = sy * e1z - sz * e1y
                    qy_triangle = sz * e1x - sx * e1z
                    qz_triangle = sx * e1y - sy * e1x
                    dot_dq = tl.fma(
                        dz,
                        qz_triangle,
                        tl.fma(dx, qx_triangle, dy * qy_triangle),
                    )
                    barycentric_v = reciprocal * dot_dq
                    dot_e2q = tl.fma(
                        e2z,
                        qz_triangle,
                        tl.fma(e2x, qx_triangle, e2y * qy_triangle),
                    )
                    box_candidate = reciprocal * dot_e2q
                    take = (
                        mask
                        & box_enabled
                        & face_enabled
                        & (
                            (previous_box != box_index)
                            | (previous_triangle != local_triangle)
                        )
                        & determinant_ok
                        & (barycentric_u >= -1.0e-6)
                        & (barycentric_u <= 1.0 + 1.0e-6)
                        & (barycentric_v >= -1.0e-6)
                        & ((barycentric_u + barycentric_v) <= 1.0 + 1.0e-6)
                        & (box_candidate > 1.0e-6)
                        & (box_candidate < best)
                    )

                    # Chroma computes the mesh normal from (v1-v0) x
                    # (v2-v1), not the edge2 used by the intersection.
                    e12x = v2x - v1x
                    e12y = v2y - v1y
                    e12z = v2z - v1z
                    raw_nx = e1y * e12z - e1z * e12y
                    raw_ny = e1z * e12x - e1x * e12z
                    raw_nz = e1x * e12y - e1y * e12x
                    normal_squared = tl.fma(
                        raw_nz,
                        raw_nz,
                        tl.fma(raw_nx, raw_nx, raw_ny * raw_ny),
                    )
                    normal_length = tl.sqrt(normal_squared)
                    candidate_nx = raw_nx / normal_length
                    candidate_ny = raw_ny / normal_length
                    candidate_nz = raw_nz / normal_length

                    best = tl.where(take, box_candidate, best)
                    kind = tl.where(take, 1, kind)
                    geometry_index = tl.where(take, box_index, geometry_index)
                    primitive_index = tl.where(
                        take, local_triangle, primitive_index
                    )
                    outward_x = tl.where(take, candidate_nx, outward_x)
                    outward_y = tl.where(take, candidate_ny, outward_y)
                    outward_z = tl.where(take, candidate_nz, outward_z)
                    surface = tl.where(take, box_surface, surface)
                    mat_inner = tl.where(take, box_inner, mat_inner)
                    mat_outer = tl.where(take, box_outer, mat_outer)

        # Periodic cylinders.  Frame projections and root solve are FP64.
        wire_best = tl.full((BLOCK_SIZE,), float("inf"), tl.float32)
        wire_index_best = tl.full((BLOCK_SIZE,), -1, tl.int32)
        wire_k_best = tl.full((BLOCK_SIZE,), -1, tl.int32)
        wire_out_x = tl.zeros((BLOCK_SIZE,), tl.float32)
        wire_out_y = tl.zeros((BLOCK_SIZE,), tl.float32)
        wire_out_z = tl.zeros((BLOCK_SIZE,), tl.float32)
        wire_surface_best = tl.full((BLOCK_SIZE,), -1, tl.int32)
        wire_inner_best = tl.full((BLOCK_SIZE,), -1, tl.int32)
        wire_outer_best = tl.full((BLOCK_SIZE,), -1, tl.int32)

        for plane_index in tl.static_range(0, NWIRE):
            array3 = plane_index * 3
            # Reproduce float3 subtraction before promotion to FP64.
            wix32 = tl.load(wire_origin_ptr + array3).to(tl.float32)
            wiy32 = tl.load(wire_origin_ptr + array3 + 1).to(tl.float32)
            wiz32 = tl.load(wire_origin_ptr + array3 + 2).to(tl.float32)
            wx = (ox - wix32).to(tl.float64)
            wy = (oy - wiy32).to(tl.float64)
            wz = (oz - wiz32).to(tl.float64)
            dx64 = dx.to(tl.float64)
            dy64 = dy.to(tl.float64)
            dz64 = dz.to(tl.float64)
            if CHROMA_WIRE_FRAME:
                # fill_state re-orthonormalizes each serialized float3 in
                # FP64.  Spell out nvcc's exact mul/FMA tree; NumPy's host
                # precomputation differs by a few double ulps for angled
                # planes and that can survive the final float32 normal cast.
                raw_ux = tl.load(wire_raw_u_ptr + array3).to(tl.float64)
                raw_uy = tl.load(wire_raw_u_ptr + array3 + 1).to(tl.float64)
                raw_uz = tl.load(wire_raw_u_ptr + array3 + 2).to(tl.float64)
                raw_vx = tl.load(wire_raw_v_ptr + array3).to(tl.float64)
                raw_vy = tl.load(wire_raw_v_ptr + array3 + 1).to(tl.float64)
                raw_vz = tl.load(wire_raw_v_ptr + array3 + 2).to(tl.float64)
                u_norm2 = tl.fma(
                    raw_uz,
                    raw_uz,
                    tl.fma(raw_ux, raw_ux, raw_uy * raw_uy),
                )
                u_inverse = 1.0 / tl.sqrt(u_norm2)
                ux = raw_ux * u_inverse
                uy = raw_uy * u_inverse
                uz = raw_uz * u_inverse
                v_dot_u = tl.fma(
                    uz,
                    raw_vz,
                    tl.fma(ux, raw_vx, uy * raw_vy),
                )
                v1x = raw_vx - ux * v_dot_u
                v1y = raw_vy - uy * v_dot_u
                v1z = raw_vz - uz * v_dot_u
                v_norm2 = tl.fma(
                    v1z, v1z, tl.fma(v1x, v1x, v1y * v1y)
                )
                v_inverse = 1.0 / tl.sqrt(v_norm2)
                vx = v1x * v_inverse
                vy = v1y * v_inverse
                vz = v1z * v_inverse
                nx = uy * vz - uz * vy
                ny = uz * vx - ux * vz
                nz = ux * vy - uy * vx
            else:
                ux = tl.load(wire_u_ptr + array3).to(tl.float64)
                uy = tl.load(wire_u_ptr + array3 + 1).to(tl.float64)
                uz = tl.load(wire_u_ptr + array3 + 2).to(tl.float64)
                vx = tl.load(wire_v_ptr + array3).to(tl.float64)
                vy = tl.load(wire_v_ptr + array3 + 1).to(tl.float64)
                vz = tl.load(wire_v_ptr + array3 + 2).to(tl.float64)
                nx = tl.load(wire_n_ptr + array3).to(tl.float64)
                ny = tl.load(wire_n_ptr + array3 + 1).to(tl.float64)
                nz = tl.load(wire_n_ptr + array3 + 2).to(tl.float64)
            du = dx64 * ux + dy64 * uy + dz64 * uz
            dv = dx64 * vx + dy64 * vy + dz64 * vz
            dn = dx64 * nx + dy64 * ny + dz64 * nz
            wu = wx * ux + wy * uy + wz * uz
            wv0 = (
                wx * vx
                + wy * vy
                + wz * vz
                - tl.load(wire_v0_ptr + plane_index).to(tl.float64)
            )
            wn0 = wx * nx + wy * ny + wz * nz
            umin = tl.load(wire_umin_ptr + plane_index).to(tl.float64)
            umax = tl.load(wire_umax_ptr + plane_index).to(tl.float64)

            parallel_u = tl.abs(du) < 1.0e-15
            valid_plane = mask & (~parallel_u | ((wu >= umin) & (wu <= umax)))
            safe_du = tl.where(parallel_u, 1.0, du)
            tu1 = (umin - wu) / safe_du
            tu2 = (umax - wu) / safe_du
            t_in = tl.where(parallel_u, -1.0e300, tl.minimum(tu1, tu2))
            t_out = tl.where(parallel_u, 1.0e300, tl.maximum(tu1, tu2))
            valid_plane &= t_in <= t_out

            pad_n = tl.load(wire_pad_n_ptr + plane_index).to(tl.float64)
            t_lo = tl.maximum(t_in, 1.0e-4)
            t_hi = tl.minimum(t_out, best.to(tl.float64))
            parallel_n = tl.abs(dn) <= 1.0e-12
            safe_dn = tl.where(parallel_n, 1.0, dn)
            tn1 = (-pad_n - wn0) / safe_dn
            tn2 = (pad_n - wn0) / safe_dn
            normal_lo = tl.minimum(tn1, tn2)
            normal_hi = tl.maximum(tn1, tn2)
            t_lo = tl.where(parallel_n, t_lo, tl.maximum(t_lo, normal_lo))
            t_hi = tl.where(parallel_n, t_hi, tl.minimum(t_hi, normal_hi))
            valid_plane &= (~parallel_n) | (tl.abs(wn0) <= pad_n)
            valid_plane &= t_hi >= t_lo

            pitch = tl.load(wire_pitch_ptr + plane_index).to(tl.float64)
            diameter = tl.load(wire_diameter_ptr + plane_index).to(tl.float64)
            parallel_v = tl.abs(dv) <= 1.0e-12
            t_span = (pitch + diameter) / tl.where(parallel_v, 1.0, tl.abs(dv))
            shorten = parallel_n & (~parallel_v)
            t_hi = tl.where(shorten, tl.minimum(t_hi, t_lo + t_span), t_hi)

            pad_v = tl.load(wire_pad_v_ptr + plane_index).to(tl.float64)
            v_entry = wv0 + dv * t_lo
            v_exit = wv0 + dv * t_hi
            v_lo = tl.minimum(v_entry, v_exit) - pad_v
            v_hi = tl.maximum(v_entry, v_exit) + pad_v
            v_lo = tl.minimum(v_lo, wv0 - pad_v)
            v_hi = tl.maximum(v_hi, wv0 + pad_v)
            inv_pitch = tl.load(wire_inv_pitch_ptr + plane_index).to(tl.float64)
            computed_start = tl.floor(v_lo * inv_pitch).to(tl.int64)
            computed_stop = tl.ceil(v_hi * inv_pitch).to(tl.int64)
            finite_start = tl.load(wire_kmin_ptr + plane_index).to(tl.int64)
            finite_stop = tl.load(wire_kmax_ptr + plane_index).to(tl.int64)
            k_lower = tl.maximum(computed_start, finite_start)
            k_upper = tl.minimum(computed_stop, finite_stop)

            a = dv * dv + dn * dn
            radius2 = tl.load(wire_radius2_ptr + plane_index).to(tl.float64)
            # Algebraically, disc(k) = r^2*A -
            # ((wv0-k*pitch)*dn - wn0*dv)^2.  Its feasible lattice integers
            # form one interval.  Intersect with an outward-padded version of
            # that interval, then retain the original B^2-A*C test below as
            # the authority.  Two extra integers on each side make this cull
            # conservative under FP64 rounding, including tangencies.
            if not CHROMA_WIRE_FULL_SCAN:
                stable_disc_interval = tl.abs(dn) > 1.0e-10
                q_disc = wv0 * dn - wn0 * dv
                disc_bound = tl.sqrt(a) * tl.sqrt(radius2)
                disc_denominator = pitch * dn
                safe_disc_denominator = tl.where(
                    stable_disc_interval, disc_denominator, 1.0
                )
                disc_root1 = (q_disc - disc_bound) / safe_disc_denominator
                disc_root2 = (q_disc + disc_bound) / safe_disc_denominator
                disc_lower = (
                    tl.floor(tl.minimum(disc_root1, disc_root2)).to(tl.int64) - 2
                )
                disc_upper = (
                    tl.ceil(tl.maximum(disc_root1, disc_root2)).to(tl.int64) + 2
                )
                k_lower = tl.where(
                    stable_disc_interval, tl.maximum(k_lower, disc_lower), k_lower
                )
                k_upper = tl.where(
                    stable_disc_interval, tl.minimum(k_upper, disc_upper), k_upper
                )
            valid_plane &= k_lower <= k_upper
            # Preserve photon.h's ascending lattice/tie order.  The padded
            # discriminant interval is normally only 5--7 integers wide.
            k = k_lower

            valid_plane &= a != 0.0
            safe_a = tl.where(a == 0.0, 1.0, a)
            eps0 = tl.maximum(1.0e-18, 1.0e-12 * radius2)

            # Candidate intervals are normally 1--3 wires.  The loop remains
            # dynamic so near-coplanar rays keep the exact conservative range.
            loop_active = valid_plane & (k <= k_upper)
            while tl.sum(loop_active.to(tl.int32), axis=0) > 0:
                wv = wv0 - k.to(tl.float64) * pitch
                b = wv * dv + wn0 * dn
                c = wv * wv + wn0 * wn0 - radius2
                discriminant = b * b - a * c
                nonnegative = discriminant >= 0.0
                root = tl.sqrt(tl.maximum(discriminant, 0.0))
                t_small = (-b - root) / safe_a
                t_large = (-b + root) / safe_a
                radius2_origin = wv * wv + wn0 * wn0
                outside = radius2_origin > radius2 + eps0
                inside = radius2_origin < radius2 - eps0
                on_boundary = ~(outside | inside)
                wire_candidate = tl.where(outside, t_small, t_large)
                wire_candidate = tl.where(on_boundary, 1.0e-4, wire_candidate)
                forward = tl.where(
                    outside, t_small > 1.0e-4, tl.where(inside, t_large > 1.0e-4, True)
                )
                uc = wu + du * wire_candidate
                candidate32 = wire_candidate.to(tl.float32)
                root_valid = (
                    loop_active
                    & nonnegative
                    & forward
                    & (uc >= umin)
                    & (uc <= umax)
                    & (wire_candidate >= t_in)
                    & (wire_candidate <= t_out)
                )
                vn_hit = wv + dv * wire_candidate
                nn_hit = wn0 + dn * wire_candidate
                length = tl.sqrt(vn_hit * vn_hit + nn_hit * nn_hit)
                root_valid &= length > 0.0
                candidate_valid = (
                    root_valid
                    & (candidate32 < best)
                    & (candidate32 < wire_best)
                )
                safe_length = tl.where(length > 0.0, length, 1.0)
                candidate_out_x = (
                    (vn_hit / safe_length) * vx + (nn_hit / safe_length) * nx
                ).to(tl.float32)
                candidate_out_y = (
                    (vn_hit / safe_length) * vy + (nn_hit / safe_length) * ny
                ).to(tl.float32)
                candidate_out_z = (
                    (vn_hit / safe_length) * vz + (nn_hit / safe_length) * nz
                ).to(tl.float32)
                wire_best = tl.where(candidate_valid, candidate32, wire_best)
                wire_index_best = tl.where(
                    candidate_valid, plane_index, wire_index_best
                )
                wire_k_best = tl.where(candidate_valid, k.to(tl.int32), wire_k_best)
                wire_out_x = tl.where(candidate_valid, candidate_out_x, wire_out_x)
                wire_out_y = tl.where(candidate_valid, candidate_out_y, wire_out_y)
                wire_out_z = tl.where(candidate_valid, candidate_out_z, wire_out_z)
                plane_surface = tl.load(wire_surface_ptr + plane_index).to(tl.int32)
                plane_inner = tl.load(wire_mat_inner_ptr + plane_index).to(tl.int32)
                plane_outer = tl.load(wire_mat_outer_ptr + plane_index).to(tl.int32)
                wire_surface_best = tl.where(
                    candidate_valid, plane_surface, wire_surface_best
                )
                wire_inner_best = tl.where(
                    candidate_valid, plane_inner, wire_inner_best
                )
                wire_outer_best = tl.where(
                    candidate_valid, plane_outer, wire_outer_best
                )
                k += 1
                loop_active = valid_plane & (k <= k_upper)

        use_wire = (
            mask
            & (wire_surface_best >= 0)
            & (wire_best.to(tl.float64) + 1.0e-12 < best.to(tl.float64))
        )
        best = tl.where(use_wire, wire_best, best)
        kind = tl.where(use_wire, 2, kind)
        geometry_index = tl.where(use_wire, wire_index_best, geometry_index)
        primitive_index = tl.where(use_wire, wire_k_best, primitive_index)
        outward_x = tl.where(use_wire, wire_out_x, outward_x)
        outward_y = tl.where(use_wire, wire_out_y, outward_y)
        outward_z = tl.where(use_wire, wire_out_z, outward_z)
        surface = tl.where(use_wire, wire_surface_best, surface)
        mat_inner = tl.where(use_wire, wire_inner_best, mat_inner)
        mat_outer = tl.where(use_wire, wire_outer_best, mat_outer)

        hit = kind != 0
        dot_raw = outward_x * (-dx) + outward_y * (-dy) + outward_z * (-dz)
        outside_now = hit & (dot_raw > 0.0)
        inside_to_outside = hit & (~outside_now)
        surface_x = tl.where(outside_now, outward_x, -outward_x)
        surface_y = tl.where(outside_now, outward_y, -outward_y)
        surface_z = tl.where(outside_now, outward_z, -outward_z)
        mat_from = tl.where(outside_now, mat_outer, mat_inner)
        mat_to = tl.where(outside_now, mat_inner, mat_outer)
        mat_from = tl.where(hit, mat_from, -1)
        mat_to = tl.where(hit, mat_to, -1)

        tl.store(out_distance_ptr + ray_index, best, mask=mask)
        tl.store(out_kind_ptr + ray_index, kind, mask=mask)
        tl.store(out_index_ptr + ray_index, geometry_index, mask=mask)
        tl.store(out_primitive_ptr + ray_index, primitive_index, mask=mask)
        tl.store(out_outward_ptr + base, outward_x, mask=mask)
        tl.store(out_outward_ptr + base + 1, outward_y, mask=mask)
        tl.store(out_outward_ptr + base + 2, outward_z, mask=mask)
        tl.store(out_surface_normal_ptr + base, surface_x, mask=mask)
        tl.store(out_surface_normal_ptr + base + 1, surface_y, mask=mask)
        tl.store(out_surface_normal_ptr + base + 2, surface_z, mask=mask)
        tl.store(out_surface_ptr + ray_index, surface, mask=mask)
        tl.store(out_mat_inner_ptr + ray_index, mat_inner, mask=mask)
        tl.store(out_mat_outer_ptr + ray_index, mat_outer, mask=mask)
        tl.store(out_mat_from_ptr + ray_index, mat_from, mask=mask)
        tl.store(out_mat_to_ptr + ray_index, mat_to, mask=mask)
        tl.store(
            out_inside_to_outside_ptr + ray_index,
            inside_to_outside.to(tl.int1),
            mask=mask,
        )


    @triton.jit(do_not_specialize=[6])
    def _compact_wire_x_candidates_kernel(
        origin_ptr,
        direction_ptr,
        incumbent_distance_ptr,
        wire_origin_ptr,
        wire_pad_n_ptr,
        candidate_ids_ptr,
        candidate_count_ptr,
        n_rays,
        launch_capacity,
        N_RAYS_IS_POINTER: tl.constexpr,
        NWIRE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compact rays whose finite segment overlaps any wire X slab.

        Target wires have normals parallel to X.  The 2 micron outward guard
        exceeds eight float32 ulps at detector coordinates.  It covers all
        rounding in the float32 broadphase; false positives are resolved by
        the unchanged FP64 cylinder test in ``_analytic_boundary_kernel``.
        """

        program_start = tl.program_id(0) * BLOCK_SIZE
        if N_RAYS_IS_POINTER:
            active_count = tl.load(n_rays).to(tl.int32)
            active_count = tl.maximum(
                0, tl.minimum(active_count, launch_capacity)
            )
            if program_start >= active_count:
                return
        else:
            active_count = n_rays
        local_lane = tl.arange(0, BLOCK_SIZE)
        ray_index = program_start + local_lane
        valid = ray_index < active_count
        base = ray_index * 3
        ox = tl.load(origin_ptr + base, mask=valid, other=0.0).to(tl.float32)
        dx = tl.load(direction_ptr + base, mask=valid, other=0.0).to(tl.float32)
        incumbent = tl.load(
            incumbent_distance_ptr + ray_index, mask=valid, other=-1.0
        ).to(tl.float32)
        candidate = tl.zeros((BLOCK_SIZE,), tl.int1)
        for plane_index in tl.static_range(0, NWIRE):
            plane_x = tl.load(wire_origin_ptr + plane_index * 3).to(tl.float32)
            pad = tl.load(wire_pad_n_ptr + plane_index).to(tl.float32) + 2.0e-3
            slab_lo = plane_x - pad
            slab_hi = plane_x + pad
            parallel = dx == 0.0
            safe_dx = tl.where(parallel, 1.0, dx)
            ta = (slab_lo - ox) / safe_dx
            tb = (slab_hi - ox) / safe_dx
            segment_lo = tl.maximum(tl.minimum(ta, tb), 1.0e-4)
            segment_hi = tl.minimum(tl.maximum(ta, tb), incumbent)
            crossing = (~parallel) & (segment_lo <= segment_hi)
            coplanar = parallel & (ox >= slab_lo) & (ox <= slab_hi) & (
                incumbent >= 1.0e-4
            )
            candidate |= crossing | coplanar
        candidate &= valid

        flag = candidate.to(tl.int32)
        local_rank = tl.cumsum(flag, axis=0)
        block_count = tl.sum(flag, axis=0)
        zero = tl.zeros((BLOCK_SIZE,), tl.int32)
        atomic_old = tl.atomic_add(
            candidate_count_ptr + zero,
            block_count + zero,
            mask=local_lane == 0,
        )
        block_base = tl.sum(
            tl.where(local_lane == 0, atomic_old, 0), axis=0
        )
        tl.store(
            candidate_ids_ptr + block_base + local_rank - 1,
            ray_index,
            mask=candidate,
        )


else:  # pragma: no cover - definition used only for a clearer error
    _analytic_boundary_kernel = None
    _compact_wire_x_candidates_kernel = None


def prepare_scene_triton(
    scene: "CompiledReflect3WiresScene", device: Any = "cuda"
) -> PreparedAnalyticScene:
    """Copy only analytic-intersection arrays to a CUDA device once."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("prepare_scene_triton requires PyTorch") from exc
    if triton is None:
        raise RuntimeError("prepare_scene_triton requires Triton")

    names = {
        "box_lo": scene.boxes.bounds_min,
        "box_hi": scene.boxes.bounds_max,
        "box_triangle": scene.boxes.triangle_vertices,
        "box_triangle_face": scene.boxes.triangle_face,
        "box_face_mask": scene.boxes.reachable_face_mask,
        "box_collision": scene.boxes.collision_enabled,
        "box_surface": scene.boxes.surface_index,
        "box_mat_inner": scene.boxes.material_inside_index,
        "box_mat_outer": scene.boxes.material_outside_index,
        "wire_origin": scene.wires.origin,
        "wire_raw_u": scene.wires.raw_u,
        "wire_raw_v": scene.wires.raw_v,
        "wire_u": scene.wires.u,
        "wire_v": scene.wires.v,
        "wire_n": scene.wires.n,
        "wire_pitch": scene.wires.pitch,
        "wire_inv_pitch": scene.wires.inv_pitch,
        "wire_radius2": scene.wires.radius2,
        "wire_diameter": scene.wires.diameter,
        "wire_pad_v": scene.wires.pad_v,
        "wire_pad_n": scene.wires.pad_n,
        "wire_umin": scene.wires.umin,
        "wire_umax": scene.wires.umax,
        "wire_v0": scene.wires.v0,
        "wire_kmin": scene.wires.kmin,
        "wire_kmax": scene.wires.kmax,
        "wire_surface": scene.wires.surface_index,
        "wire_mat_inner": scene.wires.material_inner_index,
        "wire_mat_outer": scene.wires.material_outer_index,
    }
    tensors = {
        name: torch.as_tensor(array).contiguous().to(device=device)
        for name, array in names.items()
    }
    return PreparedAnalyticScene(
        tensors=tensors,
        # Resolve generic "cuda" to the concrete ordinal used by Torch.
        device=next(iter(tensors.values())).device,
        box_count=scene.boxes.count,
        wire_count=scene.wires.count,
        wire_normals_are_x_aligned=bool(
            np.all(np.abs(scene.wires.n[:, 0]) == 1.0)
            and np.all(scene.wires.n[:, 1:] == 0.0)
        ),
    )


def intersect_scene_triton(
    scene: Union["CompiledReflect3WiresScene", PreparedAnalyticScene],
    origins: Any,
    directions: Any,
    tmax: Optional[Union[float, Any]] = None,
    *,
    last_instance: Optional[Any] = None,
    last_triangle: Optional[Any] = None,
    out: Optional[BoundaryIntersections] = None,
    block_size: int = 64,
    chroma_mesh_boxes: bool = False,
    chroma_wire_frame: bool = False,
    chroma_wire_full_scan: bool = False,
    _box_count_override: Optional[int] = None,
    _wire_count_override: Optional[int] = None,
) -> BoundaryIntersections:
    """Launch the fused macro-box/periodic-wire Triton query.

    ``origins`` and ``directions`` must be contiguous CUDA float32 Torch
    tensors.  ``last_instance``/``last_triangle`` optionally carry the shared
    transport last-hit encoding: PMTs use non-negative instance/local-triangle
    IDs, while a macro box uses ``last_instance = -(box_index + 2)`` and stores
    its face in ``last_triangle``.  Suppressing that one face reproduces
    Chroma's last-mesh-triangle rule after a reflection.  Pass a
    :class:`PreparedAnalyticScene` in steady-state code to avoid copying the
    small scene tables on every launch.  ``out`` may provide exact-shape CUDA
    tensors to overwrite, allowing a steady-state caller to avoid all result
    allocations.  The returned object is ``out`` itself.  The private count
    overrides exist for certificate backends which replay Chroma's analytic
    wire supplement without replacing any of its global mesh triangles.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("intersect_scene_triton requires PyTorch") from exc
    if triton is None or _analytic_boundary_kernel is None:
        raise RuntimeError("intersect_scene_triton requires Triton")
    if not isinstance(scene, PreparedAnalyticScene):
        scene = prepare_scene_triton(scene, device=origins.device)
    if (
        not isinstance(origins, torch.Tensor)
        or not isinstance(directions, torch.Tensor)
        or origins.shape != directions.shape
        or origins.ndim != 2
        or origins.shape[1] != 3
    ):
        raise ValueError("origins and directions must be Torch tensors of shape [N,3]")
    if not origins.is_cuda or not directions.is_cuda:
        raise ValueError("origins and directions must be CUDA tensors")
    if origins.dtype != torch.float32 or directions.dtype != torch.float32:
        raise ValueError("origins and directions must be float32")
    if not origins.is_contiguous() or not directions.is_contiguous():
        raise ValueError("origins and directions must be contiguous")
    if origins.device != directions.device or origins.device != scene.device:
        raise ValueError("rays and prepared scene must be on the same CUDA device")
    if block_size not in (64, 128, 256):
        raise ValueError("block_size must be 64, 128, or 256")

    n = origins.shape[0]
    device = origins.device
    if (last_instance is None) != (last_triangle is None):
        raise ValueError("last_instance and last_triangle must be supplied together")
    suppress_previous_box = last_instance is not None
    if suppress_previous_box:
        for value, name in (
            (last_instance, "last_instance"), (last_triangle, "last_triangle")
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (n,)
                or value.dtype != torch.int32
                or value.device != device
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous CUDA int32 [N] on the ray device"
                )
        previous_instance_pointer = last_instance
        previous_triangle_pointer = last_triangle
    else:
        # Compile-time-dead pointers in the ordinary stateless query.
        previous_instance_pointer = origins
        previous_triangle_pointer = origins
    if isinstance(tmax, torch.Tensor):
        if (
            tmax.shape != (n,)
            or tmax.dtype != torch.float32
            or tmax.device != device
            or not tmax.is_contiguous()
        ):
            raise ValueError("tensor tmax must be contiguous CUDA float32 [N]")
        tmax_pointer = tmax
        scalar_tmax = 0.0
        tmax_is_pointer = True
    else:
        scalar_tmax = float("inf") if tmax is None else float(tmax)
        if np.isnan(scalar_tmax) or scalar_tmax < 0.0:
            raise ValueError("scalar tmax must be non-negative and not NaN")
        # The constexpr branch makes this pointer dead for scalar caps.
        tmax_pointer = origins
        tmax_is_pointer = False

    if out is None:
        outward = torch.empty((n, 3), dtype=torch.float32, device=device)
        out = BoundaryIntersections(
            distance=torch.empty(n, dtype=torch.float32, device=device),
            kind=torch.empty(n, dtype=torch.int8, device=device),
            index=torch.empty(n, dtype=torch.int32, device=device),
            primitive_index=torch.empty(n, dtype=torch.int32, device=device),
            outward_normal=outward,
            surface_normal=torch.empty_like(outward),
            surface_index=torch.empty(n, dtype=torch.int32, device=device),
            material_inner_index=torch.empty(
                n, dtype=torch.int32, device=device
            ),
            material_outer_index=torch.empty(
                n, dtype=torch.int32, device=device
            ),
            material_from_index=torch.empty(
                n, dtype=torch.int32, device=device
            ),
            material_to_index=torch.empty(
                n, dtype=torch.int32, device=device
            ),
            inside_to_outside=torch.empty(n, dtype=torch.bool, device=device),
        )
    else:
        if not isinstance(out, BoundaryIntersections):
            raise TypeError("out must be a BoundaryIntersections instance")
        specifications = {
            "distance": ((n,), torch.float32),
            "kind": ((n,), torch.int8),
            "index": ((n,), torch.int32),
            "primitive_index": ((n,), torch.int32),
            "outward_normal": ((n, 3), torch.float32),
            "surface_normal": ((n, 3), torch.float32),
            "surface_index": ((n,), torch.int32),
            "material_inner_index": ((n,), torch.int32),
            "material_outer_index": ((n,), torch.int32),
            "material_from_index": ((n,), torch.int32),
            "material_to_index": ((n,), torch.int32),
            "inside_to_outside": ((n,), torch.bool),
        }
        for name, (shape, dtype) in specifications.items():
            value = getattr(out, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != shape
                or value.dtype != dtype
                or value.device != device
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"out.{name} must be contiguous CUDA {dtype} {shape} "
                    "on the ray device"
                )

    distance = out.distance
    kind = out.kind
    index = out.index
    primitive = out.primitive_index
    outward = out.outward_normal
    surface_normal = out.surface_normal
    surface = out.surface_index
    mat_inner = out.material_inner_index
    mat_outer = out.material_outer_index
    mat_from = out.material_from_index
    mat_to = out.material_to_index
    inside_to_outside = out.inside_to_outside
    table = scene.tensors
    grid = (triton.cdiv(n, block_size),)
    _analytic_boundary_kernel[grid](
        origins,
        directions,
        tmax_pointer,
        origins,  # dead ray-index pointer in the direct specialization
        origins,  # dead active-count pointer in the direct specialization
        previous_instance_pointer,
        previous_triangle_pointer,
        table["box_lo"],
        table["box_hi"],
        table["box_triangle"],
        table["box_triangle_face"],
        table["box_face_mask"],
        table["box_collision"],
        table["box_surface"],
        table["box_mat_inner"],
        table["box_mat_outer"],
        table["wire_origin"],
        table["wire_raw_u"],
        table["wire_raw_v"],
        table["wire_u"],
        table["wire_v"],
        table["wire_n"],
        table["wire_pitch"],
        table["wire_inv_pitch"],
        table["wire_radius2"],
        table["wire_diameter"],
        table["wire_pad_v"],
        table["wire_pad_n"],
        table["wire_umin"],
        table["wire_umax"],
        table["wire_v0"],
        table["wire_kmin"],
        table["wire_kmax"],
        table["wire_surface"],
        table["wire_mat_inner"],
        table["wire_mat_outer"],
        distance,
        kind,
        index,
        primitive,
        outward,
        surface_normal,
        surface,
        mat_inner,
        mat_outer,
        mat_from,
        mat_to,
        inside_to_outside,
        n,
        scalar_tmax,
        TMAX_IS_POINTER=tmax_is_pointer,
        INDIRECT=False,
        LOAD_INCUMBENT=False,
        N_RAYS_IS_POINTER=False,
        SUPPRESS_PREVIOUS_BOX=suppress_previous_box,
        CHROMA_MESH_BOXES=bool(chroma_mesh_boxes),
        CHROMA_WIRE_FRAME=bool(chroma_wire_frame),
        CHROMA_WIRE_FULL_SCAN=bool(chroma_wire_full_scan),
        NBOX=(
            scene.box_count
            if _box_count_override is None
            else int(_box_count_override)
        ),
        NWIRE=(
            scene.wire_count
            if _wire_count_override is None
            else int(_wire_count_override)
        ),
        BLOCK_SIZE=block_size,
        num_warps=max(1, block_size // 32),
    )
    return out


def allocate_split_intersection_workspace(
    capacity: int, device: Any = "cuda"
) -> SplitIntersectionWorkspace:
    """Allocate reusable candidate storage for the split analytic query."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("split intersection workspace requires PyTorch") from exc
    capacity = int(capacity)
    if capacity < 0:
        raise ValueError("workspace capacity must be non-negative")
    outward = torch.empty((capacity, 3), dtype=torch.float32, device=device)
    return SplitIntersectionWorkspace(
        candidate_ids=torch.empty(capacity, dtype=torch.int32, device=device),
        candidate_count=torch.zeros(1, dtype=torch.int32, device=device),
        distance=torch.empty(capacity, dtype=torch.float32, device=device),
        kind=torch.empty(capacity, dtype=torch.int8, device=device),
        index=torch.empty(capacity, dtype=torch.int32, device=device),
        primitive_index=torch.empty(capacity, dtype=torch.int32, device=device),
        outward_normal=outward,
        surface_normal=torch.empty_like(outward),
        surface_index=torch.empty(capacity, dtype=torch.int32, device=device),
        material_inner_index=torch.empty(
            capacity, dtype=torch.int32, device=device
        ),
        material_outer_index=torch.empty(
            capacity, dtype=torch.int32, device=device
        ),
        material_from_index=torch.empty(
            capacity, dtype=torch.int32, device=device
        ),
        material_to_index=torch.empty(
            capacity, dtype=torch.int32, device=device
        ),
        inside_to_outside=torch.empty(
            capacity, dtype=torch.bool, device=device
        ),
    )


def intersect_scene_triton_split(
    scene: Union["CompiledReflect3WiresScene", PreparedAnalyticScene],
    origins: Any,
    directions: Any,
    tmax: Optional[Union[float, Any]] = None,
    *,
    last_instance: Optional[Any] = None,
    last_triangle: Optional[Any] = None,
    workspace: Optional[SplitIntersectionWorkspace] = None,
    out: Optional[BoundaryIntersections] = None,
    block_size: int = 64,
    compact_block_size: int = 256,
    chroma_mesh_boxes: bool = False,
    chroma_wire_frame: bool = False,
    chroma_wire_full_scan: bool = False,
) -> BoundaryIntersections:
    """Exact two-stage query optimized for collision-first boundary queues.

    The first launch resolves macro faces and compacts only rays whose segment
    before that incumbent overlaps an outward-rounded wire X slab.  The second
    launch runs the unchanged FP64 periodic-cylinder code indirectly over that
    dense candidate list and scatters winners into the macro result.  Thus the
    broadphase can only add work; it cannot change a hit or tie decision.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("intersect_scene_triton_split requires PyTorch") from exc
    if triton is None or _compact_wire_x_candidates_kernel is None:
        raise RuntimeError("intersect_scene_triton_split requires Triton")
    if not isinstance(scene, PreparedAnalyticScene):
        scene = prepare_scene_triton(scene, device=origins.device)
    if not scene.wire_normals_are_x_aligned:
        raise ValueError(
            "split wire broadphase requires target wire normals parallel to X"
        )

    # Reuse the mature fused launch in its compile-time box-only specialization
    # for validation, tmax handling, output allocation, and exact macro math.
    result = intersect_scene_triton(
        scene,
        origins,
        directions,
        tmax,
        last_instance=last_instance,
        last_triangle=last_triangle,
        out=out,
        block_size=block_size,
        chroma_mesh_boxes=chroma_mesh_boxes,
        chroma_wire_frame=False,
        chroma_wire_full_scan=False,
        _wire_count_override=0,
    )
    n = int(origins.shape[0])
    if n == 0:
        return result
    if compact_block_size not in (64, 128, 256, 512):
        raise ValueError("compact_block_size must be 64, 128, 256, or 512")
    if workspace is None:
        workspace = allocate_split_intersection_workspace(n, origins.device)
    if (
        workspace.capacity < n
        or workspace.candidate_ids.dtype != torch.int32
        or workspace.candidate_count.dtype != torch.int32
        or workspace.candidate_count.numel() != 1
        or workspace.candidate_ids.device != origins.device
        or workspace.candidate_count.device != origins.device
    ):
        raise ValueError(
            "split workspace must have int32 candidate storage/counter on the ray device "
            "with capacity >= N"
        )

    workspace.candidate_count.zero_()
    table = scene.tensors
    compact_grid = (triton.cdiv(n, compact_block_size),)
    _compact_wire_x_candidates_kernel[compact_grid](
        origins,
        directions,
        result.distance,
        table["wire_origin"],
        table["wire_pad_n"],
        workspace.candidate_ids,
        workspace.candidate_count,
        n,
        n,
        N_RAYS_IS_POINTER=False,
        NWIRE=scene.wire_count,
        BLOCK_SIZE=compact_block_size,
        num_warps=max(1, compact_block_size // 32),
    )

    # Launch a worst-case grid to avoid reading the dynamic count on the host.
    # Every program reads the one-word device count; inactive lanes retire
    # before any FP64 wire arithmetic.
    grid = (triton.cdiv(n, block_size),)
    _analytic_boundary_kernel[grid](
        origins,
        directions,
        origins,  # dead tmax pointer: incumbents come from result.distance
        workspace.candidate_ids,
        workspace.candidate_count,
        origins,  # dead previous-instance pointer: this launch has NBOX=0
        origins,  # dead previous-triangle pointer: this launch has NBOX=0
        table["box_lo"],
        table["box_hi"],
        table["box_triangle"],
        table["box_triangle_face"],
        table["box_face_mask"],
        table["box_collision"],
        table["box_surface"],
        table["box_mat_inner"],
        table["box_mat_outer"],
        table["wire_origin"],
        table["wire_raw_u"],
        table["wire_raw_v"],
        table["wire_u"],
        table["wire_v"],
        table["wire_n"],
        table["wire_pitch"],
        table["wire_inv_pitch"],
        table["wire_radius2"],
        table["wire_diameter"],
        table["wire_pad_v"],
        table["wire_pad_n"],
        table["wire_umin"],
        table["wire_umax"],
        table["wire_v0"],
        table["wire_kmin"],
        table["wire_kmax"],
        table["wire_surface"],
        table["wire_mat_inner"],
        table["wire_mat_outer"],
        result.distance,
        result.kind,
        result.index,
        result.primitive_index,
        result.outward_normal,
        result.surface_normal,
        result.surface_index,
        result.material_inner_index,
        result.material_outer_index,
        result.material_from_index,
        result.material_to_index,
        result.inside_to_outside,
        n,  # ignored in favor of candidate_count
        0.0,
        TMAX_IS_POINTER=False,
        INDIRECT=True,
        LOAD_INCUMBENT=True,
        N_RAYS_IS_POINTER=True,
        SUPPRESS_PREVIOUS_BOX=False,
        CHROMA_MESH_BOXES=False,
        CHROMA_WIRE_FRAME=bool(chroma_wire_frame),
        CHROMA_WIRE_FULL_SCAN=bool(chroma_wire_full_scan),
        NBOX=0,
        NWIRE=scene.wire_count,
        BLOCK_SIZE=block_size,
        num_warps=max(1, block_size // 32),
    )
    return result


def _validate_device_count(active_count: Any, device: Any, name: str = "active_count"):
    """Validate a GPU-resident queue length without reading its value.

    The producer/consumer contract owns the dynamic invariant
    ``0 <= active_count <= capacity``.  Reading it here would turn an otherwise
    asynchronous launch chain into a host rendezvous, so this helper validates
    only the statically knowable storage contract.
    """

    import torch

    if (
        not isinstance(active_count, torch.Tensor)
        or active_count.shape != (1,)
        or active_count.dtype != torch.int32
        or active_count.device != device
        or not active_count.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous CUDA int32 [1] on the ray device"
        )


def _allocate_boundary_intersections(capacity: int, device: Any):
    import torch

    outward = torch.empty((capacity, 3), dtype=torch.float32, device=device)
    return BoundaryIntersections(
        distance=torch.empty(capacity, dtype=torch.float32, device=device),
        kind=torch.empty(capacity, dtype=torch.int8, device=device),
        index=torch.empty(capacity, dtype=torch.int32, device=device),
        primitive_index=torch.empty(capacity, dtype=torch.int32, device=device),
        outward_normal=outward,
        surface_normal=torch.empty_like(outward),
        surface_index=torch.empty(capacity, dtype=torch.int32, device=device),
        material_inner_index=torch.empty(
            capacity, dtype=torch.int32, device=device
        ),
        material_outer_index=torch.empty(
            capacity, dtype=torch.int32, device=device
        ),
        material_from_index=torch.empty(
            capacity, dtype=torch.int32, device=device
        ),
        material_to_index=torch.empty(
            capacity, dtype=torch.int32, device=device
        ),
        inside_to_outside=torch.empty(
            capacity, dtype=torch.bool, device=device
        ),
    )


def _validate_boundary_intersections(
    out: BoundaryIntersections, capacity: int, device: Any
) -> BoundaryIntersections:
    import torch

    if not isinstance(out, BoundaryIntersections):
        raise TypeError("out must be a BoundaryIntersections instance")
    specifications = {
        "distance": ((capacity,), torch.float32),
        "kind": ((capacity,), torch.int8),
        "index": ((capacity,), torch.int32),
        "primitive_index": ((capacity,), torch.int32),
        "outward_normal": ((capacity, 3), torch.float32),
        "surface_normal": ((capacity, 3), torch.float32),
        "surface_index": ((capacity,), torch.int32),
        "material_inner_index": ((capacity,), torch.int32),
        "material_outer_index": ((capacity,), torch.int32),
        "material_from_index": ((capacity,), torch.int32),
        "material_to_index": ((capacity,), torch.int32),
        "inside_to_outside": ((capacity,), torch.bool),
    }
    for field, (shape, dtype) in specifications.items():
        value = getattr(out, field)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.dtype != dtype
            or value.device != device
            or not value.is_contiguous()
        ):
            raise ValueError(
                f"out.{field} must be contiguous CUDA {dtype} {shape} "
                "on the ray device"
            )
    return out


def intersect_scene_triton_device_count(
    scene: Union["CompiledReflect3WiresScene", PreparedAnalyticScene],
    origins: Any,
    directions: Any,
    active_count: Any,
    tmax: Optional[Union[float, Any]] = None,
    *,
    last_instance: Optional[Any] = None,
    last_triangle: Optional[Any] = None,
    workspace: Optional[SplitIntersectionWorkspace] = None,
    out: Optional[BoundaryIntersections] = None,
    block_size: int = 64,
    compact_block_size: int = 256,
    split_wires: bool = True,
) -> BoundaryIntersections:
    """Launch production analytic geometry from a GPU-resident live count.

    Ray and output tensors describe the allocation *capacity*.  Only the
    prefix selected by ``active_count`` is read or written, and every
    capacity-sized launch retires wholly inactive CTAs before detector math.
    This permits a collision producer, gather, analytic query, PMT query, and
    merge to remain in one CUDA stream without a count read on the host.

    The caller must preserve ``0 <= active_count <= capacity`` until all
    enqueued consumers complete.  ``split_wires=True`` retains the production
    two-stage box/wire broadphase.  This API intentionally exposes no strict
    Chroma replay options; certificate execution continues through the
    existing synchronized APIs.
    """

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "intersect_scene_triton_device_count requires PyTorch"
        ) from exc
    if triton is None or _analytic_boundary_kernel is None:
        raise RuntimeError(
            "intersect_scene_triton_device_count requires Triton"
        )
    if not isinstance(scene, PreparedAnalyticScene):
        scene = prepare_scene_triton(scene, device=origins.device)
    if (
        not isinstance(origins, torch.Tensor)
        or not isinstance(directions, torch.Tensor)
        or origins.shape != directions.shape
        or origins.ndim != 2
        or origins.shape[1] != 3
    ):
        raise ValueError(
            "origins and directions must be Torch tensors of shape [capacity,3]"
        )
    if not origins.is_cuda or not directions.is_cuda:
        raise ValueError("origins and directions must be CUDA tensors")
    if origins.dtype != torch.float32 or directions.dtype != torch.float32:
        raise ValueError("origins and directions must be float32")
    if not origins.is_contiguous() or not directions.is_contiguous():
        raise ValueError("origins and directions must be contiguous")
    if origins.device != directions.device or origins.device != scene.device:
        raise ValueError("rays and prepared scene must be on the same CUDA device")
    if block_size not in (64, 128, 256):
        raise ValueError("block_size must be 64, 128, or 256")
    if compact_block_size not in (64, 128, 256, 512):
        raise ValueError("compact_block_size must be 64, 128, 256, or 512")
    if split_wires and not scene.wire_normals_are_x_aligned:
        raise ValueError(
            "split wire broadphase requires target wire normals parallel to X"
        )

    capacity = int(origins.shape[0])
    device = origins.device
    _validate_device_count(active_count, device)

    if (last_instance is None) != (last_triangle is None):
        raise ValueError("last_instance and last_triangle must be supplied together")
    suppress_previous_box = last_instance is not None
    if suppress_previous_box:
        for value, name in (
            (last_instance, "last_instance"),
            (last_triangle, "last_triangle"),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != (capacity,)
                or value.dtype != torch.int32
                or value.device != device
                or not value.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous CUDA int32 [capacity] "
                    "on the ray device"
                )
        previous_instance_pointer = last_instance
        previous_triangle_pointer = last_triangle
    else:
        previous_instance_pointer = origins
        previous_triangle_pointer = origins

    if isinstance(tmax, torch.Tensor):
        if (
            tmax.shape != (capacity,)
            or tmax.dtype != torch.float32
            or tmax.device != device
            or not tmax.is_contiguous()
        ):
            raise ValueError(
                "tensor tmax must be contiguous CUDA float32 [capacity]"
            )
        tmax_pointer = tmax
        scalar_tmax = 0.0
        tmax_is_pointer = True
    else:
        scalar_tmax = float("inf") if tmax is None else float(tmax)
        if np.isnan(scalar_tmax) or scalar_tmax < 0.0:
            raise ValueError("scalar tmax must be non-negative and not NaN")
        tmax_pointer = origins
        tmax_is_pointer = False

    if out is None:
        if workspace is not None:
            if workspace.capacity < capacity:
                raise ValueError(
                    "split workspace capacity is smaller than ray capacity"
                )
            out = workspace.outputs(capacity)
        else:
            out = _allocate_boundary_intersections(capacity, device)
    result = _validate_boundary_intersections(out, capacity, device)
    if capacity == 0:
        return result

    table = scene.tensors

    def launch_analytic(
        *, ray_indices, count_pointer, load_incumbent, nbox, nwire,
        suppress_previous,
    ):
        grid = (triton.cdiv(capacity, block_size),)
        _analytic_boundary_kernel[grid](
            origins,
            directions,
            tmax_pointer,
            ray_indices,
            count_pointer,
            previous_instance_pointer,
            previous_triangle_pointer,
            table["box_lo"],
            table["box_hi"],
            table["box_triangle"],
            table["box_triangle_face"],
            table["box_face_mask"],
            table["box_collision"],
            table["box_surface"],
            table["box_mat_inner"],
            table["box_mat_outer"],
            table["wire_origin"],
            table["wire_raw_u"],
            table["wire_raw_v"],
            table["wire_u"],
            table["wire_v"],
            table["wire_n"],
            table["wire_pitch"],
            table["wire_inv_pitch"],
            table["wire_radius2"],
            table["wire_diameter"],
            table["wire_pad_v"],
            table["wire_pad_n"],
            table["wire_umin"],
            table["wire_umax"],
            table["wire_v0"],
            table["wire_kmin"],
            table["wire_kmax"],
            table["wire_surface"],
            table["wire_mat_inner"],
            table["wire_mat_outer"],
            result.distance,
            result.kind,
            result.index,
            result.primitive_index,
            result.outward_normal,
            result.surface_normal,
            result.surface_index,
            result.material_inner_index,
            result.material_outer_index,
            result.material_from_index,
            result.material_to_index,
            result.inside_to_outside,
            capacity,
            scalar_tmax,
            TMAX_IS_POINTER=tmax_is_pointer,
            INDIRECT=ray_indices is not origins,
            LOAD_INCUMBENT=load_incumbent,
            N_RAYS_IS_POINTER=True,
            SUPPRESS_PREVIOUS_BOX=suppress_previous,
            CHROMA_MESH_BOXES=False,
            CHROMA_WIRE_FRAME=False,
            CHROMA_WIRE_FULL_SCAN=False,
            NBOX=nbox,
            NWIRE=nwire,
            BLOCK_SIZE=block_size,
            num_warps=max(1, block_size // 32),
        )

    if not split_wires:
        launch_analytic(
            ray_indices=origins,
            count_pointer=active_count,
            load_incumbent=False,
            nbox=scene.box_count,
            nwire=scene.wire_count,
            suppress_previous=suppress_previous_box,
        )
        return result

    if workspace is None:
        workspace = allocate_split_intersection_workspace(capacity, device)
    if (
        workspace.capacity < capacity
        or workspace.candidate_ids.dtype != torch.int32
        or workspace.candidate_count.dtype != torch.int32
        or workspace.candidate_count.shape != (1,)
        or workspace.candidate_ids.device != device
        or workspace.candidate_count.device != device
    ):
        raise ValueError(
            "split workspace must have int32 candidate storage/counter on the "
            "ray device with capacity >= ray capacity"
        )

    launch_analytic(
        ray_indices=origins,
        count_pointer=active_count,
        load_incumbent=False,
        nbox=scene.box_count,
        nwire=0,
        suppress_previous=suppress_previous_box,
    )
    if scene.wire_count == 0:
        return result

    workspace.candidate_count.zero_()
    _compact_wire_x_candidates_kernel[
        (triton.cdiv(capacity, compact_block_size),)
    ](
        origins,
        directions,
        result.distance,
        table["wire_origin"],
        table["wire_pad_n"],
        workspace.candidate_ids,
        workspace.candidate_count,
        active_count,
        capacity,
        N_RAYS_IS_POINTER=True,
        NWIRE=scene.wire_count,
        BLOCK_SIZE=compact_block_size,
        num_warps=max(1, compact_block_size // 32),
    )
    launch_analytic(
        ray_indices=workspace.candidate_ids,
        count_pointer=workspace.candidate_count,
        load_incumbent=True,
        nbox=0,
        nwire=scene.wire_count,
        suppress_previous=False,
    )
    return result


__all__ = [
    "BoundaryIntersections",
    "CHROMA_EPSILON",
    "GeometryKind",
    "PreparedAnalyticScene",
    "SplitIntersectionWorkspace",
    "WIRE_T_MIN",
    "allocate_split_intersection_workspace",
    "intersect_boxes_numpy",
    "intersect_scene_numpy",
    "intersect_scene_triton",
    "intersect_scene_triton_device_count",
    "intersect_scene_triton_split",
    "intersect_wires_bruteforce_numpy",
    "intersect_wires_numpy",
    "prepare_scene_triton",
]
