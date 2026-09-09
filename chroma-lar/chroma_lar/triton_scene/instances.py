"""Exact shared-BLAS traversal for the repeated PMTs in a compiled scene.

The reflect/reflect/three-wire detector contains 81 reachable copies of one
5,120-triangle PMT mesh.  Flattening those copies destroys cache locality and
turns a small canonical mesh into more than 400k triangles.  This module keeps
one packed BLAS and performs instance traversal in four exact GPU stages:

1. compact rays intersecting the conservatively padded union of all PMTs,
2. count every retained ray/instance AABB overlap (there is no candidate cap),
3. compact and transform precisely that many pairs into PMT-local space, and
4. traverse the shared BLAS and reduce each variable-length candidate segment.

The module also provides a fused binary top-level acceleration structure
(TLAS) over the same outward-padded instance boxes and traverses the TLAS and
canonical BLAS in one Triton kernel.  It therefore has no all-instance scan,
candidate prefix sum, dynamic pair allocation, or host count synchronization.
The production path recognizes the detector's staggered 9x9 PMT lattice.  A
conservative interval locator reduces the normal broadphase to at most sixteen
instance boxes; wider grazing rays go through the exact TLAS instead.  The
former 81-box scan remains available for foreign layouts and strict Chroma
world-space compatibility.  The fused path also remains available for
experimentation and cross-checking.
The TLAS stack capacity is derived from its immutable topology; overflow is
still observable and the diagnostic path remains available as a checked
fallback.  Conservative instance bounds may admit a false positive but can
never change the final exact triangle result.

Torch and Triton are optional import-time dependencies.  The CPU reference is
available with NumPy alone and deliberately uses the same original local
triangle and retained-instance namespaces as the accelerated path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from chroma.triton.bvh import PackedBVH, build_packed_bvh


_IMPORT_ERROR = None
try:
    import torch
    import triton
    import triton.language as tl

    from chroma.triton.bvh_kernels import (
        DevicePackedBVH,
        TraversalWorkspace,
    )
except Exception as error:  # Keep NumPy-only artifact tooling importable.
    torch = None
    triton = None
    tl = None
    DevicePackedBVH = Any
    TraversalWorkspace = Any
    _IMPORT_ERROR = error


DEFAULT_RAY_TILE = 1 << 20
DEFAULT_GRID_CANDIDATES = 16
# Below this active-ray population the extra grid-range/fallback bookkeeping
# costs more than the cache-resident 81-box scan on A100.  Selection happens
# after the existing union compaction, so sparse queues retain the lower-
# latency general path while large wavefronts receive the lattice speedup.
GRID_MIN_ACTIVE_RAYS = 1 << 18
# Zero preserves the uncapped capacity grid until end-to-end transport tuning
# identifies a robust persistent-grid size.  Isolated coherent-ray A100 tests
# favored 20--32 one-warp programs/SM, but the full simulation's divergent PMT
# population needs a wider sweep.  Positive values activate the bounded grid;
# the kernel grid-strides over all remaining compacted candidates.
DEVICE_TLAS_PROGRAMS_PER_SM = 0


def _persistent_tlas_program_count(
    ray_capacity: int,
    block_size: int,
    multiprocessor_count: int,
    programs_per_sm: Optional[int] = None,
) -> int:
    """Return a tunable bounded launch; zero programs/SM means uncapped."""

    ray_capacity = int(ray_capacity)
    block_size = int(block_size)
    multiprocessor_count = int(multiprocessor_count)
    if programs_per_sm is None:
        programs_per_sm = DEVICE_TLAS_PROGRAMS_PER_SM
    programs_per_sm = int(programs_per_sm)
    if ray_capacity < 0:
        raise ValueError("ray capacity cannot be negative")
    if block_size <= 0:
        raise ValueError("block size must be positive")
    if multiprocessor_count <= 0:
        raise ValueError("multiprocessor count must be positive")
    if programs_per_sm < 0:
        raise ValueError("programs per SM cannot be negative")
    capacity_programs = (ray_capacity + block_size - 1) // block_size
    if programs_per_sm == 0:
        return capacity_programs
    resident_programs = multiprocessor_count * programs_per_sm
    return min(capacity_programs, resident_programs)


class PMTInstanceBackendUnavailable(RuntimeError):
    """Raised when the optional CUDA/Triton instance backend is unavailable."""


@dataclass(frozen=True)
class PMTGridLocator:
    """Immutable descriptor for an ascending row-major PMT plane lattice.

    Rows advance along world Y and columns along world Z.  A column may have
    its own row-zero Y offset, which represents the target detector's stagger
    without a lookup table.  ``half_y``/``half_z`` and ``coordinate_guard``
    conservatively enclose every outward-padded instance box.
    """

    rows: int
    columns: int
    row_pitch: float
    column_pitch: float
    row_zero_min_y: float
    row_zero_max_y: float
    column_zero_z: float
    half_y: float
    half_z: float
    coordinate_guard: float
    origin_limit: float

    @property
    def maximum_candidates(self) -> int:
        return int(self.rows * self.columns)


@dataclass(frozen=True)
class PMTInstanceResult:
    """Nearest canonical-PMT intersection for each world-space ray.

    ``instance_ids`` index the retained instances in the compiled artifact.
    ``channel_ids`` are the original detector-global channel IDs and are never
    renumbered by reachability pruning.  A miss has IDs ``-1``, distance
    ``inf``, and a zero normal.
    """

    triangle_ids: Any
    distances: Any
    instance_ids: Any
    channel_ids: Any
    world_normals: Any
    overflow: Any
    candidate_counts: Any

    @property
    def local_triangle_ids(self) -> Any:
        return self.triangle_ids


@dataclass(frozen=True)
class PMTInstanceAccelerator:
    """One canonical PMT BLAS plus immutable retained-instance data."""

    host_bvh: PackedBVH
    device_bvh: Any
    channel_ids: Any
    world_to_object_rotation: Any
    world_to_object_translation: Any
    object_to_world_rotation: Any
    # Compatibility-only data.  Chroma flattens every PMT into world-space
    # float32 vertices before tracing; the fast path instead transforms rays
    # into one shared local mesh.  These optional buffers let every candidate
    # leaf reproduce the former arithmetic without burdening production.
    chroma_world_vertices: Any
    chroma_triangle_indices: Any
    # These are deliberately outward-padded broadphase bounds.
    bounds_min: Any
    bounds_max: Any
    union_bounds_min: Any
    union_bounds_max: Any
    tlas_bounds_min: Any
    tlas_bounds_max: Any
    tlas_left_child: Any
    tlas_right_child: Any
    tlas_instance: Any
    tlas_stack_capacity: int
    host_channel_ids: np.ndarray
    host_world_to_object_rotation: np.ndarray
    host_world_to_object_translation: np.ndarray
    host_object_to_world_rotation: np.ndarray
    host_chroma_world_vertices: Optional[np.ndarray]
    host_chroma_triangle_indices: Optional[np.ndarray]
    host_bounds_min: np.ndarray
    host_bounds_max: np.ndarray
    host_union_bounds_min: np.ndarray
    host_union_bounds_max: np.ndarray
    host_tlas_bounds_min: np.ndarray
    host_tlas_bounds_max: np.ndarray
    host_tlas_left_child: np.ndarray
    host_tlas_right_child: np.ndarray
    host_tlas_instance: np.ndarray
    grid_locator: Optional[PMTGridLocator]
    coarse_bounds_min: Any = None
    coarse_bounds_max: Any = None
    coarse_box_count: int = 1

    @property
    def instance_count(self) -> int:
        return int(self.host_channel_ids.size)

    @property
    def device(self) -> Any:
        return self.device_bvh.nodes.device

    def allocate_workspace(
        self,
        ray_capacity: int,
        candidate_capacity: int = 0,
        result_capacity: Optional[int] = None,
    ) -> "PMTInstanceWorkspace":
        return PMTInstanceWorkspace.allocate(
            self,
            ray_capacity,
            candidate_capacity=candidate_capacity,
            result_capacity=result_capacity,
        )


@dataclass
class PMTInstanceWorkspace:
    """Reusable scratch and opt-in result storage for exact instance queries."""

    counts: Any
    offsets: Any
    active_ray_ids: Any
    device_candidate_count: Any
    grid_fallback_flags: Any
    grid_fallback_offsets: Any
    grid_fallback_active_ids: Any
    local_origins: Any
    local_directions: Any
    candidate_tmax: Any
    candidate_last_triangle: Any
    candidate_instances: Any
    candidate_ray_ids: Any
    candidate_triangles: Any
    candidate_distances: Any
    candidate_overflow: Any
    traversal: Any
    fused_traversal: Any
    tlas_stack: Any
    sticky_overflow: Any
    bvh_dummy: Any
    out_triangles: Any
    out_distances: Any
    out_instances: Any
    out_channels: Any
    out_normals: Any
    out_overflow: Any
    out_candidate_counts: Any
    ray_capacity: int
    candidate_capacity: int
    result_capacity: int
    accelerator: PMTInstanceAccelerator

    @classmethod
    def allocate(
        cls,
        accelerator: PMTInstanceAccelerator,
        ray_capacity: int,
        *,
        candidate_capacity: int = 0,
        result_capacity: Optional[int] = None,
    ) -> "PMTInstanceWorkspace":
        _require_backend()
        if result_capacity is None:
            result_capacity = ray_capacity
        if ray_capacity < 0 or candidate_capacity < 0 or result_capacity < 0:
            raise ValueError("workspace capacities must be non-negative")
        device = accelerator.device
        ray_storage = max(1, int(ray_capacity))
        candidate_storage = max(1, int(candidate_capacity))
        result_storage = max(1, int(result_capacity))
        return cls(
            counts=torch.empty(ray_storage, dtype=torch.int32, device=device),
            offsets=torch.empty(ray_storage + 1, dtype=torch.int64, device=device),
            active_ray_ids=torch.empty(
                ray_storage, dtype=torch.int32, device=device
            ),
            device_candidate_count=torch.zeros(
                1, dtype=torch.int32, device=device
            ),
            grid_fallback_flags=torch.empty(
                ray_storage, dtype=torch.int32, device=device
            ),
            grid_fallback_offsets=torch.empty(
                ray_storage + 1, dtype=torch.int64, device=device
            ),
            grid_fallback_active_ids=torch.empty(
                ray_storage, dtype=torch.int32, device=device
            ),
            local_origins=torch.empty(
                (candidate_storage, 3), dtype=torch.float32, device=device
            ),
            local_directions=torch.empty(
                (candidate_storage, 3), dtype=torch.float32, device=device
            ),
            candidate_tmax=torch.empty(
                candidate_storage, dtype=torch.float32, device=device
            ),
            candidate_last_triangle=torch.empty(
                candidate_storage, dtype=torch.int32, device=device
            ),
            candidate_instances=torch.empty(
                candidate_storage, dtype=torch.int32, device=device
            ),
            candidate_ray_ids=torch.empty(
                candidate_storage, dtype=torch.int32, device=device
            ),
            candidate_triangles=torch.empty(
                candidate_storage, dtype=torch.int32, device=device
            ),
            candidate_distances=torch.empty(
                candidate_storage, dtype=torch.float32, device=device
            ),
            candidate_overflow=torch.empty(
                candidate_storage, dtype=torch.uint8, device=device
            ),
            traversal=accelerator.device_bvh.allocate_workspace(
                candidate_storage
            ),
            fused_traversal=accelerator.device_bvh.allocate_workspace(
                ray_storage
            ),
            tlas_stack=torch.empty(
                max(1, accelerator.tlas_stack_capacity * ray_storage),
                dtype=torch.int32,
                device=device,
            ),
            sticky_overflow=torch.zeros(1, dtype=torch.int32, device=device),
            bvh_dummy=torch.empty(1, dtype=torch.int32, device=device),
            out_triangles=torch.empty(
                result_storage, dtype=torch.int32, device=device
            ),
            out_distances=torch.empty(
                result_storage, dtype=torch.float32, device=device
            ),
            out_instances=torch.empty(
                result_storage, dtype=torch.int32, device=device
            ),
            out_channels=torch.empty(
                result_storage, dtype=torch.int32, device=device
            ),
            out_normals=torch.empty(
                (result_storage, 3), dtype=torch.float32, device=device
            ),
            out_overflow=torch.empty(
                result_storage, dtype=torch.uint8, device=device
            ),
            out_candidate_counts=torch.empty(
                result_storage, dtype=torch.int32, device=device
            ),
            ray_capacity=int(ray_capacity),
            candidate_capacity=int(candidate_capacity),
            result_capacity=int(result_capacity),
            accelerator=accelerator,
        )

    def ensure_candidate_capacity(self, required: int) -> None:
        """Grow temporary pair storage; never truncate an exact candidate list."""

        if required <= self.candidate_capacity:
            return
        if required < 0:
            raise ValueError("required candidate capacity cannot be negative")
        # Geometric growth amortizes allocations across transport epochs.  It
        # is a storage policy only; `required` candidates are always retained.
        grown = max(int(required), max(1024, 2 * self.candidate_capacity))
        device = self.accelerator.device
        self.local_origins = torch.empty(
            (grown, 3), dtype=torch.float32, device=device
        )
        self.local_directions = torch.empty_like(self.local_origins)
        self.candidate_tmax = torch.empty(
            grown, dtype=torch.float32, device=device
        )
        self.candidate_last_triangle = torch.empty(
            grown, dtype=torch.int32, device=device
        )
        self.candidate_instances = torch.empty(
            grown, dtype=torch.int32, device=device
        )
        self.candidate_ray_ids = torch.empty(
            grown, dtype=torch.int32, device=device
        )
        self.candidate_triangles = torch.empty(
            grown, dtype=torch.int32, device=device
        )
        self.candidate_distances = torch.empty(
            grown, dtype=torch.float32, device=device
        )
        self.candidate_overflow = torch.empty(
            grown, dtype=torch.uint8, device=device
        )
        self.traversal = self.accelerator.device_bvh.allocate_workspace(grown)
        self.candidate_capacity = grown

    def ensure_result_capacity(self, required: int) -> None:
        """Grow optional query outputs geometrically and retain their high-water mark."""

        required = int(required)
        if required < 0:
            raise ValueError("required result capacity cannot be negative")
        if required <= self.result_capacity:
            return
        grown = max(required, max(1024, 2 * self.result_capacity))
        device = self.accelerator.device
        self.out_triangles = torch.empty(grown, dtype=torch.int32, device=device)
        self.out_distances = torch.empty(
            grown, dtype=torch.float32, device=device
        )
        self.out_instances = torch.empty(grown, dtype=torch.int32, device=device)
        self.out_channels = torch.empty(grown, dtype=torch.int32, device=device)
        self.out_normals = torch.empty(
            (grown, 3), dtype=torch.float32, device=device
        )
        self.out_overflow = torch.empty(grown, dtype=torch.uint8, device=device)
        self.out_candidate_counts = torch.empty(
            grown, dtype=torch.int32, device=device
        )
        self.result_capacity = grown

    def outputs(self, count: int) -> PMTInstanceResult:
        """Return live workspace-owned output views for an ``out=`` query.

        A subsequent query into these views overwrites the earlier result.
        """

        count = int(count)
        self.ensure_result_capacity(count)
        return PMTInstanceResult(
            triangle_ids=self.out_triangles[:count],
            distances=self.out_distances[:count],
            instance_ids=self.out_instances[:count],
            channel_ids=self.out_channels[:count],
            world_normals=self.out_normals[:count],
            overflow=self.out_overflow[:count],
            candidate_counts=self.out_candidate_counts[:count],
        )

    def ensure_ray_capacity(self, required: int) -> None:
        """Grow sparse-ray prefix storage to the actual query size.

        Ray storage is independent of the usually much smaller PMT candidate
        list.  Keeping the two capacities separate lets callers create an
        effectively empty workspace and pay for rays only when a boundary
        queue is first observed.
        """

        required = int(required)
        if required < 0:
            raise ValueError("required ray capacity cannot be negative")
        if required <= self.ray_capacity:
            return
        device = self.accelerator.device
        self.counts = torch.empty(required, dtype=torch.int32, device=device)
        self.offsets = torch.empty(required + 1, dtype=torch.int64, device=device)
        self.active_ray_ids = torch.empty(
            required, dtype=torch.int32, device=device
        )
        self.grid_fallback_flags = torch.empty(
            required, dtype=torch.int32, device=device
        )
        self.grid_fallback_offsets = torch.empty(
            required + 1, dtype=torch.int64, device=device
        )
        self.grid_fallback_active_ids = torch.empty(
            required, dtype=torch.int32, device=device
        )
        self.fused_traversal = self.accelerator.device_bvh.allocate_workspace(
            required
        )
        self.tlas_stack = torch.empty(
            max(1, self.accelerator.tlas_stack_capacity * required),
            dtype=torch.int32,
            device=device,
        )
        self.ray_capacity = required

    def clear_sticky_overflow(self) -> None:
        """Begin an asynchronously audited sequence of PMT queries."""

        self.sticky_overflow.zero_()

    def sticky_overflowed(self) -> bool:
        """Synchronize once and report any TLAS/BLAS overflow since reset."""

        return bool(self.sticky_overflow.item())


def _require_backend() -> None:
    if torch is None or triton is None:
        detail = "" if _IMPORT_ERROR is None else ": %s" % (_IMPORT_ERROR,)
        raise PMTInstanceBackendUnavailable(
            "PyTorch and Triton are required" + detail
        )
    if not torch.cuda.is_available():
        raise PMTInstanceBackendUnavailable(
            "a CUDA device visible to PyTorch is required"
        )


def triton_instance_available(require_cuda: bool = False) -> bool:
    if torch is None or triton is None:
        return False
    return bool(torch.cuda.is_available()) if require_cuda else True


def _scene_arrays(scene: Any):
    if not hasattr(scene, "pmt") or not hasattr(scene, "instances"):
        raise TypeError("scene must expose compiled .pmt and .instances data")
    pmt = scene.pmt
    instances = scene.instances
    vertices = np.asarray(pmt.vertices, dtype=np.float32)
    triangles = np.asarray(pmt.triangles)
    channel_ids = np.asarray(instances.channel_id, dtype=np.int32)
    w2o_r = np.asarray(instances.world_to_object_rotation, dtype=np.float32)
    w2o_t = np.asarray(instances.world_to_object_translation, dtype=np.float32)
    o2w_r = np.asarray(instances.object_to_world_rotation, dtype=np.float32)
    bounds_min = np.asarray(instances.bounds_min, dtype=np.float32)
    bounds_max = np.asarray(instances.bounds_max, dtype=np.float32)

    count = len(channel_ids)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("scene.pmt.vertices must have shape (N, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("scene.pmt.triangles must have shape (M, 3)")
    if w2o_r.shape != (count, 3, 3) or o2w_r.shape != (count, 3, 3):
        raise ValueError("instance rotations must have shape (N, 3, 3)")
    if w2o_t.shape != (count, 3):
        raise ValueError("world-to-object translations must have shape (N, 3)")
    if bounds_min.shape != (count, 3) or bounds_max.shape != (count, 3):
        raise ValueError("instance bounds must have shape (N, 3)")
    if count == 0:
        raise ValueError("at least one PMT instance is required")
    if np.any(channel_ids[1:] <= channel_ids[:-1]):
        raise ValueError("global PMT channel IDs must be strictly increasing")
    if not (
        np.isfinite(vertices).all()
        and np.isfinite(w2o_r).all()
        and np.isfinite(w2o_t).all()
        and np.isfinite(o2w_r).all()
        and np.isfinite(bounds_min).all()
        and np.isfinite(bounds_max).all()
    ):
        raise ValueError("PMT geometry and transforms must be finite")
    if np.any(bounds_min > bounds_max):
        raise ValueError("instance bounds are inverted")
    return (
        vertices,
        triangles,
        channel_ids,
        w2o_r,
        w2o_t,
        o2w_r,
        bounds_min,
        bounds_max,
    )


def _padded_bounds(
    vertices: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad broadphase boxes enough to dominate FP32 transform roundoff.

    This does not alter triangles or reported distances.  The bound follows a
    standard forward-error envelope for three multiply-adds, with an extra
    barycentric tolerance term matching the BLAS triangle test.
    """

    eps = np.finfo(np.float32).eps
    world_magnitude = np.maximum(np.abs(bounds_min), np.abs(bounds_max)).max(
        axis=1, keepdims=True
    )
    local_magnitude = np.max(np.abs(vertices), initial=np.float32(0.0))
    pad = (
        np.float32(12.0 * eps)
        * (np.float32(1.0) + world_magnitude + local_magnitude)
        + np.float32(4.0e-6)
    ).astype(np.float32)
    lower = np.nextafter(bounds_min - pad, np.float32(-np.inf)).astype(
        np.float32
    )
    upper = np.nextafter(bounds_max + pad, np.float32(np.inf)).astype(
        np.float32
    )
    return np.ascontiguousarray(lower), np.ascontiguousarray(upper)


def _build_chroma_world_vertices(scene: Any) -> np.ndarray:
    """Replay ``Geometry.flatten`` for the retained PMT vertex blocks.

    Chroma applies each float32 rotation with ``np.inner``, assigns into a
    float32 destination, and then constructs a :class:`Mesh`, whose default
    final step rounds the coordinates to twelve decimal places.  The global
    duplicate-vertex pass only reorders/remaps equal words, so retaining one
    indexed world-space block per PMT is sufficient for exact leaf arithmetic.
    """

    vertices = np.asarray(scene.pmt.vertices, dtype=np.float32)
    rotations = np.asarray(
        scene.instances.object_to_world_rotation, dtype=np.float32
    )
    translations = np.asarray(
        scene.instances.object_to_world_translation, dtype=np.float32
    )
    world = np.empty(
        (len(rotations), len(vertices), 3), dtype=np.float32
    )
    for instance, (rotation, translation) in enumerate(
        zip(rotations, translations)
    ):
        # Spell this exactly as Geometry.flatten rather than using a batched
        # einsum: NumPy reduction order is part of the compatibility contract.
        world[instance] = np.inner(vertices, rotation) + translation
    return np.ascontiguousarray(world.round(decimals=12), dtype=np.float32)


def _build_instance_tlas(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build a balanced, ascending-leaf binary TLAS.

    Leaves are deliberately kept in retained-instance order.  A left-first
    traversal therefore encounters lower instance IDs first, preserving the
    legacy reduction rule when two duplicated meshes have exactly equal hit
    distances.  The target's compiler order is already a spatial 9x9 wall
    order, so contiguous median splits are also an effective hierarchy.

    Every internal bound is one representable float wider than the union of
    its children.  This makes the hierarchy conservative even if slab
    arithmetic rounds differently at a parent and leaf.
    """

    count = int(len(bounds_min))
    if count <= 0 or bounds_min.shape != (count, 3) or bounds_max.shape != (count, 3):
        raise ValueError("TLAS bounds must be non-empty float32 (N,3) arrays")

    node_min: list[np.ndarray] = []
    node_max: list[np.ndarray] = []
    left_child: list[int] = []
    right_child: list[int] = []
    instances: list[int] = []
    maximum_depth = 0

    def append_node(first: int, stop: int, depth: int) -> int:
        nonlocal maximum_depth
        node = len(instances)
        node_min.append(np.empty(3, dtype=np.float32))
        node_max.append(np.empty(3, dtype=np.float32))
        left_child.append(-1)
        right_child.append(-1)
        instances.append(-1)
        maximum_depth = max(maximum_depth, depth)
        if stop - first == 1:
            node_min[node] = bounds_min[first]
            node_max[node] = bounds_max[first]
            instances[node] = first
            return node

        middle = first + (stop - first) // 2
        left = append_node(first, middle, depth + 1)
        right = append_node(middle, stop, depth + 1)
        lower = np.minimum(node_min[left], node_min[right])
        upper = np.maximum(node_max[left], node_max[right])
        node_min[node] = np.nextafter(lower, np.float32(-np.inf))
        node_max[node] = np.nextafter(upper, np.float32(np.inf))
        left_child[node] = left
        right_child[node] = right
        return node

    root = append_node(0, count, 0)
    if root != 0:  # pragma: no cover - preorder construction makes this structural.
        raise AssertionError("TLAS root must be node zero")
    # A binary DFS pushes one right sibling per inner level.  Depth is thus an
    # exact topology-derived upper bound, with one slot retained for the
    # single-leaf case and for defensive foreign-buffer checks.
    stack_capacity = max(1, maximum_depth)
    return (
        np.ascontiguousarray(np.stack(node_min), dtype=np.float32),
        np.ascontiguousarray(np.stack(node_max), dtype=np.float32),
        np.ascontiguousarray(left_child, dtype=np.int32),
        np.ascontiguousarray(right_child, dtype=np.int32),
        np.ascontiguousarray(instances, dtype=np.int32),
        stack_capacity,
    )


def _infer_pmt_grid_locator(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> Optional[PMTGridLocator]:
    """Recognize the target's ascending square PMT lattice conservatively.

    This is deliberately a proof-by-validation rather than a detector-name
    switch.  A layout receives the fast locator only when its padded boxes are
    a row-major square grid on a common X plane, with affine row/column pitch.
    Any future geometry perturbation simply returns ``None`` and retains the
    general exact broadphase.
    """

    lower = np.asarray(bounds_min, dtype=np.float32)
    upper = np.asarray(bounds_max, dtype=np.float32)
    if (
        lower.ndim != 2
        or lower.shape[1:] != (3,)
        or upper.shape != lower.shape
        or len(lower) < 4
    ):
        return None
    side = int(round(float(np.sqrt(len(lower)))))
    if side * side != len(lower):
        return None

    centers = (
        lower.astype(np.float64) + upper.astype(np.float64)
    ) * 0.5
    half_extents = (
        upper.astype(np.float64) - lower.astype(np.float64)
    ) * 0.5
    world_magnitude = float(
        np.max(np.maximum(np.abs(lower), np.abs(upper)), initial=0.0)
    )
    # The locator calculates coordinates from two slab divisions and one
    # multiply-add.  This envelope is intentionally much wider than their
    # FP32 forward error at detector scale, while remaining tiny beside the
    # roughly 360 mm gap between neighboring padded PMT boxes.
    coordinate_guard = float(
        np.float32(128.0 * np.finfo(np.float32).eps * (1.0 + world_magnitude))
    )
    tolerance = max(1.0e-3, coordinate_guard)

    grid = centers.reshape(side, side, 3)
    # All PMTs must occupy one X slab.  The small extent tolerance admits only
    # outward-padding variation, never a second physical plane.
    if np.ptp(grid[:, :, 0]) > tolerance:
        return None
    if np.ptp(half_extents[:, 0]) > tolerance:
        return None

    z_by_row = grid[:, :, 2]
    if np.max(np.ptp(z_by_row, axis=0), initial=0.0) > tolerance:
        return None
    z_centers = z_by_row.mean(axis=0)
    column_steps = np.diff(z_centers)
    if len(column_steps) == 0:
        return None
    column_pitch = float(column_steps.mean())
    if column_pitch <= 0.0 or np.max(
        np.abs(column_steps - column_pitch), initial=0.0
    ) > tolerance:
        return None

    y_by_column = grid[:, :, 1]
    row_steps = np.diff(y_by_column, axis=0)
    row_pitch = float(row_steps.mean())
    if row_pitch <= 0.0 or np.max(
        np.abs(row_steps - row_pitch), initial=0.0
    ) > tolerance:
        return None
    row_zero = y_by_column[0]

    row_model = row_zero[None, :] + row_pitch * np.arange(side)[:, None]
    column_model = z_centers[0] + column_pitch * np.arange(side)
    row_residual = float(np.max(np.abs(y_by_column - row_model), initial=0.0))
    column_residual = float(
        np.max(np.abs(z_by_row - column_model[None, :]), initial=0.0)
    )
    half_y = float(np.max(half_extents[:, 1], initial=0.0) + row_residual)
    half_z = float(np.max(half_extents[:, 2], initial=0.0) + column_residual)
    if (
        2.0 * (half_y + coordinate_guard) >= row_pitch
        or 2.0 * (half_z + coordinate_guard) >= column_pitch
    ):
        return None

    locator = PMTGridLocator(
        rows=side,
        columns=side,
        row_pitch=row_pitch,
        column_pitch=column_pitch,
        row_zero_min_y=float(row_zero.min()),
        row_zero_max_y=float(row_zero.max()),
        column_zero_z=float(column_model[0]),
        half_y=half_y,
        half_z=half_z,
        coordinate_guard=coordinate_guard,
        origin_limit=4.0 * (1.0 + world_magnitude),
    )

    # Final containment proof: the scalar descriptor must enclose every
    # outward-padded box even after the affine model is reconstructed.
    for instance in range(len(lower)):
        row, column = divmod(instance, side)
        modeled_y = row_zero[column] + row * row_pitch
        modeled_z = locator.column_zero_z + column * locator.column_pitch
        if (
            lower[instance, 1] < modeled_y - locator.half_y
            or upper[instance, 1] > modeled_y + locator.half_y
            or lower[instance, 2] < modeled_z - locator.half_z
            or upper[instance, 2] > modeled_z + locator.half_z
        ):
            return None
    return locator


def build_pmt_instance_accelerator(
    scene: Any,
    device: Optional[Any] = None,
    *,
    chroma_world_compatibility: bool = False,
) -> PMTInstanceAccelerator:
    """Build and upload one canonical PMT BLAS and all retained transforms.

    ``chroma_world_compatibility`` additionally uploads the exact retained
    world-space vertex blocks emitted by ``Geometry.flatten``.  It is opt-in
    because the normal shared-mesh traversal needs neither their roughly
    2.5 MiB footprint nor their less cache-friendly leaf loads.
    """

    _require_backend()
    (
        vertices,
        triangles,
        channels,
        w2o_r,
        w2o_t,
        o2w_r,
        bounds_min,
        bounds_max,
    ) = _scene_arrays(scene)
    host_bvh = build_packed_bvh(vertices, triangles)
    device_bvh = host_bvh.to_triton(device=device)
    selected_device = device_bvh.nodes.device
    broadphase_min, broadphase_max = _padded_bounds(
        vertices, bounds_min, bounds_max
    )
    grid_locator = _infer_pmt_grid_locator(broadphase_min, broadphase_max)
    # A false positive here only performs the second broadphase stage.  Make
    # the union one ulp wider than the already conservative instance boxes so
    # it is impossible for an individual candidate to be rejected early.
    union_min = np.nextafter(
        broadphase_min.min(axis=0), np.float32(-np.inf)
    ).astype(np.float32)
    union_max = np.nextafter(
        broadphase_max.max(axis=0), np.float32(np.inf)
    ).astype(np.float32)
    (
        tlas_min,
        tlas_max,
        tlas_left,
        tlas_right,
        tlas_instance,
        tlas_stack_capacity,
    ) = _build_instance_tlas(broadphase_min, broadphase_max)

    def upload(value: np.ndarray, dtype: Any):
        # Copy first to avoid PyTorch's warning and mutation hazard for frozen
        # compiler artifacts.
        return torch.from_numpy(
            np.array(value, copy=True, order="C")
        ).to(device=selected_device, dtype=dtype).contiguous()

    if chroma_world_compatibility:
        host_chroma_world_vertices = _build_chroma_world_vertices(scene)
        host_chroma_triangle_indices = np.ascontiguousarray(
            triangles, dtype=np.int32
        )
        chroma_world_vertices = upload(
            host_chroma_world_vertices, torch.float32
        )
        chroma_triangle_indices = upload(
            host_chroma_triangle_indices, torch.int32
        )
    else:
        host_chroma_world_vertices = None
        host_chroma_triangle_indices = None
        chroma_world_vertices = None
        chroma_triangle_indices = None

    # Keep separated PMT walls separated in the initial cull. A single union
    # spanning both walls includes the entire empty detector interior.
    if np.any(broadphase_max[:, 0] < 0) and np.any(broadphase_min[:, 0] > 0):
        positive = broadphase_min[:, 0]+broadphase_max[:, 0] > 0
        coarse_min = np.stack([broadphase_min[mask].min(0) for mask in (~positive, positive)])
        coarse_max = np.stack([broadphase_max[mask].max(0) for mask in (~positive, positive)])
    else:
        coarse_min, coarse_max = union_min[None], union_max[None]
    return PMTInstanceAccelerator(
        host_bvh=host_bvh,
        device_bvh=device_bvh,
        channel_ids=upload(channels, torch.int32),
        world_to_object_rotation=upload(w2o_r, torch.float32),
        world_to_object_translation=upload(w2o_t, torch.float32),
        object_to_world_rotation=upload(o2w_r, torch.float32),
        chroma_world_vertices=chroma_world_vertices,
        chroma_triangle_indices=chroma_triangle_indices,
        bounds_min=upload(broadphase_min, torch.float32),
        bounds_max=upload(broadphase_max, torch.float32),
        union_bounds_min=upload(union_min, torch.float32),
        union_bounds_max=upload(union_max, torch.float32),
        tlas_bounds_min=upload(tlas_min, torch.float32),
        tlas_bounds_max=upload(tlas_max, torch.float32),
        tlas_left_child=upload(tlas_left, torch.int32),
        tlas_right_child=upload(tlas_right, torch.int32),
        tlas_instance=upload(tlas_instance, torch.int32),
        tlas_stack_capacity=tlas_stack_capacity,
        host_channel_ids=np.ascontiguousarray(channels.copy()),
        host_world_to_object_rotation=np.ascontiguousarray(w2o_r.copy()),
        host_world_to_object_translation=np.ascontiguousarray(w2o_t.copy()),
        host_object_to_world_rotation=np.ascontiguousarray(o2w_r.copy()),
        host_chroma_world_vertices=host_chroma_world_vertices,
        host_chroma_triangle_indices=host_chroma_triangle_indices,
        host_bounds_min=broadphase_min,
        host_bounds_max=broadphase_max,
        host_union_bounds_min=np.ascontiguousarray(union_min),
        host_union_bounds_max=np.ascontiguousarray(union_max),
        host_tlas_bounds_min=tlas_min,
        host_tlas_bounds_max=tlas_max,
        host_tlas_left_child=tlas_left,
        host_tlas_right_child=tlas_right,
        host_tlas_instance=tlas_instance,
        grid_locator=grid_locator,
        coarse_bounds_min=upload(coarse_min, torch.float32),
        coarse_bounds_max=upload(coarse_max, torch.float32),
        coarse_box_count=len(coarse_min),
    )


if triton is not None and torch is not None:

    @triton.jit
    def _chroma_intersection_words(
        v0x, v0y, v0z,
        v1x, v1y, v1z,
        v2x, v2y, v2z,
        ox, oy, oz,
        dx, dy, dz,
    ):
        """Return CUDA-fast-math determinant, barycentrics, and distance.

        Chroma compiles ``intersect_triangle`` with ``--use_fast_math``.
        Besides flushing denormals, NVCC keeps cross-product terms as
        separately rounded multiply/subtract instructions and implements
        ``1.0f / determinant`` with ``rcp.approx.ftz.f32``.  Ordinary Triton
        algebra contracts each cross product and emits a full division.  That
        distinction is observable for a ray aimed exactly at a shared PMT
        vertex, where one adjacent face can cross the barycentric tolerance.

        Keep this compatibility-only arithmetic opaque so later compiler
        versions cannot silently change the historical certificate result.
        """

        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .f32 e1x;
                .reg .f32 e1y;
                .reg .f32 e1z;
                .reg .f32 e2x;
                .reg .f32 e2y;
                .reg .f32 e2z;
                .reg .f32 sx;
                .reg .f32 sy;
                .reg .f32 sz;
                .reg .f32 left;
                .reg .f32 right;
                .reg .f32 hx;
                .reg .f32 hy;
                .reg .f32 hz;
                .reg .f32 partial;
                .reg .f32 determinant;
                .reg .f32 reciprocal;
                .reg .f32 dot_value;
                .reg .f32 qx;
                .reg .f32 qy;
                .reg .f32 qz;
                sub.ftz.f32 e1x, $7, $4;
                sub.ftz.f32 e1y, $8, $5;
                sub.ftz.f32 e1z, $9, $6;
                sub.ftz.f32 e2x, $10, $4;
                sub.ftz.f32 e2y, $11, $5;
                sub.ftz.f32 e2z, $12, $6;
                mul.ftz.f32 left, $17, e2z;
                mul.ftz.f32 right, $18, e2y;
                sub.ftz.f32 hx, left, right;
                mul.ftz.f32 left, $18, e2x;
                mul.ftz.f32 right, $16, e2z;
                sub.ftz.f32 hy, left, right;
                mul.ftz.f32 left, $16, e2y;
                mul.ftz.f32 right, $17, e2x;
                sub.ftz.f32 hz, left, right;
                mul.ftz.f32 partial, e1y, hy;
                fma.rn.ftz.f32 determinant, e1x, hx, partial;
                fma.rn.ftz.f32 determinant, e1z, hz, determinant;
                rcp.approx.ftz.f32 reciprocal, determinant;
                sub.ftz.f32 sx, $13, $4;
                sub.ftz.f32 sy, $14, $5;
                sub.ftz.f32 sz, $15, $6;
                mul.ftz.f32 partial, sy, hy;
                fma.rn.ftz.f32 dot_value, sx, hx, partial;
                fma.rn.ftz.f32 dot_value, sz, hz, dot_value;
                mul.ftz.f32 $1, reciprocal, dot_value;
                mul.ftz.f32 left, sy, e1z;
                mul.ftz.f32 right, sz, e1y;
                sub.ftz.f32 qx, left, right;
                mul.ftz.f32 left, sz, e1x;
                mul.ftz.f32 right, sx, e1z;
                sub.ftz.f32 qy, left, right;
                mul.ftz.f32 left, sx, e1y;
                mul.ftz.f32 right, sy, e1x;
                sub.ftz.f32 qz, left, right;
                mul.ftz.f32 partial, $17, qy;
                fma.rn.ftz.f32 dot_value, $16, qx, partial;
                fma.rn.ftz.f32 dot_value, $18, qz, dot_value;
                mul.ftz.f32 $2, reciprocal, dot_value;
                mul.ftz.f32 partial, e2y, qy;
                fma.rn.ftz.f32 dot_value, e2x, qx, partial;
                fma.rn.ftz.f32 dot_value, e2z, qz, dot_value;
                mul.ftz.f32 $3, reciprocal, dot_value;
                mov.f32 $0, determinant;
            }
            """,
            constraints=(
                "=f,=f,=f,=f,"
                "f,f,f,f,f,f,f,f,f,f,f,f,f,f,f"
            ),
            args=[
                v0x, v0y, v0z,
                v1x, v1y, v1z,
                v2x, v2y, v2z,
                ox, oy, oz,
                dx, dy, dz,
            ],
            dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )

    @triton.jit(do_not_specialize=[7, 8])
    def _initialize_instance_results_kernel(
        distances,
        triangles,
        instances,
        channels,
        normals,
        overflow,
        candidate_counts,
        n_rays,
        ray_capacity,
        N_RAYS_IS_POINTER: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_start = tl.program_id(0) * BLOCK_SIZE
        if N_RAYS_IS_POINTER:
            active_count = tl.load(n_rays).to(tl.int32)
            active_count = tl.maximum(
                0, tl.minimum(active_count, ray_capacity)
            )
            if program_start >= active_count:
                return
        else:
            active_count = n_rays
        ray = program_start + tl.arange(0, BLOCK_SIZE)
        valid = ray < active_count
        tl.store(distances + ray, float("inf"), mask=valid)
        tl.store(triangles + ray, -1, mask=valid)
        tl.store(instances + ray, -1, mask=valid)
        tl.store(channels + ray, -1, mask=valid)
        tl.store(normals + ray * 3 + 0, 0.0, mask=valid)
        tl.store(normals + ray * 3 + 1, 0.0, mask=valid)
        tl.store(normals + ray * 3 + 2, 0.0, mask=valid)
        tl.store(overflow + ray, 0, mask=valid)
        tl.store(candidate_counts + ray, 0, mask=valid)

    @triton.jit
    def _instance_aabb_hit(
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
    ):
        # Safe divisors prevent 0/0 NaNs.  Parallel axes are handled by an
        # explicit in-slab test and contribute the unbounded interval.
        nonzero_x = dx != 0.0
        nonzero_y = dy != 0.0
        nonzero_z = dz != 0.0
        safe_dx = tl.where(nonzero_x, dx, 1.0)
        safe_dy = tl.where(nonzero_y, dy, 1.0)
        safe_dz = tl.where(nonzero_z, dz, 1.0)
        tx0, tx1 = (lo_x - ox) / safe_dx, (hi_x - ox) / safe_dx
        ty0, ty1 = (lo_y - oy) / safe_dy, (hi_y - oy) / safe_dy
        tz0, tz1 = (lo_z - oz) / safe_dz, (hi_z - oz) / safe_dz
        tx_near = tl.where(nonzero_x, tl.minimum(tx0, tx1), -float("inf"))
        tx_far = tl.where(nonzero_x, tl.maximum(tx0, tx1), float("inf"))
        ty_near = tl.where(nonzero_y, tl.minimum(ty0, ty1), -float("inf"))
        ty_far = tl.where(nonzero_y, tl.maximum(ty0, ty1), float("inf"))
        tz_near = tl.where(nonzero_z, tl.minimum(tz0, tz1), -float("inf"))
        tz_far = tl.where(nonzero_z, tl.maximum(tz0, tz1), float("inf"))
        parallel_ok = (
            (nonzero_x | ((ox >= lo_x) & (ox <= hi_x)))
            & (nonzero_y | ((oy >= lo_y) & (oy <= hi_y)))
            & (nonzero_z | ((oz >= lo_z) & (oz <= hi_z)))
        )
        near = tl.maximum(tl.maximum(tx_near, ty_near), tz_near)
        far = tl.minimum(tl.minimum(tx_far, ty_far), tz_far)
        return parallel_ok & (far >= tl.maximum(near, 0.0)) & (
            tl.maximum(near, 0.0) <= ray_tmax
        )

    @triton.jit
    def _grid_candidate_range(
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
        GRID_ROWS: tl.constexpr,
        GRID_COLUMNS: tl.constexpr,
    ):
        """Return a conservative row/column rectangle for one ray.

        The interval is clipped to the complete PMT union before projection,
        so even a nearly parallel ray does not inflate the common case.  A
        non-finite interval is reported separately and sent to the TLAS.
        """

        lo_x = tl.load(union_bounds_min + 0)
        lo_y = tl.load(union_bounds_min + 1)
        lo_z = tl.load(union_bounds_min + 2)
        hi_x = tl.load(union_bounds_max + 0)
        hi_y = tl.load(union_bounds_max + 1)
        hi_z = tl.load(union_bounds_max + 2)
        nonzero_x = dx != 0.0
        nonzero_y = dy != 0.0
        nonzero_z = dz != 0.0
        safe_dx = tl.where(nonzero_x, dx, 1.0)
        safe_dy = tl.where(nonzero_y, dy, 1.0)
        safe_dz = tl.where(nonzero_z, dz, 1.0)
        tx0, tx1 = (lo_x - ox) / safe_dx, (hi_x - ox) / safe_dx
        ty0, ty1 = (lo_y - oy) / safe_dy, (hi_y - oy) / safe_dy
        tz0, tz1 = (lo_z - oz) / safe_dz, (hi_z - oz) / safe_dz
        tx_near = tl.where(nonzero_x, tl.minimum(tx0, tx1), -float("inf"))
        tx_far = tl.where(nonzero_x, tl.maximum(tx0, tx1), float("inf"))
        ty_near = tl.where(nonzero_y, tl.minimum(ty0, ty1), -float("inf"))
        ty_far = tl.where(nonzero_y, tl.maximum(ty0, ty1), float("inf"))
        tz_near = tl.where(nonzero_z, tl.minimum(tz0, tz1), -float("inf"))
        tz_far = tl.where(nonzero_z, tl.maximum(tz0, tz1), float("inf"))
        parallel_ok = (
            (nonzero_x | ((ox >= lo_x) & (ox <= hi_x)))
            & (nonzero_y | ((oy >= lo_y) & (oy <= hi_y)))
            & (nonzero_z | ((oz >= lo_z) & (oz <= hi_z)))
        )
        near = tl.maximum(tl.maximum(tx_near, ty_near), tz_near)
        far = tl.minimum(tl.minimum(tx_far, ty_far), tz_far)
        enter = tl.maximum(near, 0.0)
        leave = tl.minimum(far, ray_tmax)
        interval_valid = parallel_ok & (leave >= enter)
        direction_norm2 = dx * dx + dy * dy + dz * dz
        input_safe = (
            (tl.abs(ox) <= origin_limit)
            & (tl.abs(oy) <= origin_limit)
            & (tl.abs(oz) <= origin_limit)
            & (direction_norm2 >= 0.5)
            & (direction_norm2 <= 2.0)
        )
        interval_finite = (
            interval_valid
            & input_safe
            & (enter > -float("inf"))
            & (enter < float("inf"))
            & (leave > -float("inf"))
            & (leave < float("inf"))
        )
        safe_enter = tl.where(interval_finite, enter, 0.0)
        safe_leave = tl.where(interval_finite, leave, 0.0)
        y_enter = oy + safe_enter * dy
        y_leave = oy + safe_leave * dy
        z_enter = oz + safe_enter * dz
        z_leave = oz + safe_leave * dz
        y_min = tl.minimum(y_enter, y_leave) - half_y - coordinate_guard
        y_max = tl.maximum(y_enter, y_leave) + half_y + coordinate_guard
        z_min = tl.minimum(z_enter, z_leave) - half_z - coordinate_guard
        z_max = tl.maximum(z_enter, z_leave) + half_z + coordinate_guard

        row_first = tl.ceil(
            (y_min - row_zero_max_y) / row_pitch
        ).to(tl.int32)
        row_stop = (
            tl.floor((y_max - row_zero_min_y) / row_pitch).to(tl.int32) + 1
        )
        column_first = tl.ceil(
            (z_min - column_zero_z) / column_pitch
        ).to(tl.int32)
        column_stop = (
            tl.floor((z_max - column_zero_z) / column_pitch).to(tl.int32)
            + 1
        )
        row_first = tl.maximum(0, tl.minimum(GRID_ROWS, row_first))
        row_stop = tl.maximum(0, tl.minimum(GRID_ROWS, row_stop))
        column_first = tl.maximum(
            0, tl.minimum(GRID_COLUMNS, column_first)
        )
        column_stop = tl.maximum(
            0, tl.minimum(GRID_COLUMNS, column_stop)
        )
        row_count = tl.maximum(row_stop - row_first, 0)
        column_count = tl.maximum(column_stop - column_first, 0)
        row_count = tl.where(interval_finite, row_count, 0)
        column_count = tl.where(interval_finite, column_count, 0)
        return (
            row_first,
            row_count,
            column_first,
            column_count,
            interval_finite,
        )

    @triton.jit
    def _canonical_blas_hit(
        nodes,
        triangle_vertices,
        chroma_world_vertices,
        chroma_triangle_indices,
        leaf_instance,
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        world_ox,
        world_oy,
        world_oz,
        world_dx,
        world_dy,
        world_dz,
        upper_bound,
        previous_triangle,
        requested,
        stack,
        stack_stride,
        ray,
        world_x,
        world_y,
        world_z,
        world_scale,
        N_WORLD_VERTICES: tl.constexpr,
        CHROMA_WORLD_GEOMETRY: tl.constexpr,
        STACK_CAPACITY: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Chroma-compatible canonical BLAS traversal for TLAS leaf lanes."""

        invx, invy, invz = 1.0 / dx, 1.0 / dy, 1.0 / dz
        neg_ox_inv = -ox / dx
        neg_oy_inv = -oy / dy
        neg_oz_inv = -oz / dz
        node_index = tl.zeros((BLOCK_SIZE,), tl.int32)
        range_remaining = tl.full((BLOCK_SIZE,), 1, tl.int32)
        stack_pointer = tl.zeros((BLOCK_SIZE,), tl.int32)
        active = requested
        best_triangle = tl.full((BLOCK_SIZE,), -1, tl.int32)
        best_distance = upper_bound
        overflow = tl.zeros((BLOCK_SIZE,), tl.int1)

        # This is the same eager-sibling state machine as
        # chroma.triton.bvh_kernels._nearest_hit_kernel.  Keeping the operation
        # order here is intentional: a fused TLAS must not perturb which
        # canonical triangle wins an edge tie.
        while tl.sum(active.to(tl.int32), axis=0) != 0:
            packed_x = tl.load(
                nodes + node_index * 4 + 0, mask=active, other=0
            ).to(tl.uint32)
            packed_y = tl.load(
                nodes + node_index * 4 + 1, mask=active, other=0
            ).to(tl.uint32)
            packed_z = tl.load(
                nodes + node_index * 4 + 2, mask=active, other=0
            ).to(tl.uint32)
            packed_w = tl.load(
                nodes + node_index * 4 + 3, mask=active, other=0
            ).to(tl.uint32)

            xlo = (packed_x & 0xFFFF).to(tl.float32) * world_scale + world_x
            xhi = (packed_x >> 16).to(tl.float32) * world_scale + world_x
            ylo = (packed_y & 0xFFFF).to(tl.float32) * world_scale + world_y
            yhi = (packed_y >> 16).to(tl.float32) * world_scale + world_y
            zlo = (packed_z & 0xFFFF).to(tl.float32) * world_scale + world_z
            zhi = (packed_z >> 16).to(tl.float32) * world_scale + world_z
            tx0 = xlo * invx + neg_ox_inv
            tx1 = xhi * invx + neg_ox_inv
            ty0 = ylo * invy + neg_oy_inv
            ty1 = yhi * invy + neg_oy_inv
            tz0 = zlo * invz + neg_oz_inv
            tz1 = zhi * invz + neg_oz_inv
            txmin = tl.where(dx != 0.0, tl.minimum(tx0, tx1), 0.0)
            txmax = tl.where(dx != 0.0, tl.maximum(tx0, tx1), float("inf"))
            tymin = tl.where(dy != 0.0, tl.minimum(ty0, ty1), 0.0)
            tymax = tl.where(dy != 0.0, tl.maximum(ty0, ty1), float("inf"))
            tzmin = tl.where(dz != 0.0, tl.minimum(tz0, tz1), 0.0)
            tzmax = tl.where(dz != 0.0, tl.maximum(tz0, tz1), float("inf"))
            box_near = tl.maximum(
                tl.maximum(txmin, tymin), tl.maximum(tzmin, 0.0)
            )
            box_far = tl.minimum(tl.minimum(txmax, tymax), tzmax)
            box_hit = (
                active & (box_near <= box_far) & (box_near <= best_distance)
            )

            child_count = (packed_w >> 28).to(tl.int32)
            child = (packed_w & 0x0FFFFFFF).to(tl.int32)
            is_leaf = (
                box_hit
                & (child_count == 0)
                & (child != previous_triangle)
            )
            is_inner = box_hit & (child_count != 0)
            if CHROMA_WORLD_GEOMETRY:
                # Chroma intersects the float32 world vertices produced by
                # Geometry.flatten, not a ray transformed into a shared local
                # mesh.  Evaluate *every visited leaf* in that representation
                # so world-space rounding participates in winner selection.
                index_base = child * 3
                i0 = tl.load(
                    chroma_triangle_indices + index_base,
                    mask=is_leaf,
                    other=0,
                ).to(tl.int32)
                i1 = tl.load(
                    chroma_triangle_indices + index_base + 1,
                    mask=is_leaf,
                    other=0,
                ).to(tl.int32)
                i2 = tl.load(
                    chroma_triangle_indices + index_base + 2,
                    mask=is_leaf,
                    other=0,
                ).to(tl.int32)
                instance_base = leaf_instance * N_WORLD_VERTICES
                v0 = (instance_base + i0) * 3
                v1 = (instance_base + i1) * 3
                v2 = (instance_base + i2) * 3
                v0x = tl.load(chroma_world_vertices + v0, mask=is_leaf, other=0.0)
                v0y = tl.load(
                    chroma_world_vertices + v0 + 1, mask=is_leaf, other=0.0
                )
                v0z = tl.load(
                    chroma_world_vertices + v0 + 2, mask=is_leaf, other=0.0
                )
                v1x = tl.load(chroma_world_vertices + v1, mask=is_leaf, other=0.0)
                v1y = tl.load(
                    chroma_world_vertices + v1 + 1, mask=is_leaf, other=0.0
                )
                v1z = tl.load(
                    chroma_world_vertices + v1 + 2, mask=is_leaf, other=0.0
                )
                v2x = tl.load(chroma_world_vertices + v2, mask=is_leaf, other=0.0)
                v2y = tl.load(
                    chroma_world_vertices + v2 + 1, mask=is_leaf, other=0.0
                )
                v2z = tl.load(
                    chroma_world_vertices + v2 + 2, mask=is_leaf, other=0.0
                )
                test_ox, test_oy, test_oz = world_ox, world_oy, world_oz
                test_dx, test_dy, test_dz = world_dx, world_dy, world_dz
            else:
                triangle_base = child * 9
                v0x = tl.load(
                    triangle_vertices + triangle_base + 0,
                    mask=is_leaf,
                    other=0.0,
                )
                v0y = tl.load(
                    triangle_vertices + triangle_base + 1,
                    mask=is_leaf,
                    other=0.0,
                )
                v0z = tl.load(
                    triangle_vertices + triangle_base + 2,
                    mask=is_leaf,
                    other=0.0,
                )
                v1x = tl.load(
                    triangle_vertices + triangle_base + 3,
                    mask=is_leaf,
                    other=0.0,
                )
                v1y = tl.load(
                    triangle_vertices + triangle_base + 4,
                    mask=is_leaf,
                    other=0.0,
                )
                v1z = tl.load(
                    triangle_vertices + triangle_base + 5,
                    mask=is_leaf,
                    other=0.0,
                )
                v2x = tl.load(
                    triangle_vertices + triangle_base + 6,
                    mask=is_leaf,
                    other=0.0,
                )
                v2y = tl.load(
                    triangle_vertices + triangle_base + 7,
                    mask=is_leaf,
                    other=0.0,
                )
                v2z = tl.load(
                    triangle_vertices + triangle_base + 8,
                    mask=is_leaf,
                    other=0.0,
                )
                test_ox, test_oy, test_oz = ox, oy, oz
                test_dx, test_dy, test_dz = dx, dy, dz
            if CHROMA_WORLD_GEOMETRY:
                determinant, u, v, distance = _chroma_intersection_words(
                    v0x,
                    v0y,
                    v0z,
                    v1x,
                    v1y,
                    v1z,
                    v2x,
                    v2y,
                    v2z,
                    test_ox,
                    test_oy,
                    test_oz,
                    test_dx,
                    test_dy,
                    test_dz,
                )
            else:
                edge1x, edge1y, edge1z = v1x - v0x, v1y - v0y, v1z - v0z
                edge2x, edge2y, edge2z = v2x - v0x, v2y - v0y, v2z - v0z
                hx = test_dy * edge2z - test_dz * edge2y
                hy = test_dz * edge2x - test_dx * edge2z
                hz = test_dx * edge2y - test_dy * edge2x
                determinant = edge1x * hx + edge1y * hy + edge1z * hz
                reciprocal = 1.0 / determinant
                sx, sy, sz = test_ox - v0x, test_oy - v0y, test_oz - v0z
                u = reciprocal * (sx * hx + sy * hy + sz * hz)
                qx = sy * edge1z - sz * edge1y
                qy = sz * edge1x - sx * edge1z
                qz = sx * edge1y - sy * edge1x
                v = reciprocal * (
                    test_dx * qx + test_dy * qy + test_dz * qz
                )
                distance = reciprocal * (
                    edge2x * qx + edge2y * qy + edge2z * qz
                )
            determinant_ok = (determinant < -1.1920928955078125e-7) | (
                determinant > 1.1920928955078125e-7
            )
            # A nominally positive intersection can be smaller than one ulp
            # of a detector-scale float32 position.  Accepting it would let
            # transport change direction/last-triangle while leaving position
            # and time bitwise fixed, producing an adjacent-triangle loop.
            # Test the exact world-space update used by transport: at least one
            # component must advance to a different representable float.
            if CHROMA_WORLD_GEOMETRY:
                # Compatibility mode represents unmodified Chroma.  Its mesh
                # kernel accepts any t > CHROMA_EPSILON, including a movement
                # that later rounds to the same detector-scale position.  The
                # production path deliberately retains the loop-prevention fix.
                advances = is_leaf
            else:
                advances = (
                    ((world_ox + distance * world_dx) != world_ox)
                    | ((world_oy + distance * world_dy) != world_oy)
                    | ((world_oz + distance * world_dz) != world_oz)
                )
            triangle_hit = (
                is_leaf
                & determinant_ok
                & (u >= -1.0e-6)
                & (u <= 1.0 + 1.0e-6)
                & (v >= -1.0e-6)
                & ((u + v) <= 1.0 + 1.0e-6)
                & (distance > 1.0e-6)
                & (distance < best_distance)
                & advances
            )
            best_distance = tl.where(triangle_hit, distance, best_distance)
            best_triangle = tl.where(triangle_hit, child, best_triangle)

            can_push = stack_pointer < STACK_CAPACITY
            push = is_inner & can_push
            overflow |= is_inner & ~can_push
            packed_range = child | (child_count << 28)
            push_address = stack_pointer * stack_stride + ray
            tl.store(stack + push_address, packed_range, mask=push)
            pushed_pointer = stack_pointer + push.to(tl.int32)
            next_remaining = range_remaining - 1
            stay_in_range = active & (next_remaining > 0) & ~overflow
            should_pop = active & ~stay_in_range & ~overflow
            has_pending = should_pop & (pushed_pointer > 0)
            top = tl.maximum(pushed_pointer - 1, 0)
            popped = tl.load(
                stack + top * stack_stride + ray, mask=has_pending, other=0
            ).to(tl.uint32)
            popped_first = (popped & 0x0FFFFFFF).to(tl.int32)
            popped_count = (popped >> 28).to(tl.int32)
            stack_pointer = pushed_pointer - has_pending.to(tl.int32)
            node_index = tl.where(
                stay_in_range, node_index + 1, popped_first
            )
            range_remaining = tl.where(
                stay_in_range, next_remaining, popped_count
            )
            active = stay_in_range | has_pending

        return best_triangle, best_distance, overflow

    @triton.jit(do_not_specialize=[24, 25])
    def _nearest_pmt_tlas_kernel(
        tlas_bounds_min,
        tlas_bounds_max,
        tlas_left_child,
        tlas_right_child,
        tlas_instance,
        world_to_object_rotation,
        world_to_object_translation,
        blas_nodes,
        triangle_vertices,
        chroma_world_vertices,
        chroma_triangle_indices,
        origins,
        directions,
        tmax_values,
        last_instances,
        last_triangles,
        tlas_stack,
        blas_stack,
        out_triangles,
        out_distances,
        out_instances,
        out_overflow,
        out_candidate_counts,
        sticky_overflow,
        n_rays,
        ray_stride,
        ray_index_ptr,
        world_x,
        world_y,
        world_z,
        world_scale,
        N_WORLD_VERTICES: tl.constexpr,
        CHROMA_WORLD_GEOMETRY: tl.constexpr,
        N_RAYS_IS_POINTER: tl.constexpr,
        INDIRECT: tl.constexpr,
        TLAS_STACK_CAPACITY: tl.constexpr,
        BLAS_STACK_CAPACITY: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Traverse instance TLAS and canonical BLAS without a host rendezvous."""

        program_start = tl.program_id(0) * BLOCK_SIZE
        if N_RAYS_IS_POINTER:
            active_count = tl.load(n_rays).to(tl.int32)
            active_count = tl.maximum(
                0, tl.minimum(active_count, ray_stride)
            )
            # Fixed-capacity device scheduling launches an upper-bound grid.
            # Exit before either hierarchy walk for an entirely empty CTA.
            if program_start >= active_count:
                return
        else:
            active_count = n_rays
        # ``tl.num_programs`` is the host-selected resident grid.  Every
        # program retains its lanes and walks later tiles in grid strides,
        # so the device-resident count controls work without a host poll or a
        # capacity-sized tail of empty CTAs.
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
            best_distance = ray_tmax
            best_triangle = tl.full((BLOCK_SIZE,), -1, tl.int32)
            best_instance = tl.full((BLOCK_SIZE,), -1, tl.int32)
            candidate_count = tl.zeros((BLOCK_SIZE,), tl.int32)
            overflow = tl.zeros((BLOCK_SIZE,), tl.int1)

            node = tl.zeros((BLOCK_SIZE,), tl.int32)
            stack_pointer = tl.zeros((BLOCK_SIZE,), tl.int32)
            active = valid
            while tl.sum(active.to(tl.int32), axis=0) != 0:
                lo_x = tl.load(
                    tlas_bounds_min + node * 3 + 0,
                    mask=active,
                    other=0.0,
                )
                lo_y = tl.load(
                    tlas_bounds_min + node * 3 + 1,
                    mask=active,
                    other=0.0,
                )
                lo_z = tl.load(
                    tlas_bounds_min + node * 3 + 2,
                    mask=active,
                    other=0.0,
                )
                hi_x = tl.load(
                    tlas_bounds_max + node * 3 + 0,
                    mask=active,
                    other=0.0,
                )
                hi_y = tl.load(
                    tlas_bounds_max + node * 3 + 1,
                    mask=active,
                    other=0.0,
                )
                hi_z = tl.load(
                    tlas_bounds_max + node * 3 + 2,
                    mask=active,
                    other=0.0,
                )
                node_hit = active & _instance_aabb_hit(
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
                instance = tl.load(
                    tlas_instance + node, mask=node_hit, other=-1
                )
                leaf = node_hit & (instance >= 0)
                inner = node_hit & (instance < 0)
                candidate_count += leaf.to(tl.int32)

                # Only a leaf that can tie or improve the current result needs
                # a BLAS walk.  Count against original tmax so diagnostics are
                # identical to the all-instance reference.
                could_improve = leaf & _instance_aabb_hit(
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
                rotation = instance * 9
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
                    world_to_object_translation + instance * 3 + 0,
                    mask=could_improve,
                    other=0.0,
                )
                ty = tl.load(
                    world_to_object_translation + instance * 3 + 1,
                    mask=could_improve,
                    other=0.0,
                )
                tz = tl.load(
                    world_to_object_translation + instance * 3 + 2,
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
                    previous_instance == instance, previous_triangle, -1
                )
                local_triangle, local_distance, blas_overflow = (
                    _canonical_blas_hit(
                        blas_nodes,
                        triangle_vertices,
                        chroma_world_vertices,
                        chroma_triangle_indices,
                        instance,
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
                        N_WORLD_VERTICES=N_WORLD_VERTICES,
                        CHROMA_WORLD_GEOMETRY=CHROMA_WORLD_GEOMETRY,
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
                    improved, instance, best_instance
                )
                overflow |= blas_overflow

                can_push = stack_pointer < TLAS_STACK_CAPACITY
                push = inner & can_push
                overflow |= inner & ~can_push
                right = tl.load(
                    tlas_right_child + node, mask=push, other=0
                )
                left = tl.load(
                    tlas_left_child + node, mask=push, other=0
                )
                tl.store(
                    tlas_stack + stack_pointer * ray_stride + work_slot,
                    right,
                    mask=push,
                )
                pushed_pointer = stack_pointer + push.to(tl.int32)
                descend = push
                needs_pop = active & ~descend & ~overflow
                has_pending = needs_pop & (pushed_pointer > 0)
                top = tl.maximum(pushed_pointer - 1, 0)
                popped = tl.load(
                    tlas_stack + top * ray_stride + work_slot,
                    mask=has_pending,
                    other=0,
                )
                stack_pointer = pushed_pointer - has_pending.to(tl.int32)
                node = tl.where(descend, left, popped)
                active = descend | has_pending

            tl.store(out_triangles + ray, best_triangle, mask=valid)
            tl.store(
                out_distances + ray,
                tl.where(best_triangle >= 0, best_distance, float("inf")),
                mask=valid,
            )
            tl.store(out_instances + ray, best_instance, mask=valid)
            tl.store(
                out_overflow + ray, overflow.to(tl.uint8), mask=valid
            )
            tl.store(
                out_candidate_counts + ray, candidate_count, mask=valid
            )
            tl.atomic_or(
                sticky_overflow + tl.zeros((BLOCK_SIZE,), tl.int32),
                tl.full((BLOCK_SIZE,), 1, tl.int32),
                mask=valid & overflow,
            )
            program_start += program_stride

    @triton.jit(do_not_specialize=[17])
    def _nearest_hit_candidates_progress_kernel(
        nodes,
        triangle_vertices,
        chroma_world_vertices,
        chroma_triangle_indices,
        local_origins,
        local_directions,
        tmax_values,
        last_hit_values,
        candidate_ray_ids,
        candidate_instances,
        world_origins,
        world_directions,
        stack,
        out_triangle,
        out_distance,
        out_overflow,
        sticky_overflow,
        n_candidates,
        world_x,
        world_y,
        world_z,
        world_scale,
        N_WORLD_VERTICES: tl.constexpr,
        CHROMA_WORLD_GEOMETRY: tl.constexpr,
        STACK_CAPACITY: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        candidate = (
            tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        )
        valid = candidate < n_candidates
        ox = tl.load(
            local_origins + candidate * 3 + 0, mask=valid, other=0.0
        )
        oy = tl.load(
            local_origins + candidate * 3 + 1, mask=valid, other=0.0
        )
        oz = tl.load(
            local_origins + candidate * 3 + 2, mask=valid, other=0.0
        )
        dx = tl.load(
            local_directions + candidate * 3 + 0, mask=valid, other=1.0
        )
        dy = tl.load(
            local_directions + candidate * 3 + 1, mask=valid, other=1.0
        )
        dz = tl.load(
            local_directions + candidate * 3 + 2, mask=valid, other=1.0
        )
        upper_bound = tl.load(
            tmax_values + candidate, mask=valid, other=-1.0
        )
        previous_triangle = tl.load(
            last_hit_values + candidate, mask=valid, other=-1
        )
        ray_id = tl.load(candidate_ray_ids + candidate, mask=valid, other=0)
        instance = tl.load(
            candidate_instances + candidate, mask=valid, other=0
        )
        world_ox = tl.load(
            world_origins + ray_id * 3 + 0, mask=valid, other=0.0
        )
        world_oy = tl.load(
            world_origins + ray_id * 3 + 1, mask=valid, other=0.0
        )
        world_oz = tl.load(
            world_origins + ray_id * 3 + 2, mask=valid, other=0.0
        )
        world_dx = tl.load(
            world_directions + ray_id * 3 + 0, mask=valid, other=1.0
        )
        world_dy = tl.load(
            world_directions + ray_id * 3 + 1, mask=valid, other=1.0
        )
        world_dz = tl.load(
            world_directions + ray_id * 3 + 2, mask=valid, other=1.0
        )
        triangle, distance, overflow = _canonical_blas_hit(
            nodes,
            triangle_vertices,
            chroma_world_vertices,
            chroma_triangle_indices,
            instance,
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
            world_ox,
            world_oy,
            world_oz,
            world_dx,
            world_dy,
            world_dz,
            upper_bound,
            previous_triangle,
            valid,
            stack,
            n_candidates,
            candidate,
            world_x,
            world_y,
            world_z,
            world_scale,
            N_WORLD_VERTICES=N_WORLD_VERTICES,
            CHROMA_WORLD_GEOMETRY=CHROMA_WORLD_GEOMETRY,
            STACK_CAPACITY=STACK_CAPACITY,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        tl.store(out_triangle + candidate, triangle, mask=valid)
        tl.store(
            out_distance + candidate,
            tl.where(triangle >= 0, distance, float("inf")),
            mask=valid,
        )
        tl.store(out_overflow + candidate, overflow.to(tl.uint8), mask=valid)
        tl.atomic_or(
            sticky_overflow + tl.zeros((BLOCK_SIZE,), tl.int32),
            tl.full((BLOCK_SIZE,), 1, tl.int32),
            mask=valid & overflow,
        )

    @triton.jit(do_not_specialize=[6])
    def _mark_union_candidates_kernel(
        origins,
        directions,
        tmax_values,
        union_bounds_min,
        union_bounds_max,
        flags,
        n_rays,
        BLOCK_SIZE: tl.constexpr,
        NBOX: tl.constexpr = 1,
    ):
        ray = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = ray < n_rays
        ox = tl.load(origins + ray * 3 + 0, mask=valid, other=0.0)
        oy = tl.load(origins + ray * 3 + 1, mask=valid, other=0.0)
        oz = tl.load(origins + ray * 3 + 2, mask=valid, other=0.0)
        dx = tl.load(directions + ray * 3 + 0, mask=valid, other=0.0)
        dy = tl.load(directions + ray * 3 + 1, mask=valid, other=0.0)
        dz = tl.load(directions + ray * 3 + 2, mask=valid, other=0.0)
        ray_tmax = tl.load(tmax_values + ray, mask=valid, other=-1.0)
        hit = tl.full((BLOCK_SIZE,), False, tl.int1)
        for box in tl.static_range(NBOX):
            hit |= valid & _instance_aabb_hit(
                ox, oy, oz, dx, dy, dz,
                tl.load(union_bounds_min + box*3),
                tl.load(union_bounds_min + box*3+1),
                tl.load(union_bounds_min + box*3+2),
                tl.load(union_bounds_max + box*3),
                tl.load(union_bounds_max + box*3+1),
                tl.load(union_bounds_max + box*3+2), ray_tmax)
        tl.store(flags + ray, hit.to(tl.int32), mask=valid)

    @triton.jit(do_not_specialize=[8])
    def _compact_union_candidates_device_count_kernel(
        origins,
        directions,
        tmax_values,
        union_bounds_min,
        union_bounds_max,
        active_count,
        candidate_ray_ids,
        candidate_count,
        ray_capacity,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compact union-AABB hits without a prefix sum or host count read.

        One atomic reservation per block preserves ascending ray-row order
        within that block.  Inter-block reservation order is intentionally
        unspecified; every TLAS traversal is independent and scatters back to
        its original row, so this cannot affect hit/tie semantics.
        """

        program_start = tl.program_id(0) * BLOCK_SIZE
        live_count = tl.load(active_count).to(tl.int32)
        live_count = tl.maximum(0, tl.minimum(live_count, ray_capacity))
        if program_start >= live_count:
            return
        local_lane = tl.arange(0, BLOCK_SIZE)
        ray = program_start + local_lane
        valid = ray < live_count
        ox = tl.load(origins + ray * 3, mask=valid, other=0.0)
        oy = tl.load(origins + ray * 3 + 1, mask=valid, other=0.0)
        oz = tl.load(origins + ray * 3 + 2, mask=valid, other=0.0)
        dx = tl.load(directions + ray * 3, mask=valid, other=0.0)
        dy = tl.load(directions + ray * 3 + 1, mask=valid, other=0.0)
        dz = tl.load(directions + ray * 3 + 2, mask=valid, other=0.0)
        ray_tmax = tl.load(tmax_values + ray, mask=valid, other=-1.0)
        hit = valid & _instance_aabb_hit(
            ox,
            oy,
            oz,
            dx,
            dy,
            dz,
            tl.load(union_bounds_min),
            tl.load(union_bounds_min + 1),
            tl.load(union_bounds_min + 2),
            tl.load(union_bounds_max),
            tl.load(union_bounds_max + 1),
            tl.load(union_bounds_max + 2),
            ray_tmax,
        )
        flag = hit.to(tl.int32)
        local_rank = tl.cumsum(flag, axis=0)
        block_count = tl.sum(flag, axis=0)
        zero = tl.zeros((BLOCK_SIZE,), tl.int32)
        atomic_old = tl.atomic_add(
            candidate_count + zero,
            block_count + zero,
            mask=local_lane == 0,
        )
        block_base = tl.sum(
            tl.where(local_lane == 0, atomic_old, 0), axis=0
        )
        tl.store(
            candidate_ray_ids + block_base + local_rank - 1,
            ray,
            mask=hit,
        )

    @triton.jit(do_not_specialize=[3])
    def _compact_union_candidates_kernel(
        flags,
        offsets,
        active_ray_ids,
        n_rays,
        BLOCK_SIZE: tl.constexpr,
    ):
        ray = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = ray < n_rays
        hit = valid & (tl.load(flags + ray, mask=valid, other=0) != 0)
        destination = tl.load(offsets + ray, mask=valid, other=0)
        tl.store(active_ray_ids + destination, ray, mask=hit)

    @triton.jit(do_not_specialize=[7])
    def _count_instance_candidates_kernel(
        origins,
        directions,
        tmax_values,
        active_ray_ids,
        bounds_min,
        bounds_max,
        counts,
        n_active_rays,
        N_INSTANCES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        active_ray = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = active_ray < n_active_rays
        ray = tl.load(active_ray_ids + active_ray, mask=valid, other=0)
        ox = tl.load(origins + ray * 3 + 0, mask=valid, other=0.0)
        oy = tl.load(origins + ray * 3 + 1, mask=valid, other=0.0)
        oz = tl.load(origins + ray * 3 + 2, mask=valid, other=0.0)
        dx = tl.load(directions + ray * 3 + 0, mask=valid, other=0.0)
        dy = tl.load(directions + ray * 3 + 1, mask=valid, other=0.0)
        dz = tl.load(directions + ray * 3 + 2, mask=valid, other=0.0)
        ray_tmax = tl.load(tmax_values + ray, mask=valid, other=-1.0)
        count = tl.zeros((BLOCK_SIZE,), tl.int32)
        for instance in range(N_INSTANCES):
            lo_x = tl.load(bounds_min + instance * 3 + 0)
            lo_y = tl.load(bounds_min + instance * 3 + 1)
            lo_z = tl.load(bounds_min + instance * 3 + 2)
            hi_x = tl.load(bounds_max + instance * 3 + 0)
            hi_y = tl.load(bounds_max + instance * 3 + 1)
            hi_z = tl.load(bounds_max + instance * 3 + 2)
            hit = valid & _instance_aabb_hit(
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
            count += hit.to(tl.int32)
        tl.store(counts + active_ray, count, mask=valid)

    @triton.jit(do_not_specialize=[17])
    def _fill_instance_candidates_kernel(
        origins,
        directions,
        tmax_values,
        last_instances,
        last_triangles,
        active_ray_ids,
        bounds_min,
        bounds_max,
        world_to_object_rotation,
        world_to_object_translation,
        offsets,
        local_origins,
        local_directions,
        candidate_tmax,
        candidate_last_triangle,
        candidate_instances,
        candidate_ray_ids,
        n_active_rays,
        N_INSTANCES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        active_ray = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = active_ray < n_active_rays
        ray = tl.load(active_ray_ids + active_ray, mask=valid, other=0)
        ox = tl.load(origins + ray * 3 + 0, mask=valid, other=0.0)
        oy = tl.load(origins + ray * 3 + 1, mask=valid, other=0.0)
        oz = tl.load(origins + ray * 3 + 2, mask=valid, other=0.0)
        dx = tl.load(directions + ray * 3 + 0, mask=valid, other=0.0)
        dy = tl.load(directions + ray * 3 + 1, mask=valid, other=0.0)
        dz = tl.load(directions + ray * 3 + 2, mask=valid, other=0.0)
        ray_tmax = tl.load(tmax_values + ray, mask=valid, other=-1.0)
        previous_instance = tl.load(
            last_instances + ray, mask=valid, other=-1
        )
        previous_triangle = tl.load(
            last_triangles + ray, mask=valid, other=-1
        )
        base = tl.load(offsets + active_ray, mask=valid, other=0)
        written = tl.zeros((BLOCK_SIZE,), tl.int64)

        for instance in range(N_INSTANCES):
            lo_x = tl.load(bounds_min + instance * 3 + 0)
            lo_y = tl.load(bounds_min + instance * 3 + 1)
            lo_z = tl.load(bounds_min + instance * 3 + 2)
            hi_x = tl.load(bounds_max + instance * 3 + 0)
            hi_y = tl.load(bounds_max + instance * 3 + 1)
            hi_z = tl.load(bounds_max + instance * 3 + 2)
            hit = valid & _instance_aabb_hit(
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
            destination = base + written
            rotation = instance * 9
            r00 = tl.load(world_to_object_rotation + rotation + 0)
            r01 = tl.load(world_to_object_rotation + rotation + 1)
            r02 = tl.load(world_to_object_rotation + rotation + 2)
            r10 = tl.load(world_to_object_rotation + rotation + 3)
            r11 = tl.load(world_to_object_rotation + rotation + 4)
            r12 = tl.load(world_to_object_rotation + rotation + 5)
            r20 = tl.load(world_to_object_rotation + rotation + 6)
            r21 = tl.load(world_to_object_rotation + rotation + 7)
            r22 = tl.load(world_to_object_rotation + rotation + 8)
            tx = tl.load(world_to_object_translation + instance * 3 + 0)
            ty = tl.load(world_to_object_translation + instance * 3 + 1)
            tz = tl.load(world_to_object_translation + instance * 3 + 2)
            local_ox = r00 * ox + r01 * oy + r02 * oz + tx
            local_oy = r10 * ox + r11 * oy + r12 * oz + ty
            local_oz = r20 * ox + r21 * oy + r22 * oz + tz
            local_dx = r00 * dx + r01 * dy + r02 * dz
            local_dy = r10 * dx + r11 * dy + r12 * dz
            local_dz = r20 * dx + r21 * dy + r22 * dz
            tl.store(local_origins + destination * 3 + 0, local_ox, mask=hit)
            tl.store(local_origins + destination * 3 + 1, local_oy, mask=hit)
            tl.store(local_origins + destination * 3 + 2, local_oz, mask=hit)
            tl.store(
                local_directions + destination * 3 + 0, local_dx, mask=hit
            )
            tl.store(
                local_directions + destination * 3 + 1, local_dy, mask=hit
            )
            tl.store(
                local_directions + destination * 3 + 2, local_dz, mask=hit
            )
            tl.store(candidate_tmax + destination, ray_tmax, mask=hit)
            excluded = tl.where(
                previous_instance == instance, previous_triangle, -1
            )
            tl.store(
                candidate_last_triangle + destination, excluded, mask=hit
            )
            tl.store(candidate_instances + destination, instance, mask=hit)
            tl.store(candidate_ray_ids + destination, ray, mask=hit)
            written += hit.to(tl.int64)

    @triton.jit(do_not_specialize=[7])
    def _count_grid_candidates_kernel(
        origins,
        directions,
        tmax_values,
        active_ray_ids,
        bounds_min,
        bounds_max,
        union_bounds_min,
        union_bounds_max,
        counts,
        fallback_flags,
        n_active_rays,
        row_pitch,
        column_pitch,
        row_zero_min_y,
        row_zero_max_y,
        column_zero_z,
        half_y,
        half_z,
        coordinate_guard,
        origin_limit,
        GRID_ROWS: tl.constexpr,
        GRID_COLUMNS: tl.constexpr,
        MAX_GRID_CANDIDATES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        active_ray = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = active_ray < n_active_rays
        ray = tl.load(active_ray_ids + active_ray, mask=valid, other=0)
        ox = tl.load(origins + ray * 3 + 0, mask=valid, other=0.0)
        oy = tl.load(origins + ray * 3 + 1, mask=valid, other=0.0)
        oz = tl.load(origins + ray * 3 + 2, mask=valid, other=0.0)
        dx = tl.load(directions + ray * 3 + 0, mask=valid, other=0.0)
        dy = tl.load(directions + ray * 3 + 1, mask=valid, other=0.0)
        dz = tl.load(directions + ray * 3 + 2, mask=valid, other=0.0)
        ray_tmax = tl.load(tmax_values + ray, mask=valid, other=-1.0)
        row_first, row_count, column_first, column_count, finite = (
            _grid_candidate_range(
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
        )
        rectangle_count = row_count * column_count
        use_grid = valid & finite & (rectangle_count <= MAX_GRID_CANDIDATES)
        safe_columns = tl.maximum(column_count, 1)
        exact_count = tl.zeros((BLOCK_SIZE,), tl.int32)
        for candidate_offset in range(MAX_GRID_CANDIDATES):
            candidate_valid = use_grid & (candidate_offset < rectangle_count)
            row = row_first + candidate_offset // safe_columns
            column = column_first + candidate_offset % safe_columns
            instance = row * GRID_COLUMNS + column
            safe_instance = tl.where(candidate_valid, instance, 0)
            lo_x = tl.load(
                bounds_min + safe_instance * 3 + 0,
                mask=candidate_valid,
                other=0.0,
            )
            lo_y = tl.load(
                bounds_min + safe_instance * 3 + 1,
                mask=candidate_valid,
                other=0.0,
            )
            lo_z = tl.load(
                bounds_min + safe_instance * 3 + 2,
                mask=candidate_valid,
                other=0.0,
            )
            hi_x = tl.load(
                bounds_max + safe_instance * 3 + 0,
                mask=candidate_valid,
                other=0.0,
            )
            hi_y = tl.load(
                bounds_max + safe_instance * 3 + 1,
                mask=candidate_valid,
                other=0.0,
            )
            hi_z = tl.load(
                bounds_max + safe_instance * 3 + 2,
                mask=candidate_valid,
                other=0.0,
            )
            hit = candidate_valid & _instance_aabb_hit(
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
            exact_count += hit.to(tl.int32)
        fallback = valid & (~finite | (rectangle_count > MAX_GRID_CANDIDATES))
        tl.store(counts + active_ray, exact_count, mask=valid)
        tl.store(
            fallback_flags + active_ray, fallback.to(tl.int32), mask=valid
        )

    @triton.jit(do_not_specialize=[17])
    def _fill_grid_candidates_kernel(
        origins,
        directions,
        tmax_values,
        last_instances,
        last_triangles,
        active_ray_ids,
        bounds_min,
        bounds_max,
        union_bounds_min,
        union_bounds_max,
        world_to_object_rotation,
        world_to_object_translation,
        offsets,
        local_origins,
        local_directions,
        candidate_tmax,
        candidate_last_triangle,
        candidate_instances,
        candidate_ray_ids,
        n_active_rays,
        row_pitch,
        column_pitch,
        row_zero_min_y,
        row_zero_max_y,
        column_zero_z,
        half_y,
        half_z,
        coordinate_guard,
        origin_limit,
        GRID_ROWS: tl.constexpr,
        GRID_COLUMNS: tl.constexpr,
        MAX_GRID_CANDIDATES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        active_ray = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = active_ray < n_active_rays
        ray = tl.load(active_ray_ids + active_ray, mask=valid, other=0)
        ox = tl.load(origins + ray * 3 + 0, mask=valid, other=0.0)
        oy = tl.load(origins + ray * 3 + 1, mask=valid, other=0.0)
        oz = tl.load(origins + ray * 3 + 2, mask=valid, other=0.0)
        dx = tl.load(directions + ray * 3 + 0, mask=valid, other=0.0)
        dy = tl.load(directions + ray * 3 + 1, mask=valid, other=0.0)
        dz = tl.load(directions + ray * 3 + 2, mask=valid, other=0.0)
        ray_tmax = tl.load(tmax_values + ray, mask=valid, other=-1.0)
        previous_instance = tl.load(
            last_instances + ray, mask=valid, other=-1
        )
        previous_triangle = tl.load(
            last_triangles + ray, mask=valid, other=-1
        )
        base = tl.load(offsets + active_ray, mask=valid, other=0)
        row_first, row_count, column_first, column_count, finite = (
            _grid_candidate_range(
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
        )
        rectangle_count = row_count * column_count
        use_grid = valid & finite & (rectangle_count <= MAX_GRID_CANDIDATES)
        safe_columns = tl.maximum(column_count, 1)
        written = tl.zeros((BLOCK_SIZE,), tl.int64)
        for candidate_offset in range(MAX_GRID_CANDIDATES):
            candidate_valid = use_grid & (candidate_offset < rectangle_count)
            row = row_first + candidate_offset // safe_columns
            column = column_first + candidate_offset % safe_columns
            instance = row * GRID_COLUMNS + column
            safe_instance = tl.where(candidate_valid, instance, 0)
            lo_x = tl.load(
                bounds_min + safe_instance * 3 + 0,
                mask=candidate_valid,
                other=0.0,
            )
            lo_y = tl.load(
                bounds_min + safe_instance * 3 + 1,
                mask=candidate_valid,
                other=0.0,
            )
            lo_z = tl.load(
                bounds_min + safe_instance * 3 + 2,
                mask=candidate_valid,
                other=0.0,
            )
            hi_x = tl.load(
                bounds_max + safe_instance * 3 + 0,
                mask=candidate_valid,
                other=0.0,
            )
            hi_y = tl.load(
                bounds_max + safe_instance * 3 + 1,
                mask=candidate_valid,
                other=0.0,
            )
            hi_z = tl.load(
                bounds_max + safe_instance * 3 + 2,
                mask=candidate_valid,
                other=0.0,
            )
            hit = candidate_valid & _instance_aabb_hit(
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
            destination = base + written
            rotation = safe_instance * 9
            r00 = tl.load(
                world_to_object_rotation + rotation + 0, mask=hit, other=0.0
            )
            r01 = tl.load(
                world_to_object_rotation + rotation + 1, mask=hit, other=0.0
            )
            r02 = tl.load(
                world_to_object_rotation + rotation + 2, mask=hit, other=0.0
            )
            r10 = tl.load(
                world_to_object_rotation + rotation + 3, mask=hit, other=0.0
            )
            r11 = tl.load(
                world_to_object_rotation + rotation + 4, mask=hit, other=0.0
            )
            r12 = tl.load(
                world_to_object_rotation + rotation + 5, mask=hit, other=0.0
            )
            r20 = tl.load(
                world_to_object_rotation + rotation + 6, mask=hit, other=0.0
            )
            r21 = tl.load(
                world_to_object_rotation + rotation + 7, mask=hit, other=0.0
            )
            r22 = tl.load(
                world_to_object_rotation + rotation + 8, mask=hit, other=0.0
            )
            tx = tl.load(
                world_to_object_translation + safe_instance * 3 + 0,
                mask=hit,
                other=0.0,
            )
            ty = tl.load(
                world_to_object_translation + safe_instance * 3 + 1,
                mask=hit,
                other=0.0,
            )
            tz = tl.load(
                world_to_object_translation + safe_instance * 3 + 2,
                mask=hit,
                other=0.0,
            )
            local_ox = r00 * ox + r01 * oy + r02 * oz + tx
            local_oy = r10 * ox + r11 * oy + r12 * oz + ty
            local_oz = r20 * ox + r21 * oy + r22 * oz + tz
            local_dx = r00 * dx + r01 * dy + r02 * dz
            local_dy = r10 * dx + r11 * dy + r12 * dz
            local_dz = r20 * dx + r21 * dy + r22 * dz
            tl.store(local_origins + destination * 3 + 0, local_ox, mask=hit)
            tl.store(local_origins + destination * 3 + 1, local_oy, mask=hit)
            tl.store(local_origins + destination * 3 + 2, local_oz, mask=hit)
            tl.store(
                local_directions + destination * 3 + 0, local_dx, mask=hit
            )
            tl.store(
                local_directions + destination * 3 + 1, local_dy, mask=hit
            )
            tl.store(
                local_directions + destination * 3 + 2, local_dz, mask=hit
            )
            tl.store(candidate_tmax + destination, ray_tmax, mask=hit)
            excluded = tl.where(
                previous_instance == safe_instance, previous_triangle, -1
            )
            tl.store(
                candidate_last_triangle + destination, excluded, mask=hit
            )
            tl.store(
                candidate_instances + destination, safe_instance, mask=hit
            )
            tl.store(candidate_ray_ids + destination, ray, mask=hit)
            written += hit.to(tl.int64)

    @triton.jit(do_not_specialize=[10])
    def _reduce_instance_candidates_kernel(
        candidate_distances,
        candidate_triangles,
        candidate_overflow,
        candidate_instances,
        offsets,
        active_ray_ids,
        out_distances,
        out_triangles,
        out_instances,
        out_overflow,
        n_active_rays,
        skip_flags,
        SKIP_FLAGGED: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        active_ray = tl.program_id(0)
        lanes = tl.arange(0, BLOCK_SIZE)
        valid_ray = active_ray < n_active_rays
        if SKIP_FLAGGED:
            valid_ray &= (
                tl.load(skip_flags + active_ray, mask=valid_ray, other=1) == 0
            )
        ray = tl.load(active_ray_ids + active_ray, mask=valid_ray, other=0)
        begin = tl.load(offsets + active_ray, mask=valid_ray, other=0)
        end = tl.load(offsets + active_ray + 1, mask=valid_ray, other=0)
        candidate = begin + lanes
        valid = valid_ray & (candidate < end)
        triangle = tl.load(
            candidate_triangles + candidate, mask=valid, other=-1
        )
        distance = tl.load(
            candidate_distances + candidate,
            mask=valid & (triangle >= 0),
            other=float("inf"),
        )
        minimum = tl.min(distance, axis=0)
        winning_lane = tl.min(
            tl.where(
                valid & (triangle >= 0) & (distance == minimum),
                lanes,
                BLOCK_SIZE,
            ),
            axis=0,
        )
        has_hit = valid_ray & (winning_lane < BLOCK_SIZE)
        winner = begin + winning_lane
        winning_triangle = tl.load(
            candidate_triangles + winner, mask=has_hit, other=-1
        )
        winning_instance = tl.load(
            candidate_instances + winner, mask=has_hit, other=-1
        )
        overflow_value = tl.load(
            candidate_overflow + candidate, mask=valid, other=0
        ).to(tl.int32)
        any_overflow = tl.max(overflow_value, axis=0)
        tl.store(
            out_distances + ray,
            tl.where(has_hit, minimum, float("inf")),
            mask=valid_ray,
        )
        tl.store(out_triangles + ray, winning_triangle, mask=valid_ray)
        tl.store(out_instances + ray, winning_instance, mask=valid_ray)
        tl.store(out_overflow + ray, any_overflow.to(tl.uint8), mask=valid_ray)

    @triton.jit(do_not_specialize=[3])
    def _scatter_candidate_counts_kernel(
        counts,
        active_ray_ids,
        out_counts,
        n_active_rays,
        BLOCK_SIZE: tl.constexpr,
    ):
        active_ray = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = active_ray < n_active_rays
        ray = tl.load(active_ray_ids + active_ray, mask=valid, other=0)
        value = tl.load(counts + active_ray, mask=valid, other=0)
        tl.store(out_counts + ray, value, mask=valid)

    @triton.jit(do_not_specialize=[7, 8])
    def _finalize_instance_hits_kernel(
        triangle_vertices,
        object_to_world_rotation,
        channel_ids,
        triangles,
        instances,
        out_channels,
        out_normals,
        n_rays,
        ray_capacity,
        ray_index_ptr,
        N_RAYS_IS_POINTER: tl.constexpr,
        INDIRECT: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        program_start = tl.program_id(0) * BLOCK_SIZE
        if N_RAYS_IS_POINTER:
            active_count = tl.load(n_rays).to(tl.int32)
            active_count = tl.maximum(
                0, tl.minimum(active_count, ray_capacity)
            )
            if program_start >= active_count:
                return
        else:
            active_count = n_rays
        work_slot = program_start + tl.arange(0, BLOCK_SIZE)
        valid = work_slot < active_count
        if INDIRECT:
            ray = tl.load(
                ray_index_ptr + work_slot, mask=valid, other=0
            ).to(tl.int32)
        else:
            ray = work_slot
        triangle = tl.load(triangles + ray, mask=valid, other=-1)
        instance = tl.load(instances + ray, mask=valid, other=-1)
        hit = valid & (triangle >= 0) & (instance >= 0)
        base = triangle * 9
        v0x = tl.load(triangle_vertices + base + 0, mask=hit, other=0.0)
        v0y = tl.load(triangle_vertices + base + 1, mask=hit, other=0.0)
        v0z = tl.load(triangle_vertices + base + 2, mask=hit, other=0.0)
        e1x = (
            tl.load(triangle_vertices + base + 3, mask=hit, other=0.0) - v0x
        )
        e1y = (
            tl.load(triangle_vertices + base + 4, mask=hit, other=0.0) - v0y
        )
        e1z = (
            tl.load(triangle_vertices + base + 5, mask=hit, other=0.0) - v0z
        )
        e2x = (
            tl.load(triangle_vertices + base + 6, mask=hit, other=0.0) - v0x
        )
        e2y = (
            tl.load(triangle_vertices + base + 7, mask=hit, other=0.0) - v0y
        )
        e2z = (
            tl.load(triangle_vertices + base + 8, mask=hit, other=0.0) - v0z
        )
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        inverse_length = tl.where(
            hit,
            1.0 / tl.sqrt(nx * nx + ny * ny + nz * nz),
            0.0,
        )
        nx, ny, nz = nx * inverse_length, ny * inverse_length, nz * inverse_length
        rotation = instance * 9
        r00 = tl.load(object_to_world_rotation + rotation + 0, mask=hit, other=0.0)
        r01 = tl.load(object_to_world_rotation + rotation + 1, mask=hit, other=0.0)
        r02 = tl.load(object_to_world_rotation + rotation + 2, mask=hit, other=0.0)
        r10 = tl.load(object_to_world_rotation + rotation + 3, mask=hit, other=0.0)
        r11 = tl.load(object_to_world_rotation + rotation + 4, mask=hit, other=0.0)
        r12 = tl.load(object_to_world_rotation + rotation + 5, mask=hit, other=0.0)
        r20 = tl.load(object_to_world_rotation + rotation + 6, mask=hit, other=0.0)
        r21 = tl.load(object_to_world_rotation + rotation + 7, mask=hit, other=0.0)
        r22 = tl.load(object_to_world_rotation + rotation + 8, mask=hit, other=0.0)
        world_x = r00 * nx + r01 * ny + r02 * nz
        world_y = r10 * nx + r11 * ny + r12 * nz
        world_z = r20 * nx + r21 * ny + r22 * nz
        channel = tl.load(channel_ids + instance, mask=hit, other=-1)
        tl.store(out_channels + ray, channel, mask=valid)
        tl.store(out_normals + ray * 3 + 0, world_x, mask=valid)
        tl.store(out_normals + ray * 3 + 1, world_y, mask=valid)
        tl.store(out_normals + ray * 3 + 2, world_z, mask=valid)

    @triton.jit(do_not_specialize=[10])
    def _refine_chroma_world_hits_kernel(
        world_vertices,
        triangle_indices,
        channel_ids,
        origins,
        directions,
        triangles,
        instances,
        out_distances,
        out_channels,
        out_normals,
        n_rays,
        N_VERTICES: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Re-evaluate winning PMT leaves in Chroma's flattened coordinates."""

        ray = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = ray < n_rays
        triangle = tl.load(triangles + ray, mask=valid, other=-1).to(tl.int32)
        instance = tl.load(instances + ray, mask=valid, other=-1).to(tl.int32)
        hit = valid & (triangle >= 0) & (instance >= 0)

        index_base = triangle * 3
        i0 = tl.load(
            triangle_indices + index_base, mask=hit, other=0
        ).to(tl.int32)
        i1 = tl.load(
            triangle_indices + index_base + 1, mask=hit, other=0
        ).to(tl.int32)
        i2 = tl.load(
            triangle_indices + index_base + 2, mask=hit, other=0
        ).to(tl.int32)
        instance_base = instance * N_VERTICES
        v0 = (instance_base + i0) * 3
        v1 = (instance_base + i1) * 3
        v2 = (instance_base + i2) * 3
        v0x = tl.load(world_vertices + v0, mask=hit, other=0.0)
        v0y = tl.load(world_vertices + v0 + 1, mask=hit, other=0.0)
        v0z = tl.load(world_vertices + v0 + 2, mask=hit, other=0.0)
        v1x = tl.load(world_vertices + v1, mask=hit, other=0.0)
        v1y = tl.load(world_vertices + v1 + 1, mask=hit, other=0.0)
        v1z = tl.load(world_vertices + v1 + 2, mask=hit, other=0.0)
        v2x = tl.load(world_vertices + v2, mask=hit, other=0.0)
        v2y = tl.load(world_vertices + v2 + 1, mask=hit, other=0.0)
        v2z = tl.load(world_vertices + v2 + 2, mask=hit, other=0.0)

        ox = tl.load(origins + ray * 3, mask=hit, other=0.0)
        oy = tl.load(origins + ray * 3 + 1, mask=hit, other=0.0)
        oz = tl.load(origins + ray * 3 + 2, mask=hit, other=0.0)
        dx = tl.load(directions + ray * 3, mask=hit, other=0.0)
        dy = tl.load(directions + ray * 3 + 1, mask=hit, other=0.0)
        dz = tl.load(directions + ray * 3 + 2, mask=hit, other=0.0)

        _, barycentric_u, barycentric_v, distance = (
            _chroma_intersection_words(
                v0x,
                v0y,
                v0z,
                v1x,
                v1y,
                v1z,
                v2x,
                v2y,
                v2z,
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
            )
        )

        # Chroma's triangle normal uses (v1-v0) x (v2-v1), followed by the
        # exact normalize() instruction tree observed in its CUDA PTX.
        e1x, e1y, e1z = v1x - v0x, v1y - v0y, v1z - v0z
        e12x, e12y, e12z = v2x - v1x, v2y - v1y, v2z - v1z
        # Keep these as source-level multiply/subtract expressions.  In this
        # kernel context Triton's NVIDIA lowering contracts them exactly like
        # fill_state; spelling explicit tl.fma changes eleven detector normals.
        nx = e1y * e12z - e1z * e12y
        ny = e1z * e12x - e1x * e12z
        nz = e1x * e12y - e1y * e12x
        normal_squared = tl.fma(
            nz, nz, tl.fma(nx, nx, ny * ny)
        )
        normal_length = tl.sqrt(normal_squared)
        nx, ny, nz = nx / normal_length, ny / normal_length, nz / normal_length

        # The local BLAS already established the logical hit.  Retain the
        # barycentric calculations above deliberately: they force the same
        # dependency/instruction graph as the original CUDA intersection.
        _ = barycentric_u + barycentric_v
        tl.store(out_distances + ray, distance, mask=hit)
        channel = tl.load(channel_ids + instance, mask=hit, other=-1)
        # The fused TLAS path does not run the initializer used by the legacy
        # pipeline, so establish the miss sentinels here as well.
        tl.store(out_channels + ray, channel, mask=valid)
        tl.store(
            out_normals + ray * 3, tl.where(hit, nx, 0.0), mask=valid
        )
        tl.store(
            out_normals + ray * 3 + 1, tl.where(hit, ny, 0.0), mask=valid
        )
        tl.store(
            out_normals + ray * 3 + 2, tl.where(hit, nz, 0.0), mask=valid
        )


def _torch_rays(values: Any, name: str, device: Any) -> Any:
    if isinstance(values, torch.Tensor):
        result = values.to(device=device, dtype=torch.float32)
    else:
        result = torch.as_tensor(values, dtype=torch.float32, device=device)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("%s must have shape (N, 3)" % name)
    return result.contiguous()


def _torch_per_ray(
    values: Any,
    *,
    count: int,
    default: Any,
    dtype: Any,
    name: str,
    device: Any,
) -> Any:
    if values is None:
        return torch.full((count,), default, dtype=dtype, device=device)
    if isinstance(values, torch.Tensor):
        result = values.to(device=device, dtype=dtype)
    elif np.ndim(values) == 0:
        return torch.full((count,), values, dtype=dtype, device=device)
    else:
        result = torch.as_tensor(values, dtype=dtype, device=device)
    if result.ndim == 0:
        return torch.full((count,), result.item(), dtype=dtype, device=device)
    if result.shape != (count,):
        raise ValueError("%s must be scalar or have shape (N,)" % name)
    return result.contiguous()


def _allocate_instance_result(count: int, device: Any) -> PMTInstanceResult:
    """Allocate uninitialized outputs; one fused kernel establishes sentinels."""

    return PMTInstanceResult(
        triangle_ids=torch.empty(count, dtype=torch.int32, device=device),
        distances=torch.empty(count, dtype=torch.float32, device=device),
        instance_ids=torch.empty(count, dtype=torch.int32, device=device),
        channel_ids=torch.empty(count, dtype=torch.int32, device=device),
        world_normals=torch.empty(
            (count, 3), dtype=torch.float32, device=device
        ),
        overflow=torch.empty(count, dtype=torch.uint8, device=device),
        candidate_counts=torch.empty(
            count, dtype=torch.int32, device=device
        ),
    )


def _validate_instance_result(
    out: PMTInstanceResult, count: int, device: Any
) -> PMTInstanceResult:
    if not isinstance(out, PMTInstanceResult):
        raise TypeError("out must be a PMTInstanceResult instance")
    specifications = {
        "triangle_ids": ((count,), torch.int32),
        "distances": ((count,), torch.float32),
        "instance_ids": ((count,), torch.int32),
        "channel_ids": ((count,), torch.int32),
        "world_normals": ((count, 3), torch.float32),
        "overflow": ((count,), torch.uint8),
        "candidate_counts": ((count,), torch.int32),
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
                "on the accelerator device"
            )
    return out


def refine_pmt_hits_chroma_world(
    accelerator: PMTInstanceAccelerator,
    origins: Any,
    directions: Any,
    result: PMTInstanceResult,
) -> PMTInstanceResult:
    """Finalize PMT hit distance/normal with Chroma's world-mesh result.

    Strict compatibility traversal already evaluates every candidate leaf in
    these coordinates.  This final pass reconstructs the winning raw normal
    with Chroma's distinct ``(v1-v0) x (v2-v1)`` operation tree and writes the
    global channel.  The distance is repeated from the same words as a useful
    invariant.  Tensor storage in ``result`` is mutated in place.
    """

    _require_backend()
    _require_chroma_world_compatibility(accelerator)
    device = accelerator.device
    origin_tensor = _torch_rays(origins, "origins", device)
    direction_tensor = _torch_rays(directions, "directions", device)
    if direction_tensor.shape != origin_tensor.shape:
        raise ValueError("directions must have the same shape as origins")
    count = int(origin_tensor.shape[0])
    _validate_instance_result(result, count, device)
    if count:
        block_size = 256
        _refine_chroma_world_hits_kernel[(triton.cdiv(count, block_size),)](
            accelerator.chroma_world_vertices,
            accelerator.chroma_triangle_indices,
            accelerator.channel_ids,
            origin_tensor,
            direction_tensor,
            result.triangle_ids,
            result.instance_ids,
            result.distances,
            result.channel_ids,
            result.world_normals,
            count,
            N_VERTICES=accelerator.host_chroma_world_vertices.shape[1],
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
    return result


def _require_chroma_world_compatibility(
    accelerator: PMTInstanceAccelerator,
) -> None:
    """Validate that an accelerator owns the opt-in flattened PMT geometry."""

    if (
        accelerator.chroma_world_vertices is None
        or accelerator.chroma_triangle_indices is None
        or accelerator.host_chroma_world_vertices is None
    ):
        raise ValueError(
            "accelerator was not built with chroma_world_compatibility=True"
        )


def _nearest_hit_candidates_into(
    workspace: PMTInstanceWorkspace,
    count: int,
    world_origins: Any,
    world_directions: Any,
    *,
    chroma_world_compatibility: bool = False,
) -> None:
    """Launch the shared BLAS directly into reusable candidate outputs.

    ``nearest_hit_local`` currently allocates three result tensors on every
    call.  This internal launcher uses the same kernel and validation-proven
    workspace, but targets buffers whose lifetime follows the PMT workspace.
    """

    if count <= 0:
        return
    accelerator = workspace.accelerator
    bvh = accelerator.device_bvh
    if chroma_world_compatibility:
        chroma_world_vertices = accelerator.chroma_world_vertices
        chroma_triangle_indices = accelerator.chroma_triangle_indices
        n_world_vertices = int(
            accelerator.host_chroma_world_vertices.shape[1]
        )
    else:
        # Compile-time dead inputs: the production specialization emits no
        # loads from these pointers and allocates no compatibility geometry.
        chroma_world_vertices = bvh.triangle_vertices
        chroma_triangle_indices = workspace.bvh_dummy
        n_world_vertices = 1
    block_size = 32
    grid = (triton.cdiv(count, block_size),)
    _nearest_hit_candidates_progress_kernel[grid](
        bvh.nodes,
        bvh.triangle_vertices,
        chroma_world_vertices,
        chroma_triangle_indices,
        workspace.local_origins,
        workspace.local_directions,
        workspace.candidate_tmax,
        workspace.candidate_last_triangle,
        workspace.candidate_ray_ids,
        workspace.candidate_instances,
        world_origins,
        world_directions,
        workspace.traversal.stack,
        workspace.candidate_triangles,
        workspace.candidate_distances,
        workspace.candidate_overflow,
        workspace.sticky_overflow,
        count,
        bvh.world_origin[0],
        bvh.world_origin[1],
        bvh.world_origin[2],
        bvh.world_scale,
        N_WORLD_VERTICES=n_world_vertices,
        CHROMA_WORLD_GEOMETRY=bool(chroma_world_compatibility),
        STACK_CAPACITY=bvh.stack_capacity,
        BLOCK_SIZE=block_size,
        num_warps=1,
    )


def _nearest_pmt_hit_legacy(
    accelerator: PMTInstanceAccelerator,
    origins: Any,
    directions: Any,
    *,
    tmax: Optional[Any] = None,
    last_instance: Optional[Any] = None,
    last_triangle: Optional[Any] = None,
    ray_tile: Optional[int] = DEFAULT_RAY_TILE,
    workspace: Optional[PMTInstanceWorkspace] = None,
    out: Optional[PMTInstanceResult] = None,
    check_overflow: bool = True,
    chroma_world_compatibility: bool = False,
) -> PMTInstanceResult:
    """Find the globally nearest exact PMT triangle for world-space rays.

    A conservative union-box pass first removes rays that cannot reach any PMT
    before ``tmax``.  The retained sparse list then has no fixed capacity: an
    instance pass counts all conservative AABB overlaps and storage grows to
    the exact required size.  ``ray_tile`` only limits peak storage.  It does
    not limit candidates or alter reduction order.  ``last_instance`` is a
    retained-instance index, while ``last_triangle`` is the canonical mesh's
    original triangle ID.  They must be supplied together.  ``out`` may
    provide exact-shape CUDA tensors to overwrite; workspace-owned reusable
    views are available from :meth:`PMTInstanceWorkspace.outputs`.
    """

    _require_backend()
    if (last_instance is None) != (last_triangle is None):
        raise ValueError("last_instance and last_triangle must be supplied together")
    device = accelerator.device
    origin_tensor = _torch_rays(origins, "origins", device)
    direction_tensor = _torch_rays(directions, "directions", device)
    if direction_tensor.shape != origin_tensor.shape:
        raise ValueError("directions must have the same shape as origins")
    count = int(origin_tensor.shape[0])
    tmax_tensor = _torch_per_ray(
        tmax,
        count=count,
        default=float("inf"),
        dtype=torch.float32,
        name="tmax",
        device=device,
    )
    last_instance_tensor = _torch_per_ray(
        last_instance,
        count=count,
        default=-1,
        dtype=torch.int32,
        name="last_instance",
        device=device,
    )
    last_triangle_tensor = _torch_per_ray(
        last_triangle,
        count=count,
        default=-1,
        dtype=torch.int32,
        name="last_triangle",
        device=device,
    )

    if ray_tile is None:
        tile_size = max(1, count)
    else:
        tile_size = int(ray_tile)
        if tile_size <= 0:
            raise ValueError("ray_tile must be positive or None")
    tile_capacity = min(count, tile_size)
    owns_workspace = workspace is None
    if workspace is None:
        workspace = accelerator.allocate_workspace(
            tile_capacity, result_capacity=count
        )
    elif workspace.accelerator is not accelerator:
        raise ValueError("workspace belongs to a different PMT accelerator")
    elif workspace.ray_capacity < tile_capacity:
        raise ValueError("workspace ray capacity is smaller than ray_tile")

    if out is None and owns_workspace:
        out = workspace.outputs(count)
    result = (
        _allocate_instance_result(count, device)
        if out is None
        else _validate_instance_result(out, count, device)
    )
    distances = result.distances
    triangles = result.triangle_ids
    instances = result.instance_ids
    channels = result.channel_ids
    normals = result.world_normals
    overflow = result.overflow
    candidate_counts = result.candidate_counts
    broadphase_block = 128
    reduction_block = triton.next_power_of_2(accelerator.instance_count)
    performed_blas = False

    if count:
        _initialize_instance_results_kernel[(triton.cdiv(count, 256),)](
            distances,
            triangles,
            instances,
            channels,
            normals,
            overflow,
            candidate_counts,
            count,
            count,
            N_RAYS_IS_POINTER=False,
            BLOCK_SIZE=256,
            num_warps=8,
        )

    for start in range(0, count, tile_size):
        stop = min(count, start + tile_size)
        tile_count = stop - start
        tile_origins = origin_tensor[start:stop]
        tile_directions = direction_tensor[start:stop]
        tile_tmax = tmax_tensor[start:stop]
        tile_last_instance = last_instance_tensor[start:stop]
        tile_last_triangle = last_triangle_tensor[start:stop]
        grid = (triton.cdiv(tile_count, broadphase_block),)
        # Stage zero is a single conservative box around every PMT instance.
        # Most boundary events are capped by a nearer wire/cathode/YZ wall and
        # stop here without executing the 81-instance loop or allocating BLAS
        # work.  The same exact prefix storage is reused by the next stage.
        _mark_union_candidates_kernel[grid](
            tile_origins,
            tile_directions,
            tile_tmax,
            accelerator.coarse_bounds_min if accelerator.coarse_bounds_min is not None else accelerator.union_bounds_min,
            accelerator.coarse_bounds_max if accelerator.coarse_bounds_max is not None else accelerator.union_bounds_max,
            workspace.counts,
            tile_count,
            BLOCK_SIZE=broadphase_block,
            NBOX=accelerator.coarse_box_count,
            num_warps=4,
        )
        workspace.offsets[0].zero_()
        torch.cumsum(
            workspace.counts[:tile_count],
            dim=0,
            dtype=torch.int64,
            out=workspace.offsets[1 : tile_count + 1],
        )
        active_count = int(workspace.offsets[tile_count].item())
        if active_count == 0:
            continue
        _compact_union_candidates_kernel[grid](
            workspace.counts,
            workspace.offsets,
            workspace.active_ray_ids,
            tile_count,
            BLOCK_SIZE=broadphase_block,
            num_warps=4,
        )

        active_grid = (triton.cdiv(active_count, broadphase_block),)
        _count_instance_candidates_kernel[active_grid](
            tile_origins,
            tile_directions,
            tile_tmax,
            workspace.active_ray_ids,
            accelerator.bounds_min,
            accelerator.bounds_max,
            workspace.counts,
            active_count,
            N_INSTANCES=accelerator.instance_count,
            BLOCK_SIZE=broadphase_block,
            num_warps=4,
        )
        workspace.offsets[0].zero_()
        torch.cumsum(
            workspace.counts[:active_count],
            dim=0,
            dtype=torch.int64,
            out=workspace.offsets[1 : active_count + 1],
        )
        _scatter_candidate_counts_kernel[active_grid](
            workspace.counts,
            workspace.active_ray_ids,
            candidate_counts[start:stop],
            active_count,
            BLOCK_SIZE=broadphase_block,
            num_warps=4,
        )
        # The dynamic total permits exact-size allocation and eliminates every
        # overflow/candidate-cap failure mode.  Union-empty tiles used only the
        # earlier synchronization and reach neither this point nor the BLAS.
        total_candidates = int(workspace.offsets[active_count].item())
        workspace.ensure_candidate_capacity(total_candidates)
        if total_candidates == 0:
            continue
        performed_blas = True

        _fill_instance_candidates_kernel[active_grid](
            tile_origins,
            tile_directions,
            tile_tmax,
            tile_last_instance,
            tile_last_triangle,
            workspace.active_ray_ids,
            accelerator.bounds_min,
            accelerator.bounds_max,
            accelerator.world_to_object_rotation,
            accelerator.world_to_object_translation,
            workspace.offsets,
            workspace.local_origins,
            workspace.local_directions,
            workspace.candidate_tmax,
            workspace.candidate_last_triangle,
            workspace.candidate_instances,
            workspace.candidate_ray_ids,
            active_count,
            N_INSTANCES=accelerator.instance_count,
            BLOCK_SIZE=broadphase_block,
            num_warps=4,
        )
        _nearest_hit_candidates_into(
            workspace,
            total_candidates,
            tile_origins,
            tile_directions,
            chroma_world_compatibility=chroma_world_compatibility,
        )
        _reduce_instance_candidates_kernel[(active_count,)](
            workspace.candidate_distances,
            workspace.candidate_triangles,
            workspace.candidate_overflow,
            workspace.candidate_instances,
            workspace.offsets,
            workspace.active_ray_ids,
            distances[start:stop],
            triangles[start:stop],
            instances[start:stop],
            overflow[start:stop],
            active_count,
            workspace.counts,
            SKIP_FLAGGED=False,
            BLOCK_SIZE=reduction_block,
            num_warps=max(1, reduction_block // 32),
        )

    if count and performed_blas:
        if chroma_world_compatibility:
            refine_pmt_hits_chroma_world(
                accelerator, origin_tensor, direction_tensor, result
            )
        else:
            _finalize_instance_hits_kernel[(triton.cdiv(count, 256),)](
                accelerator.device_bvh.triangle_vertices,
                accelerator.object_to_world_rotation,
                accelerator.channel_ids,
                triangles,
                instances,
                channels,
                normals,
                count,
                count,
                triangles,
                N_RAYS_IS_POINTER=False,
                INDIRECT=False,
                BLOCK_SIZE=256,
                num_warps=8,
            )
    if (
        check_overflow
        and count
        and performed_blas
        and bool(torch.any(overflow).item())
    ):
        raise RuntimeError("canonical PMT BLAS traversal stack overflowed")
    return result


def _nearest_pmt_hit_grid(
    accelerator: PMTInstanceAccelerator,
    origins: Any,
    directions: Any,
    *,
    tmax: Optional[Any] = None,
    last_instance: Optional[Any] = None,
    last_triangle: Optional[Any] = None,
    ray_tile: Optional[int] = DEFAULT_RAY_TILE,
    workspace: Optional[PMTInstanceWorkspace] = None,
    out: Optional[PMTInstanceResult] = None,
    check_overflow: bool = True,
    chroma_world_compatibility: bool = False,
) -> PMTInstanceResult:
    """Exact production traversal for a validated regular PMT plane.

    The common path quantizes the ray segment inside the PMT union to a small
    row/column rectangle, then runs the ordinary padded-AABB test on those
    instances in ascending order.  A rectangle wider than
    :data:`DEFAULT_GRID_CANDIDATES`, or any non-finite interval, is compacted
    and passed to the independent exact TLAS.  Thus the threshold is purely a
    performance policy and never a correctness cap.
    """

    if chroma_world_compatibility:
        raise ValueError("the PMT grid path is production-only")
    locator = accelerator.grid_locator
    if locator is None:
        return _nearest_pmt_hit_legacy(
            accelerator,
            origins,
            directions,
            tmax=tmax,
            last_instance=last_instance,
            last_triangle=last_triangle,
            ray_tile=ray_tile,
            workspace=workspace,
            out=out,
            check_overflow=check_overflow,
        )
    _require_backend()
    if (last_instance is None) != (last_triangle is None):
        raise ValueError("last_instance and last_triangle must be supplied together")
    device = accelerator.device
    origin_tensor = _torch_rays(origins, "origins", device)
    direction_tensor = _torch_rays(directions, "directions", device)
    if direction_tensor.shape != origin_tensor.shape:
        raise ValueError("directions must have the same shape as origins")
    count = int(origin_tensor.shape[0])
    tmax_tensor = _torch_per_ray(
        tmax,
        count=count,
        default=float("inf"),
        dtype=torch.float32,
        name="tmax",
        device=device,
    )
    last_instance_tensor = _torch_per_ray(
        last_instance,
        count=count,
        default=-1,
        dtype=torch.int32,
        name="last_instance",
        device=device,
    )
    last_triangle_tensor = _torch_per_ray(
        last_triangle,
        count=count,
        default=-1,
        dtype=torch.int32,
        name="last_triangle",
        device=device,
    )

    if ray_tile is None:
        tile_size = max(1, count)
    else:
        tile_size = int(ray_tile)
        if tile_size <= 0:
            raise ValueError("ray_tile must be positive or None")
    tile_capacity = min(count, tile_size)
    owns_workspace = workspace is None
    if workspace is None:
        workspace = accelerator.allocate_workspace(
            tile_capacity, result_capacity=count
        )
    elif workspace.accelerator is not accelerator:
        raise ValueError("workspace belongs to a different PMT accelerator")
    elif workspace.ray_capacity < tile_capacity:
        raise ValueError("workspace ray capacity is smaller than ray_tile")

    if out is None and owns_workspace:
        out = workspace.outputs(count)
    result = (
        _allocate_instance_result(count, device)
        if out is None
        else _validate_instance_result(out, count, device)
    )
    distances = result.distances
    triangles = result.triangle_ids
    instances = result.instance_ids
    channels = result.channel_ids
    normals = result.world_normals
    overflow = result.overflow
    candidate_counts = result.candidate_counts
    broadphase_block = 128
    reduction_block = triton.next_power_of_2(DEFAULT_GRID_CANDIDATES)
    performed_blas = False

    if count:
        _initialize_instance_results_kernel[(triton.cdiv(count, 256),)](
            distances,
            triangles,
            instances,
            channels,
            normals,
            overflow,
            candidate_counts,
            count,
            count,
            N_RAYS_IS_POINTER=False,
            BLOCK_SIZE=256,
            num_warps=8,
        )

    for start in range(0, count, tile_size):
        stop = min(count, start + tile_size)
        tile_count = stop - start
        tile_origins = origin_tensor[start:stop]
        tile_directions = direction_tensor[start:stop]
        tile_tmax = tmax_tensor[start:stop]
        tile_last_instance = last_instance_tensor[start:stop]
        tile_last_triangle = last_triangle_tensor[start:stop]
        grid = (triton.cdiv(tile_count, broadphase_block),)
        _mark_union_candidates_kernel[grid](
            tile_origins,
            tile_directions,
            tile_tmax,
            accelerator.union_bounds_min,
            accelerator.union_bounds_max,
            workspace.counts,
            tile_count,
            BLOCK_SIZE=broadphase_block,
            num_warps=4,
        )
        workspace.offsets[0].zero_()
        torch.cumsum(
            workspace.counts[:tile_count],
            dim=0,
            dtype=torch.int64,
            out=workspace.offsets[1 : tile_count + 1],
        )
        active_count = int(workspace.offsets[tile_count].item())
        if active_count == 0:
            continue
        _compact_union_candidates_kernel[grid](
            workspace.counts,
            workspace.offsets,
            workspace.active_ray_ids,
            tile_count,
            BLOCK_SIZE=broadphase_block,
            num_warps=4,
        )

        active_grid = (triton.cdiv(active_count, broadphase_block),)
        use_grid_tile = active_count >= GRID_MIN_ACTIVE_RAYS
        if use_grid_tile:
            _count_grid_candidates_kernel[active_grid](
                tile_origins,
                tile_directions,
                tile_tmax,
                workspace.active_ray_ids,
                accelerator.bounds_min,
                accelerator.bounds_max,
                accelerator.union_bounds_min,
                accelerator.union_bounds_max,
                workspace.counts,
                workspace.grid_fallback_flags,
                active_count,
                locator.row_pitch,
                locator.column_pitch,
                locator.row_zero_min_y,
                locator.row_zero_max_y,
                locator.column_zero_z,
                locator.half_y,
                locator.half_z,
                locator.coordinate_guard,
                locator.origin_limit,
                GRID_ROWS=locator.rows,
                GRID_COLUMNS=locator.columns,
                MAX_GRID_CANDIDATES=DEFAULT_GRID_CANDIDATES,
                BLOCK_SIZE=broadphase_block,
                num_warps=4,
            )
        else:
            _count_instance_candidates_kernel[active_grid](
                tile_origins,
                tile_directions,
                tile_tmax,
                workspace.active_ray_ids,
                accelerator.bounds_min,
                accelerator.bounds_max,
                workspace.counts,
                active_count,
                N_INSTANCES=accelerator.instance_count,
                BLOCK_SIZE=broadphase_block,
                num_warps=4,
            )
        workspace.offsets[0].zero_()
        torch.cumsum(
            workspace.counts[:active_count],
            dim=0,
            dtype=torch.int64,
            out=workspace.offsets[1 : active_count + 1],
        )
        if use_grid_tile:
            workspace.grid_fallback_offsets[0].zero_()
            torch.cumsum(
                workspace.grid_fallback_flags[:active_count],
                dim=0,
                dtype=torch.int64,
                out=workspace.grid_fallback_offsets[1 : active_count + 1],
            )
        _scatter_candidate_counts_kernel[active_grid](
            workspace.counts,
            workspace.active_ray_ids,
            candidate_counts[start:stop],
            active_count,
            BLOCK_SIZE=broadphase_block,
            num_warps=4,
        )
        if use_grid_tile:
            _compact_union_candidates_kernel[active_grid](
                workspace.grid_fallback_flags,
                workspace.grid_fallback_offsets,
                workspace.grid_fallback_active_ids,
                active_count,
                BLOCK_SIZE=broadphase_block,
                num_warps=4,
            )

        # Both values share the same synchronization boundary.  The fallback
        # count is normally zero, while the exact fast-pair total retains the
        # existing no-cap allocation contract.
        total_candidates = int(workspace.offsets[active_count].item())
        fallback_count = (
            int(workspace.grid_fallback_offsets[active_count].item())
            if use_grid_tile
            else 0
        )
        workspace.ensure_candidate_capacity(total_candidates)

        if fallback_count:
            performed_blas = True
            fallback_active = workspace.grid_fallback_active_ids[
                :fallback_count
            ].to(dtype=torch.int64)
            fallback_rays = workspace.active_ray_ids.index_select(
                0, fallback_active
            ).to(dtype=torch.int64)
            fallback_result = _nearest_pmt_hit_tlas(
                accelerator,
                tile_origins.index_select(0, fallback_rays),
                tile_directions.index_select(0, fallback_rays),
                tmax=tile_tmax.index_select(0, fallback_rays),
                last_instance=tile_last_instance.index_select(0, fallback_rays),
                last_triangle=tile_last_triangle.index_select(0, fallback_rays),
                ray_tile=None,
                workspace=workspace,
                check_overflow=check_overflow,
            )
            for destination, source in (
                (triangles[start:stop], fallback_result.triangle_ids),
                (distances[start:stop], fallback_result.distances),
                (instances[start:stop], fallback_result.instance_ids),
                (channels[start:stop], fallback_result.channel_ids),
                (normals[start:stop], fallback_result.world_normals),
                (overflow[start:stop], fallback_result.overflow),
                (candidate_counts[start:stop], fallback_result.candidate_counts),
            ):
                destination.index_copy_(0, fallback_rays, source)

        if total_candidates:
            performed_blas = True
            if use_grid_tile:
                _fill_grid_candidates_kernel[active_grid](
                    tile_origins,
                    tile_directions,
                    tile_tmax,
                    tile_last_instance,
                    tile_last_triangle,
                    workspace.active_ray_ids,
                    accelerator.bounds_min,
                    accelerator.bounds_max,
                    accelerator.union_bounds_min,
                    accelerator.union_bounds_max,
                    accelerator.world_to_object_rotation,
                    accelerator.world_to_object_translation,
                    workspace.offsets,
                    workspace.local_origins,
                    workspace.local_directions,
                    workspace.candidate_tmax,
                    workspace.candidate_last_triangle,
                    workspace.candidate_instances,
                    workspace.candidate_ray_ids,
                    active_count,
                    locator.row_pitch,
                    locator.column_pitch,
                    locator.row_zero_min_y,
                    locator.row_zero_max_y,
                    locator.column_zero_z,
                    locator.half_y,
                    locator.half_z,
                    locator.coordinate_guard,
                    locator.origin_limit,
                    GRID_ROWS=locator.rows,
                    GRID_COLUMNS=locator.columns,
                    MAX_GRID_CANDIDATES=DEFAULT_GRID_CANDIDATES,
                    BLOCK_SIZE=broadphase_block,
                    num_warps=4,
                )
            else:
                _fill_instance_candidates_kernel[active_grid](
                    tile_origins,
                    tile_directions,
                    tile_tmax,
                    tile_last_instance,
                    tile_last_triangle,
                    workspace.active_ray_ids,
                    accelerator.bounds_min,
                    accelerator.bounds_max,
                    accelerator.world_to_object_rotation,
                    accelerator.world_to_object_translation,
                    workspace.offsets,
                    workspace.local_origins,
                    workspace.local_directions,
                    workspace.candidate_tmax,
                    workspace.candidate_last_triangle,
                    workspace.candidate_instances,
                    workspace.candidate_ray_ids,
                    active_count,
                    N_INSTANCES=accelerator.instance_count,
                    BLOCK_SIZE=broadphase_block,
                    num_warps=4,
                )
            _nearest_hit_candidates_into(
                workspace,
                total_candidates,
                tile_origins,
                tile_directions,
            )
            _reduce_instance_candidates_kernel[(active_count,)](
                workspace.candidate_distances,
                workspace.candidate_triangles,
                workspace.candidate_overflow,
                workspace.candidate_instances,
                workspace.offsets,
                workspace.active_ray_ids,
                distances[start:stop],
                triangles[start:stop],
                instances[start:stop],
                overflow[start:stop],
                active_count,
                workspace.grid_fallback_flags,
                SKIP_FLAGGED=use_grid_tile,
                BLOCK_SIZE=reduction_block,
                num_warps=max(1, reduction_block // 32),
            )

    if count and performed_blas:
        _finalize_instance_hits_kernel[(triton.cdiv(count, 256),)](
            accelerator.device_bvh.triangle_vertices,
            accelerator.object_to_world_rotation,
            accelerator.channel_ids,
            triangles,
            instances,
            channels,
            normals,
            count,
            count,
            triangles,
            N_RAYS_IS_POINTER=False,
            INDIRECT=False,
            BLOCK_SIZE=256,
            num_warps=8,
        )
    if (
        check_overflow
        and count
        and performed_blas
        and bool(torch.any(overflow).item())
    ):
        raise RuntimeError("PMT grid/TLAS traversal stack overflowed")
    return result


def _nearest_pmt_hit_tlas(
    accelerator: PMTInstanceAccelerator,
    origins: Any,
    directions: Any,
    *,
    tmax: Optional[Any] = None,
    last_instance: Optional[Any] = None,
    last_triangle: Optional[Any] = None,
    ray_tile: Optional[int] = DEFAULT_RAY_TILE,
    workspace: Optional[PMTInstanceWorkspace] = None,
    out: Optional[PMTInstanceResult] = None,
    check_overflow: bool = True,
    chroma_world_compatibility: bool = False,
) -> PMTInstanceResult:
    """Exact fused two-level traversal used by :func:`nearest_pmt_hit`."""

    _require_backend()
    if (last_instance is None) != (last_triangle is None):
        raise ValueError("last_instance and last_triangle must be supplied together")
    device = accelerator.device
    origin_tensor = _torch_rays(origins, "origins", device)
    direction_tensor = _torch_rays(directions, "directions", device)
    if direction_tensor.shape != origin_tensor.shape:
        raise ValueError("directions must have the same shape as origins")
    count = int(origin_tensor.shape[0])
    tmax_tensor = _torch_per_ray(
        tmax,
        count=count,
        default=float("inf"),
        dtype=torch.float32,
        name="tmax",
        device=device,
    )
    last_instance_tensor = _torch_per_ray(
        last_instance,
        count=count,
        default=-1,
        dtype=torch.int32,
        name="last_instance",
        device=device,
    )
    last_triangle_tensor = _torch_per_ray(
        last_triangle,
        count=count,
        default=-1,
        dtype=torch.int32,
        name="last_triangle",
        device=device,
    )
    if ray_tile is None:
        tile_size = max(1, count)
    else:
        tile_size = int(ray_tile)
        if tile_size <= 0:
            raise ValueError("ray_tile must be positive or None")
    tile_capacity = min(count, tile_size)
    owns_workspace = workspace is None
    if workspace is None:
        workspace = accelerator.allocate_workspace(
            tile_capacity, result_capacity=count
        )
    elif workspace.accelerator is not accelerator:
        raise ValueError("workspace belongs to a different PMT accelerator")
    elif workspace.ray_capacity < tile_capacity:
        raise ValueError("workspace ray capacity is smaller than ray_tile")
    if out is None and owns_workspace:
        out = workspace.outputs(count)
    result = (
        _allocate_instance_result(count, device)
        if out is None
        else _validate_instance_result(out, count, device)
    )

    block_size = 32
    bvh = accelerator.device_bvh
    if chroma_world_compatibility:
        chroma_world_vertices = accelerator.chroma_world_vertices
        chroma_triangle_indices = accelerator.chroma_triangle_indices
        n_world_vertices = int(
            accelerator.host_chroma_world_vertices.shape[1]
        )
    else:
        chroma_world_vertices = bvh.triangle_vertices
        chroma_triangle_indices = workspace.bvh_dummy
        n_world_vertices = 1
    for start in range(0, count, tile_size):
        stop = min(count, start + tile_size)
        tile_count = stop - start
        _nearest_pmt_tlas_kernel[(triton.cdiv(tile_count, block_size),)](
            accelerator.tlas_bounds_min,
            accelerator.tlas_bounds_max,
            accelerator.tlas_left_child,
            accelerator.tlas_right_child,
            accelerator.tlas_instance,
            accelerator.world_to_object_rotation,
            accelerator.world_to_object_translation,
            bvh.nodes,
            bvh.triangle_vertices,
            chroma_world_vertices,
            chroma_triangle_indices,
            origin_tensor[start:stop],
            direction_tensor[start:stop],
            tmax_tensor[start:stop],
            last_instance_tensor[start:stop],
            last_triangle_tensor[start:stop],
            workspace.tlas_stack,
            workspace.fused_traversal.stack,
            result.triangle_ids[start:stop],
            result.distances[start:stop],
            result.instance_ids[start:stop],
            result.overflow[start:stop],
            result.candidate_counts[start:stop],
            workspace.sticky_overflow,
            tile_count,
            tile_count,
            origin_tensor,
            bvh.world_origin[0],
            bvh.world_origin[1],
            bvh.world_origin[2],
            bvh.world_scale,
            N_WORLD_VERTICES=n_world_vertices,
            CHROMA_WORLD_GEOMETRY=bool(chroma_world_compatibility),
            N_RAYS_IS_POINTER=False,
            INDIRECT=False,
            TLAS_STACK_CAPACITY=accelerator.tlas_stack_capacity,
            BLAS_STACK_CAPACITY=bvh.stack_capacity,
            BLOCK_SIZE=block_size,
            num_warps=1,
        )

    if count:
        if chroma_world_compatibility:
            refine_pmt_hits_chroma_world(
                accelerator, origin_tensor, direction_tensor, result
            )
        else:
            _finalize_instance_hits_kernel[(triton.cdiv(count, 256),)](
                bvh.triangle_vertices,
                accelerator.object_to_world_rotation,
                accelerator.channel_ids,
                result.triangle_ids,
                result.instance_ids,
                result.channel_ids,
                result.world_normals,
                count,
                count,
                result.triangle_ids,
                N_RAYS_IS_POINTER=False,
                INDIRECT=False,
                BLOCK_SIZE=256,
                num_warps=8,
            )

    # Capacities are topology-derived, so a well-formed accelerator cannot
    # overflow.  Keep the check observable and fall back to the independent
    # count/compact traversal if buffers are corrupted or externally replaced.
    if check_overflow and count and bool(torch.any(result.overflow).item()):
        return _nearest_pmt_hit_legacy(
            accelerator,
            origin_tensor,
            direction_tensor,
            tmax=tmax_tensor,
            last_instance=last_instance_tensor,
            last_triangle=last_triangle_tensor,
            ray_tile=ray_tile,
            workspace=workspace,
            out=result,
            check_overflow=True,
            chroma_world_compatibility=chroma_world_compatibility,
        )
    return result


def _validate_device_count_tensor(active_count: Any, device: Any) -> None:
    """Validate count storage while deliberately avoiding a value read."""

    if (
        not isinstance(active_count, torch.Tensor)
        or active_count.shape != (1,)
        or active_count.dtype != torch.int32
        or active_count.device != device
        or not active_count.is_contiguous()
    ):
        raise ValueError(
            "active_count must be contiguous CUDA int32 [1] on the ray device"
        )


def nearest_pmt_hit_tlas_device_count(
    accelerator: PMTInstanceAccelerator,
    origins: Any,
    directions: Any,
    active_count: Any,
    *,
    launch_capacity: Optional[int] = None,
    tmax: Optional[Any] = None,
    last_instance: Optional[Any] = None,
    last_triangle: Optional[Any] = None,
    workspace: Optional[PMTInstanceWorkspace] = None,
    out: Optional[PMTInstanceResult] = None,
    programs_per_sm: Optional[int] = None,
) -> PMTInstanceResult:
    """Run the production fused TLAS/BLAS from a device-resident ray count.

    ``origins`` and ``directions`` may own more storage than the current
    scheduler batch.  ``launch_capacity`` caps the grid and returned output
    views without reading ``active_count``; only that device-selected live
    prefix is touched.  The caller maintains
    ``0 <= active_count <= launch_capacity`` until the stream completes.

    This asynchronous building block intentionally excludes Chroma's
    flattened-world compatibility path and never polls the overflow output.
    Topology overflow is accumulated in ``workspace.sticky_overflow`` for one
    deferred audit after the event.  ``programs_per_sm`` tunes the bounded
    persistent traversal grid without changing work assignment semantics;
    zero requests the former uncapped capacity grid, and ``None`` uses
    :data:`DEVICE_TLAS_PROGRAMS_PER_SM`.
    """

    _require_backend()
    if (last_instance is None) != (last_triangle is None):
        raise ValueError("last_instance and last_triangle must be supplied together")
    device = accelerator.device
    origin_storage = _torch_rays(origins, "origins", device)
    direction_storage = _torch_rays(directions, "directions", device)
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
    _validate_device_count_tensor(active_count, device)
    origin_tensor = origin_storage[:capacity]
    direction_tensor = direction_storage[:capacity]

    def per_ray_prefix(values, default, dtype, name):
        if values is None:
            return torch.full(
                (capacity,), default, dtype=dtype, device=device
            )
        if not isinstance(values, torch.Tensor):
            if np.ndim(values) == 0:
                return torch.full(
                    (capacity,), values, dtype=dtype, device=device
                )
            values = torch.as_tensor(values, dtype=dtype, device=device)
        else:
            values = values.to(device=device, dtype=dtype)
        if values.ndim == 0:
            return torch.full(
                (capacity,), values.item(), dtype=dtype, device=device
            )
        if values.ndim != 1 or values.shape[0] < capacity:
            raise ValueError(
                f"{name} must be scalar or have at least launch_capacity entries"
            )
        return values[:capacity].contiguous()

    tmax_tensor = per_ray_prefix(tmax, float("inf"), torch.float32, "tmax")
    last_instance_tensor = per_ray_prefix(
        last_instance, -1, torch.int32, "last_instance"
    )
    last_triangle_tensor = per_ray_prefix(
        last_triangle, -1, torch.int32, "last_triangle"
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
    result = _validate_instance_result(out, capacity, device)
    if capacity == 0:
        return result

    bvh = accelerator.device_bvh
    # Establish observable miss sentinels for every live boundary row before
    # compacting the small subset which can reach the PMT union.  The inactive
    # capacity suffix is deliberately left untouched for queue reuse audits.
    _initialize_instance_results_kernel[(triton.cdiv(capacity, 256),)](
        result.distances,
        result.triangle_ids,
        result.instance_ids,
        result.channel_ids,
        result.world_normals,
        result.overflow,
        result.candidate_counts,
        active_count,
        capacity,
        N_RAYS_IS_POINTER=True,
        BLOCK_SIZE=256,
        num_warps=8,
    )
    workspace.device_candidate_count.zero_()
    compact_block = 256
    _compact_union_candidates_device_count_kernel[
        (triton.cdiv(capacity, compact_block),)
    ](
        origin_tensor,
        direction_tensor,
        tmax_tensor,
        accelerator.union_bounds_min,
        accelerator.union_bounds_max,
        active_count,
        workspace.active_ray_ids,
        workspace.device_candidate_count,
        capacity,
        BLOCK_SIZE=compact_block,
        num_warps=8,
    )
    block_size = 32
    multiprocessor_count = torch.cuda.get_device_properties(
        device
    ).multi_processor_count
    traversal_programs = _persistent_tlas_program_count(
        capacity,
        block_size,
        multiprocessor_count,
        programs_per_sm=programs_per_sm,
    )
    _nearest_pmt_tlas_kernel[(traversal_programs,)](
        accelerator.tlas_bounds_min,
        accelerator.tlas_bounds_max,
        accelerator.tlas_left_child,
        accelerator.tlas_right_child,
        accelerator.tlas_instance,
        accelerator.world_to_object_rotation,
        accelerator.world_to_object_translation,
        bvh.nodes,
        bvh.triangle_vertices,
        bvh.triangle_vertices,  # compile-time-dead compatibility pointer
        workspace.bvh_dummy,  # compile-time-dead compatibility pointer
        origin_tensor,
        direction_tensor,
        tmax_tensor,
        last_instance_tensor,
        last_triangle_tensor,
        workspace.tlas_stack,
        workspace.fused_traversal.stack,
        result.triangle_ids,
        result.distances,
        result.instance_ids,
        result.overflow,
        result.candidate_counts,
        workspace.sticky_overflow,
        workspace.device_candidate_count,
        capacity,
        workspace.active_ray_ids,
        bvh.world_origin[0],
        bvh.world_origin[1],
        bvh.world_origin[2],
        bvh.world_scale,
        N_WORLD_VERTICES=1,
        CHROMA_WORLD_GEOMETRY=False,
        N_RAYS_IS_POINTER=True,
        INDIRECT=True,
        TLAS_STACK_CAPACITY=accelerator.tlas_stack_capacity,
        BLAS_STACK_CAPACITY=bvh.stack_capacity,
        BLOCK_SIZE=block_size,
        num_warps=1,
    )
    _finalize_instance_hits_kernel[(triton.cdiv(capacity, 256),)](
        bvh.triangle_vertices,
        accelerator.object_to_world_rotation,
        accelerator.channel_ids,
        result.triangle_ids,
        result.instance_ids,
        result.channel_ids,
        result.world_normals,
        workspace.device_candidate_count,
        capacity,
        workspace.active_ray_ids,
        N_RAYS_IS_POINTER=True,
        INDIRECT=True,
        BLOCK_SIZE=256,
        num_warps=8,
    )
    return result


def nearest_pmt_hit(
    accelerator: PMTInstanceAccelerator,
    origins: Any,
    directions: Any,
    *,
    tmax: Optional[Any] = None,
    last_instance: Optional[Any] = None,
    last_triangle: Optional[Any] = None,
    ray_tile: Optional[int] = DEFAULT_RAY_TILE,
    workspace: Optional[PMTInstanceWorkspace] = None,
    out: Optional[PMTInstanceResult] = None,
    check_overflow: bool = True,
    use_tlas: bool = False,
    use_grid: bool = True,
    chroma_world_compatibility: bool = False,
) -> PMTInstanceResult:
    """Find the globally nearest exact PMT triangle for world-space rays.

    The production default uses the validated regular-grid locator when one
    is present and otherwise retains the general count/compact path.  Set
    ``use_grid=False`` to force the latter, or ``use_tlas=True`` to run the
    fused two-level traversal as an independent exact cross-check.  Both paths
    reject a positive triangle distance when the corresponding float32
    world-space position update would leave all three components unchanged.
    ``chroma_world_compatibility=True`` always retains the historical 81-box
    scan, tests Chroma's pre-flattened world vertices, and intentionally
    disables that post-Chroma loop-prevention fix.
    """

    if chroma_world_compatibility:
        _require_chroma_world_compatibility(accelerator)
        implementation = _nearest_pmt_hit_legacy
    elif use_tlas:
        implementation = _nearest_pmt_hit_tlas
    elif use_grid and accelerator.grid_locator is not None:
        implementation = _nearest_pmt_hit_grid
    else:
        implementation = _nearest_pmt_hit_legacy
    result = implementation(
        accelerator,
        origins,
        directions,
        tmax=tmax,
        last_instance=last_instance,
        last_triangle=last_triangle,
        ray_tile=ray_tile,
        workspace=workspace,
        out=out,
        check_overflow=check_overflow,
        chroma_world_compatibility=chroma_world_compatibility,
    )
    return result


def _numpy_ray_inputs(
    origins: Any,
    directions: Any,
    tmax: Optional[Any],
    last_instance: Optional[Any],
    last_triangle: Optional[Any],
):
    origin_array = np.asarray(origins, dtype=np.float32)
    direction_array = np.asarray(directions, dtype=np.float32)
    if origin_array.ndim != 2 or origin_array.shape[1] != 3:
        raise ValueError("origins must have shape (N, 3)")
    if direction_array.shape != origin_array.shape:
        raise ValueError("directions must have the same shape as origins")
    if not np.isfinite(origin_array).all() or not np.isfinite(direction_array).all():
        raise ValueError("ray origins and directions must be finite")
    count = len(origin_array)

    def per_ray(value, default, dtype, name):
        if value is None:
            return np.full(count, default, dtype=dtype)
        if np.ndim(value) == 0:
            return np.full(count, value, dtype=dtype)
        result = np.asarray(value, dtype=dtype)
        if result.shape != (count,):
            raise ValueError("%s must be scalar or have shape (N,)" % name)
        return result

    if (last_instance is None) != (last_triangle is None):
        raise ValueError("last_instance and last_triangle must be supplied together")
    return (
        np.ascontiguousarray(origin_array),
        np.ascontiguousarray(direction_array),
        per_ray(tmax, np.inf, np.float32, "tmax"),
        per_ray(last_instance, -1, np.int32, "last_instance"),
        per_ray(last_triangle, -1, np.int32, "last_triangle"),
    )


def _aabb_candidates_cpu(
    origins: np.ndarray,
    directions: np.ndarray,
    tmax: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    nonzero = directions != np.float32(0.0)
    safe = np.where(nonzero, directions, np.float32(1.0))
    first = (lower - origins) / safe
    second = (upper - origins) / safe
    near_axis = np.where(nonzero, np.minimum(first, second), -np.inf)
    far_axis = np.where(nonzero, np.maximum(first, second), np.inf)
    parallel_ok = np.all(
        nonzero | ((origins >= lower) & (origins <= upper)), axis=1
    )
    near = np.max(near_axis, axis=1)
    far = np.min(far_axis, axis=1)
    forward_near = np.maximum(near, np.float32(0.0))
    return parallel_ok & (far >= forward_near) & (forward_near <= tmax)


def _nearest_hit_cpu_with_world_progress(
    bvh: PackedBVH,
    local_origins: np.ndarray,
    local_directions: np.ndarray,
    world_origins: np.ndarray,
    world_directions: np.ndarray,
    tmax: np.ndarray,
    last_hit: np.ndarray,
):
    """Brute-force canonical oracle with the representable-progress rule."""

    triangle_ids = np.full(len(local_origins), -1, dtype=np.int32)
    distances = np.full(len(local_origins), np.inf, dtype=np.float32)
    tri = bvh.triangle_vertices
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    edge1, edge2 = v1 - v0, v2 - v0
    epsilon = np.float32(1.0e-6)
    determinant_epsilon = np.finfo(np.float32).eps
    triangle_indices = np.arange(len(tri), dtype=np.int32)

    for ray_id, (origin, direction) in enumerate(
        zip(local_origins, local_directions)
    ):
        h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
        determinant = np.einsum("ij,ij->i", edge1, h, dtype=np.float32)
        accepted = (determinant < -determinant_epsilon) | (
            determinant > determinant_epsilon
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            reciprocal = np.float32(1.0) / determinant
            s = origin - v0
            u = reciprocal * np.einsum("ij,ij->i", s, h, dtype=np.float32)
            q = np.cross(s, edge1)
            v = reciprocal * np.einsum(
                "ij,j->i", q, direction, dtype=np.float32
            )
            distance = reciprocal * np.einsum(
                "ij,ij->i", edge2, q, dtype=np.float32
            )
        delta = np.multiply(
            distance[:, None], world_directions[ray_id], dtype=np.float32
        )
        updated = np.add(
            world_origins[ray_id], delta, dtype=np.float32
        )
        advances = np.any(updated != world_origins[ray_id], axis=1)
        accepted &= (
            (u >= -epsilon)
            & (u <= np.float32(1.0) + epsilon)
            & (v >= -epsilon)
            & ((u + v) <= np.float32(1.0) + epsilon)
            & (distance > epsilon)
            & (distance < tmax[ray_id])
            & advances
            & (triangle_indices != last_hit[ray_id])
        )
        if np.any(accepted):
            candidates = np.where(accepted, distance, np.float32(np.inf))
            triangle = int(np.argmin(candidates))
            triangle_ids[ray_id] = triangle
            distances[ray_id] = candidates[triangle]
    return triangle_ids, distances


def nearest_pmt_hit_cpu(
    scene_or_accelerator: Any,
    origins: Any,
    directions: Any,
    *,
    tmax: Optional[Any] = None,
    last_instance: Optional[Any] = None,
    last_triangle: Optional[Any] = None,
) -> PMTInstanceResult:
    """NumPy brute-force oracle for the complete PMT instance collection."""

    (
        origin_array,
        direction_array,
        tmax_array,
        last_instance_array,
        last_triangle_array,
    ) = _numpy_ray_inputs(
        origins, directions, tmax, last_instance, last_triangle
    )
    if isinstance(scene_or_accelerator, PMTInstanceAccelerator):
        accelerator = scene_or_accelerator
        bvh = accelerator.host_bvh
        channels_table = accelerator.host_channel_ids
        w2o_r = accelerator.host_world_to_object_rotation
        w2o_t = accelerator.host_world_to_object_translation
        o2w_r = accelerator.host_object_to_world_rotation
        bounds_min = accelerator.host_bounds_min
        bounds_max = accelerator.host_bounds_max
    else:
        (
            vertices,
            triangles_table,
            channels_table,
            w2o_r,
            w2o_t,
            o2w_r,
            raw_min,
            raw_max,
        ) = _scene_arrays(scene_or_accelerator)
        bvh = build_packed_bvh(vertices, triangles_table)
        bounds_min, bounds_max = _padded_bounds(vertices, raw_min, raw_max)

    ray_count = len(origin_array)
    best_distance = np.full(ray_count, np.inf, dtype=np.float32)
    best_triangle = np.full(ray_count, -1, dtype=np.int32)
    best_instance = np.full(ray_count, -1, dtype=np.int32)
    candidate_counts = np.zeros(ray_count, dtype=np.int32)

    for instance in range(len(channels_table)):
        candidate = _aabb_candidates_cpu(
            origin_array,
            direction_array,
            tmax_array,
            bounds_min[instance],
            bounds_max[instance],
        )
        candidate_counts += candidate.astype(np.int32)
        indices = np.flatnonzero(candidate)
        if len(indices) == 0:
            continue
        rotation = w2o_r[instance]
        local_origins = (
            origin_array[indices] @ rotation.T + w2o_t[instance]
        ).astype(np.float32)
        local_directions = (direction_array[indices] @ rotation.T).astype(
            np.float32
        )
        upper_bound = np.minimum(
            tmax_array[indices], best_distance[indices]
        ).astype(np.float32)
        excluded = np.where(
            last_instance_array[indices] == instance,
            last_triangle_array[indices],
            -1,
        ).astype(np.int32)
        local_triangle, local_distance = _nearest_hit_cpu_with_world_progress(
            bvh,
            local_origins,
            local_directions,
            origin_array[indices],
            direction_array[indices],
            upper_bound,
            excluded,
        )
        hit = local_triangle >= 0
        if np.any(hit):
            selected = indices[hit]
            # nearest_hit_cpu used best_distance as an exclusive upper bound;
            # only strict improvements can reach this branch, preserving the
            # lowest retained instance on an exact tie.
            best_distance[selected] = local_distance[hit]
            best_triangle[selected] = local_triangle[hit]
            best_instance[selected] = instance

    channel_ids = np.full(ray_count, -1, dtype=np.int32)
    normals = np.zeros((ray_count, 3), dtype=np.float32)
    hit_indices = np.flatnonzero(best_triangle >= 0)
    triangle_vertices = bvh.triangle_vertices
    for ray in hit_indices:
        triangle = int(best_triangle[ray])
        instance = int(best_instance[ray])
        points = triangle_vertices[triangle]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        normal = normal / np.linalg.norm(normal)
        normals[ray] = o2w_r[instance] @ normal
        channel_ids[ray] = channels_table[instance]

    return PMTInstanceResult(
        triangle_ids=best_triangle,
        distances=best_distance,
        instance_ids=best_instance,
        channel_ids=channel_ids,
        world_normals=normals,
        overflow=np.zeros(ray_count, dtype=np.uint8),
        candidate_counts=candidate_counts,
    )


__all__ = [
    "DEFAULT_GRID_CANDIDATES",
    "DEFAULT_RAY_TILE",
    "PMTGridLocator",
    "PMTInstanceAccelerator",
    "PMTInstanceBackendUnavailable",
    "PMTInstanceResult",
    "PMTInstanceWorkspace",
    "build_pmt_instance_accelerator",
    "nearest_pmt_hit",
    "nearest_pmt_hit_tlas_device_count",
    "nearest_pmt_hit_cpu",
    "triton_instance_available",
]
