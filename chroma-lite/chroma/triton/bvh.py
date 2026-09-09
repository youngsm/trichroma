"""Host-side construction for compact, instance-reusable triangle BVHs.

The layout intentionally follows Chroma's packed node representation.  Each
node is four uint32 values.  The first three values contain conservative
uint16 lower/upper bounds, and the fourth contains either an original triangle
ID or a child-range pointer and count.  A canonical mesh can therefore be
built once and shared by many rigid instances (for example, all detector
PMTs); rays passed to the traversal backend are expected to already be in that
canonical, instance-local coordinate system.

This module depends only on NumPy.  Importing or building a BVH does not require
PyTorch, Triton, CUDA, or a working GPU driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

DEGREE = 4
CHILD_BITS = 28
MAX_FIXED = (1 << 16) - 1
_FIXED_INTERVALS = (1 << 16) - 2


@dataclass(frozen=True)
class TraversalResult:
    """Nearest-hit result returned by the CPU and Triton frontends."""

    triangle_ids: Any
    distances: Any
    overflow: Optional[Any] = None
    visits: Optional[Any] = None
    max_stack: Optional[Any] = None


@dataclass(frozen=True)
class PackedBVH:
    """A degree-4 packed BVH and its canonical triangle data.

    ``triangle_vertices`` is indexed by the original input triangle ID, not by
    Morton order.  Leaf nodes retain those original IDs, which lets downstream
    material and surface tables use their existing indexing unchanged.
    """

    nodes: np.ndarray
    triangle_vertices: np.ndarray
    world_origin: np.ndarray
    world_scale: np.float32
    layer_offsets: Tuple[int, ...]
    layer_counts: Tuple[int, ...]
    degree: int = DEGREE

    @property
    def triangle_count(self) -> int:
        return int(self.triangle_vertices.shape[0])

    @property
    def node_count(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def depth(self) -> int:
        """Number of edges from the root layer to the leaf layer."""

        return len(self.layer_counts) - 1

    @property
    def stack_capacity(self) -> int:
        """Conservative per-ray stack capacity for this tree topology.

        Chroma traversal finishes a sibling range before popping a child range.
        At most ``degree - 1`` pending ranges are added at each deeper level.
        Using this topology-derived capacity avoids the arbitrary fixed stacks
        and possible out-of-bounds writes common in older traversal kernels.
        """

        return max(1, 1 + max(0, self.depth - 1) * (self.degree - 1))

    @property
    def nbytes(self) -> int:
        return int(self.nodes.nbytes + self.triangle_vertices.nbytes + self.world_origin.nbytes)

    def to_triton(self, device: Optional[Any] = None):
        """Upload this BVH for Triton traversal.

        The optional backend is imported only when this method is called, so a
        NumPy-only Chroma installation can freely import and use this module.
        """

        from .bvh_kernels import DevicePackedBVH

        return DevicePackedBVH.from_host(self, device=device)


def _as_mesh_arrays(vertices: Any, triangles: Optional[Any]) -> Tuple[np.ndarray, np.ndarray]:
    if triangles is None:
        if not hasattr(vertices, "vertices") or not hasattr(vertices, "triangles"):
            raise TypeError(
                "triangles is required unless vertices is a mesh-like object "
                "with .vertices and .triangles"
            )
        triangles = vertices.triangles
        vertices = vertices.vertices

    vertex_array = np.asarray(vertices, dtype=np.float32)
    triangle_array = np.asarray(triangles)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if triangle_array.ndim != 2 or triangle_array.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3)")
    if len(vertex_array) == 0:
        raise ValueError("cannot build a BVH without vertices")
    if len(triangle_array) == 0:
        raise ValueError("cannot build a BVH without triangles")
    if not np.isfinite(vertex_array).all():
        raise ValueError("vertices must all be finite")
    if not np.issubdtype(triangle_array.dtype, np.integer):
        raise TypeError("triangle indices must be integers")
    if np.any(triangle_array < 0) or np.any(triangle_array >= len(vertex_array)):
        raise ValueError("triangle index is outside the vertex array")
    if len(triangle_array) >= (1 << CHILD_BITS):
        raise ValueError("packed BVHs support fewer than 2**28 triangles")

    return (
        np.ascontiguousarray(vertex_array),
        np.ascontiguousarray(triangle_array, dtype=np.int64),
    )


def _spread3_16(values: np.ndarray) -> np.ndarray:
    """Spread each uint16 bit over every third bit of a uint64."""

    result = np.asarray(values, dtype=np.uint64)
    result = (result | (result << np.uint64(16))) & np.uint64(0x00000000FF0000FF)
    result = (result | (result << np.uint64(8))) & np.uint64(0x000000F00F00F00F)
    result = (result | (result << np.uint64(4))) & np.uint64(0x00000C30C30C30C3)
    result = (result | (result << np.uint64(2))) & np.uint64(0x0000249249249249)
    return result


def _quantize(points: np.ndarray, world_origin: np.ndarray, world_scale: np.float32) -> np.ndarray:
    # C/CUDA conversion to unsigned integer truncates non-negative values.
    scaled = (points - world_origin) / world_scale
    return np.floor(np.clip(scaled, 0.0, float(_FIXED_INTERVALS))).astype(np.uint32)


def build_packed_bvh(
    vertices: Any,
    triangles: Optional[Any] = None,
    *,
    degree: int = DEGREE,
) -> PackedBVH:
    """Build a compact degree-4 Morton BVH for a canonical triangle mesh.

    Parameters
    ----------
    vertices, triangles:
        Arrays with shapes ``(N, 3)`` and ``(M, 3)``.  As a convenience,
        ``vertices`` may instead be a mesh-like object exposing both arrays and
        ``triangles`` may be omitted.
    degree:
        Present to make the fixed degree explicit.  Only degree four is
        supported by the packed format and the Triton traversal kernel.
    """

    if degree != DEGREE:
        raise ValueError("the packed traversal format requires degree=4")

    vertex_array, triangle_array = _as_mesh_arrays(vertices, triangles)
    triangle_vertices = np.ascontiguousarray(vertex_array[triangle_array], dtype=np.float32)
    world_origin = np.ascontiguousarray(vertex_array.min(axis=0), dtype=np.float32)
    extent = np.float32(np.max(vertex_array.max(axis=0) - world_origin))
    # A fully degenerate mesh is still representable and safely traversable.
    world_scale = np.float32(extent / _FIXED_INTERVALS) if extent > 0 else np.float32(1.0)

    lower = triangle_vertices.min(axis=1)
    upper = triangle_vertices.max(axis=1)
    centroids = triangle_vertices.mean(axis=1, dtype=np.float32)
    q_lower = _quantize(lower, world_origin, world_scale)
    q_lower = np.where(q_lower > 0, q_lower - np.uint32(1), q_lower).astype(np.uint32)
    q_upper = np.minimum(
        _quantize(upper, world_origin, world_scale) + np.uint32(1),
        np.uint32(MAX_FIXED),
    ).astype(np.uint32)
    q_centroid = _quantize(centroids, world_origin, world_scale)

    morton = (
        _spread3_16(q_centroid[:, 0])
        | (_spread3_16(q_centroid[:, 1]) << np.uint64(1))
        | (_spread3_16(q_centroid[:, 2]) << np.uint64(2))
    )
    # Stable ordering makes equal-centroid meshes reproducible while keeping
    # leaf child IDs in the original triangle namespace.
    order = np.argsort(morton, kind="stable")
    q_lower = q_lower[order]
    q_upper = q_upper[order]

    leaves = np.empty((len(triangle_array), 4), dtype=np.uint32)
    leaves[:, :3] = q_lower | (q_upper << np.uint32(16))
    leaves[:, 3] = order.astype(np.uint32)

    # Build bottom-up.  Child pointers are initially relative to the next
    # layer, then rebased after all layer sizes are known.
    layers = [leaves]
    while len(layers[0]) > 1:
        children = layers[0]
        parents = _parent_layer(children, degree)
        layers.insert(0, parents)

    layer_counts = tuple(int(len(layer)) for layer in layers)
    offsets = np.cumsum((0,) + layer_counts[:-1], dtype=np.int64)
    if int(sum(layer_counts)) >= (1 << CHILD_BITS):
        raise ValueError("packed child pointers exceed 28 bits")
    for layer_number in range(len(layers) - 1):
        layers[layer_number][:, 3] += np.uint32(offsets[layer_number + 1])

    nodes = np.ascontiguousarray(np.concatenate(layers, axis=0), dtype=np.uint32)
    layer_offsets = tuple(int(value) for value in offsets)

    # Treat built geometry as immutable.  Device upload and caching rely on
    # host arrays not changing behind the BVH metadata.
    nodes.flags.writeable = False
    triangle_vertices.flags.writeable = False
    world_origin.flags.writeable = False

    return PackedBVH(
        nodes=nodes,
        triangle_vertices=triangle_vertices,
        world_origin=world_origin,
        world_scale=world_scale,
        layer_offsets=layer_offsets,
        layer_counts=layer_counts,
        degree=degree,
    )


def _parent_layer(children, degree):
    """Bit-identical packed parents, with bounded NumPy work buffers."""
    count = (len(children) + degree - 1) // degree
    parents = np.empty((count, 4), dtype=np.uint32)
    for start in range(0, count, 1 << 18):
        stop = min(count, start + (1 << 18))
        group = children[start * degree : min(stop * degree, len(children))]
        if len(group) % degree:
            padding = np.zeros((degree - len(group) % degree, 4), np.uint32)
            padding[:, :3] = np.uint32(0xFFFF)  # neutral lower=65535, upper=0
            group = np.concatenate((group, padding))
        group = group.reshape(-1, degree, 4)
        lower = (group[:, :, :3] & np.uint32(0xFFFF)).min(axis=1)
        upper = (group[:, :, :3] >> np.uint32(16)).max(axis=1)
        parents[start:stop, :3] = lower | (upper << np.uint32(16))
        first = np.arange(start, stop, dtype=np.uint32) * degree
        size = np.minimum(degree, len(children) - first).astype(np.uint32)
        parents[start:stop, 3] = (size << np.uint32(CHILD_BITS)) | first
    return parents


def _ray_inputs(
    origins: Any,
    directions: Any,
    tmax: Optional[Any],
    last_hit: Optional[Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    origin_array = np.asarray(origins, dtype=np.float32)
    direction_array = np.asarray(directions, dtype=np.float32)
    if origin_array.ndim != 2 or origin_array.shape[1] != 3:
        raise ValueError("origins must have shape (N, 3)")
    if direction_array.shape != origin_array.shape:
        raise ValueError("directions must have the same shape as origins")
    if not np.isfinite(origin_array).all() or not np.isfinite(direction_array).all():
        raise ValueError("ray origins and directions must be finite")
    if tmax is None:
        tmax_array = np.full(len(origin_array), np.inf, dtype=np.float32)
    elif np.ndim(tmax) == 0:
        tmax_array = np.full(len(origin_array), tmax, dtype=np.float32)
    else:
        tmax_array = np.asarray(tmax, dtype=np.float32)
        if tmax_array.shape != (len(origin_array),):
            raise ValueError("tmax must be scalar or have shape (N,)")
    if last_hit is None:
        last_hit_array = np.full(len(origin_array), -1, dtype=np.int32)
    elif np.ndim(last_hit) == 0:
        last_hit_array = np.full(len(origin_array), last_hit, dtype=np.int32)
    else:
        last_hit_array = np.asarray(last_hit, dtype=np.int32)
        if last_hit_array.shape != (len(origin_array),):
            raise ValueError("last_hit must be scalar or have shape (N,)")
    return origin_array, direction_array, tmax_array, last_hit_array


def _nearest_triangle(v0, edge1, edge2, origin, direction, tmax, last_hit, ids):
    """Independent Moller--Trumbore arithmetic shared by CPU oracles."""
    dtype = v0.dtype.type
    epsilon = dtype(1.0e-9 if dtype == np.float64 else 1.0e-6)
    h = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, h, dtype=dtype)
    determinant_epsilon = np.finfo(dtype).eps
    accepted = (determinant < -determinant_epsilon) | (determinant > determinant_epsilon)
    with np.errstate(divide="ignore", invalid="ignore"):
        reciprocal = dtype(1.0) / determinant
        s = origin - v0
        u = reciprocal * np.einsum("ij,ij->i", s, h, dtype=dtype)
        q = np.cross(s, edge1)
        v = reciprocal * np.einsum("ij,j->i", q, direction, dtype=dtype)
        distance = reciprocal * np.einsum("ij,ij->i", edge2, q, dtype=dtype)
    accepted &= (
        (u >= -epsilon)
        & (u <= dtype(1.0) + epsilon)
        & (v >= -epsilon)
        & (u + v <= dtype(1.0) + epsilon)
        & (distance > epsilon)
        & (distance < tmax)
        & (ids != last_hit)
    )
    if not np.any(accepted):
        return -1, np.float32(np.inf)
    candidates = np.where(accepted, distance, np.float32(np.inf))
    best = int(np.argmin(candidates))
    return int(ids[best]), candidates[best]


def nearest_hit_cpu(
    bvh: PackedBVH,
    origins: Any,
    directions: Any,
    *,
    tmax: Optional[Any] = None,
    last_hit: Optional[Any] = None,
    high_precision: bool = False,
) -> TraversalResult:
    """Float32 brute-force nearest-hit reference for local-coordinate rays.

    This is intended for validation and CPU-only testing, not production
    transport.  Directions should be normalized if returned ``t`` values are
    to represent physical distances.  ``tmax`` is an exclusive upper bound.
    """

    origin_array, direction_array, tmax_array, last_hit_array = _ray_inputs(
        origins, directions, tmax, last_hit
    )
    triangle_ids = np.full(len(origin_array), -1, dtype=np.int32)
    distances = np.full(len(origin_array), np.inf, dtype=np.float32)
    tri = bvh.triangle_vertices
    if high_precision:
        tri = tri.astype(np.float64)
        origin_array = origin_array.astype(np.float64)
        direction_array = direction_array.astype(np.float64)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    edge1, edge2 = v1 - v0, v2 - v0
    ids = np.arange(len(tri), dtype=np.int32)

    for ray_id, (origin, direction) in enumerate(zip(origin_array, direction_array)):
        triangle_ids[ray_id], distances[ray_id] = _nearest_triangle(
            v0, edge1, edge2, origin, direction, tmax_array[ray_id], last_hit_array[ray_id], ids
        )

    return TraversalResult(triangle_ids=triangle_ids, distances=distances)


def nearest_hit_bvh_cpu(
    bvh, origins, directions, *, tmax=None, last_hit=None, high_precision=False
):
    """CPU BVH broadphase followed by the brute-force oracle's exact leaf test.

    Node slabs use float64 and one extra quantization unit of padding; there
    is no nearest-hit pruning or candidate cap. Original triangle-ID sorting
    preserves the brute-force tie rule. This makes large-scene CPU transport
    tractable while keeping the unaccelerated reference available for checks.
    """
    origins, directions, limits, previous = _ray_inputs(origins, directions, tmax, last_hit)
    triangles = np.full(len(origins), -1, np.int32)
    distances = np.full(len(origins), np.inf, np.float32)
    world_origin, scale = bvh.world_origin.astype(float), float(bvh.world_scale)
    child_mask = np.uint32((1 << CHILD_BITS) - 1)
    for ray, (origin, direction) in enumerate(zip(origins, directions)):
        frontier, leaves = np.array([0], np.int64), []
        origin64, direction64 = origin.astype(float), direction.astype(float)
        parallel = direction64 == 0
        divisor = np.where(parallel, 1.0, direction64)
        while len(frontier):
            nodes = bvh.nodes[frontier]
            lower = world_origin + ((nodes[:, :3] & np.uint32(0xFFFF)).astype(float) - 1) * scale
            upper = world_origin + ((nodes[:, :3] >> np.uint32(16)).astype(float) + 1) * scale
            first, second = (lower - origin64) / divisor, (upper - origin64) / divisor
            enter = np.where(parallel, -np.inf, np.minimum(first, second)).max(axis=1)
            leave = np.where(parallel, np.inf, np.maximum(first, second)).min(axis=1)
            hit = (leave >= np.maximum(enter, 0.0)) & ~np.any(
                parallel & ((origin64 < lower) | (origin64 > upper)), axis=1
            )
            nodes = nodes[hit]
            counts = nodes[:, 3] >> np.uint32(CHILD_BITS)
            children = nodes[:, 3] & child_mask
            leaves.extend(children[counts == 0].tolist())
            inner = counts > 0
            if not inner.any():
                break
            slots = np.arange(int(counts[inner].max()))
            frontier = (children[inner, None] + slots)[slots[None, :] < counts[inner, None]]
        if not leaves:
            continue
        ids = np.unique(leaves).astype(np.int32)
        tri = bvh.triangle_vertices[ids]
        if high_precision:
            tri = tri.astype(np.float64)
            origin, direction = origin64, direction64
        triangles[ray], distances[ray] = _nearest_triangle(
            tri[:, 0],
            tri[:, 1] - tri[:, 0],
            tri[:, 2] - tri[:, 0],
            origin,
            direction,
            limits[ray],
            previous[ray],
            ids,
        )
    return TraversalResult(triangle_ids=triangles, distances=distances)


__all__ = [
    "DEGREE",
    "PackedBVH",
    "TraversalResult",
    "build_packed_bvh",
    "nearest_hit_cpu",
    "nearest_hit_bvh_cpu",
]
