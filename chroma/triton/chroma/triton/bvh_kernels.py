"""Optional Triton backend for instance-local packed-BVH traversal.

Importing this module is safe when PyTorch, Triton, or CUDA is unavailable.
Backend availability is checked when a BVH is uploaded or a kernel is run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .bvh import PackedBVH, TraversalResult


_IMPORT_ERROR = None
try:
    import torch
    import triton
    import triton.language as tl
except Exception as error:  # Optional acceleration must not break host imports.
    torch = None
    triton = None
    tl = None
    _IMPORT_ERROR = error


class TritonUnavailableError(RuntimeError):
    """Raised when the optional Triton/CUDA backend cannot be used."""


def triton_available(require_cuda: bool = False) -> bool:
    """Return whether the optional software stack (and optionally CUDA) exists."""

    if torch is None or triton is None:
        return False
    return bool(torch.cuda.is_available()) if require_cuda else True


def _require_backend() -> None:
    if torch is None or triton is None:
        detail = "" if _IMPORT_ERROR is None else ": %s" % (_IMPORT_ERROR,)
        raise TritonUnavailableError("PyTorch and Triton are required" + detail)
    if not torch.cuda.is_available():
        raise TritonUnavailableError("a CUDA device visible to PyTorch is required")


if triton is not None and torch is not None:

    @triton.jit
    def _nearest_hit_kernel(
        nodes,
        triangle_vertices,
        origins,
        directions,
        tmax_values,
        last_hit_values,
        stack,
        out_triangle,
        out_distance,
        out_overflow,
        out_visits,
        out_max_stack,
        n_rays,
        world_x,
        world_y,
        world_z,
        world_scale,
        STACK_CAPACITY: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        COLLECT_STATS: tl.constexpr,
    ):
        ray = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = ray < n_rays

        ox = tl.load(origins + ray * 3 + 0, mask=valid, other=0.0)
        oy = tl.load(origins + ray * 3 + 1, mask=valid, other=0.0)
        oz = tl.load(origins + ray * 3 + 2, mask=valid, other=0.0)
        dx = tl.load(directions + ray * 3 + 0, mask=valid, other=1.0)
        dy = tl.load(directions + ray * 3 + 1, mask=valid, other=1.0)
        dz = tl.load(directions + ray * 3 + 2, mask=valid, other=1.0)
        ray_tmax = tl.load(tmax_values + ray, mask=valid, other=0.0)
        previous_triangle = tl.load(
            last_hit_values + ray, mask=valid, other=-1
        )

        invx, invy, invz = 1.0 / dx, 1.0 / dy, 1.0 / dz
        neg_ox_inv = -ox / dx
        neg_oy_inv = -oy / dy
        neg_oz_inv = -oz / dz

        node_index = tl.zeros((BLOCK_SIZE,), tl.int32)
        range_remaining = tl.full((BLOCK_SIZE,), 1, tl.int32)
        stack_pointer = tl.zeros((BLOCK_SIZE,), tl.int32)
        active = valid
        best_triangle = tl.full((BLOCK_SIZE,), -1, tl.int32)
        best_distance = ray_tmax
        overflow = tl.zeros((BLOCK_SIZE,), tl.int1)
        if COLLECT_STATS:
            visits = tl.zeros((BLOCK_SIZE,), tl.int32)
            maximum_stack = tl.zeros((BLOCK_SIZE,), tl.int32)

        # Chroma-compatible eager sibling traversal.  Every intersected inner
        # sibling contributes a child range before the current range is popped.
        while tl.sum(active.to(tl.int32), axis=0) != 0:
            if COLLECT_STATS:
                visits += active.to(tl.int32)

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

            # Preserve Chroma's operation order at quantized faces.
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

            triangle_base = child * 9
            v0x = tl.load(
                triangle_vertices + triangle_base + 0, mask=is_leaf, other=0.0
            )
            v0y = tl.load(
                triangle_vertices + triangle_base + 1, mask=is_leaf, other=0.0
            )
            v0z = tl.load(
                triangle_vertices + triangle_base + 2, mask=is_leaf, other=0.0
            )
            v1x = tl.load(
                triangle_vertices + triangle_base + 3, mask=is_leaf, other=0.0
            )
            v1y = tl.load(
                triangle_vertices + triangle_base + 4, mask=is_leaf, other=0.0
            )
            v1z = tl.load(
                triangle_vertices + triangle_base + 5, mask=is_leaf, other=0.0
            )
            v2x = tl.load(
                triangle_vertices + triangle_base + 6, mask=is_leaf, other=0.0
            )
            v2y = tl.load(
                triangle_vertices + triangle_base + 7, mask=is_leaf, other=0.0
            )
            v2z = tl.load(
                triangle_vertices + triangle_base + 8, mask=is_leaf, other=0.0
            )

            edge1x, edge1y, edge1z = v1x - v0x, v1y - v0y, v1z - v0z
            edge2x, edge2y, edge2z = v2x - v0x, v2y - v0y, v2z - v0z
            hx = dy * edge2z - dz * edge2y
            hy = dz * edge2x - dx * edge2z
            hz = dx * edge2y - dy * edge2x
            determinant = edge1x * hx + edge1y * hy + edge1z * hz
            determinant_ok = (determinant < -1.1920928955078125e-7) | (
                determinant > 1.1920928955078125e-7
            )
            reciprocal = 1.0 / determinant
            sx, sy, sz = ox - v0x, oy - v0y, oz - v0z
            u = reciprocal * (sx * hx + sy * hy + sz * hz)
            qx = sy * edge1z - sz * edge1y
            qy = sz * edge1x - sx * edge1z
            qz = sx * edge1y - sy * edge1x
            v = reciprocal * (dx * qx + dy * qy + dz * qz)
            distance = reciprocal * (
                edge2x * qx + edge2y * qy + edge2z * qz
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
            )
            best_distance = tl.where(triangle_hit, distance, best_distance)
            best_triangle = tl.where(triangle_hit, child, best_triangle)

            can_push = stack_pointer < STACK_CAPACITY
            push = is_inner & can_push
            overflow |= is_inner & ~can_push
            packed_range = child | (child_count << 28)
            push_address = stack_pointer * n_rays + ray
            tl.store(stack + push_address, packed_range, mask=push)
            pushed_pointer = stack_pointer + push.to(tl.int32)
            if COLLECT_STATS:
                maximum_stack = tl.maximum(maximum_stack, pushed_pointer)

            next_remaining = range_remaining - 1
            stay_in_range = active & (next_remaining > 0) & ~overflow
            should_pop = active & ~stay_in_range & ~overflow
            has_pending = should_pop & (pushed_pointer > 0)
            top = tl.maximum(pushed_pointer - 1, 0)
            popped = tl.load(
                stack + top * n_rays + ray, mask=has_pending, other=0
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

        tl.store(out_triangle + ray, best_triangle, mask=valid)
        final_distance = tl.where(
            best_triangle >= 0, best_distance, float("inf")
        )
        tl.store(out_distance + ray, final_distance, mask=valid)
        tl.store(out_overflow + ray, overflow.to(tl.uint8), mask=valid)
        if COLLECT_STATS:
            tl.store(out_visits + ray, visits, mask=valid)
            tl.store(out_max_stack + ray, maximum_stack, mask=valid)


@dataclass
class TraversalWorkspace:
    """Reusable flat per-ray stack storage for nearest-hit traversal."""

    stack: Any
    ray_capacity: int
    stack_capacity: int

    @classmethod
    def allocate(
        cls, bvh: "DevicePackedBVH", ray_capacity: int
    ) -> "TraversalWorkspace":
        _require_backend()
        if ray_capacity < 0:
            raise ValueError("ray_capacity must be non-negative")
        stack = torch.empty(
            max(1, bvh.stack_capacity * ray_capacity),
            dtype=torch.int32,
            device=bvh.nodes.device,
        )
        return cls(
            stack=stack,
            ray_capacity=int(ray_capacity),
            stack_capacity=bvh.stack_capacity,
        )


@dataclass(frozen=True)
class DevicePackedBVH:
    """Packed BVH data resident on one CUDA device."""

    nodes: Any
    triangle_vertices: Any
    world_origin: tuple
    world_scale: float
    stack_capacity: int
    triangle_count: int

    @classmethod
    def from_host(
        cls, bvh: PackedBVH, device: Optional[Any] = None
    ) -> "DevicePackedBVH":
        _require_backend()
        selected_device = torch.device("cuda" if device is None else device)
        if selected_device.type != "cuda":
            raise ValueError("Triton BVHs must be uploaded to a CUDA device")
        # View uint32 node bits as int32 for compatibility with PyTorch builds
        # that do not expose uint32 CUDA tensors.  The kernel casts back.
        host_nodes = bvh.nodes.view(np.int32).copy()
        host_triangles = np.array(
            bvh.triangle_vertices, dtype=np.float32, order="C", copy=True
        )
        nodes = torch.from_numpy(host_nodes).to(selected_device).contiguous()
        triangle_vertices = (
            torch.from_numpy(host_triangles).to(selected_device).contiguous()
        )
        return cls(
            nodes=nodes,
            triangle_vertices=triangle_vertices,
            world_origin=tuple(float(value) for value in bvh.world_origin),
            world_scale=float(bvh.world_scale),
            stack_capacity=bvh.stack_capacity,
            triangle_count=bvh.triangle_count,
        )

    def allocate_workspace(self, ray_capacity: int) -> TraversalWorkspace:
        return TraversalWorkspace.allocate(self, ray_capacity)


def _ray_tensor(values: Any, name: str, device: Any) -> Any:
    if isinstance(values, torch.Tensor):
        result = values.to(device=device, dtype=torch.float32)
    else:
        result = torch.as_tensor(values, dtype=torch.float32, device=device)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("%s must have shape (N, 3)" % name)
    return result.contiguous()


def _per_ray_tensor(
    values: Optional[Any],
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
        return torch.full(
            (count,), result.item(), dtype=dtype, device=device
        )
    if result.shape != (count,):
        raise ValueError("%s must be scalar or have shape (N,)" % name)
    return result.contiguous()


def nearest_hit_local(
    bvh: DevicePackedBVH,
    origins: Any,
    directions: Any,
    *,
    tmax: Optional[Any] = None,
    last_hit: Optional[Any] = None,
    block_size: int = 32,
    workspace: Optional[TraversalWorkspace] = None,
    collect_stats: bool = False,
    check_overflow: bool = False,
) -> TraversalResult:
    """Find nearest triangles for rays in one instance's local coordinates.

    ``directions`` should be normalized for ``distances`` to be physical path
    lengths.  ``tmax`` is an optional scalar or per-ray exclusive upper bound.
    ``last_hit`` optionally excludes one original triangle ID per ray, matching
    Chroma's self-intersection behavior.

    The topology-derived stack allocation is sufficient for every path in a
    BVH produced by :func:`build_packed_bvh`.  ``overflow`` is nevertheless
    returned so corrupted or foreign node buffers fail observably rather than
    writing outside their workspace.
    """

    _require_backend()
    if block_size not in (16, 32, 64, 128, 256):
        raise ValueError("block_size must be one of 16, 32, 64, 128, or 256")
    device = bvh.nodes.device
    origin_tensor = _ray_tensor(origins, "origins", device)
    direction_tensor = _ray_tensor(directions, "directions", device)
    if direction_tensor.shape != origin_tensor.shape:
        raise ValueError("directions must have the same shape as origins")
    ray_count = int(origin_tensor.shape[0])
    tmax_tensor = _per_ray_tensor(
        tmax,
        count=ray_count,
        default=float("inf"),
        dtype=torch.float32,
        name="tmax",
        device=device,
    )
    last_hit_tensor = _per_ray_tensor(
        last_hit,
        count=ray_count,
        default=-1,
        dtype=torch.int32,
        name="last_hit",
        device=device,
    )

    if workspace is None:
        workspace = bvh.allocate_workspace(ray_count)
    elif workspace.stack.device != device:
        raise ValueError("workspace and BVH must be on the same device")
    elif workspace.stack_capacity < bvh.stack_capacity:
        raise ValueError("workspace stack capacity is too small for this BVH")
    elif workspace.ray_capacity < ray_count:
        raise ValueError("workspace ray capacity is smaller than this batch")

    triangle_ids = torch.empty(ray_count, dtype=torch.int32, device=device)
    distances = torch.empty(ray_count, dtype=torch.float32, device=device)
    overflow = torch.empty(ray_count, dtype=torch.uint8, device=device)
    if collect_stats:
        visits = torch.empty(ray_count, dtype=torch.int32, device=device)
        max_stack = torch.empty(ray_count, dtype=torch.int32, device=device)
    else:
        visits = None
        max_stack = None

    if ray_count:
        dummy = torch.empty(1, dtype=torch.int32, device=device)
        grid = (triton.cdiv(ray_count, block_size),)
        _nearest_hit_kernel[grid](
            bvh.nodes,
            bvh.triangle_vertices,
            origin_tensor,
            direction_tensor,
            tmax_tensor,
            last_hit_tensor,
            workspace.stack,
            triangle_ids,
            distances,
            overflow,
            visits if collect_stats else dummy,
            max_stack if collect_stats else dummy,
            ray_count,
            bvh.world_origin[0],
            bvh.world_origin[1],
            bvh.world_origin[2],
            bvh.world_scale,
            STACK_CAPACITY=bvh.stack_capacity,
            BLOCK_SIZE=block_size,
            COLLECT_STATS=collect_stats,
            num_warps=max(1, block_size // 32),
        )

    if check_overflow and ray_count and bool(torch.any(overflow).item()):
        raise RuntimeError("packed BVH traversal stack overflowed")
    return TraversalResult(
        triangle_ids=triangle_ids,
        distances=distances,
        overflow=overflow,
        visits=visits,
        max_stack=max_stack,
    )


__all__ = [
    "DevicePackedBVH",
    "TraversalWorkspace",
    "TritonUnavailableError",
    "nearest_hit_local",
    "triton_available",
]
