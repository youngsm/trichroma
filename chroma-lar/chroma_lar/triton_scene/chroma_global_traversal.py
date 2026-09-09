"""Strict Triton replay of an exported Chroma global mesh BVH.

This is a proof/certificate backend, not the production accelerator.  It
uploads Chroma's own flattened vertex/index buffers and recursive-grid node
topology, retains the historical global triangle namespace, and executes the
same eager-sibling traversal order as ``chroma/cuda/mesh.h``.  The arithmetic
which is sensitive at shared edges is expressed as opaque PTX matching
Chroma's CUDA ``--use_fast_math`` build.

The optimized detector backend deliberately uses analytic boxes, reachability
pruning, and a shared PMT BLAS instead.  Keeping this module independent makes
the reference implementation useful as an oracle without adding data or
branches to that production path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .chroma_global_bvh import ChromaGlobalBVHArtifact


_IMPORT_ERROR = None
try:
    import torch
    import triton
    import triton.language as tl
except Exception as error:  # Keep host artifact inspection importable.
    torch = None
    triton = None
    tl = None
    _IMPORT_ERROR = error


CHROMA_STACK_CAPACITY = 1000
DEFAULT_CERTIFICATE_RAY_TILE = 8192


class ChromaGlobalTraversalUnavailable(RuntimeError):
    """Raised when strict global traversal cannot run on this host."""


@dataclass(frozen=True)
class ChromaGlobalHit:
    """One exact flattened-mesh boundary result per input ray.

    IDs are the original Chroma flattened arrays' global IDs.  ``raw_normals``
    are geometric triangle normals; ``surface_normals`` are oriented against
    the incident direction exactly as ``fill_state`` does.  Misses have a
    triangle and all metadata IDs of ``-1``, infinite distance, zero normals,
    and false ``inside_to_outside``.
    """

    triangle_ids: Any
    distances: Any
    raw_normals: Any
    surface_normals: Any
    solid_ids: Any
    channel_ids: Any
    surface_indices: Any
    material_inner_indices: Any
    material_outer_indices: Any
    material_from_indices: Any
    material_to_indices: Any
    inside_to_outside: Any
    overflow: Any


@dataclass
class ChromaGlobalTraversalWorkspace:
    """Reusable SoA traversal stack for a bounded certificate tile."""

    stack: Any
    sticky_overflow: Any
    ray_capacity: int

    @classmethod
    def allocate(
        cls, accelerator: "ChromaGlobalBVHDevice", ray_capacity: int
    ) -> "ChromaGlobalTraversalWorkspace":
        _require_backend()
        capacity = int(ray_capacity)
        if capacity < 0:
            raise ValueError("ray_capacity must be non-negative")
        return cls(
            stack=torch.empty(
                max(1, accelerator.stack_capacity * capacity),
                dtype=torch.int32,
                device=accelerator.device,
            ),
            sticky_overflow=torch.zeros(
                1, dtype=torch.int32, device=accelerator.device
            ),
            ray_capacity=capacity,
        )

    def clear_sticky_overflow(self) -> None:
        self.sticky_overflow.zero_()

    def sticky_overflowed(self) -> bool:
        return bool(self.sticky_overflow.item())


@dataclass(frozen=True)
class ChromaGlobalBVHDevice:
    """A fingerprinted Chroma global artifact resident on one CUDA device."""

    nodes: Any
    vertices: Any
    triangles: Any
    solid_ids: Any
    channel_ids: Any
    surface_indices: Any
    material_inner_indices: Any
    material_outer_indices: Any
    colors: Any
    world_origin: tuple[float, float, float]
    world_scale: float
    mesh_md5: str
    traversal_sha256: str
    sha256: str
    triangle_count: int
    stack_capacity: int

    @property
    def device(self) -> Any:
        return self.nodes.device

    @classmethod
    def from_host(
        cls,
        artifact: ChromaGlobalBVHArtifact,
        device: Optional[Any] = None,
        *,
        expected_traversal_sha256: str,
        expected_sha256: Optional[str] = None,
        target_material_names: Optional[Sequence[str]] = None,
        target_surface_names: Optional[Sequence[str]] = None,
    ) -> "ChromaGlobalBVHDevice":
        """Validate, pin, and upload an immutable reference artifact.

        Requiring the caller's expected traversal SHA-256 prevents accidentally
        using a structurally valid BVH produced by a different Chroma build or
        detector configuration as an equivalence oracle. The traversal digest
        covers all stable topology, geometry, solid, color, and channel words.
        ``expected_sha256`` may additionally pin one exact archive, including
        Chroma's process-order-dependent material/surface index words.
        """

        _require_backend()
        artifact.validate()
        if artifact.stack_capacity > CHROMA_STACK_CAPACITY:
            raise ValueError(
                "reference BVH needs %d pending ranges, exceeding Chroma's "
                "historical stack capacity %d"
                % (artifact.stack_capacity, CHROMA_STACK_CAPACITY)
            )
        observed_traversal_sha256 = str(artifact.traversal_sha256)
        if not expected_traversal_sha256:
            raise ValueError(
                "expected_traversal_sha256 is required for fail-closed upload"
            )
        if observed_traversal_sha256 != str(expected_traversal_sha256):
            raise ValueError(
                "Chroma global BVH traversal fingerprint mismatch: expected "
                "%s, got %s"
                % (expected_traversal_sha256, observed_traversal_sha256)
            )
        observed_sha256 = str(artifact.sha256)
        if expected_sha256 is not None and observed_sha256 != str(expected_sha256):
            raise ValueError(
                "Chroma global BVH archive fingerprint mismatch: expected %s, "
                "got %s" % (expected_sha256, observed_sha256)
            )
        if (target_material_names is None) != (target_surface_names is None):
            raise ValueError(
                "target material and surface names must be supplied together"
            )
        if target_material_names is None:
            material_inner = artifact.material1_index
            material_outer = artifact.material2_index
            surface = artifact.surface_index
        else:
            material_inner, material_outer, surface = (
                artifact.remap_optical_indices(
                    target_material_names, target_surface_names
                )
            )
        selected_device = torch.device("cuda" if device is None else device)
        if selected_device.type != "cuda":
            raise ValueError("Chroma global traversal requires a CUDA device")

        def upload(value: np.ndarray, dtype: Any) -> Any:
            host = np.array(value, copy=True, order="C")
            return torch.from_numpy(host).to(
                device=selected_device, dtype=dtype
            ).contiguous()

        return cls(
            # Preserve all uint32 words through an int32 view because older
            # Torch CUDA builds do not provide every uint32 operation.
            nodes=upload(artifact.nodes.view(np.int32), torch.int32),
            vertices=upload(artifact.vertices, torch.float32),
            triangles=upload(artifact.triangles, torch.int32),
            solid_ids=upload(artifact.solid_id, torch.int32),
            channel_ids=upload(artifact.triangle_channel_index, torch.int32),
            surface_indices=upload(surface, torch.int32),
            material_inner_indices=upload(material_inner, torch.int32),
            material_outer_indices=upload(material_outer, torch.int32),
            colors=upload(artifact.colors.view(np.int32), torch.int32),
            world_origin=tuple(float(x) for x in artifact.world_origin),
            world_scale=float(artifact.world_scale),
            mesh_md5=str(artifact.mesh_md5),
            traversal_sha256=observed_traversal_sha256,
            sha256=observed_sha256,
            triangle_count=int(artifact.triangles.shape[0]),
            stack_capacity=int(artifact.stack_capacity),
        )

    def allocate_workspace(
        self, ray_capacity: int
    ) -> ChromaGlobalTraversalWorkspace:
        return ChromaGlobalTraversalWorkspace.allocate(self, ray_capacity)


def triton_chroma_global_available(*, require_cuda: bool = False) -> bool:
    """Return whether the optional Triton stack (and optionally CUDA) exists."""

    if torch is None or triton is None:
        return False
    return bool(torch.cuda.is_available()) if require_cuda else True


def _require_backend() -> None:
    if torch is None or triton is None:
        detail = "" if _IMPORT_ERROR is None else ": %s" % (_IMPORT_ERROR,)
        raise ChromaGlobalTraversalUnavailable(
            "PyTorch and Triton are required" + detail
        )
    if not torch.cuda.is_available():
        raise ChromaGlobalTraversalUnavailable(
            "a CUDA device visible to PyTorch is required"
        )


if triton is not None and torch is not None:

    @triton.jit
    def _chroma_box_words(
        packed_x,
        packed_y,
        packed_z,
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        world_x,
        world_y,
        world_z,
        world_scale,
    ):
        """Return Chroma CUDA's box ``tmin`` and intersection predicate."""

        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .u32 qlo;
                .reg .u32 qhi;
                .reg .f32 flo;
                .reg .f32 fhi;
                .reg .f32 lo;
                .reg .f32 hi;
                .reg .f32 inverse;
                .reg .f32 negative_origin_inverse;
                .reg .f32 negative_origin;
                .reg .f32 t0;
                .reg .f32 t1;
                .reg .f32 axis_near;
                .reg .f32 axis_far;
                .reg .f32 tmin;
                .reg .f32 tmax;
                .reg .pred finite_axis;
                .reg .pred misses;
                mov.f32 tmin, 0f00000000;
                mov.f32 tmax, 0f7F800000;

                and.b32 qlo, $2, 0x0000ffff;
                shr.u32 qhi, $2, 16;
                cvt.rn.f32.u32 flo, qlo;
                cvt.rn.f32.u32 fhi, qhi;
                fma.rn.ftz.f32 lo, flo, $14, $11;
                fma.rn.ftz.f32 hi, fhi, $14, $11;
                rcp.approx.ftz.f32 inverse, $8;
                neg.ftz.f32 negative_origin, $5;
                div.approx.ftz.f32 negative_origin_inverse, negative_origin, $8;
                testp.finite.f32 finite_axis, inverse;
                fma.rn.ftz.f32 t0, lo, inverse, negative_origin_inverse;
                fma.rn.ftz.f32 t1, hi, inverse, negative_origin_inverse;
                min.ftz.f32 axis_near, t0, t1;
                max.ftz.f32 axis_far, t0, t1;
                @finite_axis max.ftz.f32 tmin, tmin, axis_near;
                @finite_axis min.ftz.f32 tmax, tmax, axis_far;

                and.b32 qlo, $3, 0x0000ffff;
                shr.u32 qhi, $3, 16;
                cvt.rn.f32.u32 flo, qlo;
                cvt.rn.f32.u32 fhi, qhi;
                fma.rn.ftz.f32 lo, flo, $14, $12;
                fma.rn.ftz.f32 hi, fhi, $14, $12;
                rcp.approx.ftz.f32 inverse, $9;
                neg.ftz.f32 negative_origin, $6;
                div.approx.ftz.f32 negative_origin_inverse, negative_origin, $9;
                testp.finite.f32 finite_axis, inverse;
                fma.rn.ftz.f32 t0, lo, inverse, negative_origin_inverse;
                fma.rn.ftz.f32 t1, hi, inverse, negative_origin_inverse;
                min.ftz.f32 axis_near, t0, t1;
                max.ftz.f32 axis_far, t0, t1;
                @finite_axis max.ftz.f32 tmin, tmin, axis_near;
                @finite_axis min.ftz.f32 tmax, tmax, axis_far;

                and.b32 qlo, $4, 0x0000ffff;
                shr.u32 qhi, $4, 16;
                cvt.rn.f32.u32 flo, qlo;
                cvt.rn.f32.u32 fhi, qhi;
                fma.rn.ftz.f32 lo, flo, $14, $13;
                fma.rn.ftz.f32 hi, fhi, $14, $13;
                rcp.approx.ftz.f32 inverse, $10;
                neg.ftz.f32 negative_origin, $7;
                div.approx.ftz.f32 negative_origin_inverse, negative_origin, $10;
                testp.finite.f32 finite_axis, inverse;
                fma.rn.ftz.f32 t0, lo, inverse, negative_origin_inverse;
                fma.rn.ftz.f32 t1, hi, inverse, negative_origin_inverse;
                min.ftz.f32 axis_near, t0, t1;
                max.ftz.f32 axis_far, t0, t1;
                @finite_axis max.ftz.f32 tmin, tmin, axis_near;
                @finite_axis min.ftz.f32 tmax, tmax, axis_far;

                setp.gt.ftz.f32 misses, tmin, tmax;
                mov.f32 $0, tmin;
                selp.u32 $1, 0, 1, misses;
            }
            """,
            constraints="=f,=r,r,r,r,f,f,f,f,f,f,f,f,f,f",
            args=[
                packed_x,
                packed_y,
                packed_z,
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                world_x,
                world_y,
                world_z,
                world_scale,
            ],
            dtype=(tl.float32, tl.int32),
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _chroma_triangle_words(
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
    ):
        """Return Chroma's determinant, barycentrics, and distance words."""

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
            ],
            dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _chroma_normal_words(
        v0x,
        v0y,
        v0z,
        v1x,
        v1y,
        v1z,
        v2x,
        v2y,
        v2z,
        dx,
        dy,
        dz,
    ):
        """Return normalized raw triangle normal and its incident-ray dot."""

        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .f32 e1x;
                .reg .f32 e1y;
                .reg .f32 e1z;
                .reg .f32 e12x;
                .reg .f32 e12y;
                .reg .f32 e12z;
                .reg .f32 left;
                .reg .f32 right;
                .reg .f32 nx;
                .reg .f32 ny;
                .reg .f32 nz;
                .reg .f32 norm2;
                .reg .f32 length;
                .reg .f32 direction_product;
                .reg .f32 negative_product;
                .reg .f32 dot_value;
                sub.ftz.f32 e1x, $7, $4;
                sub.ftz.f32 e1y, $8, $5;
                sub.ftz.f32 e1z, $9, $6;
                sub.ftz.f32 e12x, $10, $7;
                sub.ftz.f32 e12y, $11, $8;
                sub.ftz.f32 e12z, $12, $9;
                mul.ftz.f32 left, e1y, e12z;
                mul.ftz.f32 right, e1z, e12y;
                sub.ftz.f32 nx, left, right;
                mul.ftz.f32 left, e1z, e12x;
                mul.ftz.f32 right, e1x, e12z;
                sub.ftz.f32 ny, left, right;
                mul.ftz.f32 left, e1x, e12y;
                mul.ftz.f32 right, e1y, e12x;
                sub.ftz.f32 nz, left, right;
                mul.ftz.f32 norm2, ny, ny;
                fma.rn.ftz.f32 norm2, nx, nx, norm2;
                fma.rn.ftz.f32 norm2, nz, nz, norm2;
                sqrt.approx.ftz.f32 length, norm2;
                div.approx.ftz.f32 $0, nx, length;
                div.approx.ftz.f32 $1, ny, length;
                div.approx.ftz.f32 $2, nz, length;
                mul.ftz.f32 direction_product, $1, $14;
                neg.ftz.f32 negative_product, direction_product;
                mul.ftz.f32 direction_product, $0, $13;
                sub.ftz.f32 dot_value, negative_product, direction_product;
                mul.ftz.f32 direction_product, $2, $15;
                sub.ftz.f32 $3, dot_value, direction_product;
            }
            """,
            constraints="=f,=f,=f,=f,f,f,f,f,f,f,f,f,f,f,f,f",
            args=[
                v0x,
                v0y,
                v0z,
                v1x,
                v1y,
                v1z,
                v2x,
                v2y,
                v2z,
                dx,
                dy,
                dz,
            ],
            dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _chroma_oriented_normal_words(nx, ny, nz, incident_dot):
        """Apply fill_state's branch/negation without losing signed zero."""

        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .pred outside_now;
                .reg .f32 flipped;
                // CUDA's source comparison is the ordered expression
                // ``incident_dot > 0.0f``.  In particular, NaN must take the
                // false/flip arm rather than being treated as greater.
                setp.gt.ftz.f32 outside_now, $6, 0f00000000;
                neg.ftz.f32 flipped, $3;
                selp.f32 $0, $3, flipped, outside_now;
                neg.ftz.f32 flipped, $4;
                selp.f32 $1, $4, flipped, outside_now;
                neg.ftz.f32 flipped, $5;
                selp.f32 $2, $5, flipped, outside_now;
            }
            """,
            constraints="=f,=f,=f,f,f,f,f",
            args=[nx, ny, nz, incident_dot],
            dtype=(tl.float32, tl.float32, tl.float32),
            is_pure=True,
            pack=1,
        )

    @triton.jit
    def _nearest_chroma_global_kernel(
        nodes,
        vertices,
        triangle_indices,
        solid_ids,
        channel_ids,
        surface_indices,
        material_inner_indices,
        material_outer_indices,
        origins,
        directions,
        last_triangles,
        stack,
        out_triangles,
        out_distances,
        out_raw_normals,
        out_surface_normals,
        out_solids,
        out_channels,
        out_surfaces,
        out_material_inner,
        out_material_outer,
        out_material_from,
        out_material_to,
        out_inside_to_outside,
        out_overflow,
        sticky_overflow,
        n_rays,
        world_x,
        world_y,
        world_z,
        world_scale,
        STACK_CAPACITY: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        ray = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        valid = ray < n_rays
        ox = tl.load(origins + ray * 3, mask=valid, other=0.0)
        oy = tl.load(origins + ray * 3 + 1, mask=valid, other=0.0)
        oz = tl.load(origins + ray * 3 + 2, mask=valid, other=0.0)
        dx = tl.load(directions + ray * 3, mask=valid, other=1.0)
        dy = tl.load(directions + ray * 3 + 1, mask=valid, other=1.0)
        dz = tl.load(directions + ray * 3 + 2, mask=valid, other=1.0)
        previous_triangle = tl.load(
            last_triangles + ray, mask=valid, other=-1
        )

        node_index = tl.zeros((BLOCK_SIZE,), tl.int32)
        range_remaining = tl.full((BLOCK_SIZE,), 1, tl.int32)
        stack_pointer = tl.zeros((BLOCK_SIZE,), tl.int32)
        active = valid
        best_triangle = tl.full((BLOCK_SIZE,), -1, tl.int32)
        best_distance = tl.full((BLOCK_SIZE,), -1.0, tl.float32)
        overflow = tl.zeros((BLOCK_SIZE,), tl.int1)

        # This state machine is the vectorized form of mesh.h's for-loop over
        # a child range plus LIFO arrays of pending child ranges.  Children are
        # visited in the exact node-array order and equal distances retain the
        # first successful leaf because the update comparison is strict.
        while tl.sum(active.to(tl.int32), axis=0) != 0:
            packed_x = tl.load(
                nodes + node_index * 4, mask=active, other=0
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
            box_near, intersects_box = _chroma_box_words(
                packed_x,
                packed_y,
                packed_z,
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                world_x,
                world_y,
                world_z,
                world_scale,
            )
            box_hit = (
                active
                & (intersects_box != 0)
                & ((best_triangle < 0) | (box_near <= best_distance))
            )

            child_count = (packed_w >> 28).to(tl.int32)
            child = (packed_w & 0x0FFFFFFF).to(tl.int32)
            is_leaf = (
                box_hit
                & (child_count == 0)
                & (child != previous_triangle)
            )
            is_inner = box_hit & (child_count != 0)

            index_base = child * 3
            i0 = tl.load(
                triangle_indices + index_base, mask=is_leaf, other=0
            ).to(tl.int32)
            i1 = tl.load(
                triangle_indices + index_base + 1, mask=is_leaf, other=0
            ).to(tl.int32)
            i2 = tl.load(
                triangle_indices + index_base + 2, mask=is_leaf, other=0
            ).to(tl.int32)
            v0 = i0 * 3
            v1 = i1 * 3
            v2 = i2 * 3
            v0x = tl.load(vertices + v0, mask=is_leaf, other=0.0)
            v0y = tl.load(vertices + v0 + 1, mask=is_leaf, other=0.0)
            v0z = tl.load(vertices + v0 + 2, mask=is_leaf, other=0.0)
            v1x = tl.load(vertices + v1, mask=is_leaf, other=0.0)
            v1y = tl.load(vertices + v1 + 1, mask=is_leaf, other=0.0)
            v1z = tl.load(vertices + v1 + 2, mask=is_leaf, other=0.0)
            v2x = tl.load(vertices + v2, mask=is_leaf, other=0.0)
            v2y = tl.load(vertices + v2 + 1, mask=is_leaf, other=0.0)
            v2z = tl.load(vertices + v2 + 2, mask=is_leaf, other=0.0)
            determinant, u, v, distance = _chroma_triangle_words(
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
            # CUDA rejects only the *open* interval (-FLT_EPSILON,
            # +FLT_EPSILON); equality at either endpoint remains a candidate.
            determinant_ok = (
                (determinant <= -1.1920928955078125e-7)
                | (determinant >= 1.1920928955078125e-7)
            )
            triangle_hit = (
                is_leaf
                & determinant_ok
                & (u >= -1.0e-6)
                & (u <= 1.0 + 1.0e-6)
                & (v >= -1.0e-6)
                & ((u + v) <= 1.0 + 1.0e-6)
                & (distance > 1.0e-6)
                & (distance < float("inf"))
                & ((best_triangle < 0) | (distance < best_distance))
            )
            best_triangle = tl.where(
                triangle_hit, child, best_triangle
            )
            best_distance = tl.where(
                triangle_hit, distance, best_distance
            )

            can_push = stack_pointer < STACK_CAPACITY
            push = is_inner & can_push
            overflow |= is_inner & ~can_push
            packed_range = child | (child_count << 28)
            tl.store(
                stack + stack_pointer * n_rays + ray,
                packed_range,
                mask=push,
            )
            pushed_pointer = stack_pointer + push.to(tl.int32)
            next_remaining = range_remaining - 1
            stay_in_range = active & (next_remaining > 0) & ~overflow
            should_pop = active & ~stay_in_range & ~overflow
            has_pending = should_pop & (pushed_pointer > 0)
            top = tl.maximum(pushed_pointer - 1, 0)
            popped = tl.load(
                stack + top * n_rays + ray,
                mask=has_pending,
                other=0,
            ).to(tl.uint32)
            stack_pointer = pushed_pointer - has_pending.to(tl.int32)
            node_index = tl.where(
                stay_in_range,
                node_index + 1,
                (popped & 0x0FFFFFFF).to(tl.int32),
            )
            range_remaining = tl.where(
                stay_in_range,
                next_remaining,
                (popped >> 28).to(tl.int32),
            )
            active = stay_in_range | has_pending

        hit = valid & (best_triangle >= 0)
        safe_triangle = tl.maximum(best_triangle, 0)
        final_index_base = safe_triangle * 3
        i0 = tl.load(
            triangle_indices + final_index_base, mask=hit, other=0
        ).to(tl.int32)
        i1 = tl.load(
            triangle_indices + final_index_base + 1, mask=hit, other=0
        ).to(tl.int32)
        i2 = tl.load(
            triangle_indices + final_index_base + 2, mask=hit, other=0
        ).to(tl.int32)
        v0 = i0 * 3
        v1 = i1 * 3
        v2 = i2 * 3
        v0x = tl.load(vertices + v0, mask=hit, other=0.0)
        v0y = tl.load(vertices + v0 + 1, mask=hit, other=0.0)
        v0z = tl.load(vertices + v0 + 2, mask=hit, other=0.0)
        v1x = tl.load(vertices + v1, mask=hit, other=0.0)
        v1y = tl.load(vertices + v1 + 1, mask=hit, other=0.0)
        v1z = tl.load(vertices + v1 + 2, mask=hit, other=0.0)
        v2x = tl.load(vertices + v2, mask=hit, other=0.0)
        v2y = tl.load(vertices + v2 + 1, mask=hit, other=0.0)
        v2z = tl.load(vertices + v2 + 2, mask=hit, other=0.0)
        nx, ny, nz, incident_dot = _chroma_normal_words(
            v0x,
            v0y,
            v0z,
            v1x,
            v1y,
            v1z,
            v2x,
            v2y,
            v2z,
            dx,
            dy,
            dz,
        )
        outside_now = hit & (incident_dot > 0.0)
        surface_x, surface_y, surface_z = _chroma_oriented_normal_words(
            nx, ny, nz, incident_dot
        )
        inner = tl.load(
            material_inner_indices + safe_triangle, mask=hit, other=-1
        ).to(tl.int32)
        outer = tl.load(
            material_outer_indices + safe_triangle, mask=hit, other=-1
        ).to(tl.int32)

        tl.store(out_triangles + ray, best_triangle, mask=valid)
        tl.store(
            out_distances + ray,
            tl.where(hit, best_distance, float("inf")),
            mask=valid,
        )
        tl.store(
            out_raw_normals + ray * 3,
            tl.where(hit, nx, 0.0),
            mask=valid,
        )
        tl.store(
            out_raw_normals + ray * 3 + 1,
            tl.where(hit, ny, 0.0),
            mask=valid,
        )
        tl.store(
            out_raw_normals + ray * 3 + 2,
            tl.where(hit, nz, 0.0),
            mask=valid,
        )
        tl.store(
            out_surface_normals + ray * 3,
            tl.where(hit, surface_x, 0.0),
            mask=valid,
        )
        tl.store(
            out_surface_normals + ray * 3 + 1,
            tl.where(hit, surface_y, 0.0),
            mask=valid,
        )
        tl.store(
            out_surface_normals + ray * 3 + 2,
            tl.where(hit, surface_z, 0.0),
            mask=valid,
        )
        tl.store(
            out_solids + ray,
            tl.load(solid_ids + safe_triangle, mask=hit, other=-1),
            mask=valid,
        )
        tl.store(
            out_channels + ray,
            tl.load(channel_ids + safe_triangle, mask=hit, other=-1),
            mask=valid,
        )
        tl.store(
            out_surfaces + ray,
            tl.load(surface_indices + safe_triangle, mask=hit, other=-1),
            mask=valid,
        )
        tl.store(out_material_inner + ray, inner, mask=valid)
        tl.store(out_material_outer + ray, outer, mask=valid)
        tl.store(
            out_material_from + ray,
            tl.where(hit, tl.where(outside_now, outer, inner), -1),
            mask=valid,
        )
        tl.store(
            out_material_to + ray,
            tl.where(hit, tl.where(outside_now, inner, outer), -1),
            mask=valid,
        )
        tl.store(
            out_inside_to_outside + ray,
            (hit & ~outside_now).to(tl.uint8),
            mask=valid,
        )
        tl.store(out_overflow + ray, overflow.to(tl.uint8), mask=valid)
        tl.atomic_or(
            sticky_overflow + tl.zeros((BLOCK_SIZE,), tl.int32),
            tl.full((BLOCK_SIZE,), 1, tl.int32),
            mask=valid & overflow,
        )


def _torch_rays(values: Any, name: str, device: Any) -> Any:
    if isinstance(values, torch.Tensor):
        result = values.to(device=device, dtype=torch.float32)
    else:
        result = torch.as_tensor(values, dtype=torch.float32, device=device)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("%s must have shape (N, 3)" % name)
    return result.contiguous()


def _torch_last_triangles(values: Any, count: int, device: Any) -> Any:
    if values is None:
        return torch.full((count,), -1, dtype=torch.int32, device=device)
    if isinstance(values, torch.Tensor):
        result = values.to(device=device, dtype=torch.int32)
    elif np.ndim(values) == 0:
        result = torch.full(
            (count,), int(values), dtype=torch.int32, device=device
        )
    else:
        result = torch.as_tensor(values, dtype=torch.int32, device=device)
    if result.ndim == 0:
        result = torch.full(
            (count,), int(result.item()), dtype=torch.int32, device=device
        )
    if result.shape != (count,):
        raise ValueError("last_triangle must be scalar or have shape (N,)")
    return result.contiguous()


def _allocate_result(count: int, device: Any) -> ChromaGlobalHit:
    def integers() -> Any:
        return torch.empty(count, dtype=torch.int32, device=device)

    return ChromaGlobalHit(
        triangle_ids=integers(),
        distances=torch.empty(count, dtype=torch.float32, device=device),
        raw_normals=torch.empty((count, 3), dtype=torch.float32, device=device),
        surface_normals=torch.empty(
            (count, 3), dtype=torch.float32, device=device
        ),
        solid_ids=integers(),
        channel_ids=integers(),
        surface_indices=integers(),
        material_inner_indices=integers(),
        material_outer_indices=integers(),
        material_from_indices=integers(),
        material_to_indices=integers(),
        inside_to_outside=torch.empty(count, dtype=torch.uint8, device=device),
        overflow=torch.empty(count, dtype=torch.uint8, device=device),
    )


def nearest_chroma_global_hit(
    accelerator: ChromaGlobalBVHDevice,
    origins: Any,
    directions: Any,
    *,
    last_triangle: Optional[Any] = None,
    ray_tile: int = DEFAULT_CERTIFICATE_RAY_TILE,
    workspace: Optional[ChromaGlobalTraversalWorkspace] = None,
    check_overflow: bool = True,
) -> ChromaGlobalHit:
    """Replay Chroma's exact flattened-mesh nearest-boundary query.

    ``last_triangle`` is in Chroma's global flattened triangle namespace.
    ``ray_tile`` bounds the certificate stack allocation; it has no effect on
    traversal or tie order.  This function intentionally has no production
    ``tmax`` optimization because historical ``intersect_mesh`` traces the
    global mesh independently before comparing its result with analytic wires.
    """

    _require_backend()
    device = accelerator.device
    origin_tensor = _torch_rays(origins, "origins", device)
    direction_tensor = _torch_rays(directions, "directions", device)
    if direction_tensor.shape != origin_tensor.shape:
        raise ValueError("directions must have the same shape as origins")
    count = int(origin_tensor.shape[0])
    last_tensor = _torch_last_triangles(last_triangle, count, device)
    tile = int(ray_tile)
    if tile <= 0:
        raise ValueError("ray_tile must be positive")
    tile_capacity = min(count, tile)
    if workspace is None:
        workspace = accelerator.allocate_workspace(tile_capacity)
    elif workspace.stack.device != device:
        raise ValueError("workspace and accelerator must use the same device")
    elif workspace.ray_capacity < tile_capacity:
        raise ValueError("workspace ray capacity is smaller than ray_tile")
    result = _allocate_result(count, device)
    block_size = 32
    for start in range(0, count, tile):
        stop = min(count, start + tile)
        tile_count = stop - start
        _nearest_chroma_global_kernel[(triton.cdiv(tile_count, block_size),)](
            accelerator.nodes,
            accelerator.vertices,
            accelerator.triangles,
            accelerator.solid_ids,
            accelerator.channel_ids,
            accelerator.surface_indices,
            accelerator.material_inner_indices,
            accelerator.material_outer_indices,
            origin_tensor[start:stop],
            direction_tensor[start:stop],
            last_tensor[start:stop],
            workspace.stack,
            result.triangle_ids[start:stop],
            result.distances[start:stop],
            result.raw_normals[start:stop],
            result.surface_normals[start:stop],
            result.solid_ids[start:stop],
            result.channel_ids[start:stop],
            result.surface_indices[start:stop],
            result.material_inner_indices[start:stop],
            result.material_outer_indices[start:stop],
            result.material_from_indices[start:stop],
            result.material_to_indices[start:stop],
            result.inside_to_outside[start:stop],
            result.overflow[start:stop],
            workspace.sticky_overflow,
            tile_count,
            accelerator.world_origin[0],
            accelerator.world_origin[1],
            accelerator.world_origin[2],
            accelerator.world_scale,
            STACK_CAPACITY=accelerator.stack_capacity,
            BLOCK_SIZE=block_size,
            num_warps=1,
        )
    if check_overflow and count and bool(torch.any(result.overflow).item()):
        raise RuntimeError("historical Chroma global BVH stack overflowed")
    return result


__all__ = [
    "CHROMA_STACK_CAPACITY",
    "DEFAULT_CERTIFICATE_RAY_TILE",
    "ChromaGlobalBVHDevice",
    "ChromaGlobalHit",
    "ChromaGlobalTraversalUnavailable",
    "ChromaGlobalTraversalWorkspace",
    "nearest_chroma_global_hit",
    "triton_chroma_global_available",
]
