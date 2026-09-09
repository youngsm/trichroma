"""Fused certified-portal classification and production boundary physics.

The device-resident scheduler normally needs three launches for a boundary
population which is known to leave the certified empty LAr box: classify the
five direct faces, materialize a compact hit record, then consume that record
with the boundary kernel.  This module removes both intermediate arrays and
the second launch.  One kernel:

* classifies the exact same conservative five-face portal;
* advances direct lanes with the corrected production boundary algorithm;
* appends active direct lanes to an existing collision-carry queue; and
* compacts only ambiguous lanes into a fallback queue for exact geometry.

This is deliberately a production-only API.  Random-tape replay, legacy
specular reflection, and Chroma-global geometry compatibility continue to use
the established unfused path.  In particular, every state store and RNG
counter increment below is masked by ``direct``: a fallback photon is
bit-for-bit untouched until the ordinary geometry/boundary pipeline consumes
it.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .portals import PortalDescriptor


_MATERIAL_TABLES = (
    "tables_material_refractive_index",
    "tables_material_absorption_length",
    "tables_material_scattering_length",
)
_SURFACE_TABLES = (
    "tables_surface_detect",
    "tables_surface_absorb",
    "tables_surface_reflect_diffuse",
    "tables_surface_reflect_specular",
)


@dataclass
class FusedPortalBoundaryWorkspace:
    """Reusable storage for the exact-geometry fallback population only."""

    buffer: Any
    count: Any

    @classmethod
    def allocate(
        cls, capacity: int, device: Any = "cuda"
    ) -> "FusedPortalBoundaryWorkspace":
        import torch

        capacity = _index(capacity, "capacity")
        if capacity < 0:
            raise ValueError("fused portal workspace capacity cannot be negative")
        return cls(
            buffer=torch.empty(capacity, dtype=torch.int32, device=device),
            count=torch.zeros(1, dtype=torch.int32, device=device),
        )

    @property
    def capacity(self) -> int:
        return int(self.buffer.numel())

    def queue(self) -> Any:
        from chroma.triton.transport import DeviceQueue

        return DeviceQueue(self.buffer, self.count)


def _index(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error


def _validate_capacities(
    input_capacity: Any,
    launch_capacity: Any,
    input_storage: int,
    fallback_storage: int,
    carry_storage: int,
) -> tuple[int, int]:
    """Validate the host-side bounds without reading a device counter."""

    input_capacity = _index(input_capacity, "input_capacity")
    launch_capacity = _index(launch_capacity, "launch_capacity")
    if input_capacity < 0 or input_capacity > input_storage:
        raise ValueError("input_capacity must fit the boundary queue buffer")
    if launch_capacity < input_capacity:
        raise ValueError(
            "launch_capacity must be at least input_capacity to avoid truncation"
        )
    if launch_capacity > fallback_storage:
        raise ValueError(
            "fallback workspace capacity must be at least launch_capacity"
        )
    if launch_capacity > carry_storage:
        raise ValueError("carry queue capacity must be at least launch_capacity")
    if launch_capacity > np.iinfo(np.int32).max:
        raise ValueError("launch_capacity exceeds the DeviceQueue int32 range")
    return input_capacity, launch_capacity


def _tensor_storage_overlaps(left: Any, right: Any) -> bool:
    if left.device != right.device or left.numel() == 0 or right.numel() == 0:
        return False
    left_begin = int(left.data_ptr())
    right_begin = int(right.data_ptr())
    left_end = left_begin + left.numel() * left.element_size()
    right_end = right_begin + right.numel() * right.element_size()
    return max(left_begin, right_begin) < min(left_end, right_end)


def _load_fused_portal_boundary_kernel():
    cached = getattr(_load_fused_portal_boundary_kernel, "_cached", None)
    if cached is not None:
        return cached
    try:
        import triton
        import triton.language as tl
        from chroma.triton.physics_kernels import (
            fresnel_step as physics_fresnel_step,
            rayleigh_scatter as physics_rayleigh_scatter,
            reflect_specular as physics_reflect_specular,
            sample_bulk_collision as physics_sample_bulk_collision,
        )
    except ImportError as exc:  # pragma: no cover - optional installation
        raise RuntimeError(
            "fused portal boundary transport requires Torch and Triton"
        ) from exc

    # Triton 3.1 resolves JIT callees from the defining module's globals.
    globals().update(
        triton=triton,
        tl=tl,
        physics_fresnel_step=physics_fresnel_step,
        physics_rayleigh_scatter=physics_rayleigh_scatter,
        physics_reflect_specular=physics_reflect_specular,
        physics_sample_bulk_collision=physics_sample_bulk_collision,
    )

    @triton.jit
    def fused_portal_boundary_kernel(
        positions,
        directions,
        polarizations,
        times,
        histories,
        rng_counters,
        last_instances,
        last_triangles,
        detected_channels,
        step_counts,
        boundary_buffer,
        boundary_count,
        fallback_buffer,
        fallback_count,
        carry_buffer,
        carry_count,
        material_refractive_index,
        material_absorption_length,
        material_scattering_length,
        surface_detect,
        surface_absorb,
        surface_reflect_diffuse,
        surface_reflect_specular,
        global_photon_ids,
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
        seed,
        photon_id_base,
        max_steps,
        USE_GLOBAL_IDS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        program_start = tl.program_id(0) * BLOCK
        live_items = tl.load(boundary_count).to(tl.int32)
        # The scheduler owns the stronger counter invariant.  Clamping keeps a
        # corrupt counter memory-safe without introducing a host read.
        live_items = tl.maximum(0, tl.minimum(live_items, input_capacity))
        if program_start >= live_items:
            return

        lane = program_start + tl.arange(0, BLOCK)
        valid = lane < live_items
        photon_id = tl.load(
            boundary_buffer + lane, mask=valid, other=0
        ).to(tl.int64)
        base = photon_id * 3
        px = tl.load(positions + base, mask=valid, other=0.0)
        py = tl.load(positions + base + 1, mask=valid, other=0.0)
        pz = tl.load(positions + base + 2, mask=valid, other=0.0)
        dx = tl.load(directions + base, mask=valid, other=1.0)
        dy = tl.load(directions + base + 1, mask=valid, other=0.0)
        dz = tl.load(directions + base + 2, mask=valid, other=0.0)

        # Match portals.portal_partition_kernel exactly, including its box
        # tie priority and conservative lower-X/self-hit fallback rules.
        finite = (
            (px == px)
            & (py == py)
            & (pz == pz)
            & (dx == dx)
            & (dy == dy)
            & (dz == dz)
        )
        inside = (
            valid
            & finite
            & (px >= lower_x)
            & (px <= upper_x)
            & (py >= lower_y)
            & (py <= upper_y)
            & (pz >= lower_z)
            & (pz <= upper_z)
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
        distance_to_boundary = tl.where(
            choose_y, ty, tl.where(choose_z, tz, tx)
        )
        lower_x_tie = (dx < 0.0) & (tx <= distance_to_boundary)
        hit_triangle = tl.where(
            choose_y,
            tl.where(dy < 0.0, 2, 3),
            tl.where(choose_z, tl.where(dz < 0.0, 4, 5), 0),
        ).to(tl.int32)
        hit_instance = tl.where(
            choose_x_high, cathode_instance, active_instance
        ).to(tl.int32)
        previous_instance = tl.load(
            last_instances + photon_id, mask=valid, other=-1
        ).to(tl.int32)
        previous_triangle = tl.load(
            last_triangles + photon_id, mask=valid, other=-1
        ).to(tl.int32)
        suppress_previous = (previous_instance == hit_instance) & (
            previous_triangle == hit_triangle
        )
        direct = (
            inside
            & (distance_to_boundary > 1.0e-6)
            & (distance_to_boundary < huge)
            & (choose_y | choose_z | choose_x_high)
            & ~lower_x_tie
            & ~suppress_previous
        )

        fallback = valid & ~direct
        fallback_flag = fallback.to(tl.int32)
        fallback_local = tl.cumsum(fallback_flag, axis=0) - fallback_flag
        fallback_n = tl.sum(fallback_flag, axis=0)
        fallback_base = tl.atomic_add(fallback_count, fallback_n)
        tl.store(
            fallback_buffer + fallback_base + fallback_local,
            photon_id.to(tl.int32),
            mask=fallback,
        )

        # Materialize the certified hit as SSA values.  These are the exact
        # words which the standalone portal kernel otherwise writes to global
        # memory for the monolithic boundary consumer.
        y_low = choose_y & (dy < 0.0)
        z_low = choose_z & (dz < 0.0)
        nx = tl.where(choose_x_high, -1.0, 0.0)
        ny = tl.where(choose_y, tl.where(y_low, 1.0, -1.0), 0.0)
        nz = tl.where(choose_z, tl.where(z_low, 1.0, -1.0), 0.0)
        from_index = tl.zeros((BLOCK,), tl.int32) + lar_material
        to_index = tl.where(
            choose_x_high, cathode_inside_material, active_outside_material
        ).to(tl.int32)
        surface_index = tl.where(
            choose_x_high, cathode_surface, active_surface
        ).to(tl.int32)

        # Corrected production monolithic boundary physics.  Keep the
        # expression ordering synchronized with triton_backend's production
        # specialization: RNG consumes two Philox counter blocks for every
        # direct hit, independently of the selected process.
        qx = tl.load(polarizations + base, mask=direct, other=0.0)
        qy = tl.load(polarizations + base + 1, mask=direct, other=1.0)
        qz = tl.load(polarizations + base + 2, mask=direct, other=0.0)
        photon_time = tl.load(times + photon_id, mask=direct, other=0.0)
        history = tl.load(histories + photon_id, mask=direct, other=0).to(
            tl.int32
        )
        rng_counter = tl.load(
            rng_counters + photon_id, mask=direct, other=0
        ).to(tl.int64)
        step_count = tl.load(
            step_counts + photon_id, mask=direct, other=0
        ).to(tl.int32)

        safe_from = tl.maximum(from_index, 0)
        safe_to = tl.maximum(to_index, 0)
        refractive1 = tl.load(
            material_refractive_index + safe_from,
            mask=direct,
            other=1.0,
        )
        refractive2 = tl.load(
            material_refractive_index + safe_to,
            mask=direct,
            other=1.0,
        )
        absorption_length = tl.load(
            material_absorption_length + safe_from,
            mask=direct,
            other=float("inf"),
        )
        scattering_length = tl.load(
            material_scattering_length + safe_from,
            mask=direct,
            other=float("inf"),
        )

        if USE_GLOBAL_IDS:
            global_id = tl.load(
                global_photon_ids + photon_id, mask=direct, other=0
            ).to(tl.int64)
        else:
            global_id = photon_id + photon_id_base.to(tl.int64)
        random_offset = global_id * 4294967296 + rng_counter
        u0, u1, u2, u3 = tl.rand4x(seed, random_offset)
        u4, u5, u6, u7 = tl.rand4x(seed, random_offset + 1)
        rng_counter += direct.to(tl.int64) * 2

        collision_distance, process = physics_sample_bulk_collision(
            absorption_length, scattering_length, u0, u1
        )
        zero_absorption = direct & (absorption_length <= 0.0)
        collision_distance = tl.where(
            zero_absorption, 0.0, collision_distance
        )
        process = tl.where(zero_absorption, 1, process)
        bulk_collision = direct & (process != 0) & (
            collision_distance <= distance_to_boundary
        )
        bulk_absorb = bulk_collision & (process == 1)
        bulk_scatter = bulk_collision & (process == 2)
        at_boundary = direct & ~bulk_collision
        new_dx, new_dy, new_dz, new_qx, new_qy, new_qz = (
            physics_rayleigh_scatter(
                dx, dy, dz, qx, qy, qz, u2, u3, u4, u5
            )
        )

        advance = tl.where(
            bulk_absorb | bulk_scatter,
            collision_distance,
            tl.where(at_boundary, distance_to_boundary, 0.0),
        )
        px += advance * dx
        py += advance * dy
        pz += advance * dz
        photon_time += advance * refractive1 / 299.792458

        history = tl.where(bulk_absorb, history | (1 << 1), history)
        history = tl.where(bulk_scatter, history | (1 << 4), history)
        dx = tl.where(bulk_scatter, new_dx, dx)
        dy = tl.where(bulk_scatter, new_dy, dy)
        dz = tl.where(bulk_scatter, new_dz, dz)
        qx = tl.where(bulk_scatter, new_qx, qx)
        qy = tl.where(bulk_scatter, new_qy, qy)
        qz = tl.where(bulk_scatter, new_qz, qz)

        has_surface = at_boundary
        safe_surface = tl.maximum(surface_index, 0)
        probability_absorb = tl.load(
            surface_absorb + safe_surface, mask=has_surface, other=0.0
        )
        probability_detect = tl.load(
            surface_detect + safe_surface, mask=has_surface, other=0.0
        )
        probability_diffuse = tl.load(
            surface_reflect_diffuse + safe_surface,
            mask=has_surface,
            other=0.0,
        )
        probability_specular = tl.load(
            surface_reflect_specular + safe_surface,
            mask=has_surface,
            other=0.0,
        )
        cut_absorb = probability_absorb
        cut_detect = cut_absorb + probability_detect
        cut_diffuse = cut_detect + probability_diffuse
        cut_specular = cut_diffuse + probability_specular
        surface_absorbed = has_surface & (u2 < cut_absorb)
        surface_detected = has_surface & (u2 >= cut_absorb) & (
            u2 < cut_detect
        )
        diffuse = has_surface & (u2 >= cut_detect) & (u2 < cut_diffuse)
        specular = has_surface & (u2 >= cut_diffuse) & (u2 < cut_specular)
        dielectric = at_boundary & (~has_surface | (u2 >= cut_specular))

        # The production diffuse sampler is fixed-work and distribution exact.
        use_xy = tl.abs(nz) < 0.9
        b1x = tl.where(use_xy, -ny, 0.0)
        b1y = tl.where(use_xy, nx, -nz)
        b1z = tl.where(use_xy, 0.0, ny)
        inv_b1 = tl.rsqrt(
            tl.maximum(b1x * b1x + b1y * b1y + b1z * b1z, 1.0e-20)
        )
        b1x, b1y, b1z = b1x * inv_b1, b1y * inv_b1, b1z * inv_b1
        b2x = ny * b1z - nz * b1y
        b2y = nz * b1x - nx * b1z
        b2z = nx * b1y - ny * b1x
        radial = tl.sqrt(u3)
        axial = tl.sqrt(tl.maximum(0.0, 1.0 - u3))
        phi = 6.283185307179586 * u4
        tangent_cos = radial * tl.cos(phi)
        tangent_sin = radial * tl.sin(phi)
        diffuse_dx = axial * nx + tangent_cos * b1x + tangent_sin * b2x
        diffuse_dy = axial * ny + tangent_cos * b1y + tangent_sin * b2y
        diffuse_dz = axial * nz + tangent_cos * b1z + tangent_sin * b2z
        pol_use_xy = tl.abs(diffuse_dz) < 0.9
        c1x = tl.where(pol_use_xy, -diffuse_dy, 0.0)
        c1y = tl.where(pol_use_xy, diffuse_dx, -diffuse_dz)
        c1z = tl.where(pol_use_xy, 0.0, diffuse_dy)
        inv_c1 = tl.rsqrt(
            tl.maximum(c1x * c1x + c1y * c1y + c1z * c1z, 1.0e-20)
        )
        c1x, c1y, c1z = c1x * inv_c1, c1y * inv_c1, c1z * inv_c1
        c2x = diffuse_dy * c1z - diffuse_dz * c1y
        c2y = diffuse_dz * c1x - diffuse_dx * c1z
        c2z = diffuse_dx * c1y - diffuse_dy * c1x
        pol_phi = 6.283185307179586 * u5
        diffuse_qx = tl.cos(pol_phi) * c1x + tl.sin(pol_phi) * c2x
        diffuse_qy = tl.cos(pol_phi) * c1y + tl.sin(pol_phi) * c2y
        diffuse_qz = tl.cos(pol_phi) * c1z + tl.sin(pol_phi) * c2z

        history = tl.where(surface_absorbed, history | (1 << 3), history)
        history = tl.where(surface_detected, history | (1 << 2), history)

        reflected_dx, reflected_dy, reflected_dz = physics_reflect_specular(
            dx, dy, dz, nx, ny, nz
        )
        dx = tl.where(specular, reflected_dx, dx)
        dy = tl.where(specular, reflected_dy, dy)
        dz = tl.where(specular, reflected_dz, dz)
        history = tl.where(specular, history | (1 << 6), history)

        dx = tl.where(diffuse, diffuse_dx, dx)
        dy = tl.where(diffuse, diffuse_dy, dy)
        dz = tl.where(diffuse, diffuse_dz, dz)
        qx = tl.where(diffuse, diffuse_qx, qx)
        qy = tl.where(diffuse, diffuse_qy, qy)
        qz = tl.where(diffuse, diffuse_qz, qz)
        history = tl.where(diffuse, history | (1 << 5), history)

        (
            fresnel_dx,
            fresnel_dy,
            fresnel_dz,
            fresnel_qx,
            fresnel_qy,
            fresnel_qz,
            fresnel_reflected,
            _,
            _,
            _,
        ) = physics_fresnel_step(
            dx,
            dy,
            dz,
            qx,
            qy,
            qz,
            nx,
            ny,
            nz,
            refractive1,
            refractive2,
            u6,
            u7,
        )
        dx = tl.where(dielectric, fresnel_dx, dx)
        dy = tl.where(dielectric, fresnel_dy, dy)
        dz = tl.where(dielectric, fresnel_dz, dz)
        qx = tl.where(dielectric, fresnel_qx, qx)
        qy = tl.where(dielectric, fresnel_qy, qy)
        qz = tl.where(dielectric, fresnel_qz, qz)
        history = tl.where(
            dielectric & fresnel_reflected, history | (1 << 6), history
        )

        step_count += direct.to(tl.int32)
        active = direct & (
            bulk_scatter | diffuse | specular | dielectric
        ) & (step_count < max_steps)
        new_last_instance = tl.where(at_boundary, hit_instance, -1)
        new_last_triangle = tl.where(at_boundary, hit_triangle, -1)

        # Crucial ownership rule: every state mutation is direct-masked.
        tl.store(last_instances + photon_id, new_last_instance, mask=direct)
        tl.store(last_triangles + photon_id, new_last_triangle, mask=direct)
        # Direct certified faces carry channel -1, so the monolithic
        # consumer's detected-channel store is a no-op as well.
        tl.store(positions + base, px, mask=direct)
        tl.store(positions + base + 1, py, mask=direct)
        tl.store(positions + base + 2, pz, mask=direct)
        tl.store(directions + base, dx, mask=direct)
        tl.store(directions + base + 1, dy, mask=direct)
        tl.store(directions + base + 2, dz, mask=direct)
        tl.store(polarizations + base, qx, mask=direct)
        tl.store(polarizations + base + 1, qy, mask=direct)
        tl.store(polarizations + base + 2, qz, mask=direct)
        tl.store(times + photon_id, photon_time, mask=direct)
        tl.store(histories + photon_id, history, mask=direct)
        tl.store(rng_counters + photon_id, rng_counter, mask=direct)
        tl.store(step_counts + photon_id, step_count, mask=direct)

        survivor_flag = active.to(tl.int32)
        survivor_local = tl.cumsum(survivor_flag, axis=0) - survivor_flag
        survivor_n = tl.sum(survivor_flag, axis=0)
        survivor_base = tl.atomic_add(carry_count, survivor_n)
        tl.store(
            carry_buffer + survivor_base + survivor_local,
            photon_id.to(tl.int32),
            mask=active,
        )

    cached = (triton, fused_portal_boundary_kernel)
    _load_fused_portal_boundary_kernel._cached = cached
    return cached


def _require_cuda_vector(
    value: Any,
    *,
    torch: Any,
    device: Any,
    dtype: Any,
    shape: Sequence[int],
    name: str,
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_cuda
        or value.device != device
        or value.dtype != dtype
        or tuple(value.shape) != tuple(shape)
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be contiguous same-device CUDA {dtype} with shape "
            f"{tuple(shape)}"
        )


def step_fused_direct_portals(
    state: Sequence[Any],
    boundary_queue: Any,
    carry_queue: Any,
    descriptor: PortalDescriptor,
    scene_device: Mapping[str, Any],
    *,
    workspace: Optional[FusedPortalBoundaryWorkspace] = None,
    input_capacity: int,
    launch_capacity: int,
    seed: int,
    photon_id_base: int,
    max_steps: int,
    global_photon_ids: Optional[Any] = None,
    block_size: int = 128,
) -> Any:
    """Advance certified direct portals and return a device fallback queue.

    ``boundary_queue.count`` remains device-resident.  The scheduler must
    maintain ``0 <= count <= input_capacity <= launch_capacity``.  The
    fallback count is reset asynchronously before the launch; ``carry_queue``
    is append-only and is intentionally never reset here.
    """

    import torch
    from chroma.triton.transport import DeviceQueue

    if len(state) != 10:
        raise ValueError("state must contain the ten production photon tensors")
    if not isinstance(boundary_queue, DeviceQueue):
        raise TypeError("boundary_queue must be a DeviceQueue")
    if not isinstance(carry_queue, DeviceQueue):
        raise TypeError("carry_queue must be a DeviceQueue")
    positions = state[0]
    if not isinstance(positions, torch.Tensor) or not positions.is_cuda:
        raise ValueError("state positions must be a CUDA tensor")
    device = positions.device
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("state positions must have shape [N,3]")
    photon_count = int(positions.shape[0])
    state_specs = (
        (torch.float32, (photon_count, 3), "positions"),
        (torch.float32, (photon_count, 3), "directions"),
        (torch.float32, (photon_count, 3), "polarizations"),
        (torch.float32, (photon_count,), "times"),
        (torch.int32, (photon_count,), "histories"),
        (torch.int64, (photon_count,), "rng_counters"),
        (torch.int32, (photon_count,), "last_instances"),
        (torch.int32, (photon_count,), "last_triangles"),
        (torch.int32, (photon_count,), "detected_channels"),
        (torch.int32, (photon_count,), "step_counts"),
    )
    for value, (dtype, shape, name) in zip(state, state_specs):
        _require_cuda_vector(
            value,
            torch=torch,
            device=device,
            dtype=dtype,
            shape=shape,
            name=f"state {name}",
        )

    def validate_queue(queue: Any, name: str) -> None:
        _require_cuda_vector(
            queue.buffer,
            torch=torch,
            device=device,
            dtype=torch.int32,
            shape=(queue.capacity,),
            name=f"{name}.buffer",
        )
        _require_cuda_vector(
            queue.count,
            torch=torch,
            device=device,
            dtype=torch.int32,
            shape=(1,),
            name=f"{name}.count",
        )

    validate_queue(boundary_queue, "boundary_queue")
    validate_queue(carry_queue, "carry_queue")
    if workspace is None:
        workspace = FusedPortalBoundaryWorkspace.allocate(
            launch_capacity, device=device
        )
    if not isinstance(workspace, FusedPortalBoundaryWorkspace):
        raise TypeError("workspace must be a FusedPortalBoundaryWorkspace")
    fallback_queue = workspace.queue()
    validate_queue(fallback_queue, "workspace fallback")
    input_capacity, launch_capacity = _validate_capacities(
        input_capacity,
        launch_capacity,
        boundary_queue.capacity,
        fallback_queue.capacity,
        carry_queue.capacity,
    )

    queues = (boundary_queue, fallback_queue, carry_queue)
    names = ("boundary", "fallback", "carry")
    for left_index in range(len(queues)):
        for right_index in range(left_index + 1, len(queues)):
            left = queues[left_index]
            right = queues[right_index]
            if _tensor_storage_overlaps(left.buffer, right.buffer):
                raise ValueError(
                    f"{names[left_index]} and {names[right_index]} queue "
                    "buffers must not alias"
                )
            if _tensor_storage_overlaps(left.count, right.count):
                raise ValueError(
                    f"{names[left_index]} and {names[right_index]} queue "
                    "counts must not alias"
                )

    if not isinstance(descriptor, PortalDescriptor):
        raise TypeError("descriptor must be a PortalDescriptor")
    lower = np.asarray(descriptor.lower, dtype=np.float32)
    upper = np.asarray(descriptor.upper, dtype=np.float32)
    if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
        raise ValueError("portal descriptor has invalid bounds")
    if block_size not in (64, 128, 256, 512):
        raise ValueError("block_size must be 64, 128, 256, or 512")
    max_steps = _index(max_steps, "max_steps")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    tables = []
    material_size = None
    surface_size = None
    for key in _MATERIAL_TABLES + _SURFACE_TABLES:
        if key not in scene_device:
            raise KeyError(f"scene_device is missing {key!r}")
        value = scene_device[key]
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError(f"scene table {key} must be one-dimensional")
        _require_cuda_vector(
            value,
            torch=torch,
            device=device,
            dtype=torch.float32,
            shape=(int(value.numel()),),
            name=f"scene table {key}",
        )
        if key in _MATERIAL_TABLES:
            material_size = (
                int(value.numel())
                if material_size is None
                else min(material_size, int(value.numel()))
            )
        else:
            surface_size = (
                int(value.numel())
                if surface_size is None
                else min(surface_size, int(value.numel()))
            )
        tables.append(value)
    assert material_size is not None and surface_size is not None
    for value, name in (
        (descriptor.lar_material, "lar_material"),
        (descriptor.active_outside_material, "active_outside_material"),
        (descriptor.cathode_inside_material, "cathode_inside_material"),
    ):
        if int(value) < 0 or int(value) >= material_size:
            raise ValueError(f"portal descriptor {name} is outside material tables")
    for value, name in (
        (descriptor.active_surface, "active_surface"),
        (descriptor.cathode_surface, "cathode_surface"),
    ):
        if int(value) < 0 or int(value) >= surface_size:
            raise ValueError(f"portal descriptor {name} is outside surface tables")

    if global_photon_ids is None:
        global_ids = state[4]  # compile-time-dead pointer
    else:
        _require_cuda_vector(
            global_photon_ids,
            torch=torch,
            device=device,
            dtype=torch.int64,
            shape=(photon_count,),
            name="global_photon_ids",
        )
        global_ids = global_photon_ids

    # Reset only the private fallback.  Collision continuations may already
    # occupy the carry prefix, so its counter is append-only here.
    fallback_queue.reset()
    if launch_capacity == 0:
        return fallback_queue
    triton, kernel = _load_fused_portal_boundary_kernel()
    kernel[(triton.cdiv(launch_capacity, block_size),)](
        *state,
        boundary_queue.buffer,
        boundary_queue.count,
        fallback_queue.buffer,
        fallback_queue.count,
        carry_queue.buffer,
        carry_queue.count,
        *tables,
        global_ids,
        float(lower[0]),
        float(lower[1]),
        float(lower[2]),
        float(upper[0]),
        float(upper[1]),
        float(upper[2]),
        int(descriptor.lar_material),
        int(descriptor.active_outside_material),
        int(descriptor.cathode_inside_material),
        int(descriptor.active_surface),
        int(descriptor.cathode_surface),
        -int(descriptor.active_box_index) - 2,
        -int(descriptor.cathode_box_index) - 2,
        input_capacity,
        int(seed),
        int(photon_id_base),
        max_steps,
        USE_GLOBAL_IDS=global_photon_ids is not None,
        BLOCK=int(block_size),
        num_warps=min(4, max(1, block_size // 32)),
    )
    return fallback_queue


__all__ = [
    "FusedPortalBoundaryWorkspace",
    "step_fused_direct_portals",
]
