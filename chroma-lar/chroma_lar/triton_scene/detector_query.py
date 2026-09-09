"""Geometry-only queries for the complete analytic/instanced detector.

This service owns acceleration structures and scratch memory, independently
of optical physics and transport scheduling. Returned tensors are workspace
views valid until the next query. Instances are sequential, single-stream
resources; concurrent simulations need separate instances.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np


class BoundaryHit(NamedTuple):
    """Nearest boundary, ordered for the spectral boundary kernel."""

    distance: Any
    normal: Any
    material_from: Any
    material_to: Any
    surface: Any
    instance: Any
    triangle: Any
    channel: Any


@dataclass(frozen=True)
class DetectorBoundaryResult:
    hit: BoundaryHit
    analytic: Any  # Includes exact primitive identity for boundary projection.


def certified_empty_lar_bounds(scene, pmt_bounds_min, pmt_bounds_max):
    """Prove the collision-only region excludes all wire and PMT obstacles.

    The compiler certifies the box/material topology; this check uses the
    accelerator's actual padded PMT bounds. Other regions use exact queries.
    """
    if scene.reachability.source_x_sign != -1:
        raise ValueError("the spectral source adapter requires source_x_sign=-1")
    active = scene.boxes.kinds.index("active")
    cathode = scene.boxes.kinds.index("cathode")
    wires = scene.wires
    selected = wires.origin[:, 0] < 0
    if not np.any(selected):
        raise ValueError("compiled scene has no negative-X wire planes")
    extent = np.max(wires.origin[selected, 0] + wires.radius[selected])
    lower = np.array(
        [
            np.nextafter(np.float32(extent), np.float32(np.inf)),
            *scene.boxes.bounds_min[active, 1:],
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            scene.boxes.bounds_min[cathode, 0],
            *scene.boxes.bounds_max[active, 1:],
        ],
        dtype=np.float64,
    )
    if np.any(lower >= upper):
        raise ValueError("compiled scene has no homogeneous LAr region")
    if np.any(np.all((pmt_bounds_max > lower) & (pmt_bounds_min < upper), axis=1)):
        raise ValueError("PMT bounds overlap the certified empty LAr region")
    if (
        not np.all(np.abs(wires.n[:, 0]) == 1.0)
        or np.any(wires.n[:, 1:] != 0.0)
        or np.any(
            (wires.origin[:, 0] + wires.radius > lower[0])
            & (wires.origin[:, 0] - wires.radius < upper[0])
        )
    ):
        raise ValueError("wire cylinders overlap the certified empty LAr region")
    return lower, upper


class DetectorBoundaryQuery:
    def __init__(
        self, scene, *, device="cuda", fused_pmt=False, region_mode="automatic", bulk_material=None
    ):
        import torch
        from .device_geometry import DeviceBoundaryRayWorkspace, DeviceBoundaryMergeWorkspace
        from .intersect import prepare_scene_triton, allocate_split_intersection_workspace
        from .instances import build_pmt_instance_accelerator

        self.device = torch.device(device)
        self.scene_device = scene.to_torch(self.device)
        self.analytic_scene = prepare_scene_triton(scene, self.device)
        self.pmt_accelerator = build_pmt_instance_accelerator(scene, device=self.device)
        self.fused_pmt = fused_pmt
        if fused_pmt and self.pmt_accelerator.grid_locator is None:
            raise ValueError("fused_pmt requires a compiler-certified regular PMT lattice")
        if region_mode not in ("automatic", "legacy", "disabled"):
            raise ValueError("region_mode must be automatic, legacy, or disabled")
        self.region_mode = region_mode
        self.regions = None
        self.bulk_region = None
        self.bulk_material = bulk_material
        if region_mode == "legacy":
            self.safe_lower, self.safe_upper = certified_empty_lar_bounds(
                scene, self.pmt_accelerator.host_bounds_min, self.pmt_accelerator.host_bounds_max
            )
        else:
            if region_mode == "automatic":
                from chroma.triton.regions import compile_regions
                from .primitive_adapter import detector_primitives

                self.primitives = detector_primitives(
                    scene,
                    self.pmt_accelerator.host_bounds_min,
                    self.pmt_accelerator.host_bounds_max,
                    mesh_exterior_material=bulk_material,
                )
                self.regions = compile_regions(self.primitives)
            self.select_bulk_region([])
        self.analytic_workspace = allocate_split_intersection_workspace(0, self.device)
        self.ray_workspace = DeviceBoundaryRayWorkspace.allocate(0, self.device)
        self.merge_workspace = DeviceBoundaryMergeWorkspace.allocate(0, self.device)
        self.pmt_workspace = self.pmt_accelerator.allocate_workspace(0, candidate_capacity=0)
        self._positive_infinity = torch.tensor(float("inf"), device=self.device)

    def select_bulk_region(self, positions, weights=None):
        """Choose a compiled certificate once per source batch, before launches."""
        if self.region_mode == "legacy":
            return
        self.bulk_region = (
            None
            if self.regions is None
            else self.regions.select(positions, material=self.bulk_material, weights=weights)
        )
        if self.bulk_region is None:
            # An empty box makes every photon fall through to exact geometry.
            self.safe_lower = self.safe_upper = np.zeros(3, dtype=np.float64)
        else:
            self.safe_lower = self.bulk_region.bounds.lower
            self.safe_upper = self.bulk_region.bounds.upper

    def begin_event(self):
        self.pmt_workspace.clear_sticky_overflow()

    def check_event(self):
        """Check once after transport synchronization, avoiding per-query stalls."""
        if self.pmt_workspace.sticky_overflowed():
            raise RuntimeError("PMT traversal overflow")

    def _reserve(self, count):
        from .device_geometry import DeviceBoundaryRayWorkspace, DeviceBoundaryMergeWorkspace
        from .intersect import allocate_split_intersection_workspace

        if count > self.ray_workspace.capacity:
            self.ray_workspace = DeviceBoundaryRayWorkspace.allocate(count, self.device)
            self.analytic_workspace = allocate_split_intersection_workspace(count, self.device)
        if count > self.merge_workspace.capacity:
            capacity = max(count, 2 * self.merge_workspace.capacity)
            self.merge_workspace = DeviceBoundaryMergeWorkspace.allocate(capacity, self.device)
        self.pmt_workspace.ensure_ray_capacity(count)

    def resolve(self, state, queue, count: int) -> DetectorBoundaryResult:
        """Resolve the live DeviceQueue prefix; count must equal queue.count.

        The scheduler already knows this count. It must consume the returned
        views before another resolve() call or any queue mutation.
        """
        import torch
        from .device_geometry import (
            gather_boundary_rays_device_count,
            merge_boundaries_device_count,
        )
        from .intersect import intersect_scene_triton_split
        from .instances import nearest_pmt_hit

        self._reserve(count)
        rays = gather_boundary_rays_device_count(
            state.pos,
            state.direction,
            state.last_instance,
            state.last_hit,
            queue,
            input_capacity=count,
            launch_capacity=count,
            out=self.ray_workspace,
        )
        analytic = intersect_scene_triton_split(
            self.analytic_scene,
            rays.origins,
            rays.directions,
            last_instance=rays.last_instances,
            last_triangle=rays.last_triangles,
            workspace=self.analytic_workspace,
            out=self.analytic_workspace.outputs(count),
        )
        # Include equal-distance PMT triangles to preserve the mesh-on-tie rule.
        torch.nextafter(analytic.distance, self._positive_infinity, out=rays.pmt_tmax)
        options = dict(
            tmax=rays.pmt_tmax,
            last_instance=rays.last_instances,
            last_triangle=rays.last_triangles,
            workspace=self.pmt_workspace,
            out=self.pmt_workspace.outputs(count),
            check_overflow=False,
        )
        if self.fused_pmt:
            from .fused_pmt import nearest_pmt_hit_fused_grid

            pmt = nearest_pmt_hit_fused_grid(
                self.pmt_accelerator,
                rays.origins,
                rays.directions,
                maximum_grid_candidates=81,
                compact_union=False,
                **options
            )
        else:
            pmt = nearest_pmt_hit(
                self.pmt_accelerator,
                rays.origins,
                rays.directions,
                ray_tile=None,
                use_grid=False,
                **options
            )
        hit = merge_boundaries_device_count(
            self.scene_device,
            analytic,
            pmt,
            rays.directions,
            queue.count,
            launch_capacity=count,
            out=self.merge_workspace,
        )
        return DetectorBoundaryResult(BoundaryHit(*hit), analytic)
