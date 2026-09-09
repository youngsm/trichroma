"""Rendering adapter for the same analytic wire queries used by transport."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from chroma.triton.viewer import BoundaryRenderHit


class AnalyticWireLayer:
    """Host-side wire description bound to an explicit compiler artifact.

    Mesh triangles remain responsible for PMTs, enclosure and cathode. This
    layer only intersects wires, so it cannot substitute a different box for
    a caller's mesh. Geometry validation checks every serialized wire frame
    and extent before the layer may handle its wireplane metadata.
    """

    features = ("wireplanes",)

    def __init__(self, scene):
        scene.wires.validate()
        self.wires = scene.wires
        self.scene = SimpleNamespace(
            wires=scene.wires,
            boxes=replace(
                scene.boxes, collision_enabled=np.zeros_like(scene.boxes.collision_enabled)
            ),
        )

    def validate_geometry(self, geometry):
        planes = tuple(getattr(geometry, "wireplanes", ()))
        if len(planes) != self.wires.count or not np.array_equal(
            self.wires.source_wireplane_index, np.arange(len(planes))
        ):
            raise ValueError("analytic renderer must retain every source wire plane in order")
        for index, plane in enumerate(planes):
            for source, compiled in (
                ("origin", "origin"),
                ("u", "raw_u"),
                ("v", "raw_v"),
                ("pitch", "pitch"),
                ("radius", "radius"),
                ("umin", "umin"),
                ("umax", "umax"),
                ("vmin", "vmin"),
                ("vmax", "vmax"),
                ("v0", "v0"),
            ):
                expected = np.asarray(plane[source], np.float32).astype(np.float64)
                if not np.array_equal(expected, getattr(self.wires, compiled)[index]):
                    raise ValueError(f"analytic renderer wire {index} has mismatched {source}")

    def prepare(self, device):
        return _DeviceAnalyticWires(self.scene, device)


class _DeviceAnalyticWires:
    def __init__(self, scene, device):
        import torch
        from .triton_scene.intersect import prepare_scene_triton

        self.scene = prepare_scene_triton(scene, device)
        self.device = self.scene.device
        self.colors = torch.as_tensor(scene.wires.color.copy().view(np.int32), device=device)
        self.workspace = None

    def reserve(self, count):
        from .triton_scene.intersect import allocate_split_intersection_workspace

        if self.workspace is None or self.workspace.capacity < count:
            self.workspace = allocate_split_intersection_workspace(count, self.device)

    def trace(self, origins, directions, tmax):
        from .triton_scene.intersect import intersect_scene_triton, intersect_scene_triton_split

        count = len(origins)
        self.reserve(count)
        if self.scene.wire_normals_are_x_aligned:
            result = intersect_scene_triton_split(
                self.scene,
                origins,
                directions,
                tmax=tmax,
                workspace=self.workspace,
                out=self.workspace.outputs(count),
            )
        else:
            result = intersect_scene_triton(
                self.scene, origins, directions, tmax=tmax, out=self.workspace.outputs(count)
            )
        return BoundaryRenderHit(result.index, result.distance, result.outward_normal, self.colors)
