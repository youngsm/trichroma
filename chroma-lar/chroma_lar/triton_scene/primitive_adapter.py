"""Translate the existing detector artifact into general region primitives.

The region compiler imports no Chroma-LAr modules and uses no detector names,
coordinates, channel counts or axis assumptions. This adapter supplies the
exterior material of the PMT assemblies from their construction contract and
the exact padded bounds used by the PMT traversal accelerator.
"""

from chroma.triton.primitives import (
    Bounds,
    BoxVolume,
    BoundedObstacle,
    PrimitiveScene,
    WireArray,
)


def instance_traversal_view(instances, *, channel_ids=None):
    """Bridge a shared mesh instance group to the existing exact GPU traverser.

    No LAr metadata or lattice assumptions are used. Triangle optical metadata
    remains the caller's responsibility; this view is for geometry queries.
    """
    from types import SimpleNamespace
    import numpy as np

    instances = tuple(instances)
    if not instances or any(i.mesh is not instances[0].mesh for i in instances):
        raise ValueError("traversal view requires a nonempty group sharing one mesh")
    count = len(instances)
    channels = np.arange(count, dtype=np.int32) if channel_ids is None else np.asarray(channel_ids)
    if (
        channels.shape != (count,)
        or not np.issubdtype(channels.dtype, np.integer)
        or np.any(channels < 0)
        or np.any(channels >= 2**31)
        or np.any(channels[1:] <= channels[:-1])
    ):
        raise ValueError("channel IDs must be strictly increasing nonnegative int32 values")
    rotation = np.asarray([i.rotation for i in instances], dtype=np.float32)
    translation = np.asarray([i.translation for i in instances], dtype=np.float32)
    inverse = rotation.transpose(0, 2, 1).copy()
    # Use the actual rounded transforms and vertices that the GPU will read.
    vertices = instances[0].mesh.vertices.astype(np.float32)
    lower, upper = [], []
    for r, t in zip(rotation, translation):
        world = vertices.astype(float) @ r.astype(float).T + t
        lower.append(np.nextafter(world.min(0).astype(np.float32), np.float32(-np.inf)))
        upper.append(np.nextafter(world.max(0).astype(np.float32), np.float32(np.inf)))
    return SimpleNamespace(
        pmt=SimpleNamespace(
            vertices=vertices, triangles=instances[0].mesh.triangles.astype(np.int32)
        ),
        instances=SimpleNamespace(
            count=count,
            channel_id=channels.astype(np.int32),
            object_to_world_rotation=rotation,
            object_to_world_translation=translation,
            world_to_object_rotation=inverse,
            world_to_object_translation=-np.einsum("nij,nj->ni", inverse, translation),
            bounds_min=np.asarray(lower),
            bounds_max=np.asarray(upper),
        ),
    )


def detector_primitives(scene, pmt_bounds_min, pmt_bounds_max, *, mesh_exterior_material):
    boxes, wires = scene.boxes, scene.wires
    volumes, obstacles = [], []
    for i in range(boxes.count):
        if not boxes.collision_enabled[i]:
            continue
        bounds = Bounds(boxes.bounds_min[i], boxes.bounds_max[i])
        name = f"box:{int(boxes.solid_id[i])}"
        if boxes.reachable_face_mask[i].all():
            volumes.append(
                BoxVolume(
                    name,
                    bounds,
                    int(boxes.material_inside_index[i]),
                    int(boxes.material_outside_index[i]),
                )
            )
        else:
            # An open collection of faces does not establish a volume interior.
            obstacles.append(BoundedObstacle(name, bounds, None))
    if len(pmt_bounds_min) != scene.instances.count or len(pmt_bounds_max) != scene.instances.count:
        raise ValueError("all mesh-instance bounds are required for certification")
    for i, (lower, upper) in enumerate(zip(pmt_bounds_min, pmt_bounds_max)):
        obstacles.append(BoundedObstacle(f"mesh:{i}", Bounds(lower, upper), mesh_exterior_material))
    arrays = tuple(
        WireArray(
            name=f"wire:{int(wires.source_wireplane_index[i])}",
            origin=wires.origin[i],
            u=wires.u[i],
            v=wires.v[i],
            pitch=float(wires.pitch[i]),
            radius=float(wires.radius[i]),
            axial_limits=(float(wires.umin[i]), float(wires.umax[i])),
            offset=float(wires.v0[i]),
            first=int(wires.kmin[i]),
            last=int(wires.kmax[i]),
            exterior_material=int(wires.material_outer_index[i]),
        )
        for i in range(wires.count)
    )
    return PrimitiveScene(tuple(volumes), tuple(obstacles), arrays)
