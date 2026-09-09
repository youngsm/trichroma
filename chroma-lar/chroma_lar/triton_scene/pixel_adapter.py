"""Region declarations for the repository's box-shaped pixel TPC builder.

The builder owns the enclosure/active/cathode box topology and the surrounding
PMT medium. This adapter supplies those facts; the generic compiler chooses
all acceleration regions. Pixel/steel surface patches remain exact boundaries
in mesh transport, including the configured area-averaged pixel approximation.
"""

import numpy as np

from chroma.triton.primitives import Bounds, BoxVolume, Mesh, MeshInstance, PrimitiveScene


def pixel_primitives(geometry):
    """Return a primitive scene and its material-name registry before flattening."""
    if hasattr(geometry, "mesh"):
        raise ValueError("pixel primitive declarations require unflattened geometry")
    channels = set(map(int, geometry.channel_index_to_solid_id))
    materials = {}

    def material_index(material):
        if material is None:
            raise ValueError("pixel volumes require explicit material ownership")
        if material.name not in materials:
            materials[material.name] = len(materials)
        return materials[material.name]

    bulk = material_index(geometry.detector_material)
    volumes, instances, meshes = [], [], {}
    for index, (solid, rotation, translation) in enumerate(
        zip(geometry.solids, geometry.solid_rotations, geometry.solid_displacements)
    ):
        name = f"solid:{index}"
        if index in channels:
            key = id(solid.mesh)
            if key not in meshes:
                meshes[key] = Mesh(solid.mesh.vertices, solid.mesh.triangles)
            instances.append(MeshInstance(name, meshes[key], rotation, translation, bulk))
            continue
        # The known pixel builder emits axis-aligned closed boxes with planar
        # surface patches. Fail instead of guessing if that contract changes.
        if not np.array_equal(rotation, np.eye(3)):
            raise ValueError("pixel box declaration requires an axis-aligned builder volume")
        vertices = solid.mesh.vertices.astype(float) + translation
        bounds = Bounds(vertices.min(0), vertices.max(0))
        faces = vertices[solid.mesh.triangles]
        twice_area = np.linalg.norm(
            np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1
        )
        covered = np.zeros(len(faces), bool)
        for axis in range(3):
            expected_area = np.prod(np.delete(bounds.upper - bounds.lower, axis))
            for edge in (bounds.lower[axis], bounds.upper[axis]):
                selected = np.all(faces[:, :, axis] == edge, axis=1)
                if not np.isclose(
                    twice_area[selected].sum() / 2, expected_area, rtol=1e-6, atol=1e-6
                ):
                    raise ValueError(
                        "pixel builder box faces no longer cover their declared bounds"
                    )
                covered |= selected
        if not covered.all():
            raise ValueError("pixel box declaration contains non-box interfaces")
        inside, outside = solid.material1[0], solid.material2[0]
        if any(m is not inside for m in solid.material1) or any(
            m is not outside for m in solid.material2
        ):
            raise ValueError("pixel box volume has ambiguous material ownership")
        volumes.append(BoxVolume(name, bounds, material_index(inside), material_index(outside)))
    if len(volumes) != 3 or not instances:
        raise ValueError("pixel adapter requires enclosure, active box, cathode, and PMTs")
    return PrimitiveScene(tuple(volumes), instances=tuple(instances)), tuple(materials)
