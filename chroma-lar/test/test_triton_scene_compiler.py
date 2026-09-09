"""CPU validation for the reflect3wires detector scene compiler."""

import numpy as np
import pytest

from chroma_lar.triton_scene import compile_reflect3wires_scene


@pytest.fixture(scope="module")
def scene():
    return compile_reflect3wires_scene()


def test_target_signature_and_global_ids(scene):
    assert scene.total_reference_solids == 165
    assert scene.total_reference_channels == 162
    assert scene.pmt.vertices.shape == (2564, 3)
    assert scene.pmt.triangles.shape == (5120, 3)
    assert scene.instances.count == 81
    np.testing.assert_array_equal(scene.instances.channel_id, np.arange(81))
    np.testing.assert_array_equal(scene.instances.solid_id, np.arange(1, 82))
    np.testing.assert_array_equal(
        scene.reachability.discarded_global_channel_id, np.arange(81, 162)
    )


def test_canonical_pmt_preserves_optical_triangle_metadata(scene):
    pmt = scene.pmt
    assert pmt.vertices.dtype == np.float32
    assert pmt.triangles.dtype == np.int32
    assert pmt.colors.dtype == np.uint32
    assert set(pmt.material_names) == {"glass", "vacuum", "liquid_argon"}
    assert set(pmt.surface_names) == {
        "glossy_surface",
        "perfect_pmt_photocathode",
    }

    # The transparent outer glass envelope is explicitly represented by -1,
    # while both inner-envelope surface types remain present.
    assert np.count_nonzero(pmt.surface_index == -1) > 0
    named_surfaces = {
        pmt.surface_names[index]
        for index in np.unique(pmt.surface_index[pmt.surface_index >= 0])
    }
    assert named_surfaces == set(pmt.surface_names)
    # Locks vertices, connectivity, all local triangle indices, and colors.
    assert pmt.sha256 == "91ea68146255c98deeed3e6d67ceccdab191705971cdfbe2a1a4a7503ee96000"


def test_optics_are_compiled_exactly_at_450nm(scene):
    tables = scene.tables
    material = {name: i for i, name in enumerate(tables.material_names)}
    np.testing.assert_array_equal(
        tables.material_refractive_index[
            [material["liquid_argon"], material["vacuum"], material["glass"], material["steel"]]
        ],
        np.asarray([1.3784, 1.0, 1.525, 1.07], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        tables.material_absorption_length[
            [material["liquid_argon"], material["vacuum"], material["glass"], material["steel"]]
        ],
        np.asarray([1e10, 1e6, 1500.0, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        tables.material_scattering_length[
            [material["liquid_argon"], material["vacuum"], material["glass"], material["steel"]]
        ],
        np.asarray([950.0, 1e6, 1e6, 0.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(tables.material_num_reemission_components, 0)

    surface = {name: i for i, name in enumerate(tables.surface_names)}
    pmt = surface["perfect_pmt_photocathode"]
    glossy = surface["glossy_surface"]
    steel = surface["polished_steel"]
    assert tables.surface_detect[pmt] == np.float32(1.0)
    assert tables.surface_reflect_diffuse[glossy] == np.float32(0.5)
    assert tables.surface_reflect_specular[glossy] == np.float32(0.5)
    assert tables.surface_absorb[steel] == np.float32(0.2)
    assert tables.surface_reflect_specular[steel] == np.float32(0.8)
    np.testing.assert_array_equal(tables.surface_default_probability_sum, 1.0)
    np.testing.assert_array_equal(tables.surface_default_transmit_probability, 0.0)
    np.testing.assert_array_equal(tables.surface_models, 0)
    np.testing.assert_array_equal(tables.surface_transmissive, 0)


def test_instance_transforms_are_exact_inverses_and_local_bounds(scene):
    instances = scene.instances
    pmt = scene.pmt
    selected = (0, 40, 80)
    for i in selected:
        rotation = instances.object_to_world_rotation[i]
        translation = instances.object_to_world_translation[i]
        inverse_rotation = instances.world_to_object_rotation[i]
        inverse_translation = instances.world_to_object_translation[i]

        local = pmt.vertices[[0, 100, -1]]
        world = local @ rotation.T + translation
        recovered = world @ inverse_rotation.T + inverse_translation
        np.testing.assert_allclose(recovered, local, rtol=0.0, atol=3e-4)

        all_world = pmt.vertices @ rotation.T + translation
        np.testing.assert_array_equal(instances.bounds_min[i], all_world.min(axis=0))
        np.testing.assert_array_equal(instances.bounds_max[i], all_world.max(axis=0))

    assert np.all(instances.object_to_world_translation[:, 0] < 0)
    np.testing.assert_allclose(
        instances.inward_normal, np.tile([1.0, 0.0, 0.0], (81, 1)), atol=1e-7
    )


def test_exact_analytic_boxes_and_reachable_faces(scene):
    boxes = scene.boxes
    assert boxes.kinds == ("cavity", "active", "cathode")
    np.testing.assert_array_equal(boxes.solid_id, [0, 163, 164])

    cathode = boxes.kinds.index("cathode")
    np.testing.assert_array_equal(boxes.bounds_min[cathode], [-3.0, -2160.0, -2160.0])
    np.testing.assert_array_equal(boxes.bounds_max[cathode], [3.0, 2160.0, 2160.0])
    np.testing.assert_array_equal(
        boxes.reachable_face_mask[cathode], [True, False, False, False, False, False]
    )

    cavity = boxes.kinds.index("cavity")
    active = boxes.kinds.index("active")
    np.testing.assert_array_equal(
        boxes.reachable_face_mask[cavity], np.ones(6, dtype=np.bool_)
    )
    assert boxes.collision_enabled[cavity]
    assert boxes.collision_enabled[active]
    assert scene.reachability.cathode_opaque
    assert scene.reachability.active_enclosure_opaque
    assert scene.reachability.cathode_probability_sum == pytest.approx(1.0)
    assert scene.reachability.active_probability_sum == pytest.approx(1.0)
    assert scene.reachability.source_component_bounds_max[0] == -3.0


def test_three_reachable_wire_lattices_have_precomputed_fp64_frames(scene):
    wires = scene.wires
    assert wires.count == 3
    np.testing.assert_array_equal(wires.source_wireplane_index, [1, 3, 5])
    np.testing.assert_array_equal(
        scene.reachability.discarded_wireplane_index, [0, 2, 4]
    )
    assert wires.origin.dtype == np.float64
    assert np.all(wires.origin[:, 0] < 0)
    np.testing.assert_allclose(wires.pitch, 3.0, rtol=0.0, atol=0.0)
    serialized_radius = np.float64(np.float32(0.075))
    np.testing.assert_allclose(
        wires.radius, serialized_radius, rtol=0.0, atol=0.0
    )
    # Regression: direct float64 promotion is observably different at a wire
    # cylinder's nominal-radius boundary.
    assert serialized_radius != np.float64(0.075)
    for field in ("pitch", "radius", "umin", "umax", "vmin", "vmax", "v0"):
        values = getattr(wires, field)
        np.testing.assert_array_equal(values, values.astype(np.float32).astype(np.float64))

    basis = np.stack((wires.u, wires.v, wires.n), axis=1)
    gram = basis @ np.transpose(basis, (0, 2, 1))
    np.testing.assert_allclose(
        gram, np.broadcast_to(np.eye(3), gram.shape), rtol=0.0, atol=2e-15
    )
    np.testing.assert_array_equal(
        wires.kmin, np.ceil((wires.vmin - wires.v0) / wires.pitch).astype(np.int32)
    )
    np.testing.assert_array_equal(
        wires.kmax, np.floor((wires.vmax - wires.v0) / wires.pitch).astype(np.int32)
    )


def test_flat_numpy_and_lazy_torch_conversions(scene):
    flat = scene.as_dict()
    assert flat["pmt_vertices"] is scene.pmt.vertices
    assert flat["instances_channel_id"] is scene.instances.channel_id
    assert flat["wires_u"].dtype == np.float64

    torch = pytest.importorskip("torch")
    tensors = scene.to_torch()
    assert tensors["pmt_vertices"].dtype == torch.float32
    assert tensors["pmt_triangles"].dtype == torch.int32
    assert tensors["wires_u"].dtype == torch.float64
    assert tensors["pmt_vertices"].data_ptr() == torch.from_numpy(scene.pmt.vertices).data_ptr()


def test_positive_half_is_exact_global_id_mirror():
    mirrored = compile_reflect3wires_scene(source_x_sign=1)
    np.testing.assert_array_equal(mirrored.instances.channel_id, np.arange(81, 162))
    np.testing.assert_array_equal(mirrored.wires.source_wireplane_index, [0, 2, 4])
    assert np.all(mirrored.instances.object_to_world_translation[:, 0] > 0)
    assert mirrored.reachability.source_component_bounds_min[0] == 3.0


def test_exact_compatibility_can_retain_all_wireplanes(scene):
    exact = compile_reflect3wires_scene(retain_all_wires=True)
    assert exact.wires.count == 6
    np.testing.assert_array_equal(
        exact.wires.source_wireplane_index, np.arange(6, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        exact.reachability.kept_wireplane_index,
        np.arange(6, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        exact.reachability.discarded_wireplane_index,
        np.empty(0, dtype=np.int32),
    )
    # Preserve the source descriptor's order and the exact serialized optical
    # identities; only the reachability mask is relaxed.
    default = scene
    np.testing.assert_array_equal(
        exact.wires.origin[default.wires.source_wireplane_index],
        default.wires.origin,
    )
    np.testing.assert_array_equal(
        exact.wires.surface_index[default.wires.source_wireplane_index],
        default.wires.surface_index,
    )
    np.testing.assert_array_equal(
        exact.wires.material_outer_index[
            default.wires.source_wireplane_index
        ],
        default.wires.material_outer_index,
    )
    np.testing.assert_array_equal(
        exact.wires.material_inner_index[
            default.wires.source_wireplane_index
        ],
        default.wires.material_inner_index,
    )


@pytest.mark.parametrize("bad_sign", [-2, 0, 2])
def test_invalid_source_half_is_rejected(bad_sign):
    with pytest.raises(ValueError, match="source_x_sign"):
        compile_reflect3wires_scene(source_x_sign=bad_sign)
