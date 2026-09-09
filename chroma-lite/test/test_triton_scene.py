"""CPU-only contract tests for the detector-independent Triton scene IR."""

from types import SimpleNamespace

import numpy as np
import pytest

from chroma.detector import Detector
from chroma.geometry import Geometry, Material, Mesh, Solid, Surface
from chroma.triton.bvh import build_packed_bvh
from chroma.triton.scene import SceneCompileError, compile_host_scene


def _material(name, refractive_index):
    material = Material(name)
    material.set("refractive_index", refractive_index)
    material.set("absorption_length", 1000.0 + refractive_index)
    material.set("scattering_length", 2000.0 + refractive_index)
    return material


def _triangle(offset=0.0):
    return Mesh(
        np.array(
            [
                [offset + 0.0, 0.0, 0.0],
                [offset + 1.0, 0.0, 0.0],
                [offset + 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        np.array([[0, 1, 2]], dtype=np.int32),
    )


def _surface(name="sensor"):
    surface = Surface(name)
    surface.set("detect", 0.25)
    surface.set("absorb", 0.25)
    surface.set("reflect_diffuse", 0.20)
    surface.set("reflect_specular", 0.10)
    return surface


def _detector():
    outside = _material("outside", 1.2)
    inside = _material("inside", 1.5)
    detector = Detector(detector_material=outside)
    detector.add_solid(
        Solid(_triangle(), material1=inside, material2=outside, surface=None)
    )
    detector.add_pmt(
        Solid(
            _triangle(),
            material1=inside,
            material2=outside,
            surface=_surface(),
        ),
        displacement=np.array([5.0, 0.0, 0.0], dtype=np.float32),
        channel_type=73,
    )
    return detector


def test_compile_flattenable_detector_preserves_global_namespaces():
    detector = _detector()
    assert not hasattr(detector, "mesh")

    scene = compile_host_scene(detector)

    assert hasattr(detector, "mesh")
    assert scene.triangle_count == 2
    np.testing.assert_array_equal(scene.solid_id, [0, 1])
    np.testing.assert_array_equal(scene.triangle_channel_index, [-1, 0])
    np.testing.assert_array_equal(scene.material1_index, detector.material1_index)
    np.testing.assert_array_equal(scene.material2_index, detector.material2_index)
    np.testing.assert_array_equal(scene.surface_index, detector.surface_index)
    assert scene.features.is_detector
    assert scene.detector.channel_count == 1
    assert scene.detector.channel_index_to_channel_type[0] == 73
    assert scene.bvh.source == "triton_cpu"
    assert len(scene.fingerprint) == 64
    scene.validate()


def test_scene_is_an_immutable_snapshot_with_deterministic_fingerprint():
    detector = _detector()
    first = compile_host_scene(detector)
    second = compile_host_scene(detector)
    assert first.fingerprint == second.fingerprint
    np.testing.assert_array_equal(first.vertices, second.vertices)

    original = first.vertices.copy()
    detector.mesh.vertices[0, 0] += np.float32(0.125)
    np.testing.assert_array_equal(first.vertices, original)
    changed = compile_host_scene(detector)
    assert changed.fingerprint != first.fingerprint

    for value in (
        first.vertices,
        first.triangles,
        first.material1_index,
        first.material2_index,
        first.surface_index,
        first.solid_id,
        first.triangle_channel_index,
        first.bvh.nodes,
        first.optics.wavelength_grid.values,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        first.vertices[0, 0] = 10.0


def test_custom_uniform_grid_and_existing_bvh_are_snapshotted():
    geometry = Geometry()
    inner = _material("inner", 1.4)
    outer = _material("outer", 1.0)
    geometry.add_solid(
        Solid(_triangle(), material1=inner, material2=outer, surface=_surface())
    )
    geometry.flatten()
    packed = build_packed_bvh(geometry.mesh)
    geometry.bvh = packed
    wavelengths = np.array([100.0, 110.0, 120.0], dtype=np.float32)
    times = np.array([0.0, 0.5, 1.0], dtype=np.float32)

    scene = compile_host_scene(geometry, wavelengths=wavelengths, times=times)

    assert scene.features.uses_existing_bvh
    assert scene.bvh.source == "geometry"
    np.testing.assert_array_equal(scene.bvh.nodes, packed.nodes)
    np.testing.assert_array_equal(scene.optics.wavelength_grid.values, wavelengths)
    np.testing.assert_array_equal(scene.optics.time_grid.values, times)
    assert tuple(scene.optics.materials.names) == tuple(
        material.name for material in geometry.unique_materials
    )


def test_chroma_structured_uint4_bvh_is_normalized_without_reindexing():
    geometry = Geometry()
    inner = _material("inner", 1.4)
    outer = _material("outer", 1.0)
    geometry.add_solid(Solid(_triangle(), inner, outer, surface=None))
    geometry.flatten()
    packed = build_packed_bvh(geometry.mesh)
    uint4 = np.empty(
        packed.node_count,
        dtype=[(axis, np.uint32) for axis in ("x", "y", "z", "w")],
    )
    for column, axis in enumerate(("x", "y", "z", "w")):
        uint4[axis] = packed.nodes[:, column]
    geometry.bvh = SimpleNamespace(
        nodes=uint4,
        world_coords=SimpleNamespace(
            world_origin=packed.world_origin,
            world_scale=packed.world_scale,
        ),
        layer_offsets=packed.layer_offsets,
    )

    scene = compile_host_scene(geometry)

    assert scene.bvh.source == "geometry"
    np.testing.assert_array_equal(scene.bvh.nodes, packed.nodes)
    np.testing.assert_array_equal(scene.solid_id, geometry.solid_id)


def test_non_detector_has_explicit_no_channel_mapping():
    geometry = Geometry()
    inner = _material("inner", 1.4)
    outer = _material("outer", 1.0)
    geometry.add_solid(Solid(_triangle(), inner, outer, surface=None))
    scene = compile_host_scene(geometry)
    assert scene.detector is None
    assert not scene.features.is_detector
    np.testing.assert_array_equal(scene.triangle_channel_index, [-1])


def test_detector_without_pmts_has_canonical_empty_channel_arrays():
    outside = _material("outside", 1.0)
    inside = _material("inside", 1.4)
    detector = Detector(detector_material=outside)
    detector.add_solid(Solid(_triangle(), inside, outside, surface=None))

    scene = compile_host_scene(detector)

    assert scene.features.is_detector
    assert scene.channel_count == 0
    assert scene.detector.channel_index_to_position.shape == (0, 3)
    np.testing.assert_array_equal(scene.triangle_channel_index, [-1])
    scene.validate()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: setattr(value, "material1_index", np.array([], np.int32)), "one value per triangle"),
        (lambda value: value.mesh.triangles.__setitem__((0, 2), 99), "outside the vertex array"),
        (lambda value: setattr(value, "unique_materials", []), "material"),
        (
            lambda value: setattr(value, "material1_index", np.array([0.5])),
            "must contain integers",
        ),
    ],
)
def test_malformed_flattened_geometry_fails_closed(mutation, message):
    inner = _material("inner", 1.4)
    outer = _material("outer", 1.0)
    geometry = Geometry()
    geometry.add_solid(Solid(_triangle(), inner, outer, surface=None))
    geometry.flatten()
    mutation(geometry)
    with pytest.raises((SceneCompileError, ValueError), match=message):
        compile_host_scene(geometry)


def test_non_geometry_duck_type_fails_with_clear_missing_metadata():
    malformed = SimpleNamespace(mesh=_triangle())
    with pytest.raises(SceneCompileError, match="material1_index"):
        compile_host_scene(malformed)
