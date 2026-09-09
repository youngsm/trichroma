"""CPU-only tests for the exact Chroma global-BVH host artifact."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from chroma_lar.triton_scene.chroma_global_bvh import (
    ChromaGlobalBVHArtifactError,
    build_chroma_global_bvh_artifact,
    load_chroma_global_bvh_artifact,
    save_chroma_global_bvh_artifact,
)


def _packed_axis(lower, upper):
    return np.uint32(lower) | (np.uint32(upper) << np.uint32(16))


def _table(value):
    return np.asarray([[400.0, value - 0.1], [500.0, value + 0.1]])


def _material(name, refractive_index, absorption_length, scattering_length):
    return SimpleNamespace(
        name=name,
        refractive_index=_table(refractive_index),
        absorption_length=_table(absorption_length),
        scattering_length=_table(scattering_length),
        comp_reemission_prob=(),
    )


def _surface(name):
    return SimpleNamespace(
        name=name,
        model=0,
        transmissive=0,
        thickness=0.25,
        detect=_table(0.20),
        absorb=_table(0.30),
        reemit=_table(0.0),
        reflect_diffuse=_table(0.10),
        reflect_specular=_table(0.15),
        eta=_table(1.0),
        k=_table(0.0),
        reemission_cdf=_table(0.0),
    )


class _Mesh:
    def __init__(self):
        self.vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.triangles = np.asarray([[0, 1, 2], [0, 3, 1]], dtype=np.int32)

    def md5(self):
        import hashlib

        digest = hashlib.md5()
        digest.update(memoryview(self.vertices).cast("B"))
        digest.update(memoryview(self.triangles).cast("B"))
        return digest.hexdigest()


def _detector():
    # Root -> two leaves.  Bounds are deliberately ordinary words; topology
    # and exact preservation, rather than ray intersection, are under test.
    bound = _packed_axis(0, 11)
    nodes = np.asarray(
        [
            [bound, bound, bound, np.uint32((2 << 28) | 1)],
            [bound, bound, bound, np.uint32(0)],
            [bound, bound, bound, np.uint32(1)],
        ],
        dtype=np.uint32,
    )
    bvh = SimpleNamespace(
        nodes=nodes,
        world_coords=SimpleNamespace(
            world_origin=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            world_scale=np.float32(0.125),
        ),
        layer_offsets=np.asarray([0, 1], dtype=np.uint32),
    )
    argon = _material("argon", 1.23, 100.0, 200.0)
    glass = _material("glass", 1.50, 50.0, 75.0)
    photocathode = _surface("photocathode")
    return SimpleNamespace(
        mesh=_Mesh(),
        bvh=bvh,
        solid_id=np.asarray([0, 1], dtype=np.uint32),
        material1_index=np.asarray([0, 1], dtype=np.int64),
        material2_index=np.asarray([1, 0], dtype=np.int64),
        surface_index=np.asarray([-1, 0], dtype=np.int16),
        colors=np.asarray([0x10203040, 0x50607080], dtype=np.uint32),
        unique_materials=(argon, glass),
        unique_surfaces=(photocathode, None),
        wireplanes=(
            {
                "origin": np.asarray([2.0, 3.0, 4.0], dtype=np.float64),
                "u": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
                "v": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
                "pitch": 0.5,
                "radius": 0.025,
                "umin": -5.0,
                "umax": 5.0,
                "vmin": -6.0,
                "vmax": 6.0,
                "v0": 0.125,
                "surface": photocathode,
                "material_outer": argon,
                "material_inner": glass,
                "color": 0x00ABCDEF,
            },
        ),
        solid_id_to_channel_index=np.asarray([-1, 0], dtype=np.int32),
        channel_index_to_solid_id=np.asarray([1], dtype=np.int32),
    )


def test_build_owns_and_locks_exact_normalized_words():
    detector = _detector()
    artifact = build_chroma_global_bvh_artifact(detector)

    assert artifact.source == "geometry"
    assert artifact.node_count == 3
    assert artifact.reachable_node_count == 3
    assert artifact.stack_capacity == 1
    assert artifact.vertex_count == 4
    assert artifact.triangle_count == 2
    assert len(artifact.traversal_sha256) == 64
    assert len(artifact.optical_semantics_sha256) == 64
    assert len(artifact.sha256) == 64
    assert artifact.optical_wavelength_nm == np.float32(450.0)
    assert artifact.mesh_md5 == detector.mesh.md5()
    assert artifact.material_names == ("argon", "glass")
    assert artifact.surface_names == ("photocathode", None)
    np.testing.assert_array_equal(artifact.nodes, detector.bvh.nodes)
    np.testing.assert_array_equal(artifact.triangle_channel_index, [-1, 0])
    np.testing.assert_array_equal(
        artifact.material_refractive_index, np.asarray([1.23, 1.5], np.float32)
    )
    np.testing.assert_array_equal(artifact.wire_source_index, [0])
    np.testing.assert_array_equal(artifact.wire_material_outer_index, [0])
    np.testing.assert_array_equal(artifact.wire_material_inner_index, [1])
    np.testing.assert_array_equal(artifact.wire_surface_index, [0])

    expected_dtypes = {
        "nodes": np.uint32,
        "world_origin": np.float32,
        "layer_offsets": np.int64,
        "vertices": np.float32,
        "triangles": np.int32,
        "solid_id": np.int32,
        "material1_index": np.int32,
        "material2_index": np.int32,
        "surface_index": np.int32,
        "colors": np.uint32,
        "triangle_channel_index": np.int32,
        "material_refractive_index": np.float32,
        "material_num_reemission_components": np.int32,
        "surface_detect": np.float32,
        "surface_models": np.int32,
        "wire_origin": np.float32,
        "wire_radius": np.float32,
        "wire_surface_index": np.int32,
        "wire_color": np.uint32,
    }
    for name, dtype in expected_dtypes.items():
        value = getattr(artifact, name)
        assert value.dtype == np.dtype(dtype)
        assert value.flags.c_contiguous
        assert value.flags.owndata
        assert not value.flags.writeable

    # Later mutation of the legacy object cannot alter the snapshot.
    detector.mesh.vertices[0, 0] = np.float32(99.0)
    detector.bvh.nodes[0, 0] = np.uint32(99)
    assert artifact.vertices[0, 0] == np.float32(0.0)
    assert artifact.nodes[0, 0] == _packed_axis(0, 11)


def test_empty_wire_table_has_a_stable_optical_digest():
    detector = _detector()
    detector.wireplanes = ()
    first = build_chroma_global_bvh_artifact(detector)
    second = build_chroma_global_bvh_artifact(detector)

    assert first.wire_source_index.size == 0
    assert first.wire_origin.shape == (0, 3)
    assert first.optical_semantics_sha256 == second.optical_semantics_sha256


@pytest.mark.parametrize("compressed", [False, True])
def test_pickle_free_archive_round_trip_and_expected_fingerprints(tmp_path, compressed):
    artifact = build_chroma_global_bvh_artifact(_detector())
    path = tmp_path / ("reference-compressed.bin" if compressed else "reference.bin")
    returned = save_chroma_global_bvh_artifact(
        artifact, path, compressed=compressed
    )
    assert returned == path
    assert path.is_file()

    loaded = load_chroma_global_bvh_artifact(
        path,
        expected_mesh_md5=artifact.mesh_md5,
        expected_traversal_sha256=artifact.traversal_sha256,
        expected_optical_semantics_sha256=artifact.optical_semantics_sha256,
        expected_sha256=artifact.sha256,
    )
    assert loaded.traversal_sha256 == artifact.traversal_sha256
    assert loaded.optical_semantics_sha256 == artifact.optical_semantics_sha256
    assert loaded.sha256 == artifact.sha256
    assert loaded.material_names == artifact.material_names
    assert loaded.surface_names == artifact.surface_names
    assert loaded.stack_capacity == artifact.stack_capacity
    for name in (
        "nodes",
        "vertices",
        "triangles",
        "solid_id",
        "material1_index",
        "material2_index",
        "surface_index",
        "colors",
        "material_refractive_index",
        "surface_default_transmit_probability",
        "wire_origin",
        "wire_surface_index",
    ):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(artifact, name))
        assert not getattr(loaded, name).flags.writeable

    with pytest.raises(ChromaGlobalBVHArtifactError, match="unexpected flattened"):
        load_chroma_global_bvh_artifact(path, expected_mesh_md5="0" * 32)
    with pytest.raises(ChromaGlobalBVHArtifactError, match="unexpected artifact"):
        load_chroma_global_bvh_artifact(path, expected_sha256="0" * 64)
    with pytest.raises(ChromaGlobalBVHArtifactError, match="unexpected traversal"):
        load_chroma_global_bvh_artifact(
            path, expected_traversal_sha256="0" * 64
        )
    with pytest.raises(
        ChromaGlobalBVHArtifactError, match="unexpected optical-semantics"
    ):
        load_chroma_global_bvh_artifact(
            path, expected_optical_semantics_sha256="0" * 64
        )


def test_traversal_fingerprint_excludes_optical_indices_and_path_provenance():
    first_detector = _detector()
    first = build_chroma_global_bvh_artifact(first_detector)
    second_detector = _detector()
    second_detector.material1_index[:] = [1, 0]
    second_detector.material2_index[:] = [0, 1]
    second_detector.surface_index[:] = [0, -1]
    second = build_chroma_global_bvh_artifact(
        second_detector, config_name="/equivalent/config/by/absolute/path.py"
    )

    assert first.traversal_sha256 == second.traversal_sha256
    assert first.optical_semantics_sha256 != second.optical_semantics_sha256
    assert first.sha256 != second.sha256


def test_optical_fingerprint_is_invariant_to_consistent_raw_id_permutation():
    first = build_chroma_global_bvh_artifact(_detector())
    permuted_detector = _detector()
    permuted_detector.unique_materials = tuple(
        reversed(permuted_detector.unique_materials)
    )
    permuted_detector.material1_index[:] = [1, 0]
    permuted_detector.material2_index[:] = [0, 1]
    permuted_detector.unique_surfaces = tuple(
        reversed(permuted_detector.unique_surfaces)
    )
    permuted_detector.surface_index[:] = [-1, 1]
    permuted = build_chroma_global_bvh_artifact(permuted_detector)

    assert first.traversal_sha256 == permuted.traversal_sha256
    assert first.optical_semantics_sha256 == permuted.optical_semantics_sha256
    assert first.sha256 != permuted.sha256


@pytest.mark.parametrize("mutation", ["table", "assignment", "wire"])
def test_optical_fingerprint_covers_tables_assignments_and_wires(mutation):
    first = build_chroma_global_bvh_artifact(_detector())
    detector = _detector()
    if mutation == "table":
        detector.unique_materials[0].refractive_index[:, 1] += 0.25
    elif mutation == "assignment":
        detector.material1_index[:] = [1, 0]
    else:
        detector.wireplanes[0]["radius"] = 0.05
    changed = build_chroma_global_bvh_artifact(detector)

    assert first.traversal_sha256 == changed.traversal_sha256
    assert first.optical_semantics_sha256 != changed.optical_semantics_sha256


def test_optical_name_remap_is_exact_immutable_and_fails_closed():
    artifact = build_chroma_global_bvh_artifact(_detector())
    material1, material2, surfaces = artifact.remap_optical_indices(
        ("glass", "argon", "steel"),
        ("mirror", "photocathode"),
    )
    np.testing.assert_array_equal(material1, [1, 0])
    np.testing.assert_array_equal(material2, [0, 1])
    np.testing.assert_array_equal(surfaces, [-1, 1])
    for value in (material1, material2, surfaces):
        assert value.dtype == np.int32
        assert value.flags.owndata
        assert not value.flags.writeable

    with pytest.raises(ChromaGlobalBVHArtifactError, match="no material"):
        artifact.remap_optical_indices(("argon",), ("photocathode",))
    with pytest.raises(ChromaGlobalBVHArtifactError, match="ambiguous"):
        artifact.remap_optical_indices(
            ("argon", "glass"), ("photocathode", "photocathode")
        )


def test_archive_payload_tampering_is_detected(tmp_path):
    artifact = build_chroma_global_bvh_artifact(_detector())
    original = tmp_path / "original.npz"
    tampered = tmp_path / "tampered.npz"
    save_chroma_global_bvh_artifact(artifact, original)

    with np.load(original, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["vertices"][0, 0] = np.float32(0.25)
    with tampered.open("wb") as output:
        np.savez(output, **payload)

    with pytest.raises(
        ChromaGlobalBVHArtifactError, match="mesh MD5|SHA-256"
    ):
        load_chroma_global_bvh_artifact(tampered)


def test_missing_or_nonreference_bvh_fails_closed():
    detector = _detector()
    detector.bvh = None
    with pytest.raises(ChromaGlobalBVHArtifactError, match="no existing BVH"):
        build_chroma_global_bvh_artifact(detector)

    detector = _detector()
    detector.bvh.source = "triton_cpu"
    with pytest.raises(ChromaGlobalBVHArtifactError, match="source='geometry'"):
        build_chroma_global_bvh_artifact(detector)


def test_invalid_child_and_leaf_topology_fails_closed():
    detector = _detector()
    detector.bvh.nodes[0, 3] = np.uint32((2 << 28) | 2)
    with pytest.raises(ChromaGlobalBVHArtifactError, match="child range"):
        build_chroma_global_bvh_artifact(detector)

    detector = _detector()
    detector.bvh.nodes[2, 3] = np.uint32(2)
    with pytest.raises(ChromaGlobalBVHArtifactError, match="leaf refers outside"):
        build_chroma_global_bvh_artifact(detector)

    detector = _detector()
    detector.bvh.nodes[2, 3] = np.uint32(0)
    with pytest.raises(ChromaGlobalBVHArtifactError, match="duplicate reachable"):
        build_chroma_global_bvh_artifact(detector)


def test_in_memory_metadata_or_fingerprint_tampering_is_detected():
    artifact = build_chroma_global_bvh_artifact(_detector())
    with pytest.raises(ChromaGlobalBVHArtifactError, match="stack_capacity"):
        replace(artifact, stack_capacity=2).validate()
    with pytest.raises(ChromaGlobalBVHArtifactError, match="SHA-256"):
        replace(artifact, sha256="0" * 64).validate()
    with pytest.raises(ChromaGlobalBVHArtifactError, match="traversal SHA-256"):
        replace(artifact, traversal_sha256="0" * 64).validate()


def test_archive_rejects_wrong_dtype_before_normalization(tmp_path):
    artifact = build_chroma_global_bvh_artifact(_detector())
    original = tmp_path / "original.npz"
    malformed = tmp_path / "malformed.npz"
    save_chroma_global_bvh_artifact(artifact, original)
    with np.load(original, allow_pickle=False) as archive:
        payload = {name: np.array(archive[name], copy=True) for name in archive.files}
    payload["triangles"] = payload["triangles"].astype(np.int64)
    with malformed.open("wb") as output:
        np.savez(output, **payload)
    with pytest.raises(ChromaGlobalBVHArtifactError, match="triangles must have dtype"):
        load_chroma_global_bvh_artifact(malformed)
