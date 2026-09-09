"""Viewer fixtures must preserve the requested detector representations."""

import numpy as np

from chroma_lar.viewer_examples import build_viewer_example


def test_pixel_config_and_explicit_resolved_closeup():
    full = build_viewer_example("pixelTPC")
    assert full.metadata["sensor_instances"] == 164
    assert full.metadata["pixel_simplified"] is True
    assert full.metadata["resolved_pad_triangles"] is False
    assert not hasattr(full.geometry, "mesh")
    closeup = build_viewer_example("pixelPads")
    assert closeup.metadata["complete_detector"] is False
    assert closeup.metadata["triangles"] == 20_480
    colors = closeup.geometry.solids[0].color
    assert np.count_nonzero(colors == 0xFFFFD700) == 8192
    assert np.count_nonzero(colors == 0xFF2E8B57) == 12288


def test_reflect_view_retains_all_six_wire_planes():
    example = build_viewer_example("reflect3wires-mesh")
    geometry = example.geometry
    assert example.metadata["sensor_instances"] == 162
    assert example.metadata["wire_solids"] == 10_750
    assert example.metadata["wire_triangles"] == 1_376_000
    assert not len(getattr(geometry, "wireplanes", ()))
    first_wire = len(geometry.channel_index_to_solid_id) + 3
    offsets = np.asarray(geometry.solid_displacements[first_wire:])[:, 0]
    np.testing.assert_array_equal(np.unique(offsets), [-2160, -2157, -2154, 2154, 2157, 2160])
    assert all(len(s.mesh.triangles) == 128 for s in geometry.solids[first_wire:])


def test_analytic_view_requires_complete_matching_wire_geometry():
    import pytest

    example = build_viewer_example("reflect3wires")
    assert example.metadata["wire_planes"] == 6
    layer = example.boundary_layers[0]
    layer.validate_geometry(example.geometry)
    example.geometry.wireplanes[0]["pitch"] += 1
    with pytest.raises(ValueError, match="mismatched pitch"):
        layer.validate_geometry(example.geometry)


def test_gpu_wire_layer_matches_independent_cpu_intersections():
    import pytest

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    from chroma_lar.triton_scene.intersect import intersect_wires_numpy

    example = build_viewer_example("reflect3wires")
    layer = example.boundary_layers[0]
    wires = layer.wires
    origins = wires.origin + 1.25 * wires.n + 123.4 * wires.u
    # Alternate actual wire intersections and clear gaps on every plane.
    origins = np.concatenate((origins, origins + 1.5 * wires.v)).astype(np.float32)
    directions = np.concatenate((-wires.n, -wires.n)).astype(np.float32)
    expected = intersect_wires_numpy(wires, origins, directions, tmax=2.0)
    actual = layer.prepare("cuda").trace(
        torch.as_tensor(origins, device="cuda"),
        torch.as_tensor(directions, device="cuda"),
        torch.full((12,), 2.0, device="cuda"),
    )
    np.testing.assert_array_equal(actual.color_ids.cpu(), expected.index)
    np.testing.assert_allclose(actual.distances.cpu(), expected.distance, atol=1e-5, rtol=0)
    np.testing.assert_allclose(
        actual.world_normals.cpu(), expected.outward_normal, atol=1e-5, rtol=0
    )
    np.testing.assert_array_equal(expected.index[:6], np.arange(6))
    assert np.all(expected.index[6:] == -1)
