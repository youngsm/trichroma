"""Detector-independent certificates and water Cherenkov fixture checks."""

from dataclasses import replace

import numpy as np
import pytest

from chroma.make import cylinder_along_z
from chroma.triton.examples.theia import (
    build_theia,
    cherenkov_photons,
    inward_rotation,
    audit_sensor_clearance,
)
from chroma.triton.primitives import Bounds, ConvexMeshVolume, Mesh, PrimitiveScene
from chroma.triton.regions import compile_regions


@pytest.mark.parametrize(
    "size,options",
    [
        (5000.0, {}),
        (5000.0, {"towards_zero": True, "offset": True}),
        ((10000.0, 8000.0, 12000.0), {}),
    ],
)
def test_theia_certificates_exclude_every_rotated_sensor(size, options):
    fixture = build_theia(size, coverage=0.05, diameter=304.8, **options)
    scene = fixture.primitives()
    result = compile_regions(scene)
    assert fixture.channel_count > 300
    assert not result.rejected_domains
    assert result.select([[0, 0, 0]]).bounds.contains([0, 0, 0])
    assert len({id(i.mesh) for i in scene.instances}) == 1
    np.testing.assert_allclose(np.linalg.det(fixture.rotations), 1.0, atol=2e-14)
    normals = fixture.rotations[:, :, 1]
    assert np.all(np.sum(normals * fixture.positions, axis=1) < 0)
    # Independently use all actual transformed vertices, including enclosure
    # triangles, rather than reusing the compiler's obstacle bounds.
    points = np.concatenate(
        [i.mesh.vertices @ i.rotation.T + i.translation for i in scene.instances]
        + [fixture.enclosure.vertices]
    )
    for region in result.regions:
        assert not region.bounds.contains(points).any()
        assert scene.volumes[0].encloses(region.bounds)


def test_convex_certificate_retains_facets_and_supports_rotation():
    original = cylinder_along_z(10.0, 20.0, points=8)
    rotation = inward_rotation([1.0, 2.0, 3.0])
    translation = np.array([950.0, -1200.0, 730.0])
    mesh = Mesh(original.vertices @ rotation.T + translation, original.triangles)
    domain = ConvexMeshVolume("arbitrary", mesh, 5, 0)
    result = compile_regions(PrimitiveScene((domain,)))
    assert result.regions
    rng = np.random.default_rng(190)
    for region in result.regions:
        world = rng.uniform(region.bounds.lower, region.bounds.upper, (10000, 3))
        local = (world - translation) @ rotation
        for angle in np.arange(8) * np.pi / 4 + np.pi / 8:
            assert np.all(
                local[:, 0] * np.cos(angle) + local[:, 1] * np.sin(angle)
                < 10 * np.cos(np.pi / 8) + 1e-6
            )
        assert np.all(np.abs(local[:, 2]) < 10.0)
    # A point inside a smooth cylinder can be outside the actual polygon.
    outside = (
        np.array([9.9 * np.cos(np.pi / 8), 9.9 * np.sin(np.pi / 8), 0.0]) @ rotation.T + translation
    )
    assert not domain.encloses(Bounds(outside, outside))
    inverted = replace(domain, mesh=Mesh(mesh.vertices, mesh.triangles[:, ::-1]))
    np.testing.assert_allclose(inverted.distances, domain.distances, atol=1e-10)


def test_main_theia_checks_every_possible_overlapping_sensor_pair():
    fixture = build_theia()
    assert fixture.channel_count == 49684
    assert fixture.packing["validated"]
    assert fixture.packing["candidate_pairs"] == 147804
    assert fixture.packing["minimum_separation_mm"] > 7.0
    assert fixture.packing["removed_edge_sensors"] == 582


def test_clearance_audit_rejects_coincident_and_intersecting_sensors():
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 1], [0, 1, -1]], np.float32)
    for shift in (0.0, 0.1):
        with pytest.raises(ValueError, match="overlap"):
            audit_sensor_clearance(
                vertices, np.array([[0.0, 0.0, 0.0], [shift, 0.0, 0.0]]), np.array([np.eye(3)] * 2)
            )


def test_convex_domains_reject_open_or_nonconvex_meshes():
    original = cylinder_along_z(10.0, 20.0, points=8)
    with pytest.raises(ValueError, match="closed"):
        ConvexMeshVolume("open", Mesh(original.vertices, original.triangles[:-1]), 0, 1)
    vertices = original.vertices.copy()
    index = np.argmax(vertices[:, 0])
    vertices[index, :2] *= 0.1
    with pytest.raises(ValueError, match="convex|faces"):
        ConvexMeshVolume("dented", Mesh(vertices, original.triangles), 0, 1)


def test_cherenkov_cone_polarization_dispersion_and_track_time():
    from chroma.demo.optics import water

    photons = cherenkov_photons(100000, seed=983, axis=[1, 2, 3], length=1000.0)
    track = np.array([1, 2, 3]) / np.sqrt(14)
    n = np.interp(photons.wavelengths, water.refractive_index[:, 0], water.refractive_index[:, 1])
    np.testing.assert_allclose(photons.dir @ track, 1 / n, rtol=2e-6)
    np.testing.assert_allclose(np.linalg.norm(photons.dir, axis=1), 1.0, atol=2e-7)
    np.testing.assert_allclose(np.linalg.norm(photons.pol, axis=1), 1.0, atol=2e-7)
    np.testing.assert_allclose(np.sum(photons.dir * photons.pol, axis=1), 0.0, atol=2e-7)
    np.testing.assert_allclose(photons.t, (photons.pos @ track + 500) / 299.792458, atol=3e-7)
    assert 300 <= photons.wavelengths.min() < photons.wavelengths.max() <= 650
    # Check the sampled spectral shape against independent numerical integration.
    grid = np.linspace(300.0, 650.0, 10001)
    n_grid = np.interp(grid, water.refractive_index[:, 0], water.refractive_index[:, 1])
    density = (1 - 1 / n_grid**2) / grid**2
    low = grid <= 400
    expected = np.trapezoid(density[low], grid[low]) / np.trapezoid(density, grid)
    assert abs(np.mean(photons.wavelengths <= 400) - expected) < 0.006
    with pytest.raises(ValueError, match="threshold"):
        cherenkov_photons(1, beta=0.5)


@pytest.mark.parametrize("axis", [[0, 1, 0], [0, -1, 0], [1, 0, 0], [0, 0, 1]])
def test_sensor_rotation_never_reflects_mesh(axis):
    rotation = inward_rotation(axis)
    np.testing.assert_allclose(rotation[:, 1], axis, atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
