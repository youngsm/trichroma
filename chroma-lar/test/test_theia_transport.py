"""Water detector GPU checks against independent NumPy geometry/optics."""

import numpy as np
import pytest

from chroma.triton.examples.theia import build_theia, cherenkov_photons
from chroma.triton.spectral import SpectralScene, SpectralSimulation
from chroma_lar.triton_scene.primitive_adapter import instance_traversal_view
from chroma_lar.triton_scene.instances import (
    build_pmt_instance_accelerator,
    nearest_pmt_hit,
    nearest_pmt_hit_cpu,
    triton_instance_available,
)

pytestmark = pytest.mark.skipif(
    not triton_instance_available(require_cuda=True), reason="local CUDA GPU required"
)


@pytest.mark.parametrize("size", [5000.0, (10000.0, 8000.0, 12000.0)])
def test_theia_shared_mesh_tlas_matches_brute_force_instances(size):
    fixture = build_theia(size, coverage=0.05, diameter=304.8, towards_zero=True)
    view = instance_traversal_view(fixture.primitives().instances)
    accelerator = build_pmt_instance_accelerator(view)
    assert accelerator.grid_locator is None
    rng = np.random.default_rng(2821)
    origins = rng.uniform(-1000.0, 1000.0, (1024, 3)).astype(np.float32)
    directions = rng.normal(size=(1024, 3))
    # Include targeted curved-sensor hits as well as generic misses/grazing rays.
    targets = fixture.positions[rng.integers(fixture.channel_count, size=512)]
    directions[:512] = targets - origins[:512]
    directions = (directions / np.linalg.norm(directions, axis=1)[:, None]).astype(np.float32)
    expected = nearest_pmt_hit_cpu(accelerator, origins, directions)
    actual = nearest_pmt_hit(accelerator, origins, directions, use_tlas=True)
    assert np.count_nonzero(expected.instance_ids >= 0) > 400
    np.testing.assert_array_equal(actual.instance_ids.cpu(), expected.instance_ids)
    np.testing.assert_array_equal(actual.triangle_ids.cpu(), expected.triangle_ids)
    np.testing.assert_array_equal(actual.channel_ids.cpu(), expected.channel_ids)
    np.testing.assert_allclose(actual.distances.cpu(), expected.distances, rtol=5e-6, atol=0.003)
    np.testing.assert_allclose(
        actual.world_normals.cpu(), expected.world_normals, rtol=1e-5, atol=1e-6
    )
    assert not actual.overflow.any().item()


def test_water_cherenkov_transport_matches_independent_cpu_reference():
    fixture = build_theia(5000.0, coverage=0.05, diameter=304.8)
    scene = SpectralScene.compile(fixture.detector(), wavelengths=np.arange(300.0, 651.0, 5.0))
    photons = cherenkov_photons(1000, seed=51)
    reference = SpectralSimulation(scene, backend="reference").simulate(
        photons, seed=44, max_steps=256
    )
    cpu_bvh = SpectralSimulation(scene, backend="reference_bvh").simulate(
        photons, seed=44, max_steps=256
    )
    for name, expected in reference.final_state.items():
        np.testing.assert_array_equal(cpu_bvh.final_state[name], expected, err_msg=name)
    gpu = SpectralSimulation(scene).simulate(photons, seed=44, max_steps=256)
    assert len(gpu.hits) > 0
    assert gpu.step_limit_count == reference.step_limit_count == 0
    np.testing.assert_array_equal(gpu.final_state["flags"], reference.final_state["flags"])
    np.testing.assert_array_equal(gpu.hits.photon_ids, reference.hits.photon_ids)
    np.testing.assert_array_equal(gpu.hits.channels, reference.hits.channels)
    np.testing.assert_allclose(gpu.hits.times, reference.hits.times, rtol=1e-5, atol=0.003)
    np.testing.assert_array_equal(gpu.hits.wavelengths, reference.hits.wavelengths)
