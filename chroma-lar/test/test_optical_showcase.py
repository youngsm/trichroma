"""Check actual path identity, source stability and visible physics in the demos."""

import numpy as np
import pytest

from chroma.event import RAYLEIGH_SCATTER, SURFACE_REEMIT, SURFACE_DETECT
from chroma_lar.optical_showcase import (
    OpticalShowcase,
    PhotonPaths,
    _path_segments,
    build_playground_scene,
    wavelength_rgb,
)


def test_showcase_source_count_does_not_change_existing_photons():
    scene = build_playground_scene("prism")
    small, large = scene.photons(17, 91), scene.photons(2000, 91)
    for field in ("pos", "direction", "polarization", "wavelengths", "global_photon_ids"):
        np.testing.assert_array_equal(getattr(small, field), getattr(large, field)[:17])
    np.testing.assert_allclose(np.sum(small.direction * small.polarization, axis=1), 0)
    np.testing.assert_allclose(np.linalg.norm(small.polarization, axis=1), 1, atol=1e-6)


def test_fluorescence_flight_subtraction_has_no_competing_bulk_processes():
    from chroma.triton.spectral import SpectralScene

    scene = build_playground_scene("fluorescence")
    compiled = SpectralScene.compile(scene.geometry, wavelengths=scene.wavelengths)
    assert np.isinf(compiled.host.optics.materials.absorption_length).all()
    assert np.isinf(compiled.host.optics.materials.scattering_length).all()
    np.testing.assert_array_equal(compiled.host.optics.materials.refractive_index, 1)


@pytest.mark.parametrize("scene", ["prism", "fluorescence", "rayleigh"])
def test_showcase_cpu_prefix_endpoints_match_full_transport(scene):
    showcase = OpticalShowcase(backend="reference_bvh", device=None)
    event = showcase.run(scene, photons=192, paths=24, seed=901)
    assert event.paths.complete
    assert not event.result.step_limit_count
    assert np.all(np.diff(event.paths.times, axis=0) >= -1e-5)
    assert event.result.scene_fingerprint
    if scene == "fluorescence":
        emitted = (event.result.final_state["flags"] & SURFACE_REEMIT) != 0
        assert 0.7 < emitted.mean() < 0.97
        assert np.all(event.result.final_state["wavelengths"][emitted] > 400)
        assert np.all((event.fluorescence_delays >= 0) & (event.fluorescence_delays <= 200.001))
    elif scene == "rayleigh":
        assert np.any(event.result.final_state["flags"] & RAYLEIGH_SCATTER)
    else:
        state = event.result.final_state
        forward = state["pos"][:, 0] > 299
        angle = np.arctan2(state["direction"][forward, 1], state["direction"][forward, 0])
        assert np.corrcoef(state["wavelengths"][forward], angle)[0, 1] > 0.95


def test_flight_color_changes_at_reemission_vertex():
    paths = PhotonPaths(
        np.array([[[0, 0, 0]], [[1, 0, 0]], [[2, 1, 0]]], float),
        np.array([[305], [500], [500]], float),
        np.array([[0], [5], [6]], float),
        np.array([[0], [SURFACE_REEMIT], [SURFACE_REEMIT | SURFACE_DETECT]], np.uint32),
        np.array([71]),
        True,
    )
    segments, colors = _path_segments(paths)
    assert segments.shape == (2, 2, 2)
    np.testing.assert_array_equal(colors, wavelength_rgb([305, 500]))
    assert len(_path_segments(paths, time_limit=4)[0]) == 0
    assert len(_path_segments(paths, time_limit=5)[0]) == 1


@pytest.mark.parametrize("scene", ["prism", "fluorescence", "rayleigh"])
def test_showcase_gpu_prefix_and_cpu_outcomes(scene):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires local CUDA")
    showcase = OpticalShowcase()
    event = showcase.run(scene, photons=1024, paths=32, seed=901)
    from chroma.triton.spectral import SpectralSimulation

    fixture, gpu = showcase.experiment(scene)
    reference = SpectralSimulation(gpu.scene, backend="reference_bvh")
    result = reference.simulate(fixture.photons(1024, 901), seed=901, max_steps=256)
    for field in ("flags", "channels", "photon_ids"):
        np.testing.assert_array_equal(event.result.final_state[field], result.final_state[field])
    np.testing.assert_allclose(
        event.result.final_state["times"], result.final_state["times"], rtol=2e-5, atol=3e-4
    )
    np.testing.assert_allclose(
        event.result.final_state["wavelengths"], result.final_state["wavelengths"], atol=1e-3
    )
