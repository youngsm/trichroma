"""Analytic optical checks and CPU/device distribution comparisons."""
import numpy as np
import pytest

from chroma.detector import Detector
from chroma.geometry import Material, Mesh, Solid, Surface
from chroma.make import box
from chroma.event import Photons, SURFACE_REEMIT, REFLECT_SPECULAR
from chroma.triton.optical_response import TabulatedCDF
from chroma.triton.spectral import SpectralScene, SpectralSimulation, group_velocity

GRID = np.arange(120., 501., 1.)


def material(name="medium", n=1.):
    m = Material(name)
    m.set("refractive_index", n)
    m.set("absorption_length", 1.e30)
    m.set("scattering_length", 1.e30)
    return m


def source(n=200, wavelength=128, z=0):
    pos = np.tile([.23, -.17, z], (n, 1))
    return Photons(pos, np.tile([0, 0, 1], (n, 1)), np.tile([1, 0, 0], (n, 1)),
                    np.broadcast_to(wavelength, (n,)), t=np.full(n, 7.))


def detector(m=None):
    m = material() if m is None else m
    d = Detector(m)
    sensor = Surface("sensor")
    sensor.set("detect", 1.)
    d.add_pmt(Solid(box(200, 200, 200), m, m, surface=sensor), displacement=np.zeros(3))
    return d


def plane(z=0):
    return Mesh([[-80, -80, z], [80, -80, z], [80, 80, z], [-80, 80, z]], [[0, 1, 2], [0, 2, 3]])


def coated_detector():
    m = material()
    m.set("group_velocity", np.where(GRID < 200, 100., 200.), GRID)
    d = detector(m)
    coat = Surface("WLS coating", model=2)
    coat.set("absorb", np.where(GRID < 200, 1., 0.), GRID)
    coat.set("reemit", 1.)
    coat.set("reemission_cdf", np.clip((GRID-410)/40, 0, 1), GRID)
    coat.reemission_time_cdf = TabulatedCDF([35, 35], [0, 1])
    coat.reemission_to_material1 = .75
    d.add_solid(Solid(plane(), m, m, surface=coat))
    return d


def run(d, p, backend="reference", **kwargs):
    return SpectralSimulation(d, wavelengths=GRID, backend=backend, tile_size=4096).simulate(p, **kwargs)


def test_group_velocity_uses_dispersion_and_explicit_override():
    n = 1.7-.001*GRID
    np.testing.assert_allclose(group_velocity(GRID, n), 299.792458/1.7, rtol=1e-6)
    m = material()
    m.set("group_velocity", 100.)
    result = run(detector(m), source(20))
    np.testing.assert_allclose(result.hits.times, 8., atol=2e-6)
    assert len(result.hits) == 20
    assert result.step_limit_count == 0


def test_wavelength_dependent_absorption_survival():
    m = material()
    m.set("absorption_length", np.where(GRID < 300, 100., 1000.), GRID)
    n = 6000
    result = run(detector(m), source(n, np.r_[np.full(n//2, 128), np.full(n//2, 450)]), seed=12)
    low = np.count_nonzero(result.hits.wavelengths < 300) / (n//2)
    high = np.count_nonzero(result.hits.wavelengths > 300) / (n//2)
    assert abs(low - np.exp(-1)) < .025
    assert abs(high - np.exp(-.1)) < .02


def test_wls_spectrum_delay_escape_side_and_transport_continues():
    result = run(coated_detector(), source(2000, z=-10), seed=36)
    state = result.final_state
    assert len(result.hits) == 2000
    assert np.all(state["flags"] & SURFACE_REEMIT)
    assert abs(result.hits.wavelengths.mean()-430) < .7
    assert np.min(result.hits.wavelengths) >= 410
    assert np.max(result.hits.wavelengths) <= 450
    after_wls_distance = np.linalg.norm(state["pos"] - [.23, -.17, 0], axis=1)
    np.testing.assert_allclose(state["times"], 7.+.1+35.+after_wls_distance/200., atol=1e-4)
    assert abs(np.mean(state["direction"][:, 2] < 0) - .75) < .035
    np.testing.assert_allclose(np.sum(state["direction"]*state["polarization"], axis=1), 0, atol=1e-5)


def test_dielectric_total_internal_reflection_and_step_limit():
    inside, outside = material("inside", 1.5), material("outside", 1.)
    d = Detector(inside)
    d.add_solid(Solid(box(200, 200, 200), inside, outside))
    p = source(1)
    p.dir[:] = [np.sqrt(.5), 0, np.sqrt(.5)]
    p.pol[:] = [0, 1, 0]
    result = run(d, p, max_steps=1)
    assert result.final_state["flags"][0] & REFLECT_SPECULAR
    assert result.step_limit_count == 1
    assert len(result.hits) == 0


def test_arbitrary_input_identity_tiling_and_out_of_band_rejection():
    p = source(63, np.linspace(121, 499, 63))
    p.t[:] = np.arange(63)
    p.evidx[:] = np.arange(63)%3
    d = detector()
    first = SpectralSimulation(d, wavelengths=GRID, backend="reference", tile_size=10).simulate(p, photon_id_base=501)
    second = SpectralSimulation(d, wavelengths=GRID, backend="reference", tile_size=100).simulate(p, photon_id_base=501)
    np.testing.assert_array_equal(first.hits.times, second.hits.times)
    np.testing.assert_array_equal(first.hits.photon_ids, np.arange(501, 564))
    np.testing.assert_array_equal(first.hits.event_indices, p.evidx)
    with pytest.raises(ValueError, match="outside"):
        run(detector(), source(2, 100))


@pytest.mark.parametrize("coated", [False, True])
def test_triton_reference_parity(coated):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    d = coated_detector() if coated else detector()
    p = source(300, z=-10 if coated else 0)
    cpu = run(d, p, seed=59)
    gpu = run(d, p, backend="triton", seed=59)
    np.testing.assert_array_equal(gpu.final_state["flags"], cpu.final_state["flags"])
    np.testing.assert_allclose(gpu.final_state["times"], cpu.final_state["times"], atol=1e-4)
    np.testing.assert_allclose(gpu.final_state["wavelengths"], cpu.final_state["wavelengths"], atol=1e-4)
    np.testing.assert_array_equal(gpu.hits.photon_ids, cpu.hits.photon_ids)


@pytest.mark.parametrize("kind", ["absorption", "rayleigh", "reflection", "dielectric", "failed_wls"])
def test_gpu_boundary_and_bulk_ensemble_parity(kind):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    m = material()
    d = detector(m)
    p = source(1200, np.linspace(125, 490, 1200))
    if kind == "absorption":
        m.set("absorption_length", np.linspace(50, 500, len(GRID)), GRID)
    elif kind == "rayleigh":
        m.set("scattering_length", 75.)
    elif kind in ("reflection", "failed_wls"):
        surface = Surface("extra", model=2 if kind == "failed_wls" else 0)
        if kind == "reflection":
            surface.set("reflect_diffuse", .4)
            surface.set("reflect_specular", .4)
            surface.set("absorb", .2)
        else:
            surface.set("absorb", 1.)
            surface.set("reemit", 0.)
        d.add_solid(Solid(plane(10), m, m, surface=surface))
    else:
        m2 = material("glass", 1.5)
        d.add_solid(Solid(box(80,80,80), m2, m))
        p.pos[:, 2] = -50
        p.dir[:] = [0, .5, np.sqrt(.75)]
    cpu = run(d, p, seed=91, max_steps=100)
    gpu = run(d, p, backend="triton", seed=91, max_steps=100)
    # Different floating-point traversal rounding can change rare boundary
    # histories after many scatterings; compare ensemble quantities as well.
    assert abs(len(cpu.hits)-len(gpu.hits)) <= 3
    assert cpu.step_limit_count == gpu.step_limit_count == 0
    if len(cpu.hits):
        np.testing.assert_allclose(np.quantile(gpu.hits.times, [.1,.5,.9]),
                                   np.quantile(cpu.hits.times, [.1,.5,.9]), atol=.02)
    assert np.isfinite(gpu.final_state["times"]).all()
    np.testing.assert_allclose(np.linalg.norm(gpu.final_state["direction"], axis=1), 1, atol=1e-4)
    np.testing.assert_allclose(np.sum(gpu.final_state["direction"]*gpu.final_state["polarization"], axis=1), 0, atol=1e-4)


def test_gpu_tiling_with_large_photon_ids():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    p = source(157, z=-10)
    d = coated_detector()
    a = SpectralSimulation(d, wavelengths=GRID, backend="triton", tile_size=157).simulate(p, photon_id_base=2**40)
    b = SpectralSimulation(d, wavelengths=GRID, backend="triton", tile_size=61).simulate(p, photon_id_base=2**40)
    np.testing.assert_array_equal(a.hits.photon_ids, b.hits.photon_ids)
    np.testing.assert_array_equal(a.hits.times, b.hits.times)
    np.testing.assert_array_equal(a.hits.wavelengths, b.hits.wavelengths)
