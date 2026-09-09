import numpy as np
import pytest

from chroma.detector import Detector
from chroma.event import Photons, SURFACE_REEMIT
from chroma.geometry import Material, Solid, Surface
from chroma.make import box
from chroma.triton.optical_response import TabulatedCDF
from chroma.triton.spectral import SpectralSimulation
from chroma_lar.geometry.pmt import build_r5912_pmt


def optics():
    materials = []
    for name in ("lar", "glass", "vacuum"):
        m = Material(name)
        m.set("refractive_index", 1.)
        m.set("absorption_length", 1e30)
        m.set("scattering_length", 1e30)
        materials.append(m)
    detector = Surface("photocathode")
    detector.set("detect", 1.)
    back = Surface("back")
    back.set("absorb", 1.)
    coat = Surface("tpb", model=2)
    coat.set("absorb", [1, 1, 0, 0], [120, 200, 201, 500])
    coat.set("reemit", 1.)
    coat.set("reemission_cdf", [0, 0, 1, 1], [120, 419, 421, 500])
    coat.reemission_time_cdf = TabulatedCDF([10, 10], [0, 1])
    coat.reemission_to_material1 = 1.
    return *materials, detector, back, coat


def test_coating_is_on_outer_window_only():
    lar, glass, vacuum, detector, back, coat = optics()
    outer, inner = build_r5912_pmt(outer_material=lar, glass=glass, vacuum=vacuum,
        photocathode_surface=detector, back_surface=back, coating_surface=coat,
        nzsteps=12, nsteps=16, return_individual_solids=True)
    coated = np.array([s is coat for s in outer.surface])
    assert np.any(coated) and np.any(~coated)
    assert all(s is None for s in outer.surface[~coated])
    assert all(s is detector or s is back for s in inner.surface)
    assert all(m is glass for m in outer.material1)
    assert all(m is lar for m in outer.material2)


def test_real_pmt_mesh_vuv_to_visible_detection_on_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    lar, glass, vacuum, sensor, back, coat = optics()
    pmt = build_r5912_pmt(outer_material=lar, glass=glass, vacuum=vacuum,
        photocathode_surface=sensor, back_surface=back, coating_surface=coat, nzsteps=12, nsteps=16)
    detector = Detector(lar)
    detector.add_pmt(pmt, displacement=np.zeros(3))
    detector.add_solid(Solid(box(1000,1000,1000), lar, lar, surface=back))
    n = 400
    photons = Photons(np.tile([.23, 50, .17], (n,1)), np.tile([0,-1,0], (n,1)),
                      np.tile([1,0,0], (n,1)), np.full(n,128.))
    result = SpectralSimulation(detector, wavelengths=np.arange(120,501), backend="triton").simulate(photons)
    assert result.step_limit_count == 0
    assert len(result.hits) > n*.4
    assert np.all(result.hits.wavelengths > 400)
    assert np.all(result.hits.times >= 10)
    assert np.all(result.final_state["flags"] & SURFACE_REEMIT)
