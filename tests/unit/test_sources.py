"""GPU photon sources: the distributions of Chroma-style voxel photon bombs."""

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _numpy(photons):
    return {name: getattr(photons, name).cpu().numpy() for name in
            ("pos", "dir", "pol", "wavelengths", "t", "last_hit_triangles", "flags", "weights", "evidx")}


def test_photon_bomb_distributions():
    import torch
    from trichroma.sources import photon_bomb

    generator = torch.Generator(device="cuda")
    generator.manual_seed(1)
    n, centre = 400_000, np.array([-1155.0, 15.0, 15.0])
    p = _numpy(photon_bomb(n, centre, voxel_size=30, wavelength=450, generator=generator))
    pos, direction, pol = (p[name].astype(np.float64) for name in ("pos", "dir", "pol"))  # float64 means
    for name in ("pos", "dir", "pol"):
        assert p[name].shape == (n, 3) and p[name].dtype == np.float32
    assert p["last_hit_triangles"].dtype == p["flags"].dtype == p["evidx"].dtype == np.int32
    np.testing.assert_allclose(np.linalg.norm(p["dir"], axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(p["pol"], axis=1), 1.0, atol=1e-5)
    assert np.abs(np.einsum("ij,ij->i", p["dir"], p["pol"])).max() < 1e-5
    assert (p["pos"] >= centre - 15 - 1e-3).all() and (p["pos"] <= centre + 15 + 1e-3).all()
    np.testing.assert_allclose(pos.mean(0), centre, atol=0.1)
    np.testing.assert_allclose(direction.mean(0), 0.0, atol=0.01)       # isotropic
    np.testing.assert_allclose((direction ** 2).mean(0), 1 / 3, atol=0.005)
    np.testing.assert_allclose(pol.mean(0), 0.0, atol=0.01)
    assert (p["wavelengths"] == 450).all() and (p["t"] == 0).all() and (p["weights"] == 1).all()
    assert (p["last_hit_triangles"] == -1).all() and (p["flags"] == 0).all() and (p["evidx"] == 0).all()


def test_photon_bombs_voxel_after_voxel_and_wavelength_range():
    from trichroma.sources import photon_bombs

    centres = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, -200.0, 50.0]])
    photons = photon_bombs(20_000, centres, voxel_size=10, wavelength=(300.0, 500.0))
    assert len(photons) == 60_000
    for i, centre in enumerate(centres):
        pos = photons[i * 20_000:(i + 1) * 20_000].pos.cpu().numpy()
        assert (np.abs(pos - centre) <= 5 + 1e-3).all()
    wavelengths = photons.wavelengths.cpu().numpy()
    assert wavelengths.min() >= 300 and wavelengths.max() < 500
    assert abs(wavelengths.mean() - 400) < 2
