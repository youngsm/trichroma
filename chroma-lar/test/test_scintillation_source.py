import numpy as np
import pytest

from chroma.triton.optical_response import TabulatedCDF
from chroma.triton.photon_input import as_photon_batch
from chroma_lar.generator.scintillation import ScintillationSource


def source():
    return ScintillationSource(TabulatedCDF([120, 136], [0, 1]), (0., 100.), (.3, .7), 200.)


def test_source_spectrum_time_and_polarization():
    p = source().photons(np.zeros((100_000, 3)), times=12, seed=83)
    as_photon_batch(p)
    assert abs(p.wavelengths.mean() - 128) < .05
    assert abs(p.times.mean() - 82) < 1
    assert abs(np.mean(p.times == 12) - .3) < .005
    assert np.max(np.abs(p.direction.mean(axis=0))) < .01
    assert np.max(np.abs((p.direction*p.polarization).sum(axis=1))) < 1e-6
    small = source().photons(np.zeros((500, 3)), times=12, seed=83, photon_id_base=500)
    np.testing.assert_array_equal(small.times, p.times[500:1000])


def test_deposition_quenching_counts_and_empty_input():
    p = source().from_depositions(np.zeros((2000, 3)), 1., quenching=.5, seed=3)
    assert abs(len(p.times) - 200_000) < 2000
    p = source().from_depositions(np.zeros((3, 3)), 0.)
    assert p.photon_count == 0
    with pytest.raises(ValueError):
        source().from_depositions(np.zeros((1, 3)), 1000, max_photons=100)


@pytest.mark.parametrize("options", [{"times": np.nan}, {"event_indices": -.5},
                                     {"event_indices": 2**32}, {"max_photons": -.1}])
def test_invalid_zero_energy_depositions_are_rejected(options):
    with pytest.raises(ValueError):
        source().from_depositions(np.zeros((1, 3)), 0., **options)


def test_fractional_photon_id_base_is_rejected():
    with pytest.raises(ValueError):
        source().photons(np.zeros((1, 3)), photon_id_base=.5)
