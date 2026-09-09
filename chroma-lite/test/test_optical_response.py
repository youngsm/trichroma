import numpy as np
import pytest

from chroma.triton.optical_response import OpticalHits, PMTResponse, TabulatedCDF, digitize


def hits(n):
    return OpticalHits(np.full(n, 10.), np.arange(n) % 2, np.arange(n), np.zeros(n, int))


def test_tts_statistics_and_identity_under_reordering():
    h = hits(100_000)
    response = PMTResponse(tts_sigma=[1., 3.], transit_time=[20., 40.])
    pe = response.apply(h, seed=21)
    for ch, mean, sigma in ((0, 30, 1), (1, 50, 3)):
        t = pe.times[pe.channels == ch]
        assert abs(t.mean() - mean) < 0.04
        assert abs(t.std() - sigma) < 0.04
    rev = OpticalHits(h.times[::-1], h.channels[::-1], h.photon_ids[::-1], h.event_indices[::-1])
    np.testing.assert_array_equal(response.apply(rev, seed=21).times[::-1], pe.times)
    np.testing.assert_array_equal(h.times, 10.)


def test_measured_cdf_plateaus_atoms_and_charge():
    cdf = TabulatedCDF([0, 1, 2, 3], [0, .5, .5, 1])
    np.testing.assert_allclose(cdf.sample([.25, .5, .75]), [.5, 2, 2.5])
    response = PMTResponse(time_cdf=TabulatedCDF([2, 2], [0, 1]),
                           charge_cdf=TabulatedCDF([3, 3], [0, 1]), gain=2)
    pe = response.apply(hits(20))
    np.testing.assert_array_equal(pe.times, 12.)
    np.testing.assert_array_equal(pe.charges, 6.)
    with pytest.raises(ValueError):
        TabulatedCDF.from_pdf([0, 1], [0, 0])


def test_digitizer_subsample_pulses_empty_events_and_saturation():
    h = OpticalHits([.5, 1.5], [0, 0], [10, 11], [7, 7])
    pe = PMTResponse().apply(h)
    wf = digitize(pe, event_indices=[7, 99], channel_count=1, start_ns=0,
                  sample_period_ns=1, sample_count=5, pulse_times_ns=[0, 1, 2],
                  pulse_adc_per_pe=[0, 10, 0], baseline=2, adc_bits=3)
    np.testing.assert_array_equal(wf.samples[0, 0], [2, 7, 7, 7, 2])
    np.testing.assert_array_equal(wf.samples[1], 2)


def test_collection_efficiency_not_applied_twice():
    h = hits(20_000)
    assert len(PMTResponse().apply(h).times) == len(h)
    assert abs(len(PMTResponse(collection_efficiency=.25).apply(h).times) / len(h) - .25) < .015
