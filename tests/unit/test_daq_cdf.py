"""The DAQ's time and charge CDFs: Chroma's _pdf_to_cdf builds y one entry short of x."""

import numpy as np
import pytest

from chroma.detector import Detector
from chroma.geometry import Material
from trichroma.engine.core import daq_cdf


def _short_cdf(n=50):
    x = np.linspace(-6.0, 6.0, n + 1)
    y = np.cumsum(np.exp(-0.5 * (x[1:] / 1.2) ** 2))
    return x, y / y[-1]  # what the unfixed Detector._pdf_to_cdf returns: len(y) == len(x) - 1


def test_fixed_tables_restore_the_leading_zero():
    x, y = _short_cdf()
    fx, fy = daq_cdf(x, y, fixes=True)
    assert len(fx) == len(fy) == 51
    assert fy[0] == 0.0 and fy[-1] == 1.0 and np.all(np.diff(fy) >= 0)


def test_legacy_tables_pad_with_the_zero_of_fresh_memory():
    x, y = _short_cdf()
    lx, ly = daq_cdf(x, y, fixes=False)
    assert len(ly) == 51 and ly[-1] == 0.0 and np.array_equal(ly[:-1], y)


def test_complete_tables_are_unchanged():
    x = np.linspace(0.0, 1.0, 11)
    y = np.linspace(0.0, 1.0, 11)
    for fixes in (True, False):
        assert np.array_equal(daq_cdf(x, y, fixes=fixes)[1], y)
    with pytest.raises(ValueError):
        daq_cdf(x, y[:-3])


def test_detector_distributions_become_complete_tables():
    det = Detector(Material("m"))
    det.set_time_dist_gaussian(1.2, -6.0, 6.0)
    det.set_charge_dist_gaussian(1.0, 0.1, 0.5, 1.5)
    for cdf in (det.time_cdf, det.charge_cdf):
        x, y = daq_cdf(*cdf, fixes=True)
        assert len(x) == len(y) and y[0] == 0.0 and y[-1] == 1.0
