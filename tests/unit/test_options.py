"""CHROMA_TRITON: the Triton backend's comma-separated options."""

import pytest

from trichroma.options import Options, options


def test_defaults():
    assert options({}) == Options()
    assert options({"CHROMA_TRITON": ""}) == Options()


def test_flags_and_values():
    opts = options({"CHROMA_TRITON": " Legacy, strict ,roulette=0.05,,no-pipeline"})
    assert opts.legacy and opts.strict and opts.legacy_wires and not opts.pipeline
    assert opts.roulette == 0.05 and opts.grid and not opts.wavefront
    opts = options({"CHROMA_TRITON": "legacy-wires,wavefront,no-grid"})
    assert opts.legacy_wires and not opts.legacy and opts.wavefront and not opts.grid


@pytest.mark.parametrize("value", ["legacyy", "strict=1", "roulette", "roulette=abc", "roulette=-1", "roulette=nan"])
def test_bad_options_raise(value):
    with pytest.raises(ValueError):
        options({"CHROMA_TRITON": value})
