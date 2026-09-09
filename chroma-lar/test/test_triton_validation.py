"""Focused tests for the statistical Triton-vs-Chroma acceptance harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_HARNESS_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "validate_triton_backend.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "validate_triton_backend_for_test", _HARNESS_PATH
)
_HARNESS = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_HARNESS)


def test_quantile_noise_floor_is_finite_and_scales_with_sample_size():
    quantiles = [5.9, 7.7, 10.1, 18.3, 36.0, 66.3, 105.7, 136.8, 208.5]
    small = {
        "aggregate": {
            "time_quantiles_ns": quantiles,
            "hit_count": 40_000,
        }
    }
    large = {
        "aggregate": {
            "time_quantiles_ns": quantiles,
            "hit_count": 160_000,
        }
    }

    small_floor = _HARNESS._quantile_noise_floors(small)
    large_floor = _HARNESS._quantile_noise_floors(large)

    assert small_floor.shape == _HARNESS.DEFAULT_QUANTILES.shape
    assert np.all(np.isfinite(small_floor))
    assert np.all(small_floor > 0.0)
    np.testing.assert_allclose(large_floor, 0.5 * small_floor, rtol=1.0e-12)


def test_large_run_work_batches_are_bounded_and_complete():
    batches = list(_HARNESS._work_batches(300_000_001, 15_000_000))

    assert len(batches) == 21
    assert batches[0] == (0, 15_000_000)
    assert batches[-1] == (300_000_000, 1)
    assert sum(count for _, count in batches) == 300_000_001
    assert max(count for _, count in batches) == 15_000_000


def test_chroma_batch_source_seeds_are_stable_and_distinct():
    seeds = [
        _HARNESS._batch_seed(8123, first, 15_000_000)
        for first, _ in _HARNESS._work_batches(45_000_000, 15_000_000)
    ]

    assert seeds[0] == 8123
    assert len(set(seeds)) == 3
    assert all(0 <= seed <= 0xFFFFFFFF for seed in seeds)
