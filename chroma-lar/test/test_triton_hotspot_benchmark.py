"""CPU-only contracts for the Triton hotspot benchmark helpers."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "profile_triton_hotspots.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "profile_triton_hotspots_for_test", _BENCHMARK_PATH
)
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _BENCHMARK
_SPEC.loader.exec_module(_BENCHMARK)


def test_positive_csv_preserves_order_and_removes_duplicates():
    assert _BENCHMARK._positive_csv("8, 1,8,4") == (8, 1, 4)
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _BENCHMARK._positive_csv("1,0,4")
    with pytest.raises(argparse.ArgumentTypeError, match="comma-separated"):
        _BENCHMARK._positive_csv("1,nope")


def test_configuration_grid_has_predictable_history_major_order():
    assert _BENCHMARK._configuration_grid((1, 2), (64, 128)) == (
        (1, 64),
        (1, 128),
        (2, 64),
        (2, 128),
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        _BENCHMARK._configuration_grid((), (64,))


def test_canonical_hit_signature_is_order_independent_and_bit_exact():
    times = np.asarray([3.5, -0.0, 3.5], dtype=np.float32)
    channels = np.asarray([7, 2, 1], dtype=np.int32)
    first = _BENCHMARK._canonical_hit_signature(times, channels)
    second = _BENCHMARK._canonical_hit_signature(
        times[[2, 0, 1]], channels[[2, 0, 1]]
    )
    changed = times.copy()
    changed[1] = np.float32(0.0)

    assert first == second
    assert first["hit_count"] == 3
    assert first["canonical_raw_sha256"] != _BENCHMARK._canonical_hit_signature(
        changed, channels
    )["canonical_raw_sha256"]


def test_canonical_hit_signature_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        _BENCHMARK._canonical_hit_signature(
            np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.int32)
        )


def test_trace_summary_exposes_survival_and_queue_pressure():
    summary = _BENCHMARK._summarize_trace(((100, 80), (80, 20), (20, 0)))

    assert summary["rounds"] == 3
    assert summary["boundary_events"] == 200
    assert summary["surviving_events"] == 100
    assert summary["terminated_events"] == 100
    assert summary["weighted_survival_fraction"] == 0.5
    assert summary["boundary_queue_max"] == 100
    with pytest.raises(ValueError, match="trace"):
        _BENCHMARK._summarize_trace(((2, 3),))


@pytest.mark.parametrize(
    ("kernel", "family"),
    (
        ("collision_first_kernel_0d1d", "bulk_collision"),
        ("_compact_wire_x_candidates_kernel", "analytic_geometry"),
        ("_compact_union_candidates_kernel", "pmt_union_broadphase"),
        ("_fill_instance_candidates_kernel", "pmt_instance_broadphase"),
        ("_fill_grid_candidates_kernel", "pmt_grid_broadphase"),
        ("_nearest_pmt_tlas_kernel", "pmt_tlas_traversal"),
        ("_nearest_hit_candidates_progress_kernel", "pmt_blas_traversal"),
        ("boundary_step_kernel", "boundary_physics"),
        ("production_boundary_classify_kernel", "boundary_classification"),
        ("production_boundary_branch_kernel", "boundary_physics"),
        ("cub::DeviceScanKernel", "queue_scan_compaction"),
        ("Memcpy DtoH (Device -> Pageable)", "host_rendezvous_transfer"),
        ("CatArrayBatchedCopy_aligned16_contig", "host_queue_bookkeeping"),
        ("unknown_science_kernel", "other_cuda"),
    ),
)
def test_kernel_family_classification(kernel, family):
    assert _BENCHMARK._kernel_family(kernel) == family


class _FakeRange:
    def __init__(self, duration):
        self.duration = duration

    def elapsed_us(self):
        return self.duration


class _FakeEvent:
    def __init__(self, name, duration, device="DeviceType.CUDA"):
        self.name = name
        self.device_type = device
        self.time_range = _FakeRange(duration)


def test_profiler_aggregation_uses_only_cuda_events_and_groups_families():
    result = _BENCHMARK._aggregate_profiler_events(
        (
            _FakeEvent("collision_first_kernel", 300.0),
            _FakeEvent("collision_first_kernel", 100.0),
            _FakeEvent("boundary_step_kernel", 200.0),
            _FakeEvent("aten::empty", 50_000.0, "DeviceType.CPU"),
        ),
        cuda_span_ms=1.0,
        top_kernels=1,
    )

    assert result["cuda_event_count"] == 3
    assert result["summed_cuda_activity_ms"] == pytest.approx(0.6)
    assert result["summed_activity_over_cuda_span"] == pytest.approx(0.6)
    assert result["families"][0] == {
        "family": "bulk_collision",
        "calls": 2,
        "cuda_time_ms": pytest.approx(0.4),
        "fraction_of_summed_cuda_activity": pytest.approx(2.0 / 3.0),
    }
    assert len(result["top_kernels"]) == 1
    assert result["top_kernels"][0]["name"] == "collision_first_kernel"


def test_derived_phase_metrics_subtract_nested_boundary_spans():
    phases = {
        "source_generation": {"total_ms": 2.0},
        "propagation_inclusive": {"total_ms": 80.0},
        "boundary_resolution": {"total_ms": 30.0},
        "boundary_physics": {"total_ms": 20.0},
        "reservoir_compaction": {"total_ms": 1.0},
        "hit_selection": {"total_ms": 2.0},
    }
    result = _BENCHMARK._derived_phase_metrics(phases, 100.0)

    assert result["collision_queue_scheduler_residual_ms"] == 30.0
    assert result["top_level_unattributed_ms"] == 15.0
    assert result["boundary_resolution_fraction_of_simulation_span"] == 0.3
    assert result["boundary_physics_fraction_of_simulation_span"] == 0.2


def test_run_summary_uses_wall_scope_and_p95_latency():
    runs = [
        {
            "wall_seconds_source_through_host_hits": value,
            "cuda_event_timing": {"end_to_end_span_ms": value * 1000.0},
        }
        for value in (1.0, 2.0, 3.0)
    ]
    summary = _BENCHMARK._summarize_runs(6_000_000, runs)

    assert summary["median_photons_per_second"] == 3_000_000.0
    assert summary["sustained_photons_per_second"] == 3_000_000.0
    assert summary["median_cuda_end_to_end_ms"] == 2000.0
    assert summary["p95_latency_photons_per_second"] < 3_000_000.0


def test_default_sweep_covers_requested_history_lengths_and_blocks():
    args = _BENCHMARK._parser().parse_args(("--json", "unused.json"))

    assert args.history_lengths == (1, 2, 3, 4, 6, 8)
    assert args.block_sizes == (64, 128, 256)
    assert args.kernel_profile == "best"
    assert args.diagnostic_photons > 0
