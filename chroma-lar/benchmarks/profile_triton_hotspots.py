#!/usr/bin/env python3
"""Sweep Triton launch policy and attribute the detector's GPU hot spots.

This benchmark is deliberately external to :mod:`chroma_lar.triton_backend`.
It temporarily wraps stable simulation methods with CUDA events, but does not
replace a kernel, alter a tensor, or select a different physics path.  The
ordinary sweep reports end-to-end throughput and inclusive phase spans.  A
separate PyTorch-profiler pass groups individual CUDA kernels into families.

An optional diagnostic pass adds GPU reductions after boundary resolution to
measure wire and PMT candidate pressure.  Those reductions preserve photon
results but perturb scheduling, so no timing from that pass is reported as a
performance result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
for source_root in (REPOSITORY / "chroma-lite", REPOSITORY / "chroma-lar"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


DEFAULT_HISTORY_LENGTHS = (1, 2, 3, 4, 6, 8)
DEFAULT_BLOCK_SIZES = (64, 128, 256)

# The order matters: specific compact/gather kernels must be classified before
# the generic scan/compaction fallbacks.
KERNEL_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("source_generation", ("generate_photon_bomb",)),
    ("bulk_collision", ("collision_first", "bulk_collision")),
    ("boundary_gather", ("gather_boundary_rays",)),
    (
        "analytic_geometry",
        ("analytic_boundary", "compact_wire_x_candidates"),
    ),
    ("boundary_classification", ("production_boundary_classify",)),
    (
        "pmt_union_broadphase",
        ("mark_union_candidates", "compact_union_candidates"),
    ),
    (
        "pmt_grid_broadphase",
        ("count_grid_candidates", "fill_grid_candidates"),
    ),
    ("pmt_fused_lattice_blas", ("nearest_pmt_fused_grid",)),
    ("pmt_tlas_traversal", ("nearest_pmt_tlas",)),
    (
        "pmt_instance_broadphase",
        (
            "count_instance_candidates",
            "fill_instance_candidates",
            "scatter_candidate_counts",
        ),
    ),
    (
        "pmt_blas_traversal",
        (
            "nearest_hit_candidates",
            "traverse_bvh",
            "nearest_bvh",
        ),
    ),
    (
        "pmt_reduce_finalize",
        (
            "initialize_instance_results",
            "reduce_instance_candidates",
            "finalize_instance_hits",
            "refine_chroma_world_hits",
        ),
    ),
    ("boundary_merge", ("boundary_merge", "chroma_global_merge")),
    (
        "boundary_physics",
        ("boundary_step", "production_boundary_branch"),
    ),
    (
        "queue_scan_compaction",
        (
            "compact_status",
            "device_scan",
            "scan_kernel",
            "cub::",
            "radix_sort",
            "sorting",
            "index_select",
        ),
    ),
    (
        "host_rendezvous_transfer",
        ("memcpy dtoh", "memcpy d2h", "device -> pageable", "device -> pinned"),
    ),
    (
        "host_queue_bookkeeping",
        ("catarraybatchedcopy",),
    ),
    (
        "memory_initialization_elementwise",
        ("memset", "fillfunctor", "elementwise_kernel", "vectorized_elementwise"),
    ),
)

HISTORY_BITS: tuple[tuple[str, int], ...] = (
    ("no_hit", 1 << 0),
    ("bulk_absorb", 1 << 1),
    ("surface_detect", 1 << 2),
    ("surface_absorb", 1 << 3),
    ("rayleigh_scatter", 1 << 4),
    ("reflect_diffuse", 1 << 5),
    ("reflect_specular", 1 << 6),
    ("surface_reemit", 1 << 7),
    ("surface_transmit", 1 << 8),
    ("bulk_reemit", 1 << 9),
    ("nan_abort", 1 << 15),
)
TERMINAL_MASK = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 15)


def _positive_csv(value: str) -> tuple[int, ...]:
    """Parse a unique, order-preserving list of positive integers."""

    try:
        raw = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not raw or any(item <= 0 for item in raw):
        raise argparse.ArgumentTypeError("all values must be positive")
    return tuple(dict.fromkeys(raw))


def _configuration_grid(
    history_lengths: Iterable[int], block_sizes: Iterable[int]
) -> tuple[tuple[int, int], ...]:
    histories = tuple(int(item) for item in history_lengths)
    blocks = tuple(int(item) for item in block_sizes)
    if not histories or not blocks:
        raise ValueError("history lengths and block sizes cannot be empty")
    if any(item <= 0 for item in histories + blocks):
        raise ValueError("history lengths and block sizes must be positive")
    return tuple((history, block) for history in histories for block in blocks)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.hex()
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _device_metadata(torch: Any, device: Any) -> dict[str, Any]:
    """Collect properties from the selected runtime device, never a label."""

    concrete = torch.device(device)
    if concrete.type != "cuda":
        raise ValueError("the hotspot benchmark requires a CUDA device")
    index = torch.cuda.current_device() if concrete.index is None else concrete.index
    properties = torch.cuda.get_device_properties(index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    capability = torch.cuda.get_device_capability(index)
    fields = (
        "major",
        "minor",
        "multi_processor_count",
        "total_memory",
        "shared_memory_per_block",
        "shared_memory_per_multiprocessor",
        "regs_per_block",
        "regs_per_multiprocessor",
        "warp_size",
        "max_threads_per_block",
        "max_threads_per_multi_processor",
        "l2_cache_size",
        "clock_rate",
        "memory_clock_rate",
        "memory_bus_width",
        "uuid",
    )
    detail = {
        name: _json_scalar(getattr(properties, name))
        for name in fields
        if hasattr(properties, name)
    }
    return {
        "requested": str(device),
        "index": int(index),
        "name": str(properties.name),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "device_count_visible": int(torch.cuda.device_count()),
        "free_memory_bytes_at_start": int(free_bytes),
        "total_memory_bytes_runtime": int(total_bytes),
        "properties": detail,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _canonical_hit_signature(times: Any, channels: Any) -> dict[str, Any]:
    """Hash an order-independent, bit-exact compact-hit ensemble."""

    time_array = np.ascontiguousarray(np.asarray(times, dtype=np.float32).reshape(-1))
    channel_array = np.ascontiguousarray(
        np.asarray(channels, dtype=np.int32).reshape(-1)
    )
    if time_array.shape != channel_array.shape:
        raise ValueError("time and channel hit arrays must have the same shape")
    time_words = time_array.view(np.uint32)
    order = np.lexsort((time_words, channel_array))
    sorted_channels = np.asarray(channel_array[order], dtype="<i4")
    sorted_time_words = np.asarray(time_words[order], dtype="<u4")
    digest = hashlib.sha256()
    digest.update(sorted_channels.tobytes(order="C"))
    digest.update(sorted_time_words.tobytes(order="C"))
    return {
        "hit_count": int(time_array.size),
        "canonical_raw_sha256": digest.hexdigest(),
    }


def _summarize_trace(trace: Sequence[Sequence[int]]) -> dict[str, Any]:
    if not trace:
        return {
            "rounds": 0,
            "boundary_events": 0,
            "surviving_events": 0,
            "terminated_events": 0,
            "weighted_survival_fraction": 0.0,
            "boundary_queue_max": 0,
            "boundary_queue_median": 0.0,
            "boundary_queue_p95": 0.0,
        }
    values = np.asarray(trace, dtype=np.int64)
    if (
        values.ndim != 2
        or values.shape[1] != 2
        or np.any(values < 0)
        or np.any(values[:, 1] > values[:, 0])
    ):
        raise ValueError(
            "trace must contain non-negative (boundary, survivor) pairs with "
            "survivors no greater than boundary events"
        )
    boundary = values[:, 0]
    surviving = values[:, 1]
    boundary_total = int(boundary.sum())
    survivor_total = int(surviving.sum())
    return {
        "rounds": int(values.shape[0]),
        "boundary_events": boundary_total,
        "surviving_events": survivor_total,
        "terminated_events": int(boundary_total - survivor_total),
        "weighted_survival_fraction": (
            float(survivor_total / boundary_total) if boundary_total else 0.0
        ),
        "boundary_queue_max": int(boundary.max(initial=0)),
        "boundary_queue_median": float(np.median(boundary)),
        "boundary_queue_p95": float(np.quantile(boundary, 0.95)),
    }


@dataclass
class _CudaSpan:
    label: str
    start: Any
    end: Any


class CudaPhaseRecorder:
    """Record inclusive CUDA-stream spans around unmodified Python calls."""

    def __init__(self, torch: Any):
        self.torch = torch
        self._spans: list[_CudaSpan] = []

    @contextmanager
    def span(self, label: str) -> Iterator[None]:
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._spans.append(_CudaSpan(label, start, end))

    def summary(self) -> dict[str, dict[str, float | int]]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for span in self._spans:
            grouped[span.label].append(float(span.start.elapsed_time(span.end)))
        return {
            label: {
                "calls": len(values),
                "total_ms": float(sum(values)),
                "mean_ms": float(statistics.fmean(values)),
                "minimum_ms": float(min(values)),
                "maximum_ms": float(max(values)),
            }
            for label, values in sorted(grouped.items())
        }


@contextmanager
def _instrument_simulation_phases(
    simulation: Any, recorder: CudaPhaseRecorder
) -> Iterator[None]:
    """Bracket stable backend methods and restore the instance afterwards."""

    methods = {
        "_new_state": "source_generation",
        "_propagate_state": "propagation_inclusive",
        "_resolve_boundaries": "boundary_resolution",
        "_step_boundaries": "boundary_physics",
        "_compact_reservoir_state": "reservoir_compaction",
        "_state_hits": "hit_selection",
    }
    originals = {name: getattr(simulation, name) for name in methods}
    prior_instance_attributes = {
        name: simulation.__dict__.get(name) for name in methods
    }
    had_instance_attribute = {
        name: name in simulation.__dict__ for name in methods
    }
    for name, label in methods.items():
        original = originals[name]

        def wrapped(*args: Any, _original=original, _label=label, **kwargs: Any):
            with recorder.span(_label):
                return _original(*args, **kwargs)

        setattr(simulation, name, wrapped)
    try:
        yield
    finally:
        for name in originals:
            if had_instance_attribute[name]:
                setattr(simulation, name, prior_instance_attributes[name])
            else:
                delattr(simulation, name)


def _derived_phase_metrics(
    phases: Mapping[str, Mapping[str, float | int]], simulation_span_ms: float
) -> dict[str, float]:
    def total(name: str) -> float:
        return float(phases.get(name, {}).get("total_ms", 0.0))

    propagation = total("propagation_inclusive")
    geometry = total("boundary_resolution")
    boundary = total("boundary_physics")
    collision_residual = max(0.0, propagation - geometry - boundary)
    top_level_accounted = (
        total("source_generation")
        + propagation
        + total("reservoir_compaction")
        + total("hit_selection")
    )
    return {
        "collision_queue_scheduler_residual_ms": collision_residual,
        "top_level_unattributed_ms": max(0.0, simulation_span_ms - top_level_accounted),
        "boundary_resolution_fraction_of_simulation_span": (
            geometry / simulation_span_ms if simulation_span_ms else 0.0
        ),
        "boundary_physics_fraction_of_simulation_span": (
            boundary / simulation_span_ms if simulation_span_ms else 0.0
        ),
    }


def _kernel_family(name: str) -> str:
    lowered = name.lower()
    for family, patterns in KERNEL_FAMILY_PATTERNS:
        if any(pattern in lowered for pattern in patterns):
            return family
    return "other_cuda"


def _is_cuda_event(event: Any) -> bool:
    return "cuda" in str(getattr(event, "device_type", "")).lower()


def _event_duration_us(event: Any) -> float:
    """Read a CUDA FunctionEvent duration across supported Torch releases."""

    time_range = getattr(event, "time_range", None)
    elapsed_us = getattr(time_range, "elapsed_us", None)
    if callable(elapsed_us):
        value = float(elapsed_us())
        if math.isfinite(value) and value >= 0.0:
            return value
    for name, scale in (
        ("duration_time_ns", 1.0e-3),
        ("duration_ns", 1.0e-3),
        ("self_device_time_total", 1.0),
        ("device_time_total", 1.0),
        ("self_cuda_time_total", 1.0),
        ("cuda_time_total", 1.0),
    ):
        value = getattr(event, name, None)
        if value is None:
            continue
        value = float(value) * scale
        if math.isfinite(value) and value >= 0.0:
            return value
    raise ValueError("CUDA profiler event has no readable duration")


def _aggregate_profiler_events(
    events: Iterable[Any], *, cuda_span_ms: float, top_kernels: int = 40
) -> dict[str, Any]:
    families: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"calls": 0, "cuda_time_us": 0.0}
    )
    kernels: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"calls": 0, "cuda_time_us": 0.0}
    )
    unreadable = 0
    for event in events:
        if not _is_cuda_event(event):
            continue
        try:
            duration = _event_duration_us(event)
        except (TypeError, ValueError, OverflowError):
            unreadable += 1
            continue
        name = str(getattr(event, "name", getattr(event, "key", "<unnamed>")))
        family = _kernel_family(name)
        families[family]["calls"] += 1
        families[family]["cuda_time_us"] += duration
        kernels[name]["calls"] += 1
        kernels[name]["cuda_time_us"] += duration

    family_rows = []
    total_us = float(sum(float(row["cuda_time_us"]) for row in families.values()))
    for family, row in sorted(
        families.items(), key=lambda item: float(item[1]["cuda_time_us"]), reverse=True
    ):
        family_rows.append(
            {
                "family": family,
                "calls": int(row["calls"]),
                "cuda_time_ms": float(row["cuda_time_us"]) / 1000.0,
                "fraction_of_summed_cuda_activity": (
                    float(row["cuda_time_us"]) / total_us if total_us else 0.0
                ),
            }
        )
    kernel_rows = [
        {
            "name": name,
            "family": _kernel_family(name),
            "calls": int(row["calls"]),
            "cuda_time_ms": float(row["cuda_time_us"]) / 1000.0,
        }
        for name, row in sorted(
            kernels.items(),
            key=lambda item: float(item[1]["cuda_time_us"]),
            reverse=True,
        )[: max(0, int(top_kernels))]
    ]
    return {
        "cuda_event_count": int(sum(int(row["calls"]) for row in families.values())),
        "unreadable_cuda_event_count": int(unreadable),
        "summed_cuda_activity_ms": total_us / 1000.0,
        # This can exceed one if kernels overlap; it is not presented as GPU
        # utilization.  On the current single stream, its complement exposes
        # host rendezvous and launch bubbles relevant to persistent execution.
        "summed_activity_over_cuda_span": (
            total_us / (1000.0 * cuda_span_ms) if cuda_span_ms else 0.0
        ),
        "families": family_rows,
        "top_kernels": kernel_rows,
    }


class BoundaryCounterCollector:
    """Queue lightweight diagnostic reductions after boundary resolution."""

    def __init__(self, torch: Any, simulation: Any):
        self.torch = torch
        self.simulation = simulation
        self.boundary_rays: list[int] = []
        self.scalars: dict[str, list[Any]] = defaultdict(list)

    def observe(self, queue: Any, merged: Sequence[Any]) -> None:
        count = int(queue.numel())
        self.boundary_rays.append(count)
        if count == 0:
            return
        torch = self.torch
        wire_count = self.simulation.analytic_workspace.candidate_count.clone()
        pmt = self.simulation.pmt_workspace.outputs(count)
        candidate_counts = pmt.candidate_counts
        distance = merged[0]
        instance = merged[5]
        finite_hit = torch.isfinite(distance)
        self.scalars["wire_candidate_rays"].append(wire_count.to(torch.int64))
        self.scalars["pmt_candidate_pairs"].append(
            candidate_counts.to(torch.int64).sum()
        )
        self.scalars["pmt_candidate_rays"].append(
            torch.count_nonzero(candidate_counts > 0).to(torch.int64)
        )
        self.scalars["pmt_candidate_max"].append(
            candidate_counts.max().to(torch.int64)
        )
        self.scalars["pmt_hits"].append(
            torch.count_nonzero(pmt.instance_ids >= 0).to(torch.int64)
        )
        self.scalars["merged_pmt"].append(
            torch.count_nonzero(instance >= 0).to(torch.int64)
        )
        self.scalars["merged_box"].append(
            torch.count_nonzero(instance <= -2).to(torch.int64)
        )
        self.scalars["merged_wire"].append(
            torch.count_nonzero((instance == -1) & finite_hit).to(torch.int64)
        )
        self.scalars["merged_no_hit"].append(
            torch.count_nonzero(~finite_hit).to(torch.int64)
        )

    def summary(self) -> dict[str, Any]:
        boundary_total = int(sum(self.boundary_rays))

        def summed(name: str) -> int:
            values = self.scalars.get(name, ())
            if not values:
                return 0
            return int(self.torch.stack(tuple(values)).sum().item())

        maxima = self.scalars.get("pmt_candidate_max", ())
        maximum = (
            int(self.torch.stack(tuple(maxima)).max().item()) if maxima else 0
        )
        pairs = summed("pmt_candidate_pairs")
        rays = summed("pmt_candidate_rays")
        return {
            "boundary_rounds_observed": len(self.boundary_rays),
            "boundary_rays": boundary_total,
            "wire_candidate_rays": summed("wire_candidate_rays"),
            "pmt_candidate_rays": rays,
            "pmt_candidate_pairs": pairs,
            "pmt_candidate_max_per_ray": maximum,
            "pmt_candidates_per_candidate_ray": float(pairs / rays) if rays else 0.0,
            "pmt_candidate_fraction_of_boundary_rays": (
                float(rays / boundary_total) if boundary_total else 0.0
            ),
            "pmt_hits_before_merge": summed("pmt_hits"),
            "merged_boundary_kinds": {
                "pmt": summed("merged_pmt"),
                "box": summed("merged_box"),
                "wire": summed("merged_wire"),
                "no_hit": summed("merged_no_hit"),
            },
        }


@contextmanager
def _collect_boundary_counters(
    simulation: Any, collector: BoundaryCounterCollector
) -> Iterator[None]:
    original = simulation._resolve_boundaries
    had_instance_attribute = "_resolve_boundaries" in simulation.__dict__
    prior_instance_attribute = simulation.__dict__.get("_resolve_boundaries")

    def wrapped(state: Any, queue: Any):
        merged = original(state, queue)
        collector.observe(queue, merged)
        return merged

    simulation._resolve_boundaries = wrapped
    try:
        yield
    finally:
        if had_instance_attribute:
            simulation._resolve_boundaries = prior_instance_attribute
        else:
            delattr(simulation, "_resolve_boundaries")


def _final_state_summary(torch: Any, states: Sequence[Sequence[Any]]) -> dict[str, Any]:
    flag_counts = {name: 0 for name, _ in HISTORY_BITS}
    photons = 0
    terminal = 0
    committed_steps = 0
    maximum_steps = 0
    rng_blocks = 0
    for state in states:
        histories = state[4]
        counters = state[5]
        steps = state[9]
        photons += int(histories.numel())
        for name, bit in HISTORY_BITS:
            flag_counts[name] += int(torch.count_nonzero(histories & bit).item())
        terminal += int(torch.count_nonzero(histories & TERMINAL_MASK).item())
        committed_steps += int(steps.to(torch.int64).sum().item())
        if steps.numel():
            maximum_steps = max(maximum_steps, int(steps.max().item()))
        rng_blocks += int(counters.to(torch.int64).sum().item())
    return {
        "photons": photons,
        "terminal_photons": terminal,
        "unterminated_photons": photons - terminal,
        # Flags are cumulative per photon.  These are incidence counts, not
        # process-event frequencies; the distinction is important for SER
        # branch-splitting decisions.
        "photons_ever_with_history_bit": flag_counts,
        "committed_interactions_from_step_counters": committed_steps,
        "maximum_interactions_per_photon": maximum_steps,
        "rng_counter_blocks_total": rng_blocks,
    }


def _run_once(
    torch: Any,
    simulation: Any,
    *,
    nphotons: int,
    center: Sequence[float],
    voxel_size: float,
    seed: int,
    max_steps: int,
) -> dict[str, Any]:
    recorder = CudaPhaseRecorder(torch)
    total_start = torch.cuda.Event(enable_timing=True)
    simulation_end = torch.cuda.Event(enable_timing=True)
    transfer_end = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats(simulation.device)
    allocated_before = int(torch.cuda.memory_allocated(simulation.device))
    reserved_before = int(torch.cuda.memory_reserved(simulation.device))
    total_start.record()
    wall_start = time.perf_counter()
    with _instrument_simulation_phases(simulation, recorder):
        result = simulation.simulate(
            nphotons,
            center,
            voxel_size=voxel_size,
            seed=seed,
            max_steps=max_steps,
        )
    simulation_end.record()
    hit_times, hit_channels = result.flat_hits.to_numpy()
    transfer_end.record()
    torch.cuda.synchronize(simulation.device)
    wall_seconds = time.perf_counter() - wall_start
    phases = recorder.summary()
    simulation_span_ms = float(total_start.elapsed_time(simulation_end))
    transfer_span_ms = float(simulation_end.elapsed_time(transfer_end))
    end_to_end_span_ms = float(total_start.elapsed_time(transfer_end))
    record = {
        "seed": int(seed),
        "wall_seconds_source_through_host_hits": wall_seconds,
        "photons_per_second_wall": float(nphotons / wall_seconds),
        "cuda_event_timing": {
            "simulation_span_ms": simulation_span_ms,
            "compact_hit_transfer_span_ms": transfer_span_ms,
            "end_to_end_span_ms": end_to_end_span_ms,
            "phases_inclusive": phases,
            "derived": _derived_phase_metrics(phases, simulation_span_ms),
        },
        "backend_stats": {
            "elapsed_seconds": float(result.stats.elapsed_seconds),
            "detections": int(result.stats.detections),
            "tiles": int(result.stats.tiles),
            "boundary_rounds": int(result.stats.boundary_rounds),
            "dense_rounds": int(result.stats.dense_rounds),
            "reservoir_rounds": int(result.stats.reservoir_rounds),
            "reservoir_photons": int(result.stats.reservoir_photons),
            "boundary_events": int(result.stats.boundary_events),
        },
        "boundary_trace": _summarize_trace(simulation.last_trace),
        "hits": _canonical_hit_signature(hit_times, hit_channels),
        "memory": {
            "allocated_before_bytes": allocated_before,
            "allocated_after_bytes": int(torch.cuda.memory_allocated(simulation.device)),
            "reserved_before_bytes": reserved_before,
            "reserved_after_bytes": int(torch.cuda.memory_reserved(simulation.device)),
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(simulation.device)
            ),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(simulation.device)),
        },
    }
    return record


def _summarize_runs(nphotons: int, runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    elapsed = np.asarray(
        [float(run["wall_seconds_source_through_host_hits"]) for run in runs],
        dtype=np.float64,
    )
    rates = nphotons / elapsed
    return {
        "median_photons_per_second": float(np.median(rates)),
        "minimum_photons_per_second": float(rates.min()),
        "maximum_photons_per_second": float(rates.max()),
        "sustained_photons_per_second": float(nphotons * len(runs) / elapsed.sum()),
        "p95_latency_photons_per_second": float(
            nphotons / np.quantile(elapsed, 0.95)
        ),
        "median_cuda_end_to_end_ms": float(
            np.median(
                [run["cuda_event_timing"]["end_to_end_span_ms"] for run in runs]
            )
        ),
    }


def _work_signature(run: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        run["hits"]["hit_count"],
        run["hits"]["canonical_raw_sha256"],
        run["backend_stats"]["boundary_events"],
    )


def _simulation_kwargs(args: argparse.Namespace, history: int, block: int) -> dict[str, Any]:
    return {
        "device": args.device,
        "tile_size": None if args.tile_size == 0 else int(args.tile_size),
        "history_length": int(history),
        "block_size": int(block),
        "reservoir_rounds": int(args.reservoir_rounds),
        "history_epochs_per_poll": int(args.history_epochs_per_poll),
        "device_scheduler": bool(args.device_scheduler),
        "device_round_batch": int(args.device_round_batch),
        "portal_boundary": bool(args.portal_boundary),
        "fused_pmt": bool(args.fused_pmt),
    }


def _new_simulation(args: argparse.Namespace, history: int, block: int) -> Any:
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    return Reflect3WiresTritonSimulation(**_simulation_kwargs(args, history, block))


def _kernel_profile(
    torch: Any,
    args: argparse.Namespace,
    history: int,
    block: int,
    nphotons: int,
) -> dict[str, Any]:
    from torch.profiler import ProfilerActivity, profile

    simulation = _new_simulation(args, history, block)
    warmup = min(int(args.warmup_photons), nphotons)
    if warmup:
        simulation.simulate(
            warmup,
            args.center,
            voxel_size=args.voxel_size,
            seed=args.seed ^ 0x61C88647,
            max_steps=args.max_steps,
        ).flat_hits.to_numpy()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        start.record()
        result = simulation.simulate(
            nphotons,
            args.center,
            voxel_size=args.voxel_size,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        hit_times, hit_channels = result.flat_hits.to_numpy()
        end.record()
        torch.cuda.synchronize(simulation.device)
    span_ms = float(start.elapsed_time(end))
    aggregate = _aggregate_profiler_events(
        prof.events(), cuda_span_ms=span_ms, top_kernels=args.top_kernels
    )
    aggregate.update(
        {
            "photons": int(nphotons),
            "history_length": int(history),
            "block_size": int(block),
            "cuda_span_ms": span_ms,
            "profiler_warning": (
                "Profiler activity is a separate diagnostic run and may add overhead; "
                "use family shares and call counts, not its throughput, for decisions."
            ),
            "hits": _canonical_hit_signature(hit_times, hit_channels),
            "boundary_trace": _summarize_trace(simulation.last_trace),
        }
    )
    return aggregate


def _diagnostic_pass(
    torch: Any,
    args: argparse.Namespace,
    history: int,
    block: int,
    nphotons: int,
) -> dict[str, Any]:
    simulation = _new_simulation(args, history, block)
    collector = BoundaryCounterCollector(torch, simulation)
    with _collect_boundary_counters(simulation, collector):
        result = simulation.simulate(
            nphotons,
            args.center,
            voxel_size=args.voxel_size,
            seed=args.seed,
            max_steps=args.max_steps,
            keep_final_states=True,
        )
    hit_times, hit_channels = result.flat_hits.to_numpy()
    torch.cuda.synchronize(simulation.device)
    return {
        "photons": int(nphotons),
        "history_length": int(history),
        "block_size": int(block),
        "timing_valid": False,
        "timing_warning": (
            "Counter reductions and keep_final_states perturb scheduling; this pass "
            "is for work/candidate counts only and is excluded from throughput."
        ),
        "boundary_candidates": collector.summary(),
        "final_state": _final_state_summary(torch, result.final_states or ()),
        "hits": _canonical_hit_signature(hit_times, hit_channels),
        "boundary_trace": _summarize_trace(simulation.last_trace),
    }


def _software_metadata(torch: Any) -> dict[str, Any]:
    import triton

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nphotons", type=int, default=5_000_000)
    parser.add_argument(
        "--history-lengths",
        type=_positive_csv,
        default=DEFAULT_HISTORY_LENGTHS,
        help="comma-separated collision history lengths",
    )
    parser.add_argument(
        "--block-sizes",
        type=_positive_csv,
        default=DEFAULT_BLOCK_SIZES,
        help="comma-separated source/collision/boundary block sizes",
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--warmup-photons", type=int, default=131_072)
    parser.add_argument("--profile-photons", type=int, default=262_144)
    parser.add_argument("--diagnostic-photons", type=int, default=262_144)
    parser.add_argument(
        "--kernel-profile",
        choices=("off", "best", "all"),
        default="best",
        help="run the intrusive CUDA kernel profiler separately",
    )
    parser.add_argument("--top-kernels", type=int, default=40)
    parser.add_argument("--tile-size", type=int, default=0, help="0 selects auto sizing")
    parser.add_argument("--reservoir-rounds", type=int, default=128)
    parser.add_argument("--history-epochs-per-poll", type=int, default=1)
    parser.add_argument(
        "--device-scheduler",
        action="store_true",
        help="profile the fixed-round device-count production scheduler",
    )
    parser.add_argument(
        "--portal-boundary",
        action="store_true",
        help="route certified box-face hits directly (device scheduler only)",
    )
    parser.add_argument(
        "--fused-pmt",
        action="store_true",
        help="use the production fused lattice/BLAS PMT query",
    )
    parser.add_argument("--device-round-batch", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument("--center", type=float, nargs=3, default=(-1000.0, 0.0, 0.0))
    parser.add_argument("--voxel-size", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.nphotons <= 0 or args.replicates < 2:
        parser.error("nphotons must be positive and replicates must be at least two")
    if args.warmup_photons < 0 or args.profile_photons < 0 or args.diagnostic_photons < 0:
        parser.error("warmup/profile/diagnostic photon counts cannot be negative")
    if args.tile_size < 0:
        parser.error("tile-size cannot be negative")
    if args.portal_boundary and not args.device_scheduler:
        parser.error("--portal-boundary requires --device-scheduler")
    if (
        args.reservoir_rounds <= 0
        or args.history_epochs_per_poll <= 0
        or args.device_round_batch <= 0
        or args.max_steps <= 0
        or args.top_kernels < 0
        or args.voxel_size < 0.0
    ):
        parser.error("scheduler sizes and max-steps must be positive")
    configurations = _configuration_grid(args.history_lengths, args.block_sizes)

    import torch

    selected_device = torch.device(args.device)
    if selected_device.type != "cuda" or not torch.cuda.is_available():
        parser.error("an available CUDA device is required")
    if selected_device.index is not None:
        torch.cuda.set_device(selected_device.index)
    selected_device = torch.device(
        "cuda", torch.cuda.current_device() if selected_device.index is None else selected_device.index
    )
    args.device = str(selected_device)

    payload: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "reflect3wires Triton hotspot and launch-policy sweep",
        "host": platform.node(),
        "command": [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        "device": _device_metadata(torch, selected_device),
        "software": _software_metadata(torch),
        "configuration": {
            "nphotons": int(args.nphotons),
            "history_lengths": list(args.history_lengths),
            "block_sizes": list(args.block_sizes),
            "replicates": int(args.replicates),
            "warmup_photons": int(args.warmup_photons),
            "profile_photons": int(args.profile_photons),
            "diagnostic_photons": int(args.diagnostic_photons),
            "tile_size": None if args.tile_size == 0 else int(args.tile_size),
            "reservoir_rounds": int(args.reservoir_rounds),
            "history_epochs_per_poll": int(args.history_epochs_per_poll),
            "device_scheduler": bool(args.device_scheduler),
            "device_round_batch": int(args.device_round_batch),
            "portal_boundary": bool(args.portal_boundary),
            "fused_pmt": bool(args.fused_pmt),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "center_mm": [float(value) for value in args.center],
            "voxel_size_mm": float(args.voxel_size),
        },
        "timing_contract": {
            "throughput": "wall clock from source generation through host-readable compact hits",
            "phase_events": "inclusive CUDA-event stream spans around unmodified backend methods",
            "kernel_profile": "separate intrusive PyTorch-profiler run, excluded from throughput ranking",
            "candidate_counters": "separate reduction pass, excluded from all performance timing",
        },
        "optimization_evidence_map": {
            "1_branch_specialization": [
                "kernel_profiles[].families[boundary_classification|boundary_physics]",
                "diagnostic_passes[].final_state.photons_ever_with_history_bit",
                "records[].runs[].boundary_trace.weighted_survival_fraction",
            ],
            "2_collision_unroll_and_block_size": [
                "records[].history_length",
                "records[].block_size",
                "kernel_profiles[].families[bulk_collision]",
            ],
            "3_common_path_fusion": [
                "records[].runs[].backend_stats.boundary_rounds",
                "records[].runs[].cuda_event_timing.phases_inclusive",
                "kernel_profiles[].families[].calls",
            ],
            "4_exact_pmt_locator": [
                "kernel_profiles[].families[pmt_grid_broadphase|pmt_tlas_traversal|pmt_blas_traversal]",
                "diagnostic_passes[].boundary_candidates",
            ],
            "5_device_resident_scheduling": [
                "kernel_profiles[].families[host_rendezvous_transfer|host_queue_bookkeeping|queue_scan_compaction]",
                "kernel_profiles[].summed_activity_over_cuda_span",
                "records[].runs[].cuda_event_timing.derived.collision_queue_scheduler_residual_ms",
            ],
        },
        "records": [],
        "kernel_profiles": [],
        "diagnostic_passes": [],
    }
    output = args.json.expanduser().resolve()
    _write_json(output, payload)

    reference_signatures: dict[int, tuple[Any, ...]] = {}
    for config_index, (history, block) in enumerate(configurations):
        simulation = _new_simulation(args, history, block)
        warmup = min(int(args.warmup_photons), int(args.nphotons))
        if warmup:
            simulation.simulate(
                warmup,
                args.center,
                voxel_size=args.voxel_size,
                seed=args.seed ^ 0x61C88647,
                max_steps=args.max_steps,
            ).flat_hits.to_numpy()
        runs = []
        for replicate in range(args.replicates):
            seed = (args.seed + replicate * 0x9E3779B1) & 0xFFFFFFFF
            run = _run_once(
                torch,
                simulation,
                nphotons=args.nphotons,
                center=args.center,
                voxel_size=args.voxel_size,
                seed=seed,
                max_steps=args.max_steps,
            )
            run["replicate"] = replicate
            signature = _work_signature(run)
            if config_index == 0:
                reference_signatures[replicate] = signature
            run["matches_first_configuration"] = (
                signature == reference_signatures[replicate]
            )
            runs.append(run)
        plan = simulation.last_tile_plan
        record = {
            "history_length": int(history),
            "block_size": int(block),
            "physics_matches_first_configuration": all(
                bool(run["matches_first_configuration"]) for run in runs
            ),
            "tile_plan": {
                "tile_capacity": int(plan.tile_capacity),
                "tile_count": int(plan.tile_count),
                "available_bytes": (
                    None if plan.available_bytes is None else int(plan.available_bytes)
                ),
                "estimated_peak_bytes": int(plan.estimated_peak_bytes),
                "explicit": bool(plan.explicit),
            },
            "summary": _summarize_runs(args.nphotons, runs),
            "runs": runs,
        }
        payload["records"].append(record)
        _write_json(output, payload)
        print(json.dumps({key: record[key] for key in ("history_length", "block_size", "physics_matches_first_configuration", "summary")}, sort_keys=True), flush=True)
        del simulation
        torch.cuda.empty_cache()

    best = max(
        payload["records"],
        key=lambda record: float(record["summary"]["median_photons_per_second"]),
    )
    payload["best_configuration"] = {
        "history_length": int(best["history_length"]),
        "block_size": int(best["block_size"]),
        **best["summary"],
    }
    if args.kernel_profile != "off" and args.profile_photons:
        selected = (
            payload["records"]
            if args.kernel_profile == "all"
            else [best]
        )
        for record in selected:
            payload["kernel_profiles"].append(
                _kernel_profile(
                    torch,
                    args,
                    int(record["history_length"]),
                    int(record["block_size"]),
                    min(int(args.profile_photons), int(args.nphotons)),
                )
            )
            _write_json(output, payload)
            torch.cuda.empty_cache()

    if args.diagnostic_photons:
        payload["diagnostic_passes"].append(
            _diagnostic_pass(
                torch,
                args,
                int(best["history_length"]),
                int(best["block_size"]),
                min(int(args.diagnostic_photons), int(args.nphotons)),
            )
        )
    payload["all_configurations_physics_matched"] = all(
        bool(record["physics_matches_first_configuration"])
        for record in payload["records"]
    )
    _write_json(output, payload)
    print(json.dumps(payload["best_configuration"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
