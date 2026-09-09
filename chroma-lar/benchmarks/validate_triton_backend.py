#!/usr/bin/env python3
"""Physics and throughput acceptance harness for the 450-nm Triton backend.

This script compares *ensembles*, not photon identities.  Chroma uses XORWOW
and an atomic work queue while the specialized backend uses counter-based
Philox, so same-seed trajectory equality is neither expected nor a useful
correctness criterion.  Instead, Chroma-versus-Chroma replicate differences
establish an empirical Monte-Carlo noise envelope.  Triton-versus-Chroma is
required to fit inside a scaled version of that envelope.

The public ``both`` mode launches each backend in a separate child process.
That is intentional: legacy Chroma owns a PyCUDA context, whereas Torch/Triton
uses CUDA's primary context.  Process isolation prevents context-stack state
from becoming part of either timing or result.

Examples
--------

Quick smoke run::

    python chroma-lar/benchmarks/validate_triton_backend.py \
        --backend both --nphotons 200000 --center -1000 0 0

Production-scale machine-readable run::

    python chroma-lar/benchmarks/validate_triton_backend.py \
        --backend both --nphotons 15000000 --center -1000 0 0 --json result.json
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Optional, Sequence

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
for _source_root in (REPOSITORY / "chroma-lite", REPOSITORY / "chroma-lar"):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

TARGET_WAVELENGTH_NM = 450.0
TARGET_PHOTONS_PER_SECOND = 5_000_000.0
DEFAULT_WORK_BATCH_PHOTONS = 15_000_000
DEFAULT_CHROMA_CONTAINER = (
    REPOSITORY.parent
    / "chroma-lar"
    / "installation"
    / "chroma3.lar-plib"
    / "chroma.simg"
)
DEFAULT_QUANTILES = np.asarray(
    [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    dtype=np.float64,
)
METRIC_LABELS = {
    "hit_fraction_abs": "absolute hit-fraction difference",
    "per_channel_rate_rmse": "per-channel rate RMSE",
    "per_channel_rate_max_abs": "largest per-channel rate difference",
    "per_channel_max_pull": "largest per-channel statistical pull",
    "channel_distribution_tv": "channel-distribution total variation",
    "time_quantile_max_abs_ns": "largest arrival-time quantile shift (ns)",
    "time_quantile_rmse_ns": "arrival-time quantile RMSE (ns)",
    "time_cdf_ks_binned": "binned arrival-time KS distance",
    "channel_time_tv_per_photon": "channel-time TV per emitted photon",
    "channel_time_shape_tv": "conditional channel-time shape TV",
}
GATED_METRICS = (
    "hit_fraction_abs",
    "per_channel_rate_max_abs",
    "channel_distribution_tv",
    "time_quantile_max_abs_ns",
    "time_cdf_ks_binned",
    "channel_time_tv_per_photon",
    "channel_time_shape_tv",
)


def _finite_float(value: Any) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _source_photons(
    nphotons: int,
    center: Sequence[float],
    voxel_size: float,
    seed: int,
):
    """Generate the production photon-bomb law with bounded host memory."""

    from chroma.event import Photons

    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    direction = np.empty((nphotons, 3), dtype=np.float32)
    costheta = (2.0 * rng.random_sample(nphotons) - 1.0).astype(np.float32)
    phi = (2.0 * np.pi * rng.random_sample(nphotons)).astype(np.float32)
    sintheta = np.sqrt(np.maximum(np.float32(0.0), 1.0 - costheta * costheta))
    direction[:, 0] = sintheta * np.cos(phi)
    direction[:, 1] = sintheta * np.sin(phi)
    direction[:, 2] = costheta

    helper = np.empty_like(direction)
    costheta = (2.0 * rng.random_sample(nphotons) - 1.0).astype(np.float32)
    phi = (2.0 * np.pi * rng.random_sample(nphotons)).astype(np.float32)
    sintheta = np.sqrt(np.maximum(np.float32(0.0), 1.0 - costheta * costheta))
    helper[:, 0] = sintheta * np.cos(phi)
    helper[:, 1] = sintheta * np.sin(phi)
    helper[:, 2] = costheta
    polarization = np.cross(direction, helper)
    polarization /= np.maximum(
        np.linalg.norm(polarization, axis=1, keepdims=True), np.float32(1.0e-20)
    )

    position = np.empty_like(direction)
    center = np.asarray(center, dtype=np.float32)
    for axis in range(3):
        position[:, axis] = (
            rng.random_sample(nphotons).astype(np.float32) * np.float32(voxel_size)
            - np.float32(0.5 * voxel_size)
            + center[axis]
        )
    wavelengths = np.full(nphotons, TARGET_WAVELENGTH_NM, dtype=np.float32)
    return Photons(
        pos=position,
        dir=direction,
        pol=polarization,
        wavelengths=wavelengths,
    )


def _work_batches(nphotons: int, batch_photons: int):
    """Yield ``(first, count)`` pairs without materializing a huge source.

    Chroma cannot split a single :class:`~chroma.event.Photons` event in its
    ``photons_per_batch`` scheduler.  The benchmark therefore constructs one
    bounded event at a time.  Triton's production scheduler instead derives
    its internal tile capacity from allocatable device memory while retaining
    a single global photon-ID stream.
    """

    nphotons = int(nphotons)
    batch_photons = int(batch_photons)
    if nphotons < 0:
        raise ValueError("nphotons must be non-negative")
    if batch_photons <= 0:
        raise ValueError("batch_photons must be positive")
    for first in range(0, nphotons, batch_photons):
        yield first, min(batch_photons, nphotons - first)


def _batch_seed(run_seed: int, first_photon: int, batch_photons: int) -> int:
    """Return a stable independent NumPy source seed for one Chroma batch."""

    batch_index = int(first_photon) // int(batch_photons)
    return int(run_seed + batch_index * 0x85EBCA6B) & 0xFFFFFFFF


def _summarize_hits(
    times: np.ndarray,
    channels: np.ndarray,
    nphotons: int,
    num_channels: int,
    time_edges: np.ndarray,
) -> dict[str, Any]:
    times = np.asarray(times, dtype=np.float64).reshape(-1)
    channels = np.asarray(channels, dtype=np.int64).reshape(-1)
    if times.shape != channels.shape:
        raise ValueError("hit time and channel arrays have different shapes")

    valid = (
        np.isfinite(times)
        & (times >= 0.0)
        & (channels >= 0)
        & (channels < num_channels)
    )
    valid_times = times[valid]
    valid_channels = channels[valid]
    counts = np.bincount(valid_channels, minlength=num_channels).astype(np.int64)

    # searchsorted gives a dedicated underflow bin 0, regular bins 1..B,
    # and an overflow bin B+1.  This retains every finite arrival time.
    time_slots = time_edges.size + 1
    time_index = np.searchsorted(time_edges, valid_times, side="right")
    flat_index = valid_channels * time_slots + time_index
    joint = np.bincount(
        flat_index, minlength=num_channels * time_slots
    ).reshape(num_channels, time_slots)
    quantiles = (
        np.quantile(valid_times, DEFAULT_QUANTILES)
        if valid_times.size
        else np.full(DEFAULT_QUANTILES.shape, np.nan)
    )
    return {
        "nphotons": int(nphotons),
        "raw_hit_count": int(times.size),
        "hit_count": int(valid_times.size),
        "invalid_hit_count": int(times.size - valid_times.size),
        "hit_fraction": float(valid_times.size / nphotons),
        "channel_counts": counts.tolist(),
        "time_quantiles_ns": [_finite_float(x) for x in quantiles],
        "joint_histogram_shape": [int(num_channels), int(time_slots)],
        "joint_histogram": joint.reshape(-1).tolist(),
    }


def _aggregate_backend(
    backend: str,
    run_records: list[dict[str, Any]],
    pooled_times: list[np.ndarray],
    pooled_channels: list[np.ndarray],
    nphotons: int,
    num_channels: int,
    time_edges: np.ndarray,
) -> dict[str, Any]:
    all_times = np.concatenate(pooled_times) if pooled_times else np.empty(0)
    all_channels = (
        np.concatenate(pooled_channels) if pooled_channels else np.empty(0, dtype=np.int64)
    )
    aggregate = _summarize_hits(
        all_times,
        all_channels,
        nphotons * len(run_records),
        num_channels,
        time_edges,
    )
    elapsed = np.asarray([record["elapsed_seconds"] for record in run_records])
    per_run_rate = nphotons / elapsed
    p95_elapsed = float(np.quantile(elapsed, 0.95))
    aggregate.update(
        elapsed_seconds_total=float(np.sum(elapsed)),
        elapsed_seconds_median=float(np.median(elapsed)),
        elapsed_seconds_p95=p95_elapsed,
        photons_per_second_sustained=float(nphotons * len(run_records) / np.sum(elapsed)),
        photons_per_second_median=float(np.median(per_run_rate)),
        photons_per_second_p95_latency=float(nphotons / p95_elapsed),
    )
    return {
        "backend": backend,
        "runs": run_records,
        "aggregate": aggregate,
    }


def _step_diagnostics(
    step_counts: np.ndarray,
    detected_step_counts: np.ndarray,
    max_steps: int,
) -> dict[str, Any]:
    """Summarize long-history tails without retaining photon state in JSON."""

    step_counts = np.asarray(step_counts, dtype=np.int64).reshape(-1)
    detected_step_counts = np.asarray(detected_step_counts, dtype=np.int64).reshape(-1)
    thresholds = (64, 100, 128)
    detected_quantiles = (
        np.quantile(detected_step_counts, DEFAULT_QUANTILES)
        if detected_step_counts.size
        else np.full(DEFAULT_QUANTILES.shape, np.nan)
    )
    tails = {}
    for threshold in thresholds:
        all_after = int(np.count_nonzero(step_counts > threshold))
        detected_after = int(np.count_nonzero(detected_step_counts > threshold))
        tails[str(threshold)] = {
            "all_photons_count_after": all_after,
            "all_photons_fraction_after": float(
                all_after / step_counts.size if step_counts.size else 0.0
            ),
            "detections_count_after": detected_after,
            "detections_fraction_after": float(
                detected_after / detected_step_counts.size
                if detected_step_counts.size
                else 0.0
            ),
        }
    at_limit = int(np.count_nonzero(step_counts >= max_steps))
    return {
        "photons": int(step_counts.size),
        "detections": int(detected_step_counts.size),
        "all_step_count_mean": float(np.mean(step_counts)) if step_counts.size else 0.0,
        "all_step_count_max": int(np.max(step_counts)) if step_counts.size else 0,
        "detection_step_count_mean": (
            float(np.mean(detected_step_counts)) if detected_step_counts.size else None
        ),
        "detection_step_count_max": (
            int(np.max(detected_step_counts)) if detected_step_counts.size else None
        ),
        "detection_step_quantiles": [
            _finite_float(value) for value in detected_quantiles
        ],
        "tails_strictly_after_step": tails,
        "photons_at_max_steps": at_limit,
        "fraction_at_max_steps": float(
            at_limit / step_counts.size if step_counts.size else 0.0
        ),
    }


def _aggregate_step_diagnostics(
    run_records: list[dict[str, Any]],
    pooled_detection_steps: list[np.ndarray],
    max_steps: int,
) -> dict[str, Any]:
    diagnostics = [record["step_diagnostics"] for record in run_records]
    photons = sum(item["photons"] for item in diagnostics)
    detections = sum(item["detections"] for item in diagnostics)
    all_step_sum = sum(
        item["all_step_count_mean"] * item["photons"] for item in diagnostics
    )
    detection_step_sum = sum(
        (item["detection_step_count_mean"] or 0.0) * item["detections"]
        for item in diagnostics
    )
    detected_steps = (
        np.concatenate(pooled_detection_steps)
        if pooled_detection_steps
        else np.empty(0, dtype=np.int64)
    )
    quantiles = (
        np.quantile(detected_steps, DEFAULT_QUANTILES)
        if detected_steps.size
        else np.full(DEFAULT_QUANTILES.shape, np.nan)
    )
    tails = {}
    for threshold in (64, 100, 128):
        entries = [
            item["tails_strictly_after_step"][str(threshold)]
            for item in diagnostics
        ]
        all_after = sum(item["all_photons_count_after"] for item in entries)
        detected_after = sum(item["detections_count_after"] for item in entries)
        tails[str(threshold)] = {
            "all_photons_count_after": int(all_after),
            "all_photons_fraction_after": float(all_after / photons if photons else 0.0),
            "detections_count_after": int(detected_after),
            "detections_fraction_after": float(
                detected_after / detections if detections else 0.0
            ),
        }
    at_limit = sum(item["photons_at_max_steps"] for item in diagnostics)
    return {
        "photons": int(photons),
        "detections": int(detections),
        "all_step_count_mean": float(all_step_sum / photons if photons else 0.0),
        "all_step_count_max": max(
            (item["all_step_count_max"] for item in diagnostics), default=0
        ),
        "detection_step_count_mean": (
            float(detection_step_sum / detections) if detections else None
        ),
        "detection_step_count_max": (
            max(
                item["detection_step_count_max"]
                for item in diagnostics
                if item["detection_step_count_max"] is not None
            )
            if detections
            else None
        ),
        "detection_step_quantiles": [_finite_float(value) for value in quantiles],
        "tails_strictly_after_step": tails,
        "photons_at_max_steps": int(at_limit),
        "fraction_at_max_steps": float(at_limit / photons if photons else 0.0),
        "max_steps": int(max_steps),
    }


def _run_chroma(args: argparse.Namespace) -> dict[str, Any]:
    """Run all Chroma replicas under one isolated PyCUDA context."""

    os.environ.setdefault("PYCUDA_CACHE_DIR", "/tmp/chroma-pycuda-cache")
    Path(os.environ["PYCUDA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    from chroma.sim import Simulation
    from chroma_lar.geometry import build_detector_from_config

    # NumPy 2 removed np.linalg.linalg, but Chroma's profile offset fallback
    # catches its LinAlgError through that old alias.  Keep the compatibility
    # shim scoped to reference geometry construction.
    had_linalg_alias = hasattr(np.linalg, "linalg")
    previous_linalg_alias = getattr(np.linalg, "linalg", None)
    if not had_linalg_alias:
        np.linalg.linalg = np.linalg  # type: ignore[attr-defined]
    try:
        detector = build_detector_from_config(
            "detector_config_reflect_reflect3wires",
            flatten=True,
            include_wires=True,
            include_active=True,
            include_cathode=True,
            include_cavity=True,
        )
    finally:
        if had_linalg_alias:
            np.linalg.linalg = previous_linalg_alias  # type: ignore[attr-defined]
        else:
            delattr(np.linalg, "linalg")
    simulation = Simulation(detector, seed=int(args.seed), photon_tracking=False)

    warm_count = min(args.nphotons, args.warmup_photons)
    if warm_count:
        warm_source = _source_photons(
            warm_count, args.center, args.voxel_size, args.seed ^ 0x61C88647
        )
        list(
            simulation.simulate(
                warm_source,
                keep_photons_beg=False,
                keep_photons_end=False,
                keep_hits=False,
                keep_flat_hits=False,
                run_daq=False,
                max_steps=args.max_steps,
                photons_per_batch=warm_count,
            )
        )

    edges = np.linspace(0.0, args.time_max, args.time_bins + 1, dtype=np.float64)
    records: list[dict[str, Any]] = []
    pooled_times: list[np.ndarray] = []
    pooled_channels: list[np.ndarray] = []
    for replicate in range(args.replicates):
        run_seed = int(args.seed + replicate * 0x9E3779B1) & 0xFFFFFFFF
        started = time.perf_counter()
        time_chunks: list[np.ndarray] = []
        channel_chunks: list[np.ndarray] = []
        for first, count in _work_batches(
            args.nphotons, args.work_batch_photons
        ):
            source = _source_photons(
                count,
                args.center,
                args.voxel_size,
                _batch_seed(run_seed, first, args.work_batch_photons),
            )
            events = simulation.simulate(
                source,
                keep_photons_beg=False,
                keep_photons_end=False,
                keep_hits=False,
                keep_flat_hits=True,
                run_daq=False,
                max_steps=args.max_steps,
                photons_per_batch=count,
            )
            event = next(events)
            try:
                next(events)
            except StopIteration:
                pass
            else:
                raise RuntimeError("expected exactly one Chroma event per batch")
            hits = event.flat_hits
            # Owning compact copies let the full Chroma photon record and its
            # GPU allocation die before the next work batch is constructed.
            time_chunks.append(np.asarray(hits.t).copy())
            channel_chunks.append(np.asarray(hits.channel).copy())
            del source, events, event, hits
        times = (
            time_chunks[0]
            if len(time_chunks) == 1
            else np.concatenate(time_chunks)
            if time_chunks
            else np.empty(0, dtype=np.float32)
        )
        channels = (
            channel_chunks[0]
            if len(channel_chunks) == 1
            else np.concatenate(channel_chunks)
            if channel_chunks
            else np.empty(0, dtype=np.int32)
        )
        elapsed = time.perf_counter() - started
        summary = _summarize_hits(
            times, channels, args.nphotons, args.num_channels, edges
        )
        summary.update(
            replicate=int(replicate),
            seed=int(run_seed),
            elapsed_seconds=float(elapsed),
            photons_per_second=float(args.nphotons / elapsed),
        )
        records.append(summary)
        pooled_times.append(times)
        pooled_channels.append(channels)
        del time_chunks, channel_chunks
        gc.collect()

    result = _aggregate_backend(
        "chroma",
        records,
        pooled_times,
        pooled_channels,
        args.nphotons,
        args.num_channels,
        edges,
    )
    import pycuda
    import pycuda.driver as cuda

    result["environment"] = {
        "device": simulation.context.get_device().name(),
        "numpy": np.__version__,
        "pycuda": pycuda.VERSION_TEXT,
        "cuda_driver_version": int(cuda.get_driver_version()),
        "runtime": "PyCUDA Chroma reference",
    }
    del simulation, detector
    gc.collect()
    return result


def _run_triton(args: argparse.Namespace) -> dict[str, Any]:
    """Run all specialized-backend replicas in the Torch primary context."""

    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    simulation = Reflect3WiresTritonSimulation(
        tile_size=args.tile_size,
        history_length=args.history_length,
        block_size=args.block_size,
        legacy_specular_reflection=args.legacy_specular_reflection,
    )
    warm_count = min(args.nphotons, args.warmup_photons)
    if warm_count:
        simulation.simulate(
            warm_count,
            args.center,
            voxel_size=args.voxel_size,
            wavelength=TARGET_WAVELENGTH_NM,
            seed=args.seed ^ 0x61C88647,
            max_steps=args.max_steps,
        )

    edges = np.linspace(0.0, args.time_max, args.time_bins + 1, dtype=np.float64)
    records: list[dict[str, Any]] = []
    pooled_times: list[np.ndarray] = []
    pooled_channels: list[np.ndarray] = []
    pooled_detection_steps: list[np.ndarray] = []
    for replicate in range(args.replicates):
        run_seed = int(args.seed + replicate * 0x9E3779B1) & 0xFFFFFFFF
        started = time.perf_counter()
        result = simulation.simulate(
            args.nphotons,
            args.center,
            voxel_size=args.voxel_size,
            wavelength=TARGET_WAVELENGTH_NM,
            seed=run_seed,
            max_steps=args.max_steps,
            keep_final_states=not args.skip_step_diagnostics,
        )
        # The acceptance target is generated photons through consumable compact
        # output, so device-to-host transfer is included in elapsed time.
        times, channels = result.flat_hits.to_numpy()
        elapsed = time.perf_counter() - started
        # Diagnostics are intentionally outside the throughput interval.  They
        # require final states only in this validation path, never production.
        all_step_chunks = []
        detection_step_chunks = []
        for state in result.final_states or ():
            all_step_chunks.append(state[9].detach().cpu().numpy())
            detected = state[8] >= 0
            detection_step_chunks.append(
                state[9][detected].detach().cpu().numpy()
            )
        all_steps = (
            np.concatenate(all_step_chunks)
            if all_step_chunks
            else np.empty(0, dtype=np.int64)
        )
        detection_steps = (
            np.concatenate(detection_step_chunks)
            if detection_step_chunks
            else np.empty(0, dtype=np.int64)
        )
        summary = _summarize_hits(
            times, channels, args.nphotons, args.num_channels, edges
        )
        summary.update(
            replicate=int(replicate),
            seed=int(run_seed),
            elapsed_seconds=float(elapsed),
            photons_per_second=float(args.nphotons / elapsed),
            boundary_rounds=int(result.stats.boundary_rounds),
            boundary_events=int(result.stats.boundary_events),
            dense_rounds=int(result.stats.dense_rounds),
            reservoir_rounds=int(result.stats.reservoir_rounds),
            reservoir_photons=int(result.stats.reservoir_photons),
            tiles=int(result.stats.tiles),
            tile_capacity=int(simulation.last_tile_plan.tile_capacity),
            tile_available_bytes=(
                None
                if simulation.last_tile_plan.available_bytes is None
                else int(simulation.last_tile_plan.available_bytes)
            ),
            tile_estimated_peak_bytes=int(
                simulation.last_tile_plan.estimated_peak_bytes
            ),
        )
        if not args.skip_step_diagnostics:
            summary["step_diagnostics"] = _step_diagnostics(
                all_steps, detection_steps, args.max_steps
            )
        records.append(summary)
        pooled_times.append(np.asarray(times, dtype=np.float64))
        pooled_channels.append(np.asarray(channels, dtype=np.int64))
        pooled_detection_steps.append(detection_steps)

    backend_result = _aggregate_backend(
        "triton",
        records,
        pooled_times,
        pooled_channels,
        args.nphotons,
        args.num_channels,
        edges,
    )
    if not args.skip_step_diagnostics:
        backend_result["aggregate"]["step_diagnostics"] = _aggregate_step_diagnostics(
            records, pooled_detection_steps, args.max_steps
        )
    import torch
    import triton

    backend_result["environment"] = {
        "device": torch.cuda.get_device_name(simulation.device),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "runtime": "Torch/Triton specialized backend",
        "legacy_specular_reflection": bool(
            simulation.legacy_specular_reflection
        ),
    }
    return backend_result


def _array(summary: dict[str, Any], key: str, dtype=np.float64) -> np.ndarray:
    values = summary[key]
    return np.asarray(
        [np.nan if value is None else value for value in values], dtype=dtype
    )


def _total_variation(left: np.ndarray, right: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(left - right)))


def _metric_differences(
    left: dict[str, Any], right: dict[str, Any]
) -> tuple[dict[str, float | None], np.ndarray]:
    nl = float(left["nphotons"])
    nr = float(right["nphotons"])
    cl = _array(left, "channel_counts")
    cr = _array(right, "channel_counts")
    rate_l, rate_r = cl / nl, cr / nr
    rate_delta = rate_l - rate_r

    hl, hr = float(left["hit_count"]), float(right["hit_count"])
    channel_l = cl / hl if hl else np.zeros_like(cl)
    channel_r = cr / hr if hr else np.zeros_like(cr)

    pooled_rate = (cl + cr) / (nl + nr)
    pull_sigma = np.sqrt(
        np.maximum(pooled_rate * (1.0 - pooled_rate) * (1.0 / nl + 1.0 / nr), 1.0e-30)
    )
    max_pull = float(np.max(np.abs(rate_delta) / pull_sigma))

    ql = _array(left, "time_quantiles_ns")
    qr = _array(right, "time_quantiles_ns")
    qdelta = np.abs(ql - qr)
    finite_q = np.isfinite(qdelta)

    shape_l = tuple(left["joint_histogram_shape"])
    shape_r = tuple(right["joint_histogram_shape"])
    if shape_l != shape_r:
        raise ValueError("histogram shapes differ")
    jl = _array(left, "joint_histogram").reshape(shape_l)
    jr = _array(right, "joint_histogram").reshape(shape_r)
    time_l, time_r = np.sum(jl, axis=0), np.sum(jr, axis=0)
    time_probability_l = time_l / hl if hl else np.zeros_like(time_l)
    time_probability_r = time_r / hr if hr else np.zeros_like(time_r)
    ks = float(
        np.max(np.abs(np.cumsum(time_probability_l) - np.cumsum(time_probability_r)))
    )
    joint_per_photon_l, joint_per_photon_r = jl / nl, jr / nr
    joint_shape_l = jl / hl if hl else np.zeros_like(jl)
    joint_shape_r = jr / hr if hr else np.zeros_like(jr)

    metrics: dict[str, float | None] = {
        "hit_fraction_abs": abs(float(left["hit_fraction"]) - float(right["hit_fraction"])),
        "per_channel_rate_rmse": float(np.sqrt(np.mean(rate_delta * rate_delta))),
        "per_channel_rate_max_abs": float(np.max(np.abs(rate_delta))),
        "per_channel_max_pull": max_pull,
        "channel_distribution_tv": _total_variation(channel_l, channel_r),
        "time_quantile_max_abs_ns": (
            float(np.max(qdelta[finite_q])) if np.any(finite_q) else None
        ),
        "time_quantile_rmse_ns": (
            float(np.sqrt(np.mean(qdelta[finite_q] ** 2))) if np.any(finite_q) else None
        ),
        "time_cdf_ks_binned": ks,
        "channel_time_tv_per_photon": _total_variation(
            joint_per_photon_l, joint_per_photon_r
        ),
        "channel_time_shape_tv": _total_variation(joint_shape_l, joint_shape_r),
    }
    return metrics, qdelta


def _categorical_tv_floor(probability: np.ndarray, samples: float) -> float:
    if samples <= 0.0:
        return 1.0
    # Expected TV between two multinomial samples under a normal approximation,
    # doubled to make this a conservative finite-sample floor.  The measured
    # Chroma pair envelope remains the primary tolerance whenever it is larger.
    expected = np.sum(
        np.sqrt(np.maximum(probability * (1.0 - probability), 0.0) / (np.pi * samples))
    )
    return float(min(1.0, 2.0 * expected))


def _analytic_floors(chroma: dict[str, Any], nphotons: int) -> dict[str, float]:
    aggregate = chroma["aggregate"]
    p_hit = float(aggregate["hit_fraction"])
    counts = _array(aggregate, "channel_counts")
    channel_rate = counts / float(aggregate["nphotons"])
    sigma_hit = math.sqrt(max(2.0 * p_hit * (1.0 - p_hit) / nphotons, 1.0 / nphotons**2))
    channel_sigma = np.sqrt(
        np.maximum(2.0 * channel_rate * (1.0 - channel_rate) / nphotons, 1.0 / nphotons**2)
    )
    hits_per_run = max(p_hit * nphotons, 1.0)
    channel_probability = counts / max(float(np.sum(counts)), 1.0)
    joint = _array(aggregate, "joint_histogram")
    joint_probability_hit = joint / max(float(np.sum(joint)), 1.0)
    joint_probability_photon = joint / float(aggregate["nphotons"])
    ks_floor = math.sqrt(
        max(0.0, -0.5 * math.log(5.0e-5) * (2.0 / hits_per_run))
    )
    return {
        "hit_fraction_abs": 5.0 * sigma_hit,
        "per_channel_rate_rmse": 5.0 * float(np.sqrt(np.mean(channel_sigma**2))),
        "per_channel_rate_max_abs": 5.0 * float(np.max(channel_sigma)),
        "per_channel_max_pull": 8.0,
        "channel_distribution_tv": _categorical_tv_floor(
            channel_probability, hits_per_run
        ),
        "time_quantile_max_abs_ns": 0.05,
        "time_quantile_rmse_ns": 0.05,
        "time_cdf_ks_binned": min(1.0, ks_floor),
        "channel_time_tv_per_photon": _categorical_tv_floor(
            joint_probability_photon, nphotons
        ),
        "channel_time_shape_tv": _categorical_tv_floor(
            joint_probability_hit, hits_per_run
        ),
    }


def _quantile_noise_floors(chroma: dict[str, Any]) -> np.ndarray:
    """Return conservative two-sample standard-error floors for quantiles.

    The maximum of only three pairwise Chroma differences is a noisy estimate
    of quantile uncertainty and can be accidentally tiny at an individual
    probability.  For a sample quantile, ``var(q_p) ~= p(1-p)/(N f(q_p)^2)``.
    Estimate the local density from adjacent reported quantiles and use five
    standard deviations for the difference of two equally sized ensembles.
    The empirical Chroma envelope remains authoritative whenever it is larger.
    """

    aggregate = chroma["aggregate"]
    quantiles = _array(aggregate, "time_quantiles_ns")
    probabilities = DEFAULT_QUANTILES
    samples = max(float(aggregate["hit_count"]), 1.0)
    density = np.zeros_like(probabilities)
    for index in range(probabilities.size):
        left = max(0, index - 1)
        right = min(probabilities.size - 1, index + 1)
        if left == right:
            continue
        delta_time = quantiles[right] - quantiles[left]
        delta_probability = probabilities[right] - probabilities[left]
        if np.isfinite(delta_time) and delta_time > 0.0:
            density[index] = delta_probability / delta_time

    variance = np.divide(
        2.0 * probabilities * (1.0 - probabilities),
        samples * density * density,
        out=np.full_like(probabilities, np.inf),
        where=density > 0.0,
    )
    return 5.0 * np.sqrt(variance)


def _calibrate(
    chroma: dict[str, Any], noise_factor: float, nphotons: int
) -> dict[str, Any]:
    noise: dict[str, list[float]] = {key: [] for key in METRIC_LABELS}
    quantile_noise: list[np.ndarray] = []
    for left, right in itertools.combinations(chroma["runs"], 2):
        metrics, qdelta = _metric_differences(left, right)
        for key, value in metrics.items():
            if value is not None and math.isfinite(value):
                noise[key].append(float(value))
        quantile_noise.append(qdelta)

    floors = _analytic_floors(chroma, nphotons)
    tolerances = {
        key: float(
            max(
                floors[key],
                noise_factor * max(values) if values else 0.0,
            )
        )
        for key, values in noise.items()
    }
    if quantile_noise:
        stacked = np.stack(quantile_noise)
        quantile_max = np.zeros(DEFAULT_QUANTILES.shape, dtype=np.float64)
        for index in range(DEFAULT_QUANTILES.size):
            finite = stacked[:, index][np.isfinite(stacked[:, index])]
            if finite.size:
                quantile_max[index] = np.max(finite)
    else:
        quantile_max = np.zeros(DEFAULT_QUANTILES.shape)
    quantile_floors = _quantile_noise_floors(chroma)
    quantile_tolerance = np.maximum(
        quantile_floors, noise_factor * quantile_max
    )
    return {
        "method": (
            "max pairwise Chroma difference times noise_factor, with analytic "
            "finite-sample floors"
        ),
        "noise_factor": float(noise_factor),
        "pair_count": int(len(quantile_noise)),
        "pairwise_noise": noise,
        "analytic_floors": floors,
        "analytic_quantile_floors_ns": quantile_floors.tolist(),
        "tolerances": tolerances,
        "time_quantile_tolerances_ns": quantile_tolerance.tolist(),
    }


def _compare_backends(
    chroma: dict[str, Any],
    triton_result: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    metrics, qdelta = _metric_differences(
        chroma["aggregate"], triton_result["aggregate"]
    )
    metric_results: dict[str, Any] = {}
    for key, value in metrics.items():
        tolerance = float(calibration["tolerances"][key])
        passed = value is not None and math.isfinite(value) and value <= tolerance
        metric_results[key] = {
            "label": METRIC_LABELS[key],
            "value": value,
            "tolerance": tolerance,
            "gate": key in GATED_METRICS,
            "pass": bool(passed),
        }

    q_tolerance = np.asarray(
        calibration["time_quantile_tolerances_ns"], dtype=np.float64
    )
    q_pass = np.isfinite(qdelta) & (qdelta <= q_tolerance)
    quantile_detail = {
        "probabilities": DEFAULT_QUANTILES.tolist(),
        "chroma_ns": chroma["aggregate"]["time_quantiles_ns"],
        "triton_ns": triton_result["aggregate"]["time_quantiles_ns"],
        "absolute_difference_ns": [_finite_float(x) for x in qdelta],
        "tolerance_ns": q_tolerance.tolist(),
        "pass": q_pass.tolist(),
    }
    invalid_ok = (
        chroma["aggregate"]["invalid_hit_count"] == 0
        and triton_result["aggregate"]["invalid_hit_count"] == 0
    )
    enough_hits = (
        chroma["aggregate"]["hit_count"] > 0
        and triton_result["aggregate"]["hit_count"] > 0
    )
    overall = (
        invalid_ok
        and enough_hits
        and bool(np.all(q_pass))
        and all(
            metric_results[key]["pass"]
            for key in GATED_METRICS
        )
    )
    return {
        "metrics": metric_results,
        "time_quantiles": quantile_detail,
        "no_invalid_hits": bool(invalid_ok),
        "both_ensembles_have_hits": bool(enough_hits),
        "correctness_pass": bool(overall),
    }


def _worker_command(
    args: argparse.Namespace, backend: str, output: Path
) -> list[str]:
    worker_arguments = [
        str(Path(__file__).resolve()),
        "--backend",
        backend,
        "--nphotons",
        str(args.nphotons),
        "--center",
        *(str(value) for value in args.center),
        "--voxel-size",
        str(args.voxel_size),
        "--seed",
        str(args.seed),
        "--replicates",
        str(args.replicates),
        "--num-channels",
        str(args.num_channels),
        "--time-max",
        str(args.time_max),
        "--time-bins",
        str(args.time_bins),
        "--max-steps",
        str(args.max_steps),
        "--tile-size",
        "auto" if args.tile_size is None else str(args.tile_size),
        "--history-length",
        str(args.history_length),
        "--block-size",
        str(args.block_size),
        "--warmup-photons",
        str(args.warmup_photons),
        "--work-batch-photons",
        str(args.work_batch_photons),
        "--_worker-output",
        str(output),
    ]
    if args.skip_step_diagnostics:
        worker_arguments.append("--skip-step-diagnostics")
    if args.legacy_specular_reflection:
        worker_arguments.append("--legacy-specular-reflection")
    use_container = backend == "chroma" and (
        args.chroma_runtime == "container"
        or (
            args.chroma_runtime == "auto"
            and Path(args.chroma_container).is_file()
        )
    )
    if use_container:
        image = Path(args.chroma_container).expanduser().resolve()
        if not image.is_file():
            raise RuntimeError("Chroma container does not exist: %s" % image)
        python_path = "%s:%s" % (
            REPOSITORY / "chroma-lite",
            REPOSITORY / "chroma-lar",
        )
        return [
            "singularity",
            "exec",
            "--nv",
            "-B",
            "/sdf:/sdf",
            "-B",
            "/tmp:/tmp",
            "--pwd",
            str(REPOSITORY),
            str(image),
            "env",
            "PYTHONPATH=" + python_path,
            "PYTHONNOUSERSITE=1",
            "PYCUDA_CACHE_DIR=/tmp/chroma-pycuda-cache",
            "TMPDIR=/tmp",
            "python",
            *worker_arguments,
        ]
    if backend == "chroma" and args.chroma_runtime == "container":
        raise RuntimeError("--chroma-runtime container requires a valid image")
    return [sys.executable, *worker_arguments]


def _run_isolated_backend(
    args: argparse.Namespace, backend: str, output: Path
) -> dict[str, Any]:
    completed = subprocess.run(
        _worker_command(args, backend, output),
        cwd=str(REPOSITORY),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 or not output.exists():
        diagnostic = (completed.stdout + "\n" + completed.stderr)[-12_000:]
        raise RuntimeError(
            "%s worker failed with status %d:\n%s"
            % (backend, completed.returncode, diagnostic)
        )
    with output.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _settings(args: argparse.Namespace) -> dict[str, Any]:
    tile_policy = "auto" if args.tile_size is None else "explicit"
    return {
        "detector_config": "detector_config_reflect_reflect3wires",
        "wavelength_nm": TARGET_WAVELENGTH_NM,
        "nphotons_per_replicate": int(args.nphotons),
        "replicates": int(args.replicates),
        "center_mm": [float(value) for value in args.center],
        "voxel_size_mm": float(args.voxel_size),
        "seed": int(args.seed),
        "num_channels": int(args.num_channels),
        "time_histogram": {
            "minimum_ns": 0.0,
            "maximum_ns": float(args.time_max),
            "regular_bins": int(args.time_bins),
            "includes_underflow_and_overflow": True,
        },
        "time_quantile_probabilities": DEFAULT_QUANTILES.tolist(),
        "max_steps": int(args.max_steps),
        "tile_size": "auto" if args.tile_size is None else int(args.tile_size),
        "tile_size_policy": tile_policy,
        "tile_size_value": (
            None
            if args.tile_size is None
            else min(int(args.nphotons), int(args.tile_size))
        ),
        "work_batch_photons": int(args.work_batch_photons),
        "throughput_target_photons_per_second": TARGET_PHOTONS_PER_SECOND,
        "timing_scope": "source generation through host-readable compact (time, channel) hits",
        "step_diagnostics": "disabled" if args.skip_step_diagnostics else "enabled outside timing",
        "legacy_specular_reflection": bool(args.legacy_specular_reflection),
        "chroma_runtime": str(args.chroma_runtime),
        "chroma_container": str(args.chroma_container),
        "chroma_reference_report": (
            None
            if args.chroma_reference_report is None
            else str(Path(args.chroma_reference_report).expanduser().resolve())
        ),
    }


def _load_chroma_reference(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a previously measured Chroma ensemble after strict provenance checks."""

    path = Path(args.chroma_reference_report).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    settings = report.get("settings", {})
    expected = {
        "detector_config": "detector_config_reflect_reflect3wires",
        "wavelength_nm": TARGET_WAVELENGTH_NM,
        "nphotons_per_replicate": int(args.nphotons),
        "replicates": int(args.replicates),
        "center_mm": [float(value) for value in args.center],
        "voxel_size_mm": float(args.voxel_size),
        "seed": int(args.seed),
        "num_channels": int(args.num_channels),
        "max_steps": int(args.max_steps),
        "work_batch_photons": int(args.work_batch_photons),
    }
    mismatches = {
        key: (settings.get(key), value)
        for key, value in expected.items()
        if settings.get(key) != value
    }
    histogram = settings.get("time_histogram", {})
    histogram_expected = {
        "minimum_ns": 0.0,
        "maximum_ns": float(args.time_max),
        "regular_bins": int(args.time_bins),
        "includes_underflow_and_overflow": True,
    }
    for key, value in histogram_expected.items():
        if histogram.get(key) != value:
            mismatches["time_histogram." + key] = (histogram.get(key), value)
    if mismatches:
        detail = ", ".join(
            "%s=%r (expected %r)" % (key, actual, expected_value)
            for key, (actual, expected_value) in sorted(mismatches.items())
        )
        raise ValueError("incompatible Chroma reference report: " + detail)
    try:
        chroma = report["backends"]["chroma"]
    except (KeyError, TypeError) as exc:
        raise ValueError("reference report contains no Chroma backend") from exc
    provenance = {
        "path": str(path),
        "schema_version": report.get("schema_version"),
        "chroma_environment": chroma.get("environment", {}),
    }
    return chroma, provenance


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    requested = ["chroma", "triton"] if args.backend == "both" else [args.backend]
    results: dict[str, Any] = {}
    reference_provenance = None
    if args.chroma_reference_report is not None:
        results["chroma"], reference_provenance = _load_chroma_reference(args)
        requested.remove("chroma")
    # Use /tmp explicitly because site TMPDIR may point at /lscratch, which is
    # not necessarily mounted inside the reference container.
    with tempfile.TemporaryDirectory(
        prefix="validate-triton-", dir="/tmp"
    ) as directory:
        for backend in requested:
            output = Path(directory) / (backend + ".json")
            results[backend] = _run_isolated_backend(args, backend, output)

    report: dict[str, Any] = {
        "schema_version": 1,
        "settings": _settings(args),
        "backends": results,
    }
    if reference_provenance is not None:
        report["chroma_reference_reuse"] = reference_provenance
    performance: dict[str, Any] = {
        "target_photons_per_second": TARGET_PHOTONS_PER_SECOND
    }
    for backend, result in results.items():
        rate = float(result["aggregate"]["photons_per_second_p95_latency"])
        performance[backend] = {
            "p95_latency_photons_per_second": rate,
            "meets_5m_target": bool(rate >= TARGET_PHOTONS_PER_SECOND),
        }
    if "chroma" in results and "triton" in results:
        performance["triton_speedup_over_chroma"] = float(
            performance["triton"]["p95_latency_photons_per_second"]
            / performance["chroma"]["p95_latency_photons_per_second"]
        )
        calibration = _calibrate(
            results["chroma"], args.noise_factor, args.nphotons
        )
        report["calibration"] = calibration
        report["comparison"] = _compare_backends(
            results["chroma"], results["triton"], calibration
        )
    report["performance"] = performance
    correctness_ok = report.get("comparison", {}).get("correctness_pass", True)
    triton_ok = performance.get("triton", {}).get("meets_5m_target", True)
    report["overall_pass"] = bool(correctness_ok and triton_ok)
    return report


def _human_report(report: dict[str, Any]) -> str:
    lines = [
        "Reflect3Wires 450-nm validation",
        "  photons: {n:,} x {r} replicas; center: {c}; voxel: {v:g} mm".format(
            n=report["settings"]["nphotons_per_replicate"],
            r=report["settings"]["replicates"],
            c=report["settings"]["center_mm"],
            v=report["settings"]["voxel_size_mm"],
        ),
    ]
    for backend, result in report["backends"].items():
        aggregate = result["aggregate"]
        lines.append(
            "  {name:7s}: {rate:,.3f} M photons/s (p95 latency), "
            "hit fraction {hit:.6f}, {count:,} pooled hits".format(
                name=backend,
                rate=aggregate["photons_per_second_p95_latency"] / 1.0e6,
                hit=aggregate["hit_fraction"],
                count=aggregate["hit_count"],
            )
        )
        if backend == "triton" and "step_diagnostics" in aggregate:
            tails = aggregate["step_diagnostics"]["tails_strictly_after_step"]
            lines.append(
                "           detected after steps 64/100/128: "
                + "/".join(
                    "{:.4%}".format(tails[str(step)]["detections_fraction_after"])
                    for step in (64, 100, 128)
                )
            )
    if "comparison" in report:
        lines.append(
            "  Chroma noise calibration: {pairs} replicate pairs, factor {factor:g}".format(
                pairs=report["calibration"]["pair_count"],
                factor=report["calibration"]["noise_factor"],
            )
        )
        for key in GATED_METRICS:
            metric = report["comparison"]["metrics"][key]
            lines.append(
                "    {status:4s} {label}: {value:.6g} <= {tol:.6g}".format(
                    status="PASS" if metric["pass"] else "FAIL",
                    label=metric["label"],
                    value=metric["value"] if metric["value"] is not None else float("nan"),
                    tol=metric["tolerance"],
                )
            )
        lines.append(
            "  correctness: "
            + ("PASS" if report["comparison"]["correctness_pass"] else "FAIL")
        )
    if "triton" in report["performance"]:
        rate = report["performance"]["triton"]["p95_latency_photons_per_second"]
        lines.append(
            "  5M photons/s target: {status} ({rate:,.3f} M/s)".format(
                status="PASS" if rate >= TARGET_PHOTONS_PER_SECOND else "FAIL",
                rate=rate / 1.0e6,
            )
        )
    lines.append("  overall: " + ("PASS" if report["overall_pass"] else "FAIL"))
    return "\n".join(lines)


def _parse_tile_size(value: str) -> Optional[int]:
    """Parse a positive manual tile or the adaptive ``auto`` policy."""

    if value.strip().lower() == "auto":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "tile size must be a positive integer or 'auto'"
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "tile size must be a positive integer or 'auto'"
        )
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the specialized Triton simulation against Chroma ensembles."
    )
    parser.add_argument(
        "--backend", choices=("triton", "chroma", "both"), default="both"
    )
    parser.add_argument("--nphotons", type=int, default=1_000_000)
    parser.add_argument(
        "--center", type=float, nargs=3, metavar=("X", "Y", "Z"), default=(-1000.0, 0.0, 0.0)
    )
    parser.add_argument("--voxel-size", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument(
        "--json",
        nargs="?",
        const="-",
        default=None,
        metavar="PATH",
        help="emit JSON to stdout, or write it to PATH",
    )
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--noise-factor", type=float, default=1.5)
    parser.add_argument("--num-channels", type=int, default=162)
    parser.add_argument("--time-max", type=float, default=100.0)
    parser.add_argument("--time-bins", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--tile-size",
        type=_parse_tile_size,
        default=None,
        metavar="N|auto",
        help=(
            "wavefront size; auto (default) derives a conservative capacity "
            "from currently allocatable CUDA memory"
        ),
    )
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--legacy-specular-reflection",
        action="store_true",
        help=(
            "use Chroma's acos/Rodrigues surface reflection in Triton; "
            "intended only for compatibility-baseline measurements"
        ),
    )
    parser.add_argument("--warmup-photons", type=int, default=65_536)
    parser.add_argument(
        "--work-batch-photons",
        type=int,
        default=DEFAULT_WORK_BATCH_PHOTONS,
        help=(
            "maximum materialized Chroma source batch (default: 15M); "
            "Triton applies its memory-derived tile plan internally"
        ),
    )
    parser.add_argument(
        "--skip-step-diagnostics",
        action="store_true",
        help="do not retain Triton final states (recommended for large scaling runs)",
    )
    parser.add_argument(
        "--chroma-runtime",
        choices=("auto", "container", "local"),
        default="auto",
        help="run Chroma in its reference container when available (default: auto)",
    )
    parser.add_argument(
        "--chroma-container",
        default=str(DEFAULT_CHROMA_CONTAINER),
        help="reference Chroma Singularity/Apptainer image",
    )
    parser.add_argument(
        "--chroma-reference-report",
        default=None,
        metavar="PATH",
        help=(
            "reuse the Chroma ensemble from a compatible prior report; "
            "all source, detector, histogram, and replicate settings are checked"
        ),
    )
    parser.add_argument("--_worker-output", default=None, help=argparse.SUPPRESS)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.nphotons <= 0:
        parser.error("--nphotons must be positive")
    if args.work_batch_photons <= 0:
        parser.error("--work-batch-photons must be positive")
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    if args.backend == "both" and args.replicates < 2:
        parser.error("--backend both needs at least two Chroma replicas for calibration")
    if args.chroma_reference_report is not None and args.backend != "both":
        parser.error("--chroma-reference-report requires --backend both")
    if args.voxel_size < 0.0:
        parser.error("--voxel-size must be non-negative")
    if args.num_channels <= 0 or args.time_bins <= 0 or args.time_max <= 0.0:
        parser.error("histogram dimensions and --time-max must be positive")
    if args.max_steps <= 0 or args.history_length <= 0:
        parser.error("transport limits must be positive")
    if args.block_size <= 0 or args.block_size & (args.block_size - 1):
        parser.error("--block-size must be a positive power of two")
    if args.noise_factor < 1.0:
        parser.error("--noise-factor must be at least one")
    if args.backend in ("triton", "both") and args.center[0] >= 0.0:
        parser.error("the current reachable-half Triton specialization requires center X < 0")


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    if args._worker_output:
        if args.backend == "both":
            parser.error("internal worker accepts one backend")
        result = _run_chroma(args) if args.backend == "chroma" else _run_triton(args)
        with Path(args._worker_output).open("w", encoding="utf-8") as stream:
            json.dump(result, stream, allow_nan=False, separators=(",", ":"))
        return 0

    try:
        report = _build_report(args)
    except Exception as error:
        parser.exit(2, "validation failed to run: %s\n" % (error,))
    if args.json == "-":
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    elif args.json:
        output = Path(args.json).expanduser().resolve()
        with output.open("w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(_human_report(report))
        print("  JSON: " + str(output))
    else:
        print(_human_report(report))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
