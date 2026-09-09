#!/usr/bin/env python3
"""Benchmark the optimized Triton backend across photon populations.

Unlike the two-backend acceptance worker, this harness keeps one simulation
and its immutable scene/workspaces resident while the population changes.  It
therefore represents a long-lived Chroma service and lets the memory-derived
tile planner respond to the allocations that are actually resident.  Every
timed run includes source generation and compact-hit transfer to NumPy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for source_root in (ROOT / "chroma-lite", ROOT / "chroma-lar"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

DEFAULT_COUNTS = (
    500_000,
    1_000_000,
    2_000_000,
    3_000_000,
    5_000_000,
    7_500_000,
    10_000_000,
    12_500_000,
    15_000_000,
    30_000_000,
    60_000_000,
    120_000_000,
    180_000_000,
    240_000_000,
    300_000_000,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=DEFAULT_COUNTS)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument("--center", type=float, nargs=3, default=(-1000.0, 0.0, 0.0))
    parser.add_argument("--warmup-photons", type=int, default=131_072)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    counts = sorted(set(int(value) for value in args.counts))
    if not counts or counts[0] <= 0 or args.replicates < 2:
        parser.error("counts must be positive and replicates must be at least two")

    import torch
    import triton

    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    device_index = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(device_index)

    output = args.json.expanduser().resolve()
    payload = {
        "schema_version": 1,
        "backend": "triton",
        "device": device_properties.name,
        "device_index": int(device_index),
        "device_total_memory_bytes": int(device_properties.total_memory),
        "host": platform.node(),
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "triton": triton.__version__,
        },
        "timing_scope": "source generation through host-readable compact hits",
        "center_mm": [float(value) for value in args.center],
        "replicates": int(args.replicates),
        "points": [],
    }
    if args.resume and output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if int(payload.get("replicates", -1)) != args.replicates:
            parser.error("resume file uses a different replicate count")
    completed = {int(point["photons"]) for point in payload.get("points", ())}

    simulation = Reflect3WiresTritonSimulation()
    if args.warmup_photons:
        simulation.simulate(
            int(args.warmup_photons),
            args.center,
            seed=args.seed ^ 0x61C88647,
        ).flat_hits.to_numpy()

    for count in counts:
        if count in completed:
            print(f"resumed {count:,}", flush=True)
            continue
        runs = []
        for replicate in range(args.replicates):
            seed = (args.seed + replicate * 0x9E3779B1) & 0xFFFFFFFF
            started = time.perf_counter()
            result = simulation.simulate(count, args.center, seed=seed)
            result.flat_hits.to_numpy()
            elapsed = time.perf_counter() - started
            plan = simulation.last_tile_plan
            runs.append(
                {
                    "replicate": replicate,
                    "seed": seed,
                    "elapsed_seconds": elapsed,
                    "photons_per_second": count / elapsed,
                    "detections": result.stats.detections,
                    "boundary_rounds": result.stats.boundary_rounds,
                    "dense_rounds": result.stats.dense_rounds,
                    "reservoir_rounds": result.stats.reservoir_rounds,
                    "reservoir_photons": result.stats.reservoir_photons,
                    "boundary_events": result.stats.boundary_events,
                    "tiles": result.stats.tiles,
                    "tile_capacity": plan.tile_capacity,
                    "tile_available_bytes": plan.available_bytes,
                    "tile_estimated_peak_bytes": plan.estimated_peak_bytes,
                }
            )
        elapsed_values = np.asarray(
            [run["elapsed_seconds"] for run in runs], dtype=np.float64
        )
        run_rates = count / elapsed_values
        point = {
            "photons": count,
            "runs": runs,
            "p95_latency_photons_per_second": float(
                count / np.quantile(elapsed_values, 0.95)
            ),
            "median_photons_per_second": float(np.median(run_rates)),
            "sustained_photons_per_second": float(
                count * len(runs) / elapsed_values.sum()
            ),
            "minimum_run_photons_per_second": float(run_rates.min()),
            "maximum_run_photons_per_second": float(run_rates.max()),
        }
        payload.setdefault("points", []).append(point)
        payload["points"].sort(key=lambda item: int(item["photons"]))
        _write(output, payload)
        print(json.dumps(point, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
