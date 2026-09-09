#!/usr/bin/env python3
"""Measure detector-specialized scheduler policies on one CUDA device.

This is a focused optimization harness, not the physics acceptance benchmark.
It holds the photon population and seed fixed while sweeping the two policies
that control long-history work: device-resident collision epochs per host poll
and the dense-to-reservoir boundary-round cutoff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time


REPOSITORY = Path(__file__).resolve().parents[2]
for source_root in (REPOSITORY / "chroma-lite", REPOSITORY / "chroma-lar"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


def _positive_csv(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nphotons", type=int, default=5_000_000)
    parser.add_argument("--warmup-photons", type=int, default=131_072)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--history-epochs", type=_positive_csv, default=(1, 2, 4))
    parser.add_argument("--reservoir-rounds", type=_positive_csv, default=(64, 128))
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=15_000_000)
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument("--center", type=float, nargs=3, default=(-1000.0, 0.0, 0.0))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.nphotons <= 0 or args.replicates <= 0:
        parser.error("nphotons and replicates must be positive")

    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    records = []
    for epochs in args.history_epochs:
        for reservoir in args.reservoir_rounds:
            simulation = Reflect3WiresTritonSimulation(
                tile_size=args.tile_size,
                history_length=args.history_length,
                reservoir_rounds=reservoir,
                history_epochs_per_poll=epochs,
            )
            simulation.simulate(
                min(args.nphotons, args.warmup_photons),
                args.center,
                seed=args.seed ^ 0x61C88647,
            )
            runs = []
            for replicate in range(args.replicates):
                seed = (args.seed + replicate * 0x9E3779B1) & 0xFFFFFFFF
                started = time.perf_counter()
                result = simulation.simulate(args.nphotons, args.center, seed=seed)
                # Include compact-hit transfer, as the production benchmark does.
                result.flat_hits.to_numpy()
                elapsed = time.perf_counter() - started
                runs.append(
                    {
                        "seed": seed,
                        "elapsed_seconds": elapsed,
                        "photons_per_second": args.nphotons / elapsed,
                        "detections": result.stats.detections,
                        "boundary_rounds": result.stats.boundary_rounds,
                        "dense_rounds": result.stats.dense_rounds,
                        "reservoir_rounds": result.stats.reservoir_rounds,
                        "reservoir_photons": result.stats.reservoir_photons,
                        "boundary_events": result.stats.boundary_events,
                    }
                )
            rates = [run["photons_per_second"] for run in runs]
            record = {
                "history_epochs_per_poll": epochs,
                "reservoir_cutoff_rounds": reservoir,
                "median_photons_per_second": statistics.median(rates),
                "minimum_photons_per_second": min(rates),
                "runs": runs,
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)

    payload = {
        "nphotons": args.nphotons,
        "replicates": args.replicates,
        "history_length": args.history_length,
        "tile_size": args.tile_size,
        "records": records,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
