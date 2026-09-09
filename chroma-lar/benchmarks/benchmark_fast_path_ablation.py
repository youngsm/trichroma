#!/usr/bin/env python3
"""Measure cumulative production optimizations with identical photon seeds."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np


CONFIGURATIONS = (
    (
        "monolithic_scan",
        dict(
            branch_specialized_boundary=False,
            portal_boundary=False,
            pmt_grid=False,
        ),
    ),
    (
        "branch_specialized",
        dict(
            branch_specialized_boundary=True,
            portal_boundary=False,
            pmt_grid=False,
        ),
    ),
    (
        "branch_plus_grid",
        dict(
            branch_specialized_boundary=True,
            portal_boundary=False,
            pmt_grid=True,
        ),
    ),
    (
        "branch_plus_portals",
        dict(
            branch_specialized_boundary=True,
            portal_boundary=True,
            pmt_grid=False,
        ),
    ),
    (
        "branch_grid_portals",
        dict(
            branch_specialized_boundary=True,
            portal_boundary=True,
            pmt_grid=True,
        ),
    ),
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nphotons", type=int, default=5_000_000)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--warmup-photons", type=int, default=131_072)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--history-epochs-per-poll", type=int, default=1)
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument("--json", type=Path, required=True)
    return parser


def _hit_signature(result):
    time_words = result.flat_hits.time.detach().cpu().numpy().view(np.uint32)
    channels = result.flat_hits.channel.detach().cpu().numpy().astype(np.int32)
    order = np.lexsort((time_words, channels))
    digest = hashlib.sha256()
    digest.update(channels[order].tobytes())
    digest.update(time_words[order].tobytes())
    return digest.hexdigest()


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.nphotons <= 0 or args.replicates < 2 or args.warmup_photons <= 0:
        raise SystemExit("positive counts and at least two replicates are required")
    import torch
    import triton
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    seeds = [
        (int(args.seed) + index * 0x9E3779B1) & 0xFFFFFFFF
        for index in range(args.replicates)
    ]
    records = []
    reference_signatures = None
    for name, switches in CONFIGURATIONS:
        simulation = Reflect3WiresTritonSimulation(
            tile_size=None,
            history_length=args.history_length,
            block_size=args.block_size,
            history_epochs_per_poll=args.history_epochs_per_poll,
            **switches,
        )
        simulation.simulate(
            min(args.nphotons, args.warmup_photons),
            (-1000.0, 0.0, 0.0),
            voxel_size=30.0,
            seed=seeds[0],
        )
        runs = []
        for replicate, seed in enumerate(seeds):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = simulation.simulate(
                args.nphotons,
                (-1000.0, 0.0, 0.0),
                voxel_size=30.0,
                seed=seed,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            runs.append(
                dict(
                    replicate=replicate,
                    seed=seed,
                    elapsed_seconds=elapsed,
                    photons_per_second=args.nphotons / elapsed,
                    detections=result.stats.detections,
                    boundary_events=result.stats.boundary_events,
                    boundary_rounds=result.stats.boundary_rounds,
                    portal_direct=int(simulation.last_portal_counts[0]),
                    portal_fallback=int(simulation.last_portal_counts[1]),
                    hit_signature=_hit_signature(result),
                )
            )
        seconds = [run["elapsed_seconds"] for run in runs]
        signatures = [run["hit_signature"] for run in runs]
        if reference_signatures is None:
            reference_signatures = signatures
        records.append(
            dict(
                name=name,
                switches=switches,
                runs=runs,
                median_photons_per_second=(
                    args.nphotons / statistics.median(seconds)
                ),
                p95_latency_photons_per_second=(
                    args.nphotons / float(np.quantile(seconds, 0.95))
                ),
                exact_hit_signatures_match_monolithic=(
                    signatures == reference_signatures
                ),
            )
        )
        print(json.dumps(records[-1], sort_keys=True), flush=True)

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = dict(
        schema_version=1,
        benchmark="reflect3wires cumulative fast-path ablation",
        command=sys.argv,
        configuration=dict(
            nphotons=args.nphotons,
            replicates=args.replicates,
            warmup_photons=args.warmup_photons,
            history_length=args.history_length,
            block_size=args.block_size,
            history_epochs_per_poll=args.history_epochs_per_poll,
            seeds=seeds,
        ),
        device=dict(
            name=properties.name,
            compute_capability=[properties.major, properties.minor],
            total_memory_bytes=properties.total_memory,
        ),
        software=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            torch_cuda=torch.version.cuda,
            triton=triton.__version__,
            numpy=np.__version__,
        ),
        host=platform.node(),
        cpu_thread_limits={
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        records=records,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
