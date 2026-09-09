#!/usr/bin/env python3
"""A/B the synchronized and fixed-round device schedulers on identical seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np


def _positive_csv(value):
    try:
        result = tuple(dict.fromkeys(int(item) for item in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return result


def _pmt_programs_csv(value):
    result = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if item in ("uncapped", "full"):
            parsed = 0
        else:
            try:
                parsed = int(item)
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    "expected positive integers or 'uncapped'"
                ) from exc
            if parsed < 0:
                raise argparse.ArgumentTypeError(
                    "programs per SM cannot be negative"
                )
        if parsed not in result:
            result.append(parsed)
    if not result:
        raise argparse.ArgumentTypeError("PMT grid sweep cannot be empty")
    return tuple(result)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nphotons", type=int, default=5_000_000)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--warmup-photons", type=int, default=262_144)
    parser.add_argument("--round-batches", type=_positive_csv, default=(1, 2, 4, 8, 16))
    parser.add_argument(
        "--portal-batches",
        type=_positive_csv,
        default=(),
        help=(
            "optional device-round batches to repeat with the certified "
            "box-face portal enabled"
        ),
    )
    parser.add_argument(
        "--fused-portal-boundary",
        action="store_true",
        help=(
            "add fused direct-portal variants beside every materialized "
            "--portal-batches variant"
        ),
    )
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--history-epochs", type=int, default=2)
    parser.add_argument(
        "--fused-pmt",
        action="store_true",
        help=(
            "add a synchronized fused-PMT A/B record and use the experimental "
            "per-ray lattice/BLAS query for every device-scheduler variant"
        ),
    )
    parser.add_argument(
        "--fused-pmt-max-candidates",
        type=int,
        default=81,
        help=(
            "maximum certified lattice rectangle before the exact all-instance "
            "fallback; requires --fused-pmt (default: 81)"
        ),
    )
    parser.add_argument(
        "--fused-pmt-routing-diagnostics",
        action="store_true",
        help=(
            "collect certified/fallback/box-visit counters after each run; "
            "requires --fused-pmt and is excluded from timed synchronization"
        ),
    )
    parser.add_argument(
        "--fused-pmt-compact-union",
        action="store_true",
        help=(
            "retain the old initialize/union-compaction/indirection route for "
            "fused-PMT A/B tests (default fused benchmark route is direct)"
        ),
    )
    parser.add_argument(
        "--pmt-programs-per-sm",
        type=_pmt_programs_csv,
        default=(0,),
        help=(
            "device-count PMT traversal grid sweep; comma-separated positive "
            "integers and/or 'uncapped'"
        ),
    )
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
    if args.fused_pmt_max_candidates <= 0:
        raise SystemExit("--fused-pmt-max-candidates must be positive")
    if not args.fused_pmt and args.fused_pmt_max_candidates != 81:
        raise SystemExit(
            "--fused-pmt-max-candidates requires --fused-pmt"
        )
    if args.fused_pmt_routing_diagnostics and not args.fused_pmt:
        raise SystemExit(
            "--fused-pmt-routing-diagnostics requires --fused-pmt"
        )
    if args.fused_pmt_compact_union and not args.fused_pmt:
        raise SystemExit("--fused-pmt-compact-union requires --fused-pmt")
    if args.fused_portal_boundary and not args.portal_batches:
        raise SystemExit("--fused-portal-boundary requires --portal-batches")

    import torch
    import triton
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation
    from chroma_lar.triton_scene import instances as pmt_instances

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    seeds = tuple(
        (int(args.seed) + index * 0x9E3779B1) & 0xFFFFFFFF
        for index in range(args.replicates)
    )
    fused_suffix = (
        "_fused_pmt_c%d" % args.fused_pmt_max_candidates
        if args.fused_pmt
        else ""
    )
    configurations = [
        ("synchronized", False, False, False, False, 1, None)
    ]
    if args.fused_pmt:
        configurations.append(
            (
                "synchronized_fused_pmt",
                False,
                False,
                False,
                True,
                1,
                None,
            )
        )
    configurations += [
        (
            "device_batch_%d_pmt_%s%s"
            % (
                batch,
                "uncapped" if programs == 0 else programs,
                fused_suffix,
            ),
            True,
            False,
            False,
            bool(args.fused_pmt),
            batch,
            programs,
        )
        for batch in args.round_batches
        for programs in args.pmt_programs_per_sm
    ] + [
        (
            "device_portal_batch_%d_pmt_%s%s"
            % (
                batch,
                "uncapped" if programs == 0 else programs,
                fused_suffix,
            ),
            True,
            True,
            False,
            bool(args.fused_pmt),
            batch,
            programs,
        )
        for batch in args.portal_batches
        for programs in args.pmt_programs_per_sm
    ]
    if args.fused_portal_boundary:
        configurations += [
            (
                "device_portal_fused_boundary_batch_%d_pmt_%s%s"
                % (
                    batch,
                    "uncapped" if programs == 0 else programs,
                    fused_suffix,
                ),
                True,
                True,
                True,
                bool(args.fused_pmt),
                batch,
                programs,
            )
            for batch in args.portal_batches
            for programs in args.pmt_programs_per_sm
        ]
    records = []
    reference_signatures = None
    original_programs_per_sm = pmt_instances.DEVICE_TLAS_PROGRAMS_PER_SM
    for (
        name,
        device_scheduler,
        portal_boundary,
        fused_portal_boundary,
        fused_pmt,
        batch,
        pmt_programs_per_sm,
    ) in configurations:
        pmt_instances.DEVICE_TLAS_PROGRAMS_PER_SM = (
            original_programs_per_sm
            if pmt_programs_per_sm is None
            else pmt_programs_per_sm
        )
        simulation = Reflect3WiresTritonSimulation(
            tile_size=None,
            history_length=args.history_length,
            block_size=args.block_size,
            history_epochs_per_poll=args.history_epochs,
            branch_specialized_boundary=False,
            portal_boundary=portal_boundary,
            fused_portal_boundary=fused_portal_boundary,
            pmt_grid=False,
            fused_pmt=fused_pmt,
            fused_pmt_max_candidates=(
                args.fused_pmt_max_candidates if fused_pmt else 81
            ),
            fused_pmt_routing_diagnostics=(
                args.fused_pmt_routing_diagnostics and fused_pmt
            ),
            fused_pmt_compact_union=(
                args.fused_pmt_compact_union if fused_pmt else True
            ),
            device_scheduler=device_scheduler,
            device_round_batch=batch,
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
            routing_counts = simulation.fused_pmt_routing_snapshot()
            runs.append(
                dict(
                    replicate=replicate,
                    seed=seed,
                    elapsed_seconds=elapsed,
                    photons_per_second=args.nphotons / elapsed,
                    detections=result.stats.detections,
                    boundary_events=result.stats.boundary_events,
                    scheduler_rounds=result.stats.boundary_rounds,
                    scheduler_host_syncs=simulation.last_device_scheduler_syncs,
                    fused_pmt_routing_counts=routing_counts,
                    hit_signature=_hit_signature(result),
                )
            )
        signatures = [run["hit_signature"] for run in runs]
        if reference_signatures is None:
            reference_signatures = signatures
        durations = [run["elapsed_seconds"] for run in runs]
        record = dict(
            name=name,
            device_scheduler=device_scheduler,
            portal_boundary=portal_boundary,
            fused_portal_boundary=fused_portal_boundary,
            fused_pmt=fused_pmt,
            fused_pmt_max_candidates=(
                args.fused_pmt_max_candidates if fused_pmt else None
            ),
            fused_pmt_compact_union=(
                args.fused_pmt_compact_union if fused_pmt else None
            ),
            device_round_batch=batch,
            pmt_programs_per_sm=(
                None
                if pmt_programs_per_sm == 0
                else pmt_programs_per_sm
            ),
            pmt_grid_mode=(
                "not_applicable"
                if not device_scheduler
                else (
                    "uncapped"
                    if pmt_programs_per_sm == 0
                    else "persistent"
                )
            ),
            runs=runs,
            exact_hit_signatures_match_synchronized=(
                signatures == reference_signatures
            ),
            median_photons_per_second=(
                args.nphotons / statistics.median(durations)
            ),
            p95_latency_photons_per_second=(
                args.nphotons / float(np.quantile(durations, 0.95))
            ),
        )
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
    pmt_instances.DEVICE_TLAS_PROGRAMS_PER_SM = original_programs_per_sm

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    report = dict(
        schema_version=1,
        benchmark="reflect3wires fixed-round device scheduler ablation",
        experimental_fused_portal_boundary=(
            dict(
                enabled_by_default=False,
                compile_pollution_ruled_out=(
                    args.warmup_photons == args.nphotons
                ),
                warmup=(
                    "one untimed warmup per configuration; use a full-size "
                    "--warmup-photons value when comparing this kernel"
                ),
                promotion_rule=(
                    "retain as experimental unless its exact isolated gain "
                    "is also stable end to end"
                ),
            )
            if args.fused_portal_boundary
            else None
        ),
        command=sys.argv,
        configuration=dict(
            nphotons=args.nphotons,
            replicates=args.replicates,
            warmup_photons=args.warmup_photons,
            round_batches=args.round_batches,
            portal_batches=args.portal_batches,
            fused_portal_boundary=bool(args.fused_portal_boundary),
            history_length=args.history_length,
            block_size=args.block_size,
            history_epochs=args.history_epochs,
            fused_pmt=bool(args.fused_pmt),
            fused_pmt_max_candidates=args.fused_pmt_max_candidates,
            fused_pmt_routing_diagnostics=bool(
                args.fused_pmt_routing_diagnostics
            ),
            fused_pmt_compact_union=bool(args.fused_pmt_compact_union),
            pmt_programs_per_sm=args.pmt_programs_per_sm,
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
