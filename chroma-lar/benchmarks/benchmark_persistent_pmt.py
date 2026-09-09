#!/usr/bin/env python3
"""Isolated device-count PMT persistent-grid sweep on the real detector."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from chroma_lar.triton_scene.compiler import compile_reflect3wires_scene
from chroma_lar.triton_scene import instances


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=int, default=262_144)
    parser.add_argument("--capacity", type=int, default=524_288)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument(
        "--programs-per-sm",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 0],
        help="persistent grid sizes; zero runs the uncapped capacity grid",
    )
    parser.add_argument("--json")
    return parser.parse_args()


def _clone_result(result, count: int):
    return {
        name: getattr(result, name)[:count].clone()
        for name in result.__dataclass_fields__
    }


def main() -> None:
    args = _arguments()
    if args.live < 0 or args.capacity < args.live:
        raise ValueError("require 0 <= live <= capacity")
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("invalid repeat count")
    if any(value < 0 for value in args.programs_per_sm):
        raise ValueError("programs-per-sm values cannot be negative")

    device = torch.device("cuda")
    scene = compile_reflect3wires_scene()
    accelerator = instances.build_pmt_instance_accelerator(
        scene, device=device
    )
    lower = torch.as_tensor(
        accelerator.host_union_bounds_min, device=device
    )
    upper = torch.as_tensor(
        accelerator.host_union_bounds_max, device=device
    )
    origins = torch.empty(
        (args.capacity, 3), dtype=torch.float32, device=device
    )
    directions = torch.zeros_like(origins)
    if args.capacity:
        origins[:, 0] = lower[0] - 1000.0
        origins[:, 1] = upper[1] + 1000.0
        origins[:, 2] = upper[2] + 1000.0
        directions[:, 0] = -1.0
    if args.live:
        row = torch.arange(args.live, device=device, dtype=torch.int64)
        # Every live ray crosses the PMT union, with a repeatable spread over
        # its Y/Z face.  Capacity-tail rays point away from it.
        y_fraction = ((row % 4093).to(torch.float32) + 0.5) / 4093.0
        z_fraction = (((row // 4093) % 4091).to(torch.float32) + 0.5) / 4091.0
        origins[: args.live, 1] = lower[1] + y_fraction * (
            upper[1] - lower[1]
        )
        origins[: args.live, 2] = lower[2] + z_fraction * (
            upper[2] - lower[2]
        )
        directions[: args.live, 0] = 1.0
    active_count = torch.tensor(
        [args.live], dtype=torch.int32, device=device
    )
    tmax = torch.full(
        (args.capacity,), float("inf"), dtype=torch.float32, device=device
    )
    previous_instance = torch.full(
        (args.capacity,), -1, dtype=torch.int32, device=device
    )
    previous_triangle = torch.full_like(previous_instance, -1)
    workspace = accelerator.allocate_workspace(
        args.capacity, result_capacity=args.capacity
    )
    output = workspace.outputs(args.capacity)

    properties = torch.cuda.get_device_properties(device)
    records = []
    reference = None
    original_programs_per_sm = instances.DEVICE_TLAS_PROGRAMS_PER_SM
    try:
        for programs_per_sm in args.programs_per_sm:
            instances.DEVICE_TLAS_PROGRAMS_PER_SM = programs_per_sm

            def launch():
                return instances.nearest_pmt_hit_tlas_device_count(
                    accelerator,
                    origins,
                    directions,
                    active_count,
                    launch_capacity=args.capacity,
                    tmax=tmax,
                    last_instance=previous_instance,
                    last_triangle=previous_triangle,
                    workspace=workspace,
                    out=output,
                )

            for _ in range(args.warmups):
                launch()
            torch.cuda.synchronize()
            elapsed_ms = []
            for _ in range(args.repeats):
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                result = launch()
                stop.record()
                stop.synchronize()
                elapsed_ms.append(float(start.elapsed_time(stop)))

            snapshot = _clone_result(result, args.live)
            if reference is None:
                reference = snapshot
            else:
                for name, expected in reference.items():
                    if not torch.equal(snapshot[name], expected):
                        raise AssertionError(
                            f"programs-per-sm={programs_per_sm} changed {name}"
                        )
            launched = instances._persistent_tlas_program_count(
                args.capacity, 32, properties.multi_processor_count
            )
            records.append(
                {
                    "programs_per_sm": programs_per_sm,
                    "grid_mode": (
                        "uncapped" if programs_per_sm == 0 else "persistent"
                    ),
                    "launched_programs": launched,
                    "median_ms": statistics.median(elapsed_ms),
                    "minimum_ms": min(elapsed_ms),
                    "samples_ms": elapsed_ms,
                    "union_candidates": int(
                        workspace.device_candidate_count.item()
                    ),
                }
            )
    finally:
        instances.DEVICE_TLAS_PROGRAMS_PER_SM = original_programs_per_sm

    report = {
        "device": properties.name,
        "multiprocessor_count": properties.multi_processor_count,
        "live": args.live,
        "capacity": args.capacity,
        "exact_across_sweep": True,
        "records": records,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
