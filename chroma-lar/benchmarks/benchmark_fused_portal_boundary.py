#!/usr/bin/env python3
"""Microbenchmark fused versus materialized direct-portal boundary physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nphotons", type=int, default=1_000_000)
    parser.add_argument("--replicates", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--collision-epochs", type=int, default=2)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument("--json", type=Path, required=True)
    return parser


def _clone_state(state):
    return tuple(value.clone() for value in state)


def _prepare_boundary_population(simulation, args):
    """Run the real LAr collision front-end and freeze its boundary queue."""

    import torch
    from chroma.triton.transport import DeviceQueue, collision_first_epoch
    from chroma_lar.triton_backend import _signed_seed32

    state = simulation._new_state(
        args.nphotons,
        (-1000.0, 0.0, 0.0),
        30.0,
        args.seed,
        0,
    )
    simulation._ensure_collision_queues(args.nphotons)
    boundary = simulation.collision_boundary_accumulator.reset()
    input_count = torch.full(
        (1,), args.nphotons, dtype=torch.int32, device=simulation.device
    )
    history = DeviceQueue(
        torch.arange(
            args.nphotons, dtype=torch.int32, device=simulation.device
        ),
        input_count,
    )
    for epoch_index in range(args.collision_epochs):
        workspace = simulation.collision_queue_workspaces[epoch_index % 2]
        epoch = collision_first_epoch(
            state[0],
            state[1],
            state[2],
            state[3],
            state[4],
            state[5],
            history,
            simulation.safe_lower,
            simulation.safe_upper,
            simulation.lar_absorption_length,
            simulation.lar_scattering_length,
            simulation.lar_refractive_index,
            seed=_signed_seed32(args.seed, 0x51A3),
            photon_id_base=0,
            max_scatter=args.history_length,
            block_size=args.block_size,
            partition="active",
            step_counts=state[9],
            max_steps=1000,
            last_instances=state[6],
            last_triangles=state[7],
            queue_workspace=workspace,
            input_capacity=args.nphotons,
            boundary_accumulator=boundary,
            append_boundary=True,
        )
        history = epoch.continuing

    torch.cuda.synchronize(simulation.device)
    boundary_items = boundary.size()
    if boundary_items == 0:
        raise RuntimeError("collision preparation produced no boundary photons")
    frozen_boundary = DeviceQueue(boundary.buffer.clone(), boundary.count.clone())
    return _clone_state(state), frozen_boundary, boundary_items


def _assert_equal_state(torch, actual, expected):
    for index, (actual_value, expected_value) in enumerate(
        zip(actual, expected)
    ):
        if not torch.equal(actual_value, expected_value):
            raise RuntimeError(f"state tensor {index} differs between variants")


def main(argv=None):
    args = _parser().parse_args(argv)
    if (
        args.nphotons <= 0
        or args.replicates < 2
        or args.warmups < 0
        or args.collision_epochs <= 0
        or args.history_length <= 0
        or args.block_size not in (64, 128, 256, 512)
    ):
        raise SystemExit("invalid positive benchmark configuration")

    import torch
    import triton
    from chroma.triton.transport import DeviceQueue
    from chroma_lar.triton_backend import (
        Reflect3WiresTritonSimulation,
        _signed_seed32,
    )
    from chroma_lar.triton_scene.fused_portal_boundary import (
        FusedPortalBoundaryWorkspace,
        step_fused_direct_portals,
    )
    from chroma_lar.triton_scene.portals import (
        PortalWorkspace,
        partition_certified_box_portals,
    )

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    simulation = Reflect3WiresTritonSimulation(
        tile_size=args.nphotons,
        history_length=args.history_length,
        block_size=args.block_size,
        reservoir_rounds=None,
        branch_specialized_boundary=False,
        portal_boundary=True,
        device_scheduler=True,
    )
    frozen_state, boundary, boundary_items = _prepare_boundary_population(
        simulation, args
    )
    materialized_workspace = PortalWorkspace.allocate(
        args.nphotons, simulation.device
    )
    fused_workspace = FusedPortalBoundaryWorkspace.allocate(
        args.nphotons, simulation.device
    )
    materialized_carry = DeviceQueue.allocate(
        args.nphotons, device=simulation.device
    )
    fused_carry = DeviceQueue.allocate(args.nphotons, device=simulation.device)
    boundary_seed = _signed_seed32(args.seed, 0x7F4A)

    def materialized_once(state):
        materialized_carry.reset()
        partition = partition_certified_box_portals(
            state[0],
            state[1],
            boundary,
            state[6],
            state[7],
            simulation.portal_descriptor,
            workspace=materialized_workspace,
            block_size=args.block_size,
            input_capacity=args.nphotons,
            launch_capacity=args.nphotons,
        )
        simulation._step_boundaries(
            state,
            partition.direct,
            partition.hit,
            boundary_seed,
            0,
            1000,
            input_capacity=args.nphotons,
            survivor_queue=materialized_carry,
            reset_survivors=False,
        )
        return partition.fallback, materialized_carry

    def fused_once(state):
        fused_carry.reset()
        fallback = step_fused_direct_portals(
            state,
            boundary,
            fused_carry,
            simulation.portal_descriptor,
            simulation.scene_device,
            workspace=fused_workspace,
            input_capacity=args.nphotons,
            launch_capacity=args.nphotons,
            seed=boundary_seed,
            photon_id_base=0,
            max_steps=1000,
            block_size=args.block_size,
        )
        return fallback, fused_carry

    # Validate the exact state and queue sets before timing.  Queue order is
    # intentionally unspecified because both paths reserve output per CTA.
    reference_state = _clone_state(frozen_state)
    fused_state = _clone_state(frozen_state)
    reference_fallback, reference_carry = materialized_once(reference_state)
    fused_fallback, actual_carry = fused_once(fused_state)
    torch.cuda.synchronize(simulation.device)
    _assert_equal_state(torch, fused_state, reference_state)
    if not torch.equal(
        torch.sort(fused_fallback.tensor()).values,
        torch.sort(reference_fallback.tensor()).values,
    ):
        raise RuntimeError("fallback populations differ")
    if not torch.equal(
        torch.sort(actual_carry.tensor()).values,
        torch.sort(reference_carry.tensor()).values,
    ):
        raise RuntimeError("direct survivor populations differ")
    fallback_items = fused_fallback.size()
    direct_items = boundary_items - fallback_items

    def timed_once(function):
        state = _clone_state(frozen_state)
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function(state)
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop) / 1000.0

    for _ in range(args.warmups):
        timed_once(materialized_once)
        timed_once(fused_once)
    durations = {"materialized": [], "fused": []}
    for replicate in range(args.replicates):
        order = (
            ("materialized", materialized_once),
            ("fused", fused_once),
        )
        if replicate & 1:
            order = tuple(reversed(order))
        for name, function in order:
            durations[name].append(timed_once(function))

    records = {}
    for name, values in durations.items():
        median = statistics.median(values)
        records[name] = {
            "seconds": values,
            "median_seconds": median,
            "boundary_photons_per_second": boundary_items / median,
            "direct_photons_per_second": direct_items / median,
        }
    payload = {
        "benchmark": "fused_portal_boundary_microbenchmark",
        "timing_metadata": {
            "compile_polluted_timing": False,
            "reason": (
                "An exact comparison call and the requested untimed warmups "
                "preceded all timed samples."
            ),
        },
        "promotion": {
            "enabled_by_default": False,
            "decision": (
                "Experimental reference only; isolated kernel speedup must "
                "also survive an end-to-end benchmark before promotion."
            ),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(simulation.device),
        "config": vars(args) | {"json": str(args.json)},
        "population": {
            "source_photons": args.nphotons,
            "boundary_photons": boundary_items,
            "direct_photons": direct_items,
            "fallback_photons": fallback_items,
            "direct_fraction": direct_items / boundary_items,
        },
        "variants": records,
        "fused_speedup": (
            records["materialized"]["median_seconds"]
            / records["fused"]["median_seconds"]
        ),
        "correctness": "all state words and output queue sets exact",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
