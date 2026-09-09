#!/usr/bin/env python3
"""Measure per-step event rates in the reference Chroma transport.

This is a diagnostic companion to ``validate_triton_backend.py``.  Chroma's
history word records whether an event ever happened, not how many times it
happened.  Running its existing single-step tracking path lets us classify
every propagation step from the post-step ``last_hit_triangle`` and terminal
flags without modifying the reference CUDA kernel.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[2]
for source_root in (REPOSITORY / "chroma-lite", REPOSITORY / "chroma-lar"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nphotons", type=int, default=100_000)
    parser.add_argument("--center", type=float, nargs=3, default=(-1000.0, 0.0, 0.0))
    parser.add_argument("--voxel-size", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()

    from chroma import gpu
    from chroma.event import (
        BULK_ABSORB,
        NAN_ABORT,
        NO_HIT,
        SURFACE_ABSORB,
        SURFACE_DETECT,
    )
    from chroma_lar.geometry import build_detector_from_config
    from validate_triton_backend import _source_photons

    # NumPy 2 compatibility, identical to the acceptance harness.
    had_alias = hasattr(np.linalg, "linalg")
    old_alias = getattr(np.linalg, "linalg", None)
    if not had_alias:
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
        if had_alias:
            np.linalg.linalg = old_alias  # type: ignore[attr-defined]
        else:
            delattr(np.linalg, "linalg")

    context = gpu.create_cuda_context()
    try:
        gpu_geometry = gpu.GPUDetector(detector)
        rng_states = gpu.get_rng_states(512 * 1024, seed=args.seed)
        source = _source_photons(
            args.nphotons, args.center, args.voxel_size, args.seed
        )
        photons = gpu.GPUPhotons(
            source, copy_flags=True, copy_triangles=False, copy_weights=False
        )
        step_ids, step_states = photons.propagate(
            gpu_geometry,
            rng_states,
            nthreads_per_block=512,
            max_blocks=1024,
            max_steps=args.max_steps,
            track=True,
        )

        terminal_mask = np.uint32(
            NO_HIT | BULK_ABSORB | SURFACE_DETECT | SURFACE_ABSORB | NAN_ABORT
        )
        total_steps = 0
        bulk_scatter = 0
        boundary_continue = 0
        terminal = {"no_hit": 0, "bulk_absorb": 0, "surface_detect": 0,
                    "surface_absorb": 0, "nan_abort": 0}
        active_by_step = []
        # Entry zero is the pre-propagation source snapshot.  Every later
        # entry contains exactly the queue that executed that one step.
        for states in step_states[1:]:
            flags = np.asarray(states.flags, dtype=np.uint32)
            last = np.asarray(states.last_hit_triangles, dtype=np.int32)
            active_by_step.append(int(flags.size))
            total_steps += int(flags.size)
            alive = (flags & terminal_mask) == 0
            bulk_scatter += int(np.count_nonzero(alive & (last < 0)))
            boundary_continue += int(np.count_nonzero(alive & (last >= 0)))
            terminal["no_hit"] += int(np.count_nonzero(flags & NO_HIT))
            terminal["bulk_absorb"] += int(np.count_nonzero(flags & BULK_ABSORB))
            terminal["surface_detect"] += int(np.count_nonzero(flags & SURFACE_DETECT))
            terminal["surface_absorb"] += int(np.count_nonzero(flags & SURFACE_ABSORB))
            terminal["nan_abort"] += int(np.count_nonzero(flags & NAN_ABORT))

        final = photons.get()
        report = {
            "nphotons": args.nphotons,
            "total_steps": total_steps,
            "steps_per_photon": total_steps / args.nphotons,
            "bulk_scatter_steps": bulk_scatter,
            "bulk_scatter_per_photon": bulk_scatter / args.nphotons,
            "boundary_continue_steps": boundary_continue,
            "boundary_continue_per_photon": boundary_continue / args.nphotons,
            "terminal_observations": terminal,
            "detections": int(np.count_nonzero(final.flags & SURFACE_DETECT)),
            "hit_fraction": float(np.count_nonzero(final.flags & SURFACE_DETECT) / args.nphotons),
            "active_by_step_first_32": active_by_step[:32],
            "tracked_rounds": len(active_by_step),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        context.pop()


if __name__ == "__main__":
    main()
