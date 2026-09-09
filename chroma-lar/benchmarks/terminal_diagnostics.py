#!/usr/bin/env python3
"""Compare terminal history/location populations while debugging parity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
for source in (ROOT / "chroma-lite", ROOT / "chroma-lar"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from validate_triton_backend import _source_photons


BITS = {
    "no_hit": 1 << 0,
    "bulk_absorb": 1 << 1,
    "surface_detect": 1 << 2,
    "surface_absorb": 1 << 3,
    "rayleigh": 1 << 4,
    "diffuse": 1 << 5,
    "specular": 1 << 6,
    "nan_abort": 1 << 15,
}


def _terminal_reason(flags):
    result = np.full(flags.shape, "alive", dtype="U16")
    result[(flags & BITS["no_hit"]) != 0] = "no_hit"
    result[(flags & BITS["bulk_absorb"]) != 0] = "bulk_absorb"
    result[(flags & BITS["surface_absorb"]) != 0] = "surface_absorb"
    result[(flags & BITS["surface_detect"]) != 0] = "surface_detect"
    result[(flags & BITS["nan_abort"]) != 0] = "nan_abort"
    return result


def _counts(values):
    labels, counts = np.unique(values, return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}


def _summary(flags, locations, times, extra=None):
    flags = np.asarray(flags, dtype=np.uint32)
    terminal = _terminal_reason(flags)
    cross = {}
    for reason in np.unique(terminal):
        cross[str(reason)] = _counts(np.asarray(locations)[terminal == reason])
    result = {
        "photons": int(flags.size),
        "terminal": _counts(terminal),
        "terminal_by_location": cross,
        "history_bits": {
            name: int(np.count_nonzero(flags & bit)) for name, bit in BITS.items()
        },
        "time_mean_ns": float(np.mean(times)),
        "time_quantiles_ns": np.quantile(
            times, [0.5, 0.9, 0.99]
        ).tolist(),
    }
    if extra:
        result.update(extra)
    return result


def _run_chroma(args):
    from chroma.sim import Simulation
    from chroma_lar.geometry import build_detector_from_config

    had_alias = hasattr(np.linalg, "linalg")
    old_alias = getattr(np.linalg, "linalg", None)
    if not had_alias:
        np.linalg.linalg = np.linalg
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
            np.linalg.linalg = old_alias
        else:
            delattr(np.linalg, "linalg")

    simulation = Simulation(detector, seed=args.seed, photon_tracking=False)
    source = _source_photons(
        args.nphotons, args.center, args.voxel_size, args.seed
    )
    event = next(simulation.simulate(
        source,
        keep_photons_beg=False,
        keep_photons_end=True,
        keep_hits=False,
        keep_flat_hits=True,
        run_daq=False,
        max_steps=args.max_steps,
        photons_per_batch=args.nphotons,
    ))
    photons = event.photons_end
    flags = np.asarray(photons.flags, dtype=np.uint32)
    last = np.asarray(photons.last_hit_triangles, dtype=np.int64)
    locations = np.full(flags.shape, "none", dtype="U16")
    locations[last == -2] = "wire"
    mesh = last >= 0
    solid = np.full(last.shape, -1, dtype=np.int64)
    solid[mesh] = detector.solid_id[last[mesh]]
    pmt_ids = set(np.asarray(detector.channel_index_to_solid_id).tolist())
    locations[np.isin(solid, list(pmt_ids))] = "pmt"
    non_pmt = sorted(set(solid[solid >= 0].tolist()) - pmt_ids)
    # Target construction order is cavity, PMTs, active, cathode.
    if non_pmt:
        locations[solid == non_pmt[0]] = "cavity"
    if len(non_pmt) >= 2:
        locations[solid == non_pmt[-2]] = "active"
        locations[solid == non_pmt[-1]] = "cathode"
    return _summary(flags, locations, np.asarray(photons.t), {
        "backend": "chroma",
        "flat_hits": int(len(event.flat_hits)),
        "non_pmt_solid_ids": non_pmt,
    })


def _run_triton(args):
    from chroma_lar.triton_backend import Reflect3WiresTritonSimulation

    simulation = Reflect3WiresTritonSimulation(
        tile_size=args.nphotons, history_length=8
    )
    result = simulation.simulate(
        args.nphotons,
        args.center,
        voxel_size=args.voxel_size,
        seed=args.seed,
        max_steps=args.max_steps,
        keep_final_states=True,
    )
    state = result.final_states[0]
    arrays = [item.detach().cpu().numpy() for item in state]
    positions, _, _, times, flags, _, last_instance, last_triangle, channels, steps = arrays
    locations = np.full(flags.shape, "none", dtype="U16")
    locations[last_instance >= 0] = "pmt"
    surface_absorb = (flags & BITS["surface_absorb"]) != 0
    scene = simulation.scene
    active = scene.boxes.kinds.index("active")
    cathode = scene.boxes.kinds.index("cathode")
    tolerance = 2.0e-2
    at_cathode = np.abs(
        positions[:, 0] - scene.boxes.bounds_min[cathode, 0]
    ) < tolerance
    lo = scene.boxes.bounds_min[active]
    hi = scene.boxes.bounds_max[active]
    at_active = np.any(
        (np.abs(positions - lo) < tolerance)
        | (np.abs(positions - hi) < tolerance), axis=1
    )
    locations[surface_absorb & at_cathode] = "cathode"
    locations[surface_absorb & at_active & ~at_cathode] = "active"
    locations[surface_absorb & (locations == "none")] = "wire"
    no_hit = (flags & BITS["no_hit"]) != 0
    no_hit_positions = positions[no_hit]
    no_hit_detail = {
        "count": int(no_hit_positions.shape[0]),
        "position_min": (
            np.min(no_hit_positions, axis=0).tolist()
            if no_hit_positions.size else None
        ),
        "position_max": (
            np.max(no_hit_positions, axis=0).tolist()
            if no_hit_positions.size else None
        ),
        "position_quantiles": (
            np.quantile(no_hit_positions, [0.01, 0.5, 0.99], axis=0).tolist()
            if no_hit_positions.size else None
        ),
        "outside_active_count": int(np.count_nonzero(
            no_hit & np.any((positions < lo) | (positions > hi), axis=1)
        )),
        "positive_cathode_side_count": int(np.count_nonzero(
            no_hit & (positions[:, 0] > scene.boxes.bounds_max[cathode, 0])
        )),
        "has_diffuse_count": int(np.count_nonzero(no_hit & ((flags & BITS["diffuse"]) != 0))),
        "has_specular_count": int(np.count_nonzero(no_hit & ((flags & BITS["specular"]) != 0))),
        "time_quantiles_ns": (
            np.quantile(times[no_hit], [0.01, 0.5, 0.99]).tolist()
            if np.any(no_hit) else None
        ),
    }
    return _summary(flags, locations, times, {
        "backend": "triton",
        "flat_hits": int(result.stats.detections),
        "step_mean": float(np.mean(steps)),
        "step_quantiles": np.quantile(steps, [0.5, 0.9, 0.99]).tolist(),
        "last_triangle_nonnegative": int(np.count_nonzero(last_triangle >= 0)),
        "detected_channel_count": int(np.count_nonzero(channels >= 0)),
        "no_hit_detail": no_hit_detail,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("chroma", "triton"), required=True)
    parser.add_argument("--nphotons", type=int, default=200_000)
    parser.add_argument("--center", type=float, nargs=3, default=(-1000.0, 0.0, 0.0))
    parser.add_argument("--voxel-size", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=8123)
    parser.add_argument("--max-steps", type=int, default=1000)
    args = parser.parse_args()
    result = _run_chroma(args) if args.backend == "chroma" else _run_triton(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
