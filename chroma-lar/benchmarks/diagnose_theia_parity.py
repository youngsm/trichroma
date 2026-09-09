"""Trace CPU/GPU divergence on the original dense water fixture."""

import argparse
import json
from pathlib import Path

import numpy as np

from chroma.triton.bvh import nearest_hit_bvh_cpu
from chroma.triton.bvh_kernels import nearest_hit_local
from chroma.triton.examples.theia import build_theia, cherenkov_photons
from chroma.triton.photon_input import as_photon_batch, slice_batch
from chroma.triton.spectral import SpectralScene, SpectralSimulation
from optical_parity import compare_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validated-layout", action="store_true")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--photon-index", type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = (
        build_theia()
        if args.validated_layout
        else build_theia(24000.0, coverage=0.9, diameter=508.0, validate_clearance=False)
    )
    scene = SpectralScene.compile(fixture.detector(), wavelengths=np.arange(300.0, 651.0, 5.0))
    cpu = SpectralSimulation(scene, backend="reference_bvh")
    gpu = SpectralSimulation(scene)
    print("Compiled dense water fixture", flush=True)
    batch = as_photon_batch(cherenkov_photons(args.count, seed=args.seed + 7))
    if args.photon_index is not None:
        batch = slice_batch(batch, args.photon_index, args.photon_index + 1)
    expected = cpu.simulate(batch, seed=args.seed, max_steps=256)
    actual = gpu.simulate(batch, seed=args.seed, max_steps=256)
    np.savez_compressed(
        args.output / "states.npz",
        **{
            f"{prefix}_{name}": value
            for prefix, result in (("cpu", expected), ("gpu", actual))
            for name, value in result.final_state.items()
        },
    )
    differences = np.flatnonzero(expected.final_state["flags"] != actual.final_state["flags"])
    report = {
        "mismatched_photons": batch.global_photon_ids[differences].tolist(),
        "traces": [],
        "configuration": vars(args) | {"output": str(args.output)},
    }
    if not len(differences):
        report["comparison"] = compare_results(actual, expected, evidence=args.output / "failure")
    args.output.joinpath("trace.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Mismatched photons", differences, flush=True)
    for index in differences:
        photon = slice_batch(batch, int(index), int(index) + 1)
        trace = {
            "photon_id": int(photon.global_photon_ids[0]),
            "steps": [],
            "input": {k: v.tolist() for k, v in vars(photon).items() if isinstance(v, np.ndarray)},
        }
        report["traces"].append(trace)
        for step in range(max(expected.steps, actual.steps) + 1):
            if step:
                a = cpu.simulate(photon, seed=args.seed, max_steps=step).final_state
                b = gpu.simulate(photon, seed=args.seed, max_steps=step).final_state
            else:
                a = b = dict(
                    pos=photon.pos,
                    direction=photon.direction,
                    last_hit=photon.last_hit_triangles,
                    flags=photon.flags,
                )
            entry = {
                "step": step,
                "cpu": {k: v.tolist() for k, v in a.items()},
                "gpu": {k: v.tolist() for k, v in b.items()},
                "queries": [],
            }
            for label, state in (("cpu", a), ("gpu", b)):
                c = nearest_hit_bvh_cpu(
                    scene.bvh,
                    state["pos"],
                    state["direction"],
                    last_hit=state["last_hit"],
                    high_precision=True,
                )
                d = nearest_hit_local(
                    gpu._device.bvh,
                    state["pos"],
                    state["direction"],
                    last_hit=state["last_hit"],
                    high_precision=True,
                )
                entry["queries"].append(
                    {
                        "input": label,
                        "cpu_triangle": c.triangle_ids.tolist(),
                        "gpu_triangle": d.triangle_ids.cpu().tolist(),
                        "cpu_distance": c.distances.tolist(),
                        "gpu_distance": d.distances.cpu().tolist(),
                        "cpu_triangle_vertices": [
                            scene.bvh.triangle_vertices[t].tolist() if t >= 0 else None
                            for t in c.triangle_ids
                        ],
                        "gpu_triangle_vertices": [
                            scene.bvh.triangle_vertices[t].tolist() if t >= 0 else None
                            for t in d.triangle_ids.cpu().numpy()
                        ],
                    }
                )
            trace["steps"].append(entry)
            print(
                index,
                step,
                "flags",
                a["flags"],
                b["flags"],
                "triangles",
                a["last_hit"],
                b["last_hit"],
                flush=True,
            )
            args.output.joinpath("trace.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
