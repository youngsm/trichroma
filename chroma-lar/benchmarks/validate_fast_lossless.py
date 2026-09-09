"""Full detector invariant: reflective wires cannot absorb photons in steel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photons", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-states", action="store_true")
    parser.add_argument("--trace-photon", type=int)
    parser.add_argument("--max-steps", type=int, default=2048)
    args = parser.parse_args()
    from chroma_lar.optical_calibration import OpticalCalibration
    from chroma_lar.spectral_backend import SpectralDetectorSimulation
    calibration = OpticalCalibration.load(Path(__file__).parent/"optical_validation/full_detector_synthetic_calibration.json")
    # A diagnostic optical configuration: steel is an immediate bulk absorber,
    # active walls/wires reflect, named PMT surfaces detect, and the outer
    # cavity absorbs. Bare PMT glass remains dielectric. A reflecting cavity
    # would trap escape rays indefinitely outside the closed active enclosure.
    # Any bulk absorption exposes an erroneous crossing into a wire or wall.
    for name, material in calibration.materials.items():
        material.set("absorption_length", 0. if name == "steel" else 1.e30)
        material.set("scattering_length", 1.e30)
        material.set("refractive_index", 1.)
    for name, surface in calibration.surfaces.items():
        surface.model = 0
        for field in ("absorb", "detect", "reflect_diffuse", "reflect_specular", "reemit"):
            surface.set(field, 0.)
        surface.set("detect" if name in ("validation_tpb", "perfect_pmt_photocathode", "glossy_surface") else "reflect_diffuse", 1.)
    calibration.surfaces["reflect00"].set("reflect_diffuse", 0.)
    calibration.surfaces["reflect00"].set("absorb", 1.)
    simulation = SpectralDetectorSimulation(calibration)
    trace = []
    if args.trace_photon is not None:
        args.photons = 1
        query = simulation.query.resolve
        def traced_query(state, queue, count):
            before = {name: state[index].cpu().numpy().tolist() for index, name in
                      ((0, "position"), (1, "direction"), (4, "flags"), (6, "last_instance"), (7, "last_triangle"))}
            resolved = query(state, queue, count)
            before["hit"] = [value.cpu().numpy().tolist() for value in resolved.hit]
            trace.append(before)
            return resolved
        simulation.query.resolve = traced_query
    rows = []
    for seed, center in ((11, (-1000., 0., 0.)), (29, (-1990., 500., 500.)), (83, (-10., -500., -500.))):
        trace.clear()
        try:
            result = simulation.simulate_voxel(args.photons, center, voxel_size=0., seed=seed,
                                           max_steps=args.max_steps, keep_final_states=True,
                                           photon_id_base=args.trace_photon or 0)
        except RuntimeError:
            args.output.with_suffix(".failure.json").write_text(json.dumps(simulation.last_failure, indent=2)+"\n")
            args.output.with_name(args.output.stem+f"_trace{seed}.json").write_text(json.dumps(trace, indent=2)+"\n")
            raise
        if args.trace_photon is not None:
            args.output.with_name(args.output.stem+f"_trace{seed}.json").write_text(json.dumps(trace, indent=2)+"\n")
        flags = result.final_state["flags"]
        if args.save_states:
            np.savez_compressed(args.output.with_name(args.output.stem+f"_seed{seed}.npz"), **result.final_state)
        counts = {"detected": int(np.count_nonzero(flags & 4)),
                  "bulk_absorbed": int(np.count_nonzero(flags & 2)),
                  "surface_absorbed": int(np.count_nonzero(flags & 8)),
                  "reflected": int(np.count_nonzero(flags & (32 | 64)))}
        cavity = simulation.scene.boxes.kinds.index("cavity")
        position = result.final_state["pos"][(flags & 8) != 0]
        lo, hi = simulation.scene.boxes.bounds_min[cavity], simulation.scene.boxes.bounds_max[cavity]
        at_cavity = np.all(np.any(np.isclose(position, lo, atol=.002, rtol=0) | np.isclose(position, hi, atol=.002, rtol=0), axis=1))
        passed = counts["detected"]+counts["surface_absorbed"] == args.photons and counts["bulk_absorbed"] == 0 and at_cavity
        row = {"seed": seed, "center_mm": center, "photons": args.photons, "counts": counts,
               "diagnostics": result.diagnostics, "maximum_interactions": int(result.final_state["steps"].max()), "passed": bool(passed)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    report = {"expected": "Every photon detected or absorbed on the outer cavity; zero bulk absorption behind perfectly reflecting wire/active/cathode boundaries.",
              "scene_fingerprint": simulation.fingerprint, "cases": rows, "passed": all(row["passed"] for row in rows),
              "diagnostic_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]:
        raise SystemExit("Lossless detector invariant failed")


if __name__ == "__main__":
    main()
