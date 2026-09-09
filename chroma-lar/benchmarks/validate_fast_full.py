"""Independent calibrated VUV/WLS yield comparison with existing analytic-wire CUDA.

The reference lacks WLS delays and group velocity, so timing laws are tested
separately. This comparison checks detection, re-emission and wavelength/channel
distributions, whose laws do not depend on those flight/emission delays.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("prepare", "cuda", "triton", "compare"), required=True)
    parser.add_argument("--count", type=int, default=500000)
    parser.add_argument("--directory", type=Path, default=Path("chroma-lar/benchmarks/optical_validation/fast_full_comparison"))
    parser.add_argument("--calibration", type=Path, default=Path("chroma-lar/benchmarks/optical_validation/full_detector_synthetic_calibration.json"))
    args = parser.parse_args()
    p = args.directory
    p.mkdir(parents=True, exist_ok=True)
    seeds = [11, 29, 83]
    if args.backend == "compare":
        from scipy.stats import ks_2samp, chi2
        records = [json.loads((p/f"{backend}.json").read_text()) for backend in ("cuda", "triton")]
        checks = []
        for seed in seeds:
            a, b = (np.load(p/f"{backend}_{seed}.npz") for backend in ("cuda", "triton"))
            ra, rb = (next(r for r in series["runs"] if r["seed"] == seed) for series in records)
            assert ra["input_sha256"] == rb["input_sha256"]
            for field in ("detected", "reemitted"):
                pa, pb = ra[field]/args.count, rb[field]/args.count
                error = np.sqrt((pa*(1-pa)+pb*(1-pb))/args.count)
                pull = abs(pa-pb)/max(error, 1/args.count)
                checks.append({"seed": seed, "metric": field, "pull": float(pull), "passed": bool(pull <= 6)})
            statistic = ks_2samp(a["wavelengths"], b["wavelengths"]).statistic
            bound = np.sqrt(-.5*np.log(1.e-6/2)*(1/len(a["wavelengths"])+1/len(b["wavelengths"])))
            checks.append({"seed": seed, "metric": "wavelength_ks", "statistic": float(statistic), "bound": float(bound), "passed": bool(statistic <= bound)})
            ca, cb = (np.bincount(r["channels"], minlength=162) for r in (a, b))
            keep = ca+cb >= 20
            ca, cb = np.r_[ca[keep], ca[~keep].sum()], np.r_[cb[keep], cb[~keep].sum()]
            keep = ca+cb > 0
            ca, cb = ca[keep], cb[keep]
            pooled = (ca+cb)/(ca.sum()+cb.sum())
            expected_a, expected_b = pooled*ca.sum(), pooled*cb.sum()
            stat = np.sum((ca-expected_a)**2/expected_a+(cb-expected_b)**2/expected_b)
            prob = chi2.sf(stat, len(ca)-1)
            checks.append({"seed": seed, "metric": "channel_distribution", "pvalue": float(prob), "passed": bool(prob >= 1.e-6)})
        report = {"checks": checks, "passed": all(c["passed"] for c in checks),
                  "physics": "same calibrated VUV source, TPB spectrum/yield, analytic wires and PMT detection tables",
                  "timing": "excluded from CUDA comparison: legacy CUDA lacks WLS delay and group velocity; separately unit-tested",
                  "runs": records}
        (p/"comparison.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
        return
    from chroma_lar.optical_calibration import OpticalCalibration
    calibration = OpticalCalibration.load(args.calibration)
    if args.backend == "prepare":
        for seed in seeds:
            positions = np.random.default_rng(seed).uniform(-15, 15, (args.count, 3))+[-1000, 0, 0]
            source = calibration.source.photons(positions, seed=seed, event_indices=11)
            np.savez(p/f"source_{seed}.npz", pos=source.pos, dir=source.direction, pol=source.polarization,
                     wavelengths=source.wavelengths, t=source.times, flags=source.flags, evidx=source.event_indices)
        return
    context = None
    if args.backend == "cuda":
        from chroma import gpu
        from chroma.triton.bvh import build_packed_bvh
        from pycuda import gpuarray as ga
        Path(os.environ.get("PYCUDA_CACHE_DIR", "/tmp/chroma-fast-full-pycuda")).mkdir(parents=True, exist_ok=True)
        context = gpu.create_cuda_context()
        geometry = calibration.build_detector("detector_config_reflect_reflect3wires", analytic_wires=True)
        geometry.flatten()
        bvh = build_packed_bvh(geometry.mesh.vertices, geometry.mesh.triangles)
        geometry.bvh = SimpleNamespace(nodes=np.array(bvh.nodes).view(ga.vec.uint4).reshape(-1),
            world_coords=SimpleNamespace(world_origin=bvh.world_origin, world_scale=bvh.world_scale))
        simulation = gpu.GPUGeometry(geometry, wavelengths=calibration.wavelengths)
    else:
        from chroma_lar.spectral_backend import SpectralDetectorSimulation
        simulation = SpectralDetectorSimulation(calibration)
    from chroma.event import Photons
    report = {"backend": args.backend, "calibration_fingerprint": calibration.fingerprint, "photons_per_seed": args.count, "runs": []}
    if args.backend == "triton":
        report["scene_fingerprint"] = simulation.fingerprint
        root = Path(__file__).resolve().parents[2]
        report["source_sha256"] = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in
            (root/"chroma-lar/chroma_lar/spectral_backend.py", root/"chroma-lar/chroma_lar/spectral_kernels.py",
             root/"chroma-lar/chroma_lar/triton_scene/compiler.py")}
    try:
        for seed in seeds:
            filename = p/f"source_{seed}.npz"
            with np.load(filename) as data:
                source = Photons(**dict(data))
            if args.backend == "cuda":
                state = gpu.GPUPhotons(source)
                rng = gpu.get_rng_states(512*1024, seed=seed)
                state.propagate(simulation, rng, nthreads_per_block=512, max_blocks=1024, max_steps=2048)
                final = state.get()
                flags, wavelengths, times = final.flags, final.wavelengths, final.t
                selected = (flags & 4) != 0
                channels = geometry.solid_id_to_channel_index[geometry.solid_id[final.last_hit_triangles[selected]]]
                nonfinite = int(np.count_nonzero(~np.isfinite(final.dir).all(1)))
                del state, rng
            else:
                final = simulation.simulate(source, seed=seed, keep_final_states=True).final_state
                flags, wavelengths, times = final["flags"], final["wavelengths"], final["times"]
                selected = (flags & 4) != 0
                channels = final["channels"][selected]
                nonfinite = int(np.count_nonzero(~np.isfinite(final["direction"]).all(1)))
            assert np.all(channels >= 0)
            np.savez(p/f"{args.backend}_{seed}.npz", wavelengths=wavelengths[selected], times=times[selected], channels=channels)
            row = {"seed": seed, "input_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
                "detected": int(selected.sum()), "reemitted": int(np.count_nonzero(flags & 128)),
                "bulk_absorbed": int(np.count_nonzero(flags & 2)), "surface_absorbed": int(np.count_nonzero(flags & 8)),
                "nonfinite": nonfinite, "aborted": int(np.count_nonzero(flags & ((1 << 31) | (1 << 15)))),
                "escaped": int(np.count_nonzero(flags & 1)), "step_limit": int(np.count_nonzero(flags & (1 << 30)))}
            report["runs"].append(row)
            (p/f"{args.backend}.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(row), flush=True)
    finally:
        if context is not None:
            context.pop()


if __name__ == "__main__":
    main()
