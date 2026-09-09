"""Capture/compare all per-photon and readout outputs across a host refactor."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from chroma_lar.optical_calibration import OpticalCalibration
from chroma_lar.optical_simulation import FastOpticalSimulation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["capture", "compare"])
    args = parser.parse_args()
    root = Path("chroma-lar/benchmarks/optical_validation/maintainability")
    root.mkdir(exist_ok=True)
    if args.mode == "capture" and (root / "before.npz").exists():
        raise FileExistsError("preserve the recorded pre-refactor baseline; use compare")
    cal = OpticalCalibration.load(root.parent / "full_detector_synthetic_calibration_noise.json")
    sim = FastOpticalSimulation(cal)
    data = {}
    for seed in [11, 101, 2027]:
        result = sim.simulate_depositions(
            [[-1000, 0, 0], [-2140, 500, 100], [-100, -500, 1500]],
            [150000 / cal.source.yield_per_mev, 100000 / cal.source.yield_per_mev, 0],
            event_indices=[11, 37, 99],
            seed=seed,
            keep_final_states=True,
        )
        for name, value in result.transport.final_state.items():
            data[f"{seed}_state_{name}"] = value
        for name, obj in [
            ("hits", result.transport.hits),
            ("pe", result.photoelectrons),
            ("wf", result.waveforms),
        ]:
            for field, value in vars(obj).items():
                if isinstance(value, np.ndarray):
                    data[f"{seed}_{name}_{field}"] = value
        data[f"{seed}_metadata"] = np.asarray(json.dumps(result.metadata, sort_keys=True))
    if args.mode == "capture":
        np.savez_compressed(root / "before.npz", **data)
    else:
        before = np.load(root / "before.npz")
        assert set(before.files) == set(data)
        for name, value in data.items():
            np.testing.assert_array_equal(value, before[name], err_msg=name)
        report = {
            "arrays_compared": len(data),
            "photons": sum(len(data[f"{s}_state_times"]) for s in [11, 101, 2027]),
            "seeds": [11, 101, 2027],
            "comparison": "exact array equality, including terminal photon states, optical hits, PE, noisy ADC waveforms and metadata",
            "baseline_sha256": hashlib.sha256((root / "before.npz").read_bytes()).hexdigest(),
        }
        (root / "equivalence.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    print(args.mode, "passed", flush=True)


if __name__ == "__main__":
    main()
