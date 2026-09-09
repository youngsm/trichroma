"""Run the full optical pipeline from NPZ inputs or a synthetic demonstration.

PYTHONPATH=../chroma-lite:. python benchmarks/run_optical_simulation.py \
    --demo --backend triton --output /tmp/optical-demo.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chroma.detector import Detector
from chroma.event import Photons
from chroma.geometry import Solid
from chroma.make import box
from chroma_lar.optical_calibration import OpticalCalibration
from chroma_lar.optical_simulation import OpticalSimulation, FastOpticalSimulation


def demo_detector(calibration):
    """Nested boxes: outer sensor, inner WLS shell; intentionally synthetic."""
    medium = calibration.materials["demo_medium"]
    detector = Detector(medium)
    detector.add_pmt(Solid(box(2400, 2400, 2400), medium, medium,
                           surface=calibration.surfaces["demo_sensor"]), displacement=np.zeros(3))
    detector.add_solid(Solid(box(2000, 2000, 2000), medium, medium,
                             surface=calibration.surfaces["demo_tpb"]))
    return detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", default=str(Path(__file__).parents[1]/"examples/synthetic_optical_calibration.json"))
    parser.add_argument("--detector-config")
    parser.add_argument("--input", help="NPZ with pos/energy_mev or pos/dir/pol/wavelengths; optional t/evidx")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--backend", choices=("triton", "triton-fast", "reference"), default="triton")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--max-photons", type=int, default=100_000_000)
    args = parser.parse_args()
    if not args.demo and (not args.detector_config or not args.input):
        parser.error("provide --demo or both --detector-config and --input")
    calibration = OpticalCalibration.load(args.calibration)
    if args.backend == "triton-fast":
        if args.demo:
            parser.error("triton-fast requires the calibrated reflect3wires detector; use --detector-config and --input")
        simulation = FastOpticalSimulation(calibration, config_name=args.detector_config)
    else:
        detector = demo_detector(calibration) if args.demo else calibration.build_detector(args.detector_config)
        simulation = OpticalSimulation(detector, calibration, backend=args.backend)
    if args.demo:
        # Include a zero-energy event to exercise empty-event output.
        result = simulation.simulate_depositions([[0, 0, 0], [0, 0, 0]], [.05, 0.],
                    event_indices=[0, 1], seed=args.seed, max_steps=args.max_steps)
    else:
        with np.load(args.input, allow_pickle=False) as data:
            t, events = data.get("t", 0.), data.get("evidx", 0)
            if "energy_mev" in data:
                result = simulation.simulate_depositions(data["pos"], data["energy_mev"], times=t,
                            event_indices=events, seed=args.seed, max_steps=args.max_steps, max_photons=args.max_photons)
            else:
                n = len(data["pos"])
                photons = Photons(data["pos"], data["dir"], data["pol"], data["wavelengths"],
                                   t=np.broadcast_to(t, (n,)), evidx=np.broadcast_to(events, (n,)))
                result = simulation.simulate_photons(photons, seed=args.seed, max_steps=args.max_steps)
    result.save(args.output)
    print(json.dumps(result.metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
