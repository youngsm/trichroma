"""Sustained local full-optical throughput, source through CPU waveform output."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=[1000000, 5000000, 30000000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--epochs-per-poll", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--fused-pmt", action="store_true")
    parser.add_argument(
        "--region-mode", choices=["automatic", "legacy", "disabled"], default="automatic"
    )
    parser.add_argument(
        "--depositions",
        action="store_true",
        help="include Poisson scintillation yield from an energy deposition",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path(
            "chroma-lar/benchmarks/optical_validation/full_detector_synthetic_calibration.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import triton
    from chroma_lar.optical_calibration import OpticalCalibration
    from chroma_lar.optical_simulation import FastOpticalSimulation

    if min(args.counts) <= 0 or args.repeats < 2:
        parser.error("counts must be positive and at least two repeats are required")
    calibration = OpticalCalibration.load(args.calibration)
    started = time.perf_counter()
    simulation = FastOpticalSimulation(
        calibration,
        history_length=args.history_length,
        epochs_per_poll=args.epochs_per_poll,
        block_size=args.block_size,
        fused_pmt=args.fused_pmt,
        region_mode=args.region_mode,
    )
    torch.cuda.synchronize()
    report = {
        "device": torch.cuda.get_device_name(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "setup_seconds": time.perf_counter() - started,
        "configuration": vars(args)
        | {"calibration": str(args.calibration), "output": str(args.output)},
        "calibration_fingerprint": calibration.fingerprint,
        "scene_fingerprint": simulation.transport.fingerprint,
        "calibration_status": calibration.manifest["status"],
        "electronics_noise_rms_adc": calibration.digitizer["noise_rms"],
        "geometry": {
            "pmts": simulation.transport.scene.instances.count,
            "wire_planes": simulation.transport.scene.wires.count,
            "all_box_faces": bool(simulation.transport.scene.boxes.reachable_face_mask.all()),
        },
        "scope": "GPU scintillation generation, spectral transport, PMT collection/TTS/charge, waveform superposition, noise/ADC and CPU-readable optical hits/PE/waveforms",
        "excluded": "detector construction, compilation warmup, file writes and optional full dead-photon debugging states",
        "source": (
            "Poisson scintillation yield from an energy deposition at (-1000,0,0) mm"
            if args.depositions
            else "specified photon count, LAr spectrum/time law, 30 mm cube centered at (-1000,0,0) mm"
        ),
        "events": [11, 99],
        "target_sustained_photons_per_second": 20000000,
        "counts": [],
    }
    report["compiled_regions"] = (
        None
        if simulation.transport.query.regions is None
        else simulation.transport.query.regions.as_dict()
    )
    root = Path(__file__).resolve().parents[2]
    report["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            root / "chroma-lar/chroma_lar/spectral_backend.py",
            root / "chroma-lar/chroma_lar/spectral_kernels.py",
            root / "chroma-lar/chroma_lar/optical_simulation.py",
            root / "chroma-lite/chroma/triton/digitizer_kernels.py",
            root / "chroma-lar/chroma_lar/triton_scene/compiler.py",
        )
    }
    for relative in (
        "chroma-lite/chroma/triton/primitives.py",
        "chroma-lite/chroma/triton/regions.py",
        "chroma-lar/chroma_lar/triton_scene/primitive_adapter.py",
        "chroma-lar/chroma_lar/spectral_state.py",
        "chroma-lar/chroma_lar/spectral_model.py",
        "chroma-lar/chroma_lar/optical_readout.py",
        "chroma-lar/chroma_lar/optical_calibration.py",
        "chroma-lar/chroma_lar/triton_scene/detector_query.py",
        "chroma-lar/chroma_lar/triton_scene/device_geometry.py",
        "chroma-lite/chroma/triton/spectral.py",
        "chroma-lite/chroma/triton/optics.py",
        "chroma-lar/chroma_lar/triton_scene/instances.py",
        "chroma-lar/chroma_lar/triton_scene/intersect.py",
        "chroma-lar/chroma_lar/generator/scintillation.py",
        "chroma-lite/chroma/triton/spectral_kernels.py",
        "chroma-lite/chroma/triton/physics_kernels.py",
        "chroma-lite/chroma/triton/optical_response.py",
    ):
        report["source_sha256"][relative] = hashlib.sha256(
            (root / relative).read_bytes()
        ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def run(count, seed):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        stages = []
        begin = time.perf_counter()
        if args.depositions:
            result = simulation.simulate_depositions(
                [[-1000, 0, 0], [-1000, 0, 0]],
                [count / calibration.source.yield_per_mev, 0],
                event_indices=[11, 99],
                seed=seed,
                max_photons=2 * count,
                timings=stages,
            )
        else:
            result = simulation.simulate_voxel(
                count, seed=seed, event_id=11, event_indices=[11, 99], timings=stages
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - begin
        actual_count = result.metadata["photons"]
        assert args.depositions or actual_count == count
        assert not result.metadata["transport_diagnostics"]["escaped"]
        assert np.isfinite(result.waveforms.samples).all()
        assert np.all(result.transport.hits.wavelengths > 200)
        if calibration.digitizer["noise_rms"] == 0:
            np.testing.assert_array_equal(
                result.waveforms.samples[1], calibration.digitizer["baseline"]
            )
        row = {
            "seed": seed,
            "photons": actual_count,
            "event_seconds": elapsed,
            "photons_per_second": actual_count / elapsed,
            "stages": stages,
            "detected": len(result.transport.hits),
            "photoelectrons": len(result.photoelectrons.times),
            "diagnostics": result.metadata["transport_diagnostics"],
            "waveform_shape": list(result.waveforms.samples.shape),
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_torch_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        print(json.dumps(row), flush=True)
        return row

    for count in args.counts:
        warmup = run(count, 1729)
        entry = {"photons": count, "warmup": warmup, "runs": []}
        report["counts"].append(entry)
        for repeat in range(args.repeats):
            entry["runs"].append(run(count, 901 + 1009 * repeat))
            rates = [r["photons_per_second"] for r in entry["runs"]]
            entry.update(
                sustained_photons_per_second=sum(r["photons"] for r in entry["runs"])
                / sum(r["event_seconds"] for r in entry["runs"]),
                median_photons_per_second=float(np.median(rates)),
                minimum_photons_per_second=min(rates),
                maximum_photons_per_second=max(rates),
            )
            entry["target_reached"] = entry["sustained_photons_per_second"] >= 20000000
            temporary = args.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            temporary.replace(args.output)


if __name__ == "__main__":
    main()
