"""Pixel-TPC spectral transport/reference comparison and complete event timing.

Uses the existing area-averaged pixel configuration, all 164 PMTs, and the
synthetic LAr/TPB/readout bundle. This is the general mesh transport engine;
compiler-discovered regions are audited but not yet consumed by this engine.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from chroma.triton.regions import compile_regions
from chroma.triton.spectral import SpectralScene, SpectralSimulation
from chroma_lar.optical_calibration import OpticalCalibration
from chroma_lar.optical_readout import OpticalReadout
from chroma_lar.triton_scene.pixel_adapter import pixel_primitives
from optical_parity import compare_results

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION = (
    Path(__file__).parent / "optical_validation/primitive_regions/pixel_synthetic_calibration.json"
)


def pixel_photons(calibration, count, seed):
    """Two equal fixed-count 30 mm source voxels, one in each TPC half."""
    positions = np.random.default_rng(seed).uniform(-15, 15, (count, 3))
    positions[:, 0] += np.where(np.arange(count) % 2, 1000.0, -1000.0)
    return calibration.source.photons(positions, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--counts", type=int, nargs="+", default=[100000, 1000000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-count", type=int, default=10000)
    parser.add_argument("--reference-seeds", type=int, nargs="+", default=[44, 101, 2027])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.counts) <= 0 or args.repeats < 2 or args.reference_count <= 0:
        parser.error("positive photon counts and at least two repeats are required")
    import torch
    import triton

    if not torch.cuda.is_available():
        parser.error("a local CUDA GPU is required")
    begin = time.perf_counter()
    calibration = OpticalCalibration.load(args.calibration)
    geometry = calibration.build_detector("detector_config_pixel")
    primitives, materials = pixel_primitives(geometry)
    regions = compile_regions(primitives)
    scene = SpectralScene.compile(geometry, wavelengths=calibration.wavelengths)
    simulation = SpectralSimulation(scene, tile_size=1 << 20)
    readout = OpticalReadout(calibration, scene.host.channel_count, device="cuda")
    torch.cuda.synchronize()
    print(
        f"Uploaded {scene.host.triangle_count:,} triangles / {scene.host.channel_count} PMTs",
        flush=True,
    )
    report = {
        "host": platform.node(),
        "device": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "configuration": vars(args)
        | {"output": str(args.output), "calibration": str(args.calibration)},
        "scope": "CPU LAr spectrum/time/position sampling, host input, general mesh spectral transport, full terminal-state download, PMT response, GPU pulse superposition, CPU noise/ADC, host-readable hits/PE/waveforms",
        "excluded": "geometry construction, compilation warmup, file writes; fixed photon counts do not time Poisson yield sampling",
        "backend": "general flattened-mesh SpectralSimulation; region certificates not used in transport",
        "pixel_model": calibration.manifest["pixel_model"],
        "calibration_status": calibration.manifest["status"],
        "calibration_provenance": calibration.manifest["provenance"],
        "calibration_fingerprint": calibration.fingerprint,
        "scene_fingerprint": scene.fingerprint,
        "source": "equal populations in 30 mm voxels centered at (-1000,0,0) and (1000,0,0) mm; calibrated VUV spectrum, singlet/triplet time and isotropic polarization",
        "geometry": {"channels": scene.host.channel_count, "triangles": scene.host.triangle_count},
        "compiled_regions": regions.as_dict(),
        "region_materials": materials,
        "construction_seconds": time.perf_counter() - begin,
        "counts": [],
    }
    paths = [
        "chroma-lite/chroma/triton/bvh.py",
        "chroma-lite/chroma/triton/bvh_kernels.py",
        "chroma-lite/chroma/triton/boundary.py",
        "chroma-lite/chroma/triton/boundary_kernels.py",
        "chroma-lite/chroma/triton/spectral.py",
        "chroma-lite/chroma/triton/spectral_kernels.py",
        "chroma-lite/chroma/triton/optical_response.py",
        "chroma-lite/chroma/triton/digitizer_kernels.py",
        "chroma-lite/chroma/triton/primitives.py",
        "chroma-lite/chroma/triton/regions.py",
        "chroma-lar/chroma_lar/triton_scene/pixel_adapter.py",
        "chroma-lar/chroma_lar/geometry/build_larcube_pixel.py",
        "chroma-lar/chroma_lar/geometry/pixelplane.py",
        "chroma-lar/chroma_lar/geometry/pmt.py",
        "chroma-lar/chroma_lar/optical_calibration.py",
        "chroma-lar/chroma_lar/optical_readout.py",
        "chroma-lar/chroma_lar/generator/scintillation.py",
        "chroma-lar/chroma_lar/config/detector_config_pixel.py",
        "chroma-lar/chroma_lar/config/detector_config_pixel.yaml",
        "chroma-lar/benchmarks/benchmark_pixel.py",
        "chroma-lar/benchmarks/optical_parity.py",
    ]
    report["source_sha256"] = {
        p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    report["cpu_comparisons"] = []
    checks = [(1000, 44)] + [(args.reference_count, seed) for seed in args.reference_seeds]
    for count, seed in dict.fromkeys(checks):
        photons = pixel_photons(calibration, count, seed + 7)
        reference_start = time.perf_counter()
        reference = SpectralSimulation(scene, backend="reference_bvh").simulate(
            photons, seed=seed, max_steps=2048
        )
        reference_seconds = time.perf_counter() - reference_start
        actual = simulation.simulate(photons, seed=seed, max_steps=2048)
        comparison = compare_results(
            actual,
            reference,
            evidence=args.output.with_name(f"pixel_failure_{count}_{seed}"),
            wavelength_atol=0.0001,
        )
        comparison.update(
            photons=count, seed=seed, source_seed=seed + 7, cpu_seconds=reference_seconds
        )
        report["cpu_comparisons"].append(comparison)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Pixel CPU/GPU comparison passed: {count:,} photons, seed {seed}", flush=True)
    readout_options = dict(seed=seed, max_steps=2048, backend="triton", photon_count=count)
    event = readout.apply(actual, [0, 1], **readout_options)
    cpu_event = OpticalReadout(calibration, scene.host.channel_count).apply(
        actual, [0, 1], **readout_options
    )
    for name in ("times", "charges", "photon_ids", "channels", "event_indices"):
        np.testing.assert_array_equal(
            getattr(event.photoelectrons, name), getattr(cpu_event.photoelectrons, name)
        )
    adc_error = int(
        np.max(
            np.abs(
                event.waveforms.samples.astype(np.int32)
                - cpu_event.waveforms.samples.astype(np.int32)
            ),
            initial=0,
        )
    )
    if adc_error > 1:
        raise AssertionError(f"GPU/CPU digitization differs by {adc_error} ADC counts")
    report["readout_comparison"] = {
        "input": "same GPU optical hits passed to GPU and CPU readout",
        "exact": "PE times, charges, photon IDs, channels and event indices",
        "maximum_adc_difference": adc_error,
        "allowed_adc_difference": 1,
        "waveform_shape": list(event.waveforms.samples.shape),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Pixel CPU/GPU optical comparison passed", flush=True)

    def run(count, seed):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        photons = pixel_photons(calibration, count, seed)
        sourced = time.perf_counter()
        result = simulation.simulate(photons, seed=seed, max_steps=2048)
        transported = time.perf_counter()
        event = readout.apply(
            result, [0, 1], seed=seed, max_steps=2048, backend="triton", photon_count=count
        )
        torch.cuda.synchronize()
        done = time.perf_counter()
        flags = result.final_state["flags"]
        failures = int(np.count_nonzero(flags & ((1 << 31) | (1 << 30) | 1)))
        if failures:
            raise RuntimeError(
                f"pixel transport has {failures} escaped, aborted or unfinished photons"
            )
        row = {
            "seed": seed,
            "photons": count,
            "event_seconds": done - started,
            "photons_per_second": count / (done - started),
            "source_seconds": sourced - started,
            "transport_seconds": transported - sourced,
            "readout_seconds": done - transported,
            "detected": len(result.hits),
            "photoelectrons": len(event.photoelectrons.times),
            "failure_count": failures,
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "waveform_shape": list(event.waveforms.samples.shape),
        }
        print(json.dumps(row), flush=True)
        return row

    for count in args.counts:
        entry = {"photons": count, "warmup": run(count, 1729), "runs": []}
        report["counts"].append(entry)
        seeds = [901 + 1009 * repeat for repeat in range(args.repeats)]
        # Different photon histories can first trigger a tiny-population JIT
        # specialization. Warm each measured seed before steady-state timing.
        entry["seed_warmups"] = [run(count, seed) for seed in seeds]
        entry["runs"] = [run(count, seed) for seed in seeds]
        rates = [r["photons_per_second"] for r in entry["runs"]]
        entry.update(
            sustained_photons_per_second=sum(r["photons"] for r in entry["runs"])
            / sum(r["event_seconds"] for r in entry["runs"]),
            minimum_photons_per_second=min(rates),
            maximum_photons_per_second=max(rates),
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
