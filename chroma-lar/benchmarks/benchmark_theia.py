"""Water Cherenkov reference-path throughput with explicit synthetic readout.

This measures the general flattened-mesh spectral engine, not the LAr fast
engine or camera rendering. Region discovery is measured at construction;
the current general transport engine does not consume its certificates yet.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from chroma.triton.examples.theia import build_theia, cherenkov_photons
from chroma.triton.optical_response import PMTResponse, TabulatedCDF
from chroma.triton.regions import compile_regions
from chroma.triton.spectral import SpectralScene, SpectralSimulation
from optical_parity import compare_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=float, nargs="+", default=[25500.0])
    parser.add_argument("--coverage", type=float, default=0.81)
    parser.add_argument("--diameter", type=float, default=508.0)
    parser.add_argument("--counts", type=int, nargs="+", default=[100000, 1000000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-count", type=int, default=10000)
    parser.add_argument("--reference-seeds", type=int, nargs="+", default=[44, 101, 2027])
    parser.add_argument(
        "--transport-reference",
        type=Path,
        help="reuse completed transport comparisons from an unchanged scene/runtime; retain the original report",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        len(args.size) not in (1, 3)
        or min(args.counts) <= 0
        or args.repeats < 2
        or args.reference_count <= 0
    ):
        parser.error(
            "provide one radius or three dimensions, positive counts, at least two repeats"
        )
    import torch
    import triton
    from chroma.triton.digitizer_kernels import digitize_gpu

    if not torch.cuda.is_available():
        parser.error("a local CUDA GPU is required")
    started = time.perf_counter()
    size = args.size[0] if len(args.size) == 1 else tuple(args.size)
    fixture = build_theia(size, coverage=args.coverage, diameter=args.diameter)
    print(f"Built {fixture.channel_count:,} sensors", flush=True)
    built = time.perf_counter()
    regions = compile_regions(fixture.primitives())
    certified = time.perf_counter()
    print(
        f"Certified {len(regions.regions)} regions in {certified-built:.3f}s; compiling mesh",
        flush=True,
    )
    scene = SpectralScene.compile(fixture.detector(), wavelengths=np.arange(300.0, 651.0, 5.0))
    simulation = SpectralSimulation(scene, tile_size=1 << 20)
    torch.cuda.synchronize()
    print(f"Uploaded {scene.host.triangle_count:,} triangles", flush=True)
    response = PMTResponse(
        tts_sigma=2.0,
        transit_time=30.0,
        collection_efficiency=0.9,
        charge_cdf=TabulatedCDF.from_pdf([0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 0.4, 1.0, 0.4, 0.0]),
    )
    digitizer = dict(
        event_indices=[0, 1],
        channel_count=fixture.channel_count,
        start_ns=0.0,
        sample_period_ns=1.0,
        sample_count=1024,
        pulse_times_ns=[0.0, 2.0, 5.0, 10.0, 18.0],
        pulse_adc_per_pe=[0.0, 4.0, 2.0, 0.5, 0.0],
        baseline=200.0,
        noise_rms=0.4,
        adc_bits=12,
    )
    report = {
        "host": platform.node(),
        "device": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "configuration": vars(args)
        | {
            "output": str(args.output),
            "transport_reference": (
                str(args.transport_reference) if args.transport_reference else None
            ),
        },
        "scope": "CPU Cherenkov source sampling, host-to-device input, general mesh spectral transport, terminal-state download, PMT response, GPU pulse superposition, CPU noise/ADC and host-readable hits/PE/waveforms",
        "excluded": "geometry construction, compilation warmup and file writes",
        "backend": "general flattened-mesh SpectralSimulation; region certificates not used in transport",
        "calibration_status": "illustrative water detector; not a Theia calibration",
        "optics": "bundled chroma.demo.optics water/glass/r7081hqe tables; black enclosure and PMT backs; scaled SNO demo PMT mesh with diameter from configuration",
        "source": "fixed-count beta=1 Cherenkov track, 1 m long along +Z at center, 300--650 nm, wavelength-dependent cone and polarization",
        "readout": {
            "tts_sigma_ns": 2.0,
            "transit_time_ns": 30.0,
            "collection_efficiency": 0.9,
            "charge_pdf": [[0.0, 0.0], [0.5, 0.4], [1.0, 1.0], [1.5, 0.4], [2.0, 0.0]],
            "digitizer": digitizer,
        },
        "geometry": {
            "channels": fixture.channel_count,
            "triangles": scene.host.triangle_count,
            "shared_sensor_triangles": len(fixture.sensor.mesh.triangles),
            "packing": fixture.packing,
        },
        "construction_seconds": {
            "fixture": built - started,
            "regions": certified - built,
            "mesh_compile_upload": time.perf_counter() - certified,
        },
        "compiled_regions": regions.as_dict(),
        "scene_fingerprint": scene.fingerprint,
        "counts": [],
    }
    root = Path(__file__).resolve().parents[2]
    paths = [
        "chroma-lite/chroma/triton/examples/theia.py",
        "chroma-lite/chroma/triton/primitives.py",
        "chroma-lite/chroma/triton/regions.py",
        "chroma-lite/chroma/triton/bvh.py",
        "chroma-lite/chroma/triton/bvh_kernels.py",
        "chroma-lite/chroma/triton/boundary.py",
        "chroma-lite/chroma/triton/boundary_kernels.py",
        "chroma-lite/chroma/triton/spectral.py",
        "chroma-lite/chroma/triton/spectral_kernels.py",
        "chroma-lite/chroma/demo/optics.py",
        "chroma-lite/chroma/demo/sno_pmt_reduced.txt",
        "chroma-lite/chroma/triton/optical_response.py",
        "chroma-lite/chroma/triton/digitizer_kernels.py",
        "chroma-lar/benchmarks/benchmark_theia.py",
        "chroma-lar/benchmarks/optical_parity.py",
    ]
    report["source_sha256"] = {
        p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Independent CPU transport comparison on this actual detector size, not
    # just the smaller pytest fixture. JIT/reference time is outside throughput.
    report["cpu_comparisons"] = []
    checks = [(1000, 44), (10000, 44), (10000, 101), (10000, 2027)] + [
        (args.reference_count, seed) for seed in args.reference_seeds
    ]
    if args.transport_reference:
        if args.transport_reference.resolve() == args.output.resolve():
            raise ValueError("retain the reference report at a distinct path")
        data = args.transport_reference.read_bytes()
        prior = json.loads(data)
        for key in (
            "device",
            "python",
            "torch",
            "triton",
            "scene_fingerprint",
            "source",
            "geometry",
            "optics",
        ):
            if prior[key] != report[key]:
                raise ValueError(f"transport reference differs: {key}")
        for key in ("size", "coverage", "diameter"):
            if prior["configuration"][key] != report["configuration"][key]:
                raise ValueError(f"transport configuration differs: {key}")
        # The harness changed only its causal readout template and this reuse
        # facility. Every imported implementation/comparator hash must match.
        for path, sha in report["source_sha256"].items():
            if (
                path != "chroma-lar/benchmarks/benchmark_theia.py"
                and prior["source_sha256"].get(path) != sha
            ):
                raise ValueError(f"transport implementation differs: {path}")
        rows = {(r["photons"], r["seed"]): r for r in prior["cpu_comparisons"]}
        for check in dict.fromkeys(checks):
            row = rows[check]
            if row["flag_mismatches"] or row["source_seed"] != check[1] + 7:
                raise ValueError(
                    "transport reference contains a failed or differently seeded comparison"
                )
            report["cpu_comparisons"].append(row)
        report["transport_comparison_provenance"] = {
            "method": "reused previously completed comparisons; not rerun in this invocation",
            "path": str(args.transport_reference),
            "sha256": hashlib.sha256(data).hexdigest(),
            "prior_harness_sha256": prior["source_sha256"][
                "chroma-lar/benchmarks/benchmark_theia.py"
            ],
            "readout_change": "pulse times shifted +2 ns to make the synthetic template causal; transport unchanged",
        }
        print(f"Reused {len(report['cpu_comparisons'])} verified transport comparisons", flush=True)
        checks = []
    for count, seed in dict.fromkeys(checks):
        photons = cherenkov_photons(count, seed=seed + 7)
        reference_started = time.perf_counter()
        reference = SpectralSimulation(scene, backend="reference_bvh").simulate(
            photons, seed=seed, max_steps=256
        )
        reference_seconds = time.perf_counter() - reference_started
        actual = simulation.simulate(photons, seed=seed, max_steps=256)
        comparison = compare_results(
            actual, reference, evidence=args.output.with_name(f"theia_failure_{count}_{seed}")
        )
        comparison.update(
            photons=count, seed=seed, source_seed=seed + 7, cpu_seconds=reference_seconds
        )
        report["cpu_comparisons"].append(comparison)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Full-size CPU/GPU comparison passed: {count:,} photons, seed {seed}", flush=True)

    def run(count, seed):
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        photons = cherenkov_photons(count, seed=seed)
        sourced = time.perf_counter()
        result = simulation.simulate(photons, seed=seed, max_steps=256)
        transported = time.perf_counter()
        pe = response.apply(result.hits, seed=seed)
        waveform = digitize_gpu(pe, seed=seed, **digitizer)
        torch.cuda.synchronize()
        done = time.perf_counter()
        flags = result.final_state["flags"]
        failures = int(np.count_nonzero(flags & ((1 << 31) | (1 << 30) | 1)))
        if failures:
            raise RuntimeError(
                f"water transport has {failures} escaped, aborted or unfinished photons"
            )
        row = {
            "seed": seed,
            "photons": count,
            "event_seconds": done - begin,
            "photons_per_second": count / (done - begin),
            "source_seconds": sourced - begin,
            "transport_seconds": transported - sourced,
            "readout_seconds": done - transported,
            "detected": len(result.hits),
            "photoelectrons": len(pe.times),
            "failure_count": failures,
            "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
            "waveform_shape": list(waveform.samples.shape),
        }
        print(json.dumps(row), flush=True)
        return row

    for count in args.counts:
        entry = {"photons": count, "warmup": run(count, 1729), "runs": []}
        report["counts"].append(entry)
        seeds = [901 + 1009 * repeat for repeat in range(args.repeats)]
        # A different trajectory can first reach nactive==1 and trigger an
        # additional Triton specialization. Warm every measured seed before
        # timing so this compilation is excluded as advertised.
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
