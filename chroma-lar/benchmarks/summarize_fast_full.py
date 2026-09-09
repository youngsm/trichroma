"""Assemble measured throughput and retain all correctness discrepancies."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).parent/"optical_validation")
    parser.add_argument("--allow-historical-runtime", action="store_true",
                        help="summarize archived evidence after runtime sources have changed")
    args = parser.parse_args()
    p = args.directory
    perf = json.loads((p/"fast_full_sustained_30m_noise.json").read_text())
    entry = perf["counts"][0]
    photons = sum(r["photons"] for r in entry["runs"])
    seconds = sum(r["event_seconds"] for r in entry["runs"])
    suites = ET.parse(p/"fast_full_tests_final.xml").getroot().findall("testsuite")
    tests = {key: sum(int(s.attrib[key]) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    memcheck = (p/"fast_full_memcheck_final.log").read_text()
    comparison = json.loads((p/"fast_full_final_comparison/comparison.json").read_text())
    failures = [row for row in comparison["checks"] if not row["passed"]]
    lossless = {backend: json.loads((p/f"fast_full_lossless_cuda/{backend}.json").read_text()) for backend in ("cuda", "triton")}
    for a, b in zip(lossless["cuda"]["runs"], lossless["triton"]["runs"]):
        assert a["seed"] == b["seed"] and a["input_sha256"] == b["input_sha256"]
    lost = {backend: sum(r["bulk_absorbed"] for r in data["runs"]) for backend, data in lossless.items()}
    source_root = Path(__file__).resolve().parents[2]
    hashes_match = all(hashlib.sha256((source_root/name).read_bytes()).hexdigest() == digest
                       for name, digest in perf["source_sha256"].items())
    summary = {"performance": perf, "tests": tests, "memcheck_zero_errors": "ERROR SUMMARY: 0 errors" in memcheck,
               "current_runtime_matches_benchmark_hashes": hashes_match,
               "legacy_cuda_comparison": comparison, "legacy_comparison_failures": failures,
               "matched_lossless_detector": lossless, "lossless_bulk_absorbed": lost,
               "universal_correctness_certified": False,
               "calibration": "Synthetic software fixture; no measured detector calibration or observations supplied."}
    (p/"fast_full_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    assert hashes_match or args.allow_historical_runtime, "runtime changed; use --allow-historical-runtime for archived evidence"
    assert not tests["failures"] and not tests["errors"] and not tests["skipped"]
    assert summary["memcheck_zero_errors"] and lost["triton"] == 0
    assert photons/seconds >= 20000000
    lines = ["# Accelerated full optical simulation", "",
        f"**{photons/seconds/1e6:.2f} million input photons/s sustained** on the local {perf['device']}. "
        f"Five independent Poisson source events emitted {photons:,} photons and completed in {seconds:.6f} s. "
        f"The slowest repeat was {entry['minimum_photons_per_second']/1e6:.2f}M/s; "
        f"the fastest whole-event repeat was {entry['maximum_photons_per_second']/1e6:.2f}M/s. No Modal was used.", "",
        "## What was timed", "",
        "The synchronized event timer includes Poisson photon yield, GPU LAr emission spectrum/time/position/direction/polarization generation, "
        "wavelength-dependent bulk absorption and polarized Rayleigh scattering, dielectric/default-surface transport, group-velocity flight time, "
        "TPB absorption/reemission spectrum/delay/escape, photocathode detection, PMT collection/TTS/charge, pulse superposition, "
        "0.4 ADC RMS electronics noise, and ADC clipping/quantization. Optical hits, PE hits and waveform arrays are CPU-readable before the timer stops.", "",
        "All **162 PMTs, six wire planes and every box face** are retained. Each event call produces "
        "two waveform events, including a zero-energy event, with shape `[2, 162, 16000]`. "
        "The scintillation deposition is at (-1000, 0, 0) mm; the configured yield gives a mean 30M photons. "
        "These are synthetic calibration values, not measured detector predictions.", "",
        "Detector construction, compilation/warmup, file compression/writes, optional CPU copies of every dead-photon state, "
        "and benchmark housekeeping between calls are excluded. All five timed repeats are retained. "
        "Sustained throughput is total actual input photons divided by total event seconds; it is not an instantaneous kernel rate.", "",
        "| Seed | Actual photons | Event time (s) | Throughput (M photons/s) |", "|---:|---:|---:|---:|"]
    for row in entry["runs"]:
        lines.append(f"| {row['seed']} | {row['photons']:,} | {row['event_seconds']:.6f} | {row['photons_per_second']/1e6:.3f} |")
    lines += ["", "Every timed event had zero escaped, unfinished, step-limited, aborted or nonfinite photons. "
        f"Maximum Torch allocation was {max(r['peak_torch_allocated_bytes'] for r in entry['runs'])/1e9:.2f} GB; "
        f"maximum reserved memory was {max(r['peak_torch_reserved_bytes'] for r in entry['runs'])/1e9:.2f} GB.", "",
        "Raw timing, stage breakdowns, environment, calibration fingerprints and runtime hashes are in "
        "[fast_full_sustained_30m_noise.json](fast_full_sustained_30m_noise.json). "
        "The earlier complete-geometry version measured 20.34M/s without electronics noise "
        "([raw run](fast_full_sustained_30m.json)); separating the PMT wall bounds removed unnecessary instance searches. "
        "The user's earlier approximately 25M/s large-batch result used the monochromatic specialization. "
        "The new measurement includes the additional optical/response stages. The original slow generic mesh measurements remain "
        "in [REPORT.md](REPORT.md); this is a different geometry engine, so those timings do not establish a pure kernel speedup.", "",
        "## Correctness findings and checks", "",
        f"- **{tests['tests']} regression tests passed**, with zero failures, errors or skips. "
        "This includes source CPU/GPU agreement, full-width Philox seeds/IDs, spectral collision handoff, group velocity, "
        "WLS spectrum/time/material side, PMT response, fractional-time waveform parity, empty events, and geometry traversal. "
        "The separated PMT bounds produced exactly the same intersection results as the complete original union search on 20,000 rays spanning both halves.",
        "- **CUDA memcheck: zero errors**, covering all 16 accelerated source/transport/response tests, including the reflective-wire detector invariant.",
        "- Fixed analytic boundary positioning: a float32 flight at metre-scale coordinates could place a reflected photon inside a 75 micrometre radius wire. "
        "The corrected engine reconstructs the cylinder/box surface in FP64 and rounds its boundary position into the outgoing material. "
        "It retains the completed flight distance/time and interaction probabilities. No photon is terminated to increase throughput.",
        "- Retained both detector halves and every box face. Coincident PMT/wall boundaries can reach the outer cavity under the existing geometry's numerical conventions; "
        "pruning the opposite half is unsafe if a cavity ray returns. The fast bulk shortcut separately verifies that its LAr region is empty.", "",
        "### Independent detector comparison", "",
        "The same persisted VUV input photons were propagated by the existing native CUDA analytic-wire implementation and the corrected Triton engine: "
        "500,000 photons per seed, three seeds, 1.5M photons per backend. Detection fractions, detected wavelength distributions and channel distributions "
        "passed their per-seed thresholds. **One of 12 checks failed:** seed 11's TPB reemission fraction differs by 6.586 standard errors "
        "against a predeclared six-standard-error threshold. The other seeds' reemission pulls are 5.337 and 5.138. "
        "The failed check is retained in [comparison.json](fast_full_final_comparison/comparison.json); thresholds were not relaxed.", "",
        "The original CUDA reference also has the wire-rounding defect. An independent detector test makes steel an immediate bulk absorber, "
        "sets wire/active/cathode surfaces perfectly reflecting, enables PMT detection and keeps the outer cavity absorbing. "
        "Every photon should either detect or terminate on the cavity; **none should bulk-absorb in steel**. "
        "On identical three-seed input batches:", "",
        "| Backend | Photons | Spurious bulk absorptions | Fraction |", "|---|---:|---:|---:|",
        f"| Original CUDA, analytic wires | 300,000 | {lost['cuda']:,} | {lost['cuda']/300000:.3%} |",
        f"| Corrected spectral Triton | 300,000 | {lost['triton']:,} | {lost['triton']/300000:.3%} |", "",
        "See the [CUDA](fast_full_lossless_cuda/cuda.json) and [Triton](fast_full_lossless_cuda/triton.json) records with matching input hashes. "
        "A separate 300,000-photon test at three detector positions also had zero false bulk losses "
        "([results](fast_full_lossless_final.json)). An earlier diagnostic made the outer cavity perfectly reflecting as well; "
        "that can trap light indefinitely outside the sensor enclosure, so its assumption that every photon must eventually detect was invalid. "
        "Those intermediate failures and traces remain in `fast_full_lossless*.json`.", "",
        "The legacy reference's demonstrated numerical losses are consistent with the corrected engine's higher detector/WLS yields. "
        "This does not prove that every part of the detector-yield difference has one cause. Exact legacy equivalence is therefore **not claimed**. "
        "Original CUDA also lacks WLS delay and group-velocity timing; those added laws are covered by independent component tests, not CUDA timing parity. "
        "The previous generic triangle-wire errors remain unresolved in that separate engine.", "",
        "## Reproduce and use", "", "From the workspace root with the local CUDA Python environment:", "", "```bash",
        "PYTHONPATH=chroma-lite:chroma-lar python chroma-lar/benchmarks/benchmark_fast_full.py \\",
        "  --depositions --counts 30000000 --repeats 5 \\",
        "  --calibration chroma-lar/benchmarks/optical_validation/full_detector_synthetic_calibration_noise.json \\",
        "  --output full_throughput.json", "```", "",
        "Use `FastOpticalSimulation(calibration)` or `run_optical_simulation.py --backend triton-fast` for deposition or prepared-photon input. "
        "See the [API and CLI guide](../../docs/full_optical_simulation.md). The CLI smoke output is `fast_full_cli_output.npz`.", "",
        "The source adapter currently accepts the negative-x detector component. The entire detector geometry is retained during transport. "
        "The source population stays on the GPU; oversized requests are not automatically tiled. Bulk reemission, weighted transport, "
        "complex films and unsupported surface models are rejected. Bulk reemission is not silently approximated: this calibration specifies no bulk reemitting components. "
        "TPB is an effective surface model; charged-particle propagation and microscopic coating transport are outside this implementation. "
        "Measured optical calibration and detector data are still required for physical validation; no universal 100% correctness certification is claimed.", "",
        "Machine-readable evidence: [summary](fast_full_summary.json), [regressions](fast_full_tests_final.xml), "
        "[memcheck](fast_full_memcheck_final.log). " +
        ("Recorded runtime hashes match the current files." if hashes_match else
         "This is archived validation of the recorded runtime hashes. Sources have since changed; "
         "see the [maintainability refactor validation](maintainability/REPORT.md) for subsequent checks."), ""]
    (p/"FAST_FULL_REPORT.md").write_text("\n".join(lines))
    print(p/"FAST_FULL_REPORT.md")


if __name__ == "__main__":
    main()
