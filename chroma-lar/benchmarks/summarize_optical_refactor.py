"""Summarize the refactor's independently recorded equivalence and GPU checks."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parents[2]
    directory = Path(__file__).parent / "optical_validation" / "maintainability"
    performance = json.loads((directory / "performance_30m.json").read_text())
    original = json.loads((directory.parent / "fast_full_sustained_30m_noise.json").read_text())
    equivalence = json.loads((directory / "equivalence.json").read_text())
    suites = ET.parse(directory / "tests_final.xml").getroot().findall("testsuite")
    tests = {key: sum(int(s.attrib[key]) for s in suites)
             for key in ("tests", "failures", "errors", "skipped")}
    hashes_match = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                       for name, digest in performance["source_sha256"].items())
    memcheck = "ERROR SUMMARY: 0 errors" in (directory / "memcheck.log").read_text()
    entry, prior = performance["counts"][0], original["counts"][0]
    for before, after in zip(prior["runs"], entry["runs"]):
        for field in ("seed", "photons", "detected", "photoelectrons", "diagnostics", "waveform_shape"):
            assert before[field] == after[field], field
    assert len(entry["runs"]) == len(prior["runs"]) == 5
    assert hashes_match and memcheck
    assert not any(tests[key] for key in ("failures", "errors", "skipped"))
    assert entry["sustained_photons_per_second"] >= 20_000_000
    summary = {"tests": tests, "equivalence": equivalence, "memcheck_zero_errors": memcheck,
               "current_runtime_matches_benchmark_hashes": hashes_match,
               "performance": performance,
               "prior_sustained_photons_per_second": prior["sustained_photons_per_second"],
               "all_five_benchmark_physics_summaries_unchanged": True}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Optical engine maintainability refactor", "",
        "Separated device table compilation, named state contracts, geometry queries, transport scheduling "
        "and shared readout. The fast path no longer constructs the legacy monochromatic transport engine. "
        "Public simulation entry points and NPZ output fields are retained. "
        "The source/bulk/surface physics kernels were unchanged.", "",
        "See [architecture and package feasibility](../../../docs/optical_architecture.md).", "",
        "## Verification", "",
        f"- **{tests['tests']} tests passed**, with zero failures, errors or skips ([log](tests_final.log)).",
        f"- **{equivalence['photons']:,} photons, three seeds, {equivalence['arrays_compared']} arrays/metadata records:** "
        "exact equality to the saved pre-refactor outputs. Includes terminal positions, directions, polarization, "
        "times, wavelengths, flags, IDs, hits, PE, noisy ADC waveforms and metadata "
        "([record](equivalence.json), [reproduction script](../../check_optical_refactor.py)).",
        "- **CUDA memcheck: zero errors** across the 16 fast optical tests and the device geometry parity test "
        "([log](memcheck.log)).",
        "- All five large benchmark events retained exactly the same photon counts, detections, PE counts, "
        "diagnostics and waveform dimensions as the earlier implementation.", "",
        "## Local A100 throughput", "",
        "Same Poisson source, full detector, synthetic calibration, nonzero electronics noise and ADC; "
        "source generation through CPU-readable hit/PE/waveform output. Setup, JIT warmup, file writes and "
        "optional terminal-state downloads are excluded.", "",
        "| Implementation | Sustained M photons/s | Slowest repeat | Fastest repeat |",
        "|---|---:|---:|---:|",
    ]
    for name, data in (("Before refactor (earlier run)", prior), ("After refactor", entry)):
        lines.append(f"| {name} | {data['sustained_photons_per_second']/1e6:.3f} | "
                     f"{data['minimum_photons_per_second']/1e6:.3f} | {data['maximum_photons_per_second']/1e6:.3f} |")
    photons = sum(r["photons"] for r in entry["runs"])
    seconds = sum(r["event_seconds"] for r in entry["runs"])
    lines += ["", f"Five measured events: {photons:,} photons in {seconds:.6f} s. "
              "These are separate measurements, not an interleaved performance experiment. "
              "Current runtime hashes match [the new benchmark record](performance_30m.json).", "",
              "The baseline capture was made before editing runtime sources. The final comparison was rerun "
              "after the refactor. The first regression run found a stale renamed-table reference in a test; "
              "that reference was corrected before the passing final suite.", "",
              "This establishes regression evidence for these configurations, not universal correctness or "
              "measured detector accuracy. The [original full validation report](../FAST_FULL_REPORT.md) "
              "retains the native-CUDA discrepancy and model limitations. No packaging migration was performed.", ""]
    (directory / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
