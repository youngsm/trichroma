"""Summarize matched input/output transport timings without discarding outliers."""
import argparse
import json
from pathlib import Path


def valid(record):
    return not any(record[key] for key in ("nonfinite", "unfinished", "aborted", "escaped"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path("chroma-lar/benchmarks/optical_validation/prepared_transport"))
    args = parser.parse_args()
    p = args.directory
    cuda = json.loads((p / "cuda.json").read_text())
    triton = json.loads((p / "triton.json").read_text())
    rows = []
    all_valid = True
    for a, b in zip(cuda["counts"], triton["counts"]):
        if a["photons"] != b["photons"] or len(a["runs"]) != len(b["runs"]):
            raise ValueError("comparison populations do not match")
        for x, y in zip([a["warmup"]] + a["runs"], [b["warmup"]] + b["runs"]):
            if x["seed"] != y["seed"] or len({x["input_sha256"], y["input_sha256"],
                                            x["resident_input_sha256"], y["resident_input_sha256"]}) != 1:
                raise ValueError("host/resident inputs do not match")
            all_valid &= valid(x) and valid(y)
        rows.append({"photons": a["photons"], "repeats": len(a["runs"]),
                     "cuda_median": a["median"], "triton_median": b["median"],
                     "triton_transport_speedup": a["median"]["transport_seconds"]/b["median"]["transport_seconds"],
                     "triton_prepared_event_speedup": a["median"]["event_seconds"]/b["median"]["event_seconds"],
                     "cuda_peak_transport_photons_per_second": a["maximum"]["transport_photons_per_second"],
                     "triton_peak_transport_photons_per_second": b["maximum"]["transport_photons_per_second"],
                     "cuda_detected_total": sum(x["detected"] for x in a["runs"]),
                     "triton_detected_total": sum(x["detected"] for x in b["runs"]),
                     "cuda_transport_range_seconds": [a["minimum"]["transport_seconds"], a["maximum"]["transport_seconds"]],
                     "triton_transport_range_seconds": [b["minimum"]["transport_seconds"], b["maximum"]["transport_seconds"]]})
    initial_failure = json.loads((p / "cuda_initial_failure.json").read_text())
    summary = {"device": cuda["device"], "identical_host_and_resident_inputs": True,
               "matching_output_fields": cuda["output_fields"], "output_bytes_per_photon": 44,
               "completed_series_all_runs_finite_and_terminated": all_valid,
               "initial_cuda_failure": initial_failure, "overall_correctness_signoff": False,
               "timings": rows,
               "comparison_limit": "Same prepared photons and output fields; existing geometry accelerations and RNG algorithms remain different. No exact legacy/physical equivalence claim."}
    (p / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    lines = ["# Matched transport comparison", "",
             "The earlier 1.96× end-to-end advantage is not a transport speedup. With identical prepared inputs, "
             "the measured median transport advantage at one million photons is 1.30×. At 100,000 photons CUDA is 2.73× faster.", "",
             "Both backends ran alone on the local A100-SXM4-40GB, with one discarded full-size warm-up and five measured repeats per size. "
             "The source is the same 450 nm photon population in a 30 mm cube centered at (-1000, 0, 0) mm. "
             "Input arrays are persisted once and their SHA-256 hashes are checked on the host and after GPU upload for both backends. "
             "Both paths download every photon's position, direction, polarization, time and history: exactly 44 bytes per input photon.", "",
             "Transport means synchronized wall time from resident state to completed propagation, including queue allocation and host scheduling. "
             "Prepared-event time adds allocation/upload and the matching output download. Source generation, input-file reads, setup and input-hash auditing are excluded. "
             "No 30-million-photon run was performed.", "",
             "| Photons | CUDA transport | Triton transport | CUDA throughput | Triton throughput | Triton speedup |",
             "|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        a, b = r["cuda_median"], r["triton_median"]
        lines.append(f"| {r['photons']:,} | {a['transport_seconds']:.4f} s | {b['transport_seconds']:.4f} s | "
                     f"{a['transport_photons_per_second']/1e6:.3f} M photons/s | {b['transport_photons_per_second']/1e6:.3f} M photons/s | {r['triton_transport_speedup']:.2f}× |")
    lines += ["", "| Photons | CUDA prepared event | Triton prepared event | Triton speedup |",
              "|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['photons']:,} | {r['cuda_median']['event_seconds']:.4f} s | "
                     f"{r['triton_median']['event_seconds']:.4f} s | {r['triton_prepared_event_speedup']:.2f}× |")
    largest = rows[-1]
    lines += ["", f"At one million photons, the fastest observed resident-transport rates were "
              f"{largest['cuda_peak_transport_photons_per_second']/1e6:.3f} M photons/s for CUDA and "
              f"{largest['triton_peak_transport_photons_per_second']/1e6:.3f} M photons/s for Triton. "
              "These are maxima over five repeats at this size, not established hardware peak rates.", "",
              "All measured repeats are retained, including the 1.533-second CUDA outlier at one million photons. "
              "CUDA uses its persistent XORWOW states and nondeterministic work queue; Triton uses its existing Philox streams. "
              "Identical inputs therefore do not imply identical random trajectories.", "",
              "## Validity and remaining differences", "",
              "An initial CUDA series stopped at one million photons, seed 1910, after finding one aborted/escaped photon and three nonfinite vector components. "
              "The harness was changed to preserve invalid states and continue recording rather than stop before writing the record. "
              "The complete CUDA series reported above was then run from fresh setup. All completed-series runs were finite and terminated, "
              "but the initial failure remains recorded in `cuda_initial_failure.json` and the initial partial timings remain in `cuda_initial_partial.json`. "
              "This is not a correctness sign-off.", "",
              "CUDA uses the repository's existing full detector with analytic wires. Triton uses its existing instance and analytic-wire acceleration, "
              "including the reachability specialization for a source on the negative-x side. The Triton harness calls the existing resident propagation loop "
              "and retains all final states; it does not compact surviving photons into the optional reservoir. "
              "No transport physics code was changed. Geometry acceleration, numerical edge behavior, and RNG algorithms still differ. "
              "Both benchmarks are 450 nm transport, without the new VUV/WLS/TTS/waveform pipeline.", "",
              "Reproduce with `benchmark_prepared_transport.py --backend prepare`, followed by separate `--backend triton` and `--backend cuda` processes "
              "in their respective local environments. Raw complete timings are in `triton.json` and `cuda.json`; input hashes are in `inputs.json`.", ""]
    (p / "REPORT.md").write_text("\n".join(lines))
    print(p / "REPORT.md")


if __name__ == "__main__":
    main()
