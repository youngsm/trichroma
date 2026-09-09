#!/usr/bin/env python3
"""Benchmark and plot Chroma/Triton throughput scaling.

Each point uses the same end-to-end timing scope as
``validate_triton_backend.py``: source generation through host-readable compact
hits.  The plotted rate is photons divided by the p95 elapsed time across the
replicates.  Per-run minima and maxima are shown as a light band.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = Path(__file__).with_name("validate_triton_backend.py")
DEFAULT_COUNTS = (
    500_000,
    1_000_000,
    2_000_000,
    3_000_000,
    5_000_000,
    7_500_000,
    10_000_000,
    12_500_000,
    15_000_000,
    30_000_000,
    60_000_000,
    120_000_000,
    180_000_000,
    240_000_000,
    300_000_000,
)

# Primary-source paper points.  These are deliberately plotted as unconnected
# reference markers: both GPU and detector workload differ from this study.
OPTICKS_REFERENCES = (
    {
        "series": "Opticks 2017 compute-only (GT 750M)",
        "photons": 500_000,
        "photons_per_second": 500_000 / 0.15,
        "source": "https://doi.org/10.1088/1742-6596/898/4/042001",
    },
    {
        "series": "Opticks 2017 compute-only (GT 750M)",
        "photons": 1_000_000,
        "photons_per_second": 1_000_000 / 0.28,
        "source": "https://doi.org/10.1088/1742-6596/898/4/042001",
    },
    {
        "series": "Opticks 2017 compute-only (GT 750M)",
        "photons": 1_000_000,
        "photons_per_second": 1_000_000 / 0.25,
        "source": "https://doi.org/10.1088/1742-6596/898/4/042001",
    },
    {
        "series": "Opticks 2025 full JUNO (RTX 5000 Ada)",
        "photons": 100_000_000,
        "photons_per_second": 100_000_000 / 10.42,
        "source": (
            "https://simoncblyth.github.io/env/presentation/"
            "opticks_20241021_krakow_chep2024.html"
        ),
        "context_source": "https://doi.org/10.1051/epjconf/202533701093",
    },
)


def _parse_seed_report(value: str) -> tuple[int, Path]:
    try:
        count_text, path_text = value.split("=", 1)
        count = int(count_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("report must be N=/path/to/report.json") from exc
    path = Path(path_text).expanduser().resolve()
    if count <= 0 or not path.is_file():
        raise argparse.ArgumentTypeError("report count must be positive and path must exist")
    return count, path


def _extract_point(report: dict[str, Any], count: int) -> dict[str, Any]:
    backends = {}
    for name in ("chroma", "triton"):
        result = report["backends"][name]
        run_rates = [float(run["photons_per_second"]) for run in result["runs"]]
        aggregate = result["aggregate"]
        backends[name] = {
            "p95_latency_photons_per_second": float(
                aggregate["photons_per_second_p95_latency"]
            ),
            "sustained_photons_per_second": float(
                aggregate["photons_per_second_sustained"]
            ),
            "run_photons_per_second": run_rates,
            "minimum_run_photons_per_second": min(run_rates),
            "maximum_run_photons_per_second": max(run_rates),
        }
    return {
        "photons": int(count),
        "replicates": int(len(report["backends"]["chroma"]["runs"])),
        "work_batch_photons": int(
            report["settings"].get("work_batch_photons", count)
        ),
        "triton_tiles_per_run": [
            int(run["tiles"]) for run in report["backends"]["triton"]["runs"]
        ],
        "backends": backends,
        "speedup_p95": (
            backends["triton"]["p95_latency_photons_per_second"]
            / backends["chroma"]["p95_latency_photons_per_second"]
        ),
    }


def _run_point(args: argparse.Namespace, count: int, report_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(VALIDATOR),
        "--backend",
        "both",
        "--nphotons",
        str(count),
        "--center",
        *(str(value) for value in args.center),
        "--replicates",
        str(args.replicates),
        "--history-length",
        str(args.history_length),
        "--warmup-photons",
        str(args.warmup_photons),
        "--work-batch-photons",
        str(args.work_batch_photons),
        "--tile-size",
        "auto",
        "--skip-step-diagnostics",
        "--json",
        str(report_path),
    ]
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="", flush=True)
    # Small batches are expected to miss the 5M acceptance target, for which
    # the validator returns one while still writing a valid benchmark report.
    if not report_path.is_file():
        raise RuntimeError(
            f"validator failed for {count:,} photons with status "
            f"{completed.returncode}"
        )
    with report_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_csv(path: Path, points: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "photons",
                "chroma_p95_photons_per_second",
                "triton_p95_photons_per_second",
                "triton_speedup_over_chroma",
            ]
        )
        for point in points:
            writer.writerow(
                [
                    point["photons"],
                    point["backends"]["chroma"][
                        "p95_latency_photons_per_second"
                    ],
                    point["backends"]["triton"][
                        "p95_latency_photons_per_second"
                    ],
                    point["speedup_p95"],
                ]
            )


def _plot(
    path: Path,
    points: list[dict[str, Any]],
    replicates: int,
    *,
    optimized_triton: bool = False,
    scale: str = "loglog",
) -> None:
    # Site home directories are read-only in batch jobs; keep font/cache files
    # in a writable location before importing matplotlib.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/triton-chroma-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.asarray([point["photons"] for point in points], dtype=np.float64) / 1.0e6
    if scale == "linear":
        fig, (ax, zoom) = plt.subplots(
            1,
            2,
            figsize=(13.0, 5.8),
            gridspec_kw={"width_ratios": (2.2, 1.0)},
            constrained_layout=True,
        )
        axes = (ax, zoom)
    else:
        fig, ax = plt.subplots(figsize=(10.5, 6.3), constrained_layout=True)
        zoom = None
        axes = (ax,)
    colors = {"chroma": "#d55e00", "triton": "#0072b2"}
    labels = {
        "chroma": "Chroma (CUDA)",
        "triton": (
            "Triton (corrected optimized)" if optimized_triton else "Triton"
        ),
    }
    for backend in ("chroma", "triton"):
        rates = np.asarray(
            [
                point["backends"][backend]["p95_latency_photons_per_second"]
                for point in points
            ]
        ) / 1.0e6
        low = np.asarray(
            [
                point["backends"][backend]["minimum_run_photons_per_second"]
                for point in points
            ]
        ) / 1.0e6
        high = np.asarray(
            [
                point["backends"][backend]["maximum_run_photons_per_second"]
                for point in points
            ]
        ) / 1.0e6
        for axis in axes:
            axis.fill_between(
                x, low, high, color=colors[backend], alpha=0.12, linewidth=0
            )
            axis.plot(
                x,
                rates,
                marker="o",
                markersize=6,
                linewidth=2.2,
                color=colors[backend],
                label=labels[backend],
            )

    reference_styles = {
        "Opticks 2017 compute-only (GT 750M)": ("D", "#6f4c9b"),
        "Opticks 2025 full JUNO (RTX 5000 Ada)": ("*", "#009e73"),
    }
    seen_reference_labels: set[str] = set()
    for reference in OPTICKS_REFERENCES:
        label = str(reference["series"])
        marker, color = reference_styles[label]
        legend_label = label if label not in seen_reference_labels else None
        seen_reference_labels.add(label)
        for axis in axes:
            axis.scatter(
                float(reference["photons"]) / 1.0e6,
                float(reference["photons_per_second"]) / 1.0e6,
                marker=marker,
                s=90 if marker == "*" else 45,
                facecolors=color if marker == "*" else "none",
                edgecolors=color,
                linewidths=1.5,
                zorder=5,
                label=legend_label if axis is ax else None,
            )

    for axis in axes:
        axis.axhline(
            5.0,
            color="#555555",
            linestyle="--",
            linewidth=1.5,
            label="5M photons/s target" if axis is ax else None,
        )
        axis.set_xscale("log" if scale == "loglog" else "linear")
        axis.set_yscale("log" if scale == "loglog" else "linear")
        axis.set_xlabel("Photons simulated per run (millions)")
        axis.grid(
            True,
            which="both" if scale == "loglog" else "major",
            color="#d0d0d0",
            linewidth=0.8,
            alpha=0.7,
        )
    ax.set_ylabel("Throughput (million photons/s)")
    if scale == "linear":
        ax.set_xlim(0.0, max(300.0, float(np.max(x)) * 1.03))
        ax.set_ylim(bottom=0.0)
        ax.set_title("Full 0–300M range")
        assert zoom is not None
        zoom.set_xlim(0.0, 16.0)
        zoom.set_ylim(bottom=0.0)
        zoom.set_title("Linear zoom: 0–15M")
    else:
        ax.set_xlim(float(np.min(x)) * 0.8, float(np.max(x)) * 1.2)
        ax.set_ylim(0.75, 40.0)
        ax.set_title("Log–log scaling: 0.5M–300M photons")
    fig.suptitle(
        f"Reflect3Wires throughput scaling on NVIDIA A100\n"
        f"Chroma/Triton: p95-latency rate across {replicates} runs; "
        f"{scale.replace('loglog', 'log–log')} axes; "
        f"shading = per-run range",
        fontsize=13,
    )
    ax.legend(frameon=True, loc="best", fontsize=8)
    ax.text(
        0.99,
        0.02,
        "Opticks markers: different GPUs and detector workloads",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _load_seed_points(paths: list[Path]) -> dict[int, dict[str, Any]]:
    points: dict[int, dict[str, Any]] = {}
    for path in paths:
        with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        default_replicates = int(payload.get("replicates", 0))
        for raw_point in payload.get("points", ()):
            point = dict(raw_point)
            point.setdefault("replicates", default_replicates)
            points[int(point["photons"])] = point
    return points


def _load_triton_scaling(path: Path) -> dict[int, dict[str, Any]]:
    with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("backend") != "triton":
        raise RuntimeError("optimized scaling input is not a Triton benchmark")
    return {int(point["photons"]): point for point in payload.get("points", ())}


def _portable_path(path: Path, *, relative_to: Path) -> str:
    """Prefer a repository-relative provenance path when one is available."""

    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(relative_to.resolve()))
    except ValueError:
        return str(resolved)


def _replace_triton_point(
    point: dict[str, Any], optimized: dict[str, Any]
) -> None:
    runs = list(optimized["runs"])
    rates = [float(run["photons_per_second"]) for run in runs]
    point["backends"]["triton"] = {
        "p95_latency_photons_per_second": float(
            optimized["p95_latency_photons_per_second"]
        ),
        "sustained_photons_per_second": float(
            optimized["sustained_photons_per_second"]
        ),
        "run_photons_per_second": rates,
        "minimum_run_photons_per_second": min(rates),
        "maximum_run_photons_per_second": max(rates),
    }
    point["replicates"] = len(runs)
    point["triton_tiles_per_run"] = [int(run["tiles"]) for run in runs]
    point["triton_tile_capacities"] = [
        int(run["tile_capacity"]) for run in runs
    ]
    point["speedup_p95"] = (
        point["backends"]["triton"]["p95_latency_photons_per_second"]
        / point["backends"]["chroma"]["p95_latency_photons_per_second"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="+", default=DEFAULT_COUNTS)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--center", type=float, nargs=3, default=(-1000.0, 0.0, 0.0))
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--warmup-photons", type=int, default=65_536)
    parser.add_argument("--work-batch-photons", type=int, default=15_000_000)
    parser.add_argument(
        "--scale",
        choices=("loglog", "linear"),
        default="loglog",
        help="axis scaling for the rendered throughput plot",
    )
    parser.add_argument(
        "--seed-report",
        action="append",
        default=[],
        type=_parse_seed_report,
        metavar="N=REPORT.json",
        help="reuse an existing both-backend validator report for one count",
    )
    parser.add_argument(
        "--seed-data",
        action="append",
        default=[],
        type=Path,
        metavar="SCALING.json",
        help="reuse matching points from an existing scaling JSON",
    )
    parser.add_argument(
        "--triton-data",
        type=Path,
        default=None,
        metavar="OPTIMIZED.json",
        help="replace seeded Triton values with a resident optimized-only sweep",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("throughput_scaling_loglog.png"),
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="durable per-count validator reports (default: OUTPUT-stem-reports)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="rerun counts even when a valid durable report already exists",
    )
    args = parser.parse_args()
    counts = sorted(set(int(value) for value in args.counts))
    if (
        not counts
        or counts[0] <= 0
        or args.replicates < 2
        or args.work_batch_photons <= 0
    ):
        parser.error("counts must be positive and replicates must be at least two")

    output = args.output.expanduser().resolve()
    report_dir = (
        args.report_dir.expanduser().resolve()
        if args.report_dir is not None
        else output.with_name(output.stem + "_reports")
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    seeded = dict(args.seed_report)
    seeded_points = _load_seed_points(args.seed_data)
    points = []
    for count in counts:
        if count in seeded_points:
            points.append(seeded_points[count])
            print(f"reused scaling point: {count:,}", flush=True)
            continue
        if count in seeded:
            report_path = seeded[count]
        else:
            report_path = report_dir / f"report-{count}.json"
            if args.no_resume or not report_path.is_file():
                report = _run_point(args, count, report_path)
                points.append(_extract_point(report, count))
                continue
        with report_path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        if int(report["settings"]["nphotons_per_replicate"]) != count:
            raise RuntimeError(f"checkpoint count mismatch in {report_path}")
        points.append(_extract_point(report, count))
        print(f"resumed validator report: {report_path}", flush=True)

    optimized_triton = (
        None
        if args.triton_data is None
        else _load_triton_scaling(args.triton_data)
    )
    if optimized_triton is not None:
        missing = [
            int(point["photons"])
            for point in points
            if int(point["photons"]) not in optimized_triton
        ]
        if missing:
            raise RuntimeError(
                "optimized Triton data is missing counts: "
                + ", ".join(str(value) for value in missing)
            )
        for point in points:
            _replace_triton_point(
                point, optimized_triton[int(point["photons"])]
            )

    data_path = (
        args.json.expanduser().resolve()
        if args.json is not None
        else output.with_suffix(".json")
    )
    csv_path = (
        args.csv.expanduser().resolve()
        if args.csv is not None
        else output.with_suffix(".csv")
    )
    payload = {
        "metric": "photons_per_second_p95_latency",
        "axes": {
            "x": "log" if args.scale == "loglog" else "linear",
            "y": "log" if args.scale == "loglog" else "linear",
        },
        "replicates": int(args.replicates),
        "work_batch_photons": int(args.work_batch_photons),
        "center_mm": [float(value) for value in args.center],
        "external_references": list(OPTICKS_REFERENCES),
        "triton_policy": (
            "corrected_optimized" if optimized_triton is not None else "paired_run"
        ),
        "optimized_triton_source": (
            None
            if args.triton_data is None
            else _portable_path(args.triton_data, relative_to=ROOT)
        ),
        "points": points,
    }
    with data_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _write_csv(csv_path, points)
    _plot(
        output,
        points,
        args.replicates,
        optimized_triton=optimized_triton is not None,
        scale=args.scale,
    )
    print(f"plot: {output}")
    print(f"data: {data_path}")
    print(f"csv:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
