"""Log-log plot of throughput versus photons per launch from ``throughput_scaling.py`` results.

    python benchmarks/plot_scaling.py scaling.png LABEL=a100_lar.json[:turing_lar.json] ...

Each argument after the output is a curve label and one or two JSON files; the first file is drawn
solid, the second dashed (for example two GPUs), in the same color.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

# Plain-number ticks (millions of photons/s) on the right-hand axis.
RIGHT_TICKS = (40, 80, 120, 160, 200)


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    handles, gpus = [], []
    for i, spec in enumerate(specs):
        label, files = spec.split("=", 1)
        for k, path in enumerate(files.split(":")):
            with open(path) as f:
                res = json.load(f)
            rows = res["rows"]
            ax.plot([r["n"] for r in rows], [r["photons_per_s"] / 1e6 for r in rows], ls="-" if k == 0 else "--",
                    marker="o" if k == 0 else "s", ms=4, lw=1.6, color=colors[i % len(colors)])
            if len(gpus) <= k:
                gpus.append(res["gpu"].replace("NVIDIA ", "").replace("GeForce ", ""))
        handles.append(Line2D([], [], color=colors[i % len(colors)], lw=2, label=label))
    for k, gpu in enumerate(gpus):
        handles.append(Line2D([], [], color="0.3", ls="-" if k == 0 else "--", marker="o" if k == 0 else "s",
                              ms=4, label=gpu))
    ax.set_xscale("log")
    ax.set_yscale("log")
    low, high = ax.get_ylim()
    ax.set_ylim(low, max(high, 1.1 * max(RIGHT_TICKS)))
    right = ax.twinx()
    right.set_yscale("log")
    right.set_ylim(ax.get_ylim())
    right.yaxis.set_minor_locator(NullLocator())
    right.yaxis.set_minor_formatter(NullFormatter())
    right.set_yticks(RIGHT_TICKS)
    right.set_yticklabels([str(v) for v in RIGHT_TICKS])
    ax.set_xlabel("photons per launch")
    ax.set_ylabel("photons / s (millions)")
    ax.set_title("Transport throughput (unweighted, photons on the GPU)")
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", alpha=0.12)
    ax.legend(handles=handles, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
