"""Log-log plot of throughput versus photons per launch from ``throughput_scaling.py`` results.

    python benchmarks/plot_scaling.py scaling.png [--legend "upper left"] LABEL=a100.json[:turing.json] ...

Each argument after the output is a curve label and one or two JSON files; the first file is drawn
solid, the second dashed (for example two GPUs), in the same color. ``--legend`` places the legend
(a matplotlib location; default lower right).
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator  # noqa: E402

# Okabe-Ito colors (distinguishable with color-vision deficiencies).
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
# Plain-number ticks (millions of photons/s) on the right-hand axis.
RIGHT_TICKS = (1, 2, 5, 10, 20, 40, 80, 120, 160, 200)
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
STYLES = (dict(ls="-", lw=2.2, marker="o", ms=4.8), dict(ls=(0, (4, 2.5)), lw=1.8, marker="s", ms=4.0))


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    legend_loc = "lower right"
    if specs[:1] == ["--legend"]:
        legend_loc, specs = specs[1], specs[2:]
    plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK,
                         "ytick.labelcolor": INK})
    fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=200)
    handles, gpus = [], []
    for i, spec in enumerate(specs):
        label, files = spec.split("=", 1)
        color = COLORS[i % len(COLORS)]
        for k, path in enumerate(files.split(":")):
            with open(path) as f:
                res = json.load(f)
            rows = res["rows"]
            ax.plot([r["n"] for r in rows], [r["photons_per_s"] / 1e6 for r in rows], color=color,
                    mew=0, solid_capstyle="round", zorder=3, **STYLES[k % 2])
            if len(gpus) <= k:
                gpus.append(res["gpu"].replace("NVIDIA ", "").replace("GeForce ", "").split("-")[0])
        handles.append(Line2D([], [], color=color, lw=2.6, label=label))
    for k, gpu in enumerate(gpus):
        handles.append(Line2D([], [], color=MUTED, mew=0, label=gpu, **STYLES[k % 2]))

    ax.set_xscale("log")
    ax.set_yscale("log")
    low, high = ax.get_ylim()
    ax.set_ylim(min(low, 0.85 * min(RIGHT_TICKS)), max(high, 1.12 * max(RIGHT_TICKS)))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, axis="x", which="major", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("photons per launch")
    ax.set_ylabel("photons / s (millions)")

    right = ax.twinx()
    right.set_yscale("log")
    right.set_ylim(ax.get_ylim())
    right.yaxis.set_minor_locator(NullLocator())
    right.yaxis.set_minor_formatter(NullFormatter())
    right.set_yticks(RIGHT_TICKS)
    right.set_yticklabels([str(v) for v in RIGHT_TICKS])
    right.tick_params(axis="y", length=3)
    for side in ("top", "left", "bottom"):
        right.spines[side].set_visible(False)
    for v in RIGHT_TICKS:
        ax.axhline(v, color="#d9d9d9", lw=0.7, ls=(0, (1, 2.5)), zorder=0)

    ax.legend(handles=handles, loc=legend_loc, frameon=False, fontsize=8.5, handlelength=2.6,
              labelspacing=0.45, borderaxespad=0.8)
    fig.tight_layout()
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
