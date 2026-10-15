"""Quality plot: gap from the reference energy versus instance size.

Produces a single figure with two side-by-side panels, (a) 5 rotamers and (b) 10 rotamers per
residue. For each method it shows the percentage gap from the reference energy: markers are the
mean over the runs, error bars are the min/max over the runs, and the vertical axis is symlog
(linear within +-SYMLOG_LINTHRESH_PCT %, logarithmic beyond). Negative values mean an energy
below the reference. Black rings mark the instances whose reference is a certified optimum.
A single legend is placed at the top, centered on the figure.

Usage:
    python -m scripts.plot_results.quality_plot_feasibility [--shared_y]

    --shared_y   use the same y scale in both panels (default: independent scales)

Input files (in DATA_DIR), for each number of rotamers {rot} in ROTAMERS. All files contain
the columns "num_res" and "num_rot"; each file has one row per run, except the reference (one row per instance):
    reference_results_{rot}rot.csv            reference: Gurobi, default configuration, 4 h time limit
                                            (columns "gs_energy" and "status"; status "OPTIMAL"
                                            marks the instances where optimality was certified)
    gurobi_results_{rot}rot_10s.csv         Gurobi, custom configuration, 10 s   (column "gs_energy")
    gurobi_results_{rot}rot_1000s.csv       Gurobi, custom configuration, 1000 s (column "gs_energy")
    hybrid_annealing_results_{rot}rot.csv   D-Wave hybrid solver, 5 s            (column "gs_energy [REU]")

Output:
    DATA_DIR/plot_results/quality_plot_feasibility.png   (300 dpi)
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullLocator

from ..config import DATA_DIR

# ----------------------------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------------------------
ROTAMERS = (5, 10)   # number of rotamers per residue: panel (a), panel (b)

# optional filter on the number of residues (None = no limit)
MIN_RES = None
MAX_RES = None

# instances where Gurobi (10 s) did not return a meaningful solution, per number of rotamers
EXCLUDED_GUROBI_10S_RESIDUES = {5: set(), 10: {197}}

EXPECTED_RUNS = 5            # runs expected per instance and method (a warning is printed otherwise)
SYMLOG_LINTHRESH_PCT = 0.1   # linear region of the y axis within +-0.1 %
Y_HEADROOM_FACTOR = 1.8      # multiplicative margin of the y limits beyond the data

FIG_SIZE = (7.16, 2.75)
FONT_SIZE = 8
LINE_WIDTH = 0.9
MARKER_SIZE = 2.8
CAP_SIZE = 1.5
ERR_LINE_WIDTH = 0.6
RING_SIZE = 40               # size of the rings that mark certified optima

METHODS = {   # internal label -> (file, energy column, legend label, color)
    "Gurobi 10s": ("gurobi_results_{rot}rot_10s.csv", "gs_energy", "Gurobi (10 s)", "tab:blue"),
    "Gurobi 1000s": ("gurobi_results_{rot}rot_1000s.csv", "gs_energy", "Gurobi (1000 s)", "tab:green"),
    "Hybrid Annealing 5s": ("hybrid_annealing_results_{rot}rot.csv", "gs_energy [REU]", "Hybrid Annealing (5 s)", "tab:red"),
}
# ----------------------------------------------------------------------------------------------


def load_and_normalize(path, num_rot):
    """Load a results file, keep the rows with the given number of rotamers and add the number of variables."""
    df = pd.read_csv(path)
    df["num_res"] = df["num_res"].astype(float).astype(int)
    df["num_rot"] = df["num_rot"].astype(float).astype(int)
    df = df[df["num_rot"] == num_rot]
    if MIN_RES is not None:
        df = df[df["num_res"] >= MIN_RES]
    if MAX_RES is not None:
        df = df[df["num_res"] <= MAX_RES]
    df = df.copy()
    df["num_var"] = df["num_res"] * df["num_rot"]
    return df


def add_run_gaps(df, reference_lookup, energy_col):
    """Add the percentage gap of every run from the reference: (E - E_ref) / |E_ref| * 100."""
    df = df.copy()
    df["ref"] = df["num_var"].map(reference_lookup)
    df["gap_pct"] = (df[energy_col] - df["ref"]) / df["ref"].abs() * 100
    return df


def style_symlog_axis(ax, lo, hi, linthresh):
    """Symlog y axis with plain tick labels (0.1, 1, 10, ...) and limits taken from the data plus a margin."""
    ax.set_yscale("symlog", linthresh=linthresh)
    lo2 = lo * Y_HEADROOM_FACTOR if lo < -1e-9 else -0.5 * linthresh
    hi2 = hi * Y_HEADROOM_FACTOR if hi > 1e-9 else 0.5 * linthresh
    ax.set_ylim(lo2, hi2)
    ticks = [0.0]
    for k in range(0, 9):
        v = linthresh * 10 ** k
        for t in (v, -v):
            if lo2 <= t <= hi2:
                ticks.append(t)
    ax.set_yticks(sorted(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", "−")))


def load_panel_data(num_rot):
    """Load the data of one panel: per-method runs with their gaps, the certified optima and the instance sizes."""
    df_optimal = load_and_normalize(DATA_DIR / f"reference_results_{num_rot}rot.csv", num_rot)
    reference = df_optimal.set_index("num_var")["gs_energy"].to_dict()
    is_optimal = df_optimal["status"].astype(str).str.strip().str.upper() == "OPTIMAL"
    certified_res = sorted(int(r) for r in df_optimal.loc[is_optimal, "num_res"].unique())
    print(f"[{num_rot} rot] Certified optimum (OPTIMAL): {certified_res}")

    runs = {}
    for label, (fname, energy_col, _, _) in METHODS.items():
        df = load_and_normalize(DATA_DIR / fname.format(rot=num_rot), num_rot)
        if label == "Gurobi 10s":
            excluded = EXCLUDED_GUROBI_10S_RESIDUES.get(num_rot, set())
            if df["num_res"].isin(excluded).any():
                print(f"[{num_rot} rot] Gurobi 10s excluded at {sorted(excluded)} residues.")
            df = df[~df["num_res"].isin(excluded)]
        runs[label] = add_run_gaps(df, reference, energy_col)
        for num_res, g in runs[label].groupby("num_res"):
            if len(g) != EXPECTED_RUNS:
                print(f"WARNING: {num_rot} rot, {label}, {num_res} residues: "
                      f"{len(g)} runs (expected {EXPECTED_RUNS}).")

    residues = sorted(set().union(*[set(df["num_res"].unique()) for df in runs.values()]))
    return runs, certified_res, residues


def draw_panel(ax, runs, certified_res, residues, title, show_ylabel=True):
    """Draw one panel: mean gap with min/max error bars for each method, the reference line and the certified optima."""
    for label, df in runs.items():
        _, _, _, color = METHODS[label]
        g = df.groupby("num_res")["gap_pct"]
        mean, lo, hi = g.mean(), g.min(), g.max()
        ax.errorbar(mean.index, mean.values,
                    yerr=[np.maximum(mean - lo, 0).values, np.maximum(hi - mean, 0).values],
                    fmt="o-", color=color, linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                    elinewidth=ERR_LINE_WIDTH, capsize=CAP_SIZE, capthick=ERR_LINE_WIDTH,
                    alpha=0.9, zorder=4)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, zorder=2)
    ax.scatter(certified_res, np.zeros(len(certified_res)), s=RING_SIZE, facecolors="none",
               edgecolors="black", linewidths=0.8, zorder=6)

    rotation = 45 if len(residues) > 15 else 0
    ax.set_xticks(residues)
    ax.set_xticklabels(residues, rotation=rotation, ha="right" if rotation else "center",
                       rotation_mode="anchor", fontsize=FONT_SIZE - 2.5)
    ax.tick_params(axis="y", labelsize=FONT_SIZE - 1)
    ax.set_xlabel("Number of residues")
    if show_ylabel:
        ax.set_ylabel("Gap from reference (%)")
    ax.set_title(title, fontsize=FONT_SIZE, pad=3)
    ax.grid(True, alpha=0.3, linewidth=0.4)


def main():
    parser = argparse.ArgumentParser(
        description="Plot the gap from the reference energy versus instance size (feasibility study)."
    )
    parser.add_argument("-shared_y", "--shared_y", action="store_true",
                        help="Same y scale in both panels (default: independent scales)")
    args = parser.parse_args()

    plt.rcParams.update({"font.size": FONT_SIZE, "axes.linewidth": 0.6})

    panels = [load_panel_data(r) for r in ROTAMERS]

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.80, wspace=0.20)   # fixed margins

    # y limits: one per panel, or shared between the panels
    def y_bounds(runs):
        all_pct = pd.concat([df["gap_pct"] for df in runs.values()])
        return min(0.0, all_pct.min()), max(0.0, all_pct.max())

    bounds = [y_bounds(p[0]) for p in panels]
    if args.shared_y:
        lo = min(b[0] for b in bounds)
        hi = max(b[1] for b in bounds)
        bounds = [(lo, hi)] * len(panels)

    for i, (ax, (runs, certified_res, residues), rot, (lo, hi)) in enumerate(
            zip(axes, panels, ROTAMERS, bounds)):
        letter = "ab"[i]
        draw_panel(ax, runs, certified_res, residues,
                   title=f"({letter}) {rot} rotamers",
                   show_ylabel=(i == 0 or not args.shared_y))
        style_symlog_axis(ax, lo, hi, SYMLOG_LINTHRESH_PCT)

    # single legend, centered on the figure, on one row
    handles = []
    for label, (_, _, legend_label, color) in METHODS.items():
        handles.append(Line2D([], [], color=color, marker="o", markersize=MARKER_SIZE,
                              linewidth=LINE_WIDTH, label=legend_label))
    handles += [Line2D([], [], color="black", linestyle="--", linewidth=0.8, label="Reference (Gurobi, 4 h)"),
                Line2D([], [], color="black", marker="o", markersize=5, markerfacecolor="none",
                       linestyle="none", markeredgewidth=0.8, label="Certified optimum")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
               fontsize=FONT_SIZE - 1, frameon=False, handlelength=1.6, columnspacing=1.4,
               labelspacing=0.2, handletextpad=0.5)

    out_dir = DATA_DIR / "plot_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "quality_plot_feasibility"
    fig.savefig(f"{out_path}.png", dpi=300)
    plt.close(fig)
    print(f"Plot saved to: {out_path}.png")


if __name__ == "__main__":
    main()