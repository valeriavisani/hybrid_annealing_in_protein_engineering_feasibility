"""QPU access time of the hybrid solver and its share of the total run time.

Usage:
    python -m scripts.plot_results.qpu_time [--min_res MIN_RES] [--max_res MAX_RES]

    --min_res / --max_res   only consider the instances with at least / at most this number of residues

Reads DATA_DIR/hybrid_annealing_results_{5,10}rot.csv (columns: num_res, num_rot, gs_energy [REU],
run_time [s], QPU_access_time [s]; one row per run) and prints, for 5 rotamers, 10 rotamers and
all the runs together:
  - the mean QPU access time per run [ms] (with minimum and maximum);
  - the mean share of the QPU in the total time, both as the mean of the ratios of the single runs
    (with minimum and maximum) and as the ratio of the sums (total QPU time / total run time).
It also saves a per-instance table to DATA_DIR/plot_results/qpu_time_per_instance.csv.
"""

import argparse
import pandas as pd

from ..config import DATA_DIR

FILES = ["hybrid_annealing_results_5rot.csv", "hybrid_annealing_results_10rot.csv"]
RUN_COL = "run_time [s]"
QPU_COL = "QPU_access_time [s]"


def summary(df, label):
    """Print the QPU time statistics of a set of runs."""
    qpu_ms = df[QPU_COL] * 1000
    frac = 100 * df[QPU_COL] / df[RUN_COL]
    print(f"\n{label}: {len(df)} runs on {df[['num_res', 'num_rot']].drop_duplicates().shape[0]} instances")
    print(f"  mean total time per run:        {df[RUN_COL].mean():.3f} s")
    print(f"  mean QPU time per run:          {qpu_ms.mean():.1f} ms  (min {qpu_ms.min():.1f}, max {qpu_ms.max():.1f})")
    print(f"  QPU / total time (mean):        {frac.mean():.2f} %  (min {frac.min():.2f}, max {frac.max():.2f})")
    print(f"  QPU / total time (sums):        {100 * df[QPU_COL].sum() / df[RUN_COL].sum():.2f} %")


def main():
    parser = argparse.ArgumentParser(
        description="QPU access time of the hybrid solver and its share of the total run time."
    )
    parser.add_argument("-min_res", "--min_res", type=int, default=None,
                        help="Minimum number of residues (default: no limit)")
    parser.add_argument("-max_res", "--max_res", type=int, default=None,
                        help="Maximum number of residues (default: no limit)")
    args = parser.parse_args()

    dfs = []
    for fname in FILES:
        df = pd.read_csv(DATA_DIR / fname)
        df["num_res"] = df["num_res"].astype(float).astype(int)
        df["num_rot"] = df["num_rot"].astype(float).astype(int)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True).dropna(subset=[RUN_COL, QPU_COL])
    if args.min_res is not None:
        df = df[df["num_res"] >= args.min_res]
    if args.max_res is not None:
        df = df[df["num_res"] <= args.max_res]

    for rot, g in df.groupby("num_rot"):
        summary(g, f"{rot} rotamers")
    summary(df, "All runs")

    # per-instance table: number of runs, mean total time, mean QPU time and mean QPU share
    per_inst = (df.assign(qpu_ms=df[QPU_COL] * 1000, qpu_pct=100 * df[QPU_COL] / df[RUN_COL])
                  .groupby(["num_rot", "num_res"])
                  .agg(n_runs=(RUN_COL, "size"), run_time_mean_s=(RUN_COL, "mean"),
                       qpu_ms_mean=("qpu_ms", "mean"), qpu_pct_mean=("qpu_pct", "mean"))
                  .reset_index())
    out = DATA_DIR / "plot_results" / "qpu_time_per_instance.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    per_inst.to_csv(out, index=False)
    print(f"\nPer-instance table saved to: {out}")


if __name__ == "__main__":
    main()