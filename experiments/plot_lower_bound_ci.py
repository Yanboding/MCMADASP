"""Plot per-run zero-penalty vs penalized lower bound with 95% CIs.

Reads the CSV produced by ``experiments.summarize_output_logs`` and draws, for
each run (one row per slurm .out file), the zero-penalty subproblem objective
mean and the final-iteration penalized objective mean, each with its 95% CI
half-window as an error bar. X axis is the run index in CSV (filename) order.

Usage:
    python -m experiments.plot_lower_bound_ci [--csv PATH] [--save-file PATH]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "results", "OUTPUT",
                           "subproblem_objective_summary.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--save-file", default=None,
                        help="default: <csv dir>/lower_bound_ci_per_run.svg")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    save_file = args.save_file or os.path.join(os.path.dirname(args.csv),
                                               "lower_bound_ci_per_run.svg")
    df = df.sort_values("final_mean").reset_index(drop=True)
    rank = np.arange(len(df))

    fig, ax_band = plt.subplots(1, 1, figsize=(14, 8))

    # CI as smooth bands over runs sorted by penalized mean — no per-run bar clutter
    ax_band.fill_between(rank, df.final_mean - df.final_half_window,
                         df.final_mean + df.final_half_window,
                         alpha=0.25, color="tab:blue")
    ax_band.plot(rank, df.final_mean, color="tab:blue", linewidth=1.5,
                 label="penalized lower bound (final iteration)")
    ax_band.fill_between(rank, df.zero_penalty_mean - df.zero_penalty_half_window,
                         df.zero_penalty_mean + df.zero_penalty_half_window,
                         alpha=0.25, color="tab:orange")
    ax_band.plot(rank, df.zero_penalty_mean, color="tab:orange", linewidth=1.5,
                 label="zero-penalty lower bound")
    ax_band.set_ylabel("lower bound (mean, shaded 95% CI)")
    ax_band.set_xlabel("run rank (sorted by penalized mean)")
    ax_band.set_title(f"zero-penalty vs penalized lower bound, {len(df)} runs "
                      "sorted by penalized mean")
    ax_band.legend(loc="upper left")
    ax_band.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_file)
    fig.savefig(os.path.splitext(save_file)[0] + ".png", dpi=150)
    print(f"saved {save_file}")


if __name__ == "__main__":
    main()
