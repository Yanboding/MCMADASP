"""Plot per-index mean and 95% CI of ALP/penalty coefficients across replications.

Each ``<job>_<rep>.jsonl`` file in the results directory holds one record with a
``coefficients`` list. Coefficients are stacked across replications and, for each
index, the sample mean and t-based 95% CI are plotted (x = coefficient index,
y = value). A zoomed panel excludes extreme-magnitude indices so the bulk of the
coefficients stays readable.

Usage:
    python -m experiments.plot_coefficients_ci [--result-dir DIR] [--zoom-pct 99]
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DEFAULT_RESULT_DIR = os.path.join(
    os.path.dirname(__file__), "results", "case_study_099_mixture_geometric_proposal_095"
)


def load_coefficients(result_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(result_dir, "*.jsonl"))):
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if "coefficients" in record:
                    rows.append(record["coefficients"])
    if not rows:
        raise ValueError(f"no coefficients found in {result_dir}")
    lengths = {len(r) for r in rows}
    if len(lengths) > 1:
        raise ValueError(f"inconsistent coefficient lengths across files: {sorted(lengths)}")
    return np.array(rows)


def mean_and_half_window(coefficients, confidence=0.95):
    n = coefficients.shape[0]
    means = coefficients.mean(axis=0)
    sem = coefficients.std(axis=0, ddof=1) / np.sqrt(n)
    t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return means, t_val * sem


def save(fig, save_file):
    fig.tight_layout()
    fig.savefig(save_file)
    fig.savefig(os.path.splitext(save_file)[0] + ".png", dpi=150)
    print(f"saved {save_file}")


def plot_all(means, half_window, save_file):
    index = np.arange(len(means))
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.fill_between(index, means - half_window, means + half_window,
                    alpha=0.3, color="tab:blue", label="95% CI")
    ax.plot(index, means, linewidth=1, color="tab:blue", label="mean")
    ax.set_xlabel("coefficient index")
    ax.set_ylabel("coefficient value")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title(f"coefficient mean and 95% CI across {N_REPS} replications")
    save(fig, save_file)


def plot_widest(means, half_window, top_k, save_file):
    widest = np.sort(np.argsort(half_window)[::-1][:top_k])
    positions = np.arange(len(widest))
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.errorbar(positions, means[widest], yerr=half_window[widest],
                fmt="o", capsize=5, linewidth=1.5, color="tab:blue")
    for x, i in zip(positions, widest):
        ax.annotate(f"{means[i]:.1f}±{half_window[i]:.1f}",
                    (x, means[i]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)
    ax.set_xticks(positions)
    ax.set_xticklabels(widest)
    ax.set_xlabel("coefficient index")
    ax.set_ylabel("coefficient value")
    ax.grid(alpha=0.3)
    ax.set_title(f"top {top_k} coefficients by CI half-window (mean ± 95% CI)")
    save(fig, save_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--top-k", type=int, default=10,
                        help="print the top-k indices by CI half-window")
    args = parser.parse_args()

    coefficients = load_coefficients(args.result_dir)
    global N_REPS
    N_REPS = coefficients.shape[0]
    means, half_window = mean_and_half_window(coefficients)
    print(f"{N_REPS} replications, {coefficients.shape[1]} coefficients")

    # relative variation: CI half-window as a fraction of |mean|
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(np.abs(means) > 0, half_window / np.abs(means), np.nan)
    order = np.argsort(half_window)[::-1][: args.top_k]
    print(f"top {args.top_k} indices by CI half-window:")
    print(f"{'index':>6} {'mean':>14} {'ci_half':>12} {'ci/|mean|':>10}")
    for i in order:
        print(f"{i:>6} {means[i]:>14.4f} {half_window[i]:>12.4f} {relative[i]:>10.4f}")
    print(f"median ci/|mean| over all indices: {np.nanmedian(relative):.4f}")

    plot_all(means, half_window, os.path.join(args.result_dir, "coefficients_ci_all.svg"))
    plot_widest(means, half_window, args.top_k,
                os.path.join(args.result_dir, f"coefficients_ci_top{args.top_k}.svg"))


if __name__ == "__main__":
    main()
