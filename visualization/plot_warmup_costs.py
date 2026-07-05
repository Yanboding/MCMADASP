"""Plot per-period ("one-time") costs over time to find the warm-up plateau.

Reads all *.jsonl result files in a results folder, groups records by
policy_id, averages the per-period costs across sample paths, and plots
mean cost vs time for the requested policies.

Usage:
    PYTHONPATH=. python visualization/plot_warmup_costs.py \
        [--results-dir experiments/results/case_study_099_mixture_geometric_proposal_095] \
        [--policies row_gen_alp myopic] \
        [--confidence 0.95] \
        [--output warmup_costs.png]
"""
import argparse
import json
from pathlib import Path
from utils import RunningStats

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RESULTS_DIR = 'experiments/results/case_study_099_mixture_geometric_proposal_095'
DEFAULT_POLICIES = ['row_gen_alp', 'myopic']
POLICY_DISPLAY_NAMES = {'row_gen_alp': 'ALP', 'myopic': 'Myopic'}


def load_costs_by_policy(results_dir, policies):
    """Return {policy_id: 2-D array of shape (num_paths, num_periods)}."""
    costs_by_policy = {policy_id: [] for policy_id in policies}
    for jsonl_file in sorted(Path(results_dir).glob('*.jsonl')):
        with open(jsonl_file) as fh:
            for line in fh:
                record = json.loads(line)
                policy_id = record.get('policy_id')
                if policy_id in costs_by_policy and 'costs' in record:
                    costs_by_policy[policy_id].append(record['costs'])
    for policy_id, costs in costs_by_policy.items():
        if not costs:
            raise ValueError(f'No records found for policy {policy_id!r} in {results_dir}')
        min_length = min(len(c) for c in costs)
        costs_by_policy[policy_id] = np.array([c[:min_length] for c in costs])
    return costs_by_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--policies', nargs='+', default=DEFAULT_POLICIES)
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='confidence level for the interval band')
    parser.add_argument('--warmup', type=int, default=300,
                        help='draw a vertical red line at this period to mark the warm-up cutoff')
    parser.add_argument('--output', default=None,
                        help='save figure to this path instead of showing it')
    args = parser.parse_args()

    costs_by_policy = load_costs_by_policy(args.results_dir, args.policies)

    fig, ax = plt.subplots(figsize=(10, 6))
    for policy_id, costs in costs_by_policy.items():
        num_paths, num_periods = costs.shape
        period_stats = [RunningStats() for _ in range(num_periods)]
        for path_costs in costs:
            for stat, cost in zip(period_stats, path_costs):
                stat.record(cost)
        means = np.array([stat.mean for stat in period_stats])
        half_windows = np.array([stat.half_window(args.confidence) for stat in period_stats])

        time = np.arange(num_periods)
        (line,) = ax.plot(time, means, linewidth=0.9,
                          label=POLICY_DISPLAY_NAMES.get(policy_id, policy_id))
        ax.fill_between(time, means - half_windows, means + half_windows,
                        color=line.get_color(), alpha=0.2)

    ax.axvline(args.warmup, color='red', linestyle='-', linewidth=1.5,
               label=f'Warm-up period = {args.warmup}')

    ax.set_xlabel('Time period')
    ax.set_ylabel('One-time cost (mean over sample paths)')
    ax.set_title('Warmup Period Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f'Saved figure to {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
