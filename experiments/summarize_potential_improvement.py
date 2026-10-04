import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import norm

from utils.stratified_running_stat import StratifiedRunningStats

PATHWISE_TOLERANCE = 1e-6
SUMMARIZED_COSTS = ('potential_gap', 'partial_information_relaxation_cost', 'penalized_information_relaxation_cost',
                    'prefix_partial_information_relaxation_cost', 'tail_error_bound', 'run_time_seconds')


def load_records(folder):
    records = {}
    for path in sorted(glob.glob(os.path.join(folder, '*.jsonl'))):
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if 'potential_gap' in record:
                    records[record['uid']] = record
    return list(records.values())


def summarize(records, alpha=0.05):
    stats = {key: StratifiedRunningStats() for key in SUMMARIZED_COSTS}
    for record in records:
        for key, stat in stats.items():
            stat.record(record[key], record['path_weight'], record['path_stratum'])
    gap = stats['potential_gap']
    standard_error = float(np.sqrt(gap.variance_of_mean()))
    weights = np.array([record['path_weight'] for record in records], dtype=float)
    negative = np.array([record['potential_gap'] < 0 for record in records], dtype=float)
    return {
        'M': gap.n,
        **{key: stats[key].mean for key in SUMMARIZED_COSTS},
        'standard_error': standard_error,
        'upper_confidence_limit': max(0.0, gap.mean + norm.ppf(1 - alpha) * standard_error),
        'alpha': alpha,
        'negative_fraction': float(weights @ negative / weights.sum()),
        'pathwise_violations': sum(
            record['schedule_penalized_information_relaxation_cost']
            < record['penalized_information_relaxation_cost'] - PATHWISE_TOLERANCE for record in records),
        'min_pathwise_slack': min(
            record['schedule_penalized_information_relaxation_cost']
            - record['penalized_information_relaxation_cost'] for record in records),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description='Summarize potential-improvement replications.')
    parser.add_argument('folder')
    parser.add_argument('--alpha', type=float, default=0.05)
    args = parser.parse_args(argv)
    summary = summarize(load_records(args.folder), alpha=args.alpha)
    for key, value in summary.items():
        print(f"{key}: {value}")
    return summary


if __name__ == '__main__':
    main()
