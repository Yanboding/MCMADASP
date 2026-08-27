"""Tests for the information-relaxation lower-bound aggregation.

Run from the repo root:  python -m test.test_ir_lower_bound_report
"""
import json
import os
import tempfile

import numpy as np
from scipy import stats

from experiments.new_result_aggregration import SimulateEvaluationResult


def _ir_record(uid, zero, penalized, weight, stratum):
    return {
        'uid': uid, 'group_id': 'g', 'experiment_name': 'x', 'mutate_val': 512,
        'warm_up_periods': 0, 'path_weight': weight, 'path_stratum': stratum,
        'policy_id': 'information_relaxation_only',
        'agent_name': 'information_relaxation_only',
        'zero_information_relaxation_cost': zero,
        'penalized_information_relaxation_cost': penalized,
        'gap_to_zero_information_relaxation': 0.0,
        'gap_to_penalized_information_relaxation': 0.0,
    }


def _load(records):
    from experiments import get_config_by_type
    env = get_config_by_type('toy').env
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, '1.jsonl'), 'w') as handle:
            for record in records:
                handle.write(json.dumps(record) + '\n')
        return SimulateEvaluationResult(tmp, '[0-9]*.jsonl', env, is_reuse=False)


def test_ir_only_records_uniform_match_numpy_and_delta_method():
    rng = np.random.default_rng(0)
    n = 200
    zero = rng.normal(1000, 100, n)
    penalized = zero + rng.normal(50, 80, n)
    records = [_ir_record(f'u{i}', float(zero[i]), float(penalized[i]), 1.0 / n, 0)
               for i in range(n)]
    # a penalty-coefficient training record in the same folder must be skipped
    records.append({'uid': 'train', 'coefficients': [1.0], 'tight_penalized_lower_bound': 1.0})
    ser = _load(records)
    key = ('g', 512)
    t = stats.t.ppf(0.975, n - 1)
    zs = ser.zero_penalized_information_relaxation_cost[key]
    ps = ser.penalized_information_relaxation_cost[key]
    ds = ser.gap_to_information_relaxation[key]
    assert zs.n == ps.n == ds.n == n
    assert np.isclose(zs.mean, zero.mean()) and np.isclose(ps.mean, penalized.mean())
    assert np.isclose(zs.half_window(0.95), t * zero.std(ddof=1) / np.sqrt(n))
    assert np.isclose(ps.half_window(0.95), t * penalized.std(ddof=1) / np.sqrt(n))
    diff = penalized - zero
    assert np.isclose(ds.mean, diff.mean())
    assert np.isclose(ds.half_window(0.95), t * diff.std(ddof=1) / np.sqrt(n))
    # relative improvement: ratio of means with a delta-method CI
    pct, hw, n_pairs = ser.information_relaxation_improvement('g', 512, 0.95)
    dm, bm = diff.mean(), zero.mean()
    cov = np.cov(np.vstack([diff, zero])) / n
    var = cov[0, 0] / bm ** 2 - 2 * dm * cov[0, 1] / bm ** 3 + dm ** 2 * cov[1, 1] / bm ** 4
    assert n_pairs == n
    assert np.isclose(pct, dm / bm * 100)
    assert np.isclose(hw, stats.norm.ppf(0.975) * np.sqrt(var) * 100)


def test_ir_only_records_two_strata_weighted_mean():
    # stratum 0: 2 paths with total weight 0.5; stratum 1: 4 paths with total weight 0.5
    zero = [100.0, 120.0, 10.0, 12.0, 14.0, 16.0]
    penalized = [130.0, 150.0, 11.0, 13.0, 15.0, 17.0]
    weights = [0.25, 0.25, 0.125, 0.125, 0.125, 0.125]
    strata = [0, 0, 1, 1, 1, 1]
    ser = _load([_ir_record(f'u{i}', zero[i], penalized[i], weights[i], strata[i])
                 for i in range(6)])
    zs = ser.zero_penalized_information_relaxation_cost[('g', 512)]
    expected_zero_mean = 0.5 * np.mean(zero[:2]) + 0.5 * np.mean(zero[2:])
    assert np.isclose(zs.mean, expected_zero_mean)
    pct, hw, n_pairs = ser.information_relaxation_improvement('g', 512, 0.95)
    diff = np.array(penalized) - np.array(zero)
    expected_diff_mean = 0.5 * diff[:2].mean() + 0.5 * diff[2:].mean()
    assert n_pairs == 6
    assert np.isclose(pct, expected_diff_mean / expected_zero_mean * 100)
    assert hw > 0


if __name__ == '__main__':
    test_ir_only_records_uniform_match_numpy_and_delta_method()
    test_ir_only_records_two_strata_weighted_mean()
    print('All IR lower-bound report tests passed.')
