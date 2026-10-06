import json

import numpy as np
from scipy.stats import norm

from experiments.summarize_potential_improvement import load_records, summarize


def _record(uid, gap, weight, stratum):
    return {'uid': uid, 'group_id': 'g', 'penalized_information_relaxation_cost': 10.0,
            'partial_information_relaxation_cost': gap + 10.0, 'potential_gap': gap,
            'prefix_partial_information_relaxation_cost': gap + 9.0, 'tail_error_bound': 1.0,
            'schedule_penalized_information_relaxation_cost': 10.5,
            'run_time_seconds': 2.0, 'path_weight': weight, 'path_stratum': stratum}


def test_iid_matches_the_section_formula():
    rng = np.random.default_rng(0)
    G = rng.normal(5.0, 3.0, 50)
    summary = summarize([_record(f'u{i}', float(g), 1.0 / 50, 0) for i, g in enumerate(G)], alpha=0.05)
    sigma = np.sqrt(((G - G.mean()) ** 2).sum() / (50 * 49))
    np.testing.assert_allclose(summary['potential_gap'], G.mean())
    np.testing.assert_allclose(summary['standard_error'], sigma)
    np.testing.assert_allclose(summary['upper_confidence_limit'], max(0.0, G.mean() + norm.ppf(0.95) * sigma))
    assert summary['M'] == 50


def test_stratified_mean_uses_path_weights():
    records = [_record('a', 1.0, 0.25, 0), _record('b', 3.0, 0.25, 0),
               _record('c', 10.0, 0.25, 1), _record('d', 14.0, 0.25, 1)]
    summary = summarize(records)
    np.testing.assert_allclose(summary['potential_gap'], 7.0)
    np.testing.assert_allclose(summary['standard_error'], np.sqrt(0.25 * 2.0 / 2 + 0.25 * 8.0 / 2))


def test_upper_limit_is_clipped_at_zero_and_negatives_are_kept():
    summary = summarize([_record('a', -5.0, 0.5, 0), _record('b', -7.0, 0.5, 0)])
    assert summary['potential_gap'] == -6.0 and summary['upper_confidence_limit'] == 0.0
    assert summary['negative_fraction'] == 1.0


def test_load_records_deduplicates_and_filters(tmp_path):
    lines = [_record('a', 1.0, 0.5, 0), _record('a', 1.0, 0.5, 0), {'uid': 'x', 'policy_id': 'row_gen_alp'}]
    (tmp_path / '1.jsonl').write_text(''.join(json.dumps(line) + '\n' for line in lines))
    assert [record['uid'] for record in load_records(str(tmp_path))] == ['a']


def test_pathwise_violations_are_counted():
    records = [_record('a', 1.0, 0.5, 0), _record('b', 2.0, 0.5, 0)]
    records[1]['schedule_penalized_information_relaxation_cost'] = 9.0
    summary = summarize(records)
    assert summary['pathwise_violations'] == 1 and summary['min_pathwise_slack'] == -1.0


def test_improvement_percentage_is_the_ratio_of_means_with_delta_method_error():
    records = [_record('a', 2.0, 0.5, 0), _record('b', 6.0, 0.5, 0)]
    records[1]['penalized_information_relaxation_cost'] = 30.0
    summary = summarize(records)
    ratio = 4.0 / 20.0
    residuals = np.array([(2.0 - ratio * 10.0) / 20.0, (6.0 - ratio * 30.0) / 20.0])
    np.testing.assert_allclose(summary['improvement_percentage'], 100 * ratio)
    np.testing.assert_allclose(summary['improvement_percentage_standard_error'],
                               100 * np.sqrt(residuals.var(ddof=1) / 2))
