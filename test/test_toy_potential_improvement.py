"""Targeted contract tests for the toy lower-bound plot's third curve.

Synthetic saved results only: these tests never construct an optimizer.
"""
import copy
import json

import numpy as np
import pytest
from scipy.stats import t


from experiments import new_result_aggregration as aggregation


def _function(name):
    assert hasattr(aggregation, name), f'Missing required aggregation function: {name}'
    return getattr(aggregation, name)


def _record(index, gain, mutate_val=.5, experiment_name='toy_lb_occupancy_alp'):
    # Equal probability strata, with unequal draws per stratum, require weights.
    weights = [.5, 1 / 6, 1 / 6, 1 / 6]
    zero = [80., 100., 120., 140.]
    anchor = [2., 4., 6., 8.]
    control = [-5., 20., -15., 25.]
    paths = []
    for i, (w, z, a, cv) in enumerate(zip(weights, zero, anchor, control)):
        fitted, alp = z + a + gain + cv, z + 3.
        paths.append(dict(
            path_id=i, stratum=0 if i == 0 else 1, weight=w,
            termination_time=i + 2, policy_control_variate=cv,
            policy_penalty_at_reference=a,
            ir={name: dict(objective=value, status='OPTIMAL')
                for name, value in [('fitted', fitted), ('alp', alp), ('zero', z)]},
            raw_gap_fitted_minus_zero=fitted - z,
            raw_gap_fitted_minus_alp=fitted - alp,
            cv_adjusted_fitted_ir_cost=fitted - cv,
            cv_adjusted_gap_fitted_minus_zero=fitted - cv - z - a,
            cv_adjusted_gap_fitted_minus_alp=fitted - cv - alp))
    lower = sum(p['weight'] * p['cv_adjusted_fitted_ir_cost'] for p in paths)
    return dict(
        uid=f'{index + 1:032x}', experiment_name=experiment_name,
        mutate_val=mutate_val, sample_path_number=4,
        coefficients=[1.] + [0.] * 16, training_time_seconds=2., paths=paths,
        tight_penalized_lower_bound=sum(p['weight'] * p['ir']['fitted']['objective'] for p in paths),
        full_objective=lower, objective_lower_bound=lower,
        objective_upper_bound=lower + 5e-7,
        master_status='OPTIMAL', termination_reason='absolute_gap_tolerance')


def _write(tmp_path, monkeypatch, records, name='toy_lb_occupancy_alp'):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / 'experiments' / 'results' / name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / '1.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    return path


def _training_row(records):
    return _function('training_potential_improvement_rows')(
        'toy_lb_occupancy_alp', expected_batches=3, expected_paths=4)


def test_saved_upper_bounds_and_anchor_corrections_are_aggregated_by_training_batch(tmp_path, monkeypatch):
    records = [_record(i, gain) for i, gain in enumerate([10., 20., 40.])]
    # Evaluation rows in the same folder must not count as training repetitions.
    evaluation = dict(uid='eval', policy_id='information_relaxation_only', mutate_val=.5)
    path = _write(tmp_path, monkeypatch, records + [evaluation])
    before = path.read_bytes()
    row = _training_row(records)[.5]
    expected = np.asarray([10., 20., 40.]) + 5e-7
    assert row['potential_improvement_mean'] == pytest.approx(expected.mean(), abs=1e-10)
    assert row['potential_improvement_variance_of_mean'] == pytest.approx(expected.var(ddof=1) / 3)
    assert row['potential_training_batches'] == 3
    assert row['potential_training_paths'] == 4
    assert row['potential_training_domain'] == 'unrestricted'
    assert 'toy_lb_occupancy_alp' in str(row['potential_training_source'])
    assert path.read_bytes() == before


def test_mutations_remain_separate_and_incoming_floor_provenance_is_preserved(tmp_path, monkeypatch):
    floor_name = 'toy_lb_occupancy_alp_floor_repeated_20261019'
    records = [_record(i, gain) for i, gain in enumerate([10., 20., 40.])]
    records += [_record(i + 3, gain, .9, floor_name) for i, gain in enumerate([3., 6., 9.])]
    path = _write(tmp_path, monkeypatch, records)
    original = copy.deepcopy(records)
    result = _training_row(records)
    assert set(result) == {.5, .9}
    assert result[.5]['potential_training_domain'] == 'unrestricted'
    assert result[.9]['potential_training_domain'] == 'alp_floor_constrained'
    assert floor_name in str(result[.9]['potential_training_source'])
    assert result[.9]['potential_improvement_mean'] == pytest.approx(6. + 5e-7, abs=1e-10)
    assert [json.loads(line) for line in path.read_text().splitlines()] == original


@pytest.mark.parametrize('bad_case', [
    'missing_batch', 'duplicate_uid', 'missing_upper', 'nan_upper',
    'nonoptimal_master', 'nonoptimal_path', 'nan_zero', 'nan_anchor',
    'negative_weight', 'bad_weight_total', 'duplicate_path_id', 'missing_path',
])
def test_incomplete_or_uncertified_training_results_are_rejected(tmp_path, monkeypatch, bad_case):
    records = [_record(i, gain) for i, gain in enumerate([10., 20., 40.])]
    if bad_case == 'missing_batch':
        records.pop()
    elif bad_case == 'duplicate_uid':
        records[1]['uid'] = records[0]['uid']
    elif bad_case == 'missing_upper':
        records[0].pop('objective_upper_bound')
    elif bad_case == 'nan_upper':
        records[0]['objective_upper_bound'] = float('nan')
    elif bad_case == 'nonoptimal_master':
        records[0]['master_status'] = 'NUMERIC'
    elif bad_case == 'nonoptimal_path':
        records[0]['paths'][0]['ir']['fitted']['status'] = 'NUMERIC'
    elif bad_case == 'nan_zero':
        records[0]['paths'][0]['ir']['zero']['objective'] = float('nan')
    elif bad_case == 'nan_anchor':
        records[0]['paths'][0]['policy_penalty_at_reference'] = float('nan')
    elif bad_case == 'negative_weight':
        records[0]['paths'][0]['weight'] = -.5
    elif bad_case == 'bad_weight_total':
        records[0]['paths'][0]['weight'] = .75
    elif bad_case == 'duplicate_path_id':
        records[0]['paths'][1]['path_id'] = records[0]['paths'][0]['path_id']
    elif bad_case == 'missing_path':
        records[0]['paths'].pop()
    _write(tmp_path, monkeypatch, records)
    function = _function('training_potential_improvement_rows')
    with pytest.raises(ValueError):
        function('toy_lb_occupancy_alp', expected_batches=3, expected_paths=4)


def test_relative_potential_ci_includes_independent_evaluation_denominator_uncertainty():
    function = _function('relative_potential_improvement')
    row = dict(potential_improvement_mean=12., potential_improvement_variance_of_mean=9.,
               potential_training_batches=3, potential_training_paths=4,
               potential_training_source='toy_lb_occupancy_alp', potential_training_domain='unrestricted')
    result = function(row, zero_mean=100., zero_variance_of_mean=25., confidence=.9)
    variance = 9. / 100. ** 2 + 12. ** 2 * 25. / 100. ** 4
    expected_half = 100. * t.ppf(.95, 2) * np.sqrt(variance)
    assert result['potential_relative_improvement_pct'] == pytest.approx(12.)
    assert result['potential_relative_improvement_half_window_pct'] == pytest.approx(expected_half)
    fixed_denominator = function(row, zero_mean=100., zero_variance_of_mean=0., confidence=.9)
    assert result['potential_relative_improvement_half_window_pct'] > fixed_denominator['potential_relative_improvement_half_window_pct']


@pytest.mark.parametrize('denominator', [0., -1.])
def test_relative_potential_is_withheld_for_nonpositive_zero_baselines(denominator):
    row = dict(potential_improvement_mean=12., potential_improvement_variance_of_mean=9.,
               potential_training_batches=3, potential_training_paths=4,
               potential_training_source='toy_lb_occupancy_alp', potential_training_domain='unrestricted')
    result = _function('relative_potential_improvement')(row, denominator, 1.)
    assert result['potential_relative_improvement_pct'] is None
    assert result['potential_relative_improvement_half_window_pct'] is None


def test_three_curve_plot_contract_and_csv_fields_include_green_potential_series():
    from matplotlib.colors import to_rgb
    series = aggregation._TOY_LB_IMPROVEMENT_SERIES
    assert len(series) == 3, 'The existing two curves need one potential-improvement curve'
    mean_key, half_key, color, label = series[2]
    assert (mean_key, half_key, label) == (
        'potential_relative_improvement_pct', 'potential_relative_improvement_half_window_pct',
        'Potential vs. Zero')
    red, green, blue = to_rgb(color)
    assert green > red and green > blue
    assert mean_key in aggregation._TOY_LOWER_BOUND_SUMMARY_COLUMNS
    assert half_key in aggregation._TOY_LOWER_BOUND_SUMMARY_COLUMNS


def test_three_series_annotations_do_not_overlap_and_potential_line_is_distinct(monkeypatch):
    rows = []
    for panel, _, _, _ in aggregation.TOY_LOWER_BOUND_SENSITIVITY_PANELS:
        for value in (.2, .5, .9):
            row = dict(panel=panel, mutate_val=value)
            for (mean_key, half_key, _, _), mean in zip(aggregation._TOY_LB_IMPROVEMENT_SERIES, [2., 300., 3.]):
                row[mean_key], row[half_key] = mean, .1
            rows.append(row)

    def inspect(figure, save_file):
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for ax in figure.axes:
            assert ax.lines[2].get_marker() == '^'
            assert ax.lines[2].get_linestyle() == '--'
            texts = ax.texts
            for i, text in enumerate(texts):
                for other in texts[i + 1:]:
                    if text.get_position()[0] == other.get_position()[0]:
                        assert not text.get_window_extent(renderer).padded(2).overlaps(
                            other.get_window_extent(renderer).padded(2))

    monkeypatch.setattr(aggregation, '_save_toy_lb_figure', inspect)
    aggregation.plot_toy_lower_bound_improvements(rows, 'unused.svg')
