import copy
import importlib
import json

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('experiments.analyze_repeated_po') is not None, 'local analysis is not implemented'
    return importlib.import_module('experiments.analyze_repeated_po')


def ir(value, lower=None):
    return dict(objective=value, lower_bound=value if lower is None else lower,
                upper_bound=value, status=2, solution_count=1,
                exact=lower is None, bound_valid=True, runtime_seconds=0.1)


def batch(rep=1):
    result = dict(schema_version=1, config_id='cfg', batch_id=f'cfg-{rep}', repetition=rep,
                  configuration={'gamma': .5}, initial_state=[1],
                  decision_set={'first': 'continuous', 'future': 'continuous'},
                  theta_domain={'coefficient_bound': None, 'alp_value_floor': False,
                                'regularization': 0, 'fixed_coefficients': {}},
                  seeds={name: rep * 100 + i for i, name in enumerate(['env', 'arrival', 'stop_time', 'initial_state'])},
                  reference={'coefficients': [2.], 'policy': 'alp', 'source': 'cache', 'source_sha256': 'sha', 'source_uid': 'cfg'},
                  sampling={'sample_path_number': 4, 'proposal': {'type': 'stratified_geometric', 'discount_factor_proposal': .5, 'num_strata': 2},
                            'tau_definition': 'arrival_length_plus_one', 'weight_convention': 'stratum_mass_over_count',
                            'quantile_clip': [1e-12, 1 - 1e-12]},
                  solver={'workers': 2, 'max_iterations': 10},
                  runtime={'git_commit': 'abc'},
                  training={'coefficients': [3.], 'full_objective': 12., 'objective_lower_bound': 12.,
                            'objective_upper_bound': 20., 'absolute_gap': 8., 'relative_gap': 2/3,
                            'master_certificate': {'sense': 'maximize', 'status': 2, 'objective': 20., 'objective_bound': 20., 'valid': True, 'reason': 'optimal'},
                            'termination_reason': 'iteration_limit', 'runtime_seconds': 1.}, paths=[])
    for i in range(4):
        result['paths'].append(dict(path_id=i, path_hash=f'path-{rep}-{i}', initial_state_hash='initial',
                                    stratum=i//2, weight=.25, arrival_length=i+1, termination_time=i+2,
                                    terminal='absorbed', policy_gradient=[1.],
                                    ir={'fitted': ir(13.), 'zero': ir(5.), 'alp': ir(10.)},
                                    raw_gap_fitted_minus_zero=8., raw_gap_fitted_minus_alp=3.))
    return result


def test_batch_baseline_uses_master_bound_and_correct_zero_constant():
    m = api()
    out = m.batch_observation(batch())
    assert out['R_alp'] == pytest.approx(10.)
    assert out['R_zero'] == pytest.approx(13.)  # 20 - (5 + 1*2)
    assert out['raw_gain_alp'] == pytest.approx(3.)
    assert out['raw_gain_zero'] == pytest.approx(8.)


def test_interrupted_baseline_uses_valid_min_lower_certificate():
    b = batch()
    b['paths'][0]['ir']['zero'] = ir(5., 1.)
    out = api().batch_observation(b)
    assert out['R_zero'] == pytest.approx(14.)
    assert out['conservative_baselines']['zero'] == 1


@pytest.mark.parametrize('mutation', ['master', 'weight', 'missing_path', 'floor', 'integer', 'duplicate_path', 'baseline', 'constant'])
def test_invalid_records_are_rejected(mutation):
    b = batch()
    if mutation == 'master': b['training']['master_certificate']['valid'] = False
    if mutation == 'weight': b['paths'][0]['weight'] = .3
    if mutation == 'missing_path': b['paths'].pop()
    if mutation == 'floor': b['theta_domain']['alp_value_floor'] = True
    if mutation == 'integer': b['decision_set']['first'] = 'integer'
    if mutation == 'duplicate_path': b['paths'][1]['path_id'] = 0
    if mutation == 'baseline': b['paths'][0]['ir']['alp']['bound_valid'] = False
    if mutation == 'constant': b['training']['full_objective'] = 13.
    with pytest.raises(ValueError): api().batch_observation(b)


def test_group_rejects_missing_batches_seeds_and_changed_reference():
    m = api()
    bs = [batch(i) for i in range(1, 4)]
    assert len(m.validate_group(bs, expected_batches=3, expected_paths=4)) == 3
    with pytest.raises(ValueError): m.validate_group(bs[:2], expected_batches=3, expected_paths=4)
    altered = copy.deepcopy(bs)
    altered[1]['seeds']['arrival'] = altered[0]['seeds']['arrival']
    with pytest.raises(ValueError): m.validate_group(altered, expected_batches=3, expected_paths=4)
    altered = copy.deepcopy(bs)
    altered[1]['reference']['coefficients'] = [4.]
    with pytest.raises(ValueError): m.validate_group(altered, expected_batches=3, expected_paths=4)


def test_student_interval_is_based_on_batch_count():
    result = api().student_summary([1., 2., 3.])
    assert result['mean'] == 2.
    assert result['standard_error'] == pytest.approx(1 / np.sqrt(3))
    assert result['ci95'][1] == pytest.approx(4.4841377117)
    assert result['batches'] == 3


def test_length_control_uses_shift_and_actual_clipped_proposal_mean():
    m = api()
    b = batch()
    assert m.expected_tau(b['sampling']) == pytest.approx(3.)
    assert m.batch_observation(b)['H'] == pytest.approx(.5)
    b['sampling']['proposal']['type'] = 'unknown'
    with pytest.raises(ValueError): m.expected_tau(b['sampling'])
    # L=ceil(log(1-U)/log(.5)), clipped U in [.5,.75]: L is 1 or 2.
    b['sampling']['proposal']['type'] = 'stratified_geometric'
    b['sampling']['quantile_clip'] = [.5, .75]
    assert m.expected_tau(b['sampling']) == pytest.approx(2.5)


def test_loo_subtracts_slope_not_intercept_and_uses_training_fold():
    m = api()
    values = np.array([8., 11., 14., 17.])
    control = np.array([-1., 0., 1., 2.])
    adjusted, slopes, fallback = m.loo_adjust(values, control)
    np.testing.assert_allclose(adjusted, 11.)
    np.testing.assert_allclose(slopes, 3.)
    assert not any(fallback)
    # Held-out extreme value must not affect its own slope.
    values[0] = 1e9
    assert m.loo_adjust(values, control)[1][0] == pytest.approx(3.)


def test_loo_repeated_bootstrap_records_exclude_position_only():
    m = api()
    values, controls = np.array([1., 1., 4., 7.]), np.array([0., 0., 1., 2.])
    adjusted, slopes, fallback = m.loo_adjust(values, controls)
    np.testing.assert_allclose(adjusted, 1.)
    np.testing.assert_allclose(slopes, 3.)
    adjusted, slopes, fallback = m.loo_adjust(values, np.zeros(4))
    np.testing.assert_allclose(adjusted, values)
    np.testing.assert_array_equal(slopes, 0.)
    assert all(fallback)


def test_cv_bootstrap_is_paired_reproducible_and_basic():
    m = api()
    values = np.array([[1., 11.], [2., 12.], [4., 14.], [8., 18.], [9., 19.]])
    control = np.zeros(5)
    first = m.cv_summary(values, control, bootstrap_count=100, seed=9)
    second = m.cv_summary(values, control, bootstrap_count=100, seed=9)
    assert first == second
    assert first['alp']['mean'] + 10 == pytest.approx(first['zero']['mean'])
    assert first['alp']['ci95'][0] + 10 == pytest.approx(first['zero']['ci95'][0])
    rng = np.random.default_rng(9)
    draws = rng.integers(0, 5, size=(100, 5))
    estimate = values[:, 0].mean()
    errors = values[draws, 0].mean(axis=1) - estimate
    assert first['alp']['ci95'][0] == pytest.approx(estimate - np.quantile(errors, .975))
    assert first['alp']['one_sided_upper95'] == pytest.approx(estimate - np.quantile(errors, .05))


def test_analyze_complete_manifest_and_missing_config_rejection(tmp_path):
    batches = [batch(i) for i in range(1, 4)]
    runs = []
    for b in batches:
        directory = tmp_path / b['batch_id']
        directory.mkdir()
        (directory / 'batch.json').write_text(json.dumps(b))
        runs.append({'batch_id': b['batch_id'], 'config_id': b['config_id'], 'params': b})
    manifest = {'runs': runs, 'configurations': [{'config_id': 'cfg'}],
                'display_settings': [{'label': 'test', 'config_id': 'cfg'}],
                'expected_batches': 3, 'expected_paths': 4}
    result = api().analyze(manifest, tmp_path)
    assert result['configurations']['cfg']['baseline']['alp']['mean'] == 10.
    manifest['configurations'].append({'config_id': 'omitted'})
    with pytest.raises(ValueError, match='configuration'):
        api().analyze(manifest, tmp_path)
