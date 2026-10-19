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
                  training={'coefficients': [3.], 'full_objective': 12., 'full_objective_exact': True,
                            'objective_lower_bound': 12.,
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


def standard_result(rep=1, result=None):
    b = batch(rep) if result is None else copy.deepcopy(result)
    b['seeds']['initial_state'] = 12345  # fixed initial state does not need resampling
    parameters = dict(uid=f'uid-{rep}', experiment_name='train-test', mutate_val=None,
        sample_path_number=4, init_state_mode='shared', init_state=[1], init_state_seed=12345,
        env_args=dict(b['configuration'], env_random_seed=b['seeds']['env'],
                      arrival_random_seed=b['seeds']['arrival'], stop_time_random_seed=b['seeds']['stop_time']),
        agent_args={'agent_args': dict(current_decision_var_type='continuous', future_decision_var_type='continuous',
            policy_centering=True, cap_policy='alp', initial_coefficients=[2.], save_path_certificates=True,
            sample_path_length_proposal=copy.deepcopy(b['sampling']['proposal']), **b['solver'])},
        training_generating_function_spec='absorption_alp_penalty')
    run = {k: copy.deepcopy(v) for k, v in b.items() if k not in ('training', 'runtime', 'paths')}
    run['parameters'] = parameters
    record = {k: copy.deepcopy(v) for k, v in b['training'].items() if k != 'runtime_seconds'}
    record.update(uid=parameters['uid'], training_time_seconds=1., paths=b['paths'])
    return run, record


def standard_manifest(runs):
    return {'runs': runs, 'configurations': [{'config_id': 'cfg'}],
            'display_settings': [{'label': 'test', 'config_id': 'cfg'}],
            'expected_batches': len(runs), 'expected_paths': 4}


def compact_result(rep=1, result=None):
    run, old = standard_result(rep, result)
    parameters = run['parameters']
    record = {key: copy.deepcopy(parameters[key]) for key in (
        'uid', 'experiment_name', 'mutate_val', 'sample_path_number')}
    record.update({key: copy.deepcopy(old[key]) for key in (
        'coefficients', 'training_time_seconds', 'full_objective',
        'objective_lower_bound', 'objective_upper_bound', 'termination_reason')})
    record['master_status'] = old['master_certificate']['status']
    record['tight_penalized_lower_bound'] = sum(
        path['weight'] * path['ir']['fitted']['objective'] for path in old['paths'])
    anchor = np.asarray(run['reference']['coefficients'])
    fitted = np.asarray(record['coefficients'])
    record['paths'] = []
    for path in old['paths']:
        compact = {key: copy.deepcopy(path[key]) for key in (
            'path_id', 'stratum', 'weight', 'termination_time',
            'raw_gap_fitted_minus_zero', 'raw_gap_fitted_minus_alp')}
        gradient = np.asarray(path['policy_gradient'])
        compact['policy_control_variate'] = float(gradient @ (fitted - anchor))
        compact['policy_penalty_at_reference'] = float(gradient @ anchor)
        compact['ir'] = {name: {key: cost[key] for key in ('objective', 'lower_bound', 'status')}
                         for name, cost in path['ir'].items()}
        record['paths'].append(compact)
    return run, record


def test_compact_record_preserves_batch_observation_without_saved_gradient_or_certificates():
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = compact_result()
    original = copy.deepcopy(record)
    converted = batch_from_training_record(record, run)
    assert api().batch_observation(converted) == api().batch_observation(batch())
    assert converted['training']['absolute_gap'] == 8.
    assert converted['training']['relative_gap'] == pytest.approx(2/3)
    assert converted['training']['runtime_seconds'] == 1.
    assert record == original


@pytest.mark.parametrize('status', [2, 'OPTIMAL'])
def test_objective_status_only_reconstructs_optimal_costs_and_preserves_cv_fields(status):
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = compact_result()
    expected = api().batch_observation(batch_from_training_record(record, run))
    for path in record['paths']:
        path.update(cv_adjusted_fitted_ir_cost=12., cv_adjusted_gap_fitted_minus_alp=2.,
                    cv_adjusted_gap_fitted_minus_zero=5.)
        for cost in path['ir'].values():
            del cost['lower_bound']
            cost['status'] = status
    original = copy.deepcopy(record)
    converted = batch_from_training_record(record, run)
    assert api().batch_observation(converted) == expected
    assert record == original
    for path in converted['paths']:
        assert path['cv_adjusted_gap_fitted_minus_zero'] == 5.
        for cost in path['ir'].values():
            assert cost['lower_bound'] == cost['objective']
            assert cost['exact']


@pytest.mark.parametrize('target', ['fitted', 'zero', 'alp'])
@pytest.mark.parametrize('status', [9, 'TIME_LIMIT'])
def test_nonoptimal_objective_without_lower_bound_is_rejected(target, status):
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = compact_result()
    cost = record['paths'][0]['ir'][target]
    cost['status'] = status
    del cost['lower_bound']
    with pytest.raises(ValueError, match='nonoptimal.*lower bound'):
        batch_from_training_record(record, run)


@pytest.mark.parametrize('factory', [standard_result, compact_result])
@pytest.mark.parametrize('field,value', [
    ('cv_adjusted_fitted_ir_cost', 12.),
    ('cv_adjusted_gap_fitted_minus_alp', 2.),
    ('cv_adjusted_gap_fitted_minus_zero', 5.),
])
def test_optional_cv_path_values_preserve_old_results_and_reject_corruption(factory, field, value):
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = factory()
    expected = api().batch_observation(batch_from_training_record(record, run))
    # Each new field is optional independently, including on older gradient records.
    record['paths'][0][field] = value
    assert api().batch_observation(batch_from_training_record(record, run)) == expected
    for invalid in (value + 1., float('nan'), None):
        record['paths'][0][field] = invalid
        with pytest.raises(ValueError, match='CV-adjusted'):
            batch_from_training_record(record, run)


@pytest.mark.parametrize('missing,field', [
    ('fitted', 'cv_adjusted_fitted_ir_cost'),
    ('fitted', 'cv_adjusted_gap_fitted_minus_alp'),
    ('fitted', 'cv_adjusted_gap_fitted_minus_zero'),
    ('alp', 'cv_adjusted_gap_fitted_minus_alp'),
    ('zero', 'cv_adjusted_gap_fitted_minus_zero'),
])
def test_cv_path_values_require_their_incumbents(missing, field):
    b = batch()
    path = b['paths'][0]
    path['ir'][missing].update(objective=None, upper_bound=None, solution_count=0, exact=False)
    for baseline in ('alp', 'zero'):
        if missing in ('fitted', baseline):
            path[f'raw_gap_fitted_minus_{baseline}'] = None
    path[field] = None
    api().validate_batch(b)
    path[field] = 0.
    with pytest.raises(ValueError, match='CV-adjusted'):
        api().validate_batch(b)


def test_cv_path_sums_recover_current_statistics_with_finite_solver_slack():
    from experiments.repeated_po_schema import batch_from_training_record
    b = batch()
    # Certified baseline lower bounds can differ from the displayed incumbents.
    b['paths'][0]['ir']['alp'] = ir(10., 8.)
    b['paths'][0]['ir']['zero'] = ir(5., 1.)
    for baseline in ('alp', 'zero'):
        b['paths'][0]['ir'][baseline]['status'] = 'TIME_LIMIT'
    run, record = compact_result(result=b)
    expected = api().batch_observation(batch_from_training_record(record, run))
    for path in record['paths']:
        path.update(cv_adjusted_fitted_ir_cost=12., cv_adjusted_gap_fitted_minus_alp=2.,
                    cv_adjusted_gap_fitted_minus_zero=5.)
    converted = batch_from_training_record(record, run)
    observation = api().batch_observation(converted)
    assert observation == expected
    paths = converted['paths']
    adjusted_mean = sum(p['weight'] * p['cv_adjusted_fitted_ir_cost'] for p in paths)
    assert adjusted_mean == record['full_objective']
    solver_slack = record['objective_upper_bound'] - adjusted_mean
    assert np.isfinite(solver_slack) and solver_slack == 8.
    for baseline in ('alp', 'zero'):
        adjusted_gap = sum(p['weight'] * p[f'cv_adjusted_gap_fitted_minus_{baseline}'] for p in paths)
        baseline_slack = sum(p['weight'] * (p['ir'][baseline]['objective'] - p['ir'][baseline]['lower_bound'])
                             for p in paths)
        assert observation[f'R_{baseline}'] == pytest.approx(adjusted_gap + solver_slack + baseline_slack)


@pytest.mark.parametrize('mode', ['baseline', 'cv'])
@pytest.mark.parametrize('save_lower_bound', [True, False])
def test_compact_outputs_preserve_repeated_batch_intervals_and_length_cv(tmp_path, mode, save_lower_bound):
    old_root, compact_root = tmp_path / 'old', tmp_path / 'compact'
    old_root.mkdir()
    compact_root.mkdir()
    runs, old_records, compact_records = [], [], []
    for rep in range(1, 6):
        b = batch(rep)
        b['training']['objective_upper_bound'] += rep**2
        b['training']['master_certificate']['objective_bound'] += rep**2
        b['training']['absolute_gap'] += rep**2
        b['training']['relative_gap'] = b['training']['absolute_gap'] / 12.
        for path in b['paths']:
            path['arrival_length'] += rep
            path['termination_time'] += rep
        run, old = standard_result(rep, b)
        _, compact = compact_result(rep, b)
        if not save_lower_bound:
            for path in compact['paths']:
                for cost in path['ir'].values():
                    del cost['lower_bound']
        runs.append(run)
        old_records.append(old)
        compact_records.append(compact)
    for root, records in ((old_root, old_records), (compact_root, compact_records)):
        (root / 'job.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    manifest = standard_manifest(runs)
    expected = api().analyze(manifest, old_root, mode=mode, bootstrap_count=30, seed=7)
    actual = api().analyze(manifest, compact_root, mode=mode, bootstrap_count=30, seed=7)
    assert actual == expected


@pytest.mark.parametrize('target', ['master', 'fitted', 'zero', 'alp'])
@pytest.mark.parametrize('status', ['NUMERIC', 'INFEASIBLE', 'UNBOUNDED', 'LOADED', 'unknown'])
def test_compact_output_rejects_status_without_valid_bound(target, status):
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = compact_result()
    if target == 'master':
        record['master_status'] = status
    else:
        record['paths'][0]['ir'][target]['status'] = status
    with pytest.raises(ValueError, match='status'):
        batch_from_training_record(record, run)


@pytest.mark.parametrize('field', ['uid', 'paths', 'coefficients', 'training_time_seconds',
    'full_objective', 'objective_lower_bound', 'objective_upper_bound', 'master_status',
    'termination_reason', 'tight_penalized_lower_bound'])
def test_compact_output_requires_core_values(field):
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = compact_result()
    del record[field]
    with pytest.raises(ValueError):
        batch_from_training_record(record, run)


@pytest.mark.parametrize('target', ['fitted', 'zero', 'alp'])
@pytest.mark.parametrize('field', ['objective', 'lower_bound'])
def test_compact_output_requires_finite_costs(target, field):
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = compact_result()
    if field == 'objective':
        del record['paths'][0]['ir'][target][field]
        with pytest.raises(ValueError):
            batch_from_training_record(record, run)
    record['paths'][0]['ir'][target][field] = float('nan')
    with pytest.raises(ValueError):
        batch_from_training_record(record, run)


@pytest.mark.parametrize('field', ['policy_control_variate', 'policy_penalty_at_reference'])
def test_compact_output_requires_both_finite_policy_products(field):
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = compact_result()
    del record['paths'][0][field]
    with pytest.raises(ValueError):
        batch_from_training_record(record, run)
    record['paths'][0][field] = float('inf')
    with pytest.raises(ValueError):
        batch_from_training_record(record, run)


@pytest.mark.parametrize('field', ['raw_gap', 'cv_mean', 'raw_mean', 'lower_bound', 'weight', 'tau'])
def test_compact_output_rejects_inconsistent_saved_values(field):
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = compact_result()
    if field == 'raw_gap': record['paths'][0]['raw_gap_fitted_minus_alp'] += 1.
    if field == 'cv_mean': record['full_objective'] += 1.
    if field == 'raw_mean': record['tight_penalized_lower_bound'] += 1.
    if field == 'lower_bound': record['objective_lower_bound'] += 1.
    if field == 'weight': record['paths'][0]['weight'] = .3
    if field == 'tau': record['paths'][0]['termination_time'] = 1.5
    with pytest.raises(ValueError):
        batch_from_training_record(record, run)


def test_standard_training_jsonl_joins_manifest_by_uid(tmp_path):
    pairs = [standard_result(i) for i in range(1, 4)]
    # Multiple jobs can append multiple training cases to the usual JSONL files.
    (tmp_path / 'job.jsonl').write_text(''.join(json.dumps(r) + '\n' for _, r in pairs))
    manifest = standard_manifest([r for r, _ in pairs])
    result = api().analyze(manifest, tmp_path)
    assert result['configurations']['cfg']['baseline']['alp']['mean'] == 10.
    assert result['configurations']['cfg']['baseline']['zero']['mean'] == 13.
    assert result['configurations']['cfg']['baseline']['alp']['batches'] == 3


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'wrong_uid', 'paths', 'manifest_uid', 'duplicate_manifest_uid'])
def test_standard_training_results_fail_closed(tmp_path, problem):
    pairs = [standard_result(i) for i in range(1, 4)]
    runs, records = [r for r, _ in pairs], [r for _, r in pairs]
    if problem == 'missing': records.pop()
    if problem == 'duplicate': records.append(copy.deepcopy(records[0]))
    if problem == 'wrong_uid': records[0]['uid'] = 'wrong'
    if problem == 'paths': del records[0]['paths']
    if problem == 'manifest_uid': del runs[0]['parameters']['uid']
    if problem == 'duplicate_manifest_uid': runs[1]['parameters']['uid'] = runs[0]['parameters']['uid']
    (tmp_path / 'job.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    with pytest.raises(ValueError):
        api().analyze(standard_manifest(runs), tmp_path)


def test_direct_converter_uses_uid_and_flat_fields_without_runtime_or_wrapper():
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = standard_result()
    assert not {'audit', 'runtime', 'audit_parameters_hash'} & record.keys()
    converted = batch_from_training_record(record, run)
    assert converted['paths'] == record['paths']
    assert 'runtime' not in converted
    assert converted['training']['runtime_seconds'] == record['training_time_seconds']
    assert api().batch_observation(converted) == api().batch_observation(batch())


@pytest.mark.parametrize('uid', ['wrong', None, ''])
def test_direct_converter_rejects_wrong_or_missing_uid(uid):
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = standard_result()
    record['uid'] = uid
    with pytest.raises(ValueError, match='UID'):
        batch_from_training_record(record, run)


@pytest.mark.parametrize('field', ['uid', 'paths', 'coefficients', 'training_time_seconds',
    'full_objective', 'full_objective_exact', 'objective_lower_bound', 'objective_upper_bound',
    'master_certificate', 'termination_reason', 'absolute_gap', 'relative_gap'])
def test_direct_converter_requires_complete_flat_output(field):
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = standard_result()
    del record[field]
    with pytest.raises(ValueError):
        batch_from_training_record(record, run)


def test_converter_does_not_accept_old_audit_wrapper():
    from experiments.repeated_po_schema import batch_from_training_record
    run, flat = standard_result()
    wrapped = {key: flat[key] for key in ('uid', 'coefficients', 'training_time_seconds')}
    wrapped['audit'] = {key: value for key, value in flat.items() if key not in wrapped}
    with pytest.raises(ValueError, match='missing'):
        batch_from_training_record(wrapped, run)


@pytest.mark.parametrize('mode', ['baseline', 'cv'])
def test_flat_training_outputs_preserve_existing_estimates(tmp_path, mode):
    legacy_root, flat_root = tmp_path / 'legacy', tmp_path / 'flat'
    legacy_root.mkdir()
    flat_root.mkdir()
    legacy_runs, runs, records = [], [], []
    for rep in range(1, 6):
        b = batch(rep)
        b['training']['objective_upper_bound'] += rep**2
        b['training']['master_certificate']['objective'] += rep**2
        b['training']['master_certificate']['objective_bound'] += rep**2
        b['training']['absolute_gap'] += rep**2
        b['training']['relative_gap'] = b['training']['absolute_gap'] / 12.
        for path in b['paths']:
            path['arrival_length'] += rep
            path['termination_time'] += rep
        folder = legacy_root / b['batch_id']
        folder.mkdir()
        (folder / 'batch.json').write_text(json.dumps(b))
        legacy_runs.append({'batch_id': b['batch_id'], 'config_id': b['config_id'], 'params': b})
        run, record = standard_result(rep, b)
        runs.append(run)
        records.append(record)
    (flat_root / 'job.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    expected = api().analyze(standard_manifest(legacy_runs), legacy_root, mode=mode, bootstrap_count=30, seed=7)
    actual = api().analyze(standard_manifest(runs), flat_root, mode=mode, bootstrap_count=30, seed=7)
    assert actual == expected


def test_group_does_not_require_or_compare_runtime_git_metadata():
    batches = [batch(i) for i in range(1, 4)]
    del batches[0]['runtime']
    batches[1]['runtime']['git_commit'] = 'different'
    assert len(api().validate_group(batches, expected_batches=3, expected_paths=4)) == 3


def test_fixed_initial_state_seed_may_repeat_across_batches():
    batches = [batch(i) for i in range(1, 4)]
    for b in batches:
        b['seeds']['initial_state'] = 12345
    assert len(api().validate_group(batches, expected_batches=3, expected_paths=4)) == 3


@pytest.mark.parametrize('field', ['proposal', 'reference', 'configuration', 'seeds', 'solver', 'decision_set', 'domain'])
def test_manifest_labels_must_describe_executed_parameters(field):
    from experiments.repeated_po_schema import batch_from_training_record
    run, record = standard_result()
    if field == 'proposal': run['sampling']['proposal']['discount_factor_proposal'] = .8
    if field == 'reference': run['reference']['coefficients'] = [4.]
    if field == 'configuration': run['configuration']['gamma'] = .8
    if field == 'seeds': run['seeds']['arrival'] = 9999
    if field == 'solver': run['solver']['workers'] = 1
    if field == 'decision_set': run['decision_set']['first'] = 'integer'
    if field == 'domain': run['theta_domain']['coefficient_bound'] = 20.
    with pytest.raises(ValueError, match='manifest'):
        batch_from_training_record(record, run)
