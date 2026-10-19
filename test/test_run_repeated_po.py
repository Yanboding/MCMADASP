"""Repeated PO uses the existing normal training route and license pool."""
import inspect
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import run


@pytest.mark.parametrize('as_file', [False, True])
def test_repeated_record_uses_existing_training_function(monkeypatch, tmp_path, as_file):
    env = Mock()
    checkout, training = Mock(return_value=env), Mock()
    monkeypatch.setattr(run, 'acquire_grb_env', checkout)
    monkeypatch.setattr(run, 'train_penalty_coefficients_for_env', training)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '64')
    record = {'uid': 'sample', 'init_state_mode': 'shared', 'sample_path_number': 512,
              'agent_args': {'agent_args': {'workers': 32}}}
    if as_file:
        source = tmp_path / 'params.json'
        source.write_text(json.dumps(record))
        args = ['--params_file', str(source)]
    else:
        args = ['--params', json.dumps(record)]
    run.main([*args, '--job_id', 'farm-17'])
    assert checkout.call_count == 33  # existing main environment + capped pool
    training.assert_called_once_with(**record, grb_env=env, grb_sub_envs=[env]*32, job_id='farm-17')


@pytest.mark.parametrize('payload', [[17], 'invalid', [], {'run_type': 'figure1_repeated_po'}])
def test_invalid_old_route_rejected_before_checkout(monkeypatch, payload):
    checkout = Mock(side_effect=AssertionError('checkout'))
    monkeypatch.setattr(run, 'acquire_grb_env', checkout)
    with pytest.raises(ValueError):
        run.main(['--params', json.dumps(payload)])
    checkout.assert_not_called()


def test_training_options_reach_existing_agent_method(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    environment = SimpleNamespace(discount_factor=.99, reset_random_seeds=Mock())
    generating = SimpleNamespace(number_of_coefficients=2)
    agent = Mock()
    agent.benders_decomposition_train.return_value = (12., np.array([1., 2.]), {'in_sample': {'mean': 12.}})
    factory = Mock(return_value=agent)
    monkeypatch.setattr(run, 'get_config_by_type', lambda *a, **k: SimpleNamespace(env=environment))
    monkeypatch.setattr(run, '_build_generating_function', lambda **k: generating)
    monkeypatch.setattr(run, 'ApproxQAgent', factory)
    monkeypatch.setattr(run, 'dispose_agent_models', Mock())
    result = run.train_penalty_coefficients_for_env(
        uid='training', experiment_name='test', mutate_val=None, sample_path_number=4,
        init_state_mode='shared', init_state=[[0], [0], [1, 1]], init_state_seed=2,
        env_args={}, agent_args={'agent_args': {'workers': 2, 'max_iterations': 1500,
            'absolute_gap_tolerance': 1e-6, 'save_path_certificates': False,
            'master_time_limit_seconds': 20., 'subproblem_time_limit_seconds': 10.}},
        training_generating_function_spec='absorption_alp_penalty',
        grb_env=None, grb_sub_envs=[], job_id='j')
    kwargs = agent.benders_decomposition_train.call_args.kwargs
    assert kwargs['max_iterations'] == 1500
    assert kwargs['absolute_gap_tolerance'] == 1e-6
    assert kwargs['master_time_limit_seconds'] == 20.
    assert kwargs['subproblem_time_limit_seconds'] == 10.
    assert 'save_path_certificates' not in kwargs
    agent.coefficient_model.evaluate_action_certificates.assert_not_called()
    assert 'workers' not in factory.call_args.kwargs
    assert 'max_iterations' not in factory.call_args.kwargs
    assert result['coefficients'] == [1., 2.]
    assert 'paths' not in result and 'objective_upper_bound' not in result



def test_training_function_keeps_original_inputs():
    assert list(inspect.signature(run.train_penalty_coefficients_for_env).parameters) == [
        'uid', 'experiment_name', 'mutate_val', 'sample_path_number', 'init_state_mode',
        'init_state', 'init_state_seed', 'env_args', 'agent_args',
        'training_generating_function_spec', 'grb_env', 'grb_sub_envs', 'job_id']


@pytest.fixture
def path_result_training(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    environment = SimpleNamespace(discount_factor=.99, reset_random_seeds=Mock())
    agent = Mock()
    agent.sample_path_weights = np.array([.25, .75])
    agent.sample_path_strata = np.array([0, 1])
    agent.subproblem_initial_cuts = {}
    path = SimpleNamespace(arrivals=np.array([[1]]), terminal=SimpleNamespace(value='absorbed'),
                           survival_weights=np.array([1.]), length=1, periods=2)
    agent.sample_paths = [path, path]
    agent._resolve_init_state_per_scenario.return_value = lambda sid: ([0], [0], [sid])
    agent.policy_cost_centering.return_value = (np.array([100., 100.]),
                                               np.array([[2.], [4.]]), np.array([1.]))
    def certificate(value):
        return dict(objective=value, lower_bound=value, upper_bound=value,
                    exact=True, bound_valid=True, status='OPTIMAL')
    workers = [SimpleNamespace(last_certificate=certificate(12.)),
               SimpleNamespace(last_certificate=certificate(22.))]
    def evaluate(theta, parallel):
        values = np.array([10. + theta[0], 20. + theta[0]])
        for worker, value in zip(workers, values):
            worker.last_certificate = certificate(float(value))
        return values, 0., [worker.last_certificate.copy() for worker in workers]
    agent.coefficient_model = SimpleNamespace(workers=workers,
                                              evaluate_action_certificates=Mock(side_effect=evaluate))
    agent.benders_decomposition_train.return_value = (
        23., np.array([2.]), {'in_sample': {'mean': 23.},
            'master_certificate': {'objective_bound': 25., 'valid': True, 'status': 'OPTIMAL'},
            'termination_reason': 'iteration_limit'})
    monkeypatch.setattr(run, 'get_config_by_type', lambda *a, **k: SimpleNamespace(env=environment))
    monkeypatch.setattr(run, '_build_generating_function', lambda **k: SimpleNamespace(number_of_coefficients=1))
    factory = Mock(return_value=agent)
    monkeypatch.setattr(run, 'ApproxQAgent', factory)
    monkeypatch.setattr(run, 'ALPRowGenerationAgent', Mock())
    dispose = Mock()
    monkeypatch.setattr(run, 'dispose_agent_models', dispose)
    parameters = dict(uid='training', experiment_name='test', mutate_val=None,
        sample_path_number=2, init_state_mode='shared', init_state=[[0], [0], [1]],
        init_state_seed=2, env_args={}, agent_args={'agent_args': {
            'save_path_certificates': True, 'policy_centering': True,
            'cap_policy': 'alp', 'initial_coefficients': [1.]}},
        training_generating_function_spec='absorption_alp_penalty',
        grb_env=None, grb_sub_envs=[], job_id='j')
    return agent, parameters, dispose, factory


@pytest.mark.parametrize('reuse_alp', [False, True])
def test_runner_evaluates_baselines_and_saves_raw_gaps_without_provenance(
        path_result_training, monkeypatch, tmp_path, reuse_alp):
    agent, parameters, dispose, _ = path_result_training
    if reuse_alp:
        agent.subproblem_initial_cuts = {
            0: (11., np.array([0.]), np.array([1.])),
            1: (21., np.array([0.]), np.array([1.])),
        }
    git_lookup = Mock(side_effect=AssertionError('No Git lookup needed to save training results'))
    monkeypatch.setattr('subprocess.check_output', git_lookup)
    result = run.train_penalty_coefficients_for_env(**parameters)
    saved = json.loads((tmp_path / 'experiments/results/test/j.jsonl').read_text())
    assert saved['full_objective'] == saved['objective_lower_bound'] == 19.5
    assert saved['objective_upper_bound'] == 25.
    assert saved['master_status'] == 'OPTIMAL'
    assert set(saved) == {'uid', 'experiment_name', 'mutate_val', 'sample_path_number',
                          'coefficients', 'training_time_seconds', 'tight_penalized_lower_bound',
                          'full_objective', 'objective_lower_bound', 'objective_upper_bound',
                          'master_status', 'termination_reason', 'paths'}
    assert saved['termination_reason'] == 'iteration_limit'
    assert saved['tight_penalized_lower_bound'] == 23.
    for row, expected in zip(saved['paths'], [(14., 11., 8., 6., 3.), (26., 21., 16., 10., 5.)]):
        assert tuple(row['ir'][k]['objective'] for k in ('fitted', 'alp', 'zero')) == expected[:3]
        assert row['raw_gap_fitted_minus_zero'] == expected[3]
        assert row['raw_gap_fitted_minus_alp'] == expected[4]
        for cert in row['ir'].values():
            assert cert['status'] == 'OPTIMAL'
            assert set(cert) == {'objective', 'status'}
    assert [row['weight'] for row in saved['paths']] == [.25, .75]
    assert [row['stratum'] for row in saved['paths']] == [0, 1]
    assert [p['policy_control_variate'] for p in saved['paths']] == [2., 4.]
    assert [p['policy_penalty_at_reference'] for p in saved['paths']] == [2., 4.]
    assert [p['cv_adjusted_fitted_ir_cost'] for p in saved['paths']] == [12., 22.]
    assert [p['cv_adjusted_gap_fitted_minus_alp'] for p in saved['paths']] == [1., 1.]
    assert [p['cv_adjusted_gap_fitted_minus_zero'] for p in saved['paths']] == [2., 2.]
    assert sum(p['weight'] * p['cv_adjusted_fitted_ir_cost'] for p in saved['paths']) == saved['full_objective']
    assert set(saved['paths'][0]) == {'path_id', 'stratum', 'weight', 'termination_time',
                                    'policy_control_variate', 'policy_penalty_at_reference', 'ir',
                                    'raw_gap_fitted_minus_zero', 'raw_gap_fitted_minus_alp',
                                    'cv_adjusted_fitted_ir_cost', 'cv_adjusted_gap_fitted_minus_zero',
                                    'cv_adjusted_gap_fitted_minus_alp'}
    cache = np.load(tmp_path / 'experiments/results/test/benders_checkpoints/training-policy-centering.npz')
    np.testing.assert_array_equal(cache['gradients'], [[2.], [4.]])
    assert cache['anchor'].tolist() == [1.]  # full reusable policy cache is preserved
    calls = agent.coefficient_model.evaluate_action_certificates.call_args_list
    # Fitted uses the final solve; a complete ALP cold-solve cache also saves its evaluation.
    assert len(calls) == (1 if reuse_alp else 2)
    assert [call.args[0].tolist() for call in calls] == ([[0.]] if reuse_alp else [[1.], [0.]])
    assert all(call.kwargs['parallel'] is True for call in calls)
    assert 'save_path_certificates' not in agent.benders_decomposition_train.call_args.kwargs
    assert dispose.call_args.args[0] is agent
    assert not {'audit', 'audit_parameters_hash', 'runtime', 'git_commit'} & saved.keys()
    git_lookup.assert_not_called()
    assert saved == result
    assert not list(tmp_path.rglob('batch.json'))
    assert not list(tmp_path.glob('*/training.jsonl'))


@pytest.mark.parametrize('problem', ['changed_seed', 'tiny_seed_change', 'missing_path',
                                    'nan_value', 'infinite_value', 'negative_infinite_value'])
def test_alp_baseline_falls_back_when_any_cold_solve_entry_cannot_be_reused(path_result_training, problem):
    agent, parameters, _, _ = path_result_training
    # Deliberately distinct cached values expose accidental partial-cache reuse.
    agent.subproblem_initial_cuts = {
        0: (-1000., np.array([0.]), np.array([1.])),
        1: (-2000., np.array([0.]), np.array([1.])),
    }
    if problem == 'missing_path':
        del agent.subproblem_initial_cuts[1]
    elif problem in ('changed_seed', 'tiny_seed_change'):
        delta = 1. if problem == 'changed_seed' else 1e-12
        agent.subproblem_initial_cuts[1] = (-2000., np.array([0.]), np.array([1. + delta]))
    else:
        value = {'nan_value': np.nan, 'infinite_value': np.inf,
                 'negative_infinite_value': -np.inf}[problem]
        agent.subproblem_initial_cuts[1] = (value, np.array([0.]), np.array([1.]))

    result = run.train_penalty_coefficients_for_env(**parameters)

    calls = agent.coefficient_model.evaluate_action_certificates.call_args_list
    assert [call.args[0].tolist() for call in calls] == [[1.], [0.]]
    assert [p['ir']['alp']['objective'] for p in result['paths']] == [11., 21.]
    assert [p['raw_gap_fitted_minus_alp'] for p in result['paths']] == [3., 5.]
    assert [p['cv_adjusted_gap_fitted_minus_alp'] for p in result['paths']] == [1., 1.]
    assert result['full_objective'] == result['objective_lower_bound'] == 19.5
    assert result['objective_upper_bound'] == 25.


def test_disabling_path_output_skips_baseline_evaluations(path_result_training):
    agent, parameters, _, _ = path_result_training
    parameters['agent_args']['agent_args']['save_path_certificates'] = False
    result = run.train_penalty_coefficients_for_env(**parameters)
    agent.coefficient_model.evaluate_action_certificates.assert_not_called()
    assert 'save_path_certificates' not in agent.benders_decomposition_train.call_args.kwargs
    assert 'paths' not in result


def test_baseline_failure_disposes_training_models_and_writes_no_result(path_result_training, tmp_path):
    agent, parameters, dispose, _ = path_result_training
    agent.coefficient_model.evaluate_action_certificates.side_effect = RuntimeError('baseline failed')
    with pytest.raises(RuntimeError, match='baseline failed'):
        run.train_penalty_coefficients_for_env(**parameters)
    assert dispose.call_args.args[0] is agent
    assert not (tmp_path / 'experiments/results/test/j.jsonl').exists()


def test_path_output_requires_policy_anchor_before_creating_agent(path_result_training):
    _, parameters, _, factory = path_result_training
    parameters['agent_args']['agent_args']['policy_centering'] = False
    with pytest.raises(ValueError, match='policy_centering'):
        run.train_penalty_coefficients_for_env(**parameters)
    factory.assert_not_called()
