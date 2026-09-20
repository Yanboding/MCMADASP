import os
import shutil
from unittest import mock

import numpy as np

import run
from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import LinearPenaltyFunction
from param_generation import cli


def make_agent():
    config = get_config_by_type('toy')
    env = config.env
    env.reset_random_seeds()
    state, _ = env.reset(**config.reset_params)
    agent = ApproxQAgent(env, discount_factor=env.discount_factor, sample_path_number=4,
                         generating_function=LinearPenaltyFunction(env))
    return agent, state


def in_sample(agent, theta):
    values, mean = agent.coefficient_model.evaluate_action(np.asarray(theta, dtype=float), parallel=False)
    return {'mean': mean, 'min': float(values.min())}


def test_min_objective_trains_from_initial_coefficients_and_reports_in_sample():
    agent, state = make_agent()
    obj_mean, theta_mean, info_mean = agent.benders_decomposition_train(init_state=state, parallel=False)
    assert 'initial_in_sample' not in info_mean and set(info_mean['in_sample']) == {'mean', 'min'}
    assert obj_mean == info_mean['in_sample']['mean']
    agent, state = make_agent()
    obj, theta, info = agent.benders_decomposition_train(
        init_state=state, parallel=False, training_objective='min', initial_coefficients=theta_mean)
    assert set(info['in_sample']) == set(info['initial_in_sample']) == {'mean', 'min'}
    expected = in_sample(agent, theta)
    for key in expected:
        assert np.isclose(info['in_sample'][key], expected[key], atol=1e-6)
    assert obj == info['in_sample']['mean']
    assert info['in_sample']['min'] >= info['initial_in_sample']['min'] - 1e-6
    assert np.isclose(info['initial_in_sample']['mean'], obj_mean, atol=1e-6)
    assert info['in_sample']['min'] >= in_sample(agent, theta_mean)['min'] - 1e-6


def test_master_builder_adds_eta_for_minimum_objectives():
    agent, state = make_agent()
    master, coefficients, theta = agent.train_master_builder_fn(objective='min')
    names = [v.VarName for v in master.getVars()]
    assert 'eta' in names
    assert master.NumConstrs >= theta.shape[0]
    try:
        agent.train_master_builder_fn(objective='max')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown objective must be rejected')


def test_runner_records_objective_and_warm_start():
    with mock.patch.object(cli, 'write_command_file'):
        (record,) = cli.main(['train', 'base_toy_study', '--dat', 'unused.dat'])
    record = dict(record)
    record['sample_path_number'] = 4
    record['experiment_name'] = record['experiment_name'] + '_objective_unittest'
    inner = record['agent_args']['agent_args']
    folder = os.path.join('experiments', 'results', record['experiment_name'])
    assert not os.path.exists(folder)
    try:
        baseline = run.train_penalty_coefficients_for_env(**record, grb_env=None, grb_sub_envs=None, job_id='obj_test')
        inner['training_objective'] = 'min'
        inner['initial_coefficients'] = baseline['coefficients']
        inner['initial_coefficients_source'] = {'file': 'unittest', 'uid': baseline['uid']}
        record['uid'] = record['uid'] + '_min'
        out = run.train_penalty_coefficients_for_env(**record, grb_env=None, grb_sub_envs=None, job_id='obj_test')
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert baseline['training_objective'] == 'mean' and baseline['initial_coefficients_source'] is None
    assert baseline['initial_in_sample'] is None and baseline['tight_penalized_lower_bound'] == baseline['in_sample']['mean']
    assert out['training_objective'] == 'min' and out['initial_coefficients_source']['uid'] == baseline['uid']
    assert out['in_sample']['min'] >= out['initial_in_sample']['min'] - 1e-6
    assert out['tight_penalized_lower_bound'] == out['in_sample']['mean']


def test_fixed_coefficients_are_pinned_in_master_and_warm_start():
    agent, state = make_agent()
    _, theta_mean, _ = agent.benders_decomposition_train(init_state=state, parallel=False)
    warm = np.array(theta_mean, dtype=float)
    warm[0] = 123.0
    agent, state = make_agent()
    obj, theta, info = agent.benders_decomposition_train(
        init_state=state, parallel=False, training_objective='min', initial_coefficients=warm,
        fixed_coefficients={0: 0.0})
    assert theta[0] == 0.0 and warm[0] == 123.0
    warm[0] = 0.0
    assert np.isclose(info['initial_in_sample']['min'], in_sample(agent, warm)['min'], atol=1e-6)
    assert info['in_sample']['min'] >= info['initial_in_sample']['min'] - 1e-6
    master, coefficients, _ = agent.train_master_builder_fn(fixed_coefficients={0: 0.0})
    master.update()
    assert coefficients[0].lb.item() == 0.0 and coefficients[0].ub.item() == 0.0


def test_runner_pins_fixed_coefficients_from_json_keys():
    with mock.patch.object(cli, 'write_command_file'):
        (record,) = cli.main(['train', 'base_toy_study', '--penalty-function', 'absorption_alp_penalty', '--fix-intercept', '--dat', 'unused.dat'])
    record = dict(record)
    record['sample_path_number'] = 4
    record['experiment_name'] = record['experiment_name'] + '_fixed_unittest'
    assert record['agent_args']['agent_args']['fixed_coefficients'] == {'0': 0.0}
    folder = os.path.join('experiments', 'results', record['experiment_name'])
    assert not os.path.exists(folder)
    try:
        out = run.train_penalty_coefficients_for_env(**record, grb_env=None, grb_sub_envs=None, job_id='fixed_test')
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert out['coefficients'][0] == 0.0 and out['fixed_coefficients'] == {'0': 0.0}


def test_workers_cold_start_at_the_initial_coefficients():
    agent, state = make_agent()
    _, theta_mean, _ = agent.benders_decomposition_train(init_state=state, parallel=False)
    warm = np.array(theta_mean, dtype=float)
    warm[0] = 5.0
    agent, state = make_agent()
    agent.benders_decomposition_train(init_state=state, parallel=False, initial_coefficients=warm,
                                      fixed_coefficients={0: 0.0})
    warm[0] = 0.0
    values, _ = agent.coefficient_model.evaluate_action(warm, parallel=False)
    for sid, (value, gradient, action) in agent.subproblem_initial_cuts.items():
        np.testing.assert_allclose(action, warm)
        assert np.isclose(value, values[sid], atol=1e-6)
    agent, state = make_agent()
    agent.benders_decomposition_train(init_state=state, parallel=False)
    for value, gradient, action in agent.subproblem_initial_cuts.values():
        assert not action.any()


def test_warm_started_training_does_not_stop_at_the_initial_coefficients():
    agent, state = make_agent()
    zeros = np.zeros(agent.generating_function.number_of_coefficients)
    obj, theta, info = agent.benders_decomposition_train(init_state=state, parallel=False, initial_coefficients=zeros)
    assert info['iterations'] > 1
    assert info['in_sample']['mean'] > info['initial_in_sample']['mean'] + 1e-6
    assert np.abs(theta).max() > 0
