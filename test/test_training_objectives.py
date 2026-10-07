import os
import shutil
from unittest import mock

import numpy as np

from metaheuristic_algorithm.benders_decomposition_solver import objective_confidence_interval

import run
from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import LinearPenaltyFunction
from generating_function.alp_penalty_function import AbsorptionALPPenaltyFunction
from param_generation import cli


def make_agent(generating_function_class=LinearPenaltyFunction):
    config = get_config_by_type('toy')
    env = config.env
    env.reset_random_seeds()
    state, _ = env.reset(**config.reset_params)
    agent = ApproxQAgent(env, discount_factor=env.discount_factor, sample_path_number=4,
                         generating_function=generating_function_class(env))
    return agent, state


def test_in_sample_reports_the_cost_mean_and_std_at_both_coefficient_sets():
    agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    init_state = distinct_states(agent.env, agent.sample_path_number)
    keys = {'mean', 'std', 'half_width'}
    warm = np.zeros(agent.generating_function.number_of_coefficients)
    obj, theta, info = agent.benders_decomposition_train(
        init_state=init_state, parallel=False, coefficient_bound=1e3, initial_coefficients=warm)
    assert set(info['in_sample']) == set(info['initial_in_sample']) == keys
    for coefficients, reported in ((np.asarray(theta, dtype=float), info['in_sample']),
                                   (warm, info['initial_in_sample'])):
        expected = in_sample(agent, coefficients)
        for key in keys:
            assert np.isclose(reported[key], expected[key], atol=1e-6)
    assert info['in_sample']['std'] > 0
    assert obj == info['in_sample']['mean']


def test_runner_records_only_the_raw_in_sample_statistics():
    with mock.patch.object(cli, 'write_command_file'):
        (record,) = cli.main(['train', 'base_toy_study', '--penalty-function', 'absorption_alp_penalty',
                              '--coefficient-bound', '1000', '--dat', 'unused.dat'])
    record = dict(record)
    record['sample_path_number'] = 4
    record['experiment_name'] = record['experiment_name'] + '_cost_unittest'
    folder = os.path.join('experiments', 'results', record['experiment_name'])
    assert not os.path.exists(folder)
    try:
        out = run.train_penalty_coefficients_for_env(**record, grb_env=None, grb_sub_envs=None, job_id='cost_test')
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert set(out['in_sample']) == {'mean', 'std', 'half_width'}
    assert out['in_sample']['std'] >= 0
    assert out['tight_penalized_lower_bound'] == out['in_sample']['mean']
    for dropped in ('in_sample_cost', 'initial_in_sample_centered', 'in_sample_centered'):
        assert dropped not in out


def test_training_records_the_paired_improvement_over_the_warm_start():
    from metaheuristic_algorithm.benders_decomposition_solver import objective_confidence_interval
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    warm = np.zeros(agent.generating_function.number_of_coefficients)
    obj, theta, info = agent.benders_decomposition_train(
        init_state=state, parallel=False, coefficient_bound=1e3, initial_coefficients=warm)
    kappa = np.asarray(agent.sample_path_weights, dtype=float)
    strata = np.asarray(agent.sample_path_strata)
    final, _ = agent.coefficient_model.evaluate_action(np.asarray(theta, dtype=float), parallel=False)
    initial, _ = agent.coefficient_model.evaluate_action(warm, parallel=False)
    mean, half_width = objective_confidence_interval(final - initial, weights=kappa, strata=strata)
    assert set(info['improvement']) == {'mean', 'half_width'}
    assert np.isclose(info['improvement']['mean'], mean, atol=1e-6)
    assert np.isclose(info['improvement']['half_width'], half_width, atol=1e-6)
    assert np.isclose(info['improvement']['mean'],
                      info['in_sample']['mean'] - info['initial_in_sample']['mean'], atol=1e-6)
    assert set(info['in_sample']) == {'mean', 'std', 'half_width'}
    assert info['in_sample']['half_width'] > 0
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    _, _, cold = agent.benders_decomposition_train(init_state=state, parallel=False, coefficient_bound=1e3)
    assert 'improvement' not in cold


def test_coefficient_blocks_expose_the_index_ranges():
    config = get_config_by_type('toy')
    env = config.env
    function = AbsorptionALPPenaltyFunction(env)
    blocks = function.coefficient_blocks()
    assert list(blocks) == ['intercept', 'regular', 'overtime', 'waitlist']
    assert blocks['intercept'] == [0]
    assert blocks['regular'] == list(range(1, 1 + env.planning_horizon))
    assert blocks['overtime'] == list(range(1 + env.planning_horizon, 1 + 2 * env.planning_horizon))
    assert blocks['waitlist'] == list(range(1 + 2 * env.planning_horizon, function.number_of_coefficients))
    try:
        LinearPenaltyFunction(env).coefficient_blocks()
    except NotImplementedError as exc:
        assert 'blocks' in str(exc)
    else:
        raise AssertionError('a generating function without named blocks must say so')


def test_evaluation_can_remove_the_pathwise_penalty_noise():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    theta = np.linspace(-3, 4, agent.generating_function.number_of_coefficients)
    agent.generating_function.set_coefficients(theta)
    path = agent.sample_paths[0]
    arrivals = np.asarray(path.arrivals)
    weights = np.asarray(path.period_weights(agent.discount_factor), dtype=float) if hasattr(path, 'period_weights') \
        else np.asarray(path.survival_weights, dtype=float)
    kwargs = {'terminal': path.terminal, 'period_weights': weights}
    raw = agent.calculate_information_relaxation_cost(state, arrivals, **kwargs)
    removed = agent.calculate_information_relaxation_cost(state, arrivals, remove_path_noise=True, **kwargs)
    assert np.isfinite(raw) and np.isfinite(removed) and not np.isclose(raw, removed)
    agent.generating_function.set_coefficients(np.zeros_like(theta))
    assert np.isclose(agent.calculate_information_relaxation_cost(state, arrivals, **kwargs),
                      agent.calculate_information_relaxation_cost(state, arrivals, remove_path_noise=True, **kwargs),
                      atol=1e-6)


def in_sample(agent, theta):
    theta = np.asarray(theta, dtype=float)
    values, _ = agent.coefficient_model.evaluate_action(theta, parallel=False)
    kappa = np.asarray(agent.sample_path_weights, dtype=float)
    costs = np.asarray(values, dtype=float)
    mean, half_width = objective_confidence_interval(costs, weights=kappa, strata=np.asarray(agent.sample_path_strata))
    return {'mean': mean, 'half_width': half_width,
            'std': float(np.sqrt(kappa @ (costs - mean) ** 2))}


def distinct_states(env, count):
    env.reset_random_seeds()
    states, seen = [], set()
    while len(states) < count:
        state = tuple(np.asarray(c) for c in env.generate_initial_state())
        key = tuple(np.concatenate([np.asarray(c, dtype=float).reshape(-1) for c in state]).tolist())
        if key not in seen:
            seen.add(key)
            states.append(state)
    return states


def test_linear_penalty_trains_the_mean():
    agent, state = make_agent()
    _, _, info = agent.benders_decomposition_train(init_state=state, parallel=False)
    assert set(info['in_sample']) == {'mean', 'std', 'half_width'}


def test_runner_records_the_warm_start():
    with mock.patch.object(cli, 'write_command_file'):
        (record,) = cli.main(['train', 'base_toy_study', '--penalty-function', 'absorption_alp_penalty', '--coefficient-bound', '1000', '--dat', 'unused.dat'])
    record = dict(record)
    record['sample_path_number'] = 4
    record['experiment_name'] = record['experiment_name'] + '_objective_unittest'
    inner = record['agent_args']['agent_args']
    folder = os.path.join('experiments', 'results', record['experiment_name'])
    assert not os.path.exists(folder)
    try:
        baseline = run.train_penalty_coefficients_for_env(**record, grb_env=None, grb_sub_envs=None, job_id='obj_test')
        inner['paths_per_state'] = 2
        inner['initial_coefficients'] = baseline['coefficients']
        inner['initial_coefficients_source'] = {'file': 'unittest', 'uid': baseline['uid']}
        record['uid'] = record['uid'] + '_wc'
        out = run.train_penalty_coefficients_for_env(**record, grb_env=None, grb_sub_envs=None, job_id='obj_test')
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert baseline['initial_coefficients_source'] is None
    assert baseline['initial_in_sample'] is None and baseline['tight_penalized_lower_bound'] == baseline['in_sample']['mean']
    assert out['initial_coefficients_source']['uid'] == baseline['uid']
    assert baseline['paths_per_state'] is None and out['paths_per_state'] == 2
    assert out['tight_penalized_lower_bound'] == out['in_sample']['mean']


def test_fixed_coefficients_are_pinned_in_master_and_warm_start():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    _, theta_mean, _ = agent.benders_decomposition_train(init_state=state, parallel=False, coefficient_bound=1e3)
    warm = np.array(theta_mean, dtype=float)
    warm[0] = 123.0
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    obj, theta, info = agent.benders_decomposition_train(
        init_state=state, parallel=False, initial_coefficients=warm,
        fixed_coefficients={0: 0.0}, coefficient_bound=1e3)
    assert theta[0] == 0.0 and warm[0] == 123.0
    warm[0] = 0.0
    assert np.isclose(info['initial_in_sample']['mean'], in_sample(agent, warm)['mean'], atol=1e-6)
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
    assert info['in_sample']['mean'] > info['initial_in_sample']['mean'] + 1e-6
    assert np.abs(theta).max() > 0


def test_warm_started_training_skips_the_iteration_pinned_at_the_warm_start():
    from metaheuristic_algorithm import BendersDecompositionSolver
    pinned_solve = BendersDecompositionSolver._solve_master_with_fixed_action
    pinned_calls = []

    def spy(solver, *args, **kwargs):
        pinned_calls.append(args[0])
        return pinned_solve(solver, *args, **kwargs)

    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    warm = np.zeros(agent.generating_function.number_of_coefficients)
    with mock.patch.object(BendersDecompositionSolver, '_solve_master_with_fixed_action', spy):
        obj, theta, info = agent.benders_decomposition_train(
            init_state=state, parallel=False, initial_coefficients=warm, coefficient_bound=1e3)
    assert not pinned_calls
    assert info['in_sample']['mean'] > info['initial_in_sample']['mean'] + 1e-6
    assert np.abs(theta).max() > 0
    solver = agent.coefficient_model
    assert solver._seeded_at(solver.workers, warm)
    assert not solver._seeded_at(solver.workers, np.ones_like(warm))
    assert not solver._seeded_at(solver.workers, None)


def test_coefficient_bound_overrides_widen_single_coefficients():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    master, coefficients, _ = agent.train_master_builder_fn(coefficient_bound=10.0, coefficient_bound_overrides={0: 1e6})
    master.update()
    assert coefficients[0].lb.item() == -1e6 and coefficients[0].ub.item() == 1e6
    assert coefficients[1].lb.item() == -10.0 and coefficients[1].ub.item() == 10.0
    warm = np.zeros(agent.generating_function.number_of_coefficients)
    warm[0] = -500.0
    obj, theta, info = agent.benders_decomposition_train(
        init_state=state, parallel=False, coefficient_bound=10.0, coefficient_bound_overrides={0: 1e6},
        initial_coefficients=warm)
    assert abs(theta[0]) <= 1e6 and max(abs(v) for v in theta[1:]) <= 10.0 + 1e-9
    assert np.isclose(info['initial_in_sample']['mean'], in_sample(agent, warm)['mean'], atol=1e-6)


def test_runner_records_bound_overrides():
    with mock.patch.object(cli, 'write_command_file'):
        (record,) = cli.main(['train', 'base_toy_study', '--penalty-function', 'absorption_alp_penalty',
                              '--coefficient-bound', '10', '--intercept-bound', '1000000', '--dat', 'unused.dat'])
    record = dict(record)
    record['sample_path_number'] = 4
    record['experiment_name'] = record['experiment_name'] + '_override_unittest'
    assert record['agent_args']['agent_args']['coefficient_bound_overrides'] == {'0': 1000000.0}
    folder = os.path.join('experiments', 'results', record['experiment_name'])
    assert not os.path.exists(folder)
    try:
        out = run.train_penalty_coefficients_for_env(**record, grb_env=None, grb_sub_envs=None, job_id='override_test')
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert out['coefficient_bound'] == 10.0 and out['coefficient_bound_overrides'] == {'0': 1000000.0}
    assert max(abs(v) for v in out['coefficients'][1:]) <= 10.0 + 1e-9


def test_generate_mode_gives_every_state_the_same_stratum_mix():
    with mock.patch.object(cli, 'write_command_file'):
        (record,) = cli.main(['train', 'base_toy_study', '--paths-per-state', '4', '--dat', 'unused.dat'])
    strata = np.array([0, 0, 1, 1, 1, 1, 1, 1])
    states = run._draw_per_scenario_init_states(record['env_args'], record['init_state_seed'], 8, 4, strata)
    keys = [tuple(np.concatenate([np.asarray(c, dtype=float).reshape(-1) for c in state]).tolist()) for state in states]
    assert len(states) == 8 and len(set(keys)) == 2
    for stratum, per_state in ((0, 1), (1, 3)):
        counts = {}
        for sid in np.flatnonzero(strata == stratum):
            counts[keys[sid]] = counts.get(keys[sid], 0) + 1
        assert sorted(counts.values()) == [per_state, per_state]
    for bad_strata, message in ((np.array([0, 0, 0, 1, 1, 1, 1, 1]), 'stratum'), (None, 'strata')):
        try:
            run._draw_per_scenario_init_states(record['env_args'], record['init_state_seed'], 8, 4, bad_strata)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f'expected a {message} error')
    try:
        run._draw_per_scenario_init_states(record['env_args'], record['init_state_seed'], 8, 3, np.zeros(8, dtype=int))
    except ValueError as exc:
        assert 'divide' in str(exc)
    else:
        raise AssertionError('paths_per_state must divide the scenario count')


