import os
import shutil
from unittest import mock

import numpy as np

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


def scenario_states(agent, init_state):
    return init_state if isinstance(init_state, list) else [init_state] * agent.sample_path_number


def state_blocks(agent, init_state):
    labels = {}
    return np.array([labels.setdefault(tuple(np.concatenate([np.asarray(c, dtype=float).reshape(-1) for c in state]).tolist()), len(labels))
                     for state in scenario_states(agent, init_state)])


def initial_state_features(agent, init_state):
    return np.array([agent.generating_function.state_features(state) for state in scenario_states(agent, init_state)])


def test_centering_shifts_the_worst_case_objective_without_touching_the_initial_state_value():
    agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    states = distinct_states(agent.env, agent.sample_path_number)
    raw, _ = make_agent(AbsorptionALPPenaltyFunction)
    _, raw_theta, raw_info = raw.benders_decomposition_train(
        init_state=states, parallel=False, training_objective='worst_case', worst_case_scope='joint',
        coefficient_bound=1e3)
    _, theta, info = agent.benders_decomposition_train(
        init_state=states, parallel=False, training_objective='worst_case', worst_case_scope='joint',
        coefficient_bound=1e3, noise_removal='mean')
    noise_mean = agent.noise_mean
    assert np.abs(noise_mean).max() > 0
    for probe in (np.asarray(theta, dtype=float), np.zeros_like(noise_mean), np.linspace(-1, 1, noise_mean.size)):
        centered_values, _ = agent.coefficient_model.evaluate_action(probe, parallel=False)
        raw_values, _ = raw.coefficient_model.evaluate_action(probe, parallel=False)
        np.testing.assert_allclose(centered_values, raw_values - probe @ noise_mean, atol=1e-6)
    theta = np.asarray(theta, dtype=float)
    assert set(info['in_sample']) == {'mean', 'std', 'half_width', 'worst_case'}
    raw_values, _ = raw.coefficient_model.evaluate_action(theta, parallel=False)
    features = initial_state_features(agent, states)
    kappa = np.asarray(agent.sample_path_weights, dtype=float)
    costs = raw_values + features @ theta
    assert np.isclose(info['in_sample']['worst_case'], float(kappa @ features @ theta) + raw_values.min(), atol=1e-6)
    assert np.isclose(info['in_sample']['mean'], float(kappa @ costs), atol=1e-6)
    assert np.isclose(info['in_sample']['std'], float(np.sqrt(kappa @ (costs - kappa @ costs) ** 2)), atol=1e-6)
    assert not np.allclose(theta, raw_theta, atol=1e-6)
    assert np.isclose(raw_info['in_sample']['worst_case'], in_sample(raw, raw_theta, states, scope='joint')['worst_case'], atol=1e-6)


def test_in_sample_reports_the_cost_mean_and_std_at_both_coefficient_sets():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    states = distinct_states(agent.env, agent.sample_path_number)
    for init_state, objective, scope, keys in ((states, 'mean', None, {'mean', 'std', 'half_width'}),
                                               (states, 'worst_case', 'joint', {'mean', 'std', 'half_width', 'worst_case'}),
                                               (state, 'worst_case', 'per_state', {'mean', 'std', 'half_width', 'worst_case'})):
        agent, _ = make_agent(AbsorptionALPPenaltyFunction)
        warm = np.zeros(agent.generating_function.number_of_coefficients)
        kwargs = {} if scope is None else {'worst_case_scope': scope}
        obj, theta, info = agent.benders_decomposition_train(
            init_state=init_state, parallel=False, training_objective=objective, coefficient_bound=1e3,
            initial_coefficients=warm, **kwargs)
        assert set(info['in_sample']) == set(info['initial_in_sample']) == keys
        for coefficients, reported in ((np.asarray(theta, dtype=float), info['in_sample']),
                                       (warm, info['initial_in_sample'])):
            expected = in_sample(agent, coefficients, init_state if objective == 'worst_case' else None,
                                 scope=scope or 'per_state')
            for key in keys:
                assert np.isclose(reported[key], expected[key], atol=1e-6)
        assert info['in_sample']['std'] > 0
        assert obj == info['in_sample']['mean']


def test_runner_records_only_the_raw_in_sample_statistics():
    with mock.patch.object(cli, 'write_command_file'):
        (record,) = cli.main(['train', 'base_toy_study', '--penalty-function', 'absorption_alp_penalty',
                              '--coefficient-bound', '1000', '--noise-removal', 'mean', '--dat', 'unused.dat'])
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
    for dropped in ('in_sample_cost', 'noise_centered', 'noise_mean',
                    'initial_in_sample_centered', 'in_sample_centered'):
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


def test_pathwise_noise_removal_subtracts_each_path_own_penalty_constant():
    raw_agent, state = make_agent(AbsorptionALPPenaltyFunction)
    _, theta_raw, _ = raw_agent.benders_decomposition_train(init_state=state, parallel=False, coefficient_bound=1e3)
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    obj, theta, info = agent.benders_decomposition_train(
        init_state=state, parallel=False, coefficient_bound=1e3, noise_removal='pathwise',
        initial_coefficients=theta_raw)
    constants = np.array([agent.subproblem_noise[sid] for sid in range(agent.sample_path_number)], dtype=float)
    assert np.abs(constants).max() > 0
    for probe in (np.asarray(theta, dtype=float), np.zeros(constants.shape[1]), np.linspace(-1, 1, constants.shape[1])):
        removed, _ = agent.coefficient_model.evaluate_action(probe, parallel=False)
        original, _ = raw_agent.coefficient_model.evaluate_action(probe, parallel=False)
        np.testing.assert_allclose(removed, original - constants @ probe, atol=1e-6)
    reference, _ = make_agent(AbsorptionALPPenaltyFunction)
    reference._build_training_workers(init_state=state, parallel=False, initial_coefficients=np.asarray(theta_raw))
    for worker in agent.coefficient_model.workers:
        value, gradient, seed = worker.initial_cut
        reference_value, reference_gradient, reference_seed = reference.subproblem_initial_cuts[worker.subproblem_id]
        np.testing.assert_allclose(seed, reference_seed, atol=1e-9)
        assert np.isclose(value, reference_value - constants[worker.subproblem_id] @ seed, atol=1e-6)
        np.testing.assert_allclose(gradient, reference_gradient - constants[worker.subproblem_id], atol=1e-6)
    kappa = np.asarray(agent.sample_path_weights, dtype=float)
    values, _ = agent.coefficient_model.evaluate_action(np.asarray(theta, dtype=float), parallel=False)
    assert np.isclose(info['in_sample']['mean'], float(kappa @ values), atol=1e-6)
    assert info['in_sample']['mean'] > info['initial_in_sample']['mean'] - 1e-6


def test_mean_noise_removal_keeps_reporting_the_original_penalty():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    obj, theta, info = agent.benders_decomposition_train(
        init_state=state, parallel=False, coefficient_bound=1e3, noise_removal='mean')
    kappa = np.asarray(agent.sample_path_weights, dtype=float)
    values, _ = agent.coefficient_model.evaluate_action(np.asarray(theta, dtype=float), parallel=False)
    shift = float(np.asarray(theta) @ agent.noise_mean)
    assert np.isclose(info['in_sample']['mean'], float(kappa @ values) + shift, atol=1e-6)


def test_unknown_noise_removal_is_rejected():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    try:
        agent.benders_decomposition_train(init_state=state, parallel=False, noise_removal='median')
    except ValueError as exc:
        assert 'noise_removal' in str(exc)
    else:
        raise AssertionError('an unknown noise_removal must be rejected')


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


def test_weighted_lower_cvar_spans_the_minimum_and_the_mean():
    from decision_maker.approximate_q_agent import weighted_lower_cvar
    values = np.array([4.0, 1.0, 3.0, 2.0])
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.isclose(weighted_lower_cvar(values, weights, 1.0), weights @ values)
    assert np.isclose(weighted_lower_cvar(values, weights, 1e-9), values.min())
    assert np.isclose(weighted_lower_cvar(values, weights, 0.3), 1.0)
    assert np.isclose(weighted_lower_cvar(values, weights, 0.4), (0.3 * 1.0 + 0.1 * 2.0) / 0.4)
    levels = [1e-9, 0.1, 0.3, 0.6, 1.0]
    got = [weighted_lower_cvar(values, weights, level) for level in levels]
    assert all(got[i] <= got[i + 1] + 1e-9 for i in range(len(got) - 1))


def test_worst_case_with_alpha_one_reproduces_the_mean_criterion():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    states = distinct_states(agent.env, agent.sample_path_number)
    _, mean_theta, mean_info = agent.benders_decomposition_train(
        init_state=states, parallel=False, coefficient_bound=1e3)
    agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    _, cvar_theta, cvar_info = agent.benders_decomposition_train(
        init_state=states, parallel=False, coefficient_bound=1e3,
        training_objective='worst_case', worst_case_scope='joint', worst_case_alpha=1.0)
    assert np.isclose(cvar_info['in_sample']['mean'], mean_info['in_sample']['mean'], atol=1e-4)
    assert np.isclose(cvar_info['in_sample']['worst_case'], cvar_info['in_sample']['mean'], atol=1e-4)
    np.testing.assert_allclose(cvar_theta, mean_theta, atol=1e-3)


def test_worst_case_alpha_interpolates_between_the_minimum_and_the_mean():
    from decision_maker.approximate_q_agent import weighted_lower_cvar
    agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    states = distinct_states(agent.env, agent.sample_path_number)
    kappa = np.asarray(agent.sample_path_weights, dtype=float)
    reported = {}
    for alpha in (None, 0.5, 1.0):
        agent, _ = make_agent(AbsorptionALPPenaltyFunction)
        _, theta, info = agent.benders_decomposition_train(
            init_state=states, parallel=False, coefficient_bound=1e3,
            training_objective='worst_case', worst_case_scope='joint', worst_case_alpha=alpha)
        theta = np.asarray(theta, dtype=float)
        residuals, _ = agent.coefficient_model.evaluate_action(theta, parallel=False)
        level = float(agent.generating_function.approximate_value(agent._state_relevance_state(states), theta))
        expected = level + weighted_lower_cvar(residuals, kappa, 1e-12 if alpha is None else alpha)
        assert np.isclose(info['in_sample']['worst_case'], expected, atol=1e-6)
        reported[alpha] = info['in_sample']['worst_case']
    assert reported[None] <= reported[0.5] + 1e-6 <= reported[1.0] + 1e-6


def test_runner_records_the_worst_case_alpha():
    with mock.patch.object(cli, 'write_command_file'):
        (record,) = cli.main(['train', 'base_toy_study', '--penalty-function', 'absorption_alp_penalty',
                              '--coefficient-bound', '1000', '--objective', 'worst_case',
                              '--worst-case-scope', 'joint', '--worst-case-alpha', '0.5',
                              '--dat', 'unused.dat'])
    record = dict(record)
    record['sample_path_number'] = 4
    record['experiment_name'] = record['experiment_name'] + '_cvar_unittest'
    assert record['agent_args']['agent_args']['worst_case_alpha'] == 0.5
    folder = os.path.join('experiments', 'results', record['experiment_name'])
    try:
        out = run.train_penalty_coefficients_for_env(**record, grb_env=None, grb_sub_envs=None, job_id='cvar_test')
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert out['worst_case_alpha'] == 0.5
    assert set(out['in_sample']) == {'mean', 'std', 'half_width', 'worst_case'}
    assert out['in_sample']['worst_case'] <= out['in_sample']['mean'] + 1e-6


def test_worst_case_aggregation_weighs_only_the_scenarios_it_is_given():
    from decision_maker.approximate_q_agent import weighted_lower_cvar
    agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    kappa = np.asarray(agent.sample_path_weights, dtype=float)
    blocks = np.array([0, 0, 1, 1])
    values = np.array([5.0, 2.0, 9.0, 4.0])
    subset = np.array([0, 2])
    for alpha in (None, 0.4, 1.0):
        objective_fn = agent.training_objective_fn('worst_case', blocks, worst_case_alpha=alpha)
        got = objective_fn(np.zeros(agent.generating_function.number_of_coefficients), values[subset], subset)
        weights = kappa[subset]
        expected = sum(
            weights[blocks[subset] == b].sum() * (
                values[subset][blocks[subset] == b].min() if alpha is None
                else weighted_lower_cvar(values[subset][blocks[subset] == b],
                                         weights[blocks[subset] == b], alpha))
            for b in np.unique(blocks[subset]))
        assert np.isclose(got, expected, atol=1e-9), f'alpha {alpha}: {got} vs {expected}'
    full = agent.training_objective_fn('worst_case', blocks, worst_case_alpha=0.4)
    everything = full(np.zeros(agent.generating_function.number_of_coefficients), values, np.arange(4))
    expected_full = sum(
        kappa[blocks == b].sum() * weighted_lower_cvar(values[blocks == b], kappa[blocks == b], 0.4)
        for b in (0, 1))
    assert np.isclose(everything, expected_full, atol=1e-9)


def test_worst_case_alpha_is_validated():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    for bad in (0.0, -0.1, 1.5):
        try:
            agent.benders_decomposition_train(init_state=state, parallel=False, training_objective='worst_case',
                                              worst_case_alpha=bad, coefficient_bound=1e3)
        except ValueError as exc:
            assert 'worst_case_alpha' in str(exc)
        else:
            raise AssertionError(f'alpha {bad} must be rejected')


def test_relevance_state_reproduces_the_weighted_average_value():
    agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    states = distinct_states(agent.env, agent.sample_path_number)
    theta = np.linspace(-2, 3, agent.generating_function.number_of_coefficients)
    relevance_state = agent._state_relevance_state(states)
    level = agent.sample_path_weights @ initial_state_features(agent, states) @ theta
    assert np.isclose(agent.generating_function.approximate_value(relevance_state, theta), level, atol=1e-9)


def test_approximate_value_is_the_affine_value_function_of_the_state():
    import gurobipy as gp
    config = get_config_by_type('toy')
    env = config.env
    function = AbsorptionALPPenaltyFunction(env)
    theta = np.arange(1, function.number_of_coefficients + 1, dtype=float)
    env.reset_random_seeds()
    state = env.generate_initial_state()
    phi = function.state_features(state)
    assert phi.shape == (function.number_of_coefficients,) and phi[0] == 1.0
    assert np.isclose(function.approximate_value(state, theta), theta @ phi)
    function.set_coefficients(theta)
    assert np.isclose(function.approximate_value(state),
                      function.W_0 + function.U @ np.asarray(state[0]) + function.V @ np.asarray(state[1])
                      + function.W @ np.asarray(state[2]))
    model = gp.Model('approximate_value')
    model.Params.OutputFlag = 0
    coefficient_vars = function.get_coefficient_var(model=model, coefficient_bound=10.0)
    model.setObjective(function.approximate_value(state, coefficient_vars), gp.GRB.MAXIMIZE)
    model.optimize()
    assert np.isclose(model.ObjVal, 10.0 * np.abs(phi).sum())
    try:
        LinearPenaltyFunction(env).approximate_value(state, theta)
    except NotImplementedError as exc:
        assert 'state features' in str(exc)
    else:
        raise AssertionError('a generating function without state features must say so')


def in_sample(agent, theta, init_state=None, scope='per_state'):
    theta = np.asarray(theta, dtype=float)
    values, _ = agent.coefficient_model.evaluate_action(theta, parallel=False)
    kappa = np.asarray(agent.sample_path_weights, dtype=float)
    costs = np.asarray(values, dtype=float)
    result = {}
    if init_state is not None:
        features = initial_state_features(agent, init_state)
        costs = costs + features @ theta
        blocks = (np.zeros(agent.sample_path_number, dtype=int) if scope == 'joint'
                  else state_blocks(agent, init_state))
        minima = np.array([values[blocks == b].min() for b in range(blocks.max() + 1)])
        result['worst_case'] = float(kappa @ features @ theta) + float(np.bincount(blocks, weights=kappa) @ minima)
    from metaheuristic_algorithm.benders_decomposition_solver import objective_confidence_interval
    mean, half_width = objective_confidence_interval(costs, weights=kappa, strata=np.asarray(agent.sample_path_strata))
    result['mean'] = mean
    result['half_width'] = half_width
    result['std'] = float(np.sqrt(kappa @ (costs - mean) ** 2))
    return result


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


def test_worst_case_objective_trains_from_initial_coefficients_and_reports_in_sample():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    obj_mean, theta_mean, info_mean = agent.benders_decomposition_train(init_state=state, parallel=False, coefficient_bound=1e3)
    assert 'initial_in_sample' not in info_mean and set(info_mean['in_sample']) == {'mean', 'std', 'half_width'}
    assert obj_mean == info_mean['in_sample']['mean']
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    obj, theta, info = agent.benders_decomposition_train(
        init_state=state, parallel=False, training_objective='worst_case', initial_coefficients=theta_mean, coefficient_bound=1e3)
    assert set(info['in_sample']) == set(info['initial_in_sample']) == {'mean', 'std', 'half_width', 'worst_case'}
    expected = in_sample(agent, theta, state)
    for key in expected:
        assert np.isclose(info['in_sample'][key], expected[key], atol=1e-6)
    assert obj == info['in_sample']['mean']
    assert info['in_sample']['worst_case'] >= info['initial_in_sample']['worst_case'] - 1e-6
    assert np.isclose(info['initial_in_sample']['mean'], obj_mean, atol=1e-6)
    assert info['in_sample']['worst_case'] >= in_sample(agent, theta_mean, state)['worst_case'] - 1e-6


def test_linear_penalty_trains_the_mean_but_not_the_worst_case():
    agent, state = make_agent()
    obj, theta, info = agent.benders_decomposition_train(init_state=state, parallel=False)
    assert set(info['in_sample']) == {'mean', 'std', 'half_width'}
    for scope in ('per_state', 'joint'):
        agent, state = make_agent()
        try:
            agent.benders_decomposition_train(init_state=state, parallel=False, training_objective='worst_case',
                                              worst_case_scope=scope, coefficient_bound=1e3)
        except NotImplementedError as exc:
            assert 'state features' in str(exc)
        else:
            raise AssertionError('the worst_case objective needs state features')


def test_master_builder_adds_one_xi_per_minimum_block():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    master, coefficients, theta = agent.train_master_builder_fn(
        objective='worst_case', xi_blocks=[0, 1, 0, 1], relevance_state=state)
    names = [v.VarName for v in master.getVars()]
    assert 'xi[0]' in names and 'xi[1]' in names and 'xi[2]' not in names
    assert master.NumConstrs >= theta.shape[0]
    joint, _, _ = agent.train_master_builder_fn(
        objective='worst_case', xi_blocks=[0, 0, 0, 0], relevance_state=state)
    names = [v.VarName for v in joint.getVars()]
    assert 'xi[0]' in names and 'xi[1]' not in names
    try:
        agent.train_master_builder_fn(objective='max')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown objective must be rejected')


def test_runner_records_objective_and_warm_start():
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
        inner['training_objective'] = 'worst_case'
        inner['paths_per_state'] = 2
        inner['worst_case_scope'] = 'per_state'
        inner['initial_coefficients'] = baseline['coefficients']
        inner['initial_coefficients_source'] = {'file': 'unittest', 'uid': baseline['uid']}
        record['uid'] = record['uid'] + '_wc'
        out = run.train_penalty_coefficients_for_env(**record, grb_env=None, grb_sub_envs=None, job_id='obj_test')
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert baseline['training_objective'] == 'mean' and baseline['initial_coefficients_source'] is None
    assert baseline['initial_in_sample'] is None and baseline['tight_penalized_lower_bound'] == baseline['in_sample']['mean']
    assert out['training_objective'] == 'worst_case' and out['initial_coefficients_source']['uid'] == baseline['uid']
    assert baseline['paths_per_state'] is None and out['paths_per_state'] == 2
    assert baseline['worst_case_scope'] is None and out['worst_case_scope'] == 'per_state'
    assert out['in_sample']['worst_case'] >= out['initial_in_sample']['worst_case'] - 1e-6
    assert out['tight_penalized_lower_bound'] == out['in_sample']['mean']


def test_fixed_coefficients_are_pinned_in_master_and_warm_start():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    _, theta_mean, _ = agent.benders_decomposition_train(init_state=state, parallel=False, coefficient_bound=1e3)
    warm = np.array(theta_mean, dtype=float)
    warm[0] = 123.0
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    obj, theta, info = agent.benders_decomposition_train(
        init_state=state, parallel=False, training_objective='worst_case', initial_coefficients=warm,
        fixed_coefficients={0: 0.0}, coefficient_bound=1e3)
    assert theta[0] == 0.0 and warm[0] == 123.0
    warm[0] = 0.0
    assert np.isclose(info['initial_in_sample']['worst_case'], in_sample(agent, warm, state)['worst_case'], atol=1e-6)
    assert info['in_sample']['worst_case'] >= info['initial_in_sample']['worst_case'] - 1e-6
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


def test_mean_centering_shifts_every_scenario_by_theta_dot_noise_mean():
    raw_agent, state = make_agent(AbsorptionALPPenaltyFunction)
    _, theta_raw, _ = raw_agent.benders_decomposition_train(init_state=state, parallel=False)
    centered_agent, state = make_agent(AbsorptionALPPenaltyFunction)
    obj, theta, info = centered_agent.benders_decomposition_train(
        init_state=state, parallel=False, noise_removal='mean', initial_coefficients=theta_raw)
    noise_mean = centered_agent.noise_mean
    seed_values, _ = centered_agent.coefficient_model.evaluate_action(np.asarray(theta_raw, dtype=float), parallel=False)
    for worker in centered_agent.coefficient_model.workers:
        assert np.isclose(worker.initial_cut[0], seed_values[worker.subproblem_id], atol=1e-6)
        assert worker.initial_cut is centered_agent.subproblem_initial_cuts[worker.subproblem_id]
    assert noise_mean.shape == (centered_agent.generating_function.number_of_coefficients,)
    assert np.abs(noise_mean).max() > 0
    for probe in (np.zeros_like(noise_mean), np.asarray(theta, dtype=float), np.linspace(-1, 1, noise_mean.size)):
        raw_values, _ = raw_agent.coefficient_model.evaluate_action(probe, parallel=False)
        centered_values, _ = centered_agent.coefficient_model.evaluate_action(probe, parallel=False)
        np.testing.assert_allclose(centered_values, raw_values - probe @ noise_mean, atol=1e-6)
    assert set(info['in_sample']) == {'mean', 'std', 'half_width'}
    assert np.isclose(info['in_sample']['mean'],
                      centered_agent.sample_path_weights @ seed_values + np.asarray(theta) @ noise_mean
                      + centered_agent.sample_path_weights @ (
                          centered_agent.coefficient_model.evaluate_action(np.asarray(theta, dtype=float), parallel=False)[0]
                          - seed_values), atol=1e-6)
    assert obj == info['in_sample']['mean']


def test_runner_records_noise_centering():
    with mock.patch.object(cli, 'write_command_file'):
        (record,) = cli.main(['train', 'base_toy_study', '--penalty-function', 'absorption_alp_penalty', '--noise-removal', 'mean', '--dat', 'unused.dat'])
    record = dict(record)
    record['sample_path_number'] = 4
    record['experiment_name'] = record['experiment_name'] + '_center_unittest'
    assert record['agent_args']['agent_args']['noise_removal'] == 'mean'
    folder = os.path.join('experiments', 'results', record['experiment_name'])
    assert not os.path.exists(folder)
    try:
        out = run.train_penalty_coefficients_for_env(**record, grb_env=None, grb_sub_envs=None, job_id='center_test')
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    assert set(out['in_sample']) == {'mean', 'std', 'half_width'}
    assert out['tight_penalized_lower_bound'] == out['in_sample']['mean']


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
        initial_coefficients=warm, noise_removal='mean')
    assert abs(theta[0]) <= 1e6 and max(abs(v) for v in theta[1:]) <= 10.0 + 1e-9
    assert np.isclose(info['initial_in_sample']['mean'], in_sample(agent, warm)['mean'] + warm @ agent.noise_mean, atol=1e-6)


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


def test_per_state_scope_averages_the_per_state_minimum_of_the_pathwise_values():
    agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    two = distinct_states(agent.env, 2)
    states = [two[sid % 2] for sid in range(agent.sample_path_number)]
    obj, theta, info = agent.benders_decomposition_train(init_state=states, parallel=False, training_objective='worst_case', coefficient_bound=1e3)
    theta = np.asarray(theta, dtype=float)
    mean_agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    mean_agent.benders_decomposition_train(init_state=states, parallel=False, coefficient_bound=1e3)
    features = initial_state_features(agent, states)
    for probe in (theta, np.zeros_like(theta), np.linspace(-1, 1, theta.size)):
        residuals, _ = agent.coefficient_model.evaluate_action(probe, parallel=False)
        pathwise, _ = mean_agent.coefficient_model.evaluate_action(probe, parallel=False)
        np.testing.assert_allclose(residuals, pathwise - features @ probe, atol=1e-6)
    expected = in_sample(agent, theta, states)
    assert np.isclose(info['in_sample']['worst_case'], expected['worst_case'], atol=1e-6)
    assert np.isclose(info['in_sample']['mean'], expected['mean'], atol=1e-6)
    assert obj == info['in_sample']['mean']
    pathwise, _ = mean_agent.coefficient_model.evaluate_action(theta, parallel=False)
    kappa = np.asarray(agent.sample_path_weights, dtype=float)
    by_hand = ((kappa[0] + kappa[2]) * min(pathwise[0], pathwise[2])
               + (kappa[1] + kappa[3]) * min(pathwise[1], pathwise[3]))
    assert np.isclose(info['in_sample']['worst_case'], by_hand, atol=1e-6)
    assert np.isclose(info['in_sample']['mean'], kappa @ pathwise, atol=1e-6)
    assert info['in_sample']['worst_case'] <= info['in_sample']['mean'] + 1e-6


def test_worst_case_with_a_shared_state_is_the_minimum_over_all_paths():
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    obj, theta, info = agent.benders_decomposition_train(init_state=state, parallel=False, training_objective='worst_case', coefficient_bound=1e3)
    theta = np.asarray(theta, dtype=float)
    mean_agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    mean_agent.benders_decomposition_train(init_state=state, parallel=False, coefficient_bound=1e3)
    pathwise, _ = mean_agent.coefficient_model.evaluate_action(theta, parallel=False)
    assert np.isclose(info['in_sample']['worst_case'], pathwise.min(), atol=1e-6)
    assert np.isclose(info['in_sample']['mean'], agent.sample_path_weights @ pathwise, atol=1e-6)
    for attribute in ('subproblem_initial_states', 'level_vector', 'scenario_features'):
        assert not hasattr(agent, attribute)
    joint, _ = make_agent(AbsorptionALPPenaltyFunction)
    _, joint_theta, joint_info = joint.benders_decomposition_train(
        init_state=state, parallel=False, training_objective='worst_case',
        worst_case_scope='joint', coefficient_bound=1e3)
    assert np.isclose(joint_info['in_sample']['worst_case'], info['in_sample']['worst_case'], atol=1e-4)
    np.testing.assert_allclose(joint_theta, theta, atol=1e-4)


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


def test_worst_case_rejects_missing_or_unrepeated_initial_states():
    agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    for init_state in (None, distinct_states(agent.env, agent.sample_path_number)):
        try:
            agent.benders_decomposition_train(init_state=init_state, parallel=False, training_objective='worst_case', coefficient_bound=1e3)
        except ValueError as exc:
            assert 'init_state' in str(exc) or 'per initial state' in str(exc)
        else:
            raise AssertionError('worst_case without repeated initial states must be rejected')
    agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    try:
        agent.benders_decomposition_train(init_state=None, parallel=False, training_objective='worst_case',
                                          worst_case_scope='joint', coefficient_bound=1e3)
    except ValueError as exc:
        assert 'init_state' in str(exc)
    else:
        raise AssertionError('the joint scope still needs initial states')
    agent, state = make_agent(AbsorptionALPPenaltyFunction)
    try:
        agent.benders_decomposition_train(init_state=state, parallel=False, training_objective='worst_case',
                                          worst_case_scope='every_path', coefficient_bound=1e3)
    except ValueError as exc:
        assert 'scope' in str(exc)
    else:
        raise AssertionError('an unknown scope must be rejected')


def test_joint_scope_takes_one_minimum_across_all_initial_states():
    agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    states = distinct_states(agent.env, agent.sample_path_number)
    obj, theta, info = agent.benders_decomposition_train(
        init_state=states, parallel=False, training_objective='worst_case',
        worst_case_scope='joint', coefficient_bound=1e3)
    theta = np.asarray(theta, dtype=float)
    mean_agent, _ = make_agent(AbsorptionALPPenaltyFunction)
    mean_agent.benders_decomposition_train(init_state=states, parallel=False, coefficient_bound=1e3)
    features = initial_state_features(agent, states)
    for probe in (theta, np.zeros_like(theta), np.linspace(-1, 1, theta.size)):
        residuals, _ = agent.coefficient_model.evaluate_action(probe, parallel=False)
        pathwise, _ = mean_agent.coefficient_model.evaluate_action(probe, parallel=False)
        np.testing.assert_allclose(residuals, pathwise - features @ probe, atol=1e-6)
    for sid, (value, gradient, seed) in agent.subproblem_initial_cuts.items():
        assert np.isclose(value, mean_agent.subproblem_initial_cuts[sid][0] - features[sid] @ seed, atol=1e-6)
        np.testing.assert_allclose(gradient, mean_agent.subproblem_initial_cuts[sid][1] - features[sid], atol=1e-6)
    expected = in_sample(agent, theta, states, scope='joint')
    assert np.isclose(info['in_sample']['worst_case'], expected['worst_case'], atol=1e-6)
    pathwise, _ = mean_agent.coefficient_model.evaluate_action(theta, parallel=False)
    assert np.isclose(info['in_sample']['mean'], agent.sample_path_weights @ pathwise, atol=1e-6)
    assert obj == info['in_sample']['mean']
    assert info['in_sample']['worst_case'] <= info['in_sample']['mean'] + 1e-6
