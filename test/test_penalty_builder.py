"""The single path builder: legacy golden regression and absorption-form checks.

Run from the repo root:  python -m test.test_penalty_builder

Part 1 replays ``test/golden/legacy_penalty.json`` (recorded by
``test/record_legacy_golden.py`` on the pre-refactor tree) through the
rewritten code and asserts every legacy number at rtol 1e-9. Part 2 checks
the properties the absorption form must have.
"""
import json
import os
import tempfile

import numpy as np
import gurobipy as gp
from gurobipy import GRB

import run
from decision_maker import ApproxQAgent, MyopicAgent
from generating_function import AbsorptionLinearPenaltyFunction, LegacyForm, LinearPenaltyFunction
from importance_sampling import FixedLengthProposal, GeometricLengthProposal, SamplePath, Terminal
from importance_sampling.proposals import MixtureGeometricStratifiedQMCProposal
from metaheuristic_algorithm import BendersDecompositionSolver
from test.record_legacy_golden import (
    GOLDEN_PATH, MIXTURE, SCENARIOS, fresh_env, make_agent, per_scenario_states, thetas,
)
from utils import acquire_grb_env

RTOL = 1e-9
# 5 paths under the golden mixture (lambda_0 = 0.5): 2 long + 3 short, so the
# stratum weights kappa are NOT uniform (4 paths would give 2 + 2, uniform).
HINDSIGHT_SCENARIOS = 5


def golden():
    with open(GOLDEN_PATH) as handle:
        return json.load(handle)


def same_states(observed, expected):
    """Component-wise equality of (regular, overtime, waitlist) state tuples."""
    return len(observed) == len(expected) and all(
        len(a) == len(b) and all(np.array_equal(np.asarray(c1, dtype=float), np.asarray(c2, dtype=float))
                                 for c1, c2 in zip(a, b))
        for a, b in zip(observed, expected))


def close(observed, expected, rtol=RTOL, atol=1e-6, what=''):
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert observed.shape == expected.shape, (what, observed.shape, expected.shape)
    if not np.allclose(observed, expected, rtol=rtol, atol=atol):
        worst = np.argmax(np.abs(observed - expected) - atol - rtol * np.abs(expected))
        raise AssertionError(f"{what}: mismatch at {worst}: {observed.flat[worst]!r} vs golden {expected.flat[worst]!r}")


# --------------------------------------------------------------------------- goldens
def test_golden_ir_evaluation(grb_env):
    block = golden()['ir_evaluation']
    config, env = fresh_env()
    state, _ = env.reset(**config.reset_params)
    paths = {int(k): np.array(v) for k, v in block['paths'].items()}
    for entry in block['entries']:
        agent = make_agent(env, grb_env, block['theta'][entry['theta']], sample_path_number=2)
        value = agent.calculate_information_relaxation_cost(
            state, paths[entry['length']], period_weights=entry['period_weights'])
        close(value, entry['value'], what=f"IR theta={entry['theta']} L={entry['length']} weighted={entry['period_weights'] is not None}")


def test_golden_training_subproblems(grb_env):
    block = golden()['training_subproblems']
    for label, proposal in (('default', None), ('mixture', MixtureGeometricStratifiedQMCProposal(**MIXTURE))):
        config, env = fresh_env()
        gamma = env.discount_factor
        agent = make_agent(env, grb_env, None, proposal=proposal)
        states = per_scenario_states(env, SCENARIOS)
        workers = agent._build_training_workers(init_state=states, parallel=False)
        close(agent.sample_path_weights, block[label]['path_weights'], what=f'{label} path weights')
        assert list(agent.sample_path_strata) == block[label]['path_strata']
        for sid, expected in enumerate(block[label]['scenarios']):
            path = agent.sample_paths[sid]
            assert np.array_equal(path.arrivals, np.array(expected['arrivals'])), f'{label} scenario {sid} arrivals'
            close(path.likelihood_ratios, expected['likelihood_ratios'], what=f'{label} {sid} ratios')
            if label == 'default':
                assert np.all(path.survival_weights == 1.0)
            else:
                close(path.survival_weights[1:], gamma * np.asarray(expected['likelihood_ratios']), what='survival')
            assert path.terminal is Terminal.ABSORBED
            close(agent.subproblem_initial_cuts[sid][0], expected['initial_cut_value'], what=f'{label} {sid} v0')
            close(agent.subproblem_initial_cuts[sid][1], expected['initial_cut_gradient'], what=f'{label} {sid} g0')
            for name, solve in expected['solves'].items():
                feasible, value, gradient = workers[sid].solve(np.asarray(block[label]['theta'][name], dtype=float))
                assert feasible
                close(value, solve['value'], what=f'{label} {sid} value({name})')
                close(gradient, solve['gradient'], what=f'{label} {sid} gradient({name})')


def test_golden_benders_training(grb_env):
    block = golden()['benders_training']
    config, env = fresh_env()
    states = per_scenario_states(env, SCENARIOS)
    assert same_states(states, block['init_states'])
    agent = make_agent(env, grb_env, None)
    objective, coefficients, _ = agent.benders_decomposition_train(
        coefficient_bound=GRB.INFINITY, init_state=states, parallel=False)
    close(objective, block['full']['objective'], what='Benders objective')
    # The SAA value at the golden optimum is the golden objective (robust to
    # ties between optimal vertices), and the optimum itself is reproduced.
    _, at_golden = agent.coefficient_model.evaluate_action(np.asarray(block['full']['coefficients']), parallel=False)
    close(at_golden, block['full']['objective'], what='SAA value at golden theta*')
    close(coefficients, block['full']['coefficients'], rtol=1e-6, atol=1e-6, what='Benders coefficients')

    def solver_for(agent_):
        workers = agent_._build_training_workers(init_state=states, parallel=False)
        master, coefficient_vars, theta_vars = agent_.train_master_builder_fn(GRB.INFINITY)
        return BendersDecompositionSolver(master_model=master, workers=workers, imm_cost=None,
                                          theta_vars=theta_vars, action_vars=coefficient_vars,
                                          scenario_weights=agent_.sample_path_weights,
                                          scenario_strata=agent_.sample_path_strata)

    def meta(checkpoint):
        with open(checkpoint + '.meta.json') as handle:
            return json.load(handle)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = os.path.join(tmp, 'checkpoint.pickle')
        _, env1 = fresh_env()
        after_3, _ = solver_for(make_agent(env1, grb_env, None)).solve(
            is_hard_bound=True, max_iter=3, parallel=False, checkpoint_path=checkpoint, min_norm_action=True)
        meta_3 = meta(checkpoint)
        _, env2 = fresh_env()
        after_resume, _ = solver_for(make_agent(env2, grb_env, None)).solve(
            is_hard_bound=True, max_iter=2, parallel=False, checkpoint_path=checkpoint,
            resume_checkpoint_path=checkpoint, min_norm_action=True)
        meta_resume = meta(checkpoint)
    for observed, returned, expected in ((meta_3, after_3, block['after_3_iterations']),
                                         (meta_resume, after_resume, block['after_resume_2_more'])):
        assert observed['iteration'] == expected['iteration'] and observed['cut_count'] == expected['cut_count']
        close(returned, expected['returned'], what='checkpoint returned bound')
        close(observed['lower_bound'], expected['lower_bound'], what='checkpoint lower bound')
        close(observed['upper_bound'], expected['upper_bound'], what='checkpoint upper bound')


def test_golden_hindsight(grb_env):
    block = golden()['hindsight']
    for label, proposal in (('default', None), ('mixture', MixtureGeometricStratifiedQMCProposal(**MIXTURE))):
        config, env = fresh_env()
        state, _ = env.reset(**config.reset_params)
        agent = make_agent(env, grb_env, block[label]['theta'], proposal=proposal, solver_name='approx_penalized_hindsight')
        for path, arrivals in zip(agent.sample_paths, block[label]['arrivals']):
            assert np.array_equal(path.arrivals, np.array(arrivals))
        objective, action, _ = agent.hindsight_solve(state, t=1, parallel=False)
        close(objective, block[label]['objective'], what=f'hindsight objective ({label})')
        for observed, expected in zip(action, block[label]['action']):
            assert np.array_equal(np.asarray(observed), np.asarray(expected)), f'hindsight action ({label})'


def test_golden_policy_accounting(grb_env):
    block = golden()['policy_accounting']
    config, env = fresh_env()
    init_state, _ = env.reset(**config.reset_params)
    assert same_states([init_state], [block['init_state']])
    sample_path = np.array(block['sample_path'])
    generating_function = LinearPenaltyFunction(env, coefficients=block['theta'])
    entries = block['entries']
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            def run_policy(policy_id, warm_up_periods, weights, warm_up_trajectory=None, return_warm_up_trajectory=False):
                return run.calculate_policy_costs_with_penalty(
                    uid='golden', experiment_name='golden_replay', policy_id=policy_id, agent_name='myopic',
                    agent_args={}, env_args=config.args, init_state=init_state, sample_path=sample_path,
                    warm_up_periods=warm_up_periods, generating_function=generating_function, grb_env=grb_env,
                    period_weights=weights, warm_up_trajectory=warm_up_trajectory,
                    return_warm_up_trajectory=return_warm_up_trajectory)

            def check(result, expected, what):
                for key in ('penalized_cost', 'total_cost', 'total_penalty', 'costs', 'penalties'):
                    close(result[key], expected[key], what=f'{what} {key}')
                assert same_states([result['warmup_state']], [expected['warmup_state']]), what
                # penalties == expected - realized wherever the terms were recorded
                # (a legacy prefix carries NaN placeholders there).
                difference = np.asarray(result['expected_terms'][:len(result['penalties'])]) - np.asarray(result['realized_terms'])
                recorded = np.isfinite(difference)
                close(difference[recorded], np.asarray(result['penalties'])[recorded], what=f'{what} terms vs penalties')

            check(run_policy('myopic_w0', 0, entries['warm_up_0']['period_weights']), entries['warm_up_0'], 'warm_up_0')
            full = run_policy('myopic_w2', 2, entries['warm_up_2']['period_weights'], return_warm_up_trajectory=True)
            check(full, entries['warm_up_2'], 'warm_up_2')
            trajectory = full['warm_up_trajectory']
            for key in ('costs', 'penalties'):
                close(trajectory[key], entries['warm_up_trajectory'][key], what=f'trajectory {key}')
            assert len(trajectory['expected_terms']) == 2 and len(trajectory['realized_terms']) == 2
            seeded = run_policy('myopic_seeded', 2, entries['seeded']['period_weights'], warm_up_trajectory=trajectory)
            check(seeded, entries['seeded'], 'seeded')
            # A legacy prefix without the per-period terms is padded and sliced away.
            legacy_trajectory = {k: v for k, v in trajectory.items() if k not in ('expected_terms', 'realized_terms')}
            legacy_seeded = run_policy('myopic_seeded_legacy', 2, entries['seeded']['period_weights'], warm_up_trajectory=legacy_trajectory)
            check(legacy_seeded, entries['seeded'], 'seeded from legacy prefix')
            assert np.isnan(legacy_seeded['expected_terms'][:2]).all()
        finally:
            os.chdir(old_cwd)


# --------------------------------------------------------------------------- absorption form
def numeric(components):
    return tuple(np.array(c, dtype=float) for c in components)


def myopic_rollout(env, myopic, gf, theta, state, path, gamma):
    """Weighted stage costs and theta . Phi of the myopic policy along ``path``."""
    form = gf.form('evaluation')
    W = form.period_weights(path, gamma)
    cost, expected_terms, realized_terms = 0.0, [], []
    for s in range(path.periods):
        _, action, _ = myopic.solve(state, t=s + 1)
        action = numeric(action)
        cost += W[s] * float(env.cost_fn(state, action, is_var=False))
        expected_terms.append(gf.expected_value(theta, state, action))
        if s < path.periods - 1:
            realized_terms.append(gf.value(theta, state, action, path.arrivals[s]))
            state = numeric(env.get_next_state(state, action, path.arrivals[s]))
    return cost, form.combine(path, gamma, expected_terms, realized_terms)


def test_absorption_bound_never_exceeds_penalized_myopic_cost(grb_env):
    config, env = fresh_env()
    state = numeric(env.reset(**config.reset_params)[0])
    theta = thetas(env)['a']
    gf = AbsorptionLinearPenaltyFunction(env, coefficients=theta)
    agent = ApproxQAgent(env, discount_factor=env.discount_factor, sample_path_number=2,
                         current_decision_var_type='integer', future_decision_var_type='continuous',
                         generating_function=gf, grb_env=grb_env, subproblem_grb_envs=[grb_env])
    myopic = MyopicAgent(env, discount_factor=env.discount_factor, grb_env=grb_env)
    for proposal, terminal in ((GeometricLengthProposal(0.7), Terminal.ABSORBED), (FixedLengthProposal(3), Terminal.TRUNCATED)):
        for path in proposal.sample_paths(env.arrival_generator, 6, env.discount_factor):
            assert path.terminal is terminal
            bound = agent.calculate_information_relaxation_cost(
                state, path.arrivals, period_weights=path.survival_weights, terminal=path.terminal)
            cost, penalty = myopic_rollout(env, myopic, gf, np.asarray(theta), state, path, env.discount_factor)
            assert bound <= cost + penalty + 1e-6 * max(1.0, abs(cost + penalty)), (bound, cost, penalty)


def test_training_worker_value_equals_ir_cost(grb_env):
    config, env = fresh_env()
    theta = np.asarray(thetas(env)['a'])
    gf = AbsorptionLinearPenaltyFunction(env, coefficients=theta.tolist())
    agent = ApproxQAgent(env, discount_factor=env.discount_factor, sample_path_number=3,
                         current_decision_var_type='continuous', future_decision_var_type='continuous',
                         generating_function=gf, sample_path_proposal=GeometricLengthProposal(0.7),
                         grb_env=grb_env, subproblem_grb_envs=[grb_env])
    state = per_scenario_states(env, 1)[0]
    workers = agent._build_training_workers(init_state=state, parallel=False)
    for worker, path in zip(workers, agent.sample_paths):
        feasible, value, gradient = worker.solve(theta)
        assert feasible
        bound = agent.calculate_information_relaxation_cost(
            state, path.arrivals, period_weights=path.survival_weights, terminal=path.terminal)
        close(value, bound, rtol=1e-7, what='worker value vs IR cost')
        # Q is concave in theta with subgradient Phi(x*): the cut at theta
        # bounds Q(0) from above and the build-time cut at 0 bounds Q(theta).
        zero_value = worker.solve(np.zeros_like(theta))[1]
        tolerance = 1e-6 * max(1.0, abs(value), abs(zero_value))
        assert zero_value <= value - float(theta @ gradient) + tolerance
        assert value <= zero_value + float(theta @ worker.initial_cut[1]) + tolerance


def test_hindsight_objective_is_path_weighted_ir_at_its_action(grb_env):
    config, env = fresh_env()
    state = numeric(env.reset(**config.reset_params)[0])
    theta = thetas(env)['a']
    gf = AbsorptionLinearPenaltyFunction(env, coefficients=theta)
    agent = ApproxQAgent(env, discount_factor=env.discount_factor, sample_path_number=HINDSIGHT_SCENARIOS,
                         current_decision_var_type='integer', future_decision_var_type='continuous',
                         generating_function=gf, sample_path_proposal=MixtureGeometricStratifiedQMCProposal(**MIXTURE),
                         solver_name='approx_penalized_hindsight', grb_env=grb_env, subproblem_grb_envs=[grb_env])
    kappa = np.asarray(agent.sample_path_weights)
    assert not np.allclose(kappa, kappa[0])
    objective, action, _ = agent.hindsight_solve(state, t=1, parallel=False)
    bounds = [agent.calculate_information_relaxation_cost(
        state, path.arrivals, period_weights=path.survival_weights, terminal=path.terminal, first_action=action)
        for path in agent.sample_paths]
    close(objective, float(kappa @ np.asarray(bounds)), rtol=1e-6, what='hindsight objective vs kappa-weighted IR')


def test_path_without_arrivals_carries_only_the_expected_term(grb_env):
    config, env = fresh_env()
    state = numeric(env.reset(**config.reset_params)[0])
    theta = np.asarray(thetas(env)['a'])
    myopic = MyopicAgent(env, discount_factor=env.discount_factor, grb_env=grb_env)
    action = numeric(myopic.solve(state, t=1)[1])
    path = SamplePath(np.zeros((0, env.num_types)), Terminal.ABSORBED, [1.0], [])
    for cls, expected in ((AbsorptionLinearPenaltyFunction, None), (LinearPenaltyFunction, 0.0)):
        gf = cls(env, coefficients=theta.tolist())
        agent = ApproxQAgent(env, discount_factor=env.discount_factor, sample_path_number=1,
                             generating_function=gf, grb_env=grb_env)
        model = gp.Model('tau1', env=grb_env)
        state_var = agent.get_state_var(model)
        action_var = agent.get_action_var(model, GRB.CONTINUOUS)
        for var_block, values in zip(state_var + action_var, state + action):
            var_block.lb = values
            var_block.ub = values
        cost, Phi, _, _ = agent.pathwise_terms(model, gf, gf.form('hindsight'), path, state_var, action_var,
                                               include_first_cost=False)
        assert cost == 0.0
        model.setObjective(theta @ Phi, GRB.MINIMIZE)
        model.optimize()
        if expected is None:
            expected = env.discount_factor * gf.expected_value(theta, state, action)
        close(model.ObjVal, expected, rtol=1e-9, what=f'{cls.__name__} tau=1 value')
        model.dispose()


def test_master_weights_follow_the_form(grb_env):
    config, env = fresh_env()
    theta = thetas(env)['a']
    for cls, expect_kappa in ((AbsorptionLinearPenaltyFunction, True), (LinearPenaltyFunction, False)):
        agent = ApproxQAgent(env, discount_factor=env.discount_factor, sample_path_number=HINDSIGHT_SCENARIOS,
                             generating_function=cls(env, coefficients=theta),
                             sample_path_proposal=MixtureGeometricStratifiedQMCProposal(**MIXTURE),
                             solver_name='approx_penalized_hindsight', grb_env=grb_env)
        weights = agent.hindsight_scenario_weights(agent.generating_function.form('hindsight'))
        assert not np.allclose(agent.sample_path_weights, agent.sample_path_weights[0])
        expected = (np.asarray(agent.sample_path_weights) if expect_kappa
                    else np.full(HINDSIGHT_SCENARIOS, 1.0 / HINDSIGHT_SCENARIOS))
        close(weights, expected, what=f'{cls.__name__} master weights')
        model, _, theta_vars, _, _ = agent.hindsight_master_builder_fn()
        close(np.asarray(theta_vars.Obj), expected, what=f'{cls.__name__} master objective coefficients')
        model.dispose()


def test_accounting_total_penalty_is_theta_dot_phi(grb_env):
    config, env = fresh_env()
    init_state = numeric(env.reset(**config.reset_params)[0])
    theta = np.asarray(thetas(env)['a'])
    gf = AbsorptionLinearPenaltyFunction(env, coefficients=theta.tolist())
    prefix = env.reset_arrivals(stop_time=2)
    tail = GeometricLengthProposal(0.7).sample_paths(env.arrival_generator, 1, env.discount_factor)[0]
    sample_path = np.concatenate((prefix, tail.arrivals), axis=0)
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            result = run.calculate_policy_costs_with_penalty(
                uid='absorption', experiment_name='absorption_accounting', policy_id='myopic', agent_name='myopic',
                agent_args={}, env_args=config.args, init_state=init_state, sample_path=sample_path,
                warm_up_periods=len(prefix), generating_function=gf, grb_env=grb_env,
                period_weights=tail.survival_weights, terminal=tail.terminal.value)
        finally:
            os.chdir(old_cwd)
    warm_up = len(prefix)
    expected_terms = result['expected_terms'][warm_up:]
    realized_terms = result['realized_terms'][warm_up:]
    assert len(expected_terms) == tail.periods and len(realized_terms) == tail.length
    recomputed = gf.form('evaluation').combine(tail, env.discount_factor, expected_terms, realized_terms)
    close(result['total_penalty'], recomputed, what='accounting total_penalty')
    close(np.asarray(result['expected_terms'][:len(result['penalties'])]) - np.asarray(result['realized_terms']),
          result['penalties'], what='penalties = expected - realized')
    close(result['total_cost'], float(np.dot(tail.survival_weights, result['costs'][warm_up:])), what='accounting total_cost')
    assert result['penalized_cost'] == result['total_cost'] + result['total_penalty']


def test_generating_function_registry():
    config, env = fresh_env()
    assert isinstance(run._build_generating_function(env, {'name': 'absorption_linear_penalty'}), AbsorptionLinearPenaltyFunction)
    assert type(run._build_generating_function(env, None)) is LinearPenaltyFunction
    try:
        run._build_generating_function(env, {'name': 'quadratic_penalty'})
    except ValueError:
        pass
    else:
        raise AssertionError('unknown generating function name accepted')


if __name__ == '__main__':
    grb_env = acquire_grb_env({'Threads': 1}, verbose=False)
    test_golden_ir_evaluation(grb_env)
    test_golden_training_subproblems(grb_env)
    test_golden_benders_training(grb_env)
    test_golden_hindsight(grb_env)
    test_golden_policy_accounting(grb_env)
    test_absorption_bound_never_exceeds_penalized_myopic_cost(grb_env)
    test_training_worker_value_equals_ir_cost(grb_env)
    test_hindsight_objective_is_path_weighted_ir_at_its_action(grb_env)
    test_path_without_arrivals_carries_only_the_expected_term(grb_env)
    test_master_weights_follow_the_form(grb_env)
    test_accounting_total_penalty_is_theta_dot_phi(grb_env)
    test_generating_function_registry()
    print('All penalty-builder tests passed.')
