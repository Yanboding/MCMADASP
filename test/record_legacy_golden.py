"""Record golden values of the legacy penalty code (toy env, fixed seeds).

Run from the repo root ONCE on the untouched tree (before the generating
function / penalty-form refactor; it was, on 2026-09-04 -- the file in
``test/golden`` is that recording, this script now only documents it and
reads the same quantities through the current ``SamplePath`` API):

    python -m test.record_legacy_golden

It writes ``test/golden/legacy_penalty.json``; ``test/test_penalty_builder.py``
replays every entry through the rewritten code and asserts agreement at
rtol 1e-9. Everything is deterministic: fixed numpy seeds, fresh env RNGs per
block, one single-threaded Gurobi env shared by all models, serial solves.

Blocks (see the plan, Task 0):
  ir_evaluation        calculate_information_relaxation_cost, 3 paths x 2 theta,
                       period_weights None and mixture-proposal weights
  training_subproblems worker.solve(theta) value + cut gradient, 4 scenarios,
                       default proposal (u == 1) and mixture proposal (u != 1),
                       per-scenario initial states, sampled paths / ratios
  benders_training     benders_decomposition_train objective + coefficients;
                       solver objective after 3 iterations with a checkpoint,
                       and after resuming 2 more iterations from it
  hindsight            hindsight_solve objective + action, default and mixture
  policy_accounting    calculate_policy_costs_with_penalty totals for
                       warm_up 0, warm_up 2, and a run seeded from a shared
                       warm-up trajectory
"""
import json
import os
import tempfile

import numpy as np
from gurobipy import GRB

import run
from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import LinearPenaltyFunction
from importance_sampling.proposals import MixtureGeometricStratifiedQMCProposal
from metaheuristic_algorithm import BendersDecompositionSolver
from param_generation.datasets import _evaluation_period_weights
from utils import acquire_grb_env

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'golden', 'legacy_penalty.json')
THETA_SEED = 2026
SCENARIOS = 4
# 4 paths: round(0.5 * 4) = 2 long + 2 short (lambda_0 = 0.1 would round to 0 long paths).
MIXTURE = dict(target_discount_factor=0.99, discount_factor_proposal=0.95, lambda_0=0.5)


def fresh_env():
    """A toy env with every RNG (init state, stop time, arrivals) at its seed."""
    config = get_config_by_type('toy')
    env = config.env
    env.reset_random_seeds()
    return config, env


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def thetas(env):
    n = LinearPenaltyFunction(env).number_of_coefficients
    rng = np.random.default_rng(THETA_SEED)
    return {'a': (rng.normal(size=n) * 5.0).tolist(), 'b': (rng.normal(size=n) * 20.0).tolist()}


def make_agent(env, grb_env, theta, proposal=None, solver_name='approx_Q', sample_path_number=SCENARIOS):
    return ApproxQAgent(env, discount_factor=env.discount_factor,
                        sample_path_number=sample_path_number,
                        current_decision_var_type='integer',
                        future_decision_var_type='continuous',
                        generating_function=LinearPenaltyFunction(env, coefficients=theta),
                        sample_path_proposal=proposal,
                        solver_name=solver_name,
                        grb_env=grb_env, subproblem_grb_envs=[grb_env])


def per_scenario_states(env, count):
    return [tuple(np.array(c) for c in env.generate_initial_state()) for _ in range(count)]


def record_ir_evaluation(grb_env):
    config, env = fresh_env()
    state, _ = env.reset(**config.reset_params)
    theta = thetas(env)
    paths = {length: env.arrival_generator.rvs(size=length) for length in (0, 2, 4)}
    mixture = MixtureGeometricStratifiedQMCProposal(**MIXTURE)
    entries = []
    for name, coefficients in theta.items():
        agent = make_agent(env, grb_env, coefficients, sample_path_number=2)
        for length, path in paths.items():
            for weighted in (False, True):
                weights = _evaluation_period_weights(mixture, env.discount_factor, length) if weighted else None
                value = agent.calculate_information_relaxation_cost(state, path, period_weights=weights)
                entries.append({'theta': name, 'length': length, 'period_weights': weights,
                                'value': float(value)})
    return {'state': jsonable(state), 'paths': {str(k): jsonable(v) for k, v in paths.items()},
            'theta': theta, 'entries': entries}


def record_training_subproblems(grb_env):
    out = {}
    for label, proposal in (('default', None), ('mixture', MixtureGeometricStratifiedQMCProposal(**MIXTURE))):
        config, env = fresh_env()
        theta = thetas(env)
        agent = make_agent(env, grb_env, None, proposal=proposal)
        states = per_scenario_states(env, SCENARIOS)
        workers = agent._build_training_workers(init_state=states, parallel=False)
        scenarios = []
        for sid, worker in enumerate(workers):
            entry = {'init_state': jsonable(states[sid]),
                     'arrivals': jsonable(agent.sample_paths[sid].arrivals),
                     'likelihood_ratios': jsonable(agent.sample_paths[sid].likelihood_ratios),
                     'initial_cut_value': float(agent.subproblem_initial_cuts[sid][0]),
                     'initial_cut_gradient': jsonable(agent.subproblem_initial_cuts[sid][1]),
                     'solves': {}}
            for name, coefficients in theta.items():
                feasible, value, gradient = worker.solve(np.asarray(coefficients, dtype=float))
                assert feasible
                entry['solves'][name] = {'value': float(value), 'gradient': jsonable(gradient)}
            scenarios.append(entry)
        out[label] = {'theta': theta, 'path_weights': jsonable(agent.sample_path_weights),
                      'path_strata': jsonable(agent.sample_path_strata), 'scenarios': scenarios}
    return out


def record_benders_training(grb_env):
    config, env = fresh_env()
    states = per_scenario_states(env, SCENARIOS)
    agent = make_agent(env, grb_env, None)
    objective, coefficients, info = agent.benders_decomposition_train(
        coefficient_bound=GRB.INFINITY, init_state=states, parallel=False)
    full = {'objective': float(objective), 'coefficients': jsonable(coefficients)}

    def solver_for(agent_):
        workers = agent_._build_training_workers(init_state=states, parallel=False)
        master, coefficient_vars, theta_vars = agent_.train_master_builder_fn(GRB.INFINITY)
        return BendersDecompositionSolver(master_model=master, workers=workers, imm_cost=None,
                                          theta_vars=theta_vars, action_vars=coefficient_vars,
                                          scenario_weights=agent_.sample_path_weights,
                                          scenario_strata=agent_.sample_path_strata)

    def checkpoint_meta(checkpoint):
        with open(checkpoint + '.meta.json') as handle:
            meta = json.load(handle)
        return {key: meta.get(key) for key in ('iteration', 'lower_bound', 'upper_bound', 'cut_count')}

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = os.path.join(tmp, 'golden-checkpoint.pickle')
        _, env1 = fresh_env()
        after_3, _ = solver_for(make_agent(env1, grb_env, None)).solve(
            is_hard_bound=True, max_iter=3, parallel=False, checkpoint_path=checkpoint, min_norm_action=True)
        meta_3 = checkpoint_meta(checkpoint)
        _, env2 = fresh_env()
        after_resume, _ = solver_for(make_agent(env2, grb_env, None)).solve(
            is_hard_bound=True, max_iter=2, parallel=False, checkpoint_path=checkpoint,
            resume_checkpoint_path=checkpoint, min_norm_action=True)
        meta_resume = checkpoint_meta(checkpoint)
    return {'init_states': jsonable(states), 'full': full,
            'after_3_iterations': {'returned': float(after_3), **meta_3},
            'after_resume_2_more': {'returned': float(after_resume), **meta_resume}}


def record_hindsight(grb_env):
    out = {}
    for label, proposal in (('default', None), ('mixture', MixtureGeometricStratifiedQMCProposal(**MIXTURE))):
        config, env = fresh_env()
        state, _ = env.reset(**config.reset_params)
        theta = thetas(env)['a']
        agent = make_agent(env, grb_env, theta, proposal=proposal, solver_name='approx_penalized_hindsight')
        objective, action, _ = agent.hindsight_solve(state, t=1, parallel=False)
        out[label] = {'state': jsonable(state), 'theta': theta, 'objective': float(objective),
                      'action': jsonable(action),
                      'arrivals': [jsonable(path.arrivals) for path in agent.sample_paths],
                      'likelihood_ratios': [jsonable(path.likelihood_ratios) for path in agent.sample_paths]}
    return out


def record_policy_accounting(grb_env):
    config, env = fresh_env()
    init_state, _ = env.reset(**config.reset_params)
    theta = thetas(env)['a']
    sample_path = env.reset_arrivals(stop_time=6)
    mixture = MixtureGeometricStratifiedQMCProposal(**MIXTURE)
    generating_function = LinearPenaltyFunction(env, coefficients=theta)
    entries = {}
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            def run_policy(policy_id, warm_up_periods, warm_up_trajectory=None, return_warm_up_trajectory=False):
                weights = _evaluation_period_weights(mixture, env.discount_factor, len(sample_path) - warm_up_periods)
                result = run.calculate_policy_costs_with_penalty(
                    uid='golden', experiment_name='golden_legacy', policy_id=policy_id, agent_name='myopic',
                    agent_args={}, env_args=config.args, init_state=init_state, sample_path=sample_path,
                    warm_up_periods=warm_up_periods, generating_function=generating_function, grb_env=grb_env,
                    period_weights=weights, warm_up_trajectory=warm_up_trajectory,
                    return_warm_up_trajectory=return_warm_up_trajectory)
                keep = {k: result[k] for k in ('penalized_cost', 'total_cost', 'total_penalty', 'costs', 'penalties', 'warmup_state')}
                keep['period_weights'] = weights
                return result, keep

            _, entries['warm_up_0'] = run_policy('myopic_w0', 0)
            full, entries['warm_up_2'] = run_policy('myopic_w2', 2, return_warm_up_trajectory=True)
            trajectory = full['warm_up_trajectory']
            entries['seeded'] = run_policy('myopic_seeded', 2, warm_up_trajectory=trajectory)[1]
            entries['warm_up_trajectory'] = trajectory
        finally:
            os.chdir(old_cwd)
    return {'init_state': jsonable(init_state), 'theta': theta, 'sample_path': jsonable(sample_path),
            'entries': jsonable(entries)}


def main():
    grb_env = acquire_grb_env({'Threads': 1}, verbose=False)
    golden = {
        'config_type': 'toy',
        'theta_seed': THETA_SEED,
        'scenarios': SCENARIOS,
        'mixture': MIXTURE,
        'ir_evaluation': record_ir_evaluation(grb_env),
        'training_subproblems': record_training_subproblems(grb_env),
        'benders_training': record_benders_training(grb_env),
        'hindsight': record_hindsight(grb_env),
        'policy_accounting': record_policy_accounting(grb_env),
    }
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    with open(GOLDEN_PATH, 'w') as handle:
        json.dump(golden, handle, indent=1)
    print(f"wrote {GOLDEN_PATH}")


if __name__ == '__main__':
    main()
