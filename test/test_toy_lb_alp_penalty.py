import os
import tempfile
from unittest import mock

import gurobipy as gp
import pytest
from gurobipy import GRB

from metaheuristic_algorithm import RowGenerationSolver
from param_generation import cli, datasets
from param_generation.datasets import generate_test_paths_and_init_state
from param_generation.experiment_specs import build_variation_test_env
from param_generation.registry import EXPERIMENT_SPECS


def _solver(separation_callback):
    def master_builder():
        model = gp.Model('master')
        model.setParam('OutputFlag', 0)
        x = model.addVar(lb=0.0, ub=100.0, name='x')
        model.setObjective(x, GRB.MAXIMIZE)
        model.update()
        return model, x, [x]

    def get_constraint_data(model, candidate):
        return model.getVarByName('x') <= candidate

    return RowGenerationSolver(master_builder, separation_callback, get_constraint_data, initial_candidates=[10.0])


def test_row_generation_converges_only_when_separation_certifies():
    solver = _solver(lambda solution: iter([(20.0, 1e-9)]))
    solver.solve(tol=1e-6, max_iter=10)
    assert solver.converged and solver.max_violation == 1e-9 and solver.iterations == 1


def test_row_generation_does_not_converge_on_a_duplicate_violated_row():
    solver = _solver(lambda solution: iter([(10.0, 5.0)]))
    solver.solve(tol=1e-6, max_iter=10)
    assert not solver.converged and solver.max_violation == 5.0


def test_row_generation_does_not_converge_when_separation_fails():
    solver = _solver(lambda solution: iter([]))
    solver.solve(tol=1e-6, max_iter=10)
    assert not solver.converged and solver.max_violation == float('inf')


def test_row_generation_does_not_converge_when_iterations_run_out():
    cuts = iter([9.0, 8.0, 7.0])
    solver = _solver(lambda solution: iter([(next(cuts), 1.0)]))
    solver.solve(tol=1e-6, max_iter=3)
    assert not solver.converged and solver.iterations == 3


def test_alp_penalty_is_a_lowerbound_only_flag():
    parser = cli.build_parser()
    assert parser.parse_args(['lowerbound', 'toy_lb_occupancy', '--alp-penalty']).alp_penalty
    with pytest.raises(SystemExit):
        parser.parse_args(['eval', 'toy_lb_occupancy', '--policies', 'myopic', '--alp-penalty'])


def test_alp_penalty_runs_get_their_own_experiment_and_group():
    tagged = cli.alp_penalty_test_envs('toy_lb_occupancy')
    untagged = cli._apply_penalty_function(
        build_variation_test_env(EXPERIMENT_SPECS['toy_lb_occupancy']), 'absorption_alp_penalty')
    assert {key[1] for key in tagged} == {'toy_lb_occupancy_alp_alpcoef'}
    assert not {key[0] for key in tagged} & {key[0] for key in untagged}


def test_alp_penalty_rejects_fitted_penalty_policies():
    with pytest.raises(ValueError, match='fitted-coefficient policies'):
        generate_test_paths_and_init_state(
            test_envs=cli.alp_penalty_test_envs('toy_lb_occupancy'), test_sample_path_num=4,
            policy_ids=['approx_penalized_hindsight'], alp_penalty=True)


def test_alp_penalty_without_a_length_proposal_fails_before_fitting():
    with tempfile.TemporaryDirectory() as tmp, \
         mock.patch.object(datasets, 'certified_alp_coefficients', side_effect=AssertionError('fitted the ALP')):
        with pytest.raises(ValueError, match='survival weights'):
            cli.main(['lowerbound', 'toy_lb_occupancy', '--variants', '0.5', '--paths', '4',
                      '--penalty-function', 'absorption_alp_penalty', '--alp-penalty',
                      '--dat', os.path.join(tmp, 'table.dat')])


def test_policy_centering_needs_the_alp_penalty():
    with pytest.raises(ValueError, match='needs alp_penalty'):
        generate_test_paths_and_init_state(
            test_envs=cli.alp_penalty_test_envs('toy_lb_occupancy'), test_sample_path_num=4, policy_centering=True)


def _centered_record(uid, zero, penalized, centered, approximate):
    return {
        'uid': uid, 'group_id': 'g', 'experiment_name': 'x', 'mutate_val': 0.5, 'warm_up_periods': 0,
        'path_weight': None, 'path_stratum': None, 'policy_id': 'information_relaxation_only',
        'agent_name': 'information_relaxation_only', 'zero_information_relaxation_cost': zero,
        'penalized_information_relaxation_cost': penalized, 'gap_to_zero_information_relaxation': 0.0,
        'gap_to_penalized_information_relaxation': 0.0, 'approximate_value_at_init_state': approximate,
        'policy_centered_information_relaxation_cost': centered, 'alp_policy_cost': centered + 100.0,
    }


def _summarize(records):
    import json
    from experiments import get_config_by_type
    from experiments.new_result_aggregration import SimulateEvaluationResult
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, '1.jsonl'), 'w') as handle:
            handle.writelines(json.dumps(record) + '\n' for record in records)
        ser = SimulateEvaluationResult(tmp, '[0-9]*.jsonl', get_config_by_type('toy').env, is_reuse=False)
    return ser.policy_centered_comparison('g', 0.5, 0.95)


def test_policy_centered_comparison_is_paired_and_matches_numpy():
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(0)
    n = 300
    zero = rng.normal(4000, 900, n)
    centered = zero + rng.normal(300, 40, n)
    row = _summarize([_centered_record(f'u{i}', zero[i], zero[i] + 5.0, centered[i], 2500.0) for i in range(n)])
    t = stats.t.ppf(0.975, n - 1)
    assert row['policy_centered_n'] == n
    assert np.isclose(row['policy_centered_mean'], centered.mean())
    assert np.isclose(row['policy_centered_improvement_mean'], (centered - zero).mean())
    assert np.isclose(row['policy_centered_improvement_half_window'], t * (centered - zero).std(ddof=1) / np.sqrt(n))
    assert np.isclose(row['policy_centered_minus_approximate_mean'], centered.mean() - 2500.0)
    assert np.isclose(row['policy_centered_minus_approximate_half_window'], t * centered.std(ddof=1) / np.sqrt(n))
    assert np.isclose(row['alp_policy_cost_mean'], centered.mean() + 100.0)


def test_policy_centered_relative_to_a_nonpositive_approximation_is_withheld():
    row = _summarize([_centered_record(f'u{i}', 1000.0 + i, 1100.0 + i, 1200.0 + i, -187.0) for i in range(4)])
    assert row['policy_centered_relative_to_approximate_pct'] is None
