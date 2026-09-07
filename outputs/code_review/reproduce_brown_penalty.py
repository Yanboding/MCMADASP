"""Read-only diagnostics for the Brown–Haugh penalty review.

Run from any directory with the project's Python environment:
  PYTHONDONTWRITEBYTECODE=1 python /absolute/path/to/this/file.py

The JSON contains counterexamples in the reviewed implementation. Successful
execution means the diagnostics ran; it does not mean the invariants pass.
No optimizer license is needed and no experiment result is written.
"""
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments import get_config_by_type
from generating_function import AbsorptionForm, AbsorptionLinearPenaltyFunction
from importance_sampling import GeometricLengthProposal, TruncatedGeometricLengthProposal
from test.test_eval_proposal_generation import generate, make_test_envs


def main():
    env = get_config_by_type('toy').env
    h, k, window = env.planning_horizon, env.num_types, env.booking_window_size
    state_a = (np.r_[1, np.zeros(h - 1, dtype=int)],
               np.zeros(h, dtype=int), np.zeros(k, dtype=int))
    state_b = tuple(np.zeros_like(part) for part in state_a)
    action = (np.zeros((window, k), dtype=int), np.zeros(h, dtype=int))
    arrival = np.r_[1, np.zeros(k - 1, dtype=int)]
    gf = AbsorptionLinearPenaltyFunction(env)
    theta = np.zeros(gf.number_of_coefficients)
    theta[0] = 1.0
    gf.set_coefficients(theta)

    exact = float(sum(prob * gf.value(theta, state_a, action, delta)
                      for prob, delta in env.arrival_generator.get_system_dynamic()))
    coded = float(gf.expected_value(theta, state_a, action))
    transitions_match = all(
        all(np.array_equal(x, y) for x, y in zip(
            env.get_next_state(state_a, action, delta),
            env.get_next_state(state_b, action, delta)))
        for _, delta in env.arrival_generator.get_system_dynamic()
    )
    output = {
        'conditional_expectation': {
            'coded': coded,
            'exact_enumeration': exact,
            'absorption_increment_mean': env.discount_factor * (coded - exact),
            'zero_mean_invariant_passes': bool(np.isclose(coded, exact, atol=1e-12, rtol=0)),
        },
        'state_value_representation': {
            'next_state_maps_match_for_every_arrival': transitions_match,
            'g_from_state_a': float(gf.value(theta, state_a, action, arrival)),
            'g_from_state_b': float(gf.value(theta, state_b, action, arrival)),
            'surviving_increment_a_with_correct_mean': env.discount_factor * exact - 1.0,
            'surviving_increment_b_with_correct_mean': 0.0,
        },
        'zero_continuation_proposals': [],
    }
    for proposal in (GeometricLengthProposal(0), TruncatedGeometricLengthProposal(0, 5)):
        path = proposal.sample_paths(env.arrival_generator, 1, .9)[0]
        penalty = AbsorptionForm().combine(path, .9, [1.] * path.periods, [1.] * path.length)
        output['zero_continuation_proposals'].append({
            'proposal': type(proposal).__name__, 'length': path.length,
            'terminal': path.terminal.value,
            'survival_weights': path.survival_weights.tolist(), 'constant_g_penalty': penalty,
        })

    spec = {'type': 'mixture_geometric', 'target_discount_factor': .99,
            'discount_factor_proposal': .5, 'lambda_0': .1}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        records = generate(spec, test_sample_path_num=4)
    periods = np.arange(1, 400)
    survival_actual = .5 ** (periods - 1)
    survival_assumed = .1 * .99 ** (periods - 1) + .9 * survival_actual
    mean_cost = float(1 + np.sum(survival_actual * .99 ** periods / survival_assumed))
    output['empty_mixture_stratum'] = {
        'record_weights': [r['path_weight'] for r in records],
        'record_strata': [r['path_stratum'] for r in records],
        'unit_stage_cost_estimator_mean': mean_cost,
        'target_unit_stage_cost': 1 / (1 - .99),
        'constant_g_penalty_mean': 1 - (1 - .99) * mean_cost,
    }

    envs = make_test_envs()
    for variant in envs.values():
        variant['agent_args']['agent_args']['penalized_lowerbound_generating_function_spec'] = {
            'name': 'absorption_linear_penalty'}
    spec.update(discount_factor_proposal=.95, lambda_0=1.)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            generate(spec, test_envs=envs)
    except ValueError as error:
        output['explicit_target_mixture_endpoint'] = str(error)
    else:
        output['explicit_target_mixture_endpoint'] = 'accepted'
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
