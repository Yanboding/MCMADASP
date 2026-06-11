"""Integration test for the PO-derived policy training pipeline.

Runs the three-stage ``ApproxQAgent.po_policy_train`` on the toy config:
1. Benders pathwise optimization with the linear penalty basis;
2. extraction of the regression dataset: X = the per-scenario initial states
   used in training, Y = the per-scenario information-relaxation costs at the
   trained coefficients (|X| = |Y| = sample_path_number);
3. least-squares fit of the quadratic value function and a greedy solve.
"""
import numpy as np

from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import MulticlassLinearPenaltyFunction, MulticlassQuadraticPenaltyFunction


def main():
    config = get_config_by_type('toy')
    env = config.env
    penalty_generating_function = MulticlassLinearPenaltyFunction(env)
    agent = ApproxQAgent(env,
                         discount_factor=env.discount_factor,
                         sample_path_number=4,
                         paths_per_init_state=2,
                         generating_function=penalty_generating_function,
                         solver_name='approx_Q')

    value_generating_function = MulticlassQuadraticPenaltyFunction(env)
    po_lower_bound, value_coefficients, info = agent.po_policy_train(
        value_generating_function=value_generating_function,
        parallel=False,
        regularization=1e-6,
        verbose=False,
    )
    print(f"PO lower bound: {po_lower_bound:.4f}")
    print(f"regression MSE: {info['regression_mse']:.6e} over {info['num_regression_samples']} samples")

    # Dataset checks (re-extract; reads directly from the trained subproblems).
    agent.generating_function = penalty_generating_function
    X, Y = agent.extract_po_regression_targets()
    num_groups = agent.sample_path_number // agent.paths_per_init_state
    assert len(X) == len(Y) == num_groups, (len(X), num_groups)
    assert info['num_regression_samples'] == num_groups
    for (u, v, w), y in zip(X, Y):
        assert u.shape == (env.planning_horizon,)
        assert v.shape == (env.planning_horizon,)
        assert w.shape == (env.num_types,)
        assert np.isfinite(y)
    # Scenarios within a group must share the recorded initial state, X must
    # be exactly those group states, and mean(Y) must equal the PO lower bound
    # (each Y is the group-average information-relaxation cost).
    for group_index, state in enumerate(X):
        group_anchor = group_index * agent.paths_per_init_state
        for omega in range(group_anchor, group_anchor + agent.paths_per_init_state):
            recorded = agent.train_init_states[omega]
            assert all(np.allclose(a, b) for a, b in zip(state, recorded))
    # mean(Y) equals the Benders cost-to-go estimate (the subproblem-side
    # bound); it matches the returned bound exactly only at full convergence,
    # so allow the residual Benders gap.
    assert np.isclose(np.mean(Y), po_lower_bound, rtol=1e-3), (np.mean(Y), po_lower_bound)
    print(f"extracted {len(X)} regression samples "
          f"({agent.sample_path_number} paths, {agent.paths_per_init_state} per state); "
          f"mean(Y) = {np.mean(Y):.4f}")
    agent.generating_function = value_generating_function

    # The agent must now hold the quadratic generating function with fitted
    # coefficients and be able to act greedily.
    assert agent.generating_function is value_generating_function
    assert agent.is_trained
    assert len(value_coefficients) == value_generating_function.number_of_coefficients
    state, _ = env.reset(**config.reset_params)
    cost, action, _ = agent.approx_Q_solve(state, t=0)
    print(f"greedy PO policy objective: {cost:.4f}")
    print("All po_policy_train tests passed.")


if __name__ == "__main__":
    main()
