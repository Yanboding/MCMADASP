"""Verify regression_train against closed-form ridge least squares.

Builds the explicit design matrix Phi for the quadratic basis (one column per
coefficient, via unit-coefficient evaluations), generates synthetic targets
Y = Phi @ kappa* + noise, and checks that the Gurobi QP in
``ApproxQAgent.regression_train`` attains the same training MSE as the exact
closed-form ridge solution. Agreement means the solver implementation is
correct and any residual error on real data is a property of the data/basis,
not the optimizer.
"""
import numpy as np

from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import MulticlassLinearPenaltyFunction, MulticlassQuadraticPenaltyFunction


def build_design_matrix(generating_function, states):
    n_coeff = generating_function.number_of_coefficients
    Phi = np.zeros((len(states), n_coeff))
    for k in range(n_coeff):
        unit = np.zeros(n_coeff)
        unit[k] = 1.0
        blocks = generating_function.get_coefficients(unit)
        for i, state in enumerate(states):
            Phi[i, k] = generating_function.calculate_state_value(state, coefficients=blocks)
    return Phi


def main():
    rng = np.random.default_rng(0)
    config = get_config_by_type('toy')
    env = config.env
    value_fn = MulticlassQuadraticPenaltyFunction(env)
    agent = ApproxQAgent(env,
                         discount_factor=env.discount_factor,
                         sample_path_number=2,
                         generating_function=value_fn,
                         solver_name='approx_Q')

    # Random states with realistic magnitudes.
    n_samples = 200
    states = [(rng.integers(0, env.regular_capacity + 1, env.planning_horizon).astype(float),
               rng.integers(0, env.overtime_capacity + 1, env.planning_horizon).astype(float),
               rng.integers(0, 15, env.num_types).astype(float))
              for _ in range(n_samples)]

    Phi = build_design_matrix(value_fn, states)
    kappa_true = rng.normal(0, 50, value_fn.number_of_coefficients)
    noise = rng.normal(0, 500, n_samples)
    Y = Phi @ kappa_true + noise

    regularization = 1e-6
    # Exact closed-form ridge: min (1/N)||Phi k - Y||^2 + reg ||k||^2
    n = len(Y)
    kappa_exact = np.linalg.solve(Phi.T @ Phi / n + regularization * np.eye(Phi.shape[1]),
                                  Phi.T @ Y / n)
    mse_exact = float(np.mean((Phi @ kappa_exact - Y) ** 2))

    coefficients, mse_gurobi = agent.regression_train(list(states), Y, regularization=regularization)
    kappa_gurobi = np.asarray(coefficients)
    mse_recomputed = float(np.mean((Phi @ kappa_gurobi - Y) ** 2))

    print(f"closed-form ridge MSE: {mse_exact:.6f}")
    print(f"regression_train MSE:  {mse_gurobi:.6f} (recomputed {mse_recomputed:.6f})")
    print(f"max |kappa_gurobi - kappa_exact|: {np.max(np.abs(kappa_gurobi - kappa_exact)):.6e}")
    assert np.isclose(mse_gurobi, mse_exact, rtol=1e-4), (mse_gurobi, mse_exact)
    assert np.allclose(kappa_gurobi, kappa_exact, atol=1e-3 * max(1.0, np.max(np.abs(kappa_exact))))
    print("regression_train matches closed-form ridge least squares.")


if __name__ == "__main__":
    main()
