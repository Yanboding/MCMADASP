"""Test least-squares training of linear value-function coefficients.

Generates synthetic (state, value) data from a ground-truth coefficient
vector theta*, fits a fresh ``LinearPenaltyFunction`` with
``ApproxQAgent.regression_train`` (a Gurobi QP), and checks that the fitted
V_theta(s) reproduces the targets on training and held-out states. Finally
verifies that the trained agent can solve the decision model.
"""
import numpy as np

from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import LinearPenaltyFunction


def sample_states(env, size, rng):
    states = []
    for _ in range(size):
        u = rng.integers(0, env.regular_capacity + 1, size=env.planning_horizon).astype(float)
        v = rng.integers(0, env.overtime_capacity + 1, size=env.planning_horizon).astype(float)
        w = rng.integers(0, 20, size=env.num_types).astype(float)
        states.append((u, v, w))
    return states


def main():
    config = get_config_by_type('toy')
    env = config.env
    rng = np.random.default_rng(42)

    T, K, W = env.planning_horizon, env.num_types, env.booking_window_size
    sizes = [K * T, K * T, K * K, K * W * K, K * T]
    # Ground truth: state blocks (theta_u, theta_v, theta_w) are random; the
    # action blocks (theta_x, theta_y) are zero because phi(s) is state-only.
    true_theta = np.concatenate([
        rng.normal(0.0, 1.0, sizes[0]),
        rng.normal(0.0, 1.0, sizes[1]),
        rng.normal(0.0, 1.0, sizes[2]),
        np.zeros(sizes[3]),
        np.zeros(sizes[4]),
    ])
    truth = LinearPenaltyFunction(env, coefficients=true_theta)

    X_train = sample_states(env, 300, rng)
    Y_train = np.array([truth.calculate_state_value(s) for s in X_train])
    X_test = sample_states(env, 50, rng)
    Y_test = np.array([truth.calculate_state_value(s) for s in X_test])

    agent = ApproxQAgent(env,
                         discount_factor=env.discount_factor,
                         sample_path_number=2,
                         generating_function=LinearPenaltyFunction(env))

    # The agent must refuse to act before training.
    try:
        agent.approx_Q_solve(config.init_state, t=0)
        raise AssertionError("approx_Q_solve should fail before training")
    except RuntimeError:
        print("untrained approx_Q_solve correctly raised RuntimeError")

    coefficients, train_mse = agent.regression_train(X_train, Y_train, regularization=1e-8)
    predictions = np.array([agent.generating_function.calculate_state_value(s) for s in X_test])
    test_mse = float(np.mean((predictions - Y_test) ** 2))
    scale = max(1.0, float(np.mean(Y_test ** 2)))
    print(f"train MSE: {train_mse:.6e}")
    print(f"test  MSE: {test_mse:.6e} (target scale {scale:.3e})")
    assert train_mse <= 1e-6 * scale, "training error too large"
    assert test_mse <= 1e-6 * scale, "generalization error too large"

    state_block_size = sum(sizes[:3])
    coefficient_error = float(np.max(np.abs(np.asarray(coefficients)[:state_block_size] - true_theta[:state_block_size])))
    print(f"max |theta_hat - theta*| on state blocks: {coefficient_error:.3e}")

    state, info = env.reset(**config.reset_params)
    cost, action, _ = agent.approx_Q_solve(state, t=0)
    print(f"approx_Q_solve objective with trained coefficients: {cost:.4f}")
    print("All value-function regression tests passed.")


if __name__ == "__main__":
    main()
