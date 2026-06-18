"""Tests for the PO-derived quadratic generating function.

Checks, on the toy configuration:
1. coefficient flat-vector <-> block round trip and shapes;
2. closed-form E[V_kappa(f(s,a,delta))] equals exact enumeration over the
   truncated-Poisson/multinomial arrival distribution;
3. the reduced penalty equals the full penalty
       z_full = E[V(f(s,a,delta'))] - V(f(s,a,delta))
   minus its decision-independent terms, and has zero mean over arrivals;
4. ``ApproxQAgent.regression_train`` recovers a ground-truth kappa* from
   state-value data (the basis is linear in kappa);
5. the trained agent solves the (nonconvex quadratic) decision model.
"""
from itertools import product

import numpy as np
from scipy.stats import multinomial

from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import MulticlassQuadraticPenaltyFunction


def sample_states(env, size, rng):
    states = []
    for _ in range(size):
        u = rng.integers(0, env.regular_capacity + 1, size=env.planning_horizon).astype(float)
        v = rng.integers(0, env.overtime_capacity + 1, size=env.planning_horizon).astype(float)
        w = rng.integers(0, 20, size=env.num_types).astype(float)
        states.append((u, v, w))
    return states


def enumerate_arrivals(arrival_generator):
    """All (delta, probability) pairs of the truncated multinomial-Poisson."""
    pmf = arrival_generator.truncate_poisson_pmf
    probs = arrival_generator.type_probs
    num_types = len(probs)
    maximum = arrival_generator.maximum_arrival
    pairs = []
    for delta in product(range(maximum + 1), repeat=num_types):
        total = sum(delta)
        if total > maximum:
            continue
        probability = pmf[total] * multinomial(total, probs).pmf(np.array(delta))
        pairs.append((np.array(delta, dtype=float), float(probability)))
    assert abs(sum(p for _, p in pairs) - 1.0) < 1e-9
    return pairs


def main():
    config = get_config_by_type('toy')
    env = config.env
    rng = np.random.default_rng(7)

    generating_function = MulticlassQuadraticPenaltyFunction(env)
    n = generating_function.number_of_coefficients
    T, I = env.planning_horizon, env.num_types
    assert n == 1 + 2 * T + 2 * I + 2 * I * T

    # --- 1. round trip ------------------------------------------------------
    kappa_star = rng.normal(0.0, 1.0, n)
    generating_function.set_coefficients(kappa_star)
    blocks = (generating_function.kappa_0, generating_function.kappa_u,
              generating_function.kappa_v, generating_function.kappa_w,
              generating_function.kappa_uw, generating_function.kappa_vw,
              generating_function.kappa_ww)
    shapes = [(1,), (T,), (T,), (I,), (I, T), (I, T), (I,)]
    assert [b.shape for b in blocks] == shapes
    assert np.allclose(np.concatenate([b.reshape(-1) for b in blocks]), kappa_star)
    print("coefficient round-trip OK")

    # --- 2. closed-form expected continuation value -------------------------
    state, info = env.reset(**config.reset_params)
    action = list(env.valid_actions(state))[-1]
    arrivals = enumerate_arrivals(env.arrival_generator)

    mean_check = sum(p * d for d, p in arrivals)
    m2_check = sum(p * d ** 2 for d, p in arrivals)
    assert np.allclose(mean_check, generating_function.arrival_mean)
    assert np.allclose(m2_check, generating_function.arrival_second_moment)
    print("arrival moments OK")

    closed_form = generating_function.calculate_expected_continuation_value(state, action)
    enumerated = sum(
        p * generating_function.calculate_state_value(env.get_next_state(state, action, d, is_var=False))
        for d, p in arrivals)
    assert abs(closed_form - enumerated) < 1e-8 * max(1.0, abs(enumerated)), (closed_form, enumerated)
    print(f"closed-form expected continuation value OK ({closed_form:.6f})")

    # --- 3. penalty identity and zero mean -----------------------------------
    mean = generating_function.arrival_mean
    second_moment = generating_function.arrival_second_moment
    kappa_w = generating_function.kappa_w
    kappa_ww = generating_function.kappa_ww
    for delta, _ in arrivals:
        next_state = env.get_next_state(state, action, delta, is_var=False)
        full_penalty = closed_form - generating_function.calculate_state_value(next_state)
        omitted = (mean - delta) @ kappa_w + kappa_ww @ (second_moment - delta ** 2)
        reduced = generating_function.calculate_penalty(state, action, delta, is_var=False)
        assert abs(reduced - (full_penalty - omitted)) < 1e-8, (reduced, full_penalty - omitted)
    expected_penalty = sum(
        p * generating_function.calculate_penalty(state, action, d, is_var=False) for d, p in arrivals)
    assert abs(expected_penalty) < 1e-8, expected_penalty
    print("penalty identity and zero-mean OK")

    # --- 4. regression recovery ----------------------------------------------
    truth = MulticlassQuadraticPenaltyFunction(env, coefficients=kappa_star)
    X_train = sample_states(env, max(400, 4 * n), rng)
    Y_train = np.array([truth.calculate_state_value(s) for s in X_train])
    X_test = sample_states(env, 50, rng)
    Y_test = np.array([truth.calculate_state_value(s) for s in X_test])

    agent = ApproxQAgent(env,
                         discount_factor=env.discount_factor,
                         sample_path_number=2,
                         generating_function=MulticlassQuadraticPenaltyFunction(env))
    coefficients, train_mse = agent.regression_train(X_train, Y_train, regularization=1e-8)
    predictions = np.array([agent.generating_function.calculate_state_value(s) for s in X_test])
    test_mse = float(np.mean((predictions - Y_test) ** 2))
    scale = max(1.0, float(np.mean(Y_test ** 2)))
    print(f"train MSE: {train_mse:.6e}")
    print(f"test  MSE: {test_mse:.6e} (target scale {scale:.3e})")
    assert train_mse <= 1e-6 * scale, "training error too large"
    assert test_mse <= 1e-6 * scale, "generalization error too large"
    coefficient_error = float(np.max(np.abs(np.asarray(coefficients) - kappa_star)))
    print(f"max |kappa_hat - kappa*|: {coefficient_error:.3e}")

    # --- 5. greedy policy from the fitted value function ----------------------
    state, info = env.reset(**config.reset_params)
    cost, greedy_action, _ = agent.approx_Q_solve(state, t=0)
    print(f"approx_Q_solve objective with fitted kappa: {cost:.4f}")
    print("All PO quadratic generating-function tests passed.")


if __name__ == "__main__":
    main()
