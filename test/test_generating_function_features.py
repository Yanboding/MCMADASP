"""Feature interface of the generating functions (toy env).

Run from the repo root:  python -m test.test_generating_function_features
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from decision_maker import MyopicAgent
from experiments import get_config_by_type
from generating_function import AbsorptionLinearPenaltyFunction, Features, LinearPenaltyFunction
from utils import acquire_grb_env


def setup(seed=0):
    config = get_config_by_type('toy')
    env = config.env
    env.reset_random_seeds()
    grb_env = acquire_grb_env({'Threads': 1}, verbose=False)
    myopic = MyopicAgent(env, discount_factor=env.discount_factor, grb_env=grb_env)
    rng = np.random.default_rng(seed)
    n = LinearPenaltyFunction(env).number_of_coefficients
    theta = rng.normal(size=n) * 3.0
    samples = []
    for _ in range(4):
        state = tuple(np.array(c, dtype=float) for c in env.generate_initial_state())
        _, action, _ = myopic.solve(state, t=1)
        action = tuple(np.array(c, dtype=float) for c in action)
        samples.append((state, action, env.arrival_generator.rvs(size=1)[0]))
    return env, grb_env, theta, samples


def legacy_penalty(env, theta, state, action, arrival):
    """The pre-refactor formula ``(sum mean - sum delta) * theta . phi_0``."""
    post = env.post_action_state(state, action, is_var=False)
    x, y = action
    phi_0 = np.concatenate([post[0], post[1], post[2], x.reshape(-1), y])
    return float(np.sum(env.arrival_generator.mean_by_type - arrival)) * float(theta @ phi_0)


def test_features_block_algebra():
    f = Features(4, [(0, 2, np.array([1.0, 2.0])), (2, 4, np.array([3.0, 4.0]))])
    assert np.allclose(f.dense(), [1, 2, 3, 4]) and f.dot([1, 1, 1, 1]) == 10.0
    assert f.scaled(0.0).blocks == [] and np.allclose(f.scaled(2.0).dense(), [2, 4, 6, 8])
    assert np.allclose((f - f.scaled(0.5)).dense(), [0.5, 1, 1.5, 2])


def test_numeric_identities_match_legacy_formula():
    env, _, theta, samples = setup()
    gf = LinearPenaltyFunction(env, coefficients=theta.tolist())
    for state, action, arrival in samples:
        expected = gf.expected_value(theta, state, action)
        realized = gf.value(theta, state, action, arrival)
        assert np.isclose(expected - realized, legacy_penalty(env, theta, state, action, arrival), rtol=1e-12)
        gradient = gf.penalty_features(state, action, arrival, 1.0, 1.0).dense()
        assert np.isclose(theta @ gradient, expected - realized, rtol=1e-12)
        for c_e, c_r in ((1.0, 1.0), (0.7, 0.0), (0.0, 0.3), (0.99, 1.1)):
            combined = gf.penalty_features(state, action, arrival, c_e, c_r).dense()
            direct = c_e * gf.expected_features(state, action).dense() - c_r * gf.features(state, action, arrival).dense()
            assert np.allclose(combined, direct, rtol=1e-12)
        assert gf.penalty_features(state, action, None, 0.5, 0.0).dot(theta) == 0.5 * expected
    # An arrival equal to the mean total gives an empty (all-zero) feature.
    state, action, _ = samples[0]
    mean_total = float(np.sum(env.arrival_generator.mean_by_type))
    if float(mean_total).is_integer():
        arrival = np.array([mean_total] + [0.0] * (env.num_types - 1))
        assert gf.penalty_features(state, action, arrival, 1.0, 1.0).blocks == []


def test_expected_features_equal_monte_carlo_mean():
    env, _, theta, samples = setup(1)
    gf = LinearPenaltyFunction(env)
    state, action, _ = samples[0]
    arrivals = env.arrival_generator.rvs(size=20000)
    mean = np.mean([gf.features(state, action, delta).dense() for delta in arrivals], axis=0)
    expected = gf.expected_features(state, action).dense()
    scale = max(1.0, np.abs(expected).max())
    assert np.allclose(mean, expected, atol=0.02 * scale), np.abs(mean - expected).max() / scale


def test_absorption_class_continuation_is_expected_value():
    env, _, theta, samples = setup(2)
    linear = LinearPenaltyFunction(env, coefficients=theta.tolist())
    absorption = AbsorptionLinearPenaltyFunction(env, coefficients=theta.tolist())
    for state, action, arrival in samples:
        post = env.post_action_state(state, action, is_var=False)
        x, y = action
        phi_0 = np.concatenate([post[0], post[1], post[2], x.reshape(-1), y])
        mean_total = float(np.sum(env.arrival_generator.mean_by_type))
        assert np.isclose(absorption.expected_value(theta, state, action), mean_total * float(theta @ phi_0), rtol=1e-12)
        assert np.isclose(absorption.calculate_expected_continuation_value(state, action),
                          absorption.expected_value(theta, state, action), rtol=1e-12)
        workload = float((post[2] + env.arrival_generator.mean_by_type).sum())
        assert np.isclose(linear.calculate_expected_continuation_value(state, action), workload * float(theta @ phi_0), rtol=1e-12)
        # Same penalty features on both classes; different forms.
        assert np.allclose(absorption.penalty_features(state, action, arrival, 0.3, 0.8).dense(),
                           linear.penalty_features(state, action, arrival, 0.3, 0.8).dense())
    assert type(absorption.forms['training']).__name__ == 'AbsorptionForm'
    assert linear.forms['training'].mode == 'training' and linear.forms['evaluation'].mode == 'evaluation'


def test_symbolic_features_are_affine_and_agree_with_numeric():
    env, grb_env, theta, samples = setup(3)
    gf = LinearPenaltyFunction(env, coefficients=theta.tolist())
    myopic = MyopicAgent(env, discount_factor=env.discount_factor, grb_env=grb_env)
    for state, action, arrival in samples[:2]:
        model = gp.Model('features', env=grb_env)
        state_var = myopic.get_state_var(model)
        action_var = myopic.get_action_var(model, GRB.CONTINUOUS)
        for var_block, values in zip(state_var + action_var, state + action):
            var_block.lb = values
            var_block.ub = values
        objective = gf.penalty_features(state_var, action_var, arrival, 0.9, 1.0, is_var=True).dot(theta)
        model.setObjective(objective, GRB.MINIMIZE)
        model.update()
        assert model.IsQP == 0
        model.optimize()
        numeric = gf.penalty_features(state, action, arrival, 0.9, 1.0).dot(theta)
        assert np.isclose(model.ObjVal, numeric, rtol=1e-9, atol=1e-9), (model.ObjVal, numeric)
        # Symbolic theta (MVar) against numeric features: the Benders-master view.
        theta_var = gf.get_coefficient_var(model, GRB.INFINITY)
        theta_var.lb = theta
        theta_var.ub = theta
        model.setObjective(gf.penalty_features(state, action, arrival, 0.9, 1.0).dot(theta_var), GRB.MINIMIZE)
        model.optimize()
        assert np.isclose(model.ObjVal, numeric, rtol=1e-9, atol=1e-9)
        model.dispose()


if __name__ == '__main__':
    test_features_block_algebra()
    test_numeric_identities_match_legacy_formula()
    test_expected_features_equal_monte_carlo_mean()
    test_absorption_class_continuation_is_expected_value()
    test_symbolic_features_are_affine_and_agree_with_numeric()
    print('All generating-function feature tests passed.')
