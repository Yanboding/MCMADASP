import numpy as np
import gurobipy as gp
from gurobipy import GRB

from decision_maker import ALPRowGenerationAgent, MyopicAgent
from experiments import get_config_by_type
from generating_function import AbsorptionALPPenaltyFunction, AbsorptionForm
from importance_sampling import SamplePath, Terminal
from utils import acquire_grb_env


def setup(seed=0, samples=4):
    env = get_config_by_type('toy').env
    env.reset_random_seeds()
    grb_env = acquire_grb_env({'Threads': 1}, verbose=False)
    myopic = MyopicAgent(env, discount_factor=env.discount_factor, grb_env=grb_env)
    rng = np.random.default_rng(seed)
    n = 1 + 2 * env.planning_horizon + env.num_types
    theta = rng.normal(size=n) * 3.0
    drawn = []
    for _ in range(samples):
        state = tuple(np.array(c, dtype=float) for c in env.generate_initial_state())
        _, action, _ = myopic.solve(state, t=1)
        action = tuple(np.array(c, dtype=float) for c in action)
        drawn.append((state, action, env.arrival_generator.rvs(size=1)[0].astype(float)))
    return env, grb_env, myopic, theta, drawn


def alp_value(env, grb_env, theta, state):
    agent = ALPRowGenerationAgent(env, discount_factor=env.discount_factor,
                                  coefficients=theta.tolist(), grb_env=grb_env)
    return float(agent.get_approx_value_fn(state, agent.W_0, agent.U, agent.V, agent.W))


def test_value_is_alp_value_of_the_next_state():
    env, grb_env, _, theta, drawn = setup()
    gf = AbsorptionALPPenaltyFunction(env, coefficients=theta.tolist())
    assert gf.number_of_coefficients == len(theta) == 1 + 2 * env.planning_horizon + env.num_types
    assert np.allclose(gf.coefficient_vector(), theta)
    mean = np.asarray(env.arrival_generator.mean_by_type, dtype=float)
    for state, action, arrival in drawn:
        realized = alp_value(env, grb_env, theta, env.get_next_state(state, action, arrival))
        expected = alp_value(env, grb_env, theta, env.get_next_state(state, action, mean))
        assert np.isclose(gf.value(theta, state, action, arrival), realized, rtol=1e-12)
        assert np.isclose(gf.expected_value(theta, state, action), expected, rtol=1e-12)
        assert np.isclose(gf.calculate_expected_continuation_value(state, action), expected, rtol=1e-12)
        W_0, U, V, W = gf.get_coefficients(theta)
        assert W_0 == theta[0] and np.allclose(U, theta[1:1 + env.planning_horizon])
        assert np.allclose(W, theta[-env.num_types:])


def test_expected_features_equal_monte_carlo_mean():
    env, _, _, theta, drawn = setup(1, samples=1)
    gf = AbsorptionALPPenaltyFunction(env)
    state, action, _ = drawn[0]
    arrivals = env.arrival_generator.rvs(size=20000)
    mean = np.mean([gf.features(state, action, delta).dense() for delta in arrivals], axis=0)
    expected = gf.expected_features(state, action).dense()
    scale = max(1.0, np.abs(expected).max())
    assert np.allclose(mean, expected, atol=0.02 * scale), np.abs(mean - expected).max() / scale


def test_penalty_features_identities():
    env, _, _, theta, drawn = setup(2)
    gf = AbsorptionALPPenaltyFunction(env, coefficients=theta.tolist())
    mean = np.asarray(env.arrival_generator.mean_by_type, dtype=float)
    K = env.num_types
    for state, action, arrival in drawn:
        expected = gf.expected_features(state, action).dense()
        realized = gf.features(state, action, arrival).dense()
        for c_e, c_r in ((1.0, 1.0), (0.7, 0.0), (0.0, 0.3), (0.99, 1.1)):
            combined = gf.penalty_features(state, action, arrival, c_e, c_r).dense()
            assert np.allclose(combined, c_e * expected - c_r * realized, rtol=1e-12, atol=1e-12)
        unit = gf.penalty_features(state, action, arrival, 1.0, 1.0).dense()
        assert np.allclose(unit[:-K], 0.0) and np.allclose(unit[-K:], mean - arrival)
        assert np.allclose(gf.penalty_features(state, action, None, 0.5, 0.0).dense(), 0.5 * expected)


def test_symbolic_features_agree_with_numeric():
    env, grb_env, myopic, theta, drawn = setup(3, samples=2)
    gf = AbsorptionALPPenaltyFunction(env, coefficients=theta.tolist())
    for state, action, arrival in drawn:
        model = gp.Model('alp_features', env=grb_env)
        state_var = myopic.get_state_var(model)
        action_var = myopic.get_action_var(model, GRB.CONTINUOUS)
        for var_block, values in zip(state_var + action_var, state + action):
            var_block.lb = values
            var_block.ub = values
        numeric = gf.penalty_features(state, action, arrival, 0.9, 1.0).dot(theta)
        model.setObjective(gf.penalty_features(state_var, action_var, arrival, 0.9, 1.0, is_var=True).dot(theta), GRB.MINIMIZE)
        model.update()
        assert model.IsQP == 0
        model.optimize()
        assert np.isclose(model.ObjVal, numeric, rtol=1e-9, atol=1e-9), (model.ObjVal, numeric)
        model.setObjective(gf.calculate_expected_continuation_value(state_var, action_var, is_var=True), GRB.MINIMIZE)
        model.optimize()
        assert np.isclose(model.ObjVal, gf.expected_value(theta, state, action), rtol=1e-9, atol=1e-9)
        theta_var = gf.get_coefficient_var(model, GRB.INFINITY)
        theta_var.lb = theta
        theta_var.ub = theta
        model.setObjective(gf.penalty_features(state, action, arrival, 0.9, 1.0).dot(theta_var), GRB.MINIMIZE)
        model.optimize()
        assert np.isclose(model.ObjVal, numeric, rtol=1e-9, atol=1e-9)
        model.dispose()


def test_wrong_coefficient_length_is_refused():
    env = get_config_by_type('toy').env
    n = 1 + 2 * env.planning_horizon + env.num_types
    for bad in ([0.0] * (n - 1), [0.0] * (n + 1)):
        try:
            AbsorptionALPPenaltyFunction(env, coefficients=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'length {len(bad)} accepted for {n} coefficients')


def test_path_penalty_is_the_two_case_martingale_difference():
    env, grb_env, myopic, theta, _ = setup(4, samples=0)
    gf = AbsorptionALPPenaltyFunction(env, coefficients=theta.tolist())
    gamma = env.discount_factor
    mean = np.asarray(env.arrival_generator.mean_by_type, dtype=float)
    arrivals = env.arrival_generator.rvs(size=6).astype(float)
    weights = np.array([1.0, 0.9, 0.8, 0.85, 0.7, 0.6, 0.5])
    state = tuple(np.array(c, dtype=float) for c in env.generate_initial_state())
    expected_terms, realized_terms, manual = [], [], 0.0
    for t in range(len(arrivals) + 1):
        _, action, _ = myopic.solve(state, t=t + 1)
        action = tuple(np.array(c, dtype=float) for c in action)
        expected_terms.append(gf.expected_value(theta, state, action))
        continuation = alp_value(env, grb_env, theta, env.get_next_state(state, action, mean))
        manual += gamma * weights[t] * continuation
        if t < len(arrivals):
            realized_terms.append(gf.value(theta, state, action, arrivals[t]))
            state = env.get_next_state(state, action, arrivals[t])
            manual -= weights[t + 1] * alp_value(env, grb_env, theta, state)
    absorbed = SamplePath(arrivals, Terminal.ABSORBED, weights)
    truncated = SamplePath(arrivals, Terminal.TRUNCATED, weights)
    form = gf.form('evaluation')
    assert isinstance(form, AbsorptionForm)
    assert np.isclose(form.combine(absorbed, gamma, expected_terms, realized_terms), manual, rtol=1e-12)
    last = gamma * weights[-1] * expected_terms[-1]
    assert np.isclose(form.combine(truncated, gamma, expected_terms, realized_terms), manual - last, rtol=1e-12)


def test_registered_with_the_builders():
    from param_generation.generating_functions import GENERATING_FUNCTION_CLASSES, build_generating_function
    from param_generation.training import zero_penalty_coefficients
    import run
    env = get_config_by_type('toy').env
    n = 1 + 2 * env.planning_horizon + env.num_types
    assert GENERATING_FUNCTION_CLASSES['absorption_alp_penalty'] is AbsorptionALPPenaltyFunction
    assert run._GENERATING_FUNCTION_CLASSES['absorption_alp_penalty'] is AbsorptionALPPenaltyFunction
    gf = build_generating_function(env, {'name': 'absorption_alp_penalty'})
    assert np.allclose(gf.coefficient_vector(), np.zeros(n))
    assert len(zero_penalty_coefficients(env, {'name': 'absorption_alp_penalty'})) == n
    assert len(zero_penalty_coefficients(env)) == 2 * env.planning_horizon + env.num_types + env.booking_window_size * env.num_types + env.planning_horizon
    assert all(isinstance(gf.form(consumer), AbsorptionForm) for consumer in ('training', 'hindsight', 'evaluation'))


if __name__ == '__main__':
    test_value_is_alp_value_of_the_next_state()
    test_expected_features_equal_monte_carlo_mean()
    test_penalty_features_identities()
    test_symbolic_features_agree_with_numeric()
    test_wrong_coefficient_length_is_refused()
    test_path_penalty_is_the_two_case_martingale_difference()
    test_registered_with_the_builders()
    print('All ALP penalty function tests passed.')
