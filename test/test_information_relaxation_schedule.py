import numpy as np

from decision_maker import ALPRowGenerationAgent, ApproxQAgent
from generating_function import AbsorptionALPPenaltyFunction
from test.toy_potential_improvement import TOY_ALP_COEFFICIENTS, toy_env

INIT_STATE = (np.array([5, 5, 5, 5, 5, 5, 0]), np.zeros(7, dtype=int), np.array([1, 2]))


def toy_relaxation_agent(env):
    return ApproxQAgent(env, discount_factor=env.discount_factor, current_decision_var_type='integer',
                        future_decision_var_type='integer', penalty_ratio=0.0,
                        generating_function=AbsorptionALPPenaltyFunction(env, TOY_ALP_COEFFICIENTS))


def _replay(env, actions, arrivals):
    env.reset(init_state=INIT_STATE, t=1, new_arrivals=np.asarray(arrivals))
    costs = []
    for action in actions:
        _, cost, _, _ = env.step(action)
        costs.append(float(cost))
    return np.asarray(costs)


def _alp_rollout(env, arrivals):
    agent = ALPRowGenerationAgent(env, discount_factor=env.discount_factor, coefficients=TOY_ALP_COEFFICIENTS)
    state, _ = env.reset(init_state=INIT_STATE, t=1, new_arrivals=np.asarray(arrivals))
    actions, costs = [], []
    for k in range(len(arrivals) + 1):
        _, action, _ = agent.solve(state, k + 1)
        state, cost, _, _ = env.step(action)
        actions.append(action)
        costs.append(float(cost))
    return actions, np.asarray(costs)


def test_schedule_objective_equals_replayed_discounted_cost():
    env = toy_env()
    arrivals = env.arrival_generator.rvs(size=7)
    weights = env.discount_factor ** np.arange(8)
    objective, actions, info = toy_relaxation_agent(env).information_relaxation_schedule(
        INIT_STATE, arrivals, weights, 'absorbed')
    assert len(actions) == 8
    np.testing.assert_allclose(weights @ _replay(env, actions, arrivals), objective, rtol=1e-9, atol=1e-6)
    assert info['gap'] <= 1e-9 or abs(info['objective'] - info['bound']) <= 1e-6


def test_schedule_is_no_worse_than_alp_and_accepts_alp_start():
    env = toy_env()
    arrivals = env.arrival_generator.rvs(size=9)
    alp_actions, alp_costs = _alp_rollout(env, arrivals)
    weights = env.discount_factor ** np.arange(10)
    objective, _, _ = toy_relaxation_agent(env).information_relaxation_schedule(
        INIT_STATE, arrivals, weights, 'absorbed', start_actions=alp_actions)
    assert objective <= weights @ alp_costs + 1e-6
