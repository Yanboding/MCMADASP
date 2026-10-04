import numpy as np

from decision_maker import ALPRowGenerationAgent
from policy_evaluator.potential_improvement import arrival_informed_schedule_cost
from test.test_information_relaxation_schedule import toy_relaxation_agent
from test.toy_potential_improvement import TOY_ALP_COEFFICIENTS, toy_env

INIT_STATE = (np.array([5, 5, 5, 5, 5, 5, 0]), np.zeros(7, dtype=int), np.array([1, 2]))


def _agents(env):
    alp = ALPRowGenerationAgent(env, discount_factor=env.discount_factor, coefficients=TOY_ALP_COEFFICIENTS)
    return alp, toy_relaxation_agent(env)


def _alp_costs(env, alp, state, arrivals):
    state, _ = env.reset(init_state=state, t=1, new_arrivals=np.asarray(arrivals))
    costs = []
    for k in range(len(arrivals) + 1):
        _, action, _ = alp.solve(state, k + 1)
        state, cost, _, _ = env.step(action)
        costs.append(float(cost))
    return np.asarray(costs)


def test_zero_prefix_reduces_to_alp_rollout():
    env = toy_env()
    alp, schedule_agent = _agents(env)
    stream = env.arrival_generator.rvs(size=6)
    weights = np.concatenate(([1.0], 0.99 * np.ones(6)))
    result = arrival_informed_schedule_cost(env, alp, schedule_agent, INIT_STATE, stream, 0, 6, weights)
    assert result['prefix_cost'] == 0.0
    np.testing.assert_allclose(result['continuation_cost'], weights @ _alp_costs(env, alp, INIT_STATE, stream),
                               rtol=1e-12)
    assert result['schedule_cost'] == result['continuation_cost']
    assert 'mip_objective' not in result


def test_prefix_cost_equals_mip_objective_and_continuation_starts_at_gamma_T():
    env = toy_env()
    alp, schedule_agent = _agents(env)
    T, L = 4, 5
    stream = env.arrival_generator.rvs(size=T + L + 2)
    weights = np.concatenate(([1.0], 0.99 * np.ones(L)))
    result = arrival_informed_schedule_cost(env, alp, schedule_agent, INIT_STATE, stream, T, L, weights)
    np.testing.assert_allclose(result['prefix_cost'], result['mip_objective'], rtol=1e-9, atol=1e-6)
    assert result['mip_objective'] <= result['alp_prefix_cost'] + 1e-6
    np.testing.assert_allclose(result['schedule_cost'], result['prefix_cost'] + result['continuation_cost'])
    _, actions, _ = schedule_agent.information_relaxation_schedule(
        INIT_STATE, stream[:T - 1], 0.99 ** np.arange(T), 'absorbed')
    state, _ = env.reset(init_state=INIT_STATE, t=1, new_arrivals=np.asarray(stream[:T + L]))
    for action in actions:
        state, _, _, _ = env.step(action)
    tail_costs = _alp_costs(env, alp, state, stream[T:T + L])
    np.testing.assert_allclose(result['continuation_cost'], 0.99 ** T * weights @ tail_costs, rtol=1e-9)
