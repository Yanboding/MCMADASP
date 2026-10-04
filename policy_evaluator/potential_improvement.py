import numpy as np


def _alp_rollout(env, alp_agent, init_state, arrivals):
    state, _ = env.reset(init_state=init_state, t=1, new_arrivals=np.asarray(arrivals))
    actions, costs = [], []
    for k in range(len(arrivals) + 1):
        _, action, _ = alp_agent.solve(state, k + 1)
        state, cost, _, _ = env.step(action)
        actions.append(action)
        costs.append(float(cost))
    return actions, np.asarray(costs)


def arrival_informed_schedule_cost(env, alp_agent, relaxation_agent, init_state, arrival_stream,
                                   prefix_periods, continuation_length, continuation_period_weights,
                                   action_periods=0):
    gamma = env.discount_factor
    T = int(prefix_periods)
    cost_periods = T + continuation_length + 1
    replay_periods = max(cost_periods, int(action_periods))
    stream = np.asarray(arrival_stream)
    if len(stream) < replay_periods - 1:
        raise ValueError(f"arrival stream has {len(stream)} periods; need {replay_periods - 1}")
    continuation_period_weights = np.asarray(continuation_period_weights, dtype=float)
    if continuation_period_weights.shape != (continuation_length + 1,):
        raise ValueError(f"continuation weights need {continuation_length + 1} entries; "
                         f"got {continuation_period_weights.shape}")
    weights = np.concatenate((gamma ** np.arange(T), gamma ** T * continuation_period_weights))
    result = {'prefix_periods': T}
    prefix_actions = []
    if T > 0:
        alp_actions, alp_costs = _alp_rollout(env, alp_agent, init_state, stream[:T - 1])
        result['alp_prefix_cost'] = float(weights[:T] @ alp_costs)
        _, prefix_actions, info = relaxation_agent.information_relaxation_schedule(
            init_state, stream[:T - 1], weights[:T], 'absorbed', start_actions=alp_actions)
        result.update({'mip_objective': info['objective'], 'mip_bound': info['bound'],
                       'mip_gap': info['gap'], 'mip_runtime': info['runtime']})
    state, _ = env.reset(init_state=init_state, t=1, new_arrivals=stream[:replay_periods - 1])
    actions, costs = [], []
    for k in range(replay_periods):
        action = prefix_actions[k] if k < T else alp_agent.solve(state, k + 1)[1]
        state, cost, _, _ = env.step(action)
        actions.append(action)
        costs.append(float(cost))
    result['actions'] = actions
    costs = np.asarray(costs[:cost_periods])
    result['prefix_cost'] = float(weights[:T] @ costs[:T])
    result['continuation_cost'] = float(weights[T:] @ costs[T:])
    result['schedule_cost'] = result['prefix_cost'] + result['continuation_cost']
    return result
