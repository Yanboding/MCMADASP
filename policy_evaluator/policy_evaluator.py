import time
from collections import defaultdict
from utils import iter_to_tuple, RunningStats


class PolicyEvaluator:

    def __init__(self, env, agent, discount_factor, V=None, is_inf=True):
        self.env = env
        self.agent = agent
        self.discount_factor = discount_factor
        self.is_inf = is_inf
        if V is None:
            self.V = {}

    def evaluate(self, state, t):
        state_tuple = iter_to_tuple(state)
        action = self.agent.policy(state, t)
        q = 0
        for prob, next_state, cost, done in self.env.transition_dynamic(state, action, t):
            next_state_tuple = iter_to_tuple(next_state)
            if (next_state_tuple, t + 1) in self.V:
                next_state_val = self.V[(next_state_tuple, t + 1)]
            elif done:
                next_state_val = 0
            else:
                next_state_val = self.evaluate(next_state, t + 1)
            q += prob * (cost + self.discount_factor * next_state_val)
        self.V[(state_tuple, t)] = q
        return self.V[(state_tuple, t)]

    def sample_path_evaluate(self, state, t, sample_path, action=None):
        states = []
        rewards = []
        s, info = self.env.reset(state, t, sample_path)
        for tau in range(len(sample_path)):
            states.append(s)
            if tau == 0 and action:
                a = action
            else:
                a = self.agent.policy(s, t + tau)
            next_state, reward, done, info = self.env.step(a)
            rewards.append(reward)
            s = next_state
            if done:
                break
        return states, rewards

    def simulation_evaluate_helper(self, state, t, sample_paths, action=None):
        if self.is_inf:
            self.discount_factor = 1
        sample_average_V = defaultdict(lambda: RunningStats())
        for sample_path in sample_paths:
            states, rewards = self.sample_path_evaluate(state,t,sample_path, action=action)
            G = 0.0
            for tau in reversed(range(len(states))):
                G = self.discount_factor * G + rewards[tau]
                s = states[tau]
                sample_average_V[(iter_to_tuple(s), t + tau)] += G
        return sample_average_V

    def sample_path_optimality_gap_evaluate(self, lower_bound_solver, state, t, sample_path, action=None):
        abs_gap, benchmark_value = self.sample_path_absolute_gap_evaluate(lower_bound_solver, state, t, sample_path, action=action)
        if abs_gap == 0:
            return 0
        return abs_gap / benchmark_value * 100 if benchmark_value != 0 else float('inf')
    
    def sample_path_absolute_gap_evaluate(self, benchmark_solver, state, t, sample_path, action=None):
        state_tuple = iter_to_tuple(state)
        # sample_path should start from period t+1 
        benchmark_solver.set_sample_path(sample_path)
        _, benchmark_value, info = benchmark_solver.solve(state, t, action=action)
        sample_average_V = self.simulation_evaluate_helper(state, t, [sample_path], action=action)
        upper_bound = sample_average_V[(state_tuple, t)].mean
        abs_gap = max(upper_bound - benchmark_value, 0)
        return abs_gap, benchmark_value



if __name__ == '__main__':
    from experiments import get_config_by_type
    from decision_maker import InfiniteSAAAgent, ALPEJORColumnGenerationAgent
    config = get_config_by_type('ejor_default')
    env = config.env
    t = 1
    sample_path = [[2, 5, 4, 8, 6], [4, 3, 3, 5, 8], [2, 4, 5, 8, 4], [3, 4, 3, 10, 3], [4, 4, 7, 12, 10], [2, 10, 4, 6, 5], [4, 5, 4, 8, 6], [4, 5, 4, 6, 4], [5, 3, 2, 11, 6], [2, 1, 6, 6, 4], [2, 1, 6, 4, 4], [1, 3, 5, 12, 9], [3, 4, 1, 3, 3], [5, 5, 3, 7, 16], [1, 5, 6, 4, 14], [1, 4, 7, 8, 9], [3, 3, 1, 3, 7], [2, 4, 6, 6, 3], [7, 6, 4, 6, 7], [1, 7, 5, 2, 9], [5, 5, 9, 2, 5], [2, 1, 7, 3, 6], [3, 4, 4, 5, 5], [3, 5, 5, 5, 7], [4, 5, 5, 7, 7], [1, 4, 7, 7, 6], [3, 8, 2, 4, 2], [5, 6, 5, 4, 11], [3, 1, 5, 2, 9], [1, 4, 7, 5, 7], [6, 6, 5, 4, 5], [2, 7, 4, 5, 5], [3, 2, 10, 4, 5], [4, 7, 5, 6, 9], [5, 2, 4, 7, 6], [3, 4, 4, 8, 6], [6, 5, 10, 9, 5], [3, 2, 2, 10, 5], [6, 5, 6, 8, 6], [2, 3, 4, 6, 8], [3, 0, 4, 7, 8], [6, 5, 4, 11, 5], [4, 2, 7, 3, 2], [2, 6, 5, 11, 7], [4, 2, 7, 8, 9], [5, 8, 6, 6, 6], [2, 1, 5, 4, 5], [2, 5, 3, 6, 5], [3, 3, 7, 10, 6], [1, 2, 5, 10, 6], [1, 2, 2, 3, 5], [3, 4, 3, 9, 8], [2, 5, 6, 3, 13], [3, 8, 9, 9, 6], [1, 5, 3, 4, 7], [1, 6, 8, 3, 4], [7, 2, 1, 9, 11], [4, 2, 2, 7, 3], [3, 3, 5, 6, 5], [1, 4, 6, 4, 9], [1, 1, 4, 4, 14], [3, 6, 2, 9, 5], [2, 0, 7, 9, 13], [3, 3, 7, 6, 8], [5, 5, 6, 6, 5], [2, 7, 5, 2, 10], [1, 6, 4, 1, 9], [1, 8, 3, 5, 7], [5, 1, 8, 11, 12], [1, 5, 9, 9, 8], [2, 5, 4, 8, 10], [2, 4, 6, 4, 9], [7, 6, 8, 4, 6], [4, 5, 9, 5, 4], [3, 2, 7, 8, 4], [4, 4, 1, 12, 4], [4, 4, 4, 5, 4], [5, 3, 3, 6, 9], [1, 4, 6, 5, 6], [3, 3, 11, 4, 8], [6, 3, 8, 4, 9], [5, 6, 5, 11, 11], [4, 7, 3, 7, 9], [0, 5, 1, 7, 4], [4, 7, 5, 3, 9], [4, 1, 4, 8, 12], [4, 2, 7, 7, 11], [2, 7, 5, 9, 2], [4, 3, 7, 6, 5], [2, 3, 5, 6, 9], [5, 5, 1, 4, 5], [2, 8, 7, 8, 9], [6, 8, 2, 7, 14], [1, 7, 3, 4, 7], [3, 4, 5, 9, 13], [5, 2, 2, 5, 7], [2, 4, 4, 4, 4], [2, 2, 5, 8, 11], [5, 1, 6, 9, 8], [1, 3, 6, 5, 6], [3, 9, 6, 8, 8], [2, 5, 4, 6, 7], [2, 3, 3, 5, 8], [3, 4, 3, 6, 6], [4, 3, 4, 5, 3], [4, 3, 4, 7, 11], [2, 3, 3, 9, 5], [5, 2, 5, 12, 3], [5, 2, 3, 5, 9], [0, 5, 6, 1, 7], [5, 3, 4, 7, 7], [2, 8, 5, 6, 14], [2, 2, 4, 3, 9], [4, 7, 6, 8, 9], [2, 2, 6, 5, 4], [4, 4, 5, 6, 7], [2, 4, 3, 4, 7], [2, 9, 5, 11, 12], [2, 2, 5, 8, 8], [5, 2, 6, 12, 6], [3, 4, 6, 4, 11], [6, 2, 5, 4, 7], [3, 1, 3, 8, 7], [2, 5, 7, 9, 8], [5, 6, 6, 6, 8], [4, 3, 5, 7, 10], [2, 4, 7, 11, 5], [2, 5, 3, 3, 4], [2, 2, 6, 4, 7], [2, 0, 6, 8, 10], [1, 2, 5, 5, 5], [3, 3, 4, 7, 9], [3, 4, 8, 8, 10], [1, 6, 4, 5, 7], [3, 6, 6, 4, 8], [3, 3, 6, 7, 2], [7, 5, 4, 7, 6], [6, 4, 4, 7, 10], [3, 4, 6, 6, 10], [3, 2, 6, 5, 8], [3, 2, 5, 6, 7], [4, 1, 3, 4, 6], [3, 4, 7, 3, 13], [3, 5, 9, 10, 5], [5, 6, 8, 6, 1], [4, 2, 9, 6, 8], [0, 2, 5, 7, 7], [7, 5, 7, 2, 7], [2, 5, 6, 3, 6], [4, 3, 4, 10, 7], [1, 4, 0, 6, 3], [4, 3, 4, 5, 7], [2, 4, 5, 5, 5], [1, 2, 7, 3, 4], [2, 3, 7, 4, 6], [4, 4, 5, 8, 11], [1, 5, 9, 4, 3], [2, 3, 4, 6, 4], [1, 2, 8, 7, 9], [5, 7, 3, 8, 6], [4, 4, 4, 5, 6], [4, 2, 8, 2, 16], [2, 5, 8, 8, 4], [2, 0, 9, 4, 10], [5, 3, 9, 4, 10], [6, 2, 6, 9, 6], [1, 4, 5, 10, 6], [3, 6, 7, 3, 3], [4, 4, 4, 10, 3], [2, 4, 5, 5, 10], [3, 4, 9, 8, 7], [4, 7, 7, 10, 5], [1, 2, 6, 4, 5], [0, 3, 4, 9, 9], [6, 4, 6, 7, 10], [2, 1, 11, 4, 6], [4, 5, 9, 6, 4], [3, 3, 8, 2, 5], [4, 5, 3, 11, 8], [1, 3, 4, 5, 7], [2, 5, 7, 4, 4], [2, 2, 1, 8, 8], [3, 5, 3, 4, 5], [3, 2, 8, 9, 5], [3, 6, 8, 3, 10], [3, 2, 4, 6, 5], [2, 5, 5, 1, 8], [1, 8, 3, 8, 4], [3, 3, 4, 6, 6], [3, 3, 8, 9, 4], [0, 5, 7, 4, 3], [3, 3, 4, 11, 5], [1, 5, 6, 6, 7], [2, 6, 3, 4, 4], [3, 5, 5, 6, 5], [6, 4, 6, 2, 5], [2, 3, 5, 3, 5], [2, 4, 3, 6, 8], [4, 3, 4, 9, 7], [6, 5, 7, 4, 9], [5, 5, 2, 9, 13], [5, 3, 4, 8, 8], [1, 4, 6, 4, 9], [2, 1, 4, 8, 5], [3, 6, 6, 5, 7], [4, 5, 5, 6, 6], [6, 3, 7, 8, 6], [5, 3, 5, 5, 4], [3, 3, 5, 8, 8], [2, 4, 4, 4, 5], [5, 2, 5, 4, 6], [5, 5, 2, 8, 4], [4, 4, 4, 7, 10], [3, 5, 5, 4, 6], [4, 4, 6, 5, 7], [4, 4, 8, 8, 13], [4, 5, 10, 6, 8], [1, 4, 6, 7, 7], [4, 4, 3, 4, 6], [0, 6, 2, 8, 10], [3, 6, 4, 6, 10], [6, 4, 3, 8, 5], [1, 8, 4, 9, 6], [2, 6, 2, 3, 8], [4, 3, 2, 7, 6], [4, 4, 0, 7, 5], [2, 3, 2, 4, 6], [1, 5, 4, 10, 4], [1, 6, 5, 8, 9], [1, 5, 3, 5, 9], [3, 5, 10, 4, 12], [3, 4, 3, 5, 8], [1, 7, 6, 6, 3], [4, 4, 6, 9, 7], [3, 7, 8, 1, 8], [4, 4, 2, 5, 10], [4, 3, 3, 8, 3], [3, 5, 3, 4, 3], [1, 4, 6, 3, 6], [2, 4, 6, 2, 6], [3, 2, 6, 5, 8], [3, 4, 6, 5, 5], [3, 5, 2, 6, 9], [4, 3, 5, 6, 4], [5, 3, 10, 5, 4], [2, 8, 5, 9, 9], [3, 6, 6, 7, 7], [4, 6, 3, 5, 8], [3, 2, 4, 9, 4], [4, 3, 5, 4, 13], [5, 2, 5, 7, 11], [4, 1, 2, 2, 10], [7, 4, 5, 6, 8], [2, 5, 8, 8, 9], [4, 5, 4, 4, 4], [1, 2, 4, 4, 9], [3, 5, 4, 9, 7], [4, 4, 2, 4, 11], [2, 4, 7, 3, 2], [0, 8, 9, 8, 1], [5, 3, 5, 12, 5], [5, 2, 5, 8, 8], [3, 3, 4, 2, 5], [1, 2, 3, 7, 4], [1, 7, 5, 6, 6], [2, 2, 9, 8, 6], [5, 5, 9, 5, 8], [2, 3, 5, 6, 12], [5, 7, 4, 6, 5], [7, 5, 3, 8, 8], [6, 5, 5, 7, 1], [2, 4, 5, 5, 4], [4, 1, 4, 5, 5], [1, 4, 3, 8, 7], [1, 6, 9, 9, 6], [1, 5, 4, 5, 10], [6, 2, 4, 2, 3], [5, 7, 4, 10, 6], [6, 7, 4, 9, 4], [3, 2, 4, 5, 3], [0, 5, 6, 6, 12], [2, 2, 5, 3, 8], [4, 3, 2, 4, 4], [4, 6, 6, 3, 5], [5, 7, 4, 4, 8], [1, 4, 4, 4, 9], [3, 5, 3, 8, 5], [4, 2, 7, 11, 6], [1, 5, 2, 4, 7], [5, 4, 5, 6, 7], [2, 4, 6, 3, 7], [1, 3, 4, 3, 7], [3, 4, 5, 7, 10], [4, 4, 5, 7, 7], [7, 1, 3, 7, 6], [2, 4, 4, 8, 6], [3, 2, 8, 2, 8], [5, 4, 8, 2, 8], [2, 2, 0, 4, 6], [5, 3, 0, 5, 6], [3, 5, 3, 5, 2], [1, 4, 3, 7, 4], [2, 3, 7, 9, 5], [2, 4, 6, 4, 8], [4, 4, 7, 7, 5], [3, 2, 4, 5, 11], [4, 6, 7, 7, 5], [4, 6, 8, 3, 7], [2, 3, 1, 5, 5], [4, 3, 1, 6, 4], [3, 4, 5, 3, 12], [6, 4, 6, 3, 11], [7, 6, 8, 4, 5], [3, 3, 4, 4, 4], [3, 2, 8, 4, 9], [0, 6, 4, 4, 6], [4, 4, 4, 6, 4], [5, 4, 7, 8, 6], [3, 6, 6, 5, 6], [4, 9, 6, 4, 7], [2, 4, 7, 5, 8], [1, 6, 7, 8, 10], [5, 4, 3, 4, 6], [2, 1, 6, 5, 11], [3, 12, 6, 7, 5], [3, 2, 5, 5, 14], [3, 3, 2, 4, 5], [2, 5, 1, 7, 6]]
    config.reset_params['new_arrivals'] = sample_path
    # restart env
    # retest it
    alp_agent = ALPEJORColumnGenerationAgent(env=env, discount_factor=0.99, pretrain=True)
    #print("Action:", sa_advance_agent.policy(init_state, t))
    alp_policy_evaluator = PolicyEvaluator(env, alp_agent, discount_factor=env.discount_factor, is_inf=True)
    init_state, info = env.reset(**config.reset_params)
    states, rewards = alp_policy_evaluator.sample_path_evaluate(init_state, t, sample_path)
    print(sum(rewards)) # 339483.0489337634

    myopic_agent = InfiniteSAAAgent(env, discount_factor=env.discount_factor, is_myopic=True)
    init_state, info = env.reset(**config.reset_params)
    myopic_policy_evaluator = PolicyEvaluator(env, myopic_agent, discount_factor=env.discount_factor, is_inf=True)
    states, rewards = myopic_policy_evaluator.sample_path_evaluate(init_state, t, sample_path)
    print(sum(rewards))



