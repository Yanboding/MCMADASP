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
        actions = []
        rewards = []
        s, info = self.env.reset(state, t, sample_path)
        for tau in range(len(sample_path[t-1:])):
            print("Current time step:", t + tau)
            states.append(s)
            if tau == 0 and action:
                a = action
            else:
                start = time.time()
                a = self.agent.policy(s, t + tau)
                end = time.time()
                print(f"Policy {t + tau} computation time: {end - start} seconds")
            actions.append(a)
            next_state, reward, done, info = self.env.step(a)
            rewards.append(reward)
            s = next_state
            if done:
                break
        return states, actions, rewards

    def simulation_evaluate_helper(self, state, t, sample_paths, action=None):
        if self.is_inf:
            self.discount_factor = 1
        sample_average_V = defaultdict(lambda: RunningStats())
        for sample_path in sample_paths:
            states, actions, rewards = self.sample_path_evaluate(state,t,sample_path, action=action)
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
    
    def penalized_sample_path_evaluate(self, state, t, sample_path, action=None):
        states = []
        actions = []
        rewards = []
        penalties = []
        s, info = self.env.reset(state, t, sample_path)
        tau = 0
        if tau == 0 and action:
            a = action
        else:
            start = time.time()
            a = self.agent.policy(s, t + tau)
            end = time.time()
            print(f"Policy {t + tau} computation time: {end - start} seconds")
        next_state, reward, done, info = self.env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(reward)
        penalties.append(0)
        s = next_state
        for tau, new_arrival in enumerate(sample_path[t:], start=1):
            print("Current time step:", t + tau)
            states.append(s)
            if tau == 0 and action:
                a = action
            else:
                start = time.time()
                a = self.agent.policy(s, t + tau)
                end = time.time()
                print(f"Policy {t + tau} computation time: {end - start} seconds")
            actions.append(a)
            next_state, reward, done, info = self.env.step(a)
            (next_regular_booking, next_overtime, next_waitlist) = next_state
            (regular_booking, overtime, waitlist) = s
            (advance_scheduling_decision, overtime_decision) = a
            arrival_difference = self.env.arrival_generator.mean_by_type - new_arrival
            total_booked_slots = (next_regular_booking + next_overtime).sum()
            penalty = 2 * (waitlist - advance_scheduling_decision.sum(axis=0) + total_booked_slots) @ arrival_difference
            penalties.append(penalty)
            rewards.append(reward)
            s = next_state
            if done:
                break
        return states, actions, rewards, penalties



if __name__ == '__main__':
    from experiments import get_config_by_type
    from decision_maker import InfiniteSAAAgent, ALPRowGenerationAgent, InfinitePenalizedSAAAgent, MyopicAgent
    config = get_config_by_type('toy')
    env = config.env
    t = 2
    sample_path = [[3, 4, 4], [2, 6, 1], [3, 3, 1], [2, 1, 1], [4, 2, 2], [3, 5, 3], [2, 4, 2], [3, 3, 2], [2, 2, 1], [4, 3, 2], [3,4,3], [2,5,2], [3,4,1], [2,2,2], [4,3,3], [3,4,2], [2,3,1], [3,2,2], [2,1,1], [4,2,2]]
    config.reset_params['new_arrivals'] = sample_path
    init_state, info = env.reset(**config.reset_params)
    print('Initial state:', init_state)
    alp_agent = MyopicAgent(env=env, discount_factor=0.99)
    #print("Action:", sa_advance_agent.policy(init_state, t))
    alp_policy_evaluator = PolicyEvaluator(env, alp_agent, discount_factor=env.discount_factor, is_inf=True)
    states, actions, rewards, penalties = alp_policy_evaluator.penalized_sample_path_evaluate(init_state, t, sample_path)
    print(penalties)
    print(rewards) # 339483.0489337634
