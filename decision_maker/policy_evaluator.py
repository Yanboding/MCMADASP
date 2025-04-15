import os
import time
from collections import defaultdict
from pprint import pprint
from joblib import Parallel, delayed
import numpy as np

from decision_maker import OptimalAgent, SAAdvanceAgent
from environment import MultiClassPoissonArrivalGenerator, AdvanceSchedulingEnv
from utils import iter_to_tuple
from utils.running_stat import RunningStat


class PolicyEvaluator:

    def __init__(self, env, agent, discount_factor, V=None):
        self.env = env
        self.agent = agent
        self.discount_factor = discount_factor
        if V is None:
            self.V = {}
        self.sample_average_V = defaultdict(lambda: RunningStat(1))

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
        '''
        print('Time:', t)
        print('State:', state, 'Action:', action)
        print('Total_cost:', q)
        print()
        '''
        self.V[(state_tuple, t)] = q
        return self.V[(state_tuple, t)]

    def sample_path_evaluate(self, state, t, sample_path):
        states = []
        rewards = []
        s, info = self.env.reset(state, t, sample_path)
        for tau in range(len(sample_path)):
            states.append(s)
            a = self.agent.policy(s, t + tau)
            next_state, reward, done, info = self.env.step(a)
            rewards.append(reward)
            s = next_state
            if done:
                break
        return states, rewards

    def simulation_evaluate_helper(self, state, t, sample_paths):
        sample_average_V = defaultdict(lambda: RunningStat(1))
        for sample_path in sample_paths:
            states, rewards = self.sample_path_evaluate(state,t,sample_path)
            G = 0.0
            for tau in reversed(range(len(states))):
                G = self.discount_factor * G + rewards[tau]
                s = states[tau]
                sample_average_V[(iter_to_tuple(s), t + tau)].record(G)
        return sample_average_V

    def simulation_evaluate(self, state, t, replication, confidence):
        num_cpus = os.cpu_count()
        sample_paths = [self.env.reset_arrivals(t=t) for _ in range(replication)]
        responses = Parallel(n_jobs=-1)(
            delayed(self.simulation_evaluate_helper)(state=state, t=t, sample_paths=sample_paths[i::num_cpus])for i in range(num_cpus))
        for sample_average_V in responses:
            for state_tuple, value_stat in sample_average_V.items():
                self.sample_average_V[state_tuple].merge(value_stat)
        state_tuple = iter_to_tuple(state)
        mean = self.sample_average_V[(state_tuple, t)].mean()
        half_window = self.sample_average_V[(state_tuple, t)].half_window(confidence)
        return mean, mean-half_window, mean+half_window

if __name__ == '__main__':
    decision_epoch = 20
    class_number = 2
    arrival_generator = MultiClassPoissonArrivalGenerator(3, 1, [1 / class_number] * class_number)

    env_params = {
        'treatment_pattern': [[2, 1]],
        'decision_epoch': decision_epoch,
        'arrival_generator': arrival_generator,
        'holding_cost': [10, 5],
        'overtime_cost': 40,
        'duration': 1,
        'regular_capacity': 5,
        'discount_factor': 1,
        'problem_type': 'advance'
    }
    bookings = np.array([0])
    future_schedule = np.array([[0] * class_number for i in range(decision_epoch)])
    new_arrival = np.array([3, 3])
    init_state = (bookings, new_arrival, future_schedule)
    t = 1
    env = AdvanceSchedulingEnv(**env_params)
    sa_advance_agent = SAAdvanceAgent(env=env, discount_factor=env_params['discount_factor'], sample_path_number=50)
    #print("Action:", sa_advance_agent.policy(init_state, t))
    policy_evaluator = PolicyEvaluator(env, sa_advance_agent, discount_factor=env_params['discount_factor'])
    start = time.time()
    mean, cl, ch = policy_evaluator.simulation_evaluate(init_state, t, 8, confidence=0.95)
    print(mean, (cl, ch))
    print(time.time() - start)
    #print(policy_evaluator.evaluate(init_state, t))


