import numpy as np

from environment import MultiClassPoissonArrivalGenerator, SchedulingEnv
from utils import str2treatment_patterns, wait_time


class ExperimentConfig:

    def __init__(self, env, reset_params, init_state):
        self.env = env
        self.reset_params = reset_params
        self.init_state = init_state

    @classmethod
    def from_multiappt_default_case(cls):
        decision_epoch = 3
        class_number = 2
        arrival_generator = MultiClassPoissonArrivalGenerator(3, 4, [1 / class_number] * class_number,
                                                              is_precompute_state=True)
        treatment_pattern = np.array([[2, 1],
                                      [1, 0],
                                      [1, 1]])
        holding_cost = [10 - i * 5 / max((class_number - 1), 1) for i in range(class_number)]
        holding_cost_fn = lambda t, i: holding_cost[i]
        env_params = {
            'treatment_pattern': treatment_pattern,
            'decision_epoch': decision_epoch,
            'arrival_generator': arrival_generator,
            'holding_cost': holding_cost_fn,
            'overtime_cost': 40,
            'duration': 1,
            'regular_capacity': 5,
            'discount_factor': 0.99,
        }
        env = SchedulingEnv(**env_params)
        reset_params = {
            'percentage_occupied': 0,
            't': 1
        }
        bookings = np.array([0]*(decision_epoch+len(treatment_pattern)-1))
        delta = np.array([6, 6])
        init_state = (bookings, delta)
        return cls(env, reset_params, init_state)

    @classmethod
    def from_EJOR_case(cls):
        decision_epoch = 30
        total_arrival_rate_mean = 8.25
        type_probs = np.array(
            [0.19, 0.11, 0.11, 1.43, 0.59, 0.45, 1.42, 1.36, 0.57, 0.38, 0.18, 0.18, 0.29, 0.21, 0.3, 0.29, 0.15,
             0.04]) / total_arrival_rate_mean
        arrival_generator = MultiClassPoissonArrivalGenerator(total_arrival_rate_mean, 2, type_probs,
                                                              is_precompute_state=True)

        patterns = ['1 * 2 + 4 * 1',
                    '1 * 2',
                    '1 * 2 + 3 * 1',
                    '1 * 2 + 15 * 1',
                    '1 * 2 + 15 * 1 + 1 * 2 + 3 * 1',
                    '1 * 3 + 15 * 2',
                    '1 * 2',
                    '1 * 2 + 4 * 1',
                    '1 * 2 + 9 * 1',
                    '1 * 2 + 3 * 1',
                    '1 * 2 + 14 * 1',
                    '1 * 1',
                    '1 * 2 + 19 * 1',
                    '1 * 3 + 34 * 2',
                    '1 * 2 + 32 * 1',
                    '1 * 2 + 36 * 1',
                    '1 * 2 + 21 * 1 + 1 * 2 + 14 * 1',
                    '1 * 2 + 32 * 1']
        treatment_pattern = str2treatment_patterns(patterns)
        l1_3 = [(0, 1, 0), (1, 5, 100), (5, 100, 150)]
        l4_6 = [(0, 10, 0), (10, 20, 50), (20, 40, 100), (40, 100, 150)]
        l7_12 = [(0, 5, 0), (5, 10, 65), (10, 40, 100), (40, 100, 150)]
        l13_14 = [(0, 5, 0), (5, 10, 80), (10, 100, 150)]
        l15_17 = [(0, 10, 0), (10, 20, 40), (20, 30, 80), (30, 40, 100), (40, 100, 150)]
        l18 = [(0, 10, 0), (10, 20, 50), (20, 30, 90), (30, 40, 100), (40, 100, 150)]
        class_number = len(patterns)
        holding_cost = []
        for i in range(class_number):
            if 0 <= i < 3:
                wait_cost_by_day = wait_time(l1_3)
            elif 3 <= i < 6:
                wait_cost_by_day = wait_time(l4_6)
            elif 6 <= i < 12:
                wait_cost_by_day = wait_time(l7_12)
            elif 12 <= i < 14:
                wait_cost_by_day = wait_time(l13_14)
            elif 14 <= i < 17:
                wait_cost_by_day = wait_time(l15_17)
            elif 17 <= i < 18:
                wait_cost_by_day = wait_time(l18)
            holding_cost.append(wait_cost_by_day)
        holding_cost = np.array(holding_cost).T
        holding_cost_fn = lambda t, i: holding_cost[t, i]
        env_params = {
            'treatment_pattern': treatment_pattern,
            'decision_epoch': decision_epoch,
            'arrival_generator': arrival_generator,
            'holding_cost': holding_cost_fn,
            'overtime_cost': 100,
            'duration': 1,
            'regular_capacity': 120,
            'discount_factor': 0.99,
        }
        env = SchedulingEnv(**env_params)

        reset_params = {
            'percentage_occupied': 0.9,
            't': 1
        }
        occupied_capacity = env_params['regular_capacity'] * reset_params['percentage_occupied']
        bookings = np.array([occupied_capacity] * (decision_epoch + len(treatment_pattern) - 1))
        delta = np.array([1]*class_number)
        init_state = (bookings, delta)
        return cls(env, reset_params, init_state)

    @classmethod
    def from_adjust_EJOR_case(cls):
        decision_epoch = 30
        class_number = 18
        arrival_rates = np.array(
            [0.19, 0.11, 0.11, 1.43, 0.59, 0.45, 1.42, 1.36, 0.57, 0.38, 0.18, 0.18, 0.29, 0.21, 0.3, 0.29, 0.15,
             0.04])[:class_number]
        patterns = ['1* 2 + 4 * 1',
                    '1*2',
                    '1*2+3*1',
                    '1* 2 + 15 * 1',
                    '1*2 + 15*1 + 1*2 + 3*1',
                    '1* 3 + 15 * 2',
                    '1* 2',
                    '1* 2 + 4 * 1',
                    '1* 2 + 9 * 1',
                    '1 * 2 + 3 * 1',
                    '1 * 2 + 14 * 1',
                    '1 * 1',
                    '1 * 2 + 19 * 1',
                    '1 * 3 + 34 * 2',
                    '1 * 2 + 32 * 1',
                    '1 * 2 + 36 * 1',
                    '1 * 2 + 21 * 1 + 1 * 2 + 14 * 1',
                    '1 * 2 + 32 * 1'][:class_number]
        treatment_pattern = str2treatment_patterns(patterns)
        holding_cost = [132.5] * 3 + [100] * 3 + [66.25] * 6 + [27.5] * 2 + [25] * 3 + [20] * 1
        holding_cost = holding_cost[:class_number]
        total_arrival_rate_mean = np.sum(arrival_rates)
        type_probs = arrival_rates / total_arrival_rate_mean
        arrival_generator = MultiClassPoissonArrivalGenerator(total_arrival_rate_mean, 25, type_probs,
                                                              is_precompute_state=False)
        holding_cost_fn = lambda t, i: holding_cost[i]
        env_params = {
            'treatment_pattern': treatment_pattern,
            'decision_epoch': decision_epoch,
            'arrival_generator': arrival_generator,
            'holding_cost': holding_cost_fn,
            'overtime_cost': 100,
            'duration': 1,
            'regular_capacity': 120,
            'discount_factor': 0.99,
        }
        env = SchedulingEnv(**env_params)

        reset_params = {
            'percentage_occupied': 0.9,
            't': 1
        }
        occupied_capacity = env_params['regular_capacity'] * reset_params['percentage_occupied']
        bookings = np.array([occupied_capacity] * (decision_epoch + len(treatment_pattern) - 1))
        delta = np.array([1] * class_number)
        init_state = (bookings, delta)
        return cls(env, reset_params, init_state)

def get_config_by_type(case_type):
    if case_type == "default":
        config = ExperimentConfig.from_multiappt_default_case()
    elif case_type == 'ejor':
        config = ExperimentConfig.from_EJOR_case()
    elif case_type == 'adjust_ejor':
        config = ExperimentConfig.from_adjust_EJOR_case()
    return config
