import numpy as np

from environment import MultiClassPoissonArrivalGenerator
from utils import str2treatment_patterns


class Config:
    def __init__(self, decision_epoch, class_number, arrival_generator, env_params, init_arrival, init_state):
        self.decision_epoch = decision_epoch
        self.class_number = class_number
        self.arrival_generator = arrival_generator
        self.env_params = env_params
        self.init_arrival = init_arrival
        self.init_state = init_state

    @classmethod
    def from_default(cls):
        decision_epoch = 10
        class_number = 7
        arrival_generator = MultiClassPoissonArrivalGenerator(3, 4, [1 / class_number] * class_number, is_precompute_state=True)
        env_params = {
            'treatment_pattern': [[i for i in reversed(range(1, class_number+1))]],
            'decision_epoch': decision_epoch,
            'arrival_generator': arrival_generator,
            'holding_cost': [10 - i * 5 / (class_number - 1) for i in range(class_number)],
            'overtime_cost': 40,
            'duration': 1,
            'regular_capacity': 5,
            'discount_factor': 1,
            'problem_type': 'advance'
        }
        init_arrival = np.array([6]*class_number)
        num_types = len(env_params['treatment_pattern'][0])
        bookings = np.array([0] * len(env_params['treatment_pattern']))
        future_schedule = np.array([[0] * num_types for _ in range(env_params['decision_epoch'])])
        init_state = (bookings, init_arrival, future_schedule)
        return cls(decision_epoch, class_number, arrival_generator, env_params, init_arrival, init_state)

    @classmethod
    def from_real_scale(cls):
        decision_epoch = 20
        class_number = 20
        arrival_generator = MultiClassPoissonArrivalGenerator(10, 30, [1 / class_number] * class_number, is_precompute_state=False)
        env_params = {
            'treatment_pattern': [[1]*class_number],
            'decision_epoch': decision_epoch,
            'arrival_generator': arrival_generator,
            'holding_cost': [10 - i * 5 / (class_number - 1) for i in range(class_number)],
            'overtime_cost': 40,
            'duration': 1,
            'regular_capacity': 10,
            'discount_factor': 0.99,
            'problem_type': 'advance'
        }
        init_arrival = np.array([5]*class_number)
        num_types = len(env_params['treatment_pattern'][0])
        bookings = np.array([0] * len(env_params['treatment_pattern']))
        future_schedule = np.array([[0] * num_types for _ in range(env_params['decision_epoch'])])
        init_state = (bookings, init_arrival, future_schedule)
        return cls(decision_epoch, class_number, arrival_generator, env_params, init_arrival, init_state)

    @classmethod
    def from_multiappt_default_case(cls):
        decision_epoch = 2
        class_number = 1
        arrival_generator = MultiClassPoissonArrivalGenerator(3, 4, [1 / class_number] * class_number,
                                                              is_precompute_state=True)
        env_params = {
            'treatment_pattern': [[0]*class_number,[1]*class_number],
            'decision_epoch': decision_epoch,
            'arrival_generator': arrival_generator,
            'holding_cost': [10 - i * 5 / max((class_number - 1), 1) for i in range(class_number)],
            'overtime_cost': 40,
            'duration': 1,
            'regular_capacity': 5,
            'discount_factor': 1,
            'problem_type': 'advance'
        }
        init_arrival = np.array([6] * class_number)
        num_types = len(env_params['treatment_pattern'][0])
        bookings = np.array([0] * len(env_params['treatment_pattern']))
        future_schedule = np.array([[0] * num_types for _ in range(env_params['decision_epoch'])])
        init_state = (bookings, init_arrival, future_schedule)
        return cls(decision_epoch, class_number, arrival_generator, env_params, init_arrival, init_state)

    @classmethod
    def from_EJOR_case(cls):
        decision_epoch = 20
        total_arrival_rate_mean = 8.25
        type_probs = np.array([0.19, 0.11,0.11,1.43,0.59,0.45,1.42,1.36,0.57,0.38,0.18,0.18,0.29,0.21,0.3,0.29,0.15,0.04])/total_arrival_rate_mean
        arrival_generator = MultiClassPoissonArrivalGenerator(total_arrival_rate_mean, 25, type_probs,
                                                              is_precompute_state=False)

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
                    '1 * 2 + 32 * 1']
        treatment_pattern = str2treatment_patterns(patterns)
        class_number = len(patterns)
        holding_cost = [146.5]*3 + [115.0]*3 + [123.25]*6 + [139.0]*2 + [112.0]*3 + [114.0]*1
        env_params = {
            'treatment_pattern': treatment_pattern,
            'decision_epoch': decision_epoch,
            'arrival_generator': arrival_generator,
            'holding_cost': holding_cost,
            'overtime_cost': 100,
            'duration': 1,
            'regular_capacity': 120,
            'discount_factor': 0.99,
            'problem_type': 'advance'
        }
        init_arrival = np.array([5] * class_number)
        num_types = len(env_params['treatment_pattern'][0])
        bookings = np.array([0] * len(env_params['treatment_pattern']))
        future_schedule = np.array([[0] * num_types for _ in range(env_params['decision_epoch'])])
        init_state = (bookings, init_arrival, future_schedule)
        return cls(decision_epoch, class_number, arrival_generator, env_params, init_arrival, init_state)

