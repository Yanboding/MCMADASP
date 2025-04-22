import numpy as np

from environment import MultiClassPoissonArrivalGenerator


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
        decision_epoch = 1
        class_number = 2
        arrival_generator = MultiClassPoissonArrivalGenerator(3, 4, [1 / class_number] * class_number, is_precompute_state=True)
        env_params = {
            'treatment_pattern': [[2, 1]],
            'decision_epoch': decision_epoch,
            'arrival_generator': arrival_generator,
            'holding_cost': [10 - i * 5 / (class_number - 1) for i in range(class_number)],
            'overtime_cost': 40,
            'duration': 1,
            'regular_capacity': 5,
            'discount_factor': 1,
            'problem_type': 'advance'
        }
        init_arrival = np.array([6, 6])
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
