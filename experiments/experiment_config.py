import math

import numpy as np
import numbers

from environment import MultiClassPoissonArrivalGenerator, RTEnv
from utils import str2treatment_patterns, wait_time
from scipy.stats import geom

class HoldingCostCalculator:
    """
    A picklable, function-like object that calculates holding cost.
    """
    def __init__(self, cost_data):
        # Store the cost data when the object is created
        self.cost_data = cost_data

    def __call__(self, t, i):
        if self.cost_data.ndim == 1:
            holding_cost = self.cost_data[i]
        else:
            holding_cost = self.cost_data[t, i]
        return holding_cost
    
    def get_waiting_target(self, i):
        waiting_target = np.argmax(self.cost_data[:, i] > 0) if np.any(self.cost_data[:, i] > 0) else float('inf')
        return waiting_target - 1 # think carefully about the -1 here

class OvertimeCostCalculator:
    """
    A picklable, function-like object that calculates holding cost.
    """
    def __init__(self, cost_data):
        # Store the cost data when the object is created
        self.cost_data = cost_data

    def __call__(self, t):
        # This method is executed when you call the instance.
        # It ignores 't', just like the original lambda.
        if isinstance(self.cost_data, numbers.Number):
            overtime_cost = self.cost_data
        else:
            overtime_cost = self.cost_data[t]
        return overtime_cost

class PostponingCostCalculator:
    """
    A picklable, function-like object that calculates holding cost.
    """
    def __init__(self, cost_data):
        # Store the cost data when the object is created
        self.cost_data = cost_data

    def __call__(self, i):
        # This method is executed when you call the instance.
        # It ignores 't', just like the original lambda.
        if isinstance(self.cost_data, numbers.Number):
            postponing_cost = self.cost_data
        else:
            postponing_cost = self.cost_data[i]
        return postponing_cost

class ExperimentConfig:

    def __init__(self, env, reset_params, init_state, valid_action=None,args=None):
        self.env = env
        self.reset_params = reset_params
        self.init_state = init_state
        self.valid_action = valid_action
        self.args = args if args is not None else {}

    @classmethod
    def from_ejor_default_case(cls):
        treatment_patterns = ["1 * 1", "2 * 1", "3 * 1", "4 * 1"]
        booking_window_size = 15
        treatment_num = len(treatment_patterns)
        arrival_rates = [4, 6, 8, 10][:treatment_num]
        l = [[(0, 5, 0), (5, 100, 50)],
             [(0, 7, 0), (7, 100, 50)],
             [(0, 9, 0), (9, 100, 50)],
             [(0, 11, 0), (11, 100, 50)]]
        holding_cost = [wait_time(l[i]) for i in range(len(treatment_patterns))]
        holding_cost = np.array(holding_cost).T
        env_args = {
            "booking_window_size": booking_window_size,
            "arrival_rates": arrival_rates,
            "patterns": treatment_patterns,
            "holding_cost_by_day_by_type": holding_cost.tolist(),
            "overtime_cost_by_day": 100,
            "postponing_cost": 2000,
            "duration": 1,
            "regular_capacity": 60,
            "overtime_capacity": 30,
            "discount_factor": 0.95,
            "reset_params": {
                'percentage_occupied': 0,
                't': 1
            },
            "maximum_total_arrival": 75,
            "init_state": None,
            "valid_action": None,
            "env_random_seed": 0,
            "stop_time_random_seed": 1,
            "arrival_random_seed": 42,
        }
        return cls.from_custom_case(**env_args)
    
    @classmethod
    def from_small_case(cls):
        treatment_patterns = ["1 * 2 + 4 * 1", 
                              "1 * 2 + 15 * 1",
                              "1 * 2 + 9 * 1",
                              "1 * 2 + 19 * 1",
                              "1 * 2 + 32 * 1",
                              "1 * 2 + 32 * 1"]
        '''
        1*2 + 15*1 + 1*2 + 3*1
        treatment_patterns = ["1*2 + 4*1", 
                              "1*3 + 15*2 + 4*1",
                              "1*2 + 14*1",
                              "1*3 + 19*2 + 15*1",
                              "1*2 + 21*1 + 1*2 + 14*1",
                              "1 * 2 + 32 * 1"]
        '''
        booking_window_size = 25
        arrival_rates = [0.41, 2.47, 4.09, 0.5, 0.74, 0.04]
        l = [[(0, 1, 0), (1, 5, 100), (5, 100, 150)],
             [(0, 10, 0), (10, 20, 50), (20, 40, 100), (40, 100, 150)],
             [(0, 5, 0), (5, 10, 65), (10, 40, 100), (40, 100, 150)],
             [(0, 5, 0), (5, 10, 80), (10, 100, 150)],
             [(0, 10, 0), (10, 20, 40), (20, 30, 80), (30, 40, 100), (40, 100, 150)],
             [(0, 10, 0), (10, 20, 50), (20, 30, 90), (30, 40, 100), (40, 100, 150)]]
        holding_cost = [wait_time(l[i]) for i in range(len(treatment_patterns))]
        holding_cost = np.array(holding_cost).T
        env_args = {
            "booking_window_size": booking_window_size,
            "arrival_rates": arrival_rates,
            "patterns": treatment_patterns,
            "holding_cost_by_day_by_type": holding_cost.tolist(),
            "overtime_cost_by_day": 100,
            "postponing_cost": 2000,
            "duration": 1,
            "regular_capacity": 120,
            "overtime_capacity": 15,
            "discount_factor": 0.99,
            "reset_params": {
                'percentage_occupied': 0,
                't': 1
            },
            "maximum_total_arrival": math.ceil(sum(arrival_rates) * 3),
            "init_state": None,
            "valid_action": None,
            "env_random_seed": 0,
            "stop_time_random_seed": 1,
            "arrival_random_seed": 42,
        }
        return cls.from_custom_case(**env_args)
    
    @classmethod
    def from_toy_case(cls):
        booking_window_size = 7
        treatment_patterns = ["1 * 3", 
                              "1 * 2"]
        arrival_rates = [1, 2]
        l = [[(0, 1, 0), (1, booking_window_size, 100)],
             [(0, 1, 0), (1, booking_window_size, 10)]]
        regular_capacity = 5
        overtime_capacity = 5
        holding_cost = [wait_time(l[i]) for i in range(len(treatment_patterns))]
        holding_cost = np.array(holding_cost).T
        treatment_pattern = str2treatment_patterns(treatment_patterns)
        planning_horizon = booking_window_size + treatment_pattern.shape[0] - 1
        regular_bookings = [regular_capacity]*planning_horizon
        regular_bookings[-1] = 0
        overtime_bookings = [0]*planning_horizon
        initial_state = (regular_bookings, overtime_bookings, arrival_rates.copy())
        env_args = {
            "booking_window_size": booking_window_size,
            "arrival_rates": arrival_rates,
            "patterns": treatment_patterns,
            "holding_cost_by_day_by_type": holding_cost.tolist(),
            "overtime_cost_by_day": 100,
            "postponing_cost": 2000,
            "duration": 1,
            "regular_capacity": regular_capacity,
            "overtime_capacity": overtime_capacity,
            "discount_factor": 0.99,
            "reset_params": {
                "init_state": initial_state,
            },
            "maximum_total_arrival": math.ceil(sum(arrival_rates) * 3),
            "init_state": None,
            "valid_action": None,
            "env_random_seed": 0,
            "stop_time_random_seed": 1,
            "arrival_random_seed": 42,
        }
        return cls.from_custom_case(**env_args)

    @classmethod
    def from_ejor_base_case(cls):
        class_num = 18
        l1_3 = [(0, 1, 0), (1, 5, 100), (5, 100, 150)]
        l4_6 = [(0, 10, 0), (10, 20, 50), (20, 40, 100), (40, 100, 150)]
        l7_12 = [(0, 5, 0), (5, 10, 65), (10, 40, 100), (40, 100, 150)]
        l13_14 = [(0, 5, 0), (5, 10, 80), (10, 100, 150)]
        l15_17 = [(0, 10, 0), (10, 20, 40), (20, 30, 80), (30, 40, 100), (40, 100, 150)]
        l18 = [(0, 10, 0), (10, 20, 50), (20, 30, 90), (30, 40, 100), (40, 100, 150)]
        holding_cost = []
        for i in range(class_num):
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
        arrival_rates = [0.19, 0.11, 0.11, 1.43, 0.59, 0.45, 1.42, 1.36, 0.57, 0.38, 0.18, 0.18, 0.29, 0.21, 0.3,
                         0.29, 0.15, 0.04][:class_num]
        total_arrival_rate = sum(arrival_rates)
        booking_window_size = 20
        patterns = ['1 * 2 + 4 * 1',
                    '1 * 2',
                    '1 * 2 + 3 * 1',
                    '1 * 2 + 15 * 1',
                    '1 * 2 + 15*1 + 1*2 + 3*1',
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
                    '1 * 2 + 32 * 1'][:class_num]
        treatment_pattern = str2treatment_patterns(patterns)
        planning_horizon = booking_window_size + treatment_pattern.shape[0] - 1
        initial_state = ([0]*planning_horizon, [0]*planning_horizon, np.round(arrival_rates).astype(int).tolist())
        env_args = {
            "booking_window_size": booking_window_size,
            'arrival_rates': arrival_rates,
            'patterns': patterns,
            'holding_cost_by_day_by_type': holding_cost[:, :class_num].tolist(),
            'overtime_cost_by_day': 100,
            "postponing_cost": 2000,
            'duration': 1,
            'regular_capacity': 120,
            "overtime_capacity": 15,
            'discount_factor': 0.99,
            'reset_params': {
                "init_state": initial_state,
            },
            'maximum_total_arrival': math.ceil(total_arrival_rate * 3),
            'init_state': None,
            'valid_action': None,
            'env_random_seed': 0,
            'stop_time_random_seed':1,
            'arrival_random_seed': 42
        }
        return cls.from_custom_case(**env_args)
    
    @classmethod
    def from_custom_case(cls,
                              booking_window_size,
                              arrival_rates,
                              patterns,
                              holding_cost_by_day_by_type,
                              overtime_cost_by_day,
                              postponing_cost,
                              duration,
                              regular_capacity,
                              overtime_capacity,
                              discount_factor,
                              reset_params,
                              maximum_total_arrival=None,
                              init_state=None,
                              valid_action=None,
                              env_random_seed=None,
                              stop_time_random_seed=None,
                              arrival_random_seed=None,
                              ):
        args = {
            'booking_window_size': booking_window_size,
            'arrival_rates': arrival_rates,
            'patterns': patterns,
            'holding_cost_by_day_by_type': holding_cost_by_day_by_type,
            'overtime_cost_by_day': overtime_cost_by_day,
            'postponing_cost': postponing_cost,
            'duration': duration,
            'regular_capacity': regular_capacity,
            'overtime_capacity': overtime_capacity,
            'discount_factor': discount_factor,
            'reset_params': reset_params,
            'maximum_total_arrival': maximum_total_arrival,
            'init_state': init_state,
            'valid_action': valid_action,
            'env_random_seed': env_random_seed,
            'stop_time_random_seed': stop_time_random_seed,
            'arrival_random_seed': arrival_random_seed
        }
        treatment_pattern = str2treatment_patterns(patterns)
        arrival_rates = np.array(arrival_rates)
        total_arrival_rate_mean = np.sum(arrival_rates)
        type_probs = arrival_rates / total_arrival_rate_mean
        if maximum_total_arrival is None:
            maximum_total_arrival = 3 * total_arrival_rate_mean
        geom_p = 1 - discount_factor
        arrival_generator = MultiClassPoissonArrivalGenerator(total_arrival_rate_mean, maximum_total_arrival,
                                                              type_probs,
                                                              random_seed=arrival_random_seed,
                                                              use_qmc=True, 
                                                              max_periods=int(geom.ppf(0.9999, p=geom_p)), # to ensure that the probability of generating more than max_periods arrivals is very small
                                                              geom_p=geom_p,
                                                              is_precompute_state=False)
        holding_cost_by_day_by_type = np.array(holding_cost_by_day_by_type)
        holding_cost_fn = HoldingCostCalculator(holding_cost_by_day_by_type)
        overtime_cost_fn = OvertimeCostCalculator(overtime_cost_by_day)
        postponing_cost_fn = PostponingCostCalculator(postponing_cost)
        env_params = {
            'treatment_pattern': treatment_pattern,
            'booking_window_size': booking_window_size,
            'arrival_generator': arrival_generator,
            'holding_cost': holding_cost_fn,
            'overtime_cost': overtime_cost_fn,
            'postponing_cost': postponing_cost_fn,
            'duration': duration,
            'regular_capacity': regular_capacity,
            'overtime_capacity': overtime_capacity,
            'discount_factor': discount_factor,
            'init_state_random_seed': env_random_seed,
            'stop_time_random_seed': stop_time_random_seed
        }
        env = RTEnv(**env_params)
        if init_state == None:
            bookings = np.array([0] * env.planning_horizon)
            overtimes = np.array([0] * env.planning_horizon)
            waitlists = np.array([1000] * env.num_types)
            advance_scheduling_decision = np.array(
                [[0] * env.num_types for _ in range(env.booking_window_size - 1)]+[waitlists])
            overtime_decision = np.maximum(
                np.minimum(
                    bookings + env.convert_action_to_booking_slots(advance_scheduling_decision) - regular_capacity,
                    overtime_capacity), 0)
            init_state = (bookings, overtimes, waitlists)
            valid_action = (advance_scheduling_decision, overtime_decision)
        return cls(env, reset_params, init_state, valid_action, args)

def get_config_by_type(case_type, args=None):
    if args is None:
        args = {}
    if case_type == 'ejor':
        config = ExperimentConfig.from_ejor_base_case()
    elif case_type == 'ejor_default':
        config = ExperimentConfig.from_ejor_default_case()
    elif case_type == 'small':
        config = ExperimentConfig.from_small_case()
    elif case_type == 'toy':
        config = ExperimentConfig.from_toy_case()
    elif case_type == 'infinite_custom':
        config = ExperimentConfig.from_custom_case(**args)
    return config

if __name__ == '__main__':
    config = get_config_by_type('toy')
    print(config.args['reset_params'])
    


