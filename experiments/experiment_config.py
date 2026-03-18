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
        return cls.from_ejor_custom_case(**env_args)
    
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
        return cls.from_ejor_custom_case(**env_args)
    
    @classmethod
    def from_toy_case(cls):
        booking_window_size = 3
        treatment_patterns = ["1 * 2", 
                              "1 * 2"]
        arrival_rates = [1, 2]
        l = [[(0, 1, 0), (1, 5, 100), (5, 100, 150)],
             [(0, 1, 0), (1, 10, 20), (10, 100, 150)]]
        # treatment_patterns = ["1 * 2"]
        # arrival_rates = [3]
        # l = [[(0, 1, 10), (1, 5, 100), (5, 100, 150)]]
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
            "regular_capacity": 7,
            "overtime_capacity": 2,
            "discount_factor": 0.95,
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
        return cls.from_ejor_custom_case(**env_args)

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
        env_args = {
            "booking_window_size": 20,
            'arrival_rates': arrival_rates,
            'patterns': ['1 * 2 + 4 * 1',
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
                         '1 * 2 + 32 * 1'][:class_num],
            'holding_cost_by_day_by_type': holding_cost[:, :class_num].tolist(),
            'overtime_cost_by_day': 100,
            "postponing_cost": 2000,
            'duration': 1,
            'regular_capacity': 120,
            "overtime_capacity": 15,
            'discount_factor': 0.99,
            'reset_params': {
                'percentage_occupied': 0.99,
                't': 1
            },
            'maximum_total_arrival': math.ceil(total_arrival_rate * 3),
            'init_state': None,
            'valid_action': None,
            'env_random_seed': 0,
            'stop_time_random_seed':1,
            'arrival_random_seed': 42
        }
        return cls.from_ejor_custom_case(**env_args)
    
    @classmethod
    def from_ejor_custom_case(cls,
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
        arrival_generator = MultiClassPoissonArrivalGenerator(total_arrival_rate_mean, maximum_total_arrival,
                                                              type_probs,
                                                              random_seed=arrival_random_seed,
                                                              use_qmc=True, 
                                                              max_periods=int(geom.ppf(0.999, p=1-discount_factor)), # to ensure that the probability of generating more than max_periods arrivals is very small
                                                              geom_p=(1-discount_factor),
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
        config = ExperimentConfig.from_ejor_custom_case(**args)
    return config

if __name__ == '__main__':
    import re
    from utils.treatment_pattern import treatment_pattern2str
    def slot_session_rate(treatment_pattern_str):
        '''
        1*2 + 15*1 + 1*2 + 3*1
        '''
        term_re = re.compile(r'\s*(\d+)\s*\*\s*(\d+)\s*')
        slot_num = 0
        session_num = 0
        for term in treatment_pattern_str.split('+'):
            term = term.strip()
            m = term_re.fullmatch(term)
            if not m:
                raise ValueError(f"Invalid term: '{term}' (expected form 'x*y')")
            x, y = map(int, m.groups())
            slot_num += x * y
            session_num += x
        return slot_num / session_num
    type1 = ['1 * 2 + 4 * 1',
             '1 * 2',
             '1 * 2 + 3 * 1',]
    rate1 = np.array([0.19, 0.11, 0.11,])
    prob1 = rate1 / np.sum(rate1)
    treatment_pattern1 = str2treatment_patterns(type1)
    pattern1 = np.round(prob1 @ treatment_pattern1.T).astype(int)
    pattern1_str = treatment_pattern2str(pattern1)
    print(pattern1_str)
    type2 = ['1 * 2 + 15 * 1',
             '1 * 2 + 15*1 + 1*2 + 3*1',
             '1 * 3 + 15 * 2',]
    rate2 = np.array([1.43, 0.59, 0.45,])
    prob2 = rate2 / np.sum(rate2)
    treatment_pattern2 = str2treatment_patterns(type2)
    pattern2 = np.round(prob2 @ treatment_pattern2.T).astype(int)
    pattern2_str = treatment_pattern2str(pattern2)
    print(pattern2_str)
    type3 = ['1 * 2',
             '1 * 2 + 4 * 1',
             '1 * 2 + 9 * 1',
             '1 * 2 + 3 * 1',
             '1 * 2 + 14 * 1',
             '1 * 1',]
    rate3 = np.array([1.42, 1.36, 0.57, 0.38, 0.18, 0.18,])
    prob3 = rate3 / np.sum(rate3)
    treatment_pattern3 = str2treatment_patterns(type3)
    pattern3 = np.round(prob3 @ treatment_pattern3.T).astype(int)
    pattern3_str = treatment_pattern2str(pattern3)
    print(pattern3_str)
    type4 = ['1 * 2 + 19 * 1',
             '1 * 3 + 34 * 2',]
    rate4 = np.array([0.29, 0.21,])
    prob4 = rate4 / np.sum(rate4)
    treatment_pattern4 = str2treatment_patterns(type4)
    pattern4 = np.round(prob4 @ treatment_pattern4.T).astype(int)
    pattern4_str = treatment_pattern2str(pattern4)
    print(pattern4_str)
    type5 = ['1 * 2 + 32 * 1',
             '1 * 2 + 36 * 1',
             '1 * 2 + 21 * 1 + 1 * 2 + 14 * 1',]
    rate5 = np.array([0.3, 0.29, 0.15,])
    prob5 = rate5 / np.sum(rate5)
    treatment_pattern5 = str2treatment_patterns(type5)
    pattern5 = np.round(prob5 @ treatment_pattern5.T).astype(int)
    pattern5_str = treatment_pattern2str(pattern5)
    print(pattern5_str)
    type6 = ['1 * 2 + 32 * 1']
    rate6 = np.array([0.04])
    prob6 = rate6 / np.sum(rate6)
    treatment_pattern6 = str2treatment_patterns(type6)
    pattern6 = np.round(prob6 @ treatment_pattern6.T).astype(int)
    pattern6_str = treatment_pattern2str(pattern6)
    print(pattern6_str)

    treatment_pattern = type1 + type2 + type3 + type4 + type5 + type6
    arrival_rates = np.concatenate([rate1, rate2, rate3, rate4, rate5, rate6])
    average_workload = 0
    for tratment_pattern_str, rate in zip(treatment_pattern, arrival_rates):
        ss_rate = slot_session_rate(tratment_pattern_str)
        workload = rate * ss_rate
        average_workload += workload
    print('Average workload:', average_workload)

    treatment_pattern = [pattern1_str, pattern2_str, pattern3_str, pattern4_str, pattern5_str, pattern6_str]
    arrival_rates = np.array([np.sum(rate1), np.sum(rate2), np.sum(rate3), np.sum(rate4), np.sum(rate5), np.sum(rate6)])
    print('Arrival rates:', arrival_rates)
    average_workload = 0
    for tratment_pattern_str, rate in zip(treatment_pattern, arrival_rates):
        ss_rate = slot_session_rate(tratment_pattern_str)
        workload = rate * ss_rate
        average_workload += workload
    print('Average workload:', average_workload)


