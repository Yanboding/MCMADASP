import glob
import json
import os
from pprint import pprint
import pickle

import numpy as np
from collections import defaultdict

from scipy.stats import geom

from utils import RunningStats, load_pickle_if_exists
from visualization import approximate_value_plot_from_running_stats_dict

# --- 1. Top-level Factory Functions (Required for Pickling) ---

def dd_int_factory():
    """Returns a defaultdict of integers (0)."""
    return defaultdict(int)

def dd_rs_factory():
    """Returns a defaultdict of RunningStats."""
    return defaultdict(RunningStats)

def dd_dd_rs_factory():
    """Returns a 2-level nested defaultdict of RunningStats."""
    return defaultdict(dd_rs_factory)

def dd_float_factory():
    """Returns a defaultdict of floats (0.0)."""
    return defaultdict(float)

def dd_dd_float_factory():
    """Returns a 2-level nested defaultdict of floats (0.0)."""
    return defaultdict(dd_float_factory)

def dd_dd_dd_float_factory():
    """Returns a 3-level nested defaultdict of floats (0.0)."""
    return defaultdict(dd_dd_float_factory)

'''
cumulateive costs: [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
algo: 
1. calculate cumulative costs after warm-up period
'''
class SimulateEvaluationResult:

    def __init__(self,directory_path, file_pattern, is_reuse=False):
        self.directory_path = directory_path
        self.file_pattern = file_pattern
        self.is_reuse = is_reuse
        # Using named functions instead of lambdas
        self.scenario_results = dd_dd_dd_float_factory()
        
        # Built-in types (int, list) and Classes (RunningStats) are already picklable
        self.policy_costs = defaultdict(RunningStats)
        self.zero_penalized_information_relaxation_cost = defaultdict(RunningStats)
        self.penalized_information_relaxation_cost = defaultdict(RunningStats)
        self.zero_penalized_gap = defaultdict(RunningStats)
        self.penalized_gap = defaultdict(RunningStats)
        self.zero_penalized_improvement = defaultdict(RunningStats)
        self.penalized_improvement = defaultdict(RunningStats)
        
        self.gap_to_information_relaxation = defaultdict(RunningStats)
        self.improvement = defaultdict(RunningStats)
        

        pickle_file = os.path.join(self.directory_path, 'simulate_evaluation_result', 'scenario_results.pickle')
        # Make sure the parent directories exist
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        data = load_pickle_if_exists(pickle_file)
        if data != None and self.is_reuse:
            self.policy_costs = data['policy_costs']
            self.zero_penalized_gap = data['zero_penalized_gap']
            self.penalized_gap = data['penalized_gap']
            self.zero_penalized_information_relaxation_cost = data['zero_penalized_information_relaxation_cost']
            self.penalized_information_relaxation_cost = data['penalized_information_relaxation_cost']
            self.gap_to_information_relaxation = data['gap_to_information_relaxation']
        else:
            pattern = os.path.join(self.directory_path, self.file_pattern)
            jsonl_files = glob.glob(pattern)
            for file_path in jsonl_files:
                with open(file_path, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        self.load(data)
            # calculate gap
            with open(pickle_file, 'wb') as f:
                res = {
                    'policy_costs': self.policy_costs,
                    'zero_penalized_gap': self.zero_penalized_gap,
                    'penalized_gap': self.penalized_gap,
                    'zero_penalized_information_relaxation_cost': self.zero_penalized_information_relaxation_cost,
                    'penalized_information_relaxation_cost': self.penalized_information_relaxation_cost,
                    'gap_to_information_relaxation': self.gap_to_information_relaxation,
                    }
                pickle.dump(res, f)
        for (group_id, policy_id), stats in self.zero_penalized_gap.items():
            self.zero_penalized_improvement[(group_id, policy_id)] = self.zero_penalized_gap[(group_id, policy_id)] / self.policy_costs[(group_id, policy_id)].mean / 0.01
        for (group_id, policy_id), stats in self.penalized_gap.items():
            self.penalized_improvement[(group_id, policy_id)] = self.penalized_gap[(group_id, policy_id)] / self.policy_costs[(group_id, policy_id)].mean / 0.01
        for group_id, stats in self.gap_to_information_relaxation.items():
            self.improvement[group_id] = self.gap_to_information_relaxation[group_id] / self.policy_costs[(group_id, policy_id)].mean / 0.01
    
    def load(self, data):
        policy_id = data['policy_id']
        group_id = data['group_id']
        print(f"Loading data for group_id: {group_id}, policy_id: {policy_id}")
        self.policy_costs[(group_id, policy_id)] += data['total_cost']
        if policy_id == 'approx_hindsight':
            self.zero_penalized_information_relaxation_cost[group_id] += data['zero_information_relaxation_cost']
            self.penalized_information_relaxation_cost[group_id] += data['penalized_information_relaxation_cost']
            self.gap_to_information_relaxation[group_id] += data['penalized_information_relaxation_cost'] - data['zero_information_relaxation_cost']
        self.zero_penalized_gap[(group_id, policy_id)] += data['gap_to_zero_information_relaxation']
        self.penalized_gap[(group_id, policy_id)] += data['gap_to_penalized_information_relaxation']


if __name__ == "__main__":
    directory_path = os.path.join('.', 'experiments', 'results', "waiting_penalty_impact")
    # 5: 33.1989634321917 0.13896181129865617
    # 10: 36.687370600414376 0.1486390341192171
    # 20: 39.3842249382221 0.5255526412672854
    file_pattern = '[0-9]*.jsonl'
    ser = SimulateEvaluationResult(directory_path, file_pattern, is_reuse=False)
    print("Policy Costs")
    pprint(ser.policy_costs)
    print("Zero Penalized Gap")
    pprint(ser.zero_penalized_gap)
    print("Penalized Gap")
    pprint(ser.penalized_gap)
    print("Zero Penalized Improvement")
    pprint(ser.zero_penalized_improvement)
    print("Penalized Improvement")
    pprint(ser.penalized_improvement)
    print("Zero Penalized Information Relaxation Cost")
    pprint(ser.zero_penalized_information_relaxation_cost)
    print("Penalized Information Relaxation Cost")
    pprint(ser.penalized_information_relaxation_cost)

    print("Gap to Information Relaxation")
    pprint(ser.gap_to_information_relaxation)
    print("Improvement")
    pprint(ser.improvement)
        