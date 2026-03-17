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
        self.zero_penalized_gap = defaultdict(RunningStats)
        self.penalized_gap = defaultdict(RunningStats)
        self.zero_penalized_information_relaxation_cost = defaultdict(RunningStats)
        self.penalized_information_relaxation_cost = defaultdict(RunningStats)
        self.zero_penalized_improvement = defaultdict(RunningStats)
        self.penalized_improvement = defaultdict(RunningStats)
        

        pickle_file = os.path.join(self.directory_path, 'simulate_evaluation_result', 'scenario_results.pickle')
        # Make sure the parent directories exist
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        data = load_pickle_if_exists(pickle_file)
        if data != None and self.is_reuse:
            self.zero_penalized_gap = data['zero_penalized_gap']
            self.penalized_gap = data['penalized_gap']
            self.zero_penalized_information_relaxation_cost = data['zero_penalized_information_relaxation_cost']
            self.penalized_information_relaxation_cost = data['penalized_information_relaxation_cost']
        else:
            pattern = os.path.join(self.directory_path, self.file_pattern)
            jsonl_files = glob.glob(pattern)
            for file_path in jsonl_files:
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            self.load(data)
                        except Exception :
                            print(line)
            # calculate gap
            with open(pickle_file, 'wb') as f:
                res = {
                    'zero_penalized_gap': self.zero_penalized_gap,
                    'penalized_gap': self.penalized_gap,
                    'zero_penalized_information_relaxation_cost': self.zero_penalized_information_relaxation_cost,
                    'penalized_information_relaxation_cost': self.penalized_information_relaxation_cost,
                }
                pickle.dump(res, f)
        for agent_name, stats in self.zero_penalized_information_relaxation_cost.items():
            self.zero_penalized_improvement[agent_name] = self.zero_penalized_gap[agent_name] / stats.mean / 0.01
        for agent_name, stats in self.penalized_information_relaxation_cost.items():
            self.penalized_improvement[agent_name] = self.penalized_gap[agent_name] / stats.mean / 0.01
        print('Zero Penalized Gap, Penalized Gap, Zero Penalized Improvement, Penalized Improvement:')
        pprint(self.zero_penalized_gap)
        pprint(self.penalized_gap)
        pprint(self.zero_penalized_improvement)
        pprint(self.penalized_improvement)
    
    def load(self, data):
        agent_name = json.dumps(data['agent_name'])
        self.zero_penalized_gap[agent_name] += data['total_cost'] - data['information_relaxation_cost_penalty_ratio_0']
        self.penalized_gap[agent_name] += data['penalized_cost'] - data['information_relaxation_cost_penalty_ratio_1']
        self.zero_penalized_information_relaxation_cost[agent_name] += data['information_relaxation_cost_penalty_ratio_0']
        self.penalized_information_relaxation_cost[agent_name] += data['information_relaxation_cost_penalty_ratio_1']


if __name__ == "__main__":
    directory_path = os.path.join('.', 'experiments', 'results', "waiting_penalty_impact")
    # 5: 33.1989634321917 0.13896181129865617
    # 10: 36.687370600414376 0.1486390341192171
    # 20: 39.3842249382221 0.5255526412672854
    file_pattern = '[0-9]*.jsonl'
    ser = SimulateEvaluationResult(directory_path, file_pattern, is_reuse=True)
        