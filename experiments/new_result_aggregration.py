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

        self.waiting_time_target_ptc_by_day = defaultdict(dd_rs_factory)
        
        # Triple-nested structure
        self.waiting_time_target_ptc_by_day_type = defaultdict(dd_dd_rs_factory)
        
        self.one_time_cost_by_policy = dd_dd_rs_factory()
        self.number_of_periods = None

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
            self.one_time_cost_by_policy = data['one_time_cost_by_policy']
            self.waiting_time_target_ptc_by_day = data['waiting_time_target_ptc_by_day']
            self.waiting_time_target_ptc_by_day_type = data['waiting_time_target_ptc_by_day_type']
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
                    'one_time_cost_by_policy': self.one_time_cost_by_policy,
                    'waiting_time_target_ptc_by_day': self.waiting_time_target_ptc_by_day,
                    'waiting_time_target_ptc_by_day_type': self.waiting_time_target_ptc_by_day_type,
                    }
                pickle.dump(res, f)
        '''
        for (group_id, policy_id), stats in self.zero_penalized_gap.items():
            self.zero_penalized_improvement[(group_id, policy_id)] = self.zero_penalized_gap[(group_id, policy_id)] / self.penalized_information_relaxation_cost[(group_id)].mean / 0.01
        for (group_id, policy_id), stats in self.penalized_gap.items():
            self.penalized_improvement[(group_id, policy_id)] = self.penalized_gap[(group_id, policy_id)] / self.penalized_information_relaxation_cost[(group_id)].mean / 0.01
        for group_id, stats in self.gap_to_information_relaxation.items():
            self.improvement[group_id] = self.gap_to_information_relaxation[group_id] / self.policy_costs[(group_id, policy_id)].mean / 0.01
        '''
    def load(self, data):
        policy_id = data['policy_id']
        group_id = data['group_id']
        self.policy_costs[(group_id, policy_id)] += data['total_cost']
        if policy_id == 'approx_hindsight':
            self.zero_penalized_information_relaxation_cost[group_id] += data['zero_information_relaxation_cost']
            self.penalized_information_relaxation_cost[group_id] += data['penalized_information_relaxation_cost']
            self.gap_to_information_relaxation[group_id] += data['penalized_information_relaxation_cost'] - data['zero_information_relaxation_cost']
        self.zero_penalized_gap[(group_id, policy_id)] += data['gap_to_zero_information_relaxation']
        self.penalized_gap[(group_id, policy_id)] += data['gap_to_penalized_information_relaxation']
        if self.number_of_periods is None:
            self.number_of_periods = len(data['costs'])
        for t, cost in enumerate(data['costs']):
            self.one_time_cost_by_policy[policy_id][t] += cost
        
        warm_up_periods = data["warm_up_periods"]
        
        scheduled_patients = np.array(data["scheduled_patients"])[warm_up_periods:].sum(axis=0) if len(data["scheduled_patients"]) > warm_up_periods else np.array(data["scheduled_patients"]).sum(axis=0)
        
        total_scheduled_patients = scheduled_patients.sum()

        cum_scheduled_patients = scheduled_patients.cumsum(axis=0)
        total_scheduled_patients_by_type = scheduled_patients.sum(axis=0)
        scheduled_patients_ptc_by_day = cum_scheduled_patients/total_scheduled_patients_by_type * 100
        cum_total_scheduled_patients_by_day = scheduled_patients.sum(axis=1).cumsum(axis=0)
        total_scheduled_patients_ptc_by_day = cum_total_scheduled_patients_by_day/total_scheduled_patients * 100

        for day in range(len(scheduled_patients)):
            for treatment_type in range(len(scheduled_patients[day])):
                self.waiting_time_target_ptc_by_day_type[policy_id][treatment_type][day] += scheduled_patients_ptc_by_day[day][treatment_type]
                self.waiting_time_target_ptc_by_day[policy_id][day] += total_scheduled_patients_ptc_by_day[day]

    def generate_table(self):
        opc_20 = 'acbffa87277103d172340d09fb3d6714'
        opc_50 = 'b0b4f1c19b307d6a79f44e80a39b44cf'
        opc_80 = 'b25851a57c0a8ed02e9956116cf3c659'
        table = f"""
        Hindsight & & & & & & \\\\ 
        \quad Policy cost & \({self.policy_costs[(opc_20, 'approx_hindsight')].confidence_interval()}\) & & \({self.policy_costs[(opc_50, 'approx_hindsight')].confidence_interval()}\) & &\({self.policy_costs[(opc_80, 'approx_hindsight')].confidence_interval()}\) & \\\\
        \quad Zero penalty gap & \({self.zero_penalized_gap[(opc_20, 'approx_hindsight')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_20, 'approx_hindsight')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_50, 'approx_hindsight')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_50, 'approx_hindsight')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_80, 'approx_hindsight')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_80, 'approx_hindsight')].confidence_interval()}\)\\\\
        \quad Max penalty gap & \({self.penalized_gap[(opc_20, 'approx_hindsight')].confidence_interval()}\) & \({self.penalized_improvement[(opc_20, 'approx_hindsight')].confidence_interval()}\) & \({self.penalized_gap[(opc_50, 'approx_hindsight')].confidence_interval()}\) & \({self.penalized_improvement[(opc_50, 'approx_hindsight')].confidence_interval()}\) & \({self.penalized_gap[(opc_80, 'approx_hindsight')].confidence_interval()}\) & \({self.penalized_improvement[(opc_80, 'approx_hindsight')].confidence_interval()}\) \\\\
        Penalized Hindsight & & & & & & \\\\
        \quad Policy cost & \({self.policy_costs[(opc_20, 'approx_penalized_hindsight')].confidence_interval()}\) & & \({self.policy_costs[(opc_50, 'approx_penalized_hindsight')].confidence_interval()}\) & &\({self.policy_costs[(opc_80, 'approx_penalized_hindsight')].confidence_interval()}\) & \\\\
        \quad Zero penalty gap & \({self.zero_penalized_gap[(opc_20, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_20, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_50, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_50, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_80, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_80, 'approx_penalized_hindsight')].confidence_interval()}\)\\\\
        \quad Max penalty gap & \({self.penalized_gap[(opc_20, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.penalized_improvement[(opc_20, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.penalized_gap[(opc_50, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.penalized_improvement[(opc_50, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.penalized_gap[(opc_80, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.penalized_improvement[(opc_80, 'approx_penalized_hindsight')].confidence_interval()}\)\\\\
        Myopic & & & & & & \\\\
        \quad Policy cost & \({self.policy_costs[(opc_20, 'myopic')].confidence_interval()}\) & & \({self.policy_costs[(opc_50, 'myopic')].confidence_interval()}\) & &\({self.policy_costs[(opc_80, 'myopic')].confidence_interval()}\) & \\\\
        \quad Zero penalty gap & \({self.zero_penalized_gap[(opc_20, 'myopic')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_20, 'myopic')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_50, 'myopic')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_50, 'myopic')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_80, 'myopic')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_80, 'myopic')].confidence_interval()}\)\\\\
        \quad Max penalty gap & \({self.penalized_gap[(opc_20, 'myopic')].confidence_interval()}\) & \({self.penalized_improvement[(opc_20, 'myopic')].confidence_interval()}\) & \({self.penalized_gap[(opc_50, 'myopic')].confidence_interval()}\) & \({self.penalized_improvement[(opc_50, 'myopic')].confidence_interval()}\) & \({self.penalized_gap[(opc_80, 'myopic')].confidence_interval()}\) & \({self.penalized_improvement[(opc_80, 'myopic')].confidence_interval()}\)\\\\
        ALP & & & & & & \\\\
        \quad Policy cost & \({self.policy_costs[(opc_20, 'row_gen_alp')].confidence_interval()}\) & & \({self.policy_costs[(opc_50, 'row_gen_alp')].confidence_interval()}\) & &\({self.policy_costs[(opc_80, 'row_gen_alp')].confidence_interval()}\) & \\\\
        \quad Zero penalty gap & \({self.zero_penalized_gap[(opc_20, 'row_gen_alp')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_20, 'row_gen_alp')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_50, 'row_gen_alp')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_50, 'row_gen_alp')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_80, 'row_gen_alp')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_80, 'row_gen_alp')].confidence_interval()}\)\\\\
        \quad Max penalty gap & \({self.penalized_gap[(opc_20, 'row_gen_alp')].confidence_interval()}\) & \({self.penalized_improvement[(opc_20, 'row_gen_alp')].confidence_interval()}\) & \({self.penalized_gap[(opc_50, 'row_gen_alp')].confidence_interval()}\) & \({self.penalized_improvement[(opc_50, 'row_gen_alp')].confidence_interval()}\) & \({self.penalized_gap[(opc_80, 'row_gen_alp')].confidence_interval()}\) & \({self.penalized_improvement[(opc_80, 'row_gen_alp')].confidence_interval()}\)\\\\"""
        print(table)
    
    def format_table(self):
        kys = self.waiting_time_target_ptc_by_day_type.keys()
        experiment_labels = 'ALP'
        for agent_name in kys:
            table = defaultdict(list)
            table2 = {}
            for day in [1, 5, 10, 15, 20]:
                total_mean = self.waiting_time_target_ptc_by_day[agent_name][day].mean
                total_hw = self.waiting_time_target_ptc_by_day[agent_name][day].half_window(0.95)
                table2[day] = (round(total_mean, 2), round(total_hw, 3))
                for type in range(len(self.waiting_time_target_ptc_by_day_type[agent_name])):
                    mean = self.waiting_time_target_ptc_by_day_type[agent_name][type][day].mean
                    hw = self.waiting_time_target_ptc_by_day_type[agent_name][type][day].half_window(0.95)
                    table[type+1].append((day, round(mean), round(hw)))
            print(table2)
            print('ALP')
            for type in sorted(table.keys()):
                line = f'{type} '
                for (day, ptc, hw) in table[type]:
                    line += f' & {ptc} $\pm$ {hw}'
                line += r' \\'
                print(line)
            line = r'\textbf{Total} '
            for day, (ptc, hw) in table2.items():
                line += f' & {ptc} $\pm$ {hw}'
            line += r' \\'
            print(line)

if __name__ == "__main__":
    directory_path = os.path.join('.', 'experiments', 'results', "case_study")
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

    print('waiting_time_target_ptc_by_day_type')
    pprint(ser.waiting_time_target_ptc_by_day_type)
    print('waiting_time_target_ptc_by_day')
    pprint(ser.waiting_time_target_ptc_by_day)

    ser.format_table()

    # ser.generate_table()
    # print(ser.one_time_cost_by_policy)
    # approximate_value_plot_from_running_stats_dict(running_stats_dict=ser.one_time_cost_by_policy,
    #                                                x_vals=None,
    #                                                xticks=None,
    #                                                xticklabels=None,
    #                                                xlabel='Time step',
    #                                                ylabel="One-time cost",
    #                                                plot_labels={'approx_hindsight': "Hindsight", 'approx_penalized_hindsight': "Penalized Hindsight", 'myopic': "Myopic", 'row_gen_alp': "ALP"},
    #                                                title=None,
    #                                                save_file='one_time_cost_by_policy.svg',
    #                                                is_show_text=False,
    #                                                is_set_x_color=True)
        