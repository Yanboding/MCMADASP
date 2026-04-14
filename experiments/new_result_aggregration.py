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

    _CACHE_KEYS = (
        'policy_costs',
        'zero_penalized_gap',
        'penalized_gap',
        'zero_penalized_information_relaxation_cost',
        'penalized_information_relaxation_cost',
        'gap_to_information_relaxation',
        'one_time_cost_by_policy',
        'after_warmup_policy_costs',
        'waiting_time_target_ptc_by_day',
        'waiting_time_target_ptc_by_type_day',
        'waiting_time_violation',
        'overtime_utilization'
    )

    def __init__(self,directory_path, file_pattern, env_info, is_reuse=False):
        self.directory_path = directory_path
        self.file_pattern = file_pattern
        self.env_info = env_info
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

        # Discountred total cost gap after warmup period (if the information relaxation includes the discount factor, then we should use discounted gap.)
        self.after_warmup_policy_costs = defaultdict(RunningStats)
        # Waiting time target violation
        self.waiting_time_target_ptc_by_type_day = defaultdict(dd_dd_rs_factory)
        self.waiting_time_target_ptc_by_day = defaultdict(dd_rs_factory)
        self.waiting_time_violation = defaultdict(RunningStats)
        
        # Average ovetime utilization per day
        self.overtime_utilization = defaultdict(RunningStats)
        # Average postponement rate
        self.postponement_rate = defaultdict(RunningStats)
        
        self.gap_to_information_relaxation = defaultdict(RunningStats)
        self.improvement = defaultdict(RunningStats)

        self.one_time_cost_by_policy = dd_dd_rs_factory()
        self.number_of_periods = None
        self.information_relaxation_id = None

        pickle_file = os.path.join(self.directory_path, 'simulate_evaluation_result', 'scenario_results.pickle')
        # Make sure the parent directories exist
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        cached_data = load_pickle_if_exists(pickle_file)
        if self.is_reuse and self._has_valid_cache(cached_data):
            self._load_from_cache(cached_data)
        else:
            self._load_from_jsonl()
            self._save_cache(pickle_file)
        for (group_id, policy_id), stats in self.zero_penalized_gap.items():
            self.zero_penalized_improvement[(group_id, policy_id)] = self.zero_penalized_gap[(group_id, policy_id)] / self.penalized_information_relaxation_cost[(group_id)].mean / 0.01
        for (group_id, policy_id), stats in self.penalized_gap.items():
            self.penalized_improvement[(group_id, policy_id)] = self.penalized_gap[(group_id, policy_id)] / self.penalized_information_relaxation_cost[(group_id)].mean / 0.01
        for group_id, stats in self.gap_to_information_relaxation.items():
            self.improvement[group_id] = self.gap_to_information_relaxation[group_id] / self.policy_costs[(group_id, policy_id)].mean / 0.01

    def _has_valid_cache(self, data):
        if data is None:
            return False
        return all(key in data for key in self._CACHE_KEYS)

    def _load_from_cache(self, data):
        for key in self._CACHE_KEYS:
            setattr(self, key, data[key])

    def _load_from_jsonl(self):
        pattern = os.path.join(self.directory_path, self.file_pattern)
        jsonl_files = sorted(glob.glob(pattern))
        for file_path in jsonl_files:
            with open(file_path, 'r') as f:
                for line in f:
                    self.load(json.loads(line))

    def _save_cache(self, pickle_file):
        res = {key: getattr(self, key) for key in self._CACHE_KEYS}
        with open(pickle_file, 'wb') as f:
            pickle.dump(res, f)
        
    def load(self, data):
        policy_id = data['policy_id']
        group_id = data['group_id']
        self.policy_costs[(group_id, policy_id)] += data['total_cost']
        # use the first loaded policy as the information relaxation benchmark
        if self.information_relaxation_id is None:
            self.information_relaxation_id = policy_id
        if policy_id == self.information_relaxation_id:
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
        # becarful abount the warm-up period.
        costs_after_warmup = data['costs'][warm_up_periods:] if len(data['costs']) > warm_up_periods else data['costs']

        self.after_warmup_policy_costs[(group_id, policy_id)] += sum(cost * (0.99 ** t) for t, cost in enumerate(costs_after_warmup))
        
        scheduled_patients = np.array(data["scheduled_patients"])[warm_up_periods:].sum(axis=0) if len(data["scheduled_patients"]) > warm_up_periods else np.array(data["scheduled_patients"]).sum(axis=0)
        
        total_scheduled_patients = scheduled_patients.sum()

        cum_scheduled_patients = scheduled_patients.cumsum(axis=0)
        total_scheduled_patients_by_type = scheduled_patients.sum(axis=0)
        scheduled_patients_ptc_by_day = cum_scheduled_patients/total_scheduled_patients_by_type * 100
        cum_total_scheduled_patients_by_day = scheduled_patients.sum(axis=1).cumsum(axis=0)
        total_scheduled_patients_ptc_by_day = cum_total_scheduled_patients_by_day/total_scheduled_patients * 100

        for day in range(len(scheduled_patients)):
            for treatment_type in range(len(scheduled_patients[day])):
                self.waiting_time_target_ptc_by_type_day[policy_id][treatment_type][day] += scheduled_patients_ptc_by_day[day][treatment_type]
                self.waiting_time_target_ptc_by_day[policy_id][day] += total_scheduled_patients_ptc_by_day[day]
        
        for day in range(len(data["overtime"])):
            self.overtime_utilization[(group_id, policy_id)] += data["overtime"][day] / self.env_info['overtime_capacity'] * 100
        # calculate the waiting time violation rate
        # scheduled_patients is a 2D array of shape (num_days, num_types), where each entry represents the number of patients of a certain type scheduled on a certain day. We need to calculate the percentage of patients that are scheduled outside of their waiting time target. For each treatment type, we have a waiting time target (e.g., 1 day, 5 days, etc.). We can calculate the cumulative percentage of patients scheduled by each day and compare it to the waiting time target to determine the violation rate.
        
        waiting_time_targets = self.env_info.get('waiting_time_targets', [])
        num_types = len(waiting_time_targets)
        patients_outside_target = sum(
            total_scheduled_patients_by_type[t] - cum_scheduled_patients[waiting_time_targets[t] - 1][t]
            for t in range(num_types)
            if total_scheduled_patients_by_type[t] > 0
        )
        if total_scheduled_patients > 0:
            self.waiting_time_violation[(group_id, policy_id)] += patients_outside_target / total_scheduled_patients * 100
            

    def generate_table(self):
        opc_20 = 'd9b05dbc43a20cbb3bdcb288a172b634'
        opc_50 = '8d8f27dc138e77346d4d15c71219a2cf'
        opc_80 = 'e2899a935c322cd59ea015f835bb9498'
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
        kys = self.waiting_time_target_ptc_by_type_day.keys()
        experiment_labels = 'ALP'
        for agent_name in kys:
            table = defaultdict(list)
            table2 = {}
            for day in [1, 5, 10, 15, 20]:
                total_mean = self.waiting_time_target_ptc_by_day[agent_name][day].mean
                total_hw = self.waiting_time_target_ptc_by_day[agent_name][day].half_window(0.95)
                table2[day] = (round(total_mean, 2), round(total_hw, 3))
                for type in range(len(self.waiting_time_target_ptc_by_type_day[agent_name])):
                    mean = self.waiting_time_target_ptc_by_type_day[agent_name][type][day].mean
                    hw = self.waiting_time_target_ptc_by_type_day[agent_name][type][day].half_window(0.95)
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
                line += f' & {round(ptc)} $\pm$ {round(hw)}'
            line += r' \\'
            print(line)
    
    def summary_table(self):
        group_id = 'f4e3ac161730cb1a133fc82f962b8c4f'
        table = f"""
        Myopic & ${self.after_warmup_policy_costs[(group_id, 'myopic')].confidence_interval()}$ & ${self.waiting_time_violation[(group_id, 'myopic')].confidence_interval()}$ & ${self.overtime_utilization[(group_id, 'myopic')].confidence_interval()}$ \\\\
        ALP & ${self.after_warmup_policy_costs[(group_id, 'row_gen_alp')].confidence_interval()}$ & ${self.waiting_time_violation[(group_id, 'row_gen_alp')].confidence_interval()}$ & ${self.overtime_utilization[(group_id, 'row_gen_alp')].confidence_interval()}$ \\\\
        """
        return table

if __name__ == "__main__":
    directory_path = os.path.join('.', 'experiments', 'results', "case_study")
    file_pattern = '[0-9]*.jsonl'
    env_info = {
        'waiting_time_targets': [1]*3 + [10]*3 + [5]*8 + [10]*4,
        'overtime_capacity': 15,
    }
    ser = SimulateEvaluationResult(directory_path, file_pattern, env_info, is_reuse=False)

    # print("Gap to Information Relaxation")
    # pprint(ser.gap_to_information_relaxation)
    # print("Improvement")
    # pprint(ser.improvement)

    # print('waiting_time_target_ptc_by_day_type')
    # pprint(ser.waiting_time_target_ptc_by_day_type)
    # print('waiting_time_target_ptc_by_day')
    # pprint(ser.waiting_time_target_ptc_by_day)

    # ser.generate_table()
    # print('self.after_warmup_policy_costs')
    # print(ser.after_warmup_policy_costs)

    pprint(ser.waiting_time_target_ptc_by_type_day)

    ser.format_table()
    pprint(ser.overtime_utilization)
    pprint(ser.waiting_time_violation)
    print(ser.summary_table())
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
        