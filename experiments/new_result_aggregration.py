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
from visualization.line_plot import approximate_value_plot_from_running_stats

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


EXPERIMENT_PLOT_CONFIGS = [
    {
        'name': 'initial_state_congestion',
        'scale': 100,
        'xlabel': 'Initial State Congestion (%)',
    },
    {
        'name': 'high_priority_proportion',
        'scale': 100,
        'xlabel': 'High Priority Proportion (%)',
    },
    {
        'name': 'low_priority_waiting_time_target',
        'scale': None,
        'xlabel': 'Low Priority Waiting Time Target (Days)',
    },
    {
        'name': 'high_priority_waiting_time_penalty',
        'scale': None,
        'xlabel': 'High Priority Waiting Time Penalty (Cost)',
    },
    {
        'name': 'total_arrival_rate',
        'scale': None,
        'xlabel': 'Total Arrival Rate (Mean)',
    },
    {
        'name': 'overtime_cost',
        'scale': None,
        'xlabel': 'Overtime Cost (Cost)',
    },
    {
        'name': 'type_1_treatment_pattern',
        'scale': None,
        'xlabel': 'Type 1 Treatment Pattern (Pattern)',
    },
]

IMPROVEMENT_YLABEL = "Lower Bound Improvement (%)"

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
        'overtime_utilization',
        'solving_time_per_state'
    )

    def __init__(self,directory_path, file_pattern, env_info, group_ids=None, is_reuse=False):
        self.directory_path = directory_path
        self.file_pattern = file_pattern
        self.env_info = env_info
        self.group_ids = group_ids if group_ids is not None else []
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
        self.solving_time_per_state = defaultdict(RunningStats)
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
        # for group_id, stats in self.gap_to_information_relaxation.items():
        #     self.improvement[group_id] = self.gap_to_information_relaxation[group_id] / self.policy_costs[(group_id, policy_id)].mean / 0.01
        # # for (group_id, mutate_val), stats in self.zero_penalized_gap.items():
        #     self.zero_penalized_improvement[(group_id, mutate_val)] = self.zero_penalized_gap[(group_id, mutate_val)] / self.zero_penalized_information_relaxation_cost[(group_id, mutate_val)].mean / 0.01
    
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
                    # self.lowerbound_load(json.loads(line))
                    self.load(json.loads(line))

    def _save_cache(self, pickle_file):
        res = {key: getattr(self, key) for key in self._CACHE_KEYS}
        with open(pickle_file, 'wb') as f:
            pickle.dump(res, f)
    
    def lowerbound_load(self, data):
        group_id = data['group_id']
        mutate_val = data['mutate_val']
        self.zero_penalized_information_relaxation_cost[(group_id, mutate_val)] += data['zero_penalized_lower_bound_objective']
        self.penalized_information_relaxation_cost[(group_id, mutate_val)] += data['penalized_lower_bound_objective']
        self.zero_penalized_gap[(group_id, mutate_val)] += data['gap_between_penalized_and_zero']

        
    def load(self, data):
        policy_id = data['policy_id']
        group_id = data['group_id']
        if self.group_ids and group_id not in self.group_ids:
            self.group_ids.append(group_id)
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
        scheduled_patients_ptc_by_day = np.divide(
                                                cum_scheduled_patients * 100,
                                                total_scheduled_patients_by_type,
                                                out=np.zeros_like(cum_scheduled_patients, dtype=float),
                                                where=total_scheduled_patients_by_type != 0
                                            )
        cum_total_scheduled_patients_by_day = scheduled_patients.sum(axis=1).cumsum(axis=0)
        total_scheduled_patients_ptc_by_day = cum_total_scheduled_patients_by_day/total_scheduled_patients * 100 if total_scheduled_patients > 0 else np.zeros_like(cum_total_scheduled_patients_by_day)

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
        
        self.solving_time_per_state[(group_id, policy_id)] += data.get('solving_time_per_state', 0)
            

    def generate_table(self):
        opc_20 = self.group_ids[0]
        opc_50 = self.group_ids[1]
        opc_80 = self.group_ids[2]
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
    
    def plot_percentage_improvement(self, scale, xlabel, ylabel, file_name):
        # Plot percentage improvement of myopic and ALP over the information relaxation benchmark
        plot_stats = {
            (mutate_val * scale if scale is not None else mutate_val): self.zero_penalized_improvement[(group_id, mutate_val)]
            for (group_id, mutate_val) in self.zero_penalized_improvement.keys()
        }
        approximate_value_plot_from_running_stats(running_stats_dict=plot_stats,
                                                  xlabel=xlabel,
                                                  ylabel=ylabel,
                                                title=None,
                                                save_file=os.path.join(self.directory_path,
                                                                        file_name))
    
    def last_decision_period_distribution_table(self):
        group_id ='3ce2a2f68baf077097e28e1f33c60462'
        table = f"""
        \quad $\gamma_q=0.99$ & ${self.after_warmup_policy_costs[(group_id, 'approx_penalized_hindsight_geometric_0_99')].confidence_interval()}$ & ${self.zero_penalized_gap[(group_id, 'approx_penalized_hindsight_geometric_0_99')].confidence_interval()}$ & ${self.zero_penalized_improvement[(group_id, 'approx_penalized_hindsight_geometric_0_99')].confidence_interval()}$ & ${self.penalized_gap[(group_id, 'approx_penalized_hindsight_geometric_0_99')].confidence_interval()}$ & ${self.penalized_improvement[(group_id, 'approx_penalized_hindsight_geometric_0_99')].confidence_interval()}$ \\
        \quad $\gamma_q=0.98$ & ${self.after_warmup_policy_costs[(group_id, 'approx_penalized_hindsight_geometric_0_98')].confidence_interval()}$ & ${self.zero_penalized_gap[(group_id, 'approx_penalized_hindsight_geometric_0_98')].confidence_interval()}$ & ${self.zero_penalized_improvement[(group_id, 'approx_penalized_hindsight_geometric_0_98')].confidence_interval()}$ & ${self.penalized_gap[(group_id, 'approx_penalized_hindsight_geometric_0_98')].confidence_interval()}$ & ${self.penalized_improvement[(group_id, 'approx_penalized_hindsight_geometric_0_98')].confidence_interval()}$ \\
        \quad $\gamma_q=0.96$ & ${self.after_warmup_policy_costs[(group_id, 'approx_penalized_hindsight_geometric_0_96')].confidence_interval()}$ & ${self.zero_penalized_gap[(group_id, 'approx_penalized_hindsight_geometric_0_96')].confidence_interval()}$ & ${self.zero_penalized_improvement[(group_id, 'approx_penalized_hindsight_geometric_0_96')].confidence_interval()}$ & ${self.penalized_gap[(group_id, 'approx_penalized_hindsight_geometric_0_96')].confidence_interval()}$ & ${self.penalized_improvement[(group_id, 'approx_penalized_hindsight_geometric_0_96')].confidence_interval()}$ \\
        \quad $\gamma_q=0.95$ & ${self.after_warmup_policy_costs[(group_id, 'approx_penalized_hindsight_geometric_0_95')].confidence_interval()}$ & ${self.zero_penalized_gap[(group_id, 'approx_penalized_hindsight_geometric_0_95')].confidence_interval()}$ & ${self.zero_penalized_improvement[(group_id, 'approx_penalized_hindsight_geometric_0_95')].confidence_interval()}$ & ${self.penalized_gap[(group_id, 'approx_penalized_hindsight_geometric_0_95')].confidence_interval()}$ & ${self.penalized_improvement[(group_id, 'approx_penalized_hindsight_geometric_0_95')].confidence_interval()}$ \\
        """
        return table
    
    def solver_compare_table(self):
        approx_penalized_hindsight_approx_penalized_hindsight = 'e654df848c9ae807dfbe799f66450025'
        approx_penalized_hindsight_approx_Q = 'bb0efe73303985a5a64575177e78b6dd'
        table = f"""
        Penalized Hindsight & \({self.policy_costs[(approx_penalized_hindsight_approx_penalized_hindsight, 'approx_penalized_hindsight_approx_penalized_hindsight')].confidence_interval()}\) & \({self.zero_penalized_improvement[(approx_penalized_hindsight_approx_penalized_hindsight, 'approx_penalized_hindsight_approx_penalized_hindsight')].confidence_interval()}\) & \({self.penalized_improvement[(approx_penalized_hindsight_approx_penalized_hindsight, 'approx_penalized_hindsight_approx_penalized_hindsight')].confidence_interval()}\) & \({self.solving_time_per_state[(approx_penalized_hindsight_approx_penalized_hindsight, 'approx_penalized_hindsight_approx_penalized_hindsight')].mean}\) \\
        Approximate Q Greedy & \({self.policy_costs[(approx_penalized_hindsight_approx_Q, 'approx_penalized_hindsight_approx_Q')].confidence_interval()}\) & \({self.zero_penalized_improvement[(approx_penalized_hindsight_approx_Q, 'approx_penalized_hindsight_approx_Q')].confidence_interval()}\) & \({self.penalized_improvement[(approx_penalized_hindsight_approx_Q, 'approx_penalized_hindsight_approx_Q')].confidence_interval()}\) & \({self.solving_time_per_state[(approx_penalized_hindsight_approx_Q, 'approx_penalized_hindsight_approx_Q')].mean}\) \\
        ALP & \({self.policy_costs[(approx_penalized_hindsight_approx_penalized_hindsight, 'row_gen_alp')].confidence_interval()}\) & \({self.zero_penalized_improvement[(approx_penalized_hindsight_approx_penalized_hindsight, 'row_gen_alp')].confidence_interval()}\) & \({self.penalized_improvement[(approx_penalized_hindsight_approx_penalized_hindsight, 'row_gen_alp')].confidence_interval()}\) & \({self.solving_time_per_state[(approx_penalized_hindsight_approx_penalized_hindsight, 'row_gen_alp')].mean}\) \\
        """
        return table
    
    def initial_state_congestion_distribution_table(self):
        opc_20 = '347989b0945decac3603070c5485b3f8'
        opc_50 = 'd371745175d939eb1e1cb94afa2d0651'
        opc_80 = '49a2d779c9d5c1c801c5a6f9fa26c0a3'
        table = f"""
        Penalized Hindsight & & & & & & \\\\
        \quad Policy cost & \({self.policy_costs[(opc_20, 'approx_penalized_hindsight')].confidence_interval()}\) & & \({self.policy_costs[(opc_50, 'approx_penalized_hindsight')].confidence_interval()}\) & &\({self.policy_costs[(opc_80, 'approx_penalized_hindsight')].confidence_interval()}\) & \\\\
        \quad Zero penalty gap & \({self.zero_penalized_gap[(opc_20, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_20, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_50, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_50, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_80, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_80, 'approx_penalized_hindsight')].confidence_interval()}\)\\\\
        \quad Max penalty gap & \({self.penalized_gap[(opc_20, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.penalized_improvement[(opc_20, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.penalized_gap[(opc_50, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.penalized_improvement[(opc_50, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.penalized_gap[(opc_80, 'approx_penalized_hindsight')].confidence_interval()}\) & \({self.penalized_improvement[(opc_80, 'approx_penalized_hindsight')].confidence_interval()}\)\\\\
        Approximate Q Greedy & & & & & & \\\\
        \quad Policy cost & \({self.policy_costs[(opc_20, 'approx_Q')].confidence_interval()}\) & & \({self.policy_costs[(opc_50, 'approx_Q')].confidence_interval()}\) & &\({self.policy_costs[(opc_80, 'approx_Q')].confidence_interval()}\) & \\\\
        \quad Zero penalty gap & \({self.zero_penalized_gap[(opc_20, 'approx_Q')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_20, 'approx_Q')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_50, 'approx_Q')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_50, 'approx_Q')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_80, 'approx_Q')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_80, 'approx_Q')].confidence_interval()}\)\\\\
        \quad Max penalty gap & \({self.penalized_gap[(opc_20, 'approx_Q')].confidence_interval()}\) & \({self.penalized_improvement[(opc_20, 'approx_Q')].confidence_interval()}\) & \({self.penalized_gap[(opc_50, 'approx_Q')].confidence_interval()}\) & \({self.penalized_improvement[(opc_50, 'approx_Q')].confidence_interval()}\) & \({self.penalized_gap[(opc_80, 'approx_Q')].confidence_interval()}\) & \({self.penalized_improvement[(opc_80, 'approx_Q')].confidence_interval()}\)\\\\
        ALP & & & & & & \\\\
        \quad Policy cost & \({self.policy_costs[(opc_20, 'row_gen_alp')].confidence_interval()}\) & & \({self.policy_costs[(opc_50, 'row_gen_alp')].confidence_interval()}\) & &\({self.policy_costs[(opc_80, 'row_gen_alp')].confidence_interval()}\) & \\\\
        \quad Zero penalty gap & \({self.zero_penalized_gap[(opc_20, 'row_gen_alp')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_20, 'row_gen_alp')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_50, 'row_gen_alp')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_50, 'row_gen_alp')].confidence_interval()}\) & \({self.zero_penalized_gap[(opc_80, 'row_gen_alp')].confidence_interval()}\) & \({self.zero_penalized_improvement[(opc_80, 'row_gen_alp')].confidence_interval()}\)\\\\
        \quad Max penalty gap & \({self.penalized_gap[(opc_20, 'row_gen_alp')].confidence_interval()}\) & \({self.penalized_improvement[(opc_20, 'row_gen_alp')].confidence_interval()}\) & \({self.penalized_gap[(opc_50, 'row_gen_alp')].confidence_interval()}\) & \({self.penalized_improvement[(opc_50, 'row_gen_alp')].confidence_interval()}\) & \({self.penalized_gap[(opc_80, 'row_gen_alp')].confidence_interval()}\) & \({self.penalized_improvement[(opc_80, 'row_gen_alp')].confidence_interval()}\)\\\\"""
        return table


def run_improvement_plots(base_results_dir, file_pattern, env_info, group_ids, is_reuse=False):
    for config in EXPERIMENT_PLOT_CONFIGS:
        directory_path = os.path.join(base_results_dir, config['name'])
        ser = SimulateEvaluationResult(
            directory_path,
            file_pattern,
            env_info,
            group_ids=group_ids,
            is_reuse=is_reuse,
        )
        ser.plot_percentage_improvement(
            scale=config['scale'],
            xlabel=config['xlabel'],
            ylabel=IMPROVEMENT_YLABEL,
            file_name=f"{config['name']}_lower_bound_improvement.svg",
        )

if __name__ == "__main__":

    base_results_dir = os.path.join('.', 'experiments', 'results', 'initial_state_congestion')
    file_pattern = '[0-9]*.jsonl'
    env_info = {
        'waiting_time_targets': [1]*2,
        'overtime_capacity': 5,
    }
    # group_ids = ['73d11360affe39305e7716cf5c42ac04', '841708e72300000ddfd948daf08d6805', 'a3202d39ed34711b47ecebb72aabad43']
    # run_improvement_plots(
    #     base_results_dir=base_results_dir,
    #     file_pattern=file_pattern,
    #     env_info=env_info,
    #     group_ids=group_ids,
    #     is_reuse=False,
    # )
    ser = SimulateEvaluationResult(
            base_results_dir,
            file_pattern,
            env_info,
            is_reuse=True,
        )
    
    # print(ser.last_decision_period_distribution_table())
    print("Gap to Information Relaxation")
    pprint(ser.gap_to_information_relaxation)
    # print("Improvement")
    # pprint(ser.improvement)

    # print('waiting_time_target_ptc_by_day_type')
    # pprint(ser.waiting_time_target_ptc_by_type_day)
    # print('waiting_time_target_ptc_by_day')
    # pprint(ser.waiting_time_target_ptc_by_day)

    # ser.generate_table()
    print('self.policy_costs')
    print(ser.policy_costs)
    print('penalized_improvement')
    print(ser.penalized_improvement)
    print('zero_improvement')
    print(ser.zero_penalized_improvement)
    print('Solving time per state')
    print(ser.solving_time_per_state)
    print(ser.group_ids)
    print(ser.initial_state_congestion_distribution_table())

    # To inspect one experiment interactively, instantiate SimulateEvaluationResult
    # with a specific directory and use the helper methods below.

    # ser.format_table()
    # pprint(ser.overtime_utilization)
    # pprint(ser.waiting_time_violation)
    # print(ser.summary_table())
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
    print(geom.ppf(0.985, 0.01))
        