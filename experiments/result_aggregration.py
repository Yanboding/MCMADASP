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
        self.total_cost = defaultdict(RunningStats)
        self.total_discounted_cost = defaultdict(RunningStats)
        self.total_cost_after_warmup = defaultdict(RunningStats)
        self.total_discounted_cost_after_warmup = defaultdict(RunningStats)
        self.gap_after_warmup = defaultdict(RunningStats)
        self.ptc_gap_after_warmup = defaultdict(RunningStats)

        self.total_penalty = defaultdict(RunningStats)
        self.total_discounted_penalty = defaultdict(RunningStats)
        self.total_penalty_after_warmup = defaultdict(RunningStats)
        self.total_discounted_penalty_after_warmup = defaultdict(RunningStats)
        
        # Nested structures
        self.cost_by_day = defaultdict(dd_rs_factory)
        self.cumulative_cost_by_day = defaultdict(dd_rs_factory)

        self.average_waiting_time = defaultdict(RunningStats)
        self.average_overtime = defaultdict(RunningStats)
        self.average_postponing_decision = defaultdict(RunningStats)

        self.waiting_time_by_type = defaultdict(dd_rs_factory)
        self.waiting_time_target_ptc_by_day = defaultdict(dd_rs_factory)
        
        # Triple-nested structure
        self.waiting_time_target_ptc_by_day_type = defaultdict(dd_dd_rs_factory)

        self.overtime_by_day = defaultdict(dd_rs_factory)
        self.overtime_ptc_by_day = defaultdict(dd_rs_factory)
        self.total_overtime_used_ptc = defaultdict(RunningStats)

        self.postponing_decision_by_type = defaultdict(dd_rs_factory)
        self.experiment_labels = {
            'myopic': 'Myopic Policy',
            'row_gen_alp': 'ALP',
            'hindsight_approx': 'Hindsight Approximation',
            'lowerbound': 'Lower Bound',
            'hindsight_approx_MC': 'Hindsight Approximation via MC',
            'hindsight_approx_QMC': 'Hindsight Approximation via QMC',
            'lowerbound_hindsight_approx_MC': 'Lower Bound',
            'lowerbound_hindsight_approx_QMC': 'QMC Lower Bound',
            # 'penalized_lowerbound': 'Penalized Lower Bound',
            'penalized_lowerbound_hindsight_approx_MC': 'Penalized Lower Bound',
            'penalized_lowerbound_hindsight_approx_QMC': 'QMC Penalized Lower Bound',
        }
        self.lowerbound_labels = {
            'penalized_lowerbound_hindsight_approx': 'Hindsight Policy vs. Penalized Lower Bound',
            'lowerbound_hindsight_approx': 'Hindsight Policy vs. Lower Bound',
            'penalized_lowerbound_row_gen_alp': 'ALP vs. Penalized Lower Bound'
        }
        pickle_file = os.path.join(self.directory_path, 'simulate_evaluation_result', 'scenario_results.pickle')
        # Make sure the parent directories exist
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        data = load_pickle_if_exists(pickle_file)
        if data != None and self.is_reuse:
            self.scenario_results = data['scenario_results']
            self.total_cost = data['total_cost']
            self.total_discounted_cost = data['total_discounted_cost']
            self.total_cost_after_warmup = data['total_cost_after_warmup']
            self.total_discounted_cost_after_warmup = data['total_discounted_cost_after_warmup']
            self.gap_after_warmup = data['gap_after_warmup']
            self.ptc_gap_after_warmup = data['ptc_gap_after_warmup']

            self.total_penalty = data['total_penalty']
            self.total_discounted_penalty = data['total_discounted_penalty']
            self.total_penalty_after_warmup = data['total_penalty_after_warmup']
            self.total_discounted_penalty_after_warmup = data['total_discounted_penalty_after_warmup']
            self.cost_by_day = data['cost_by_day']
            self.cumulative_cost_by_day = data['cumulative_cost_by_day']

            self.average_waiting_time = data['average_waiting_time']
            self.average_overtime = data['average_overtime']
            self.average_postponing_decision = data['average_postponing_decision']

            self.waiting_time_by_type = data['waiting_time_by_type']
            self.waiting_time_target_ptc_by_day = data['waiting_time_target_ptc_by_day']
            self.waiting_time_target_ptc_by_day_type = data['waiting_time_target_ptc_by_day_type']

            self.overtime_by_day = data['overtime_by_day']
            self.overtime_ptc_by_day = data['overtime_ptc_by_day']
            self.total_overtime_used_ptc = data['total_overtime_used_ptc']

            self.postponing_decision_by_type = data['postponing_decision_by_type']
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
                    'scenario_results': self.scenario_results,
                    'total_cost': self.total_cost,
                    'total_discounted_cost': self.total_discounted_cost,
                    'total_cost_after_warmup': self.total_cost_after_warmup,
                    'total_discounted_cost_after_warmup': self.total_discounted_cost_after_warmup,
                    'gap_after_warmup': self.gap_after_warmup,
                    'ptc_gap_after_warmup': self.ptc_gap_after_warmup,

                    'total_penalty': self.total_penalty,
                    'total_discounted_penalty': self.total_discounted_penalty,
                    'total_penalty_after_warmup': self.total_penalty_after_warmup,
                    'total_discounted_penalty_after_warmup': self.total_discounted_penalty_after_warmup,
                    'cost_by_day': self.cost_by_day,
                    'cumulative_cost_by_day': self.cumulative_cost_by_day,

                    'average_waiting_time': self.average_waiting_time,
                    'average_overtime': self.average_overtime,
                    'average_postponing_decision': self.average_postponing_decision,

                    'waiting_time_by_type': self.waiting_time_by_type,
                    'waiting_time_target_ptc_by_day': self.waiting_time_target_ptc_by_day,
                    'waiting_time_target_ptc_by_day_type': self.waiting_time_target_ptc_by_day_type,

                    'overtime_by_day': self.overtime_by_day,
                    'overtime_ptc_by_day': self.overtime_ptc_by_day,
                    'total_overtime_used_ptc': self.total_overtime_used_ptc,

                    'postponing_decision_by_type': self.postponing_decision_by_type
                }
                pickle.dump(res, f)
        self.calculate_gaps()
        #print(self.scenario_results[uid][('penalized_lowerbound_', '{"coefficients": 1, "current_decision_var_type": "integer", "future_decision_var_type": "continuous", "is_include_discount_factor": false, "is_myopic": false}')])
        print(self.gap_after_warmup)

    def load(self, data):
        agent_name = json.dumps(data['agent_name'])
        for day, cost in enumerate(data['costs']):
            self.cost_by_day[agent_name][day] += cost
        warm_up_periods = data["warm_up_periods"]
        costs = data['costs'][warm_up_periods:]
        penalties = data['penalties'] if len(data['penalties']) > 0 else []
        cumulative_costs = np.cumsum(data['costs'][warm_up_periods:])
        for day, cumulative_cost in enumerate(cumulative_costs):
            self.cumulative_cost_by_day[agent_name][day] += cumulative_cost
        penalties_sum = sum(penalties)
        scheduled_patients = np.array(data["scheduled_patients"])[warm_up_periods:].sum(axis=0) if len(data["scheduled_patients"]) > warm_up_periods else np.array(data["scheduled_patients"]).sum(axis=0)
        cum_scheduled_patients = scheduled_patients.cumsum(axis=0)
        total_scheduled_patients_by_type = scheduled_patients.sum(axis=0)
        scheduled_patients_ptc_by_day = cum_scheduled_patients/total_scheduled_patients_by_type * 100
        total_scheduled_patients = scheduled_patients.sum()
        cum_total_scheduled_patients_by_day = scheduled_patients.sum(axis=1).cumsum(axis=0)
        total_scheuled_patients_ptc_by_day = cum_total_scheduled_patients_by_day/total_scheduled_patients * 100
        total_cost_sum = sum(costs)
        penalized_total_cost = total_cost_sum + penalties_sum
        self.total_cost_after_warmup[agent_name] += penalized_total_cost
        self.total_penalty[agent_name] += penalties_sum
        for day in range(len(scheduled_patients)):
            for type in range(len(scheduled_patients[day])):
                self.average_waiting_time[agent_name].record_batch(day, scheduled_patients[day][type])
                self.waiting_time_by_type[agent_name][type].record_batch(day, scheduled_patients[day][type])
                self.waiting_time_target_ptc_by_day_type[agent_name][type][day] += scheduled_patients_ptc_by_day[day][type]
                self.waiting_time_target_ptc_by_day[agent_name][day] += total_scheuled_patients_ptc_by_day[day]


        overtimes = data['overtime'][warm_up_periods:] if len(data["overtime"]) > warm_up_periods else data["overtime"]
        for day, overtime in enumerate(overtimes):
            self.average_overtime[agent_name] += overtime
            self.overtime_by_day[agent_name][day] += overtime
        '''
        postponing_decisions = data['postponing_decision'][warm_up_periods:]
        for day in range(len(scheduled_patients)):
            for type in range(len(scheduled_patients[day])):
                self.average_postponing_decision[agent_name] += postponing_decisions[day][type]
                self.postponing_decision_by_type[agent_name][type] += postponing_decisions[day][type]
        '''
        '''
        scenario_results = {uid: {policy:{(agent_name, args_id):cost}, (lowerbound, args_id):{(agent_name, args_id):cost}, (penalized_lowerbound, args_id):{(agent_name, args_id):cost}}}
        '''
        name, args, lowerbound_args = data['agent_name']['agent_name'], json.dumps(data['agent_name'].get('args', {}), sort_keys=True), json.dumps(data['agent_name'].get('lowerbound_args', {}), sort_keys=True)

        if name.startswith("penalized_lowerbound_"):
            base_name = name.replace("penalized_lowerbound_", "")
            self.scenario_results[data['uid']][("penalized_lowerbound_", lowerbound_args)][(base_name, args)] = penalized_total_cost
        elif name.startswith("lowerbound_"):
            base_name = name.replace("lowerbound_", "")
            self.scenario_results[data['uid']][("lowerbound_", lowerbound_args)][(base_name, args)] = penalized_total_cost
        else:
            #self.scenario_results[data['uid']]["policy"][(name, args)] = penalized_total_cost
            self.scenario_results[data['uid']]["policy"][(name, args, lowerbound_args)] = penalized_total_cost
    
    def calculate_gaps(self):
        for uid, results in self.scenario_results.items():
            self.calculate_gap(uid, results)
    
    def calculate_gap(self, uid, scenario_result):
        # 1. Parse and categorize entries
        policies = scenario_result.get("policy", {})
        for key, r in scenario_result.items():
            if key != "policy":
                bound_type, bound_args = key
                for (name, args), cost in r.items():
                    policy_cost = policies.get((name, args, bound_args), None)
                    if policy_cost is not None:
                        gap = policy_cost - cost
                        if gap < 0:
                            print("uid", uid, gap, bound_type)
                        self.gap_after_warmup[(bound_type, name, args, bound_args)] += gap
                        self.ptc_gap_after_warmup[(bound_type, name, args, bound_args)] += gap / policy_cost * 100 if policy_cost > 0 else 0
                # calculate the gap between penalized_lowerbound and zero penalized lowerbound
                bound_args_dict = json.loads(bound_args)
                if bound_type == "penalized_lowerbound_" and bound_args_dict['coefficients'] == 1:
                    bound_args_dict['coefficients'] = 0
                    zero_penalized_lowerbound_args = json.dumps(bound_args_dict, sort_keys=True)
                    penalized_lowerbound_args = bound_args
                    for key, cost in scenario_result[("penalized_lowerbound_", penalized_lowerbound_args)].items():
                        gap = cost - scenario_result[("penalized_lowerbound_", zero_penalized_lowerbound_args)][key]
                        self.gap_after_warmup[(bound_type, bound_type, penalized_lowerbound_args, zero_penalized_lowerbound_args)] += gap
    
    def get_experiment_labels(self, stats):
        experiment_labels = {}
        for k in stats.keys():
            agent = eval(k)
            experiment_labels[k] = self.experiment_labels[agent['agent_name']]
        return experiment_labels

    def plot(self):
        '''
        Waiting time by type
        overtime by day
        overtime ptc by day
        '''
        key = list(self.waiting_time_by_type.keys())[0]
        x_values = sorted(list(self.waiting_time_by_type[key].keys()))
        experiment_labels = self.get_experiment_labels(self.waiting_time_by_type)
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.waiting_time_by_type,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel="Patient Type",
                                                       ylabel="Waiting Time (days)",
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path, f'waiting_time_by_type.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)
        x_values = sorted(list(self.overtime_by_day[key].keys()))
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.overtime_by_day,
                                                       x_vals=x_values,
                                                       xticks=None,
                                                       xticklabels=None,
                                                       xlabel="Day",
                                                       ylabel="Number of overtime (slots)",
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'overtime_used_by_day.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)
        x_values = sorted(list(self.cost_by_day[key].keys()))

        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.cost_by_day,
                                                       x_vals=x_values,
                                                       xticks=None,
                                                       xticklabels=None,
                                                       xlabel="Day",
                                                       ylabel="One-time Cost ($)",
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'cost_by_day.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)
    def plot_opt_gap(self, order_by_agent, x_values, xlabel, ylabel, file_name):
        opt_gap_by_policy = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        experiment_labels = {}
        for name, orders in order_by_agent.items():
            for agent, x in zip(orders, x_values):
                agent_name, args, bound_args = agent['agent_name'], json.dumps(agent.get('args', {}), sort_keys=True), json.dumps(agent.get('lowerbound_args', {}), sort_keys=True)
                if agent_name.startswith("penalized_lowerbound_"):
                    base_name = agent_name.replace("penalized_lowerbound_", "")
                    key = ("penalized_lowerbound_", base_name, args, bound_args)
                elif agent_name.startswith("lowerbound_"):
                    base_name = agent_name.replace("lowerbound_", "")
                    key = ("lowerbound_", base_name, args, bound_args)
                experiment_labels[name] = self.lowerbound_labels[name]
                opt_gap_by_policy[name][x] = self.gap_after_warmup[key]
        print(opt_gap_by_policy)
        approximate_value_plot_from_running_stats_dict(running_stats_dict=opt_gap_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=xlabel,
                                                       ylabel=ylabel,
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              file_name),
                                                       is_show_text=True,
                                                       is_set_x_color=False,
                                                       ncol=2)
    
    def plot_ptc_opt_gap(self, order_by_agent, x_values, xlabel, ylabel, file_name):
        ptc_gap_by_policy = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        experiment_labels = {}
        for name, orders in order_by_agent.items():
            for agent, x in zip(orders, x_values):
                agent_name, args, bound_args = agent['agent_name'], json.dumps(agent.get('args', {}), sort_keys=True), json.dumps(agent.get('lowerbound_args', {}), sort_keys=True)
                if agent_name.startswith("penalized_lowerbound_"):
                    base_name = agent_name.replace("penalized_lowerbound_", "")
                    key = ("penalized_lowerbound_", base_name, args, bound_args)
                elif agent_name.startswith("lowerbound_"):
                    base_name = agent_name.replace("lowerbound_", "")
                    key = ("lowerbound_", base_name, args, bound_args)
                experiment_labels[name] = self.lowerbound_labels[name]
                ptc_gap_by_policy[name][x] = self.ptc_gap_after_warmup[key]
        approximate_value_plot_from_running_stats_dict(running_stats_dict=ptc_gap_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=xlabel,
                                                       ylabel=ylabel,
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              file_name),
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)

    def plot_overtime(self, order_by_agent, x_values, xlabel, ylabel, file_name):
        overtime_by_policy = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        experiment_labels = {}
        for name, orders in order_by_agent.items():
            experiment_labels[name] = self.experiment_labels[name]
            for agent, x in zip(orders, x_values):
                overtime_by_policy[name][x] = self.average_overtime[json.dumps(agent)]
                
        approximate_value_plot_from_running_stats_dict(running_stats_dict=overtime_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=xlabel,
                                                       ylabel=ylabel,
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              file_name),
                                                       is_show_text=True,
                                                       is_set_x_color=False,
                                                       ncol=2)
    
    def plot_average_waiting_time(self, order_by_agent, x_values, xlabel, ylabel, file_name):
        waiting_time_by_policy = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        experiment_labels = {}
        for name, orders in order_by_agent.items():
            experiment_labels[name] = self.experiment_labels[name]
            for agent, x in zip(orders, x_values):
                waiting_time_by_policy[name][x] = self.average_waiting_time[json.dumps(agent)]
        
        approximate_value_plot_from_running_stats_dict(running_stats_dict=waiting_time_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=xlabel,
                                                       ylabel=ylabel,
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              file_name),
                                                       is_show_text=True,
                                                       is_set_x_color=False,
                                                       ncol=2)

    def format_table(self):
        kys = self.waiting_time_target_ptc_by_day_type.keys()
        experiment_labels = self.get_experiment_labels(self.waiting_time_target_ptc_by_day_type)
        for agent_name in kys:
            table = defaultdict(list)
            table2 = {}
            for day in [1, 5, 10, 15, 20, 25, 30]:
                total_mean = self.waiting_time_target_ptc_by_day[agent_name][day].mean
                total_hw = self.waiting_time_target_ptc_by_day[agent_name][day].half_window(0.95)
                table2[day] = (round(total_mean, 2), round(total_hw, 3))
                for type in range(len(self.waiting_time_target_ptc_by_day_type[agent_name])):
                    mean = self.waiting_time_target_ptc_by_day_type[agent_name][type][day].mean
                    hw = self.waiting_time_target_ptc_by_day_type[agent_name][type][day].half_window(0.95)
                    table[type+1].append((day, round(mean), round(hw)))
            print(table2)
            print(experiment_labels[agent_name])
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

    def report(self):
        for agent_key, agent_name in self.experiment_labels.items():
            line = (
                f"{agent_name} & "
                f"{self.total_cost_after_warmup[agent_key].mean:.2f}"
                f" $\\pm$ {self.total_cost_after_warmup[agent_key].half_window(0.95):.3f} & "
                f"{self.total_penalty[agent_key].mean:.2f}"
                f" $\\pm$ {self.total_penalty[agent_key].half_window(0.95):.3f} & "
                f"{self.average_waiting_time[agent_key].mean:.2f}"
                f" $\\pm$ {self.average_waiting_time[agent_key].half_window(0.95):.3f} & "
                f"{self.average_overtime[agent_key].mean:.2f}"
                f" $\\pm$ {self.average_overtime[agent_key].half_window(0.95):.3f}"
                r" \\ "
            )
            print(line)

    def plot_value_function(self, order_by_agent, x_values, xlabel, ylabel, file_name):
        self.total_cost_after_warmup_by_x = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        print(self.total_cost_after_warmup.keys())
        experiment_labels = {}
        for name, orders in order_by_agent.items():
            experiment_labels[name] = self.experiment_labels[name]
            for agent, x in zip(orders, x_values):
                self.total_cost_after_warmup_by_x[name][x] = self.total_cost_after_warmup[str(agent)]
        #print(self.total_cost_after_warmup_by_x)

        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.total_cost_after_warmup_by_x,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=xlabel,
                                                       ylabel=ylabel,
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              file_name),
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)

    def plot_warmup_relative_error(self,order_by_agent, x_values, xlabel, ylabel, file_name):
        relative_error_by_day = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        experiment_labels = {}
        mark = {}  # agent_name -> ptc -> day
        for name, orders in order_by_agent.items():
            experiment_labels[name] = self.experiment_labels[name]
            for agent, x in zip(orders, x_values):
                relative_error = self.cost_by_day[str(agent)][x].half_window(0.9) / \
                                 self.cost_by_day[str(agent)][x].mean * 100
                relative_error_by_day[name][x] += relative_error
                if name not in mark:
                    mark[name] = {}
                if 7 not in mark[name] and relative_error < 7:
                    mark[name][7] = x
                if 6 not in mark[name] and relative_error < 6:
                    mark[name][6] = x
                elif 5 not in mark[name] and relative_error < 5:
                    mark[name][5] = x
                elif 4 not in mark[name] and relative_error < 4:
                    mark[name][4] = x
                elif 3 not in mark[name] and relative_error < 3:
                    mark[name][3] = x
        print('Relative Error by Day:')
        approximate_value_plot_from_running_stats_dict(running_stats_dict=relative_error_by_day,
                                                       x_vals=np.array(x_values),
                                                       xticks=None,
                                                       xticklabels=None,
                                                       xlabel=xlabel,
                                                       ylabel=ylabel,
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              file_name),
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)


if __name__=='__main__':
    directory_path = os.path.join('.', 'experiments', 'results', "small_scale_problem")
    # 5: 33.1989634321917 0.13896181129865617
    # 10: 36.687370600414376 0.1486390341192171
    # 20: 39.3842249382221 0.5255526412672854
    file_pattern = '[0-9]*.jsonl'
    fuck = 250
    ser = SimulateEvaluationResult(directory_path, file_pattern)
    '''
    x_values = [i for i in range(2000-fuck)]
    xlabel = "Periods"
    order_by_agent = {
        "myopic": [{'agent_name': 'myopic','args': {}} for i in x_values],
        # "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
        #             "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
        #                              97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
        #                              93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
        #                              89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
        #                              86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
        #                              82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
        #                              78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
        #                              75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
        #                              72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
        #                              69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
        #                              66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
        #                              64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
        #                              61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
        #                              59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
        #                              4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
        #                              6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
        #                              8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
        #                              9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
        #                              1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
        #                              1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
        #                              1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
        #                              1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
        #                              7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
        #                              4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
        #                              3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
        #                              -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #                              0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
        #                              1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
        #                         for _ in x_values],
        # "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
        #                           'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
        #                                    'future_decision_var_type': 'continuous', 'is_myopic': False,
        #                                    'sample_path_length': None, 'is_include_discount_factor': False,
        #                                    'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
        #                                    "geom_p": 0.05}} for i in x_values],
        # "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
        #                          'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
        #                                   'future_decision_var_type': 'continuous', 'is_myopic': False,
        #                                   'sample_path_length': None, 'is_include_discount_factor': False,
        #                                   'is_quasi_MC': True, "max_periods": int(geom.ppf(0.995, 0.05)),
        #                                   "geom_p": 0.05}} for i in x_values]
    }

    ser.plot_relative_error(order_by_agent, x_values=x_values, xlabel=xlabel, ylabel="Cumulative Cost Relative Error (%)",
                                   file_name="relative_error_sample_path_total_cost.svg")
    x_values = [i for i in range(2000)]
    order_by_agent = {
        "myopic": [{'agent_name': 'myopic', 'args': {}} for i in x_values],
        # "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
        #             "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
        #                              97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
        #                              93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
        #                              89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
        #                              86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
        #                              82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
        #                              78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
        #                              75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
        #                              72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
        #                              69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
        #                              66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
        #                              64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
        #                              61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
        #                              59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
        #                              4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
        #                              6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
        #                              8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
        #                              9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
        #                              1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
        #                              1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
        #                              1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
        #                              1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
        #                              7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
        #                              4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
        #                              3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
        #                              -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #                              0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
        #                              1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
        #                         for _ in x_values],
        # "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
        #                           'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
        #                                    'future_decision_var_type': 'continuous', 'is_myopic': False,
        #                                    'sample_path_length': None, 'is_include_discount_factor': False,
        #                                    'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
        #                                    "geom_p": 0.05}} for i in x_values],
        # "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
        #                          'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
        #                                   'future_decision_var_type': 'continuous', 'is_myopic': False,
        #                                   'sample_path_length': None, 'is_include_discount_factor': False,
        #                                   'is_quasi_MC': True, "max_periods": int(geom.ppf(0.995, 0.05)),
        #                                   "geom_p": 0.05}} for i in x_values]
    }
    ser.plot_warmup_relative_error(order_by_agent, x_values=x_values, xlabel=xlabel, ylabel="One-Time Cost Relative Error (%)",
                                   file_name="relative_error_warmup_period_onetime_cost.svg")
    ser.format_table()

    """
    1. Number of scenarios vs. total cost
    """
    # scenarios_values = [128, 256, 512]
    # scenarios_xlabel = "Number of Scenarios"
    # order_by_agent = {
    #     "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
    #                              'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
    #                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                       'sample_path_length': None, 'is_include_discount_factor': False,
    #                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
    #                                       "geom_p": 0.05}} for n in scenarios_values],
    #     "lowerbound_hindsight_approx_MC": [{'agent_name': 'lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
    #                                                  "geom_p": 0.05}}
    #                                        for n in scenarios_values],
    #     "penalized_lowerbound_hindsight_approx_MC": [{'agent_name': 'penalized_lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
    #                                                  "geom_p": 0.05}}
    #                                        for n in scenarios_values],
    #     "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
    #                           'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
    #                                    'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                    'sample_path_length': None, 'is_include_discount_factor': False,
    #                                    'is_quasi_MC': True, "max_periods": int(geom.ppf(0.995, 0.05)), "geom_p": 0.05}}
    #                          for n in scenarios_values],
    #     "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
    #         "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
    #                          97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
    #                          93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
    #                          89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
    #                          86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
    #                          82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
    #                          78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
    #                          75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
    #                          72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
    #                          69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
    #                          66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
    #                          64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
    #                          61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
    #                          59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
    #                          4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
    #                          6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
    #                          8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
    #                          9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
    #                          1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
    #                          1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
    #                          1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
    #                          1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
    #                          7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
    #                          4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
    #                          3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
    #                          -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                          0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
    #                          1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
    #                     for _ in scenarios_values]
    # }
    # ser.plot_value_function(order_by_agent, x_values=scenarios_values, xlabel=scenarios_xlabel,
    #                         ylabel="Total Cost After Warm-up ($)",
    #                         file_name="scenario_total_cost.svg")
    #
    # """
    # 2. discount factor vs. total cost
    # """
    # discount_values = [0.9, 0.95, 0.98]
    # discount_xlabel = "Discount Factor"
    # order_by_agent = {
    #     "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
    #                              'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                       'sample_path_length': None, 'is_include_discount_factor': False,
    #                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 1 - gamma)),
    #                                       "geom_p": round(1 - gamma, 2)}}
    #                             for gamma in discount_values],
    #     "lowerbound_hindsight_approx_MC": [{'agent_name': 'lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False,
    #                                                  "max_periods": int(geom.ppf(0.995, 1 - gamma)),
    #                                                  "geom_p": round(1 - gamma, 2)}}
    #                                        for gamma in discount_values],
    #     "penalized_lowerbound_hindsight_approx_MC": [{'agent_name': 'penalized_lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False,
    #                                                  "max_periods": int(geom.ppf(0.995, 1 - gamma)),
    #                                                  "geom_p": round(1 - gamma, 2)}}
    #                                        for gamma in discount_values],
    #     "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
    #                           'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                    'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                    'sample_path_length': None, 'is_include_discount_factor': False,
    #                                    'is_quasi_MC': True, "max_periods": int(geom.ppf(0.995, 1-gamma)), "geom_p": round(1-gamma, 2)}}
    #                          for gamma in discount_values],
    #     "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
    #         "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
    #                          97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
    #                          93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
    #                          89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
    #                          86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
    #                          82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
    #                          78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
    #                          75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
    #                          72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
    #                          69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
    #                          66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
    #                          64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
    #                          61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
    #                          59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
    #                          4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
    #                          6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
    #                          8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
    #                          9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
    #                          1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
    #                          1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
    #                          1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
    #                          1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
    #                          7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
    #                          4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
    #                          3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
    #                          -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                          0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
    #                          1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
    #                     for gamma in discount_values]
    # }
    # ser.plot_value_function(order_by_agent, x_values=discount_values, xlabel=discount_xlabel,
    #                         ylabel="Total Cost After Warm-up ($)",
    #                         file_name="discount_total_cost.svg")
    # """
    # 3. truncation level vs. total cost
    # """
    # truncation_values = [50, 80, 99.5]
    # truncation_xlabel = "Truncation Level (%)"
    # order_by_agent = {
    #     "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
    #                              'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                       'sample_path_length': None, 'is_include_discount_factor': False,
    #                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(truncation / 100, 0.05)),
    #                                       "geom_p": 0.05}}
    #                             for truncation in truncation_values],
    #     "lowerbound_hindsight_approx_MC": [{'agent_name': 'lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False,
    #                                                  "max_periods": int(geom.ppf(truncation / 100, 0.05)),
    #                                                  "geom_p": 0.05}}
    #                                        for truncation in truncation_values],
    #     "penalized_lowerbound_hindsight_approx_MC": [{'agent_name': 'penalized_lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False,
    #                                                  "max_periods": int(geom.ppf(truncation / 100, 0.05)),
    #                                                  "geom_p": 0.05}}
    #                                        for truncation in truncation_values],
    #     "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
    #                           'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                    'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                    'sample_path_length': None, 'is_include_discount_factor': False,
    #                                    'is_quasi_MC': True, "max_periods": int(geom.ppf(truncation/100, 0.05)),
    #                                    "geom_p": 0.05}}
    #                          for truncation in truncation_values],
    #     "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
    #         "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
    #                          97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
    #                          93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
    #                          89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
    #                          86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
    #                          82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
    #                          78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
    #                          75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
    #                          72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
    #                          69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
    #                          66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
    #                          64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
    #                          61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
    #                          59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
    #                          4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
    #                          6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
    #                          8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
    #                          9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
    #                          1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
    #                          1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
    #                          1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
    #                          1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
    #                          7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
    #                          4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
    #                          3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
    #                          -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                          0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
    #                          1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
    #                     for gamma in truncation_values]
    # }
    # '''
    # Combine plots into one figure
    # '''
    # ser.plot_value_function(order_by_agent, x_values=truncation_values, xlabel=truncation_xlabel,
    #                         ylabel="Total Cost After Warm-up ($)",
    #                         file_name="truncation_total_cost.svg")
