import json
import os
from collections import defaultdict

import pandas as pd
import numpy as np
import glob

from utils import RunningStats, safe_execute
from visualization import approximate_value_plot_from_running_stats_dict

class ExperimentResult:

    def __init__(self, experiment_result_builder):
        self.experiment_result_builder = experiment_result_builder
        self.directory_path = experiment_result_builder.directory_path
        self.file_pattern = experiment_result_builder.file_pattern
        self.experiment_label = experiment_result_builder.experiment_label
        self.performance_interest_list = experiment_result_builder.performance_interest_list
        # we can combine performance interest to two or three dim cases
        self.value_function_stats_by_policy = defaultdict(lambda:defaultdict(lambda: RunningStats()))
        self.abs_gap_stats_by_policy = defaultdict(lambda:defaultdict(lambda: RunningStats()))
        self.pct_opt_gap_stats_by_policy = defaultdict(lambda:defaultdict(lambda: RunningStats()))

        self.average_waiting_time_stats_by_policy = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        self.average_waiting_time_stats_by_type_policy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: RunningStats())))

        self.average_waiting_time_target_violation_stats_by_policy = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        self.average_waiting_time_target_violation_stats_by_type_policy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: RunningStats())))

        self.average_overtime_stats_by_policy = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        self.overtime_used_ptc_stats_by_policy = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        self.average_overtime_stats_by_day_by_policy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: RunningStats())))

        self.average_postponing_decision_stats_by_policy = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        self.average_postponing_decision_stats_by_type_policy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: RunningStats())))

        self.average_postponing_decision_ptc_stats_by_policy = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        self.average_postponing_decision_ptc_stats_by_type_policy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: RunningStats())))

        pattern = os.path.join(self.directory_path, self.file_pattern)
        jsonl_files = glob.glob(pattern)
        uids = set()
        for file_path in jsonl_files:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    param_value = data.get('param_value')
                    experiment_name = data.get('experiment_name')
                    uid = data.get('uid')
                    if uid in uids:
                        continue
                    else:
                        uids.add(uid)
                        self.load(data)

    def plot(self):
        key = list(self.value_function_stats_by_policy.keys())[0]
        x_values = sorted(list(self.value_function_stats_by_policy[key].keys()))
        #for performance_interest in self.performance_interest_list:

        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.value_function_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Value Function",
                                                       plot_labels=plot_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path, f'value_function.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)
        hindsight_labels = {'hindsight_approx': 'Hindsight Policy',
                            'benchmark_value': "Lower Bound",
                            #'penalized_lower_bound': "Penalized Lower Bound"
                            }
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.value_function_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Value Function",
                                                       plot_labels=hindsight_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'value_function_hindsight_policy.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)
        myopic_alp_label = {'myopic': 'Myopic Policy', 'row_gen_alp': 'ALP', }
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.value_function_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Value Function",
                                                       plot_labels=myopic_alp_label,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'value_function_myopic_alp.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)
        plot_labels.pop('benchmark_value')
        plot_labels.pop('penalized_lower_bound', None)
        hindsight_label = {'hindsight_approx': 'Hindsight Policy vs. LB',
                           #'hindsight_approx_penalized': 'Hindsight Policy vs. PLB'
                           }
        myopic_alp_label = {'myopic': 'Myopic Policy vs. LB',
                            'row_gen_alp': 'ALP vs. LB',
                            #'myopic_penalized': 'Myopic Policy vs. PLB',
                            #'row_gen_alp_penalized': 'ALP vs. PLB'
                            }
        all_labels = hindsight_label | myopic_alp_label
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.abs_gap_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Absolute optimality gap (UB)",
                                                       plot_labels=all_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'abs_opt_gap.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.abs_gap_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Absolute optimality gap (UB)",
                                                       plot_labels=hindsight_label,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'abs_opt_gap_hindsight_policy.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.abs_gap_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Absolute optimality gap (UB)",
                                                       plot_labels=myopic_alp_label,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'abs_opt_gap_myopic_alp.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.pct_opt_gap_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Percentage optimality gap (UB %)",
                                                       plot_labels=all_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'pct_opt_gap.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.pct_opt_gap_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Percentage optimality gap (UB %)",
                                                       plot_labels=hindsight_label,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'pct_opt_gap_hindsight_policy.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.pct_opt_gap_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Percentage optimality gap (UB %)",
                                                       plot_labels=myopic_alp_label,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'pct_opt_gap_myopic_alp_policy.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)

        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.average_waiting_time_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Waiting time (days)",
                                                       plot_labels=plot_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path, f'average_wait_time.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)

        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.average_waiting_time_target_violation_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Total waiting time violation (%)",
                                                       plot_labels=plot_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'waiting_time_violation.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)

        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.average_overtime_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Number of overtime (slots)",
                                                       plot_labels=plot_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path, f'average_overtime_used.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)

        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.overtime_used_ptc_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Overtime utilization (%)",
                                                       plot_labels=plot_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'overtime_used_ptc.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)

        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.average_postponing_decision_stats_by_policy,
                                                        x_vals=x_values,
                                                        xticks=x_values,
                                                        xticklabels=x_values,
                                                        xlabel=self.experiment_label,
                                                        ylabel="Deferred decisions (treatments)",
                                                        plot_labels=plot_labels,
                                                        title=None,
                                                        save_file=os.path.join(self.directory_path, f'postponing_decision.svg'),
                                                        is_show_text=False,
                                                        is_set_x_color=False)

        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.average_postponing_decision_ptc_stats_by_policy,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=self.experiment_label,
                                                       ylabel="Deferred decisions (%)",
                                                       plot_labels=plot_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path, 'postponing_decision_ptc.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)

        plot_labels.pop('benchmark_value', None)
        plot_labels.pop('penalized_lower_bound', None)
        print(plot_labels)
        for x_value in x_values:
            key = list(self.average_waiting_time_stats_by_type_policy[x_value].keys())[0]
            treatment_types = np.array(sorted(list(self.average_waiting_time_stats_by_type_policy[x_value][key].keys())))
            x_value_str = str(x_value).replace('.', '_')
            approximate_value_plot_from_running_stats_dict(running_stats_dict=self.average_waiting_time_stats_by_type_policy[x_value],
                                                           x_vals=treatment_types,
                                                           xticks=treatment_types,
                                                           xticklabels=treatment_types + 1,
                                                           xlabel='Treatment type',
                                                           ylabel="Waiting time (days)",
                                                           plot_labels=plot_labels,
                                                           title=f"Waiting Time with {self.experiment_label}: {x_value}",
                                                           save_file=os.path.join(self.directory_path,
                                                                                  f'average_wait_time_type_{x_value_str}.svg'),
                                                           is_show_text=False,
                                                           is_set_x_color=False)
            approximate_value_plot_from_running_stats_dict(
                running_stats_dict=self.average_waiting_time_target_violation_stats_by_type_policy[x_value],
                x_vals=treatment_types,
                xticks=treatment_types,
                xticklabels=treatment_types + 1,
                xlabel='Treatment type',
                ylabel="Waiting time Violation (%)",
                plot_labels=plot_labels,
                title=f"Waiting time violation with {self.experiment_label}: {x_value}",
                save_file=os.path.join(self.directory_path,
                                       f'waiting_time_violation_{x_value_str}.svg'),
                is_show_text=False,
                is_set_x_color=False)

            approximate_value_plot_from_running_stats_dict(
                running_stats_dict=self.average_postponing_decision_stats_by_type_policy[x_value],
                x_vals=treatment_types,
                xticks=treatment_types,
                xticklabels=treatment_types + 1,
                xlabel='Treatment type',
                ylabel="Deferred decisions (treatments)",
                plot_labels=plot_labels,
                title=f"Deferred Decisions with {self.experiment_label}: {x_value}",
                save_file=os.path.join(self.directory_path,
                                       f'postponing_decisions_type_{x_value_str}.svg'),
                is_show_text=False,
                is_set_x_color=False)
            approximate_value_plot_from_running_stats_dict(
                running_stats_dict=self.average_postponing_decision_ptc_stats_by_type_policy[x_value],
                x_vals=treatment_types,
                xticks=treatment_types,
                xticklabels=treatment_types + 1,
                xlabel='Treatment type',
                ylabel="Deferred decisions (%)",
                plot_labels=plot_labels,
                title=f"Deferred Decisions % with {self.experiment_label}: {x_value}",
                save_file=os.path.join(self.directory_path,
                                       f'ptc_postponing_decisions_{x_value_str}.svg'),
                is_show_text=False,
                is_set_x_color=False)

    def load(self, data):
        param_value = data.get('param_value')
        for agent_result in data.get('result', []):
            agent_name = agent_result['agent_name']
            value_function = agent_result['value_function']
            abs_gap_value = value_function - agent_result['perfect_info_lower_bound']
            abs_gap_with_penalized_value = value_function - agent_result['penalized_lower_bound']
            pct_opt_gap_value = abs_gap_value / agent_result['perfect_info_lower_bound'] * 100 if agent_result['perfect_info_lower_bound'] != 0 else 0.0
            pct_opt_gap_with_penalized_value = abs_gap_value / agent_result['perfect_info_lower_bound'] * 100 if agent_result['perfect_info_lower_bound'] != 0 else 0.0
            self.value_function_stats_by_policy[agent_name][param_value] += value_function
            self.abs_gap_stats_by_policy[agent_name][param_value] += abs_gap_value
            self.abs_gap_stats_by_policy[agent_name + "_penalized"][param_value] += abs_gap_with_penalized_value
            self.pct_opt_gap_stats_by_policy[agent_name][param_value] += pct_opt_gap_value
            self.pct_opt_gap_stats_by_policy[agent_name + "_penalized"][param_value] += pct_opt_gap_with_penalized_value

            if agent_name == 'myopic':
                self.value_function_stats_by_policy['benchmark_value'][param_value] += agent_result['perfect_info_lower_bound']
                self.value_function_stats_by_policy['penalized_lower_bound'][param_value] += agent_result['penalized_lower_bound']

            for wait_time_by_type in agent_result['wait_time_by_type']:
                treatment_type = wait_time_by_type['treatment_type']
                wait_time_stats = RunningStats(wait_time_by_type['count'], wait_time_by_type['expect'],
                                               wait_time_by_type['varSum'])
                self.average_waiting_time_stats_by_policy[agent_name][param_value] += wait_time_stats
                self.average_waiting_time_stats_by_type_policy[param_value][agent_name][treatment_type] += wait_time_stats

            for waiting_time_target_violations in agent_result['waiting_time_target_violations']:
                treatment_type = waiting_time_target_violations['treatment_type']
                waiting_time_target_violation_stats = RunningStats(waiting_time_target_violations['count'],
                                                                   waiting_time_target_violations['expect'] * 100,
                                                                   waiting_time_target_violations['varSum'])
                self.average_waiting_time_target_violation_stats_by_policy[agent_name][
                    param_value] += waiting_time_target_violation_stats
                self.average_waiting_time_target_violation_stats_by_type_policy[param_value][agent_name][treatment_type] += waiting_time_target_violation_stats

            for day, overtime in enumerate(agent_result['overtime']):
                self.average_overtime_stats_by_policy[agent_name][param_value] += overtime
                self.overtime_used_ptc_stats_by_policy[agent_name][param_value] += overtime / 6 * 100 # need to modify the capacity
                self.average_overtime_stats_by_day_by_policy[param_value][agent_name][day] += overtime

            for day, (postponing_decisions, waitlist) in enumerate(zip(agent_result['postponing_decision_number'], agent_result['waiting_number'])):
                for treatment_type, (postponing_decision, waiting) in enumerate(zip(postponing_decisions, waitlist)):
                    self.average_postponing_decision_stats_by_policy[agent_name][param_value] += postponing_decision
                    self.average_postponing_decision_ptc_stats_by_policy[agent_name][param_value] += postponing_decision/waiting * 100 if postponing_decision != 0 else 0.0
                    self.average_postponing_decision_stats_by_type_policy[param_value][agent_name][treatment_type] += postponing_decision
                    self.average_postponing_decision_ptc_stats_by_type_policy[param_value][agent_name][treatment_type] += postponing_decision/waiting * 100 if postponing_decision != 0 else 0.0

class ExperimentResultBuilder:

    def __init__(self, directory_path, file_pattern, experiment_label):
        self.directory_path = directory_path
        self.file_pattern = file_pattern
        self.experiment_label = experiment_label
        self.performance_interest_list = []

    def with_value_function(self, policy_names):
        performance_interest = {
            'name': 'value_function',
            'policy_names': policy_names
        }
        self.performance_interest_list.append(performance_interest)
        return self

    def with_abs_gap(self, policy_names, benchmark_name):
        performance_interest = {
            'name': 'abs_gap',
            'get_data_fn': lambda data: data['value_function'] - data[benchmark_name],
        }
        self.performance_interest_list.append(performance_interest)
        return self

    def build(self):
        return ExperimentResult(self)
@safe_execute(debug_mode=True)
def plot_experiment_result(directory_path, plot_labels, experiment_lables):
    """
    Loads a .jsonl result file and processes it into a pandas DataFrame.

    Args:
        file_path (str): The path to the .jsonl file.

    Returns:
        pd.DataFrame: A DataFrame containing the processed results,
                      with one row per agent per simulation run.
    """
    '''
    value_function_stats_by_policy
    abs_gap_stats_by_policy
    pct_opt_gap_stats_by_policy
    
    average_waiting_time_stats_by_policy
    average_waiting_time_stats_by_type_policy
    average_waiting_time_target_violation_stats_by_type_policy
    
    average_overtime_stats_by_policy
    average_overtime_stats_by_day_by_policy
    
    average_postponing_decision_stats_by_policy
    average_postponing_decision_ptc_stats_by_policy
    average_postponing_decision_stats_by_type_policy
    '''
    abs_gap = defaultdict(lambda:defaultdict(lambda: RunningStats()))
    pct_opt_gap = defaultdict(lambda:defaultdict(lambda: RunningStats()))
    value_function_by_agent = defaultdict(lambda:defaultdict(lambda: RunningStats()))
    waiting_time_target_violation_by_type_by_agent = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: RunningStats())))
    wait_time_by_type_by_agent = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: RunningStats())))
    total_average_wait_time = defaultdict(lambda: defaultdict(lambda: RunningStats()))
    total_waiting_time_violation_by_agent = defaultdict(lambda: defaultdict(lambda: RunningStats()))
    total_overtime_by_day_by_agent = defaultdict(lambda: defaultdict(lambda: RunningStats()))

    x_values = set()
    treatment_types = set()
    pattern = os.path.join(directory_path, '[0-9]*.jsonl')
    jsonl_files = glob.glob(pattern)
    count = 0
    uids = set()
    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                param_value = data.get('param_value')
                experiment_name = data.get('experiment_name')
                uid = data.get('uid')
                if uid in uids:
                    continue
                else:
                    uids.add(uid)
                    count += 1
                    x_values.add(param_value)
                    for agent_result in data.get('result', []):
                        agent_name = agent_result['agent_name']
                        print('agent_name:', agent_name)
                        value_function = agent_result['value_function']
                        #abs_gap[agent_name][param_value] += abs(value_function - agent_result['benchmark_value'])
                        #pct_opt_gap[agent_name][param_value] += abs(value_function - agent_result['benchmark_value']) / abs(agent_result['benchmark_value']) * 100 if agent_result['benchmark_value'] !=0 else 0.0
                        abs_gap_value = value_function - agent_result['perfect_info_lower_bound']
                        abs_gap_with_penalized_value = value_function - agent_result['penalized_lower_bound']
                        abs_gap[agent_name][param_value] += abs_gap_value
                        abs_gap[agent_name+"_penalized"][param_value] += abs_gap_with_penalized_value
                        pct_opt_gap[agent_name][param_value] += abs_gap_value / agent_result['perfect_info_lower_bound'] * 100 if agent_result['perfect_info_lower_bound'] != 0 else 0.0
                        pct_opt_gap[agent_name+"_penalized"][param_value] += abs_gap_value / agent_result['penalized_lower_bound'] * 100 if agent_result['penalized_lower_bound'] != 0 else 0.0
                        value_function_by_agent[agent_name][param_value] += value_function
                        if agent_name == 'myopic':
                            value_function_by_agent['benchmark_value'][param_value] += agent_result['perfect_info_lower_bound']
                            value_function_by_agent['penalized_lower_bound'][param_value] += agent_result['penalized_lower_bound']

                        for wait_time_by_type in agent_result['wait_time_by_type']:
                            treatment_type = wait_time_by_type['treatment_type']
                            treatment_types.add(treatment_type)
                            wait_time_stats = RunningStats(wait_time_by_type['count'], wait_time_by_type['expect'], wait_time_by_type['varSum'])
                            wait_time_by_type_by_agent[param_value][agent_name][treatment_type]+=wait_time_stats
                            total_average_wait_time[agent_name][param_value]+=wait_time_stats
                        for waiting_time_target_violations in agent_result['waiting_time_target_violations']:
                            treatment_type = waiting_time_target_violations['treatment_type']
                            treatment_types.add(treatment_type)
                            waiting_time_target_violation_stats = RunningStats(waiting_time_target_violations['count'], waiting_time_target_violations['expect'] * 100, waiting_time_target_violations['varSum'])
                            waiting_time_target_violation_by_type_by_agent[param_value][agent_name][treatment_type] += waiting_time_target_violation_stats
                            total_waiting_time_violation_by_agent[agent_name][param_value] += waiting_time_target_violation_stats
                        for overtime in agent_result['overtime']:
                            total_overtime_by_day_by_agent[agent_name][param_value].record(overtime)
    print("Sample path number:", count)
    x_values = sorted(list(x_values))
    print('value_function_by_agent')
    for agent, occupency_stats in value_function_by_agent.items():
        print(f"Agent: {agent}")
        for occupency_level, stats in occupency_stats.items():
            print(f"  Occupancy Level: {occupency_level}, Mean Value Function: {stats.mean}, half window: {stats.half_window(0.95)}")
    print('total_waiting_time_violation_by_agent')
    for agent, occupency_stats in total_waiting_time_violation_by_agent.items():
        print(f"Agent: {agent}")
        for occupency_level, stats in occupency_stats.items():
            print(f"  Occupancy Level: {occupency_level}, Mean Value Function: {stats.mean}, half window: {stats.half_window(0.95)}")
    print('total_overtime_by_day_by_agent')
    for agent, occupency_stats in total_overtime_by_day_by_agent.items():
        print(f"Agent: {agent}")
        for occupency_level, stats in occupency_stats.items():
            print(f"  Occupancy Level: {occupency_level}, Mean Value Function: {stats.mean}, half window: {stats.half_window(0.95)}")
    print(abs_gap.keys())
    approximate_value_plot_from_running_stats_dict(running_stats_dict=value_function_by_agent,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Value Function",
                                                   plot_labels=plot_labels,
                                                   title=None,
                                                   save_file=os.path.join(directory_path, f'value_function.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)
    hindsight_labels = {'hindsight_approx': 'Hindsight Policy',
                        'benchmark_value': "Lower Bound",
                        #'penalized_lower_bound': "Penalized Lower Bound"
                        }
    approximate_value_plot_from_running_stats_dict(running_stats_dict=value_function_by_agent,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Value Function",
                                                   plot_labels=hindsight_labels,
                                                   title=None,
                                                   save_file=os.path.join(directory_path,
                                                                          f'value_function_hindsight_policy.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)
    myopic_alp_label = {'myopic': 'Myopic Policy', 'row_gen_alp': 'ALP',}
    approximate_value_plot_from_running_stats_dict(running_stats_dict=value_function_by_agent,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Value Function",
                                                   plot_labels=myopic_alp_label,
                                                   title=None,
                                                   save_file=os.path.join(directory_path,
                                                                          f'value_function_myopic_alp.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)
    plot_labels.pop('benchmark_value')
    plot_labels.pop('penalized_lower_bound', None)
    hindsight_label = {'hindsight_approx': 'Hindsight Policy vs. LB',
                       #'hindsight_approx_penalized': 'Hindsight Policy vs. PLB'
                    }
    myopic_alp_label = {'myopic': 'Myopic Policy vs. LB',
                        'row_gen_alp': 'ALP vs. LB',
                        #'myopic_penalized': 'Myopic Policy vs. PLB',
                        #'row_gen_alp_penalized': 'ALP vs. PLB'
                        }
    all_labels = hindsight_label | myopic_alp_label
    approximate_value_plot_from_running_stats_dict(running_stats_dict=abs_gap,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Absolute Optimality Gap (UB)",
                                                   plot_labels=all_labels,
                                                   title=None,
                                                   save_file=os.path.join(directory_path,
                                                                          f'abs_opt_gap.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)
    approximate_value_plot_from_running_stats_dict(running_stats_dict=abs_gap,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Absolute Optimality Gap (UB)",
                                                   plot_labels=hindsight_label,
                                                   title=None,
                                                   save_file=os.path.join(directory_path,
                                                                          f'abs_opt_gap_hindsight_policy.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)
    approximate_value_plot_from_running_stats_dict(running_stats_dict=abs_gap,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Absolute Optimality Gap (UB)",
                                                   plot_labels=myopic_alp_label,
                                                   title=None,
                                                   save_file=os.path.join(directory_path,
                                                                          f'abs_opt_gap_myopic_alp.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)
    approximate_value_plot_from_running_stats_dict(running_stats_dict=pct_opt_gap,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Percentage Optimality Gap (UB %)",
                                                   plot_labels=all_labels,
                                                   title=None,
                                                   save_file=os.path.join(directory_path,
                                                                          f'pct_opt_gap.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)
    approximate_value_plot_from_running_stats_dict(running_stats_dict=pct_opt_gap,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Percentage Optimality Gap (UB %)",
                                                   plot_labels=hindsight_label,
                                                   title=None,
                                                   save_file=os.path.join(directory_path,
                                                                          f'pct_opt_gap_hindsight_policy.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)
    approximate_value_plot_from_running_stats_dict(running_stats_dict=pct_opt_gap,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Percentage Optimality Gap (UB %)",
                                                   plot_labels=myopic_alp_label,
                                                   title=None,
                                                   save_file=os.path.join(directory_path,
                                                                          f'pct_opt_gap_myopic_alp_policy.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)

    approximate_value_plot_from_running_stats_dict(running_stats_dict=total_average_wait_time,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Waiting time (days)",
                                                   plot_labels=plot_labels,
                                                   title=None,
                                                   save_file=os.path.join(directory_path, f'average_wait_time.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)

    approximate_value_plot_from_running_stats_dict(running_stats_dict=total_waiting_time_violation_by_agent,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Total Waiting time Violation (%)",
                                                   plot_labels=plot_labels,
                                                   title=None,
                                                   save_file=os.path.join(directory_path,
                                                                          f'waiting_time_violation.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)

    approximate_value_plot_from_running_stats_dict(running_stats_dict=total_overtime_by_day_by_agent,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Number of Overtime (slots)",
                                                   plot_labels=plot_labels,
                                                   title=None,
                                                   save_file=os.path.join(directory_path, f'average_overtime_used.svg'),
                                                   is_show_text=False,
                                                   is_set_x_color=False)
    treatment_types = np.array(sorted(list(treatment_types)))
    for x_value in x_values:
        print(x_value)
        x_value_str = str(x_value).replace('.', '_')
        approximate_value_plot_from_running_stats_dict(running_stats_dict=wait_time_by_type_by_agent[x_value],
                                                       x_vals=treatment_types,
                                                       xticks=treatment_types,
                                                       xticklabels=treatment_types + 1,
                                                       xlabel='Treatment type',
                                                       ylabel="Waiting time (days)",
                                                       plot_labels=plot_labels,
                                                       title=f"Waiting Time with {experiment_lables[experiment_name]}: {x_value}",
                                                       save_file=os.path.join(directory_path, f'average_wait_time_type_{x_value_str}.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)
        approximate_value_plot_from_running_stats_dict(running_stats_dict=waiting_time_target_violation_by_type_by_agent[x_value],
                                                       x_vals=treatment_types,
                                                       xticks=treatment_types,
                                                       xticklabels=treatment_types + 1,
                                                       xlabel='Treatment type',
                                                       ylabel="Waiting time Violation (%)",
                                                       plot_labels=plot_labels,
                                                       title=f"Waiting time violation with {experiment_lables[experiment_name]}: {x_value}",
                                                       save_file=os.path.join(directory_path,
                                                                              f'waiting_time_violation_{x_value_str}.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)
        print('Finished plotting.')

# {"result": [{"agent_name": "myopic", "value_function": 6080686.35843262, "benchmark_value": 3133560.6074154405, "wait_time_by_type": [{"treatment_type": 0, "expect": 12.28888888888889, "varSum": 1791.2444444444443, "count": 45}, {"treatment_type": 1, "expect": 13.03225806451613, "varSum": 2585.9354838709673, "count": 62}, {"treatment_type": 2, "expect": 18.530303030303024, "varSum": 3912.87878787879, "count": 132}, {"treatment_type": 3, "expect": 21.886075949367093, "varSum": 4349.924050632907, "count": 237}, {"treatment_type": 4, "expect": 21.42142857142857, "varSum": 4302.271428571428, "count": 280}], "waiting_time_target_violations": [{"treatment_type": 0, "expect": 0.8888888888888888, "varSum": 4.444444444444446, "count": 45}, {"treatment_type": 1, "expect": 0.8064516129032258, "varSum": 9.677419354838714, "count": 62}, {"treatment_type": 2, "expect": 0.9545454545454546, "varSum": 5.727272727272728, "count": 132}, {"treatment_type": 3, "expect": 0.9662447257383966, "varSum": 7.729957805907173, "count": 237}, {"treatment_type": 4, "expect": 0.9571428571428572, "varSum": 11.485714285714282, "count": 280}], "overtime": [4, 1, 0, 0, 0, 0, 0, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 0, 0, 0]}, {"agent_name": "col_gen_alp", "value_function": 5192275.103651488, "benchmark_value": 3133560.6074154405, "wait_time_by_type": [{"treatment_type": 0, "expect": 3.4615384615384617, "varSum": 1864.1538461538462, "count": 65}, {"treatment_type": 1, "expect": 2.7936507936507935, "varSum": 1776.31746031746, "count": 63}, {"treatment_type": 2, "expect": 12.389705882352942, "varSum": 13232.345588235294, "count": 136}, {"treatment_type": 3, "expect": 22.602409638554217, "varSum": 5213.638554216868, "count": 249}, {"treatment_type": 4, "expect": 23.437735849056605, "varSum": 2183.2226415094347, "count": 265}], "waiting_time_target_violations": [{"treatment_type": 0, "expect": 0.3076923076923077, "varSum": 13.846153846153848, "count": 65}, {"treatment_type": 1, "expect": 0.2222222222222222, "varSum": 10.888888888888888, "count": 63}, {"treatment_type": 2, "expect": 0.6544117647058821, "varSum": 30.757352941176467, "count": 136}, {"treatment_type": 3, "expect": 0.963855421686747, "varSum": 8.674698795180724, "count": 249}, {"treatment_type": 4, "expect": 0.9886792452830189, "varSum": 2.966037735849056, "count": 265}], "overtime": [6, 6, 6, 6, 6, 6, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]}, {"agent_name": "row_gen_alp", "value_function": 5176766.931456326, "benchmark_value": 3133560.6074154405, "wait_time_by_type": [{"treatment_type": 0, "expect": 3.0895522388059704, "varSum": 1527.4626865671642, "count": 67}, {"treatment_type": 1, "expect": 3.774193548387097, "varSum": 1232.8387096774195, "count": 62}, {"treatment_type": 2, "expect": 11.992537313432836, "varSum": 8586.992537313432, "count": 134}, {"treatment_type": 3, "expect": 21.814345991561172, "varSum": 3891.8312236286956, "count": 237}, {"treatment_type": 4, "expect": 23.91071428571428, "varSum": 122.76785714285734, "count": 280}], "waiting_time_target_violations": [{"treatment_type": 0, "expect": 0.23880597014925373, "varSum": 12.17910447761194, "count": 67}, {"treatment_type": 1, "expect": 0.3548387096774194, "varSum": 14.193548387096774, "count": 62}, {"treatment_type": 2, "expect": 0.7761194029850746, "varSum": 23.28358208955224, "count": 134}, {"treatment_type": 3, "expect": 1.0, "varSum": 0.0, "count": 237}, {"treatment_type": 4, "expect": 1.0, "varSum": 0.0, "count": 280}], "overtime": [6, 6, 6, 6, 1, 6, 6, 4, 5, 6, 6, 4, 6, 6, 6, 6, 2, 6, 6, 5, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0]}], "uid": "49d90f3650fbd45d40d6acd2338695b9", "experiment_name": "occupancy_level", "param_value": 0.95}
def combine_files(directory_path, write_file_path):
    pattern = os.path.join(directory_path, '[0-9]*.jsonl')
    jsonl_files = glob.glob(pattern)
    count = 0
    uids = set()
    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            for line in f:
                values = []
                try:
                    data = json.loads(line)
                    for res in data.get('result', []):
                        if res['agent_name'] == 'hindsight_approx':
                            values.append(res)
                    data['result'] = values
                    with open(write_file_path, 'a') as f:
                        f.write(json.dumps(data) + '\n')
                except Exception as e:
                    print(f"Skipping malformed or incomplete line: {line.strip()} - Error: {e}")

def extract_lower_bound_from_file(directory_path, write_file_path):
    result_by_uid = {}
    pattern = os.path.join(directory_path, '[0-9]*.jsonl')
    jsonl_files = glob.glob(pattern)
    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'benchmark_value' in data:
                    uid = data.get('uid')
                    benchmark_value = data.get('benchmark_value')
                    result_by_uid[uid] = data
    print(result_by_uid)
    jsonl_files = glob.glob(pattern)
    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data:
                        for res in data.get('result', []):
                            res["benchmark_value"] = result_by_uid[data["uid"]]['benchmark_value']
                        with open(write_file_path, 'a') as f:
                            f.write(json.dumps(data) + '\n')
                except Exception as e:
                    print(f"Skipping malformed or incomplete line: {line.strip()} - Error: {e}")

def extract_penalized_lower_bound_from_file(directory_path, write_file_path):
    result_by_uid = {}
    pattern = os.path.join(directory_path, '[0-9]*.jsonl')
    jsonl_files = glob.glob(pattern)
    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                if 'penalized_lower_bound' in data:
                    uid = data.get('uid')
                    benchmark_value = data.get('penalized_lower_bound')
                    result_by_uid[uid] = data
    print(result_by_uid)
    jsonl_files = glob.glob(pattern)
    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data:
                        for res in data.get('result', []):
                            res["penalized_lower_bound"] = result_by_uid[data["uid"]]['penalized_lower_bound']
                        with open(write_file_path, 'a') as f:
                            f.write(json.dumps(data) + '\n')
                except Exception as e:
                    print(f"Skipping malformed or incomplete line: {line.strip()} - Error: {e}")

def rename_keys_in_file(directory_path, write_file_path):
    pattern = os.path.join(directory_path, '[0-9]*.jsonl')
    jsonl_files = glob.glob(pattern)
    for file_path in jsonl_files:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'result' in data:
                        for res in data.get('result', []):
                            res["benchmark_value"] = res['perfect_info_lower_bound']
                            res.pop('perfect_info_lower_bound', None)
                        with open(write_file_path, 'a') as f:
                            f.write(json.dumps(data) + '\n')
                except Exception as e:
                    print(f"Skipping malformed or incomplete line: {line.strip()} - Error: {e}")


if __name__ == '__main__':
    # Define the path to your results file
    plot_labels = {
        #'hindsight_approx_with_penalty': 'Hindsight Policy with Penalty',
        'hindsight_approx': 'Hindsight Policy',
        'myopic': 'Myopic Policy',
        #'col_gen_alp': 'ALP',
        'row_gen_alp': 'ALP',
        'benchmark_value': "Lower Bound",
        #'penalized_lower_bound': "Penalized Lower Bound",

    }
    experiment_lables = {'demand_rate': 'Arrival Rate',
                         'decision_epoch': 'Decision Epoch',
                         'overtime_cost_by_day':'Overtime Cost',
                         'occupancy_level': 'Occupancy Level',
                         'discount_factor': 'Discount Factor'}
    # Load and process the data
    # plot_experiment_result('results/demand_rate', plot_labels, experiment_lables)
    # extract_lower_bound_from_file('results/occupancy_level', write_file_path='results/inform_data/234_corrected_lower_bound.jsonl')
    #extract_penalized_lower_bound_from_file('results/occupancy_level', write_file_path='results/inform_data_with_penalized/234_penalized_lower_bound.jsonl')
    #rename_keys_in_file('results/demand_rate', write_file_path='results/inform_demand_rate_data/123_renamed_lower_bound.jsonl')
    #plot_experiment_result('results/demand_rate', plot_labels, experiment_lables)
    #combine_files('results/occupancy_level_backup', write_file_path='results/occupancy_level/123hindsight_approx_only.jsonl')
    #plot_experiment_result('results/discount_factor', plot_labels, experiment_lables)
    #plot_experiment_result('results/discount_factor_alp_only', plot_labels, experiment_lables)
    #plot_experiment_result('results/decision_epoch', plot_labels, experiment_lables)

    '''
    if not results_df.empty:
        # Analyze and plot the results for the 'demand_rate' experiment
        analyze_and_plot_results(results_df, experiment_name_filter='demand_rate')
    else:
        print("Could not generate plots because no data was loaded.")
    '''
    result = ExperimentResultBuilder(directory_path='results/occupancy_level_ejor_case_study',
                                     file_pattern='[0-9]*.jsonl',
                                     experiment_label='Occupancy level').build()
    result.plot()
