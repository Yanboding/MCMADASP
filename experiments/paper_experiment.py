import json
import os
from collections import defaultdict

import pandas as pd
import numpy as np
import glob

from utils import RunningStats
from visualization import approximate_value_plot_from_running_stats_dict


def plot_experiment_result(directory_path, plot_labels, experiment_lables):
    """
    Loads a .jsonl result file and processes it into a pandas DataFrame.

    Args:
        file_path (str): The path to the .jsonl file.

    Returns:
        pd.DataFrame: A DataFrame containing the processed results,
                      with one row per agent per simulation run.
    """
    print('parent_directory', directory_path)
    pct_opt_gap = defaultdict(lambda:defaultdict(lambda: RunningStats()))
    wait_time_by_type_by_agent = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: RunningStats())))
    total_average_wait_time = defaultdict(lambda: defaultdict(lambda: RunningStats()))
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
                try:
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
                            pct_opt_gap_val = agent_result['opt_gap']
                            if pct_opt_gap_val == float('inf'):
                                continue
                            pct_opt_gap[agent_name][param_value].record(pct_opt_gap_val)
                            for wait_time_by_type in agent_result['wait_time_by_type']:
                                treatment_type = wait_time_by_type['treatment_type']
                                treatment_types.add(treatment_type)
                                wait_time_stats = RunningStats(wait_time_by_type['count'], wait_time_by_type['expect'], wait_time_by_type['varSum'])
                                wait_time_by_type_by_agent[param_value][agent_name][treatment_type]+=wait_time_stats
                                total_average_wait_time[agent_name][param_value]+=wait_time_stats
                            for overtime in agent_result['overtime']:
                                total_overtime_by_day_by_agent[agent_name][param_value].record(overtime)
                except Exception as e:
                    print(f"Skipping malformed or incomplete line: {line.strip()} - Error: {e}")
    print("Sample path number:", count)
    x_values = sorted(list(x_values))
    os.path.join(directory_path, f'percentage_optimality_gap_by_{experiment_name}')
    approximate_value_plot_from_running_stats_dict(running_stats_dict=pct_opt_gap,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Percentage Optimality Gap (%)",
                                                   plot_labels=plot_labels,
                                                   title=None,
                                                   save_file=os.path.join(directory_path, f'percentage_optimality_gap_by_{experiment_name}'),
                                                   is_show_text=True,
                                                   is_set_x_color=False)

    approximate_value_plot_from_running_stats_dict(running_stats_dict=total_average_wait_time,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Waiting time (days)",
                                                   plot_labels=plot_labels,
                                                   title=None,
                                                   save_file=os.path.join(directory_path, f'average_wait_time_by_{experiment_name}'),
                                                   is_show_text=True,
                                                   is_set_x_color=False)

    approximate_value_plot_from_running_stats_dict(running_stats_dict=total_overtime_by_day_by_agent,
                                                   x_vals=x_values,
                                                   xticks=x_values,
                                                   xticklabels=x_values,
                                                   xlabel=experiment_lables[experiment_name],
                                                   ylabel="Number of Overtime (slots)",
                                                   plot_labels=plot_labels,
                                                   title=None,
                                                   save_file=os.path.join(directory_path, f'average_overtime_used_by_{experiment_name}'),
                                                   is_show_text=True,
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
                                                       save_file=os.path.join(directory_path, f'average_wait_time_type_{x_value_str}_{experiment_name}'),
                                                       is_show_text=False,
                                                       is_set_x_color=False)


if __name__ == '__main__':
    # Define the path to your results file
    plot_labels = {'alp': 'ALP Policy'}
    experiment_lables = {'demand_rate': 'Arrival Rate',
                         'decision_epoch': 'Decision Epoch',
                         'overtime_cost_by_day':'Overtime Cost',
                         'occupancy_level': 'Occupancy Level'}
    # Load and process the data
    #plot_experiment_result('results/demand_rate', plot_labels, experiment_lables)
    #plot_experiment_result('results/occupancy_level', plot_labels, experiment_lables)
    plot_experiment_result('results/demand_rate', plot_labels, experiment_lables)
    #plot_experiment_result('results/decision_epoch', plot_labels, experiment_lables)

    '''
    if not results_df.empty:
        # Analyze and plot the results for the 'demand_rate' experiment
        analyze_and_plot_results(results_df, experiment_name_filter='demand_rate')
    else:
        print("Could not generate plots because no data was loaded.")
    '''
