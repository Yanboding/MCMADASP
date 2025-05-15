import os

import numpy as np
import pandas as pd
from collections import defaultdict

from utils import RunningStat
from visualization import error_bar_plot, approximate_value_plot_from_running_stats, approximate_value_plot, \
    approximate_value_plot_from_running_stats_dict, error_bar_plot_from_running_stats_dict, \
    approximate_value_plot_from_multid_running_stats

file_name = 'adjust_ejor_policy_value.csv'
data_path = os.path.join('.', file_name)
type_num = 18
days = 66
agents = ['hindsight_approx', 'myopic']
column_path = os.path.join('.', 'columns.txt')
with open("columns.txt", "r", encoding="utf-8") as f:
    result_header = [line.strip() for line in f]
print(result_header)
result_df = pd.read_csv(data_path, names=result_header, header=None)
print(result_df)

value_by_agent = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: RunningStat(1))))
wait_time_by_type_by_agent = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: RunningStat(1)))))
overtime_by_day_by_agent = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: RunningStat(days))))
hindsight_lower_bound_dict = defaultdict(lambda:defaultdict(lambda: RunningStat(1)))
run_time_dict = defaultdict(lambda:defaultdict(lambda: RunningStat(1)))

for i in range(len(result_df)):
    row = result_df.iloc[i]
    sample_path_number = row['sample_path_number']
    occupancy_percentage = row['occupancy_percentage']
    for agent_name in agents:
        value_by_agent[occupancy_percentage][agent_name][sample_path_number].record(row[agent_name+'_value'])
        for j in range(type_num):
            wait_time_stats = RunningStat(1)
            wait_time_stats.expect = np.array([row['expect_' + agent_name + '_' + str(j)]])
            wait_time_stats.varSum = np.array([row['varSum_' + agent_name + '_' + str(j)]])
            wait_time_stats.count = row['count_' + agent_name + '_' + str(j)]
            wait_time_by_type_by_agent[occupancy_percentage][sample_path_number][agent_name][j].merge(wait_time_stats)
        overtime_columns = [agent_name+'_overtime_'+str(day) for day in range(days)]
        overtime_by_day_by_agent[occupancy_percentage][sample_path_number][agent_name].record(row[overtime_columns])
    hindsight_lower_bound_stats = RunningStat(1)
    hindsight_lower_bound_stats.expect = np.array([row['expect_hindsight_approx']])
    hindsight_lower_bound_stats.varSum = np.array([row['varSum_hindsight_approx']])
    hindsight_lower_bound_stats.count = row['count_hindsight_approx']
    #value_by_agent[occupancy_percentage]['lower_bound'][sample_path_number].merge(hindsight_lower_bound_stats)
    run_time_stats = RunningStat(1)
    run_time_stats.expect = np.array([row['expect_hindsight_approx_runtime']])
    run_time_stats.varSum = np.array([row['varSum_hindsight_approx_runtime']])
    run_time_stats.count = row['count_hindsight_approx_runtime']
    run_time_dict[occupancy_percentage][sample_path_number].merge(run_time_stats)
'''
for occupancy_percentage, value_by_agent_by_sample_path in value_by_agent.items():
    print(occupancy_percentage)
    sample_path_numbers = [300, 400, 500]
    if occupancy_percentage != 0.5:
        approximate_value_plot_from_running_stats_dict(running_stats_dict=value_by_agent[occupancy_percentage],
                                                      x_vals=sample_path_numbers,
                                                      xlabel='Sample Path Number',
                                                      ylabel='Value Function',
                                                      plot_labels={'hindsight_approx':'Hindsight Approx Policy', 'myopic': 'Myopic Policy', 'lower_bound':'Hindsight Approx Lower Bound'},
                                                      title='Occupancy: '+str(occupancy_percentage),
                                                      save_file='value_function_occ_'+str(int(occupancy_percentage*100)))

x_types = np.arange(type_num)
print(wait_time_by_type_by_agent.keys())
for occupancy_percentage, wait_time_by_sample_path in wait_time_by_type_by_agent.items():
    print(wait_time_by_sample_path[500])
    if occupancy_percentage != 0.5:
        error_bar_plot_from_running_stats_dict(running_stats_dict=wait_time_by_sample_path[500],
                       x_vals=x_types,
                       xlabel="Treatment type",
                       ylabel="Waiting time (days)",
                       plot_labels={'hindsight_approx':'Hindsight Approx Policy', 'myopic': 'Myopic Policy', 'lower_bound':'Hindsight Approx Lower Bound'},
                       save_file='wait_time_'+str(int(occupancy_percentage*100)))
'''
x_days = np.arange(days)
approximate_value_plot_from_multid_running_stats(running_stats_dict=overtime_by_day_by_agent[0.8][500],
                                          x_vals=x_days,
                                          xlabel='Days',
                                          ylabel='Number of slot used',
                                          plot_labels={'hindsight_approx':'Hindsight Approx Policy', 'myopic': 'Myopic Policy', 'lower_bound':'Hindsight Approx Lower Bound'},
                                          title='Occupancy: '+str(0.8),
                                          save_file='slot_used')
