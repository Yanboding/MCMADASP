import os
from collections import defaultdict

import numpy as np
import pandas as pd

from utils import RunningStat
from visualization import approximate_value_plot, error_bar_plot_from_running_stats_dict, \
    approximate_value_plot_from_running_stats_dict

file_name = 'adjust_ejor_policy_value.csv'
data_path = os.path.join('.', file_name)
type_num = 18
days = 56
lower_bound_replication = 30
agents = ['hindsight_approx', 'myopic']
column_path = os.path.join('.', 'columns.txt')
with open("columns.txt", "r", encoding="utf-8") as f:
    result_header = [line.strip() for line in f]
print(result_header)
result_df = pd.read_csv(data_path, names=result_header, header=None)
print(result_df)

improvement_by_agent = defaultdict(lambda:defaultdict(lambda: RunningStat(1)))
wait_time_by_type_by_agent = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: RunningStat(1))))
total_average_wait_time = defaultdict(lambda:defaultdict(lambda: RunningStat(1)))
total_overtime_by_day_by_agent = defaultdict(lambda:defaultdict(lambda: RunningStat(1)))
for i in range(len(result_df)):
    row = result_df.iloc[i]
    sample_path_number = row['sample_path_number']
    occupancy_percentage = row['occupancy_percentage']
    for agent_name in agents:
        for r in range(lower_bound_replication):
            hindsight_lower_bound = row['hindsight_lower_bound_'+str(r)]
            improvement = (row[agent_name+'_value'] - hindsight_lower_bound) / hindsight_lower_bound * 100
            improvement_by_agent[agent_name][occupancy_percentage].record(improvement)
        for j in range(type_num):
            wait_time_stats = RunningStat(1)
            wait_time_stats.expect = np.array([row['expect_' + agent_name + '_' + str(j)]])
            wait_time_stats.varSum = np.array([row['varSum_' + agent_name + '_' + str(j)]])
            wait_time_stats.count = row['count_' + agent_name + '_' + str(j)]
            wait_time_by_type_by_agent[occupancy_percentage][agent_name][j].merge(wait_time_stats)
            total_average_wait_time[agent_name][occupancy_percentage].merge(wait_time_stats)
        for day in range(days):
            total_overtime_by_day_by_agent[agent_name][occupancy_percentage].record(row[agent_name+'_overtime_'+str(day)])
occupancy_percentages = [0.2, 0.5, 0.8]

approximate_value_plot_from_running_stats_dict(running_stats_dict=improvement_by_agent,
                   x_vals=sorted(occupancy_percentages),
                   xticklabels=sorted(occupancy_percentages),
                   xlabel="Occupancy Percentage",
                   ylabel="Percentage Optimality Gap (%)",
                   plot_labels={'hindsight_approx':'Hindsight Approx Policy', 'myopic': 'Myopic Policy'},
                   title=None,
                   save_file='percentage_optimality_gap',
                   is_show_text=True)

approximate_value_plot_from_running_stats_dict(running_stats_dict=total_average_wait_time,
                   x_vals=sorted(occupancy_percentages),
                   xticklabels=sorted(occupancy_percentages),
                   xlabel="Occupancy Percentage",
                   ylabel="Waiting time (days)",
                   plot_labels={'hindsight_approx':'Hindsight Approx Policy', 'myopic': 'Myopic Policy'},
                   title=None,
                   save_file='average_wait_time',
                   is_show_text=True)

approximate_value_plot_from_running_stats_dict(running_stats_dict=total_overtime_by_day_by_agent,
                   x_vals=sorted(occupancy_percentages),
                   xticklabels=sorted(occupancy_percentages),
                   xlabel="Occupancy Percentage",
                   ylabel="Number of Overtime (slots)",
                   plot_labels={'hindsight_approx':'Hindsight Approx Policy', 'myopic': 'Myopic Policy'},
                   title=None,
                   save_file='average_overtime_used',
                   is_show_text=True)

x_types = np.arange(type_num)
for occupancy_percentage in occupancy_percentages:
    approximate_value_plot_from_running_stats_dict(running_stats_dict=wait_time_by_type_by_agent[occupancy_percentage],
                                                   x_vals=x_types,
                                                   xticklabels=x_types + 1,
                                                   xlabel='Treatment type',
                                                   ylabel="Waiting time (days)",
                                                   plot_labels={'hindsight_approx':'Hindsight Approx Policy', 'myopic': 'Myopic Policy'},
                                                   title="Waiting time with Capacity Occupancy: " + str(occupancy_percentage),
                                                   save_file='average_wait_time_type_'+str(int(occupancy_percentage*100)),
                                                   is_show_text=False)