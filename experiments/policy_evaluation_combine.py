import os

import numpy as np
import pandas as pd
from collections import defaultdict

from utils import RunningStat
from visualization import error_bar_plot, approximate_value_plot_from_running_stats, approximate_value_plot

file_name = 'adjust_ejor_policy_value.csv'
data_path = os.path.join('.', 'experiments', file_name)
type_num = 18
days = 56
result_header = ['value', 'sample_path_number', 'hindsight_value', 'run_time', 'command_id', 'expect_0', 'varSum_0', 'count_0', 'expect_1', 'varSum_1', 'count_1', 'expect_2', 'varSum_2', 'count_2', 'expect_3', 'varSum_3', 'count_3', 'expect_4', 'varSum_4', 'count_4', 'expect_5', 'varSum_5', 'count_5', 'expect_6', 'varSum_6', 'count_6', 'expect_7', 'varSum_7', 'count_7', 'expect_8', 'varSum_8', 'count_8', 'expect_9', 'varSum_9', 'count_9', 'expect_10', 'varSum_10', 'count_10', 'expect_11', 'varSum_11', 'count_11', 'expect_12', 'varSum_12', 'count_12', 'expect_13', 'varSum_13', 'count_13', 'expect_14', 'varSum_14', 'count_14', 'expect_15', 'varSum_15', 'count_15', 'expect_16', 'varSum_16', 'count_16', 'expect_17', 'varSum_17', 'count_17', 'slot_number_0', 'slot_number_1', 'slot_number_2', 'slot_number_3', 'slot_number_4', 'slot_number_5', 'slot_number_6', 'slot_number_7', 'slot_number_8', 'slot_number_9', 'slot_number_10', 'slot_number_11', 'slot_number_12', 'slot_number_13', 'slot_number_14', 'slot_number_15', 'slot_number_16', 'slot_number_17', 'slot_number_18', 'slot_number_19', 'slot_number_20', 'slot_number_21', 'slot_number_22', 'slot_number_23', 'slot_number_24', 'slot_number_25', 'slot_number_26', 'slot_number_27', 'slot_number_28', 'slot_number_29', 'slot_number_30', 'slot_number_31', 'slot_number_32', 'slot_number_33', 'slot_number_34', 'slot_number_35', 'slot_number_36', 'slot_number_37', 'slot_number_38', 'slot_number_39', 'slot_number_40', 'slot_number_41', 'slot_number_42', 'slot_number_43', 'slot_number_44', 'slot_number_45', 'slot_number_46', 'slot_number_47', 'slot_number_48', 'slot_number_49', 'slot_number_50', 'slot_number_51', 'slot_number_52', 'slot_number_53', 'slot_number_54', 'slot_number_55']
result_df = pd.read_csv(data_path, names=result_header, header=None)
print(result_df)
value_dict = defaultdict(lambda: RunningStat(1))
hindsight_value_dict = defaultdict(lambda: RunningStat(1))
run_time_dict = defaultdict(lambda: RunningStat(1))
wait_time_by_type = defaultdict(lambda:defaultdict(lambda: RunningStat(1)))
slot_number_by_day = defaultdict(lambda: RunningStat(days))
slot_number_columns = ['slot_number_'+str(d) for d in range(days)]
for i in range(len(result_df)):
    row = result_df.iloc[i]
    sample_path_number = row['sample_path_number']
    value_dict[sample_path_number].record(row['value'])
    hindsight_value_dict[sample_path_number].record(row['hindsight_value'])
    run_time_dict[sample_path_number].record(row['run_time'])
    for j in range(type_num):
        wait_time_stats = RunningStat(1)
        wait_time_stats.expect = np.array([row['expect_'+str(j)]])
        wait_time_stats.varSum = np.array([row['varSum_' + str(j)]])
        wait_time_stats.count = row['count_' + str(j)]
        wait_time_by_type[sample_path_number][j].merge(wait_time_stats)
    slot_number_by_day[sample_path_number].record(row[slot_number_columns])
df = pd.DataFrame([{
        'num_sample_path': int(sample_path_number),
        'SAAdvanceAgent': value_dict[sample_path_number].expect[0],
        'SAAdvanceAgent_lower': value_dict[sample_path_number].expect[0]-value_dict[sample_path_number].half_window(0.95)[0],
        'SAAdvanceAgent_upper': value_dict[sample_path_number].expect[0]+value_dict[sample_path_number].half_window(0.95)[0],
        'SAAdvanceAgent_hindsight': hindsight_value_dict[sample_path_number].expect[0],
        'SAAdvanceAgent_hindsight_lower': hindsight_value_dict[sample_path_number].expect[0]-hindsight_value_dict[sample_path_number].half_window(0.95)[0],
        'SAAdvanceAgent_hindsight_upper': hindsight_value_dict[sample_path_number].expect[0]+hindsight_value_dict[sample_path_number].half_window(0.95)[0],
        'SAAdvanceAgent_runtime': run_time_dict[sample_path_number].expect[0],
        'SAAdvanceAgent_runtime_lower': run_time_dict[sample_path_number].expect[0]-run_time_dict[sample_path_number].half_window(0.95)[0],
        'SAAdvanceAgent_runtime_upper': run_time_dict[sample_path_number].expect[0]+run_time_dict[sample_path_number].half_window(0.95)[0],
    } for sample_path_number in value_dict])
print(df)
approximate_value_plot(df,
                           xlabel='Number of Sample Paths',
                           ylabel='Value function',
                           approx_labels=['SAAdvanceAgent', 'SAAdvanceAgent_hindsight'],
                           text_labels=['SAAdvanceAgent', 'SAAdvanceAgent_hindsight'],
                           plot_labels={'SAAdvanceAgent': "Hindsight Policy Performance", 'SAAdvanceAgent_hindsight': "Lower Bound Sample Average Approximation"},
                           save_file='case_study_value_function',
                           x_val_col='num_sample_path')
approximate_value_plot(df,
                       xlabel='Number of Sample Paths',
                       ylabel='Seconds',
                       approx_labels=['SAAdvanceAgent_runtime'],
                       text_labels=['SAAdvanceAgent_runtime'],
                       plot_labels={'SAAdvanceAgent_runtime': "Advance Hindsight Policy"},
                       save_file='case_study_run_time',
                       x_val_col='num_sample_path')
x_types = np.arange(type_num)
error_bar_plot(stats_by_sample_path=wait_time_by_type,
               x_vals=x_types,
               xlabel="Treatment type",
               ylabel="Waiting time (days)",
               plot_labels={500: 'sample path number: 500'},
               save_file='wait_time')

x_days = np.arange(days)
approximate_value_plot_from_running_stats(running_stats=slot_number_by_day,
                                          x_vals=x_days,
                                          xlabel='Days',
                                          ylabel='Number of slot used',
                                          plot_labels={500: 'sample path number: 500'},
                                          save_file='slot_used')
