import os
import pandas as pd
from collections import defaultdict

from utils import RunningStat

file_name = 'ejor_policy_value.csv'
data_path = os.path.join('.', file_name)
result_header = ['value', 'sample_path_number', 'hindsight_value', 'run_time', 'command_id']
result_df = pd.read_csv(data_path, names=result_header, header=None)

value_dict = defaultdict(lambda: RunningStat(1))
hindsight_value_dict = defaultdict(lambda: RunningStat(1))
run_time_dict = defaultdict(lambda: RunningStat(1))
for i in range(len(result_df)):
    value, sample_path_number, hindsight_value, run_time, command_id = result_df.iloc[i]
    value_dict[sample_path_number].record(value)
    hindsight_value_dict[sample_path_number].record(hindsight_value)
    run_time_dict[sample_path_number].record(run_time)
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
df.to_csv('value_function_data.csv')
