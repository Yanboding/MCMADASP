import os
import pandas as pd
from collections import defaultdict

from utils import RunningStat


data_path = os.path.join('.', 'real_scale_policy_value.csv')
result_header = ['value', 'sample_path_number', 'hindsight_value', 'command_id']
result_df = pd.read_csv(data_path, names=result_header, header=None)

value_dict = defaultdict(lambda: RunningStat(1))
hindsight_value_dict = defaultdict(lambda: RunningStat(1))
for i in range(len(result_df)):
    value, sample_path_number, hindsight_value, command_id = result_df.iloc[i]
    value_dict[sample_path_number].record(value)
    hindsight_value_dict[sample_path_number].record(hindsight_value)
df = pd.DataFrame([{
        'num_sample_path': int(sample_path_number),
        'SAAdvanceAgent': value_dict[sample_path_number].expect[0],
        'SAAdvanceAgent_lower': value_dict[sample_path_number].expect[0]-value_dict[sample_path_number].half_window(0.95)[0],
        'SAAdvanceAgent_upper': value_dict[sample_path_number].expect[0]+value_dict[sample_path_number].half_window(0.95)[0],
        'SAAdvanceAgent_hindsight': hindsight_value_dict[sample_path_number].expect[0],
        'SAAdvanceAgent_hindsight_lower': hindsight_value_dict[sample_path_number].expect[0]-hindsight_value_dict[sample_path_number].half_window(0.95)[0],
        'SAAdvanceAgent_hindsight_upper': hindsight_value_dict[sample_path_number].expect[0]+hindsight_value_dict[sample_path_number].half_window(0.95)[0],
    } for sample_path_number in value_dict])

print(df)
df.to_csv('value_function_data.csv')
