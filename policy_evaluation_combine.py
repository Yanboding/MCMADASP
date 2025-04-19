import os
import pandas as pd
from collections import defaultdict

from utils import RunningStat


data_path = os.path.join('.', 'real_scale_policy_value.csv')
result_header = ['value', 'sample_path_number']
result_df = pd.read_csv(data_path, names=result_header, header=None)

value_dict = defaultdict(lambda: RunningStat(1))
for i in range(len(result_df)):
    value, sample_path_number = result_df.iloc[i]
    value_dict[sample_path_number].record(value)
df = pd.DataFrame([{
        'num_sample_path': int(sample_path_number),
        'SAAdvanceAgent': running_stat.expect[0],
        'SAAdvanceAgent_lower': running_stat.expect[0]-running_stat.half_window(0.95)[0],
        'SAAdvanceAgent_upper': running_stat.expect[0]+running_stat.half_window(0.95)[0],
    } for sample_path_number, running_stat in value_dict.items()])

print(df)
df.to_csv('value_function_data.csv')