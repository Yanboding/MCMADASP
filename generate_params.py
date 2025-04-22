import json
import pandas as pd
from environment import AdvanceSchedulingEnv
from experiments import Config
from utils import iter_to_tuple, iter_to_list
from pathlib import Path

def build_request_df():
    config = Config.from_real_scale()
    env_params = config.env_params
    init_state = config.init_state
    t = 1
    env = AdvanceSchedulingEnv(**env_params)
    replication = 1000
    num_sample_path = 1000
    df = []
    for command_id in range(replication):
        sample_path = env.reset_arrivals(t=t)
        sample_path_numbers = [1]+[i for i in range(200, num_sample_path+1, 200)]
        for sample_path_number in sample_path_numbers:
            parameter = {
                    "sample_path": sample_path.tolist(),
                    "sample_path_number": sample_path_number,
                    "command_id": command_id
                }
            df.append(parameter)
    df = pd.DataFrame(df)
    return df

def generate_params(request_path, result_path, dat_file='table.dat'):
    request_file = Path(request_path)
    if request_file.exists():
        request_df = pd.read_csv(request_file, index_col=0)
    else:
        request_df = build_request_df()
        request_df.to_csv(request_file)
    result_file = Path(result_path)
    if result_file.exists():
        result_header = ['value', 'sample_path_number', 'hindsight_value', 'command_id']
        result_df = pd.read_csv(result_file, names=result_header, header=None)
        merge_df = pd.merge(request_df, result_df, on=['command_id', 'sample_path_number'], how='left', suffixes=('_expect', '_current'))
        df = merge_df.loc[merge_df['value'].isna(), request_df.columns.tolist()].groupby("command_id", as_index=False).agg(
            sample_path        = ("sample_path", "first"),
            sample_path_number = ("sample_path_number", list)
        ).rename(columns={'sample_path_number': 'sample_path_numbers'})
    else:
        df = request_df
    with open(dat_file, 'w') as f:
        for parameter in df.to_dict(orient='records'):
            line = "python run.py --params '" + json.dumps(parameter) + "'\n"
            f.write(line)



if __name__ == '__main__':
    generate_params('request.csv', 'real_scale_policy_value.csv')