import json
import pandas as pd
from environment import AdvanceSchedulingEnv
from experiments import Config
from utils import iter_to_tuple, iter_to_list
from pathlib import Path

def build_request_df(case_type):
    if case_type == 'real_scale':
        config = Config.from_real_scale()
    elif case_type == 'ejor':
        config = Config.from_EJOR_case()
    env_params = config.env_params
    t = 1
    env = AdvanceSchedulingEnv(**env_params)
    replication = 2000
    num_sample_path = 700
    df = []
    for command_id in range(replication):
        sample_path = env.reset_arrivals(t=t)
        sample_path_numbers = [1]+[i for i in range(100, num_sample_path+1, 100)]
        for sample_path_number in sample_path_numbers:
            parameter = {
                    "sample_path": sample_path.tolist(),
                    "sample_path_number": sample_path_number,
                    "command_id": command_id
                }
            df.append(parameter)
    df = pd.DataFrame(df)
    return df

def generate_params(request_path, result_path, dat_file='table.dat', case_type='ejor'):
    request_file = Path(request_path)
    if request_file.exists():
        request_df = pd.read_csv(request_file, index_col=0)
        request_df['sample_path'] = request_df['sample_path'].apply(eval)
    else:
        request_df = build_request_df(case_type)
        request_df.to_csv(request_file)
    result_file = Path(result_path)
    if result_file.exists():
        result_header = ['value', 'sample_path_number', 'hindsight_value', 'run_time', 'command_id']
        result_df = pd.read_csv(result_file, names=result_header, header=None)
        merge_df = pd.merge(request_df, result_df, on=['command_id', 'sample_path_number'], how='left', suffixes=('_expect', '_current'))
        df = merge_df.loc[merge_df['value'].isna(), request_df.columns.tolist()]
    else:
        df = request_df
    df = df.groupby("command_id", as_index=False).agg(
            sample_path        = ("sample_path", "first"),
            sample_path_number = ("sample_path_number", list)
        ).rename(columns={'sample_path_number': 'sample_path_numbers'})
    df['output_file'] = result_path
    df['case_type'] = case_type
    with open(dat_file, 'w') as f:
        for parameter in df.to_dict(orient='records'):
            line = "python run.py --params '" + json.dumps(parameter) + "'\n"
            f.write(line)



if __name__ == '__main__':
    generate_params('ejor_request.csv', 'ejor_policy_value.csv')