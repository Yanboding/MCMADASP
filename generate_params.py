import json
from pprint import pprint

import pandas as pd
from environment import AdvanceSchedulingEnv
from experiments import Config
from experiments.config import get_config_by_type
from utils import iter_to_tuple, iter_to_list
from pathlib import Path

def build_request_df(case_type):
    config = get_config_by_type(case_type)
    env_params = config.env_params
    t = 1
    env = AdvanceSchedulingEnv(**env_params)
    test_sample_path_num = 2000
    replication = 10
    sample_path_numbers = [300, 400, 500]
    occ_pcts = [0, 0.2, 0.5, 0.8]
    df = []
    for command_id in range(test_sample_path_num):
        sample_path = env.reset_arrivals(t=t)
        for sample_path_number in sample_path_numbers:
            for occ_pct in occ_pcts:
                parameter = {
                        "sample_path": sample_path.tolist(),
                        "sample_path_number": sample_path_number,
                        "command_id": command_id,
                        "replication": replication,
                        "occupancy_percentage": occ_pct,
                        "agents": [{'agent_name': 'hindsight_approx', 'args':{}}, {'agent_name': 'myopic', 'args':{}}]
                    }
                df.append(parameter)
    df = pd.DataFrame(df)
    return df

def generate_params(request_path, result_path, dat_file='table.dat', case_type='ejor', is_reuse=False):
    '''
        {
        'sample_path': list, # the sample path we want to evaluate
        'sample_path_numbers': list, # the number of sample path to generate to estimate SA
        'replication': int, # for each sample path number, how many replications to do for lower bound
        'agents': dict # agent name with its parameters default is {}. The agents we want to test
        'occupancy_percentage': float # number of percentage
        'command_id': id,
        'output_file': str,
        'case_type': str,
        }
    '''
    request_file = Path(request_path)
    if request_file.exists() and is_reuse:
        request_df = pd.read_csv(request_file, index_col=0)
        request_df['sample_path'] = request_df['sample_path'].apply(eval)
        request_df['agents'] = request_df['agents'].apply(eval)
    else:
        request_df = build_request_df(case_type)
        request_df.to_csv(request_file)
    result_file = Path(result_path)
    if result_file.exists() and is_reuse:
        with open("columns.txt", "r", encoding="utf-8") as f:
            result_header = [line.strip() for line in f]
        result_df = pd.read_csv(result_file, names=result_header, header=None, index_col=False)
        merge_df = pd.merge(request_df, result_df, on=['command_id', 'sample_path_number','occupancy_percentage'], how='left', suffixes=('_expect', '_current'))
        df = merge_df.loc[merge_df['hindsight_approx_value'].isna(), request_df.columns.tolist()]
    else:
        df = request_df
    df = df.groupby(["command_id", "occupancy_percentage"], as_index=False).agg(
            sample_path        = ("sample_path", "first"),
            sample_path_number = ("sample_path_number", list),
            replication = ("replication", "first"),
            agents = ("agents", "first")
        ).rename(columns={'sample_path_number': 'sample_path_numbers'})
    df['output_file'] = result_path
    df['case_type'] = case_type
    with open(dat_file, 'w') as f:
        for parameter in df.to_dict(orient='records'):
            line = "python run.py --params '" + json.dumps(parameter) + "'\n"
            f.write(line)

if __name__ == '__main__':
    generate_params(request_path='adjust_ejor_request.csv',
                    result_path='experiments/adjust_ejor_policy_value.csv',
                    dat_file='table.dat',
                    case_type='adjust_ejor',
                    is_reuse=True)

