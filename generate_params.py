import json
from pprint import pprint

import pandas as pd
import numpy as np
from environment import SchedulingEnv
from experiments import get_config_by_type
from utils import iter_to_tuple, iter_to_list
from pathlib import Path
import hashlib
import json
def get_uid(parameter):
    # Ensure the dict is serialized consistently
    param_str = json.dumps(parameter, sort_keys=True)
    return hashlib.md5(param_str.encode('utf-8')).hexdigest()

def different_demand_rate_request_df(total_rate_list, test_sample_path_num, request_path):
    env_args = get_config_by_type('base_case').args
    arrival_rates = np.array(env_args['arrival_rates'])
    total_arrival_rate_mean = np.sum(arrival_rates)
    type_probs = arrival_rates / total_arrival_rate_mean
    result = {}
    with open(request_path, "w") as f:
        for total_rate in total_rate_list:
            arrival_rates = total_rate * type_probs
            env_args['arrival_rates'] = arrival_rates.tolist()
            for command_id in range(test_sample_path_num):
                # random seed is not good
                env_args['arrival_random_seed'] = command_id
                env_args['env_random_seed'] = command_id + 1
                config = get_config_by_type('custom', args=env_args)
                env = config.env
                sample_path = env.reset_arrivals(t=1)
                agent_args = [{'agent_name': 'hindsight_approx', 'args':{'sample_path_number': 500,'current_decision_var_type':'integer', 'future_decision_var_type':'continuous'}}, {'agent_name': 'myopic', 'args':{}}]
                parameter = {
                    "sample_path": sample_path.tolist(),
                    "env_args": env_args,
                    "agent_args": agent_args,
                    }
                uid = get_uid(parameter)
                parameter["uid"] = uid
                parameter_str = json.dumps(parameter)
                result[uid] = parameter
                f.write(parameter_str + '\n')
    return result

def different_overtime_cost(overtime_cost_list, test_sample_path_num, request_path):
    env_args = get_config_by_type('base_case').args
    result = {}
    with open(request_path, "w") as f:
        for overtime_cost in overtime_cost_list:
            # Use random seed 0 to generate sample paths, then use different random seeds to generate sample paths for hindsight_approx agent
            env_args['overtime_cost_by_day'] = overtime_cost
            env_args['arrival_random_seed'] = 0
            config = get_config_by_type('custom', args=env_args)
            for command_id in range(test_sample_path_num):
                # random seed is not good
                env_args['arrival_random_seed'] = command_id+1
                env_args['env_random_seed'] = command_id + 2
                env = config.env
                sample_path = env.reset_arrivals(t=1)
                agent_args = [{'agent_name': 'hindsight_approx', 'args':{'sample_path_number': 500,'current_decision_var_type':'integer', 'future_decision_var_type':'continuous'}}, {'agent_name': 'myopic', 'args':{}}]
                parameter = {
                    "sample_path": sample_path.tolist(),
                    "env_args": env_args,
                    "agent_args": agent_args,
                    }
                uid = get_uid(parameter)
                parameter["uid"] = uid
                parameter_str = json.dumps(parameter)
                result[uid] = parameter
                f.write(parameter_str + '\n')
    return result

def generate_params(request_path, result_path, dat_file='table.dat', is_reuse=False):
    request_file = Path(request_path)
    if request_file.exists() and is_reuse:
        request_dict = {}
        with open(request_file, 'r') as f:
            for line in f:
                request = json.loads(line)
                request_dict[request['uid']] = request
    else:
        #request_dict = different_demand_rate_request_df(total_rate_list=[4, 8, 12], test_sample_path_num=10, request_path=request_path)
        request_dict = different_overtime_cost(overtime_cost_list=[150], test_sample_path_num=2000, request_path=request_path)
    result_file = Path(result_path)
    if result_file.exists() and is_reuse:
        result_dict = {}
        with open(result_file, 'r') as f:
            for line in f:
                result = json.loads(line)
                result_dict[result['uid']] = result
    else:
        result_dict = {}
    # Filter out requests that have already been processed
    with open(dat_file, 'w') as f:
        for k, parameter in request_dict.items(): 
            if k not in result_dict:
                parameter['output_file'] = result_path
                line = "python run.py --params '" + json.dumps(parameter) + "'\n"
                f.write(line)

if __name__ == '__main__':
    generate_params(request_path='different_overtime_cost_request.jsonl',
                    result_path='experiments/data_result/different_overtime_cost_results.jsonl',
                    dat_file='table.dat',
                    is_reuse=False)