import json
import math
import random
import pickle
from pprint import pprint

import pandas as pd
import numpy as np
from scipy.stats import geom

from decision_maker import ALPEJORColumnGenerationAgent
from experiments import get_config_by_type
from utils import iter_to_tuple, iter_to_list, get_uid, read_lines_with_pattern, RunningStats, encode, decode, wait_time
from pathlib import Path
import hashlib
import glob
import json
import copy
import os

def _generate_alp_train_params(config_type, experiment_name, param_name, param_values, base_env_args_overrides=None, param_modifier_fn=None):
    # Start with a base configuration.
    base_env_args = get_config_by_type(config_type).args
    if base_env_args_overrides:
        base_env_args.update(base_env_args_overrides)
    print(f"Generating parameters for {experiment_name}...")

    # Iterate over each value of the parameter being tested.
    for value in param_values:
        # Optimization: Use copy.deepcopy for more efficient object copying.
        env_args_for_value = copy.deepcopy(base_env_args)

        # Modify the environment arguments for the current value.
        if param_modifier_fn:
            env_args_for_value = param_modifier_fn(env_args_for_value, value)
        else:
            # Handle simple or nested parameter updates using dot notation.
            keys = param_name.split('.')
            d = env_args_for_value
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = value
        yield env_args_for_value, value

def generate_alp_train_params(experiment_configs, dat_file):
    with open(dat_file, 'w') as f:
        for name, config in experiment_configs.items():
            for env_arg, param_value in  _generate_alp_train_params(
                                       config_type=config['config_type'],
                                       experiment_name=name,
                                       param_name=config['param_name'],
                                       param_values=config['param_values'],
                                       base_env_args_overrides=config.get('base_env_args_overrides'),
                                       param_modifier_fn=config.get('param_modifier_fn')):
                print(get_uid(env_arg))
                for i, train_type in enumerate(["row_gen"], start=1):
                    line = f"{i} python run.py --params '" + json.dumps({'env_args':env_arg, 'experiment_name': name, "param_value": param_value, 'train_type': train_type}) + "'\n"
                    f.write(line)

def generate_simulation_params(config_type, experiment_name, warm_up_periods, test_sample_path_num, num_periods, dat_file):
    '''
    1. start from config_type, get environment arguments
    2. create a copy of arguments for sample path simulation
    3. generate a sample path using the copied arguments
    4. make sure the occumency percantage is 0 in the reset parameter
    5. add warm_up_period to the final requrest body
    6. generate uid for this parameter
    7. for each agent, generate a line to evaluate the performance of the agent on this sample path
    '''
    env_args = get_config_by_type(config_type).args
    env_args['reset_params']['percentage_occupied'] = 0
    env_uid = get_uid(env_args)
    print(env_uid)
    alp_train_res = {}
    directory_path = os.path.join('experiments', 'results', experiment_name)
    for line in read_lines_with_pattern(directory_path, 'alp_train*.jsonl'):
        line = json.loads(line)
        agent_type = line['result']['agent_name']
        alp_train_res[line['uid']+agent_type] = line['result']
    col_alp_args = alp_train_res.get(env_uid+'col_gen_alp', {'agent_name': 'alp', 'args': {'coefficients': None}})
    row_alp_args = alp_train_res.get(env_uid+'row_gen_alp', {'agent_name': 'alp', 'args': {'coefficients': None}})
    coeffecients = row_alp_args['args']['coefficients']
    agent_args = [
                #{'agent_name': 'lowerbound', 'args': {'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'is_include_discount_factor':False}},
                #{'agent_name': 'myopic', 'args': {}},
                row_alp_args,             
                
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 128, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':False, 'max_periods':int(geom.ppf(0.995, 0.05)), "geom_p":0.05}},
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 128, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':True, 'max_periods':int(geom.ppf(0.995, 0.05)), "geom_p":0.05}},
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':False, 'max_periods':int(geom.ppf(0.995, 0.05)), "geom_p":0.05}},
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':True, 'max_periods':int(geom.ppf(0.995, 0.05)), "geom_p":0.05}},
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 512, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':False, 'max_periods':int(geom.ppf(0.995, 0.05)), "geom_p":0.05}},
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 512, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':True, 'max_periods':int(geom.ppf(0.995, 0.05)), "geom_p":0.05}},

                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':False, 'max_periods':int(geom.ppf(0.995, 0.1)), "geom_p":0.1}},
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':True, 'max_periods':int(geom.ppf(0.995, 0.1)), "geom_p":0.1}},
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':False, 'max_periods':int(geom.ppf(0.995, 0.02)), "geom_p":0.02}},
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':True, 'max_periods':int(geom.ppf(0.995, 0.02)), "geom_p":0.02}},
                
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':False, 'max_periods':int(geom.ppf(0.5, 0.05)), "geom_p":0.05}},
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':True, 'max_periods':int(geom.ppf(0.5, 0.05)), "geom_p":0.05}},
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':False, 'max_periods':int(geom.ppf(0.8, 0.05)), "geom_p":0.05}},
                # {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'sample_path_length': None, 'is_include_discount_factor':False, 'is_quasi_MC':True, 'max_periods':int(geom.ppf(0.8, 0.05)), "geom_p":0.05}},
                # {'agent_name': 'hindsight_approx_with_penalty', 'args': {'sample_path_number': 350, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'coeffecients':col_alp_args['args']['coefficients']}},
                
                #{'agent_name': 'penalized_lowerbound', 'args':{'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'is_include_discount_factor':False, 'coefficients':coeffecients}}
                #{'agent_name': 'penalized_lowerbound', 'args':{'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'is_include_discount_factor':False, 'coefficients':[1]}},
                #{'agent_name': 'penalized_lowerbound', 'args':{'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'is_include_discount_factor':False}}
                ]
    lines_to_write = []
    max_length = 0
    for command_id in range(test_sample_path_num):
        sample_gen_args = copy.deepcopy(env_args)
        sample_gen_args['arrival_random_seed'] = command_id + 100 # Seed for sample path generation
        sample_gen_args['stop_time_random_seed'] = command_id + 400
        config_for_sample_path = get_config_by_type('infinite_custom', args=sample_gen_args)
        env_for_sample_path = config_for_sample_path.env
        if num_periods is None:
            sample_path = env_for_sample_path.reset_arrivals(stop_time=warm_up_periods)
            additional_sample_path = env_for_sample_path.reset_arrivals()
            sample_path = np.append(sample_path, additional_sample_path, axis=0) if len(sample_path)>0 else additional_sample_path
        else:
            sample_path = env_for_sample_path.reset_arrivals(stop_time=num_periods)
        sample_path = sample_path.tolist() if hasattr(sample_path, 'tolist') else sample_path
        max_length = max(max_length, len(sample_path))
        params = {
            'sample_path': sample_path,
            'warm_up_periods':warm_up_periods,
            'env_args':env_args,
        }
        uid = get_uid(params)
        for i, agent_arg in enumerate(agent_args, start=1):
            save_params = {
                'uid': uid,
                'experiment_name': experiment_name,
                'agent_arg': agent_arg,
                **params
            }
            lines_to_write.append(f"{command_id+1} python run.py --params '" + json.dumps(save_params) + "'\n")
    print("max sample path length:", max_length)
    with open(dat_file, 'w') as f:
        f.writelines(lines_to_write)

def generate_simulation_paths(env_args, test_sample_path_num, init_state=None, agent_args=None, warm_up_periods=0, num_periods=None):
    sample_paths = []
    max_length = 0
    for command_id in range(test_sample_path_num):
        sample_gen_args = copy.deepcopy(env_args)
        sample_gen_args['arrival_random_seed'] = command_id + 100 # Seed for sample path generation
        sample_gen_args['stop_time_random_seed'] = command_id + 400
        config_for_sample_path = get_config_by_type('infinite_custom', args=sample_gen_args)
        env_for_sample_path = config_for_sample_path.env
        if init_state is None:
            init_state = env_for_sample_path.generate_initial_state()
        if num_periods is None:
            sample_path = env_for_sample_path.reset_arrivals(stop_time=warm_up_periods)
            additional_sample_path = env_for_sample_path.reset_arrivals()
            sample_path = np.append(sample_path, additional_sample_path, axis=0) if len(sample_path)>0 else additional_sample_path
        else:
            sample_path = env_for_sample_path.reset_arrivals(stop_time=num_periods)
        sample_paths.append(sample_path.tolist() if hasattr(sample_path, 'tolist') else sample_path)
        max_length = max(max_length, len(sample_path))
        params = {
            'init_state': init_state,
            'sample_path': sample_path,
            'warm_up_periods':warm_up_periods,
            'env_args':env_args,
        }
        uid = get_uid(params)
        if agent_args is None:
            agent_args = [None]
        for i, agent_arg in enumerate(agent_args, start=1):
            save_params = {
                'uid': uid,
                'agent_arg': agent_arg,
                **params
            }
    print("max sample path length:", max_length)
    return sample_paths

def generate_waiting_penalty_params(dat_file):
    # This function can be implemented to generate parameters for testing the impact of different waiting penalties on the performance of the agents.
    experiment_name = 'waiting_penalty_impact'
    config_type = 'toy'
    env_args = get_config_by_type(config_type).args
    l = [[[(0, 1, 0), (1, 3, 100)],
         [(0, 1, 0), (1, 3, 20)]],
        [[(0, 1, 0), (1, 3, 100)],
         [(0, 1, 0), (1, 3, 50)]],
        [[(0, 1, 0), (1, 3, 100)],
         [(0, 1, 0), (1, 3, 80)]]]
    lines_to_write = []
    for i, waiting_penalty in enumerate(l, start=1):
        holding_cost = [wait_time(waiting_penalty[i]) for i in range(len(waiting_penalty))]
        holding_cost = np.array(holding_cost).T
        env_args['holding_cost_by_day_by_type'] = holding_cost.tolist()
        save_params = {
                        'experiment_name': experiment_name,
                        'env_args': env_args,
                      }
        lines_to_write.append(f"{i} python run.py --params '" + json.dumps(save_params) + "'\n")
    with open(dat_file, 'w') as f:
        f.writelines(lines_to_write)

def generate_high_priority_arrival_rate(dat_file):
    # This function can be implemented to generate parameters for testing the impact of different high priority arrival rates on the performance of the agents.
    experiment_name = 'high_priority_arrival_rate_impact'
    config_type = 'toy'
    env_args = get_config_by_type(config_type).args
    arrival_rates = [[0.6, 2.4],
                     [1.5, 1.5],
                     [2.4, 0.6]]
    lines_to_write = []
    for i, arrival_rate in enumerate(arrival_rates, start=1):
        env_args['arrival_rates'] = arrival_rate
        save_params = {
                        'experiment_name': experiment_name,
                        'env_args': env_args,
                      }
        lines_to_write.append(f"{i} python run.py --params '" + json.dumps(save_params) + "'\n")
    with open(dat_file, 'w') as f:
        f.writelines(lines_to_write)

if __name__ == '__main__':
    # EXPERIMENT_CONFIGS = {
    #     'toy_problem': {
    #         'config_type': 'toy',
    #         'param_name': 'reset_params.percentage_occupied',
    #         'param_values': [0],
    #     }
    # }
    
    # --- Run All Experiments ---
    # generate_alp_train_params(
    #     experiment_configs=EXPERIMENT_CONFIGS,
    #     dat_file = 'table_alp_toy_train.dat',
    # )
    
    # generate_simulation_params(config_type='toy', 
    #                            experiment_name='toy_problem', 
    #                            warm_up_periods=250,
    #                            test_sample_path_num=1,
    #                            num_periods=None,
    #                            dat_file='table.dat')
    
    #generate_waiting_penalty_params(dat_file='table_waiting_penalty.dat')
    generate_high_priority_arrival_rate(dat_file='table_high_priority_arrival_rate.dat')
    
