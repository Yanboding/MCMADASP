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
from utils import iter_to_tuple, iter_to_list, get_uid, read_lines_with_pattern, RunningStats, encode, decode
from pathlib import Path
import hashlib
import glob
import json
import copy
import os

def _generate_experiment_parameters(config_type, experiment_name, param_name, param_values, test_sample_path_num, request_path, base_env_args_overrides=None, param_modifier_fn=None):
    """
    A generic function to generate experiment parameter files.

    This function handles the core logic of iterating through parameter values and trials,
    generating unique configurations, and writing them to a request file.

    Args:
        experiment_name (str): The name of the current experiment.
        param_name (str): The key of the environment argument to modify.
                          Use dot notation for nested keys (e.g., 'reset_params.percentage_occupied').
        param_values (list): A list of values for the parameter to be tested.
        test_sample_path_num (int): The number of random trials to generate for each parameter value.
        request_path (str or Path): The path to the output request file.
        base_env_args_overrides (dict, optional): A dictionary of arguments to override in the base configuration.
        param_modifier_fn (function, optional): A function for complex parameter modifications.
                                                It should take (env_args, value) and return modified_env_args.

    Returns:
        dict: A dictionary of the generated parameters, keyed by their UID.
    """
    # Start with a base configuration.
    base_env_args = get_config_by_type(config_type).args
    if base_env_args_overrides:
        base_env_args.update(base_env_args_overrides)

    # Define a standard set of agent arguments.
    result_dict = {}
    lines_to_write = [] # Optimization: Collect lines to write in a list
    print(f"Generating parameters for {experiment_name}...")
    directory_path = os.path.join('experiments', 'results', experiment_name)
    alp_train_res = {}
    for line in read_lines_with_pattern(directory_path, 'alp_train*.jsonl'):
        line = json.loads(line)
        agent_type = line['result']['agent_name']
        alp_train_res[line['uid']+agent_type] = line['result']
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
        env_uid = get_uid(env_args_for_value)
        print(env_uid)
        # Generate multiple random trials for each parameter value.
        sample_path_stats = RunningStats()
        for command_id in range(test_sample_path_num):
            # 1. Generate the sample path with a specific, isolated random seed.
            sample_gen_args = copy.deepcopy(env_args_for_value)
            sample_gen_args['arrival_random_seed'] = command_id + 100 # Seed for sample path generation
            sample_gen_args['stop_time_random_seed'] = command_id + 400
            stop_time = 100
            config_for_sample_path = get_config_by_type('infinite_custom', args=sample_gen_args)
            env_for_sample_path = config_for_sample_path.env
            sample_path = env_for_sample_path.reset_arrivals(stop_time=stop_time) if env_for_sample_path else [[]]
            sample_path = sample_path.tolist() if hasattr(sample_path, 'tolist') else sample_path
            sample_path_stats += len(sample_path)
            col_alp_args = alp_train_res.get(env_uid+'col_gen_alp', {'agent_name': 'alp', 'args': {'coefficients': None}})
            row_alp_args = alp_train_res.get(env_uid+'row_gen_alp', {'agent_name': 'alp', 'args': {'coefficients': None}})
            agent_args = [
                {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 350, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False}},
                #{'agent_name': 'hindsight_approx_with_penalty', 'args': {'sample_path_number': 350, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'coeffecients':col_alp_args['args']['coefficients']}},
                {'agent_name': 'myopic', 'args': {}},
                #col_alp_args,
                row_alp_args
                ]

            # 2. Prepare the final parameters for the actual simulation run with a different seed.
            trial_env_args = copy.deepcopy(env_args_for_value)
            trial_env_args['arrival_random_seed'] = command_id + 200 # Seed for agent's future paths
            trial_env_args['env_random_seed'] = command_id + 300     # Seed for the main environment simulation

            parameter = {
                "experiment_name": experiment_name,
                "param_value": value,
                "sample_path": sample_path,
                "coefficients": row_alp_args['args']['coefficients'],
                "env_args": trial_env_args
            }
            
            # Generate a unique ID and save the parameter set.
            uid = get_uid(parameter)
            parameter["uid"] = uid
            parameter["agent_args"] = agent_args
            parameter_str = json.dumps(parameter)
            result_dict[uid] = parameter
            lines_to_write.append(parameter_str + '\n')
        print(sample_path_stats)
    # Optimization: Write all lines to the file at once to reduce I/O operations.
    print(f"Writing {len(lines_to_write)} parameters to {request_path}...")
    with open(request_path, "w") as f:
        f.writelines(lines_to_write)
    
    print("Parameter generation complete.")
    return result_dict

def generate_all_experiments(config_type, experiment_configs, test_sample_path_num_map, dat_file='table.dat', is_reuse=False):
    """
    Generates parameters for all defined experiments and writes them to a single DAT file.

    Args:
        experiment_configs (dict): A dictionary defining all available experiments.
        test_sample_path_num_map (dict): A map from experiment_name to the number of trials.
        dat_file (str): The name of the output DAT file for all commands.
        is_reuse (bool): If True, reuse existing request and result files.
    """
    all_requests = {}
    all_results = {}

    # --- 1. Generate or Load all Request Files ---
    print("--- Processing all experiments ---")
    for name, config in experiment_configs.items():
        request_path = config.get('request_path', f'{name}_request.jsonl')
        request_file = Path(request_path)
        
        # Ensure parent directory exists
        request_file.parent.mkdir(parents=True, exist_ok=True)

        if request_file.exists() and is_reuse:
            print(f"Reusing existing request file: {request_path}")
            with open(request_file, 'r') as f:
                for line in f:
                    request = json.loads(line)
                    all_requests[request['uid']] = request
        else:
            test_sample_path_num = test_sample_path_num_map.get(name, 2000) # Default to 2000 if not specified
            generated_requests = _generate_experiment_parameters(
                config_type=config_type,
                experiment_name=name,
                param_name=config['param_name'],
                param_values=config['param_values'],
                test_sample_path_num=test_sample_path_num,
                request_path=request_path,
                base_env_args_overrides=config.get('base_env_args_overrides'),
                param_modifier_fn=config.get('param_modifier_fn')
            )
            all_requests.update(generated_requests)

    # --- 2. Load all existing Result Files ---
    print("\n--- Loading all existing results ---")
    for name, config in experiment_configs.items():
        result_path = config.get('result_path', f'experiments/data_result/{name}_results.jsonl')
        result_file = Path(result_path)
        if result_file.exists():
            print(f"Found existing result file: {result_path}")
            with open(result_file, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        all_results[result['uid']] = result
                    except (json.JSONDecodeError, KeyError):
                        print(f"Warning: Skipping malformed or key-missing line in {result_path}")

    # --- 3. Generate a single DAT file for all pending requests ---
    print(f"\n--- Generating combined DAT file: {dat_file} ---")
    pending_requests = 0
    uids = list(all_requests.keys())
    random.shuffle(uids)
    with open(dat_file, 'w') as f:
        for uid in uids:
            parameter = all_requests[uid]
            if uid not in all_results:
                # Determine the correct output file for this specific parameter
                exp_name = parameter['experiment_name']
                exp_config = experiment_configs[exp_name]
                line = "python run.py --params '" + json.dumps(parameter) + "'\n"
                f.write(line)
                pending_requests += 1
    
    print(f"Generated {pending_requests} commands for pending requests across all experiments.")

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
                for train_type in ["row_gen"]:
                    line = "python run.py --params '" + json.dumps({'env_args':env_arg, 'experiment_name': name, "param_value": param_value, 'train_type': train_type}) + "'\n"
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
        for agent_arg in agent_args:
            save_params = {
                'uid': uid,
                'experiment_name': experiment_name,
                'agent_arg': agent_arg,
                **params
            }
            lines_to_write.append("python run.py --params '" + json.dumps(save_params) + "'\n")
    print("max sample path length:", max_length)
    with open(dat_file, 'w') as f:
        f.writelines(lines_to_write)


if __name__ == '__main__':
    # --- Define Experiment-Specific Logic ---

    def demand_rate_modifier(env_args, total_rate):
        """Modifier function for the demand rate experiment."""
        base_arrival_rates = np.array(env_args['arrival_rates'])
        total_arrival_rate_mean = np.sum(base_arrival_rates)
        
        if total_arrival_rate_mean == 0:
            # Avoid division by zero if base rates are all zero.
            num_types = len(base_arrival_rates) if len(base_arrival_rates) > 0 else 1
            type_probs = np.ones(num_types) / num_types
        else:
            type_probs = base_arrival_rates / total_arrival_rate_mean
        
        new_arrival_rates = total_rate * type_probs
        env_args['arrival_rates'] = new_arrival_rates.tolist()
        env_args['maximum_total_arrival'] = math.ceil(total_rate * 3)
        return env_args

    # --- Central Configuration for All Experiments ---
    '''
    EXPERIMENT_CONFIGS = {

        'demand_rate': {
            'param_name': 'total_arrival_rate',
            'param_values': [4, 8, 12],
            'param_modifier_fn': demand_rate_modifier
        },

        'decision_epoch': {
            'param_name': 'decision_epoch',
            'param_values': [20, 30],
        },

        'overtime_cost_by_day': {
            'param_name': 'overtime_cost_by_day',
            'param_values': [100, 150, 200],
        },

        'occupancy_level': {
            'param_name': 'reset_params.percentage_occupied',
            'param_values': [0.2, 0.5, 0.8],
            'base_env_args_overrides': {'decision_epoch': 30}
        }
    }
    '''
    '''
    'occupancy_level': {
        'config_type': 'ejor_default',
        'param_name': 'reset_params.percentage_occupied',
        'param_values': [0.5, 0.7, 0.9],
    },
    '''
    '''
    EXPERIMENT_CONFIGS = {
        'postponing_cost': {
            'config_type': 'ejor_default',
            'param_name': 'postponing_cost',
            'param_values': [500, 600, 700],
        }
    }
    '''
    EXPERIMENT_CONFIGS = {
        'toy_problem': {
            'config_type': 'toy',
            'param_name': 'reset_params.percentage_occupied',
            'param_values': [0],
        }
    }
    '''
    booking window size
    '''
    # --- Specify the number of trials for each experiment ---
    # You can customize the number of samples for each experiment here.
    TEST_SAMPLE_NUM_MAP = {
        'demand_rate': 2000,
        'decision_epoch': 2000,
        'overtime_cost_by_day': 2000,
        'occupancy_level': 2000,
        'discount_factor':2000,
        'percentage_occupied':2000,
        'postponing_cost': 2000
    }
    
    # --- Run All Experiments ---
    # generate_alp_train_params(
    #     experiment_configs=EXPERIMENT_CONFIGS,
    #     dat_file = 'table_alp_toy_train.dat',
    # )
    
    
    '''
    generate_all_experiments(
        config_type='ejor_default',
        experiment_configs=EXPERIMENT_CONFIGS,
        test_sample_path_num_map=TEST_SAMPLE_NUM_MAP,
        dat_file='table.dat',
        is_reuse=False # Set to True to avoid regenerating files and only create the .dat
    )
    '''
    
    generate_simulation_params(config_type='toy', 
                               experiment_name='toy_problem', 
                               warm_up_periods=250,
                               test_sample_path_num=5000,
                               num_periods=None,
                               dat_file='table.dat')
    
