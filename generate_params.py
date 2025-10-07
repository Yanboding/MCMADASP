import json
import math
from pprint import pprint

import pandas as pd
import numpy as np

from decision_maker import ALPEJORColumnGenerationAgent
from experiments import get_config_by_type
from utils import iter_to_tuple, iter_to_list, get_uid, read_lines_with_pattern
from pathlib import Path
import hashlib
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
    agent_args = [
        {'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 350, 'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False}},
        {'agent_name': 'myopic', 'args': {'is_myopic': True}},
        {'agent_name': 'alp', 'args': {'coefficients': [-195284.271323261, 123.537317331, 122.301944158, 121.078924717, 119.868135469, 118.669454115, 117.482759574, 116.307931978, 115.144852658, 113.993404131, 112.85347009, 111.724935389, 110.607686035, 109.501609175, 108.406593083, 107.322527152, 106.249301881, 105.186808862, 104.134940773, 103.093591366, 102.062655452, 101.042028898, 100.031608609, 99.031292522, 98.040979597, 97.060569801, 96.089964103, 95.129064462, 94.177773818, 93.235996079, 92.303636119, 91.380599757, 90.46679376, 89.562125822, 88.666504564, 87.779839518, 86.902041123, 86.033020712, 85.172690505, 84.3209636, 83.477753964, 82.642976424, 81.81654666, 80.998381193, 80.188397381, 79.386513408, 78.592648274, 77.806721791, 77.028654573, 76.258368027, 75.495784347, 74.740826503, 73.993418238, 73.253484056, 72.520949215, 71.795739723, 71.077782326, 70.367004503, 69.663334458, 68.966701113, 68.277034102, 67.594263761, 66.918321123, 66.249137912, 65.586646533, 64.930780068, 64.281472267, 63.638657544, 63.002270969, 62.372248259, 61.748525777, 61.131040519, 60.519730114, 59.914532813, 59.315387484, 58.72223361, 58.135011273, 57.553661161, 56.978124549, 56.408343304, 55.844259871, 55.285817272, 54.732959099, 54.185629508, 53.643773213, 53.107335481, 0.0, 23.537317331, 23.301944158, 23.068924717, 22.838235469, 22.609853115, 22.383754584, 22.159917038, 21.938317867, 21.718934689, 21.501745342, 21.286727888, 21.073860609, 20.863122003, 20.654490783, 20.447945875, 20.243466417, 20.041031753, 19.840621435, 19.642215221, 19.445793068, 19.251335138, 19.058821786, 18.868233569, 18.679551233, 18.492755721, 18.307828163, 18.124749882, 17.943502383, 17.764067359, 17.586426685, 17.410562419, 17.236456794, 17.064092226, 16.893451304, 16.724516791, 16.557271623, 16.391698907, 16.227781918, 16.065504099, 15.904849058, 15.745800567, 15.588342562, 15.432459136, 15.278134545, 15.125353199, 14.974099667, 14.82435867, 14.676115084, 14.529353933, 14.384060393, 14.24021979, 14.097817592, 13.956839416, 13.817271021, 13.679098311, 13.542307328, 13.406884255, 13.272815412, 13.140087258, 2e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 728.993093121, 247.074634663, 610.323639007, 1789.204068776, 2263.926831342, 3465.554667463, 237.338908229, 700.26785657, 1253.362150343, 586.274452439, 1779.349320377, 118.669454115, 2279.557885455, 7157.009484284, 3298.362359767, 3617.528300023, 3699.344846683, 3298.362359767]}}

    ]
    result_dict = {}
    lines_to_write = [] # Optimization: Collect lines to write in a list
    print(f"Generating parameters for {experiment_name}...")
    directory_path = os.path.join('experiments', 'results', experiment_name)
    alp_train_res = {}
    for line in read_lines_with_pattern(directory_path, 'alp_train*.jsonl'):
        line = json.loads(line)
        alp_train_res[line['uid']] = line['result']
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
        uid = get_uid(env_args_for_value)
        #agent_args.append(alp_train_res.get(uid, {'agent_name': 'alp', 'args': {'coefficients': [-195284.271323261, 123.537317331, 122.301944158, 121.078924717, 119.868135469, 118.669454115, 117.482759574, 116.307931978, 115.144852658, 113.993404131, 112.85347009, 111.724935389, 110.607686035, 109.501609175, 108.406593083, 107.322527152, 106.249301881, 105.186808862, 104.134940773, 103.093591366, 102.062655452, 101.042028898, 100.031608609, 99.031292522, 98.040979597, 97.060569801, 96.089964103, 95.129064462, 94.177773818, 93.235996079, 92.303636119, 91.380599757, 90.46679376, 89.562125822, 88.666504564, 87.779839518, 86.902041123, 86.033020712, 85.172690505, 84.3209636, 83.477753964, 82.642976424, 81.81654666, 80.998381193, 80.188397381, 79.386513408, 78.592648274, 77.806721791, 77.028654573, 76.258368027, 75.495784347, 74.740826503, 73.993418238, 73.253484056, 72.520949215, 71.795739723, 71.077782326, 70.367004503, 69.663334458, 68.966701113, 68.277034102, 67.594263761, 66.918321123, 66.249137912, 65.586646533, 64.930780068, 64.281472267, 63.638657544, 63.002270969, 62.372248259, 61.748525777, 61.131040519, 60.519730114, 59.914532813, 59.315387484, 58.72223361, 58.135011273, 57.553661161, 56.978124549, 56.408343304, 55.844259871, 55.285817272, 54.732959099, 54.185629508, 53.643773213, 53.107335481, 0.0, 23.537317331, 23.301944158, 23.068924717, 22.838235469, 22.609853115, 22.383754584, 22.159917038, 21.938317867, 21.718934689, 21.501745342, 21.286727888, 21.073860609, 20.863122003, 20.654490783, 20.447945875, 20.243466417, 20.041031753, 19.840621435, 19.642215221, 19.445793068, 19.251335138, 19.058821786, 18.868233569, 18.679551233, 18.492755721, 18.307828163, 18.124749882, 17.943502383, 17.764067359, 17.586426685, 17.410562419, 17.236456794, 17.064092226, 16.893451304, 16.724516791, 16.557271623, 16.391698907, 16.227781918, 16.065504099, 15.904849058, 15.745800567, 15.588342562, 15.432459136, 15.278134545, 15.125353199, 14.974099667, 14.82435867, 14.676115084, 14.529353933, 14.384060393, 14.24021979, 14.097817592, 13.956839416, 13.817271021, 13.679098311, 13.542307328, 13.406884255, 13.272815412, 13.140087258, 2e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 1e-09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 728.993093121, 247.074634663, 610.323639007, 1789.204068776, 2263.926831342, 3465.554667463, 237.338908229, 700.26785657, 1253.362150343, 586.274452439, 1779.349320377, 118.669454115, 2279.557885455, 7157.009484284, 3298.362359767, 3617.528300023, 3699.344846683, 3298.362359767]}}))
        # Generate multiple random trials for each parameter value.
        for command_id in range(test_sample_path_num):
            # 1. Generate the sample path with a specific, isolated random seed.
            sample_gen_args = copy.deepcopy(env_args_for_value)
            sample_gen_args['arrival_random_seed'] = command_id + 1 # Seed for sample path generation
            sample_gen_args['stop_time_random_seed'] = command_id + 4
            
            config_for_sample_path = get_config_by_type('infinite_custom', args=sample_gen_args)
            env_for_sample_path = config_for_sample_path.env
            sample_path = env_for_sample_path.reset_arrivals() if env_for_sample_path else [[]]

            # 2. Prepare the final parameters for the actual simulation run with a different seed.
            trial_env_args = copy.deepcopy(env_args_for_value)
            trial_env_args['arrival_random_seed'] = command_id + 2 # Seed for agent's future paths
            trial_env_args['env_random_seed'] = command_id + 3     # Seed for the main environment simulation

            parameter = {
                "experiment_name": experiment_name,
                "param_value": value,
                "sample_path": sample_path.tolist() if hasattr(sample_path, 'tolist') else sample_path,
                "env_args": trial_env_args
            }
            
            # Generate a unique ID and save the parameter set.
            uid = get_uid(parameter)
            parameter["uid"] = uid
            parameter["agent_args"] = agent_args
            parameter_str = json.dumps(parameter)
            result_dict[uid] = parameter
            lines_to_write.append(parameter_str + '\n')

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
    with open(dat_file, 'w') as f:
        for uid, parameter in all_requests.items():
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
        yield env_args_for_value

def generate_alp_train_params(experiment_configs, dat_file):
    with open(dat_file, 'w') as f:
        for name, config in experiment_configs.items():
            for env_arg in  _generate_alp_train_params(
                                       config_type=config['config_type'],
                                       experiment_name=name,
                                       param_name=config['param_name'],
                                       param_values=config['param_values'],
                                       base_env_args_overrides=config.get('base_env_args_overrides'),
                                       param_modifier_fn=config.get('param_modifier_fn')):
                line = "python run.py --params '" + json.dumps({'env_args':env_arg, 'experiment_name': name}) + "'\n"
                f.write(line)

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
    EXPERIMENT_CONFIGS = {
        'discount_factor': {
            'config_type': 'ejor',
            'param_name': 'discount_factor',
            'param_values': [0.9, 0.95, 0.99],
        },
        'booking_window_size': {
            'config_type': 'ejor',
            'param_name': 'booking_window_size',
            'param_values': [80, 90, 100],
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
        'occupancy_level': 2000
    }
    
    # --- Run All Experiments ---
    generate_alp_train_params(
        experiment_configs=EXPERIMENT_CONFIGS,
        dat_file = 'table.dat',
    )
    '''
    
    generate_all_experiments(
        config_type='ejor',
        experiment_configs=EXPERIMENT_CONFIGS,
        test_sample_path_num_map=TEST_SAMPLE_NUM_MAP,
        dat_file='table.dat',
        is_reuse=False # Set to True to avoid regenerating files and only create the .dat
    )
    '''