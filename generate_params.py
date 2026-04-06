import json
import math
import random
import pickle
from pprint import pprint

import pandas as pd
import numpy as np
from scipy.stats import geom
from experiments import get_config_by_type
from utils import iter_to_tuple, iter_to_list, get_uid, read_lines_with_pattern, RunningStats, encode, decode, wait_time, acquire_grb_env
from pathlib import Path
import hashlib
import glob
import json
import copy
import os
from gurobipy import GRB

from decision_maker import InfinitePenalizedSAAAgent, LinearPenaltyFunction, ALPRowGenerationAgent


def _load_cached_training_result(experiment_name, file_name, env_args):
    file_path = os.path.join('experiments', 'results', experiment_name, file_name)
    if not os.path.exists(file_path):
        return None
    env_uid = get_uid(env_args)
    cached_record = None
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('uid') == env_uid:
                cached_record = record
    return cached_record


def _save_training_result(experiment_name, file_name, env_args, agent_name, obj_val, coefficients, info=None):
    file_path = os.path.join('experiments', 'results', experiment_name, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    record = {
        'uid': get_uid(env_args),
        'result': {
            'agent_name': agent_name,
            'obj_val': obj_val,
            'args': {'coefficients': coefficients},
            'info': info or {},
        },
    }
    with open(file_path, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record

def _split_list_into_groups(results, num_groups, dat_file=None):
    n = num_groups if num_groups and num_groups > 0 else len(results)
    # Split results into n groups as evenly as possible
    groups = [results[i::n] for i in range(min(n, len(results)))]
    lines_to_write = []
    for line_index, group in enumerate(groups, start=1):
        lines_to_write.append(
            f"{line_index} python run.py --params '" + json.dumps(group) + "'\n"
        )
    if dat_file:
        with open(dat_file, 'w') as f:
            f.writelines(lines_to_write)
        print(f"Saved {len(lines_to_write)} group commands to {dat_file}")
    return lines_to_write

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
         [(0, 1, 0), (1, 3, 80)]]][:]
    lines_to_write = []
    test_params = {}
    for i, waiting_penalty in enumerate(l, start=1):
        holding_cost = [wait_time(waiting_penalty[i]) for i in range(len(waiting_penalty))]
        holding_cost = np.array(holding_cost).T
        env_args['holding_cost_by_day_by_type'] = holding_cost.tolist()
        save_params = {
                        'experiment_name': experiment_name,
                        'env_args': env_args,
                      }
        env_uid = get_uid(env_args)
        lines_to_write.append(f"{i} python run.py --params '" + json.dumps(save_params) + "'\n")
        test_params[env_uid] = copy.deepcopy(env_args)
    with open(dat_file, 'w') as f:
        f.writelines(lines_to_write)
    return test_params, experiment_name

def generate_high_priority_arrival_rate(dat_file):
    # This function can be implemented to generate parameters for testing the impact of different high priority arrival rates on the performance of the agents.
    experiment_name = 'high_priority_arrival_rate_impact'
    config_type = 'toy'
    env_args = get_config_by_type(config_type).args
    arrival_rates = [[0.6, 2.4],
                     [1.5, 1.5],
                     [2.4, 0.6]]
    lines_to_write = []
    test_params = {}
    for i, arrival_rate in enumerate(arrival_rates, start=1):
        env_args['arrival_rates'] = arrival_rate
        save_params = {
                        'experiment_name': experiment_name,
                        'env_args': env_args,
                      }
        env_uid = get_uid(env_args)
        lines_to_write.append(f"{i} python run.py --params '" + json.dumps(save_params) + "'\n")
        test_params[env_uid] = copy.deepcopy(env_args)
    with open(dat_file, 'w') as f:
        f.writelines(lines_to_write)
    return test_params, experiment_name

def generate_inital_state_variation(dat_file):
    # This function can be implemented to generate parameters for testing the impact of different initial states on the performance of the agents.
    experiment_name = 'initial_state_variation_impact'
    config_type = 'toy'
    env_args = get_config_by_type(config_type).args
    initial_states = [([2, 2, 0], [0, 0, 0], [1, 2]),
                      ([5, 5, 0], [0, 0, 0], [1, 2]),
                      ([5, 5, 0], [3, 3, 0], [1, 2]),][:]
    lines_to_write = []
    test_params = {}
    for i, initial_state in enumerate(initial_states, start=1):
        reset_params = {
            "init_state": initial_state,
        }
        env_args['reset_params'] = reset_params
        save_params = {
                        'experiment_name': experiment_name,
                        'env_args': env_args,
                      }
        env_uid = get_uid(env_args)
        lines_to_write.append(f"{i} python run.py --params '" + json.dumps(save_params) + "'\n")
        test_params[env_uid] = copy.deepcopy(env_args)
    with open(dat_file, 'w') as f:
        f.writelines(lines_to_write)
    return test_params, experiment_name

def generate_steady_state_distribution_variation(dat_file):
    # This function can be implemented to generate parameters for testing the impact of different steady state distributions on the performance of the agents.
    experiment_name = 'steady_state_distribution_variation_impact'
    config_type = 'toy'
    env_args = get_config_by_type(config_type).args
    initial_state = ([0, 0, 0], [0, 0, 0], [1, 2])
    reset_params = {
            "init_state": initial_state,
        }
    env_args['reset_params'] = reset_params
    lines_to_write = []
    test_params = {}
    save_params = {
                    'experiment_name': experiment_name,
                    'env_args': env_args,
                    }
    env_uid = get_uid(env_args)
    lines_to_write.append(f"{1} python run.py --params '" + json.dumps(save_params) + "'\n")
    test_params[env_uid] = copy.deepcopy(env_args)
    with open(dat_file, 'w') as f:
        f.writelines(lines_to_write)
    return test_params, experiment_name


def train_penalty_coefficients(env_args, experiment_name, sample_path_number):
    '''
    This function can be implemented to train the coefficients for the penalty function used in the hindsight approximation with penalty agent. The training can be done using a simple grid search or a more sophisticated optimization algorithm.
    '''
    cached = _load_cached_training_result(experiment_name, 'penalty_train.jsonl', env_args)
    if cached is not None:
        cached_result = cached.get('result', {})
        cached_obj = cached_result.get('obj_val')
        cached_coefficients = cached_result.get('args', {}).get('coefficients', None)
        cached_info = cached_result.get('info', {})
        print(f"Use cached penalty coefficients for env_uid={cached.get('uid')}")
        return cached_obj, cached_coefficients, cached_info

    config_for_train = get_config_by_type('infinite_custom', args=env_args)
    env = config_for_train.env
    generating_function = LinearPenaltyFunction(env=env)
    agent = InfinitePenalizedSAAAgent(env=env, discount_factor=env.discount_factor, 
                                      sample_path_number=sample_path_number, 
                                      current_decision_var_type='continuous', 
                                      future_decision_var_type='continuous',
                                      generating_function=generating_function, 
                                      is_myopic=False)
    init_state = None if 'init_state' not in env_args.get('reset_params', {}) else env_args['reset_params']['init_state']
    print(f"Training penalty coefficients for env_uid {get_uid(env_args)} with init_state: {init_state} and sample_path_number: {sample_path_number}")
    if init_state is not None:
        init_state = tuple(np.array(item) for item in init_state)
    env.reset_random_seeds()  # Reset random seeds before training again to ensure the same sample paths
    obj, direct_coefficients, info = agent.benders_decomposition_train(coefficient_bound=GRB.INFINITY, init_state = init_state, parallel=True, verbose=False)
    print('Obejctive from Benders decomposition training:', obj) # Full MILP:39050.05571672409 # LP: 21524.097118570513
    print('Coefficients from Benders decomposition training:', direct_coefficients)

    # # direct_coefficients = [4.5050629088130085, 19.007403595529222, 210.52084330182836, 0.6690122556727325, 17.81240359552962, 210.1258433018285, 469.6670526138236, 469.667052613824, 383.49566005697466, 395.9814464036324, 383.1616355414874, 395.8926130702999, 9.70567848715938e-13, 0.0, -0.015222222219714846, 0.19500000000251028, 0.0]
    # direct_coefficients = direct_coefficients = [5.799735521230385, 20.437279922776042, 209.3610285715884, 1.1901698841543076, 18.11727992276178, 208.3710285715395, 444.697203860891, 444.69720386089654, 377.872261776431, 390.7855706566974, 377.61857065670546, 390.68657065670726, -7.140954494389007e-13, 0.0, -0.3300000000072032, -5.093170329928398e-11, 0.0]
    # env.reset_random_seeds()
    # obj, coefficients, info = agent.sample_mean_penalized_lowerbound(coefficients=direct_coefficients, ratio=0, verbose=False)
    # print('Objective from sample mean penalized lower bound evaluation using original problem coefficients:', obj) # Full MILP:39049.9991672409 # LP: 39036.989105384375 # Zero penalized: 35918.338281242075
    _save_training_result(
        experiment_name=experiment_name,
        file_name='penalty_train.jsonl',
        env_args=env_args,
        agent_name='hindsight_approx_with_penalty',
        obj_val=obj,
        coefficients=direct_coefficients,
        info=info,
    )

    return obj, direct_coefficients, info

def train_alp_coefficients(env_args, experiment_name):
    '''
    This function can be implemented to train the coefficients for the ALP row generation agent. The training can be done using a simple grid search or a more sophisticated optimization algorithm.
    '''
    cached = _load_cached_training_result(experiment_name, 'alp_train.jsonl', env_args)
    if cached is not None:
        cached_result = cached.get('result', {})
        cached_obj = cached_result.get('obj_val')
        cached_coefficients = cached_result.get('args', {}).get('coefficients', None)
        print(f"Use cached ALP coefficients for env_uid={cached.get('uid')}")
        return cached_obj, cached_coefficients

    config_for_train = get_config_by_type('infinite_custom', args=env_args)
    env = config_for_train.env
    alp_agent = ALPRowGenerationAgent(env=env, discount_factor=env.discount_factor)
    obj, oefficients = alp_agent.train(debug=False, verbose=False)
    _save_training_result(
        experiment_name=experiment_name,
        file_name='alp_train.jsonl',
        env_args=env_args,
        agent_name='row_gen_alp',
        obj_val=obj,
        coefficients=oefficients,
        info={},
    )
    return obj, oefficients


def generate_test_paths_and_init_state(test_envs, experiment_name, test_sample_path_num, warm_up_periods=0, num_periods=None, dat_file=None, num_groups=None):
    '''
    Inital state is considered as period 1. sample path will start from period 2.
    '''
    results = []
    
    for env_uid, env_args in test_envs.items():
        print(f"Processing env_uid: {env_uid}")
        obj_alp_train, alp_coefficients = train_alp_coefficients(env_args=env_args, experiment_name=experiment_name)
        obj, direct_coefficients, info = train_penalty_coefficients(env_args=env_args, experiment_name=experiment_name, sample_path_number=256)
        print(f"Trained penalty coefficients for env_uid {env_uid}: {direct_coefficients}")
        max_length = 0
        sample_gen_args = copy.deepcopy(env_args)
        sample_gen_args["env_random_seed"] = env_args.get("env_random_seed", 0) + 1000 # make sure the random seed for sample path generation is different from the random seed for training ALP
        sample_gen_args['arrival_random_seed'] = env_args.get("arrival_random_seed", 1) + 1000 # Seed for sample path generation
        sample_gen_args['stop_time_random_seed'] = env_args.get("stop_time_random_seed", 0) + 1000
        config_for_sample_path = get_config_by_type('infinite_custom', args=sample_gen_args)
        env_for_sample_path = config_for_sample_path.env
        for command_id in range(test_sample_path_num):
            init_state = env_for_sample_path.generate_initial_state() if 'init_state' not in env_args.get('reset_params', {}) else env_args['reset_params']['init_state']
            init_state = tuple(np.array(item).tolist() for item in init_state)
            if num_periods is None:
                sample_path = env_for_sample_path.reset_arrivals(stop_time=warm_up_periods-1)
                additional_sample_path = env_for_sample_path.reset_arrivals()
                print("sample path: ")
                print(len(sample_path))
                sample_path = np.append(sample_path, additional_sample_path, axis=0) if len(sample_path) > 0 else additional_sample_path
            else:
                sample_path = env_for_sample_path.reset_arrivals(stop_time=num_periods)
            sample_path = sample_path.tolist() if hasattr(sample_path, 'tolist') else sample_path
            max_length = max(max_length, len(sample_path))
            params = {
                'init_state': init_state,
                'sample_path': sample_path,
                'warm_up_periods': warm_up_periods,
                'env_args': env_args,
            }
            uid = get_uid(params)
            save_params = {
                'uid': uid,
                "experiment_name": experiment_name,
                **params,
                "penalty_coefficients": direct_coefficients,
                "alp_coefficients": alp_coefficients,
                "group_id": env_uid,
            }
            results.append(save_params)
        print("max sample path length:", max_length)

    # Write to dat file if specified
    if dat_file:
        lines_to_write = []
        n = num_groups if num_groups and num_groups > 0 else len(results)
        # Split results into n groups as evenly as possible
        groups = [results[i::n] for i in range(min(n, len(results)))]
        for line_index, group in enumerate(groups, start=1):
            lines_to_write.append(
                f"{line_index} python run.py --params '" + json.dumps(group) + "'\n"
            )
        with open(dat_file, 'w') as f:
            f.writelines(lines_to_write)
        print(f"Saved {len(lines_to_write)} group commands to {dat_file}")

    return results

def generate_train_env(test_envs, experiment_name, number_replication, dat_file, num_groups=None):
    # This function can be implemented to generate parameters for training the agents. The parameters can include different environment configurations, initial states, and sample paths.
    results = []
    for env_uid, env_args in test_envs.items():
        for i in range(number_replication):
            save_env_args = copy.deepcopy(env_args)
            save_env_args['env_random_seed'] = env_args.get('env_random_seed', 0) + i
            save_env_args['arrival_random_seed'] = env_args.get('arrival_random_seed', 1) + i
            save_env_args['stop_time_random_seed'] = env_args.get('stop_time_random_seed', 42) + i
            save_params = {
                'uid': get_uid(save_env_args),
                'experiment_name': experiment_name,
                'env_args': save_env_args,
                'group_id': env_uid,
            }
            results.append(save_params)
    _split_list_into_groups(results, num_groups=num_groups, dat_file=dat_file)
    return results


if __name__ == '__main__':
    # test_envs, experiment_name = generate_waiting_penalty_params(dat_file='table_waiting_penalty.dat')
    # test_envs, experiment_name = generate_high_priority_arrival_rate(dat_file='table_high_priority_arrival_rate.dat')
    # test_envs, experiment_name = generate_inital_state_variation(dat_file='initial_state_variation_impact.dat')
    test_envs, experiment_name = generate_steady_state_distribution_variation(dat_file='steady_state_distribution_variation_impact.dat')
    results = generate_test_paths_and_init_state(
        test_envs=test_envs,
        experiment_name=experiment_name, 
        test_sample_path_num=5,
        warm_up_periods=1,
        num_periods=5,
        dat_file='table.dat',
        num_groups=998,  # divide into N groups
    )
    # test_envs, experiment_name = generate_inital_state_variation(dat_file='initial_state_variation_impact.dat')
    # results = generate_train_env(
    #     test_envs=test_envs,
    #     experiment_name=experiment_name,
    #     number_replication=5000,
    #     dat_file='table.dat',
    #     num_groups=998,  # divide into N groups
    # )




