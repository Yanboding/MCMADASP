import argparse
import json
import os
import time
from pprint import pprint

import numpy as np
import pandas as pd
from gurobipy import GRB

from experiments.experiment_config import get_config_by_type
from utils import iter_to_tuple, get_uid, safe_open
from decision_maker import ALPEJORColumnGenerationAgent, InfiniteSAAAgent, InfinitePenalizedSAAAgent, MyopicAgent, ALPRowGenerationAgent
from policy_evaluator import PolicyEvaluator

def run_lower_bound_solver(experiment_name, param_value, env_args, agent_args, sample_path, uid, job_id):
    res = {}
    res["uid"] = uid
    res["experiment_name"] = experiment_name
    res["param_value"] = param_value
    t = 1  # Assuming a single time step for the experiment
    config = get_config_by_type(case_type='infinite_custom',args=env_args)
    env = config.env
    config.reset_params['new_arrivals'] = sample_path
    perfect_info_lower_bound_solver = InfiniteSAAAgent(env, discount_factor=env.discount_factor,current_decision_var_type=GRB.INTEGER,
                                            future_decision_var_type=GRB.CONTINUOUS, sample_path=sample_path, is_include_discount_factor=True)
    state, info = env.reset(**config.reset_params)
    _, benchmark_value, info = perfect_info_lower_bound_solver.solve(state, t)
    res['benchmark_value'] = benchmark_value
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    # Make sure the parent directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')

def experiment(experiment_name, param_value, env_args, coefficients, agent_args, sample_path, uid, job_id):
    res = {'result':[]}
    res["uid"] = uid
    res["experiment_name"] = experiment_name
    res["param_value"] = param_value
    sample_path = np.array(sample_path)
    print('uid:', uid)
    print('test sample_path:', sample_path)
    t = 1  # Assuming a single time step for the experiment
    config = get_config_by_type(case_type='infinite_custom',args=env_args)
    env = config.env
    config.reset_params['new_arrivals'] = sample_path
    state, info = env.reset(**config.reset_params)
    print('init state:', state)
    perfect_info_lower_bound_solver = InfiniteSAAAgent(env, discount_factor=env.discount_factor,current_decision_var_type=GRB.INTEGER,
                                            future_decision_var_type=GRB.INTEGER, sample_path=sample_path, is_include_discount_factor=True)
    _, perfect_info_lower_bound, info = perfect_info_lower_bound_solver.solve(state, t)
    print('benchmark_value:', perfect_info_lower_bound)
    penalized_lower_bound_solver = InfinitePenalizedSAAAgent(env, discount_factor=env.discount_factor,current_decision_var_type=GRB.INTEGER,
                                            future_decision_var_type=GRB.INTEGER, sample_path=sample_path, is_include_discount_factor=True, coeffecients=coefficients)
    _, penalized_lower_bound, info = penalized_lower_bound_solver.solve(state, t)
    print('benchmark_value:', penalized_lower_bound)
    for agent in agent_args:
        config = get_config_by_type(case_type='infinite_custom',args=env_args)
        env = config.env
        agent_name, args = agent['agent_name'], agent['args']
        stats = {'agent_name': agent_name}
        config.reset_params['new_arrivals'] = sample_path
        state, info = env.reset(**config.reset_params)
        print('agent init state:', state)
        print('agent_name:', agent_name)
        if agent_name in {"hindsight_approx"}:
            agent_instance = InfiniteSAAAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name in {"hindsight_approx_with_penalty"}:
            agent_instance = InfinitePenalizedSAAAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == "myopic":
            agent_instance = MyopicAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == 'col_gen_alp':
            agent_instance = ALPEJORColumnGenerationAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == 'row_gen_alp':
            agent_instance = ALPRowGenerationAgent(env, discount_factor=env.discount_factor, **args)
        evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
        states, rewards = evaluator.sample_path_evaluate(state, t, sample_path)
        G = 0.0
        for tau in reversed(range(len(rewards))):
            G = env.discount_factor * G + rewards[tau]
        stats['value_function'] = G
        stats['perfect_info_lower_bound'] = perfect_info_lower_bound
        stats['penalized_lower_bound'] = penalized_lower_bound
        wait_time_by_type =[]
        waiting_time_target_violations = []
        for type_i, running_stat in env.wait_time_by_type.items():
            wait_time_by_type.append({"treatment_type":type_i,
                                      "expect":running_stat.mean,
                                      "varSum":running_stat.var_sum,
                                      "count":int(running_stat.n)})
        stats['wait_time_by_type'] = wait_time_by_type
        for type_i, running_stat in env.waiting_time_target_violations.items():
            waiting_time_target_violations.append({"treatment_type":type_i,
                                                    "expect":running_stat.mean,
                                                    "varSum":running_stat.var_sum,
                                                    "count":int(running_stat.n)})
            
        stats['waiting_time_target_violations'] = waiting_time_target_violations
        stats['overtime'] = env.overtime.tolist()
        res['result'].append(stats)
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    # Make sure the parent directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')

def value_function_experiment(experiment_name, param_value, env_args, agent_args, sample_path, uid, job_id):
    res = {'result':[]}
    res["uid"] = uid
    res["experiment_name"] = experiment_name
    res["param_value"] = param_value
    sample_path = np.array(sample_path)
    t = 1  # Assuming a single time step for the experiment
    print('agent_args:', len(agent_args))
    for agent in agent_args:
        config = get_config_by_type(case_type='infinite_custom',args=env_args)
        env = config.env
        agent_name, args = agent['agent_name'], agent['args']
        stats = {'agent_name': agent_name}
        config.reset_params['new_arrivals'] = sample_path
        state, info = env.reset(**config.reset_params)
        print(config.reset_params)
        print('init state:', state)
        if agent_name in {"hindsight_approx", "hindsight_value"}:
            agent_instance = InfiniteSAAAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == "myopic":
            agent_instance = MyopicAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == 'alp':
            agent_instance = ALPEJORColumnGenerationAgent(env, discount_factor=env.discount_factor, **args)
        evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
        states, rewards = evaluator.sample_path_evaluate(state, t, sample_path)
        stats['value_function'] = sum(rewards)
        '''
        if agent_name == "hindsight_value":
            benchmark_solver = InfiniteSAAAgent(env, discount_factor=env.discount_factor, **args)
            _, benchmark_value, info = benchmark_solver.solve(state, t)
        '''
        wait_time_by_type =[]
        for type_i, running_stat in env.wait_time_by_type.items():
            wait_time_by_type.append({"treatment_type":type_i,
                                      "expect":running_stat.mean,
                                      "varSum":running_stat.var_sum,
                                      "count":int(running_stat.n)})
        stats['wait_time_by_type'] = wait_time_by_type
        stats['overtime'] = env.overtime.tolist()
        res['result'].append(stats)
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    # Make sure the parent directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')

def alp_train(env_args, experiment_name, train_type="col_gen", job_id=None):
    print('Training ALP agent with args:', env_args)
    config_for_train = get_config_by_type(case_type='infinite_custom',args=env_args)
    env_for_train = config_for_train.env
    if train_type == "col_gen":
        agent = ALPEJORColumnGenerationAgent(env=env_for_train, discount_factor=env_for_train.discount_factor)
    else:
        agent = ALPRowGenerationAgent(env=env_for_train, discount_factor=env_for_train.discount_factor)
    obj_val, coefficients = agent.train(debug=False, verbose=True)
    output_file = os.path.join('experiments','results',experiment_name, f'alp_train_{train_type}_{job_id}.jsonl' if job_id else f'alp_train_{train_type}.jsonl')
    with safe_open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps({'uid':get_uid(env_args), 'result': {'agent_name': f'{train_type}_alp', 'obj_val': obj_val, 'args': {'coefficients':coefficients}}}) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of using argparse to pass in a list of lists.")
    parser.add_argument('--params', help='Input JSON-encoded list of lists', type=str)
    parser.add_argument('--job_id', help='Input METAJOB_ID', type=str)
    args = parser.parse_args()
    params = json.loads(args.params)
    #alp_train(**params, job_id=args.job_id)
    experiment(**params, job_id=args.job_id)
    #value_function_experiment(**params, job_id=args.job_id)
    #run_lower_bound_solver(**params, job_id=args.job_id)
    