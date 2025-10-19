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


def experiment(experiment_name, param_value, env_args, agent_args, sample_path, uid, job_id):
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
                                            future_decision_var_type=GRB.INTEGER, sample_path=sample_path)
    _, benchmark_value, info = perfect_info_lower_bound_solver.solve(state, t)
    print('benchmark_value:', benchmark_value)
    for agent in agent_args:
        config = get_config_by_type(case_type='infinite_custom',args=env_args)
        env = config.env
        agent_name, args = agent['agent_name'], agent['args']
        stats = {'agent_name': agent_name}
        config.reset_params['new_arrivals'] = sample_path
        state, info = env.reset(**config.reset_params)
        print('agent_name:', agent_name)
        if agent_name in {"hindsight_approx", "hindsight_value", "myopic"}:
            agent_instance = InfiniteSAAAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name in {"hindsight_approx_with_penalty"}:
            agent_instance = InfinitePenalizedSAAAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == 'alp':
            agent_instance = ALPEJORColumnGenerationAgent(env, discount_factor=env.discount_factor, **args)
        evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
        states, rewards = evaluator.sample_path_evaluate(state, t, sample_path)
        value_function = sum(rewards)
        stats['abs_gap'] = max(value_function - benchmark_value, 0)
        stats['benchmark_value'] = benchmark_value
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
        if agent_name in {"hindsight_approx", "hindsight_value", "myopic"}:
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

def alp_train(env_args, experiment_name, job_id=None):
    print('Training ALP agent with args:', env_args)
    config_for_train = get_config_by_type(case_type='infinite_custom',args=env_args)
    env_for_train = config_for_train.env
    agent = ALPRowGenerationAgent(env=env_for_train, discount_factor=env_for_train.discount_factor)
    coefficients = agent.train(debug=False, verbose=True, max_iter=5000)
    output_file = os.path.join('experiments','results',experiment_name, f'alp_train{job_id}.jsonl' if job_id else 'alp_train.jsonl')
    with safe_open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps({'uid':get_uid(env_args), 'result': {'agent_name': 'alp', 'args': {'coefficients':coefficients}}}) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of using argparse to pass in a list of lists.")
    parser.add_argument('--params', help='Input JSON-encoded list of lists', type=str)
    parser.add_argument('--job_id', help='Input METAJOB_ID', type=str)
    args = parser.parse_args()
    params = json.loads(args.params)
    alp_train(**params, job_id=args.job_id)
    #experiment(**params, job_id=args.job_id)
    #value_function_experiment(**params, job_id=args.job_id)
    