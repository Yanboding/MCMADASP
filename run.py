import argparse
import json
import os
import time
from pprint import pprint

import numpy as np
import pandas as pd
from gurobipy import GRB

from decision_maker.memory_efficient_mcma_agent import SAAdvanceFastAgent
from experiments.experiment_config import get_config_by_type
from utils import iter_to_tuple, get_uid, safe_open
from decision_maker import SAAdvanceAgent, PolicyEvaluator, ALPAgent


def experiment(experiment_name, param_value, env_args, agent_args, sample_path, uid, job_id):
    res = {'result':[]}
    res["uid"] = uid
    res["experiment_name"] = experiment_name
    res["param_value"] = param_value
    sample_path = np.array(sample_path)
    t = 1  # Assuming a single time step for the experiment
    for agent in agent_args:
        config = get_config_by_type(case_type='custom',args=env_args)
        env = config.env
        agent_name, args = agent['agent_name'], agent['args']
        stats = {'agent_name': agent_name}
        state, info = env.reset(**config.reset_params)
        if agent_name == "hindsight_approx":
            agent_instance = SAAdvanceAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == "myopic":
            agent_instance = SAAdvanceFastAgent(env, discount_factor=env.discount_factor, **args)
        elif agent_name == 'alp':
            agent_instance = ALPAgent(env, discount_factor=env.discount_factor, **args)
        evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
        lower_bound_solver = SAAdvanceAgent(env, discount_factor=env.discount_factor,current_decision_var_type=GRB.INTEGER,
                                            future_decision_var_type=GRB.INTEGER)
        opt_gap = evaluator.sample_path_optimality_gap_evaluate(lower_bound_solver, state, t, sample_path)
        stats['opt_gap'] = opt_gap
        wait_time_by_type =[]
        for type_i, running_stat in env.wait_time_by_type.items():
            wait_time_by_type.append({"treatment_type":type_i,
                                      "expect":running_stat.expect.tolist()[0],
                                      "varSum":running_stat.varSum.tolist()[0],
                                      "count":int(running_stat.count)})
        stats['wait_time_by_type'] = wait_time_by_type
        stats['overtime'] = env.overtime.tolist()
        res['result'].append(stats)
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')

def alp_train(env_args, experiment_name, job_id=None):
    print('Training ALP agent with args:', env_args)
    config_for_train = get_config_by_type(case_type='custom',args=env_args)
    env_for_train = config_for_train.env
    agent = ALPAgent(env=env_for_train, discount_factor=env_for_train.discount_factor)
    coefficients = agent.train(debug=False)
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
    
