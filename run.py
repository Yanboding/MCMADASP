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
from utils import iter_to_tuple
from decision_maker import SAAdvanceAgent, PolicyEvaluator

def experiment(env_args, agent_args, sample_path, uid, output_file):
    res = {'result':[]}
    res["uid"] = uid
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
    print(res)
    with open(output_file, 'a') as f:  # 'a' will create the file if not present
        f.write(json.dumps(res) + '\n')
    


def experiment_revise(command_id, sample_path, sample_path_numbers, replication, agents, occupancy_percentage, output_file, case_type):
    '''
    {
        "command_id"
        "sample_path_number",
        "occupancy_percentage",
        "{agent_name}_value",
        "expect_hindsight_approx",
        "varSum_hindsight_approx",
        "count_hindsight_approx",
        "expect_hindsight_approx_runtime",
        "varSum_hindsight_approx_runtime",
        "count_hindsight_approx_runtime",
        "expect_{agent_name}_type_{i}_wait_time",
        "varSum_{agent_name}_type_{i}_wait_time",
        "{agent_name}_overtime_{day}"
    }
    '''
    t = 1
    # For each different number of sample paths
    for M in sample_path_numbers:
        # set result identifier
        res = {}
        res["command_id"] = command_id
        res["sample_path_number"] = M
        res["occupancy_percentage"] = occupancy_percentage
        # evaluate agent performance
        for agent in agents:
            config = get_config_by_type(case_type, random_seed=command_id + 1)
            env = config.env
            agent_name, args = agent['agent_name'], agent['args']
            state, info = env.reset(t=t, percentage_occupied=occupancy_percentage, new_arrivals=sample_path, seed=command_id)
            if agent_name == "hindsight_approx":
                agent_instance = SAAdvanceAgent(env, discount_factor=env.discount_factor, **args)
            elif agent_name == "myopic":
                agent_instance = SAAdvanceFastAgent(env, discount_factor=env.discount_factor, **args)
            agent_instance.set_sample_paths(M)
            evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
            sample_average_V = evaluator.simulation_evaluate_helper(state, t=t,
                                                                    sample_paths=[np.array(sample_path)])
            res[agent_name+'_value'] = sample_average_V[(iter_to_tuple(state), t)].expect[0]

            for type_i, running_stat in env.wait_time_by_type.items():
                res['expect_' + agent_name + '_' + str(type_i)] = running_stat.expect[0]
                res['varSum_' + agent_name + '_' + str(type_i)] = running_stat.varSum[0]
                res['count_' + agent_name + '_' + str(type_i)] = running_stat.count
            for day in range(len(env.overtime)):
                res[agent_name + '_overtime_' + str(day)] = env.overtime[day]

        config = get_config_by_type(case_type, random_seed=command_id + 1)
        env = config.env
        state, info = env.reset(t=t, percentage_occupied=occupancy_percentage, new_arrivals=sample_path,
                                seed=command_id)
        hindsight_lower_bound_agent = SAAdvanceAgent(env, discount_factor=env.discount_factor)
        hindsight_lower_bound_agent.set_real_sample_paths([sample_path])
        action, overtime, obj_value = hindsight_lower_bound_agent.solve(state, t,
                                                                        current_decision_var_type=GRB.INTEGER,
                                                                        future_decision_var_type=GRB.INTEGER)
        res['hindsight_lower_bound'] = obj_value
        df = pd.DataFrame(
            [res]
        )
        with open("columns.txt", "w", encoding="utf‑8") as f:
            for col in df.columns:
                f.write(f"{col}\n")
        data_path = os.path.join('.', output_file)
        df.to_csv(data_path, mode='a', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of using argparse to pass in a list of lists.")
    parser.add_argument('--params', help='Input JSON-encoded list of lists', type=str)
    args = parser.parse_args()
    params = json.loads(args.params)
    experiment(**params)
    
