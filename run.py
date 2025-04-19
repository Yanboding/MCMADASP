import argparse
import json
import os
import numpy as np
import pandas as pd
from utils import iter_to_tuple
from experiments import Config
from environment import AdvanceSchedulingEnv  # Replace 'some_module' with the actual module name where AdvanceSchedulingEnv is defined
from decision_maker import SAAdvanceAgent, PolicyEvaluator

def experiment(sample_path, sample_path_number):
    config = Config.from_real_scale()
    env_params = config.env_params
    init_state = config.init_state
    t = 1
    env = AdvanceSchedulingEnv(**env_params)
    agent_instance = SAAdvanceAgent(env, discount_factor=env_params['discount_factor'], sample_path_number=sample_path_number)
    evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
    sample_average_V = evaluator.simulation_evaluate_helper(init_state, t=t, sample_paths=[np.array(sample_path)])
    value = sample_average_V[(iter_to_tuple(init_state), t)].expect[0]
    df = pd.DataFrame(
        {
            'value': [value],
            'sample_path_number': [sample_path_number]
        }
    )
    data_path = os.path.join('.', 'real_scale_policy_value.csv')
    df.to_csv(data_path, mode='a', index=False, header=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of using argparse to pass in a list of lists.")
    parser.add_argument('--params', help='Input JSON-encoded list of lists', type=str)
    args = parser.parse_args()
    params = json.loads(args.params)
    experiment(**params)
    
