import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from utils import iter_to_tuple
from experiments import Config
from environment import AdvanceSchedulingEnv  # Replace 'some_module' with the actual module name where AdvanceSchedulingEnv is defined
from decision_maker import SAAdvanceAgent, PolicyEvaluator

def experiment(sample_path, sample_path_numbers, command_id, output_file='real_scale_policy_value.csv', case_type='ejor'):
    for sample_path_number in sample_path_numbers:
        print('sample_path_number:', sample_path_number)
        config = None
        if case_type == 'real_scale':
            config = Config.from_real_scale()
        elif case_type == 'ejor':
            config = Config.from_EJOR_case()
        elif case_type == 'adjust_ejor':
            config = Config.from_adjust_EJOR_case()
        env_params = config.env_params
        bookings, _, future_schedule = config.init_state
        init_state = (bookings, np.array(sample_path[0]), future_schedule)
        t = 1
        env = AdvanceSchedulingEnv(**env_params)
        agent_instance = SAAdvanceAgent(env, discount_factor=env_params['discount_factor'])
        agent_instance.set_sample_paths(sample_path_number)
        evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
        start = time.time()
        action, overtime, obj_value = agent_instance.solve(init_state, t)
        end = time.time() - start
        sample_average_V = evaluator.simulation_evaluate_helper(init_state, t=t, sample_paths=[np.array(sample_path)])
        value = sample_average_V[(iter_to_tuple(init_state), t)].expect[0]
        res = {}
        res['value'] = value
        res['sample_path_number'] = sample_path_number
        res['hindsight_value'] = obj_value
        res['run_time'] = end
        res['command_id'] = command_id
        for type_i, running_stat in env.wait_time_by_type.items():
            res['expect_'+str(type_i)] = running_stat.expect[0]
            res['varSum_' + str(type_i)] = running_stat.varSum[0]
            res['count_' + str(type_i)] = running_stat.count
        for day in range(len(env.overtime)):
            res['slot_number_'+str(day)] = env.overtime[day]
        df = pd.DataFrame(
            [res]
        )
        print(df.columns.tolist())
        data_path = os.path.join('.', output_file)
        df.to_csv(data_path, mode='a', index=False, header=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of using argparse to pass in a list of lists.")
    parser.add_argument('--params', help='Input JSON-encoded list of lists', type=str)
    args = parser.parse_args()
    params = json.loads(args.params)
    experiment(**params)
    
