import time
from pprint import pprint

import numpy as np
import pandas as pd
from decision_maker.so_allocation_to_advance_agent import SAAllocationAdvanceAgent
from environment import MultiClassPoissonArrivalGenerator, AdvanceSchedulingEnv
from visualization import opt_plot


def sample_path_experiment(env_params, t, num_sample_path, class_numbers):
    sa_value_function_df = {}
    runtime_df = {}
    Ms = [1]+[i for i in range(200, num_sample_path+1, 200)]
    sa_value_function_df['num_sample_path'] = Ms
    runtime_df['num_sample_path'] = Ms
    for class_number in class_numbers:
        arrival_generator = MultiClassPoissonArrivalGenerator(10, 30, [1 / class_number] * class_number)
        holding_cost = [10 - i * (5 / (class_number - 1)) for i in range(class_number)]
        treatment_pattern = [[1]*class_number]
        env_params['treatment_pattern'] = treatment_pattern
        env_params['arrival_generator'] = arrival_generator
        env_params['holding_cost'] = holding_cost
        env_params['regular_capacity'] = 10
        bookings = np.array([0])
        future_schedule = np.array([[0] * class_number for _ in range(env_params['decision_epoch'])])
        new_arrival = np.array([6]*class_number)
        init_state = (bookings, new_arrival, future_schedule)
        #pprint(env_params)
        sa_values = []
        optimal_actions = []
        runtimes = []
        env = AdvanceSchedulingEnv(**env_params)
        so_agent = SAAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'])
        for M in Ms:
            print(M)
            start = time.time()
            action, solution_y, obj_value = so_agent.solve(init_state, t, 0, M)
            runtime = time.time() - start
            runtimes.append(runtime)
            sa_values.append(obj_value)
            optimal_actions.append(action)
        sa_value_function_df['sa_values_'+str(class_number)] = sa_values
        runtime_df['runtimes_'+str(class_number)] = runtimes
        pprint(optimal_actions)
    sa_value_function_df = pd.DataFrame(sa_value_function_df)
    runtime_df = pd.DataFrame(runtime_df)
    sa_value_function_df.to_csv('converge_test.csv')
    runtime_df.to_csv('runtime.csv')
    print(sa_value_function_df)
    sa_value_function_text_labels = [label for label in sa_value_function_df.drop(columns=['num_sample_path'])]
    opt_plot(sa_value_function_df, 'Number of Sample Path', [str(i)+' types' for i in class_numbers], 'converge_test', sa_value_function_text_labels, x_val_col='num_sample_path')
    runtime_text_labels = [label for label in runtime_df.drop(columns=['num_sample_path'])]
    opt_plot(runtime_df, 'Number of Sample Path', [str(i)+' types' for i in class_numbers], 'runtime_test', runtime_text_labels, ylabel='Seconds', x_val_col='num_sample_path')

def advance_scheduling_experiment(env_params, t, num_sample_path, class_numbers):
    sa_value_function_df = {}
    runtime_df = {}
    Ms = [1]+[i for i in range(200, num_sample_path+1, 200)]
    sa_value_function_df['num_sample_path'] = Ms
    runtime_df['num_sample_path'] = Ms
    for class_number in class_numbers:
        arrival_generator = MultiClassPoissonArrivalGenerator(10, 30, [1 / class_number] * class_number)
        holding_cost = [10 - i * (5 / (class_number - 1)) for i in range(class_number)]
        treatment_pattern = [[1]*class_number]
        env_params['treatment_pattern'] = treatment_pattern
        env_params['arrival_generator'] = arrival_generator
        env_params['holding_cost'] = holding_cost
        env_params['regular_capacity'] = 10
        bookings = np.array([0])
        future_schedule = np.array([[0] * class_number for _ in range(env_params['decision_epoch'])])
        new_arrival = np.array([6]*class_number)
        init_state = (bookings, new_arrival, future_schedule)
        #pprint(env_params)
        sa_values = []
        optimal_actions = []
        runtimes = []
        env = AdvanceSchedulingEnv(**env_params)
        so_agent = SAAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'])
        for M in Ms:
            print(M)
            start = time.time()
            action, solution_y, obj_value = so_agent.solve(init_state, t, 0, M)
            runtime = time.time() - start
            runtimes.append(runtime)
            sa_values.append(obj_value)
            optimal_actions.append(action)
        sa_value_function_df['sa_values_'+str(class_number)] = sa_values
        runtime_df['runtimes_'+str(class_number)] = runtimes
        pprint(optimal_actions)
    sa_value_function_df = pd.DataFrame(sa_value_function_df)
    runtime_df = pd.DataFrame(runtime_df)
    sa_value_function_df.to_csv('converge_test.csv')
    runtime_df.to_csv('runtime.csv')
    print(sa_value_function_df)
    sa_value_function_text_labels = [label for label in sa_value_function_df.drop(columns=['num_sample_path'])]
    opt_plot(sa_value_function_df, 'Number of Sample Path', [str(i)+' types' for i in class_numbers], 'converge_test', sa_value_function_text_labels, x_val_col='num_sample_path')
    runtime_text_labels = [label for label in runtime_df.drop(columns=['num_sample_path'])]
    opt_plot(runtime_df, 'Number of Sample Path', [str(i)+' types' for i in class_numbers], 'runtime_test', runtime_text_labels, ylabel='Seconds', x_val_col='num_sample_path')


if __name__ == '__main__':
    from environment import AdvanceSchedulingEnv

    env_params = {
        'decision_epoch': 20,
        'overtime_cost': 30,
        'duration': 1,
        'regular_capacity': 12,
        'discount_factor': 0.99,
        'problem_type': 'allocation'
    }

    sample_path_experiment(env_params=env_params, t=1, num_sample_path=3000, class_numbers=[2])