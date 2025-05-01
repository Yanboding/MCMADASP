import os
import time
from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd

from decision_maker import PolicyEvaluator, SAAdvanceAgent
from decision_maker.so_allocation_to_advance_agent import SAAllocationAdvanceAgent
from environment import MultiClassPoissonArrivalGenerator, AdvanceSchedulingEnv
from experiments.config import Config
from visualization import opt_plot, approximate_value_plot


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

def period_to_go_experiment(env_params, agents, period_num, init_arrival, plot_labels, data_file_path, image_path, shortcut=False):
    if os.path.exists(data_file_path) and shortcut:
        print(f"Found '{data_file_path}'. Reading CSV file...")
        df = pd.read_csv(data_file_path,index_col=0)
        txt_labels = []
        for agent_name, agent in agents.items():
            txt_labels.append(agent_name)
        print("File loaded successfully!")
        print("Here are the first 5 rows:")
        print(df.head())
        opt_plot(df, 'period to go', plot_labels, image_path, text_labels=txt_labels)
    else:
        decision_epochs = [decision_epoch for decision_epoch in range(1, period_num + 1)]
        values = defaultdict(list)
        txt_labels = set()
        approx_labels = set()
        for decision_epoch in decision_epochs:
            print('decision epoch:', decision_epoch)
            env_params['decision_epoch'] = decision_epoch
            env = AdvanceSchedulingEnv(**env_params)
            num_types = len(env_params['treatment_pattern'][0])
            bookings = np.array([0]* len(env_params['treatment_pattern']))
            future_schedule = np.array([[0] * num_types for _ in range(env_params['decision_epoch'])])
            init_state = (bookings, init_arrival, future_schedule)

            for agent_name, agent in agents.items():
                txt_labels.add(agent_name+'_hindsight_value')
                txt_labels.add(agent_name)
                approx_labels.add(agent_name)
                print(f'Waiting for {agent_name}...')
                agent_instance = agent(env=env, discount_factor=env_params['discount_factor'])
                evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
                start = time.time()
                solution_a, y_t_solution, hindsight_value = agent_instance.solve(init_state, 1)
                print('solve time:', time.time() - start)
                start = time.time()
                policy_value, policy_value_lower, policy_value_upper = evaluator.simulation_evaluate(init_state, 1, replication=3000, confidence=.95, max_periods=100)
                print('evaluate time:', time.time() - start)
                values[agent_name+'_hindsight_value'].append(hindsight_value)
                values[agent_name].append(policy_value[0])
                values[agent_name+'_lower'].append(policy_value_lower[0])
                values[agent_name + '_upper'].append(policy_value_upper[0])
                print(f'{agent_name} Done:', policy_value, f'{agent_name} action:', agent_instance.policy(init_state, 1))
        values['decision_epoch'] = decision_epochs
        df = pd.DataFrame(values)
        df.to_csv(data_file_path)
        approximate_value_plot(df,
                               xlabel='period to go',
                               ylabel='Value function',
                               approx_labels=list(approx_labels),
                               text_labels=list(txt_labels),
                               plot_labels=plot_labels,
                               save_file=image_path,
                               x_val_col='decision_epoch')
        #opt_plot(df, 'period to go', plot_labels, 'test')

def sample_size_experiments(env_params, agents, simple_size, init_state, value_function_data_path, runtime_data_path, value_plot_labels, runtime_plot_labels, value_image_path, runtime_image_path,shortcut=True):
    if os.path.exists(value_function_data_path) and shortcut:
        print(f"Found '{value_function_data_path}'. Reading CSV file...")
        sa_value_function_df = pd.read_csv(value_function_data_path,index_col=0)
        runtime_df = pd.read_csv(runtime_data_path,index_col=0)
        txt_labels = set()
        approx_labels = set()
        for agent_name, agent in agents.items():
            txt_labels.add(agent_name + '_hindsight')
            txt_labels.add(agent_name)
            approx_labels.add(agent_name)
        print("File loaded successfully!")
        print("Here are the first 5 rows:")
        print(sa_value_function_df.head())
        print(runtime_df.head())
    else:
        sa_value_function_df = defaultdict(list)
        runtime_df = defaultdict(list)
        Ms = [1] + [i for i in range(200, simple_size + 1, 200)]
        sa_value_function_df['num_sample_path'] = Ms
        runtime_df['num_sample_path'] = Ms
        env = AdvanceSchedulingEnv(**env_params)
        txt_labels = set()
        approx_labels = set()
        for agent_name, agent in agents.items():
            txt_labels.add(agent_name + '_hindsight')
            txt_labels.add(agent_name)
            approx_labels.add(agent_name)
            for M in Ms:
                print('Sample Size:', M)
                agent_instance = agent(env, discount_factor=env_params['discount_factor'])
                agent_instance.set_sample_paths(M)
                evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
                start = time.time()
                policy_value, policy_value_cl, policy_value_ch = evaluator.simulation_evaluate(init_state, t=1, replication=300, confidence=0.95)
                policy_evaluate_runtime = time.time() - start
                sa_value_function_df[agent_name].append(policy_value[0])
                sa_value_function_df[agent_name + '_lower'].append(policy_value_cl[0])
                sa_value_function_df[agent_name + '_upper'].append(policy_value_ch[0])
                print('Policy Value:', policy_value[0], 'Confidence interval:', (policy_value_cl[0], policy_value_ch[0]), 'Time:', policy_evaluate_runtime)
                start = time.time()
                solution_a, y_t_solution, hindsight_value = agent_instance.solve(init_state, 1)
                runtime = time.time() - start
                print('Hindsight Value:', hindsight_value, 'Time:', runtime)
                sa_value_function_df[agent_name + '_hindsight'].append(hindsight_value)
                runtime_df[agent_name].append(runtime)
        sa_value_function_df = pd.DataFrame(sa_value_function_df)
        runtime_df = pd.DataFrame(runtime_df)
        sa_value_function_df.to_csv(value_function_data_path)
        runtime_df.to_csv(runtime_data_path)
    approximate_value_plot(sa_value_function_df,
                           xlabel='Number of Sample Paths',
                           ylabel='Value function',
                           approx_labels=list(approx_labels),
                           text_labels=list(txt_labels),
                           plot_labels=value_plot_labels,
                           save_file=value_image_path,
                           x_val_col='num_sample_path')
    approximate_value_plot(runtime_df,
                           xlabel='Number of Sample Paths',
                           ylabel='Seconds',
                           approx_labels=[],
                           text_labels=list(agents.keys()),
                           plot_labels=runtime_plot_labels,
                           save_file=runtime_image_path,
                           x_val_col='num_sample_path')
    return

if __name__ == '__main__':
    from environment import AdvanceSchedulingEnv
    config = Config.from_multiappt_default_case()
    agents = {
        'SAAdvanceAgent': SAAdvanceAgent
    }
    sample_size_experiments(config.env_params, agents, 200, init_state=config.init_state,
                            value_function_data_path='data/sa_advance_real_scale_sample_size_experiment_value_function.csv',
                            runtime_data_path='data/sa_advance_real_scale_sample_size_experiment_runtime.csv',
                            value_plot_labels={'SAAdvanceAgent':"Advance Hindsight Policy Value", 'SAAdvanceAgent_hindsight':"Advance Expected Hindsight Value"},
                            runtime_plot_labels={'SAAdvanceAgent': "Advance Hindsight Policy"},
                            value_image_path='figures/sa_advance_real_scale_sample_size_experiment_value_function',
                            runtime_image_path='figures/sa_advance_real_scale_sample_size_experiment_runtime',shortcut=True)
