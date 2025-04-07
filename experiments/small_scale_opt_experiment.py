import os
import time
from pprint import pprint

import numpy as np
import pandas as pd

from decision_maker import SAAdvanceAgent, OptimalAgent, PolicyEvaluator, SAAllocationAdvanceAgent, \
    ApproxAllocationAdvanceAgent, SAMultiClassMultiAppntAllocationAdvanceAgent
from visualization import opt_plot
from environment import MultiClassPoissonArrivalGenerator, AdvanceSchedulingEnv

def basic_converge_experiment(env_params, agents, init_state, t, M):
    env = AdvanceSchedulingEnv(**env_params)
    value_function_df = {}
    best_action_df = {}
    print('Waiting for Optimal...')
    optimal_agent = OptimalAgent(env=env, discount_factor=env_params['discount_factor'])
    optimal_agent.train(init_state, t)
    optimal_value = optimal_agent.get_state_value(init_state, t)
    optimal_action = optimal_agent.policy(init_state, t)
    value_function_df['OPT'] = [optimal_value]
    best_action_df['OPT'] = [optimal_action]
    for agent_name, agent in agents.items():
        print(f'Waiting for {agent_name}...')
        agent_instance = agent(env=env, discount_factor=env_params['discount_factor'])
        action, solution_y, obj_value = agent_instance.solve(init_state, t, 0, M)
        value_function_df[agent_name] = [obj_value]
        best_action_df[agent_name] = [action]
    value_function_df = pd.DataFrame(value_function_df)
    best_action_df = pd.DataFrame(best_action_df)
    return value_function_df, best_action_df

def period_to_go_experiment(env_params, agents, period_num, plot_labels, data_file_path, image_path):
    if os.path.exists(data_file_path) and False:
        print(f"Found '{data_file_path}'. Reading CSV file...")
        df = pd.read_csv(data_file_path,index_col=0)

        print("File loaded successfully!")
        print("Here are the first 5 rows:")
        print(df.head())
        opt_plot(df, 'period to go', plot_labels, image_path)
    else:
        decision_epochs = [decision_epoch for decision_epoch in range(1, period_num + 1)]
        values = {agent_name: [] for agent_name in agents}
        values['OptimalAgent'] = []
        txt_labels = ['OptimalAgent']
        for decision_epoch in decision_epochs:
            print('decision epoch:', decision_epoch)
            env_params['decision_epoch'] = decision_epoch
            env = AdvanceSchedulingEnv(**env_params)
            num_types = len(env_params['treatment_pattern'][0])
            bookings = np.array([0]* len(env_params['treatment_pattern']))
            future_schedule = np.array([[0] * num_types for _ in range(env_params['decision_epoch'])])
            new_arrival = np.array([3]*num_types)
            init_state = (bookings, new_arrival, future_schedule)

            print('Waiting for Optimal...')
            optimal_agent = OptimalAgent(env=env, discount_factor=env_params['discount_factor'])
            optimal_agent.train(init_state, 1)
            print(init_state)
            optimal_value = optimal_agent.get_state_value(init_state, 1)
            values['OptimalAgent'].append(optimal_value)
            print('Optimal Done:', optimal_value, 'Optimal action:', optimal_agent.policy(init_state, 1))

            for agent_name, agent in agents.items():
                txt_labels.append(agent_name)
                print(f'Waiting for {agent_name}...')
                agent_instance = agent(env=env, discount_factor=env_params['discount_factor'])
                evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
                policy_value = evaluator.evaluate(init_state, 1)
                values[agent_name].append(policy_value)
                print(f'{agent_name} Done:', policy_value, f'{agent_name} action:', agent_instance.policy(init_state, 1))
        values['decision_epoch'] = decision_epochs
        df = pd.DataFrame(values)
        df.to_csv(data_file_path)
        opt_plot(df, 'period to go', plot_labels+['Optimal Policy'], image_path, text_labels=txt_labels)
        #opt_plot(df, 'period to go', plot_labels, 'test')

def sample_path_experiment(env_params, agents, sample_path_number, plot_labels, data_file_path, image_path):
    values = {agent_name: [] for agent_name in agents}
    Ms = [1]+[i for i in range(200, sample_path_number+1, 200)]
    values['num_sample_path'] = Ms
    env = AdvanceSchedulingEnv(**env_params)
    num_types = len(env_params['treatment_pattern'][0])

    # set up init state
    bookings = np.array([0])
    future_schedule = np.array([[0] * num_types for _ in range(env_params['decision_epoch'])])
    new_arrival = np.array([6]*num_types)
    init_state = (bookings, new_arrival, future_schedule)

    print('Waiting for Optimal...')
    optimal_agent = OptimalAgent(env=env, discount_factor=env_params['discount_factor'])
    optimal_agent.train(init_state, 1)
    print(init_state)
    optimal_value = optimal_agent.get_state_value(init_state, 1)
    print('Optimal Done:', optimal_value, 'Optimal action:', optimal_agent.policy(init_state, 1))

    txt_labels = []
    for agent_name, agent in agents.items():
        txt_labels.append(agent_name)
        print(f'Waiting for {agent_name}...')
        agent_instance = agent(env=env, discount_factor=env_params['discount_factor'])
        for M in Ms:
            print(f'Waiting for {agent_name} with sample path is {M}...')
            action, solution_y, obj_value = agent_instance.solve(init_state, 1, 0, M)
            values[agent_name].append(obj_value)
            print(f'{agent_name} Done:', obj_value, 'agent action:', action, 'optimal action value:', optimal_agent.get_action_value(init_state, action, 1))
    df = pd.DataFrame(values)
    df.to_csv(data_file_path)
    print(df)
    opt_plot(df=df,
             xlable='Number of Sample Path',
             plot_labels=plot_labels,
             save_file=image_path,
             text_labels=txt_labels,
             ylabel='Value Function',
             x_val_col='num_sample_path')


if __name__ == '__main__':

    decision_epoch = 4
    class_number = 2
    arrival_generator = MultiClassPoissonArrivalGenerator(3, 4, [1 / class_number] * class_number)
    env_params = {
        'treatment_pattern': [[2, 1]],
        'decision_epoch': decision_epoch,
        'arrival_generator': arrival_generator,
        'holding_cost': [10, 5],
        'overtime_cost': 40,
        'duration': 1,
        'regular_capacity': 5,
        'discount_factor': 0.99,
        'problem_type':'allocation'
    }

    agents = {
        'SAAllocationAdvanceAgent': SAAllocationAdvanceAgent
    }

    period_to_go_experiment(env_params, agents, 9, ['Sample Average Translate Policy'],
                            'data/sa_opt_allocation_compare.csv',"figures/sa_opt_allocation_compare")      
    '''
    sample_path_number = 3000
    sample_path_experiment(env_params=env_params,
                           agents=agents,
                           sample_path_number=sample_path_number,
                           plot_labels=[str(len(env_params['treatment_pattern'][0]))],
                           data_file_path='data/small_size_converge_test.csv',
                           image_path='figures/small_size_converge_test')
    

    bookings = np.array([0])
    future_schedule = np.array([[0] * class_number for i in range(decision_epoch)])
    new_arrival = np.array([5, 6])
    init_state = (bookings, new_arrival, future_schedule)
    t=1
    M=3000
    value_function_df, best_action_df = basic_converge_experiment(env_params, agents, init_state, t, M)
    print(value_function_df)
    print(best_action_df)
    '''


