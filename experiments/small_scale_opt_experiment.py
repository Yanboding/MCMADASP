import os
import time
from collections import defaultdict
from pprint import pprint
from utils import iter_to_tuple, iter_to_list, RunningStat

import numpy as np
import pandas as pd

from decision_maker import SAAdvanceAgent, OptimalAgent, PolicyEvaluator, \
    ApproxAllocationAdvanceAgent
from visualization import opt_plot, approximate_value_plot_from_running_stats_dict
from environment import MultiClassPoissonArrivalGenerator

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
        agent_instance = agent(env=env, discount_factor=env_params['discount_factor'], sample_path_number=M)
        action, solution_y, obj_value = agent_instance.solve(init_state, t, 0)
        value_function_df[agent_name] = [obj_value]
        best_action_df[agent_name] = [action]
    value_function_df = pd.DataFrame(value_function_df)
    best_action_df = pd.DataFrame(best_action_df)
    return value_function_df, best_action_df

def basic_policy_evaluation_experiment(env_params, agents, init_state, t, M):
    env = AdvanceSchedulingEnv(**env_params)
    value_function_df = {}
    for agent_name, agent in agents.items():
        print(f'Waiting for {agent_name}...')
        agent_instance = agent(env=env, discount_factor=env_params['discount_factor'], sample_path_number=M)
        evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
        policy_value = evaluator.evaluate(init_state, t)
        value_function_df[agent_name] = [policy_value]
    value_function_df = pd.DataFrame(value_function_df)
    return value_function_df

def period_to_go_experiment(env_params, agents, period_num, init_arrival, plot_labels, data_file_path, image_path, skip_optimal=False, shortcut=False):
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
            init_state = (bookings, init_arrival, future_schedule)
            if skip_optimal == False:
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
        if skip_optimal == False:
            opt_plot(df, 'period to go', plot_labels+['Optimal Policy'], image_path, text_labels=txt_labels)
        else:
            opt_plot(df, 'period to go', plot_labels, image_path, text_labels=txt_labels)
        #opt_plot(df, 'period to go', plot_labels, 'test')

def sample_path_experiment(env_params, agents, sample_path_number, init_state, plot_labels, data_file_path, image_path, shortcut=False):
    if os.path.exists(data_file_path) and shortcut:
        print(f"Found '{data_file_path}'. Reading CSV file...")
        df = pd.read_csv(data_file_path,index_col=0)

        print("File loaded successfully!")
        print("Here are the first 5 rows:")
        print(df.head())
        txt_labels = []
        for agent_name, agent in agents.items():
            txt_labels.append(agent_name)
        opt_plot(df=df,
                 xlable='Number of Sample Path',
                 plot_labels=plot_labels + ['Optimal Value Function'],
                 save_file=image_path,
                 text_labels=txt_labels,
                 ylabel='Value Function',
                 x_val_col='num_sample_path')
    else:
        values = {agent_name: [] for agent_name in agents}
        Ms = [1]+[i for i in range(200, sample_path_number+1, 200)]
        values['num_sample_path'] = Ms
        env = AdvanceSchedulingEnv(**env_params)

        print('Waiting for Optimal...')
        optimal_agent = OptimalAgent(env=env, discount_factor=env_params['discount_factor'])
        optimal_agent.train(init_state, 1)
        print(init_state)
        optimal_value = optimal_agent.get_state_value(init_state, 1)
        print('Optimal Done:', optimal_value, 'Optimal action:', optimal_agent.policy(init_state, 1))
        values['OptimalAgent'] = [optimal_value for _ in Ms]
        txt_labels = []
        for agent_name, agent in agents.items():
            txt_labels.append(agent_name)
            print(f'Waiting for {agent_name}...')
            for M in Ms:
                print(f'Waiting for {agent_name} with sample path is {M}...')
                agent_instance = agent(env=env, discount_factor=env_params['discount_factor'], sample_path_number=M)
                action, solution_y, obj_value = agent_instance.solve(init_state, 1, 0)
                values[agent_name].append(obj_value)
                print(f'{agent_name} Done:', obj_value, 'agent action:', action, 'optimal action value:', optimal_agent.get_action_value(init_state, action, 1))
        df = pd.DataFrame(values)
        df.to_csv(data_file_path)
        print(df)
        opt_plot(df=df,
                 xlable='Number of Sample Path',
                 plot_labels=plot_labels+['Optimal Value Function'],
                 save_file=image_path,
                 text_labels=txt_labels,
                 ylabel='Value Function',
                 x_val_col='num_sample_path')

def action_value_function_compare_experiment(config):
    '''
    1. index actions by integers
    2. plot exact q value function on each action and information relaxation bound on each action
    '''
    env = config.env
    init_state = config.init_state
    optimal_agent = OptimalAgent(env=env, discount_factor=env.discount_factor)
    agent = SAAdvanceAgent(env=env, discount_factor=env.discount_factor)
    optimal_agent.train(init_state, 1)
    print('finish optimal')
    state_tuple = iter_to_tuple(init_state)
    x = []
    xticks = []
    xticklabels = []
    actions = []
    action_values = defaultdict(lambda:defaultdict(lambda: RunningStat(1)))
    min_value = float('inf')
    best_action = None
    best_index = None
    optimal_action = optimal_agent.policy(init_state, 1)
    optimal_action_tuple = iter_to_tuple(optimal_action)
    for i, (action, qValue) in enumerate(optimal_agent.Q[(state_tuple, 1)].items()):
        action_now = np.array(iter_to_list(action))
        x.append(i)
        actions.append(action)
        action_values['optimal_policy'][i].record(qValue)
        for j in range(10):
            agent.set_sample_paths(500)
            _, _, approximate_value = agent.solve(init_state, 1, action=action_now)
            action_values['hindsight_policy'][i].record(approximate_value)
        if action_values['hindsight_policy'][i].expect < min_value:
            min_value = action_values['hindsight_policy'][i].expect
            best_action = action_now
            best_index = i
        if action == optimal_action_tuple:
            xticks.append(i)
            xticklabels.append(str(optimal_action))
    xticks.append(best_index)
    xticklabels.append(str(best_action))
    print(optimal_action)
    print(best_action)

    approximate_value_plot_from_running_stats_dict(running_stats_dict=action_values,
                                                   x_vals=sorted(x),
                                                   xticks=xticks,
                                                   xticklabels=xticklabels,
                                                   xlabel='Best Action',
                                                   ylabel="Value Function",
                                                   plot_labels={'optimal_policy': 'Optimal Policy',
                                                                'hindsight_policy': 'Hindsight Approx Policy'
                                                                },
                                                   title=None,
                                                   save_file='action_value_comparison',
                                                   is_show_text=False)


if __name__ == '__main__':
    from experiments.experiment_config import ExperimentConfig
    config = ExperimentConfig.from_multiappt_default_case()
    action_value_function_compare_experiment(config)