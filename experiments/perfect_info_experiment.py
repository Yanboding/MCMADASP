import time
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

from decision_maker.approx_allocation_to_advance_agent import ApproxAllocationAdvanceAgent
from decision_maker.myopic_allocation_to_advance_agent import MyopicAllocationAdvanceAgent
from decision_maker.policy_evaluator import PolicyEvaluator
from decision_maker.so_allocation_to_advance_agent import SAAllocationAdvanceAgent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decision_maker import OptimalAgent, HindsightApproximationAgent
from environment import AdvanceSchedulingEnv


def set_fontsize(ax, size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)

def opt_plot(df, xlable, plot_labels, save_file, text_labels=[], ylabel='Value Function', x_val_col='decision_epoch'):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    x_vals = df[x_val_col]
    for column_name, label in zip(df.drop(columns=[x_val_col]), plot_labels):
        ax.plot(x_vals, df[column_name], label=label, marker='o')
    for text_label in text_labels:
        for l, txt in enumerate(df[text_label]):
            ax.text(x_vals[l], df[text_label][l], str(round(txt, 3)), ha='center', va='bottom', fontsize=20)

    set_fontsize(ax, 20)
    # To handle multiple lines with the same label, we need to manually create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = sorted(list(set(labels)))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_vals, rotation=0, fontsize=20)
    ax.set_xlabel(xlable, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    # Create legend
    ax.legend(unique_handles, unique_labels, fontsize=20)
    fig.tight_layout()
    plt.savefig(save_file)
    plt.show()

def period_to_go_experiment(env_params, period_num):
    optimal_values = []
    mip_values = []
    mip_uppers = []
    mip_lowers = []
    num_types = len(env_params['treatment_pattern'])
    for decision_epoch in range(1, period_num+1):
        print('decision epoch:', decision_epoch)
        env_params['decision_epoch'] = decision_epoch
        env = AdvanceSchedulingEnv(**env_params)
        optimal_agent = OptimalAgent(env=env, discount_factor=0.99)
        # I need a distribution to initiate the first waiting list
        bookings = np.array([0])
        future_schedule = np.array([[0]*num_types for _ in range(decision_epoch)])
        new_arrival = np.array([0]*num_types)
        init_state = (bookings, new_arrival, future_schedule)
        optimal_agent.train(init_state, 1)
        optimal_value = optimal_agent.get_state_value(init_state, 1)
        optimal_values.append(optimal_value)
        evaluator = PolicyEvaluator(env, None, env.discount_factor)
        mip_value, mip_lower, mip_upper  = evaluator.simulation_evaluate(init_state, 1000)
        mip_values.append(mip_value)
        mip_uppers.append(mip_upper[0])
        mip_lowers.append(mip_lower[0])
    df = pd.DataFrame({
        'decision_epoch': [i+1 for i in range(period_num)],
        'optimal_value': optimal_values,
        'mip_value': mip_values,
        'mip_upper': mip_uppers,
        'mip_lower': mip_lowers
    })
    opt_plot(df, 'period to go', ['Optimal policy', 'Perfect Info Relaxation'], 'test')

def opt_hindsight_experiment(env_params, period_num):
    optimal_values = []
    hindsight_values = []
    num_types = len(env_params['treatment_pattern'])
    for decision_epoch in range(1, period_num + 1):
        env_params['decision_epoch'] = decision_epoch
        # Env and init states
        env = AdvanceSchedulingEnv(**env_params)
        # I need a distribution to initiate the first waiting list
        bookings = np.array([0])
        future_schedule = np.array([[0] * num_types for _ in range(decision_epoch)])
        new_arrival = np.array([0] * num_types)
        init_state = (bookings, new_arrival, future_schedule)
        # Optimal Agent
        optimal_agent = OptimalAgent(env=env, discount_factor=env_params['discount_factor'])
        optimal_agent.train(init_state, 1)
        optimal_value = optimal_agent.get_state_value(init_state, 1)
        optimal_values.append(optimal_value)
        # Hindsight Agent
        hindsight_agent = HindsightApproximationAgent(env, discount_factor=env_params['discount_factor'], num_sample_paths=10)
        evaluator = PolicyEvaluator(env, hindsight_agent, env.discount_factor)
        hindsight_values.append(evaluator.evaluate(init_state, 1))
    df = pd.DataFrame({
        'decision_epoch': [i + 1 for i in range(period_num)],
        'optimal_value': optimal_values,
        'hindsight_values': hindsight_values,
    })
    opt_plot(df, 'period to go', ['Optimal policy', 'Hindsight approximation'], 'test')

def opt_myopic_experiment(env_params, period_num):
    optimal_values = []
    myopic_values = []
    for decision_epoch in range(1, period_num + 1):
        env_params['decision_epoch'] = decision_epoch
        # Env and init states
        env = AdvanceSchedulingEnv(**env_params)
        # I need a distribution to initiate the first waiting list
        bookings = np.array([0])
        future_schedule = np.array([[0] * env.num_types for _ in range(decision_epoch)])
        new_arrival = np.array([2] * env.num_types)
        init_state = (bookings, new_arrival, future_schedule)
        # Optimal Agent
        optimal_agent = OptimalAgent(env=env, discount_factor=env_params['discount_factor'])
        optimal_agent.train(init_state, 1)
        optimal_value = optimal_agent.get_state_value(init_state, 1)
        optimal_values.append(optimal_value)
        # Myopic Agent
        myopic_agent = MyopicAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'])
        evaluator = PolicyEvaluator(env, myopic_agent, env.discount_factor)
        myopic_values.append(evaluator.evaluate(init_state, 1))
    df = pd.DataFrame({
        'decision_epoch': [i + 1 for i in range(period_num)],
        'optimal_value': optimal_values,
        'hindsight_values': myopic_values,
    })
    opt_plot(df, 'period to go', ['Optimal policy', 'Myopic Allocation Translation'], 'test')

def opt_approximation_experiment(env_params, period_num):
    optimal_values = []
    myopic_values = []
    so_values = []
    approx_values = []
    for decision_epoch in range(1, period_num + 1):
        print('decision_epoch:', decision_epoch)
        env_params['decision_epoch'] = decision_epoch
        # Env and init states
        env = AdvanceSchedulingEnv(**env_params)
        # I need a distribution to initiate the first waiting list
        bookings = np.array([0])
        future_schedule = np.array([[0] * env.num_types for _ in range(decision_epoch)])
        new_arrival = np.array([3] * env.num_types)
        init_state = (bookings, new_arrival, future_schedule)
        # Optimal Agent
        optimal_agent = OptimalAgent(env=env, discount_factor=env_params['discount_factor'])
        optimal_agent.train(init_state, 1)
        optimal_value = optimal_agent.get_state_value(init_state, 1)
        optimal_values.append(optimal_value)
        '''
        # Myopic Agent
        myopic_agent = MyopicAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'])
        myopic_evaluator = PolicyEvaluator(env, myopic_agent, env.discount_factor)
        myopic_values.append(myopic_evaluator.evaluate(init_state, 1))
        '''
        # Simulation Optimization Agent
        so_agent = SAAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'])
        so_evaluator = PolicyEvaluator(env, so_agent, env.discount_factor)
        so_values.append(so_evaluator.evaluate(init_state, 1))
        # Stochastic Approximate Agent
        approx_agent = ApproxAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'])
        approx_evaluator = PolicyEvaluator(env, approx_agent, env.discount_factor)
        approx_values.append(approx_evaluator.evaluate(init_state, 1))
    df = pd.DataFrame({
        'decision_epoch': [i + 1 for i in range(period_num)],
        'optimal_value': optimal_values,
        'so_values': so_values,
        'approx_values': approx_values
    })
    df.to_csv('allocation.csv')
    txt_labels = ['optimal_value', 'so_values','approx_values']
    opt_plot(df, 'period to go', ['Optimal', 'Simulation Optimization', 'Stochastic'], 'allocation', txt_labels)

def arrival_rate_experiment(env_params, arrival_rates):
    sa_values = []
    for arrival_rate in arrival_rates:
        system_dynamic = get_system_dynamic(arrival_rate, 3*arrival_rate, [0.5, 0.5])
        regular_capacity = arrival_rate
        env_params['system_dynamic'] = system_dynamic
        env_params['regular_capacity'] = regular_capacity
        env = AdvanceSchedulingEnv(**env_params)
        so_agent = SAAllocationAdvanceAgent(env, discount_factor=env_params['discount_factor'])

def sample_path_experiment(env_params, state, t, num_sample_path):
    sa_value_function_df = {}
    runtime_df = {}
    Ms = [1]+[i for i in range(200, num_sample_path+1, 200)]
    sa_value_function_df['num_sample_path'] = Ms
    runtime_df['num_sample_path'] = Ms
    class_numbers = [2, 5]
    for class_number in [2, 5]:
        probability = 1 / class_number
        system_dynamic = get_system_dynamic_fast(10, 30, [probability] * class_number)
        holding_cost = [10 - i * (5 / (class_number - 1)) for i in range(class_number)]
        treatment_pattern = [[1]*class_number]
        env_params['treatment_pattern'] = treatment_pattern
        env_params['system_dynamic'] = system_dynamic
        env_params['holding_cost'] = holding_cost
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
    sa_value_function_df = pd.DataFrame(sa_value_function_df)
    runtime_df = pd.DataFrame(runtime_df)
    pprint(optimal_actions)
    sa_value_function_df.to_csv('converge_test.csv')
    runtime_df.to_csv('runtime.csv')
    print(sa_value_function_df)
    sa_value_function_text_labels = [label for label in sa_value_function_df.drop(columns=['num_sample_path'])]
    opt_plot(sa_value_function_df, 'Number of Sample Path', [str(i)+' types' for i in class_numbers], 'converge_test', sa_value_function_text_labels, x_val_col='num_sample_path')
    runtime_text_labels = [label for label in runtime_df.drop(columns=['num_sample_path'])]
    opt_plot(runtime_df, 'Number of Sample Path', [str(i)+' types' for i in class_numbers], 'runtime_test', runtime_text_labels, ylabel='Seconds', x_val_col='num_sample_path')

if __name__ == '__main__':
    from environment import AdvanceSchedulingEnv
    from environment.utility import get_system_dynamic, get_system_dynamic_fast
    '''
    env_params = {
        'decision_epoch': 20,
        'overtime_cost': 30,
        'duration': 1,
        'regular_capacity': 12,
        'discount_factor': 0.99,
        'problem_type': 'allocation'
    }
    '''
    env_params = {
        'treatment_pattern': [[2, 1]],
        'decision_epoch': 3,
        'arrival_generator': None,
        'system_dynamic': get_system_dynamic(3, 4, [0.5, 0.5]),
        'holding_cost': [10, 5],
        'overtime_cost': 40,
        'duration': 1,
        'regular_capacity': 12,
        'discount_factor': 0.99,
        'problem_type': 'allocation'
    }
    opt_approximation_experiment(env_params, 4)

    '''
    num_types = len(env_params['treatment_pattern'][0])
    bookings = np.array([0])
    future_schedule = np.array([[0] * num_types for _ in range(env_params['decision_epoch'])])
    new_arrival = np.array([6, 6])
    init_state = (bookings, new_arrival, future_schedule)

    sample_path_experiment(env_params, init_state, 1, 400)
    '''