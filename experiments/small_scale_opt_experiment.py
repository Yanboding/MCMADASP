import json
import os
import time
from collections import defaultdict
from pprint import pprint

from decision_maker.alp_cg_ejor_agent import ALPEJORColumnGenerationAgent
from decision_maker.alp_rg_ejor_agent import ALPEJORRowGenerationAgent
from decision_maker.hindsight_sa_agent import HindsightSAAgent
from utils import iter_to_tuple, iter_to_list, RunningStats, get_uid

import numpy as np
import pandas as pd

from decision_maker import SAAdvanceAgent, OptimalAgent, PolicyEvaluator, HeterogeneousALPColumnGenerationAgent, \
    HeterogeneousALPRowGenerationAgent
from visualization import opt_plot, approximate_value_plot_from_running_stats_dict
from environment import MultiClassPoissonArrivalGenerator

def action_value_function_compare_experiment(config, agent_configs, plot_labels):
    '''
    1. index actions by integers
    2. plot exact q value function on each action and information relaxation bound on each action
    '''
    env = config.env
    init_state = config.init_state
    agents = {agent_name: agent(env=env, discount_factor=env.discount_factor, **args) for agent_name, (agent, args) in agent_configs.items()}
    x = []
    xticks = []
    xticklabels = []
    action_values = defaultdict(lambda:defaultdict(lambda: RunningStats()))
    agent_best_action = defaultdict(lambda: (float('inf'), None, None))
    for agent_name, agent_instance in agents.items():
        best_action, action_val, info = agent_instance.solve(init_state, 1)
        agent_best_action[agent_name] = (action_val, None, best_action)
    df = []
    for i, action in enumerate(env.valid_actions(init_state, 1)):
        x.append(i)
        record = {'action':action}
        for agent_name, agent_instance in agents.items():
            _, action_val, info = agent_instance.solve(init_state, 1, action)
            record[agent_name] = action_val
            record.update(info)
            action_values[agent_name][i].record(action_val)
            if agent_best_action[agent_name][2] is not None and np.array_equal(agent_best_action[agent_name][2][0], action[0]):
                agent_best_action[agent_name] = (action_val, i, action[0])
        df.append(record)
    for agent_name, (action_val, i, action) in agent_best_action.items():
        xticks.append(i)
        xticklabels.append(str(action))
    print(agent_best_action)
    print(x)
    approximate_value_plot_from_running_stats_dict(running_stats_dict=action_values,
                                                   x_vals=sorted(x),
                                                   xticks=xticks,
                                                   xticklabels=xticklabels,
                                                   xlabel='Best Action',
                                                   ylabel="Value Function",
                                                   plot_labels=plot_labels,
                                                   title=None,
                                                   save_file='action_value_comparison_ALP',
                                                   is_show_text=False,
                                                   is_set_x_color=True)
    action_value_df = pd.DataFrame(df)
    action_value_df.to_excel('action_value.xlsx')

def decision_epoch_experiment(config, agents, decision_epochs, plot_labels, ylabel, replication=500):
    x = []
    value_fuc_stats = defaultdict(lambda: defaultdict(lambda: RunningStats()))
    env = config.env
    for decision_epoch in decision_epochs:
        print('decision_epoch:', decision_epoch)
        x.append(decision_epoch)
        env.set_decision_epoch(decision_epoch)
        init_state, info = env.reset(**config.reset_params)
        #lower_bound_agent = SAAdvanceAgent(env=env, discount_factor=env.discount_factor)
        lower_bound_agent = HindsightSAAgent(env=env, discount_factor=env.discount_factor)
        sample_paths = [env.reset_arrivals(t=1) for _ in range(replication)]
        for agent_name, (agent, args) in agents.items():
            agent_instant = agent(env=env, discount_factor=env.discount_factor, **args)
            policy_evaluator = PolicyEvaluator(env, agent_instant, env.discount_factor)
            for sid, sample_path in enumerate(sample_paths):
                uid = get_uid(sample_path.tolist())
                pct_gap = policy_evaluator.sample_path_optimality_gap_evaluate(lower_bound_agent, init_state, 1,
                                                                               sample_path)
                print(f'decision_epoch {decision_epoch}, agent_name {agent_name}, sample path id {uid} - {sid}', pct_gap)
                value_fuc_stats[agent_name][decision_epoch].record(pct_gap)
    approximate_value_plot_from_running_stats_dict(running_stats_dict=value_fuc_stats,
                                                   x_vals=sorted(x),
                                                   xticks=x,
                                                   xticklabels=x,
                                                   xlabel='Number of periods',
                                                   ylabel=ylabel,
                                                   plot_labels= plot_labels,
                                                   title=None,
                                                   save_file='decision_epoch_value_comparison',
                                                   is_show_text=True,
                                                   is_set_x_color=False)
    '''
    with open('decision_epoch_experiment.jsonl', 'w') as f:
        f.write(json.dumps({'decision_epoch':decision_epoch, 'agent_name':agent_name, 'uid':uid, 'pct_gap':pct_gap}))
    '''

def coefficient_plot(config, agents, prefix):
    x = []
    coefficient_fuc_stats = defaultdict(lambda: defaultdict(lambda: RunningStats()))
    env = config.env
    for agent_name, (agent, args) in agents.items():
        agent_instant = agent(env=env, discount_factor=env.discount_factor, **args)
        for n in range(1, env.booking_window_size + 1):
            x.append(n)
            for i in range(env.num_types):
                coefficient_fuc_stats[i][n].record(agent_instant.coeff_C(0, n))
    plot_labels = {key: f"{prefix} {key}" for key in coefficient_fuc_stats}
    approximate_value_plot_from_running_stats_dict(running_stats_dict=coefficient_fuc_stats,
                                                   x_vals=sorted(x),
                                                   xticks=x,
                                                   xticklabels=x,
                                                   xlabel='Workday',
                                                   ylabel="Coefficient Cin",
                                                   plot_labels=plot_labels,
                                                   title=None,
                                                   save_file='ejor_basecase',
                                                   is_show_text=False,
                                                   is_set_x_color=False)


def action_opt_pct_compare_experiment(config, agent_configs, plot_labels, replication=1000):
    env = config.env
    init_state = config.init_state
    new_arrival_t = config.init_state[1]
    x = set()
    xticks = []
    xticklabels = []
    action_values = defaultdict(lambda: defaultdict(lambda: RunningStats()))
    agent_best_action = defaultdict(lambda: (None, None))
    #sample_paths = [env.reset_arrivals(t=1) for _ in range(replication)]
    agents = {agent_name: agent(env=env, discount_factor=env.discount_factor, **args) for agent_name, (agent, args) in
              agent_configs.items()}
    for agent_name, agent_instance in agents.items():
        best_action, action_val, info = agent_instance.solve(init_state, 1)
        agent_best_action[agent_name] = (None, best_action)
    lower_bound_agent = SAAdvanceAgent(env=env, discount_factor=env.discount_factor)
    for agent_name, agent_instance in agents.items():
        policy_evaluator = PolicyEvaluator(env, agent_instance, env.discount_factor)
        for i, action in enumerate(env.valid_actions(init_state, 1)):
            x.add(i)
            if agent_best_action[agent_name][1] is not None and np.array_equal(agent_best_action[agent_name][1][0], action[0]):
                agent_best_action[agent_name] = (i, action[0])
            for _ in range(replication):
                sample_path = env.reset_arrivals(t=1)
                sample_path[0] = new_arrival_t
                uid = get_uid(sample_path.tolist())
                pct_gap = policy_evaluator.sample_path_optimality_gap_evaluate(lower_bound_agent, init_state, 1,
                                                                               sample_path, action)
                print('sample num:', _, 'uid:', uid, 'pct_gap:', pct_gap)
                action_values[agent_name][i].record(pct_gap)
    for agent_name, (i, action) in agent_best_action.items():
        xticks.append(i)
        xticklabels.append(str(action))
    approximate_value_plot_from_running_stats_dict(running_stats_dict=action_values,
                                                   x_vals=sorted(list(x)),
                                                   xticks=xticks,
                                                   xticklabels=xticklabels,
                                                   xlabel='Best Action',
                                                   ylabel="Percentage Optimality Gap (%)",
                                                   plot_labels=plot_labels,
                                                   title=None,
                                                   save_file='action_value_comparison_ALP',
                                                   is_show_text=False,
                                                   is_set_x_color=True)

if __name__ == '__main__':
    from experiments import get_config_by_type

    config = get_config_by_type('ejor_default')
    #action_value_function_compare_experiment(config)
    '''
    agent_configs = {
        'Myopic Policy': (HindsightSAAgent, {'sample_path_number': 500, 'is_myopic': True}),
        'Hindsight Approx Policy': (HindsightSAAgent, {'sample_path_number': 500}),
        #'Optimal Policy': (OptimalAgent, {'pretrain':True, 'state':config.init_state, 't':1})
    }
    '''
    agent_configs = {'ALP CG': (ALPEJORColumnGenerationAgent, {'pretrain': True})}
    plot_labels = {key:key for key in agent_configs}

    #plot_labels = {key: key for key in agent_configs}
    #action_value_function_compare_experiment(config, agent_configs,  plot_labels)
    #decision_epoch_experiment(config, agent_configs, [decision_epoch for decision_epoch in range(1, 11)], plot_labels, ylabel='Percentage Optimality Gap (%)')
    coefficient_plot(config, agent_configs, prefix='type')
    #action_opt_pct_compare_experiment(config, agent_configs, plot_labels)