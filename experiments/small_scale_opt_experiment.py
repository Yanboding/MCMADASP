import json
import os
import time
from collections import defaultdict
from pprint import pprint

from decision_maker.heterogeneous_alp_agent import HeterogeneousALPAgent
from decision_maker.memory_efficient_mcma_agent import SAAdvanceFastAgent
from utils import iter_to_tuple, iter_to_list, RunningStat, get_uid

import numpy as np
import pandas as pd

from decision_maker import SAAdvanceAgent, OptimalAgent, PolicyEvaluator, \
    ApproxAllocationAdvanceAgent, ALPAgent
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
    action_values = defaultdict(lambda:defaultdict(lambda: RunningStat(1)))
    agent_best_action = defaultdict(lambda: (float('inf'), None, None))
    for agent_name, agent_instance in agents.items():
        best_action, action_val = agent_instance.solve(init_state, 1)
        agent_best_action[agent_name] = (action_val, None, best_action)
    for i, action in enumerate(env.valid_actions(init_state, 1)):
        x.append(i)
        for agent_name, agent_instance in agents.items():
            _, action_val = agent_instance.solve(init_state, 1, action)
            action_values[agent_name][i].record(action_val)
            if agent_best_action[agent_name][2] is not None and np.array_equal(agent_best_action[agent_name][2], action):
                agent_best_action[agent_name] = (action_val, i, action)
    for agent_name, (action_val, i, action) in agent_best_action.items():
        xticks.append(i)
        xticklabels.append(str(action))
    approximate_value_plot_from_running_stats_dict(running_stats_dict=action_values,
                                                   x_vals=sorted(x),
                                                   xticks=xticks,
                                                   xticklabels=xticklabels,
                                                   xlabel='Best Action',
                                                   ylabel="Value Function",
                                                   plot_labels=plot_labels,
                                                   title=None,
                                                   save_file='action_value_comparison',
                                                   is_show_text=False,
                                                   is_set_x_color=True)

def decision_epoch_experiment(config, agents, decision_epochs, plot_labels, replication=1000):
    x = []
    value_fuc_stats = defaultdict(lambda: defaultdict(lambda: RunningStat(1)))
    env = config.env
    for decision_epoch in decision_epochs:
        print('decision_epoch:', decision_epoch)
        x.append(decision_epoch)
        env.decision_epoch = decision_epoch
        init_state, info = env.reset(**config.reset_params)
        lower_bound_agent = SAAdvanceAgent(env=env, discount_factor=env.discount_factor)
        sample_paths = [env.reset_arrivals(t=1) for _ in range(replication)]
        for agent_name, (agent, args) in agents.items():
            agent_instant = agent(env=env, discount_factor=env.discount_factor, **args)
            policy_evaluator = PolicyEvaluator(env, agent_instant, env.discount_factor)
            for sample_path in sample_paths:
                uid = get_uid(sample_path.tolist())
                pct_gap = policy_evaluator.sample_path_optimality_gap_evaluate(lower_bound_agent, init_state, 1,
                                                                               sample_path)
                print(f'decision_epoch {decision_epoch}, agent_name {agent_name}, sample path id {uid}', pct_gap)
                value_fuc_stats[agent_name][decision_epoch].record(pct_gap)
    approximate_value_plot_from_running_stats_dict(running_stats_dict=value_fuc_stats,
                                                   x_vals=sorted(x),
                                                   xticks=x,
                                                   xticklabels=x,
                                                   xlabel='Number of periods',
                                                   ylabel="Value Function",
                                                   plot_labels= plot_labels,
                                                   title=None,
                                                   save_file='decision_epoch_value_comparison',
                                                   is_show_text=False,
                                                   is_set_x_color=False)
    '''
    with open('decision_epoch_experiment.jsonl', 'w') as f:
        f.write(json.dumps({'decision_epoch':decision_epoch, 'agent_name':agent_name, 'uid':uid, 'pct_gap':pct_gap}))
    '''


if __name__ == '__main__':
    from experiments import get_config_by_type

    config = get_config_by_type('default')
    #action_value_function_compare_experiment(config)
    agent_configs = {
        'Myopic Policy': (SAAdvanceAgent, {'sample_path_number': 500, 'is_myopic': True}),
        'Modify Myopic Policy': (SAAdvanceFastAgent, {'sample_path_number': 500, 'is_myopic': True}),
        'Homogeneous ALP Policy': (ALPAgent, {'pretrain': True}),
        'Heterogeneous ALP Policy': (HeterogeneousALPAgent, {'pretrain': True}),
        'Hindsight Approx Policy': (SAAdvanceAgent, {'sample_path_number': 500})
    }
    plot_labels = {
        'Myopic Policy': 'Myopic Policy',
        'Modify Myopic Policy': 'Modify Myopic Policy',
        'Homogeneous ALP Policy': 'Homogeneous ALP Policy',
        'Heterogeneous ALP Policy': 'Heterogeneous ALP Policy',
        'Hindsight Approx Policy': 'Hindsight Approx Policy',
    }
    action_value_function_compare_experiment(config, agent_configs,  plot_labels)
    decision_epoch_experiment(config, agent_configs, [decision_epoch for decision_epoch in range(3, 11)], plot_labels)