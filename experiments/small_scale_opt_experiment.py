import os
import time
from collections import defaultdict
from pprint import pprint
from utils import iter_to_tuple, iter_to_list, RunningStat

import numpy as np
import pandas as pd

from decision_maker import SAAdvanceAgent, OptimalAgent, PolicyEvaluator, \
    ApproxAllocationAdvanceAgent, ALPAgent
from visualization import opt_plot, approximate_value_plot_from_running_stats_dict
from environment import MultiClassPoissonArrivalGenerator

def action_value_function_compare_experiment(config):
    '''
    1. index actions by integers
    2. plot exact q value function on each action and information relaxation bound on each action
    '''
    env = config.env
    init_state = config.init_state
    optimal_agent = OptimalAgent(env=env, discount_factor=env.discount_factor)
    agent = SAAdvanceAgent(env=env, discount_factor=env.discount_factor)
    alp_agent = ALPAgent(env=env, discount_factor=env.discount_factor)
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
    alp_min_value = float('inf')
    alp_best_action = None
    alp_best_index = None
    optimal_action = optimal_agent.policy(init_state, 1)
    optimal_action_tuple = iter_to_tuple(optimal_action)
    for i, (action, qValue) in enumerate(optimal_agent.Q[(state_tuple, 1)].items()):
        action_now = np.array(iter_to_list(action))
        x.append(i)
        actions.append(action)
        action_values['optimal_policy'][i].record(qValue)
        # alp policy
        _, approximate_value = alp_agent.solve(init_state, 1, action_now)
        action_values['alp_policy'][i].record(approximate_value)
        if action_values['alp_policy'][i].expect < alp_min_value:
            alp_min_value = action_values['alp_policy'][i].expect
            alp_best_action = action_now
            alp_best_index = i
        # hindsight policy
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
    xticks.append(alp_best_index)
    xticklabels.append(str(alp_best_action))
    xticks.append(best_index)
    xticklabels.append(str(best_action))
    print(optimal_action)
    print(alp_best_action)
    print(best_action)

    approximate_value_plot_from_running_stats_dict(running_stats_dict=action_values,
                                                   x_vals=sorted(x),
                                                   xticks=xticks,
                                                   xticklabels=xticklabels,
                                                   xlabel='Best Action',
                                                   ylabel="Value Function",
                                                   plot_labels={'optimal_policy': 'Optimal Policy',
                                                                'alp_policy': 'ALP Policy',
                                                                'hindsight_policy': 'Hindsight Approx Policy',
                                                                },
                                                   title=None,
                                                   save_file='action_value_comparison',
                                                   is_show_text=False,
                                                   is_set_x_color=True)

def decision_epoch_experiment(config, agents, decision_epochs, replication=1000):
    x = []
    value_fuc_stats = defaultdict(lambda: defaultdict(lambda: RunningStat(1)))
    env = config.env
    for decision_epoch in decision_epochs:
        x.append(decision_epoch)
        env.decision_epoch = decision_epoch
        init_state, info = env.reset(**config.reset_params)
        lower_bound_agent = SAAdvanceAgent(env=env, discount_factor=env.discount_factor)
        for agent_name, (agent, args) in agents.items():
            agent_instant = agent(env=env, discount_factor=env.discount_factor, **args)
            policy_evaluator = PolicyEvaluator(env, agent_instant, env.discount_factor)
            for r in range(replication):
                sample_path = env.reset_arrivals(t=1)
                pct_gap = policy_evaluator.sample_path_optimality_gap_evaluate(lower_bound_agent, init_state, 1, sample_path)
                #print(f'decision_epoch {decision_epoch}, agent_name {agent_name}, replication {r}', pct_gap)
                value_fuc_stats[agent_name][decision_epoch].record(pct_gap)
    approximate_value_plot_from_running_stats_dict(running_stats_dict=value_fuc_stats,
                                                   x_vals=sorted(x),
                                                   xticks=x,
                                                   xticklabels=x,
                                                   xlabel='Number of periods',
                                                   ylabel="Value Function",
                                                   plot_labels= {agent_name: agent_name for agent_name in agents.keys()},
                                                   title=None,
                                                   save_file='decision_epoch_value_comparison',
                                                   is_show_text=False,
                                                   is_set_x_color=False)


if __name__ == '__main__':
    from experiments import get_config_by_type

    config = get_config_by_type('default', 42)
    #action_value_function_compare_experiment(config)
    agents = {
        'ALP Policy': (ALPAgent, {}),
        'Hindsight Approx Policy': (SAAdvanceAgent, {'sample_path_number': 500})
    }
    decision_epoch_experiment(config, agents, [decision_epoch for decision_epoch in range(1,11)])