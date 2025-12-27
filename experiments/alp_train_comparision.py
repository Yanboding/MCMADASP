import json
import os
from collections import defaultdict
from decision_maker import ALPEJORColumnGenerationAgent, ALPRowGenerationAgent

import matplotlib.pyplot as plt
import numpy as np

from utils import read_lines_with_pattern, RunningStats
from visualization import approximate_value_plot_from_running_stats_dict


def set_fontsize(ax, size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)

def coefficient_plot(directory_path, plot_labels, save_file='coefficient_plot.png'):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    alp_train_res = []
    for line in read_lines_with_pattern(directory_path, 'alp_train*.jsonl'):
        line = json.loads(line)
        res = line['result']
        agent_name = res['agent_name']
        coefficient = res['args']['coefficients'][1:]
        print('total:',len(coefficient))
        ax.plot(coefficient, label=plot_labels[agent_name], marker='o')
    set_fontsize(ax, 20)
    # To handle multiple lines with the same label, we need to manually create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = sorted(list(set(labels)))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    #ax.set_xticks(x_vals)
    #ax.set_xticklabels(x_vals, rotation=0, fontsize=20)
    ax.set_xlabel("Coefficient index", fontsize=20)
    ax.set_ylabel("Coefficient Value", fontsize=20)
    # Create legend
    ax.legend(unique_handles, unique_labels, fontsize=20)
    fig.tight_layout()
    plt.savefig(os.path.join(directory_path,save_file))
    plt.show()

def policy_coefficient_plot(directory_path, config, agents, prefix, title_labels):
    env = config.env
    for line in read_lines_with_pattern(directory_path, 'alp_train*.jsonl'):
        line = json.loads(line)
        agent = line['result']
        agent_name = agent['agent_name']
        args = agent['args']
        if agent_name == 'row_gen_alp':
            agent_instant = ALPRowGenerationAgent(env=env, discount_factor=env.discount_factor, **args)
        else:
            agent_instant = ALPEJORColumnGenerationAgent(env=env, discount_factor=env.discount_factor, **args)
        x = []
        coefficient_fuc_stats = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        for n in range(1, env.booking_window_size + 1):
            x.append(n)
            for i in range(env.num_types):
                coefficient_fuc_stats[i][n].record(agent_instant.coeff_C(i, n))
        plot_labels = {key: f"{prefix} {key}" for key in coefficient_fuc_stats}
        save_file = os.path.join(directory_path, f'{agent_name}_Cin_coefficients')
        approximate_value_plot_from_running_stats_dict(running_stats_dict=coefficient_fuc_stats,
                                                       x_vals=sorted(x),
                                                       xticks=x,
                                                       xticklabels=x,
                                                       xlabel='Workday',
                                                       ylabel="Coefficient Cin",
                                                       plot_labels=plot_labels,
                                                       title=f'{title_labels[agent_name]}, Obj Value: {agent["obj_val"]}',
                                                       save_file=save_file,
                                                       is_show_text=False,
                                                       is_set_x_color=False)

def penalty_coefficient_plot(directory_path, config):
    env = config.env
    coefficient_fuc_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: RunningStats())))
    x = []
    for line in read_lines_with_pattern(directory_path, 'alp_train*.jsonl'):
        line = json.loads(line)
        agent = line['result']
        agent_name = agent['agent_name']
        args = agent['args']
        print('agent_name:', agent_name)
        if agent_name == 'row_gen_alp':
            agent_instant = ALPRowGenerationAgent(env=env, discount_factor=env.discount_factor, **args)
        else:
            agent_instant = ALPEJORColumnGenerationAgent(env=env, discount_factor=env.discount_factor, **args)
        print(agent_instant.W)
        x = []
        param_value = agent['param_value']
        for i, w in enumerate(agent_instant.W):
            x.append(i)
            coefficient_fuc_stats[param_value][agent_name][i] += w
    plot_labels = {
        'row_gen_alp': "Row Generation ALP",
        #'col_gen_alp': 'Column Generation ALP',
    }
    for param_value in coefficient_fuc_stats:
        param_value_str = str(param_value).replace('.', '_')
        save_file = os.path.join(directory_path, f'alp_weights_by_{param_value_str}')
        title = f'Parameter Value: {param_value}'
        approximate_value_plot_from_running_stats_dict(running_stats_dict=coefficient_fuc_stats[param_value],
                                                       x_vals=sorted(x),
                                                       xticks=x,
                                                       xticklabels=np.array(x)+1,
                                                       xlabel='Type',
                                                       ylabel="Weight",
                                                       plot_labels=plot_labels,
                                                       title=title,
                                                       save_file=save_file,
                                                       is_show_text=True,
                                                       is_set_x_color=False)

if __name__ == '__main__':
    from experiments import get_config_by_type

    config = get_config_by_type('ejor')
    # Define the path to your results file
    plot_labels = {
                   #'col_gen_alp': 'Column Generation ALP',
                   'row_gen_alp': 'ALP',
    }
    experiment_lables = {'demand_rate': 'Arrival Rate',
                         'decision_epoch': 'Decision Epoch',
                         'overtime_cost_by_day':'Overtime Cost',
                         'occupancy_level': 'Occupancy Level',
                         'discount_factor': 'Discount Factor'}
    # Load and process the data
    #plot_experiment_result('results/demand_rate', plot_labels, experiment_lables)
    directory_path = 'results/occupancy_level_ejor_case_study'
    #coefficient_plot(directory_path, plot_labels

    # plot_labels = {key: key for key in agent_configs}
    # action_value_function_compare_experiment(config, agent_configs,  plot_labels)
    # decision_epoch_experiment(config, agent_configs, [decision_epoch for decision_epoch in range(1, 11)], plot_labels, ylabel='Percentage Optimality Gap (%)')
    penalty_coefficient_plot(directory_path, config)