import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

def approximate_value_plot(df, xlabel, ylabel, approx_labels, text_labels, plot_labels, save_file, x_val_col='decision_epoch'):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    x_vals = df[x_val_col]
    for column_name, label in plot_labels.items():
        ax.plot(x_vals, df[column_name], label=label, marker='o')
    for text_label in text_labels:
        for l, txt in enumerate(df[text_label]):
            ax.text(x_vals[l], df[text_label][l], str(round(txt, 3)), ha='center', va='bottom', fontsize=20)
    for approx_label in approx_labels:
        ax.fill_between(x_vals, df[approx_label+'_lower'], df[approx_label+'_upper'], alpha=0.2)

    set_fontsize(ax, 20)
    # To handle multiple lines with the same label, we need to manually create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = sorted(list(set(labels)))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_vals, rotation=0, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    # Create legend
    ax.legend(unique_handles, unique_labels, fontsize=20)
    fig.tight_layout()
    plt.savefig(save_file)
    plt.show()

def approximate_value_plot_from_running_stats_dict(running_stats_dict, x_vals, xticks, xticklabels, xlabel, ylabel, plot_labels, title, save_file, is_show_text=True, is_set_x_color=False):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    lines = []
    for label in plot_labels:
        running_stats_by_x_val = running_stats_dict[label]
        means = []
        half_window = []
        for x_val in x_vals:
            running_stats = running_stats_by_x_val[x_val]
            means.append(running_stats.mean)
            half_window.append(running_stats.half_window(0.95))
        means = np.array(means).reshape(-1)
        print('this is means', means)
        half_window = np.array(half_window).reshape(-1)
        (line,) = ax.plot(x_vals, means, label=plot_labels[label], marker='o')
        lines.append(line)
        ax.fill_between(x_vals, means-half_window, means+half_window, alpha=0.2)
        if is_show_text:
            for x, y, hw in zip(x_vals, means, half_window):
                # offset = max(hw * 1.1, 0.02)  # Ensure a minimum offset
                ax.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=16)
    set_fontsize(ax, 30)
    # To handle multiple lines with the same label, we need to manually create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = sorted(list(set(labels)))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    if xticklabels is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0, fontsize=20)
        xtick_labels = ax.get_xticklabels()
        if is_set_x_color:
            for label, line in zip(xtick_labels, lines):
                label.set_color(line.get_color())
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)
    ax.set_title(title, fontsize=30)
    if title != None:
        ax.set_title(title, fontsize=30)
    # Create legend
    ax.legend(unique_handles, unique_labels, fontsize=20)
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(save_file)
    plt.show()

def approximate_value_plot_from_multid_running_stats(running_stats_dict, x_vals, xlabel, ylabel, plot_labels, title, save_file):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    for label, running_stats in running_stats_dict.items():
        means = running_stats.expect
        half_window = running_stats.half_window(0.95)
        ax.plot(x_vals, means, label=plot_labels[label], marker='o')
        ax.fill_between(x_vals, means-half_window, means+half_window, alpha=0.2)
        for l, txt in enumerate(means):
            ax.text(x_vals[l], means[l], str(round(txt, 3)), ha='center', va='bottom', fontsize=20)
    set_fontsize(ax, 20)
    # To handle multiple lines with the same label, we need to manually create a custom legend
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = sorted(list(set(labels)))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_vals, rotation=0, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_title(title, fontsize=20)
    # Create legend
    ax.legend(unique_handles, unique_labels, fontsize=20)
    fig.tight_layout()
    plt.savefig(save_file)
    plt.show()


if __name__ == "__main__":
    df = pd.DataFrame({
        'decision_epoch': [1, 2, 3],
        'policy_value': [10, 20, 30],
        'policy_value_lower': [5, 5, 20],
        'policy_value_upper': [15, 25, 33],
        'hindsight_value': [5, 7, 9]
    })
    approximate_value_plot(df,
                           xlabel='period to go',
                           ylabel='Value function',
                           approx_labels=['policy_value'],
                           text_labels=['policy_value', 'hindsight_value'],
                           plot_labels={'policy_value':"Policy Value", 'hindsight_value':"Hindsight Value"},
                           save_file='test1',
                           x_val_col='decision_epoch')