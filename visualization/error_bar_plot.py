import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .utility import set_fontsize
def error_bar_plot(stats_by_sample_path, x_vals, xlabel, ylabel, plot_labels, save_file):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    for column_name, label in plot_labels.items():
        stats = stats_by_sample_path[column_name]
        means = [stats[j].mean()[0] for j in stats]
        ci_half = [stats[j].half_window(0.95)[0] for j in stats]
        # ------------------------------------------------------------------
        # Plot (one chart, default style)
        ax.errorbar(x_vals, means, yerr=ci_half, label=label, fmt='o', capsize=5, linewidth=1.5)
        for x, y in zip(x_vals, means):
            plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=20)
    ax.set_xticks(x_vals)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    set_fontsize(ax, 20)
    fig.tight_layout()
    plt.savefig(save_file)
    plt.show()

def error_bar_plot_from_running_stats_dict(running_stats_dict, x_vals, xticklabels, xlabel, ylabel, plot_labels, title, save_file, is_show_text=True):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    for label in plot_labels:
        running_stats_by_x_val = running_stats_dict[label]
        means = []
        half_window = []
        for x_val in x_vals:
            running_stats = running_stats_by_x_val[x_val]
            means.append(running_stats.expect[0])
            half_window.append(running_stats.half_window(0.95)[0])
        means = np.array(means)
        half_window = np.array(half_window)
        print(label, means, half_window)
        # ------------------------------------------------------------------
        # Plot (one chart, default style)
        ax.errorbar(x_vals, means, yerr=half_window, label=plot_labels[label], fmt='o', capsize=5, linewidth=1.5)
        if is_show_text:
            for x, y in zip(x_vals, means):
                plt.text(x, y, f"{y:.2f}", ha='center', va='bottom', fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = sorted(list(set(labels)))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    ax.set_xticks(x_vals)
    ax.set_xticklabels(xticklabels, rotation=0, fontsize=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title != None:
        ax.set_title(title, fontsize=20)
    set_fontsize(ax, 20)
    # Create legend
    ax.legend(unique_handles, unique_labels, fontsize=20)
    fig.tight_layout()
    print(save_file)
    plt.savefig(save_file)
    plt.show()