import matplotlib.pyplot as plt
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