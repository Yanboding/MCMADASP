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

def approximate_value_plot_from_running_stats_dict(running_stats_dict, x_vals, xticks, xticklabels, xlabel, ylabel, plot_labels, title, save_file, is_show_text=True, is_set_x_color=False, ncol=2):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    lines = []
    if x_vals is None:
        x_vals = sorted(list(next(iter(running_stats_dict.values())).keys()))
    for label in plot_labels:
        running_stats_by_x_val = running_stats_dict[label]
        means = []
        half_window = []
        for x_val in x_vals:
            running_stats = running_stats_by_x_val[x_val]
            means.append(running_stats.mean)
            half_window.append(running_stats.half_window(0.95))
        means = np.array(means).reshape(-1)
        half_window = np.array(half_window).reshape(-1)
        (line,) = ax.plot(x_vals, means, label=plot_labels[label], marker='o')
        lines.append(line)
        ax.fill_between(x_vals, means-half_window, means+half_window, alpha=0.2)
        if is_show_text:
            for x, y, hw in zip(x_vals, means, half_window):
                # offset = max(hw * 1.1, 0.02)  # Ensure a minimum offset
                ax.text(x, y, f"{y:.0f}", ha='center', va='bottom', fontsize=20)
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
    ax.set_xlabel(xlabel, fontsize=33)
    ax.set_ylabel(ylabel, fontsize=33)
    ax.set_title(title, fontsize=33)
    if title != None:
        ax.set_title(title, fontsize=30)
    # --- MODIFIED LEGEND SECTION ---
    # 1. bbox_to_anchor=(0.5, -0.15):
    #    x=0.5 centers it horizontally.
    #    y=-0.15 pushes it down below the axis.
    #    (You may need to adjust -0.15 to -0.20 if your xlabel is very tall)
    # 2. loc='upper center':
    #    Aligns the TOP CENTER of the legend box to the anchor point defined above.
    ax.legend(
        unique_handles,
        unique_labels,
        fontsize=25,
        borderaxespad=0.,
        bbox_to_anchor=(0.5, -0.15),
        ncol=ncol,
        loc='upper center'
    )
    ax.grid(True)
    # This targets the '1e6' text at the top of the axis
    ax.yaxis.get_offset_text().set_fontsize(20)
    fig.tight_layout()
    plt.savefig(save_file, bbox_inches='tight', format='svg')
    # plt.show()


def approximate_value_plot_from_running_stats(
    running_stats_dict,
    xlabel,
    ylabel,
    title,
    save_file,
    line_label=None,
):
    # ----- Prepare data -----
    x_vals = sorted(running_stats_dict.keys())
    means = np.array([running_stats_dict[x].mean for x in x_vals]).reshape(-1)
    half_window = np.array([running_stats_dict[x].half_window(0.95) for x in x_vals]).reshape(-1)

    # ----- Figure / axes -----
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    line_color = "#2f6df6"

    # ----- Main plot -----
    ax.plot(
        x_vals,
        means,
        marker="o",
        markersize=10,
        linewidth=3,
        color=line_color,
        label=line_label,
        zorder=3,
    )

    ax.fill_between(
        x_vals,
        means - half_window,
        means + half_window,
        color=line_color,
        alpha=0.18,
        zorder=2,
    )

    # ----- Point annotations -----
    y_range = max(means.max() - means.min(), 1e-8)
    text_offset = 0.03 * y_range

    for x, y in zip(x_vals, means):
        ax.text(
            x,
            y + text_offset,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=16,
            color="#222222",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
            zorder=4,
        )

    # ----- Labels / title -----
    ax.set_xlabel(xlabel, fontsize=20, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=20, labelpad=12)
    ax.set_title(title, fontsize=24, weight="bold", pad=18)

    # ----- Ticks -----
    ax.set_xticks(x_vals)
    ax.tick_params(axis="both", labelsize=18)

    # ----- Grid / spines -----
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(axis="x", visible=False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ----- Limits / margins -----
    lower = np.min(means - half_window)
    upper = np.max(means + half_window)
    pad = 0.08 * max(upper - lower, 1e-8)
    ax.set_ylim(lower - pad, upper + 2 * pad)

    # ----- Legend -----
    if line_label is not None:
        ax.legend(frameon=False, fontsize=18)

    # ----- Save / show -----
    fig.tight_layout()
    fig.savefig(save_file, bbox_inches="tight", dpi=300)
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