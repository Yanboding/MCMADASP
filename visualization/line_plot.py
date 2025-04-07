import matplotlib.pyplot as plt

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