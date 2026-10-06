import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.summarize_potential_improvement import load_records, summarize

SERIES_COLOR = '#2a78d6'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID_COLOR = '#e4e3df'


def occupancy_summaries(results_dir, experiment_prefix, occupancies):
    return {occupancy: summarize(load_records(os.path.join(results_dir, f'{experiment_prefix}{occupancy}')))
            for occupancy in occupancies}


def plot_improvement_by_occupancy(summaries, save_file, z=1.96):
    occupancies = sorted(summaries)
    percentages = np.array([summaries[o]['improvement_percentage'] for o in occupancies])
    half_widths = z * np.array([summaries[o]['improvement_percentage_standard_error'] for o in occupancies])
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(occupancies, percentages, color=SERIES_COLOR, linewidth=2, zorder=2)
    ax.errorbar(occupancies, percentages, yerr=half_widths, fmt='o', color=SERIES_COLOR, markersize=8,
                markeredgecolor='white', markeredgewidth=2, elinewidth=2, capsize=4, zorder=3)
    for occupancy, percentage, half_width in zip(occupancies, percentages, half_widths):
        ax.annotate(f'{percentage:.1f}%', (occupancy, percentage + half_width), textcoords='offset points',
                    xytext=(0, 6), ha='center', fontsize=10, color=TEXT_PRIMARY)
    ax.set_xticks(occupancies)
    ax.set_xticklabels([f'{o}%' for o in occupancies])
    ax.set_xlim(occupancies[0] - 10, occupancies[-1] + 10)
    ax.set_ylim(0, (percentages + half_widths).max() * 1.2)
    ax.set_xlabel('Initial occupancy level', fontsize=11, color=TEXT_PRIMARY)
    ax.set_ylabel('Potential improvement\n(% of penalized lower bound)', fontsize=11, color=TEXT_PRIMARY)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=10, length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID_COLOR)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
    fig.savefig(save_file, bbox_inches='tight')
    fig.savefig(os.path.splitext(save_file)[0] + '.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Potential improvement by initial occupancy level.')
    parser.add_argument('--results-dir', default=os.path.join('experiments', 'results'))
    parser.add_argument('--prefix', default='toy_improvement_occ')
    parser.add_argument('--occupancies', default='20,50,90')
    parser.add_argument('--save', default=os.path.join('experiments', 'figures', 'toy_potential_improvement_by_occupancy.svg'))
    args = parser.parse_args(argv)
    summaries = occupancy_summaries(args.results_dir, args.prefix, [int(o) for o in args.occupancies.split(',')])
    for occupancy, summary in sorted(summaries.items()):
        print(f"{occupancy}%: M={summary['M']}, gap {summary['potential_gap']:.1f} +/- "
              f"{1.96 * summary['standard_error']:.1f}, penalized LB {summary['penalized_information_relaxation_cost']:.1f}, "
              f"improvement {summary['improvement_percentage']:.2f}% +/- "
              f"{1.96 * summary['improvement_percentage_standard_error']:.2f}%, "
              f"violations {summary['pathwise_violations']}")
    plot_improvement_by_occupancy(summaries, args.save)
    print(f"Saved {args.save}")


if __name__ == '__main__':
    main()
