"""Extract per-iteration Benders training statistics from a slurm log.

Parses ``UB: ...`` and ``Iteration k, subproblem objective mean m +/- h (95%
CI, N=n)`` lines emitted by ``BendersDecompositionSolver.solve`` during penalty
coefficient training, writes them to a CSV, and renders two figures:

1. Upper bound vs. subproblem objective mean per iteration (convergence view).
2. Subproblem objective mean with its 95% CI band (sampling-variance view).

Each figure has a full-range panel (symlog y: early iterations are off-scale
by design — the cut-less master pushes coefficients to huge magnitudes) and a
linear zoom on the last iterations where the converged scale is readable.

Run from the repo root:

    python -m test.report_benders_subproblem_ci experiments/results/slurm-19386787.out
"""
import argparse
import csv
import os
import re

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

UB_LINE = re.compile(r'^UB: ([-+0-9.eE]+), LB: ([-+0-9.eE]+), Gap: ')
CI_LINE = re.compile(
    r'^Iteration (\d+), subproblem objective mean ([-+0-9.eE]+) '
    r'\+/- ([-+0-9.eE]+) \(95% CI, N=(\d+)\)'
)

# Validated 2-series categorical pair (CVD-safe; see dataviz palette).
COLOR_UB = '#2a78d6'
COLOR_MEAN = '#e8963a'
INK = '#1a1a19'
INK_MUTED = '#52514e'


def parse_log(log_path):
    """Return one row per iteration: the CI line closes the iteration and is
    paired with the most recent UB/LB line above it."""
    rows = []
    last_bounds = None
    with open(log_path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            bound_match = UB_LINE.match(line)
            if bound_match:
                last_bounds = (float(bound_match.group(1)), float(bound_match.group(2)))
                continue
            ci_match = CI_LINE.match(line)
            if ci_match:
                if last_bounds is None:
                    raise ValueError(f'CI line before any UB line: {line!r}')
                rows.append({
                    'iteration': int(ci_match.group(1)),
                    'upper_bound': last_bounds[0],
                    'lower_bound': last_bounds[1],
                    'subproblem_obj_mean': float(ci_match.group(2)),
                    'ci_half_width': float(ci_match.group(3)),
                    'n_scenarios': int(ci_match.group(4)),
                })
    return rows


def write_csv(rows, csv_path):
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _style_axis(ax):
    ax.grid(True, alpha=0.25, linewidth=0.5)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(INK_MUTED)


def plot_convergence(rows, out_path, zoom_last=200):
    iterations = [r['iteration'] for r in rows]
    ub = [r['upper_bound'] for r in rows]
    mean = [r['subproblem_obj_mean'] for r in rows]

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(10, 8))
    for ax, (xs, ys_ub, ys_mean) in (
        (ax_full, (iterations, ub, mean)),
        (ax_zoom, (iterations[-zoom_last:], ub[-zoom_last:], mean[-zoom_last:])),
    ):
        ax.plot(xs, ys_ub, color=COLOR_UB, linewidth=2, label='Upper bound (UB)')
        ax.plot(xs, ys_mean, color=COLOR_MEAN, linewidth=2,
                label='Subproblem objective mean')
        _style_axis(ax)
        ax.set_xlabel('Iteration', color=INK)
        ax.set_ylabel('Objective value', color=INK)

    ax_full.set_yscale('symlog')
    ax_full.set_title('Full run (symlog scale: early iterations are off-scale)',
                      color=INK, fontsize=10)
    ax_zoom.set_title(f'Last {zoom_last} iterations (linear scale)',
                      color=INK, fontsize=10)
    ax_full.legend(frameon=False, labelcolor=INK)
    fig.suptitle('Benders training convergence: UB vs. subproblem objective mean',
                 color=INK)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ci_band(rows, out_path, zoom_last=200):
    iterations = [r['iteration'] for r in rows]
    mean = [r['subproblem_obj_mean'] for r in rows]
    low = [r['subproblem_obj_mean'] - r['ci_half_width'] for r in rows]
    high = [r['subproblem_obj_mean'] + r['ci_half_width'] for r in rows]

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(10, 8))
    for ax, sl in ((ax_full, slice(None)), (ax_zoom, slice(-zoom_last, None))):
        ax.fill_between(iterations[sl], low[sl], high[sl], color=COLOR_MEAN,
                        alpha=0.25, linewidth=0, label='95% CI band')
        ax.plot(iterations[sl], mean[sl], color=COLOR_MEAN, linewidth=2,
                label='Subproblem objective mean')
        ax.axhline(0, color=INK_MUTED, linewidth=0.5, alpha=0.5)
        _style_axis(ax)
        ax.set_xlabel('Iteration', color=INK)
        ax.set_ylabel('Objective value', color=INK)

    ax_full.set_yscale('symlog')
    ax_full.set_title('Full run (symlog scale)', color=INK, fontsize=10)
    ax_zoom.set_title(f'Last {zoom_last} iterations (linear scale): the CI band '
                      'dwarfs the mean', color=INK, fontsize=10)
    ax_full.legend(frameon=False, labelcolor=INK)
    fig.suptitle('Subproblem objective mean with 95% confidence band', color=INK)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('log_path', nargs='?',
                        default=os.path.join('experiments', 'results',
                                             'slurm-19386787.out'))
    parser.add_argument('--output-dir', default=None,
                        help='defaults to the log file directory')
    parser.add_argument('--zoom-last', type=int, default=200,
                        help='iterations shown in the linear zoom panels')
    args = parser.parse_args()

    rows = parse_log(args.log_path)
    if not rows:
        raise SystemExit(f'No iteration statistics found in {args.log_path}')

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.log_path))
    stem = os.path.splitext(os.path.basename(args.log_path))[0]
    csv_path = os.path.join(out_dir, f'{stem}_subproblem_ci.csv')
    fig1_path = os.path.join(out_dir, f'{stem}_convergence.svg')
    fig2_path = os.path.join(out_dir, f'{stem}_ci_band.svg')

    write_csv(rows, csv_path)
    plot_convergence(rows, fig1_path, zoom_last=args.zoom_last)
    plot_ci_band(rows, fig2_path, zoom_last=args.zoom_last)

    final = rows[-1]
    relative = (final['ci_half_width'] / abs(final['subproblem_obj_mean'])
                if final['subproblem_obj_mean'] else float('inf'))
    print(f'Parsed {len(rows)} iterations from {args.log_path}')
    print(f'CSV:      {csv_path}')
    print(f'Figure 1: {fig1_path}')
    print(f'Figure 2: {fig2_path}')
    print(f"Final iteration {final['iteration']}: "
          f"subproblem objective mean = {final['subproblem_obj_mean']:.4f}, "
          f"95% CI half-width = {final['ci_half_width']:.4f} "
          f"(N={final['n_scenarios']}, relative half-width = {relative:.1%})")


if __name__ == '__main__':
    main()
