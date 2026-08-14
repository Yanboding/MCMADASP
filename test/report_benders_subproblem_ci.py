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

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 13,
    'figure.titlesize': 17,
})

UB_LINE = re.compile(r'^UB: ([-+0-9.eE]+), LB: ([-+0-9.eE]+), Gap: ')
CI_LINE = re.compile(
    r'^Iteration (\d+), subproblem objective mean ([-+0-9.eE]+) '
    r'\+/- ([-+0-9.eE]+) \(95% CI, N=(\d+)\)'
)
ZERO_LINE = re.compile(
    r'^Zero-penalty subproblem objective mean ([-+0-9.eE]+) '
    r'\+/- ([-+0-9.eE]+) \(95% CI, N=(\d+)\)'
)

# Validated 2-series categorical pair (CVD-safe; see dataviz palette).
COLOR_UB = '#2a78d6'
COLOR_MEAN = '#e8963a'
INK = '#1a1a19'
INK_MUTED = '#52514e'


def parse_log(log_path):
    """Return one row per iteration: the CI line closes the iteration and is
    paired with the most recent UB/LB line above it.

    A ``Zero-penalty subproblem objective mean ...`` line (printed next to the
    a=0 seed cuts, before iteration 1) becomes iteration 0; its upper bound is
    the pre-training cap (the first iteration's UB) and its lower bound is the
    zero-penalty mean itself (the a=0 bound the seed line reports).
    """
    rows = []
    zero_row = None
    last_bounds = None
    with open(log_path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            zero_match = ZERO_LINE.match(line)
            if zero_match:
                zero_row = {
                    'iteration': 0,
                    'upper_bound': None,  # filled from the first iteration's UB
                    'lower_bound': float(zero_match.group(1)),
                    'subproblem_obj_mean': float(zero_match.group(1)),
                    'ci_half_width': float(zero_match.group(2)),
                    'n_scenarios': int(zero_match.group(3)),
                }
                continue
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
    if zero_row is not None and rows:
        zero_row['upper_bound'] = rows[0]['upper_bound']
        rows.insert(0, zero_row)
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


def _burn_in_cutoff(rows, scale_factor=5.0):
    """First index from which UB and the mean stay within ``scale_factor`` times
    the final magnitudes. The cut-less early master pushes coefficients to
    astronomic values; everything before this index is off any readable scale."""
    final = rows[-1]
    limit = scale_factor * max(abs(final['upper_bound']),
                               abs(final['subproblem_obj_mean']), 1.0)
    for index, row in enumerate(rows):
        if (abs(row['upper_bound']) <= limit
                and abs(row['subproblem_obj_mean']) <= limit):
            return index
    return 0


def plot_convergence(rows, out_path, zoom_last=200):
    start = _burn_in_cutoff(rows)
    shown = rows[start:]
    iterations = [r['iteration'] for r in shown]
    ub = [r['upper_bound'] for r in shown]
    mean = [r['subproblem_obj_mean'] for r in shown]

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(10, 8))
    for ax, sl in ((ax_full, slice(None)), (ax_zoom, slice(-zoom_last, None))):
        ax.plot(iterations[sl], ub[sl], color=COLOR_UB, linewidth=2,
                label='Upper bound (UB)')
        ax.plot(iterations[sl], mean[sl], color=COLOR_MEAN, linewidth=2,
                label='Subproblem objective mean')
        _style_axis(ax)
        ax.set_xlabel('Iteration', color=INK)
        ax.set_ylabel('Objective value', color=INK)

    skipped = rows[start]['iteration'] - rows[0]['iteration'] if start else 0
    if skipped:
        print(f'Convergence figure starts at iteration {iterations[0]}: '
              f'first {skipped} burn-in iterations omitted (values off scale)')
    ax_full.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_ci_band(rows, out_path, zoom_last=200):
    iterations = [r['iteration'] for r in rows]
    half = [r['ci_half_width'] for r in rows]
    final = rows[-1]
    final_scale = abs(final['subproblem_obj_mean'])
    initial = rows[0] if rows[0]['iteration'] == 0 else None

    fig, (ax_width, ax_zoom) = plt.subplots(2, 1, figsize=(10, 8))

    # Top: the CI half-width itself. Always positive, so an ordinary log axis
    # shows the whole decay without any sign gymnastics.
    ax_width.plot(iterations, half, color=COLOR_MEAN, linewidth=2,
                  label='95% CI half-width')
    ax_width.axhline(final_scale, color=COLOR_UB, linewidth=1.5,
                     linestyle='--',
                     label='Fitted-penalty lower bound')
    if initial is not None:
        ax_width.axhline(abs(initial['subproblem_obj_mean']), color=INK_MUTED,
                         linewidth=1.5, linestyle=':',
                         label='Zero-penalty lower bound')
    ax_width.set_yscale('log')
    width_notes = [
        f"Fitted-penalty lower bound: {final['subproblem_obj_mean']:.4f} "
        f"+/- {final['ci_half_width']:.4f}"
    ]
    if initial is not None:
        width_notes.append(
            f"Zero-penalty lower bound: {initial['subproblem_obj_mean']:.4f} "
            f"+/- {initial['ci_half_width']:.4f}")
    ax_width.text(0.32, 0.6, '\n'.join(width_notes),
                  transform=ax_width.transAxes, color=INK, linespacing=1.6)
    _style_axis(ax_width)
    ax_width.set_xlabel('Iteration', color=INK)
    ax_width.set_ylabel('CI half-width', color=INK)
    ax_width.legend(frameon=False, labelcolor=INK, loc='upper center',
                    bbox_to_anchor=(0.5, -0.28), ncol=3, columnspacing=1.2,
                    handletextpad=0.6)

    # Bottom: linear zoom on the tail — the band around the mean stays wider
    # than the mean itself.
    tail = rows[-zoom_last:]
    tail_iters = [r['iteration'] for r in tail]
    tail_mean = [r['subproblem_obj_mean'] for r in tail]
    tail_low = [r['subproblem_obj_mean'] - r['ci_half_width'] for r in tail]
    tail_high = [r['subproblem_obj_mean'] + r['ci_half_width'] for r in tail]
    ax_zoom.fill_between(tail_iters, tail_low, tail_high, color=COLOR_MEAN,
                         alpha=0.25, linewidth=0, label='95% CI band')
    ax_zoom.plot(tail_iters, tail_mean, color=COLOR_MEAN, linewidth=2,
                 label='Subproblem objective mean')
    ax_zoom.axhline(0, color=INK_MUTED, linewidth=0.5, alpha=0.5)
    _style_axis(ax_zoom)
    ax_zoom.set_xlabel('Iteration', color=INK)
    ax_zoom.set_ylabel('Objective value', color=INK)
    ax_zoom.legend(frameon=False, labelcolor=INK, loc='upper center',
                   bbox_to_anchor=(0.5, -0.28), ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)


def _sci_label(value):
    """Format 1.419e+09 as mathtext ``1.419 x 10^9``."""
    mantissa, exponent = f'{value:.3e}'.split('e')
    return f'${mantissa} \\times 10^{{{int(exponent)}}}$'


def plot_variance(rows, out_path):
    iterations = [r['iteration'] for r in rows]
    variance = [r['sample_average_variance'] for r in rows]
    initial = rows[0] if rows[0]['iteration'] == 0 else None
    final = rows[-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iterations, variance, color=COLOR_MEAN, linewidth=2,
            label='Sample average variance')
    ax.set_yscale('log')
    notes = [
        f"Fitted-penalty variance: {_sci_label(final['sample_average_variance'])}"
    ]
    if initial is not None:
        notes.append(
            f"Zero-penalty variance: {_sci_label(initial['sample_average_variance'])}")
    ax.text(0.45, 0.6, '\n'.join(notes), transform=ax.transAxes, color=INK,
            linespacing=1.6)
    _style_axis(ax)
    ax.set_xlabel('Iteration', color=INK)
    ax.set_ylabel('Sample average variance', color=INK)
    ax.legend(frameon=False, labelcolor=INK, loc='upper center',
              bbox_to_anchor=(0.5, -0.22), ncol=1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight')
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
    parser.add_argument('--initial-mean', type=float, default=None,
                        help='zero-penalty (iteration 0) subproblem objective '
                             'mean, for logs predating the Zero-penalty line')
    parser.add_argument('--initial-half-width', type=float, default=None,
                        help='zero-penalty (iteration 0) 95% CI half-width')
    parser.add_argument('--initial-n', type=int, default=None,
                        help='zero-penalty scenario count (defaults to the '
                             'first iteration\'s N)')
    args = parser.parse_args()

    rows = parse_log(args.log_path)
    if not rows:
        raise SystemExit(f'No iteration statistics found in {args.log_path}')
    if rows[0]['iteration'] != 0 and args.initial_mean is not None:
        if args.initial_half_width is None:
            raise SystemExit('--initial-mean requires --initial-half-width')
        rows.insert(0, {
            'iteration': 0,
            'upper_bound': rows[0]['upper_bound'],
            'lower_bound': args.initial_mean,
            'subproblem_obj_mean': args.initial_mean,
            'ci_half_width': args.initial_half_width,
            'n_scenarios': args.initial_n or rows[0]['n_scenarios'],
        })

    # Variance of the sample average, backed out of the reported 95% half-width
    # (half = 1.96 * SE, so Var(mean) = SE^2 = (half / 1.96)^2).
    for row in rows:
        row['sample_average_variance'] = (row['ci_half_width'] / 1.96) ** 2

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.log_path))
    stem = os.path.splitext(os.path.basename(args.log_path))[0]
    csv_path = os.path.join(out_dir, f'{stem}_subproblem_ci.csv')
    fig1_path = os.path.join(out_dir, f'{stem}_convergence.svg')
    fig2_path = os.path.join(out_dir, f'{stem}_ci_band.svg')
    fig3_path = os.path.join(out_dir, f'{stem}_variance.svg')

    write_csv(rows, csv_path)
    plot_convergence(rows, fig1_path, zoom_last=args.zoom_last)
    plot_ci_band(rows, fig2_path, zoom_last=args.zoom_last)
    plot_variance(rows, fig3_path)

    final = rows[-1]
    relative = (final['ci_half_width'] / abs(final['subproblem_obj_mean'])
                if final['subproblem_obj_mean'] else float('inf'))
    print(f'Parsed {len(rows)} iterations from {args.log_path}')
    print(f'CSV:      {csv_path}')
    print(f'Figure 1: {fig1_path}')
    print(f'Figure 2: {fig2_path}')
    print(f'Figure 3: {fig3_path}')
    print(f"Final iteration {final['iteration']}: "
          f"subproblem objective mean = {final['subproblem_obj_mean']:.4f}, "
          f"95% CI half-width = {final['ci_half_width']:.4f} "
          f"(N={final['n_scenarios']}, relative half-width = {relative:.1%})")


if __name__ == '__main__':
    main()
