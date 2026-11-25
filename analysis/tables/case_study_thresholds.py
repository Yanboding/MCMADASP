"""Table tab:case_study_thresholds: percentage of treatments initiated within selected waiting-time thresholds.

Columns: PH, ALP and M for 1, 5, 10 and 15 workdays, then one column for 20 workdays, where the three policies'
entries are equal and shown once. In each type row and the Total row, the largest unrounded mean of each threshold
is bold. Every cell keeps its 95% confidence-interval half-width.

usage: python -m analysis.tables.case_study_thresholds [--results-dir DIR] [--out-dir DIR] [--refresh]

Numbers come from SimulateEvaluationResult (experiments/new_result_aggregration.py) on the results folder
(cs_alp_steady_state) and are cached in <results folder>/analysis_cache/case_study_thresholds.pkl; later runs
read only that file. --refresh rebuilds them from the records (is_reuse=False in the loader).
"""
import argparse
import os

from analysis.common import DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR, cached, check_out_dir

LABEL = 'tab:case_study_thresholds'
CACHE_FILE, CACHE_VERSION = 'case_study_thresholds.pkl', 1
TEX_FILE = 'case_study_thresholds.tex'
POLICIES = (('approx_penalized_hindsight', 'PH'), ('row_gen_alp', 'ALP'), ('myopic', 'M'))
DAYS = (1, 5, 10, 15, 20)
SHOWN_DAYS = (1, 5, 10, 15)  # one column per policy
MERGED_DAY = 20  # one entry for the three policies, whose entries are equal as displayed (checked in latex())
CONFIDENCE = 0.95


def build_numbers(results_dir, refresh=False):
    """Mean and half-width of every cell, read from the loader's statistics."""
    from experiments import get_config_by_type
    from experiments.new_result_aggregration import SimulateEvaluationResult
    ser = SimulateEvaluationResult(results_dir, '[0-9]*.jsonl', get_config_by_type('ejor').env, is_reuse=not refresh)
    by_type, by_day = ser.waiting_time_target_ptc_by_type_day, ser.waiting_time_target_ptc_by_day

    def pair(stats):
        return (float(stats.mean), float(stats.half_window(CONFIDENCE)))

    n_types = max(len(by_type[pid]) for pid, _ in POLICIES)
    return {'days': list(DAYS), 'policies': [name for _, name in POLICIES], 'confidence': CONFIDENCE,
            'n_paths': int(by_day[POLICIES[0][0]][DAYS[0] - 1].n),
            'types': {t + 1: {name: {d: pair(by_type[pid][t][d - 1]) for d in DAYS} for pid, name in POLICIES}
                      for t in range(n_types)},
            'total': {name: {d: pair(by_day[pid][d - 1]) for d in DAYS} for pid, name in POLICIES}}


def load_numbers(results_dir=DEFAULT_RESULTS_DIR, refresh=False):
    return cached(results_dir, CACHE_FILE, CACHE_VERSION, lambda: build_numbers(results_dir, refresh), refresh)


def cell(pair, bold=False):
    body = f'{pair[0]:.0f}{{\\pm}}{pair[1]:.1f}'
    return f'\\(\\mathbf{{{body}}}\\)' if bold else f'\\({body}\\)'


def check_merged_day(numbers):
    """The 20-workday column shows one entry for the three policies, so their entries must be equal as displayed."""
    rows = list(numbers['types'].values()) + [numbers['total']]
    for row in rows:
        shown = {(f'{row[name][MERGED_DAY][0]:.0f}', f'{row[name][MERGED_DAY][1]:.1f}') for name in numbers['policies']}
        if len(shown) > 1:
            raise ValueError(f'the {MERGED_DAY}-workday entries of the three policies differ ({sorted(shown)}), '
                             'so they cannot be shown once')


def threshold_cells(row, names, d):
    """Cells of one threshold; the policies with the largest unrounded mean are bold, unless all three are equal."""
    means = [row[name][d][0] for name in names]
    top, all_equal = max(means), len(set(means)) == 1
    return [cell(row[name][d], bold=not all_equal and row[name][d][0] == top) for name in names]


def row_cells(row, names):
    cells = [c for d in SHOWN_DAYS for c in threshold_cells(row, names, d)]
    return cells + [cell(row[names[0]][MERGED_DAY])]


def latex(numbers):
    check_merged_day(numbers)
    names = numbers['policies']
    k = len(names)
    header = '\n'.join(f'& \\multicolumn{{{k}}}{{c}}{{{d} workday{"s" if d > 1 else ""}}}' for d in SHOWN_DAYS)
    header += f'\n& {MERGED_DAY} workdays'
    rules = '\n'.join(f'\\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}' for i in range(len(SHOWN_DAYS)))
    last = 2 + len(SHOWN_DAYS) * k
    rules += f'\n\\cmidrule(lr){{{last}-{last}}}'
    sub = ' '.join(f'& {name}' for _ in SHOWN_DAYS for name in names) + ' & ' + ', '.join(names)
    rows = '\n'.join(f'{t} & ' + ' & '.join(row_cells(numbers['types'][t], names)) + ' \\\\'
                     for t in sorted(numbers['types']))
    total = 'Total\n' + '\n'.join('& ' + ' & '.join(threshold_cells(numbers['total'], names, d)) for d in SHOWN_DAYS)
    total += f"\n& {cell(numbers['total'][names[0]][MERGED_DAY])} \\\\"
    level = round(numbers['confidence'] * 100)
    return f"""\\begin{{table}}[!htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Percentage of treatments initiated within selected waiting-time thresholds.}}
\\label{{{LABEL}}}
\\fontsize{{7.5}}{{8.6}}\\selectfont
\\setlength{{\\tabcolsep}}{{0pt}}
\\renewcommand{{\\arraystretch}}{{1.02}}
\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}l*{{{k * len(SHOWN_DAYS) + 1}}}{{c}}@{{}}}}
\\toprule
Type
{header} \\\\
{rules}
{sub} \\\\
\\midrule
{rows}
\\midrule
{total}
\\bottomrule
\\end{{tabular*}}
\\begin{{tablenotes}}[flushleft]
\\fontsize{{7.5}}{{8.6}}\\selectfont
\\item \\textit{{Note.}} Entries are percentages reported as sample means \\(\\pm\\) {level}\\% confidence-interval half-widths over {numbers['n_paths']:,} evaluation sample paths. For the {MERGED_DAY}-workday threshold, the entries of the three policies are equal and are shown once. Bold entries are the largest among the three policies for the same type and threshold, compared before rounding. PH, ALP, and M denote the ALP-based penalized hindsight policy, the ALP policy, and the myopic policy, respectively. Treatment inputs appear in Table~\\ref{{tab:case_study_inputs}}.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--refresh', action='store_true', help='rebuild the numbers from the records')
    args = parser.parse_args(argv)
    try:
        out_dir = check_out_dir(args.out_dir)
    except ValueError as error:
        parser.error(str(error))
    numbers, source = load_numbers(args.results_dir, args.refresh)
    text = latex(numbers)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, TEX_FILE)
    with open(path, 'w') as handle:
        handle.write(text + '\n')
    print(text)
    print('cache:', 'used' if source == 'cache' else 'built', os.path.join(args.results_dir, 'analysis_cache', CACHE_FILE))
    print('saved', path)


if __name__ == '__main__':
    main()
