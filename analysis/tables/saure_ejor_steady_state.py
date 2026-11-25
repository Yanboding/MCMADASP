"""Table tab:saure_ejor_steady_state: estimated case-study policy performance starting from the steady state under
the ALP policy.

usage: python -m analysis.tables.saure_ejor_steady_state [--results-dir DIR] [--out-dir DIR] [--refresh]

Numbers come from experiments/new_result_aggregration.py on the results folder (cs_alp_steady_state) and on the
partial-information-relaxation folder next to it (<results folder>_partial_ir). They are cached in
<results folder>/analysis_cache/saure_ejor_steady_state.pkl; later runs read only that file.
--refresh rebuilds them from the records (is_reuse=False in the loader).
"""
import argparse
import os

from analysis.common import DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR, cached, check_out_dir

LABEL = 'tab:saure_ejor_steady_state'
CACHE_FILE, CACHE_VERSION = 'saure_ejor_steady_state.pkl', 1
TEX_FILE = 'saure_ejor_steady_state.tex'
POLICIES = (('approx_penalized_hindsight', 'ALP-based PH'), ('row_gen_alp', 'ALP'), ('myopic', 'Myopic'))
BASELINE = 'myopic'
CONFIDENCE = 0.95
# (key, column header, decimals of the mean, larger is better)
COLUMNS = (('cost', 'Discounted\\\\cost', 0, False),
           ('improvement', 'Improvement\\\\over Myopic (\\%)', 1, True),
           ('violation', 'Wait-time\\\\violations', 0, False),
           ('overtime', 'Overtime\\\\utilization', 0, False),
           ('zero_gap', '\\% gap to\\\\zero-penalty LB', 1, False),
           ('penalized_gap', '\\% gap to\\\\penalized LB', 1, False),
           ('partial_gap', '\\% gap to partial\\\\relaxation LB', 1, False))


def partial_dir(results_dir):
    return os.path.normpath(results_dir) + '_partial_ir'


def build_numbers(results_dir, refresh=False):
    """Mean and half-width of every cell, from the loader and the helpers the old table function uses."""
    from experiments import get_config_by_type
    from experiments.new_result_aggregration import (SimulateEvaluationResult, partial_relaxation_costs,
                                                     partial_relaxation_gaps)
    ser = SimulateEvaluationResult(results_dir, '[0-9]*.jsonl', get_config_by_type('ejor').env, is_reuse=not refresh)
    groups = sorted({gid for gid, _ in ser.policy_costs}, key=str)
    if len(groups) != 1:
        raise ValueError(f'expected one group in {results_dir}, found {len(groups)}')
    gid = groups[0]
    partial = partial_relaxation_costs(partial_dir(results_dir))

    def pair(stats):
        return (float(stats.mean), float(stats.half_window(CONFIDENCE)))

    cells = {}
    for pid, _ in POLICIES:
        key = (gid, pid)
        improvement = None
        if pid != BASELINE:
            mean, half, _ = ser.improvement_over_baseline(gid, pid, BASELINE, CONFIDENCE, cost_by_uid=ser.cost_by_uid)
            improvement = (float(mean), float(half))
        gaps = partial_relaxation_gaps(ser, partial, gid, pid, CONFIDENCE)
        cells[pid] = {'cost': pair(ser.policy_costs[key]), 'improvement': improvement,
                      'violation': pair(ser.waiting_time_violation[key]),
                      'overtime': pair(ser.overtime_utilization[key]),
                      'zero_gap': pair(ser.zero_penalized_improvement[key]),
                      'penalized_gap': pair(ser.penalized_improvement[key]),
                      'partial_gap': (float(gaps['gap_pct']), float(gaps['gap_half_window_pct']))}
    n_paths = max(ser.policy_costs[(gid, pid)].n for pid, _ in POLICIES)
    return {'n_paths': int(n_paths), 'confidence': CONFIDENCE, 'cells': cells}


def load_numbers(results_dir=DEFAULT_RESULTS_DIR, refresh=False):
    return cached(results_dir, CACHE_FILE, CACHE_VERSION, lambda: build_numbers(results_dir, refresh), refresh)


def number(value, decimals):
    return f'{value:,.{decimals}f}'.replace(',', '{,}')


def cell(pair, decimals, bold=False):
    body = f'{number(pair[0], decimals)} \\pm {number(pair[1], 1)}'
    return f'\\(\\mathbf{{{body}}}\\)' if bold else f'\\({body}\\)'


def best(cells, key, larger_is_better):
    present = {pid: c[key][0] for pid, c in cells.items() if c[key] is not None}
    return (max if larger_is_better else min)(present, key=present.get)


def latex(numbers):
    cells = numbers['cells']
    bold = {key: best(cells, key, larger) for key, _, _, larger in COLUMNS}
    header = 'Policy\n' + '\n'.join(f'& \\makecell{{{title}}}' for _, title, _, _ in COLUMNS) + '\\\\'
    rows = []
    for pid, name in POLICIES:
        entries = ['---' if cells[pid][key] is None else cell(cells[pid][key], decimals, bold=bold[key] == pid)
                   for key, _, decimals, _ in COLUMNS]
        rows.append(f'{name} & ' + ' & '.join(entries) + ' \\\\')
    body = '\n'.join(rows)
    return f"""\\begin{{table}}[!htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Estimated case-study policy performance starting from the steady state under the ALP policy with $\\gamma=0.99$.}}
\\label{{{LABEL}}}

\\scriptsize
\\setlength{{\\tabcolsep}}{{2.4pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}

\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} l{'c' * len(COLUMNS)} @{{}}}}
\\toprule
{header}
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}

\\begin{{tablenotes}}[flushleft]
\\footnotesize
\\item \\textit{{Note.}} Bold entries are the best in each column.
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
