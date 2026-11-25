"""Figure fig:cs_matched_states: waiting times of the treatments booked by the ALP policy and the
ALP-based penalized hindsight policy at the same first-decision states.

usage: python -m analysis.figures.cs_matched_states [--results-dir DIR] [--out-dir DIR] [--refresh]

The bars are computed in memory from the simulation records in the results folder (cs_alp_steady_state):
for every uid, the state after the warm-up (warmup_state of the ALP record) and the first decision after the
warm-up (scheduled_patients[warm_up_periods]) of the ALP policy and of the ALP-based penalized hindsight policy.
The first run saves these arrays to <results folder>/analysis_cache/cs_matched_states.pkl; later runs read that
cache, and --refresh rebuilds it. No other file is written in the results tree.
Types, waiting-time targets, numbers of sessions and capacities come from get_config_by_type('ejor').env.
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

from analysis.common import CACHE_DIR, DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR, REPO_ROOT, cached, check_out_dir

FILE_NAME = 'cs_first_decision_waiting_time_by_type_sessions.svg'

POLICIES = (('row_gen_alp', 'ALP'), ('approx_penalized_hindsight', 'PH'))
CACHE_FILE, CACHE_VERSION = 'cs_matched_states.pkl', 1
GROUPS = ('low', 'mid', 'high')
LOWER_PERCENTILE, UPPER_PERCENTILE = 33.3, 66.7

# shade level of a type from its number of sessions
FEW_SESSIONS, MEDIUM_SESSIONS = 5, 20

# class of a type from its waiting-time target in workdays (target index + 1)
CLASS_OF_WORKDAYS = {1: 'urgent', 5: 'five', 10: 'ten'}
ROW_TITLES = {'urgent': '(a) Urgent\ntreatments (%)', 'five': '(b) Five-day target\ntreatments (%)',
              'ten': '(c) Ten-day target\ntreatments (%)'}
# waiting-time bins (label, first day, last day + 1); the five-day row splits off the last day before the target
BINS4 = [('[0, 1)', 0, 1), ('[1, 5)', 1, 5), ('[5, 10)', 5, 10), ('[10, 20)', 10, 20)]
BINS5 = [('[0, 1)', 0, 1), ('[1, 4)', 1, 4), ('[4, 5)', 4, 5), ('[5, 10)', 5, 10), ('[10, 20)', 10, 20)]
BINS = {'urgent': BINS4, 'five': BINS5, 'ten': BINS4}

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
RAMP = {'ALP': ['#86b6ef', '#2a78d6', '#104281'], 'PH': ['#f0a084', '#eb6834', '#9c3a0e']}


def level_of(sessions):
    return 0 if sessions <= FEW_SESSIONS else 1 if sessions <= MEDIUM_SESSIONS else 2


def type_list_label(types):
    types = list(types)
    if len(types) == 1:
        return f'Type {types[0]}'
    if all(b - a == 1 for a, b in zip(types, types[1:])):
        return f'Types {types[0]}–{types[-1]}'
    return 'Types ' + ', '.join(str(t) for t in types)


def cell_label(types, sessions):
    lo, hi = min(sessions), max(sessions)
    count = f'{lo}' if lo == hi else f'{lo}–{hi}'
    return f'{type_list_label(types)} ({count} sessions)'


def type_spec(env):
    """Classes, shade levels and row legend labels of the case-study types (types numbered from 1)."""
    pattern = np.asarray(env.treatment_pattern)
    sessions = {i + 1: int((pattern[:, i] > 0).sum()) for i in range(env.num_types)}
    workdays = {i + 1: int(env.holding_cost.get_waiting_target(i)) + 1 for i in range(env.num_types)}
    class_of = {t: CLASS_OF_WORKDAYS[w] for t, w in workdays.items()}  # KeyError for a target other than 1, 5, 10
    classes = {c: tuple(t for t in sessions if class_of[t] == c)
               for c in CLASS_OF_WORKDAYS.values() if c in class_of.values()}
    level = {t: level_of(s) for t, s in sessions.items()}
    labels = {}
    for c, types in classes.items():
        labels[c] = {}
        for lv in sorted(set(level[t] for t in types)):
            cell = [t for t in types if level[t] == lv]
            labels[c][lv] = cell_label(cell, [sessions[t] for t in cell])
    class_workdays = {c: workdays[types[0]] for c, types in classes.items()}
    return {'classes': classes, 'level': level, 'labels': labels, 'workdays': class_workdays}


def _read_records(path):
    """(policy, uid, warmup_state, first decision after warm-up) of the ALP and PH records of one jsonl file."""
    wanted = {policy for policy, _ in POLICIES}
    out = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record['policy_id'] in wanted:
                w = record['warm_up_periods']
                out.append((record['policy_id'], record['uid'], record['warmup_state'], record['scheduled_patients'][w]))
    return out


def read_results(results_dir, processes=6):
    """Read the [0-9]*.jsonl records; keep one record per (policy, uid), the first in sorted file order."""
    paths = sorted(glob.glob(os.path.join(results_dir, '[0-9]*.jsonl')))
    if processes > 1:
        with Pool(processes) as pool:
            chunks = pool.map(_read_records, paths)
    else:
        chunks = [_read_records(p) for p in paths]
    kept = {policy: {} for policy, _ in POLICIES}
    for chunk in chunks:
        for policy, uid, state, decision in chunk:
            kept[policy].setdefault(uid, (state, decision))
    alp, ph = (kept[policy] for policy, _ in POLICIES)
    if set(alp) != set(ph):
        raise ValueError(f'uids without both policies: {sorted(set(alp) ^ set(ph))}')
    uids = sorted(alp)
    for uid in uids:
        if not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(alp[uid][0], ph[uid][0])):
            raise ValueError(f'warmup_state of uid {uid} differs between the two policies')
    return {'uids': uids,
            'u': np.array([alp[uid][0][0] for uid in uids]),
            'v': np.array([alp[uid][0][1] for uid in uids]),
            'w': np.array([alp[uid][0][2] for uid in uids]),
            'decisions': {name: np.array([kept[policy][uid][1] for uid in uids]) for policy, name in POLICIES}}


def load_data(results_dir=DEFAULT_RESULTS_DIR, refresh=False, processes=6):
    """Return (data, 'cache' or 'built'). The cache is the only file written inside the results folder."""
    return cached(results_dir, CACHE_FILE, CACHE_VERSION, lambda: read_results(results_dir, processes), refresh)


def first_decision_table(data, n_days, n_types, capacity):
    """Group the first-decision states by the share of the booking window booked under ALP and average the
    bookings per type.

    Returns (cutoffs in percent, uids per group, table) where table[(group, policy, type)] holds
    n_states, the mean number booked on each of the n_days booking days (d) and the mean left on the waitlist.
    """
    uids = list(data['uids'])
    regular = np.asarray(data['u'], float)[:, :n_days].sum(1)
    overtime = np.asarray(data['v'], float)[:, :n_days].sum(1)
    booked = regular + overtime
    share = booked / capacity
    cutoffs = (round(np.percentile(share, LOWER_PERCENTILE) * 100), round(np.percentile(share, UPPER_PERCENTILE) * 100))
    label = np.where(booked < cutoffs[0] * capacity / 100, 0, np.where(booked < cutoffs[1] * capacity / 100, 1, 2))
    groups = {GROUPS[k]: [u for i, u in enumerate(uids) if label[i] == k] for k in range(3)}
    waitlist = np.asarray(data['w'], float)[:, :n_types]

    table = {}
    for k, g in enumerate(GROUPS):
        members = np.flatnonzero(label == k)
        for _, name in POLICIES:
            bookings = np.asarray(data['decisions'][name], float)[members]
            for i in range(n_types):
                rows = np.column_stack([bookings[:, :, i], waitlist[members, i] - bookings[:, :, i].sum(1)])
                mean = np.mean(rows, axis=0)
                table[(g, name, i + 1)] = {'n_states': len(members), 'd': mean[:-1], 'left': float(mean[-1])}
    return cutoffs, groups, table


def shares(table, group, policy, types, bins, level):
    """Percentage of the waiting treatments of the given types booked in each bin, split by shade level."""
    total = sum(table[(group, policy, t)]['d'].sum() for t in types) + sum(table[(group, policy, t)]['left'] for t in types)
    arr = np.zeros((len(bins), 3))
    for b, (_, lo, hi) in enumerate(bins):
        for t in types:
            arr[b, level[t]] += table[(group, policy, t)]['d'][lo:hi].sum()
    return 100 * arr / total


def dashed_after(bins, workdays):
    """Number of bins that lie within the waiting-time target."""
    return sum(1 for _, _, hi in bins if hi <= workdays)


def bar_totals(table, spec):
    lines = []
    for c, types in spec['classes'].items():
        cells = []
        for g in GROUPS:
            tot = {p: shares(table, g, p, types, BINS[c], spec['level']).sum(1) for _, p in POLICIES}
            cells.append(' '.join(f'{a:.0f} {b:.0f}' for a, b in zip(tot['ALP'], tot['PH'])))
        lines.append(f'{c} | ' + ' | '.join(cells))
    return lines


def font_name():
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return 'Times New Roman' if 'Times New Roman' in have else 'Liberation Serif'


def plot_figure(table, spec, cutoffs, out_path, dpi=300):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.patches import Patch

    rc = {'font.family': [font_name()], 'mathtext.fontset': 'stix', 'svg.fonttype': 'path', 'svg.hashsalt': 'cs_matched_states',
          'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
          'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 9,
          'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6}
    titles = (f'Less than {cutoffs[0]}% booked', f'{cutoffs[0]}% to {cutoffs[1]}% booked', f'At least {cutoffs[1]}% booked')
    rows = list(spec['classes'])
    hm = {tuple: HandlerTuple(ndivide=None, pad=0.0)}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(3, 3, figsize=(6.5, 7.25), sharey=True)
        w = 0.38
        for ci, (g, title) in enumerate(zip(GROUPS, titles)):
            for ri, c in enumerate(rows):
                ax = axes[ri, ci]
                bins = BINS[c]
                x = np.arange(len(bins))
                for p, off in (('ALP', -w / 2 - 0.02), ('PH', w / 2 + 0.02)):
                    arr = shares(table, g, p, spec['classes'][c], bins, spec['level'])
                    bottom = np.zeros(len(bins))
                    for lv in range(3):
                        ax.bar(x + off, arr[:, lv], w, bottom=bottom, color=RAMP[p][lv], edgecolor='white', linewidth=0.5)
                        bottom += arr[:, lv]
                    for xi, tot in zip(x, bottom):
                        ax.text(xi + off, tot + 1.5, f'{tot:.0f}', ha='center', va='bottom', fontsize=6.6, color=INK)
                ax.set_xticks(x)
                ax.set_xticklabels([b[0] for b in bins])
                if len(bins) == 5:
                    ax.tick_params(axis='x', labelsize=7.6)
                ax.set_ylim(0, 112)
                ax.set_yticks([0, 25, 50, 75, 100])
                ax.yaxis.grid(True, color=GRID, linewidth=0.6)
                ax.set_axisbelow(True)
                for s in ('top', 'right'):
                    ax.spines[s].set_visible(False)
                for s in ('left', 'bottom'):
                    ax.spines[s].set_color(MUTED)
                ax.tick_params(colors=INK, length=2.5)
                ax.axvline(dashed_after(bins, spec['workdays'][c]) - 0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
                ax.set_xlabel('Waiting time (days)')
                if ri == 0:
                    ax.set_title(title, pad=23)
                if ci == 0:
                    ax.set_ylabel(ROW_TITLES[c])
        pol = [tuple(Patch(facecolor=col, edgecolor='white', linewidth=0.5) for col in RAMP[p]) for p in ('ALP', 'PH')]
        fig.legend(pol, ['ALP policy', 'ALP-based penalized hindsight policy'], loc='upper center', ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 1.0), handler_map=hm, handlelength=2.6, columnspacing=2.0)
        fig.tight_layout(rect=(0, 0, 1, 0.965), w_pad=0.8, h_pad=2.9)
        for ri, c in enumerate(rows):
            lab = spec['labels'][c]
            hs = [tuple(Patch(facecolor=RAMP[p][lv], edgecolor='white', linewidth=0.5) for p in ('ALP', 'PH')) for lv in lab]
            p0, p2 = axes[ri, 0].get_position(), axes[ri, 2].get_position()
            fig.legend(hs, list(lab.values()), loc='lower center', bbox_to_anchor=((p0.x0 + p2.x1) / 2, p0.y1 + 0.005),
                       ncol=len(lab), handler_map=hm, handlelength=1.5, fontsize=7.8, columnspacing=1.6,
                       handletextpad=0.45, frameon=False, borderaxespad=0.0, borderpad=0.15)
        metadata = {'Date': None} if out_path.endswith('.svg') else None
        fig.savefig(out_path, dpi=dpi, facecolor='white', metadata=metadata)
        plt.close(fig)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--refresh', action='store_true', help='rebuild the cache from the jsonl records')
    args = parser.parse_args(argv)
    try:
        out_dir = check_out_dir(args.out_dir)
    except ValueError as error:
        parser.error(str(error))

    from experiments import get_config_by_type
    env = get_config_by_type('ejor').env
    data, source = load_data(args.results_dir, refresh=args.refresh)
    capacity = env.booking_window_size * (env.regular_capacity + env.overtime_capacity)
    cutoffs, _, table = first_decision_table(data, env.booking_window_size, env.num_types, capacity)
    spec = type_spec(env)
    os.makedirs(out_dir, exist_ok=True)
    path = plot_figure(table, spec, cutoffs, os.path.join(out_dir, FILE_NAME))
    print('font:', font_name())
    print('cache:', 'used' if source == 'cache' else 'built',
          os.path.join(args.results_dir, CACHE_DIR, CACHE_FILE))
    for line in bar_totals(table, spec):
        print(line)
    print('saved', path)


if __name__ == '__main__':
    main()
