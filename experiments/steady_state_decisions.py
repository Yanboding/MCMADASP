import glob
import json
import os
import pickle

import numpy as np

from experiments import get_config_by_type
from param_generation.command_files import read_command_file_records

POLICIES = (('approx_penalized_hindsight', 'Penalized hindsight', '#2a78d6'),
            ('row_gen_alp', 'ALP', '#eb6834'),
            ('myopic', 'Myopic', '#1baf7a'))
PATIENT_GROUPS = {'urgent': [0, 1, 2], 'target5': list(range(6, 14)), 'target10': [3, 4, 5, 14, 15, 16, 17]}
GROUP_LABELS = {'urgent': 'Urgent\n(target 1 day)', 'target5': 'Target 5 days', 'target10': 'Target 10 days\n(long courses)'}
NEAR_TERM_DAYS = 5
LOAD_BINS = ((0.0, 0.8), (0.8, 0.9), (0.9, 0.97), (0.97, 1.01))
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'


def replay_steady_state_periods(source_dat, results_dir, cache_file=None):
    if cache_file is not None and os.path.exists(cache_file):
        with open(cache_file, 'rb') as handle:
            return pickle.load(handle)
    source = {record['uid']: record for record in read_command_file_records(source_dat)}
    env = get_config_by_type('infinite_custom', args=next(iter(source.values()))['env_args']).env
    late = env.waiting_cost_coeffs > 0
    days = np.arange(env.booking_window_size)[:, None]
    policy_index = {policy_id: index for index, (policy_id, _, _) in enumerate(POLICIES)}
    columns = ['policy', 'weight', 'cost', 'waiting', 'overtime_cost', 'regular_near', 'overtime_booked_near',
               'new_overtime', 'new_overtime_near']
    for group in PATIENT_GROUPS:
        columns += [f'{group}_waitlist', f'{group}_booked', f'{group}_lead', f'{group}_late']
    rows = []
    max_cost_error = 0.0
    for path in sorted(glob.glob(os.path.join(results_dir, '[0-9]*.jsonl'))):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                src = source[record['uid']]
                w = record['warm_up_periods']
                weights = np.asarray(src['period_weights'], dtype=float)
                arrivals = np.asarray(src['sample_path'][w:], dtype=int).reshape(-1, env.num_types)
                state = tuple(np.asarray(component, dtype=float) for component in record['warmup_state'])
                for t, weight in enumerate(weights):
                    regular, overtime, waitlist = state
                    bookings = np.asarray(record['scheduled_patients'][w + t], dtype=float)
                    new_overtime = np.maximum(np.round(
                        regular + env.convert_action_to_booking_slots(bookings) - env.regular_capacity), 0.0)
                    cost = float(env.cost_fn(state, (bookings, new_overtime)))
                    max_cost_error = max(max_cost_error, abs(cost - record['costs'][w + t]))
                    row = [policy_index[record['policy_id']], weight, cost, float((env.waiting_cost_coeffs * bookings).sum()),
                           float(env.overtime_cost_coeffs @ new_overtime), regular[:NEAR_TERM_DAYS].sum(),
                           overtime[:NEAR_TERM_DAYS].sum(), new_overtime.sum(), new_overtime[:NEAR_TERM_DAYS].sum()]
                    for types in PATIENT_GROUPS.values():
                        booked = bookings[:, types]
                        row += [waitlist[types].sum(), booked.sum(), float((days * booked).sum()),
                                float((late[:, types] * booked).sum())]
                    rows.append(row)
                    arrival = arrivals[t] if t < len(arrivals) else np.zeros(env.num_types)
                    state = env.post_action_state_to_new_state(
                        env.post_action_state(state, (bookings, new_overtime)), arrival, is_var=False)
    if max_cost_error > 1e-6:
        raise ValueError(f'replayed costs differ from the recorded costs by up to {max_cost_error}')
    replay = {'columns': columns, 'rows': np.asarray(rows, dtype=float),
              'near_term_capacity': NEAR_TERM_DAYS * env.regular_capacity}
    if cache_file is not None:
        with open(cache_file, 'wb') as handle:
            pickle.dump(replay, handle)
    return replay


def replay_booking_profiles(source_dat, results_dir, cache_file=None):
    if cache_file is not None and os.path.exists(cache_file):
        with open(cache_file, 'rb') as handle:
            return pickle.load(handle)
    source = {record['uid']: record for record in read_command_file_records(source_dat)}
    env = get_config_by_type('infinite_custom', args=next(iter(source.values()))['env_args']).env
    policy_index = {policy_id: index for index, (policy_id, _, _) in enumerate(POLICIES)}
    features, histograms = [], []
    for path in sorted(glob.glob(os.path.join(results_dir, '[0-9]*.jsonl'))):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                src = source[record['uid']]
                w = record['warm_up_periods']
                weights = np.asarray(src['period_weights'], dtype=float)
                arrivals = np.asarray(src['sample_path'][w:], dtype=int).reshape(-1, env.num_types)
                state = tuple(np.asarray(component, dtype=float) for component in record['warmup_state'])
                for t, weight in enumerate(weights):
                    regular, overtime, waitlist = state
                    bookings = np.asarray(record['scheduled_patients'][w + t], dtype=float)
                    new_overtime = np.maximum(np.round(
                        regular + env.convert_action_to_booking_slots(bookings) - env.regular_capacity), 0.0)
                    features.append([policy_index[record['policy_id']], weight, regular[:4].sum(), regular[4:10].sum(),
                                     regular[10:20].sum(), overtime[:4].sum(), regular[0], overtime[0],
                                     new_overtime[:4].sum(), new_overtime[4:].sum(),
                                     *[waitlist[types].sum() for types in PATIENT_GROUPS.values()]])
                    histograms.append(np.stack([bookings[:, types].sum(axis=1) for types in PATIENT_GROUPS.values()]))
                    arrival = arrivals[t] if t < len(arrivals) else np.zeros(env.num_types)
                    state = env.post_action_state_to_new_state(
                        env.post_action_state(state, (bookings, new_overtime)), arrival, is_var=False)
    profiles = {
        'columns': ['policy', 'weight', 'regular_days_1_4', 'regular_days_5_10', 'regular_days_11_20',
                    'overtime_days_1_4', 'regular_today', 'overtime_today', 'new_overtime_days_1_4',
                    'new_overtime_later', *[f'{group}_waitlist' for group in PATIENT_GROUPS]],
        'features': np.asarray(features, dtype=np.float32),
        'histograms': np.asarray(histograms, dtype=np.int16),
        'regular_capacity': env.regular_capacity,
        'overtime_capacity': env.overtime_capacity,
    }
    if cache_file is not None:
        with open(cache_file, 'wb') as handle:
            pickle.dump(profiles, handle)
    return profiles


def _weighted(replay, mask, numerator, denominator=None):
    column = {name: index for index, name in enumerate(replay['columns'])}
    rows = replay['rows'][mask]
    weights = rows[:, column['weight']]
    total = (rows[:, column[numerator]] * weights).sum()
    base = weights.sum() if denominator is None else (rows[:, column[denominator]] * weights).sum()
    return float(total / base) if base > 0 else float('nan')


def _load(replay):
    column = replay['columns'].index('regular_near')
    return replay['rows'][:, column] / replay['near_term_capacity']


def full_load_summary(replay, threshold=0.97):
    policy = replay['rows'][:, 0]
    load = _load(replay)
    summary = {}
    for index, (policy_id, _, _) in enumerate(POLICIES):
        mask = (policy == index) & (load >= threshold)
        everything = policy == index
        row = {'share_of_periods': float(replay['rows'][mask, 1].sum() / replay['rows'][everything, 1].sum())}
        for name in ('cost', 'waiting', 'overtime_cost', 'new_overtime', 'new_overtime_near', 'overtime_booked_near'):
            row[name] = _weighted(replay, mask, name)
        row['new_overtime_far'] = row['new_overtime'] - row['new_overtime_near']
        for group in PATIENT_GROUPS:
            row[f'{group}_late_pct'] = 100 * _weighted(replay, mask, f'{group}_late', f'{group}_booked')
            row[f'{group}_lead'] = _weighted(replay, mask, f'{group}_lead', f'{group}_booked')
        summary[policy_id] = row
    return summary


def load_profile(replay):
    policy = replay['rows'][:, 0]
    load = _load(replay)
    profile = {}
    for index, (policy_id, _, _) in enumerate(POLICIES):
        everything = policy == index
        rows = []
        for low, high in LOAD_BINS:
            mask = everything & (load >= low) & (load < high)
            rows.append({'share_of_periods': float(replay['rows'][mask, 1].sum() / replay['rows'][everything, 1].sum()),
                         'cost': _weighted(replay, mask, 'cost'),
                         'urgent_late_pct': 100 * _weighted(replay, mask, 'urgent_late', 'urgent_booked')})
        profile[policy_id] = rows
    return profile


def _style(ax):
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(GRID)


def _grouped_bars(ax, categories, values_by_policy, value_format):
    width = 0.8 / len(POLICIES)
    x = np.arange(len(categories))
    for index, (policy_id, label, color) in enumerate(POLICIES):
        values = values_by_policy[policy_id]
        bars = ax.bar(x + (index - (len(POLICIES) - 1) / 2) * width, values, width * 0.92, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.annotate(value_format.format(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=7.5,
                        color=TEXT_SECONDARY)
    ax.set_xticks(x, categories)
    _style(ax)


def _save(figure, save_file):
    os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
    figure.savefig(save_file, bbox_inches='tight')
    figure.savefig(os.path.splitext(save_file)[0] + '.png', dpi=200, bbox_inches='tight')
    print(f'Saved {save_file}')


def plot_full_load_decisions(summary, save_file):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    _grouped_bars(axes[0], ['Waiting', 'Overtime'],
                  {p: [summary[p]['waiting'], summary[p]['overtime_cost']] for p, _, _ in POLICIES}, '{:,.0f}')
    axes[0].set_title('Cost per period by component', fontsize=10, color=TEXT_PRIMARY)
    _grouped_bars(axes[1], [f'Days 1-{NEAR_TERM_DAYS}', f'Day {NEAR_TERM_DAYS + 1} onward'],
                  {p: [summary[p]['new_overtime_near'], summary[p]['new_overtime_far']] for p, _, _ in POLICIES},
                  '{:.1f}')
    axes[1].set_title('New overtime slots per period, by service day', fontsize=10, color=TEXT_PRIMARY)
    _grouped_bars(axes[2], [GROUP_LABELS[group] for group in PATIENT_GROUPS],
                  {p: [summary[p][f'{group}_late_pct'] for group in PATIENT_GROUPS] for p, _, _ in POLICIES}, '{:.0f}%')
    axes[2].set_title('Patients booked beyond their waiting target', fontsize=10, color=TEXT_PRIMARY)
    axes[2].set_ylabel('% of booked patients', fontsize=9, color=TEXT_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.08))
    figure.tight_layout()
    _save(figure, save_file)
    plt.close(figure)


def plot_load_profile(profile, save_file):
    import matplotlib.pyplot as plt
    labels = [f'< {LOAD_BINS[0][1]:.0%}'] + [f'{low:.0%}-{high:.0%}' for low, high in LOAD_BINS[1:-1]] + [f'≥ {LOAD_BINS[-1][0]:.0%}']
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _grouped_bars(axes[0], labels, {p: [row['share_of_periods'] * 100 for row in profile[p]] for p, _, _ in POLICIES}, '{:.0f}%')
    axes[0].set_title('Share of periods spent in each load range', fontsize=10, color=TEXT_PRIMARY)
    _grouped_bars(axes[1], labels, {p: [row['cost'] for row in profile[p]] for p, _, _ in POLICIES}, '{:,.0f}')
    axes[1].set_title('Cost per period', fontsize=10, color=TEXT_PRIMARY)
    for ax in axes:
        ax.set_xlabel(f'Regular-capacity load over the next {NEAR_TERM_DAYS} days', fontsize=9, color=TEXT_SECONDARY)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    figure.tight_layout()
    _save(figure, save_file)
    plt.close(figure)


def full_load_table(summary):
    lines = []
    for policy_id, label, _ in POLICIES:
        row = summary[policy_id]
        lines.append(
            f"{label} & {100 * row['share_of_periods']:.0f} & {row['cost']:,.0f} & {row['waiting']:,.0f} & "
            f"{row['overtime_cost']:,.0f} & {row['new_overtime_near']:.1f} & {row['new_overtime_far']:.1f} & "
            f"{row['urgent_late_pct']:.1f} & {row['target5_late_pct']:.1f} & {row['target10_late_pct']:.1f} \\\\")
    body = '\n'.join(lines).replace(',', '{,}')
    return f"""\\begin{{table}}[!htbp]
\\centering
\\caption{{Scheduling decisions in fully loaded states of the case study (regular-capacity load of at least 97\\% over the next {NEAR_TERM_DAYS} days), starting from the ALP steady state.}}
\\label{{tab:steady_state_full_load_decisions}}
\\small
\\begin{{tabular}}{{lccccccccc}}
\\toprule
& & \\multicolumn{{3}}{{c}}{{Cost per period}} & \\multicolumn{{2}}{{c}}{{New overtime slots}} & \\multicolumn{{3}}{{c}}{{\\% booked beyond target}}\\\\
\\cmidrule(lr){{3-5}}\\cmidrule(lr){{6-7}}\\cmidrule(lr){{8-10}}
Policy & \\% periods & Total & Waiting & Overtime & Days 1--{NEAR_TERM_DAYS} & Later & Urgent & 5-day & 10-day\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
