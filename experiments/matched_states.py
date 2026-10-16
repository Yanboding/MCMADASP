import csv
import glob
import json
import os

import numpy as np

from experiments import get_config_by_type
from param_generation.command_files import read_command_file_records, write_grouped_command_file
from utils import get_uid

POLICIES = (('approx_penalized_hindsight', 'ALP-based penalized hindsight', '#2a78d6'),
            ('row_gen_alp', 'ALP', '#eb6834'),
            ('myopic', 'Myopic', '#1baf7a'))
PATIENT_CLASSES = (('urgent', 'Urgent', [0, 1, 2]),
                   ('target5', 'Five-day target', list(range(6, 14))),
                   ('target10', 'Ten-day target', [3, 4, 5, 14, 15, 16, 17]))
NEAR_DAYS = 5
OUTCOMES = (('on_time', 'On time'), ('late', 'Beyond target'), ('unscheduled', 'Not scheduled'))


def _environment(source_records):
    return get_config_by_type('infinite_custom', args=source_records[0]['env_args']).env


def _replay(record, src, env):
    w = record['warm_up_periods']
    weights = np.asarray(src['period_weights'], dtype=float)
    arrivals = np.asarray(src['sample_path'][w:], dtype=int).reshape(-1, env.num_types)
    state = tuple(np.asarray(component, dtype=float) for component in record['warmup_state'])
    for t, weight in enumerate(weights):
        bookings = np.asarray(record['scheduled_patients'][w + t], dtype=float)
        yield t, w + t + 1, float(weight), state, bookings
        regular = state[0]
        new_overtime = np.maximum(np.round(
            regular + env.convert_action_to_booking_slots(bookings) - env.regular_capacity), 0.0)
        arrival = arrivals[t] if t < len(arrivals) else np.zeros(env.num_types)
        state = env.post_action_state_to_new_state(
            env.post_action_state(state, (bookings, new_overtime)), arrival, is_var=False)


def eligible_periods(source_dat, results_dir, threshold=0.97):
    source_records = read_command_file_records(source_dat)
    source = {record['uid']: record for record in source_records}
    env = _environment(source_records)
    capacity = NEAR_DAYS * env.regular_capacity
    pool = []
    for path in sorted(glob.glob(os.path.join(results_dir, '[0-9]*.jsonl'))):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                for t, period, weight, state, _ in _replay(record, source[record['uid']], env):
                    if state[0][:NEAR_DAYS].sum() >= threshold * capacity:
                        pool.append({'source_policy': record['policy_id'], 'source_uid': record['uid'],
                                     'path_stratum': record['path_stratum'], 'path_weight': record['path_weight'],
                                     'period_index': t, 'period': period, 'weight': weight})
    return pool


def sample_matched_states(pool, per_policy, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for policy_id, _, _ in POLICIES:
        candidates = [state for state in pool if state['source_policy'] == policy_id]
        probabilities = np.array([state['weight'] * state['path_weight'] for state in candidates])
        chosen = rng.choice(len(candidates), size=min(per_policy, len(candidates)), replace=False,
                            p=probabilities / probabilities.sum())
        for index in sorted(chosen):
            samples.append({**candidates[index], 'sampling_seed': seed, 'eligible_states': len(candidates)})
    return samples


def attach_states(source_dat, results_dir, samples):
    source_records = read_command_file_records(source_dat)
    source = {record['uid']: record for record in source_records}
    env = _environment(source_records)
    wanted = {}
    for sample in samples:
        wanted.setdefault((sample['source_uid'], sample['source_policy']), []).append(sample)
    for path in sorted(glob.glob(os.path.join(results_dir, '[0-9]*.jsonl'))):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                targets = wanted.get((record['uid'], record['policy_id']))
                if not targets:
                    continue
                by_index = {target['period_index']: target for target in targets}
                for t, _, _, state, bookings in _replay(record, source[record['uid']], env):
                    if t in by_index:
                        by_index[t]['state'] = [component.astype(int).tolist() for component in state]
                        by_index[t]['recorded_bookings'] = bookings.astype(int).tolist()
    missing = [sample for sample in samples if 'state' not in sample]
    if missing:
        raise ValueError(f'{len(missing)} sampled states could not be reconstructed')
    for state_id, sample in enumerate(samples):
        sample['state_id'] = state_id
        sample['state_uid'] = get_uid({key: value for key, value in sample.items() if key != 'state_id'})
    return samples


def matched_state_command_records(source_dat, samples, experiment_name, batch_size):
    source_records = read_command_file_records(source_dat)
    template = source_records[0]
    records = []
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        record = {'matched_state_decision': True, 'experiment_name': experiment_name,
                  'env_args': template['env_args'], 'policy_specs': template['policy_specs'], 'states': batch}
        record['uid'] = get_uid({'states': [state['state_uid'] for state in batch], 'experiment_name': experiment_name})
        records.append(record)
    return records


def write_matched_state_dat(source_dat, results_dir, dat_file, experiment_name, per_policy=300, seed=20261015,
                            batch_size=10, samples_file=None):
    pool = eligible_periods(source_dat, results_dir)
    samples = attach_states(source_dat, results_dir, sample_matched_states(pool, per_policy, seed))
    if samples_file is not None:
        with open(samples_file, 'w') as handle:
            handle.writelines(json.dumps(sample) + '\n' for sample in samples)
    records = matched_state_command_records(source_dat, samples, experiment_name, batch_size)
    write_grouped_command_file(results=records, num_groups=len(records), dat_file=dat_file)
    counts = {policy_id: sum(1 for state in pool if state['source_policy'] == policy_id) for policy_id, _, _ in POLICIES}
    print(f'{len(pool)} fully loaded states {counts}; sampled {len(samples)} into {len(records)} commands')
    return samples


def type_parameters(env):
    rows = []
    for name, label, types in PATIENT_CLASSES:
        for i in types:
            late_days = np.flatnonzero(env.waiting_cost_coeffs[:, i] > 0)
            target = int(late_days[0]) if late_days.size else env.booking_window_size
            rows.append({'type': i, 'class': name, 'class_label': label, 'waiting_target_days': target,
                         'waiting_cost_per_day_after_target': float(env.holding_cost(target, i)) if target < env.booking_window_size else 0.0,
                         'waiting_cost_first_late_booking': float(env.waiting_cost_coeffs[target, i]) if target < env.booking_window_size else 0.0,
                         'sessions': int((env.treatment_pattern[:, i] > 0).sum()),
                         'slots_per_course': float(env.treatment_pattern[:, i].sum())})
    return rows


def decision_metrics(env, state, bookings, overtime):
    bookings = np.asarray(bookings, dtype=float)
    overtime = np.asarray(overtime, dtype=float)
    waitlist = np.asarray(state[2], dtype=float)
    window, types = bookings.shape
    mapping = env.booking_mapping_matrix[:NEAR_DAYS].reshape(NEAR_DAYS, window, types)
    near_capacity = (mapping * bookings[None]).sum(axis=(0, 1))
    late = env.waiting_cost_coeffs > 0
    booked = bookings.sum(axis=0)
    on_time = (bookings * ~late).sum(axis=0)
    beyond = (bookings * late).sum(axis=0)
    return {
        'overtime_near': float(overtime[:NEAR_DAYS].sum()),
        'overtime_far': float(overtime[NEAR_DAYS:].sum()),
        'overtime_total': float(overtime.sum()),
        'near_capacity_by_type': near_capacity,
        'waitlist_by_type': waitlist,
        'on_time_by_type': on_time,
        'late_by_type': beyond,
        'unscheduled_by_type': waitlist - booked,
    }


def load_matched_state_results(results_dir):
    records = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '[0-9]*.jsonl'))):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                records[record['uid']] = record
    return sorted(records.values(), key=lambda record: record['state_id'])


def matched_state_table(records, env):
    rows = []
    for record in records:
        for policy_id, _, _ in POLICIES:
            decision = record['decisions'][policy_id]
            metrics = decision_metrics(env, record['state'], decision['bookings'], decision['overtime'])
            row = {'state_id': record['state_id'], 'source_policy': record['source_policy'],
                   'source_uid': record['source_uid'], 'path_stratum': record['path_stratum'],
                   'period': record['period'], 'state_weight': record['weight'], 'path_weight': record['path_weight'],
                   'sampling_seed': record['sampling_seed'], 'policy': policy_id,
                   'regular_occupancy_near': float(np.sum(record['state'][0][:NEAR_DAYS]) / (NEAR_DAYS * env.regular_capacity)), 'overtime_near': metrics['overtime_near'],
                   'overtime_far': metrics['overtime_far'], 'overtime_total': metrics['overtime_total'],
                   'remaining_overtime_near': float(NEAR_DAYS * env.overtime_capacity - np.sum(record['state'][1][:NEAR_DAYS])),
                   'reproduces_recorded': policy_id == record['source_policy'] and decision['bookings'] == record['recorded_bookings']}
            for i in range(env.num_types):
                for key in ('near_capacity', 'waitlist', 'on_time', 'late', 'unscheduled'):
                    row[f'{key}_{i}'] = float(metrics[f'{key}_by_type'][i])
            rows.append(row)
    return rows


def _cluster_mean(values, clusters, strata, design_strata):
    values = np.asarray(values, dtype=float)
    keep = ~np.isnan(values)
    values, clusters, strata, design_strata = values[keep], clusters[keep], strata[keep], design_strata[keep]
    labels = np.unique(design_strata)
    estimate = 0.0
    influence = np.zeros_like(values)
    for label in labels:
        in_stratum = design_strata == label
        mean = values[in_stratum].mean()
        estimate += mean / len(labels)
        influence[in_stratum] = (values[in_stratum] - mean) / (len(labels) * in_stratum.sum())
    totals = {}
    for cluster, stratum, z in zip(clusters, strata, influence):
        totals[(stratum, cluster)] = totals.get((stratum, cluster), 0.0) + z
    by_stratum = {}
    for (stratum, _), total in totals.items():
        by_stratum.setdefault(stratum, []).append(total)
    variance = 0.0
    for group in by_stratum.values():
        group = np.asarray(group)
        if len(group) > 1:
            variance += len(group) / (len(group) - 1) * ((group - group.mean()) ** 2).sum()
        else:
            variance += (group ** 2).sum()
    return estimate, 1.96 * np.sqrt(variance), int(keep.sum())


def _ratio_by_state(rows, numerator, denominator):
    num = np.array([row[numerator] for row in rows])
    den = np.array([row[denominator] for row in rows])
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def summarize_matched_states(rows, env, superstrata=16):
    by_policy = {policy_id: [row for row in rows if row['policy'] == policy_id] for policy_id, _, _ in POLICIES}
    reference = by_policy[POLICIES[0][0]]
    clusters = np.array([row['source_uid'] for row in reference])
    strata = np.array([row['path_stratum'] * superstrata // 1024 for row in reference])
    design = np.array([row['source_policy'] for row in reference])
    summary = {'n_states': len(reference), 'n_paths': len(set(clusters)), 'policies': {}, 'differences': {}}

    def per_state(policy_rows):
        values = {key: np.array([row[key] for row in policy_rows]) for key in ('overtime_near', 'overtime_far', 'overtime_total')}
        for name, _, types in PATIENT_CLASSES:
            values[f'near_capacity_{name}'] = np.array([sum(row[f'near_capacity_{i}'] for i in types) for row in policy_rows])
            waitlist = np.array([sum(row[f'waitlist_{i}'] for i in types) for row in policy_rows])
            for outcome, _ in OUTCOMES:
                count = np.array([sum(row[f'{outcome}_{i}'] for i in types) for row in policy_rows])
                with np.errstate(invalid='ignore', divide='ignore'):
                    values[f'{outcome}_{name}'] = np.where(waitlist > 0, 100 * count / waitlist, np.nan)
        for i in range(env.num_types):
            values[f'near_capacity_type{i}'] = np.array([row[f'near_capacity_{i}'] for row in policy_rows])
            for outcome, _ in OUTCOMES:
                values[f'{outcome}_type{i}'] = 100 * _ratio_by_state(policy_rows, f'{outcome}_{i}', f'waitlist_{i}')
        return values

    values = {policy_id: per_state(policy_rows) for policy_id, policy_rows in by_policy.items()}
    for policy_id, policy_values in values.items():
        summary['policies'][policy_id] = {key: _cluster_mean(v, clusters, strata, design) for key, v in policy_values.items()}
    for policy_id, _, _ in POLICIES[1:]:
        summary['differences'][policy_id] = {
            key: _cluster_mean(values[POLICIES[0][0]][key] - values[policy_id][key], clusters, strata, design)
            for key in values[policy_id]}
    summary['reproduction'] = {
        policy_id: float(np.mean([row['reproduces_recorded'] for row in by_policy[policy_id]
                                  if row['source_policy'] == policy_id])) for policy_id, _, _ in POLICIES}
    return summary


def write_matched_state_exports(records, rows, summary, env, out_prefix):
    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
    with open(f'{out_prefix}_decisions.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['state_id', 'policy', 'variable', 'day', 'type', 'value'])
        for record in records:
            for i, count in enumerate(record['state'][2]):
                writer.writerow([record['state_id'], 'pre_decision', 'w', '', i, count])
            for policy_id, _, _ in POLICIES:
                decision = record['decisions'][policy_id]
                for n, day in enumerate(decision['bookings']):
                    for i, count in enumerate(day):
                        if count:
                            writer.writerow([record['state_id'], policy_id, 'x', n + 1, i, count])
                for j, slots in enumerate(decision['overtime']):
                    writer.writerow([record['state_id'], policy_id, 'y', j + 1, '', slots])
    with open(f'{out_prefix}_by_state_and_type.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(f'{out_prefix}_type_parameters.csv', 'w', newline='') as handle:
        parameters = type_parameters(env)
        writer = csv.DictWriter(handle, fieldnames=list(parameters[0]))
        writer.writeheader()
        writer.writerows(parameters)
    with open(f'{out_prefix}_summary.csv', 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['quantity', 'metric', 'estimate', 'half_width_95', 'n_states'])
        for policy_id, metrics in summary['policies'].items():
            for metric, (estimate, half_width, n) in metrics.items():
                writer.writerow([policy_id, metric, estimate, half_width, n])
        for policy_id, metrics in summary['differences'].items():
            for metric, (estimate, half_width, n) in metrics.items():
                writer.writerow([f'penalized_hindsight_minus_{policy_id}', metric, estimate, half_width, n])
    print(f'Saved {out_prefix}_decisions.csv, _by_state_and_type.csv, _type_parameters.csv and _summary.csv')


def plot_matched_states(summary, env, save_file):
    import matplotlib.pyplot as plt
    text_secondary, grid = '#52514e', '#e4e3df'
    parameters = type_parameters(env)
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.0), gridspec_kw={'width_ratios': [1, 1.6, 1.6]})

    def style(ax):
        ax.grid(axis='y', color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=text_secondary, labelsize=8.5, length=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(grid)

    def grouped(ax, keys, labels):
        width = 0.8 / len(POLICIES)
        x = np.arange(len(keys))
        for index, (policy_id, label, color) in enumerate(POLICIES):
            means = [summary['policies'][policy_id][key][0] for key in keys]
            errors = [summary['policies'][policy_id][key][1] for key in keys]
            ax.bar(x + (index - 1) * width, means, width * 0.92, yerr=errors, color=color, label=label,
                   error_kw={'elinewidth': 1, 'capsize': 2, 'ecolor': '#333333'})
        ax.set_xticks(x, labels)
        style(ax)

    grouped(axes[0], ['overtime_near', 'overtime_far'], [f'Days 1-{NEAR_DAYS}', f'Day {NEAR_DAYS + 1} to H'])
    axes[0].set_ylabel('New overtime slots per state', fontsize=9, color=text_secondary)
    axes[0].set_title('(a) Overtime added, by service day', fontsize=10)

    class_labels = []
    for name, label, types in PATIENT_CLASSES:
        class_rows = [row for row in parameters if row['class'] == name]
        targets = sorted({row['waiting_target_days'] for row in class_rows})
        costs = sorted({row['waiting_cost_per_day_after_target'] for row in class_rows})
        sessions = [row['sessions'] for row in class_rows]
        cost_text = f'{min(costs):g}' if min(costs) == max(costs) else f'{min(costs):g}-{max(costs):g}'
        class_labels.append(f"{label}\n{'/'.join(str(t) for t in targets)}-day target\n"
                            f"{cost_text} per late day\n{min(sessions)}-{max(sessions)} sessions")
    grouped(axes[1], [f'near_capacity_{name}' for name, _, _ in PATIENT_CLASSES], class_labels)
    axes[1].set_ylabel(f'New session slots in days 1-{NEAR_DAYS} per state', fontsize=9, color=text_secondary)
    axes[1].set_title(f'(b) Near-term capacity added, by patient class', fontsize=10)

    ax = axes[2]
    hatches = {'on_time': '', 'late': '////', 'unscheduled': '....'}
    width = 0.8 / len(POLICIES)
    for c, (name, label, _) in enumerate(PATIENT_CLASSES):
        for index, (policy_id, _, color) in enumerate(POLICIES):
            bottom = 0.0
            position = c + (index - 1) * width
            for outcome, _ in OUTCOMES:
                mean, half_width, n = summary['policies'][policy_id][f'{outcome}_{name}']
                ax.bar(position, mean, width * 0.92, bottom=bottom, color=color if outcome == 'on_time' else 'white',
                       edgecolor=color, hatch=hatches[outcome], linewidth=1)
                bottom += mean
        n = summary['policies'][POLICIES[0][0]][f'on_time_{name}'][2]
        ax.annotate(f'n = {n}', (c, 102), ha='center', fontsize=8, color=text_secondary)
    ax.set_xticks(range(len(PATIENT_CLASSES)), [label for _, label, _ in PATIENT_CLASSES])
    ax.set_ylim(0, 110)
    ax.set_ylabel('% of the class waitlist before the decision', fontsize=9, color=text_secondary)
    ax.set_title('(c) Scheduling outcome of the current waitlist', fontsize=10)
    style(ax)
    from matplotlib.patches import Patch
    outcome_handles = [Patch(facecolor='#9a9a9a' if outcome == 'on_time' else 'white', edgecolor='#555555',
                             hatch=hatches[outcome], label=label) for outcome, label in OUTCOMES]
    ax.legend(handles=outcome_handles, fontsize=8, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    figure.suptitle(f"Matched states: {summary['n_states']} fully loaded states from {summary['n_paths']} sample paths, "
                    'each solved by all three policies', fontsize=10, y=1.02)
    figure.tight_layout()
    os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
    root = os.path.splitext(save_file)[0]
    for extension in ('.svg', '.pdf', '.png'):
        figure.savefig(root + extension, bbox_inches='tight', dpi=200)
    plt.close(figure)
    print(f'Saved {root}.svg, .pdf and .png')


def matched_state_report(results_dir, source_dat='table_alp_steady_state.dat',
                         out_prefix='experiments/results/cs_matched_states',
                         figure_file='experiments/figures/cs_matched_states.svg'):
    env = _environment(read_command_file_records(source_dat))
    records = load_matched_state_results(results_dir)
    rows = matched_state_table(records, env)
    summary = summarize_matched_states(rows, env)
    write_matched_state_exports(records, rows, summary, env, out_prefix)
    plot_matched_states(summary, env, figure_file)
    print(f"{summary['n_states']} states from {summary['n_paths']} sample paths; "
          f"source policy reproduces its recorded decision: {summary['reproduction']}")
    for quantity, metrics in (('policy', summary['policies']), ('H minus', summary['differences'])):
        for policy_id, values in metrics.items():
            for metric, (estimate, half_width, n) in values.items():
                if 'type' not in metric:
                    print(f'{quantity:<7} {policy_id:<27} {metric:<24} {estimate:8.2f} +- {half_width:6.2f} (n={n})')
    return records, rows, summary
