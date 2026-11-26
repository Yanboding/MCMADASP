"""Figure 1 drawn from all post-warm-up periods: waiting times of the treatments booked by the ALP policy and the
ALP-based penalized hindsight policy, each on the periods of its own run.

usage: python -m analysis.figures.cs_all_states [--results-dir DIR] [--out-dir DIR] [--refresh] [--source-dat FILE]

The results keep the state only at the end of the warm-up. Every later state is rebuilt by replaying the recorded
decisions: the overtime of a period is the part of the new bookings above the regular capacity, and the new arrivals
come from the sample path of the same uid in the command file of the experiment (table_alp_steady_state.dat). The
replayed cost of every period is compared with the recorded cost.
For each policy, the counts are summed over the periods by the number of slots booked in the booking window before
the decision, so any cut points can be applied later. The counts are saved to
<results folder>/analysis_cache/cs_all_states.pkl; no other file is written in the results tree.
The figure is the drawing of cs_matched_states with these counts; every period has equal weight, and a treatment
left on the waitlist is waiting again in the next period.
"""
import argparse
import glob
import json
import os

import numpy as np

from analysis.common import CACHE_DIR, DEFAULT_OUT_DIR, DEFAULT_RESULTS_DIR, REPO_ROOT, cached, check_out_dir
from analysis.figures import cs_matched_states as fig

FILE_NAME = 'cs_all_states_waiting_time_by_type_sessions.svg'
CACHE_FILE, CACHE_VERSION = 'cs_all_states.pkl', 1
CUTOFFS = (60, 69)               # the cut points of the first-state figure
DEFAULT_SOURCE_DAT = os.path.join(REPO_ROOT, 'table_alp_steady_state.dat')
_ARRIVALS = None


def replay(record, arrivals, env):
    """Yield (slots booked in the booking window, decision, waitlist) for every post-warm-up period of a record."""
    w0 = record['warm_up_periods']
    n_days = env.booking_window_size
    state = tuple(np.asarray(component, float) for component in record['warmup_state'])
    tail = np.asarray(arrivals)[w0:]
    for t in range(len(record['costs']) - w0):
        bookings = np.asarray(record['scheduled_patients'][w0 + t], float)
        regular, overtime, waitlist = state
        yield int(round((regular[:n_days] + overtime[:n_days]).sum())), bookings.astype(int), waitlist.astype(int)
        new_overtime = np.maximum(np.round(regular + env.convert_action_to_booking_slots(bookings) - env.regular_capacity), 0.0)
        arrival = tail[t] if t < len(tail) else np.zeros(env.num_types)
        state = env.post_action_state_to_new_state(env.post_action_state(state, (bookings, new_overtime)), arrival, is_var=False)


def replayed_cost_difference(record, arrivals, env):
    """Largest absolute difference between the replayed and the recorded cost of the post-warm-up periods."""
    w0 = record['warm_up_periods']
    state = tuple(np.asarray(component, float) for component in record['warmup_state'])
    tail = np.asarray(arrivals)[w0:]
    worst = 0.0
    for t in range(len(record['costs']) - w0):
        bookings = np.asarray(record['scheduled_patients'][w0 + t], float)
        new_overtime = np.maximum(np.round(state[0] + env.convert_action_to_booking_slots(bookings) - env.regular_capacity), 0.0)
        worst = max(worst, abs(float(env.cost_fn(state, (bookings, new_overtime))) - record['costs'][w0 + t]))
        arrival = tail[t] if t < len(tail) else np.zeros(env.num_types)
        state = env.post_action_state_to_new_state(env.post_action_state(state, (bookings, new_overtime)), arrival, is_var=False)
    return worst


def empty_counts(capacity, n_days, n_types):
    size = capacity + 1
    return {'periods': np.zeros(size, np.int64), 'booked': np.zeros((size, n_days, n_types), np.int64),
            'waiting': np.zeros((size, n_types), np.int64), 'first_periods': np.zeros(size, np.int64),
            'first_booked': np.zeros((size, n_days, n_types), np.int64), 'first_waiting': np.zeros((size, n_types), np.int64)}


def add_path(counts, periods):
    """Add the (slots booked, decision, waitlist) periods of one path; the first one also goes to the first_ counts."""
    for t, (booked, bookings, waitlist) in enumerate(periods):
        counts['periods'][booked] += 1
        counts['booked'][booked] += bookings
        counts['waiting'][booked] += waitlist
        if t == 0:
            counts['first_periods'][booked] += 1
            counts['first_booked'][booked] += bookings
            counts['first_waiting'][booked] += waitlist


def all_states_table(counts, cutoffs, capacity, first_only=False):
    """Table in the format of cs_matched_states.first_decision_table (sums instead of means) and periods per group."""
    prefix = 'first_' if first_only else ''
    slots = np.arange(capacity + 1)
    label = np.where(slots < cutoffs[0] * capacity / 100, 0, np.where(slots < cutoffs[1] * capacity / 100, 1, 2))
    table, n_periods = {}, {}
    for _, name in fig.POLICIES:
        c = counts[name]
        for k, g in enumerate(fig.GROUPS):
            rows = label == k
            n_periods[(g, name)] = int(c[prefix + 'periods'][rows].sum())
            booked = c[prefix + 'booked'][rows].sum(0).astype(float)
            waiting = c[prefix + 'waiting'][rows].sum(0).astype(float)
            for i in range(booked.shape[1]):
                table[(g, name, i + 1)] = {'n_states': n_periods[(g, name)], 'd': booked[:, i],
                                           'left': float(waiting[i] - booked[:, i].sum())}
    return table, n_periods


def _init(arrivals):
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    global _ARRIVALS
    _ARRIVALS = arrivals


def _loads(line):
    try:
        import orjson
        return orjson.loads(line)
    except ImportError:
        return json.loads(line)


def _read_file(path):
    """Replay the ALP and PH records of one jsonl file."""
    from experiments import get_config_by_type
    env = get_config_by_type('ejor').env
    wanted = {policy: name for policy, name in fig.POLICIES}
    out = []
    with open(path, 'rb') as handle:
        for line in handle:
            if not line.strip():
                continue
            record = _loads(line)
            if record['policy_id'] not in wanted:
                continue
            arrivals = _ARRIVALS[record['uid']]
            periods = [(b, x.astype(np.int16), w.astype(np.int16)) for b, x, w in replay(record, arrivals, env)]
            out.append((wanted[record['policy_id']], record['uid'], periods,
                        replayed_cost_difference(record, arrivals, env)))
    return out


def read_results(results_dir, source_dat, processes=7):
    """Replay every (policy, uid) once, the first record in sorted file order, and sum the counts per policy."""
    import multiprocessing as mp
    from experiments import get_config_by_type
    from param_generation.command_files import read_command_file_records
    env = get_config_by_type('ejor').env
    arrivals = {}
    for record in read_command_file_records(source_dat):
        if record['uid'] in arrivals:
            raise ValueError(f'uid {record["uid"]} appears twice in {source_dat}')
        arrivals[record['uid']] = np.asarray(record['sample_path'], np.int16).reshape(-1, env.num_types)
    capacity = env.booking_window_size * (env.regular_capacity + env.overtime_capacity)
    counts = {name: empty_counts(capacity, env.booking_window_size, env.num_types) for _, name in fig.POLICIES}
    seen, worst = {name: set() for _, name in fig.POLICIES}, 0.0
    paths = sorted(glob.glob(os.path.join(results_dir, '[0-9]*.jsonl')))
    with mp.get_context('spawn').Pool(processes, initializer=_init, initargs=(arrivals,)) as pool:
        for chunk in pool.imap(_read_file, paths):
            for name, uid, periods, difference in chunk:
                if uid in seen[name]:
                    continue
                seen[name].add(uid)
                add_path(counts[name], periods)
                worst = max(worst, difference)
    alp, ph = (seen[name] for _, name in fig.POLICIES)
    if alp != ph:
        raise ValueError(f'uids without both policies: {sorted(alp ^ ph)}')
    return {'counts': counts, 'n_paths': len(alp), 'max_abs_cost_difference': worst}


def load_data(results_dir=DEFAULT_RESULTS_DIR, refresh=False, source_dat=DEFAULT_SOURCE_DAT, processes=7):
    """Return (data, 'cache' or 'built'). The cache is the only file written inside the results folder."""
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(key, '1')
    return cached(results_dir, CACHE_FILE, CACHE_VERSION, lambda: read_results(results_dir, source_dat, processes), refresh)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--results-dir', default=DEFAULT_RESULTS_DIR)
    parser.add_argument('--out-dir', default=DEFAULT_OUT_DIR)
    parser.add_argument('--source-dat', default=DEFAULT_SOURCE_DAT)
    parser.add_argument('--refresh', action='store_true', help='replay the jsonl records again')
    args = parser.parse_args(argv)
    try:
        out_dir = check_out_dir(args.out_dir)
    except ValueError as error:
        parser.error(str(error))

    from experiments import get_config_by_type
    env = get_config_by_type('ejor').env
    data, source = load_data(args.results_dir, refresh=args.refresh, source_dat=args.source_dat)
    capacity = env.booking_window_size * (env.regular_capacity + env.overtime_capacity)
    table, n_periods = all_states_table(data['counts'], CUTOFFS, capacity)
    spec = fig.type_spec(env)
    os.makedirs(out_dir, exist_ok=True)
    path = fig.plot_figure(table, spec, CUTOFFS, os.path.join(out_dir, FILE_NAME))
    print('cache:', 'used' if source == 'cache' else 'built', os.path.join(args.results_dir, CACHE_DIR, CACHE_FILE))
    print('paths:', data['n_paths'], 'max |replayed - recorded cost|:', data['max_abs_cost_difference'])
    print('periods per group:', {f'{g} {p}': n for (g, p), n in n_periods.items()})
    for line in fig.bar_totals(table, spec):
        print(line)
    print('saved', path)


if __name__ == '__main__':
    main()
