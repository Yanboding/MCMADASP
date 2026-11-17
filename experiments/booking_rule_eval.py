"""Evaluate ALP policies with a start-day booking rule on existing case-study sample paths.

The agent lives in decision_maker/alp_booking_rule_agent.py. This module only adds the
agent name 'alp_booking_rule' to run.build_policy_agent, without editing run.py.
"""
import run
from decision_maker.alp_booking_rule_agent import ALPBookingRuleAgent

_run_build_policy_agent = run.build_policy_agent


def build_policy_agent(env, agent_name, agent_args, grb_env=None, grb_sub_envs=None):
    if agent_name == 'alp_booking_rule':
        return ALPBookingRuleAgent(env, discount_factor=env.discount_factor, grb_env=grb_env, **dict(agent_args))
    return _run_build_policy_agent(env, agent_name, agent_args, grb_env=grb_env, grb_sub_envs=grb_sub_envs)


def install():
    """Route run.build_policy_agent through this module (used by the evaluation runner)."""
    run.build_policy_agent = build_policy_agent


LONG_TYPES = [5, 13, 14, 15, 16, 17]       # Types 6, 14-18: 33 or more capacity slots per course
TEN_DAY_TYPES = [3, 4, 5, 14, 15, 16, 17]  # Types 4-6, 15-18: ten-day waiting-time target
VARIANTS = [(f'alp_rule_{name}_R{reserve}', types, reserve)
            for name, types in (('long', LONG_TYPES), ('tenday', TEN_DAY_TYPES))
            for reserve in (0, 10, 20)]


def variant_records(source_records, experiment_name, limit=None, variants=None, include_alp=True):
    """Copy each source path record, keep its ALP spec first, append the booking-rule variants.

    ``variants`` selects and orders variant ids (default: all six); ``include_alp`` keeps the ALP policy.
    """
    chosen = [v for v in VARIANTS] if variants is None else [next(v for v in VARIANTS if v[0] == name)
                                                                for name in variants]
    records = []
    for source in source_records[:limit]:
        alp_spec = next(spec for spec in source['policy_specs'] if spec['policy_id'] == 'row_gen_alp')
        specs = ([alp_spec] if include_alp else []) + [
            {'policy_id': policy_id, 'agent_name': 'alp_booking_rule',
             'agent_args': {'coefficients': alp_spec['agent_args']['coefficients'],
                            'restricted_types': list(types), 'reserve': reserve}}
            for policy_id, types, reserve in chosen]
        records.append({**source, 'policy_specs': specs, 'experiment_name': experiment_name,
                        'skip_information_relaxation': True})
    return records


def remaining_batch(source_records, done_uids, n_batches, batch):
    """Batch ``batch`` of the unfinished paths: systematic sample of every n_batches-th path in stratum order."""
    remaining = sorted((r for r in source_records if r['uid'] not in done_uids),
                       key=lambda r: r.get('path_stratum', 0))
    return remaining[batch::n_batches]


def from_stored_warm_up(records, saved_warm_up_states):
    """Start each path from its saved post-warm-up state instead of rerunning the warm-up."""
    out = []
    for record in records:
        state = saved_warm_up_states[record['uid']]
        w = record['warm_up_periods']
        out.append({**record, 'init_state': state, 'sample_path': record['sample_path'][w:],
                    'warm_up_periods': 0, 'warm_up_policy_id': None})
    return out


def load_saved_warm_up_states(results_dir, policy_id='row_gen_alp'):
    """{uid: warmup_state} from existing evaluation records of the same paths."""
    import glob
    import json
    import os
    states = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '[0-9]*.jsonl'))):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                if record.get('policy_id') == policy_id:
                    states.setdefault(record['uid'], record['warmup_state'])
    return states


def finished_uids(results_dir, variants=None):
    """uids that have a record for every requested variant in results_dir (empty set if no folder)."""
    import glob
    import json
    import os
    if not results_dir:
        return set()
    wanted = set(variants or [v[0] for v in VARIANTS])
    seen = {}
    for path in glob.glob(os.path.join(results_dir, '*.jsonl')):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                seen.setdefault(record['uid'], set()).add(record['policy_id'])
    return {uid for uid, ids in seen.items() if wanted <= ids}


def _evaluate_chunk(args):
    import inspect
    import os
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from utils.tools import acquire_grb_env
    records, job_id = args
    install()
    parameters = inspect.signature(run.evaluate_policy_costs_with_information_relaxation).parameters
    grb_env = acquire_grb_env({'Threads': 1}, verbose=False, wait=15)
    try:
        for record in records:
            kwargs = {k: v for k, v in record.items() if k in parameters}
            run.evaluate_policy_costs_with_information_relaxation(
                **kwargs, grb_env=grb_env, grb_sub_envs=[], job_id=job_id)
            print(f'[{job_id}] done {record["uid"]}', flush=True)
    finally:
        grb_env.dispose()
    return job_id


def main(argv=None):
    import argparse
    import json
    from param_generation.command_files import read_command_file_records
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dat', default='table_alp_steady_state.dat')
    parser.add_argument('--name', required=True, help='results folder under experiments/results')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--offset', type=int, default=0, help='skip the first OFFSET source paths')
    parser.add_argument('--variants', default=None, help='comma-separated variant ids (default: all six)')
    parser.add_argument('--no-alp', action='store_true', help='do not evaluate the ALP policy itself')
    parser.add_argument('--batch', default=None, metavar='B/K',
                        help='run batch B of K of the paths not yet finished in --done-from (systematic by stratum)')
    parser.add_argument('--done-from', default=None, metavar='DIR',
                        help='results folder whose finished uids (all requested variants present) are skipped')
    parser.add_argument('--warm-up-states-from', default=None, metavar='DIR',
                        help='results folder whose row_gen_alp records give each path its saved warm-up state')
    parser.add_argument('--warm-up-states-json', default=None, metavar='FILE',
                        help='json file {uid: saved warm-up state}, used instead of --warm-up-states-from')
    parser.add_argument('--job_id', default=None,
                        help='Meta Farm meta-job id; output files are named <job_id>-w<k>.jsonl')
    args = parser.parse_args(argv)
    source = read_command_file_records(args.source_dat)[args.offset:]
    variants = args.variants.split(',') if args.variants else None
    if args.batch:
        b, k = map(int, args.batch.split('/'))
        source = remaining_batch(source, finished_uids(args.done_from, variants), k, b)
    records = variant_records(source, args.name, args.limit, variants=variants, include_alp=not args.no_alp)
    if args.warm_up_states_from:
        records = from_stored_warm_up(records, load_saved_warm_up_states(args.warm_up_states_from))
    if args.warm_up_states_json:
        with open(args.warm_up_states_json) as handle:
            records = from_stored_warm_up(records, json.load(handle))
    prefix = f'{args.job_id}-' if args.job_id else ''
    chunks = [(records[k::args.workers], f'{prefix}w{k}') for k in range(args.workers)]
    _run_chunks(chunks, args.workers)


def _run_chunks(chunks, workers):
    from multiprocessing import get_context
    if workers == 1:
        for chunk in chunks:
            print(f'worker {_evaluate_chunk(chunk)} finished', flush=True)
        return
    with get_context('spawn').Pool(workers) as pool:
        for job_id in pool.imap_unordered(_evaluate_chunk, chunks):
            print(f'worker {job_id} finished', flush=True)


if __name__ == '__main__':
    main()
