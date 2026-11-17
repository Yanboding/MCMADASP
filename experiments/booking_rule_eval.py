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


def variant_records(source_records, experiment_name, limit=None):
    """Copy each source path record, keep its ALP spec first, append the six booking-rule variants."""
    records = []
    for source in source_records[:limit]:
        alp_spec = next(spec for spec in source['policy_specs'] if spec['policy_id'] == 'row_gen_alp')
        specs = [alp_spec] + [
            {'policy_id': policy_id, 'agent_name': 'alp_booking_rule',
             'agent_args': {'coefficients': alp_spec['agent_args']['coefficients'],
                            'restricted_types': list(types), 'reserve': reserve}}
            for policy_id, types, reserve in VARIANTS]
        records.append({**source, 'policy_specs': specs, 'experiment_name': experiment_name,
                        'skip_information_relaxation': True})
    return records


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
    from multiprocessing import get_context
    from param_generation.command_files import read_command_file_records
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dat', default='table_alp_steady_state.dat')
    parser.add_argument('--name', required=True, help='results folder under experiments/results')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args(argv)
    records = variant_records(read_command_file_records(args.source_dat), args.name, args.limit)
    chunks = [(records[k::args.workers], f'w{k}') for k in range(args.workers)]
    with get_context('spawn').Pool(args.workers) as pool:
        for job_id in pool.imap_unordered(_evaluate_chunk, chunks):
            print(f'worker {job_id} finished', flush=True)


if __name__ == '__main__':
    main()
