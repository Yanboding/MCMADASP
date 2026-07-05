"""Benchmark hindsight_solve subproblem runtime (cold vs warm LP solves).

Reproduces exactly what ``./single_case.sh ./table.dat <line>`` does for the
``approx_penalized_hindsight`` policy, but only for the first few decision
epochs, and instruments ``SubproblemWorker.solve`` so cold/warm LP solve times
are reported separately.

Usage:
    PYTHONPATH=. python test/benchmark_hindsight_solve.py \
        [--table table.dat] [--line 1] [--command 0] \
        [--paths 64] [--epochs 2]
"""
import argparse
import json
import os
import sys
import time
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import run as run_module
from experiments.experiment_config import get_config_by_type
from decision_maker import ApproxQAgent
from metaheuristic_algorithm.benders_decomposition_solver import SubproblemWorker
from utils import acquire_grb_env


def load_command(table_path, line_number, command_index):
    with open(table_path) as fh:
        for i, line in enumerate(fh, start=1):
            if i == line_number:
                break
        else:
            raise ValueError(f'{table_path} has fewer than {line_number} lines')
    start = line.index("--params '") + len("--params '")
    end = line.index("'", start)
    params = json.loads(line[start:end])
    if isinstance(params, dict):
        params = [params]
    return params[command_index]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--table', default='table.dat')
    parser.add_argument('--line', type=int, default=1)
    parser.add_argument('--command', type=int, default=0)
    parser.add_argument('--paths', type=int, default=64,
                        help='override agent sample_path_number (0 = keep spec value)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of decision epochs (hindsight_solve calls) to run')
    parser.add_argument('--label', default='run')
    args = parser.parse_args()

    record = load_command(args.table, args.line, args.command)
    spec = next(s for s in record['policy_specs']
                if s['agent_name'] == 'approx_penalized_hindsight')

    config = get_config_by_type(case_type='infinite_custom', args=record['env_args'])
    env = config.env

    agent_args = dict(spec['agent_args'])
    if args.paths:
        agent_args['sample_path_number'] = args.paths
    gf_spec = run_module._normalize_generating_function_spec(
        agent_args.pop('generating_function_spec', None))
    agent_args['generating_function'] = run_module._build_generating_function(env=env, spec=gf_spec)
    run_module._set_sample_path_proposal(agent_args)

    num_cpus = os.cpu_count() or 1
    num_sub_envs = min(agent_args['sample_path_number'], num_cpus)
    grb_env = acquire_grb_env({"Threads": 1}, verbose=False, wait=15)
    grb_sub_envs = [acquire_grb_env({"Threads": 1}, verbose=False, wait=15)
                    for _ in range(num_sub_envs)]
    print(f'[bench] paths={agent_args["sample_path_number"]} sub_envs={len(grb_sub_envs)}')

    agent = ApproxQAgent(env, discount_factor=env.discount_factor,
                         grb_env=grb_env, subproblem_grb_envs=grb_sub_envs,
                         **agent_args)

    # --- instrument SubproblemWorker.solve ---------------------------------
    lock = threading.Lock()
    solve_log = []          # (epoch, seconds)
    current_epoch = {'value': 0}
    original_solve = SubproblemWorker.solve

    def timed_solve(self, action_values, verbose=False):
        t0 = time.time()
        result = original_solve(self, action_values, verbose)
        dt = time.time() - t0
        with lock:
            solve_log.append((current_epoch['value'], dt))
        return result

    SubproblemWorker.solve = timed_solve

    sample_path = np.array(record['sample_path'])
    state, _ = env.reset(init_state=record['init_state'], t=1, new_arrivals=sample_path)

    epoch_summaries = []
    for epoch in range(1, args.epochs + 1):
        current_epoch['value'] = epoch
        t0 = time.time()
        obj, action, info = agent.solve(state, epoch)
        wall = time.time() - t0
        times = [dt for (e, dt) in solve_log if e == epoch]
        epoch_summaries.append({
            'epoch': epoch,
            'objective': obj,
            'wall_seconds': wall,
            'subproblem_solves': len(times),
            'subproblem_total_seconds': sum(times),
            'subproblem_mean_seconds': (sum(times) / len(times)) if times else 0.0,
            'subproblem_max_seconds': max(times) if times else 0.0,
        })
        print(f"[bench:{args.label}] epoch {epoch}: obj={obj}, wall={wall:.1f}s, "
              f"subproblem solves={len(times)}, total={sum(times):.1f}s, "
              f"mean={epoch_summaries[-1]['subproblem_mean_seconds']:.2f}s, "
              f"max={epoch_summaries[-1]['subproblem_max_seconds']:.2f}s")
        state, cost, done, _ = env.step(action)
        if done:
            break

    print(f"\n[bench:{args.label}] SUMMARY")
    for s in epoch_summaries:
        print(json.dumps(s))


if __name__ == '__main__':
    main()
