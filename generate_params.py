import json
import math
import random
import pickle
from dataclasses import dataclass, field
from pprint import pprint
from typing import Callable, List, Optional, Sequence

import pandas as pd
import numpy as np
from scipy.stats import geom
from experiments import get_config_by_type
from importance_sampling import build_proposal
from utils import iter_to_tuple, iter_to_list, get_uid, read_lines_with_pattern, RunningStats, encode, decode, wait_time, acquire_grb_env, str2treatment_patterns
from pathlib import Path
import hashlib
import glob
import json
import copy
import os
from gurobipy import GRB

from decision_maker import ALPRowGenerationAgent, ApproxQAgent
from generating_function import MulticlassLinearPenaltyFunction, LinearPenaltyFunction


def _training_uid(env_args, agent_args=None):
    """Cache uid that includes agent_args (e.g. proposal spec) when present.

    When agent_args is None or empty, falls back to the env-only uid for
    backward compatibility with previously cached records.
    """
    if not agent_args:
        return get_uid(env_args)
    return get_uid({'env_args': env_args, 'agent_args': agent_args})


def _load_cached_training_result(experiment_name, file_name, env_args, agent_args):
    # useful parameters for coefficient training
    file_path = os.path.join('experiments', 'results', experiment_name, file_name)
    if not os.path.exists(file_path):
        return None
    target_uid = _training_uid(env_args, agent_args)
    cached_record = None
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('uid') == target_uid:
                cached_record = record
    return cached_record


def _save_training_result(experiment_name, file_name, env_args, agent_name, obj_val, coefficients, info=None, agent_args=None):
    file_path = os.path.join('experiments', 'results', experiment_name, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    record = {
        'uid': _training_uid(env_args, agent_args),
        'result': {
            'agent_name': agent_name,
            'obj_val': obj_val,
            'args': {'coefficients': coefficients, 'agent_args': agent_args or {}},
            'info': info or {},
        },
    }
    with open(file_path, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record

def _split_list_into_groups(results, num_groups, dat_file=None):
    n = num_groups if num_groups and num_groups > 0 else len(results)
    # Split results into n groups as evenly as possible
    groups = [results[i::n] for i in range(min(n, len(results)))]
    lines_to_write = []
    for line_index, group in enumerate(groups, start=1):
        lines_to_write.append(
            f"{line_index} python run.py --params '" + json.dumps(group) + "'\n"
        )
    if dat_file:
        with open(dat_file, 'w') as f:
            f.writelines(lines_to_write)
        print(f"Saved {len(lines_to_write)} group commands to {dat_file}")
    return lines_to_write

def _holding_cost_from_schedule(schedule):
    """Convert a per-type list of (start, end, cost) segments to the
    `holding_cost_by_day_by_type` matrix expected by env_args."""
    return np.array([wait_time(s) for s in schedule]).T.tolist()


# ---------------------------------------------------------------------------
# Experiment specification
# ---------------------------------------------------------------------------
# Each experiment sweeps a parameter over `val_args`. For every value in
# `val_args`, the corresponding mutator is called with `(env_args, val)` and
# mutates `env_args` in place. A mutator MUST be a module-level function
# (no nested closures) so experiments remain easy to find, test, and extend.
#
# To add a new experiment:
#   1. Write a `_mutate_<name>(env_args, val)` function at module level.
#   2. Add an `ExperimentSpec(...)` entry to `EXPERIMENT_SPECS` below.
#   3. (Optional) expose a thin `generate_<name>(dat_file)` wrapper if you
#      want the old calling style.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    config_type: str
    val_args: Sequence
    mutate: Optional[Callable] = None  # signature: (env_args, val) -> None
    agent_mutate: Optional[Callable] = None  # signature: (agent_args, val) -> None

    @property
    def is_single_variant(self) -> bool:
        return len(self.val_args) == 1 and self.val_args[0] is None


def build_variation_test_env(spec: ExperimentSpec):
    """Materialize the experiment's variants.

    Returns a dict keyed by `env_uid` for single-variant experiments, else by
    `(env_uid, experiment_name, val)`. Each value is a dict with two keys:
    ``env_args`` (env-level configuration) and ``agent_args`` (agent-level
    keyword arguments, e.g. proposal specs).
    """
    base_env_args = get_config_by_type(spec.config_type).args
    test_params = {}
    for i, val in enumerate(spec.val_args, start=1):
        env_args = copy.deepcopy(base_env_args)
        agent_args = {
            'policy_id': 'approx_penalized_hindsight',
            'agent_name': 'approx_penalized_hindsight',
            'agent_args': {
                'sample_path_number': 256,
                'current_decision_var_type': 'integer',
                'future_decision_var_type': 'continuous',
                'penalty_ratio': 1,
            },
        }
        if spec.mutate is not None:
            spec.mutate(env_args, val)
        if spec.agent_mutate is not None:
            spec.agent_mutate(agent_args, val)
        group_uid = get_uid({'env_args': env_args, 'agent_args': agent_args})
        key = group_uid if spec.is_single_variant else (group_uid, spec.name, val)
        test_params[key] = {'env_args': env_args, 'agent_args': agent_args}
    return test_params


# ---------------------------------------------------------------------------
# Module-level mutators (one per experiment that needs parameter sweeping).
# Keep these small and self-contained: everything they need should be
# derivable from `env_args` and `val`.
# ---------------------------------------------------------------------------

def _mutate_initial_state_congestion(env_args, occupancy_level):
    pprint(env_args)
    total_capacity = env_args['regular_capacity'] + env_args['overtime_capacity']
    treatment_pattern = str2treatment_patterns(env_args['patterns'])
    planning_horizon = env_args['booking_window_size'] + treatment_pattern.shape[0] - 1
    required_slots = total_capacity * occupancy_level
    regular_booking = min(required_slots, env_args['regular_capacity'])
    overtime_booking = required_slots - regular_booking
    regular_bookings = [regular_booking] * planning_horizon
    regular_bookings[-1] = 0
    overtime_bookings = [overtime_booking] * planning_horizon
    overtime_bookings[-1] = 0
    env_args['reset_params'] = {
        'init_state': (regular_bookings, overtime_bookings, env_args['arrival_rates']),
    }


def _mutate_high_priority_proportion(env_args, proportion):
    total = sum(env_args['arrival_rates'])
    env_args['arrival_rates'] = [total * proportion, total * (1 - proportion)]


def _mutate_low_priority_waiting_time_target(env_args, waiting_time_target):
    bw = env_args['booking_window_size']
    schedule = [
        [(0, 1, 0), (1, bw, 100)],
        [(0, waiting_time_target, 0), (waiting_time_target, bw, 10)],
    ]
    env_args['holding_cost_by_day_by_type'] = _holding_cost_from_schedule(schedule)


def _mutate_high_priority_waiting_time_penalty(env_args, waiting_penalty):
    bw = env_args['booking_window_size']
    schedule = [
        [(0, 1, 0), (1, bw, waiting_penalty)],
        [(0, 1, 0), (1, bw, 10)],
    ]
    env_args['holding_cost_by_day_by_type'] = _holding_cost_from_schedule(schedule)

def _mutate_total_arrival_rate(env_args, total_arrival_rate):
    arrival_rates = np.array(env_args['arrival_rates'])
    env_args['arrival_rates'] = (arrival_rates / sum(arrival_rates) * total_arrival_rate).tolist()

def _mutate_type_1_treatment_pattern(env_args, pattern):
    env_args['patterns'][0] = pattern
    treatment_pattern = str2treatment_patterns(env_args['patterns'])
    booking_window_size = env_args['booking_window_size']
    regular_capacity = env_args['regular_capacity']
    arrival_rates = env_args['arrival_rates']
    planning_horizon = booking_window_size + treatment_pattern.shape[0] - 1
    regular_bookings = [regular_capacity]*planning_horizon
    regular_bookings[-1] = 0
    overtime_bookings = [0]*planning_horizon
    initial_state = (regular_bookings, overtime_bookings, arrival_rates.copy())
    env_args['reset_params'] = {'init_state': initial_state}

def _mutate_overtime_cost(env_args, overtime_cost):
    env_args['overtime_cost_by_day'] = overtime_cost

def _mutate_discount_factor(env_args, discount_factor):
    env_args['discount_factor'] = discount_factor

def _mutate_discount_factor_for_sample_path_length_proposal(agent_args, discount_factor):
    discount_factor_str = str(discount_factor).replace('.', '_')
    agent_args.update({
            'policy_id': 'approx_penalized_hindsight_geometric_' + discount_factor_str,
            'agent_name': 'approx_penalized_hindsight',
        })
    agent_args['agent_args'].update({
                'sample_path_length_proposal': {
                    'type': 'geometric',
                    'discount_factor_proposal': discount_factor,
                }
            })

def _mutate_fixed_length_for_sample_path_length_proposal(agent_args, max_length):
    agent_args.update(
            {
                'policy_id': 'approx_penalized_hindsight_truncated_horizon_' + str(max_length),
                'agent_name': 'approx_penalized_hindsight',
            }
    )
    agent_args['agent_args'].update({
                    'sample_path_length_proposal': {
                        'type': 'fixed',
                        'max_length': max_length-1,
                    }
                })

def _mutate_solver(agent_args, solver_name):
    agent_args.update(
        {
            'policy_id': 'approx_penalized_hindsight_' + solver_name,
            'agent_name': 'approx_penalized_hindsight',
        }
    )
    agent_args['agent_args'].update({
        'solver_name': solver_name,
    })

# ---------------------------------------------------------------------------
# Registry of all experiments. Add new experiments here.
# ---------------------------------------------------------------------------

EXPERIMENT_SPECS = {
    spec.name: spec for spec in [
        ExperimentSpec(
            name='initial_state_congestion',
            config_type='toy',
            val_args=[0., 0.5, 1.],
            mutate=_mutate_initial_state_congestion,
        ),
        ExperimentSpec(
            name='high_priority_proportion',
            config_type='toy',
            val_args=[0.1, 0.5, 0.9],
            mutate=_mutate_high_priority_proportion,
        ),
        ExperimentSpec(
            name='low_priority_waiting_time_target',
            config_type='toy',
            val_args=[1, 3, 5],
            mutate=_mutate_low_priority_waiting_time_target,
        ),
        ExperimentSpec(
            name='high_priority_waiting_time_penalty',
            config_type='toy',
            val_args=[10, 100, 200],
            mutate=_mutate_high_priority_waiting_time_penalty,
        ),
        ExperimentSpec(
            name='total_arrival_rate',
            config_type='toy',
            val_args=[24/7, 30/7, 36/7],
            mutate=_mutate_total_arrival_rate,
        ),
        ExperimentSpec(
            name='type_1_treatment_pattern',
            config_type='toy',
            val_args=['1 * 3', '1 * 2 + 1 * 1', '3 * 1'],
            mutate=_mutate_type_1_treatment_pattern,
        ),
        ExperimentSpec(
            name='overtime_cost',
            config_type='toy',
            val_args=[10, 100, 200],
            mutate=_mutate_overtime_cost,
        ),
        ExperimentSpec(
            name='case_study_discount_factor',
            config_type='toy',
            val_args=[0.95, 0.96, 0.97, 0.98],
            mutate=_mutate_discount_factor,
        ),
        ExperimentSpec(
            name='sample_path_length_proposal_geometric',
            config_type='toy',
            val_args=[0.95, 0.96, 0.98, 0.99],
            agent_mutate=_mutate_discount_factor_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='sample_path_length_proposal_fixed',
            config_type='toy',
            val_args=[100, 200, 400],
            agent_mutate=_mutate_fixed_length_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='solver_comparison',
            config_type='toy',
            val_args=[0.],
            mutate=_mutate_initial_state_congestion,
        ),
        ExperimentSpec(
            name='multiclass_LP_solver_comparison',
            config_type='toy',
            val_args=[0.],
            mutate=_mutate_initial_state_congestion,
        ),
    ]
}

def generate_experiment(experiment_name: str):
    """Preferred entry point: look up a spec by name and build its dat file."""
    return build_variation_test_env(EXPERIMENT_SPECS[experiment_name])



def train_penalty_coefficients(env_args, experiment_name, agent_args=None):
    '''
    Train coefficients for the penalty function used in the hindsight
    approximation with penalty agent. ``agent_args`` may include a
    JSON-serializable ``sample_path_length_proposal`` spec dict that is
    materialized into a SamplePathLengthProposal before constructing the agent.
    '''
    train_params = {
                    'agent_name': agent_args['agent_name'],
                    'agent_args': {
                        'sample_path_number': agent_args['agent_args']['sample_path_number'],
                        'current_decision_var_type': agent_args['agent_args']['current_decision_var_type'],
                        'future_decision_var_type': agent_args['agent_args']['future_decision_var_type'],
                        'penalty_ratio': agent_args['agent_args']['penalty_ratio'],
                    },
                }
    agent_args = dict(agent_args or {})
    cached = _load_cached_training_result(experiment_name, 'penalty_train.jsonl', env_args, agent_args=train_params)
    if cached is not None:
        cached_result = cached.get('result', {})
        cached_obj = cached_result.get('obj_val')
        cached_coefficients = cached_result.get('args', {}).get('coefficients', None)
        cached_info = cached_result.get('info', {})
        print(f"Use cached penalty coefficients for uid={cached.get('uid')}")
        return cached_obj, cached_coefficients, cached_info

    config_for_train = get_config_by_type('infinite_custom', args=env_args)
    env = config_for_train.env
    # generating_function = LinearPenaltyFunction(env=env)
    generating_function = MulticlassLinearPenaltyFunction(env=env)
    inner = dict(agent_args.get('agent_args', {}))
    inner['generating_function'] = generating_function
    if 'sample_path_length_proposal' in inner:
        inner['sample_path_length_proposal'] = build_proposal(
            inner['sample_path_length_proposal']
        )
    pprint(inner)
    agent = ApproxQAgent(env=env, discount_factor=env.discount_factor,
                                      **inner)
    init_state = None if 'init_state' not in env_args.get('reset_params', {}) else env_args['reset_params']['init_state']
    sample_path_number = agent_args['agent_args']['sample_path_number']
    print(f"Training penalty coefficients for env_uid {get_uid(env_args)} with init_state: {init_state}, sample_path_number: {sample_path_number}, agent_args: {agent_args}")
    if init_state is not None:
        init_state = tuple(np.array(item) for item in init_state)
    env.reset_random_seeds()  # Reset random seeds before training again to ensure the same sample paths
    obj, direct_coefficients, info = agent.benders_decomposition_train(coefficient_bound=GRB.INFINITY, init_state=init_state, parallel=True, verbose=False)
    print('Obejctive from Benders decomposition training:', obj)
    print('Coefficients from Benders decomposition training:', direct_coefficients)
    _save_training_result(
        experiment_name=experiment_name,
        file_name='penalty_train.jsonl',
        env_args=env_args,
        agent_name='hindsight_approx_with_penalty',
        obj_val=obj,
        coefficients=direct_coefficients,
        info=info,
        agent_args=train_params,
    )

    return obj, direct_coefficients, info

def train_alp_coefficients(env_args, experiment_name):
    '''
    This function can be implemented to train the coefficients for the ALP row generation agent. The training can be done using a simple grid search or a more sophisticated optimization algorithm.
    '''
    cached = _load_cached_training_result(experiment_name, 'alp_train.jsonl', env_args, agent_args={})
    if cached is not None:
        cached_result = cached.get('result', {})
        cached_obj = cached_result.get('obj_val')
        cached_coefficients = cached_result.get('args', {}).get('coefficients', None)
        print(f"Use cached ALP coefficients for env_uid={cached.get('uid')}")
        return cached_obj, cached_coefficients

    config_for_train = get_config_by_type('infinite_custom', args=env_args)
    env = config_for_train.env
    alp_agent = ALPRowGenerationAgent(env=env, discount_factor=env.discount_factor)
    obj, coefficients = alp_agent.train(debug=False, verbose=False)
    _save_training_result(
        experiment_name=experiment_name,
        file_name='alp_train.jsonl',
        env_args=env_args,
        agent_name='row_gen_alp',
        obj_val=obj,
        coefficients=coefficients,
        info={},
    )
    return obj, coefficients


def generate_test_paths_and_init_state(test_envs, test_sample_path_num, warm_up_periods=0, num_periods=None, dat_file=None, num_groups=None, is_require_penalty_coefficients=True, policy_ids=[]):
    '''
    Inital state is considered as period 1. sample path will start from period 2.

    ``policy_ids`` is an optional iterable of policy_ids registered in
    ``POLICY_SPECS``. The resolved policy spec dicts are embedded in every
    saved params record under the ``policies`` key so the runner knows which
    policies to evaluate against the corresponding sample path.
    '''
    results = []
    for (env_uid, experiment_name, mutate_val), variant in test_envs.items():
        env_args = variant['env_args']
        env = get_config_by_type('infinite_custom', args=env_args).env
        print(f"Processing env_uid: {env_uid}, experiment_name: {experiment_name}, mutate_val: {mutate_val}")
        policies = []
        if 'row_gen_alp' in policy_ids:
            obj_alp_train, alp_coefficients = train_alp_coefficients(env_args=env_args, experiment_name=experiment_name)
            policies.append({
                'policy_id': 'row_gen_alp',
                'agent_name': 'row_gen_alp',
                'agent_args': {
                    'coefficients': alp_coefficients,
                },
            })
        if 'myopic' in policy_ids:
            policies.append({
                'policy_id': 'myopic',
                'agent_name': 'myopic',
                'agent_args': {
                },
            })
        if 'approx_hindsight' in policy_ids:
            policies.append({
                'policy_id': 'approx_hindsight',
                'agent_name': 'approx_hindsight',
                'agent_args': {
                    'sample_path_number': 256,
                    'current_decision_var_type': 'integer',
                    'future_decision_var_type': 'continuous',
                    'is_myopic': False,
                    'penalty_ratio': 0,
                    'is_quasi_MC': True,
                },
            })
        if 'approx_penalized_hindsight' in policy_ids:
            agent_args = copy.deepcopy(variant['agent_args'])
            print(agent_args)
            if is_require_penalty_coefficients:
                obj, direct_coefficients, info = train_penalty_coefficients(env_args=env_args, agent_args=agent_args, experiment_name=experiment_name)
            else:
                direct_coefficients = [0] * (env.planning_horizon * 2 + env.num_types + env.booking_window_size * env.num_types + env.planning_horizon)
            agent_args['agent_args']['penalty_coefficients'] = direct_coefficients
            agent_args.update(
                {
                    'policy_id': 'approx_penalized_hindsight',
                    'agent_name': 'approx_penalized_hindsight',
                }
            )
            agent_args['agent_args'].update({
                'solver_name': 'approx_penalized_hindsight',
            })
            policies.append(agent_args)
        if 'approx_Q' in policy_ids:
            agent_args = copy.deepcopy(variant['agent_args'])
            if is_require_penalty_coefficients:
                obj, direct_coefficients, info = train_penalty_coefficients(env_args=env_args, agent_args=agent_args, experiment_name=experiment_name)
            else:
                direct_coefficients = [0] * (env.planning_horizon * 2 + env.num_types + env.booking_window_size * env.num_types + env.planning_horizon)
            agent_args['agent_args']['penalty_coefficients'] = direct_coefficients
            agent_args.update(
                {
                    'policy_id': 'approx_Q',
                    'agent_name': 'approx_penalized_hindsight',
                }
            )
            agent_args['agent_args'].update({
                'solver_name': 'approx_Q',
            })
            policies.append(agent_args)
        # Inject the freshly trained coefficients into the matching policy specs
        # for this variant so each saved record carries everything the runner
        # needs to instantiate its agents.
        max_length = 0
        sample_gen_args = copy.deepcopy(env_args)
        sample_gen_args["env_random_seed"] = env_args.get("env_random_seed", 0) + 1001 # make sure the random seed for sample path generation is different from the random seed for training ALP
        sample_gen_args['arrival_random_seed'] = env_args.get("arrival_random_seed", 42) + 1001 # Seed for sample path generation
        sample_gen_args['stop_time_random_seed'] = env_args.get("stop_time_random_seed", 1) + 1001
        config_for_sample_path = get_config_by_type('infinite_custom', args=sample_gen_args)
        env_for_sample_path = config_for_sample_path.env
        average_sample_path_length = 0
        for command_id in range(test_sample_path_num):
            init_state = env_for_sample_path.generate_initial_state() if 'init_state' not in env_args.get('reset_params', {}) else env_args['reset_params']['init_state']
            init_state = tuple(np.array(item).tolist() for item in init_state)
            if num_periods is None:
                sample_path = env_for_sample_path.reset_arrivals(stop_time=warm_up_periods)
                additional_sample_path = env_for_sample_path.reset_arrivals()
                sample_path = np.append(sample_path, additional_sample_path, axis=0) if len(sample_path) > 0 else additional_sample_path
            else:
                sample_path = env_for_sample_path.reset_arrivals(stop_time=num_periods)
            sample_path = sample_path.tolist() if hasattr(sample_path, 'tolist') else sample_path
            max_length = max(max_length, len(sample_path))
            average_sample_path_length += len(sample_path)
            params = {
                'init_state': init_state,
                'sample_path': sample_path,
                'warm_up_periods': warm_up_periods,
                'env_args': env_args,
            }
            uid = get_uid(params)
            save_params = {
                'uid': uid,
                "experiment_name": experiment_name,
                "mutate_val": mutate_val,
                **params,
                "penalty_coefficients": direct_coefficients,
                "group_id": env_uid,
                "policy_specs": policies,
            }
            results.append(save_params)
        print("max sample path length:", max_length)
        print("average sample path length:", average_sample_path_length / test_sample_path_num)
    # Write to dat file if specified
    if dat_file:
        lines_to_write = []
        n = num_groups if num_groups and num_groups > 0 else len(results)
        # Split results into n groups as evenly as possible
        groups = [results[i::n] for i in range(min(n, len(results)))]
        for line_index, group in enumerate(groups, start=1):
            lines_to_write.append(
                f"{line_index} python run.py --params '" + json.dumps(group) + "'\n"
            )
        with open(dat_file, 'w') as f:
            f.writelines(lines_to_write)
        print(f"Saved {len(lines_to_write)} group commands to {dat_file}")

    return results

def generate_train_env(test_envs, dat_file=None):
    # This function can be implemented to generate parameters for training the agents. The parameters can include different environment configurations, initial states, and sample paths.
    results = []
    for (env_uid, experiment_name, mutate_val), variant in test_envs.items():
        env_args = variant['env_args']
        agent_args = variant.get('agent_args', {})
        save_params = {
                "experiment_name": experiment_name,
                "mutate_val": mutate_val,
                "sample_path_number": 256,
                'env_args': env_args,
                'agent_args': agent_args,
            }
        results.append(save_params)
    if dat_file:
        lines_to_write = []
        for line_index, result in enumerate(results, start=1):
            lines_to_write.append(
                f"{line_index} python run.py --params '" + json.dumps(result) + "'\n"
            )
        with open(dat_file, 'w') as f:
            f.writelines(lines_to_write)
        print(f"Saved {len(lines_to_write)} group commands to {dat_file}")
    return results


if __name__ == '__main__':
    # test_envs = {}
    # experiments = list(EXPERIMENT_SPECS.keys())
    # for experiment_name in experiments:
    #     test_envs.update(build_variation_test_env(EXPERIMENT_SPECS[experiment_name]))
    # test_envs = build_variation_test_env(EXPERIMENT_SPECS['sample_path_length_proposal_fixed'])
    test_envs = build_variation_test_env(EXPERIMENT_SPECS['multiclass_LP_solver_comparison'])
    results = generate_test_paths_and_init_state(
        test_envs=test_envs,
        test_sample_path_num=5000,
        warm_up_periods=100,
        num_periods=None,
        dat_file='table.dat',
        num_groups=998,  # divide into N groups
        is_require_penalty_coefficients=True,
        policy_ids=['approx_penalized_hindsight', 'approx_Q'],
    )
    # test_envs = build_variation_test_env(EXPERIMENT_SPECS['case_study_discount_factor'])
    # results = generate_train_env(
    #     test_envs=test_envs,
    #     dat_file='table.dat'
    # )
    #     dat_file='table.dat'
    # )



