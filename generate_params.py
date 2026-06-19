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
from generating_function import MulticlassQuadraticPenaltyFunction, LinearPenaltyFunction


def _normalize_generating_function_spec(spec):
    if spec is None:
        return {'name': 'linear_penalty'}
    if isinstance(spec, str):
        return {'name': spec}
    if isinstance(spec, dict):
        spec = dict(spec)
        if 'name' not in spec and 'type' in spec:
            spec['name'] = spec.pop('type')
        spec.setdefault('name', 'linear_penalty')
        return spec
    raise ValueError(f"Unsupported generating function spec: {spec}")


def _build_generating_function(env, spec, coefficients=None):
    spec = _normalize_generating_function_spec(spec)
    name = spec.get('name')
    if coefficients is None:
        coefficients = spec.get('coefficients')
    if name == 'linear_penalty':
        generating_function = LinearPenaltyFunction(env=env, coefficients=coefficients)
    elif name in {'multiclass_quadratic_penalty', 'quadratic_penalty'}:
        generating_function = MulticlassQuadraticPenaltyFunction(env=env, coefficients=coefficients)
    else:
        raise ValueError(f"Unsupported generating function name: {name}")
    if coefficients is None:
        # No coefficients supplied anywhere -> treat all coefficients as zeros.
        generating_function.set_coefficients([0.0] * generating_function.number_of_coefficients)
    return generating_function


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
    print(f"Looking for cached training result with uid={target_uid} in {file_path}...")
    cached_record = None
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            print(record)
            if record.get('uid') == target_uid:
                cached_record = record
    print('cached_record')
    print(cached_record)
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
                'generating_function_spec': {'name': 'linear_penalty'},
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

def _mutate_is_proposal_098_const(agent_args, val):
    """Always set a geometric IS proposal with discount_factor_proposal=0.98.

    Used together with a 0.99 target discount-factor env to importance-sample
    longer paths from the cheaper 0.98 geometric distribution.
    """
    _mutate_discount_factor_for_sample_path_length_proposal(agent_args, 0.98)


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
        ExperimentSpec(
            name='case_study',
            config_type='ejor',
            val_args=[0.95],
            mutate=_mutate_discount_factor,
        ),
        ExperimentSpec(
            name='case_study_098',
            config_type='ejor',
            val_args=[0.98],
            mutate=_mutate_discount_factor,
        ),
        ExperimentSpec(
            name='case_study_099',
            config_type='ejor',
            val_args=[0.99],
            mutate=_mutate_discount_factor,
        ),
        ExperimentSpec(
            name='case_study_099_is_098',
            config_type='toy',
            val_args=[0.99],
            mutate=_mutate_discount_factor,
            agent_mutate=_mutate_is_proposal_098_const,
        ),
        ExperimentSpec(
            name='case_study_099_fixed_length',
            config_type='ejor',
            val_args=[20, 40, 100, 160, 200],
            agent_mutate=_mutate_fixed_length_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='case_study_099_base_case',
            config_type='toy',
            val_args=[0.99],
            agent_mutate=_mutate_discount_factor_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='case_study_099_importance_sampling_proposal_098',
            config_type='ejor',
            val_args=[0.98],
            agent_mutate=_mutate_discount_factor_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='case_study_099_importance_sampling_proposal_095',
            config_type='toy',
            val_args=[0.95],
            agent_mutate=_mutate_discount_factor_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='toy_study_base_case',
            config_type='toy',
            val_args=[0.99],
            agent_mutate=_mutate_discount_factor_for_sample_path_length_proposal,
        ),
    ]
}
# [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 22.756075675673316, 5.894690591201279, 2.816597653273641, 1.50114430484255, 0.8218579465833417, -0.5004316436927129, 0.0, -4.156766115621662, -7.05872867959643, -7.485017207490414, -8.148766425983194, -8.146939862015877, -7.88218278691842, 0.0, 19.782585035499658, 3.5526319213952844, 2.0003054355612004, 0.8390483647272565, 0.9481841830252683, -0.8081045585832958, 0.0, -5.037070467540705, -7.35438894284113, -7.412090644464743, -8.375699506923448, -8.423755598372676, -8.146059498365254, 0.0, 358.9574271739, 192.65655887638724]
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
    agent_args = dict(agent_args or {})
    train_agent_args = {
        'sample_path_number': agent_args['agent_args']['sample_path_number'],
        'current_decision_var_type': agent_args['agent_args']['current_decision_var_type'],
        'future_decision_var_type': agent_args['agent_args']['future_decision_var_type'],
        'penalty_ratio': agent_args['agent_args']['penalty_ratio'],
    }
    # Include the IS proposal spec in the cache key so variants with different
    # proposals are stored and retrieved independently.
    for _key in ('sample_path_length_proposal', 'sample_path_proposal'):
        if _key in agent_args.get('agent_args', {}):
            train_agent_args['sample_path_length_proposal'] = agent_args['agent_args'][_key]
            break
    train_agent_args['training_generating_function_spec'] = _normalize_generating_function_spec(
        agent_args.get('agent_args', {}).get('training_generating_function_spec')
    )
    train_params = {
        'agent_name': agent_args['agent_name'],
        'agent_args': train_agent_args,
    }
    cached = _load_cached_training_result(experiment_name, 'penalty_train.jsonl', env_args, agent_args=train_params)
    if cached is not None:
        cached_result = cached.get('result', {})
        cached_obj = cached_result.get('obj_val')
        cached_coefficients = cached_result.get('args', {}).get('coefficients', None)
        cached_info = cached_result.get('info', {})
        if isinstance(cached_info, dict) and cached_info.get('debug'):
            print(
                f"Ignore cached failed penalty coefficients for uid={cached.get('uid')} "
                f"(debug={cached_info.get('debug')}); retraining."
            )
        else:
            print(f"Use cached penalty coefficients for uid={cached.get('uid')}")
            return cached_obj, cached_coefficients, cached_info

    config_for_train = get_config_by_type('infinite_custom', args=env_args)
    env = config_for_train.env
    training_generating_function_spec = _normalize_generating_function_spec(
        agent_args.get('agent_args', {}).get('training_generating_function_spec')
    )
    generating_function = _build_generating_function(env=env, spec=training_generating_function_spec)
    # generating_function = MulticlassLinearPenaltyFunction(env=env)
    inner = dict(agent_args.get('agent_args', {}))
    for key in (
        'policy_generating_function_spec',
        'zero_lowerbound_generating_function_spec',
        'penalized_lowerbound_generating_function_spec',
        'training_generating_function_spec',
    ):
        inner.pop(key, None)
    inner['generating_function'] = generating_function
    # Materialize the IS proposal spec (legacy key ``sample_path_length_proposal``
    # or ``sample_path_proposal``) into a built proposal instance for the agent.
    proposal_spec = inner.pop('sample_path_length_proposal', None)
    if 'sample_path_proposal' in inner:
        proposal_spec = inner['sample_path_proposal']
    proposal = build_proposal(proposal_spec)
    if proposal is not None:
        inner['sample_path_proposal'] = proposal
    else:
        inner.pop('sample_path_proposal', None)
    pprint(inner)
    agent = ApproxQAgent(env=env, discount_factor=env.discount_factor,
                                      **inner)
    init_state = None if 'init_state' not in env_args.get('reset_params', {}) else env_args['reset_params']['init_state']
    sample_path_number = agent_args['agent_args']['sample_path_number']
    print(f"Training penalty coefficients for env_uid {get_uid(env_args)} with init_state: {init_state}, sample_path_number: {sample_path_number}, agent_args: {agent_args}")
    if init_state is not None:
        init_state = tuple(np.array(item) for item in init_state)
    required_bookings = [(env.regular_capacity + env.overtime_capacity) * env.discount_factor**(j) for j in range(env.planning_horizon)]
    required_bookings[-1] = 0
    required_bookings = np.array(required_bookings)
    E_u_alpha = np.minimum(required_bookings, env.regular_capacity)
    E_v_alpha = required_bookings - E_u_alpha
    E_w_alpha = env.arrival_generator.mean_by_type
    init_state = (E_u_alpha, E_v_alpha, E_w_alpha)
    env.reset_random_seeds()  # Reset random seeds before training again to ensure the same sample paths
    print(init_state)
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


def _zero_penalty_coefficients(env):
    return [0] * (
        env.planning_horizon * 2
        + env.num_types
        + env.booking_window_size * env.num_types
        + env.planning_horizon
    )


def _build_penalty_policy(base_agent_args, policy_id, solver_name, penalty_coefficients, generating_function_spec=None):
    policy = copy.deepcopy(base_agent_args)
    policy.update(
        {
            'policy_id': policy_id,
            'agent_name': 'approx_penalized_hindsight',
        }
    )
    policy['agent_args']['solver_name'] = solver_name
    policy['agent_args'].pop('penalty_coefficients', None)
    policy['agent_args'].pop('policy_generating_function_spec', None)
    policy['agent_args'].pop('zero_lowerbound_generating_function_spec', None)
    policy['agent_args'].pop('penalized_lowerbound_generating_function_spec', None)
    policy['agent_args'].pop('training_generating_function_spec', None)
    # Self-contained spec: embed the policy's penalty coefficients in the spec.
    spec = _normalize_generating_function_spec(generating_function_spec)
    spec['coefficients'] = penalty_coefficients
    policy['agent_args']['generating_function_spec'] = spec
    return policy


def generate_test_paths_and_init_state(test_envs, test_sample_path_num, warm_up_periods=0, num_periods=None, dat_file=None, num_groups=None, is_require_penalty_coefficients=True, policy_ids=None):
    '''
    Inital state is considered as period 1. sample path will start from period 2.

    ``policy_ids`` is an optional iterable of policy_ids registered in
    ``POLICY_SPECS``. The resolved policy spec dicts are embedded in every
    saved params record under the ``policies`` key so the runner knows which
    policies to evaluate against the corresponding sample path.
    '''
    policy_ids = list(policy_ids or [])
    policy_id_set = set(policy_ids)
    results = []
    for (env_uid, experiment_name, mutate_val), variant in test_envs.items():
        env_args = variant['env_args']
        env = get_config_by_type('infinite_custom', args=env_args).env
        print(f"Processing env_uid: {env_uid}, experiment_name: {experiment_name}, mutate_val: {mutate_val}")
        policy_generating_function_spec = _normalize_generating_function_spec(
            variant.get('agent_args', {}).get('agent_args', {}).get('policy_generating_function_spec')
        )
        lowerbound_generating_function_spec = _normalize_generating_function_spec(
            variant.get('agent_args', {}).get('agent_args', {}).get('penalized_lowerbound_generating_function_spec')
        )

        policies = []
        if 'row_gen_alp' in policy_id_set:
            _, alp_coefficients = train_alp_coefficients(env_args=env_args, experiment_name=experiment_name)
            policies.append({
                'policy_id': 'row_gen_alp',
                'agent_name': 'row_gen_alp',
                'agent_args': {
                    'coefficients': alp_coefficients,
                },
            })
        if 'myopic' in policy_id_set:
            policies.append({
                'policy_id': 'myopic',
                'agent_name': 'myopic',
                'agent_args': {
                },
            })

        direct_coefficients = _zero_penalty_coefficients(env)
        if is_require_penalty_coefficients:
            agent_args = copy.deepcopy(variant['agent_args'])
            _, direct_coefficients, _ = train_penalty_coefficients(
                env_args=env_args,
                agent_args=agent_args,
                experiment_name=experiment_name,
            )

        if 'approx_hindsight' in policy_id_set:
            policies.append(
                _build_penalty_policy(
                    base_agent_args=variant['agent_args'],
                    policy_id='approx_hindsight',
                    solver_name='approx_penalized_hindsight',
                    penalty_coefficients= _zero_penalty_coefficients(env),
                    generating_function_spec=policy_generating_function_spec,
                )
            )

        if 'approx_penalized_hindsight' in policy_id_set:
            policies.append(
                _build_penalty_policy(
                    base_agent_args=variant['agent_args'],
                    policy_id='approx_penalized_hindsight',
                    solver_name='approx_penalized_hindsight',
                    penalty_coefficients=direct_coefficients,
                    generating_function_spec=policy_generating_function_spec,
                )
            )
        if 'approx_Q' in policy_id_set:
            policies.append(
                _build_penalty_policy(
                    base_agent_args=variant['agent_args'],
                    policy_id='approx_Q',
                    solver_name='approx_Q',
                    penalty_coefficients=direct_coefficients,
                    generating_function_spec=policy_generating_function_spec,
                )
            )

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
        for _ in range(test_sample_path_num):
            init_state = env_for_sample_path.generate_initial_state() if 'init_state' not in env_args.get('reset_params', {}) else env_args['reset_params']['init_state']
            init_state = tuple(np.array(item).tolist() for item in init_state)
            if num_periods is None:
                warm_up_path = env_for_sample_path.reset_arrivals(stop_time=warm_up_periods)
                sampled_path = env_for_sample_path.reset_arrivals()
                sample_path = np.concatenate((warm_up_path, sampled_path), axis=0) if len(warm_up_path) > 0 else sampled_path
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
                "generating_function_spec": {**lowerbound_generating_function_spec, 'coefficients': direct_coefficients},
                "group_id": env_uid,
                "policy_specs": policies,
            }
            results.append(save_params)
        print("max sample path length:", max_length)
        print("average sample path length:", average_sample_path_length / test_sample_path_num)
    # Write to dat file if specified
    if dat_file:
        _split_list_into_groups(results=results, num_groups=num_groups, dat_file=dat_file)

    return results

def generate_train_env(
    test_envs,
    dat_file=None,
    num_init_states=256,
    sample_path_number=256,
    init_state_seed=12345,
    sample_paths_seed=42,
):
    """Generate (X, Y) training-data commands for the tight penalized
    information-relaxation lower bound.

    For each variant in ``test_envs``:
            * X = ``num_init_states`` initial states generated from the environment's
                ``generate_initial_state()`` quasi-Monte Carlo reference distribution
                (daily total bookings use inverse-binomial sampling with
                horizon-decaying occupancy probability; waitlists use inverse sampling
                from the one-period arrival distribution). This emphasizes
                representative congestion levels for training.
      * Y (computed by ``run.py`` when each command runs) = the tight
        penalized information-relaxation lower bound at that initial state,
        obtained by Benders training on a *fixed* set of arrival sample paths.
        We force the same sample paths across all initial states by pinning
        ``arrival_random_seed`` (and the related env seeds) to a constant
        value in every emitted ``env_args``; ``reset_random_seeds()`` is then
        called inside the trainer before sampling, so every initial state
        sees the identical Monte-Carlo set.

    One command (one initial state) is written per line so the workload can be
    farmed out to job-array schedulers.
    """
    results = []
    for (env_uid, experiment_name, mutate_val), variant in test_envs.items():
        init_state_rng = np.random.default_rng(init_state_seed)
        base_env_args = copy.deepcopy(variant['env_args'])
        agent_args = copy.deepcopy(variant.get('agent_args', {}))

        # Pin the arrival/sampling seeds so every initial state for this
        # variant trains against the *same* set of arrival sample paths.
        # NOTE: in experiment_config.py the env's ``init_state_random_seed``
        # is derived from ``env_random_seed``; we therefore use
        # ``env_random_seed`` ONLY for arrival/sampling and override it on a
        # *separate* sampler env (below) when drawing X.
        base_env_args['arrival_random_seed'] = sample_paths_seed
        base_env_args['stop_time_random_seed'] = sample_paths_seed
        base_env_args['env_random_seed'] = sample_paths_seed

        # Build a sampler env with a *different* env_random_seed so the
        # initial-state RNG is independent of the (pinned) sample-path RNGs.
        sampler_env_args = copy.deepcopy(base_env_args)
        sampler_env_args['env_random_seed'] = int(init_state_rng.integers(0, 2**31 - 1))
        sampler_env = get_config_by_type('infinite_custom', args=sampler_env_args).env

        for k in range(num_init_states):
            init_state = sampler_env.generate_initial_state()
            init_state = tuple(np.array(item).tolist() for item in init_state)

            env_args_k = copy.deepcopy(base_env_args)
            env_args_k['reset_params'] = dict(env_args_k.get('reset_params', {}))
            env_args_k['reset_params']['init_state'] = init_state

            save_params = {
                'experiment_name': experiment_name,
                'mutate_val': mutate_val,
                'sample_path_number': sample_path_number,
                'init_state_index': k,
                'init_state': init_state,
                'env_args': env_args_k,
                'agent_args': agent_args,
                'training_generating_function_spec': _normalize_generating_function_spec(
                    agent_args.get('agent_args', {}).get('generating_function_spec')
                ),
            }
            save_params['uid'] = get_uid({
                'env_args': env_args_k,
                'agent_args': agent_args,
                'init_state': init_state,
            })
            results.append(save_params)

    if dat_file:
        lines_to_write = []
        for line_index, result in enumerate(results, start=1):
            lines_to_write.append(
                f"{line_index} python run.py --params '" + json.dumps(result) + "'\n"
            )
        with open(dat_file, 'w') as f:
            f.writelines(lines_to_write)
        print(f"Saved {len(lines_to_write)} commands to {dat_file}")
    return results

def generate_policy_efficientcy_data(
    test_envs,
    is_require_penalty_coefficients=False,
    dat_file=None,
    num_init_states=256,
    sample_path_number=256,
    init_state_seed=12345,
    sample_paths_seed=42,
):
    results = []
    for (env_uid, experiment_name, mutate_val), variant in test_envs.items():
        init_state_rng = np.random.default_rng(init_state_seed)
        base_env_args = copy.deepcopy(variant['env_args'])
        env = get_config_by_type('infinite_custom', args=base_env_args).env
        agent_args = copy.deepcopy(variant.get('agent_args', {}))
        direct_coefficients = _zero_penalty_coefficients(env)
        if is_require_penalty_coefficients:
            agent_args = copy.deepcopy(variant['agent_args'])
            _, direct_coefficients, _ = train_penalty_coefficients(
                env_args=base_env_args,
                agent_args=agent_args,
                experiment_name=experiment_name,
            )
        agent_args['agent_args']['generating_function_spec']['coefficients'] = direct_coefficients
        agent_args['agent_args']['solver_name'] = 'approx_penalized_hindsight'
        # Pin the arrival/sampling seeds so every initial state for this
        # variant trains against the *same* set of arrival sample paths.
        # NOTE: in experiment_config.py the env's ``init_state_random_seed``
        # is derived from ``env_random_seed``; we therefore use
        # ``env_random_seed`` ONLY for arrival/sampling and override it on a
        # *separate* sampler env (below) when drawing X.
        base_env_args['arrival_random_seed'] = sample_paths_seed
        base_env_args['stop_time_random_seed'] = sample_paths_seed
        base_env_args['env_random_seed'] = sample_paths_seed

        # Build a sampler env with a *different* env_random_seed so the
        # initial-state RNG is independent of the (pinned) sample-path RNGs.
        sampler_env_args = copy.deepcopy(base_env_args)
        sampler_env_args['env_random_seed'] = int(init_state_rng.integers(0, 2**31 - 1))
        sampler_env = get_config_by_type('infinite_custom', args=sampler_env_args).env

        for k in range(num_init_states):
            init_state = sampler_env.generate_initial_state()
            init_state = tuple(np.array(item).tolist() for item in init_state)

            env_args_k = copy.deepcopy(base_env_args)
            env_args_k['reset_params'] = dict(env_args_k.get('reset_params', {}))
            env_args_k['reset_params']['init_state'] = init_state

            save_params = {
                'experiment_name': experiment_name,
                'mutate_val': mutate_val,
                'sample_path_number': sample_path_number,
                'init_state_index': k,
                'init_state': init_state,
                'env_args': env_args_k,
                'agent_args': agent_args
            }
            save_params['uid'] = get_uid({
                'env_args': env_args_k,
                'agent_args': agent_args,
                'init_state': init_state,
            })
            results.append(save_params)

    if dat_file:
        lines_to_write = []
        for line_index, result in enumerate(results, start=1):
            lines_to_write.append(
                f"{line_index} python run.py --params '" + json.dumps(result) + "'\n"
            )
        with open(dat_file, 'w') as f:
            f.writelines(lines_to_write)
        print(f"Saved {len(lines_to_write)} commands to {dat_file}")
    return results


if __name__ == '__main__':
    # test_envs = {}
    # experiments = list(EXPERIMENT_SPECS.keys())
    # for experiment_name in experiments:
    #     test_envs.update(build_variation_test_env(EXPERIMENT_SPECS[experiment_name]))
    # test_envs = build_variation_test_env(EXPERIMENT_SPECS['sample_path_length_proposal_fixed'])

    # --- IS training: proposal gamma=0.98, target gamma=0.99 ---
    # Builds env with discount_factor=0.99 and agent with geometric IS proposal 0.98.
    # Calling generate_test_paths_and_init_state with is_require_penalty_coefficients=True
    # will trigger train_penalty_coefficients for each variant, which constructs
    # ApproxQAgent(sample_path_proposal=GeometricLengthProposal(0.98)) so every
    # Benders subproblem objective is weighted by (0.99/0.98)^(t-1).
    # test_envs = build_variation_test_env(EXPERIMENT_SPECS['case_study_099_is_098'])
    # print(test_envs)
    # results = generate_test_paths_and_init_state(
    #     test_envs=test_envs,
    #     test_sample_path_num=1,
    #     warm_up_periods=750,
    #     num_periods=None,
    #     dat_file='table.dat',
    #     num_groups=998,  # divide into N groups
    #     is_require_penalty_coefficients=True,
    #     policy_ids=['approx_penalized_hindsight','row_gen_alp', 'myopic'],
    # )
    test_envs = build_variation_test_env(EXPERIMENT_SPECS['case_study_099_fixed_length'])
    # results = generate_train_env(
    #     test_envs=test_envs,
    #     dat_file='table.dat',
    #     num_init_states=1,
    #     sample_path_number=256,
    #     init_state_seed=12345,
    #     sample_paths_seed=42,
    # )

    results = generate_policy_efficientcy_data(
        test_envs=test_envs,
        is_require_penalty_coefficients=False,
        dat_file='table.dat',
        num_init_states=1,
        sample_path_number=256,
        init_state_seed=12345,
        sample_paths_seed=42,
    )
    '''
    Iteration 65, master solved in 0.057006120681762695 seconds
    Iteration 65, master memory used: 0.0894 GB (peak 0.0894 GB)
    Iteration 65, action from master: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.021504743196445, 10.053358168725675, 7.063043451772355, 6.216953928591073, 5.947031007584353, 5.810820287738708, 0.0, 11.187008919361203, 7.254799818243785, 6.4588507728931335, 6.234492388031058, 6.1830370923598865, 6.152579129839666, 0.0, 19.50692893662462, 8.474659528470424, 6.056375910077659, 5.460441131475307, 5.28643882624245, 5.179953111836649, 0.0, 8.574288536055477, 6.31731618984728, 5.816184723235822, 5.704460757858072, 5.789927111145126, 5.831256131673501, 0.0, 203.11528608052782, 148.07798996576534]
    Iteration 65, subproblems solved in 0.35s                                                                                                                                                                                                                                                                                  
    Iteration 65, master memory used: 0.0894 GB (peak 0.0894 GB)
    Iteration 65, subproblem memory used: 0.4997 GB (peak 0.6339 GB across 256 workers)
    Iteration 65, adding 256 optimality cuts
    UB: 13320.421376476756, LB: 13320.421375945056, Gap: 5.316996976034716e-07, First-stage cost: 0.0, Cost-to-go estimate: 13320.421375945056
    obj=13320.421376476756, elapsed=312.2s
    '''

    '''
    Iteration 117, master solved in 0.018671035766601562 seconds
    Iteration 117, master memory used: 0.0516 GB (peak 0.0516 GB)
    Iteration 117, action from master: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.616730590447517, 10.030054928650797, 7.342823067108933, 6.654157808104911, 6.4464816885102225, 6.410584260999075, 0.0, 11.033619329902345, 7.288276331903743, 6.734505712965286, 6.53726614555766, 6.5283657524539205, 6.5284141456860345, 0.0, 22.3526664371209, 9.8860296843891, 7.1631333979262015, 6.401340865619632, 6.222369897551891, 6.3568993797326305, 0.0, 9.797455963185683, 7.086005954265959, 6.724634542591972, 6.509801728794599, 6.442347525009969, 6.517717763684011, 0.0, 185.20931236956602, -471.225770275197]
    Iteration 117, subproblems solved in 0.20s                                                                                                                                                                                                                                                                                 
    Iteration 117, master memory used: 0.0516 GB (peak 0.0516 GB)
    Iteration 117, subproblem memory used: 0.4575 GB (peak 0.4708 GB across 62 workers)
    Iteration 117, adding 62 optimality cuts
    UB: 13619.654740421676, LB: 13619.654740421674, Gap: 1.8189894035458565e-12, First-stage cost: 0.0, Cost-to-go estimate: 13619.654740421674
    obj=13619.654740421676, elapsed=675.7s
    '''

    '''
    Iteration 59, master solved in 0.03646492958068848 seconds
    Iteration 59, master memory used: 0.0791 GB (peak 0.0791 GB)
    Iteration 59, action from master: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.748031621453524, 9.918235566889901, 6.789947415560918, 5.7070523630518535, 5.233641515655337, 5.025104748457616, 0.0, 11.28160334967443, 7.564451997461522, 6.751135923582664, 6.467225605886129, 6.320329947364923, 6.2789287499572035, 0.0, 16.78486889821841, 7.627737793048174, 5.126671249519612, 3.987258536644841, 3.89802930953473, 3.8762243691548863, 0.0, 7.1606919986247926, 6.065562187121943, 5.362411455521592, 5.315588459573747, 5.469605558957678, 5.77783654895791, 0.0, -872.4827337558502, 169.54391212467092]
    Iteration 59, subproblems solved in 0.19s                                                                                                                                                                                                                                                                                  
    Iteration 59, master memory used: 0.0791 GB (peak 0.0791 GB)
    Iteration 59, subproblem memory used: 0.2732 GB (peak 0.3542 GB across 256 workers)
    Iteration 59, adding 256 optimality cuts
    UB: 12178.941557563008, LB: 12178.941557563005, Gap: 3.637978807091713e-12, First-stage cost: 0.0, Cost-to-go estimate: 12178.941557563005
    obj=12178.941557563008, elapsed=129.1s
    '''



