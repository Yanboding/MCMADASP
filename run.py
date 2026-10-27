import argparse
import glob
import json
import pickle
import os
import time
import re
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint

from generate_params import _save_training_result, _load_cached_training_result
from param_generation.training import train_alp_coefficients
import numpy as np
import pandas as pd
from gurobipy import GRB
from scipy.stats import geom

from experiments.experiment_config import get_config_by_type
from importance_sampling import build_proposal
from importance_sampling.sample_path import SamplePath
from utils import iter_to_tuple, get_uid, safe_open, RunningStats, encode, decode, get_solution_value, acquire_grb_env, read_lines_with_pattern
from decision_maker import MyopicAgent, ALPRowGenerationAgent, ApproxQAgent
from decision_maker.approximate_q_agent import dispose_agent_models
from policy_evaluator import PolicyEvaluator
from policy_evaluator.potential_improvement import arrival_informed_schedule_cost
from generating_function import AbsorptionALPPenaltyFunction, AbsorptionLinearPenaltyFunction, LinearPenaltyFunction
from importance_sampling.sample_path import sample_path_from_record


POLICY_EVALUATION_CACHE_VERSION = 3


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
    if name not in _GENERATING_FUNCTION_CLASSES:
        raise ValueError(f"Unsupported generating function name: {name}; use one of {sorted(_GENERATING_FUNCTION_CLASSES)}")
    generating_function = _GENERATING_FUNCTION_CLASSES[name](env=env, coefficients=coefficients)
    if coefficients is None:
        generating_function.set_coefficients([0.0] * generating_function.number_of_coefficients)
    return generating_function


_GENERATING_FUNCTION_CLASSES = {
    cls.spec_name: cls
    for cls in (LinearPenaltyFunction, AbsorptionLinearPenaltyFunction, AbsorptionALPPenaltyFunction)
}


_GENERATING_FUNCTION_SPEC_KEYS = (
    'generating_function_spec',
    'policy_generating_function_spec',
    'zero_lowerbound_generating_function_spec',
    'penalized_lowerbound_generating_function_spec',
    'training_generating_function_spec',
)


def _set_sample_path_proposal(agent_args):
    spec = agent_args.pop('sample_path_length_proposal', None)
    if 'sample_path_proposal' in agent_args:
        spec = agent_args['sample_path_proposal']
    proposal = build_proposal(spec)
    if proposal is not None:
        agent_args['sample_path_proposal'] = proposal
    else:
        agent_args.pop('sample_path_proposal', None)
    return agent_args


def _resolve_period_weights(period_weights, num_periods):
    if period_weights is None:
        return np.ones(num_periods)
    weights = np.asarray(period_weights, dtype=float)
    if weights.shape[0] < num_periods:
        raise ValueError(
            f"period_weights has {weights.shape[0]} entries but {num_periods} "
            "evaluation periods need weighting."
        )
    return weights[:num_periods]


def jsonl_result_exists(path, uid, policy_id):
    if not os.path.isfile(path):
        return False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('uid') == uid and row.get('policy_id') == policy_id:
                return True
    return False

def jsonl_uid_exists(path, uid):
    if not os.path.isfile(path):
        return False
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('uid') == uid:
                return True
    return False

def partial_relaxation_written(folder, uid):
    key = f'"uid": "{uid}"'
    for path in glob.glob(os.path.join(folder, '*.jsonl')):
        with open(path) as handle:
            for line in handle:
                if key not in line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cost = row.get('partial_information_relaxation_cost')
                if row.get('uid') == uid and isinstance(cost, (int, float)) and np.isfinite(cost):
                    return True
    return False

def load_pickle_if_exists(path):
    if os.path.isfile(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError) as exc:
            # A previous interrupted write can leave an empty/truncated pickle.
            # Remove bad checkpoint/result so the caller can recompute safely.
            print(f"Warning: ignoring corrupted pickle at {path}: {exc}")
            try:
                os.remove(path)
            except OSError:
                pass
            return None
    return None


def atomic_pickle_dump(path, payload):
    tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

def build_policy_agent(env, agent_name, agent_args, grb_env=None, grb_sub_envs=None):
    runtime_agent_args = dict(agent_args)
    policy_generating_function_spec = _normalize_generating_function_spec(
        runtime_agent_args.pop('generating_function_spec', None))
    if agent_name in {"approx_hindsight", "approx_penalized_hindsight"}:
        runtime_agent_args['generating_function'] = _build_generating_function(
            env=env, spec=policy_generating_function_spec)
        _set_sample_path_proposal(runtime_agent_args)
        return ApproxQAgent(env, discount_factor=env.discount_factor, grb_env=grb_env,
                            subproblem_grb_envs=grb_sub_envs, **runtime_agent_args)
    if agent_name == "myopic":
        return MyopicAgent(env, discount_factor=env.discount_factor, grb_env=grb_env, **runtime_agent_args)
    if agent_name == 'row_gen_alp':
        return ALPRowGenerationAgent(env, discount_factor=env.discount_factor, grb_env=grb_env, **runtime_agent_args)
    raise ValueError(f"Unsupported policy: {agent_name}")


def matched_state_decisions(uid, experiment_name, env_args, policy_specs, states, grb_env, grb_sub_envs, job_id):
    output_file = _result_file(experiment_name, job_id)
    pending = [state for state in states if not jsonl_uid_exists(output_file, state['state_uid'])]
    if not pending:
        print(f"Skip existing matched-state batch uid={uid}")
        return []
    env = get_config_by_type(case_type='infinite_custom', args=env_args).env
    agents = {spec['policy_id']: build_policy_agent(env, spec['agent_name'], spec['agent_args'], grb_env, grb_sub_envs)
              for spec in policy_specs}
    records = []
    for state in pending:
        s = tuple(np.asarray(component) for component in state['state'])
        decisions = {}
        for policy_id, agent in agents.items():
            start = time.time()
            _, (bookings, overtime), _ = agent.solve(s, state['period'])
            decisions[policy_id] = {
                'bookings': np.asarray(bookings).astype(int).tolist(),
                'overtime': np.asarray(overtime).astype(int).tolist(),
                'solve_seconds': time.time() - start,
            }
        record = {key: value for key, value in state.items() if key != 'state_uid'}
        record.update({'uid': state['state_uid'], 'policy_id': 'matched_state_decisions', 'batch_uid': uid,
                       'decisions': decisions})
        with open(output_file, 'a') as handle:
            handle.write(json.dumps(record) + '\n')
        records.append(record)
        print(f"Matched state {state['state_uid']}: " + ', '.join(
            f"{policy_id} {decision['solve_seconds']:.1f}s" for policy_id, decision in decisions.items()))
    return records


def calculate_policy_costs_with_penalty(uid,
                           experiment_name,
                           policy_id,
                           agent_name,
                           agent_args,
                           env_args,
                           init_state,
                           sample_path,
                           warm_up_periods,
                           generating_function,
                           grb_env=None,
                           grb_sub_envs=None,
                           period_weights=None,
                           warm_up_trajectory=None,
                           return_warm_up_trajectory=False,
                           terminal=None):
    def _to_float(v):
        if hasattr(v, "getValue"):
            return float(v.getValue())
        return float(v)

    def _to_jsonable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_jsonable(v) for v in value]
        return value

    # Validate the post-warm-up path before any cache/checkpoint lookup. In
    # particular, an old result must not hide missing absorption metadata.
    sample_path = np.asarray(sample_path)
    tail_path = sample_path_from_record(sample_path[warm_up_periods:], period_weights, terminal)
    config = get_config_by_type(case_type='infinite_custom', args=env_args)
    env = config.env
    evaluation_form = generating_function.form('evaluation')
    evaluation_form.period_weights(tail_path, env.discount_factor)
    cost_weights = _resolve_period_weights(tail_path.survival_weights, tail_path.periods)
    if warm_up_trajectory is not None and len(warm_up_trajectory['states']) != warm_up_periods:
        raise ValueError(
            f"warm_up_trajectory has {len(warm_up_trajectory['states'])} periods "
            f"but warm_up_periods={warm_up_periods}."
        )

    base_dir = os.path.join('experiments', 'results', experiment_name, 'pickles')
    os.makedirs(base_dir, exist_ok=True)
    # The supplied uid is not proof that a record is unchanged. Both results
    # and checkpoints depend on the complete rollout and accounting inputs.
    signature_payload = {
        'cache_version': POLICY_EVALUATION_CACHE_VERSION,
        'agent_name': agent_name,
        'agent_args': _to_jsonable({k: v for k, v in dict(agent_args).items() if k != 'grb_env'}),
        'env_args': _to_jsonable(env_args),
        'init_state': _to_jsonable(init_state),
        'sample_path': _to_jsonable(sample_path),
        'warm_up_periods': warm_up_periods,
        'warm_up_trajectory': _to_jsonable(warm_up_trajectory),
        'terminal': tail_path.terminal.value,
        'period_weights': _to_jsonable(cost_weights),
        'penalty_coefficients': _to_jsonable(getattr(generating_function, 'coefficients', None)),
        'penalty_function': getattr(generating_function, 'spec_name', None),
    }
    policy_signature = get_uid(signature_payload)
    checkpoint_file = os.path.join(base_dir, f'{uid}-{policy_id}-{policy_signature}-checkpoint.pickle')
    result_file = os.path.join(base_dir, f'{uid}-{policy_id}-{policy_signature}-result.pickle')

    cached_result = load_pickle_if_exists(result_file)
    # A cached result is only reusable if it already carries the warm-up
    # trajectory when the caller asks for one.
    if cached_result is not None and not (
        return_warm_up_trajectory and 'warm_up_trajectory' not in cached_result
    ):
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        return cached_result

    local_generating_function = generating_function
    agent_instance = build_policy_agent(env, agent_name, agent_args, grb_env, grb_sub_envs)

    states = []
    actions = []
    costs = []
    expected_terms = []
    realized_terms = []
    theta = local_generating_function.coefficient_vector()
    checkpoint = load_pickle_if_exists(checkpoint_file)
    if checkpoint is not None:
        states = checkpoint['states']
        actions = checkpoint['actions']
        costs = checkpoint['costs']
        expected_terms = checkpoint['expected_terms']
        realized_terms = checkpoint['realized_terms']
        t = checkpoint['t']
        s = checkpoint['s']
        solving_time_per_state = checkpoint.get('solving_time_per_state', RunningStats())
        s, _ = env.reset(init_state=s, t=t, new_arrivals=sample_path)
    elif warm_up_trajectory is not None:
        states = [tuple(np.array(component) for component in state) for state in warm_up_trajectory['states']]
        actions = [tuple(np.array(component) for component in action) for action in warm_up_trajectory['actions']]
        costs = list(warm_up_trajectory['costs'])
        expected_terms = list(warm_up_trajectory['expected_terms'])
        realized_terms = list(warm_up_trajectory['realized_terms'])
        t = warm_up_periods + 1
        end_state = tuple(np.array(component) for component in warm_up_trajectory['end_state'])
        s, _ = env.reset(init_state=end_state, t=t, new_arrivals=sample_path)
        solving_time_per_state = RunningStats()
    else:
        t = 1
        s, _ = env.reset(init_state=init_state, t=t, new_arrivals=sample_path)
        solving_time_per_state = RunningStats()
    for tau in range(len(sample_path) - t + 2):
        current_t = t + tau
        start_time = time.time()
        _, action, _ = agent_instance.solve(s, current_t)
        if tau > 0: # Skip the first period because it requires building the model.
            solving_time_per_state += time.time() - start_time
        next_state, cost, done, _ = env.step(action)

        expected_terms.append(_to_float(local_generating_function.expected_value(theta, s, action)))
        realized_term = 0.0
        if current_t <= len(sample_path):
            new_arrivals = sample_path[current_t - 1]
            realized_term = _to_float(local_generating_function.value(theta, s, action, new_arrivals))
        realized_terms.append(realized_term)

        states.append(s)
        actions.append(action)
        costs.append(_to_float(cost))
        s = next_state

        if current_t % 10 == 0:
            atomic_pickle_dump(
                checkpoint_file,
                {
                    's': s,
                    't': current_t + 1,
                    'states': states,
                    'actions': actions,
                    'costs': costs,
                    'expected_terms': expected_terms,
                    'realized_terms': realized_terms,
                    'solving_time_per_state': solving_time_per_state,
                },
            )
        if done:
            break

    atomic_pickle_dump(
        checkpoint_file,
        {
            's': s,
            't': len(costs) + 1,
            'states': states,
            'actions': actions,
            'costs': costs,
            'expected_terms': expected_terms,
            'realized_terms': realized_terms,
            'solving_time_per_state': solving_time_per_state,
        },
    )

    scheduled_patients = []
    overtime = np.zeros(len(sample_path) + env.planning_horizon)
    postponing_decisions = []
    for idx, (state_t, action_t) in enumerate(zip(states, actions)):
        _, _, waitlist = state_t
        advance_scheduling_decision, overtime_decision = action_t
        scheduled_patients.append(advance_scheduling_decision.tolist())
        overtime[idx:idx + len(overtime_decision)] += overtime_decision
        postponing_decisions.append(waitlist - advance_scheduling_decision.sum(axis=0))

    postponing_decisions = np.array(postponing_decisions).sum(axis=0) if postponing_decisions else np.zeros(env.num_types)
    tail_costs = costs[warm_up_periods:]
    total_cost = float(np.dot(cost_weights, tail_costs))
    total_penalty = evaluation_form.combine(
        tail_path, env.discount_factor, expected_terms[warm_up_periods:], realized_terms[warm_up_periods:-1])
    penalized_cost = total_cost + total_penalty
    warmup_state = tuple(np.array(item).tolist() for item in states[warm_up_periods])

    result = {
        'policy_id': policy_id,
        'agent_name': agent_name,
        'agent_args': signature_payload['agent_args'],
        'penalized_cost': penalized_cost,
        'total_cost': total_cost,
        'total_penalty': total_penalty,
        "warmup_state": warmup_state,
        'costs': [float(v) for v in costs],
        'expected_terms': [float(v) for v in expected_terms],
        'realized_terms': [float(v) for v in realized_terms],
        'scheduled_patients': scheduled_patients,
        'overtime': overtime.tolist(),
        'postponing_decisions': postponing_decisions.tolist(),
        'solving_time_per_state': solving_time_per_state.mean,
    }

    if return_warm_up_trajectory:
        result['warm_up_trajectory'] = {
            'states': [_to_jsonable(state) for state in states[:warm_up_periods]],
            'actions': [_to_jsonable(action) for action in actions[:warm_up_periods]],
            'costs': [float(v) for v in costs[:warm_up_periods]],
            'expected_terms': [float(v) for v in expected_terms[:warm_up_periods]],
            'realized_terms': [float(v) for v in realized_terms[:warm_up_periods]],
            'end_state': warmup_state,
        }

    atomic_pickle_dump(result_file, result)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return result

def information_relaxation_bounds(env, generating_function_spec, penalty_ratios, state, sample_path_tail,
                                  period_weights, terminal, grb_env, grb_sub_envs, remove_path_noise=False,
                                  fixed_actions=None, relax_first_decision=False):
    bounds = {}
    for penalty_ratio in penalty_ratios:
        agent = ApproxQAgent(
            env,
            discount_factor=env.discount_factor,
            current_decision_var_type='continuous' if relax_first_decision else 'integer',
            future_decision_var_type='continuous',
            generating_function=_build_generating_function(env=env, spec=generating_function_spec),
            penalty_ratio=penalty_ratio,
            grb_env=grb_env,
            subproblem_grb_envs=grb_sub_envs,
        )
        start = time.time()
        bound = agent.calculate_information_relaxation_cost(
            state, sample_path=sample_path_tail, period_weights=period_weights, terminal=terminal,
            remove_path_noise=remove_path_noise, fixed_actions=fixed_actions)
        print(f"Information relaxation cost at penalty ratio {penalty_ratio:g} computed in "
              f"{time.time() - start:.1f} seconds: {bound}")
        bounds[float(penalty_ratio)] = bound
    return bounds


def estimate_potential_improvement(uid, experiment_name, mutate_val, group_id, env_args, init_state,
                                   arrival_stream, baseline_length, baseline_period_weights, baseline_terminal,
                                   prefix_periods, continuation_length, continuation_period_weights,
                                   path_weight, path_stratum, generating_function_spec, alp_coefficients,
                                   grb_env, grb_sub_envs, job_id):
    output_file = _result_file(experiment_name, job_id)
    if jsonl_uid_exists(output_file, uid):
        print(f"Skip existing potential-improvement replication uid={uid}")
        return None
    start = time.time()
    env = get_config_by_type(case_type='infinite_custom', args=env_args).env
    init_state = tuple(np.array(component) for component in init_state)
    stream = np.array(arrival_stream)
    penalized_cost = float(information_relaxation_bounds(
        env, generating_function_spec, [1.0], init_state, stream[:baseline_length], baseline_period_weights,
        baseline_terminal, grb_env, grb_sub_envs, remove_path_noise=True)[1.0])
    alp_agent = ALPRowGenerationAgent(env, discount_factor=env.discount_factor,
                                      coefficients=alp_coefficients, grb_env=grb_env)
    relaxation_agent = ApproxQAgent(
        env,
        discount_factor=env.discount_factor,
        current_decision_var_type='integer',
        future_decision_var_type='integer',
        generating_function=_build_generating_function(env=env, spec=generating_function_spec),
        penalty_ratio=0.0,
        grb_env=grb_env,
        subproblem_grb_envs=grb_sub_envs,
    )
    schedule = arrival_informed_schedule_cost(env, alp_agent, relaxation_agent, init_state, stream,
                                              prefix_periods, continuation_length, continuation_period_weights,
                                              action_periods=baseline_length + 1)
    schedule_penalized_cost = float(information_relaxation_bounds(
        env, generating_function_spec, [1.0], init_state, stream[:baseline_length], baseline_period_weights,
        baseline_terminal, grb_env, grb_sub_envs, remove_path_noise=True,
        fixed_actions=schedule['actions'][:baseline_length + 1])[1.0])
    record = {
        'uid': uid,
        'group_id': group_id,
        'penalized_information_relaxation_cost': penalized_cost,
        'partial_information_relaxation_cost': schedule['schedule_cost'],
        'potential_gap': schedule['schedule_cost'] - penalized_cost,
        'prefix_partial_information_relaxation_cost': schedule['prefix_cost'],
        'tail_error_bound': schedule['continuation_cost'],
        'schedule_penalized_information_relaxation_cost': schedule_penalized_cost,
        'run_time_seconds': time.time() - start,
        'path_weight': path_weight,
        'path_stratum': path_stratum,
    }
    with open(output_file, 'a') as handle:
        handle.write(json.dumps(record) + '\n')
    print(f"Potential improvement uid={uid}: partial {record['partial_information_relaxation_cost']:.4f}, "
          f"penalized {penalized_cost:.4f}, gap {record['potential_gap']:.4f}")
    return record


def partial_information_relaxation_for_path(uid, source_uid, experiment_name, mutate_val, group_id, env_args,
                                            init_state, arrival_stream, path_weight, path_stratum, grb_env,
                                            grb_sub_envs, job_id, policy_ids=None,
                                            future_decision_var_type='continuous'):
    output_file = _result_file(experiment_name, job_id)
    if partial_relaxation_written(os.path.dirname(output_file), uid):
        print(f"Skip existing partial-relaxation path uid={uid}")
        return None
    start = time.time()
    env = get_config_by_type(case_type='infinite_custom', args=env_args).env
    stream = np.array(arrival_stream)
    period_weights = (env.discount_factor ** np.arange(len(stream) + 1)).tolist()
    state = tuple(np.array(component) for component in init_state)
    integer_fields = {}
    if future_decision_var_type == 'integer':
        agent = ApproxQAgent(env, discount_factor=env.discount_factor, current_decision_var_type='integer',
                             future_decision_var_type='integer',
                             generating_function=_build_generating_function(env=env, spec={'name': 'absorption_alp_penalty'}),
                             penalty_ratio=0.0, grb_env=grb_env, subproblem_grb_envs=grb_sub_envs)
        solve = agent.information_relaxation_bound(state, stream, period_weights, 'absorbed')
        cost = solve['bound']
        integer_fields = {'partial_information_relaxation_objective': solve['objective'], 'mip_gap': solve['gap'],
                          'solver_status': solve['status'], 'future_decision_var_type': 'integer'}
    else:
        cost = information_relaxation_bounds(
            env, {'name': 'absorption_alp_penalty'}, [0.0], state, stream, period_weights, 'absorbed', grb_env,
            grb_sub_envs)[0.0]
    record = {
        'uid': uid,
        'source_uid': source_uid,
        'policy_ids': policy_ids,
        'experiment_name': experiment_name,
        'group_id': group_id,
        'mutate_val': mutate_val,
        'policy_id': 'partial_information_relaxation',
        'partial_information_relaxation_cost': float(cost),
        **integer_fields,
        'horizon': len(period_weights),
        'path_weight': path_weight,
        'path_stratum': path_stratum,
        'run_time_seconds': time.time() - start,
    }
    _append_jsonl_record(output_file, record)
    print(f"Partial information relaxation uid={uid}: {cost:.4f} over {len(period_weights)} periods "
          f"in {record['run_time_seconds']:.1f} seconds")
    return record


def _result_file(experiment_name, job_id):
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    return output_file


def _append_jsonl_record(output_file, record):
    if jsonl_result_exists(output_file, record['uid'], record['policy_id']):
        print(f"Skip saving duplicate result: uid={record['uid']}, policy_id={record['policy_id']}")
        return
    with open(output_file, 'a') as f:
        f.write(json.dumps(record) + '\n')


def information_relaxation_cost_fields(bounds):
    if bounds is None:
        return {'zero_information_relaxation_cost': None, 'penalized_information_relaxation_cost': None}
    return {'zero_information_relaxation_cost': float(bounds[0.0]),
            'penalized_information_relaxation_cost': float(bounds[1.0])}


def approximate_value_fields(generating_function, init_state):
    try:
        value = generating_function.approximate_value(init_state)
    except NotImplementedError:
        return {}
    return {'approximate_value_at_init_state': float(value)}


def policy_gap_fields(policy_result, bounds):
    if bounds is None:
        return {'gap_to_zero_information_relaxation': None, 'gap_to_penalized_information_relaxation': None}
    return {'gap_to_zero_information_relaxation': float(policy_result['total_cost'] - bounds[0.0]),
            'gap_to_penalized_information_relaxation': float(policy_result['penalized_cost'] - bounds[1.0])}


def _order_policy_specs_for_warm_up(policy_specs, warm_up_policy_id):
    if warm_up_policy_id is None:
        return list(policy_specs)
    warm_up_specs = [spec for spec in policy_specs if spec['policy_id'] == warm_up_policy_id]
    if not warm_up_specs:
        raise ValueError(
            f"warm_up_policy_id '{warm_up_policy_id}' is not among the evaluated "
            f"policies {[spec['policy_id'] for spec in policy_specs]}."
        )
    return warm_up_specs + [spec for spec in policy_specs if spec['policy_id'] != warm_up_policy_id]


def _alp_policy_cost_and_gradient(env, alp_spec, init_state, sample_path_tail, period_weights, terminal,
                                  grb_env, grb_sub_envs):
    centering_agent = ApproxQAgent(
        env, discount_factor=env.discount_factor,
        current_decision_var_type='integer', future_decision_var_type='continuous',
        generating_function=_build_generating_function(env=env, spec=alp_spec),
        penalty_ratio=1.0, grb_env=grb_env, subproblem_grb_envs=grb_sub_envs)
    alp_agent = ALPRowGenerationAgent(env, discount_factor=env.discount_factor,
                                      coefficients=np.asarray(alp_spec['coefficients'], dtype=float), grb_env=grb_env)
    path = SamplePath(sample_path_tail, terminal,
                      None if period_weights is None else period_weights[:len(sample_path_tail) + 1], None)
    actions = centering_agent._rollout_actions(alp_agent, init_state, path)
    return centering_agent.policy_cost_and_gradient(
        init_state, sample_path_tail, period_weights=period_weights, terminal=terminal,
        fixed_actions=actions)


def _policy_centered_relaxation_fields(env, generating_function_spec, init_state, sample_path_tail,
                                       period_weights, terminal, penalized, grb_env, grb_sub_envs):
    penalized_policy_cost, gradient = _alp_policy_cost_and_gradient(
        env, generating_function_spec, init_state, sample_path_tail, period_weights, terminal,
        grb_env, grb_sub_envs)
    policy_penalty = float(gradient @ np.asarray(generating_function_spec['coefficients'], dtype=float))
    return {
        'policy_centered_information_relaxation_cost': float(penalized) - policy_penalty,
        'alp_policy_cost': float(penalized_policy_cost) - policy_penalty,
    }


def _centered_relaxation_fields(env, generating_function_spec, alp_coefficients, init_state,
                                sample_path_tail, period_weights, terminal, uncentered_fitted,
                                grb_env, grb_sub_envs, relax_first_decision=False):
    # The centering term vanishes at the anchor, so the centered ALP value is the plain ALP
    # relaxation value. Neither uses remove_path_noise: centering already removes the policy's
    # sampled penalty variation, and stacking both would subtract the action-independent part
    # under two different coefficients.
    anchor = np.asarray(alp_coefficients, dtype=float)
    fitted = np.asarray(generating_function_spec.get('coefficients', anchor), dtype=float)
    alp_spec = dict(generating_function_spec)
    alp_spec['coefficients'] = anchor.tolist()
    _, gradient = _alp_policy_cost_and_gradient(
        env, alp_spec, init_state, sample_path_tail, period_weights, terminal, grb_env, grb_sub_envs)
    centered_alp = float(information_relaxation_bounds(
        env, alp_spec, [1.0], init_state, sample_path_tail, period_weights, terminal,
        grb_env, grb_sub_envs, relax_first_decision=relax_first_decision)[1.0])
    centered_fitted = float(uncentered_fitted) - float(gradient @ (fitted - anchor))
    return {
        'centered_fitted_information_relaxation_cost': centered_fitted,
        'centered_alp_information_relaxation_cost': centered_alp,
        'center_potential_gap': centered_fitted - centered_alp,
    }


def evaluate_policy_costs_with_information_relaxation(uid,
                                                      experiment_name,
                                                      mutate_val,
                                                      init_state,
                                                      sample_path,
                                                      warm_up_periods,
                                                      env_args,
                                                      policy_specs,
                                                      group_id,
                                                      grb_env,
                                                      grb_sub_envs,
                                                      job_id,
                                                      generating_function_spec,
                                                      period_weights=None,
                                                      path_weight=None,
                                                      path_stratum=None,
                                                      warm_up_policy_id=None,
                                                      skip_information_relaxation=False,
                                                      penalty_ratios=None,
                                                      coefficients_source=None,
                                                      remove_path_noise=False,
                                                      alp_coefficients=None,
                                                      policy_centering=False,
                                                      terminal=None,
                                                      relax_first_decision=False):
    init_state = tuple(np.array(item) for item in init_state)
    sample_path = np.array(sample_path)
    sample_path_tail = sample_path[warm_up_periods:]

    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    env = get_config_by_type(case_type='infinite_custom', args=env_args).env
    generating_function_spec = _normalize_generating_function_spec(generating_function_spec)
    policy_costs_generating_function = _build_generating_function(env=env, spec=generating_function_spec)
    ratios = [0.0, 1.0] if penalty_ratios is None else sorted({0.0, 1.0, *map(float, penalty_ratios)})
    base_record = {
        'uid': uid,
        'group_id': group_id,
        'experiment_name': experiment_name,
        'mutate_val': mutate_val,
        'warm_up_periods': warm_up_periods,
        'path_weight': path_weight,
        'path_stratum': path_stratum,
    }
    if relax_first_decision:
        base_record['relax_first_decision'] = True
        base_record['information_relaxation_decision_types'] = {
            'current': 'continuous', 'future': 'continuous'}
    ordered_policy_specs = _order_policy_specs_for_warm_up(policy_specs, warm_up_policy_id)
    if policy_centering and ordered_policy_specs:
        raise ValueError('policy_centering centers the information-relaxation-only bound; the record '
                         f'also evaluates policies {[spec["policy_id"] for spec in ordered_policy_specs]}')

    if not ordered_policy_specs:
        bounds = None if skip_information_relaxation else information_relaxation_bounds(
            env, generating_function_spec, ratios, init_state, sample_path_tail, period_weights, terminal,
            grb_env, grb_sub_envs, remove_path_noise, relax_first_decision=relax_first_decision)
        centered = {}
        if policy_centering and bounds is not None:
            if remove_path_noise:
                raise ValueError('policy_centering and remove_path_noise cannot be combined: the policy '
                                 'centering already removes the action-independent penalty terms')
            centered = _policy_centered_relaxation_fields(
                env, generating_function_spec, init_state, sample_path_tail, period_weights, terminal,
                bounds[1.0], grb_env, grb_sub_envs)
        if alp_coefficients is not None and bounds is not None:
            if remove_path_noise:
                raise ValueError('alp_coefficients and remove_path_noise cannot be combined: '
                                 'policy cost centering already removes the sampled penalty')
            centered = _centered_relaxation_fields(
                env, generating_function_spec, alp_coefficients, init_state, sample_path_tail,
                period_weights, terminal, bounds[1.0], grb_env, grb_sub_envs,
                relax_first_decision=relax_first_decision)
        record = {
            **base_record,
            'policy_id': 'information_relaxation_only',
            'agent_name': 'information_relaxation_only',
            **information_relaxation_cost_fields(bounds),
            **centered,
            **approximate_value_fields(policy_costs_generating_function, init_state),
            'gap_to_zero_information_relaxation': 0.0,
            'gap_to_penalized_information_relaxation': 0.0,
            'warmup_state': tuple(np.array(item).tolist() for item in init_state),
        }
        if penalty_ratios is not None:
            record['penalty_ratios'] = ratios
            record['information_relaxation_cost_by_penalty_ratio'] = (
                None if bounds is None else [[t, float(bounds[t])] for t in ratios])
        if penalty_ratios is not None or coefficients_source is not None:
            record['coefficients_source'] = coefficients_source
        _append_jsonl_record(output_file, record)
        return [record]

    summary_rows = []
    shared_warm_up_trajectory = None
    bounds_by_state = {}
    for policy_spec in ordered_policy_specs:
        is_warm_up_policy = warm_up_policy_id is not None and policy_spec['policy_id'] == warm_up_policy_id
        policy_result = calculate_policy_costs_with_penalty(
            uid=uid,
            experiment_name=experiment_name,
            policy_id=policy_spec['policy_id'],
            agent_name=policy_spec['agent_name'],
            agent_args=policy_spec['agent_args'],
            env_args=env_args,
            init_state=init_state,
            sample_path=sample_path,
            warm_up_periods=warm_up_periods,
            generating_function=policy_costs_generating_function,
            grb_env=grb_env,
            grb_sub_envs=grb_sub_envs,
            period_weights=period_weights,
            warm_up_trajectory=shared_warm_up_trajectory,
            return_warm_up_trajectory=is_warm_up_policy,
            terminal=terminal,
        )
        if is_warm_up_policy:
            shared_warm_up_trajectory = policy_result.pop('warm_up_trajectory')

        warmup_state = tuple(np.array(item) for item in policy_result.get('warmup_state', init_state))
        state_key = iter_to_tuple(warmup_state)
        if not skip_information_relaxation and state_key not in bounds_by_state:
            bounds_by_state[state_key] = information_relaxation_bounds(
                env, generating_function_spec, ratios, warmup_state, sample_path_tail, period_weights, terminal,
                grb_env, grb_sub_envs, remove_path_noise, relax_first_decision=relax_first_decision)
        bounds = bounds_by_state.get(state_key)
        policy_result.update({
            **base_record,
            **information_relaxation_cost_fields(bounds),
            **policy_gap_fields(policy_result, bounds),
        })
        _append_jsonl_record(output_file, policy_result)

        summary_rows.append({key: policy_result[key] for key in (
            'policy_id',
            'agent_name',
            'penalized_cost',
            'total_cost',
            'total_penalty',
            'zero_information_relaxation_cost',
            'penalized_information_relaxation_cost',
            'gap_to_zero_information_relaxation',
            'gap_to_penalized_information_relaxation',
        )})

    return summary_rows

def _draw_per_scenario_init_states(env_args, init_state_seed, sample_path_number, paths_per_state=None, strata=None):
    state_number = sample_path_number
    if paths_per_state is not None:
        if paths_per_state < 1 or sample_path_number % paths_per_state:
            raise ValueError(f"paths_per_state must divide sample_path_number ({sample_path_number}); got {paths_per_state}")
        state_number = sample_path_number // paths_per_state
        if strata is None:
            raise ValueError('paths_per_state needs the sample-path strata to balance the states')
        strata = np.asarray(strata).reshape(-1)
        labels, counts = np.unique(strata, return_counts=True)
        unbalanced = {str(label): int(count) for label, count in zip(labels, counts) if count % state_number}
        if unbalanced:
            raise ValueError(
                f"every stratum must split evenly over the {state_number} initial states so that each of them "
                f"draws the same mix of sample paths; stratum sizes {unbalanced} do not")
    sampler_env_args = copy.deepcopy(env_args)
    sampler_env_args['env_random_seed'] = init_state_seed
    sampler_env = get_config_by_type('infinite_custom', args=sampler_env_args).env
    sampler_env.reset_random_seeds()
    states = [
        tuple(np.array(component) for component in sampler_env.generate_initial_state())
        for _ in range(state_number)
    ]
    if paths_per_state is None:
        return states
    drawn = {}
    assignment = []
    for stratum in strata:
        position = drawn.get(stratum, 0)
        assignment.append(states[position % state_number])
        drawn[stratum] = position + 1
    return assignment

def _by_coefficient_index(json_mapping):
    return {int(index): float(value) for index, value in (json_mapping or {}).items()}


def train_penalty_coefficients_for_env(
    uid,
    experiment_name,
    mutate_val,
    sample_path_number,
    init_state_mode,
    init_state,
    init_state_seed,
    env_args,
    agent_args,
    training_generating_function_spec,
    grb_env,
    grb_sub_envs,
    job_id,
):
    solver_choice = os.environ.get('PENALTY_TRAIN_SOLVER', 'benders').strip().lower()
    if solver_choice != 'benders':
        raise ValueError(
            f"Unsupported PENALTY_TRAIN_SOLVER={solver_choice!r}; "
            "only Benders training is implemented. Use PENALTY_TRAIN_SOLVER=benders."
        )
    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if jsonl_uid_exists(output_file, uid):
        print(f"Skip duplicate penalty-training result: uid={uid}")
        return None

    config = get_config_by_type('infinite_custom', args=env_args)
    env = config.env

    training_generating_function_spec = _normalize_generating_function_spec(
        training_generating_function_spec
    )
    training_generating_function_spec.pop('coefficients', None)
    generating_function = _build_generating_function(
        env=env,
        spec=training_generating_function_spec,
    )

    inner = dict((agent_args or {}).get('agent_args', {}))
    # Path output is handled here; only solver options reach the training method.
    save_path_certificates = bool(inner.pop('save_path_certificates', False))
    training_options = {key: inner.pop(key) for key in (
        'max_iterations', 'absolute_gap_tolerance',
        'master_time_limit_seconds',
        'subproblem_time_limit_seconds') if key in inner}
    inner.pop('workers', None)  # the shared pool is allocated once by main()
    for spec_key in _GENERATING_FUNCTION_SPEC_KEYS:
        inner.pop(spec_key, None)
    inner['generating_function'] = generating_function
    _set_sample_path_proposal(inner)
    inner['sample_path_number'] = sample_path_number
    coefficient_bound = inner.pop('coefficient_bound', None)
    coefficient_bound = GRB.INFINITY if coefficient_bound is None else float(coefficient_bound)
    initial_coefficients = inner.pop('initial_coefficients', None)
    initial_coefficients_source = inner.pop('initial_coefficients_source', None)
    fixed_coefficients = inner.pop('fixed_coefficients', None)
    min_norm_slack = float(inner.pop('min_norm_slack', 0.0) or 0.0)
    min_norm_every_iteration = bool(inner.pop('min_norm_every_iteration', False))
    alp_value_floor = bool(inner.pop('alp_value_floor', False))
    alp_value_floor_tolerance = float(inner.pop('alp_value_floor_tolerance', 1e-6) or 1e-6)
    coefficient_bound_overrides = inner.pop('coefficient_bound_overrides', None)
    paths_per_state = inner.pop('paths_per_state', None)
    paths_per_state = None if paths_per_state is None else int(paths_per_state)
    policy_centering_enabled = bool(inner.pop('policy_centering', False))
    if save_path_certificates and not policy_centering_enabled:
        raise ValueError('save_path_certificates requires policy_centering for the fixed ALP reference')
    policy_cap_scenarios = int(inner.pop('policy_cap_scenarios', 64) or 64)
    cap_policy = inner.pop('cap_policy', None) or 'penalized_hindsight'
    if cap_policy not in ('alp', 'penalized_hindsight'):
        raise ValueError(f"cap_policy must be 'alp' or 'penalized_hindsight'; got {cap_policy!r}")
    if paths_per_state is not None and init_state_mode != 'generate':
        raise ValueError(f"paths_per_state applies to init_state_mode 'generate' only; got {init_state_mode!r}")
    if initial_coefficients is not None and len(initial_coefficients) != generating_function.number_of_coefficients:
        raise ValueError(
            f"initial_coefficients has {len(initial_coefficients)} entries but "
            f"{training_generating_function_spec['name']} has {generating_function.number_of_coefficients}")

    agent = ApproxQAgent(
        env=env,
        discount_factor=env.discount_factor,
        grb_env=grb_env,
        subproblem_grb_envs=grb_sub_envs,
        **inner,
    )

    if init_state_mode == 'generate':
        resolved_init_state = _draw_per_scenario_init_states(
            env_args=env_args,
            init_state_seed=init_state_seed,
            sample_path_number=sample_path_number,
            paths_per_state=paths_per_state,
            strata=None if paths_per_state is None else agent.sample_path_strata,
        )
    elif init_state_mode == 'per_scenario':
        resolved_init_state = [
            tuple(np.array(component) for component in state) for state in init_state
        ]
    elif init_state_mode == 'shared':
        resolved_init_state = tuple(np.array(component) for component in init_state)
    else:
        raise ValueError(f"Unknown init_state_mode: {init_state_mode!r}")

    # Reset to this command's seeds so policy evaluation and all three penalties
    # use the same paths; repeated commands supply different sampling seeds.
    env.reset_random_seeds()

    checkpoint_dir = os.path.join(os.path.dirname(output_file), 'benders_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    policy_centering = None
    if policy_centering_enabled:
        if initial_coefficients is None:
            raise ValueError(
                'policy_centering needs initial_coefficients: they anchor both the penalty and the policy')
        cap_start = time.time()
        if training_generating_function_spec['name'] != 'absorption_alp_penalty':
            raise ValueError(
                "policy_centering needs --penalty-function absorption_alp_penalty: ALPRowGenerationAgent reads the "
                f"coefficient vector as [W_0, U, V, W], which does not match "
                f"{training_generating_function_spec['name']!r}")
        if cap_policy == 'alp':
            policy_agent = ALPRowGenerationAgent(
                env, discount_factor=env.discount_factor, grb_env=grb_env,
                coefficients=initial_coefficients)
        else:
            policy_args = dict(inner)
            policy_args['sample_path_number'] = int(policy_cap_scenarios)
            # inner already holds a built proposal object, so rebuild from the raw spec:
            # the training proposal stratifies for sample_path_number paths and the policy
            # draws far fewer, which cannot fill that many strata.
            raw_proposal = ((agent_args or {}).get('agent_args') or {}).get('sample_path_length_proposal')
            if isinstance(raw_proposal, dict) and 'num_strata' in raw_proposal:
                raw_proposal = dict(raw_proposal)
                raw_proposal['num_strata'] = max(1, min(int(raw_proposal['num_strata']),
                                                        int(policy_cap_scenarios) // 2))
                policy_args['sample_path_proposal'] = build_proposal(raw_proposal)
            policy_args['generating_function'] = _build_generating_function(
                env=env, spec=dict(training_generating_function_spec), coefficients=initial_coefficients)
            policy_args['solver_name'] = 'approx_penalized_hindsight'
            policy_agent = ApproxQAgent(
                env=env, discount_factor=env.discount_factor, grb_env=grb_env,
                subproblem_grb_envs=grb_sub_envs, **policy_args)
        cache_path = os.path.join(checkpoint_dir, f'{uid}-policy-centering.npz')
        cached = np.load(cache_path) if os.path.exists(cache_path) else None
        if cached is not None and cached['costs'].shape == (sample_path_number,):
            policy_centering = (cached['costs'], cached['gradients'], cached['anchor'])
            print(f"  reusing cached policy centering from {cache_path}")
        else:
            policy_centering = agent.policy_cost_centering(
                policy_agent, init_state=resolved_init_state, coefficients=initial_coefficients)
            np.savez(cache_path, costs=policy_centering[0], gradients=policy_centering[1],
                     anchor=policy_centering[2])
        # The policy models are dead weight once the centering data exists; releasing them
        # keeps them out of memory while training holds its own subproblems.
        dispose_agent_models(policy_agent)
        env.reset_random_seeds()
        costs = policy_centering[0]
        print(f"  policy centering ready in {time.time() - cap_start:.1f}s: "
              f"mean policy cost {float(np.mean(costs)):.4f}, min {float(np.min(costs)):.4f}")

    print(
        f"Training penalty coefficients for uid={uid}, mode={init_state_mode}, "
        f"sample_path_number={sample_path_number}"
    )
    start = time.time()
    checkpoint_path = os.path.join(
        checkpoint_dir, f'{uid}-penalty-checkpoint.pickle'
    )
    try:
        obj, coefficients, info = agent.benders_decomposition_train(
            coefficient_bound=coefficient_bound,
            init_state=resolved_init_state,
            checkpoint_path=checkpoint_path,
            resume_checkpoint_path=checkpoint_path,
            initial_coefficients=initial_coefficients,
            fixed_coefficients=_by_coefficient_index(fixed_coefficients),
            min_norm_slack=min_norm_slack,
            min_norm_every_iteration=min_norm_every_iteration,
            alp_value_floor=alp_value_floor,
            alp_value_floor_tolerance=alp_value_floor_tolerance,
            coefficient_bound_overrides=_by_coefficient_index(coefficient_bound_overrides),
            policy_centering=policy_centering,
            **training_options,
        )
        path_results = {}
        if save_path_certificates:
            from experiments.po_certificates import path_record

            _, gradients, anchor = policy_centering
            gradients, anchor = np.asarray(gradients, dtype=float), np.asarray(anchor, dtype=float)
            fitted = np.asarray(coefficients, dtype=float)
            if anchor.shape != fitted.shape or not np.isfinite(anchor).all():
                raise ValueError('policy_centering anchor must match fitted coefficients')
            # The final training evaluation already solved every path at the fitted
            # coefficients. Snapshot it before reusing the workers for the two baselines.
            centered = [copy.deepcopy(worker.last_certificate)
                        for worker in agent.coefficient_model.workers[:sample_path_number]]
            alp_initial_cuts = [agent.subproblem_initial_cuts.get(sid) for sid in range(sample_path_number)]
            raw_by_name = {}
            for name, theta in [('fitted', fitted), ('alp', anchor), ('zero', np.zeros_like(fitted))]:
                if name == 'fitted':
                    certificates = copy.deepcopy(centered)
                elif name == 'alp' and all(
                        cut is not None and np.isfinite(cut[0]) and np.array_equal(cut[2], anchor)
                        for cut in alp_initial_cuts):
                    # Initial cuts exist only for OPTIMAL cold solves; at the anchor,
                    # their centered objective also equals the raw ALP baseline.
                    certificates = [dict(objective=value, lower_bound=value, upper_bound=value,
                                         status='OPTIMAL') for value, _, _ in alp_initial_cuts]
                else:
                    _, _, certificates = agent.coefficient_model.evaluate_action_certificates(theta, parallel=True)
                # Workers solve the control-variate objective. Restore raw IR costs
                # and their MIN bounds before computing the paired path differences.
                shift = gradients @ (theta - anchor)
                for sid, certificate in enumerate(certificates):
                    for field in ('objective', 'lower_bound', 'upper_bound'):
                        if certificate[field] is not None:
                            certificate[field] += float(shift[sid])
                raw_by_name[name] = certificates
            weights = np.asarray(agent.sample_path_weights, dtype=float)

            def weighted(field):
                if any(row[field] is None for row in centered):
                    return None
                return float(weights @ [row[field] for row in centered])

            lower = weighted('lower_bound') if all(row['bound_valid'] for row in centered) else None
            certificate = copy.deepcopy(info.get('master_certificate'))
            if certificate is None:
                certificate = dict(sense='maximize', status='NOT_SOLVED', objective=None,
                                   objective_bound=None, valid=False, reason='No unrestricted primary MAX solve')
            upper = certificate['objective_bound'] if certificate['valid'] else None
            if upper is not None and lower is not None and lower > upper + 1e-6 * max(1., abs(upper), abs(lower)):
                raise RuntimeError('Certified candidate lower bound exceeds primary master upper bound')
            policy_control_variates = gradients @ (fitted - anchor)
            reference_policy_penalties = gradients @ anchor
            paths = [path_record(sid, path, agent.sample_path_strata[sid], weights[sid],
                        policy_control_variates[sid], reference_policy_penalties[sid],
                        {name: rows[sid] for name, rows in raw_by_name.items()})
                     for sid, path in enumerate(agent.sample_paths)]
            path_results = dict(paths=paths, full_objective=weighted('objective'),
                                objective_lower_bound=lower, objective_upper_bound=upper,
                                master_status=certificate['status'], termination_reason=info['termination_reason'])
    except Exception:
        dispose_agent_models(agent)
        raise
    elapsed = time.time() - start
    print(f"  solver={solver_choice}, obj={obj}, elapsed={elapsed:.1f}s")

    if hasattr(coefficients, 'tolist'):
        coefficients_jsonable = coefficients.tolist()
    else:
        coefficients_jsonable = list(coefficients)

    record = {
        'uid': uid,
        'experiment_name': experiment_name,
        'mutate_val': mutate_val,
        'sample_path_number': sample_path_number,
        'tight_penalized_lower_bound': float(obj),
        'coefficients': coefficients_jsonable,
        'training_time_seconds': elapsed,
    }
    if not save_path_certificates:
        # Ordinary training keeps its existing summary; repeated PO saves the
        # path data needed to recompute statistics instead of duplicating them.
        record.update({
            'init_state_mode': init_state_mode,
            'init_state_seed': init_state_seed,
            'coefficient_bound': None if coefficient_bound == GRB.INFINITY else coefficient_bound,
            'coefficient_bound_overrides': coefficient_bound_overrides,
            'policy_centering': policy_centering_enabled,
            'cap_policy': cap_policy if policy_centering_enabled else None,
            'policy_cap_scenarios': policy_cap_scenarios if policy_centering_enabled and cap_policy != 'alp' else None,
            'policy_centering_info': info.get('policy_centering'),
            'paths_per_state': paths_per_state,
            'min_norm': info.get('min_norm'),
            'min_norm_every_iteration': min_norm_every_iteration,
            'alp_value_floor': alp_value_floor,
            'alp_value_floor_info': info.get('alp_value_floor'),
            'initial_coefficients_source': initial_coefficients_source,
            'fixed_coefficients': fixed_coefficients,
            'initial_in_sample': info.get('initial_in_sample'),
            'improvement': info.get('improvement'),
            'in_sample': info['in_sample'],
        })
    dispose_agent_models(agent)
    record.update(path_results)
    with open(output_file, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record

def train_alp_coefficients_for_env(uid, experiment_name, env_args, grb_env, grb_sub_envs, job_id):
    # The training cache is the single ALP output, reused across job IDs.
    return train_alp_coefficients(env_args=env_args, experiment_name=experiment_name)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Example of using argparse to pass in a list of lists.")
    parser.add_argument('--params', help='Input JSON-encoded list of lists', type=str)
    parser.add_argument('--params_file', help='Path to JSON file containing params payload', type=str)
    parser.add_argument('--job_id', help='Input METAJOB_ID', type=str)
    args = parser.parse_args(argv)
    if args.params_file:
        with open(args.params_file, 'r') as f:
            params = json.load(f)
    elif args.params:
        params = json.loads(args.params)
    else:
        raise ValueError('Either --params or --params_file must be provided.')

    if isinstance(params, dict):
        params = [params]
    if not isinstance(params, list) or not params or any(not isinstance(p, dict) for p in params):
        raise ValueError('Params must be a record or a nonempty list of records.')
    if any('run_type' in p for p in params):
        raise ValueError('Use ordinary train records with init_state_mode; run_type is unsupported.')
    # Dynamically allocate a pool of Gurobi tokens (envs) for the subproblems.
    # A Gurobi env is NOT thread-safe for concurrent optimization, so the
    # minimum number of tokens needed equals the number of subproblems solved
    # concurrently, i.e. the number of available CPUs. Subproblems are then
    # partitioned across this pool and the ones sharing an env are solved
    # sequentially (see SubproblemWorker grouping in the Benders solver), so we
    # only hold `min(sample_path_number, num_cpus)` tokens instead of one per
    # subproblem.
    def _sample_path_number_for_param(param):
        for spec in param.get('policy_specs', []) or []:
            n = spec.get('agent_args', {}).get('sample_path_number', 0)
            if n:
                return n
        n = param.get('sample_path_number', 0)
        if n:
            return n
        return (
            param.get('agent_args', {})
            .get('agent_args', {})
            .get('sample_path_number', 0)
        )
    sample_path_number = max(
                (_sample_path_number_for_param(param) for param in params),
                default=0,
            )
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    try:
        num_cpus = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    except ValueError:
        num_cpus = os.cpu_count() or 1
    budget = os.environ.get('MCMADASP_CPU_BUDGET')
    if budget is not None:
        if int(budget) < 1:
            raise ValueError('MCMADASP_CPU_BUDGET must be positive')
        num_cpus = min(num_cpus, int(budget))
    requested_workers = [((p.get('agent_args') or {}).get('agent_args') or {}).get('workers', num_cpus)
                         for p in params]
    if any(isinstance(n, bool) or int(n) != n or n < 1 for n in requested_workers):
        raise ValueError('workers must be a positive integer')
    num_sub_envs = min(sample_path_number, num_cpus, max(requested_workers)) if sample_path_number else 0
    grb_env = acquire_grb_env({"Threads": 1}, verbose=False, wait=15)
    grb_sub_envs = []
    try:
        for _ in range(num_sub_envs):
            grb_sub_envs.append(acquire_grb_env({"Threads": 1}, verbose=False, wait=15))
        print('grb_sub_envs:',len(grb_sub_envs))
        for param in params:
            # Dispatch on the shape of the params record:
            #   * generate_test_paths_and_init_state -> contains 'policy_specs'
            #     -> evaluate policy costs against information-relaxation bounds.
            #   * generate_penalty_coefficient_training_env -> contains
            #     'init_state_mode' -> train one set of penalty coefficients per env
            #     over sample_path_number scenarios.
            if param.pop('matched_state_decision', False):
                matched_state_decisions(
                    **param,
                    grb_env=grb_env,
                    grb_sub_envs=grb_sub_envs,
                    job_id=args.job_id,
                )
            elif 'policy_specs' in param:
                print('Wow, this is an evaluation record with policy_specs:')
                evaluate_policy_costs_with_information_relaxation(
                    **param,
                    grb_env=grb_env,
                    grb_sub_envs=grb_sub_envs,
                    job_id=args.job_id,
                )
            elif param.pop('partial_information_relaxation', False):
                partial_information_relaxation_for_path(
                    **param,
                    grb_env=grb_env,
                    grb_sub_envs=grb_sub_envs,
                    job_id=args.job_id,
                )
            elif param.pop('potential_improvement', False):
                estimate_potential_improvement(
                    **param,
                    grb_env=grb_env,
                    grb_sub_envs=grb_sub_envs,
                    job_id=args.job_id,
                )
            elif 'init_state_mode' in param:
                train_penalty_coefficients_for_env(
                    **param,
                    grb_env=grb_env,
                    grb_sub_envs=grb_sub_envs,
                    job_id=args.job_id,
                )
            elif param.pop('alp_training', False):
                train_alp_coefficients_for_env(
                    **param,
                    grb_env=grb_env,
                    grb_sub_envs=grb_sub_envs,
                    job_id=args.job_id,
                )
            else:
                raise ValueError(
                    "Unrecognized params payload: expected 'matched_state_decision', 'policy_specs' "
                    "(evaluation), 'partial_information_relaxation', 'potential_improvement', "
                    "'init_state_mode' (penalty-coefficient training) or 'alp_training' in record."
                )
    finally:
        for environment in reversed(grb_sub_envs):
            environment.dispose()
        grb_env.dispose()


if __name__ == '__main__':
    main()
