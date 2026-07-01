import argparse
import json
import pickle
import os
import time
import re
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from pprint import pprint

from generate_params import _save_training_result, _load_cached_training_result
import numpy as np
import pandas as pd
from gurobipy import GRB
from scipy.stats import geom

from experiments.experiment_config import get_config_by_type
from importance_sampling import build_proposal
from utils import iter_to_tuple, get_uid, safe_open, RunningStats, encode, decode, get_solution_value, acquire_grb_env, read_lines_with_pattern
from decision_maker import MyopicAgent, ALPRowGenerationAgent, ApproxQAgent
from policy_evaluator import PolicyEvaluator
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


# Config-only keys describing a generating function. They never belong in the
# runtime keyword arguments forwarded to an agent constructor.
_GENERATING_FUNCTION_SPEC_KEYS = (
    'generating_function_spec',
    'policy_generating_function_spec',
    'zero_lowerbound_generating_function_spec',
    'penalized_lowerbound_generating_function_spec',
    'training_generating_function_spec',
)


def _set_sample_path_proposal(agent_args):
    """Materialize the IS proposal into ``agent_args['sample_path_proposal']``.

    Config payloads carry a JSON-serializable proposal spec under either
    ``sample_path_length_proposal`` (legacy) or ``sample_path_proposal``.
    ``ApproxQAgent`` expects a built ``SamplePathLengthProposal`` instance via
    ``sample_path_proposal``, so convert the spec here (in the caller) and drop
    the legacy key. ``build_proposal(None)`` returns ``None`` and lets the agent
    fall back to its default proposal.
    """
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
    """Return a length-``num_periods`` array of per-period importance weights.

    ``period_weights`` is the reweighting vector for the post-warm-up evaluation
    horizon produced by the sampling proposal (see
    ``generate_test_paths_and_init_state``). ``None`` means the horizon was drawn
    from the target geometric distribution, so every period weight is 1 and the
    estimator reduces to the plain (non-importance-sampling) sum.
    """
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
    """Return True if (uid, policy_id) already exists in JSONL output."""
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
    """Return True if a record with ``uid`` already exists in JSONL output."""
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

def load_pickle_if_exists(path):
    """Return file contents if the file exists, otherwise return None."""
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
    """Atomically persist a pickle payload to avoid truncated files."""
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
                           period_weights=None):
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

    base_dir = os.path.join('experiments', 'results', experiment_name, 'pickles')
    os.makedirs(base_dir, exist_ok=True)
    # The cache/checkpoint key must capture the policy's *actual settings*, not
    # just (uid, policy_id). Several policies can share one policy_id (e.g.
    # multiple "approx_penalized_hindsight" variants) while differing in
    # agent_args (penalty coefficients, IS proposal, ...) and/or in the penalty
    # used for cost accounting. Hashing those into the filename prevents a later
    # policy from silently reusing an earlier policy's cached result.
    policy_signature = get_uid({
        'agent_name': agent_name,
        'agent_args': _to_jsonable({k: v for k, v in dict(agent_args).items() if k != 'grb_env'}),
        'penalty_coefficients': _to_jsonable(getattr(generating_function, 'coefficients', None)),
    })
    checkpoint_file = os.path.join(base_dir, f'{uid}-{policy_id}-{policy_signature}-checkpoint.pickle')
    result_file = os.path.join(base_dir, f'{uid}-{policy_id}-{policy_signature}-result.pickle')

    cached_result = load_pickle_if_exists(result_file)
    if cached_result is not None:
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        return cached_result

    sample_path = np.array(sample_path)

    config = get_config_by_type(case_type='infinite_custom', args=env_args)
    env = config.env
    runtime_agent_args = dict(agent_args)
    # The policy carries its own self-contained generating-function spec
    # (type + coefficients). It is independent from the accounting penalty
    # (``generating_function``) applied to the executed trajectory below.
    policy_generating_function_spec = _normalize_generating_function_spec(
        runtime_agent_args.pop('generating_function_spec', None)
    )
    serializable_agent_args = {k: _to_jsonable(v) for k, v in runtime_agent_args.items() if k != 'grb_env'}
    serializable_agent_args['generating_function_spec'] = policy_generating_function_spec

    # Penalty used for cost accounting on the executed trajectory. Built by the
    # caller as a separate instance sharing the lower-bound generating-function
    # spec/coefficients.
    local_generating_function = generating_function

    if agent_name in {"approx_hindsight", "approx_penalized_hindsight"}:
        runtime_agent_args['generating_function'] = _build_generating_function(
            env=env,
            spec=policy_generating_function_spec,
        )
        _set_sample_path_proposal(runtime_agent_args)
        agent_instance = ApproxQAgent(env, discount_factor=env.discount_factor,
                                        grb_env=grb_env,
                                        subproblem_grb_envs=grb_sub_envs,
                                        **runtime_agent_args
                                    )
    elif agent_name == "myopic":
        agent_instance = MyopicAgent(env, discount_factor=env.discount_factor, grb_env=grb_env, **runtime_agent_args)
    elif agent_name == 'row_gen_alp':
        agent_instance = ALPRowGenerationAgent(env, discount_factor=env.discount_factor, grb_env=grb_env, **runtime_agent_args)
    else:
        raise ValueError(f"Unsupported policy: {agent_name}")

    states = []
    actions = []
    costs = []
    penalties = []
    checkpoint = load_pickle_if_exists(checkpoint_file)
    if checkpoint is not None:
        states = checkpoint['states']
        actions = checkpoint['actions']
        costs = checkpoint['costs']
        penalties = checkpoint['penalties']
        t = checkpoint['t']
        s = checkpoint['s']
        solving_time_per_state = checkpoint.get('solving_time_per_state', RunningStats())
        s, _ = env.reset(init_state=s, t=t, new_arrivals=sample_path)
    else:
        t = 1
        s, _ = env.reset(init_state=init_state, t=t, new_arrivals=sample_path)
        solving_time_per_state = RunningStats()
    for tau in range(len(sample_path) - t + 2):
        current_t = t + tau
        start_time = time.time()
        _, action, _ = agent_instance.solve(s, current_t)
        solving_time_per_state += time.time() - start_time
        next_state, cost, done, _ = env.step(action)

        if current_t <= len(sample_path):
            new_arrivals = sample_path[current_t - 1]
            penalty = local_generating_function.calculate_penalty(s, action, new_arrivals)
            penalties.append(_to_float(penalty))

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
                    'penalties': penalties,
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
            'penalties': penalties,
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
    # Reweight each post-warm-up period by its importance-sampling weight so the
    # estimator stays unbiased when the sample path was drawn from a proposal
    # whose length distribution differs from the target geometric horizon. With
    # no proposal (period_weights is None) the weights are all 1 and this reduces
    # to the plain sum. costs[warm_up:] has one more entry than penalties[warm_up:]
    # (the trailing period carries a stage cost but no arrival/penalty).
    tail_costs = costs[warm_up_periods:]
    tail_penalties = penalties[warm_up_periods:]
    cost_weights = _resolve_period_weights(period_weights, len(tail_costs))
    penalty_weights = _resolve_period_weights(period_weights, len(tail_penalties))
    total_cost = float(np.dot(cost_weights, tail_costs))
    total_penalty = float(np.dot(penalty_weights, tail_penalties))
    penalized_cost = total_cost + total_penalty
    warmup_state = tuple(np.array(item).tolist() for item in states[warm_up_periods])

    result = {
        'policy_id': policy_id,
        'agent_name': agent_name,
        'agent_args': serializable_agent_args,
        'penalized_cost': penalized_cost,
        'total_cost': total_cost,
        'total_penalty': total_penalty,
        "warmup_state": warmup_state,
        'costs': [float(v) for v in costs],
        'penalties': [float(v) for v in penalties],
        'scheduled_patients': scheduled_patients,
        'overtime': overtime.tolist(),
        'postponing_decisions': postponing_decisions.tolist(),
        'solving_time_per_state': solving_time_per_state.mean,
    }

    atomic_pickle_dump(result_file, result)

    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return result

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
                                                      period_weights=None):
    '''
    This function evaluates the costs of different policies and their gaps to the information relaxation lower bounds.
    '''
    init_state = tuple(np.array(item) for item in init_state)
    sample_path = np.array(sample_path)

    output_file = os.path.join('experiments', 'results', experiment_name, f'{job_id}.jsonl')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    config = get_config_by_type(case_type='infinite_custom', args=env_args)
    env = config.env

    # One self-contained spec (type + coefficients) drives the zero lower
    # bound, the penalized lower bound, and the penalty accounting on each
    # policy. Build three SEPARATE instances that share this spec/coefficients.
    generating_function_spec = _normalize_generating_function_spec(generating_function_spec)
    zero_lowerbound_generating_function = _build_generating_function(env=env, spec=generating_function_spec)
    penalized_lowerbound_generating_function = _build_generating_function(env=env, spec=generating_function_spec)
    policy_costs_generating_function = _build_generating_function(env=env, spec=generating_function_spec)


    zero_lowerbound_args = {
        'current_decision_var_type': 'integer',
        'future_decision_var_type': 'continuous',
        'generating_function': zero_lowerbound_generating_function,
        'penalty_ratio': 0,
        'grb_env': grb_env,
        'subproblem_grb_envs': grb_sub_envs,
    }
    penalized_lowerbound_args = {
        'current_decision_var_type': 'integer',
        'future_decision_var_type': 'continuous',
        'generating_function': penalized_lowerbound_generating_function,
        'penalty_ratio': 1,
        'grb_env': grb_env,
        'subproblem_grb_envs': grb_sub_envs,
    }

    zero_lowerbound_instance = ApproxQAgent(env, discount_factor=env.discount_factor, **zero_lowerbound_args)
    penalized_lowerbound_instance = ApproxQAgent(env, discount_factor=env.discount_factor, **penalized_lowerbound_args)

    sample_path_tail = sample_path[warm_up_periods:]

    if not policy_specs:
        start = time.time()
        zero_information_relaxation_cost = zero_lowerbound_instance.calculate_information_relaxation_cost(
            init_state,
            sample_path=sample_path_tail,
            period_weights=period_weights,
        )
        print(
            f"Zero information relaxation cost computed in {time.time() - start:.1f} seconds: "
            f"{zero_information_relaxation_cost}"
        )
        start = time.time()
        penalized_information_relaxation_cost = penalized_lowerbound_instance.calculate_information_relaxation_cost(
            init_state,
            sample_path=sample_path_tail,
            period_weights=period_weights,
        )
        print(
            f"Penalized information relaxation cost computed in {time.time() - start:.1f} seconds: "
            f"{penalized_information_relaxation_cost}"
        )

        record = {
            'uid': uid,
            'group_id': group_id,
            'experiment_name': experiment_name,
            'mutate_val': mutate_val,
            'warm_up_periods': warm_up_periods,
            'policy_id': 'information_relaxation_only',
            'agent_name': 'information_relaxation_only',
            'zero_information_relaxation_cost': float(zero_information_relaxation_cost),
            'penalized_information_relaxation_cost': float(penalized_information_relaxation_cost),
            'gap_to_zero_information_relaxation': 0.0,
            'gap_to_penalized_information_relaxation': 0.0,
            'warmup_state': tuple(np.array(item).tolist() for item in init_state),
        }

        if not jsonl_result_exists(output_file, uid, record['policy_id']):
            with open(output_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        else:
            print(f"Skip saving duplicate result: uid={uid}, policy_id={record['policy_id']}")

        return [record]

    summary_rows = []
    for policy_spec in policy_specs:
        policy_result = calculate_policy_costs_with_penalty(
            uid=uid,
            experiment_name=experiment_name,
            policy_id=policy_spec['policy_id'],
            agent_name=policy_spec['agent_name'],
            agent_args=policy_spec['agent_args'],
            env_args=env_args,
            init_state=tuple(np.array(item) for item in init_state),
            sample_path=sample_path,
            warm_up_periods=warm_up_periods,
            generating_function=policy_costs_generating_function,
            grb_env=grb_env,
            grb_sub_envs=grb_sub_envs,
            period_weights=period_weights,
        )
        warmup_sate = tuple(np.array(item) for item in policy_result.get('warmup_state', init_state))
        start = time.time()
        zero_information_relaxation_cost = zero_lowerbound_instance.calculate_information_relaxation_cost(warmup_sate, sample_path=sample_path_tail, period_weights=period_weights)
        print(f"Zero information relaxation cost computed in {time.time() - start:.1f} seconds: {zero_information_relaxation_cost}")
        start = time.time()
        penalized_information_relaxation_cost = penalized_lowerbound_instance.calculate_information_relaxation_cost(warmup_sate, sample_path=sample_path_tail, period_weights=period_weights)
        print(f"Penalized information relaxation cost computed in {time.time() - start:.1f} seconds: {penalized_information_relaxation_cost}")
        policy_result.update({
            'uid': uid,
            'group_id': group_id,
            'experiment_name': experiment_name,
            'mutate_val': mutate_val,
            'warm_up_periods': warm_up_periods,
            'zero_information_relaxation_cost': float(zero_information_relaxation_cost),
            'penalized_information_relaxation_cost': float(penalized_information_relaxation_cost),
            'gap_to_zero_information_relaxation': float(policy_result['total_cost'] - zero_information_relaxation_cost),
            'gap_to_penalized_information_relaxation': float(policy_result['penalized_cost'] - penalized_information_relaxation_cost),
        })

        if not jsonl_result_exists(output_file, uid, policy_result['policy_id']):
            with open(output_file, 'a') as f:
                f.write(json.dumps(policy_result) + '\n')
        else:
            print(f"Skip saving duplicate result: uid={uid}, policy_id={policy_result['policy_id']}")

        summary_rows.append({
            'policy_id': policy_result['policy_id'],
            'agent_name': policy_result['agent_name'],
            'penalized_cost': policy_result['penalized_cost'],
            'total_cost': policy_result['total_cost'],
            'total_penalty': policy_result['total_penalty'],
            'zero_information_relaxation_cost': float(zero_information_relaxation_cost),
            'penalized_information_relaxation_cost': float(penalized_information_relaxation_cost),
            'gap_to_zero_information_relaxation': policy_result['gap_to_zero_information_relaxation'],
            'gap_to_penalized_information_relaxation': policy_result['gap_to_penalized_information_relaxation'],
        })

    return summary_rows

def train_lowerbound_for_init_state(
    uid,
    experiment_name,
    mutate_val,
    sample_path_number,
    init_state_index,
    init_state,
    env_args,
    agent_args,
    grb_env,
    grb_sub_envs,
    job_id,
    training_generating_function_spec=None,
):
    """Compute the tight penalized information-relaxation lower bound at a
    single supplied initial state.

    X = ``init_state`` (one of the states drawn by ``generate_train_env``).
    Y = the Benders-trained tight penalized lower bound at X using the same
    Monte-Carlo arrival sample paths for every command (forced by
    ``env.reset_random_seeds()`` and the pinned arrival seeds in env_args).

    The generating function used for the penalty is created entirely from
    ``training_generating_function_spec`` (its ``name``/type). The penalty
    coefficients are NOT supplied to this function: they are the decision
    variables optimized by the Benders training below, so the generating
    function is built without coefficients and any coefficients that happen to
    be present in the spec are ignored.

    One JSONL record per init_state is appended to
    ``experiments/results/<experiment_name>/<job_id>.jsonl`` so the workload
    can be split across a job array. Records that already exist (matched on
    ``uid``) are skipped.
    """
    output_file = os.path.join(
        'experiments', 'results', experiment_name, f'{job_id}.jsonl'
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if jsonl_uid_exists(output_file, uid):
        print(
            f"Skip duplicate training result: uid={uid}, "
            f"init_state_index={init_state_index}"
        )
        return None

    config = get_config_by_type('infinite_custom', args=env_args)
    env = config.env
    # Build the generating function solely from the self-contained training
    # spec. Coefficients are trained, not given: drop any that slipped into the
    # spec so the trainer always optimizes them from scratch.
    training_generating_function_spec = _normalize_generating_function_spec(
        training_generating_function_spec
    )
    training_generating_function_spec.pop('coefficients', None)
    generating_function = _build_generating_function(
        env=env,
        spec=training_generating_function_spec,
    )

    # Runtime keyword arguments for the trainer come from the nested
    # ``agent_args``; strip the config-only generating-function spec keys so
    # they are not forwarded to the agent constructor.
    inner = dict((agent_args or {}).get('agent_args', {}))
    for spec_key in _GENERATING_FUNCTION_SPEC_KEYS:
        inner.pop(spec_key, None)
    inner['generating_function'] = generating_function
    _set_sample_path_proposal(inner)
    inner['sample_path_number'] = sample_path_number

    agent = ApproxQAgent(
        env=env,
        discount_factor=env.discount_factor,
        grb_env=grb_env,
        subproblem_grb_envs=grb_sub_envs,
        **inner,
    )

    init_state_tuple = tuple(np.array(item) for item in init_state)
    # Reset RNGs so every command (every init_state) sees the *same*
    # Monte-Carlo arrival sample paths. The pinned env_random_seed /
    # arrival_random_seed in env_args (set by generate_train_env) ensures the
    # underlying generators are identical across commands; this call rewinds
    # them to the start before sampling.
    env.reset_random_seeds()

    print(
        f"Training tight penalized lower bound for uid={uid}, "
        f"init_state_index={init_state_index}, "
        f"sample_path_number={sample_path_number}"
    )
    start = time.time()
    # obj, coefficients, info = agent.extensive_form_train(
    #     coefficient_bound=GRB.INFINITY,
    #     init_state=init_state_tuple,
    #     crossover=False,
    #     verbose=False,
    # )
    checkpoint_dir = os.path.join(
        'experiments', 'results', experiment_name, 'benders_checkpoints'
    )    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f'{uid}-{init_state_index}-checkpoint.pickle'
    )
    obj, coefficients, info = agent.benders_decomposition_train(coefficient_bound=GRB.INFINITY, init_state=init_state_tuple, checkpoint_path=checkpoint_path, resume_checkpoint_path=checkpoint_path)
    elapsed = time.time() - start
    print(
        f"  obj={obj}, elapsed={elapsed:.1f}s"
    )

    init_state_jsonable = [np.asarray(item).tolist() for item in init_state]
    if hasattr(coefficients, 'tolist'):
        coefficients_jsonable = coefficients.tolist()
    else:
        coefficients_jsonable = list(coefficients)

    record = {
        'uid': uid,
        'experiment_name': experiment_name,
        'mutate_val': mutate_val,
        'init_state_index': init_state_index,
        'init_state': init_state_jsonable,
        'sample_path_number': sample_path_number,
        'tight_penalized_lower_bound': float(obj),
        'coefficients': coefficients_jsonable,
        'training_time_seconds': elapsed,
    }

    with open(output_file, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record

def _draw_per_scenario_init_states(env_args, init_state_seed, sample_path_number):
    """Draw one reproducible initial state per scenario from a sampler env.

    The sampler env is a copy of ``env_args`` with ``env_random_seed`` (and
    hence the derived ``init_state_random_seed``) set to ``init_state_seed``, so
    the per-scenario initial states are reproducible and independent of the
    pinned arrival/sample-path seeds. States are drawn single-threaded here so
    the subsequent parallel Benders worker build stays deterministic.
    """
    sampler_env_args = copy.deepcopy(env_args)
    sampler_env_args['env_random_seed'] = init_state_seed
    sampler_env = get_config_by_type('infinite_custom', args=sampler_env_args).env
    sampler_env.reset_random_seeds()
    return [
        tuple(np.array(component) for component in sampler_env.generate_initial_state())
        for _ in range(sample_path_number)
    ]

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
    """Train one set of penalty coefficients for an env variant by Benders
    decomposition over ``sample_path_number`` arrival sample paths.

    Emitted by ``generate_penalty_coefficient_training_env``. ``init_state_mode``
    selects how each scenario's starting state is chosen:

      * ``'generate'``: draw one initial state per scenario from a sampler env
        seeded by ``init_state_seed`` (reproducible); ``init_state`` is ``None``.
      * ``'shared'``: every scenario starts from the single ``init_state``.
      * ``'per_scenario'``: scenario ``i`` starts from ``init_state[i]`` and
        ``len(init_state) == sample_path_number``.

    Like ``train_lowerbound_for_init_state`` the penalty coefficients are the
    Benders decision variables (NOT supplied): the generating function is built
    from ``training_generating_function_spec`` with any coefficients stripped.
    One JSONL record (uid, coefficients, objective) is appended to
    ``experiments/results/<experiment_name>/<job_id>.jsonl``; duplicates
    (matched on ``uid``) are skipped.
    """
    output_file = os.path.join(
        'experiments', 'results', experiment_name, f'{job_id}.jsonl'
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if jsonl_uid_exists(output_file, uid):
        print(f"Skip duplicate penalty-training result: uid={uid}")
        return None

    config = get_config_by_type('infinite_custom', args=env_args)
    env = config.env

    # Build the generating function solely from the self-contained training
    # spec; coefficients are trained, not given.
    training_generating_function_spec = _normalize_generating_function_spec(
        training_generating_function_spec
    )
    training_generating_function_spec.pop('coefficients', None)
    generating_function = _build_generating_function(
        env=env,
        spec=training_generating_function_spec,
    )

    inner = dict((agent_args or {}).get('agent_args', {}))
    for spec_key in _GENERATING_FUNCTION_SPEC_KEYS:
        inner.pop(spec_key, None)
    inner['generating_function'] = generating_function
    _set_sample_path_proposal(inner)
    inner['sample_path_number'] = sample_path_number

    agent = ApproxQAgent(
        env=env,
        discount_factor=env.discount_factor,
        grb_env=grb_env,
        subproblem_grb_envs=grb_sub_envs,
        **inner,
    )

    # Resolve the unified init_state argument from the init_state mode:
    #   generate     -> one drawn state per scenario (list of length N)
    #   per_scenario -> the supplied list of states, one per scenario
    #   shared       -> a single state shared across all scenarios
    if init_state_mode == 'generate':
        resolved_init_state = _draw_per_scenario_init_states(
            env_args=env_args,
            init_state_seed=init_state_seed,
            sample_path_number=sample_path_number,
        )
    elif init_state_mode == 'per_scenario':
        resolved_init_state = [
            tuple(np.array(component) for component in state) for state in init_state
        ]
    elif init_state_mode == 'shared':
        resolved_init_state = tuple(np.array(component) for component in init_state)
    else:
        raise ValueError(f"Unknown init_state_mode: {init_state_mode!r}")

    # Reset RNGs so every command sees the *same* Monte-Carlo arrival sample
    # paths (the pinned arrival/env seeds in env_args make them identical).
    env.reset_random_seeds()

    print(
        f"Training penalty coefficients for uid={uid}, mode={init_state_mode}, "
        f"sample_path_number={sample_path_number}"
    )
    start = time.time()
    # Solver selection (default: Benders cutting plane). Set
    # PENALTY_TRAIN_SOLVER=extensive to solve the monolithic extensive-form LP
    # instead.
    solver_choice = os.environ.get('PENALTY_TRAIN_SOLVER', 'benders').lower()
    if solver_choice in ('extensive', 'extensive_form', 'ef', 'deterministic_equivalent'):
        # Deterministic-equivalent / extensive-form LP: build the whole
        # monolithic LP once and solve it in a single barrier pass. Exact
        # optimum + exact kappa (coupling-row shadow prices); no cut iteration,
        # no growing master re-solved O(iters) times like Benders. No checkpoint
        # needed (one shot).
        ef_crossover = os.environ.get('PENALTY_EF_CROSSOVER', '0') == '1'
        obj, coefficients, info = agent.extensive_form_train(
            coefficient_bound=GRB.INFINITY,
            init_state=resolved_init_state,
            crossover=ef_crossover,
            verbose=True,
        )
    else:
        checkpoint_dir = os.path.join(
            'experiments', 'results', experiment_name, 'benders_checkpoints'
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f'{uid}-penalty-checkpoint.pickle'
        )
        obj, coefficients, info = agent.benders_decomposition_train(
            coefficient_bound=GRB.INFINITY,
            init_state=resolved_init_state,
            checkpoint_path=checkpoint_path,
            resume_checkpoint_path=checkpoint_path,
        )
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
        'init_state_mode': init_state_mode,
        'init_state_seed': init_state_seed,
        'tight_penalized_lower_bound': float(obj),
        'coefficients': coefficients_jsonable,
        'training_time_seconds': elapsed,
    }

    with open(output_file, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record

def evaluated_hindsight_policy_solving_time(uid,
                                            experiment_name,
                                            mutate_val,
                                            sample_path_number,
                                            init_state_index,
                                            init_state,
                                            env_args,
                                            agent_args,
                                            grb_env,
                                            grb_sub_envs,
                                            job_id):
    output_file = os.path.join(
        'experiments', 'results', experiment_name, f'{job_id}.jsonl'
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if jsonl_uid_exists(output_file, uid):
        print(
            f"Skip duplicate training result: uid={uid}, "
            f"init_state_index={init_state_index}"
        )
        return None
    config = get_config_by_type(case_type='infinite_custom', args=env_args)
    env = config.env
    policy_spec = dict(agent_args or {})
    agent_name = policy_spec.get('agent_name', 'approx_penalized_hindsight')
    policy_id = policy_spec.get('policy_id', agent_name)
    runtime_agent_args = dict(policy_spec.get('agent_args', {}))
    policy_generating_function_spec = _normalize_generating_function_spec(
        runtime_agent_args.pop('generating_function_spec', None)
    )
    runtime_agent_args['generating_function'] = _build_generating_function(
            env=env,
            spec=policy_generating_function_spec,
        )
    _set_sample_path_proposal(runtime_agent_args)
    agent_instance = ApproxQAgent(env, discount_factor=env.discount_factor,
                                    grb_env=grb_env,
                                    subproblem_grb_envs=grb_sub_envs,
                                    **runtime_agent_args
                                )
    obj, action_t, info = agent_instance.solve(tuple(np.array(item) for item in init_state), init_state_index)

    start = time.time()
    obj, action_t, info = agent_instance.solve(tuple(np.array(item) for item in init_state), init_state_index)
    elapsed = time.time() - start

    init_state_jsonable = [np.asarray(item).tolist() for item in init_state]

    record = {
        'uid': uid,
        'experiment_name': experiment_name,
        'mutate_val': mutate_val,
        'policy_id': policy_id,
        'agent_name': agent_name,
        'objective': float(obj),
        'init_state_index': init_state_index,
        'init_state': init_state_jsonable,
        'action': [np.asarray(part).tolist() for part in action_t],
        'sample_path_number': sample_path_number,
        'training_time_seconds': elapsed,
    }
    with open(output_file, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of using argparse to pass in a list of lists.")
    parser.add_argument('--params', help='Input JSON-encoded list of lists', type=str)
    parser.add_argument('--params_file', help='Path to JSON file containing params payload', type=str)
    parser.add_argument('--job_id', help='Input METAJOB_ID', type=str)
    args = parser.parse_args()
    if args.params_file:
        with open(args.params_file, 'r') as f:
            params = json.load(f)
    elif args.params:
        params = json.loads(args.params)
    else:
        raise ValueError('Either --params or --params_file must be provided.')

    if isinstance(params, dict):
        params = [params]
    grb_env = acquire_grb_env({"Threads": 1}, verbose=False, wait=15)
    # Dynamically allocate a pool of Gurobi tokens (envs) for the subproblems.
    # A Gurobi env is NOT thread-safe for concurrent optimization, so the
    # minimum number of tokens needed equals the number of subproblems solved
    # concurrently, i.e. the number of available CPUs. Subproblems are then
    # partitioned across this pool and the ones sharing an env are solved
    # sequentially (see SubproblemWorker grouping in the Benders solver), so we
    # only hold `min(sample_path_number, num_cpus)` tokens instead of one per
    # subproblem.
    def _sample_path_number_for_param(param):
        # 1) Evaluation records embed it under each policy_spec.agent_args.
        for spec in param.get('policy_specs', []) or []:
            n = spec.get('agent_args', {}).get('sample_path_number', 0)
            if n:
                return n
        # 2) Training records (generate_train_env) put it at the top level
        #    and / or under agent_args.agent_args.
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
    num_sub_envs = min(sample_path_number, num_cpus) if sample_path_number else 0
    grb_sub_envs = [
        acquire_grb_env({"Threads": 1}, verbose=False, wait=15)
        for _ in range(num_sub_envs)
    ]
    print('grb_sub_envs:',len(grb_sub_envs))
    def _is_hindsight_timing_record(param):
        spec = ((param.get('agent_args') or {}).get('agent_args') or {}).get('generating_function_spec')
        return isinstance(spec, dict) and 'coefficients' in spec

    for param in params:
        # Dispatch on the shape of the params record:
        #   * generate_test_paths_and_init_state -> contains 'policy_specs'
        #     -> evaluate policy costs against information-relaxation bounds.
        #   * generate_penalty_coefficient_training_env -> contains
        #     'init_state_mode' -> train one set of penalty coefficients per env
        #     over sample_path_number scenarios.
        #   * generate_train_env -> contains 'init_state_index' (and no
        #     'policy_specs') -> train the tight penalized lower bound at a
        #     single initial state and emit one (X, Y) record.
        if 'policy_specs' in param:
            print('Wow, this is an evaluation record with policy_specs:')
            evaluate_policy_costs_with_information_relaxation(
                **param,
                grb_env=grb_env,
                grb_sub_envs=grb_sub_envs,
                job_id=args.job_id,
            )
        elif _is_hindsight_timing_record(param):
            evaluated_hindsight_policy_solving_time(
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
        elif 'init_state_index' in param:
            train_lowerbound_for_init_state(
                **param,
                grb_env=grb_env,
                grb_sub_envs=grb_sub_envs,
                job_id=args.job_id,
            )
        else:
            raise ValueError(
                "Unrecognized params payload: expected 'policy_specs' "
                "(evaluation), 'init_state_mode' (penalty-coefficient training), "
                "or 'init_state_index' (lower-bound training) in record."
            )