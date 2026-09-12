import copy
import os

import numpy as np

from experiments import get_config_by_type
from importance_sampling import build_proposal
from importance_sampling.sample_path import sample_path_from_record
from utils import get_uid, is_single_init_state

from param_generation.caching import (
    load_trained_coefficient_record_from_folder as _load_trained_coefficient_record_from_folder,
    load_trained_coefficients_from_folder as _load_trained_coefficients_from_folder,
)
from param_generation.command_files import (
    write_command_file,
    write_grouped_command_file as _split_list_into_groups,
)
from param_generation.generating_functions import (
    normalize_generating_function_spec as _normalize_generating_function_spec,
)
from param_generation.policies import build_penalty_policy as _build_penalty_policy
from param_generation.training import (
    train_alp_coefficients,
    train_penalty_coefficients,
    zero_penalty_coefficients as _zero_penalty_coefficients,
)


def _pin_sampling_seeds(env_args, seed):
    env_args['arrival_random_seed'] = seed
    env_args['stop_time_random_seed'] = seed
    env_args['env_random_seed'] = seed


def offset_sample_generation_seeds(env_args, seed_offset):
    sample_gen_args = copy.deepcopy(env_args)
    sample_gen_args['env_random_seed'] = env_args.get('env_random_seed', 0) + seed_offset
    sample_gen_args['arrival_random_seed'] = env_args.get('arrival_random_seed', 42) + seed_offset
    sample_gen_args['stop_time_random_seed'] = env_args.get('stop_time_random_seed', 1) + seed_offset
    return sample_gen_args


def _make_sampler_env(base_env_args, init_state_rng):
    sampler_env_args = copy.deepcopy(base_env_args)
    sampler_env_args['env_random_seed'] = int(init_state_rng.integers(0, 2**31 - 1))
    return get_config_by_type('infinite_custom', args=sampler_env_args).env


def _state_to_jsonable(state):
    return [np.asarray(component).tolist() for component in state]


def _evaluation_period_weights(proposal, discount_factor, tail_length, arrival_generator=None):
    if proposal is None:
        return None
    weights = proposal.survival_weights([tail_length], discount_factor, arrival_generator)[0]
    return [float(w) for w in weights]


def _evaluation_terminal(proposal, tail_length, arrival_generator=None):
    if proposal is None:
        return None
    return proposal.terminal_for([tail_length], arrival_generator)[0].value


def _normalize_penalty_training_init_state(init_state, sample_path_number):
    if init_state is None:
        return 'generate', None
    if is_single_init_state(init_state):
        return 'shared', _state_to_jsonable(init_state)
    if not isinstance(init_state, (list, tuple)):
        raise ValueError(
            "init_state must be None, a single (regular, overtime, waitlist) "
            "state, or a list of such states."
        )
    init_states = list(init_state)
    if len(init_states) != sample_path_number:
        raise ValueError(
            "init_state list must have length sample_path_number "
            f"({sample_path_number}); got {len(init_states)}."
        )
    normalized = []
    for index, state in enumerate(init_states):
        if not is_single_init_state(state):
            raise ValueError(
                f"init_state[{index}] is not a valid (regular, overtime, "
                "waitlist) initial state."
            )
        normalized.append(_state_to_jsonable(state))
    return 'per_scenario', normalized


_COEFFICIENTS_SOURCE_KEYS = (
    'uid', 'file', 'sample_path_number', 'init_state_mode', 'init_state_seed',
    'tight_penalized_lower_bound',
)


def normalize_penalty_ratios(penalty_ratios):
    ratios = {float(ratio) for ratio in penalty_ratios}
    ratios.update((0.0, 1.0))
    return sorted(ratios)


def _coefficients_source(record):
    return {key: record.get(key) for key in _COEFFICIENTS_SOURCE_KEYS}


def _warn_initial_state_mismatch(coefficients_source, is_random_initial_state):
    mode = coefficients_source.get('init_state_mode')
    if mode == 'generate' and not is_random_initial_state:
        print("WARNING: initial-state distribution mismatch: the coefficients were "
              "trained with per-scenario random initial states "
              "(init_state_mode='generate') but the evaluation starts every path "
              "from the fixed reset initial state; pass --random-init to match.")
    elif mode == 'shared' and is_random_initial_state:
        print("WARNING: initial-state distribution mismatch: the coefficients were "
              "trained from one shared initial state (init_state_mode='shared') "
              "but the evaluation draws random initial states (--random-init); "
              "use --init-occupancy to match the training state instead.")


def generate_test_paths_and_init_state(test_envs, test_sample_path_num, warm_up_periods=0, num_periods=None, dat_file=None, num_groups=None, is_require_penalty_coefficients=True, is_random_initial_state=False, policy_ids=None, warm_up_policy_id=None, evaluation_proposal_spec=None, penalty_coefficients_dir=None, warm_up_paths=None, sample_gen_seed_offset=1001, penalty_ratios=None):
    policy_ids = list(policy_ids or [])
    policy_id_set = set(policy_ids)
    if penalty_ratios is not None:
        penalty_ratios = normalize_penalty_ratios(penalty_ratios)
        if not is_require_penalty_coefficients:
            raise ValueError('penalty_ratios requires is_require_penalty_coefficients=True.')
    if warm_up_policy_id is not None and warm_up_policy_id not in policy_id_set:
        raise ValueError(
            f"warm_up_policy_id '{warm_up_policy_id}' must be one of the "
            f"evaluated policy_ids {policy_ids}."
        )
    if warm_up_paths is not None:
        if num_periods is not None:
            raise ValueError(
                "warm_up_paths is only supported on the proposal-driven branch "
                "(num_periods=None)."
            )
        if len(warm_up_paths) != test_sample_path_num:
            raise ValueError(
                "warm_up_paths must have length test_sample_path_num "
                f"({test_sample_path_num}); got {len(warm_up_paths)}."
            )
        warm_up_paths = [np.asarray(path) for path in warm_up_paths]
        for index, path in enumerate(warm_up_paths):
            if len(path) != warm_up_periods:
                raise ValueError(
                    f"warm_up_paths[{index}] has {len(path)} periods but "
                    f"warm_up_periods={warm_up_periods}."
                )
    results = []
    for (env_uid, experiment_name, mutate_val), variant in test_envs.items():
        env_args = variant['env_args']
        env = get_config_by_type('infinite_custom', args=env_args).env
        print(f"Processing env_uid: {env_uid}, experiment_name: {experiment_name}, mutate_val: {mutate_val}")
        inner_agent_args = variant.get('agent_args', {}).get('agent_args', {})
        policy_generating_function_spec = _normalize_generating_function_spec(
            inner_agent_args.get('policy_generating_function_spec')
            or inner_agent_args.get('generating_function_spec')
        )
        lowerbound_generating_function_spec = _normalize_generating_function_spec(
            inner_agent_args.get('penalized_lowerbound_generating_function_spec')
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

        direct_coefficients = _zero_penalty_coefficients(env, lowerbound_generating_function_spec)
        coefficients_source = None
        if is_require_penalty_coefficients:
            sample_path_number = variant.get('agent_args', {}).get('agent_args', {}).get('sample_path_number')
            if penalty_ratios is not None:
                record = _load_trained_coefficient_record_from_folder(
                    experiment_name=experiment_name,
                    mutate_val=mutate_val,
                    sample_path_number=sample_path_number,
                    folder_path=penalty_coefficients_dir,
                )
                if record is None:
                    raise ValueError(
                        "penalty_ratios requires trained coefficients, but none were "
                        f"found for experiment={experiment_name}, mutate_val={mutate_val}, "
                        f"sample_path_number={sample_path_number} in "
                        f"{penalty_coefficients_dir or os.path.join('experiments', 'results', experiment_name)}; "
                        "refusing to train."
                    )
                direct_coefficients = record['coefficients']
                coefficients_source = _coefficients_source(record)
                print(
                    f"Loaded trained penalty coefficients for the penalty-ratio grid "
                    f"from {record['file']} (uid={coefficients_source['uid']}, "
                    f"init_state_mode={coefficients_source['init_state_mode']}, "
                    f"in-sample objective={coefficients_source['tight_penalized_lower_bound']})."
                )
                _warn_initial_state_mismatch(coefficients_source, is_random_initial_state)
            else:
                loaded_coefficients = _load_trained_coefficients_from_folder(
                    experiment_name=experiment_name,
                    mutate_val=mutate_val,
                    sample_path_number=sample_path_number,
                    folder_path=penalty_coefficients_dir,
                )
                if loaded_coefficients is not None:
                    direct_coefficients = loaded_coefficients
                    print(
                        f"Loaded trained penalty coefficients from folder for "
                        f"experiment={experiment_name}, mutate_val={mutate_val}."
                    )
                else:
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
                    penalty_coefficients=_zero_penalty_coefficients(env, policy_generating_function_spec),
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
            approx_q_coefficients = direct_coefficients
            approx_q_generating_function_spec = policy_generating_function_spec
            policies.append(
                _build_penalty_policy(
                    base_agent_args=variant['agent_args'],
                    policy_id='approx_Q',
                    solver_name='approx_Q',
                    penalty_coefficients=approx_q_coefficients,
                    generating_function_spec=approx_q_generating_function_spec,
                )
            )

        max_length = 0
        # Path-generation seeds are offset from the training seeds so evaluation
        # paths stay independent of the ALP/penalty training paths; distinct
        # offsets keep different experiments' tails independent of each other.
        sample_gen_args = offset_sample_generation_seeds(env_args, sample_gen_seed_offset)
        config_for_sample_path = get_config_by_type('infinite_custom', args=sample_gen_args)
        env_for_sample_path = config_for_sample_path.env
        eval_proposal_spec = (
            evaluation_proposal_spec
            if evaluation_proposal_spec is not None
            else inner_agent_args.get('sample_path_length_proposal')
            or inner_agent_args.get('sample_path_proposal')
        )
        sample_path_proposal = build_proposal(eval_proposal_spec)
        if (lowerbound_generating_function_spec['name'] == 'absorption_linear_penalty'
                and (num_periods is not None or sample_path_proposal is None)):
            raise ValueError(
                "the absorption penalty (absorption_linear_penalty) needs every evaluation "
                "record to carry survival weights and a terminal outcome: draw the tails from "
                "a length proposal (pass --eval-proposal, e.g. "
                "'{\"type\": \"geometric\", \"discount_factor_proposal\": 0.99}')")
        # Generate the whole evaluation batch directly from the proposal in a
        # single call. The proposal's stratified/QMC length sampling is defined
        # over the FULL sample size (e.g. a mixture deterministically assigns a
        # ``lambda_0`` fraction of the paths to the long/target component), so it
        # must see all ``test_sample_path_num`` paths at once. Drawing one path at
        # a time (the previous ``size=1`` call) rounds ``lambda_0 * 1`` down to
        # zero long paths, collapsing the mixture onto its short component and
        # shrinking the mean horizon. ``sample_arrival_paths`` draws from
        # ``env_for_sample_path``'s arrival RNG, whose seeds are offset by
        # ``sample_gen_seed_offset`` from the training seeds, so the evaluation
        # paths stay independent of the training sample paths.
        proposal_tails = None
        path_weights = path_strata = None
        if num_periods is None and sample_path_proposal is not None:
            # Validate the allocation before sampling: every positive-mass
            # stratum needs a sample for the weighted estimator to be valid.
            path_weights = sample_path_proposal.path_weights(test_sample_path_num)
            path_strata = sample_path_proposal.path_strata(test_sample_path_num)
            proposal_tails, _ = sample_path_proposal.sample_arrival_paths(
                arrival_generator=env_for_sample_path.arrival_generator,
                size=test_sample_path_num,
            )
        average_sample_path_length = 0
        for path_index in range(test_sample_path_num):
            init_state = env_for_sample_path.generate_initial_state() if ('init_state' not in env_args.get('reset_params', {})) or is_random_initial_state else env_args['reset_params']['init_state']
            init_state = tuple(np.array(item).tolist() for item in init_state)
            if num_periods is None:
                warm_up_path = (
                    warm_up_paths[path_index]
                    if warm_up_paths is not None
                    else env_for_sample_path.reset_arrivals(stop_time=warm_up_periods)
                )
                if proposal_tails is not None:
                    sampled_path = proposal_tails[path_index]
                else:
                    sampled_path = env_for_sample_path.reset_arrivals()
                sample_path = np.concatenate((warm_up_path, sampled_path), axis=0) if len(warm_up_path) > 0 else sampled_path
                period_weights = _evaluation_period_weights(
                    sample_path_proposal,
                    env_for_sample_path.discount_factor,
                    len(sampled_path),
                    env_for_sample_path.arrival_generator,
                )
                terminal = _evaluation_terminal(
                    sample_path_proposal, len(sampled_path), env_for_sample_path.arrival_generator)
            else:
                sample_path = env_for_sample_path.reset_arrivals(stop_time=num_periods)
                period_weights = None
                terminal = None
            sample_path = sample_path.tolist() if hasattr(sample_path, 'tolist') else sample_path
            max_length = max(max_length, len(sample_path))
            average_sample_path_length += len(sample_path)
            params = {
                'init_state': init_state,
                'sample_path': sample_path,
                'warm_up_periods': warm_up_periods,
                'env_args': env_args,
            }
            save_params = {
                "experiment_name": experiment_name,
                "mutate_val": mutate_val,
                **params,
                "period_weights": period_weights,
                "terminal": terminal,
                "path_weight": None if path_weights is None else float(path_weights[path_index]),
                "path_stratum": None if path_strata is None else int(path_strata[path_index]),
                "generating_function_spec": {**lowerbound_generating_function_spec, 'coefficients': direct_coefficients},
                "group_id": env_uid,
                "policy_specs": policies,
            }
            if warm_up_policy_id is not None:
                save_params['warm_up_policy_id'] = warm_up_policy_id
            if penalty_ratios is not None:
                save_params['penalty_ratios'] = list(penalty_ratios)
                save_params['coefficients_source'] = coefficients_source
            # Include all record semantics in duplicate detection, including
            # the drawing law, policy/accounting coefficients and shared warm-up.
            # Hash canonical metadata for exactly the post-warm-up tail.
            tail = sample_path_from_record(sample_path[warm_up_periods:], period_weights, terminal)
            weights = np.ones(tail.periods) if tail.survival_weights is None else tail.survival_weights
            save_params['uid'] = get_uid({
                'cache_version': 2,
                **save_params,
                'terminal': tail.terminal.value,
                'period_weights': weights.tolist(),
            })
            results.append(save_params)
        print("max sample path length:", max_length)
        print("average sample path length:", average_sample_path_length / test_sample_path_num)
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
    results = []
    for (env_uid, experiment_name, mutate_val), variant in test_envs.items():
        init_state_rng = np.random.default_rng(init_state_seed)
        base_env_args = copy.deepcopy(variant['env_args'])
        agent_args = copy.deepcopy(variant.get('agent_args', {}))

        _pin_sampling_seeds(base_env_args, sample_paths_seed)
        sampler_env = _make_sampler_env(base_env_args, init_state_rng)

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

    write_command_file(results, dat_file)
    return results


def generate_penalty_coefficient_training_env(
    test_envs,
    dat_file=None,
    init_state=None,
    sample_path_number=256,
    init_state_seed=12345,
    sample_paths_seed=42,
):
    init_state_mode, normalized_init_state = _normalize_penalty_training_init_state(
        init_state, sample_path_number
    )
    results = []
    for (env_uid, experiment_name, mutate_val), variant in test_envs.items():
        base_env_args = copy.deepcopy(variant['env_args'])
        agent_args = copy.deepcopy(variant.get('agent_args', {}))

        _pin_sampling_seeds(base_env_args, sample_paths_seed)

        save_params = {
            'experiment_name': experiment_name,
            'mutate_val': mutate_val,
            'sample_path_number': sample_path_number,
            'init_state_mode': init_state_mode,
            'init_state': normalized_init_state,
            'init_state_seed': init_state_seed,
            'env_args': base_env_args,
            'agent_args': agent_args,
            'training_generating_function_spec': _normalize_generating_function_spec(
                agent_args.get('agent_args', {}).get('generating_function_spec')
            ),
        }
        save_params['uid'] = get_uid({
            'env_args': base_env_args,
            'agent_args': agent_args,
            'init_state_mode': init_state_mode,
            'init_state': normalized_init_state,
            'init_state_seed': init_state_seed,
            'sample_path_number': sample_path_number,
        })
        results.append(save_params)

    write_command_file(results, dat_file)
    return results


def generate_policy_efficiency_data(
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

        _pin_sampling_seeds(base_env_args, sample_paths_seed)
        sampler_env = _make_sampler_env(base_env_args, init_state_rng)

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

    write_command_file(results, dat_file)
    return results
