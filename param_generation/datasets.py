"""Generators that turn experiment variants into job-array .dat datasets.

Three datasets are produced:

* :func:`generate_test_paths_and_init_state` -- test sample paths + initial
  states with resolved policy specs (grouped command file).
* :func:`generate_train_env` -- (X, Y) training commands for the tight
  penalized lower bound (one command per initial state).
* :func:`generate_policy_efficiency_data` -- policy-efficiency commands
  (one command per initial state).
"""

import copy

import numpy as np

from experiments import get_config_by_type
from importance_sampling import build_proposal
from utils import get_uid, is_single_init_state

from param_generation.caching import (
    load_regression_training_data as _load_regression_training_data,
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
    train_value_function_coefficients as _train_value_function_coefficients,
    zero_penalty_coefficients as _zero_penalty_coefficients,
)


def _pin_sampling_seeds(env_args, seed):
    """Pin arrival/stop-time/env seeds so every initial state for a variant
    trains against the *same* set of arrival sample paths.

    NOTE: in experiment_config.py the env's ``init_state_random_seed`` is
    derived from ``env_random_seed``; we therefore use ``env_random_seed`` ONLY
    for arrival/sampling and override it on a *separate* sampler env (see
    :func:`_make_sampler_env`) when drawing initial states.
    """
    env_args['arrival_random_seed'] = seed
    env_args['stop_time_random_seed'] = seed
    env_args['env_random_seed'] = seed


def offset_sample_generation_seeds(env_args, seed_offset):
    """Return a deepcopy of ``env_args`` whose arrival/stop-time/env seeds are
    shifted by ``seed_offset``, so path generation draws from a random stream
    disjoint from training (and from any other offset)."""
    sample_gen_args = copy.deepcopy(env_args)
    sample_gen_args['env_random_seed'] = env_args.get('env_random_seed', 0) + seed_offset
    sample_gen_args['arrival_random_seed'] = env_args.get('arrival_random_seed', 42) + seed_offset
    sample_gen_args['stop_time_random_seed'] = env_args.get('stop_time_random_seed', 1) + seed_offset
    return sample_gen_args


def _make_sampler_env(base_env_args, init_state_rng):
    """Build a sampler env with a *different* env_random_seed so the
    initial-state RNG is independent of the (pinned) sample-path RNGs."""
    sampler_env_args = copy.deepcopy(base_env_args)
    sampler_env_args['env_random_seed'] = int(init_state_rng.integers(0, 2**31 - 1))
    return get_config_by_type('infinite_custom', args=sampler_env_args).env


def _state_to_jsonable(state):
    """Normalize a single initial state to a 3-list of plain Python lists."""
    return [np.asarray(component).tolist() for component in state]


def _evaluation_period_weights(proposal, discount_factor, tail_length):
    """Per-period importance-sampling weights for the post-warm-up evaluation
    horizon, or ``None`` when no reweighting is needed.

    When ``proposal`` is ``None`` the tail was drawn from the target geometric
    horizon by the legacy path and no reweighting is applied, so we return
    ``None`` to keep the evaluation byte-identical to the
    non-importance-sampling path.

    Otherwise the tail was drawn from ``proposal``. Rolling the policy over
    ``tail_length`` sampled arrivals visits ``tail_length + 1`` decision
    periods (the trailing period carries a stage cost but no arrival), so
    period ``s`` is visited iff the sampled length ``L >= s - 1`` and its
    unbiased weight is ``gamma ** (s - 1) / P_proposal(L >= s - 1)``. In terms
    of the proposal's per-period ratios ``u_t = gamma ** (t - 1) / P(L >= t)``
    this is ``w_1 = 1`` and ``w_s = gamma * u_{s-1}`` for ``s >= 2``; deriving
    the weights from ``period_likelihood_ratios`` keeps its validation (gamma
    match, positive survival). For a fixed-length proposal the weights are
    unchanged (``gamma ** (s - 1)``). The agent-side use of
    ``period_likelihood_ratios`` intentionally differs: its scenarios weight
    exactly ``L`` periods, for which dividing by ``P(L >= s)`` is correct.
    """
    if proposal is None:
        return None
    num_periods = tail_length + 1
    unshifted = proposal.period_likelihood_ratios(
        target_discount_factor=discount_factor,
        lengths=[num_periods],
    )[0]
    return [1.0] + [float(discount_factor * w) for w in unshifted[:-1]]


def _normalize_penalty_training_init_state(init_state, sample_path_number):
    """Validate and JSON-normalize the ``init_state`` argument of
    :func:`generate_penalty_coefficient_training_env`.

    Returns ``(init_state_mode, normalized_init_state)`` where:

      * ``('generate', None)`` -- ``init_state is None``; the runner draws one
        initial state per scenario from a seeded sampler env.
      * ``('shared', state)`` -- a single state shared across all scenarios.
      * ``('per_scenario', [state, ...])`` -- one state per scenario; the list
        length must equal ``sample_path_number``.
    """
    if init_state is None:
        return 'generate', None
    if is_single_init_state(init_state):
        return 'shared', _state_to_jsonable(init_state)
    # Otherwise it must be a per-scenario sequence of states.
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



def generate_test_paths_and_init_state(test_envs, test_sample_path_num, warm_up_periods=0, num_periods=None, dat_file=None, num_groups=None, is_require_penalty_coefficients=True, is_random_initial_state=False, policy_ids=None, train_data_dir=None, warm_up_policy_id=None, evaluation_proposal_spec=None, penalty_coefficients_dir=None, warm_up_paths=None, sample_gen_seed_offset=1001):
    '''
    Inital state is considered as period 1. sample path will start from period 2.

    ``policy_ids`` is an optional iterable of policy_ids registered in
    ``POLICY_SPECS``. The resolved policy spec dicts are embedded in every
    saved params record under the ``policies`` key so the runner knows which
    policies to evaluate against the corresponding sample path.

    ``train_data_dir`` is an optional path to a folder of training-result JSONL
    files holding (X, Y) = (``init_state``, ``tight_penalized_lower_bound``)
    pairs. When provided and ``'approx_Q'`` is in ``policy_ids``, the approx_Q
    value-function coefficients are fitted by least-squares regression on that
    data (instead of the Benders ``direct_coefficients``) and written into the
    approx_Q ``policy_generating_function_spec``.

    ``warm_up_policy_id`` optionally names one of ``policy_ids`` (e.g.
    ``'row_gen_alp'``) as the shared warm-up policy. By default every policy
    warms itself up over the first ``warm_up_periods`` periods of its sample
    path before costs are counted. With ``warm_up_policy_id`` set, the runner
    instead rolls ONLY that policy over the warm-up prefix of each sample path
    and starts every other policy from the resulting per-path warm-up state,
    evaluating them on the post-warm-up tail only. This makes all policies
    start from the warm-up policy's steady state.

    ``evaluation_proposal_spec`` optionally decouples the EVALUATION-path
    proposal from the agent's own IS proposal: when given (a
    ``build_proposal``-style spec dict), the post-warm-up tails and
    ``period_weights`` are drawn from it while the embedded ``policy_specs``
    keep the untouched agent proposal. When ``None`` the tails fall back to
    the agent's spec (legacy coupled behavior).

    ``penalty_coefficients_dir`` optionally points the trained-coefficient
    lookup at another experiment's results folder (passed through as
    ``folder_path``), so a new experiment name can reuse coefficients without
    copying files or retraining.

    ``warm_up_paths`` optionally supplies one pre-drawn warm-up arrival prefix
    (shape ``(warm_up_periods, num_types)``) per sample path. Use it to make
    several experiments share byte-identical warm-up prefixes while their
    post-warm-up tails are still drawn per experiment. Only supported when
    ``num_periods`` is None.

    ``sample_gen_seed_offset`` shifts the arrival/stop-time/env seeds used for
    path generation (default 1001, the historical offset). Give different
    experiments different offsets so their evaluation tails come from disjoint
    random streams.
    '''
    policy_ids = list(policy_ids or [])
    policy_id_set = set(policy_ids)
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
        # Policy basis: prefer an explicit ``policy_generating_function_spec``;
        # otherwise fall back to the agent's ``generating_function_spec`` so the
        # spec configured in ``build_variation_test_env`` actually drives the
        # approx_Q / hindsight policy basis.
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

        direct_coefficients = _zero_penalty_coefficients(env)
        if is_require_penalty_coefficients:
            sample_path_number = variant.get('agent_args', {}).get('agent_args', {}).get('sample_path_number')
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
            approx_q_coefficients = direct_coefficients
            approx_q_generating_function_spec = policy_generating_function_spec
            if train_data_dir is not None:
                X, Y = _load_regression_training_data(train_data_dir)
                approx_q_coefficients = _train_value_function_coefficients(
                    env=env,
                    generating_function_spec=policy_generating_function_spec,
                    X=X,
                    Y=Y,
                )
                # Update coefficients in the approx_Q policy generating function spec.
                approx_q_generating_function_spec = {
                    **policy_generating_function_spec,
                    'coefficients': approx_q_coefficients,
                }
            policies.append(
                _build_penalty_policy(
                    base_agent_args=variant['agent_args'],
                    policy_id='approx_Q',
                    solver_name='approx_Q',
                    penalty_coefficients=approx_q_coefficients,
                    generating_function_spec=approx_q_generating_function_spec,
                )
            )

        # Inject the freshly trained coefficients into the matching policy specs
        # for this variant so each saved record carries everything the runner
        # needs to instantiate its agents.
        max_length = 0
        # Path-generation seeds are offset from the training seeds so evaluation
        # paths stay independent of the ALP/penalty training paths; distinct
        # offsets keep different experiments' tails independent of each other.
        sample_gen_args = offset_sample_generation_seeds(env_args, sample_gen_seed_offset)
        config_for_sample_path = get_config_by_type('infinite_custom', args=sample_gen_args)
        env_for_sample_path = config_for_sample_path.env
        # Draw the post-warm-up evaluation tails from ``evaluation_proposal_spec``
        # when given, else from the SAME importance-sampling proposal used for
        # penalty-coefficient training. Falls back to the target geometric
        # horizon (``build_proposal(None) is None``) when no proposal is
        # configured, keeping the legacy behaviour unchanged.
        eval_proposal_spec = (
            evaluation_proposal_spec
            if evaluation_proposal_spec is not None
            else inner_agent_args.get('sample_path_length_proposal')
            or inner_agent_args.get('sample_path_proposal')
        )
        sample_path_proposal = build_proposal(eval_proposal_spec)
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
            proposal_tails, _ = sample_path_proposal.sample_arrival_paths(
                arrival_generator=env_for_sample_path.arrival_generator,
                size=test_sample_path_num,
            )
            # Record order equals proposal order (single batch draw), so the
            # per-path stratum weights align positionally with the tails.
            try:
                path_weights = sample_path_proposal.path_weights(test_sample_path_num)
                path_strata = sample_path_proposal.path_strata(test_sample_path_num)
            except ValueError as exc:
                # Degenerate allocation (e.g. a tiny smoke run that cannot
                # represent every stratum): emit unweighted records so the
                # plumbing keeps working; such runs are not statistically
                # meaningful either way.
                print(f"WARNING: proposal path weights unavailable ({exc}); "
                      "records will aggregate equal-weight.")
                path_weights = path_strata = None
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
                )
            else:
                sample_path = env_for_sample_path.reset_arrivals(stop_time=num_periods)
                period_weights = None
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
                "period_weights": period_weights,
                "path_weight": None if path_weights is None else float(path_weights[path_index]),
                "path_stratum": None if path_strata is None else int(path_strata[path_index]),
                "generating_function_spec": {**lowerbound_generating_function_spec, 'coefficients': direct_coefficients},
                "group_id": env_uid,
                "policy_specs": policies,
            }
            if warm_up_policy_id is not None:
                save_params['warm_up_policy_id'] = warm_up_policy_id
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
    """Generate penalty-coefficient training commands (one command per variant).

    Unlike :func:`generate_train_env` -- which sweeps many initial states and
    emits one command each to build an ``(X, Y)`` regression dataset -- this
    emits a *single* command per variant in ``test_envs``. Each command trains
    one set of penalty coefficients by Benders decomposition over
    ``sample_path_number`` arrival sample paths. The sample paths are held fixed
    across commands by pinning the arrival/stop-time/env seeds to
    ``sample_paths_seed`` (``reset_random_seeds()`` is called inside the trainer
    before sampling).

    ``init_state`` controls the starting state of the training scenarios:

      * ``None`` (default): the runner (``run.py``) draws one initial state per
        scenario from the env's reference distribution, seeded by
        ``init_state_seed`` for reproducibility ("let the solver generate"). The
        emitted record carries ``init_state_seed`` so the draw is reproducible.
      * a single initial state -- a 3-tuple ``(regular, overtime, waitlist)`` of
        per-class arrays: every scenario starts from this same state.
      * a list of initial states with length ``sample_path_number``: scenario
        ``i`` starts from ``init_state[i]``.

    The emitted records are consumed by ``train_penalty_coefficients_for_env``
    in ``run.py`` (dispatched on the ``init_state_mode`` key).
    """
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
