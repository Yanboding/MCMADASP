"""Coefficient training for penalty / ALP / value-function policies."""

from pprint import pprint

import numpy as np
from gurobipy import GRB

from decision_maker import ALPRowGenerationAgent, ApproxQAgent
from experiments import get_config_by_type
from importance_sampling import build_proposal
from utils import get_uid

from param_generation.caching import (
    load_cached_training_result as _load_cached_training_result,
    save_training_result as _save_training_result,
)
from param_generation.generating_functions import (
    build_generating_function as _build_generating_function,
    normalize_generating_function_spec as _normalize_generating_function_spec,
)


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
        'generating_function_spec',
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


def zero_penalty_coefficients(env):
    return [0] * (
        env.planning_horizon * 2
        + env.num_types
        + env.booking_window_size * env.num_types
        + env.planning_horizon
    )


def train_value_function_coefficients(env, generating_function_spec, X, Y, regularization=1e-6):
    """Fit approx_Q value-function coefficients on (X, Y) via least squares.

    Builds the basis named by ``generating_function_spec`` and fits its
    coefficients with :meth:`ApproxQAgent.regression_train`. Returns the fitted
    coefficients as a plain list (JSON-serializable).
    """
    spec = _normalize_generating_function_spec(generating_function_spec)
    generating_function = _build_generating_function(env=env, spec={'name': spec['name']})
    agent = ApproxQAgent(
        env=env,
        discount_factor=env.discount_factor,
        sample_path_number=1,
        generating_function=generating_function,
        solver_name='approx_Q',
    )
    coefficients, training_mse = agent.regression_train(X, Y, regularization=regularization)
    print(
        f"approx_Q value-function regression on '{spec['name']}' basis: "
        f"{len(X)} samples, training RMSE={np.sqrt(training_mse):.4f}"
    )
    return coefficients
