import copy

from param_generation.generating_functions import (
    normalize_generating_function_spec as _normalize_generating_function_spec,
)


def build_penalty_policy(base_agent_args, policy_id, solver_name, penalty_coefficients, generating_function_spec=None):
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
    spec = _normalize_generating_function_spec(generating_function_spec)
    spec['coefficients'] = penalty_coefficients
    policy['agent_args']['generating_function_spec'] = spec
    return policy


def build_atomic_policy(training_record, policy_id='atomic_ppo', action_selection='greedy', seed=None):
    """Policy spec of the atomic policy that the training record trains. run.py loads its file by uid."""
    if action_selection not in ('greedy', 'sample'):
        raise ValueError(f"action_selection must be 'greedy' or 'sample'; got {action_selection!r}")
    agent_args = {
        'training_experiment_name': training_record['experiment_name'],
        'training_uid': training_record['uid'],
        'action_selection': action_selection,
    }
    if action_selection == 'sample':
        agent_args['seed'] = 0 if seed is None else int(seed)
    return {'policy_id': policy_id, 'agent_name': 'atomic_ppo', 'agent_args': agent_args}
