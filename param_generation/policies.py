"""Builders for runtime policy specs embedded in saved params records."""

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
    # Self-contained spec: embed the policy's penalty coefficients in the spec.
    spec = _normalize_generating_function_spec(generating_function_spec)
    spec['coefficients'] = penalty_coefficients
    policy['agent_args']['generating_function_spec'] = spec
    return policy
