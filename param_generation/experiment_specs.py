import copy

from experiments import get_config_by_type
from utils import get_uid

DEFAULT_AGENT_ARGS = {
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


class ExperimentSpec:
    def __init__(self, name, config_type, val_args, mutate=None, agent_mutate=None):
        self.name = name
        self.config_type = config_type
        self.val_args = val_args
        self.mutate = mutate
        self.agent_mutate = agent_mutate

    def is_single_variant(self):
        return len(self.val_args) == 1 and self.val_args[0] is None


def build_variation_test_env(spec):
    base_env_args = get_config_by_type(spec.config_type).args
    test_params = {}
    for val in spec.val_args:
        env_args = copy.deepcopy(base_env_args)
        agent_args = copy.deepcopy(DEFAULT_AGENT_ARGS)
        if spec.mutate is not None:
            spec.mutate(env_args, val)
        if spec.agent_mutate is not None:
            spec.agent_mutate(agent_args, val)
        group_uid = get_uid({'env_args': env_args, 'agent_args': agent_args})
        key = group_uid if spec.is_single_variant() else (group_uid, spec.name, val)
        test_params[key] = {'env_args': env_args, 'agent_args': agent_args}
    return test_params
