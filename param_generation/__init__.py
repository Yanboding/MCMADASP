"""Parameter-generation pipeline for MCMADASP experiments.

This package splits the former monolithic ``generate_params.py`` into cohesive
modules:

* :mod:`param_generation.generating_functions` -- penalty/basis spec handling.
* :mod:`param_generation.caching` -- training-result cache + regression-data I/O.
* :mod:`param_generation.command_files` -- ``run.py`` job-array ``.dat`` writers.
* :mod:`param_generation.mutators` -- per-experiment ``env_args``/``agent_args`` mutators.
* :mod:`param_generation.experiment_specs` -- ``ExperimentSpec`` + variant builder.
* :mod:`param_generation.registry` -- the ``EXPERIMENT_SPECS`` registry.
* :mod:`param_generation.training` -- penalty/ALP/value-function training.
* :mod:`param_generation.policies` -- policy-spec assembly.
* :mod:`param_generation.datasets` -- the top-level ``generate_*`` entry points.
* :mod:`param_generation.cli` -- runnable dataset-generation recipes.
"""

from param_generation.experiment_specs import (
    DEFAULT_AGENT_ARGS,
    ExperimentSpec,
    build_variation_test_env,
)
from param_generation.registry import EXPERIMENT_SPECS, generate_experiment
from param_generation.datasets import (
    generate_penalty_coefficient_training_env,
    generate_policy_efficiency_data,
    generate_test_paths_and_init_state,
    generate_train_env,
)
from param_generation.training import (
    train_alp_coefficients,
    train_penalty_coefficients,
)

__all__ = [
    'DEFAULT_AGENT_ARGS',
    'ExperimentSpec',
    'build_variation_test_env',
    'EXPERIMENT_SPECS',
    'generate_experiment',
    'generate_penalty_coefficient_training_env',
    'generate_policy_efficiency_data',
    'generate_test_paths_and_init_state',
    'generate_train_env',
    'train_alp_coefficients',
    'train_penalty_coefficients',
]
