"""Backward-compatible entry point for parameter/experiment generation.

The implementation now lives in the :mod:`param_generation` package. This thin
shim re-exports the public API and the CLI entry point so that existing imports
(e.g. ``from generate_params import generate_test_paths_and_init_state`` or the
private cache helpers used by ``run.py``) and ``python generate_params.py`` keep
working unchanged.
"""

from param_generation import (
    EXPERIMENT_SPECS,
    build_variation_test_env,
    generate_experiment,
    generate_policy_efficiency_data,
    generate_test_paths_and_init_state,
    generate_train_env,
    train_alp_coefficients,
    train_penalty_coefficients,
)
from param_generation.caching import (
    load_cached_training_result as _load_cached_training_result,
    save_training_result as _save_training_result,
)
from param_generation.cli import main

__all__ = [
    'EXPERIMENT_SPECS',
    'build_variation_test_env',
    'generate_experiment',
    'generate_policy_efficiency_data',
    'generate_test_paths_and_init_state',
    'generate_train_env',
    'train_alp_coefficients',
    'train_penalty_coefficients',
    'main',
]



if __name__ == '__main__':
    main()
