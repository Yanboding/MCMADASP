"""Command-line recipes for generating experiment parameters and data files.

Each ``recipe_*`` function is a self-contained driver that builds the relevant
test environments and writes the corresponding ``.dat`` command file. ``main``
runs the recipe that is currently active for the project; switch the call in
``main`` (or invoke a recipe directly) to produce a different dataset.
"""

import os
from pprint import pprint

from param_generation.datasets import (
    generate_penalty_coefficient_training_env,
    generate_policy_efficiency_data,
    generate_test_paths_and_init_state,
    generate_train_env,
)
from param_generation.experiment_specs import build_variation_test_env
from param_generation.registry import EXPERIMENT_SPECS


def recipe_case_study_099_fixed_length():
    """Generate test paths + initial states for ``case_study_099_fixed_length``.

    Penalty coefficients are trained per variant (importance-sampling aware),
    initial states are randomised, and the cases are written to ``table.dat``
    split into 998 groups.
    """
    folder_path = 'case_study_099_fixed_length'
    test_envs = build_variation_test_env(EXPERIMENT_SPECS[folder_path])
    print(test_envs)
    generate_test_paths_and_init_state(
        test_envs=test_envs,
        test_sample_path_num=5000,
        warm_up_periods=0,
        num_periods=None,
        dat_file='table.dat',
        num_groups=998,  # divide into N groups
        is_require_penalty_coefficients=True,
        is_random_initial_state=True,
        policy_ids=[],
        train_data_dir=os.path.join('experiments', 'results', folder_path),
    )


def recipe_all_experiments_test_envs():
    """Build (without emitting) the test environments for every experiment."""
    test_envs = {}
    for experiment_name in EXPERIMENT_SPECS:
        test_envs.update(build_variation_test_env(EXPERIMENT_SPECS[experiment_name]))
    return test_envs


def recipe_importance_sampling_098():
    """IS training recipe: proposal gamma=0.98, target gamma=0.99.

    Builds an env with ``discount_factor=0.99`` and an agent with a geometric IS
    proposal of 0.98. Because ``is_require_penalty_coefficients=True``,
    ``train_penalty_coefficients`` runs for each variant and constructs
    ``ApproxQAgent(sample_path_proposal=GeometricLengthProposal(0.98))`` so every
    Benders subproblem objective is weighted by ``(0.99/0.98)**(t-1)``.
    """
    test_envs = build_variation_test_env(EXPERIMENT_SPECS['case_study_099_is_098'])
    generate_test_paths_and_init_state(
        test_envs=test_envs,
        test_sample_path_num=1,
        warm_up_periods=750,
        num_periods=None,
        dat_file='table.dat',
        num_groups=998,  # divide into N groups
        is_require_penalty_coefficients=True,
        policy_ids=['approx_Q', 'row_gen_alp', 'myopic'],
    )


def recipe_case_study_099_mixture_geometric_proposal_095():
    """Train penalty coefficients on the case study with a mixture-geometric IS proposal.

    Sweeps the mixture mass ``lambda_0 in {0.05, 0.1, 0.15}`` on the long
    (target gamma=0.99) component; the short component uses proposal
    discount factor q=0.95. Because ``is_require_penalty_coefficients=True``,
    ``train_penalty_coefficients`` runs per variant and weights each Benders
    subproblem by the mixture's per-period likelihood ratio (bounded in
    ``[1, 1 / lambda_0]``), yielding cheaper-but-unbiased training paths.
    """
    test_envs = build_variation_test_env(
        EXPERIMENT_SPECS['case_study_099_mixture_geometric_proposal_095']
    )
    generate_penalty_coefficient_training_env(
        test_envs,
        dat_file='table.dat',
        init_state=None,
        sample_path_number=256,
        init_state_seed=12345,
        sample_paths_seed=42,
    )
    


def recipe_toy_study_train_env():
    """Emit the regression training environments for ``toy_study_base_case``.

    Fits approx_Q value-function coefficients by least-squares regression on the
    (X, Y) training data in ``experiments/results/toy_study_train``
    (X=init_state, Y=tight_penalized_lower_bound) and embeds them in the
    approx_Q ``policy_generating_function_spec`` before emitting the test cases.
    """
    test_envs = build_variation_test_env(EXPERIMENT_SPECS['toy_study_base_case'])
    generate_train_env(
        test_envs=test_envs,
        dat_file='table.dat',
        num_init_states=256,
        sample_path_number=256,
        init_state_seed=12345,
        sample_paths_seed=42,
    )


def recipe_policy_efficiency():
    """Generate policy-efficiency evaluation data for ``toy_study_base_case``."""
    test_envs = build_variation_test_env(EXPERIMENT_SPECS['toy_study_base_case'])
    generate_policy_efficiency_data(
        test_envs=test_envs,
        is_require_penalty_coefficients=False,
        dat_file='table.dat',
        num_init_states=1,
        sample_path_number=256,
        init_state_seed=12345,
        sample_paths_seed=42,
    )

def recipe_toy_study_099_mixture_geometric_proposal_095():
    """Train penalty coefficients on the case study with a mixture-geometric IS proposal.

    Sweeps the mixture mass ``lambda_0 in {0.05, 0.1, 0.15}`` on the long
    (target gamma=0.99) component; the short component uses proposal
    discount factor q=0.95. Because ``is_require_penalty_coefficients=True``,
    ``train_penalty_coefficients`` runs per variant and weights each Benders
    subproblem by the mixture's per-period likelihood ratio (bounded in
    ``[1, 1 / lambda_0]``), yielding cheaper-but-unbiased training paths.
    """
    test_envs = build_variation_test_env(
        EXPERIMENT_SPECS['toy_study_099_mixture_geometric_proposal_095']
    )
    generate_penalty_coefficient_training_env(
        test_envs,
        dat_file='table.dat',
        init_state=None,
        sample_path_number=256,
        init_state_seed=12345,
        sample_paths_seed=42,
    )

def recipe_toy_study_099_mixture_geometric_proposal_095_policy_evaluation():
    """IS training recipe: proposal gamma=0.98, target gamma=0.99.

    Builds an env with ``discount_factor=0.99`` and an agent with a geometric IS
    proposal of 0.98. Because ``is_require_penalty_coefficients=True``,
    ``train_penalty_coefficients`` runs for each variant and constructs
    ``ApproxQAgent(sample_path_proposal=GeometricLengthProposal(0.98))`` so every
    Benders subproblem objective is weighted by ``(0.99/0.98)**(t-1)``.
    """
    test_envs = build_variation_test_env(EXPERIMENT_SPECS['toy_study_099_mixture_geometric_proposal_095'])
    generate_test_paths_and_init_state(
        test_envs=test_envs,
        test_sample_path_num=5000,
        warm_up_periods=None,
        num_periods=None,
        dat_file='table.dat',
        num_groups=998,  # divide into N groups
        is_require_penalty_coefficients=True,
        policy_ids=['approx_penalized_hindsight', 'row_gen_alp', 'myopic'],
    )

def main():
    """Run the currently-active dataset-generation recipe."""
    recipe_toy_study_099_mixture_geometric_proposal_095_policy_evaluation()


if __name__ == '__main__':
    main()
