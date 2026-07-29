"""Command-line recipes for generating experiment parameters and data files.

Each ``recipe_*`` function is a self-contained driver that builds the relevant
test environments and writes the corresponding ``.dat`` command file. ``main``
runs the recipe that is currently active for the project; switch the call in
``main`` (or invoke a recipe directly) to produce a different dataset.
"""

import os
import shutil
from pprint import pprint

from scipy.stats import geom

from param_generation.caching import load_trained_coefficients_from_folder
from param_generation.command_files import write_grouped_command_file
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
        EXPERIMENT_SPECS['toy_study_099_mixture_geometric_proposal_095_test']
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
    test_envs = build_variation_test_env(EXPERIMENT_SPECS['toy_study_099_mixture_geometric_proposal_095_test'])
    generate_test_paths_and_init_state(
        test_envs=test_envs,
        test_sample_path_num=1,
        warm_up_periods=5,
        num_periods=None,
        dat_file='table.dat',
        num_groups=998,  # divide into N groups
        is_require_penalty_coefficients=True,
        policy_ids=['approx_penalized_hindsight', 'row_gen_alp', 'myopic'],
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
        EXPERIMENT_SPECS['case_study_099_mixture_geometric_proposal_095_overtime_50']
    )
    generate_penalty_coefficient_training_env(
        test_envs,
        dat_file='table.dat',
        init_state=None,
        sample_path_number=256,
        init_state_seed=12345,
        sample_paths_seed=42,
    )

def recipe_case_study_099_mixture_geometric_proposal_095_policy_evaluation():
    """IS training recipe: proposal gamma=0.98, target gamma=0.99.

    Builds an env with ``discount_factor=0.99`` and an agent with a geometric IS
    proposal of 0.98. Because ``is_require_penalty_coefficients=True``,
    ``train_penalty_coefficients`` runs for each variant and constructs
    ``ApproxQAgent(sample_path_proposal=GeometricLengthProposal(0.98))`` so every
    Benders subproblem objective is weighted by ``(0.99/0.98)**(t-1)``.
    """
    test_envs = build_variation_test_env(EXPERIMENT_SPECS['case_study_099_mixture_geometric_proposal_095'])
    generate_test_paths_and_init_state(
        test_envs=test_envs,
        test_sample_path_num=2,
        warm_up_periods=300,
        num_periods=None,
        dat_file='table.dat',
        num_groups=500,  # divide into N groups
        is_require_penalty_coefficients=True,
        is_random_initial_state=True,
        policy_ids=['approx_penalized_hindsight'],
    )

def recipe_toy_study_train_env():
    """Train penalty coefficients on the case study with a mixture-geometric IS proposal.

    Sweeps the mixture mass ``lambda_0 in {0.05, 0.1, 0.15}`` on the long
    (target gamma=0.99) component; the short component uses proposal
    discount factor q=0.95. Because ``is_require_penalty_coefficients=True``,
    ``train_penalty_coefficients`` runs per variant and weights each Benders
    subproblem by the mixture's per-period likelihood ratio (bounded in
    ``[1, 1 / lambda_0]``), yielding cheaper-but-unbiased training paths.
    """
    test_envs = build_variation_test_env(
        EXPERIMENT_SPECS['base_toy_study']
    )
    generate_penalty_coefficient_training_env(
        test_envs,
        dat_file='table.dat',
        init_state=None,
        sample_path_number=256,
        init_state_seed=12345,
        sample_paths_seed=42,
    )

def recipe_toy_study_policy_evaluation():
    """Train penalty coefficients on the case study with a mixture-geometric IS proposal.

    Sweeps the mixture mass ``lambda_0 in {0.05, 0.1, 0.15}`` on the long
    (target gamma=0.99) component; the short component uses proposal
    discount factor q=0.95. Because ``is_require_penalty_coefficients=True``,
    ``train_penalty_coefficients`` runs per variant and weights each Benders
    subproblem by the mixture's per-period likelihood ratio (bounded in
    ``[1, 1 / lambda_0]``), yielding cheaper-but-unbiased training paths.
    """
    test_envs = build_variation_test_env(
        EXPERIMENT_SPECS['base_toy_study']
    )
    generate_test_paths_and_init_state(
            test_envs=test_envs,
            test_sample_path_num=4096,
            warm_up_periods=0,
            num_periods=None,
            dat_file='table.dat',
            num_groups=500,  # divide into N groups
            is_require_penalty_coefficients=True,
            is_random_initial_state=False,
            policy_ids=['approx_penalized_hindsight','row_gen_alp', 'myopic'],
        )

def recipe_steady_state_toy_study_policy_evaluation():
    """Train penalty coefficients on the case study with a mixture-geometric IS proposal.

    Sweeps the mixture mass ``lambda_0 in {0.05, 0.1, 0.15}`` on the long
    (target gamma=0.99) component; the short component uses proposal
    discount factor q=0.95. Because ``is_require_penalty_coefficients=True``,
    ``train_penalty_coefficients`` runs per variant and weights each Benders
    subproblem by the mixture's per-period likelihood ratio (bounded in
    ``[1, 1 / lambda_0]``), yielding cheaper-but-unbiased training paths.
    """
    test_envs = build_variation_test_env(
        EXPERIMENT_SPECS['steady_state_toy_study']
    )
    generate_test_paths_and_init_state(
            test_envs=test_envs,
            test_sample_path_num=4096,
            warm_up_periods=100,
            num_periods=None,
            dat_file='table.dat',
            num_groups=500,  # divide into N groups
            is_require_penalty_coefficients=True,
            is_random_initial_state=False,
            policy_ids=['approx_penalized_hindsight','row_gen_alp', 'myopic'],
        )
def recipe_alp_steady_state_toy_study_policy_evaluation():
    """Evaluate all policies from the ALP per-sample-path steady state.

    Same env as ``steady_state_toy_study`` (toy env, initial-state congestion
    0.5), but with ``warm_up_policy_id='row_gen_alp'``: the runner rolls ONLY
    the row-generation ALP policy over the first ``warm_up_periods`` periods of
    each sample path, then starts every policy (including ALP itself) from the
    resulting warm-up state and accumulates costs on the post-warm-up tail.
    This isolates steady-state policy performance from the warm-up transient of
    each individual policy.
    """
    experiment_name = 'alp_steady_state_toy_study'
    # The env is identical to steady_state_toy_study, so its trained ALP and
    # penalty coefficients apply verbatim; copy the caches over (if present) so
    # generation does not retrain them from scratch.
    source_dir = os.path.join('experiments', 'results', 'steady_state_toy_study')
    target_dir = os.path.join('experiments', 'results', experiment_name)
    os.makedirs(target_dir, exist_ok=True)
    for cache_file in ('alp_train.jsonl', 'penalty_coefficients.jsonl'):
        source_path = os.path.join(source_dir, cache_file)
        target_path = os.path.join(target_dir, cache_file)
        if os.path.isfile(source_path) and not os.path.isfile(target_path):
            shutil.copyfile(source_path, target_path)

    test_envs = build_variation_test_env(EXPERIMENT_SPECS[experiment_name])
    generate_test_paths_and_init_state(
            test_envs=test_envs,
            test_sample_path_num=4096,
            warm_up_periods=100,
            num_periods=None,
            dat_file='table.dat',
            num_groups=500,  # divide into N groups
            is_require_penalty_coefficients=True,
            is_random_initial_state=False,
            policy_ids=['approx_penalized_hindsight', 'row_gen_alp', 'myopic'],
            warm_up_policy_id='row_gen_alp',
        )

def recipe_mixture_probability_toy_study_train_env():
    """Train penalty coefficients on the case study with a mixture-geometric IS proposal.

    Sweeps the mixture mass ``lambda_0 in {0.05, 0.1, 0.15}`` on the long
    (target gamma=0.99) component; the short component uses proposal
    discount factor q=0.95. Because ``is_require_penalty_coefficients=True``,
    ``train_penalty_coefficients`` runs per variant and weights each Benders
    subproblem by the mixture's per-period likelihood ratio (bounded in
    ``[1, 1 / lambda_0]``), yielding cheaper-but-unbiased training paths.
    """
    test_envs = build_variation_test_env(
        EXPERIMENT_SPECS['mixture_probability_toy_study']
    )
    generate_penalty_coefficient_training_env(
        test_envs,
        dat_file='table.dat',
        init_state=None,
        sample_path_number=256,
        init_state_seed=12345,
        sample_paths_seed=42,
    )

def recipe_mixture_probability_toy_study_policy_evaluation():
    """Train penalty coefficients on the case study with a mixture-geometric IS proposal.

    Sweeps the mixture mass ``lambda_0 in {0.05, 0.1, 0.15}`` on the long
    (target gamma=0.99) component; the short component uses proposal
    discount factor q=0.95. Because ``is_require_penalty_coefficients=True``,
    ``train_penalty_coefficients`` runs per variant and weights each Benders
    subproblem by the mixture's per-period likelihood ratio (bounded in
    ``[1, 1 / lambda_0]``), yielding cheaper-but-unbiased training paths.
    """
    test_envs = build_variation_test_env(
        EXPERIMENT_SPECS['mixture_probability_toy_study']
    )
    generate_test_paths_and_init_state(
            test_envs=test_envs,
            test_sample_path_num=4096,
            warm_up_periods=0,
            num_periods=None,
            dat_file='table.dat',
            num_groups=500,  # divide into N groups
            is_require_penalty_coefficients=True,
            is_random_initial_state=False,
            policy_ids=['approx_penalized_hindsight'],
        )

def recipe_case_study_099_mixture_geometric_proposal_095_overtime_50_train_env():
    test_envs = build_variation_test_env(
        EXPERIMENT_SPECS['case_study_099_mixture_geometric_proposal_095_overtime_50']
    )
    generate_penalty_coefficient_training_env(
        test_envs,
        dat_file='table.dat',
        init_state=None,
        sample_path_number=256,
        init_state_seed=12345,
        sample_paths_seed=42,
    )

def recipe_case_study_099_mixture_geometric_proposal_095_overtime_50_policy_evaluation():
    test_envs = build_variation_test_env(
        EXPERIMENT_SPECS['case_study_099_mixture_geometric_proposal_095_overtime_50']
    )
    generate_test_paths_and_init_state(
        test_envs=test_envs,
        test_sample_path_num=4096,
        warm_up_periods=300,
        num_periods=None,
        dat_file='table.dat',
        num_groups=500,  # divide into N groups
        is_require_penalty_coefficients=True,
        is_random_initial_state=True,
        policy_ids=['approx_penalized_hindsight'],
    )

def recipe_toy_eval_proposal_comparison(
    test_sample_path_num=4096,
    num_groups=500,
    dat_file='table.dat',
    penalty_coefficients_dir=os.path.join(
        'experiments', 'results', 'mixture_probability_toy_study'
    ),
):
    """Evaluation-proposal comparison on the toy env (spec:
    docs/superpowers/specs/2026-07-29-eval-proposal-comparison-design.md).

    ONE fixed approx_penalized_hindsight policy (internal mixture IS proposal
    lambda_0=0.1, trained coefficients reused from
    ``mixture_probability_toy_study``) is evaluated on tails drawn from three
    different length proposals; each record carries the matching
    ``period_weights`` so ``run.py`` stays unbiased per arm. All
    3 x ``test_sample_path_num`` records go into ONE grouped dat file.
    """
    # Fail fast if the trained coefficients are missing: silently retraining
    # per arm would be slow and would break the fixed-policy comparison.
    coefficients = load_trained_coefficients_from_folder(
        experiment_name='mixture_probability_toy_study',
        mutate_val=0.1,
        sample_path_number=256,
        folder_path=penalty_coefficients_dir,
    )
    if coefficients is None:
        raise RuntimeError(
            'No trained penalty coefficients (mutate_val=0.1, '
            f'sample_path_number=256) found in {penalty_coefficients_dir}.'
        )

    # 459 = 0.99-quantile of Geom(1 - 0.99). The fixed proposal samples
    # max_length = 458 arrivals so the rollout visits exactly 459 weighted
    # decision periods (see the spec: FixedLengthProposal(459) would raise in
    # _evaluation_period_weights).
    fixed_horizon = int(geom.ppf(0.99, 1 - 0.99))
    assert fixed_horizon == 459, fixed_horizon

    arms = [
        ('toy_eval_proposal_geometric_099',
         {'type': 'geometric', 'discount_factor_proposal': 0.99}),
        ('toy_eval_proposal_fixed_459',
         {'type': 'fixed', 'max_length': fixed_horizon - 1}),
        ('toy_eval_proposal_mixture_095_l01',
         {'type': 'mixture_geometric', 'target_discount_factor': 0.99,
          'discount_factor_proposal': 0.95, 'lambda_0': 0.1}),
    ]
    all_records = []
    for experiment_name, evaluation_proposal_spec in arms:
        test_envs = build_variation_test_env(EXPERIMENT_SPECS[experiment_name])
        all_records.extend(
            generate_test_paths_and_init_state(
                test_envs=test_envs,
                test_sample_path_num=test_sample_path_num,
                warm_up_periods=0,
                num_periods=None,
                dat_file=None,
                is_require_penalty_coefficients=True,
                is_random_initial_state=False,
                policy_ids=['row_gen_alp'],
                evaluation_proposal_spec=evaluation_proposal_spec,
                penalty_coefficients_dir=penalty_coefficients_dir,
            )
        )
    write_grouped_command_file(
        results=all_records, num_groups=num_groups, dat_file=dat_file
    )
    return all_records


def main():
    """Run the currently-active dataset-generation recipe."""
    recipe_toy_eval_proposal_comparison()
    #recipe_case_study_099_mixture_geometric_proposal_095_overtime_50_policy_evaluation()
    #recipe_case_study_099_mixture_geometric_proposal_095_policy_evaluation()


if __name__ == '__main__':
    main()
