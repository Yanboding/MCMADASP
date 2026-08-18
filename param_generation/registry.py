"""Registry of all experiments. Add new experiments here.

Maps each experiment ``name`` to its :class:`ExperimentSpec`. ``generate_experiment``
is the preferred entry point: look up a spec by name and materialize its variants.
"""

from param_generation.experiment_specs import ExperimentSpec, build_variation_test_env
from param_generation.mutators import (
    mutate_initial_state_congestion,
    mutate_initial_state_congestion_05_const,
    mutate_mixture_geometric_proposal_lambda_0,
    mutate_mixture_target_discount_factor,
    mutate_mixture_target_discount_factor_overtime_5,
    mutate_mixture_target_discount_factor_overtime_50,
    mutate_sample_path_number_mixture_geometric_l01,
)

EXPERIMENT_SPECS = {
    spec.name: spec for spec in [
        # 0.99-target case study, mixture-geometric IS proposal (lambda_0=0.1).
        ExperimentSpec(
            name='case_study_099_mixture_geometric_proposal_095',
            config_type='ejor',
            val_args=[0.1],
            mutate=mutate_mixture_target_discount_factor,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
        ExperimentSpec(
            name='case_study_099_mixture_geometric_proposal_095_overtime_50',
            config_type='ejor',
            val_args=[0.1],
            mutate=mutate_mixture_target_discount_factor_overtime_50,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
        ExperimentSpec(
            name='case_study_099_mixture_geometric_proposal_095_overtime_5',
            config_type='ejor',
            val_args=[0.1],
            mutate=mutate_mixture_target_discount_factor_overtime_5,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
        # Scenario-count sweep: how the Benders scenario count drives the
        # per-iteration 95% CI of the subproblem objectives. val = N.
        ExperimentSpec(
            name='case_study_099_scenario_number',
            config_type='ejor',
            val_args=[64, 128, 256, 512],
            mutate=mutate_mixture_target_discount_factor,
            agent_mutate=mutate_sample_path_number_mixture_geometric_l01,
        ),
        # Saure EJOR case-study pair (spec:
        # docs/superpowers/specs/2026-08-02-saure-ejor-case-study-design.md).
        ExperimentSpec(
            name='case_study_ejor_replication',
            config_type='ejor',
            val_args=[0.1],
            mutate=mutate_mixture_target_discount_factor,
        ),
        ExperimentSpec(
            name='case_study_ejor_alp_steady_state',
            config_type='ejor',
            val_args=[0.1],
            mutate=mutate_mixture_target_discount_factor,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
        # Toy-study conditions used by the policy-performance tables.
        ExperimentSpec(
            name='base_toy_study',
            config_type='toy',
            val_args=[0.5],
            mutate=mutate_initial_state_congestion,
        ),
        ExperimentSpec(
            name='steady_state_toy_study',
            config_type='toy',
            val_args=[0.5],
            mutate=mutate_initial_state_congestion,
        ),
        ExperimentSpec(
            name='alp_steady_state_toy_study',
            config_type='toy',
            val_args=[0.5],
            mutate=mutate_initial_state_congestion,
        ),
        # Defensive-mixing-probability sweep (lambda_0).
        ExperimentSpec(
            name='mixture_probability_toy_study',
            config_type='toy',
            val_args=[0.05, 0.1, 0.15, 1.0],
            mutate=mutate_initial_state_congestion_05_const,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
        # Evaluation-proposal comparison arms (see
        # docs/superpowers/specs/2026-07-29-eval-proposal-comparison-design.md):
        # env and policy are byte-identical to mixture_probability_toy_study
        # val=0.1; only the name (= results directory) differs. The evaluation
        # proposal itself is supplied per-arm by the CLI recipe.
        ExperimentSpec(
            name='toy_eval_proposal_geometric_099',
            config_type='toy',
            val_args=[0.1],
            mutate=mutate_initial_state_congestion_05_const,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
        ExperimentSpec(
            name='toy_eval_proposal_fixed_459',
            config_type='toy',
            val_args=[0.1],
            mutate=mutate_initial_state_congestion_05_const,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
        ExperimentSpec(
            name='toy_eval_proposal_mixture_095_l01',
            config_type='toy',
            val_args=[0.1],
            mutate=mutate_initial_state_congestion_05_const,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
    ]
}


def generate_experiment(experiment_name: str):
    """Preferred entry point: look up a spec by name and build its dat file."""
    return build_variation_test_env(EXPERIMENT_SPECS[experiment_name])
