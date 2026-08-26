"""Registry of all experiments. Add new experiments here.

Maps each experiment ``name`` to its :class:`ExperimentSpec`. ``generate_experiment``
is the preferred entry point: look up a spec by name and materialize its variants.
"""

from functools import partial

from param_generation.experiment_specs import ExperimentSpec, build_variation_test_env
from param_generation.mutators import (
    mutate_initial_state_congestion,
    mutate_initial_state_congestion_05_const,
    mutate_mixture_geometric_proposal_lambda_0,
    mutate_mixture_geometric_scenario_const,
    mutate_mixture_target_initial_state_congestion,
    mutate_mixture_target_discount_factor,
    mutate_mixture_target_discount_factor_overtime_5,
    mutate_mixture_target_discount_factor_overtime_50,
    mutate_pathwise_safety,
    mutate_sample_path_number_mixture_geometric_l01,
)

# Initial-occupancy grid for the 0.99-target case study: penalty coefficients
# trained from ONE fixed initial state at each occupancy level (generate the
# commands with ``train --reset-init-state``) under a mixture-geometric proposal
# with long-component mass ``epsilon`` (= lambda_0; short component 0.95) and
# a fixed Benders scenario count. One experiment per (epsilon, scenario count),
# named ``case_study_099_occupancy_l<epsilon digits>_scenario_<N>`` (e.g.
# epsilon 0.125 -> ``l0125``); val = occupancy fraction. Extend the lists to
# add grid points.
OCCUPANCY_LEVELS = [0.3, 0.5, 0.9]
OCCUPANCY_EPSILONS = [0.125, 0.2, 0.5, 1.0]  # 1.0 = no importance sampling (target Geom(0.99))
OCCUPANCY_SCENARIO_NUMBERS = [256, 512]


def occupancy_experiment_name(epsilon, sample_path_number):
    return (f"case_study_099_occupancy_l{str(epsilon).replace('.', '')}"
            f"_scenario_{sample_path_number}")


def _occupancy_experiment_specs():
    return [
        ExperimentSpec(
            name=occupancy_experiment_name(epsilon, sample_path_number),
            config_type='ejor',
            val_args=list(OCCUPANCY_LEVELS),
            mutate=mutate_mixture_target_initial_state_congestion,
            agent_mutate=partial(mutate_mixture_geometric_scenario_const,
                                 lambda_0=epsilon, sample_path_number=sample_path_number),
        )
        for epsilon in OCCUPANCY_EPSILONS
        for sample_path_number in OCCUPANCY_SCENARIO_NUMBERS
    ]


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
        # Initial-occupancy grid (see ``_occupancy_experiment_specs``): one
        # experiment per (epsilon, scenario count), sweeping the occupancy.
        *_occupancy_experiment_specs(),
        # Pathwise safety sweep on the case study: hard constraint vs soft
        # (rho=1); the unconstrained baseline is
        # case_study_099_mixture_geometric_proposal_095. val = mode.
        ExperimentSpec(
            name='case_study_099_pathwise_safety',
            config_type='ejor',
            val_args=['hard', 1.0],
            mutate=mutate_mixture_target_discount_factor,
            agent_mutate=mutate_pathwise_safety,
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
