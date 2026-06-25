"""Registry of all experiments. Add new experiments here.

Maps each experiment ``name`` to its :class:`ExperimentSpec`. ``generate_experiment``
is the preferred entry point: look up a spec by name and materialize its variants.
"""

from param_generation.experiment_specs import ExperimentSpec, build_variation_test_env
from param_generation.mutators import (
    mutate_discount_factor,
    mutate_discount_factor_for_sample_path_length_proposal,
    mutate_fixed_length_for_sample_path_length_proposal,
    mutate_high_priority_proportion,
    mutate_high_priority_waiting_time_penalty,
    mutate_initial_state_congestion,
    mutate_is_proposal_098_const,
    mutate_low_priority_waiting_time_target,
    mutate_mixture_geometric_proposal_lambda_0,
    mutate_mixture_target_discount_factor,
    mutate_overtime_cost,
    mutate_total_arrival_rate,
    mutate_type_1_treatment_pattern,
)

EXPERIMENT_SPECS = {
    spec.name: spec for spec in [
        ExperimentSpec(
            name='initial_state_congestion',
            config_type='toy',
            val_args=[0., 0.5, 1.],
            mutate=mutate_initial_state_congestion,
        ),
        ExperimentSpec(
            name='high_priority_proportion',
            config_type='toy',
            val_args=[0.1, 0.5, 0.9],
            mutate=mutate_high_priority_proportion,
        ),
        ExperimentSpec(
            name='low_priority_waiting_time_target',
            config_type='toy',
            val_args=[1, 3, 5],
            mutate=mutate_low_priority_waiting_time_target,
        ),
        ExperimentSpec(
            name='high_priority_waiting_time_penalty',
            config_type='toy',
            val_args=[10, 100, 200],
            mutate=mutate_high_priority_waiting_time_penalty,
        ),
        ExperimentSpec(
            name='total_arrival_rate',
            config_type='toy',
            val_args=[24/7, 30/7, 36/7],
            mutate=mutate_total_arrival_rate,
        ),
        ExperimentSpec(
            name='type_1_treatment_pattern',
            config_type='toy',
            val_args=['1 * 3', '1 * 2 + 1 * 1', '3 * 1'],
            mutate=mutate_type_1_treatment_pattern,
        ),
        ExperimentSpec(
            name='overtime_cost',
            config_type='toy',
            val_args=[10, 100, 200],
            mutate=mutate_overtime_cost,
        ),
        ExperimentSpec(
            name='case_study_discount_factor',
            config_type='toy',
            val_args=[0.95, 0.96, 0.97, 0.98],
            mutate=mutate_discount_factor,
        ),
        ExperimentSpec(
            name='sample_path_length_proposal_geometric',
            config_type='toy',
            val_args=[0.95, 0.96, 0.98, 0.99],
            agent_mutate=mutate_discount_factor_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='sample_path_length_proposal_fixed',
            config_type='toy',
            val_args=[100, 200, 400],
            agent_mutate=mutate_fixed_length_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='solver_comparison',
            config_type='toy',
            val_args=[0.],
            mutate=mutate_initial_state_congestion,
        ),
        ExperimentSpec(
            name='multiclass_LP_solver_comparison',
            config_type='toy',
            val_args=[0.],
            mutate=mutate_initial_state_congestion,
        ),
        ExperimentSpec(
            name='case_study',
            config_type='ejor',
            val_args=[0.95],
            mutate=mutate_discount_factor,
        ),
        ExperimentSpec(
            name='case_study_098',
            config_type='ejor',
            val_args=[0.98],
            mutate=mutate_discount_factor,
        ),
        ExperimentSpec(
            name='case_study_099',
            config_type='ejor',
            val_args=[0.99],
            mutate=mutate_discount_factor,
        ),
        ExperimentSpec(
            name='case_study_099_is_098',
            config_type='toy',
            val_args=[0.99],
            mutate=mutate_discount_factor,
            agent_mutate=mutate_is_proposal_098_const,
        ),
        ExperimentSpec(
            name='case_study_099_fixed_length',
            config_type='ejor',
            val_args=[20, 40],
            agent_mutate=mutate_fixed_length_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='case_study_099_base_case',
            config_type='toy',
            val_args=[0.99],
            agent_mutate=mutate_discount_factor_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='case_study_099_importance_sampling_proposal_098',
            config_type='ejor',
            val_args=[0.98],
            agent_mutate=mutate_discount_factor_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='case_study_099_importance_sampling_proposal_095',
            config_type='toy',
            val_args=[0.95],
            agent_mutate=mutate_discount_factor_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='case_study_099_mixture_geometric_proposal_095',
            config_type='ejor',
            val_args=[0.05, 0.1, 0.15],
            mutate=mutate_mixture_target_discount_factor,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
        ExperimentSpec(
            name='toy_study_base_case',
            config_type='toy',
            val_args=[0.99],
            agent_mutate=mutate_discount_factor_for_sample_path_length_proposal,
        ),
        ExperimentSpec(
            name='toy_study_099_mixture_geometric_proposal_095_test',
            config_type='toy',
            val_args=[0.05, 0.1, 0.15, 1],
            mutate=mutate_mixture_target_discount_factor,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
    ]
}


def generate_experiment(experiment_name: str):
    """Preferred entry point: look up a spec by name and build its dat file."""
    return build_variation_test_env(EXPERIMENT_SPECS[experiment_name])
