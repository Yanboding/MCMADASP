import numpy as np
from scipy.stats import geom

from experiments import get_config_by_type
from generating_function import AbsorptionALPPenaltyFunction
from importance_sampling import build_proposal
from param_generation.caching import load_trained_coefficient_record_from_folder
from param_generation.datasets import offset_sample_generation_seeds
from utils import get_uid


def default_prefix_periods(discount_factor, quantile=0.998):
    return int(geom.ppf(quantile, 1.0 - discount_factor))


def _period_weights(proposal, length, discount_factor, generator):
    return [float(w) for w in proposal.survival_weights([length], discount_factor, generator)[0]]


def draw_replications(env, proposal, size, prefix_periods, is_random_initial_state, fixed_init_state):
    generator = env.arrival_generator
    discount_factor = env.discount_factor
    path_weights = proposal.path_weights(size)
    path_strata = proposal.path_strata(size)
    baseline_lengths = np.asarray(proposal.sample_lengths(generator, size), dtype=int)
    continuation_lengths = generator.rng.permutation(np.asarray(proposal.sample_lengths(generator, size), dtype=int))
    replications = []
    for omega in range(size):
        baseline_length = int(baseline_lengths[omega])
        continuation_length = int(continuation_lengths[omega])
        init_state = env.generate_initial_state() if is_random_initial_state else fixed_init_state
        stream = generator.rvs(size=max(baseline_length, prefix_periods + continuation_length))
        replications.append({
            'init_state': [np.asarray(component).tolist() for component in init_state],
            'arrival_stream': np.asarray(stream).tolist(),
            'baseline_length': baseline_length,
            'baseline_period_weights': _period_weights(proposal, baseline_length, discount_factor, generator),
            'baseline_terminal': proposal.terminal_for([baseline_length], generator)[0].value,
            'prefix_periods': int(prefix_periods),
            'continuation_length': continuation_length,
            'continuation_period_weights': _period_weights(proposal, continuation_length, discount_factor, generator),
            'path_weight': float(path_weights[omega]),
            'path_stratum': int(path_strata[omega]),
        })
    return replications


def generate_potential_improvement_records(test_envs, size, proposal_spec, penalty_dir, seed_offset,
                                           is_random_initial_state, experiment_name, prefix_periods=None):
    records = []
    for (env_uid, spec_experiment_name, mutate_val), variant in test_envs.items():
        env_args = variant['env_args']
        coefficient_record = load_trained_coefficient_record_from_folder(
            spec_experiment_name, mutate_val=mutate_val, folder_path=penalty_dir)
        if coefficient_record is None:
            raise ValueError(f"no theta_ALP coefficients for mutate_val={mutate_val} under {penalty_dir}")
        coefficients = [float(value) for value in coefficient_record['coefficients']]
        env = get_config_by_type('infinite_custom', args=offset_sample_generation_seeds(env_args, seed_offset)).env
        T = default_prefix_periods(env.discount_factor) if prefix_periods is None else int(prefix_periods)
        fixed_init_state = env_args.get('reset_params', {}).get('init_state')
        if not is_random_initial_state and fixed_init_state is None:
            raise ValueError("the variant has no reset init_state; pass --random-init")
        for replication in draw_replications(env, build_proposal(proposal_spec), size, T,
                                             is_random_initial_state, fixed_init_state):
            record = {
                'potential_improvement': True,
                'experiment_name': experiment_name,
                'mutate_val': mutate_val,
                'group_id': env_uid,
                'env_args': env_args,
                **replication,
                'generating_function_spec': {'name': AbsorptionALPPenaltyFunction.spec_name, 'coefficients': coefficients},
                'alp_coefficients': coefficients,
            }
            record['uid'] = get_uid(record)
            records.append(record)
        print(f"{experiment_name}: {size} replications, T = {T}, theta_ALP from {coefficient_record['file']}")
    return records
