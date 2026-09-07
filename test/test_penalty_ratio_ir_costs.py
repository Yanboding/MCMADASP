"""Information-relaxation bounds at several penalty ratios (toy env).

Run from the repo root:  python -m test.test_penalty_ratio_ir_costs
"""
import numpy as np

from experiments import get_config_by_type
from param_generation.training import zero_penalty_coefficients
import run


def test_bounds_per_ratio_are_concave_in_t():
    config = get_config_by_type('toy')
    env = config.env
    env.reset_random_seeds()
    state, _ = env.reset(**config.reset_params)
    tail = env.reset_arrivals(stop_time=3)
    rng = np.random.default_rng(0)
    coefficients = (rng.normal(0.0, 1.0, len(zero_penalty_coefficients(env))) * 5.0).tolist()
    spec = {'name': 'linear_penalty', 'coefficients': coefficients}

    instances = run._build_lowerbound_instances(env, spec, None, None, penalty_ratios=(0, 0.5, 1))
    assert sorted(instances) == [0.0, 0.5, 1.0]
    bounds = run._information_relaxation_bounds(instances, state, tail, None)
    assert sorted(bounds) == [0.0, 0.5, 1.0]
    # t -> V_t is a minimum of affine functions of t, hence concave.
    assert bounds[0.5] >= 0.5 * (bounds[0.0] + bounds[1.0]) - 1e-6, bounds
    # t = 0 is the zero-penalty bound regardless of the coefficients.
    zero_spec = {'name': 'linear_penalty'}
    zero_only = run._build_lowerbound_instances(env, zero_spec, None, None, penalty_ratios=(1,))
    zero_bound = run._information_relaxation_bounds(zero_only, state, tail, None)[1.0]
    assert np.isclose(bounds[0.0], zero_bound, rtol=1e-9, atol=1e-6), (bounds[0.0], zero_bound)
    # Legacy default grid.
    legacy = run._build_lowerbound_instances(env, spec, None, None)
    assert sorted(legacy) == [0.0, 1.0]


if __name__ == '__main__':
    test_bounds_per_ratio_are_concave_in_t()
    print('All penalty-ratio IR cost tests passed.')
