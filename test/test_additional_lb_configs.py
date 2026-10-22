import copy

import gurobipy as gp
import numpy as np
import pytest

from experiments import get_config_by_type
from param_generation.cli import _select_variants
from param_generation.experiment_specs import build_variation_test_env
from param_generation.registry import EXPERIMENT_SPECS


TOY_CASES = [
    ('toy_lb_treatment_pattern', ['3 * 1', '1 * 2'], [1, 2], 7, 9, [100, 10], [1, 2]),
    ('toy_lb_arrival_rates', ['1 * 3', '1 * 2'], [9 / 7, 18 / 7], 7, 7,
     [100, 10], [1, 2]),
    ('toy_lb_pattern_arrival_rates', ['3 * 1', '1 * 2'], [9 / 7, 18 / 7], 7, 9,
     [100, 10], [1, 2]),
    ('toy_lb_booking_window', ['1 * 3', '1 * 2'], [1, 2], 21, 21,
     [100, 10], [1, 2]),
    ('toy_lb_waiting_costs', ['1 * 3'] * 3 + ['1 * 2'] * 3,
     [1 / 3] * 3 + [2 / 3] * 3, 7, 7, [150, 100, 50, 15, 10, 5], [0, 1, 0, 0, 2, 0]),
]


@pytest.fixture(autouse=True)
def forbid_solver_initialization(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('configuration generation must not initialize a solver')

    monkeypatch.setattr(gp, 'Env', fail)
    monkeypatch.setattr(gp, 'Model', fail)


def single_environment(name):
    assert name in EXPERIMENT_SPECS, f'missing appendix experiment {name}'
    spec = EXPERIMENT_SPECS[name]
    expected_values = {
        'toy_lb_treatment_pattern': 3,
        'toy_lb_arrival_rates': 9 / 7,
        'toy_lb_pattern_arrival_rates': 9 / 7,
        'toy_lb_booking_window': 21,
        'toy_lb_waiting_costs': 6,
        'small_lb': 6,
    }
    assert spec.val_args == [expected_values[name]]
    variants = build_variation_test_env(spec)
    assert len(variants) == 1
    key = next(iter(variants))
    assert isinstance(key, tuple) and len(key) == 3
    assert key[1:] == (name, expected_values[name])
    assert _select_variants(variants, str(expected_values[name])) == variants
    return next(iter(variants.values()))['env_args']


@pytest.mark.parametrize('name,patterns,rates,window,horizon,costs,waitlist', TOY_CASES)
def test_toy_appendix_configs_preserve_the_agreed_controls(
        name, patterns, rates, window, horizon, costs, waitlist):
    env_args = single_environment(name)
    assert env_args['patterns'] == patterns
    assert env_args['arrival_rates'] == rates
    assert env_args['booking_window_size'] == window
    assert env_args['maximum_total_arrival'] == 9
    assert env_args['regular_capacity'] == 7
    assert env_args['overtime_capacity'] == 3
    assert env_args['overtime_cost_by_day'] == 100
    assert env_args['postponing_cost'] == 2000
    assert env_args['discount_factor'] == 0.99
    assert env_args['duration'] == 1
    regular, overtime, actual_waitlist = env_args['reset_params']['init_state']
    assert regular == [5] * 6 + [0] * (horizon - 6)
    assert overtime == [0] * horizon
    assert actual_waitlist == waitlist
    assert env_args['holding_cost_by_day_by_type'] == [[0] * len(costs)] + [costs] * (window - 1)

    # Construct the real environment to catch stale horizon or cost dimensions.
    env = get_config_by_type('infinite_custom', args=env_args).env
    assert env.planning_horizon == horizon
    assert env.num_types == len(costs)
    assert env.waiting_cost_coeffs.shape == (window, len(costs))
    expected_last_costs = np.asarray(costs) * sum(0.99 ** day for day in range(1, window))
    np.testing.assert_allclose(env.waiting_cost_coeffs[-1], expected_last_costs)


def test_small_appendix_config_is_the_exact_original_case():
    env_args = single_environment('small_lb')
    assert env_args == get_config_by_type('small').args
    env = get_config_by_type('infinite_custom', args=env_args).env
    assert (env.booking_window_size, env.planning_horizon, env.num_types) == (25, 57, 6)
    assert (env.regular_capacity, env.overtime_capacity) == (120, 15)
    assert env.arrival_generator.maximum_arrival == 25


def test_appendix_variants_do_not_modify_base_configs_or_each_other():
    base_toy = get_config_by_type('toy').args
    original_toy = copy.deepcopy(base_toy)
    for name, *_ in TOY_CASES:
        env_args = single_environment(name)
        env_args['arrival_rates'][0] = -1
        env_args['holding_cost_by_day_by_type'][1][0] = -1
        env_args['reset_params']['init_state'][0][0] = -1
        fresh = single_environment(name)
        assert fresh['arrival_rates'][0] >= 0
        assert fresh['holding_cost_by_day_by_type'][1][0] > 0
        assert fresh['reset_params']['init_state'][0][0] == 5
    assert base_toy == original_toy
    assert get_config_by_type('toy').args == original_toy
