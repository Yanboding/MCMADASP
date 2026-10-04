from experiments import get_config_by_type

TOY_ENV_ARGS = {
    "booking_window_size": 7, "arrival_rates": [1, 2], "patterns": ["1 * 3", "1 * 2"],
    "holding_cost_by_day_by_type": [[0, 0], [100, 10], [100, 10], [100, 10], [100, 10], [100, 10], [100, 10]],
    "overtime_cost_by_day": 100, "postponing_cost": 2000, "duration": 1, "regular_capacity": 7,
    "overtime_capacity": 3, "discount_factor": 0.99,
    "reset_params": {"init_state": [[5, 5, 5, 5, 5, 5, 0], [0, 0, 0, 0, 0, 0, 0], [1, 2]]},
    "maximum_total_arrival": 9, "init_state": None, "valid_action": None,
    "env_random_seed": 0, "stop_time_random_seed": 1, "arrival_random_seed": 42,
}

TOY_ALP_COEFFICIENTS = [
    -887.3514752347272, 100.00000000000001, 99.00000000000001, 98.01, 97.0299, 96.059601, 95.09900499,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 300.00000000000006, 200.00000000000003,
]


def toy_env():
    return get_config_by_type('infinite_custom', args=TOY_ENV_ARGS).env
