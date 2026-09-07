"""Exact expectations under the arrival generator's truncated distribution."""
import numpy as np
import pytest

from environment.arrival_generator import MultiClassPoissonArrivalGenerator
from experiments import get_config_by_type
from generating_function import AbsorptionLinearPenaltyFunction, LinearPenaltyFunction


@pytest.mark.parametrize("rate, cap", [(3.0, 9), (8.0, 2), (3.0, 0), (0.0, 9)])
def test_mean_by_type_matches_enumerated_distribution(rate, cap):
    generator = MultiClassPoissonArrivalGenerator(
        mean_arrival_rate=rate,
        maximum_arrival=cap,
        type_probs=[0.2, 0.3, 0.5],
        use_qmc=False,
    )
    probabilities, arrivals = zip(*generator.get_system_dynamic())
    exact_mean = np.asarray(probabilities) @ np.asarray(arrivals)

    np.testing.assert_allclose(generator.mean_by_type, exact_mean, rtol=1e-12, atol=1e-12)
    assert generator.mean_arrival == rate  # Preserve the nominal Poisson parameter.


@pytest.mark.parametrize("penalty_class", [LinearPenaltyFunction, AbsorptionLinearPenaltyFunction])
def test_expected_penalty_features_match_enumerated_arrivals(penalty_class):
    env = get_config_by_type('toy').env
    state = (
        np.full(env.planning_horizon, 2.0),
        np.ones(env.planning_horizon),
        np.full(env.num_types, 3.0),
    )
    scheduling = np.zeros((env.booking_window_size, env.num_types))
    scheduling[0] = 1.0
    action = (scheduling, np.ones(env.planning_horizon))
    penalty = penalty_class(env)
    exact_features = sum(
        probability * penalty.features(state, action, arrival).dense()
        for probability, arrival in env.arrival_generator.get_system_dynamic()
    )

    np.testing.assert_allclose(
        penalty.expected_features(state, action).dense(),
        exact_features,
        rtol=1e-12,
        atol=1e-12,
    )
