import numpy as np

from decision_maker import ALPRowGenerationAgent
from decision_maker.alp_rg_agent import alp_expected_initial_state
from experiments import get_config_by_type


def test_expected_state_matches_alp_relevance_moments():
    env = get_config_by_type('toy').env
    regular, overtime, waitlist = alp_expected_initial_state(env)
    total = env.regular_capacity + env.overtime_capacity
    required = total * 0.95 ** np.arange(env.planning_horizon)
    required[-1] = 0
    np.testing.assert_allclose(regular, np.minimum(required, env.regular_capacity))
    np.testing.assert_allclose(overtime, required - np.minimum(required, env.regular_capacity))
    np.testing.assert_allclose(waitlist, env.arrival_generator.mean_by_type)
    assert regular[-1] == 0 and overtime[-1] == 0
    assert regular.dtype == float and (np.concatenate([regular, overtime, waitlist]) % 1 != 0).any()
