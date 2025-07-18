import unittest

import numpy as np

from decision_maker import SAAdvanceAgent
from experiments import get_config_by_type


class TestSAAdvanceAgent(unittest.TestCase):

    def test_lower_bound(self):
        config = get_config_by_type('default', 42)
        env = config.env
        env.decision_epoch = 6
        init_state, info = env.reset(**config.reset_params)
        sample_path = np.array([[0, 0], [0, 0], [0, 0], [0,0], [0,0], [0,0]])
        init_state = (init_state[0], sample_path[0])
        lower_bound_agent = SAAdvanceAgent(env=env, discount_factor=env.discount_factor)
        lower_bound_agent.set_sample_path(sample_path)
        _, _, lower_bound = lower_bound_agent.solve(init_state, 1)
        self.assertEqual(lower_bound, 0)

if __name__ == "__main_":
    unittest.main()
