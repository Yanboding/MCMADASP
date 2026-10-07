import unittest

import gurobipy as gp
import numpy as np

from decision_maker import ALPRowGenerationAgent, ApproxQAgent
from experiments import get_config_by_type
from generating_function import AbsorptionALPPenaltyFunction
from importance_sampling import FixedLengthProposal, SamplePath, Terminal

GOLDEN_THETA = [0.0, -15.152024161256973, -20.10202397087601, -25.002523970876275,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
GOLDEN_IN_SAMPLE = {'mean': 609.3977233925422, 'half_width': 105.56467999037841,
                    'std': 120.43357167423369, 'centered_mean': 1016.3355458625759,
                    'centered_half_width': 100.41585349043687, 'centered_std': 114.55952776697775}
GOLDEN_IMPROVEMENT = {'mean': 14.713061369728269, 'half_width': 14.08503994941022}
GOLDEN_POLICY_GAP = 2034.224742386182


class TestSubproblemCentering(unittest.TestCase):
    def setUp(self):
        self.grb = gp.Env(empty=True)
        self.grb.setParam('OutputFlag', 0)
        self.grb.setParam('Threads', 1)
        self.grb.start()
        self.env = get_config_by_type('toy').env
        self.state = (np.array([3.] * (self.env.planning_horizon - 1) + [0.]),
                      np.zeros(self.env.planning_horizon), np.array([1., 2.]))
        self.anchor = np.linspace(-20.0, 20.0,
                                  AbsorptionALPPenaltyFunction(self.env).number_of_coefficients)

    def tearDown(self):
        self.grb.dispose()

    def make_agent(self, scenarios=6):
        gf = AbsorptionALPPenaltyFunction(self.env)
        gf.set_coefficients(self.anchor)
        agent = ApproxQAgent(self.env, self.env.discount_factor, sample_path_number=scenarios,
                             sample_path_proposal=FixedLengthProposal(24), generating_function=gf,
                             grb_env=self.grb, subproblem_grb_envs=[self.grb])
        rng = np.random.default_rng(7)
        agent.sample_paths = [
            SamplePath(rng.integers(0, 3, size=(24, self.env.num_types)).astype(float),
                       Terminal.ABSORBED, [0.99 ** t for t in range(25)], [1.] * 24)
            for _ in range(scenarios)]
        return agent

    def policy(self):
        return ALPRowGenerationAgent(self.env, discount_factor=self.env.discount_factor,
                                     coefficients=self.anchor, grb_env=self.grb)

    def test_the_subproblem_reports_the_centered_value_and_gradient(self):
        plain = self.make_agent(scenarios=2)
        centering = plain.policy_cost_centering(self.policy(), init_state=self.state,
                                                coefficients=self.anchor)
        _, gradients, anchor = centering
        probe = self.anchor + np.linspace(2.0, -3.0, self.anchor.size)

        plain.workers = plain._build_training_workers(init_state=self.state, parallel=False)
        centered = self.make_agent(scenarios=2)
        centered.workers = centered._build_training_workers(
            init_state=self.state, parallel=False, policy_centering=centering)

        for sid in range(2):
            ok_plain, value_plain, grad_plain = plain.workers[sid].solve(probe)
            ok_centered, value_centered, grad_centered = centered.workers[sid].solve(probe)
            self.assertTrue(ok_plain and ok_centered)
            shift = float(gradients[sid] @ (probe - anchor))
            self.assertAlmostEqual(value_centered, value_plain - shift, places=6)
            np.testing.assert_allclose(grad_centered, grad_plain - gradients[sid],
                                       rtol=1e-7, atol=1e-7)

    def test_training_with_centering_reproduces_the_recorded_solution(self):
        agent = self.make_agent()
        centering = agent.policy_cost_centering(self.policy(), init_state=self.state,
                                                coefficients=self.anchor)
        _, theta, info = agent.benders_decomposition_train(
            coefficient_bound=1e4, init_state=self.state,
            initial_coefficients=self.anchor.tolist(), policy_centering=centering,
            parallel=False, verbose=False)

        np.testing.assert_allclose(theta, GOLDEN_THETA, rtol=1e-6, atol=1e-6)
        for key, expected in GOLDEN_IN_SAMPLE.items():
            self.assertAlmostEqual(info['in_sample'][key], expected, places=4, msg=key)
        for key, expected in GOLDEN_IMPROVEMENT.items():
            self.assertAlmostEqual(info['improvement'][key], expected, places=4, msg=key)
        self.assertAlmostEqual(info['policy_centering']['mean_policy_relaxation_gap'],
                               GOLDEN_POLICY_GAP, places=4)


if __name__ == '__main__':
    unittest.main()
