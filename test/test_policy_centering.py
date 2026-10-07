import unittest

import gurobipy as gp
import numpy as np

from decision_maker import ApproxQAgent, MyopicAgent
from experiments import get_config_by_type
from generating_function import AbsorptionLinearPenaltyFunction
from importance_sampling import FixedLengthProposal, SamplePath, Terminal


class TestPolicyCentering(unittest.TestCase):
    def setUp(self):
        self.grb = gp.Env(empty=True)
        self.grb.setParam('OutputFlag', 0)
        self.grb.setParam('Threads', 1)
        self.grb.start()
        self.env = get_config_by_type('toy').env
        self.state = (np.array([5.] * (self.env.planning_horizon - 1) + [0.]),
                      np.zeros(self.env.planning_horizon), np.array([1., 2.]))

    def tearDown(self):
        self.grb.dispose()

    def make_agent(self, sample_path_number=3, coefficients=None):
        gf = AbsorptionLinearPenaltyFunction(self.env)
        gf.set_coefficients(np.zeros(gf.number_of_coefficients) if coefficients is None else coefficients)
        agent = ApproxQAgent(
            self.env, self.env.discount_factor, sample_path_number=sample_path_number,
            sample_path_proposal=FixedLengthProposal(3), generating_function=gf,
            grb_env=self.grb, subproblem_grb_envs=[self.grb])
        rng = np.random.default_rng(0)
        agent.sample_paths = [
            SamplePath(rng.integers(0, 2, size=(3, self.env.num_types)).astype(float),
                       Terminal.TRUNCATED,
                       [1., 0.99, 0.99 ** 2, 0.99 ** 3], [1., 1., 1.])
            for _ in range(sample_path_number)]
        return agent

    def policy(self):
        return MyopicAgent(self.env, discount_factor=self.env.discount_factor, grb_env=self.grb)

    def test_policy_cost_is_affine_in_theta_with_the_reported_gradient(self):
        agent = self.make_agent(sample_path_number=1)
        rng = np.random.default_rng(1)
        size = agent.generating_function.number_of_coefficients
        first, second = rng.normal(size=size) * 10, rng.normal(size=size) * 10
        path = agent.sample_paths[0]
        actions = agent._rollout_actions(self.policy(), self.state, path)
        args = dict(period_weights=path.survival_weights, terminal=path.terminal, fixed_actions=actions)

        agent.generating_function.set_coefficients(first)
        value_first, gradient = agent.policy_cost_and_gradient(self.state, path.arrivals, **args)
        agent.generating_function.set_coefficients(second)
        value_second, gradient_second = agent.policy_cost_and_gradient(self.state, path.arrivals, **args)

        self.assertAlmostEqual(value_second, value_first + gradient @ (second - first), places=6)
        np.testing.assert_allclose(gradient, gradient_second, rtol=1e-9, atol=1e-9)

    def test_centering_vanishes_at_the_anchor(self):
        agent = self.make_agent()
        anchor = np.full(agent.generating_function.number_of_coefficients, 0.5)
        gradients = np.arange(3 * anchor.size, dtype=float).reshape(3, anchor.size)
        objective_fn = agent.training_objective_fn(
            'mean', policy_centering=(np.zeros(3), gradients, anchor))
        values = np.array([10.0, 20.0, 30.0])
        kappa = np.asarray(agent.sample_path_weights, dtype=float)
        self.assertAlmostEqual(objective_fn(anchor, values, np.arange(3)), float(kappa @ values))

    def test_centering_subtracts_the_policy_gradient_term(self):
        agent = self.make_agent()
        size = agent.generating_function.number_of_coefficients
        anchor = np.zeros(size)
        gradients = np.tile(np.arange(size, dtype=float), (3, 1))
        objective_fn = agent.training_objective_fn(
            'mean', policy_centering=(np.zeros(3), gradients, anchor))
        values = np.array([10.0, 20.0, 30.0])
        action = np.full(size, 2.0)
        kappa = np.asarray(agent.sample_path_weights, dtype=float)
        expected = float(kappa @ (values - gradients @ action))
        self.assertAlmostEqual(objective_fn(action, values, np.arange(3)), expected)

    def test_master_adds_one_policy_cut_per_scenario(self):
        agent = self.make_agent()
        size = agent.generating_function.number_of_coefficients
        centering = (np.array([1.0, 2.0, 3.0]), np.zeros((3, size)), np.zeros(size))
        master_model, _, theta_vars = agent.train_master_builder_fn(policy_centering=centering)
        rows = [c for c in master_model.getConstrs() if c.ConstrName.startswith('policy_cut')]
        self.assertEqual(len(rows), 3)
        np.testing.assert_allclose(theta_vars.UB, np.full(3, 1e8))
        master_model.dispose()

    def test_master_without_centering_has_no_policy_cut(self):
        agent = self.make_agent()
        master_model, _, _ = agent.train_master_builder_fn()
        self.assertFalse([c for c in master_model.getConstrs()
                          if c.ConstrName.startswith('policy_cut')])
        master_model.dispose()

    def test_centering_triple_predicts_the_policy_cost_at_other_coefficients(self):
        agent = self.make_agent(sample_path_number=2)
        size = agent.generating_function.number_of_coefficients
        rng = np.random.default_rng(2)
        anchor, elsewhere = rng.normal(size=size) * 5, rng.normal(size=size) * 5
        costs, gradients, reported = agent.policy_cost_centering(
            self.policy(), init_state=self.state, coefficients=anchor)

        np.testing.assert_allclose(reported, anchor)
        self.assertEqual(costs.shape, (2,))
        self.assertEqual(gradients.shape, (2, size))

        agent.generating_function.set_coefficients(elsewhere)
        for sid in range(2):
            path = agent.sample_paths[sid]
            actions = agent._rollout_actions(self.policy(), self.state, path)
            direct, _ = agent.policy_cost_and_gradient(
                self.state, path.arrivals, period_weights=path.survival_weights,
                terminal=path.terminal, fixed_actions=actions)
            predicted = costs[sid] + gradients[sid] @ (elsewhere - anchor)
            self.assertAlmostEqual(predicted, direct, places=5)

    def test_centering_restores_the_generating_function(self):
        agent = self.make_agent(sample_path_number=2)
        before = np.array(agent.generating_function.coefficient_vector())
        agent.policy_cost_centering(
            self.policy(), init_state=self.state,
            coefficients=np.full(before.size, -2.0))
        np.testing.assert_allclose(agent.generating_function.coefficient_vector(), before)

    def test_dispose_agent_models_releases_the_subproblem_models(self):
        from decision_maker.approximate_q_agent import dispose_agent_models
        agent = self.make_agent(sample_path_number=2)
        agent.workers = agent._build_training_workers(init_state=self.state, parallel=False)
        self.assertTrue(agent.workers)
        agent.workers[0].model.getVars()
        dispose_agent_models(agent)
        with self.assertRaises(Exception):
            agent.workers[0].model.getVars()

    def test_dispose_agent_models_tolerates_an_agent_without_models(self):
        from decision_maker.approximate_q_agent import dispose_agent_models
        dispose_agent_models(None)
        dispose_agent_models(self.make_agent(sample_path_number=1))


if __name__ == '__main__':
    unittest.main()
