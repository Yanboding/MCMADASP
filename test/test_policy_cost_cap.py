import unittest

import gurobipy as gp
import numpy as np

from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import AbsorptionLinearPenaltyFunction
from importance_sampling import FixedLengthProposal, SamplePath, Terminal


class TestPolicyCostCap(unittest.TestCase):
    def setUp(self):
        self.grb = gp.Env(empty=True)
        self.grb.setParam('OutputFlag', 0)
        self.grb.setParam('Threads', 1)
        self.grb.start()
        self.env = get_config_by_type('toy').env
        self.agents = []

    def tearDown(self):
        for agent in self.agents:
            if agent.coefficient_model is not None:
                agent.coefficient_model.master_model.dispose()
        self.grb.dispose()

    def make_agent(self, sample_path_number=3):
        gf = AbsorptionLinearPenaltyFunction(self.env)
        gf.set_coefficients(np.zeros(gf.number_of_coefficients))
        agent = ApproxQAgent(
            self.env, self.env.discount_factor, sample_path_number=sample_path_number,
            sample_path_proposal=FixedLengthProposal(1), generating_function=gf,
            grb_env=self.grb, subproblem_grb_envs=[self.grb],
        )
        agent.sample_paths = [
            SamplePath(np.zeros((1, self.env.num_types)), Terminal.TRUNCATED,
                       [1., self.env.discount_factor], [1.])
            for _ in range(sample_path_number)]
        self.agents.append(agent)
        return agent

    def test_master_bounds_each_scenario_epigraph_by_its_cap(self):
        agent = self.make_agent()
        caps = np.array([10.0, -5.0, 300.0])
        master_model, _, theta_vars = agent.train_master_builder_fn(policy_cost_caps=caps)
        np.testing.assert_allclose(theta_vars.UB, caps)
        master_model.dispose()

    def test_master_without_caps_keeps_the_default_upper_bound(self):
        agent = self.make_agent()
        master_model, _, theta_vars = agent.train_master_builder_fn()
        np.testing.assert_allclose(theta_vars.UB, np.full(3, 1e8))
        master_model.dispose()

    def test_capped_objective_takes_the_minimum_of_value_and_cap(self):
        agent = self.make_agent()
        agent.sample_path_weights = np.array([0.5, 0.25, 0.25])
        caps = np.array([10.0, 100.0, 0.0])
        objective_fn = agent.training_objective_fn('mean', policy_cost_caps=caps)
        values = np.array([40.0, 20.0, -7.0])
        expected = 0.5 * 10.0 + 0.25 * 20.0 + 0.25 * -7.0
        self.assertAlmostEqual(objective_fn(None, values, np.arange(3)), expected)

    def test_capped_objective_respects_the_supplied_scenario_subset(self):
        agent = self.make_agent()
        agent.sample_path_weights = np.array([0.5, 0.25, 0.25])
        caps = np.array([10.0, 100.0, 0.0])
        objective_fn = agent.training_objective_fn('mean', policy_cost_caps=caps)
        self.assertAlmostEqual(objective_fn(None, np.array([40.0, -7.0]), np.array([0, 2])),
                               0.5 * 10.0 + 0.25 * -7.0)

class TestPolicyCostCapRollout(unittest.TestCase):
    def setUp(self):
        self.grb = gp.Env(empty=True)
        self.grb.setParam('OutputFlag', 0)
        self.grb.setParam('Threads', 1)
        self.grb.start()
        self.env = get_config_by_type('toy').env

    def tearDown(self):
        self.grb.dispose()

    def make_agent(self, sample_path_number=2):
        gf = AbsorptionLinearPenaltyFunction(self.env)
        gf.set_coefficients(np.zeros(gf.number_of_coefficients))
        agent = ApproxQAgent(
            self.env, self.env.discount_factor, sample_path_number=sample_path_number,
            sample_path_proposal=FixedLengthProposal(2), generating_function=gf,
            grb_env=self.grb, subproblem_grb_envs=[self.grb])
        rng = np.random.default_rng(0)
        agent.sample_paths = [
            SamplePath(rng.integers(0, 2, size=(2, self.env.num_types)).astype(float),
                       Terminal.TRUNCATED,
                       [1., self.env.discount_factor, self.env.discount_factor ** 2], [1., 1.])
            for _ in range(sample_path_number)]
        return agent

    def test_caps_match_the_relaxation_value_of_the_rolled_out_actions(self):
        agent = self.make_agent()
        from decision_maker import MyopicAgent
        policy = MyopicAgent(self.env, discount_factor=self.env.discount_factor, grb_env=self.grb)
        state = (np.zeros(self.env.planning_horizon),
                 np.zeros(self.env.planning_horizon), np.array([1., 2.]))
        caps = agent.policy_cost_caps(policy, init_state=state)
        self.assertEqual(caps.shape, (2,))
        self.assertTrue(np.all(np.isfinite(caps)))
        for sid in range(2):
            path = agent.sample_paths[sid]
            actions = agent._rollout_actions(policy, state, path)
            expected = agent.calculate_information_relaxation_cost(
                state, path.arrivals, period_weights=path.survival_weights,
                terminal=path.terminal, fixed_actions=actions)
            self.assertAlmostEqual(caps[sid], expected, places=6)

    def test_cap_is_at_least_the_hindsight_relaxation_value(self):
        agent = self.make_agent()
        from decision_maker import MyopicAgent
        policy = MyopicAgent(self.env, discount_factor=self.env.discount_factor, grb_env=self.grb)
        state = (np.zeros(self.env.planning_horizon),
                 np.zeros(self.env.planning_horizon), np.array([1., 2.]))
        caps = agent.policy_cost_caps(policy, init_state=state)
        for sid in range(2):
            path = agent.sample_paths[sid]
            relaxation = agent.calculate_information_relaxation_cost(
                state, path.arrivals, period_weights=path.survival_weights, terminal=path.terminal)
            self.assertGreaterEqual(caps[sid] + 1e-6, relaxation)

    def test_caps_are_evaluated_at_the_supplied_coefficients(self):
        agent = self.make_agent()
        from decision_maker import MyopicAgent
        policy = MyopicAgent(self.env, discount_factor=self.env.discount_factor, grb_env=self.grb)
        state = (np.zeros(self.env.planning_horizon),
                 np.zeros(self.env.planning_horizon), np.array([1., 2.]))
        theta = np.full(agent.generating_function.number_of_coefficients, -3.0)
        zero_caps = agent.policy_cost_caps(policy, init_state=state)
        theta_caps = agent.policy_cost_caps(policy, init_state=state, coefficients=theta)
        self.assertFalse(np.allclose(zero_caps, theta_caps))

    def test_caps_restore_the_generating_function_coefficients(self):
        agent = self.make_agent()
        from decision_maker import MyopicAgent
        policy = MyopicAgent(self.env, discount_factor=self.env.discount_factor, grb_env=self.grb)
        state = (np.zeros(self.env.planning_horizon),
                 np.zeros(self.env.planning_horizon), np.array([1., 2.]))
        before = np.array(agent.generating_function.coefficient_vector())
        agent.policy_cost_caps(policy, init_state=state,
                               coefficients=np.full(before.size, -3.0))
        np.testing.assert_allclose(agent.generating_function.coefficient_vector(), before)

    def test_pathwise_noise_removal_subtracts_the_same_shift_as_the_subproblem(self):
        agent = self.make_agent()
        from decision_maker import MyopicAgent
        policy = MyopicAgent(self.env, discount_factor=self.env.discount_factor, grb_env=self.grb)
        state = (np.array([5.] * (self.env.planning_horizon - 1) + [0.]),
                 np.zeros(self.env.planning_horizon), np.array([1., 2.]))
        theta = np.full(agent.generating_function.number_of_coefficients, -3.0)
        plain = agent.policy_cost_caps(policy, init_state=state, coefficients=theta)
        denoised = agent.policy_cost_caps(policy, init_state=state, coefficients=theta,
                                          noise_removal='pathwise')
        agent.generating_function.set_coefficients(theta)
        for sid in range(2):
            path = agent.sample_paths[sid]
            actions = agent._rollout_actions(policy, state, path)
            expected = agent.calculate_information_relaxation_cost(
                state, path.arrivals, period_weights=path.survival_weights,
                terminal=path.terminal, fixed_actions=actions, remove_path_noise=True)
            self.assertAlmostEqual(denoised[sid], expected, places=6)

    def test_mean_noise_removal_shifts_every_cap_by_one_constant(self):
        agent = self.make_agent()
        from decision_maker import MyopicAgent
        policy = MyopicAgent(self.env, discount_factor=self.env.discount_factor, grb_env=self.grb)
        state = (np.zeros(self.env.planning_horizon),
                 np.zeros(self.env.planning_horizon), np.array([1., 2.]))
        theta = np.full(agent.generating_function.number_of_coefficients, -3.0)
        plain = agent.policy_cost_caps(policy, init_state=state, coefficients=theta)
        shifted = agent.policy_cost_caps(policy, init_state=state, coefficients=theta,
                                         noise_removal='mean')
        offsets = plain - shifted
        np.testing.assert_allclose(offsets, np.full_like(offsets, offsets[0]))

    def test_alp_penalty_caps_move_when_the_path_noise_is_removed(self):
        from decision_maker import MyopicAgent
        from generating_function.alp_penalty_function import AbsorptionALPPenaltyFunction
        gf = AbsorptionALPPenaltyFunction(self.env)
        gf.set_coefficients(np.zeros(gf.number_of_coefficients))
        agent = ApproxQAgent(
            self.env, self.env.discount_factor, sample_path_number=2,
            sample_path_proposal=FixedLengthProposal(4), generating_function=gf,
            grb_env=self.grb, subproblem_grb_envs=[self.grb])
        policy = MyopicAgent(self.env, discount_factor=self.env.discount_factor, grb_env=self.grb)
        state = (np.array([5.] * (self.env.planning_horizon - 1) + [0.]),
                 np.zeros(self.env.planning_horizon), np.array([1., 2.]))
        theta = np.full(gf.number_of_coefficients, -3.0)
        plain = agent.policy_cost_caps(policy, init_state=state, coefficients=theta)
        denoised = agent.policy_cost_caps(policy, init_state=state, coefficients=theta,
                                          noise_removal='pathwise')
        self.assertFalse(np.allclose(plain, denoised))


if __name__ == '__main__':
    unittest.main()
