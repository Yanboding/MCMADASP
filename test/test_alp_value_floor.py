import unittest

import gurobipy as gp
import numpy as np

from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import AbsorptionALPPenaltyFunction, LinearPenaltyFunction
from importance_sampling import FixedLengthProposal, SamplePath, Terminal


class TestAlpValueFloor(unittest.TestCase):
    def setUp(self):
        self.grb = gp.Env(empty=True)
        self.grb.setParam('OutputFlag', 0)
        self.grb.setParam('Threads', 1)
        self.grb.start()
        self.env = get_config_by_type('toy').env
        self.anchor = np.linspace(-20.0, 20.0,
                                  AbsorptionALPPenaltyFunction(self.env).number_of_coefficients)

    def tearDown(self):
        self.grb.dispose()

    def state(self, fill):
        return (np.array([float(fill)] * (self.env.planning_horizon - 1) + [0.]),
                np.zeros(self.env.planning_horizon), np.array([1., 2.]))

    def make_agent(self, scenarios=3):
        gf = AbsorptionALPPenaltyFunction(self.env)
        gf.set_coefficients(self.anchor)
        agent = ApproxQAgent(self.env, self.env.discount_factor, sample_path_number=scenarios,
                             sample_path_proposal=FixedLengthProposal(3), generating_function=gf,
                             grb_env=self.grb, subproblem_grb_envs=[self.grb])
        rng = np.random.default_rng(0)
        agent.sample_paths = [
            SamplePath(rng.integers(0, 2, size=(3, self.env.num_types)).astype(float),
                       Terminal.ABSORBED, [1., 0.99, 0.99 ** 2, 0.99 ** 3], [1.] * 3)
            for _ in range(scenarios)]
        return agent

    def test_state_features_are_the_affine_basis(self):
        gf = AbsorptionALPPenaltyFunction(self.env)
        state = self.state(3)
        phi = gf.state_features(state)
        expected = np.concatenate(([1.0], state[0], state[1], state[2]))
        self.assertEqual(phi.shape, (gf.number_of_coefficients,))
        np.testing.assert_allclose(phi, expected)

    def test_approximate_value_is_theta_dot_features(self):
        gf = AbsorptionALPPenaltyFunction(self.env)
        state = self.state(3)
        np.testing.assert_allclose(gf.approximate_value(state, self.anchor),
                                   self.anchor @ gf.state_features(state))
        with self.assertRaises(NotImplementedError):
            LinearPenaltyFunction(self.env).approximate_value(state, self.anchor)

    def test_floors_are_the_alp_value_at_each_scenario_state(self):
        agent = self.make_agent()
        shared = self.state(3)
        floors = agent.alp_value_floors(shared, self.anchor)
        gf = agent.generating_function
        self.assertEqual(floors.shape, (3,))
        np.testing.assert_allclose(floors, gf.approximate_value(shared, self.anchor))

        per_scenario = [self.state(2), self.state(3), self.state(2)]
        floors = agent.alp_value_floors(per_scenario, self.anchor)
        np.testing.assert_allclose(
            floors, [gf.approximate_value(s, self.anchor) for s in per_scenario])

    def test_master_without_centering_floors_the_scenario_variable(self):
        agent = self.make_agent()
        floors = np.array([10.0, 20.0, 30.0])
        master, _, theta_vars = agent.train_master_builder_fn(alp_value_floors=floors)
        master.update()
        np.testing.assert_allclose(theta_vars.LB, floors)
        self.assertFalse([c for c in master.getConstrs() if c.ConstrName.startswith('alp_value_floor')])
        master.dispose()

    def test_master_with_centering_adds_the_control_variate_back(self):
        agent = self.make_agent()
        size = agent.generating_function.number_of_coefficients
        gradients = np.tile(np.arange(size, dtype=float), (3, 1))
        centering = (np.full(3, 1e6), gradients, self.anchor)
        floors = np.array([10.0, 20.0, 30.0])
        master, coefficients, theta_vars = agent.train_master_builder_fn(
            policy_centering=centering, alp_value_floors=floors)
        master.update()
        rows = [c for c in master.getConstrs() if c.ConstrName.startswith('alp_value_floor')]
        self.assertEqual(len(rows), 3)
        # xi is centered, so the floor must not be imposed on it directly.
        self.assertTrue(np.all(np.asarray(theta_vars.LB) < floors.min()))
        master.dispose()

    def test_centered_floor_reduces_to_the_plain_floor_at_the_anchor(self):
        agent = self.make_agent()
        size = agent.generating_function.number_of_coefficients
        gradients = np.tile(np.arange(size, dtype=float), (3, 1))
        floors = np.array([10.0, 20.0, 30.0])
        master, coefficients, theta_vars = agent.train_master_builder_fn(
            policy_centering=(np.full(3, 1e6), gradients, self.anchor),
            alp_value_floors=floors)
        for index in range(size):
            coefficients[index].lb = self.anchor[index]
            coefficients[index].ub = self.anchor[index]
        master.setObjective(theta_vars.sum(), gp.GRB.MINIMIZE)
        master.optimize()
        self.assertEqual(master.Status, gp.GRB.OPTIMAL)
        np.testing.assert_allclose(theta_vars.X, floors, atol=1e-6)
        master.dispose()

    def test_training_reports_the_floor_diagnostics_and_respects_them(self):
        agent = self.make_agent(scenarios=4)
        state = self.state(3)
        _, theta, info = agent.benders_decomposition_train(
            coefficient_bound=1e4, init_state=state, initial_coefficients=self.anchor.tolist(),
            alp_value_floor=True, parallel=False)
        report = info['alp_value_floor']
        floors = agent.alp_value_floors(state, self.anchor)
        self.assertEqual(report['violations'], 0)
        self.assertTrue(report['verified'])
        self.assertGreaterEqual(report['min_margin'], -report['tolerance'])
        np.testing.assert_allclose(report['floors'][0], floors[0])

    def test_training_without_the_flag_reports_nothing(self):
        agent = self.make_agent(scenarios=4)
        _, _, info = agent.benders_decomposition_train(
            coefficient_bound=1e4, init_state=self.state(3),
            initial_coefficients=self.anchor.tolist(), parallel=False)
        self.assertNotIn('alp_value_floor', info)

    def test_the_floor_needs_initial_coefficients(self):
        agent = self.make_agent(scenarios=4)
        with self.assertRaises(ValueError) as caught:
            agent.benders_decomposition_train(
                coefficient_bound=1e4, init_state=self.state(3), alp_value_floor=True,
                parallel=False)
        self.assertIn('initial_coefficients', str(caught.exception))

    def test_floors_ignore_fixed_coefficient_overrides(self):
        # The floor is hat V at theta*_ALP; pinning a coefficient during the fit must not move it.
        agent = self.make_agent(scenarios=4)
        state = self.state(3)
        expected = agent.generating_function.approximate_value(state, self.anchor)
        _, _, info = agent.benders_decomposition_train(
            coefficient_bound=1e4, init_state=state, initial_coefficients=self.anchor.tolist(),
            fixed_coefficients={0: 0.0}, alp_value_floor=True, parallel=False)
        np.testing.assert_allclose(info['alp_value_floor']['floors'][0], expected)

    def test_cli_rejects_the_floor_without_init_coefficients(self):
        from unittest import mock
        from param_generation import cli
        with mock.patch.object(cli, 'write_command_file'):
            with self.assertRaises(ValueError) as caught:
                cli.main(['train', 'base_toy_study', '--penalty-function', 'absorption_alp_penalty',
                          '--alp-value-floor', '--dat', 'unused.dat'])
        self.assertIn('--init-coefficients', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
