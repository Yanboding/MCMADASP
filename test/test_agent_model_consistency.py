"""Regression tests for cached objectives and returned scheduling actions."""
import unittest
from pathlib import Path
import tempfile

import gurobipy as gp
import numpy as np

from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import AbsorptionLinearPenaltyFunction, LinearPenaltyFunction
from importance_sampling import FixedLengthProposal, SamplePath, Terminal


class TestAgentModelConsistency(unittest.TestCase):
    def setUp(self):
        self.grb = gp.Env(empty=True)
        self.grb.setParam('OutputFlag', 0)
        self.grb.setParam('Threads', 1)
        self.grb.start()
        self.env = get_config_by_type('toy').env
        self.state = (np.zeros(self.env.planning_horizon),
                      np.zeros(self.env.planning_horizon), np.array([1., 2.]))
        self.zero_action = (np.zeros((self.env.booking_window_size, self.env.num_types)),
                            np.zeros(self.env.planning_horizon))
        self.agents = []

    def tearDown(self):
        for agent in self.agents:
            if agent.decision_model is not None:
                agent.decision_model.dispose()
            for worker in agent.workers or []:
                worker.model.dispose()
            if agent.coefficient_model is not None:
                agent.coefficient_model.master_model.dispose()
                for worker in agent.coefficient_model.workers:
                    worker.model.dispose()
        self.grb.dispose()

    def make_agent(self, solver='approx_Q', coefficients=None, penalty_class=AbsorptionLinearPenaltyFunction):
        gf = penalty_class(self.env)
        gf.set_coefficients(np.zeros(gf.number_of_coefficients) if coefficients is None else coefficients)
        agent = ApproxQAgent(
            self.env, self.env.discount_factor, sample_path_number=1,
            sample_path_proposal=FixedLengthProposal(1), generating_function=gf,
            solver_name=solver, grb_env=self.grb, subproblem_grb_envs=[self.grb],
        )
        # A deterministic two-period scenario makes fresh/reused comparisons exact.
        agent.sample_paths = [SamplePath(np.zeros((1, self.env.num_types)),
                                        Terminal.TRUNCATED, [1., self.env.discount_factor], [1.])]
        self.agents.append(agent)
        return agent

    def solve_raw(self, agent, action=None):
        if agent.solver_name == 'approx_Q':
            return agent.approx_Q_solve(self.state, 1, action=action)
        return agent.hindsight_solve(self.state, 1, action=action, parallel=False)

    def waitlist_coefficients(self):
        theta = np.zeros(AbsorptionLinearPenaltyFunction(self.env).number_of_coefficients)
        theta[2 * self.env.planning_horizon:2 * self.env.planning_horizon + self.env.num_types] = -1000.
        return theta

    def test_policy_caches_follow_coefficient_changes(self):
        for solver in ('approx_Q', 'approx_penalized_hindsight'):
            with self.subTest(solver=solver):
                agent = self.make_agent(solver)
                before = self.solve_raw(agent, self.zero_action)[0]
                theta = self.waitlist_coefficients()
                agent.generating_function.set_coefficients(theta)
                observed = self.solve_raw(agent, self.zero_action)[0]
                expected = self.solve_raw(self.make_agent(solver, theta), self.zero_action)[0]
                self.assertNotAlmostEqual(before, expected)
                self.assertAlmostEqual(observed, expected, places=6)

    def test_policy_caches_follow_ratio_and_function_changes(self):
        theta = self.waitlist_coefficients()
        for solver in ('approx_Q', 'approx_penalized_hindsight'):
            with self.subTest(solver=solver):
                agent = self.make_agent(solver, theta)
                self.solve_raw(agent, self.zero_action)
                agent.penalty_ratio = 0.25
                fresh = self.make_agent(solver, theta)
                fresh.penalty_ratio = 0.25
                self.assertAlmostEqual(self.solve_raw(agent, self.zero_action)[0],
                                       self.solve_raw(fresh, self.zero_action)[0], places=6)
                agent.generating_function = LinearPenaltyFunction(self.env, coefficients=theta)
                fresh = self.make_agent(solver, theta, LinearPenaltyFunction)
                fresh.penalty_ratio = 0.25
                self.assertAlmostEqual(self.solve_raw(agent, self.zero_action)[0],
                                       self.solve_raw(fresh, self.zero_action)[0], places=6)

    def test_fixed_action_does_not_lock_later_free_solve(self):
        agent = self.make_agent()
        self.solve_raw(agent, self.zero_action)
        observed, action, _ = self.solve_raw(agent)
        expected = self.solve_raw(self.make_agent())[0]
        self.assertAlmostEqual(observed, expected, places=6)
        self.assertGreater(action[0].sum(), 0)

    def test_infeasible_fixed_action_restores_bounds(self):
        agent = self.make_agent()
        invalid = tuple(part.copy() for part in self.zero_action)
        invalid[0][0, 0] = 100
        with self.assertRaises(RuntimeError):
            self.solve_raw(agent, invalid)
        self.assertAlmostEqual(self.solve_raw(agent)[0],
                               self.solve_raw(self.make_agent())[0], places=6)

    def test_public_solve_preserves_explicit_fixed_action(self):
        fixed = tuple(part.copy() for part in self.zero_action)
        fixed[0][0, 0] = 1
        fixed = (fixed[0], self.env.convert_action_to_booking_slots(fixed[0]))
        for solver in ('approx_Q', 'approx_penalized_hindsight'):
            with self.subTest(solver=solver):
                agent = self.make_agent(solver)
                objective, returned, _ = agent.solve(self.state, 1, action=fixed, parallel=False)
                for actual, requested in zip(returned, fixed):
                    np.testing.assert_array_equal(actual, requested)
                expected = self.solve_raw(self.make_agent(solver), fixed)[0]
                self.assertAlmostEqual(objective, expected, places=6)

    def test_repaired_action_is_reevaluated(self):
        theta = np.zeros(AbsorptionLinearPenaltyFunction(self.env).number_of_coefficients)
        theta[self.env.planning_horizon:2 * self.env.planning_horizon] = -1000.
        for solver in ('approx_Q', 'approx_penalized_hindsight'):
            with self.subTest(solver=solver):
                agent = self.make_agent(solver, theta)
                raw_objective, raw_action, _ = self.solve_raw(agent)
                canonical = agent.regular_first_overtime(self.state, raw_action[0])
                self.assertFalse(np.array_equal(raw_action[1], canonical))
                objective, returned, info = agent.solve(self.state, 1, parallel=False)
                expected = self.solve_raw(self.make_agent(solver, theta), returned)[0]
                self.assertAlmostEqual(objective, expected, places=6)
                self.assertAlmostEqual(info['raw_objective'], raw_objective, places=6)
                np.testing.assert_array_equal(returned[1],
                                              agent.regular_first_overtime(self.state, returned[0]))

    def test_retraining_uses_new_regularizer_bounds_and_initial_state(self):
        agent = self.make_agent()
        agent.benders_decomposition_train(init_state=self.state, coefficient_bound=100.,
                                         regularization={'type': 'l2', 'lambda': 0.001}, parallel=False)
        state = (*self.state[:2], np.array([50., 50.]))
        options = dict(init_state=state, coefficient_bound=0., parallel=False,
                       regularization={'type': 'l1', 'lambda': 1e6, 'scale': 'none'})
        observed, coefficients, info = agent.benders_decomposition_train(**options)
        expected, _, _ = self.make_agent().benders_decomposition_train(**options)
        np.testing.assert_allclose(coefficients, 0., atol=1e-8)
        np.testing.assert_array_equal(info['regularization']['scale'], 1.)
        self.assertFalse(agent.coefficient_model.master_model.IsQP)
        self.assertAlmostEqual(observed, expected, places=6)

    def test_unchanged_penalty_reuses_policy_models(self):
        agent = self.make_agent()
        self.solve_raw(agent, self.zero_action)
        model = agent.decision_model
        self.solve_raw(agent, self.zero_action)
        self.assertIs(agent.decision_model, model)
        agent.solver_name = 'approx_penalized_hindsight'
        self.solve_raw(agent, self.zero_action)
        worker = agent.workers[0]
        self.solve_raw(agent, self.zero_action)
        self.assertIs(agent.workers[0], worker)

    def test_continuous_fixed_action_preserves_fractions_and_objective(self):
        fixed = tuple(part.copy() for part in self.zero_action)
        fixed[0][0, 0] = 0.4
        for solver in ('approx_Q', 'approx_penalized_hindsight'):
            with self.subTest(solver=solver):
                agent = self.make_agent(solver)
                agent.current_decision_var_type = gp.GRB.CONTINUOUS
                objective, returned, _ = agent.solve(self.state, 1, action=fixed, parallel=False)
                for actual, requested in zip(returned, fixed):
                    np.testing.assert_allclose(actual, requested, atol=1e-9)
                self.assertAlmostEqual(objective, self.env.cost_fn(self.state, fixed), places=6)

    def test_continuous_regular_first_preserves_fractional_overtime(self):
        self.state[0][:] = self.env.regular_capacity - 0.25
        for solver in ('approx_Q', 'approx_penalized_hindsight'):
            with self.subTest(solver=solver):
                agent = self.make_agent(solver)
                agent.current_decision_var_type = gp.GRB.CONTINUOUS
                objective, action, _ = agent.solve(self.state, 1, parallel=False)
                expected_overtime = np.maximum(
                    self.state[0] + self.env.convert_action_to_booking_slots(action[0])
                    - self.env.regular_capacity, 0.)
                np.testing.assert_allclose(action[1], expected_overtime, atol=1e-8)
                self.assertAlmostEqual(objective, self.solve_raw(agent, action)[0], places=6)

    def test_training_checkpoint_resumes_only_the_same_models(self):
        options = dict(init_state=self.state, coefficient_bound=5., parallel=False,
                       regularization={'type': 'l2', 'lambda': 0.001, 'scale': 'none'})
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / 'training.pickle')
            agent = self.make_agent()
            expected, _, _ = agent.benders_decomposition_train(**options, checkpoint_path=checkpoint)
            resumed, _, _ = self.make_agent().benders_decomposition_train(
                **options, resume_checkpoint_path=checkpoint)
            self.assertAlmostEqual(resumed, expected, places=6)
            for changes in ({'init_state': (*self.state[:2], np.array([50., 50.]))},
                            {'coefficient_bound': 0.},
                            {'regularization': {'type': 'l1', 'lambda': 1e6, 'scale': 'none'}}):
                with self.subTest(changes=changes):
                    with self.assertRaisesRegex(ValueError, 'checkpoint.*incompatible'):
                        agent.benders_decomposition_train(
                            **{**options, **changes}, resume_checkpoint_path=checkpoint)

    def test_unidentified_training_checkpoint_is_rejected(self):
        options = dict(init_state=self.state, coefficient_bound=5., parallel=False,
                       regularization={'type': 'l2', 'lambda': 0.001, 'scale': 'none'})
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = str(Path(tmp) / 'training.pickle')
            agent = self.make_agent()
            agent.benders_decomposition_train(**options, checkpoint_path=checkpoint)
            identity = Path(checkpoint + '.training.json')
            if identity.exists():
                identity.unlink()
            with self.assertRaisesRegex(ValueError, 'checkpoint.*identity'):
                self.make_agent().benders_decomposition_train(**options, resume_checkpoint_path=checkpoint)


if __name__ == '__main__':
    unittest.main()
