import unittest
from unittest.mock import patch

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

    def test_the_aggregate_objective_takes_the_subproblem_values_as_given(self):
        # The subproblems now report the centered value, so the aggregation must not
        # subtract the policy term a second time.
        agent = self.make_agent()
        anchor = np.full(agent.generating_function.number_of_coefficients, 0.5)
        objective_fn = agent.training_objective_fn()
        values = np.array([10.0, 20.0, 30.0])
        kappa = np.asarray(agent.sample_path_weights, dtype=float)
        for action in (anchor, anchor + 4.0):
            self.assertAlmostEqual(objective_fn(action, values, np.arange(3)), float(kappa @ values))

    def test_master_caps_each_scenario_at_its_policy_cost(self):
        agent = self.make_agent()
        size = agent.generating_function.number_of_coefficients
        costs = np.array([1.0, 2.0, 3.0])
        centering = (costs, np.zeros((3, size)), np.zeros(size))
        master_model, _, theta_vars = agent.train_master_builder_fn(policy_centering=centering)
        np.testing.assert_allclose(theta_vars.UB, costs)
        master_model.dispose()

    def test_master_without_centering_leaves_the_scenario_cap(self):
        agent = self.make_agent()
        master_model, _, theta_vars = agent.train_master_builder_fn()
        master_model.update()
        np.testing.assert_allclose(theta_vars.UB, np.full(3, 1e8))
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

    def test_the_information_relaxation_solves_release_their_model(self):
        agent = self.make_agent(sample_path_number=1)
        path = agent.sample_paths[0]
        args = dict(period_weights=path.survival_weights, terminal=path.terminal)
        built = []
        original = agent._information_relaxation_model

        def spy(*call_args, **call_kwargs):
            result = original(*call_args, **call_kwargs)
            built.append(result[0])
            return result

        agent._information_relaxation_model = spy
        agent.calculate_information_relaxation_cost(self.state, path.arrivals, **args)
        agent.policy_cost_and_gradient(self.state, path.arrivals, **args)
        self.assertEqual(len(built), 2)
        for model in built:
            with self.assertRaises(Exception):
                model.getVars()

    def test_reused_training_workers_match_direct_absorption_alp_ir_costs(self):
        from generating_function.alp_penalty_function import AbsorptionALPPenaltyFunction
        agent = self.make_agent(sample_path_number=2)
        agent.current_decision_var_type = agent.future_decision_var_type = gp.GRB.CONTINUOUS
        agent.generating_function = AbsorptionALPPenaltyFunction(self.env)
        size = agent.generating_function.number_of_coefficients
        anchor, fitted = np.linspace(-2., 3., size), np.linspace(4., -3., size)
        agent.generating_function.set_coefficients(anchor)
        states = [self.state, (np.array([3.] * (self.env.planning_horizon - 1) + [0.]),
                              np.zeros(self.env.planning_horizon), np.array([2., 1.]))]
        agent.sample_paths[0].terminal = Terminal.ABSORBED
        agent.sample_paths[0].survival_weights = np.array([1., 1.1, 1.3, 1.5])
        centering = agent.policy_cost_centering(self.policy(), init_state=states, coefficients=anchor)
        _, gradients, _ = centering
        workers = agent._build_training_workers(init_state=states, parallel=False,
                                                initial_coefficients=anchor, policy_centering=centering)
        try:
            for theta in (fitted, anchor, np.zeros(size)):
                agent.generating_function.set_coefficients(theta)
                for sid, worker in enumerate(workers):
                    with self.subTest(theta=theta.tolist(), scenario=sid):
                        feasible, adjusted, _ = worker.solve(theta)
                        self.assertTrue(feasible)
                        path = agent.sample_paths[sid]
                        direct = agent.calculate_information_relaxation_cost(
                            states[sid], path.arrivals, period_weights=path.survival_weights,
                            terminal=path.terminal)
                        self.assertAlmostEqual(adjusted + gradients[sid] @ (theta - anchor), direct, places=6)
        finally:
            for worker in workers:
                worker.model.dispose()

    def test_final_training_evaluation_reuses_matching_worker_results(self):
        from decision_maker.approximate_q_agent import dispose_agent_models
        from metaheuristic_algorithm import BendersDecompositionSolver
        agent = self.make_agent(sample_path_number=2)
        anchor = np.zeros(agent.generating_function.number_of_coefficients)
        centering = agent.policy_cost_centering(self.policy(), init_state=self.state, coefficients=anchor)
        evaluate = BendersDecompositionSolver.evaluate_action
        calls = []

        def counted(solver, coefficients, **kwargs):
            calls.append(np.array(coefficients, copy=True))
            return evaluate(solver, coefficients, **kwargs)

        try:
            with patch.object(BendersDecompositionSolver, 'evaluate_action', counted):
                _, theta, info = agent.benders_decomposition_train(
                    coefficient_bound=100., init_state=self.state, initial_coefficients=anchor,
                    policy_centering=centering, parallel=False, max_iterations=3)
            self.assertEqual(len(calls), 0, 'Final reporting must reuse the last Benders evaluation')
            values, _ = evaluate(agent.coefficient_model, theta, parallel=False)
            expected = agent._aggregate_in_sample(agent._uncentered(values, theta), theta)
            self.assertAlmostEqual(info['in_sample']['mean'], expected['in_sample']['mean'], places=7)
            for worker in agent.coefficient_model.workers:
                np.testing.assert_array_equal(worker.last_action, theta)
        finally:
            dispose_agent_models(agent)

    def test_rejected_min_norm_candidate_requires_an_exact_coefficient_match(self):
        from decision_maker.approximate_q_agent import dispose_agent_models
        from metaheuristic_algorithm import BendersDecompositionSolver
        agent = self.make_agent(sample_path_number=2)
        anchor = np.zeros(agent.generating_function.number_of_coefficients)
        evaluate = BendersDecompositionSolver.evaluate_action
        calls = []

        def counted(solver, coefficients, **kwargs):
            calls.append(np.array(coefficients, copy=True))
            return evaluate(solver, coefficients, **kwargs)

        def rejected(solver, info, slack, parallel, verbose):
            # Mimic a rejected minimum-norm trial leaving workers at a different
            # candidate. Even a change below np.allclose's tolerance is a miss.
            candidate = info['action'].copy()
            candidate[0] += 1e-10
            solver.evaluate_action(candidate, parallel=parallel)

        try:
            with patch.object(BendersDecompositionSolver, 'evaluate_action', counted), \
                 patch.object(BendersDecompositionSolver, '_relax_action_towards_the_minimum_norm', rejected):
                _, theta, info = agent.benders_decomposition_train(
                    coefficient_bound=100., init_state=self.state, initial_coefficients=anchor,
                    parallel=False, max_iterations=3, min_norm_slack=.05)
            self.assertEqual(len(calls), 2, 'Rejected trial must be followed by a fresh final evaluation')
            np.testing.assert_array_equal(calls[-1], theta)
            self.assertFalse(np.array_equal(calls[0], calls[-1]))
            for worker in agent.coefficient_model.workers:
                np.testing.assert_array_equal(worker.last_action, theta)
            self.assertAlmostEqual(info['in_sample']['mean'],
                                   evaluate(agent.coefficient_model, theta, parallel=False)[1], places=7)
        finally:
            dispose_agent_models(agent)

    def test_worker_caches_a_copy_and_nonoptimal_results_are_not_reused(self):
        from decision_maker.approximate_q_agent import dispose_agent_models
        agent = self.make_agent(sample_path_number=1)
        anchor = np.zeros(agent.generating_function.number_of_coefficients)
        try:
            agent.benders_decomposition_train(coefficient_bound=100., init_state=self.state,
                initial_coefficients=anchor, parallel=False, max_iterations=2)
            worker = agent.coefficient_model.workers[0]
            theta = np.asarray(agent.coefficients).copy()
            worker.solve(theta)
            saved_theta = theta.copy()
            theta[0] += 1
            np.testing.assert_array_equal(worker.last_action, saved_theta)
            worker.last_certificate['status'] = 'TIME_LIMIT'
            with patch.object(agent.coefficient_model, 'evaluate_action',
                              wraps=agent.coefficient_model.evaluate_action) as evaluated:
                agent._in_sample_values(saved_theta, parallel=False)
            evaluated.assert_called_once()
            self.assertEqual(worker.last_certificate['status'], 'OPTIMAL')
        finally:
            dispose_agent_models(agent)

    def test_dispose_agent_models_tolerates_an_agent_without_models(self):
        from decision_maker.approximate_q_agent import dispose_agent_models
        dispose_agent_models(None)
        dispose_agent_models(self.make_agent(sample_path_number=1))


if __name__ == '__main__':
    unittest.main()
