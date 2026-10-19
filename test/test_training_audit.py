"""Certificate plumbing tests use model doubles without license checkout."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from gurobipy import GRB
from metaheuristic_algorithm import SubproblemWorker, BendersDecompositionSolver
from decision_maker import ApproxQAgent


def model(status=GRB.OPTIMAL, value=7., bound=7.):
    return SimpleNamespace(Status=status, SolCount=1, ObjVal=value, ObjBound=bound,
                           Runtime=.1, Params=SimpleNamespace(), ModelSense=GRB.MAXIMIZE, optimize=Mock(),
                           setParam=Mock())


def test_interrupted_subproblem_never_requests_feasibility_ray():
    worker = SubproblemWorker(model(GRB.TIME_LIMIT), [], [], 3)
    worker._get_feasibility_ray_for_link_rows = Mock(side_effect=AssertionError('ray'))
    with pytest.raises(RuntimeError, match='TIME_LIMIT'):
        worker.solve(np.array([0.]))
    assert worker.last_certificate['status'] == 'TIME_LIMIT'
    worker._get_feasibility_ray_for_link_rows.assert_not_called()


def test_primary_certificate_retains_best_bound_and_excludes_pinned_solve():
    solver = BendersDecompositionSolver.__new__(BendersDecompositionSolver)
    solver.master_model = model(value=7., bound=8.)
    solver._record_primary_master_certificate(global_certificate=True)
    solver.master_model.ObjBound = 10.
    solver._record_primary_master_certificate(global_certificate=True)
    solver.master_model.ObjVal = solver.master_model.ObjBound = 2.
    solver._record_primary_master_certificate(global_certificate=False)
    assert solver.best_master_certificate['objective_bound'] == 8.
    assert len(solver.master_certificate_history) == 2


def test_evaluate_certificates_snapshots_order_and_values():
    solver = BendersDecompositionSolver.__new__(BendersDecompositionSolver)
    solver.theta_vars = np.zeros(2)
    solver.objective_fn = lambda a, v, ids: float(np.mean(v))
    workers = []
    for sid in range(2):
        w = SimpleNamespace(subproblem_id=sid, grb_env=None)
        def solve(action, worker=w, value=float(sid+2)):
            worker.last_certificate = {'objective': value, 'lower_bound': value}
            return True, value, np.array([0.])
        w.solve = solve
        workers.append(w)
    solver.workers = workers
    values, objective, certs = solver.evaluate_action_certificates([0.], parallel=False)
    assert values.tolist() == [2., 3.] and objective == 2.5
    workers[0].last_certificate['objective'] = -1
    assert certs[0]['objective'] == 2.




def test_primary_master_snapshot_precedes_min_norm_objective(monkeypatch):
    import metaheuristic_algorithm.benders_decomposition_solver as module
    solver = BendersDecompositionSolver.__new__(BendersDecompositionSolver)
    solver.master_model = model(value=7., bound=8.)
    solver.action_vars = SimpleNamespace(X=np.array([2.e6]))
    solver.theta_vars = SimpleNamespace(X=np.array([7.]))
    solver._cut_purge_enabled = False
    solver._report_memory_usage = Mock()
    solver._at_bound_mask = lambda action: np.array([False])
    def norm(action, theta, verbose):
        solver.master_model.ObjVal = solver.master_model.ObjBound = 999.
        return action, theta
    solver._min_norm_master_action = norm
    monkeypatch.setattr(module, 'solve_and_handle_errors', lambda *a, **kw: True)
    solver._solve_master_step(1, 1e-6, True, False)
    assert solver.best_master_certificate['objective_bound'] == 8.


def test_pinned_master_explicitly_suppresses_global_certificate():
    solver = BendersDecompositionSolver.__new__(BendersDecompositionSolver)
    solver.action_vars = SimpleNamespace(shape=(1,), lb=np.array([-9.]), ub=np.array([9.]))
    solver._solve_master_step = Mock(return_value=(np.array([1.]), 5., np.array([5.])))
    solver._solve_master_with_fixed_action([1.], 1, 1e-6, False)
    assert solver._solve_master_step.call_args.kwargs['global_certificate'] is False
    assert solver.action_vars.lb.tolist() == [-9.]


def test_training_forwards_limits_and_preserves_original_outputs(monkeypatch):
    import decision_maker.approximate_q_agent as module
    agent = ApproxQAgent.__new__(ApproxQAgent)
    agent.coefficient_model = None
    agent.subproblem_initial_cuts = {}
    agent.subproblem_cold_solve_seconds = {}
    agent.sample_path_weights = np.array([1.])
    agent.sample_path_strata = np.array([0])
    agent.sample_path_number = 1
    agent._build_training_workers = Mock(return_value=[])
    master = SimpleNamespace(Params=SimpleNamespace())
    agent.train_master_builder_fn = Mock(return_value=(master, SimpleNamespace(X=[2.]), np.array([0.])))
    agent._prepare_training_checkpoint = Mock()
    agent._build_importance_sampling_info = Mock(return_value={})
    agent._require_generating_function = Mock(return_value=SimpleNamespace(set_coefficients=Mock()))
    agent._in_sample_values = lambda *a, **kw: {'in_sample': {'mean': 9.}, 'costs':np.array([9.])}
    solver = SimpleNamespace(solve=Mock(return_value=(11., {'action':[2.], 'termination_reason':'max_iterations'})))
    monkeypatch.setattr(module, 'BendersDecompositionSolver', Mock(return_value=solver))
    centering = (np.array([20.]), np.array([[1.]]), np.array([1.]))
    objective, coeff, info = agent.benders_decomposition_train(
        init_state=([0],[0],[0]), policy_centering=centering,
        max_iterations=17, absolute_gap_tolerance=.02,
        master_time_limit_seconds=4., subproblem_time_limit_seconds=3.)
    options = solver.solve.call_args.kwargs
    assert options['max_iter'] == 17 and options['tol'] == .02
    assert options['min_norm_action'] is True
    assert master.Params.TimeLimit == 4.
    assert agent._training_subproblem_time_limit_seconds == 3.
    master_options = agent.train_master_builder_fn.call_args.kwargs
    assert 'scenario_upper_bound' not in master_options
    assert master_options['policy_centering'] is centering
    assert 'audit' not in info and objective == 9.
    assert info['in_sample']['mean'] == 9.
    assert info['in_sample']['centered_mean'] == 8.
    assert coeff == [2.]
    assert 'paths' not in info and 'full_objective' not in info




def test_training_has_no_baseline_flags_or_extra_reference_inputs():
    import inspect
    parameters = inspect.signature(ApproxQAgent.benders_decomposition_train).parameters
    assert 'reference_coefficients' not in parameters
    assert 'save_path_certificates' not in parameters
    assert not hasattr(ApproxQAgent, '_training_path_results')
    assert 'capture_certificates' not in inspect.signature(ApproxQAgent._in_sample_values).parameters




def test_master_uses_fixed_policy_cost_cap_even_above_historical_default():
    import gurobipy as gp
    with gp.Env(empty=True) as license_env:
        license_env.setParam('OutputFlag', 0)
        license_env.start()
        agent = ApproxQAgent.__new__(ApproxQAgent)
        agent.grb_env = license_env
        agent.sample_path_number = 2
        agent.sample_path_weights = np.array([.5,.5])
        agent.generating_function = SimpleNamespace(get_coefficient_var=lambda model, coefficient_bound:
            model.addMVar(1, lb=-coefficient_bound, ub=coefficient_bound))
        costs = np.array([2.e8,5.e7])
        model, _, theta = agent.train_master_builder_fn(policy_centering=(costs, np.zeros((2,1)),np.zeros(1)))
        try:
            np.testing.assert_array_equal(theta.UB, costs)
        finally:
            model.dispose()
        model, _, theta = agent.train_master_builder_fn()
        try:
            np.testing.assert_array_equal(theta.UB, [1.e8,1.e8])
        finally:
            model.dispose()
