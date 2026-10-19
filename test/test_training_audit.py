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


def test_agent_audit_uses_existing_paths_and_shifts_raw_certificates_once():
    agent = ApproxQAgent.__new__(ApproxQAgent)
    agent.sample_path_weights = np.array([.25, .75])
    agent.sample_path_strata = np.array([0, 1])
    path = SimpleNamespace(arrivals=np.array([[1]]), terminal=SimpleNamespace(value='absorbed'),
                           survival_weights=np.array([1.]), length=1, periods=2)
    agent.sample_paths = [path, path]
    agent._training_policy_centering = (np.array([100.,100.]), np.array([[2.],[4.]]), np.array([1.]))
    def evaluate(theta, parallel):
        v = [10.+float(theta[0]), 20.+float(theta[0])]
        return np.array(v), 0., [dict(objective=x, lower_bound=x, upper_bound=x,
                   exact=True, bound_valid=True, status='OPTIMAL') for x in v]
    agent.coefficient_model = SimpleNamespace(evaluate_action_certificates=evaluate,
                   best_master_certificate={'objective_bound': 25., 'valid':True})
    audit = agent._build_training_audit([2.], [1.], ([0],[0],[0]), False,
                                      {'termination_reason':'max_iterations'})
    assert audit['full_objective'] == 19.5
    assert audit['objective_lower_bound'] == 19.5
    assert audit['objective_upper_bound'] == 25.
    assert audit['absolute_gap'] == 5.5
    assert audit['paths'][0]['ir']['fitted']['objective'] == 14.
    assert audit['paths'][0]['ir']['alp']['objective'] == 11.
    assert audit['paths'][0]['ir']['zero']['objective'] == 8.


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


def test_training_forwards_limits_and_keeps_policy_cap_with_unbounded_scenario_cap(monkeypatch):
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
    agent._build_training_audit = Mock(return_value={'paths': []})
    solver = SimpleNamespace(solve=Mock(return_value=(11., {'action':[2.], 'termination_reason':'max_iterations'})))
    monkeypatch.setattr(module, 'BendersDecompositionSolver', Mock(return_value=solver))
    centering = (np.array([20.]), np.array([[1.]]), np.array([1.]))
    objective, coeff, info = agent.benders_decomposition_train(
        init_state=([0],[0],[0]), policy_centering=centering,
        max_iterations=17, absolute_gap_tolerance=.02, save_path_certificates=True,
        reference_coefficients=[1.], master_time_limit_seconds=4., subproblem_time_limit_seconds=3.)
    options = solver.solve.call_args.kwargs
    assert options['max_iter'] == 17 and options['tol'] == .02
    assert options['min_norm_action'] is True
    assert master.Params.TimeLimit == 4.
    assert agent._training_subproblem_time_limit_seconds == 3.
    master_options = agent.train_master_builder_fn.call_args.kwargs
    assert master_options['scenario_upper_bound'] == GRB.INFINITY
    assert master_options['policy_centering'] is centering
    assert info['audit'] == {'paths': []} and objective == 9.


def test_final_value_evaluation_can_capture_certificates_without_an_extra_solve():
    agent = ApproxQAgent.__new__(ApproxQAgent)
    certificates = [{'objective':3.}]
    certified = Mock(return_value=(np.array([3.]),3.,certificates))
    agent.coefficient_model = SimpleNamespace(evaluate_action_certificates=certified,
                                               evaluate_action=Mock(side_effect=AssertionError('extra solve')))
    agent._training_policy_centering = None
    agent._aggregate_in_sample = lambda v,t: {'costs':v}
    result = agent._in_sample_values([1.], False, capture_certificates=True)
    assert result['certificates'] == certificates
    certified.assert_called_once()
