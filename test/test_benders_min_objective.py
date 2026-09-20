import numpy as np
import gurobipy as gp
from gurobipy import GRB

from metaheuristic_algorithm import BendersDecompositionSolver, SubproblemWorker

N_COEFF = 4
SCENARIO_COSTS = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 3.0], [2.5, 3.5, 1.5, 0.5], [0.5, 4.0, 0.5, 2.0]])
WEIGHTS = np.array([0.4, 0.3, 0.2, 0.1])


def build_workers():
    workers = []
    for sid, cost in enumerate(SCENARIO_COSTS):
        sub = gp.Model(f"toy_sub_{sid}")
        sub.Params.OutputFlag = 0
        x = sub.addMVar(N_COEFF, lb=0.0, ub=1.0, name="x")
        feature = sub.addMVar(N_COEFF, lb=-GRB.INFINITY, name="feature")
        sub.addConstr(x.sum() >= 2.0, name="cover")
        sub.addConstr(feature == x - 0.5, name="feature_link")
        sub.setObjective(cost @ x, GRB.MINIMIZE)
        sub.optimize()
        initial_cut = (float(sub.ObjVal), np.array(feature.X, dtype=float))

        def objective_builder_fn(model, action_values, feature=feature):
            feature.Obj = np.asarray(action_values, dtype=float)

        def cut_gradient_fn(model, action_values, feature=feature):
            return np.array(feature.X, dtype=float)

        workers.append(SubproblemWorker(model=sub, link_rows=None, state_linking_constraints=None, subproblem_id=sid,
                                        objective_builder_fn=objective_builder_fn, cut_gradient_fn=cut_gradient_fn,
                                        verbose=False, initial_cut=initial_cut))
    return workers


def build_solver(objective, weights=None):
    master = gp.Model("toy_master")
    master.Params.OutputFlag = 0
    theta = master.addMVar(len(SCENARIO_COSTS), lb=-GRB.INFINITY, ub=1e8, name="theta")
    coeff = master.addMVar(N_COEFF, lb=-5.0, ub=5.0, name="coeff")
    kappa = np.full(len(SCENARIO_COSTS), 1.0 / len(SCENARIO_COSTS)) if weights is None else weights
    if objective == 'mean':
        master.setObjective(theta @ kappa, GRB.MAXIMIZE)
    else:
        eta = master.addVar(lb=-GRB.INFINITY, name="eta")
        master.addConstr(eta <= theta, name="eta_le_theta")
        master.setObjective(eta, GRB.MAXIMIZE)
    master.update()
    return BendersDecompositionSolver(master_model=master, workers=build_workers(), imm_cost=None, theta_vars=theta,
                                      action_vars=coeff, scenario_weights=weights, objective=objective)


def test_scenario_objective_aggregations():
    values = np.array([3.0, 1.0, 2.0, 4.0])
    assert build_solver('mean').scenario_objective(values) == 2.5
    assert build_solver('min').scenario_objective(values) == 1.0
    assert build_solver('mean', WEIGHTS).scenario_objective(values) == WEIGHTS @ values


def test_unknown_objective_is_rejected():
    try:
        build_solver('max')
    except ValueError:
        pass
    else:
        raise AssertionError('unknown objective must be rejected')


def test_min_objective_converges_and_beats_mean_optimum_on_the_minimum():
    mean_solver = build_solver('mean')
    mean_solver.solve(parallel=False, max_iter=200)
    theta_mean = np.array(mean_solver.action_vars.X)
    min_solver = build_solver('min')
    value, info = min_solver.solve(parallel=False, max_iter=200)
    assert info['gap'] < 1e-6
    theta_min = np.array(min_solver.action_vars.X)
    values_at_min, _ = min_solver.evaluate_action(theta_min, parallel=False)
    values_at_mean, _ = min_solver.evaluate_action(theta_mean, parallel=False)
    values_at_zero, _ = min_solver.evaluate_action(np.zeros(N_COEFF), parallel=False)
    assert np.isclose(info['evaluated_value'], values_at_min.min(), atol=1e-6)
    assert values_at_min.min() >= values_at_mean.min() - 1e-6
    assert values_at_min.min() >= values_at_zero.min() - 1e-6


def test_seed_cuts_are_anchored_at_the_initial_cut_action():
    solver = build_solver('mean')
    anchor = np.array([0.5, -0.5, 1.0, 0.0])
    for worker in solver.workers:
        values, _ = solver.evaluate_action(anchor, parallel=False)
        break
    for worker, value in zip(solver.workers, values):
        worker.solve(anchor, False)
        gradient = worker.cut_gradient_fn(worker.model, anchor)
        worker.initial_cut = (float(value), gradient, anchor)
    solver.solve(parallel=False, max_iter=0)
    solver.master_model.update()
    seeds = [c for c in solver.master_model.getConstrs() if c.ConstrName.startswith('benders_seed_cut_')]
    assert len(seeds) == len(solver.workers)
    for worker, constr in zip(solver.workers, seeds):
        value, gradient, action = worker.initial_cut
        assert np.isclose(constr.RHS, value - gradient @ action, atol=1e-9)


def test_seed_cuts_with_mixed_anchors_stay_valid():
    solver = build_solver('mean')
    anchor = np.array([0.5, -0.5, 1.0, 0.0])
    values, _ = solver.evaluate_action(anchor, parallel=False)
    for sid, worker in enumerate(solver.workers):
        if sid % 2:
            worker.solve(anchor, False)
            worker.initial_cut = (float(values[sid]), worker.cut_gradient_fn(worker.model, anchor), anchor)
    solver.solve(parallel=False, max_iter=0)
    solver.master_model.update()
    seeds = {c.ConstrName: c for c in solver.master_model.getConstrs() if c.ConstrName.startswith('benders_seed_cut_')}
    assert len(seeds) == len(solver.workers)
    theta = np.array([1.0, -2.0, 0.5, 3.0])
    values_at_theta, _ = solver.evaluate_action(theta, parallel=False)
    for worker in solver.workers:
        value, gradient = worker.initial_cut[0], worker.initial_cut[1]
        action = worker.initial_cut[2] if len(worker.initial_cut) > 2 else np.zeros(N_COEFF)
        assert values_at_theta[worker.subproblem_id] <= value + gradient @ (theta - action) + 1e-9


def build_solver_with_flat_coefficient(min_norm):
    master = gp.Model("toy_master_flat")
    master.Params.OutputFlag = 0
    theta = master.addMVar(len(SCENARIO_COSTS), lb=-GRB.INFINITY, ub=1e8, name="theta")
    coeff = master.addMVar(N_COEFF + 1, lb=-5.0, ub=5.0, name="coeff")
    master.setObjective(theta.sum() / theta.shape[0], GRB.MAXIMIZE)
    master.update()
    workers = []
    for sid, cost in enumerate(SCENARIO_COSTS):
        sub = gp.Model(f"toy_sub_flat_{sid}")
        sub.Params.OutputFlag = 0
        x = sub.addMVar(N_COEFF, lb=0.0, ub=1.0, name="x")
        feature = sub.addMVar(N_COEFF + 1, lb=-GRB.INFINITY, name="feature")
        sub.addConstr(x.sum() >= 2.0, name="cover")
        sub.addConstr(feature[:N_COEFF] == x - 0.5, name="feature_link")
        sub.addConstr(feature[N_COEFF] == 0.0, name="flat_feature")
        sub.setObjective(cost @ x, GRB.MINIMIZE)
        sub.optimize()
        initial_cut = (float(sub.ObjVal), np.array(feature.X, dtype=float))

        def objective_builder_fn(model, action_values, feature=feature):
            feature.Obj = np.asarray(action_values, dtype=float)

        def cut_gradient_fn(model, action_values, feature=feature):
            return np.array(feature.X, dtype=float)

        workers.append(SubproblemWorker(model=sub, link_rows=None, state_linking_constraints=None, subproblem_id=sid,
                                        objective_builder_fn=objective_builder_fn, cut_gradient_fn=cut_gradient_fn,
                                        verbose=False, initial_cut=initial_cut))
    solver = BendersDecompositionSolver(master_model=master, workers=workers, imm_cost=None, theta_vars=theta, action_vars=coeff)
    value, info = solver.solve(parallel=False, max_iter=200, min_norm_action=min_norm)
    return solver, value, info


def test_min_norm_zeroes_flat_coefficients_that_sit_on_the_box():
    solver, value, info = build_solver_with_flat_coefficient(min_norm=True)
    action = np.asarray(info['action'])
    assert abs(action[N_COEFF]) < 1e-9
    values, mean = solver.evaluate_action(action, parallel=False)
    assert np.isclose(mean, value, atol=1e-6) and info['gap'] < 1e-6
    _, _, info_free = build_solver_with_flat_coefficient(min_norm=False)
    assert abs(np.asarray(info_free['action'])[N_COEFF]) == 5.0


def test_action_at_bound_ignores_fixed_coefficients():
    solver = build_solver('mean')
    solver.action_vars[0].lb = 0.0
    solver.action_vars[0].ub = 0.0
    solver.master_model.update()
    assert not solver._action_at_bound(np.array([0.0, 1.0, -1.0, 2.0]))
    assert solver._action_at_bound(np.array([0.0, 5.0, -1.0, 2.0]))
    assert solver._action_at_bound(np.array([0.0, 1.0, -5.0, 2.0]))
