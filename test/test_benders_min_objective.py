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


def weighted_mean_fn(weights):
    return lambda action, values, ids: float(np.asarray(weights)[np.asarray(ids)] @ np.asarray(values))


def build_solver(objective='mean', weights=None):
    master = gp.Model("toy_master")
    master.Params.OutputFlag = 0
    theta = master.addMVar(len(SCENARIO_COSTS), lb=-GRB.INFINITY, ub=1e8, name="theta")
    coeff = master.addMVar(N_COEFF, lb=-5.0, ub=5.0, name="coeff")
    kappa = np.full(len(SCENARIO_COSTS), 1.0 / len(SCENARIO_COSTS)) if weights is None else weights
    master.setObjective(theta @ kappa, GRB.MAXIMIZE)
    master.update()
    return BendersDecompositionSolver(master_model=master, workers=build_workers(), imm_cost=None, theta_vars=theta,
                                      action_vars=coeff, objective_fn=None if weights is None else weighted_mean_fn(weights))


def test_objective_value_uses_the_supplied_aggregation():
    values = np.array([3.0, 1.0, 2.0, 4.0])
    assert build_solver('mean').objective_value(np.zeros(N_COEFF), values) == 2.5
    assert build_solver('mean', WEIGHTS).objective_value(np.zeros(N_COEFF), values) == WEIGHTS @ values
    shared = build_worst_case_solver(np.tile(LEVEL, (len(SCENARIO_COSTS), 1)))
    for theta in (np.zeros(N_COEFF), np.array([1.0, -2.0, 0.5, 3.0])):
        assert np.isclose(shared.objective_value(theta, values), 1.0)


def test_shared_state_worst_case_maximizes_the_minimum_scenario_value():
    mean_solver = build_solver('mean')
    mean_solver.solve(parallel=False, max_iter=200)
    theta_mean = np.array(mean_solver.action_vars.X)
    solver = build_worst_case_solver(np.tile(LEVEL, (len(SCENARIO_COSTS), 1)))
    value, info = solver.solve(parallel=False, max_iter=200)
    assert info['gap'] < 1e-6
    theta = np.asarray(info['action'])
    values_at_theta, _ = solver.evaluate_action(theta, parallel=False)
    values_at_mean, _ = solver.evaluate_action(theta_mean, parallel=False)
    values_at_zero, _ = solver.evaluate_action(np.zeros(N_COEFF), parallel=False)
    assert np.isclose(info['evaluated_value'], values_at_theta.min(), atol=1e-6)
    assert values_at_theta.min() >= values_at_mean.min() - 1e-6
    assert values_at_theta.min() >= values_at_zero.min() - 1e-6


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


LEVEL = np.array([1.0, 0.5, -0.5, 2.0])
FEATURES = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [2.0, 0.5, 0.5, 0.0]])


def worst_case_fn(features):
    features = np.asarray(features, dtype=float)

    def objective_fn(action, values, ids):
        action = np.asarray(action, dtype=float)
        return float(LEVEL @ action + (np.asarray(values) - features[np.asarray(ids)] @ action).min())

    return objective_fn


def build_worst_case_solver(features):
    master = gp.Model("toy_master_wc")
    master.Params.OutputFlag = 0
    theta = master.addMVar(len(SCENARIO_COSTS), lb=-GRB.INFINITY, ub=1e8, name="theta")
    coeff = master.addMVar(N_COEFF, lb=-5.0, ub=5.0, name="coeff")
    eta = master.addVar(lb=-GRB.INFINITY, name="eta")
    for n in range(len(SCENARIO_COSTS)):
        master.addConstr(eta <= theta[n] - features[n] @ coeff, name=f"eta_le_theta_{n}")
    master.setObjective(LEVEL @ coeff + eta, GRB.MAXIMIZE)
    master.update()
    solver = BendersDecompositionSolver(master_model=master, workers=build_workers(), imm_cost=None, theta_vars=theta,
                                        action_vars=coeff, objective_fn=worst_case_fn(features))
    solver.scenario_features = np.asarray(features, dtype=float)
    return solver


def worst_case_value(solver, theta):
    values, _ = solver.evaluate_action(theta, parallel=False)
    return float(LEVEL @ theta + (values - solver.scenario_features @ theta).min())


def test_worst_case_objective_converges_to_level_plus_worst_residual():
    solver = build_worst_case_solver(FEATURES)
    value, info = solver.solve(parallel=False, max_iter=300)
    theta = np.asarray(info['action'])
    assert info['gap'] < 1e-6
    assert np.isclose(info['evaluated_value'], worst_case_value(solver, theta), atol=1e-6)
    assert np.isclose(solver.evaluate_action(theta, parallel=False)[1], info['evaluated_value'], atol=1e-6)
    for probe in (np.zeros(N_COEFF), np.array([1.0, -1.0, 0.5, 2.0]), np.array([-3.0, 2.0, 2.0, -1.0])):
        assert worst_case_value(solver, theta) >= worst_case_value(solver, probe) - 1e-6


def test_epsilon_min_norm_trades_a_bounded_objective_loss_for_a_smaller_action():
    tight = build_solver('mean')
    tight_value, tight_info = tight.solve(parallel=False, max_iter=300, min_norm_action=True)
    tight_action = np.asarray(tight_info['action'], dtype=float)

    slack = 0.05
    relaxed = build_solver('mean')
    relaxed_value, relaxed_info = relaxed.solve(parallel=False, max_iter=300, min_norm_action=True,
                                                min_norm_slack=slack)
    relaxed_action = np.asarray(relaxed_info['action'], dtype=float)

    assert np.abs(relaxed_action).sum() <= np.abs(tight_action).sum() + 1e-9
    assert relaxed_value >= tight_value - slack - 1e-6
    assert np.isclose(relaxed.evaluate_action(relaxed_action, parallel=False)[1], relaxed_info['evaluated_value'],
                      atol=1e-6)
    assert relaxed_info['min_norm_slack'] == slack
    assert relaxed_info['min_norm_loss'] <= slack + 1e-6

    huge = build_solver('mean')
    huge_value, huge_info = huge.solve(parallel=False, max_iter=300, min_norm_action=True, min_norm_slack=1e6)
    assert np.abs(np.asarray(huge_info['action'], dtype=float)).sum() <= np.abs(relaxed_action).sum() + 1e-9
    assert huge_info['min_norm_loss'] <= 1e6


def test_epsilon_min_norm_is_off_by_default():
    solver = build_solver('mean')
    _, info = solver.solve(parallel=False, max_iter=300, min_norm_action=True)
    assert info.get('min_norm_slack') in (None, 0.0)


def test_min_norm_re_solve_can_run_repeatedly():
    solver = build_solver('mean')
    solver.solve(parallel=False, max_iter=300)
    solver.master_model.optimize()
    reference = solver.master_model.ObjVal
    first, _ = solver._min_norm_master_action(np.zeros(N_COEFF), None, slack=0.05, reference_value=reference)
    second, _ = solver._min_norm_master_action(np.zeros(N_COEFF), None, slack=0.01, reference_value=reference)
    assert np.abs(first).sum() <= np.abs(second).sum() + 1e-9
