"""L1/L2 regularization of the Benders master (toy generalized-Benders problem).

The toy master is regularized here the same way
``ApproxQAgent.train_master_builder_fn`` does it (L2: ``- lambda ||s theta||^2``
on the objective; L1: ``coefficient_abs`` epigraph rows); the agent-level
behaviour and input validation are covered by ``test_agent_regularization``.

Run from the repo root:  python -m test.test_benders_regularization
"""
import os
import tempfile

import numpy as np
from gurobipy import GRB

from metaheuristic_algorithm.benders_decomposition_solver import BendersDecompositionSolver
from test.test_benders_cut_purging import build_solver, COEFF_BOUND, N_COEFF


def build_regularized(kind, lam, scale=None):
    """Toy solver whose master carries ``- lam * R(scale * coeff)``; also
    returns the numeric ``coeff -> lam * R(scale * coeff)``."""
    base = build_solver()
    master, coeff = base.master_model, base.action_vars
    scale = np.ones(N_COEFF) if scale is None else np.asarray(scale, dtype=float)
    scaled = scale * coeff
    if kind == 'l2':
        penalty = scaled @ scaled

        def regularizer(action):
            v = scale * np.asarray(action, dtype=float)
            return lam * float(v @ v)
    else:
        abs_vars = master.addMVar(N_COEFF, lb=0.0, name="coefficient_abs")
        master.addConstr(abs_vars >= scaled, name="l1_abs_pos")
        master.addConstr(abs_vars >= -scaled, name="l1_abs_neg")
        penalty = abs_vars.sum()

        def regularizer(action):
            v = scale * np.asarray(action, dtype=float)
            return lam * float(np.abs(v).sum())
    master.setObjective(master.getObjective() - lam * penalty, GRB.MAXIMIZE)
    master.update()
    solver = BendersDecompositionSolver(
        master_model=master, workers=base.workers, imm_cost=None,
        theta_vars=base.theta_vars, action_vars=coeff)
    return solver, regularizer


def solve(solver, **kwargs):
    obj, info = solver.solve(parallel=False, max_iter=300, tol=1e-8, **kwargs)
    assert 'debug' not in info, info
    return obj, np.array(solver.action_vars.X, dtype=float)


def test_lambda_zero_matches_unregularized():
    obj_free, _ = solve(build_solver())
    for kind in ('l1', 'l2'):
        solver, _ = build_regularized(kind, 0.0)
        obj, _ = solve(solver)
        assert abs(obj - obj_free) < 1e-6, (kind, obj, obj_free)


def test_l2_shrinks_and_gap_closes():
    obj_free, theta_free = solve(build_solver())
    norms = []
    for lam in (0.0, 0.05, 0.5, 5.0):
        solver, regularizer = build_regularized('l2', lam)
        obj, theta = solve(solver)
        values, saa = solver.evaluate_action(theta, parallel=False)
        assert saa <= obj_free + 1e-6
        assert abs(obj - (saa - regularizer(theta))) < 1e-5, (obj, saa, regularizer(theta))
        norms.append(float(np.linalg.norm(theta)))
    for a, b in zip(norms, norms[1:]):
        assert b <= a + 1e-7, norms


def test_l1_sparsity_and_large_lambda_zero():
    counts = []
    for lam in (0.0, 0.1, 1.0, 10.0):
        solver, _ = build_regularized('l1', lam)
        obj, theta = solve(solver)
        counts.append(int(np.count_nonzero(np.abs(theta) > 1e-6)))
    for a, b in zip(counts, counts[1:]):
        assert b <= a, counts
    assert np.max(np.abs(theta)) < 1e-6, theta
    bound_at_zero = float(np.mean([w.initial_cut[0] for w in solver.workers]))
    assert abs(obj - bound_at_zero) < 1e-6, (obj, bound_at_zero)


def test_scaled_regularizer_values():
    scale = np.array([1.0, 2.0, 0.5, 4.0])
    a = np.array([1.0, -1.0, 2.0, 0.0])
    _, l1 = build_regularized('l1', 0.3, scale)
    assert np.isclose(l1(a), 0.3 * np.abs(scale * a).sum())
    _, l2 = build_regularized('l2', 0.3, scale)
    assert np.isclose(l2(a), 0.3 * float((scale * a) @ (scale * a)))


def test_evaluate_action_matches_worker_solve():
    solver = build_solver()
    a = np.array([0.5, -1.0, 2.0, -3.0])
    values, mean = solver.evaluate_action(a, parallel=False)
    expected = np.array([w.solve(a)[1] for w in solver.workers])
    assert np.allclose(values, expected) and np.isclose(mean, expected.mean())


def test_resume_from_unregularized_checkpoint_matches_fresh():
    # Restored bounds are reporting only; the gap rule never consults them.
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, 'free_ckpt')
        free = build_solver()
        solve(free, checkpoint_path=ckpt)
        fresh, _ = build_regularized('l2', 5.0)
        obj_fresh, _ = solve(fresh)
        resumed, _ = build_regularized('l2', 5.0)
        obj_resumed, _ = solve(resumed, resume_checkpoint_path=ckpt)
        assert abs(obj_fresh - obj_resumed) < 1e-5, (obj_fresh, obj_resumed)


def test_init_solution_hard_bound_carries_the_regularizer():
    # The pinned first iteration's value under the full (regularized) master
    # objective is saa(theta_0) - R(theta_0); the hard bound must use exactly
    # that, and the solve must still reach the fresh optimum.
    # L1 keeps the master objective linear, so the hard bound is a linear row.
    theta_0 = np.array([0.5, -1.0, 2.0, -3.0])
    fresh, regularizer = build_regularized('l1', 0.5)
    obj_fresh, _ = solve(fresh)
    solver, regularizer = build_regularized('l1', 0.5)
    _, saa_0 = solver.evaluate_action(theta_0, parallel=False)
    obj, theta = solve(solver, init_solution=theta_0, is_hard_bound=True)
    assert abs(obj - obj_fresh) < 1e-5, (obj, obj_fresh)
    # The pinned bounds were restored: the returned action is the optimum, not theta_0.
    assert not np.allclose(theta, theta_0)
    assert np.allclose(np.array(solver.action_vars.lb), -COEFF_BOUND) and np.allclose(np.array(solver.action_vars.ub), COEFF_BOUND)
    solver.master_model.update()
    constr = solver.master_model.getConstrByName('init_solution_lower_bound')
    assert constr is not None
    assert abs(constr.RHS - (saa_0 - regularizer(theta_0))) < 1e-6, (constr.RHS, saa_0, regularizer(theta_0))


if __name__ == '__main__':
    test_lambda_zero_matches_unregularized()
    test_l2_shrinks_and_gap_closes()
    test_l1_sparsity_and_large_lambda_zero()
    test_scaled_regularizer_values()
    test_evaluate_action_matches_worker_solve()
    test_resume_from_unregularized_checkpoint_matches_fresh()
    test_init_solution_hard_bound_carries_the_regularizer()
    print('All Benders regularization tests passed.')
