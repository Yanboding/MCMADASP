"""Pathwise safety constraints in ``BendersDecompositionSolver.solve``.

Reuses the toy generalized-Benders problem of ``test_benders_cut_purging``.
Run from the repo root:  python -m test.test_pathwise_safety
"""
import numpy as np

from test.test_benders_cut_purging import build_solver


def zero_penalty_values(solver):
    return np.array([w.initial_cut[0] for w in solver.workers], dtype=float)


def final_subproblem_values(solver):
    action = np.array(solver.action_vars.X, dtype=float)
    return np.array([w.solve(action)[1] for w in solver.workers], dtype=float)


def test_hard_mode_enforces_pathwise_bound():
    unconstrained = build_solver()
    obj_free, _ = unconstrained.solve(max_iter=200, tol=1e-9)
    hard = build_solver()
    obj_hard, info = hard.solve(max_iter=200, tol=1e-9, pathwise_safety='hard')
    q0 = zero_penalty_values(hard)
    q_star = final_subproblem_values(hard)
    assert np.all(q_star >= q0 - 1e-6), (q_star, q0)
    assert obj_hard <= obj_free + 1e-6
    assert info['pathwise_safety']['mode'] == 'hard'
    assert info['pathwise_safety']['violated_paths'] == 0
    # The unconstrained optimum genuinely violates on some path, so the
    # constraint is active in this toy (otherwise the test proves nothing).
    q_free = final_subproblem_values(unconstrained)
    assert np.any(q_free < zero_penalty_values(unconstrained) - 1e-6)


def test_soft_mode_limits():
    free = build_solver()
    obj_free, _ = free.solve(max_iter=200, tol=1e-9)
    soft_zero = build_solver()
    obj_soft_zero, _ = soft_zero.solve(max_iter=200, tol=1e-9, pathwise_safety=0.0)
    assert np.isclose(obj_soft_zero, obj_free, atol=1e-6)
    hard = build_solver()
    obj_hard, _ = hard.solve(max_iter=200, tol=1e-9, pathwise_safety='hard')
    soft_big = build_solver()
    obj_soft_big, info = soft_big.solve(max_iter=200, tol=1e-9, pathwise_safety=1e4)
    assert np.isclose(obj_soft_big, obj_hard, atol=1e-5)
    assert info['pathwise_safety']['mode'] == 1e4


def test_soft_mode_converges_with_consistent_gap():
    solver = build_solver()
    obj, info = solver.solve(max_iter=200, tol=1e-6, pathwise_safety=1.0)
    assert 'debug' not in info
    assert np.isfinite(obj)


def test_missing_initial_cut_raises():
    solver = build_solver()
    solver.workers[0].initial_cut = None
    try:
        solver.solve(max_iter=5, pathwise_safety='hard')
    except ValueError as exc:
        assert 'initial_cut' in str(exc)
    else:
        raise AssertionError('expected ValueError when a worker lacks initial_cut')


if __name__ == '__main__':
    test_hard_mode_enforces_pathwise_bound()
    test_soft_mode_limits()
    test_soft_mode_converges_with_consistent_gap()
    test_missing_initial_cut_raises()
    print('All pathwise-safety tests passed.')
