"""Benders termination on the current iteration's gap.

The solver stops when the cut-model gap at its proposed action (master
epigraph values minus subproblem values, scenario-weighted) is below ``tol``;
the returned (last) master action is therefore tol-optimal. No historical
best is kept.

Run from the repo root:  python -m test.test_benders_termination
"""
import json
import os
import tempfile

import numpy as np

from test.test_benders_cut_purging import build_solver


def _last_action_value(solver):
    action = np.array(solver.action_vars.X, dtype=float)
    _, value = solver.evaluate_action(action, parallel=False)
    return action, value


def test_loose_tolerance_certifies_the_returned_action():
    solver = build_solver()
    master_value, info = solver.solve(parallel=False, max_iter=300, tol=0.3)
    assert 'debug' not in info, info
    _, value = _last_action_value(solver)
    # master_obj >= F(theta) always (cuts overestimate); the stop rule bounds the difference.
    assert value <= master_value + 1e-9
    assert master_value - value < 0.3, (master_value, value)


def test_tight_tolerance_reaches_the_exact_optimum():
    solver = build_solver()
    master_value, info = solver.solve(parallel=False, max_iter=300, tol=1e-6)
    assert 'debug' not in info, info
    _, value = _last_action_value(solver)
    assert abs(master_value - value) < 1e-6


def test_reported_bound_is_the_current_iterate_not_a_historical_best():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, 'ckpt')
        solver = build_solver()
        solver.solve(parallel=False, max_iter=2, tol=1e-12, checkpoint_path=ckpt)
        with open(ckpt + '.meta.json') as handle:
            meta = json.load(handle)
        _, value = _last_action_value(solver)
        # Maximization master: 'lower_bound' is this iteration's evaluated value.
        assert np.isclose(meta['lower_bound'], value), (meta['lower_bound'], value)
        # On this toy the seed (a = 0) value 2.25 beats the second iterate, so a
        # historical-best rule would have reported 2.25 instead.
        assert meta['lower_bound'] < 2.25


if __name__ == '__main__':
    test_loose_tolerance_certifies_the_returned_action()
    test_tight_tolerance_reaches_the_exact_optimum()
    test_reported_bound_is_the_current_iterate_not_a_historical_best()
    print('All Benders termination tests passed.')
