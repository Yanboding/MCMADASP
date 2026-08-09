"""Tests for inactive-cut purging in ``BendersDecompositionSolver.solve``.

Builds a small generalized-Benders toy that mirrors the coefficient-training
setup of ``ApproxQAgent.benders_decomposition_train``: a maximization master
over bounded coefficients ``a`` with one theta per scenario, and per-scenario
LP subproblems

    Q_s(a) = min_x  c_s . x + a . (x - 0.5)   s.t.  x in [0,1]^n, sum(x) >= 2,

whose Benders cut at ``a_k`` is ``theta_s <= Q_s(a_k) + phi_s(x*) . (a - a_k)``
with ``phi_s(x*) = x* - 0.5`` read off an auxiliary feature variable, exactly
like the training subproblems. The ``- 0.5`` shift makes the feature gradient
change sign across coordinates so the master genuinely zigzags for several
iterations and old cuts go slack.

Verifies that
1. purging drops master cuts that stay inactive for ``purge_after``
   consecutive master solves without changing the converged objective, while
   every theta keeps at least one supporting cut; and
2. with checkpointing enabled the cuts file is rewritten to exactly the
   surviving cuts (file records == meta cut_count == master cut rows, and
   fewer than the baseline run's total), and resuming from the purged
   checkpoint reproduces the same objective.

Run from the repo root:  python -m test.test_benders_cut_purging
"""
import json
import os
import tempfile

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from metaheuristic_algorithm import BendersDecompositionSolver, SubproblemWorker


N_COEFF = 4
COEFF_BOUND = 5.0
SCENARIO_COSTS = np.array([
    [1.0, 2.0, 3.0, 4.0],
    [4.0, 1.0, 2.0, 3.0],
    [2.5, 3.5, 1.5, 0.5],
    [0.5, 4.0, 0.5, 2.0],
])


def build_solver():
    master = gp.Model("toy_master")
    master.Params.OutputFlag = 0
    master.Params.Method = 1
    theta = master.addMVar(len(SCENARIO_COSTS), lb=-GRB.INFINITY, ub=1e8, name="theta")
    coeff = master.addMVar(N_COEFF, lb=-COEFF_BOUND, ub=COEFF_BOUND, name="coeff")
    master.setObjective(theta.sum() / theta.shape[0], GRB.MAXIMIZE)
    master.update()

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

        workers.append(SubproblemWorker(
            model=sub,
            link_rows=None,
            state_linking_constraints=None,
            subproblem_id=sid,
            objective_builder_fn=objective_builder_fn,
            cut_gradient_fn=cut_gradient_fn,
            verbose=False,
            initial_cut=initial_cut,
        ))

    return BendersDecompositionSolver(
        master_model=master,
        workers=workers,
        imm_cost=None,
        theta_vars=theta,
        action_vars=coeff,
    )


def count_cut_records(solver, checkpoint_path):
    cuts_path = solver._checkpoint_paths(checkpoint_path)[1]
    return sum(1 for _ in solver._iter_cut_records(cuts_path))


def test_purging_bounds_master_and_preserves_objective():
    baseline = build_solver()
    obj_baseline, _ = baseline.solve(parallel=False, max_iter=100)
    baseline.master_model.update()
    rows_baseline = baseline.master_model.NumConstrs

    purged = build_solver()
    obj_purged, _ = purged.solve(parallel=False, max_iter=100, purge_after=1)
    purged.master_model.update()
    rows_purged = purged.master_model.NumConstrs

    assert abs(obj_baseline - obj_purged) < 1e-5, (
        f"Purged objective {obj_purged} differs from baseline {obj_baseline}")
    assert rows_purged < rows_baseline, (
        f"Expected purging to shrink the master: purged rows {rows_purged} "
        f"vs baseline rows {rows_baseline}")
    for scenario_id in range(len(SCENARIO_COSTS)):
        column = purged.master_model.getCol(purged.theta_vars[scenario_id].item())
        assert column.size() >= 1, (
            f"theta_{scenario_id} lost every supporting cut after purging")
    print(f"PASS: objective {obj_purged:.6f} preserved, "
          f"master rows {rows_baseline} -> {rows_purged}")


def test_purged_checkpoint_rewrite_and_resume():
    with tempfile.TemporaryDirectory() as tmp_dir:
        baseline_ckpt = os.path.join(tmp_dir, 'baseline_ckpt')
        baseline = build_solver()
        obj_baseline, _ = baseline.solve(
            parallel=False, max_iter=100, checkpoint_path=baseline_ckpt)
        total_records = count_cut_records(baseline, baseline_ckpt)

        purged_ckpt = os.path.join(tmp_dir, 'purged_ckpt')
        purged = build_solver()
        obj_purged, _ = purged.solve(
            parallel=False, max_iter=100, checkpoint_path=purged_ckpt,
            purge_after=1)
        surviving_records = count_cut_records(purged, purged_ckpt)
        meta_path = purged._checkpoint_paths(purged_ckpt)[0]
        with open(meta_path, 'r', encoding='utf-8') as handle:
            meta = json.load(handle)
        purged.master_model.update()
        rows_purged = purged.master_model.NumConstrs

        assert surviving_records < total_records, (
            f"Expected the rewritten cuts file to shrink: {surviving_records} "
            f"records vs baseline {total_records}")
        assert meta['cut_count'] == surviving_records, (
            f"meta cut_count {meta['cut_count']} != cuts-file records "
            f"{surviving_records}")
        assert rows_purged == surviving_records, (
            f"master rows {rows_purged} != checkpointed surviving cuts "
            f"{surviving_records}")

        resumed = build_solver()
        obj_resumed, _ = resumed.solve(
            parallel=False, max_iter=100, resume_checkpoint_path=purged_ckpt)
        assert abs(obj_resumed - obj_baseline) < 1e-5, (
            f"Resume from purged checkpoint gave {obj_resumed}, "
            f"baseline {obj_baseline}")
    print(f"PASS: checkpoint rewritten {total_records} -> {surviving_records} "
          f"records, resume objective {obj_resumed:.6f} matches")


def test_train_enables_purging_by_default():
    """``benders_decomposition_train`` must forward purging to the solver.

    Runs a real (tiny) coefficient training on the toy config and checks that
    the underlying ``BendersDecompositionSolver`` ran with inactive-cut
    purging enabled -- the whole point of the feature is that the long
    case-study trainings get it without touching run.py.
    """
    from decision_maker import ApproxQAgent
    from experiments import get_config_by_type
    from generating_function import LinearPenaltyFunction

    config = get_config_by_type('toy')
    env = config.env
    generating_function = LinearPenaltyFunction(env=env)
    generating_function.set_coefficients(
        [0.0] * generating_function.number_of_coefficients)
    agent = ApproxQAgent(env,
                         discount_factor=env.discount_factor,
                         sample_path_number=2,
                         generating_function=generating_function,
                         solver_name='approx_Q')
    obj, coefficients, _ = agent.benders_decomposition_train(parallel=False)
    assert agent.coefficient_model._cut_purge_enabled, (
        "benders_decomposition_train should enable inactive-cut purging by default")
    assert np.isfinite(obj)
    assert len(coefficients) == generating_function.number_of_coefficients
    print(f"PASS: training ran with purging enabled, objective {obj:.4f}")


def main():
    test_purging_bounds_master_and_preserves_objective()
    test_purged_checkpoint_rewrite_and_resume()
    test_train_enables_purging_by_default()
    print("All cut-purging tests passed.")


if __name__ == '__main__':
    main()
