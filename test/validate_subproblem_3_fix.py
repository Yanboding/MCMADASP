"""Validate the true-LP reformulation of the training subproblem builder.

`train_subproblem_builder_fn` now builds the scenario subproblem as a genuine
LP (penalty coefficients as constants) and returns

    (sub_model, None, objective_builder_fn, cut_gradient_fn)

The Benders cut is recovered from the penalty feature vector phi(x*) instead of
the linking-constraint duals `.Pi`. By the envelope theorem this gradient equals
the dual of the old ``coefficients == action`` linking constraint, so the cut is
mathematically identical to the previous bilinear-QP formulation -- but the LP
can be warm-started by simplex.

This script checks:

  PART 1 (the exact case that crashed): build the LP for subproblem 3 and confirm
    the cold solve at zero coefficients reproduces the captured QP oracle
    objective (ObjVal ~= 1110174.6).

  PART 2 (LP == QP equivalence, on the fastest scenario for speed): rebuild the
    OLD bilinear-QP subproblem inline and compare, at several coefficient
    vectors a:
      * ObjVal(LP) == ObjVal(QP)                          (must match tightly)
      * cut_gradient_fn(LP) == linking-constraint .Pi(QP) (subgradient match)
    plus a finite-difference check that cut_gradient_fn is a true subgradient of
    the LP value function Q(a).

Run (after the usual module/venv setup):
    python -u test/validate_subproblem_3_fix.py
"""
import os
import sys
import json
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import acquire_grb_env, set_link_rhs, get_status_string, flatten
from decision_maker import ApproxQAgent
from run import (
    get_config_by_type,
    _build_generating_function,
    _normalize_generating_function_spec,
    _set_sample_path_proposal,
    _draw_per_scenario_init_states,
    _GENERATING_FUNCTION_SPEC_KEYS,
)

SCENARIO_ID = 3
PARAMS_FILE = os.environ.get("PARAMS_FILE", "/tmp/params_case2.json")

# Captured from the previous bilinear-QP solve of subproblem 3 at zero
# coefficients (TMP/debug_sub3.log). Used as a regression oracle for ObjVal.
ORACLE_OBJVAL = 1110174.6
ORACLE_DUALS_NORM = 85851.3
ORACLE_DUALS_NNZ = 282


def _flatten_constrs(obj):
    if obj is None:
        return []
    if hasattr(obj, "tolist"):
        return _flatten_constrs(obj.tolist())
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(_flatten_constrs(x))
        return out
    return [obj]


def build_qp_subproblem(agent, grb_env, scenario_id, init_state):
    """Faithful rebuild of the PREVIOUS bilinear-QP subproblem builder.

    Coefficients are Gurobi variables pinned by ``coefficients == action``
    linking constraints, so the penalty contributes ``coefficient_var *
    decision_var`` quadratic terms. Returns (model, linking_constraints).
    """
    sub_model = gp.Model(f"QP_Subproblem_{scenario_id}", env=grb_env)
    sub_model.setParam("Method", 2)
    sub_model.setParam("Crossover", 1)
    sub_model.setParam("LPWarmStart", 2)
    sub_model.setParam("InfUnbdInfo", 1)
    sub_model.setParam("NumericFocus", 2)
    sub_model.setParam("DualReductions", 0)
    sub_model.setParam("MultiObjPre", 0)
    sub_model.setParam("FeasibilityTol", 1e-9)
    sub_model.setParam("OptimalityTol", 1e-9)

    gen = agent._require_generating_function()
    coefficient_vars = gen.get_coefficient_var(model=sub_model, coefficient_bound=GRB.INFINITY)
    coefficient_blocks = gen.get_coefficients(coefficient_vars)
    linking = gen.build_coefficient_linking_constraints(sub_model, coefficient_vars)

    state_var = agent.get_state_var(sub_model)
    state_linking = agent.build_state_linking_constraints(sub_model, state_var)
    set_link_rhs(state_linking, flatten(init_state))
    action_var = agent.get_action_var(model=sub_model, advance_scheduling_type=agent.future_decision_var_type)
    agent.add_action_space_constraints(model=sub_model, state_var=state_var, action_var=action_var)

    # Bilinear QP: the coefficient VARIABLES times the path's penalty feature.
    cost, Phi, _, _ = agent.pathwise_terms(
        sub_model, gen, gen.form('training'), agent.sample_paths[scenario_id], state_var, action_var)
    sub_model.setObjective(cost + coefficient_vars @ Phi, GRB.MINIMIZE)
    sub_model.Params.OutputFlag = 0
    return sub_model, _flatten_constrs(linking)


def solve_qp(qp_model, qp_link_rows, action):
    set_link_rhs(qp_link_rows, np.asarray(action, dtype=float).tolist())
    qp_model.optimize()
    if qp_model.Status != GRB.OPTIMAL:
        return None, None
    obj = qp_model.ObjVal
    duals = np.array([c.Pi for c in qp_link_rows], dtype=float)
    return obj, duals


def solve_lp(lp_model, objective_builder_fn, cut_gradient_fn, action):
    objective_builder_fn(lp_model, np.asarray(action, dtype=float))
    lp_model.optimize()
    if lp_model.Status != GRB.OPTIMAL:
        return None, None
    obj = lp_model.ObjVal
    duals = np.asarray(cut_gradient_fn(lp_model, np.asarray(action, dtype=float)), dtype=float)
    return obj, duals


def make_agent():
    params = json.load(open(PARAMS_FILE))
    env_args = params["env_args"]
    sample_path_number = params["sample_path_number"]
    init_state_seed = params["init_state_seed"]
    agent_args = params["agent_args"]

    training_spec = _normalize_generating_function_spec(params["training_generating_function_spec"])
    training_spec.pop("coefficients", None)

    config = get_config_by_type("infinite_custom", args=env_args)
    env = config.env
    generating_function = _build_generating_function(env=env, spec=training_spec)

    inner = dict((agent_args or {}).get("agent_args", {}))
    for spec_key in _GENERATING_FUNCTION_SPEC_KEYS:
        inner.pop(spec_key, None)
    inner["generating_function"] = generating_function
    _set_sample_path_proposal(inner)
    inner["sample_path_number"] = sample_path_number

    grb_env = acquire_grb_env({"Threads": 1}, verbose=False, wait=15)
    agent = ApproxQAgent(
        env,
        discount_factor=env.discount_factor,
        grb_env=grb_env,
        subproblem_grb_envs=None,
        **inner,
    )
    resolved_init_state = _draw_per_scenario_init_states(
        env_args=env_args,
        init_state_seed=init_state_seed,
        sample_path_number=sample_path_number,
    )
    env.reset_random_seeds()
    return agent, env, grb_env, resolved_init_state, sample_path_number


def main():
    rng = np.random.default_rng(0)
    agent, env, grb_env, resolved_init_state, sample_path_number = make_agent()
    results = []

    # ---- PART 1: subproblem 3 reproduces the captured QP oracle ObjVal -------
    print(f"\n=== PART 1: subproblem {SCENARIO_ID} LP cold solve vs QP oracle ===")
    t0 = time.time()
    lp_model, link_rows, obj_builder, cut_grad = agent.train_subproblem_builder_fn(
        env=grb_env, scenario_id=SCENARIO_ID, init_state=resolved_init_state[SCENARIO_ID],
    )
    print(f"Built LP subproblem {SCENARIO_ID} in {time.time() - t0:.1f}s "
          f"(link_rows is None -> {link_rows is None})")

    # The cold solve already optimized at zero coefficients.
    cold_status_ok = lp_model.Status == GRB.OPTIMAL
    cold_obj = lp_model.ObjVal if cold_status_ok else float("nan")
    obj_err = abs(cold_obj - ORACLE_OBJVAL) / abs(ORACLE_OBJVAL)
    check1 = cold_status_ok and obj_err < 1e-4
    print(f"CHECK 1: cold solve status={get_status_string(lp_model.Status)}, "
          f"ObjVal={cold_obj:.6g} vs oracle {ORACLE_OBJVAL:.6g} "
          f"(rel err {obj_err:.2e}) -> {'PASS' if check1 else 'FAIL'}")
    results.append(check1)

    # Cut gradient at the cold (zero-coefficient) optimum (informative: may pick a
    # different vertex than the QP barrier solve when the optimum is degenerate).
    ncoef_full = agent._require_generating_function().number_of_coefficients
    zero_a = np.zeros(ncoef_full)
    duals0 = np.asarray(cut_grad(lp_model, zero_a), dtype=float)
    print(f"         cut gradient @0: ||.||={np.linalg.norm(duals0):.6g} "
          f"(oracle ~{ORACLE_DUALS_NORM:.6g}), "
          f"nnz={(np.abs(duals0) > 1e-9).sum()}/{len(duals0)} "
          f"(oracle ~{ORACLE_DUALS_NNZ})")

    # ---- PART 2: LP == QP on a small but NON-TRIVIAL scenario ----------------
    # Pick the shortest sample path whose length is >= 10 so the dual / finite-
    # difference checks are meaningful (a length-1 path is degenerate), while
    # still building and solving quickly.
    lengths = sorted((agent.sample_paths[s].length, s) for s in range(sample_path_number))
    non_trivial = [s for (L, s) in lengths if L >= 10]
    fast_sid = non_trivial[0] if non_trivial else lengths[-1][1]
    print(f"\n=== PART 2: LP vs QP equivalence on scenario {fast_sid} "
          f"(len {agent.sample_paths[fast_sid].length}) ===")

    t0 = time.time()
    lp2, lp2_link, obj_builder2, cut_grad2 = agent.train_subproblem_builder_fn(
        env=grb_env, scenario_id=fast_sid, init_state=resolved_init_state[fast_sid],
    )
    print(f"Built LP scenario {fast_sid} in {time.time() - t0:.1f}s")
    t0 = time.time()
    qp2, qp2_link = build_qp_subproblem(agent, grb_env, fast_sid, resolved_init_state[fast_sid])
    print(f"Built QP scenario {fast_sid} in {time.time() - t0:.1f}s "
          f"({len(qp2_link)} linking constraints)")

    ncoef = len(qp2_link)
    test_actions = [
        np.zeros(ncoef),
        rng.normal(scale=20.0, size=ncoef),
        rng.normal(scale=100.0, size=ncoef),
    ]
    for idx, a in enumerate(test_actions):
        lp_obj, lp_duals = solve_lp(lp2, obj_builder2, cut_grad2, a)
        qp_obj, qp_duals = solve_qp(qp2, qp2_link, a)
        if lp_obj is None or qp_obj is None:
            print(f"CHECK 2.{idx}: a-case {idx}: a solve was non-optimal "
                  f"(LP {lp_obj}, QP {qp_obj}) -> FAIL")
            results.append(False)
            continue
        obj_rel = abs(lp_obj - qp_obj) / max(1.0, abs(qp_obj))
        dual_abs = np.linalg.norm(lp_duals - qp_duals)
        dual_rel = dual_abs / max(1.0, np.linalg.norm(qp_duals))
        obj_ok = obj_rel < 1e-5
        dual_ok = dual_rel < 1e-5
        print(f"CHECK 2.{idx}: ||a||={np.linalg.norm(a):.4g} | "
              f"ObjVal LP={lp_obj:.8g} QP={qp_obj:.8g} (rel {obj_rel:.2e}) "
              f"{'OK' if obj_ok else 'MISMATCH'} | "
              f"duals rel diff {dual_rel:.2e} "
              f"{'OK' if dual_ok else 'DIFF (alt-optima?)'} "
              f"-> {'PASS' if obj_ok else 'FAIL'}")
        # ObjVal equivalence is the unambiguous correctness signal; duals can
        # legitimately differ at degenerate optima (both are valid subgradients).
        results.append(obj_ok)

    # ---- Finite-difference check that cut_gradient_fn is a true subgradient ---
    a_fd = rng.normal(scale=50.0, size=ncoef)
    base_obj, base_duals = solve_lp(lp2, obj_builder2, cut_grad2, a_fd)
    eps = 1e-2
    coords = rng.choice(ncoef, size=min(8, ncoef), replace=False)
    fd_errors = []
    for k in coords:
        a_pert = a_fd.copy()
        a_pert[k] += eps
        pert_obj, _ = solve_lp(lp2, obj_builder2, cut_grad2, a_pert)
        fd = (pert_obj - base_obj) / eps
        denom = max(1.0, abs(base_duals[k]))
        fd_errors.append(abs(fd - base_duals[k]) / denom)
    fd_errors = np.array(fd_errors)
    check3 = np.median(fd_errors) < 1e-3 and np.max(fd_errors) < 5e-2
    print(f"\nCHECK 3 (finite-diff subgradient): median rel err {np.median(fd_errors):.2e}, "
          f"max {np.max(fd_errors):.2e} over {len(coords)} coords "
          f"-> {'PASS' if check3 else 'FAIL'}")
    results.append(check3)

    ok = all(results)
    print("\nRESULT:", "ALL CHECKS PASSED" if ok else "FAILURE",
          f"({sum(results)}/{len(results)} checks)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
