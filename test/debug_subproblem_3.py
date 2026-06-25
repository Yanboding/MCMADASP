"""Standalone reproduction + root-cause analysis for the
"Failed to retrieve Pi dual values for subproblem 3" bug.

Subproblem 3 is the longest sample path (length 313) in case 2. With
``future_decision_var_type = continuous`` the subproblem is a *pure LP*
(IsMIP=0), so the Benders worker extracts duals through

    duals = [c.Pi for c in self.link_rows]          # solver, LP branch

That requires a simplex *basis*. The training builder solves the cold model
with barrier (Method=2) + crossover, and only switches to dual simplex
(Method=1) when the cold solve took <= 60s. On the busy compute node
subproblem 3's cold solve took 256s, so it KEEPS barrier -- and a
barrier-optimal point without a usable basis has no ``.Pi``.

This script rebuilds ONLY subproblem 3 and then re-runs the LP with the exact
production parameters, with the Gurobi log turned ON, so we can see why the
solver does (or does not) return dual values. Each solve is time-limited so the
whole run is bounded.

Run (after the usual module/venv setup):
    python -u test/debug_subproblem_3.py
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

from utils import acquire_grb_env, set_link_rhs, get_status_string
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
PER_SOLVE_TIME_LIMIT = float(os.environ.get("PER_SOLVE_TIME_LIMIT", "300"))
os.makedirs("TMP", exist_ok=True)


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


def build_subproblem(scenario_id):
    params = json.load(open(PARAMS_FILE))
    env_args = params["env_args"]
    sample_path_number = params["sample_path_number"]
    init_state_seed = params["init_state_seed"]
    agent_args = params["agent_args"]

    training_spec = _normalize_generating_function_spec(
        params["training_generating_function_spec"]
    )
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

    print(f"sample path length for scenario {scenario_id} = {len(agent.delta[scenario_id])}")

    resolved_init_state = _draw_per_scenario_init_states(
        env_args=env_args,
        init_state_seed=init_state_seed,
        sample_path_number=sample_path_number,
    )
    env.reset_random_seeds()

    t0 = time.time()
    sub_model, link_rows = agent.train_subproblem_builder_fn(
        env=grb_env,
        scenario_id=scenario_id,
        init_state=resolved_init_state[scenario_id],
    )
    print(f"Built subproblem {scenario_id} (incl. cold solve) in {time.time() - t0:.1f}s")
    return sub_model, _flatten_constrs(link_rows)


def read_pi(link_rows, tag):
    """Try to read the link-row duals exactly like the Benders worker does."""
    try:
        duals = np.array([c.Pi for c in link_rows], dtype=float)
        print(f"  [{tag}] Pi OK   ||duals||={np.linalg.norm(duals):.6g} "
              f"nnz={(np.abs(duals) > 1e-9).sum()}/{len(duals)}")
        return True
    except gp.GurobiError as exc:
        print(f"  [{tag}] Pi FAILED -> GurobiError {exc.errno}: {exc}")
    except AttributeError as exc:
        print(f"  [{tag}] Pi FAILED -> AttributeError: {exc}")
    return False


def main():
    sub_model, link_rows = build_subproblem(SCENARIO_ID)

    print("-" * 72)
    print(f"IsMIP={sub_model.IsMIP} NumVars={sub_model.NumVars} "
          f"NumConstrs={sub_model.NumConstrs} NumIntVars={sub_model.NumIntVars}")
    print(f"num link rows = {len(link_rows)}   build-time link RHS[0]={link_rows[0].RHS}")
    print(f"per-solve TimeLimit = {PER_SOLVE_TIME_LIMIT}s")
    zero = [0.0] * len(link_rows)

    def configure(method, crossover, numfocus, feastol, opttol):
        sub_model.Params.OutputFlag = 1
        sub_model.Params.LogToConsole = 1
        sub_model.Params.TimeLimit = PER_SOLVE_TIME_LIMIT
        sub_model.Params.Method = method
        sub_model.Params.Crossover = crossover
        sub_model.Params.NumericFocus = numfocus
        sub_model.Params.FeasibilityTol = feastol
        sub_model.Params.OptimalityTol = opttol
        sub_model.Params.LPWarmStart = 2
        sub_model.Params.InfUnbdInfo = 1
        sub_model.Params.DualReductions = 0
        sub_model.Params.MultiObjPre = 0

    def solve(tag, method, crossover, numfocus, feastol, opttol, reset):
        if reset:
            sub_model.reset(1)  # clear solution AND warm-start basis -> cold solve
        configure(method, crossover, numfocus, feastol, opttol)
        set_link_rhs(link_rows, zero)
        print("\n" + "=" * 72)
        print(f"[{tag}] Method={method} Crossover={crossover} NumericFocus={numfocus} "
              f"FeasTol={feastol} OptTol={opttol} reset={reset}")
        sys.stdout.flush()
        t0 = time.time()
        sub_model.optimize()
        dt = time.time() - t0
        print(f"  [{tag}] status={sub_model.Status} "
              f"({get_status_string(sub_model.Status)}) "
              f"SolCount={sub_model.SolCount} time={dt:.1f}s")
        if sub_model.SolCount > 0:
            print(f"  [{tag}] ObjVal={sub_model.ObjVal:.8g}")
        ok = read_pi(link_rows, tag)
        sys.stdout.flush()
        return ok

    # --- T1: exact production COLD solve (barrier + crossover, tol 1e-9). ----
    # This is what train_subproblem_builder_fn runs at build time.
    solve("T1 cold barrier+crossover tol1e-9",
          method=2, crossover=1, numfocus=2, feastol=1e-9, opttol=1e-9, reset=True)

    # --- T2: production ITERATION re-solve when barrier is KEPT (>60s case). -
    # Same params, NO reset -> warm. Reads .Pi exactly like the LP branch of
    # SubproblemWorker.solve. THIS reproduces the reported failure.
    solve("T2 warm barrier re-solve (BUG repro)",
          method=2, crossover=1, numfocus=2, feastol=1e-9, opttol=1e-9, reset=False)

    # --- T3: candidate FIX -- dual simplex recovers a basis -> duals. --------
    solve("T3 dual simplex tol1e-9 (FIX)",
          method=1, crossover=1, numfocus=2, feastol=1e-9, opttol=1e-9, reset=False)

    # --- T4: cold dual simplex from scratch (how slow without warm basis?). --
    solve("T4 cold dual simplex tol1e-9",
          method=1, crossover=1, numfocus=2, feastol=1e-9, opttol=1e-9, reset=True)

    # --- T5: cold barrier + crossover with default-ish tol 1e-6. -------------
    solve("T5 cold barrier+crossover tol1e-6",
          method=2, crossover=1, numfocus=2, feastol=1e-6, opttol=1e-6, reset=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
