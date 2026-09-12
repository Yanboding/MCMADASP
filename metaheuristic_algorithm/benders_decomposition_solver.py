import copy
import gzip
import json
import os
import time

import numpy as np
from gurobipy import GRB
from tqdm.auto import tqdm

from utils import get_status_string, solve_and_handle_errors, set_link_rhs
from concurrent.futures import ThreadPoolExecutor


def _resolve_parallel_workers(max_workers, worker_count):
    if worker_count <= 0:
        return 1
    if max_workers is not None:
        return max(1, min(int(max_workers), worker_count))

    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus is not None:
        try:
            return max(1, min(int(slurm_cpus), worker_count))
        except ValueError:
            pass

    return max(1, min(os.cpu_count() or 1, worker_count))


def _group_workers_by_env(workers):
    groups = {}
    order = []
    for worker in workers:
        env = getattr(worker, "grb_env", None)
        key = id(env) if env is not None else ("__private__", id(worker))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(worker)
    return [groups[key] for key in order]


def _dispatch_subproblem_solves(env_groups, active_workers, solve_one, executor, progress=None):
    def _solve_and_tick(worker):
        result = solve_one(worker)
        if progress is not None:
            progress.update(1)
        return result

    if executor is None:
        result_by_worker = {id(w): _solve_and_tick(w) for group in env_groups for w in group}
    else:
        def run_group(group):
            return [(w, _solve_and_tick(w)) for w in group]

        result_by_worker = {}
        for future in [executor.submit(run_group, group) for group in env_groups]:
            for worker, result in future.result():
                result_by_worker[id(worker)] = result
    return [result_by_worker[id(w)] for w in active_workers]


def benders_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        x_vars = model._action_vars
        theta_vars = model._theta_vars
        tol = model._tol
        use_pareto_cuts = model._use_pareto
        pareto_epsilon = model._pareto_epsilon
        active_workers = model._workers[:theta_vars.shape[0]]
        env_groups = model._env_groups
        executor = model._executor

        x_vals = model.cbGetSolution(x_vars)
        theta_vals = model.cbGetSolution(theta_vars)

        model._cb_iter += 1
        k = model._cb_iter
        if model._core_point is None:
            model._core_point = np.copy(x_vals)
        else:
            alpha = 1.0 / (k + 1.0)
            model._core_point = (1.0 - alpha) * model._core_point + alpha * x_vals

        current_core = model._core_point

        # 3. Solve subproblems in parallel. Different env groups run concurrently
        # on the persistent executor; workers sharing a Gurobi env are solved
        # sequentially within their group (Gurobi envs are not thread-safe for
        # concurrent optimize).
        if use_pareto_cuts:
            solve_one = lambda w: w.solve_pareto(x_vals, current_core, pareto_epsilon)
        else:
            solve_one = lambda w: w.solve(x_vals)
        if model._verbose:
            print(f"Callback iter {k}, action from incumbent: {x_vals.tolist()}")
        progress = tqdm(
            total=len(active_workers),
            desc=f"Callback iter {k} subproblems",
            leave=False,
            dynamic_ncols=True,
        ) if model._verbose else None
        start_sub = time.time()
        try:
            results = _dispatch_subproblem_solves(
                env_groups, active_workers, solve_one, executor, progress=progress
            )
        except RuntimeError as exc:
            if "can't start new thread" not in str(exc):
                raise
            # Threads exhausted: disable the pool for this and every subsequent
            # callback in this solve and fall back to sequential solves.
            print("Falling back to sequential Benders subproblem solves: unable to start worker threads.")
            model._executor = None
            results = _dispatch_subproblem_solves(
                env_groups, active_workers, solve_one, None, progress=progress
            )
        finally:
            if progress is not None:
                progress.close()
        subproblem_time = time.time() - start_sub

        feasibility_cuts_added = 0
        optimality_cuts_added = 0
        cost_to_go = 0.0
        action_delta = x_vars - x_vals
        for i, (is_feasible, obj_val, duals) in enumerate(results):
            expr = obj_val + duals @ action_delta

            if not is_feasible:
                model.cbLazy(expr >= 0)
                feasibility_cuts_added += 1
            else:
                cost_to_go += obj_val
                if model.ModelSense == GRB.MINIMIZE:
                    if theta_vals[i] < (obj_val - tol):
                        model.cbLazy(theta_vars[i] >= expr)
                        optimality_cuts_added += 1
                else:
                    if theta_vals[i] > (obj_val + tol):
                        model.cbLazy(theta_vars[i] <= expr)
                        optimality_cuts_added += 1

        worker_count = len(active_workers)
        mean_theta = float(np.mean(theta_vals)) if worker_count else 0.0
        mean_cost_to_go = cost_to_go / worker_count if worker_count else 0.0
        incumbent_obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        first_stage_cost = incumbent_obj - mean_theta
        evaluated_obj = first_stage_cost + mean_cost_to_go
        print(
            f"Callback iter {k}: master {incumbent_obj:.4f} "
            f"MIP gap {abs(incumbent_obj - evaluated_obj):.4f} | added {optimality_cuts_added} opt / "
            f"{feasibility_cuts_added} feas cuts | first-stage {first_stage_cost:.4f}, "
            f"cost-to-go {mean_cost_to_go:.4f}, evaluated obj {evaluated_obj:.4f} | "
            f"subproblems {subproblem_time:.2f}s"
        )

class SubproblemWorker:

    def __init__(self, model, link_rows, state_linking_constraints, subproblem_id, verbose: bool = True,
                 objective_builder_fn=None, cut_gradient_fn=None, grb_env=None, initial_cut=None):
        self.model = model
        # The Gurobi environment this worker's model lives in. Workers that
        # share an env must never be optimized concurrently (a Gurobi Env is
        # not thread-safe), so the solver groups workers by this env and solves
        # each group sequentially while running different envs in parallel.
        self.grb_env = grb_env
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
        self.link_rows = _flatten_constrs(link_rows)
        self._link_indices = [c.index for c in self.link_rows]
        self.state_linking_constraints = state_linking_constraints
        self.subproblem_id = subproblem_id
        self.verbose = verbose
        self.objective_builder_fn = objective_builder_fn
        self.cut_gradient_fn = cut_gradient_fn
        self.initial_cut = initial_cut

    def _set_output_flag(self, model, verbose: bool):
        model.Params.OutputFlag = 1 if verbose else 0

    def _get_link_rows_in_derived_model(self, derived_model):
        rows_in_model = derived_model.getConstrs()
        return [rows_in_model[i] for i in self._link_indices]

    def _get_feasibility_ray_for_link_rows(self, action_values, verbose: bool = False):
        relax_model = self.model.relax() if self.model.IsMIP else self.model
        try:
            self._set_output_flag(relax_model, verbose)
            # Set every param the Farkas extraction needs HERE, lazily, so the
            # workers can be built with fast IR-style defaults:
            #   InfUnbdInfo=1    -> compute the FarkasDual ray;
            #   DualReductions=0 -> unambiguous INFEASIBLE (never INF_OR_UNBD);
            #   Method=1         -> FarkasDual requires a simplex solve.
            relax_model.Params.InfUnbdInfo = 1
            relax_model.Params.DualReductions = 0
            relax_model.Params.Method = 1
            relax_link_rows = self._get_link_rows_in_derived_model(relax_model)
            set_link_rhs(relax_link_rows, action_values)
            relax_model.optimize()
            if relax_model.Status == GRB.INFEASIBLE:
                v = sum(c.FarkasDual * c.RHS for c in relax_model.getConstrs())
                ray = np.array([c.FarkasDual for c in relax_link_rows], dtype=float)
                return v, ray
            raise RuntimeError(f"Relaxation not infeasible (Status {relax_model.Status})")
        finally:
            if self.model.IsMIP: relax_model.dispose()

    def solve(self, action_values, verbose: bool = False):
        if self.link_rows is not None and len(self.link_rows) > 0:
            set_link_rhs(self.link_rows, action_values)
        if self.objective_builder_fn is not None:
            self.objective_builder_fn(self.model, action_values)
        self._set_output_flag(self.model, verbose)
        self.model.optimize()
        if self.model.Status == GRB.OPTIMAL:
            v = self.model.ObjVal
            if self.cut_gradient_fn is not None:
                duals = np.asarray(self.cut_gradient_fn(self.model, action_values), dtype=float)
            else:
                if self.model.IsMIP:
                    fixed = self.model.fixed()
                    fixed.optimize()
                    fixed_rows = self._get_link_rows_in_derived_model(fixed)
                    duals = np.array([c.Pi for c in fixed_rows], dtype=float)
                    fixed.dispose()
                else:
                    duals = np.array([c.Pi for c in self.link_rows], dtype=float)
            return True, v, duals
        else:
            if self.link_rows is None or len(self.link_rows) == 0:
                raise RuntimeError(
                    f"Subproblem {self.subproblem_id} is infeasible/unbounded but has no linking constraints for feasibility rays"
                )
            v, ray = self._get_feasibility_ray_for_link_rows(action_values, verbose)
            return False, v, ray

    def solve_pareto(self, action_values, core_point, epsilon=1e-4, verbose: bool = False):
        if self.link_rows is None or len(self.link_rows) == 0:
            return self.solve(action_values, verbose)
        set_link_rhs(self.link_rows, action_values)
        if self.objective_builder_fn is not None:
            self.objective_builder_fn(self.model, action_values)
        self._set_output_flag(self.model, verbose)
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            v, ray = self._get_feasibility_ray_for_link_rows(action_values, verbose)
            return False, v, ray

        z_star = self.model.ObjVal

        if self.model.IsMIP:
            lp_model = self.model.fixed()
        else:
            lp_model = self.model.copy()

        try:
            self._set_output_flag(lp_model, verbose)
            lp_link_rows = self._get_link_rows_in_derived_model(lp_model)
            
            # Magnanti-Wong/Papadakos perturbation
            perturbed_rhs = [(1 - epsilon) * action_values[i] + epsilon * core_point[i]
                             for i in range(len(action_values))]
            
            set_link_rhs(lp_link_rows, perturbed_rhs)
            lp_model.optimize()

            if lp_model.Status != GRB.OPTIMAL:
                # Fallback to standard duals if perturbation causes numerical issues
                set_link_rhs(lp_link_rows, action_values)
                lp_model.optimize()
            
            duals = np.array([c.Pi for c in lp_link_rows], dtype=float)
            return True, z_star, duals
        finally:
            lp_model.dispose()


def objective_confidence_interval(values, weights=None, strata=None):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    if weights is None:
        weights = np.full(values.size, 1.0 / values.size)
    else:
        weights = np.asarray(weights, dtype=float)
    if strata is None:
        strata = np.zeros(values.size, dtype=int)
    else:
        strata = np.asarray(strata)
    mean = float(np.dot(weights, values))
    variance = 0.0
    for label in np.unique(strata):
        in_stratum = strata == label
        n_h = int(np.count_nonzero(in_stratum))
        if n_h <= 1:
            continue
        total_weight = float(weights[in_stratum].sum())
        sample_var = float(values[in_stratum].var(ddof=1))
        variance += total_weight ** 2 * sample_var / n_h
    return mean, 1.96 * float(np.sqrt(variance))


class BendersDecompositionSolver:

    def __init__(self, master_model, workers, imm_cost, theta_vars, action_vars,
                 scenario_weights=None, scenario_strata=None):
        self.master_model = master_model
        self.workers = workers
        self.imm_cost = imm_cost
        self.theta_vars = theta_vars
        self.action_vars = action_vars
        scenario_count = theta_vars.shape[0]
        if scenario_weights is None:
            self._scenario_weights = np.full(scenario_count, 1.0 / scenario_count)
        else:
            self._scenario_weights = np.asarray(scenario_weights, dtype=float)
        if scenario_strata is None:
            self._scenario_strata = np.zeros(scenario_count, dtype=int)
        else:
            self._scenario_strata = np.asarray(scenario_strata)
        self._action_abs_vars = None
        # Purge bookkeeping; solve() re-initializes these per call. Set here
        # so _register_cut is safe to reach before/outside solve().
        self._cut_purge_enabled = False
        self._cut_registry = []
        self._latest_cut_by_scenario = {}

    def solve(self, 
              init_solution=None,
              is_hard_bound=False,
              tol=1e-6,
              max_iter=150,
              use_pareto_cuts=False,
              pareto_epsilon=1e-4,
              core_alpha=None,
              verbose=False,
              parallel=True,
              max_workers=None,
              checkpoint_path=None,
              resume_checkpoint_path=None,
              min_norm_action=False,
              purge_after=None,
              purge_slack_tol=1e-6):
        info = {}
        summary = {}
        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        core_point = None
        is_min = self.master_model.ModelSense == GRB.MINIMIZE
        scenario_count = self.theta_vars.shape[0]

        self._cut_purge_enabled = purge_after is not None
        self._cut_registry = []
        self._latest_cut_by_scenario = {}

        active_workers = self.workers[:scenario_count]
        env_groups = _group_workers_by_env(active_workers)
        resolved_max_workers = _resolve_parallel_workers(max_workers, len(env_groups))
        executor = ThreadPoolExecutor(max_workers=resolved_max_workers) if parallel else None

        if checkpoint_path and not resume_checkpoint_path:
            self._reset_checkpoint_store(checkpoint_path)

        checkpoint_state = self._load_cut_checkpoint(resume_checkpoint_path)
        if checkpoint_state.get('core_point') is not None:
            core_point = np.asarray(checkpoint_state['core_point'], dtype=float)
        lower_bound = checkpoint_state.get('lower_bound', lower_bound) or lower_bound
        upper_bound = checkpoint_state.get('upper_bound', upper_bound) or upper_bound
        iteration_offset = int(checkpoint_state.get('iteration', 0) or 0)
        cut_count = int(checkpoint_state.get('cut_count', 0) or 0)
        # Restored bounds are reporting only: the first iteration's master
        # solve and gap overwrite both before any convergence decision.

        # Seed the master with the build-time (a=0) Benders cuts -- one per
        # scenario -- before the first master solve. They were produced for free
        # by each subproblem's cold solve during construction, so injecting them
        # lets the solver skip the iteration that would otherwise just re-derive
        # them and starts the master from a gradient-informed approximation.
        # Skipped only when a checkpoint actually restored cuts (the reloaded
        # store already contains them). Callers routinely pass
        # ``resume_checkpoint_path`` unconditionally (pointing at a file that
        # does not exist yet on a fresh run), so keying on the path alone would
        # skip seeding on fresh runs and let the cut-less master push the
        # coefficients to absurd magnitudes (numerically breaking the
        # subproblems).
        if not checkpoint_state.get('cut_count'):
            lower_bound, upper_bound, cut_count = self._seed_initial_cuts(
                active_workers, iteration_offset, lower_bound, upper_bound,
                cut_count, checkpoint_path, core_point)
        try:
            for iteration in range(1, max_iter + 1):
                global_iteration = iteration_offset + iteration
                if init_solution is not None and iteration == 1:
                    action, master_obj, theta_values = self._solve_master_with_fixed_action(
                        init_solution, global_iteration, purge_slack_tol, verbose)
                    print(f"Iteration {global_iteration}, action pinned at init_solution")
                else:
                    action, master_obj, theta_values = self._solve_master_step(
                        global_iteration, purge_slack_tol, min_norm_action, verbose)
                if is_min:
                    lower_bound = master_obj
                else:
                    upper_bound = master_obj
                if core_point is None:
                    core_point = copy.deepcopy(action)

                print(f"Iteration {global_iteration}, action from master: {action.tolist()}")

                if use_pareto_cuts:
                    solve_one = lambda w: w.solve_pareto(action, core_point, pareto_epsilon, verbose)
                else:
                    solve_one = lambda w: w.solve(action, verbose)
                start_sub = time.time()
                results, executor = self._solve_all_subproblems(
                    active_workers, env_groups, solve_one, executor, global_iteration
                )
                parallel = executor is not None
                print(f"Iteration {global_iteration}, subproblems solved in {time.time() - start_sub:.2f}s")
                self._report_memory_usage(global_iteration, active_workers)

                all_feasible, feasibility_cuts, optimality_cuts, new_cut_records, cost_to_go_estimation = \
                    self._build_cuts(results, active_workers, action)

                if not all_feasible:
                    self._add_cuts_to_master(feasibility_cuts, new_cut_records,
                                             global_iteration, 'feasibility')
                    print('-' * 20)
                    continue

                self._add_cuts_to_master(optimality_cuts, new_cut_records,
                                         global_iteration, 'optimality')
                # Convergence is judged on THIS iteration only, through the
                # cut-model gap at the master's action: the weighted sum of
                # (master epigraph value - subproblem value) per scenario. Cuts
                # bound every Q_s from the master's side, so the gap is >= 0 in
                # exact arithmetic; every other master term (first-stage cost,
                # regularizer) is the same on both sides and cancels. When the
                # gap is within tol the cut model is exact at the returned
                # action and that action is tol-optimal. No historical best is
                # kept: an earlier, better incumbent would not be the returned
                # solution.
                scenario_gaps = self._scenario_gaps(results, active_workers, theta_values)
                scenario_ids = [worker.subproblem_id for worker in active_workers]
                gap = float(self._scenario_weights[scenario_ids] @ scenario_gaps)
                model_cost_to_go = float(self._scenario_weights @ theta_values)
                if is_min:
                    upper_bound = master_obj + gap
                else:
                    lower_bound = master_obj - gap
                summary = {
                    'master_objective': float(master_obj),
                    'evaluated_value': float(upper_bound if is_min else lower_bound),
                    'gap': gap,
                    'iterations': global_iteration,
                }

                if init_solution is not None and iteration == 1 and is_hard_bound:
                    master_obj_expr = self.master_model.getObjective()
                    if is_min:
                        self.master_model.addConstr(master_obj_expr <= upper_bound, name="init_solution_upper_bound")
                        print(f"Added hard master upper bound from init_solution: {upper_bound}")
                    else:
                        self.master_model.addConstr(master_obj_expr >= lower_bound, name="init_solution_lower_bound")
                        print(f"Added hard master lower bound from init_solution: {lower_bound}")

                # Update core point AFTER we have a valid x_k from the master.
                core_point = self.update_core_point(core_point, action, iteration, alpha=core_alpha)

                print(f"UB: {upper_bound}, LB: {lower_bound}, Gap: {gap}, "
                      f"Master cost-to-go: {model_cost_to_go}, Subproblem cost-to-go: {cost_to_go_estimation}, "
                      f"max scenario gap: {float(np.max(scenario_gaps)):.6g}")

                obj_mean, obj_half_width = objective_confidence_interval(
                    [v for (is_feasible, v, _) in results if is_feasible],
                    weights=self._scenario_weights, strata=self._scenario_strata)
                print(f"Iteration {global_iteration}, subproblem objective mean {obj_mean:.4f} "
                      f"+/- {obj_half_width:.4f} (95% CI, N={scenario_count})")

                cut_count = self._checkpoint_iteration(
                    checkpoint_path, global_iteration, lower_bound, upper_bound,
                    core_point, cut_count, new_cut_records, purge_after)

                if abs(gap) < tol:
                    info = {}
                    break
                if gap < 0:
                    # A negative gap beyond tol means some master epigraph value
                    # lies on the wrong side of its subproblem value: an invalid
                    # cut (dual rays / gradients / scenario weights) or LP
                    # tolerances far beyond the master's. Print the raw pieces.
                    worst = int(np.argmin(scenario_gaps))
                    print("Error: Lower bound exceeded upper bound (check dual rays/bounds). "
                          f"master objective {master_obj:.10g}, "
                          f"master cost-to-go {model_cost_to_go:.10g}, "
                          f"subproblem cost-to-go {cost_to_go_estimation:.10g}, "
                          f"gap {gap:.6g} (tol {tol:g}), "
                          f"worst scenario {scenario_ids[worst]} gap {float(scenario_gaps[worst]):.6g}")
                    info = {'debug': 'lower_bound_exceeded_upper_bound'}
                    break
                print('-' * 20)
            else:
                print('Max iterations reached')
        finally:
            if executor is not None:
                executor.shutdown(wait=False)
        info.update(summary)
        return upper_bound, info

    def solve_with_callback(self, tol=1e-6,
                            max_iter=150,
                            use_pareto_cuts=False,
                            pareto_epsilon=1e-4,
                            max_workers=None,
                            parallel=True,
                            verbose=False):
        # 1. Mandatory Parameter for Lazy Constraints
        self.master_model.Params.MIPGap = 0.0
        self.master_model.Params.LazyConstraints = 1

        active_workers = self.workers[:self.theta_vars.shape[0]]
        env_groups = _group_workers_by_env(active_workers)
        resolved_max_workers = _resolve_parallel_workers(max_workers, len(env_groups))
        executor = (
            ThreadPoolExecutor(max_workers=resolved_max_workers)
            if parallel and resolved_max_workers > 1
            else None
        )

        # We use the underscore prefix (_) to avoid namespace collisions
        self.master_model._action_vars = self.action_vars
        self.master_model._theta_vars = self.theta_vars
        self.master_model._workers = self.workers
        self.master_model._env_groups = env_groups
        self.master_model._executor = executor
        self.master_model._max_iterations = max_iter
        self.master_model._tol = tol
        self.master_model._use_pareto = use_pareto_cuts
        self.master_model._pareto_epsilon = pareto_epsilon
        self.master_model._core_point = None
        self.master_model._cb_iter = 0
        self.master_model._max_workers = max_workers
        self.master_model._verbose = verbose

        print("Starting Benders with Lazy Constraint Callback...")
        try:
            self.master_model.optimize(benders_callback)
        finally:
            if executor is not None:
                executor.shutdown(wait=False)
        info = {}
        if self.master_model.Status == GRB.OPTIMAL:
            return self.master_model.ObjVal, info
        return None, info

    def evaluate_action(self, action, parallel=True, max_workers=None):
        action = np.asarray(action, dtype=float)
        active_workers = self.workers[:self.theta_vars.shape[0]]
        env_groups = _group_workers_by_env(active_workers)
        executor = None
        if parallel and len(env_groups) > 1:
            executor = ThreadPoolExecutor(
                max_workers=_resolve_parallel_workers(max_workers, len(env_groups)))
        try:
            results = _dispatch_subproblem_solves(
                env_groups, active_workers, lambda w: w.solve(action), executor)
        finally:
            if executor is not None:
                executor.shutdown(wait=False)
        infeasible = [w.subproblem_id for w, (ok, _, _) in zip(active_workers, results) if not ok]
        if infeasible:
            raise RuntimeError(f"evaluate_action: scenarios {infeasible[:5]} infeasible at the given action")
        values = np.array([v for (_, v, _) in results], dtype=float)
        return values, float(self._scenario_weights @ values)

    def _solve_master_step(self, global_iteration, purge_slack_tol, min_norm_action, verbose):
        start = time.time()
        solved = solve_and_handle_errors(self.master_model, verbose=verbose)
        retry_cold = not solved and self.master_model.Status in (
            GRB.UNBOUNDED, GRB.INF_OR_UNBD, GRB.NUMERIC)
        if retry_cold:
            print(f"Master reported {get_status_string(self.master_model.Status)}; "
                  "retrying once from a cold solve with unchanged parameters.")
            self.master_model.reset()
            solved = solve_and_handle_errors(self.master_model, verbose=verbose)
        elif not solved and self.master_model.Status == GRB.INFEASIBLE:
            # A badly scaled cut matrix can make Gurobi presolve report a false
            # INFEASIBLE status even though, for example, action=0 with
            # sufficiently low theta is feasible. Keep presolve disabled for
            # subsequent master solves after this occurs; the saved case that
            # motivated this retry solves to OPTIMAL with identical tolerances
            # once presolve is bypassed.
            print(
                "Master reported INFEASIBLE; retrying with Presolve=0 to rule out "
                "a numerical presolve misclassification."
            )
            self.master_model.reset()
            self.master_model.Params.Presolve = 0
            solved = solve_and_handle_errors(self.master_model, verbose=verbose)
        if not solved:
            # A failed cold retry must not trigger another diagnostic solve.
            diagnostics = self._save_master_failure_diagnostics(
                global_iteration, clarify_status=not retry_cold)
            raise RuntimeError(
                f"Master model optimal solution not found: {diagnostics['reason']} "
                f"Diagnostics saved to {diagnostics['directory']}"
            )
        print(f"Iteration {global_iteration}, master solved in {time.time() - start} seconds")
        self._report_memory_usage(global_iteration)
        action = np.array(self.action_vars.X, dtype=float)
        theta_values = np.array(self.theta_vars.X, dtype=float)
        if self._cut_purge_enabled:
            self._update_cut_activity(global_iteration, purge_slack_tol)
        master_obj = self.master_model.ObjVal
        if min_norm_action and np.max(np.abs(action)) > 1e6:
            # Degenerate vertex on an under-constrained optimal face:
            # re-solve for the minimum-norm optimal action (bounds use the
            # primary ObjVal captured above). Sane actions skip the re-solve
            # so the endgame is untouched.
            action, theta_values = self._min_norm_master_action(
                action, theta_values, verbose=verbose)
        return action, master_obj, theta_values

    def _solve_master_with_fixed_action(self, fixed_action, global_iteration,
                                        purge_slack_tol, verbose):
        fixed = np.asarray(fixed_action, dtype=float).reshape(-1)
        if fixed.shape != (self.action_vars.shape[0],):
            raise ValueError(f"init_solution must have {self.action_vars.shape[0]} entries; got {fixed.shape}")
        original_lb = np.array(self.action_vars.lb, dtype=float)
        original_ub = np.array(self.action_vars.ub, dtype=float)
        self.action_vars.lb = fixed
        self.action_vars.ub = fixed
        try:
            _, master_obj, theta_values = self._solve_master_step(
                global_iteration, purge_slack_tol, False, verbose)
        finally:
            self.action_vars.lb = original_lb
            self.action_vars.ub = original_ub
        return fixed, master_obj, theta_values

    def _scenario_gaps(self, results, active_workers, theta_values):
        is_min = self.master_model.ModelSense == GRB.MINIMIZE
        gaps = np.empty(len(active_workers), dtype=float)
        for i, (worker, (is_feasible, v, _)) in enumerate(zip(active_workers, results)):
            assert is_feasible
            theta = float(theta_values[worker.subproblem_id])
            gaps[i] = (v - theta) if is_min else (theta - v)
        return gaps

    def _save_master_failure_diagnostics(self, global_iteration, *, clarify_status=True):
        model = self.master_model
        original_status = int(model.Status)
        clarified_status = original_status
        errors = []

        # Presolve reductions can make infeasible and unbounded indistinguishable.
        # Re-solving without dual reductions is Gurobi's documented way to
        # distinguish the two before producing an IIS or an unbounded diagnosis.
        if clarify_status and original_status == GRB.INF_OR_UNBD:
            try:
                original_dual_reductions = model.Params.DualReductions
                model.Params.DualReductions = 0
                model.optimize()
                clarified_status = int(model.Status)
                model.Params.DualReductions = original_dual_reductions
            except Exception as exc:
                errors.append(f"status clarification failed: {exc}")

        model_name = model.ModelName.strip() or "unnamed_master"
        safe_model_name = ''.join(
            character if character.isalnum() or character in ('-', '_') else '_'
            for character in model_name
        )
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        directory = os.path.abspath(os.path.join(
            "master_failure_diagnostics",
            f"{safe_model_name}_iter_{global_iteration}_{timestamp}_pid_{os.getpid()}",
        ))
        os.makedirs(directory, exist_ok=True)

        for extension in ("lp", "mps", "prm"):
            path = os.path.join(directory, f"master.{extension}")
            try:
                model.write(path)
            except Exception as exc:
                errors.append(f"could not write {extension}: {exc}")

        def read_attr(attribute):
            try:
                return model.getAttr(attribute)
            except Exception:
                return None

        min_coefficient = read_attr(GRB.Attr.MinCoeff)
        max_coefficient = read_attr(GRB.Attr.MaxCoeff)
        coefficient_ratio = (
            float(max_coefficient / min_coefficient)
            if min_coefficient is not None and max_coefficient is not None and min_coefficient > 0
            else None
        )
        variables = model.getVars()
        unbounded_below = sum(variable.LB <= -0.5 * GRB.INFINITY for variable in variables)
        unbounded_above = sum(variable.UB >= 0.5 * GRB.INFINITY for variable in variables)
        analysis = []
        if coefficient_ratio is not None and coefficient_ratio >= 1e9:
            analysis.append(
                f"matrix coefficients span a ratio of {coefficient_ratio:.3e}, which can cause numerical instability"
            )
        if unbounded_below or unbounded_above:
            analysis.append(
                f"the model has {unbounded_below} variables unbounded below and "
                f"{unbounded_above} variables unbounded above"
            )

        status_name = get_status_string(clarified_status)
        if clarified_status == GRB.INFEASIBLE:
            reason = "the master is infeasible; inspect master.ilp for the conflicting constraints or variable bounds"
            try:
                model.computeIIS()
                model.write(os.path.join(directory, "master.ilp"))
            except Exception as exc:
                errors.append(f"IIS computation failed: {exc}")
        elif clarified_status == GRB.UNBOUNDED:
            reason = "the master objective is unbounded; theta/action variables are missing a supporting cut, bound, or valid objective direction"
        elif clarified_status == GRB.NUMERIC:
            scaling_detail = f" (matrix coefficient ratio {coefficient_ratio:.3e})" if coefficient_ratio else ""
            reason = f"Gurobi stopped because of numerical difficulties{scaling_detail}; inspect the saved model"
        elif clarified_status == GRB.INTERRUPTED:
            reason = "the master optimization was interrupted before proving optimality"
        elif clarified_status in (GRB.TIME_LIMIT, GRB.NODE_LIMIT, GRB.ITERATION_LIMIT, GRB.SOLUTION_LIMIT):
            reason = f"the master stopped at its configured {status_name.lower()} before proving optimality"
        elif clarified_status == GRB.INF_OR_UNBD:
            reason = "the master is infeasible or unbounded; the diagnostic re-solve could not distinguish the two"
        else:
            reason = f"Gurobi returned non-optimal status {status_name} ({clarified_status})"

        summary = {
            "model_name": model_name,
            "iteration": int(global_iteration),
            "original_status": {
                "code": original_status,
                "name": get_status_string(original_status),
            },
            "clarified_status": {
                "code": clarified_status,
                "name": status_name,
            },
            "reason": reason,
            "model_sense": "minimize" if model.ModelSense == GRB.MINIMIZE else "maximize",
            "solution_count": read_attr(GRB.Attr.SolCount),
            "variables": read_attr(GRB.Attr.NumVars),
            "constraints": read_attr(GRB.Attr.NumConstrs),
            "nonzeros": read_attr(GRB.Attr.NumNZs),
            "unbounded_variable_counts": {
                "below": unbounded_below,
                "above": unbounded_above,
            },
            "coefficient_ranges": {
                "matrix_min": min_coefficient,
                "matrix_max": max_coefficient,
                "matrix_ratio": coefficient_ratio,
                "objective_min": read_attr(GRB.Attr.MinObjCoeff),
                "objective_max": read_attr(GRB.Attr.MaxObjCoeff),
                "rhs_min": read_attr(GRB.Attr.MinRHS),
                "rhs_max": read_attr(GRB.Attr.MaxRHS),
                "bound_min": read_attr(GRB.Attr.MinBound),
                "bound_max": read_attr(GRB.Attr.MaxBound),
            },
            "analysis": analysis,
            "errors": errors,
        }
        summary_path = os.path.join(directory, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

        print(f"Master failure diagnosis: {reason}")
        print(f"Master failure artifacts: {directory}")
        return {"directory": directory, "reason": reason, "summary": summary}

    def _min_norm_master_action(self, fallback_action, fallback_theta, verbose=False):
        model = self.master_model
        original_sense = model.ModelSense
        primary_objective = model.getObjective()
        objective_value = model.ObjVal
        if self._action_abs_vars is None:
            abs_vars = model.addMVar(
                self.action_vars.shape[0], lb=0.0, name="action_abs")
            model.addConstr(abs_vars >= self.action_vars, name="action_abs_pos")
            model.addConstr(abs_vars >= -self.action_vars, name="action_abs_neg")
            self._action_abs_vars = abs_vars
        # Pin the primary objective at its optimum, with a RELATIVE slack.
        # An exact pin is numerically infeasible in the early degenerate
        # iterations (the optimum sits at the theta upper-bound cap, reached
        # only along a near-unbounded direction), which silently falls back to
        # the degenerate vertex. The slack cannot stall the Benders endgame
        # because this re-solve only runs while max|a| > 1e6 (see solve());
        # once the cuts pin the action to sane magnitudes the loop uses the
        # untouched first-phase vertex.
        slack = 1e-9 * max(1.0, abs(objective_value))
        if original_sense == GRB.MINIMIZE:
            guard = model.addConstr(
                primary_objective <= objective_value + slack, name="min_norm_guard")
        else:
            guard = model.addConstr(
                primary_objective >= objective_value - slack, name="min_norm_guard")
        try:
            model.setObjective(self._action_abs_vars.sum(), GRB.MINIMIZE)
            model.optimize()
            if model.Status == GRB.OPTIMAL:
                return (np.array(self.action_vars.X, dtype=float),
                        np.array(self.theta_vars.X, dtype=float))
            print(f"Min-norm re-solve not optimal (Status {model.Status}); "
                  f"keeping first-phase master action")
            return fallback_action, fallback_theta
        finally:
            model.remove(guard)
            model.setObjective(primary_objective, original_sense)
            model.update()

    def _seed_initial_cuts(self, active_workers, iteration, lower_bound, upper_bound,
                           cut_count, checkpoint_path, core_point):
        is_min = self.master_model.ModelSense == GRB.MINIMIZE
        seeded_workers, seeded_results = [], []
        for worker in active_workers:
            initial_cut = getattr(worker, "initial_cut", None)
            if initial_cut is None:
                continue
            v0, g0 = initial_cut
            seeded_workers.append(worker)
            seeded_results.append((True, float(v0), np.asarray(g0, dtype=float)))
        if not seeded_workers:
            return lower_bound, upper_bound, cut_count

        zero_action = np.zeros(self.action_vars.shape[0], dtype=float)
        _, _, optimality_cuts, cut_records, cost_to_go = self._build_cuts(
            seeded_results, seeded_workers, zero_action)
        seed_constrs = self.master_model.addConstrs(
            (optimality_cuts[i] for i in range(len(optimality_cuts))),
            name="benders_seed_cut_",
        )
        for i, record in enumerate(cut_records):
            self._register_cut(seed_constrs[i], record, iteration)
        self.master_model.update()

        all_seeded = len(seeded_workers) == len(active_workers)
        # Weights sum to 1 over ALL active workers, so the weighted sum is the
        # weighted average exactly when every worker contributed a seed cut.
        bound_at_zero = cost_to_go if all_seeded else None
        print(f"Seeded master with {len(optimality_cuts)} build-time (a=0) cuts"
              + (f"; bound at zero coefficients = {bound_at_zero}" if bound_at_zero is not None else ""))
        seeded_ids = [worker.subproblem_id for worker in seeded_workers]
        zero_mean, zero_half_width = objective_confidence_interval(
            [v for _, v, _ in seeded_results],
            weights=self._scenario_weights[seeded_ids],
            strata=self._scenario_strata[seeded_ids])
        print(f"Zero-penalty subproblem objective mean {zero_mean:.4f} +/- "
              f"{zero_half_width:.4f} (95% CI, N={len(seeded_results)})")

        if bound_at_zero is not None:
            # The optimum is at least as good as all-zero coefficients, so
            # the a=0 value is a valid incumbent bound to start the gap from.
            if is_min:
                upper_bound = min(upper_bound, bound_at_zero)
            else:
                lower_bound = max(lower_bound, bound_at_zero)
        if cut_records and checkpoint_path:
            cut_count += len(cut_records)
            self._save_cut_checkpoint(
                checkpoint_path,
                {
                    'iteration': iteration,
                    'lower_bound': None if np.isinf(lower_bound) else float(lower_bound),
                    'upper_bound': None if np.isinf(upper_bound) else float(upper_bound),
                    'core_point': None if core_point is None else np.asarray(core_point, dtype=float).tolist(),
                    'cut_count': cut_count,
                },
                new_cut_records=cut_records,
            )
        return lower_bound, upper_bound, cut_count

    def _build_cuts(self, results, active_workers, action):
        is_min = self.master_model.ModelSense == GRB.MINIMIZE
        action_values = np.asarray(action, dtype=float)
        feasibility_cuts, optimality_cuts, cut_records = [], [], []
        cost_to_go = 0.0
        all_feasible = True
        for worker, (is_feasible, v, duals) in zip(active_workers, results):
            scenario_id = worker.subproblem_id
            # theta[s] >= v + duals^T (x - x_k)  (>= flips to <= for maximization)
            cut_expr = v + duals @ (self.action_vars - action)
            if not is_feasible:
                all_feasible = False
                feasibility_cuts.append(cut_expr >= 0)
                cut_records.append(
                    self._cut_to_record('feasibility', scenario_id, duals, v, action_values, 'ge')
                )
            else:
                cost_to_go += self._scenario_weights[scenario_id] * v
                if is_min:
                    optimality_cuts.append(self.theta_vars[scenario_id] >= cut_expr)
                    cut_records.append(
                        self._cut_to_record('optimality', scenario_id, duals, v, action_values, 'ge')
                    )
                else:
                    optimality_cuts.append(self.theta_vars[scenario_id] <= cut_expr)
                    cut_records.append(
                        self._cut_to_record('optimality', scenario_id, duals, v, action_values, 'le')
                    )
        return all_feasible, feasibility_cuts, optimality_cuts, cut_records, cost_to_go

    def _add_cuts_to_master(self, cuts, records, iteration, kind):
        print(f"Iteration {iteration}, adding {len(cuts)} {kind} cuts")
        prefix = 'feas' if kind == 'feasibility' else 'opt'
        constrs = self.master_model.addConstrs(
            (cuts[i] for i in range(len(cuts))),
            name=f"{prefix}_cut_{iteration}_",
        )
        kind_records = [r for r in records if r['kind'] == kind]
        for i, record in enumerate(kind_records):
            self._register_cut(constrs[i], record, iteration)

    def _register_cut(self, constr, record, iteration):
        if not self._cut_purge_enabled:
            return
        # Cuts built from MVar expressions come back as 0-d/1-element MConstr;
        # bulk getAttr/remove need the scalar Constr.
        if hasattr(constr, 'tolist'):
            constr = constr.tolist()
        while isinstance(constr, (list, tuple)):
            constr = constr[0]
        entry = {'constr': constr, 'record': record, 'last_active': iteration}
        self._cut_registry.append(entry)
        if record['kind'] == 'optimality':
            self._latest_cut_by_scenario[record['scenario_id']] = entry

    def _update_cut_activity(self, iteration, slack_tol):
        entries = [e for e in self._cut_registry if e['record']['kind'] == 'optimality']
        if not entries:
            return
        constrs = [e['constr'] for e in entries]
        slacks = self.master_model.getAttr(GRB.Attr.Slack, constrs)
        rhs_values = self.master_model.getAttr(GRB.Attr.RHS, constrs)
        for entry, slack, rhs in zip(entries, slacks, rhs_values):
            if abs(slack) <= slack_tol * max(1.0, abs(rhs)):
                entry['last_active'] = iteration

    def _purge_inactive_cuts(self, iteration, purge_after):
        protected = {id(entry) for entry in self._latest_cut_by_scenario.values()}
        keep, stale = [], []
        for entry in self._cut_registry:
            is_stale = (entry['record']['kind'] == 'optimality'
                        and id(entry) not in protected
                        and iteration - entry['last_active'] >= purge_after)
            (stale if is_stale else keep).append(entry)
        if stale:
            self.master_model.remove([entry['constr'] for entry in stale])
            self._cut_registry = keep
        return len(stale)

    def _checkpoint_paths(self, checkpoint_path):
        return f"{checkpoint_path}.meta.json", f"{checkpoint_path}.cuts.jsonl.gz"

    def _encode_vector(self, values, zero_tol=1e-12, sparse_density=0.35):
        vector = np.asarray(values, dtype=float).ravel()
        nonzero_idx = np.flatnonzero(np.abs(vector) > zero_tol)

        if nonzero_idx.size == 0:
            return {
                'format': 'zero',
                'size': int(vector.size),
            }

        if nonzero_idx.size <= sparse_density * vector.size:
            return {
                'format': 'sparse',
                'size': int(vector.size),
                'indices': nonzero_idx.tolist(),
                'values': vector[nonzero_idx].tolist(),
            }

        return {
            'format': 'dense',
            'values': vector.tolist(),
        }

    def _decode_vector(self, payload):
        if isinstance(payload, list):
            return np.asarray(payload, dtype=float)

        vector_format = payload.get('format', 'dense')
        if vector_format == 'zero':
            return np.zeros(int(payload['size']), dtype=float)
        if vector_format == 'sparse':
            vector = np.zeros(int(payload['size']), dtype=float)
            indices = np.asarray(payload['indices'], dtype=int)
            values = np.asarray(payload['values'], dtype=float)
            vector[indices] = values
            return vector
        if vector_format == 'dense':
            return np.asarray(payload['values'], dtype=float)

        raise ValueError(f"Unsupported vector format: {vector_format}")

    def _normalize_cut_record(self, record):
        coefficients = self._decode_vector(record['coefficients'])
        if 'intercept' in record:
            intercept = float(record['intercept'])
        else:
            action_values = self._decode_vector(record['action_values'])
            intercept = float(record['rhs_value']) - float(np.dot(coefficients, action_values))

        return {
            'kind': record['kind'],
            'scenario_id': int(record['scenario_id']),
            'sense': record.get('sense', 'ge'),
            'intercept': intercept,
            'coefficients': self._encode_vector(coefficients),
        }

    def _cut_to_record(self, kind, scenario_id, coefficients, rhs_value, action_values, sense='ge'):
        coefficients = np.asarray(coefficients, dtype=float)
        action_values = np.asarray(action_values, dtype=float)
        intercept = float(rhs_value) - float(np.dot(coefficients, action_values))
        return {
            'kind': kind,
            'scenario_id': int(scenario_id),
            'sense': sense,
            'intercept': intercept,
            'coefficients': self._encode_vector(coefficients),
        }

    def _cut_record_to_constraint(self, record):
        record = self._normalize_cut_record(record)
        coefficients = self._decode_vector(record['coefficients'])
        intercept = float(record['intercept'])
        cut_expr = intercept + sum(coefficients[j] * self.action_vars[j] for j in range(len(coefficients)))

        if record['kind'] == 'optimality':
            theta_var = self.theta_vars[record['scenario_id']]
            if record['sense'] == 'le':
                return theta_var <= cut_expr
            return theta_var >= cut_expr

        if record['kind'] == 'feasibility':
            if record['sense'] == 'le':
                return cut_expr <= 0
            return cut_expr >= 0

        raise ValueError(f"Unsupported cut kind: {record['kind']}")

    def _reset_checkpoint_store(self, checkpoint_path):
        if not checkpoint_path:
            return

        meta_path, cuts_path = self._checkpoint_paths(checkpoint_path)
        for path in (checkpoint_path, meta_path, cuts_path):
            if os.path.exists(path):
                os.remove(path)

    def _read_json_file(self, path):
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as handle:
                return json.load(handle)
        except (OSError, gzip.BadGzipFile):
            with open(path, 'r', encoding='utf-8') as handle:
                return json.load(handle)

    def _iter_cut_records(self, path):
        if not os.path.exists(path):
            return

        try:
            with gzip.open(path, 'rt', encoding='utf-8') as handle:
                for line in handle:
                    if line.strip():
                        yield json.loads(line)
            return
        except (OSError, gzip.BadGzipFile):
            pass

        with open(path, 'r', encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def _load_cut_checkpoint(self, checkpoint_path):
        empty_state = {
            'cuts': [],
            'cut_count': 0,
            'core_point': None,
            'lower_bound': None,
            'upper_bound': None,
            'iteration': 0,
        }
        if not checkpoint_path:
            return empty_state

        meta_path, cuts_path = self._checkpoint_paths(checkpoint_path)

        if os.path.exists(meta_path):
            state = self._read_json_file(meta_path)
            reload_iteration = int(state.get('iteration', 0) or 0)
            cut_records = []
            for cut_index, record in enumerate(self._iter_cut_records(cuts_path)):
                normalized = self._normalize_cut_record(record)
                cut_records.append(normalized)
                constr = self.master_model.addConstr(
                    self._cut_record_to_constraint(normalized),
                    name=f"reloaded_{normalized['kind']}_cut_{cut_index}",
                )
                self._register_cut(constr, normalized, reload_iteration)
            state['cuts'] = cut_records
            state['cut_count'] = len(cut_records)
            return state

        if not os.path.exists(checkpoint_path):
            return empty_state

        state = self._read_json_file(checkpoint_path)
        reload_iteration = int(state.get('iteration', 0) or 0)
        normalized_records = []
        for cut_index, record in enumerate(state.get('cuts', [])):
            normalized = self._normalize_cut_record(record)
            normalized_records.append(normalized)
            constr = self.master_model.addConstr(
                self._cut_record_to_constraint(normalized),
                name=f"reloaded_{normalized['kind']}_cut_{cut_index}",
            )
            self._register_cut(constr, normalized, reload_iteration)

        state['cuts'] = normalized_records
        state['cut_count'] = len(normalized_records)
        return state

    def _save_cut_checkpoint(self, checkpoint_path, state, new_cut_records=None):
        if not checkpoint_path:
            return

        meta_path, cuts_path = self._checkpoint_paths(checkpoint_path)

        if new_cut_records:
            with gzip.open(cuts_path, 'at', encoding='utf-8') as handle:
                for record in new_cut_records:
                    handle.write(json.dumps(record, separators=(',', ':')))
                    handle.write('\n')

        metadata = {
            'iteration': int(state.get('iteration', 0) or 0),
            'lower_bound': state.get('lower_bound'),
            'upper_bound': state.get('upper_bound'),
            'core_point': state.get('core_point'),
            'cut_count': int(state.get('cut_count', 0) or 0),
        }
        tmp_path = f"{meta_path}.tmp"
        with open(tmp_path, 'w', encoding='utf-8') as handle:
            json.dump(metadata, handle, separators=(',', ':'))
        os.replace(tmp_path, meta_path)

    def _rewrite_cut_checkpoint_store(self, checkpoint_path):
        cuts_path = self._checkpoint_paths(checkpoint_path)[1]
        tmp_path = f"{cuts_path}.tmp"
        with gzip.open(tmp_path, 'wt', encoding='utf-8') as handle:
            for entry in self._cut_registry:
                handle.write(json.dumps(entry['record'], separators=(',', ':')) + '\n')
        os.replace(tmp_path, cuts_path)

    def _checkpoint_iteration(self, checkpoint_path, global_iteration, lower_bound,
                              upper_bound, core_point, cut_count, new_cut_records,
                              purge_after):
        purged_count = 0
        if self._cut_purge_enabled:
            purged_count = self._purge_inactive_cuts(global_iteration, purge_after)
            if purged_count:
                print(f"Iteration {global_iteration}, purged {purged_count} inactive cuts "
                      f"({len(self._cut_registry)} cuts remain)")

        checkpoint_state = {
            'iteration': global_iteration,
            'lower_bound': None if np.isinf(lower_bound) else float(lower_bound),
            'upper_bound': None if np.isinf(upper_bound) else float(upper_bound),
            'core_point': None if core_point is None else np.asarray(core_point, dtype=float).tolist(),
            'cut_count': cut_count + len(new_cut_records),
        }
        if purged_count:
            # Rewrite the cuts store to the surviving cuts (this iteration's
            # cuts are already in the registry) so a resume does not
            # re-inflate the master with purged cuts.
            checkpoint_state['cut_count'] = len(self._cut_registry)
            if checkpoint_path:
                self._rewrite_cut_checkpoint_store(checkpoint_path)
                self._save_cut_checkpoint(checkpoint_path, checkpoint_state, new_cut_records=None)
        else:
            self._save_cut_checkpoint(checkpoint_path, checkpoint_state, new_cut_records=new_cut_records)
        return checkpoint_state['cut_count']

    def _solve_all_subproblems(self, active_workers, env_groups, solve_one, executor, global_iteration):
        progress_kwargs = dict(
            total=len(active_workers),
            desc=f"Iter {global_iteration} subproblems",
            leave=False,
            dynamic_ncols=True,
        )
        if executor is not None:
            try:
                with tqdm(**progress_kwargs) as progress:
                    results = _dispatch_subproblem_solves(
                        env_groups, active_workers, solve_one, executor, progress=progress
                    )
                return results, executor
            except RuntimeError as exc:
                if "can't start new thread" not in str(exc):
                    raise
                tqdm.write("Falling back to sequential Benders subproblem solves: unable to start worker threads.")
                executor.shutdown(wait=True, cancel_futures=True)
                executor = None

        results = []
        with tqdm(**progress_kwargs) as progress:
            for idx, worker in enumerate(active_workers):
                start = time.time()
                results.append(solve_one(worker))
                progress.update(1)
                tqdm.write(
                    f"Iteration {global_iteration}, subproblem {idx} solved in {time.time() - start:.2f}s"
                )
        return results, executor

    def update_core_point(self, core: np.ndarray, xk: np.ndarray, k: int, alpha: float | None = None) -> np.ndarray:
        xk = np.asarray(xk, dtype=float)
        core = np.asarray(core, dtype=float)
        if alpha is None:
            alpha = 1.0 / (k + 1.0)
        return (1.0 - alpha) * core + alpha * xk

    def _get_model_memory_usage(self, model):
        return {
            'mem_used_gb': float(model.getAttr(GRB.Attr.MemUsed)),
            'max_mem_used_gb': float(model.getAttr(GRB.Attr.MaxMemUsed)),
        }

    def _report_memory_usage(self, iteration, active_workers=None):
        master_memory = self._get_model_memory_usage(self.master_model)
        print(
            f"Iteration {iteration}, master memory used: {master_memory['mem_used_gb']:.4f} GB "
            f"(peak {master_memory['max_mem_used_gb']:.4f} GB)"
        )
        if active_workers is None:
            return master_memory, None

        subproblem_memories = [self._get_model_memory_usage(worker.model) for worker in active_workers]
        total_mem_used = sum(memory['mem_used_gb'] for memory in subproblem_memories)/len(subproblem_memories)
        total_peak_mem_used = max(memory['max_mem_used_gb'] for memory in subproblem_memories)
        print(
            f"Iteration {iteration}, subproblem memory used: {total_mem_used:.4f} GB "
            f"(peak {total_peak_mem_used:.4f} GB across {len(subproblem_memories)} workers)"
        )
        return master_memory, subproblem_memories
