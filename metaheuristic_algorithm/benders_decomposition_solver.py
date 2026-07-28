import copy
import gzip
import json
import os
import time

import numpy as np
from gurobipy import GRB
from tqdm.auto import tqdm

from utils import solve_and_handle_errors, set_link_rhs
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
    """Partition workers into groups that share a Gurobi environment.

    A Gurobi ``Env`` is not thread-safe for concurrent optimization, so two
    workers whose models live in the same env must not be solved at the same
    time. Returning one list per distinct env lets the caller solve each group
    sequentially (on a single thread) while running different groups in
    parallel. Workers without a recorded env (``grb_env is None``) are treated
    as having their own private env, preserving the original one-thread-per-
    worker parallelism.
    """
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
    """Solve every worker via ``solve_one(worker)``, respecting env grouping.

    Workers in the same env group are solved sequentially on a single thread,
    while different groups run concurrently on ``executor`` (a Gurobi ``Env`` is
    not thread-safe for concurrent optimization). When ``executor`` is ``None``
    the solves run sequentially. Results are returned in ``active_workers``
    order. If ``progress`` is provided, ``progress.update(1)`` is called after
    each worker's solve completes (tqdm is thread-safe). Any ``RuntimeError``
    raised while starting worker threads propagates to the caller so it can
    fall back to a sequential strategy.
    """
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
        # The worker set, their env grouping and the thread pool are fixed for
        # the whole solve, so they were built ONCE in solve_with_callback and are
        # reused here instead of being rebuilt at every incumbent.
        env_groups = model._env_groups
        executor = model._executor

        # 1. Get current candidate solution (xk). ``x_vars``/``theta_vars`` are
        # MVars (see flatten/hindsight_master_builder_fn), so cbGetSolution
        # already returns ndarrays -- no extra conversion needed.
        x_vals = model.cbGetSolution(x_vars)
        theta_vals = model.cbGetSolution(theta_vars)

        # 2. Update the core point with diminishing step size: alpha = 1 / (k + 1)
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

        # 4. Process results and add Lazy Constraints
        feasibility_cuts_added = 0
        optimality_cuts_added = 0
        cost_to_go = 0.0
        # Shared affine term (x - x_k) as a single MLinExpr; each cut is then one
        # vectorized dot product instead of a per-coefficient Python loop.
        action_delta = x_vars - x_vals
        for i, (is_feasible, obj_val, duals) in enumerate(results):
            # Cut expression: theta[i] >= obj_val + duals^T * (x - x_vals)
            expr = obj_val + duals @ action_delta

            if not is_feasible:
                # Feasibility cut (Farkas Ray)
                model.cbLazy(expr >= 0)
                feasibility_cuts_added += 1
            else:
                cost_to_go += obj_val
                # Optimality cut
                if model.ModelSense == GRB.MINIMIZE:
                    if theta_vals[i] < (obj_val - tol):
                        model.cbLazy(theta_vars[i] >= expr)
                        optimality_cuts_added += 1
                else:
                    if theta_vals[i] > (obj_val + tol):
                        model.cbLazy(theta_vars[i] <= expr)
                        optimality_cuts_added += 1

        # 5. Intermediate progress output (mirrors solve()'s per-iteration log).
        # Each MIPSOL is one incumbent, so report the incumbent objective against
        # Gurobi's best bound (their difference is the live MIP gap), the cuts
        # injected for this incumbent, the evaluated objective at this action
        # (first-stage cost + averaged subproblem cost-to-go -- a valid bound just
        # like solve() reports), and the time spent solving subproblems.
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
    """
    One worker per scenario. Owns its own gp.Env and gp.Model.
    Build once, then call solve(action) repeatedly.
    """

    def __init__(self, model, link_rows, state_linking_constraints, subproblem_id, verbose: bool = True,
                 objective_builder_fn=None, cut_gradient_fn=None, grb_env=None, initial_cut=None):
        # Build the model and linking constraints inside THIS env.
        self.model = model
        # The Gurobi environment this worker's model lives in. Workers that
        # share an env must never be optimized concurrently (a Gurobi Env is
        # not thread-safe), so the solver groups workers by this env and solves
        # each group sequentially while running different envs in parallel.
        self.grb_env = grb_env
        # Normalize link_rows to a flat list of scalar Constr objects so that
        # per-element attributes like .index/.Pi/.FarkasDual/.RHS work uniformly.
        # The input may be an MConstr, a plain list of Constr, or a mixed list
        # containing MConstr elements (e.g. from addConstr on MVar scalars).
        def _flatten_constrs(obj):
            if obj is None:
                return []
            if hasattr(obj, "tolist"):  # MConstr or numpy array
                return _flatten_constrs(obj.tolist())
            if isinstance(obj, (list, tuple)):
                out = []
                for x in obj:
                    out.extend(_flatten_constrs(x))
                return out
            return [obj]  # scalar Constr
        self.link_rows = _flatten_constrs(link_rows)
        self._link_indices = [c.index for c in self.link_rows]
        self.state_linking_constraints = state_linking_constraints
        self.subproblem_id = subproblem_id
        self.verbose = verbose
        # Optional hooks for generalized Benders: treat first-stage values as constants
        # in subproblem objective and return custom cut gradients.
        self.objective_builder_fn = objective_builder_fn
        self.cut_gradient_fn = cut_gradient_fn
        # Build-time Benders cut (v0, g0) obtained by solving this subproblem at
        # action a = 0 during construction (v0 = Q_s(0), g0 = subgradient at 0).
        # The solver uses it to seed the master with one valid optimality cut per
        # scenario before the first master solve. ``None`` for classical workers.
        self.initial_cut = initial_cut

    def _set_output_flag(self, model, verbose: bool):
        model.Params.OutputFlag = 1 if verbose else 0

    def _get_link_rows_in_derived_model(self, derived_model):
        rows_in_model = derived_model.getConstrs()
        return [rows_in_model[i] for i in self._link_indices]

    def _get_feasibility_ray_for_link_rows(self, action_values, verbose: bool = False):
        """Standard Farkas Dual logic for infeasible LPs or LP-relaxations."""
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
                # For standard solve, we extract duals from the fixed MILP or the LP
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
        """
        Fix-then-Perturb strategy: Solve MILP once, fix integers, 
        then solve perturbed LP for Pareto-optimal duals.
        """
        if self.link_rows is None or len(self.link_rows) == 0:
            # Fallback for generalized workers that provide custom subgradients
            # and do not expose classical linking-constraint duals.
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

        # For MILP, we fix the optimal integer solution to find the strongest duals
        # for that specific realization of the first-stage variables.
        if self.model.IsMIP:
            lp_model = self.model.fixed()
        else:
            lp_model = self.model.copy() # For pure LP, just copy to avoid modifying original

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


class BendersDecompositionSolver:
    """
    Bender's decomposition solver for two-stage stochastic programs.
    The master problem is built by master_builder_fn, and the subproblems
    are built by subproblem_builder_fn.

    Args:
        master_builder_fn: function() -> (gp.Model, list of gp.Var)
            Builds the master problem and returns it along with the action variables.
        subproblem_builder_fn: function() -> (gp.Model, list of gp.Constr)
            Builds a subproblem and returns it along with the linking constraints.
        subproblem_builder_args: dict
            Arguments to pass to subproblem_builder_fn.
        num_subproblems: int
            Number of subproblems (scenarios).
        sense: GRB.MINIMIZE or GRB.MAXIMIZE
        tol: float
            Tolerance for convergence.
        max_iter: int
            Maximum number of Bender iterations.
        verbose: bool
            Whether to print detailed logs.
    """

    def __init__(self, master_model, workers, imm_cost, theta_vars, action_vars):
        self.master_model = master_model
        self.workers = workers
        self.imm_cost = imm_cost
        self.theta_vars = theta_vars
        self.action_vars = action_vars
        # Lazily-created |action| epigraph variables for the min-norm
        # optimal-face re-solve (see _min_norm_master_action).
        self._action_abs_vars = None

    def _min_norm_master_action(self, fallback_action, verbose=False):
        """Return the minimum-L1-norm action on the master's optimal face.

        The master objective only involves the theta epigraph variables, so
        with few cuts its optimal face is typically unbounded in the action
        space and simplex returns a degenerate vertex with astronomically
        large action values (which numerically break the subproblems). This
        re-solve keeps the PRIMARY objective pinned at its optimal value and,
        among all optimal solutions, picks the one minimizing ``sum |a_j|``.
        The action variables themselves stay UNBOUNDED and the master optimum
        is unchanged -- this is a tie-break on the optimal face, not a bound.

        Any Benders optimality cut is valid at any action point, so cutting at
        the min-norm optimum preserves correctness. On any failure the
        first-phase ``fallback_action`` is returned unchanged.
        """
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
                return np.array(self.action_vars.X, dtype=float)
            print(f"Min-norm re-solve not optimal (Status {model.Status}); "
                  f"keeping first-phase master action")
            return fallback_action
        finally:
            model.remove(guard)
            model.setObjective(primary_objective, original_sense)
            model.update()

    @staticmethod
    def _checkpoint_paths(checkpoint_path):
        return f"{checkpoint_path}.meta.json", f"{checkpoint_path}.cuts.jsonl.gz"

    @staticmethod
    def _encode_vector(values, zero_tol=1e-12, sparse_density=0.35):
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

    @staticmethod
    def _decode_vector(payload):
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

    @staticmethod
    def _read_json_file(path):
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as handle:
                return json.load(handle)
        except (OSError, gzip.BadGzipFile):
            with open(path, 'r', encoding='utf-8') as handle:
                return json.load(handle)

    @staticmethod
    def _iter_cut_records(path):
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
            cut_records = []
            for cut_index, record in enumerate(self._iter_cut_records(cuts_path)):
                normalized = self._normalize_cut_record(record)
                cut_records.append(normalized)
                self.master_model.addConstr(
                    self._cut_record_to_constraint(normalized),
                    name=f"reloaded_{normalized['kind']}_cut_{cut_index}",
                )
            state['cuts'] = cut_records
            state['cut_count'] = len(cut_records)
            return state

        if not os.path.exists(checkpoint_path):
            return empty_state

        state = self._read_json_file(checkpoint_path)
        normalized_records = []
        for cut_index, record in enumerate(state.get('cuts', [])):
            normalized = self._normalize_cut_record(record)
            normalized_records.append(normalized)
            self.master_model.addConstr(
                self._cut_record_to_constraint(normalized),
                name=f"reloaded_{normalized['kind']}_cut_{cut_index}",
            )

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

    def update_core_point(self, core: np.ndarray, xk: np.ndarray, k: int, alpha: float | None = None) -> np.ndarray:
        xk = np.asarray(xk, dtype=float)
        core = np.asarray(core, dtype=float)
        if alpha is None:
            alpha = 1.0 / (k + 1.0)  # diminishing step
        return (1.0 - alpha) * core + alpha * xk

    @staticmethod
    def _get_model_memory_usage(model):
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

    def _solve_all_subproblems(self, active_workers, env_groups, solve_one, executor, global_iteration):
        """Solve every subproblem for the current master action.

        Uses the parallel env-group dispatch when ``executor`` is available and
        transparently falls back to a sequential solve (with per-subproblem
        timing) if worker threads cannot be started. Returns the results aligned
        to ``active_workers`` together with the (possibly disabled) executor.
        A tqdm progress bar tracks per-worker completion (with ETA) regardless
        of which path is taken.
        """
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

    def _build_cuts(self, results, active_workers, action):
        """Convert subproblem results into Benders cuts and checkpoint records.

        Returns ``(all_feasible, feasibility_cuts, optimality_cuts, cut_records,
        cost_to_go)`` where ``cost_to_go`` is the sum of feasible subproblem
        objectives (the caller averages it across scenarios).
        """
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
                cost_to_go += v
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

    def _seed_initial_cuts(self, active_workers):
        """Seed the master with each worker's build-time (a=0) optimality cut.

        Every subproblem was solved once at coefficients ``a = 0`` while it was
        built, producing ``Q_s(0)`` and the subgradient ``g_s = phi_s(x*(0))``.
        Since ``Q_s`` is concave in ``a``, ``theta_s <= Q_s(0) + g_s . a`` is a
        globally valid optimality cut (a tangent that upper-bounds the concave
        ``Q_s`` for the maximization master; the inequality flips for a
        minimization master). Injecting all available cuts BEFORE the first
        master solve means the solver starts from the approximation it would
        otherwise spend a full iteration (re-)deriving at ``a = 0`` -- so it
        converges in fewer iterations and the first master action is
        gradient-informed rather than an arbitrary extreme point.

        The value at ``a = 0``, ``(1/N) sum_s Q_s(0)``, is also a valid bound on
        the optimum (the optimum is no worse than all-zero coefficients); it is
        returned so the caller can initialize the incumbent bound. Returns
        ``(bound_at_zero, cut_records)`` -- ``(None, [])`` when no worker carries
        a build-time cut, and ``bound_at_zero is None`` when only some workers do
        (a partial average is not a valid bound on the full-scenario objective).
        """
        seeded_workers, seeded_results = [], []
        for worker in active_workers:
            initial_cut = getattr(worker, "initial_cut", None)
            if initial_cut is None:
                continue
            v0, g0 = initial_cut
            seeded_workers.append(worker)
            seeded_results.append((True, float(v0), np.asarray(g0, dtype=float)))
        if not seeded_workers:
            return None, []

        zero_action = np.zeros(self.action_vars.shape[0], dtype=float)
        _, _, optimality_cuts, cut_records, cost_to_go = self._build_cuts(
            seeded_results, seeded_workers, zero_action)
        self.master_model.addConstrs(
            (optimality_cuts[i] for i in range(len(optimality_cuts))),
            name="benders_seed_cut_",
        )
        self.master_model.update()

        all_seeded = len(seeded_workers) == len(active_workers)
        bound_at_zero = (cost_to_go / len(active_workers)) if all_seeded else None
        print(f"Seeded master with {len(optimality_cuts)} build-time (a=0) cuts"
              + (f"; bound at zero coefficients = {bound_at_zero}" if bound_at_zero is not None else ""))
        return bound_at_zero, cut_records

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
              min_norm_action=False):
        info = {}
        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        core_point = None  # For Pareto cuts; initialized after the first master solve.
        is_min = self.master_model.ModelSense == GRB.MINIMIZE
        scenario_count = self.theta_vars.shape[0]

        # The worker set and their env grouping are fixed for the whole solve,
        # so compute them once. Workers sharing an env must be solved
        # sequentially (envs are not thread-safe for concurrent optimize), which
        # makes the env group, not the individual worker, the unit of parallelism.
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
            bound_at_zero, seed_records = self._seed_initial_cuts(active_workers)
            if bound_at_zero is not None:
                # The optimum is at least as good as all-zero coefficients, so
                # the a=0 value is a valid incumbent bound to start the gap from.
                if is_min:
                    upper_bound = min(upper_bound, bound_at_zero)
                else:
                    lower_bound = max(lower_bound, bound_at_zero)
            if seed_records and checkpoint_path:
                cut_count += len(seed_records)
                self._save_cut_checkpoint(
                    checkpoint_path,
                    {
                        'iteration': iteration_offset,
                        'lower_bound': None if np.isinf(lower_bound) else float(lower_bound),
                        'upper_bound': None if np.isinf(upper_bound) else float(upper_bound),
                        'core_point': None if core_point is None else np.asarray(core_point, dtype=float).tolist(),
                        'cut_count': cut_count,
                    },
                    new_cut_records=seed_records,
                )
        try:
            for iteration in range(1, max_iter + 1):
                global_iteration = iteration_offset + iteration
                start = time.time()
                if init_solution is not None and iteration == 1:
                    # Use the provided initial solution instead of solving the master.
                    action = np.asarray(init_solution, dtype=float)
                    core_point = copy.deepcopy(action)
                    print(f"Iteration {global_iteration}, using init_solution (skipping master solve)")
                else:
                    if not solve_and_handle_errors(self.master_model, verbose=verbose):
                        raise RuntimeError("Master model optimal solution not found")
                    print(f"Iteration {global_iteration}, master solved in {time.time() - start} seconds")
                    self._report_memory_usage(global_iteration)
                    action = self.action_vars.X
                    if is_min:
                        lower_bound = self.master_model.ObjVal
                    else:
                        upper_bound = self.master_model.ObjVal
                    if min_norm_action and np.max(np.abs(action)) > 1e6:
                        # Degenerate vertex on an under-constrained optimal
                        # face: re-solve for the minimum-norm optimal action
                        # (bounds above use the primary ObjVal). Sane actions
                        # skip the re-solve so the endgame is untouched.
                        action = self._min_norm_master_action(action, verbose=verbose)
                    if core_point is None:
                        core_point = copy.deepcopy(action)

                print(f"Iteration {global_iteration}, action from master: {action.tolist()}")

                # Solve every subproblem for this candidate action.
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
                    # Some scenario infeasible: add feasibility cuts and repeat.
                    print(f"Iteration {global_iteration}, adding {len(feasibility_cuts)} feasibility cuts")
                    self.master_model.addConstrs(
                        (feasibility_cuts[i] for i in range(len(feasibility_cuts))),
                        name=f"feas_cut_{global_iteration}_",
                    )
                    print('-' * 20)
                    continue

                # All scenarios feasible: add optimality cuts and refresh bounds.
                print(f"Iteration {global_iteration}, adding {len(optimality_cuts)} optimality cuts")
                self.master_model.addConstrs(
                    (optimality_cuts[i] for i in range(len(optimality_cuts))),
                    name=f"opt_cut_{global_iteration}_",
                )
                cost_to_go_estimation /= scenario_count  # Average cost-to-go across scenarios for reporting.
                first_stage_cost = self.imm_cost.getValue() if hasattr(self.imm_cost, 'getValue') else float(self.imm_cost or 0.0)
                # Keep the incumbent (subproblem-evaluated) bound monotone so the
                # build-time a=0 seed bound is never lost and the gap shrinks
                # monotonically: the evaluated value at any action is a valid
                # lower bound (maximization) / upper bound (minimization).
                if is_min:
                    upper_bound = min(upper_bound, first_stage_cost + cost_to_go_estimation)
                else:
                    lower_bound = max(lower_bound, first_stage_cost + cost_to_go_estimation)

                # If an init_solution was supplied, its evaluated cost is a valid
                # bound on the master's optimum. Add it once as a hard constraint
                # to prune the master's search space.
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

                print(f"UB: {upper_bound}, LB: {lower_bound}, Gap: {abs(upper_bound - lower_bound)}, "
                      f"First-stage cost: {first_stage_cost}, Cost-to-go estimate: {cost_to_go_estimation}")
                checkpoint_state = {
                    'iteration': global_iteration,
                    'lower_bound': None if np.isinf(lower_bound) else float(lower_bound),
                    'upper_bound': None if np.isinf(upper_bound) else float(upper_bound),
                    'core_point': None if core_point is None else np.asarray(core_point, dtype=float).tolist(),
                    'cut_count': cut_count + len(new_cut_records),
                }
                self._save_cut_checkpoint(checkpoint_path, checkpoint_state, new_cut_records=new_cut_records)
                cut_count += len(new_cut_records)

                if abs(upper_bound - lower_bound) < tol:
                    info = {}
                    break
                if lower_bound > upper_bound:
                    print("Error: Lower bound exceeded upper bound (check dual rays/bounds).")
                    info = {'debug': 'lower_bound_exceeded_upper_bound'}
                    break
                print('-' * 20)
            else:
                print('Max iterations reached')
        finally:
            if executor is not None:
                executor.shutdown(wait=False)
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

        # 2. Pre-processing: Initialize Core Point if using Pareto
        # Often helpful to solve the LP relaxation first to get a good core point
        # self.master_model.optimize()
        # initial_action = np.array([v.X for v in self.action_vars])
        # print('Initial action for Pareto cuts:', initial_action)

        # 2b. Subproblem parallelism, set up ONCE for the whole solve. Env groups
        # are the unit of parallelism -- workers sharing a Gurobi env must be
        # solved sequentially (envs are not thread-safe for concurrent optimize)
        # while different env groups run concurrently. A single persistent
        # ThreadPoolExecutor is reused by every callback so incumbents don't each
        # pay thread-pool build/teardown; it is sized to the number of env groups
        # (capped by max_workers / SLURM CPUs / cpu_count).
        active_workers = self.workers[:self.theta_vars.shape[0]]
        env_groups = _group_workers_by_env(active_workers)
        resolved_max_workers = _resolve_parallel_workers(max_workers, len(env_groups))
        executor = (
            ThreadPoolExecutor(max_workers=resolved_max_workers)
            if parallel and resolved_max_workers > 1
            else None
        )

        # 3. Attach variables/data to the model object for the callback to access
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

        # 4. Start the single optimization call
        print("Starting Benders with Lazy Constraint Callback...")
        try:
            self.master_model.optimize(benders_callback)
        finally:
            if executor is not None:
                executor.shutdown(wait=False)
        # if not solve_and_handle_errors(self.master_model, verbose=verbose):
        #     raise RuntimeError("Master model optimal solution not found")
        # 5. Extract results
        info = {}
        if self.master_model.Status == GRB.OPTIMAL:
            return self.master_model.ObjVal, info
        return None, info

    def update_master_problem(self, new_scenario_number):
        # No need to update anything in the master problem
        # Add the new variable to the model and the dictionary
        # set imm_cost and a cost to go lb
        start_index = len(self.theta_vars)
        self.theta_vars += [self.master_model.addVar(vtype=GRB.CONTINUOUS, name=f"theta_{omega}") for omega in range(start_index, start_index + new_scenario_number)]
        z = self.imm_cost + sum(self.theta_vars) / len(self.theta_vars)
        current_sense = self.master_model.ModelSense
        self.master_model.setObjective(z, current_sense)
        self.master_model.update()

    def adaptive_solve(self, batch_size=64,
                       adaptive_tol=0.05,
                       gap_tol=1e-6,
                       max_iter=150,
                       use_pareto_cuts=True,
                       pareto_epsilon=1e-4,
                       core_alpha=None,
                       verbose=False):
        converge = False
        prev = -float('inf')
        info = {}
        number_of_workers = 0
        while not converge:
            # Include more workers if needed
            # just activate next batch of workers
            number_of_workers += batch_size
            self.update_master_problem(new_scenario_number=batch_size)
            # add theta vars for new workers
            obj_val, info = self.solve_with_callback(tol=gap_tol,
                                       max_iter=max_iter,
                                       use_pareto_cuts=use_pareto_cuts,
                                       pareto_epsilon=pareto_epsilon,
                                       verbose=verbose)
            if obj_val is None:
                info['debug'] = 'no_objective_value'
                break
            obj_val = float(obj_val)
            print(f"Adaptive solve: current objective value = {obj_val}, previous = {prev}", info, abs(obj_val - prev), number_of_workers, len(self.workers))
            if 'debug' in info:
                break
            ptc_subgap = abs(obj_val - prev)/obj_val if obj_val > 0 else float('inf')
            if ptc_subgap < adaptive_tol or number_of_workers >= len(self.workers):
                info['number_of_workers'] = number_of_workers
                converge = True
            prev = obj_val
        return prev, info


if __name__ == "__main__":
    action = (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))