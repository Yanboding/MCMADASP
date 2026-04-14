import copy
import time

import numpy as np
from gurobipy import GRB

from utils import solve_and_handle_errors, get_solution_value, set_link_rhs
from concurrent.futures import ThreadPoolExecutor


def benders_callback(model, where):
    if where == GRB.Callback.MIPSOL:
        x_vars = model._action_vars
        theta_vars = model._theta_vars
        workers = model._workers
        tol = model._tol
        use_pareto_cuts = model._use_pareto
        pareto_epsilon = model._pareto_epsilon
        max_workers = model._max_workers

        # 1. Get current candidate solution (xk)
        x_vals = np.array(model.cbGetSolution(x_vars))
        theta_vals = np.array(model.cbGetSolution(theta_vars))

        # 2. Update the core point with diminishing step size: alpha = 1 / (k + 1)
        model._cb_iter += 1
        k = model._cb_iter
        if model._core_point is None:
            model._core_point = np.copy(x_vals)
        else:
            alpha = 1.0 / (k + 1.0)
            model._core_point = (1.0 - alpha) * model._core_point + alpha * x_vals

        current_core = model._core_point

        # 3. Solve subproblems (Parallel)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            if use_pareto_cuts:
                futures = [executor.submit(w.solve_pareto, x_vals, current_core, pareto_epsilon)
                           for w in workers[:len(theta_vars)]]
            else:
                futures = [executor.submit(w.solve, x_vals)
                           for w in workers[:len(theta_vars)]]
            results = [f.result() for f in futures]

        # 4. Process results and add Lazy Constraints
        for i, (is_feasible, obj_val, duals) in enumerate(results):
            # Cut expression: theta[i] >= obj_val + duals^T * (x - x_vals)
            expr = obj_val + sum(duals[j] * (x_vars[j] - x_vals[j]) for j in range(len(x_vals)))

            if not is_feasible:
                # Feasibility cut (Farkas Ray)
                model.cbLazy(expr >= 0)
            else:
                # Optimality cut
                if model.ModelSense == GRB.MINIMIZE:
                    if theta_vals[i] < (obj_val - tol):
                        model.cbLazy(theta_vars[i] >= expr)
                else:
                    if theta_vals[i] > (obj_val + tol):
                        model.cbLazy(theta_vars[i] <= expr)

class SubproblemWorker:
    """
    One worker per scenario. Owns its own gp.Env and gp.Model.
    Build once, then call solve(action) repeatedly.
    """

    def __init__(self, model, link_rows, state_linking_constraints, subproblem_id, verbose: bool = True,
                 objective_builder_fn=None, cut_gradient_fn=None):
        # Build the model and linking constraints inside THIS env.
        self.model = model
        self.link_rows = link_rows
        self.state_linking_constraints = state_linking_constraints
        self.subproblem_id = subproblem_id
        self.verbose = verbose
        # Optional hooks for generalized Benders: treat first-stage values as constants
        # in subproblem objective and return custom cut gradients.
        self.objective_builder_fn = objective_builder_fn
        self.cut_gradient_fn = cut_gradient_fn

    def _set_output_flag(self, model, verbose: bool):
        model.Params.OutputFlag = 1 if verbose else 0

    def _get_link_rows_in_derived_model(self, derived_model):
        current_indices = [c.index for c in self.link_rows]
        rows_in_model = derived_model.getConstrs()
        return [rows_in_model[i] for i in current_indices]

    def _get_feasibility_ray_for_link_rows(self, action_values, verbose: bool = False):
        """Standard Farkas Dual logic for infeasible LPs or LP-relaxations."""
        relax_model = self.model.relax() if self.model.IsMIP else self.model
        try:
            self._set_output_flag(relax_model, verbose)
            relax_model.Params.InfUnbdInfo = 1
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

    def update_core_point(self, core: np.ndarray, xk: np.ndarray, k: int, alpha: float = None) -> np.ndarray:
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
        total_mem_used = sum(memory['mem_used_gb'] for memory in subproblem_memories)
        total_peak_mem_used = sum(memory['max_mem_used_gb'] for memory in subproblem_memories)
        print(
            f"Iteration {iteration}, subproblem memory used: {total_mem_used:.4f} GB "
            f"(peak {total_peak_mem_used:.4f} GB across {len(subproblem_memories)} workers)"
        )
        return master_memory, subproblem_memories

    def solve(self, 
              init_solution=None,
              tol=1e-6,
              max_iter=150,
              use_pareto_cuts=False,
              pareto_epsilon=1e-4,
              core_alpha=None,
              verbose=False,
              parallel=True,
              max_workers=None):
        info = {}
        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        core_point = None  # For Pareto cuts; could be average of previous actions, will initialize after first master solve
        executor = ThreadPoolExecutor(max_workers=max_workers) if parallel else None
        try:
            for iteration in range(1, max_iter + 1):
                start = time.time()
                if init_solution is not None and iteration == 1:
                    # Use the provided initial solution instead of solving the master
                    action = np.asarray(init_solution, dtype=float)
                    core_point = copy.deepcopy(action)
                    print(f"Iteration {iteration}, using init_solution (skipping master solve)")
                else:
                    if not solve_and_handle_errors(self.master_model, verbose=verbose):
                        raise RuntimeError("Master model optimal solution not found")
                    print(f"Iteration {iteration}, master solved in {time.time() - start} seconds")
                    self._report_memory_usage(iteration)
                    action = get_solution_value(self.action_vars).astype(float)
                    if core_point is None:
                        core_point = copy.deepcopy(action)
                
                    if self.master_model.ModelSense == GRB.MINIMIZE:
                        lower_bound = self.master_model.ObjVal
                    else:
                        upper_bound = self.master_model.ObjVal

                # Ask all workers to solve for this action
                # futures = [ex.submit(w.solve, action_t, verbose) for w in workers]
                start_sub = time.time()
                active_workers = self.workers[:self.theta_vars.shape[0]]  # Only use as many workers as we have theta variables (scenarios)
                # Execute subproblems
                if parallel:
                    if use_pareto_cuts:
                        futures = [executor.submit(w.solve_pareto, action, core_point, pareto_epsilon, verbose)
                                   for w in active_workers]
                    else:
                        futures = [executor.submit(w.solve, action, verbose) for w in active_workers]
                    results = [f.result() for f in futures]
                else:
                    results = []
                    for w in active_workers:
                        if use_pareto_cuts:
                            results.append(w.solve_pareto(action, core_point, pareto_epsilon, verbose))
                        else:
                            results.append(w.solve(action, verbose))
                print(f"Iteration {iteration}, subproblems solved in {time.time() - start_sub:.2f}s")
                self._report_memory_usage(iteration, active_workers)
                
                feasibility_cuts = []
                optimality_cuts = []
                cost_to_go_estimation = 0.0
                all_feasible = True

                for idx, (is_feasible, v, duals) in enumerate(results):
                    scenario_id = active_workers[idx].subproblem_id

                    if not is_feasible:
                        all_feasible = False
                        cut_expr = v + np.dot(duals, self.action_vars - action)
                        feasibility_cuts.append(cut_expr >= 0)
                        # Note: We continue the loop to collect all possible feasibility cuts
                        # rather than breaking, which helps the Master converge faster.
                    else:
                        cost_to_go_estimation += v
                        cut_rhs = v + np.dot(duals, self.action_vars - action)
                        if self.master_model.ModelSense == GRB.MINIMIZE:
                            optimality_cuts.append(self.theta_vars[scenario_id] >= cut_rhs)
                        else:
                            optimality_cuts.append(self.theta_vars[scenario_id] <= cut_rhs)
                
                if not all_feasible:
                    print(f"Iteration {iteration}, adding {len(feasibility_cuts)} feasibility cuts")
                    # Some scenario infeasible: add feasibility cuts and repeat
                    self.master_model.addConstrs((feasibility_cuts[i] for i in range(len(feasibility_cuts))),
                                                name=f"feas_cut_{iteration}_")
                else:
                    print(f"Iteration {iteration}, adding {len(optimality_cuts)} optimality cuts")
                    # All scenarios feasible: add optimality cuts and continue
                    self.master_model.addConstrs((optimality_cuts[i] for i in range(len(optimality_cuts))), name=f"opt_cut_{iteration}_")
                    cost_to_go_estimation = cost_to_go_estimation / self.theta_vars.shape[0]  # Average cost-to-go across scenarios for reporting
                    first_stage_cost = self.imm_cost.getValue() if hasattr(self.imm_cost, 'getValue') else float(self.imm_cost or 0.0)
                    if self.master_model.ModelSense == GRB.MINIMIZE:
                        upper_bound = first_stage_cost + cost_to_go_estimation
                    else:
                        lower_bound = first_stage_cost + cost_to_go_estimation

                    # update core point AFTER you have a valid x_k from the master
                    core_point = self.update_core_point(core_point, action, iteration, alpha=core_alpha)

                    print(f"UB: {upper_bound}, LB: {lower_bound}, Gap: {abs(upper_bound - lower_bound)}, First-stage cost: {first_stage_cost}, Cost-to-go estimate: {cost_to_go_estimation}")
                    # Average the future cost across scenarios like in direct solution
                    if abs(upper_bound - lower_bound) < tol:
                        info = {}
                        break
                    if lower_bound > upper_bound:
                        print("Error: Lower bound exceeded upper bound (check dual rays/bounds).")
                        # save more state here for debugging
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
                            verbose=False):
        # 1. Mandatory Parameter for Lazy Constraints
        self.master_model.Params.MIPGap = 0.0
        self.master_model.Params.LazyConstraints = 1

        # 2. Pre-processing: Initialize Core Point if using Pareto
        # Often helpful to solve the LP relaxation first to get a good core point
        # self.master_model.optimize()
        # initial_action = np.array([v.X for v in self.action_vars])
        # print('Initial action for Pareto cuts:', initial_action)

        # 3. Attach variables/data to the model object for the callback to access
        # We use the underscore prefix (_) to avoid namespace collisions
        self.master_model._action_vars = self.action_vars
        self.master_model._theta_vars = self.theta_vars
        self.master_model._workers = self.workers
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
        self.master_model.optimize(benders_callback)
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