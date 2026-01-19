import copy
import time

import numpy as np
from gurobipy import GRB

from utils import solve_and_handle_errors, get_solution_value, set_link_rhs

class SubproblemWorker:
    """
    One worker per scenario. Owns its own gp.Env and gp.Model.
    Build once, then call solve(action) repeatedly.
    """

    def __init__(self, model, link_rows, state_linking_constraints, subproblem_id, verbose: bool = True):
        # Build the model and linking constraints inside THIS env.
        self.model = model
        self.link_rows = link_rows
        self.state_linking_constraints = state_linking_constraints
        self.subproblem_id = subproblem_id
        self.verbose = verbose

    def solve(self, action_values, verbose: bool = False):
        """
        Set links to the candidate master action and optimize the subproblem.
        Return (is_feasible, objective_value, dual_vector_or_ray_on_link_rows).
        """
        set_link_rhs(self.link_rows, action_values)

        if verbose:
            self.model.Params.OutputFlag = 1
        else:
            self.model.Params.OutputFlag = 0

        self.model.optimize()

        if self.model.Status == GRB.OPTIMAL:
            v = self.model.ObjVal
            duals = np.array([c.Pi for c in self.link_rows], dtype=float)
            return True, v, duals
        else:
            # Infeasible: use Farkas duals / ray
            v = sum(c.FarkasDual * c.RHS for c in self.model.getConstrs())
            ray = np.array([c.FarkasDual for c in self.link_rows], dtype=float)
            return False, v, ray

    def solve_pareto(self, action_values, core_point, epsilon=1e-4, verbose: bool = False):
        """
        Implements a simplified Magnanti-Wong/Papadakos cut.
        core_point: a point in the interior of the feasible region (e.g., average of previous actions).
        """
        set_link_rhs(self.link_rows, action_values)
        self.model.Params.OutputFlag = 1 if verbose else 0

        # --- Step 1: Solve standard subproblem ---
        self.model.optimize()

        if self.model.Status != GRB.OPTIMAL:
            # If infeasible, handle with Farkas (Standard Benders)
            v = sum(c.FarkasDual * c.RHS for c in self.model.getConstrs())
            ray = np.array([c.FarkasDual for c in self.link_rows], dtype=float)
            return False, v, ray

        # Current optimal value
        z_star = self.model.ObjVal

        # --- Step 2: Solve for Pareto-Optimal Duals ---
        # We fix the objective value to z_star and change the objective to maximize
        # the cut value at the 'core_point'.

        # 1. Add a temporary constraint to maintain optimality: dual_obj == z_star
        # Note: In the primal, this means fixing the objective.
        # In practice, it's easier to use Gurobi's 'Secondary Objective' or
        # fix the primal variables that were basic.

        # Shift the RHS toward the core point by a small epsilon
        perturbed_rhs = [(1 - epsilon) * action_values[i] + epsilon * core_point[i]
                         for i in range(len(action_values))]

        set_link_rhs(self.link_rows, perturbed_rhs)
        self.model.optimize()

        # The duals from this slightly perturbed problem are biased toward the core point
        v = self.model.ObjVal
        duals = np.array([c.Pi for c in self.link_rows], dtype=float)

        # Reset RHS for next iteration
        set_link_rhs(self.link_rows, action_values)

        return True, z_star, duals

    def dispose(self):
        self.model.dispose()


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

    def solve(self, tol=1e-6,
              max_iter=150,
              use_pareto_cuts=True,
              pareto_epsilon=1e-4,
              core_alpha=None,
              verbose=False):
        info = {}
        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        core_point = None  # For Pareto cuts; could be average of previous actions, will initialize after first master solve
        for iteration in range(1, max_iter + 1):
            start = time.time()
            if not solve_and_handle_errors(self.master_model, verbose=verbose):
                raise RuntimeError("Master model optimal solution not found")
            end = time.time()
            print(f"Iteration {iteration}, master solved in {end - start} seconds")
            action = get_solution_value(self.action_vars).astype(float)
            if core_point is None:
                core_point = copy.deepcopy(action)

            lower_bound = self.master_model.ObjVal

            # Ask all workers to solve for this action
            # futures = [ex.submit(w.solve, action_t, verbose) for w in workers]
            feasibility_cuts = []
            optimality_cuts = []
            cost_to_go_estimation = 0.0
            all_feasible = True
            start = time.time()
            for w in self.workers[:len(self.theta_vars)]:
                scenario_id = w.subproblem_id
                if use_pareto_cuts:
                    is_feasible, v, duals = w.solve_pareto(
                        action, core_point, epsilon=pareto_epsilon, verbose=verbose
                    )
                else:
                    is_feasible, v, duals = w.solve(action, verbose=verbose)
                if not is_feasible:
                    print(f"Iteration {iteration}, scenario {scenario_id} infeasible; adding feasibility cut")
                    all_feasible = False
                    # Add feasibility cut to master
                    cut_expr = v + np.dot(duals, self.action_vars - action)
                    feasibility_cuts.append(cut_expr >= 0)
                    break
                else:
                    # If feasible, generate the strengthened cut using the dynamic method
                    cost_to_go_estimation += v
                    # cut = @constraint(model, θ >= ret.obj + sum(ret.π .* (x .- x_k)))
                    cut_rhs = v + np.dot(duals, self.action_vars - action)
                    optimality_cuts.append(self.theta_vars[scenario_id] >= cut_rhs)
            end = time.time()
            print(f"Iteration {iteration}, subproblems solved in {end - start} seconds")
            if not all_feasible:
                print(f"Iteration {iteration}, adding {len(feasibility_cuts)} feasibility cuts")
                # Some scenario infeasible: add feasibility cuts and repeat
                self.master_model.addConstrs((feasibility_cuts[i] for i in range(len(feasibility_cuts))),
                                             name="feas_cut_")
            else:
                print(f"Iteration {iteration}, adding {len(optimality_cuts)} optimality cuts")
                # All scenarios feasible: add optimality cuts and continue
                self.master_model.addConstrs((optimality_cuts[i] for i in range(len(optimality_cuts))), name="opt_cut_")
                cost_to_go_estimation = cost_to_go_estimation / len(self.theta_vars)
                upper_bound = self.imm_cost.getValue() + cost_to_go_estimation

                # update core point AFTER you have a valid x_k from the master
                core_point = self.update_core_point(core_point, action, iteration, alpha=core_alpha)
                # Average the future cost across scenarios like in direct solution
                if abs(upper_bound - lower_bound) < tol:
                    info = {}
                    break
                if lower_bound > upper_bound:
                    print('Wrong upper_bound:', upper_bound)
                    print('Wrong lower_bound:', lower_bound)
                    print("Lower bound exceeded upper bound")
                    # save more state here for debugging
                    info = {'debug': 'lower_bound_exceeded_upper_bound'}
                    break
            print('upper_bound:', upper_bound)
            print('lower_bound:', lower_bound)
            print('-' * 20)
        else:
            print('Max iterations reached')
        return upper_bound, info

    def update_master_problem(self, new_scenario_number):
        # No need to update anything in the master problem
        # Add the new variable to the model and the dictionary
        # set imm_cost and a cost to go lb
        start_index = len(self.theta_vars)
        self.theta_vars += [self.master_model.addVar(vtype=GRB.CONTINUOUS, name=f"theta_{omega}") for omega in range(start_index, start_index + new_scenario_number)]
        z = self.imm_cost + sum(self.theta_vars) / len(self.theta_vars)
        self.master_model.setObjective(z, GRB.MINIMIZE)
        self.master_model.update()

    def adaptive_solve(self, batch_size=64,
                       adaptive_tol=1e-5,
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
            obj_val, info = self.solve(tol=gap_tol,
                                       max_iter=max_iter,
                                       use_pareto_cuts=use_pareto_cuts,
                                       pareto_epsilon=pareto_epsilon,
                                       core_alpha=core_alpha,
                                       verbose=verbose)
            print(f"Adaptive solve: current objective value = {obj_val}, previous = {prev}", info, abs(obj_val - prev), number_of_workers, len(self.workers))
            if 'debug' in info:
                break
            if abs(obj_val - prev) < adaptive_tol or number_of_workers >= len(self.workers):
                converge = True
            prev = obj_val
        return prev, info

    def warmup_solve(self, warmup_size=64,
                       gap_tol=1e-6,
                       max_iter=150,
                       use_pareto_cuts=True,
                       pareto_epsilon=1e-4,
                       core_alpha=None,
                       verbose=False):
        self.update_master_problem(new_scenario_number=warmup_size)
        # add theta vars for new workers
        obj_val, info = self.solve(tol=gap_tol,
                                   max_iter=max_iter,
                                   use_pareto_cuts=use_pareto_cuts,
                                   pareto_epsilon=pareto_epsilon,
                                   core_alpha=core_alpha,
                                   verbose=verbose)
        self.update_master_problem(new_scenario_number=len(self.workers)-warmup_size)
        # add theta vars for new workers
        obj_val, info = self.solve(tol=gap_tol,
                                   max_iter=max_iter,
                                   use_pareto_cuts=use_pareto_cuts,
                                   pareto_epsilon=pareto_epsilon,
                                   core_alpha=core_alpha,
                                   verbose=verbose)
        return obj_val, info


if __name__ == "__main__":
    action = (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))