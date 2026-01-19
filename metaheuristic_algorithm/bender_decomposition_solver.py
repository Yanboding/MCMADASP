import copy
import time

import numpy as np
from gurobipy import GRB

from utils import solve_and_handle_errors


def flatten(vars):
    list = []
    for item in vars:
        list.extend(item.reshape(-1))
    return np.array(list)

def set_link_rhs(linking_constraints, rhs_values):
    for i, constr in enumerate(linking_constraints):
        constr.setAttr("RHS", float(rhs_values[i]))

class SubproblemWorker:
    """
    One worker per scenario. Owns its own gp.Env and gp.Model.
    Build once, then call solve(action) repeatedly.
    """
    def __init__(self, builder_fn, builder_args, subproblem_id:int, verbose:bool=True):
        self.verbose = verbose
        # Build the model and linking constraints inside THIS env.
        builder_args['scenario_id'] = subproblem_id
        start = time.time()
        self.model, self.link_rows, self.state_linking_constraints = builder_fn(**builder_args)
        end = time.time()
        print(f"Finished building subproblem for scenario {subproblem_id} in {end - start} seconds")
        self.subproblem_id = subproblem_id

    def set_link_rhs(self, action_values):
        """
        Update RHS of linking constraints so they enforce: (action vars) == (action values).
        Assumes link rows were built as equality rows var == 0 initially.
        """
        for i, constr in enumerate(self.link_rows):
            constr.setAttr("RHS", float(action_values[i]))

    def solve(self, action_values, verbose:bool=False):
        """
        Set links to the candidate master action and optimize the subproblem.
        Return (is_feasible, objective_value, dual_vector_or_ray_on_link_rows).
        """
        self.set_link_rhs(action_values)

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
        self.set_link_rhs(action_values)
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

        self.set_link_rhs(perturbed_rhs)
        self.model.optimize()

        # The duals from this slightly perturbed problem are biased toward the core point
        v = self.model.ObjVal
        duals = np.array([c.Pi for c in self.link_rows], dtype=float)

        # Reset RHS for next iteration
        self.set_link_rhs(action_values)

        return True, z_star, duals

    def dispose(self):
        self.model.dispose()

class BenderDecompositionSolver:
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
    def __init__(self, master_builder_fn, master_builder_args, subproblem_builder_fn, subproblem_builder_args, get_solution, flatten_fn, num_subproblems):
        """
        Initialize the Bender's decomposition solver.
        Args:
            master_builder_fn: function() -> (gp.Model, gp.Var, list of gp.Var, gp.Var)
                Builds the master problem and returns it along with the immediate cost variable,
                the theta variables for each scenario, and the action variable.
            subproblem_builder_fn: function() -> (gp.Model, list of gp.Constr)
                Builds a subproblem and returns it along with the linking constraints.
            get_solution: function(gp.Var, is_final=False) -> np.array
                Extracts the solution from the action variable.
            flatten_fn: function(action) -> np.array
                Flattens the action into a 1D numpy array for linking constraints.
            num_subproblems: int
                Number of subproblems (scenarios).
        """
        self.master_builder_fn = master_builder_fn
        self.subproblem_builder_fn = subproblem_builder_fn
        self.get_solution = get_solution
        self.flatten_fn = flatten_fn
        if flatten_fn is None:
            self.flatten_fn = flatten
        self.num_subproblems = num_subproblems
        self.master_builder_args= master_builder_args
        self.subproblem_builder_args = subproblem_builder_args
        #self.master_model, self.imm_cost, self.theta_vars, self.action_t_var, self.state_linking_constraints = self.master_builder_fn(**master_builder_args)
        # Build one worker per scenario once, then reuse

        self.workers = [
            SubproblemWorker(self.subproblem_builder_fn, self.subproblem_builder_args, sid)
            for sid in range(self.num_subproblems)
        ]

    def update_core_point(self, core: np.ndarray, xk: np.ndarray, k: int, alpha: float = None) -> np.ndarray:
        xk = np.asarray(xk, dtype=float)
        core = np.asarray(core, dtype=float)
        if alpha is None:
            alpha = 1.0 / (k + 1.0)  # diminishing step
        return (1.0 - alpha) * core + alpha * xk

    def solve(self, state, action=None, tol=1e-6, max_iter=150, verbose=False,
              use_pareto_cuts=True, pareto_epsilon=1e-4, core_alpha=None):
        self.master_model, self.imm_cost, self.theta_vars, self.action_t_var, self.state_linking_constraints = self.master_builder_fn(
            **self.master_builder_args)
        flatten_state = flatten(state)
        set_link_rhs(self.state_linking_constraints, flatten_state)
        flat_action_t_var = flatten(self.action_t_var)
        for worker in self.workers:
            set_link_rhs(worker.state_linking_constraints, flatten_state)
            worker.model.reset()

        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        core_point = None  # For Pareto cuts; could be average of previous actions, will initialize after first master solve
        for iteration in range(1, max_iter + 1):
            start = time.time()
            if not solve_and_handle_errors(self.master_model, verbose=verbose):
                raise RuntimeError("Master model optimal solution not found")
            end = time.time()
            print(f"Iteration {iteration}, master solved in {end - start} seconds")
            action_t = self.get_solution(self.action_t_var)
            flat_action_t = self.flatten_fn(action_t)

            if core_point is None:
                core_point = copy.deepcopy(flat_action_t)
                print('core_point:', core_point)

            lower_bound = self.master_model.ObjVal

            # Ask all workers to solve for this action
            # futures = [ex.submit(w.solve, action_t, verbose) for w in workers]
            feasibility_cuts = []
            optimality_cuts = []
            cost_to_go_estimation = 0.0
            all_feasible = True
            start = time.time()
            for w in self.workers:
                scenario_id = w.subproblem_id
                if use_pareto_cuts:
                    is_feasible, v, duals = w.solve_pareto(
                        flat_action_t, core_point, epsilon=pareto_epsilon, verbose=verbose
                    )
                else:
                    is_feasible, v, duals = w.solve(flat_action_t, verbose=verbose)
                if not is_feasible:
                    print(f"Iteration {iteration}, scenario {scenario_id} infeasible; adding feasibility cut")
                    all_feasible = False
                    # Add feasibility cut to master
                    cut_expr = v + np.dot(duals, flat_action_t_var - flat_action_t)
                    feasibility_cuts.append(cut_expr >= 0)
                    break
                else:
                    # If feasible, generate the strengthened cut using the dynamic method
                    cost_to_go_estimation += v
                    # cut = @constraint(model, θ >= ret.obj + sum(ret.π .* (x .- x_k)))
                    cut_rhs = v + np.dot(duals, flat_action_t_var - flat_action_t)
                    optimality_cuts.append(self.theta_vars[scenario_id] >= cut_rhs)
            end = time.time()
            print(f"Iteration {iteration}, subproblems solved in {end - start} seconds")
            if not all_feasible:
                print(f"Iteration {iteration}, adding {len(feasibility_cuts)} feasibility cuts")
                # Some scenario infeasible: add feasibility cuts and repeat
                self.master_model.addConstrs((feasibility_cuts[i] for i in range(len(feasibility_cuts))), name="feas_cut_")
            else:
                print(f"Iteration {iteration}, adding {len(optimality_cuts)} optimality cuts")
                # All scenarios feasible: add optimality cuts and continue
                self.master_model.addConstrs((optimality_cuts[i] for i in range(len(optimality_cuts))), name="opt_cut_")
                cost_to_go_estimation = cost_to_go_estimation / self.num_subproblems
                upper_bound = self.imm_cost.getValue() + cost_to_go_estimation

                # update core point AFTER you have a valid x_k from the master
                core_point = self.update_core_point(core_point, flat_action_t, iteration, alpha=core_alpha)
                # Average the future cost across scenarios like in direct solution
                if abs(upper_bound - lower_bound) < tol:
                    action_t = self.get_solution(self.action_t_var, is_final=True)
                    return action_t, upper_bound, {}
                if lower_bound > upper_bound:
                    print('Wrong upper_bound:', upper_bound)
                    print('Wrong lower_bound:', lower_bound)
                    print("Lower bound exceeded upper bound")
                    # save more state here for debugging
                    action_t = self.get_solution(self.action_t_var, is_final=True)
                    return action_t, upper_bound, {'debug':'lower_bound_exceeded_upper_bound'}
            print('upper_bound:', upper_bound)
            print('lower_bound:', lower_bound)
            print('-' * 20)
        print('Max iterations reached')
        action_t = self.get_solution(self.action_t_var, is_final=True)
        return action_t, upper_bound, {}
    
    def _solve(self, master_model, imm_cost, theta_vars, action_t_var, tol=1e-6, max_iter=150, use_pareto_cuts=True, pareto_epsilon=1e-4, core_alpha=None, verbose=False):
        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        flat_action_t_var = flatten(action_t_var)
        info = {}
        core_point = None
        workers = self.workers[:len(theta_vars)]
        for iteration in range(1, max_iter + 1):
            start = time.time()
            if not solve_and_handle_errors(master_model, verbose=verbose):
                raise RuntimeError("Master model optimal solution not found")
            end = time.time()
            print(f"Iteration {iteration}, master solved in {end - start} seconds")
            action_t = self.get_solution(action_t_var)
            flat_action_t = self.flatten_fn(action_t)

            lower_bound = master_model.ObjVal

            # Ask all workers to solve for this action
            # futures = [ex.submit(w.solve, action_t, verbose) for w in workers]
            feasibility_cuts = []
            optimality_cuts = []
            cost_to_go_estimation = 0.0
            all_feasible = True
            start = time.time()
            print('subproblem workers:', len(self.workers), 'theta vars:', len(theta_vars))
            for w in workers:
                scenario_id = w.subproblem_id
                is_feasible, v, duals = w.solve(flat_action_t, verbose=verbose)
                if not is_feasible:
                    print(f"Iteration {iteration}, scenario {scenario_id} infeasible; adding feasibility cut")
                    all_feasible = False
                    # Add feasibility cut to master
                    cut_expr = v + np.dot(duals, flat_action_t_var - flat_action_t)
                    feasibility_cuts.append(cut_expr >= 0)
                    break
                else:
                    # If feasible, generate the strengthened cut using the dynamic method
                    cost_to_go_estimation += v
                    # cut = @constraint(model, θ >= ret.obj + sum(ret.π .* (x .- x_k)))
                    cut_rhs = v + np.dot(duals, flat_action_t_var - flat_action_t)
                    optimality_cuts.append(theta_vars[scenario_id] >= cut_rhs)
            end = time.time()
            print(f"Iteration {iteration}, subproblems solved in {end - start} seconds")
            if not all_feasible:
                print(f"Iteration {iteration}, adding {len(feasibility_cuts)} feasibility cuts")
                # Some scenario infeasible: add feasibility cuts and repeat
                master_model.addConstrs((feasibility_cuts[i] for i in range(len(feasibility_cuts))), name="feas_cut_")
            else:
                print(f"Iteration {iteration}, adding {len(optimality_cuts)} optimality cuts")
                # All scenarios feasible: add optimality cuts and continue
                master_model.addConstrs((optimality_cuts[i] for i in range(len(optimality_cuts))), name="opt_cut_")
                cost_to_go_estimation = cost_to_go_estimation / self.num_subproblems
                upper_bound = imm_cost.getValue() + cost_to_go_estimation
                # Average the future cost across scenarios like in direct solution
                if abs(upper_bound - lower_bound) < tol:
                    info = {}
                    break
                if lower_bound > upper_bound:
                    print('Wrong upper_bound:', upper_bound)
                    print('Wrong lower_bound:', lower_bound)
                    print("Lower bound exceeded upper bound")
                    info = {'debug':'lower_bound_exceeded_upper_bound'}
                    break
            print('upper_bound:', upper_bound)
            print('lower_bound:', lower_bound)
            print('-' * 20)
        else:
            print('Max iterations reached')
        action_t = self.get_solution(action_t_var, is_final=True)
        return action_t, upper_bound, info

    def update_master_problem(self, model, imm_cost, theta_vars, new_scenario_number):
        # No need to update anything in the master problem
        # Add the new variable to the model and the dictionary
        # set imm_cost and a cost to go lb
        start_index = len(theta_vars)
        print('start_index:', start_index)
        theta_vars += [model.addVar(vtype=GRB.CONTINUOUS, name=f"theta_{omega}") for omega in range(start_index, start_index + new_scenario_number)]
        z = imm_cost + sum(theta_vars) / len(theta_vars)
        model.setObjective(z, GRB.MINIMIZE)
        model.update()
    
    def adaptive_solve(self, state, action=None, batch_size=64, adaptive_tol=1e-5, gap_tol=1e-6, max_iter=150,
                       use_pareto_cuts=True, pareto_epsilon=1e-4, core_alpha=None, verbose=False):
        master_model, imm_cost, theta_vars, action_t_var, state_linking_constraints = self.master_builder_fn(**self.master_builder_args)
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        converge = False
        prev = -float('inf')
        info = {}
        number_of_workers = 0
        while not converge:
            # incluede more workers if needed
            # just activate next batch of workers
            for worker in self.workers[number_of_workers: number_of_workers + batch_size]:
                set_link_rhs(worker.state_linking_constraints, flatten_state)
                worker.model.reset()
            number_of_workers += batch_size
            print('theta_vars,', theta_vars)
            self.update_master_problem(master_model, imm_cost, theta_vars, new_scenario_number=batch_size)
            # add theta vars for new workers
            action_t, obj_val, info = self._solve(master_model, imm_cost, theta_vars, action_t_var, tol=gap_tol, max_iter=max_iter, verbose=verbose)
            if 'debug' in info:
                return action_t, obj_val, info
            if abs(obj_val - prev) < adaptive_tol or number_of_workers >= len(self.workers):
                converge = True
            prev = obj_val
        return action_t, obj_val, info

if __name__ == "__main__":
    action = (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
    print(flatten(action))