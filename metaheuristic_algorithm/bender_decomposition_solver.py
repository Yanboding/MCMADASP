import numpy as np
from gurobipy import GRB

from utils import solve_and_handle_errors


def flatten(action):
    list = []
    for item in action:
        list.extend(item.reshape(-1))
    return np.array(list)

class SubproblemWorker:
    """
    One worker per scenario. Owns its own gp.Env and gp.Model.
    Build once, then call solve(action) repeatedly.
    """
    def __init__(self, builder_fn, builder_args, subproblem_id:int, verbose:bool=True):
        self.verbose = verbose
        # Build the model and linking constraints inside THIS env.
        builder_args['scenario_id'] = subproblem_id
        self.model, self.link_rows = builder_fn(**builder_args)
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
    def __init__(self, master_builder_fn, subproblem_builder_fn, get_solution, flatten_fn, num_subproblems):
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

    def solve(self, master_builder_args, subproblem_builder_args, tol=1e-6, max_iter=15000, verbose=False):
        lower_bound = -GRB.INFINITY
        upper_bound = GRB.INFINITY
        master_model, imm_cost, theta_vars, action_t_var = self.master_builder_fn(**master_builder_args)
        flat_action_t_var = self.flatten_fn(action_t_var)
        # Build one worker per scenario once, then reuse
        workers = [
            SubproblemWorker(self.subproblem_builder_fn, subproblem_builder_args, sid, verbose)
            for sid in range(self.num_subproblems)
        ]
        for iteration in range(1, max_iter + 1):
            if not solve_and_handle_errors(master_model, verbose=verbose):
                raise RuntimeError("Master model optimal solution not found")
            action_t = self.get_solution(action_t_var)
            flat_action_t = self.flatten_fn(action_t)

            lower_bound = master_model.ObjVal

            # Ask all workers to solve for this action
            # futures = [ex.submit(w.solve, action_t, verbose) for w in workers]
            feasibility_cuts = []
            optimality_cuts = []
            cost_to_go_estimation = 0.0
            all_feasible = True
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
                    action_t = self.get_solution(action_t_var, is_final=True)
                    return action_t, upper_bound, {}
            print('upper_bound:', upper_bound)
            print('lower_bound:', lower_bound)

            print('-' * 20)
        print('Max iterations reached')
        action_t = self.get_solution(action_t_var, is_final=True)
        return action_t, upper_bound, {}

if __name__ == "__main__":
    action = (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]))
    print(flatten(action))