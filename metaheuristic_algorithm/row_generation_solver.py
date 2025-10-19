import gurobipy as gp
import numpy as np
from gurobipy import GRB
from utils import solve_and_handle_errors

class RowGenerationSolver:
    """
    A class to solve a Linear Program with a very large number of constraints
    using the row generation (cutting-plane) method.
    """

    def __init__(self, master_builder, separation_callback, get_constraint_data, initial_candidates):
        """
        Initializes the RowGenerationSolver.

        Args:
            master_builder (function): A function that takes no arguments and
                returns a gurobipy.Model object. This model should contain all
                the variables of the problem but can start with a minimal set
                of constraints (or none).

            separation_callback (function): The separation oracle.
                - It takes one argument: a dictionary mapping Gurobi variable
                  objects to their current solution values (primal values, var.X).
                - It must return a list of violated constraints. Each item in the
                  list should be a tuple: (constraint_data, violation_amount),
                  where constraint_data is itself a tuple (lhs, sense, rhs)
                  that can be passed to Gurobi's model.addConstr().
        """
        self.master_model, self.objective, self.coefficient_vars = master_builder()
        self.l2_penalty = gp.quicksum(variable*variable for variable in self.coefficient_vars)
        # --- Define L1 penalty term using absolute-value auxiliary vars ---
        self.abs_vars = []
        for v in self.coefficient_vars:
            abs_v = self.master_model.addVar(lb=0.0, name=f"abs_{v.VarName}")
            self.master_model.addConstr(abs_v >= v, name=f"abs_pos_{v.VarName}")
            self.master_model.addConstr(abs_v >= -v, name=f"abs_neg_{v.VarName}")
            self.abs_vars.append(abs_v)
        self.l1_penalty = gp.quicksum(self.abs_vars)
        self.master_model.update()
        
        self.separation_callback = separation_callback
        self.get_constraint_data = get_constraint_data
        self.initial_candidates = initial_candidates
        # Use a set to keep track of added constraints to avoid duplicates
        self.added_constraints = set()
        self.candidates_list = []

        # --- penalty parameters ---
        self.penalty_weight_l2 = 1e-3       # initial penalty weight
        self.max_penalty_weight_l2 = 1e-3   # upper bound
        self.min_penalty_weight_l2 = 1e-6   # lower bound
        self.adapt_factor_up_l2 = 2.0       # factor to increase penalty when unstable
        self.adapt_factor_down_l2 = 0.99     # factor to decrease when stable

        self.penalty_weight_l1 = 1e-3       # initial penalty weight
        self.max_penalty_weight_l1 = 1e-3   # upper bound
        self.min_penalty_weight_l1 = 1e-6   # lower bound
        self.adapt_factor_up_l1 = 2.0       # factor to increase penalty when unstable
        self.adapt_factor_down_l1 = 0.99     # factor to decrease when stable
        
        self.stable_iters = 0
        self.stable_patience = 5
        self.stable_threshold = 1e-4

    def add_constraint(self, candidate, constr_name):
        """
        Adds a new constraint (row) to the master model.

        Args:
            candidate (tuple): A tuple (lhs, sense, rhs) representing the
                constraint to be added.
            constr_name (str): The name for the new constraint.
        """
        constraint = self.get_constraint_data(model=self.master_model,
                                              candidate=candidate)
        # Create a string representation to check for duplicates
        constraint_str = f"{constraint}"
        if constraint_str in self.added_constraints:
            print(f"Warning: Attempting to add duplicate constraint: {constraint_str}")
            return False
        self.master_model.addConstr(constraint, name=constr_name)
        self.master_model.update()
        self.added_constraints.add(constraint_str)
        self.candidates_list.append(candidate)
        return True
    
    def calculate_penalty_weight(self, drift, max_violation, tol):
        """
        Adaptively updates the L2 penalty weight (rho) based on system stability.

        Args:
            drift (float): Euclidean norm of coefficient change between iterations.
            max_violation (float): Maximum constraint violation from separation callback.
            tol (float): Tolerance threshold for violations.

        Returns:
            float: Updated penalty weight.
        """
        print(f"Drift: {drift:.6f}, Max Violation: {max_violation:.6f}, Current L2 Penalty Weight: {self.penalty_weight_l2:.6f}, Current L1 Penalty Weight: {self.penalty_weight_l1:.6f}")
        new_weight_l2 = max(self.min_penalty_weight_l2, self.penalty_weight_l2 * self.adapt_factor_down_l2)
        new_weight_l1 = max(self.min_penalty_weight_l1, self.penalty_weight_l1 * self.adapt_factor_down_l1)
        #new_weight_l2 = self.penalty_weight_l2
        #new_weight_l1 = self.penalty_weight_l1
        return new_weight_l2, new_weight_l1


    def solve(self, tol=1e-6, max_iter=1000):
        """
        Executes the row generation loop to solve the problem.

        Args:
            tol (float): The tolerance for constraint violation. If the most
                violated constraint has a violation less than this, the
                algorithm terminates.
            max_iter (int): The maximum number of row generation iterations.

        Returns:
            gurobipy.Model: The final solved master model.
        """
        previous_solution = [float('inf') for v in self.master_model.getVars()]
        drift = float('inf')
        for iteration in range(max_iter):
            print(f"\n--- Iteration {iteration + 1} ---")
            max_violation = 0.0
            if iteration == 0:
                # add initial columns to the master model
                for i, candidate in enumerate(self.initial_candidates):
                    self.add_constraint(candidate=candidate, constr_name=f"init_{i + 1}")
            else:
                # 2. Get the current primal solution
                solution = [v.X for v in self.master_model.getVars()]
                # separation_callback yields the row and violation in decreasing order
                for candidate, violation in self.separation_callback(solution):
                    max_violation = max(max_violation, violation)
                    if violation > tol and self.add_constraint(candidate=candidate, constr_name=f'constr_{iteration}'):
                        break
                else:
                    print(f"Optimal solution found after {iteration} iterations.")
                    break  # No new, valid, improving column was found
            
            self.penalty_weight_l2, self.penalty_weight_l1 = self.calculate_penalty_weight(
                                                                                            drift=drift,
                                                                                            max_violation=max_violation,
                                                                                            tol=tol
                                                                                        )
            self.master_model.setObjective(self.objective - self.penalty_weight_l2 * self.l2_penalty - self.penalty_weight_l1 * self.l1_penalty, self.master_model.ModelSense)
            self.master_model.update()
            # 1. Optimize the current relaxed master model
            if not solve_and_handle_errors(self.master_model):
                print("Master problem could not be solved to optimality. Aborting.")
                break
            current_solution = np.array([v.X for v in self.master_model.getVars()])
            drift = np.linalg.norm(current_solution - previous_solution)
            previous_solution = current_solution
            print(f"Relaxed master objective: {self.master_model.ObjVal:.6f}")
        return self.master_model
