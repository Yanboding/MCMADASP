import gurobipy as gp
import numpy as np
from gurobipy import GRB
from utils import solve_and_handle_errors

class RowGenerationSolver:

    def __init__(self, master_builder, separation_callback, get_constraint_data, initial_candidates):
        self.master_model, self.objective, self.coefficient_vars = master_builder()
        self.l2_penalty = gp.quicksum(variable*variable for variable in self.coefficient_vars)
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
        self.added_constraints = set()
        self.candidates_list = []

        self.penalty_weight_l2 = 1e-3
        self.max_penalty_weight_l2 = 1e-3
        self.min_penalty_weight_l2 = 1e-6
        self.adapt_factor_up_l2 = 2.0
        self.adapt_factor_down_l2 = 0.99

        self.penalty_weight_l1 = 1e-3
        self.max_penalty_weight_l1 = 1e-3
        self.min_penalty_weight_l1 = 1e-6
        self.adapt_factor_up_l1 = 2.0
        self.adapt_factor_down_l1 = 0.99
        
        self.stable_iters = 0
        self.stable_patience = 5
        self.stable_threshold = 1e-4

    def add_constraint(self, candidate, constr_name):
        constraint = self.get_constraint_data(model=self.master_model,
                                              candidate=candidate)
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
        print(f"Drift: {drift:.6f}, Max Violation: {max_violation:.6f}, Current L2 Penalty Weight: {self.penalty_weight_l2:.6f}, Current L1 Penalty Weight: {self.penalty_weight_l1:.6f}")
        new_weight_l2 = max(self.min_penalty_weight_l2, self.penalty_weight_l2 * self.adapt_factor_down_l2)
        new_weight_l1 = max(self.min_penalty_weight_l1, self.penalty_weight_l1 * self.adapt_factor_down_l1)
        return new_weight_l2, new_weight_l1

    def solve(self, tol=1e-6, max_iter=1000):
        previous_solution = [float('inf') for v in self.master_model.getVars()]
        drift = float('inf')
        for iteration in range(max_iter):
            print(f"\n--- Iteration {iteration + 1} ---")
            max_violation = 0.0
            if iteration == 0:
                for i, candidate in enumerate(self.initial_candidates):
                    self.add_constraint(candidate=candidate, constr_name=f"init_{i + 1}")
            else:
                solution = [v.X for v in self.master_model.getVars()]
                for candidate, violation in self.separation_callback(solution):
                    max_violation = max(max_violation, violation)
                    if violation > tol and self.add_constraint(candidate=candidate, constr_name=f'constr_{iteration}'):
                        break
                else:
                    print(f"Optimal solution found after {iteration} iterations.")
                    break
            
            self.penalty_weight_l2, self.penalty_weight_l1 = self.calculate_penalty_weight(
                                                                                            drift=drift,
                                                                                            max_violation=max_violation,
                                                                                            tol=tol
                                                                                        )
            self.penalty_weight_l2 = 0
            self.penalty_weight_l1 = 0
            self.master_model.setObjective(self.objective - self.penalty_weight_l2 * self.l2_penalty - self.penalty_weight_l1 * self.l1_penalty, self.master_model.ModelSense)
            self.master_model.update()
            if not solve_and_handle_errors(self.master_model):
                print("Master problem could not be solved to optimality. Aborting.")
                break
            current_solution = np.array([v.X for v in self.master_model.getVars()])
            drift = np.linalg.norm(current_solution - previous_solution)
            previous_solution = current_solution
            print(f"Relaxed master objective: {self.master_model.ObjVal:.6f}")
        return self.master_model
