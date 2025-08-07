import gurobipy as gp

from utils import solve_and_handle_errors


class ColumnGenerationSolver:
    def __init__(self, master_builder, pricing_callback, initial_columns, get_constr_coefficients, get_obj_coefficient):
        """
        master_builder: function(model, columns) -> None
            Add variables and constraints to the model, given columns.
        pricing_callback: function(dual_values) -> list of columns
            Given dual values, generate new columns (can be empty if optimal).
        initial_columns: list
            Initial list of columns for the master problem.
        sense: GRB.MINIMIZE or GRB.MAXIMIZE
        """
        self.pricing_callback = pricing_callback
        self.get_constr_coefficients = get_constr_coefficients
        self.get_obj_coefficient = get_obj_coefficient
        self.master_model = master_builder()
        self.candidates = set()
        # add initial columns to the master model
        for i, candidate in enumerate(initial_columns):
            self.add_column(candidate=candidate, col_name=f"init_{i + 1}")

    def add_column(self, candidate, col_name):
        new_col = gp.Column()
        for j, coefficient in enumerate(self.get_constr_coefficients(candidate)):
            new_col.addTerms([float(coefficient)], [self.master_model.getConstrs()[j]])

        self.master_model.addVar(obj=self.get_obj_coefficient(candidate), column=new_col, name=col_name)
        self.master_model.update()
        candidate_str = str(candidate)
        if candidate_str in self.candidates:
            raise ValueError(f'already add this candidate: {candidate_str}')
        self.candidates.add(candidate_str)

    def solve(self, tol=1e-4, max_iter=3000):
        iteration = 0
        while iteration < max_iter:
            # 1. optimize the model
            self.master_model.optimize()
            if self.master_model.Status != gp.GRB.OPTIMAL:
                # Print a more user-friendly explanation
                if self.master_model.status == gp.GRB.INFEASIBLE:
                    self.master_model.write('infeasible.lp')
                    print("Model is infeasible.")
                elif self.master_model.status == gp.GRB.UNBOUNDED:
                    print("Model is unbounded.")
                elif self.master_model.status == gp.GRB.INF_OR_UNBD:
                    print("Model is infeasible or unbounded.")
                elif self.master_model.status == gp.GRB.TIME_LIMIT:
                    print("Time limit reached before optimality.")
                elif self.master_model.status == gp.GRB.INTERRUPTED:
                    print("Optimization was interrupted.")
                elif self.master_model.status == gp.GRB.NUMERIC:
                    print("Numerical issues encountered.")
                else:
                    print("See Gurobi documentation for other status codes.")
                raise ValueError(f"Master model returned status {self.master_model.Status}")
            print('optimal value:', self.master_model.ObjVal)
            # 2. Get dual values
            duals = [constr.Pi for constr in self.master_model.getConstrs()]
            # 3. Pricing (column generation)
            is_column_added = False
            for candidate, reduce_cost in self.pricing_callback(duals):
                candidate_cost = self.get_obj_coefficient(candidate)
                candidate_coeffs = self.get_constr_coefficients(candidate)
                # reduced cost = cost − ∑ dual[j] * coeffs[j]
                rc = candidate_cost
                for j, coeff in enumerate(candidate_coeffs):
                    rc -= duals[j] * coeff
                if abs(rc -reduce_cost) > tol:
                    print(duals)
                    print(f"Warning: calculated reduced cost {rc} differs from pricing callback {reduce_cost}")
                    raise ValueError("Inconsistent reduced cost calculation.")
                if -reduce_cost < tol:
                    print('reduce_cost:', reduce_cost)
                    break
                if str(candidate) not in self.candidates:
                    # print(f'iterations: {iteration}, try to add {candidate} with reduce_cost {reduce_cost}')
                    self.add_column(candidate=candidate, col_name=f"x_{iteration}")
                    is_column_added = True
                    break
            if not is_column_added:
                print(f"Optimal solution found after {iteration} iterations.")
                print(duals)
                break  # No new, valid, improving column was found
            iteration += 1
        if iteration >= max_iter:
            solve_and_handle_errors(self.master_model)
            self.master_model.optimize()
            if self.master_model.Status != gp.GRB.OPTIMAL:
                # Print a more user-friendly explanation
                if self.master_model.status == gp.GRB.INFEASIBLE:
                    self.master_model.write('infeasible.lp')
                    print("Model is infeasible.")
                elif self.master_model.status == gp.GRB.UNBOUNDED:
                    print("Model is unbounded.")
                elif self.master_model.status == gp.GRB.INF_OR_UNBD:
                    print("Model is infeasible or unbounded.")
                elif self.master_model.status == gp.GRB.TIME_LIMIT:
                    print("Time limit reached before optimality.")
                elif self.master_model.status == gp.GRB.INTERRUPTED:
                    print("Optimization was interrupted.")
                elif self.master_model.status == gp.GRB.NUMERIC:
                    print("Numerical issues encountered.")
                else:
                    print("See Gurobi documentation for other status codes.")
                raise ValueError(f"Master model returned status {self.master_model.Status}")
            print('optimal value:', self.master_model.ObjVal)
            print(f"Reached max iterations ({max_iter}).")
