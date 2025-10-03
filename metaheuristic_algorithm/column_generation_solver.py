import gurobipy as gp

from utils import solve_and_handle_errors, clean_value


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
        self.master_model = master_builder()
        self.pricing_callback = pricing_callback
        self.initial_candidates = initial_columns
        self.get_constr_coefficients = get_constr_coefficients
        self.get_obj_coefficient = get_obj_coefficient
        self.candidates = set()
        self.candidates_list = []

    def add_column(self, candidate, col_name):
        new_col = gp.Column()
        if self.get_constr_coefficients != None:
            for j, coefficient in enumerate(self.get_constr_coefficients(candidate)):
                new_col.addTerms([float(coefficient)], [self.master_model.getConstrs()[j]])
        kwargs = {'column': new_col, 'name': col_name, 'lb': 0}
        if self.get_obj_coefficient != None:
            kwargs['obj'] = self.get_obj_coefficient(candidate)
        self.master_model.addVar(**kwargs)
        self.master_model.update()
        candidate_str = str(candidate)
        if candidate_str in self.candidates:
            raise ValueError(f'already add this candidate: {candidate_str}')
        self.candidates.add(candidate_str)
        self.candidates_list.append(candidate)
        return True

    def initial_columns_solve(self, tol=1e-6, max_iter=3000, verbose=False):
        for iteration in range(max_iter):
            print(f"\n--- Iteration {iteration + 1} ---")
            if iteration == 0:
                # 0. add initial columns to the master model
                for i, candidate in enumerate(self.initial_candidates):
                    self.add_column(candidate=candidate, col_name=f"init_X({i + 1})")
            else:
                duals = [clean_value(constr.Pi, 1e-8) for constr in self.master_model.getConstrs()]
                is_column_added = False
                # separation_callback yields the row and violation in decreasing order
                for candidate, reduce_cost in self.pricing_callback(duals):
                    if -reduce_cost > tol and (is_column_added :=self.add_column(candidate=candidate, col_name=f'X({iteration})')):
                        break
                if not is_column_added:
                    print(f"Optimal solution found after {iteration+1} iterations.")
                    break  # No new, valid, improving column was found
            # 3. Optimize the current relaxed master model
            if not solve_and_handle_errors(self.master_model):
                if verbose:
                    print("Master problem could not be solved to optimality. Aborting.")
                break
            if clean_value(self.master_model.ObjVal, 1e-8) < tol:
                if verbose:
                    print([clean_value(constr.Pi, tol) for constr in self.master_model.getConstrs()])
                return self.candidates_list
            if verbose:
                print('objective:', self.master_model.ObjVal, 'dual:', [clean_value(constr.Pi, tol) for constr in self.master_model.getConstrs()])
        if verbose:
            print("Final: ", self.master_model.ObjVal)
        raise ValueError('Finding feasible columns failed!')

    def solve(self, tol=1e-6, max_iter=30000):
        for iteration in range(max_iter):
            print(f"\n--- Iteration {iteration + 1} ---")
            if iteration == 0:
                # 0. add initial columns to the master model
                for i, candidate in enumerate(self.initial_candidates):
                    self.add_column(candidate=candidate, col_name=f"init_X({i + 1})")
            else:
                # 2. Get dual values
                duals = [clean_value(constr.Pi, 1e-12) for constr in self.master_model.getConstrs()]
                is_column_added = False
                # separation_callback yields the row and violation in decreasing order
                for candidate, reduce_cost in self.pricing_callback(duals):
                    if -reduce_cost > tol and self.add_column(candidate=candidate, col_name=f'X({iteration})'):
                        is_column_added = True
                        break
                if not is_column_added:
                    print(f"Optimal solution found after {iteration} iterations.")
                    break  # No new, valid, improving column was found
            # 3. Optimize the current relaxed master model
            if not solve_and_handle_errors(self.master_model):
                print("Master problem could not be solved to optimality. Aborting.")
                break
            print(f"Relaxed master objective: {self.master_model.ObjVal:.6f}")
        return self.master_model
