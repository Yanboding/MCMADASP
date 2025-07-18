import gurobipy as gp

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
        # add initial columns to the master model
        for i, candidate in enumerate(initial_columns):
            self.add_column(candidate=candidate, col_name=f"init_{i + 1}")

    def add_column(self, candidate, col_name):
        new_col = gp.Column()
        for j, coefficient in enumerate(self.get_constr_coefficients(candidate)):
            new_col.addTerms([float(coefficient)], [self.master_model.getConstrs()[j]])

        self.master_model.addVar(obj=self.get_obj_coefficient(candidate), column=new_col, name=col_name)
        self.master_model.update()

    def solve(self, tol=1e-20, max_iter=1000):
        iteration = 0
        while True:
            #self.master_model.write(f"model_{iteration}.rlp")
            # 1. optimize the model
            self.master_model.optimize()
            if self.master_model.Status != gp.GRB.OPTIMAL:
                raise ValueError(f"Master model returned status {self.master_model.Status}")
            # 2. Get dual values
            duals = [constr.Pi for constr in self.master_model.getConstrs()]
            # 3. Pricing (column generation)
            candidate, reduce_cost = self.pricing_callback(duals)
            if not candidate or iteration >= max_iter or -reduce_cost < tol:
                break
            self.add_column(candidate=candidate, col_name=f"x_{iteration}")
            iteration += 1
