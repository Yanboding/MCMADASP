from utils import clean_value, solve_and_handle_errors

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
        self.master_model = master_builder()
        self.separation_callback = separation_callback
        self.get_constraint_data = get_constraint_data
        self.initial_candidates = initial_candidates
        # Use a set to keep track of added constraints to avoid duplicates
        self.added_constraints = set()
        self.candidates_list = []

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
        for iteration in range(max_iter):
            print(f"\n--- Iteration {iteration + 1} ---")
            if iteration == 0:
                # add initial columns to the master model
                for i, candidate in enumerate(self.initial_candidates):
                    self.add_constraint(candidate=candidate, constr_name=f"init_{i + 1}")
            else:
                # 2. Get the current primal solution
                solution = [clean_value(v.X, tol) for v in self.master_model.getVars()]
                is_constraint_added = False
                # separation_callback yields the row and violation in decreasing order
                for candidate, violation in self.separation_callback(solution):
                    if violation > tol and self.add_constraint(candidate=candidate, constr_name=f'constr_{iteration}'):
                        is_constraint_added = True
                        break
                if not is_constraint_added:
                    print(f"Optimal solution found after {iteration} iterations.")
                    break  # No new, valid, improving column was found
            # 1. Optimize the current relaxed master model
            if not solve_and_handle_errors(self.master_model):
                print("Master problem could not be solved to optimality. Aborting.")
                break
            print(f"Relaxed master objective: {self.master_model.ObjVal:.6f}")
        return self.master_model