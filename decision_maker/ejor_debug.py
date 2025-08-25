import gurobipy as gp
from gurobipy import GRB

def solve_master_rmp():
    """
    Sets up and solves the specified linear programming problem using Gurobi.
    """
    try:
        # Create a new model
        m = gp.Model("MasterRMP")

        # --- 1. Define Variables ---
        # The problem statement includes constraints like U^1_0 >= 0.
        # These are handled by setting the lower bound (lb) of the variables.
        # The default lower bound is 0, so we only need to specify it for
        # variables that can be negative.

        # W variables
        # W^1_0 and W^2_0 appear to be unrestricted (free variables)
        w10 = m.addVar(name="W^1_0")
        w20 = m.addVar(name="W^2_0")
        # constr_8 sets W^3_0 <= 0, which we can set as an upper bound
        w30 = m.addVar(lb=-GRB.INFINITY, ub=0, name="W^3_0")

        w11 = m.addVar(lb=0, name="W^1_1")
        w21 = m.addVar(lb=0, name="W^2_1")
        w31 = m.addVar(lb=0, name="W^3_1")

        # U variables (all are non-negative, which is the default)
        u10 = m.addVar(name="U^1_0")
        u11 = m.addVar(name="U^1_1")
        u12 = m.addVar(name="U^1_2")
        u20 = m.addVar(name="U^2_0")
        u21 = m.addVar(name="U^2_1")
        u30 = m.addVar(name="U^3_0")

        # --- 2. Set Objective Function ---
        objective = (w10 + w20 + w30
                     + 2.5 * (u10 + u11 + u12 + u20 + u21 + u30)
                     + 3 * (w11 + w21 + w31))
        m.setObjective(objective, GRB.MAXIMIZE)

        # --- 3. Add Constraints ---
        # Note: Non-negativity constraints and constr_8 (W^3_0 <= 0) were
        # handled during variable creation.

        m.addConstr(w10 - 0.99*w20 + 5*u10 + 5*u11 + 5*u12 - 4.95*u20 - 4.95*u21 + 9*w11 - 2.97*w21 <= 810, "init_1")
        m.addConstr(w20 - 0.99*w30 + 5*u20 + 5*u21 - 4.95*u30 + 9*w21 - 2.97*w31 <= 810, "init_2")
        m.addConstr(w30 + 5*u30 + 9*w31 <= 810, "init_3")

        m.addConstr(w20 - 0.99*w30 - 2.97*w31 <= 0, "constr_1")
        m.addConstr(w10 - 0.99*w20 - 2.97*w21 <= 0, "constr_2")
        m.addConstr(w10 - 0.99*w20 + 5*u10 - 2.97*w21 <= 0, "constr_3")
        m.addConstr(w10 - 0.99*w20 + 5*u11 - 2.97*w21 <= 198, "constr_4")
        m.addConstr(w10 - 0.99*w20 + 5*u12 - 4.95*u21 - 2.97*w21 <= 0, "constr_5")
        m.addConstr(w10 - 0.99*w20 + 5*u11 + 5*u12 - 4.95*u20 + 3*w11 - 2.97*w21 <= 266.02, "constr_6")
        m.addConstr(w20 - 0.99*w30 + 5*u20 + 5*u21 - 4.95*u30 - 2.97*w31 <= 0, "constr_7")
        # constr_8 is handled in variable definition
        m.addConstr(w30 + 5*u30 <= 0, "constr_9")
        m.addConstr(w10 - 0.99*w20 - 4.95*u20 - 4.95*u21 + 9*w11 - 2.97*w21 <= 297.607, "constr_10")
        m.addConstr(w20 - 0.99*w30 - 4.95*u30 + 6*w21 - 2.97*w31 <= 169.3, "constr_11")
        m.addConstr(w10 - 0.99*w20 + 5*u11 + u12 - 4.95*u21 + 5*w11 - 2.97*w21 <= 327.402, "constr_12")
        m.addConstr(w10 - 0.99*w20 + 5*u10 + 5*u11 + 5*u12 - 4.95*u20 - 4.95*u21 - 2.97*w21 <= 0, "constr_13")
        m.addConstr(w30 + 9*w31 <= 610, "constr_14")
        m.addConstr(w30 + 3*w31 <= 70, "constr_15")
        m.addConstr(w10 - 0.99*w20 - 3.96*u21 + 4*w11 - 2.97*w21 <= 79.402, "constr_16")
        m.addConstr(w10 - 0.99*w20 + 5*u12 - 3.96*u20 - 4.95*u21 + 4*w11 - 2.97*w21 <= 59.8, "constr_17")
        m.addConstr(w10 - 0.99*w20 + 5*u11 - 4.95*u20 + 2*w11 - 2.97*w21 <= 20, "constr_18")
        m.addConstr(w10 - 0.99*w20 + u10 + u11 - 4.95*u20 - 3.96*u21 + 6*w11 - 2.97*w21 <= 119.202, "constr_19")
        m.addConstr(w20 - 0.99*w30 + u20 + u21 - 4.95*u30 + 4*w21 - 2.97*w31 <= 59.8, "constr_20")
        m.addConstr(w30 + 2*w31 <= 20, "constr_21")
        m.addConstr(w10 - 0.99*w20 + u10 + 5*u12 - 4.95*u21 + 2*w11 - 2.97*w21 <= 20, "constr_22")
        m.addConstr(w10 - 0.99*w20 + u10 + 5*u11 - 4.95*u20 + 2*w11 - 2.97*w21 <= 20, "constr_23")
        m.addConstr(w20 - 0.99*w30 + 5*u20 - 3.96*u30 + 2*w21 - 2.97*w31 <= 39.8, "constr_24")
        m.addConstr(w20 - 0.99*w30 + 5*u20 + u21 - 4.95*u30 + 2*w21 - 2.97*w31 <= 39.8, "constr_25")

        # --- 4. Solve the Model ---
        m.optimize()

        # --- 5. Display Results ---
        print("-" * 50)
        if m.Status == GRB.OPTIMAL:
            print(f"✅ Optimal objective value found: {m.ObjVal:.4f}")
            print("\nNon-zero variable values:")
            for v in m.getVars():
                # Only print variables with a value significantly different from zero
                if abs(v.X) > 1e-6:
                    print(f"   {v.VarName:<6} = {v.X:.4f}")
        else:
            print(f"❌ Optimization was not successful. Status code: {m.Status}")
        print("-" * 50)

    except gp.GurobiError as e:
        print(f"Error code {e.errno}: {e}")
    except AttributeError:
        print("Could not find Gurobi. Please ensure it is installed and licensed.")

# Run the solver
solve_master_rmp()