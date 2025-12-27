import gurobipy as gp
from gurobipy import GRB


def solve_with_behavior(reset_model=False):
    m = gp.Model()
    m.Params.OutputFlag = 0  # Quiet

    # 1. Create variables
    x = m.addVar(name="x")
    y = m.addVar(name="y")

    # 2. Initial Objective and Constraints
    m.setObjective(10 * x + 10 * y, GRB.MAXIMIZE)
    c1 = m.addConstr(x + y <= 100)
    c2 = m.addConstr(x <= 60)

    m.optimize()
    print(f"First Solve - Obj: {m.ObjVal}, Iterations: {m.IterCount}")

    # --- THE DYNAMIC CHANGE ---
    # Change objective to favor Y and loosen X constraint
    m.setObjective(10 * x + 10.1 * y, GRB.MAXIMIZE)
    c2.RHS = 80

    if reset_model:
        m.reset(0)  # Clearing the "pollution"
        print("Model Reset Performed.")

    m.optimize()
    print(f"Second Solve - Obj: {m.ObjVal}, Iterations: {m.IterCount}")
    print("-" * 30)


print("Running WITHOUT reset (Warm Start):")
solve_with_behavior(reset_model=False)

print("Running WITH reset (Cold Start):")
solve_with_behavior(reset_model=True)