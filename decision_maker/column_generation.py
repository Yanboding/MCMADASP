import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from itertools import product

# Problem data
widths = [20, 35, 50]  # Demand widths (cm)
demands = [50, 40, 30]  # Number of rolls needed for each width
roll_width = 100  # Width of large roll (cm)

# --- Column Generation Approach ---
# Initial patterns: one for each width, cutting as many as possible
initial_patterns = [
    [int(roll_width // widths[0]), 0, 0],  # e.g., 5 pieces of 20 cm
    [0, int(roll_width // widths[1]), 0],  # e.g., 2 pieces of 35 cm
    [0, 0, int(roll_width // widths[2])]   # e.g., 2 pieces of 50 cm
]
print(initial_patterns)
# Master Problem (Restricted Master Problem, RMP)
def solve_master(patterns, demands):
    model = gp.Model("Master_Problem")
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output
    x = model.addVars(len(patterns), vtype=GRB.CONTINUOUS, lb=0, name="x")
    model.setObjective(gp.quicksum(x[j] for j in range(len(patterns))), GRB.MINIMIZE)
    constraints = model.addConstrs(
        (gp.quicksum(patterns[j][i] * x[j] for j in range(len(patterns))) >= demands[i]
         for i in range(len(widths))),
        name="Demand"
    )
    model.optimize()
    duals = [constraints[i].Pi for i in range(len(widths))]
    obj_value = model.objVal
    x_values = [x[j].x for j in range(len(patterns))]
    return duals, obj_value, x_values

# Subproblem: Find a new pattern with negative reduced cost
def solve_subproblem(duals, widths, roll_width):
    model = gp.Model("Subproblem")
    model.setParam("OutputFlag", 0)
    a = model.addVars(len(widths), vtype=GRB.INTEGER, lb=0, name="a")
    model.setObjective(gp.quicksum(duals[i] * a[i] for i in range(len(widths))), GRB.MAXIMIZE)
    model.addConstr(
        gp.quicksum(widths[i] * a[i] for i in range(len(widths))) <= roll_width,
        name="Width_Limit"
    )
    model.optimize()
    pattern = [int(a[i].x) for i in range(len(widths))]
    reduced_cost = 1 - sum(duals[i] * pattern[i] for i in range(len(widths)))
    return reduced_cost, pattern

# Column Generation Loop
def column_generation():
    patterns = initial_patterns.copy()
    while True:
        duals, obj_value, x_values = solve_master(patterns, demands)
        reduced_cost, new_pattern = solve_subproblem(duals, widths, roll_width)
        if reduced_cost >= -1e-6:  # Stop if no improving pattern
            break
        patterns.append(new_pattern)
    return patterns, x_values, obj_value

# Solve final master problem as integer program
def solve_integer_master(patterns, demands):
    model = gp.Model("Integer_Master_Problem")
    model.setParam("OutputFlag", 0)
    x = model.addVars(len(patterns), vtype=GRB.INTEGER, lb=0, name="x")
    model.setObjective(gp.quicksum(x[j] for j in range(len(patterns))), GRB.MINIMIZE)
    model.addConstrs(
        (gp.quicksum(patterns[j][i] * x[j] for j in range(len(patterns))) >= demands[i]
         for i in range(len(widths))),
        name="Demand"
    )
    model.optimize()
    obj_value = model.objVal
    x_values = [x[j].x for j in range(len(patterns))]
    return obj_value, x_values

# --- Full Enumeration Approach ---
# Generate all feasible patterns
def generate_all_patterns(widths, roll_width):
    patterns = []
    max_cuts = [int(roll_width // w) + 1 for w in widths]  # Max number of cuts for each width
    for combo in product(*[range(m + 1) for m in max_cuts]):
        pattern = list(combo)
        if sum(pattern[i] * widths[i] for i in range(len(widths))) <= roll_width:
            patterns.append(pattern)
    return patterns

# Solve ILP with all patterns
def solve_full_enumeration(patterns, demands):
    model = gp.Model("Full_Enumeration")
    model.setParam("OutputFlag", 0)
    x = model.addVars(len(patterns), vtype=GRB.INTEGER, lb=0, name="x")
    model.setObjective(gp.quicksum(x[j] for j in range(len(patterns))), GRB.MINIMIZE)
    model.addConstrs(
        (gp.quicksum(patterns[j][i] * x[j] for j in range(len(patterns))) >= demands[i]
         for i in range(len(widths))),
        name="Demand"
    )
    model.optimize()
    obj_value = model.objVal
    x_values = [x[j].x for j in range(len(patterns))]
    return obj_value, x_values

# --- Compare Running Speeds ---
# Column Generation
start_time = time.time()
patterns_cg, x_values_lp, obj_lp = column_generation()
obj_cg, x_values_cg = solve_integer_master(patterns_cg, demands)
cg_time = time.time() - start_time

# Full Enumeration
start_time = time.time()
all_patterns = generate_all_patterns(widths, roll_width)
obj_fe, x_values_fe = solve_full_enumeration(all_patterns, demands)
fe_time = time.time() - start_time

# --- Results ---
print("=== Column Generation Results ===")
print(f"Objective (min rolls): {obj_cg:.0f}")
print(f"Running time: {cg_time:.4f} seconds")
print(f"Number of patterns generated: {len(patterns_cg)}")
print("Patterns and usage:")
for j, pattern in enumerate(patterns_cg):
    if x_values_cg[j] > 0:
        print(f"Pattern {j}: {pattern} used {x_values_cg[j]:.0f} times")

print("\n=== Full Enumeration Results ===")
print(f"Objective (min rolls): {obj_fe:.0f}")
print(f"Running time: {fe_time:.4f} seconds")
print(f"Number of patterns generated: {len(all_patterns)}")
print("Patterns and usage:")
for j, pattern in enumerate(all_patterns):
    if x_values_fe[j] > 0:
        print(f"Pattern {j}: {pattern} used {x_values_fe[j]:.0f} times")

print("\n=== Comparison ===")
print(f"Column Generation Time: {cg_time:.4f} seconds")
print(f"Full Enumeration Time: {fe_time:.4f} seconds")
print(f"Speedup (Full Enum / Col Gen): {fe_time / cg_time:.2f}x")