import numpy as np
from gurobipy import Model, GRB, quicksum
from scipy.stats import poisson

# Parameters
s = 10.0  # current state
action_lb = 0.0
action_ub = 10.0
holding_cost = 1.0
ordering_cost = 2.0
discount = 1.0

# Demand distribution (discretized Poisson)
max_demand = 15
d_vals = np.arange(max_demand + 1)
p_vals = poisson.pmf(d_vals, mu=5)

# Value function at t+1 on a grid
state_grid = np.linspace(0, 20, 101)
V_next = np.maximum(0, 20 - state_grid)  # Example: V(s) = 20 - s

# Build Gurobi model
m = Model()
m.setParam("OutputFlag", 0)

# Decision variable: action
a = m.addVar(lb=action_lb, ub=action_ub, name="a")

# Auxiliary variables: interpolated V_{t+1}(s + a - d) for each d
v_interp = {}
for i, d in enumerate(d_vals):
    # Compute next state range for interpolation
    next_s_expr = s + a - d
    # Find interpolation indices
    idx = np.searchsorted(state_grid, s + action_ub - d)
    idx = min(idx, len(state_grid) - 2)

    # Linear interpolation: s' = λ * s1 + (1 - λ) * s0
    λ = m.addVar(lb=0, ub=1, name=f"lambda_{i}")
    v = m.addVar(lb=-GRB.INFINITY, name=f"v_interp_{i}")

    s0 = state_grid[idx]
    s1 = state_grid[idx + 1]
    V0 = V_next[idx]
    V1 = V_next[idx + 1]

    # Enforce interpolation: v = λ * V1 + (1 - λ) * V0
    m.addConstr(v == λ * V1 + (1 - λ) * V0)

    # Enforce interpolation position: next_s = λ * s1 + (1 - λ) * s0
    m.addConstr(s + a - d == λ * s1 + (1 - λ) * s0)

    v_interp[d] = v

# Objective: minimize cost + expected future value
immediate_cost = ordering_cost * a + holding_cost * s
expected_future = quicksum(p_vals[i] * v_interp[d] for i, d in enumerate(d_vals))
m.setObjective(immediate_cost + discount * expected_future, GRB.MINIMIZE)

m.optimize()

# Output result
print(f"Optimal action: {a.X:.4f}")
print(f"Value function at state s={s:.1f}: V_t(s) = {m.ObjVal:.4f}")