import gurobipy as gp
from gurobipy import GRB
import numpy as np

# --- 1. Problem Data Generation ---
# We'll create a smaller, illustrative problem instance.
I = 2  # Number of treatment types
N = 10  # Total number of periods in the horizon
H = 5  # Overtime horizon
L = 1  # Max treatment length
M = 10  # Number of scenarios
t = 0  # Current time period (we solve for the decision at t=0)
CAPACITY = 5  # Daily capacity C
GAMMA = 0.95  # Discount factor


def generate_problem_data():
    """Generates random data for a runnable example."""
    np.random.seed(42)
    data = {}
    data['c'] = np.random.randint(10, 30, size=(N + 1, I))  # Appointment costs
    data['o'] = np.random.randint(50, 100, size=(H + 1))  # Overtime costs
    data['r'] = np.array([[2, 1]])

    # Initial state and current demand
    data['u_t'] = np.zeros(N - t + 1)
    data['delta_t'] = np.random.randint(5, 10, size=I)

    # Future demand scenarios
    data['delta_future'] = {
        m: np.random.randint(2, 8, size=(N - t - 1, I)) for m in range(M)
    }
    return data


# --- 2. The Master Problem ---
def create_master_problem(data):
    """Creates the Benders master problem."""
    master = gp.Model("Master")

    # First-stage decision variables
    x_t = master.addVars(N - t + 1, I, vtype=GRB.INTEGER, name="x_t")
    y_t = master.addVars(H - t + 1, vtype=GRB.INTEGER, name="y_t")
    theta = master.addVar(name="theta", lb=0)  # Approximation of future costs

    # Objective: Minimize immediate cost + approximated future cost
    immediate_cost = gp.quicksum(data['c'][j, i] * x_t[j, i] for j in range(N - t + 1) for i in range(I)) + \
                     gp.quicksum(data['o'][j] * y_t[j] for j in range(H - t + 1))
    master.setObjective(immediate_cost + theta, GRB.MINIMIZE)

    # First-stage constraints (only involving t)
    # Constraint (1d): Fulfill current demand
    master.addConstrs((x_t.sum('*', i) == data['delta_t'][i] for i in range(I)), name="demand_t")

    # Helper variables for state transition
    u_bar_t = master.addVars(N - t + 1, name="u_bar_t")

    # Constraint (1a): Post-decision state
    for j in range(N - t + 1):
        resource_sum = gp.quicksum(x_t[k, i] * data['r'][j - k, i]
                                   for i in range(I)
                                   for k in range(max(j - L + 1, 0), j + 1))
        # Handle overtime index carefully
        overtime_term = y_t[j] if j <= H - t else 0
        master.addConstr(u_bar_t[j] == data['u_t'][j] + resource_sum - overtime_term, name=f"post_state_{j}")

    # Constraint (1c): Capacity
    master.addConstrs((u_bar_t[j] <= CAPACITY for j in range(N - t + 1)), name="capacity_t")

    master._vars = {'x_t': x_t, 'y_t': y_t, 'theta': theta}
    master._u_bar_t = u_bar_t  # Expose for use in subproblems
    return master


# --- 3. The Subproblem ---
def solve_subproblem(data, master_solution, scenario_omega):
    """
    Solves the second-stage problem for a single scenario, given the master's decision.
    Returns the objective value and duals for the Benders cut.
    """
    subproblem = gp.Model(f"Subproblem_{scenario_omega}")
    subproblem.setParam('OutputFlag', 0)

    # Calculate the initial state u_t+1 based on the master solution
    u_t1 = np.zeros(N - (t + 1) + 1)
    for j in range(len(u_t1)):
        u_t1[j] = master_solution['u_bar_t'][j + 1]

    # --- Subproblem Variables (from t+1 to N) ---
    x = {}
    y = {}
    for tau in range(1, N - t):
        x[tau] = subproblem.addVars(N - t - tau + 1, I, vtype=GRB.CONTINUOUS, name=f"x_{t + tau}")
        y[tau] = subproblem.addVars(H - t - tau + 1, vtype=GRB.CONTINUOUS, name=f"y_{t + tau}")

    # --- Subproblem Objective ---
    future_cost = gp.quicksum(
        data['c'][j, i] * x[tau][j, i]
        for tau in range(1, N - t) for j in range(N - t - tau + 1) for i in range(I)
    ) + gp.quicksum(
        data['o'][j] * y[tau][j]
        for tau in range(1, N - t) for j in range(H - t - tau + 1)
    )
    subproblem.setObjective((GAMMA / M) * future_cost, GRB.MINIMIZE)

    # --- Subproblem Constraints ---
    u = {tau: subproblem.addVars(N - t - tau + 2, name=f"u_{t + tau}") for tau in range(1, N - t + 1)}
    u_bar = {tau: subproblem.addVars(N - t - tau + 1, name=f"u_bar_{t + tau}") for tau in range(1, N - t)}

    # Linking constraint (1b): Initialize the state from master's decision
    # We add this as a constraint to get its dual value for the cut.
    linking_constrs = subproblem.addConstrs((u[1][j] == u_t1[j] for j in range(N - t)), name="linking")

    # All future constraints from tau=1 to N-t
    for tau in range(1, N - t):
        # Post-decision state (1e)
        for j in range(N - t - tau + 1):
            resource_sum = gp.quicksum(x[tau][k, i] * data['r'][j - k, i]
                                       for i in range(I)
                                       for k in range(max(j - L + 1, 0), j + 1))
            overtime_term = y[tau][j] if j <= H - t - tau else 0
            subproblem.addConstr(u_bar[tau][j] == u[tau][j] + resource_sum - overtime_term)

        # State transition (1f)
        subproblem.addConstrs((u[tau + 1][j] == u_bar[tau][j + 1] for j in range(N - t - tau)))

        # Capacity (1g)
        subproblem.addConstrs((u_bar[tau][j] <= CAPACITY for j in range(N - t - tau + 1)))

        # Demand fulfillment (1h)
        demand = data['delta_future'][scenario_omega][tau - 1]
        subproblem.addConstrs((x[tau].sum('*', i) == demand[i] for i in range(I)))

    # --- Solve and Get Cut Information ---
    # To generate a valid Benders cut, we need duals from the LP relaxation.
    lp_subproblem = subproblem.relax()
    lp_subproblem.optimize()

    # Case 1: Subproblem is feasible
    if lp_subproblem.Status == GRB.OPTIMAL:
        # We still need the true integer objective for the upper bound calculation
        subproblem.optimize()
        if subproblem.Status != GRB.OPTIMAL:
            # This can happen in rare cases. Treat as infeasible for simplicity.
            return {'status': 'infeasible', 'farkas_duals': None, 'obj': None}
        duals = [linking_constrs[j].Pi for j in range(N - t)]
        return {'status': 'optimal', 'duals': duals, 'obj': subproblem.ObjVal}

    # Case 2: Subproblem is infeasible
    elif lp_subproblem.Status == GRB.INFEASIBLE:
        # Generate a feasibility cut using Farkas duals
        farkas_duals = [linking_constrs[j].FarkasDual for j in range(N - t)]
        return {'status': 'infeasible', 'farkas_duals': farkas_duals, 'obj': None}
    else:
        # Other statuses (e.g., unbounded) indicate an issue
        raise Exception(f"Subproblem {scenario_omega} has unhandled status: {lp_subproblem.Status}")


# --- 4. The Main Benders Loop ---
def run_benders_decomposition():
    """Manages the iterative process of solving master and subproblems."""
    data = generate_problem_data()
    master = create_master_problem(data)

    lower_bound = -GRB.INFINITY
    upper_bound = GRB.INFINITY

    print("--- Starting Benders Decomposition ---")
    for i in range(50):  # Max iterations
        print(f"\nIteration {i + 1}:")

        # Solve the master problem
        master.setParam('OutputFlag', 0)
        master.optimize()

        if master.Status != GRB.OPTIMAL:
            print("Master problem could not be solved to optimality. Stopping.")
            break

        lower_bound = master.ObjVal
        master_solution = {
            'x_t': master.getAttr('X', master._vars['x_t']),
            'y_t': master.getAttr('X', master._vars['y_t']),
            'theta': master._vars['theta'].X,
            'u_bar_t': master.getAttr('X', master._u_bar_t)
        }

        print(f"  Master solved. Lower Bound = {lower_bound:.2f}")

        # Solve subproblems for all scenarios
        total_subproblem_obj = 0
        cut_expression = 0
        infeasibility_found = False

        for m in range(M):
            result = solve_subproblem(data, master_solution, m)

            if result['status'] == 'infeasible':
                # Add a feasibility cut
                print(f"  Scenario {m} is infeasible. Adding a feasibility cut.")
                duals = result['farkas_duals']
                u_bar_t = master._u_bar_t
                # The cut forces the master to a region where this infeasibility is avoided
                cut = gp.quicksum(duals[j] * u_bar_t[j + 1] for j in range(N - t)) <= gp.quicksum(
                    duals[j] * master_solution['u_bar_t'][j + 1] for j in range(N - t)) - 1e-4
                master.addConstr(cut)
                infeasibility_found = True
                break  # Go to the next master iteration immediately

            total_subproblem_obj += result['obj']
            duals = result['duals']
            u_bar_t = master._u_bar_t

            # Build the optimality cut expression
            # theta >= E[ duals * (u_bar_t - u_bar_t_val) + subproblem_obj_val ]
            cut_expression += gp.quicksum(
                duals[j] * (u_bar_t[j + 1] - master_solution['u_bar_t'][j + 1]) for j in range(N - t)) + result['obj']

        if infeasibility_found:
            continue

        # Update upper bound
        immediate_cost_val = master.ObjVal - master_solution['theta']
        current_upper_bound = immediate_cost_val + total_subproblem_obj
        upper_bound = min(upper_bound, current_upper_bound)

        print(f"  Subproblems solved. Expected Future Cost = {total_subproblem_obj:.2f}")
        print(f"  Current Upper Bound = {current_upper_bound:.2f}")
        print(f"  GAP = {((upper_bound - lower_bound) / (abs(upper_bound) + 1e-6)) * 100:.2f}%")

        # Check for convergence
        if upper_bound - lower_bound <= 1e-4:
            print("\n--- Benders has converged! ---")
            break

        # Add optimality cut
        master.addConstr(master._vars['theta'] >= cut_expression)

    print("\n--- Benders Final Optimal Solution ---")
    print(f"Optimal First-Stage Cost: ${master.ObjVal - master._vars['theta'].X:,.2f}")
    print(f"Expected Total Cost (Upper Bound): ${upper_bound:,.2f}")
    return data  # Return data for the exact solver


# --- 5. The Exact Solution (Extensive Form) ---
def solve_extensive_form(data):
    """Solves the problem as a single, large ILP (deterministic equivalent)."""
    model = gp.Model("ExtensiveForm")

    # --- First-stage variables ---
    x_t = model.addVars(N - t + 1, I, vtype=GRB.INTEGER, name="x_t")
    y_t = model.addVars(H - t + 1, vtype=GRB.INTEGER, name="y_t")

    # --- Second-stage variables for ALL scenarios ---
    x = model.addVars(M, N - t, N - t + 1, I, vtype=GRB.INTEGER, name="x_future")
    y = model.addVars(M, N - t, H - t + 1, vtype=GRB.INTEGER, name="y_future")
    u = model.addVars(M, N - t + 1, N - t + 2, name="u_future")
    u_bar = model.addVars(M, N - t, N - t + 1, name="u_bar_future")

    # --- Objective Function ---
    immediate_cost = gp.quicksum(data['c'][j, i] * x_t[j, i] for j in range(N - t + 1) for i in range(I)) + \
                     gp.quicksum(data['o'][j] * y_t[j] for j in range(H - t + 1))

    future_cost = gp.quicksum(
        (GAMMA / M) * (
                data['c'][j, i] * x[m, tau, j, i] +
                data['o'][j] * y[m, tau, j]
        )
        for m in range(M)
        for tau in range(1, N - t)
        for j in range(N - t - tau + 1) for i in range(I)
        if j <= H - t - tau
    )
    model.setObjective(immediate_cost + future_cost, GRB.MINIMIZE)

    # --- First-stage constraints ---
    model.addConstrs((x_t.sum('*', i) == data['delta_t'][i] for i in range(I)), name="demand_t")
    u_bar_t = model.addVars(N - t + 1, name="u_bar_t")
    for j in range(N - t + 1):
        resource_sum = gp.quicksum(
            x_t[k, i] * data['r'][j - k, i] for i in range(I) for k in range(max(j - L + 1, 0), j + 1))
        overtime_term = y_t[j] if j <= H - t else 0
        model.addConstr(u_bar_t[j] == data['u_t'][j] + resource_sum - overtime_term)
    model.addConstrs((u_bar_t[j] <= CAPACITY for j in range(N - t + 1)), name="capacity_t")

    # --- Linking and Second-stage constraints for EACH scenario ---
    for m in range(M):
        # Linking constraint (1b)
        model.addConstrs((u[m, 1, j] == u_bar_t[j + 1] for j in range(N - t)), name=f"linking_{m}")

        for tau in range(1, N - t):
            # Post-decision state (1e)
            for j in range(N - t - tau + 1):
                resource_sum = gp.quicksum(
                    x[m, tau, k, i] * data['r'][j - k, i] for i in range(I) for k in range(max(j - L + 1, 0), j + 1))
                overtime_term = y[m, tau, j] if j <= H - t - tau else 0
                model.addConstr(u_bar[m, tau, j] == u[m, tau, j] + resource_sum - overtime_term)

            # State transition (1f)
            model.addConstrs((u[m, tau + 1, j] == u_bar[m, tau, j + 1] for j in range(N - t - tau)))
            # Capacity (1g)
            model.addConstrs((u_bar[m, tau, j] <= CAPACITY for j in range(N - t - tau + 1)))
            # Demand fulfillment (1h)
            demand = data['delta_future'][m][tau - 1]
            model.addConstrs(
                (gp.quicksum(x[m, tau, j, i] for j in range(N - t - tau + 1)) == demand[i] for i in range(I)))

    # --- Solve ---
    print("\n--- Solving with Exact (Extensive Form) Method ---")
    model.setParam('OutputFlag', 1)
    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print("\n--- Exact Form Final Optimal Solution ---")
        print(f"Expected Total Cost: ${model.ObjVal:,.2f}")
    else:
        print("Exact form could not be solved to optimality.")


if __name__ == "__main__":
    #problem_data = run_benders_decomposition()
    data = generate_problem_data()
    solve_extensive_form(data)