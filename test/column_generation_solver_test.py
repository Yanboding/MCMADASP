import random
import numpy as np
from scipy.stats import poisson
from itertools import product

import gurobipy as gp
from gurobipy import GRB

from utils import ColumnGenerationSolver


def compute_b_matrix(N, T, f, gamma):
    """
    Compute the matrix b of shape (N, I) where each entry b(n, i) is defined as:

      b(n, i) = sum_{k=1}^{n - T[i]} gamma^(k-1) * f[i]   if n > T[i],
                0                                          otherwise.

    For gamma != 1, this sum can be computed in closed form as:

      b(n, i) = f[i] * (1 - gamma^(n-T[i])) / (1 - gamma)

    and for gamma == 1, b(n, i) = f[i] * (n - T[i]).

    Parameters:
      N     : int
              Total number of days (rows of the matrix).
      T     : array-like of length I
              Threshold day for each priority.
      f     : array-like of length I
              Waiting penalty for each priority.
      gamma : float
              Discount factor.

    Returns:
      b     : NumPy array of shape (N, I)
              The computed cost matrix b.
    """
    # Convert T and f to NumPy arrays for vectorized operations.
    T = np.array(T)  # shape (I,)
    f = np.array(f)  # shape (I,)

    # Create an array of day indices from 1 to N.
    days = np.arange(1, N + 1)  # shape (N,)

    # For each day n and each priority i, compute the difference n - T[i]
    # This will be used to determine if the threshold has been exceeded.
    diff = days[:, None] - T[None, :]  # shape (N, I)

    # If gamma equals 1, then the sum becomes a linear count.
    if gamma == 1:
        b = np.where(diff > 0, f[None, :] * diff, 0)
    else:
        # Compute the closed-form geometric series sum:
        # sum_{j=0}^{m-1} gamma^j = (1 - gamma^m) / (1 - gamma),
        # where m = diff (i.e., n - T[i])
        b = np.where(diff > 0, f[None, :] * (1 - gamma ** diff) / (1 - gamma), 0)

    return b.T

def truncated_poisson_means(arrival_rates):
    """
    Calculate the truncated Poisson mean for each arrival rate in a list.
    Each Poisson distribution is truncated at 3 * lambda.

    Parameters:
        arrival_rates (list or array): A list of λ values.

    Returns:
        list: A list of truncated means corresponding to each λ.
    """
    truncated_means = []
    truncated_pmf_list = []
    for lam in arrival_rates:
        # Define the truncation value (three times the mean)
        max_val = int(3 * lam)
        # Create an array of x values from 0 to max_val (inclusive)
        x = np.arange(0, max_val + 1)
        # Compute the Poisson PMF for these values
        pmf = poisson.pmf(x, lam)
        # Renormalize the PMF so it sums to 1 over the truncated range
        truncated_pmf = pmf / pmf.sum()
        # Calculate the truncated mean and add it to the list
        truncated_pmf_list.append(truncated_pmf)
        truncated_means.append(np.sum(x * truncated_pmf))
    return truncated_pmf_list, truncated_means


def compute_custom_distributions(N, C1, arrival_rates, alpha_func_X=None, alpha_func_Y=None):
    """
    Compute custom ranges and probability distributions.

    Parameters:
        N (int): Number of rows for the fixed range and alpha_X.
        C1 (int): Upper bound for the fixed range [0, C1] used for range_x and alpha_X.
        arrival_rates (list or array): List of arrival rate values. For each λ, the support for range_Y and alpha_Y is [0, int(3*λ)].
        alpha_func_X (callable, optional): Function to generate weights for alpha_X.
            It should accept an array of x values and return weights (default: uniform over the array).
        alpha_func_Y (callable, optional): Function to generate weights for alpha_Y.
            It should accept (lam, x_array) and return weights (default: uniform over the array).

    Returns:
        tuple: (range_x, range_Y, alpha_X, alpha_Y)
            - range_x: NumPy array of shape (N, C1+1) where each row is np.arange(0, C1+1).
            - range_Y: List of NumPy arrays, each corresponding to np.arange(0, int(3*λ)+1) for each arrival rate λ.
            - alpha_X: NumPy array of shape (N, C1+1) with a probability distribution over [0, C1].
            - alpha_Y: List of NumPy arrays with probability distributions over [0, int(3*λ)] for each λ.
    """
    # Create the fixed range_x with shape (N, C1+1)
    x_fixed = np.arange(0, C1 + 1)
    range_x = np.tile(x_fixed, (N, 1))

    # Default function for alpha_X: uniform distribution over x_fixed
    if alpha_func_X is None:
        def alpha_func_X(x_array):
            return np.ones_like(x_array, dtype=float) / len(x_array)

    # Default function for alpha_Y: uniform distribution over the provided x_array for each λ
    if alpha_func_Y is None:
        def alpha_func_Y(lam, x_array):
            return np.ones_like(x_array, dtype=float) / len(x_array)

    # Compute alpha_X: same support for all rows (each row gets the same distribution)
    alpha_X = np.zeros((N, C1 + 1))
    weights_X = alpha_func_X(x_fixed)
    weights_X = weights_X / np.sum(weights_X)  # ensure normalization
    for i in range(N):
        alpha_X[i, :] = weights_X

    # Compute range_Y and alpha_Y for each arrival rate (each can be different)
    range_Y = []
    alpha_Y = []
    for lam in arrival_rates:
        # Support for Y: [0, 1, ..., int(3*lam)]
        max_val_Y = int(3 * lam)
        x_y = np.arange(0, max_val_Y + 1)
        range_Y.append(x_y)

        weights_Y = alpha_func_Y(lam, x_y)
        weights_Y = weights_Y / np.sum(weights_Y)  # ensure normalization
        alpha_Y.append(weights_Y)

    return range_x, range_Y, alpha_X, alpha_Y
def column_generation(params):
    """
    Performs column generation for the ALP.  All helper routines are nested inside.

    params must contain:
        'gamma'            : discount factor
        'N'                : number of flow constraints
        'I'                : number of resource constraints
        'E_X (alpha)'      : list of length N (flow RHS)
        'E_Y (alpha)'      : list of length I (resource RHS)
        'b'                : booking cost matrix (I x N)
        'f'                : waiting cost vector (length I)
        'd'                : diversion cost vector (length I)
        'E_Y'              : arrival rates (length I)  # only used in generate_initial_feasible_columns
        'C1'               : capacity parameter for x + sum(a)
        'C2'               : capacity parameter for sum(z)
        'max_arrival'      : upper bound on y
    """

    # Extract parameters once
    gamma       = params['gamma']
    N           = params['N']
    I           = params['I']
    E_X_alpha   = params['E_X (alpha)']
    E_Y_alpha   = params['E_Y (alpha)']
    b_mat       = params['b']
    f_vec       = params['f']
    d_vec       = params['d']
    E_Y         = params['E_Y']
    C1          = params['C1']
    C2          = params['C2']
    max_arrival = params['max_arrival']

    # -----------------------------
    # (1) Compute coefficients for a given candidate tuple
    # -----------------------------
    def compute_coeffs(candidate):
        x, y, a, z = candidate
        # coeff[0] = 1 - gamma
        coeffs = {0: 1.0 - gamma}

        # Flow constraints (indices 1..N)
        for n in range(N):
            if n == N - 1:
                coeffs[n + 1] = x[n]
            else:
                next_sum = gamma * x[n + 1] + gamma * sum(a[i][n + 1] for i in range(I))
                coeffs[n + 1] = x[n] - next_sum

        # Resource constraints (indices N+1 .. N+I)
        for i in range(I):
            total_a_i = sum(a[i][n] for n in range(N))
            coeffs[N + 1 + i] = (1.0 - gamma) * y[i] + gamma * (total_a_i + z[i] - E_Y[i])

        return coeffs

    # -----------------------------
    # (2) Compute total cost for a given candidate tuple
    # -----------------------------
    def compute_cost(candidate):
        x, y, a, z = candidate
        # Booking cost
        booking = sum(b_mat[i][n] * a[i][n] for i in range(I) for n in range(N))
        # Diversion cost
        diversion = sum(d_vec[i] * z[i] for i in range(I))
        # Waiting cost
        booked = [sum(a[i]) for i in range(I)]
        waiting = sum(f_vec[i] * (y[i] - booked[i] - z[i]) for i in range(I))
        return booking + diversion + waiting

    # -----------------------------
    # (3) Generate one candidate by solving the pricing subproblem (continuous LP)
    # -----------------------------
    def generate_candidate(dual_values):
        # Extract duals
        W0 = dual_values[0]
        V  = dual_values[1 : N + 1]
        W  = dual_values[N + 1 : N + 1 + I]

        # Build M_a (I x N) for a[i,n]
        M_a = np.empty((I, N))
        for i in range(I):
            for n in range(N):
                V_prev = V[n - 1] if n > 0 else 0.0
                M_a[i, n] = b_mat[i][n] + gamma * V_prev - f_vec[i] - gamma * W[i]

        # Coefficients for x[n]
        v_x = np.array([
            (gamma * (V[n - 1] if n > 0 else 0.0) - V[n])
            for n in range(N)
        ])

        # Coefficients for z[i], y[i]
        c_z = np.array([d_vec[i] - f_vec[i] - gamma * W[i] for i in range(I)])
        c_y = np.array([f_vec[i] + (gamma - 1.0) * W[i]   for i in range(I)])

        # Constant term in objective
        constant = gamma * np.dot(W, E_Y) - (1.0 - gamma) * W0

        # Build pricing model
        m = gp.Model("PricingSubproblem")
        m.setParam('OutputFlag',      0)
        m.setParam('FeasibilityTol',  1e-9)
        m.setParam('OptimalityTol',   1e-9)
        m.setParam('IntFeasTol',      1e-9)
        m.setParam('NumericFocus',    3)
        m.setParam('Method',          2)
        m.setParam('Presolve',        0)

        # Decision variables
        x_vars = m.addVars(N, vtype=GRB.CONTINUOUS, name="x")
        y_vars = m.addVars(I, ub=max_arrival, vtype=GRB.CONTINUOUS, name="y")
        z_vars = m.addVars(I, vtype=GRB.CONTINUOUS, name="z")
        a_vars = m.addVars(I, N, vtype=GRB.CONTINUOUS, name="a")

        # Constraints: x[n] + sum_i a[i,n] ≤ C1
        for n in range(N):
            m.addConstr(x_vars[n] + gp.quicksum(a_vars[i, n] for i in range(I)) <= C1,
                        name=f"constr_x_a_{n}")

        # Constraint: sum_i z[i] ≤ C2
        m.addConstr(gp.quicksum(z_vars[i] for i in range(I)) <= C2,
                    name="constr_sum_z")

        # For each i: sum_n a[i,n] + z[i] ≤ y[i]
        for i in range(I):
            m.addConstr(gp.quicksum(a_vars[i, n] for n in range(N)) + z_vars[i] <= y_vars[i],
                        name=f"constr_a_z_y_{i}")

        # Build objective expression
        obj_expr = gp.LinExpr()
        # a[i,n] terms
        a_list  = [a_vars[i, n] for i in range(I) for n in range(N)]
        coeff_a = M_a.flatten()
        obj_expr.add(gp.quicksum(coeff_a[k] * a_list[k] for k in range(I * N)))

        # x terms
        obj_expr.add(gp.quicksum(v_x[n] * x_vars[n] for n in range(N)))
        # z terms
        obj_expr.add(gp.quicksum(c_z[i] * z_vars[i] for i in range(I)))
        # y terms
        obj_expr.add(gp.quicksum(c_y[i] * y_vars[i] for i in range(I)))
        # constant
        obj_expr.addConstant(constant)

        m.setObjective(obj_expr, GRB.MINIMIZE)
        m.optimize()

        status = m.status
        if status == GRB.OPTIMAL:
            x_sol = [x_vars[n].x for n in range(N)]
            y_sol = [y_vars[i].x for i in range(I)]
            a_sol = [[a_vars[i, n].x for n in range(N)] for i in range(I)]
            z_sol = [z_vars[i].x for i in range(I)]
            return (x_sol, y_sol, a_sol, z_sol), m.objVal
        elif status == GRB.INFEASIBLE:
            print("Pricing subproblem infeasible.")
        elif status == GRB.UNBOUNDED:
            print("Pricing subproblem unbounded.")
        else:
            print("Pricing returned status", status)

        return None, None

    # -----------------------------
    # (4) Pricing subproblem wrapper: check reduced cost
    # -----------------------------
    def pricing_subproblem(dual_values):
        tol = 1e-20
        candidate, obj_val = generate_candidate(dual_values)
        if candidate is None:
            return None

        candidate_cost   = compute_cost(candidate)
        candidate_coeffs = compute_coeffs(candidate)
        # reduced cost = cost − ∑ dual[j] * coeffs[j]
        rc = candidate_cost
        for j, coeff in candidate_coeffs.items():
            rc -= dual_values[j] * coeff
        print('rc:', rc, 'obj_val:',obj_val)
        # Debug prints
        # print(candidate)
        # print(f"Candidate rc = {rc:.3e}, pricing-obj = {obj_val:.3e}")

        if rc < -tol:
            return candidate, candidate_cost, candidate_coeffs, rc
        return None

    # -----------------------------
    # (5) Generate an initial feasible set of columns
    # -----------------------------
    def generate_initial_feasible_columns():
        # Dummy candidate: x = [C1] * N, y = [max_arrival] * I, a = zeros, z = E_Y
        x0 = [C1 for _ in range(N)]
        y0 = [float(max_arrival) for _ in range(I)]
        a0 = [[0.0] * N for _ in range(I)]
        z0 = list(E_Y)
        return [(x0, y0, a0, z0)]

    # -----------------------------
    # (6) Create the Restricted Master Problem (RMP)
    # -----------------------------
    def create_rmp():
        model = gp.Model("MasterRMP")
        model.ModelSense = GRB.MINIMIZE
        model.setParam('OutputFlag',      0)
        model.setParam('FeasibilityTol',  1e-9)
        model.setParam('OptimalityTol',   1e-9)
        model.setParam('IntFeasTol',      1e-9)
        model.setParam('NumericFocus',    3)
        model.setParam('Method',          2)
        model.setParam('Presolve',        0)

        constraints = []
        # Row 0: normalization (1 - gamma) * ∑ π_c = 1
        constraints.append(model.addConstr(gp.LinExpr() == 1.0, name="constr_norm"))

        # Rows 1..N: flow constraints ≥ E_X_alpha[n]
        for n in range(N):
            constraints.append(
                model.addConstr(gp.LinExpr() >= E_X_alpha[n], name=f"constr_flow_{n}")
            )

        # Rows N+1..N+I: resource constraints ≥ E_Y_alpha[i]
        for i in range(I):
            constraints.append(
                model.addConstr(gp.LinExpr() >= E_Y_alpha[i], name=f"constr_resource_{i}")
            )

        model.update()
        return model, constraints

    # -----------------------------
    # (7) Add a column (variable) to the RMP
    # -----------------------------
    def add_column(model, constraints, candidate, col_name):
        candidate_coeffs = compute_coeffs(candidate)
        candidate_cost   = compute_cost(candidate)

        obj_coeff = 1e8 if col_name.startswith("init_") else float(candidate_cost)
        new_col = gp.Column()
        for j, coeff in candidate_coeffs.items():
            new_col.addTerms([float(coeff)], [constraints[j]])

        model.addVar(obj=obj_coeff, column=new_col, name=col_name)
        model.update()

    # -----------------------------
    # (8) Solve the RMP and return duals
    # -----------------------------
    def solve_rmp(model):
        model.optimize()
        if model.status != GRB.OPTIMAL:
            raise RuntimeError("RMP did not solve optimally!")
        return [c.Pi for c in model.getConstrs()]

    # -----------------------------
    # (9) Main column-generation loop
    # -----------------------------
    model, constraints = create_rmp()
    col_candidates = {}

    # (9a) Add initial feasible columns
    for idx, cand in enumerate(generate_initial_feasible_columns()):
        name = f"init_{idx + 1}"
        add_column(model, constraints, cand, name)
        col_candidates[name] = cand

    iteration = 0
    max_iter = 200
    while iteration < max_iter:
        iteration += 1
        # (9b) Solve RMP → get duals
        dual_vals = solve_rmp(model)

        # (9c) Pricing
        pr = pricing_subproblem(dual_vals)
        if pr is None:
            break

        candidate, cost_c, coeffs_c, rc = pr
        col_name = f"x_{iteration}"
        add_column(model, constraints, candidate, col_name)
        col_candidates[col_name] = candidate

    # (9d) Final solve
    model.optimize()

    # Extract final dual values by constraint name
    final_duals = {c.ConstrName: c.Pi for c in constraints}

    # Extract each column variable’s value and objective coefficient
    var_x    = {v.varName: v.x   for v in model.getVars() if v.varName in col_candidates}
    var_coefs= {v.varName: v.Obj for v in model.getVars() if v.varName in col_candidates}

    # Return: (objective, duals, primal column weights, column costs, column dictionary)
    return model.objVal, final_duals, var_x, var_coefs, col_candidates


def create_rmp():
    # Extract parameters once
    gamma = params['gamma']
    N = params['N']
    I = params['I']
    E_X_alpha = params['E_X (alpha)']
    E_Y_alpha = params['E_Y (alpha)']
    b_mat = params['b']
    f_vec = params['f']
    d_vec = params['d']
    E_Y = params['E_Y']
    C1 = params['C1']
    C2 = params['C2']
    max_arrival = params['max_arrival']
    model = gp.Model("MasterRMP")
    model.ModelSense = GRB.MINIMIZE
    model.setParam('OutputFlag',      0)
    model.setParam('FeasibilityTol',  1e-9)
    model.setParam('OptimalityTol',   1e-9)
    model.setParam('IntFeasTol',      1e-9)
    model.setParam('NumericFocus',    3)
    model.setParam('Method',          2)
    model.setParam('Presolve',        0)

    constraints = []
    # Row 0: normalization (1 - gamma) * ∑ π_c = 1
    constraints.append(model.addConstr(gp.LinExpr() == 1.0, name="constr_norm"))

    # Rows 1..N: flow constraints ≥ E_X_alpha[n]
    for n in range(N):
        constraints.append(
            model.addConstr(gp.LinExpr() >= E_X_alpha[n], name=f"constr_flow_{n}")
        )

    # Rows N+1..N+I: resource constraints ≥ E_Y_alpha[i]
    for i in range(I):
        constraints.append(
            model.addConstr(gp.LinExpr() >= E_Y_alpha[i], name=f"constr_resource_{i}")
        )

    model.update()
    return model

def generate_initial_feasible_columns():
    max_arrival = params['max_arrival']
    # Dummy candidate: x = [C1] * N, y = [max_arrival] * I, a = zeros, z = E_Y
    x0 = [C1 for _ in range(N)]
    y0 = [float(max_arrival) for _ in range(I)]
    a0 = [[0.0] * N for _ in range(I)]
    z0 = list(E_Y)
    return [(x0, y0, a0, z0)]

def set_val(vars, vals):
    for i in vars:
        vars[i].lb = vals[i]
        vars[i].ub = vals[i]

def generate_candidate(dual_values, candidate=None):
    if candidate != None:
        x_can, y_can, a_can, z_can = candidate
    gamma = params['gamma']
    N = params['N']
    I = params['I']
    b_mat = params['b']
    f_vec = params['f']
    d_vec = params['d']
    E_Y = params['E_Y']
    C1 = params['C1']
    C2 = params['C2']
    max_arrival = params['max_arrival']

    # Extract duals
    W0 = dual_values[0]
    V  = dual_values[1 : N + 1]
    W  = dual_values[N + 1 : N + 1 + I]

    # Build M_a (I x N) for a[i,n]
    M_a = np.empty((I, N))
    for i in range(I):
        for n in range(N):
            V_prev = V[n - 1] if n > 0 else 0.0
            M_a[i, n] = b_mat[i][n] + gamma * V_prev - f_vec[i] - gamma * W[i]

    # Coefficients for x[n]
    v_x = np.array([
        (gamma * (V[n - 1] if n > 0 else 0.0) - V[n])
        for n in range(N)
    ])

    # Coefficients for z[i], y[i]
    c_z = np.array([d_vec[i] - f_vec[i] - gamma * W[i] for i in range(I)])
    c_y = np.array([f_vec[i] + (gamma - 1.0) * W[i]   for i in range(I)])

    # Constant term in objective
    constant = gamma * np.dot(W, E_Y) - (1.0 - gamma) * W0

    # Build pricing model
    m = gp.Model("PricingSubproblem")
    m.setParam('OutputFlag',      0)
    m.setParam('FeasibilityTol',  1e-9)
    m.setParam('OptimalityTol',   1e-9)
    m.setParam('IntFeasTol',      1e-9)
    m.setParam('NumericFocus',    3)
    m.setParam('Method',          2)
    m.setParam('Presolve',        0)

    # Decision variables
    x_vars = m.addVars(N, vtype=GRB.CONTINUOUS, name="x")
    y_vars = m.addVars(I, ub=max_arrival, vtype=GRB.CONTINUOUS, name="y")
    z_vars = m.addVars(I, vtype=GRB.CONTINUOUS, name="z")
    a_vars = m.addVars(I, N, vtype=GRB.CONTINUOUS, name="a")
    if candidate != None:
        set_val(x_vars, x_can)
        set_val(y_vars, y_can)
        set_val(z_vars, z_can)
        print(a_vars)
        for i in range(I):
            for t in range(N):
                a_vars[i,t].lb = a_can[i][t]
                a_vars[i, t].ub = a_can[i][t]
    # Constraints: x[n] + sum_i a[i,n] ≤ C1
    for n in range(N):
        m.addConstr(x_vars[n] + gp.quicksum(a_vars[i, n] for i in range(I)) <= C1,
                    name=f"constr_x_a_{n}")

    # Constraint: sum_i z[i] ≤ C2
    m.addConstr(gp.quicksum(z_vars[i] for i in range(I)) <= C2,
                name="constr_sum_z")

    # For each i: sum_n a[i,n] + z[i] ≤ y[i]
    for i in range(I):
        m.addConstr(gp.quicksum(a_vars[i, n] for n in range(N)) + z_vars[i] <= y_vars[i],
                    name=f"constr_a_z_y_{i}")

    # Build objective expression
    obj_expr = gp.LinExpr()
    # a[i,n] terms
    a_list  = [a_vars[i, n] for i in range(I) for n in range(N)]
    coeff_a = M_a.flatten()
    obj_expr.add(gp.quicksum(coeff_a[k] * a_list[k] for k in range(I * N)))

    # x terms
    obj_expr.add(gp.quicksum(v_x[n] * x_vars[n] for n in range(N)))
    # z terms
    obj_expr.add(gp.quicksum(c_z[i] * z_vars[i] for i in range(I)))
    # y terms
    obj_expr.add(gp.quicksum(c_y[i] * y_vars[i] for i in range(I)))
    # constant
    obj_expr.addConstant(constant)

    m.setObjective(obj_expr, GRB.MINIMIZE)
    m.optimize()

    status = m.status
    if status == GRB.OPTIMAL:
        x_sol = [x_vars[n].x for n in range(N)]
        y_sol = [y_vars[i].x for i in range(I)]
        a_sol = [[a_vars[i, n].x for n in range(N)] for i in range(I)]
        z_sol = [z_vars[i].x for i in range(I)]
        return [((x_sol, y_sol, a_sol, z_sol), m.objVal)]
    elif status == GRB.INFEASIBLE:
        print("Pricing subproblem infeasible.")
    elif status == GRB.UNBOUNDED:
        print("Pricing subproblem unbounded.")
    else:
        print("Pricing returned status", status)

    return [(None, None)]

def compute_coeffs(candidate):
    x, y, a, z = candidate
    # coeff[0] = 1 - gamma
    coeffs = [0]*(N+I+1)
    coeffs[0] = 1 - gamma
    # Flow constraints (indices 1..N)
    for n in range(N):
        if n == N - 1:
            coeffs[n + 1] = x[n]
        else:
            next_sum = gamma * x[n + 1] + gamma * sum(a[i][n + 1] for i in range(I))
            coeffs[n + 1] = x[n] - next_sum

    # Resource constraints (indices N+1 .. N+I)
    for i in range(I):
        total_a_i = sum(a[i][n] for n in range(N))
        coeffs[N + 1 + i] = (1.0 - gamma) * y[i] + gamma * (total_a_i + z[i] - E_Y[i])

    return coeffs

# -----------------------------
# (2) Compute total cost for a given candidate tuple
# -----------------------------
def compute_cost(candidate):
    N = params['N']
    I = params['I']
    b_mat = params['b']
    f_vec = params['f']
    d_vec = params['d']
    x, y, a, z = candidate
    # Booking cost
    booking = sum(b_mat[i][n] * a[i][n] for i in range(I) for n in range(N))
    # Diversion cost
    diversion = sum(d_vec[i] * z[i] for i in range(I))
    # Waiting cost
    booked = [sum(a[i]) for i in range(I)]
    waiting = sum(f_vec[i] * (y[i] - booked[i] - z[i]) for i in range(I))
    return booking + diversion + waiting

if __name__ == '__main__':

    # Parameters for demonstration.
    random.seed(42)
    np.random.seed(42)
    N = 3  # number of days
    I = 1  # number of priority classes
    C1 = 2  # daily base capacity
    C2 = 1  # maximum diverted patients per day
    arrival_rates = np.array([3])  # Arrival rate of the priority i patient
    Q_list = arrival_rates * 3  # The Poisson distribution is truncated at three times the mean

    gamma = 0.99  # discount_factor
    T_target = [1]  # Wait-time target of the priority i patient
    f = [20]  # late-booking cost of the priority i patient
    d = [100]  # Diversting costss
    b = compute_b_matrix(N, T_target, f, gamma)  # Booking costs

    truncated_pmf_list, E_Y = truncated_poisson_means(arrival_rates)

    alpha_0_X = [np.full(C1 + 1, 1 / (C1 + 1)) for _ in
                 range(N)]  # Initial distribution of x. It should be also sampled from this for simulation
    # alpha_0_Y = truncated_pmf_list # Initial distribution of y. It should be also sampled from this for simulation

    uniform_list = []
    for m in arrival_rates:
        L = 3 * m + 1  # desired vector length
        v = np.ones(L, dtype=int)  # start all-zeros
        v = v / L  # make it one-hot at index 0
        uniform_list.append(v)
    alpha_0_Y = uniform_list

    range_X, range_Y, alpha_X_1, alpha_Y_1 = compute_custom_distributions(N, C1, arrival_rates, alpha_func_X=None,
                                                                          alpha_func_Y=None)

    # alpha_0_X[-1][:] = 0
    # alpha_0_X[-1][0] = 1

    E_X_alpha_1 = np.sum(range_X * alpha_0_X, axis=1)
    # E_X_alpha_1 += 10
    E_Y_alpha_1 = [np.sum(range_Y[i] * alpha_0_Y[i]) for i in range(len(range_Y))]

    nu_0_X = [np.full(C1 + 1, 1 / (C1 + 1)) for _ in range(N)]
    nu_0_Y = [np.ones(3 * m + 1) / (3 * m + 1) for m in arrival_rates]

    params = {
        'gamma': gamma,
        'N': N,  # number of flow constraints
        'I': I,  # number of resource constraints
        'C1': C1,  # daily base capacity
        'C2': C2,  # maximum diverted patients per day
        'max_arrival': 3 * arrival_rates,
        'Q': Q_list,
        'E_Y': E_Y,  # arrival rate of paitent i (E_Y)
        'trun_pmf': truncated_pmf_list,
        'E_X (alpha)': E_X_alpha_1,  # Expected number of occupied slots s
        'E_Y (alpha)': E_Y_alpha_1,  # alpha_weighted
        'T': T_target,
        'b': b,
        'f': f,
        'd': d,
        'nu_0_X': nu_0_X,
        'nu_0_Y': nu_0_Y,
    }

    all_candidates = list(enumerate_candidates(params))

    primal = solve_dual_alp_from_tuples(all_candidates, params)

    theoritical_W0 = compute_W0(d, gamma, I, T_target, E_Y, C1)

    dual_complete = solve_primal_alp_from_tuples(all_candidates, params)

    cg_solver = ColumnGenerationSolver(master_builder=create_rmp,
                                       pricing_callback=generate_candidate,
                                       initial_columns=generate_initial_feasible_columns(),
                                       get_constr_coefficients=compute_coeffs,
                                       get_obj_coefficient=compute_cost)

    cg_solver.solve()

    print(cg_solver.candidates)
    duals = [41935.11637221304, 0.0, 0.0, 0.0, 0.0]
    candidate = ([0, 0, 0], [0], [[0.0, 0.0, 0.0]], [0])
    print(generate_candidate(duals, candidate))
