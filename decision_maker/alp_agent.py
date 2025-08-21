# -*- coding: utf-8 -*-
"""
Multiclass column generation with a persistent Phase‑I driver:
- Phase‑I (Appendix A) kept PERSISTENT & INCREMENTAL (adds one X_j per column via gp.Column)
- Pricing (eq. (14)) with MVar arrays (u, v, w, x, y)
- Restricted primal (eq. (12)) persistent; one inequality per column

Implements:
  • State/action feasibility (1)–(3), transitions (4)–(7), cost (8)
  • Affine ALP (11), dual master (13), pricing (14), Phase‑I (Appendix A)

Reference: Sauré, Patrick, Tyldesley, Puterman (2012), EJOR 223:573–584.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from scipy.stats import uniform


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _ensure_objval(model: gp.Model, name: str = ""):
    """Ensure model has a valid ObjVal, retry without dual reductions if INF_OR_UNBD."""
    st = model.Status
    if st != GRB.OPTIMAL:
        model.write(f'notOptimal_{name}.lp')
        raise ValueError(f"Master model returned status {model.Status}")
    if st in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return
    if st == GRB.INF_OR_UNBD:
        model.Params.DualReductions = 0
        model.optimize()
        st = model.Status
        if st in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return
    raise gp.GurobiError(
        f"{name} solve returned status {st} — no ObjVal available. "
        f"Inspect model.Status; if infeasible, consider model.computeIIS()."
    )


# -----------------------------------------------------------------------------
# Data model (multiclass)
# -----------------------------------------------------------------------------
@dataclass
class MultiClassData:
    Cr: int                      # regular capacity per day
    Co: int                      # overtime capacity per day
    N: int                       # booking window
    kappa: float                 # discount factor
    h: float                     # overtime cost base
    I: int                       # number of classes
    L: List[int]                 # sessions per class i
    R: List[List[int]]           # R[i][j] = slots in session j (0-based j)
    mu: List[float]              # arrivals/day per class
    F: Optional[List[List[float]]] = None
    wait_target: Optional[List[int]] = None
    wait_penalty: Optional[List[float]] = None
    Eu: Optional[List[float]] = None     # length M
    Ev: Optional[List[float]] = None     # length M
    Ew: Optional[List[float]] = None     # length I
    G: Optional[List[float]] = None      # postponement penalty per class
    Wmax: Optional[List[int]] = None     # backlog cap per class (finite state)
    ban_postponements: bool = True       # if True: sum_n x_{in} == w_i in pricing
    M: int = 0

    def finalize(self):
        self.M = self.N + max(self.L) - 1
        if self.Eu is None:
            self.Eu = [uniform(loc=0, scale=self.Cr).mean()*0.8] * self.M
            self.Eu[-1] = 0
        if self.Ev is None:
            self.Ev = [uniform(loc=0, scale=self.Co).mean()*0.8] * self.M
            self.Ev[-1] = 0
        if self.Ew is None:
            self.Ew = list(self.mu)
        if self.G  is None:
            self.G  = [10] * self.I
        if self.Wmax is None:
            # finite-state cap: moderately high vs. mu (tune as needed)
            self.Wmax = [max(5, int(np.ceil(50.0 * self.mu[i]))) for i in range(self.I)]
        print(self.Eu)
        print(self.Ev)
        print(self.Ew)
        print('E_u_alpha:', self.Eu)
        print('E_v_alpha:', self.Ev)
        print('E_w_alpha:', self.Ew)


def make_example_data() -> MultiClassData:
    """
    Base case (§6.1a): Cr=50, Co=6, N=25, κ=0.99, h=100;
    one treatment type: 5 sessions, 1 slot each; μ=10/day.
    """
    Cr, Co, N, kappa, h = 50, 6, 25, 0.99, 100.0
    L = [5]
    R = [[1, 1, 1, 1, 1]]
    mu = [10.0]
    wait_target = [10]
    wait_penalty = [50.0]
    Wmax = [50000]
    data = MultiClassData(Cr=Cr, Co=Co, N=N, kappa=kappa, h=h,
                          I=len(L), L=L, R=R, mu=mu,
                          wait_target=wait_target, wait_penalty=wait_penalty,
                          Wmax=Wmax, ban_postponements=False)
    data.finalize()
    return data


# -----------------------------------------------------------------------------
# Costs c_{i n} and h_m  (eq. 8, 0-based m)
# -----------------------------------------------------------------------------
def make_costs(data: MultiClassData) -> Tuple[List[List[float]], List[float]]:
    κ = data.kappa
    # h_m = κ^m * h  (paper is 1-based; here m = 0..M-1)
    h_m = [(κ ** m) * data.h for m in range(data.M)]
    c_in: List[List[float]] = []
    for i in range(data.I):
        if data.F is not None and data.F[i] is not None:
            f = list(data.F[i])
            if len(f) < data.N: f += [f[-1]] * (data.N - len(f))
        else:
            tgt = data.wait_target[i] if data.wait_target else 10
            pen = data.wait_penalty[i] if data.wait_penalty else 50.0
            f = [0.0] * tgt + [pen] * max(0, data.N - tgt)
            if len(f) < data.N: f += [pen] * (data.N - len(f))
        # c_{in} = sum_{k=1..n} κ^{k-1} f_{i,k}
        c_i = []
        for n in range(1, data.N + 1):
            s = 0.0
            for k in range(1, n + 1):
                s += (κ ** (k - 1)) * f[k - 1]
            c_i.append(s)
        c_in.append(c_i)
    return c_in, h_m

# -----------------------------------------------------------------------------
# Helpers (0-based indices)
# -----------------------------------------------------------------------------
def day_load_expr(model: gp.Model, day: int, x: gp.MVar, data: MultiClassData) -> gp.LinExpr:
    """
    Total slots scheduled on day `day` across all classes, as a Gurobi linear expr.
    x has shape (I, N) -> start of class i on offset n.
    """
    expr = gp.LinExpr(0.0)
    for i in range(data.I):
        Li = data.L[i]
        k_lo = max(0, day - Li + 1)
        k_hi = min(day, data.N - 1)
        for k in range(k_lo, k_hi + 1):
            j = day - k  # session index 0..Li-1
            expr += data.R[i][j] * x[i, k]
    return expr


def day_load_value(day: int, x_val: np.ndarray, data: MultiClassData) -> int:
    """
    Numeric day load: x_val is shape (I, N), integer or int-like.
    """
    s = 0
    for i in range(data.I):
        Li = data.L[i]
        k_lo = max(0, day - Li + 1)
        k_hi = min(day, data.N - 1)
        for k in range(k_lo, k_hi + 1):
            j = day - k
            s += data.R[i][j] * int(round(x_val[i, k]))
    return s


# -----------------------------------------------------------------------------
# Initial columns (Appendix A) — one per class
#   v=y=0; w_i = 3 μ_i; others 0; maximize g s.t. g ≤ ℓ_m
# -----------------------------------------------------------------------------
def build_initial_column_for_class(data: MultiClassData, cls: int):
    κ = data.kappa

    m = gp.Model(f"init_col_cls{cls}")
    m.Params.OutputFlag = 0

    # build u as a list of Vars (robust X retrieval)
    u_vars = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=data.Cr, name=f'u[{j}]') for j in range(data.M)]
    x_i = m.addMVar(shape=data.N, vtype=GRB.INTEGER, lb=0, name="x_i")
    g = m.addVar(lb=-GRB.INFINITY, name="g")

    w_i = int(round(3.0 * data.mu[cls]))
    m.addConstr(x_i.sum() <= w_i, name="backlog_cap")

    # capacity with y=0
    for day in range(data.M):
        sched = gp.LinExpr(0.0)
        Li = data.L[cls]
        k_lo = max(0, day - Li + 1)
        k_hi = min(day, data.N - 1)
        for k in range(k_lo, k_hi + 1):
            j = day - k
            sched += data.R[cls][j] * x_i[k]
        m.addConstr(u_vars[day] + sched <= data.Cr, name=f"cap[{day}]")

    # end-of-horizon empty
    m.addConstr(u_vars[data.M - 1] == 0, name="uM_zero")

    # g ≤ ℓ_m = u_m - κ u'_m  (with y=0)
    for day in range(data.M):
        if day < data.M - 1:
            Li = data.L[cls]
            k_lo = max(0, (day + 1) - Li + 1)
            k_hi = min(day + 1, data.N - 1)
            sched_next = gp.LinExpr(0.0)
            for k in range(k_lo, k_hi + 1):
                j = (day + 1) - k
                sched_next += data.R[cls][j] * x_i[k]
            u_prime = u_vars[day + 1] + sched_next  # y=0
        else:
            u_prime = 0.0
        m.addConstr(g <= u_vars[day] - κ * u_prime, name=f"g_le_l[{day}]")

    m.setObjective(g, GRB.MAXIMIZE)
    m.optimize()
    _ensure_objval(m, "InitColumn")

    # Collect values
    u_val = np.array([int(round(v.X)) for v in u_vars], dtype=int)
    v_val = np.zeros(data.M, dtype=int)
    y_val = np.zeros(data.M, dtype=int)
    w_vec = np.zeros(data.I, dtype=int); w_vec[cls] = w_i
    x_val = np.zeros((data.I, data.N), dtype=int)
    x_val[cls, :] = np.rint(x_i.X).astype(int)
    candidate = ((u_val, v_val, w_vec), (x_val, y_val))
    return candidate


def build_initial_columns(data: MultiClassData):
    return [build_initial_column_for_class(data, i) for i in range(data.I)]


# -----------------------------------------------------------------------------
# Phase‑I (A.1) — persistent & incremental (adds one X_j per column)
# -----------------------------------------------------------------------------
class PhaseIInc:
    def __init__(self, data: MultiClassData):
        self.data = data
        self.m = gp.Model("PhaseI_inc")
        self.m.Params.OutputFlag = 0
        self.m.Params.Method = 1  # dual simplex

        self.sigma = self.m.addVar(lb=0.0, name="sigma")
        # Mass equality, start with 0 == 1; we add (1-κ) via columns
        self.mass = self.m.addConstr(gp.LinExpr(0.0) == 1.0, name="mass")

        self.u_ge = [self.m.addConstr(self.sigma >= self.data.Eu[d], name=f"u_ge[{d}]")
                     for d in range(self.data.M)]
        self.v_ge = [self.m.addConstr(self.sigma >= self.data.Ev[d], name=f"v_ge[{d}]")
                     for d in range(self.data.M)]
        self.w_ge = [self.m.addConstr(self.sigma >= self.data.Ew[i], name=f"w_ge[{i}]")
                     for i in range(self.data.I)]
        self.X_vars: List[gp.Var] = []
        self._num_cols = 0
        self.m.setObjective(self.sigma, GRB.MINIMIZE)
        self.m.update()

    def add_column(self, candidate, col_name):
        coeffs = get_constr_coefficients(self.data, candidate)
        colobj = gp.Column(coeffs=coeffs, constrs=self.m.getConstrs())  # coeffs FIRST
        Xj = self.m.addVar(lb=0.0, obj=0.0, column=colobj, name=col_name)
        self.X_vars.append(Xj)
        self._num_cols += 1

    def add_columns(self, candidates):
        for i, c in enumerate(candidates): self.add_column(c, f'init_{i}')

    def solve(self):
        self.m.optimize()
        _ensure_objval(self.m, "Phase-I")
        sigma = self.m.ObjVal
        theta = self.mass.Pi
        U = [c.Pi for c in self.u_ge]
        V = [c.Pi for c in self.v_ge]
        W = [c.Pi for c in self.w_ge]
        clamp = lambda x: 0.0 if abs(x) < 1e-10 else x
        print('coefficient:')
        print([clamp(theta)]+ [clamp(x) for x in U]+[clamp(x) for x in V]+[clamp(x) for x in W])
        return sigma, clamp(theta), [clamp(x) for x in U], [clamp(x) for x in V], [clamp(x) for x in W]


# -----------------------------------------------------------------------------
# Pricing (14) with MVar arrays
# -----------------------------------------------------------------------------
def solve_pricing(
    data: MultiClassData,
    W0: float, U: List[float], V: List[float], W: List[float]):
    κ = data.kappa
    c_in, h_m = make_costs(data)

    m = gp.Model("pricing")
    m.Params.OutputFlag = 0
    m.Params.Method = 1

    # STATE
    u  = m.addMVar(shape=data.M, vtype=GRB.INTEGER, lb=0, ub=data.Cr, name="u")
    v  = m.addMVar(shape=data.M, vtype=GRB.INTEGER, lb=0, ub=data.Co, name="v")
    wv = m.addMVar(shape=data.I, vtype=GRB.INTEGER, lb=0, ub=np.array(data.Wmax), name="w")

    # ACTION
    x = m.addMVar(shape=(data.I, data.N), vtype=GRB.INTEGER, lb=0, name="x")
    y = m.addMVar(shape=data.M, vtype=GRB.INTEGER, lb=0, ub=data.Co, name="y")

    # backlog feasibility
    if data.ban_postponements:
        for i in range(data.I):
            m.addConstr(gp.quicksum(x[i, n] for n in range(data.N)) == wv[i], name=f"backlog_eq[{i}]")
    else:
        m.addConstr(x.sum(axis=1) <= wv, name="backlog_le")

    # per-day capacity & validity; end-of-horizon emptiness
    for day in range(data.M):
        load = day_load_expr(m, day, x, data)
        m.addConstr(u[day] + load <= data.Cr + y[day], name=f"cap[{day}]")
        m.addConstr(v[day] + y[day] <= data.Co,          name=f"ot_cap[{day}]")
        m.addConstr(y[day] <= load,                      name=f"y_le_load[{day}]")
    m.addConstr(u[data.M - 1] == 0, name="uM_zero")
    m.addConstr(v[data.M - 1] == 0, name="vM_zero")

    # ℓ_m, m_m
    l_terms, m_terms = [], []
    for day in range(data.M):
        if day < data.M - 1:
            load_next = day_load_expr(m, day + 1, x, data)
            u_prime = u[day + 1] + load_next - y[day + 1]
            v_prime = v[day + 1] + y[day + 1]
        else:
            u_prime = 0.0
            v_prime = 0.0
        l_terms.append(u[day] - κ * u_prime)
        m_terms.append(v[day] - κ * v_prime)

    # x_i(s,a) = (1-κ)w_i + κ Σ_n x_{in} - κ μ_i
    xi_expr = []
    for i in range(data.I):
        xi_expr.append((1 - κ) * wv[i] +
                       κ * gp.quicksum(x[i, n] for n in range(data.N)) -
                       κ * data.mu[i])

    # c(s,a)
    cost_expr = gp.LinExpr(0.0)
    for i in range(data.I):
        cost_expr += gp.quicksum(c_in[i][n] * x[i, n] for n in range(data.N))
        cost_expr += data.G[i] * (wv[i] - gp.quicksum(x[i, n] for n in range(data.N)))
    cost_expr += gp.quicksum(h_m[d] * y[d] for d in range(data.M))

    # Reduced cost
    red_cost = cost_expr - (1 - κ) * W0 \
               - gp.quicksum(l_terms[d] * U[d] for d in range(data.M)) \
               - gp.quicksum(m_terms[d] * V[d] for d in range(data.M)) \
               - gp.quicksum(xi_expr[i] * W[i] for i in range(data.I))

    m.setObjective(red_cost, GRB.MINIMIZE)
    m.optimize()
    _ensure_objval(m, "Pricing")
    rc = m.ObjVal

    # Extract new column
    u_val = np.rint(u.X).astype(int)
    v_val = np.rint(v.X).astype(int)
    y_val = np.rint(y.X).astype(int)
    w_val = np.rint(wv.X).astype(int)
    x_val = np.rint(x.X).astype(int)  # (I,N)
    state = (u_val, v_val, w_val)
    action = (x_val, y_val)
    candidate = (state, action)
    return candidate, rc

def get_constr_coefficients(data, candidate):
    κ = data.kappa
    state, action = candidate
    u, v, w = state
    x, y = action
    l_vec, m_vec = [], []
    for day in range(data.M):
        if day < data.M - 1:
            load_next = day_load_value(day + 1, x, data)
            u_prime = u[day + 1] + load_next - y[day + 1]
            v_prime = v[day + 1] + y[day + 1]
        else:
            u_prime = 0
            v_prime = 0
        l_vec.append(u[day] - κ * u_prime)
        m_vec.append(v[day] - κ * v_prime)
    # x_vec
    x_vec = [(1 - κ) * w[i] + κ * int(x[i, :].sum()) - κ * data.mu[i]
             for i in range(data.I)]
    return [1 - data.kappa] + l_vec + m_vec + x_vec
def run_column_generation_multiclass(
    data: MultiClassData,
    max_phaseI_iter: int = 100,
    max_main_iter: int = 500,
    tol_sigma: float = 1e-6,
    tol_rc: float = 1e-8,
    verbose: bool = True,):
    """
    1) Build Appendix‑A initial columns (one per class).
    2) Persistent Phase‑I (incremental): minimize sigma, adding columns via pricing until feasible.
    3) Restricted primal loop with pricing until no violated constraint (rc ≥ 0).
    """
    # 1) Initial columns
    #cols: List[Column] = build_initial_columns(data)
    candidates =  build_initial_columns(data)
    candidate_strs = set()
    if verbose:
        print(f"Initial columns: {len(candidates)}")

    # 2) Phase‑I
    pi = PhaseIInc(data)
    pi.add_columns(candidates)

    for it in range(max_phaseI_iter):
        sigma, theta, U, V, W = pi.solve()
        if verbose:
            print(f"[Phase‑I] iter={it:03d}  sigma={sigma:.6g}  |cols|={len(candidates)}")
        if sigma <= tol_sigma:
            if verbose:
                print("  -> Phase‑I feasible (sigma≈0).")
            break
        candidate, rc = solve_pricing(data, theta, U, V, W)
        candidate_str = str(candidate)
        if verbose:
            print(f"     pricing reduced cost = {rc:.6g}")
        if rc < -tol_rc:
            pi.add_column(candidate, f'phase_I_{it}')
            candidates.append(candidate)
            if candidate_str in candidate_strs:
                print('Warning: added!')
            candidate_strs.add(candidate_str)
        else:
            if verbose:
                print("     No improving column found for Phase‑I (rc≥0).")
            break

    return candidates


# -----------------------------------------------------------------------------
# Minimal demo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    data = make_example_data()
    data.finalize()
    '''
    result = run_column_generation_multiclass(
        data,
        max_phaseI_iter=1000,
        max_main_iter=200,
        tol_sigma=1e-6,
        tol_rc=1e-8,
        verbose=True
    )
    print("\n=== Summary ===")
    print(result[0])
    
    u  = np.ones((data.M,))
    u[:14] = 50
    v = np.ones((data.M,))
    w = np.ones((data.I,))
    w[0] = 30
    x = np.ones((data.I, data.N))
    y = np.ones((data.M,))
    candidiate = ((u, v, w), (x, y))
    print(get_constr_coefficients(data, candidiate))
    '''
    print(solve_pricing(data, 1, [2]*data.M, [1]*data.M, [1]*data.I))
    # [0.010000000000000009, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -9.600000000000001]
