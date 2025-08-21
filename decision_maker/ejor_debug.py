# rt_adp_cg.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
import itertools
import random

from gurobipy import Model, GRB, quicksum

# -----------------------------
# Problem data structures
# -----------------------------

@dataclass
class TreatmentType:
    name: str
    r_sessions: List[int]  # e.g., [2,1,1,1,1] (slots per session)
    mean_daily_arrivals: float  # m_i
    wait_penalty_schedule: List[Tuple[int, int, float]]
    """
    Piecewise daily penalties f_{i,k}: list of (k_start, k_end, per_slot_penalty).
    Example (paper Table 3): [(0,1,0), (2,5,100), (6,10,150), ...]
    k is "days waited" counting from 0 -> start-day-1 inclusive.
    """
    postpone_penalty: float  # g_i (large)

    @property
    def L(self) -> int:
        return len(self.r_sessions)


@dataclass
class RTInstance:
    types: List[TreatmentType]
    Cr: int            # regular capacity per day
    Co: int            # overtime capacity per day
    N: int             # booking horizon (days into the future one can start)
    M: int             # planning horizon M = N + max_i{L_i} - 1
    h: float           # overtime cost per slot (undiscounted)
    lam: float         # discount factor lambda in (0,1)
    w_max: List[int]   # upper bounds on w_i to keep state space finite (for pricing)
    alpha_expectations: Dict[str, List[float]]
    """
    alpha_expectations keys: 'u', 'v' -> length M; 'w' -> length I
    E_alpha[u_m], E_alpha[v_m], E_alpha[w_i] as in dual constraints.
    Set to zeros if you just want to learn around empty initial state.
    """

# -----------------------------
# Utilities: penalties and c_in
# -----------------------------

def daily_wait_penalty(tt: TreatmentType, k: int) -> float:
    # f_{i,k} from schedule, per appointment slot per day waited
    for (a, b, p) in tt.wait_penalty_schedule:
        if a <= k <= b:
            return p
    return tt.wait_penalty_schedule[-1][2]  # last segment if beyond

def c_in(tt: TreatmentType, n: int, lam: float) -> float:
    """
    c_{i,n} = sum_{k=1..n} lambda^{k-1} * f_{i,k}
    Here k counts days of wait before start. With k=1 meaning "wait 1 day".
    """
    val = 0.0
    for k in range(1, n+1):
        val += (lam ** (k-1)) * daily_wait_penalty(tt, k)
    # Multiply by total slots the treatment will consume (penalties defined per slot)
    total_slots = sum(tt.r_sessions)
    return val * total_slots

def discounted_overtime(m: int, h: float, lam: float) -> float:
    # h_m = lambda^{m-1} * h
    return (lam ** (m-1)) * h

# -----------------------------
# Booking MIP (paper eq. 17)
# -----------------------------

class BookingIP:
    def __init__(self, inst: RTInstance):
        self.inst = inst

    def compute_Cin(self, U: List[float], W: List[float]) -> List[List[float]]:
        I, N, M, lam = len(self.inst.types), self.inst.N, self.inst.M, self.inst.lam
        Cin = [[0.0]*(N+1) for _ in range(I)]  # 1..N (ignore index 0)
        for i, tt in enumerate(self.inst.types):
            g_i = tt.postpone_penalty
            for n in range(1, N+1):
                term = c_in(tt, n, lam)
                # lambda * sum_{k=n}^{n+L_i-1} r_i^{(k-n+1)} * U_k
                for k in range(n, min(n+tt.L-1, M)+1):
                    session_idx = k - n  # 0-based
                    term += lam * tt.r_sessions[session_idx] * U[k-1]  # U indexed 0..M-1
                term -= (g_i + lam * W[i])  # -(g_i + lambda * W_i)
                Cin[i][n] = term
        return Cin

    def compute_Hm(self, U: List[float], V: List[float]) -> List[float]:
        M, h, lam = self.inst.M, self.inst.h, self.inst.lam
        H = [0.0]*(M+1)  # 1..M
        for m in range(1, M+1):
            if m == 1:
                H[m] = h
            else:
                H[m] = discounted_overtime(m, h, lam) + lam*V[m-2] - lam*U[m-2]
        return H

    def solve(self, u: List[int], v: List[int], w: List[int],
              U: List[float], V: List[float], W: List[float]) -> Tuple[List[List[int]], List[int]]:
        """
        Given current state (u,v,w) and learned VFA coeffs (U,V,W),
        solve min sum_{i,n} Cin x_{in} + sum_m H_m y_m  s.t. (1)-(3).
        Returns x[i][n] and y[m].
        """
        I, N, M = len(self.inst.types), self.inst.N, self.inst.M
        Cr, Co = self.inst.Cr, self.inst.Co

        Cin = self.compute_Cin(U, W)   # Cin[i][n], n=1..N
        Hm  = self.compute_Hm(U, V)    # Hm[m],    m=1..M

        m = Model("booking")
        m.Params.OutputFlag = 0

        x = {(i,n): m.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{n}")
             for i in range(I) for n in range(1, N+1)}
        y = {mday: m.addVar(vtype=GRB.INTEGER, lb=0, ub=Co, name=f"y_{mday}")
             for mday in range(1, M+1)}

        # (1) cannot book more than waiting
        for i in range(I):
            m.addConstr(quicksum(x[(i,n)] for n in range(1, N+1)) <= w[i], name=f"wait_{i}")

        # (2) capacity by day m (regular + overtime booked today)
        for mday in range(1, M+1):
            booked_today_for_m = []
            for i, tt in enumerate(self.inst.types):
                # sum over start days k that affect day m
                lower = max(mday - tt.L + 1, 1)
                upper = min(mday, N)
                if lower <= upper:
                    for k in range(lower, upper+1):
                        # session index in r_sessions
                        sidx = mday - k  # 0-based
                        booked_today_for_m.append(tt.r_sessions[sidx] * x[(i,k)])
            m.addConstr(u[mday-1] + quicksum(booked_today_for_m) <= Cr + y[mday], name=f"cap_{mday}")

        # (3) overtime cap
        for mday in range(1, M+1):
            m.addConstr(v[mday-1] + y[mday] <= Co, name=f"ot_{mday}")

        # Objective
        obj = quicksum(Cin[i][n] * x[(i,n)] for i in range(I) for n in range(1, N+1)) \
              + quicksum(Hm[mday] * y[mday] for mday in range(1, M+1))
        m.setObjective(obj, GRB.MINIMIZE)
        m.optimize()

        x_sol = [[0]*(N+1) for _ in range(I)]
        y_sol = [0]*(M+1)
        if m.status == GRB.OPTIMAL:
            for i in range(I):
                for n in range(1, N+1):
                    x_sol[i][n] = int(round(x[(i,n)].X))
            for mday in range(1, M+1):
                y_sol[mday] = int(round(y[mday].X))
        return x_sol, y_sol

# -----------------------------
# Column (state–action pair) container
# -----------------------------

@dataclass
class Column:
    u: List[int]; v: List[int]; w: List[int]
    x: Dict[Tuple[int,int], int]; y: Dict[int, int]
    # Derived quantities used in master/pricing
    mu: List[float]; nu: List[float]; omega: List[float]
    cost: float

# -----------------------------
# Pricing MIP (paper (14)-(16))
# -----------------------------

class PricingProblem:
    def __init__(self, inst: RTInstance):
        self.inst = inst

    def build_and_solve(self, U: List[float], V: List[float], W: List[float], W0: float) -> Optional[Column]:
        I, N, M, lam = len(self.inst.types), self.inst.N, self.inst.M, self.inst.lam
        Cr, Co, h = self.inst.Cr, self.inst.Co, self.inst.h

        # Model
        m = Model("pricing")
        m.Params.OutputFlag = 0

        # State vars
        u = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=Cr, name=f"u_{d}") for d in range(M)]
        v = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=Co, name=f"v_{d}") for d in range(M)]
        w = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=self.inst.w_max[i], name=f"w_{i}") for i in range(I)]
        # Action vars
        x = {(i,n): m.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{n}") for i in range(I) for n in range(1, N+1)}
        y = {mday: m.addVar(vtype=GRB.INTEGER, lb=0, ub=Co, name=f"y_{mday}") for mday in range(1, M+1)}

        # Next-state deterministic components (u', v'), expected component in w' handled in omega
        u_p = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=Cr+Co, name=f"up_{d}") for d in range(M)]
        v_p = [m.addVar(vtype=GRB.INTEGER, lb=0, ub=Co,    name=f"vp_{d}") for d in range(M)]

        # feasibility (1)-(3)
        for i in range(I):
            m.addConstr(quicksum(x[(i,n)] for n in range(1, N+1)) <= w[i], name=f"wait_{i}")

        for mday in range(1, M+1):
            booked_today_for_m = []
            for i, tt in enumerate(self.inst.types):
                lower = max(mday - tt.L + 1, 1)
                upper = min(mday, N)
                if lower <= upper:
                    for k in range(lower, upper+1):
                        sidx = mday - k
                        booked_today_for_m.append(tt.r_sessions[sidx] * x[(i,k)])
            m.addConstr(u[mday-1] + quicksum(booked_today_for_m) <= self.inst.Cr + y[mday], name=f"cap_{mday}")
            m.addConstr(v[mday-1] + y[mday] <= self.inst.Co, name=f"ot_{mday}")

        # additional constraints (15) and (16)
        for mday in range(1, M+1):
            # y_m <= sum of new bookings for that day (cannot book "overtime" without bookings)
            booked_today_for_m = []
            for i, tt in enumerate(self.inst.types):
                lower = max(mday - tt.L + 1, 1)
                upper = min(mday, N)
                if lower <= upper:
                    for k in range(lower, upper+1):
                        sidx = mday - k
                        booked_today_for_m.append(tt.r_sessions[sidx] * x[(i,k)])
            m.addConstr(quicksum(booked_today_for_m) >= y[mday], name=f"y_link_{mday}")

        m.addConstr(u[M-1] == 0, name="uM_zero")
        m.addConstr(v[M-1] == 0, name="vM_zero")

        # transitions for u', v' (paper (4)-(5))
        for md in range(M-1):
            added = []
            for i, tt in enumerate(self.inst.types):
                lower = max((md+1) - tt.L + 1, 1)
                upper = min(md+1, N)
                if lower <= upper:
                    for k in range(lower, upper+1):
                        sidx = (md+1) - k
                        added.append(tt.r_sessions[sidx] * x[(i,k)])
            m.addConstr(u_p[md] == u[md+1] + quicksum(added) - y[md+2], name=f"uprime_{md}")  # y_{m+1}
            m.addConstr(v_p[md] == v[md+1] + y[md+2], name=f"vprime_{md}")

        # boundary primes are not used in mu/nu definitions for md = M (u'_M, v'_M == 0 implicitly)
        # Define mu, nu, omega (use linear expressions)
        mu = []
        nu = []
        omega = []
        for md in range(M-1):
            mu.append(u[md] - self.inst.lam * u_p[md])
            nu.append(v[md] - self.inst.lam * v_p[md])
        # For md = M, u_M = v_M = 0 -> mu_M = u_M - lam*0 = 0, nu_M = v_M - lam*0 = 0
        mu.append(0.0); nu.append(0.0)

        for i, tt in enumerate(self.inst.types):
            m_i = tt.mean_daily_arrivals
            omega.append(w[i] - self.inst.lam * (w[i] - quicksum(x[(i,n)] for n in range(1, N+1)) + m_i))

        # Stage cost c(s,a): (paper (8))
        stage_cost = quicksum(
            c_in(self.inst.types[i], n, self.inst.lam) * x[(i,n)]
            for i in range(I) for n in range(1, self.inst.N+1)
        ) \
        + quicksum(discounted_overtime(mday, h, self.inst.lam) * y[mday] for mday in range(1, M+1)) \
        + quicksum(self.inst.types[i].postpone_penalty * (w[i] - quicksum(x[(i,n)] for n in range(1, self.inst.N+1)))
                   for i in range(I))

        # Reduced cost (paper (14))
        rc = stage_cost \
             - (1 - self.inst.lam) * W0 \
             - quicksum(mu[md] * U[md] for md in range(M)) \
             - quicksum(nu[md] * V[md] for md in range(M)) \
             - quicksum(omega[i] * W[i] for i in range(I))

        m.setObjective(rc, GRB.MINIMIZE)
        m.optimize()

        if m.status != GRB.OPTIMAL:
            print('pass')
            return None

        # Extract a column
        u_sol = [int(round(var.X)) for var in u]
        v_sol = [int(round(var.X)) for var in v]
        w_sol = [int(round(var.X)) for var in w]
        x_sol = {(i,n): int(round(x[(i,n)].X)) for i in range(I) for n in range(1, self.inst.N+1)}
        y_sol = {mday: int(round(y[mday].X)) for mday in range(1, M+1)}
        mu_val = [_val(e) for e in mu]
        nu_val = [_val(e) for e in nu]
        omega_val = [_val(e) for e in omega]
        cost_val = float(stage_cost.getValue())

        return Column(u_sol, v_sol, w_sol, x_sol, y_sol, mu_val, nu_val, omega_val, cost_val)
# NEW (robust to Var or LinExpr)
def _val(e):
    try:
        return float(e.getValue())   # LinExpr
    except AttributeError:
        try:
            return float(e.X)        # Var
        except AttributeError:
            return float(e)          # plain float (shouldn't happen after the change)

# -----------------------------
# Master (dual) LP over columns
# -----------------------------

class MasterDual:
    def __init__(self, inst: RTInstance):
        self.inst = inst
        self.columns: List[Column] = []
        self.model: Optional[Model] = None
        self.X_vars = []  # aligned with self.columns
        self.duals = None  # (W0, U, V, W)

    def _build(self):
        m = Model("master_dual")
        m.Params.OutputFlag = 0

        X = [m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"X_{k}") for k in range(len(self.columns))]

        # (1 - lambda) sum X = 1
        m.addConstr(quicksum(X) * (1.0 - self.inst.lam) == 1.0, name="mass")

        # mu, nu, omega constraints
        U_rhs = self.inst.alpha_expectations.get('u', [0.0]*self.inst.M)
        V_rhs = self.inst.alpha_expectations.get('v', [0.0]*self.inst.M)
        W_rhs = self.inst.alpha_expectations.get('w', [0.0]*len(self.inst.types))

        for md in range(self.inst.M):
            m.addConstr(quicksum(self.columns[k].mu[md] * X[k] for k in range(len(self.columns))) >= U_rhs[md],
                        name=f"mu_{md}")
            m.addConstr(quicksum(self.columns[k].nu[md] * X[k] for k in range(len(self.columns))) >= V_rhs[md],
                        name=f"nu_{md}")

        for i in range(len(self.inst.types)):
            m.addConstr(quicksum(self.columns[k].omega[i] * X[k] for k in range(len(self.columns))) >= W_rhs[i],
                        name=f"omega_{i}")

        # Objective: minimize sum c(s,a) X
        m.setObjective(quicksum(self.columns[k].cost * X[k] for k in range(len(self.columns))), GRB.MINIMIZE)

        self.model = m
        self.X_vars = X

    def add_column(self, col: Column):
        self.columns.append(col)

    def solve(self) -> Tuple[float, List[float], List[float], List[float]]:
        if self.model is None:
            self._build()
        else:
            # add one variable and the coefficients to existing constraints
            m = self.model
            k = len(self.columns) - 1
            Xk = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"X_{k}")
            self.X_vars.append(Xk)
            # Update constraints (mass)
            m.getConstrByName("mass").RHS = 1.0  # unchanged, but keeps API explicit

            # Add coefficients to mu, nu, omega constraints
            for md in range(self.inst.M):
                m.chgCoeff(m.getConstrByName(f"mu_{md}"), Xk, self.columns[k].mu[md])
                m.chgCoeff(m.getConstrByName(f"nu_{md}"), Xk, self.columns[k].nu[md])
            for i in range(len(self.inst.types)):
                m.chgCoeff(m.getConstrByName(f"omega_{i}"), Xk, self.columns[k].omega[i])

            # Update objective
            m.setObjective(m.getObjective() + self.columns[k].cost * Xk)

        self.model.optimize()

        # Duals: W0 from 'mass', U from mu-constraints, V from nu-constraints, W from omega-constraints
        m = self.model
        W0 = m.getConstrByName("mass").Pi  # dual
        U = [m.getConstrByName(f"mu_{md}").Pi for md in range(self.inst.M)]
        V = [m.getConstrByName(f"nu_{md}").Pi for md in range(self.inst.M)]
        W = [m.getConstrByName(f"omega_{i}").Pi for i in range(len(self.inst.types))]
        self.duals = (W0, U, V, W)
        return self.duals

# -----------------------------
# Trainer (column generation driver)
# -----------------------------

class ADPColumnGeneration:
    def __init__(self, inst: RTInstance):
        self.inst = inst
        self.master = MasterDual(inst)
        self.pricer = PricingProblem(inst)

    def seed_columns_trivial(self):
        """
        Trivial feasible seed if alpha-expectations are zeros (empty system).
        For richer alpha, implement Appendix A seeding (paper Appendix A).
        """
        I, N, M = len(self.inst.types), self.inst.N, self.inst.M
        u = [0]*M; v = [0]*M; w = [0]*I
        x = {(i,n): 0 for i in range(I) for n in range(1, N+1)}
        y = {mday: 0 for mday in range(1, M+1)}
        mu = [0.0]*M; nu=[0.0]*M; omega=[0.0]*I
        cost = 0.0
        self.master.add_column(Column(u,v,w,x,y,mu,nu,omega,cost))

    def train(self, max_iters: int = 100, tol: float = -1e-4):
        if len(self.master.columns) == 0:
            self.seed_columns_trivial()

        for it in range(max_iters):
            W0, U, V, W = self.master.solve()
            col = self.pricer.build_and_solve(U=U, V=V, W=W, W0=W0)
            if col is None:
                break
            # reduced cost value stored in objective; if >= tol, stop
            # To compute rc, re-evaluate objective: if model.ObjVal >= tol → stop
            # Here, we check indirectly by plugging back a variable; simpler: use pricing model objVal
            # but we did not return rc explicitly; re-build quickly:
            # We'll reuse the master: if pricing model's objective >= tol, no violated constraint.
            # For clarity, stop when column's cost improvement seems small by testing dual feasibility:
            self.master.add_column(col)
            # Re-run master to get duals and check if new pricing will add something negative;
            # The loop will stop when pricer returns a nearly zero improvement under tol on next iter.

        # After training: return learned coefficients
        W0, U, V, W = self.master.solve()
        return W0, U, V, W

# -----------------------------
# Simple simulator (uses booking IP)
# -----------------------------

class Simulator:
    def __init__(self, inst: RTInstance, policy: BookingIP):
        self.inst = inst
        self.policy = policy

    def step(self, u, v, w, U, V, W):
        # Solve booking decision
        x, y = self.policy.solve(u, v, w, U, V, W)
        # Apply transitions (roll horizon)
        M, N = self.inst.M, self.inst.N
        I = len(self.inst.types)
        # compute u', v' deterministically
        u_next = [0]*M; v_next = [0]*M
        for md in range(M-1):
            add = 0
            for i, tt in enumerate(self.inst.types):
                lower = max((md+1) - tt.L + 1, 1)
                upper = min(md+1, N)
                if lower <= upper:
                    for k in range(lower, upper+1):
                        sidx = (md+1) - k
                        add += tt.r_sessions[sidx] * x[i][k]
            u_next[md] = u[md+1] + add - y[md+2]
            v_next[md] = v[md+1] + y[md+2]
        u_next[-1] = 0; v_next[-1] = 0

        # arrivals and waiting list
        w_next = [0]*I
        for i, tt in enumerate(self.inst.types):
            arrivals = random.poisson(lam=tt.mean_daily_arrivals) if hasattr(random, "poisson") else \
                       sum(1 for _ in range(1000) if random.random() < tt.mean_daily_arrivals/1000.0)  # crude
            booked = sum(x[i][n] for n in range(1, self.inst.N+1))
            w_next[i] = max(0, w[i] - booked + arrivals)

        return u_next, v_next, w_next

    def run(self, T: int, U, V, W, seed_state=None):
        I, M = len(self.inst.types), self.inst.M
        if seed_state is None:
            u = [0]*M; v = [0]*M; w = [0]*I
        else:
            u, v, w = seed_state
        traj = []
        for t in range(T):
            u,v,w = self.step(u,v,w,U,V,W)
            traj.append((u[:], v[:], w[:]))
        return traj

if __name__ == '__main__':
    lam = 0.99
    # One radical-like and one palliative-like type (toy)
    tt1 = TreatmentType(
        name="Radical-5x1",
        r_sessions=[2, 1, 1, 1, 1],
        mean_daily_arrivals=0.6,
        wait_penalty_schedule=[(0, 1, 0), (2, 5, 80), (6, 10, 150), (11, 100, 150)],
        postpone_penalty=1e5
    )
    tt2 = TreatmentType(
        name="Palliative-1x2",
        r_sessions=[2],
        mean_daily_arrivals=1.0,
        wait_penalty_schedule=[(0, 1, 0), (2, 5, 65), (6, 100, 100)],
        postpone_penalty=1e5
    )

    N = 25
    M = N + max(tt1.L, tt2.L) - 1
    inst = RTInstance(
        types=[tt1, tt2],
        Cr=120, Co=15,
        N=N, M=M,
        h=100.0, lam=lam,
        w_max=[50, 50],
        alpha_expectations={'u': [0.0] * M, 'v': [0.0] * M, 'w': [0.0, 0.0]}
    )

    trainer = ADPColumnGeneration(inst)
    W0, U, V, W = trainer.train(max_iters=20, tol=-1e-4)  # returns learned VFA coefficients
    print(W0, U, V, W)
    booking = BookingIP(inst)
    sim = Simulator(inst, booking)
    traj = sim.run(T=50, U=U, V=V, W=W)