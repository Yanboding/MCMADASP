from collections import defaultdict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import iter_to_tuple


class SAAdvanceAgent:

    def __init__(self, env, discount_factor, sample_path_number=3000, V=None, Q=None):
        self.env = env
        self.discount_factor = discount_factor
        self.V = V
        self.Q = Q
        self.sample_path_number = sample_path_number
        delta = []
        for omega in range(self.sample_path_number):
            new_arrivals = self.env.reset_arrivals(1)
            delta.append(new_arrivals)
        self.delta = np.array(delta)
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(int))
        if V is None:
            self.V = {}
        self.action_map = {}

    def solve(self, state, t, x=None, action=None):
        # ---------- shortcuts ----------
        N = self.env.decision_epoch
        I = self.env.num_types
        H = N - t  # remaining horizon
        M = self.sample_path_number
        gamma = self.discount_factor
        w = np.asarray(self.env.holding_cost)
        r = np.asarray(self.env.treatment_pattern)[0]  # (I,)
        C = self.env.regular_capacity
        O = self.env.overtime_cost

        bookings, delta_t, b = state  # b shape = (H+1, I)
        b = np.asarray(b, dtype=int)
        delta = self.delta  # shape = (M, H, I)
        delta_max = delta.max(axis=(0, 1))

        # ---------- model ----------
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.start()
            with gp.Model("SA_Advance", env=env) as m:
                m.setParam("OutputFlag", 0)
                m.setParam("LogToConsole", 0)
                # ---------- 1. today’s increments ----------
                a_t = m.addVars(H + 1, I, vtype=GRB.CONTINUOUS, name="a_t")
                if action is not None:
                    for j in range(H + 1):
                        for i in range(I):
                            a_t[j, i].lb = a_t[j, i].ub = int(action[j, i])

                # ---------- 2. triangular index sets ----------
                idx_a_fut = [
                    (omega, tau, j, i)
                    for tau in range(1, H + 1)  # stage
                    for j in range(H - tau + 1)  # slot
                    for omega in range(M)  # scenario
                    for i in range(I)  # class
                ]
                a_fut = m.addVars(idx_a_fut, lb=0, vtype=GRB.CONTINUOUS, name="a_fut")
                idx_b_fut = [
                    (omega, tau, j, i)
                    for tau in range(1, H + 1)
                    for j in range(H - tau + 1)
                    for omega in range(M)
                    for i in range(I)
                ]
                b_fut = m.addVars(
                    idx_b_fut,
                    lb=0,
                    vtype=GRB.CONTINUOUS,
                    name="b_fut",
                )

                # current cumulative schedule
                b_now = m.addVars(
                    H + 1,
                    I,
                    lb=b,
                    ub=b + delta_max,
                    vtype=GRB.CONTINUOUS,
                    name="b_now",
                )

                # overtime
                y_now = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_now")
                y_future = m.addVars(M, range(1, H + 1), lb=0, vtype=GRB.CONTINUOUS, name="y_fut")

                # ---------- 3. definitions ----------
                # today
                m.addConstrs(
                    (b_now[j, i] == b[j, i] + a_t[j, i] for j in range(H + 1) for i in range(I)),
                    name="def_today",
                )
                # tau = 1
                m.addConstrs(
                    (
                        b_fut[omega, 1, j, i]
                        == b_now[j + 1, i] + a_fut[omega, 1, j, i]
                        for omega in range(M)
                        for j in range(H)
                        for i in range(I)
                    ),
                    name="link_tau1",
                )
                # tau ≥ 2 cascade
                m.addConstrs(
                    (
                        b_fut[omega, tau, j, i]
                        == b_fut[omega, tau - 1, j + 1, i] + a_fut[omega, tau, j, i]
                        for tau in range(2, H + 1)
                        for j in range(H - tau + 1)
                        for omega in range(M)
                        for i in range(I)
                    ),
                    name="cascade",
                )

                # ---------- 4. demand constraints ----------
                # today
                m.addConstrs(
                    (
                        gp.quicksum(a_t[j, i] for j in range(H + 1)) == delta_t[i]
                        for i in range(I)
                    ),
                    name="demand_today",
                )
                # future
                m.addConstrs(
                    (
                        gp.quicksum(a_fut[omega, tau, j, i] for j in range(H - tau + 1)) == delta[omega, tau, i]
                        for tau in range(1, H + 1)
                        for omega in range(M)
                        for i in range(I)
                    ),
                    name="demand_future",
                )

                # ---------- 5. capacity (overtime) ----------
                m.addConstr(
                    y_now >= gp.quicksum(b_now[0, i] * r[i] for i in range(I)) - C,
                    name="cap_today",
                )
                m.addConstrs(
                    (
                        y_future[omega, tau]
                        >= gp.quicksum(b_fut[omega, tau, 0, i] * r[i] for i in range(I)) - C
                        for tau in range(1, H + 1)
                        for omega in range(M)
                    ),
                    name="cap_future",
                )

                # ---------- 6. objective ----------
                imm_cost = (
                        gp.quicksum(w[i] * gp.quicksum(b_now[j, i] for j in range(H + 1)) for i in range(I))
                        + O * y_now
                )
                fut_cost = (
                        1.0
                        / M
                        * gp.quicksum(
                    gamma ** tau
                    * (
                            gp.quicksum(
                                w[i] * gp.quicksum(b_fut[omega, tau, j, i] for j in range(H - tau + 1))
                                for i in range(I)
                            )
                            + O * y_future[omega, tau]
                    )
                    for tau in range(1, H + 1)
                    for omega in range(M)
                )
                )
                m.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)

                # ---------- 7. solve ----------
                m.setParam("Presolve", 2)
                m.setParam("Threads", 0)
                m.optimize()
                # ---------- 8. return ----------
                if m.Status == GRB.OPTIMAL:
                    a_now = np.zeros((H + 1, I), dtype=int)
                    for (j, i), v in m.getAttr("X", a_t).items():
                        a_now[j, i] = int(round(v))
                    return a_now, y_now.X, m.ObjVal
                else:
                    raise RuntimeError("Optimal solution not found")

    def policy(self, state, t):
        state_tuple = iter_to_tuple(state)
        if (state_tuple, t) in self.action_map:
            return self.action_map[(state_tuple, t)]
        action, overtime, obj_value = self.solve(state, t)
        self.action_map[(state_tuple, t)] = action
        return action

if __name__ =="__main__":
    from experiments.config import Config
    from environment import AdvanceSchedulingEnv, MultiClassPoissonArrivalGenerator
    from environment.utility import get_system_dynamic
    import time

    config = Config.from_real_scale()
    env_params = config.env_params
    state = config.init_state
    # opt: 200.9459842557252
    env = AdvanceSchedulingEnv(**env_params)
    agent = SAAdvanceAgent(env, discount_factor=env_params['discount_factor'], sample_path_number=1000)
    print(state)
    start = time.time()
    solution_a, y_t_solution, approx_obj_value = agent.solve(state, 1, 0)
    print(time.time() - start)
    print(approx_obj_value)

