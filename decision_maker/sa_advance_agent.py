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
        γ = self.discount_factor
        w = np.asarray(self.env.holding_cost)
        r = np.asarray(self.env.treatment_pattern)[0]  # (I,)
        C = self.env.regular_capacity
        O = self.env.overtime_cost

        bookings, delta_t, b = state  # b shape = (H+1, I)
        b = np.asarray(b, dtype=int)
        delta = self.delta  # shape = (M, H, I)
        delta_max = delta.max(axis=(0, 1))

        # ---------- model ----------
        with gp.Model("SA_Advance") as m:
            m.setParam("OutputFlag", 0)
            m.setParam("LogToConsole", 0)

            # ---------- 1. today’s increments ----------
            a_t = m.addVars(H + 1, I, vtype=GRB.INTEGER, name="a_t")
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
            a_fut = m.addVars(idx_a_fut, lb=0, vtype=GRB.INTEGER, name="a_fut")

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
                vtype=GRB.INTEGER,
                name="b_fut",
            )

            # current cumulative schedule
            b_now = m.addVars(
                H + 1,
                I,
                lb=b,
                ub=b + delta_max,
                vtype=GRB.INTEGER,
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
                γ ** tau
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

    def exact_solve(self, state, t, x=None, action=None):
        N = self.env.decision_epoch
        I = self.env.num_types
        gamma = self.discount_factor
        w = self.env.holding_cost
        bookings, delta_t, b = state
        r = self.env.treatment_pattern
        C = self.env.regular_capacity
        O = self.env.overtime_cost
        P, delta = self.env.arrival_generator.get_sample_paths_with_prob(N - t - x)
        sample_path_number = len(delta)
        with gp.Model("SA_Advance") as model:
            model.setParam('OutputFlag', 0)
            model.setParam('LogToConsole', 0)

            # ----------------------
            # 1. Define the sets
            # ----------------------
            # i in {0,...,I-1}  -- for convenience we'll shift Python indexing to 0..I-1
            # j in {0,...,N-t}
            # tau in {1,...,N-t}
            # omega in {0,...,M-1} in Python indexing
            # (Adjust as needed if your data is 1-indexed.)

            # ----------------------
            # 2. Add decision variables
            # ----------------------

            # a^t_{j,i}: # of new treatments allocated in period t + j (for each i)
            #   j = 0..(N - t), i = 0..(I-1)
            #   must be nonnegative integers
            a_t = model.addVars(N-t+1, I, vtype=GRB.INTEGER, name="a_t")
            if action is not None:
                for j in range(N-t+1):
                    for i in range(I):
                        a_t[j, i].lb = action[j, i]
                        a_t[j, i].ub = action[j, i]

            # a^{t+tau}_{j,i,omega}: # of new treatments allocated in period (t + tau + j) in scenario omega
            #   tau = 1..(N - t)
            #   j = 0..(N - t - tau)
            #   i = 0..(I-1)
            #   omega = 0..(M - 1)
            a_future = {}
            for tau in range(1, N - t + 1):
                a_t_tau_omega = model.addVars(sample_path_number, N - t - tau + 1, I, vtype=GRB.INTEGER, name="a_t_j_tau_omega")
                a_future[tau] = a_t_tau_omega

            # Overtime in period t
            y_t = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_t")

            # Overtime in period (t + tau) under scenario omega
            y_future = model.addVars(sample_path_number, range(1, N - t + 1), lb=0, vtype=GRB.CONTINUOUS, name="y_future")
            b_bar_t = model.addVars(N-t+1, I, vtype=GRB.INTEGER, name="b_bar_t")
            for j in range(N-t+1):
                for i in range(I):
                    model.addConstr(b_bar_t[j, i] == b[j, i] + a_t[j, i])
            # New schedule in period (t+\tau) under scenario omega
            b_bar_future = {}
            for tau in range(1, N - t + 1):
                b_bar_t_tau_omega = model.addVars(sample_path_number, N - t - tau + 1, I, vtype=GRB.INTEGER,
                                                name="b_bar_t_j_tau_omega")
                b_bar_future[tau] = b_bar_t_tau_omega
                for omega in range(sample_path_number):
                    for j in range(N - t - tau + 1):
                        for i in range(I):
                            model.addConstr(b_bar_future[tau][omega, j, i] == b[tau + j, i] + a_t[tau + j, i] + gp.quicksum(
                                a_future[k][omega, tau + j - k, i] for k in range(1, tau + 1)))

            # immediate_cost: sum_{i,j} w_i [ b^t_{j,i} + a^t_{j,i} ]
            immediate_cost = gp.quicksum(w[i] * gp.quicksum(b_bar_t[j, i] for j in range(N-t+1))
                                        for i in range(I)) + O * y_t

            # future_cost: (1/M)* sum_{omega} sum_{tau} gamma^tau [
            #                sum_{i,j} w_i * ( b^t_{\tau+j,i} + a^t_{\tau+j,i}
            #                    + sum_{k=1}^tau a^{t+k}_{\tau+j-k, i, omega} )
            #                + O * y^{t+tau}_omega
            #              ]
            # \bar{b}_{j,i, \omega}^{t+\tau} = gp.quicksum(b[tau+j, i] + a_t[tau+j, i]
            #             + gp.quicksum(a_future[k][omega, tau+j-k, i] for k in range(1, tau+1))
            #             for j in range(N-t- tau + 1))
            future_cost = gp.quicksum(P[omega]*gp.quicksum(gamma ** tau * (gp.quicksum(w[i] * gp.quicksum(b_bar_future[tau][omega, j, i]
                                    for j in range(N-t- tau + 1))
                                    for i in range(I)) + O * y_future[omega, tau])
                                    for tau in range(1, N-t+1))
                                    for omega in range(sample_path_number))

            model.setObjective(immediate_cost + future_cost, GRB.MINIMIZE)

            # ----------------------
            # 3. Define the constraints
            # ----------------------

            #
            # (cons3)  sum_{j=0..N-t} a^t_{j,i} = delta_i^t
            #
            # for i in 1..I
            #
            for i in range(I):
                model.addConstr(
                    gp.quicksum(a_t[j, i] for j in range(N - t + 1)) == delta_t[i],
                    name=f"cons3_delta_t_{i}"
                )

            #
            # (cons4)  sum_{j=0..N-t-tau} a^{t+tau}_{j,i,omega} = delta_i^{t+tau}
            #
            # for tau = 1..(N-t), i=1..I, omega=1..M
            # --> Each scenario must meet the same demand for category i in period t+tau
            #     (You need to have delta_i^{t+tau} for each tau, presumably the same across scenarios;
            #      if each scenario has a *different* demand, adapt accordingly.)
            #
            # Suppose you have a data structure delta_t_tau[tau][i] that gives delta_i^{t+tau}.
            #
            for tau in range(1, N - t + 1):
                for i in range(I):
                    for omega in range(sample_path_number):
                        model.addConstr(gp.quicksum(a_future[tau][omega, j, i] for j in range(N - t - tau + 1)) == delta[omega, tau, i], name=f"cons4_delta_{tau}_{i}_om{omega}")

            #
            # (cons5) and (cons6):
            #   y^t >= sum_{i=1}^I (b^t_{0,i} + a^t_{0,i}) * r_i - C
            #   and y^t >= 0 (already handled by lb=0)
            #
            # If b_t[j,i] is given by b_t[j][i], then for j=0:
            model.addConstr(
                y_t >= gp.quicksum(b_bar_t[0, i] * r[0, i] for i in range(I)) - C,
                name="cons5"
            )
            #
            # (cons7) and (cons8):
            #   for tau=1..N−t, omega=1..M:
            #     y^{t+tau}_omega >= sum_{i=1}^I [ b^t_{\tau+0,i} + a^t_{\tau+0,i} + sum_{k=1}^tau a^{t+k}_{\tau+0−k,i,omega} ] * r_i  - C
            #   and y^{t+tau}_omega >= 0 (again lb=0 handles the non-negativity).
            #
            # Here we need the “0th” job in period t+tau, that is j=0 in the expression \bar{b}^{t+tau}_{0,i,omega}.
            # Notice that j=0 => \bar{b}^{t+tau}_{0,i,omega} = b^t_{\tau,i} + a^t_{\tau,i} + sum_{k=1}^tau a^{t+k}_{\tau−k,i,omega}.
            #
            for tau in range(1, N - t + 1):
                for omega in range(sample_path_number):
                    model.addConstr(
                        y_future[omega, tau] >= gp.quicksum(b_bar_future[tau][omega, 0, i] * r[0, i] for i in range(I)) - C,
                        name=f"cons7_tau{tau}_om{omega}")

            model.optimize()
            # Retrieve solution (example)
            if model.status == GRB.OPTIMAL:
                a_t_solution = model.getAttr('X', a_t)
                y_t_solution = y_t.X
                solution_a = np.zeros((N-t+1, I))
                for (t, i), value in a_t_solution.items():
                    solution_a[t, i] = value
                solution_a = solution_a.astype(int)
                obj_value = model.objVal
                print(solution_a)
                return solution_a, y_t_solution, obj_value
            else:
                print("No optimal solution found")
        return

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

