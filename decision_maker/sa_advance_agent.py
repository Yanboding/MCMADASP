from collections import defaultdict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import iter_to_tuple

class SAAdvanceAgent:
    TOKEN_WAIT = 15
    def __init__(self, env, discount_factor, V=None, Q=None):
        self.env = env
        self.discount_factor = discount_factor
        self.V = V
        self.Q = Q
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(int))
        if V is None:
            self.V = {}
        self.action_map = {}
        self.grb_env = self._acquire_grb_env()
    
    def set_sample_paths(self, sample_path_number):
        self.sample_path_number = sample_path_number
        delta = []
        for omega in range(self.sample_path_number):
            new_arrivals = self.env.reset_arrivals(1)
            delta.append(new_arrivals)
        self.delta = np.array(delta)
    
    # ------------------------------------------------------------------
    # Helper: wait‑until‑token‑free loop
    # ------------------------------------------------------------------
    def _acquire_grb_env(self, silent=True, wait=TOKEN_WAIT):
        """
        Try to create and start a gp.Env.  If all tokens are in use,
        wait <wait> seconds and retry indefinitely.
        """
        while True:
            try:
                grb_env = gp.Env(empty=True)    # no token yet
                if silent:
                    grb_env.setParam("OutputFlag", 0)
                grb_env.start()                 # tries to grab ONE token
                print('Get one token...')
                return grb_env                  # success
            except gp.GurobiError as e:
                if "All tokens currently in use" in str(e):
                    print('Waiting...')
                    time.sleep(wait)            # back‑off and try again
                else:
                    raise                       # some other licence error

    def solve(self, state, t, x=None, action=None):
        # ---------- shortcuts ----------
        N = self.env.decision_epoch
        I = self.env.num_types
        H = N - t  # remaining horizon
        M = self.sample_path_number
        gamma = self.discount_factor
        w = np.asarray(self.env.holding_cost)
        r = np.asarray(self.env.treatment_pattern)  # (l, I)
        C = self.env.regular_capacity
        O = self.env.overtime_cost
        l = self.env.num_sessions

        z, delta_t, b = state  # b shape = (H+1, I)
        b = np.asarray(b, dtype=int)
        delta = self.delta  # shape = (M, H, I)
        decision_variable_type = GRB.INTEGER
        # ---------- model ----------
        with gp.Model("SA_Advance", env=self.grb_env) as m:
            m.setParam("OutputFlag", 0)
            m.setParam("LogToConsole", 0)
            # ---------- 1. today’s increments ----------
            a_t = m.addVars(H + 1, I, vtype=decision_variable_type, name="a_t")
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
            a_fut = m.addVars(idx_a_fut, lb=0, vtype=decision_variable_type, name="a_fut")
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
                vtype=decision_variable_type,
                name="b_fut",
            )
            # current cumulative schedule
            b_now = m.addVars(
                H + 1,
                I,
                lb=0,
                vtype=decision_variable_type,
                name="b_now",
            )
            # N-t+l-1 = H + l-1
            # overtime
            y_now = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name="y_now")
            y_future = m.addVars(M, range(1, H + l), lb=0, vtype=GRB.CONTINUOUS, name="y_fut")
            #print('N:', N, 't:', t)
            idx_z_fut = [
                (omega, tau, j)
                for tau in range(H + 1)
                for j in range(l)
                for omega in range(M)
            ]
            z_future = m.addVars(idx_z_fut, lb=0, vtype=GRB.CONTINUOUS, name="z_fut")

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
            #  booked appointment slots at the end of today
            m.addConstrs(
                (z_future[omega, 0, j] == z[j] + gp.quicksum(b_now[0, i] * r[j, i] for i in range(I))
                 for j in range(l)
                 for omega in range(M)),
                name="first_booked_slot"
            )
            m.addConstrs(
                (z_future[omega, tau, j] == z_future[omega, tau-1, j + 1] + gp.quicksum(b_fut[omega, tau, 0, i] * r[j, i] for i in range(I))
                    for tau in range(1, H+1)
                    for j in range(l - 1)
                    for omega in range(M)),
                name="booked_slot_future"
            )
            m.addConstrs(
                (z_future[omega, tau, l-1] == gp.quicksum(b_fut[omega, tau, 0, i] * r[l-1, i] for i in range(I))
                    for tau in range(1, H+1)
                    for omega in range(M)),
                name="booked_slot_last_day"
            )
            # ---------- 5. capacity (overtime) ----------
            m.addConstr(
                y_now >= z[0] + gp.quicksum(b_now[0, i] * r[0, i] for i in range(I)) - C,
                name="cap_today",
            )
            m.addConstrs(
                (
                    y_future[omega, tau] >= z_future[omega, tau, 0] - C
                    for tau in range(1, H + 1)
                    for omega in range(M)
                ),
                name="cap_future",
            )
            m.addConstrs(
                (
                    y_future[omega, H + tau] >= z_future[omega, H, tau] - C
                    for tau in range(1, l)
                    for omega in range(M)
                ),
                name="cap_future_last",
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
    # 54946.988268116984
    config = Config.from_multiappt_default_case()
    env_params = config.env_params
    state = config.init_state
    # opt: 200.9459842557252
    env = AdvanceSchedulingEnv(**env_params)
    agent = SAAdvanceAgent(env, discount_factor=env_params['discount_factor'])
    agent.set_sample_paths(1)
    print(state)
    start = time.time()
    solution_a, y_t_solution, approx_obj_value = agent.solve(state, 1, 0)
    print(time.time() - start)
    print(solution_a)
    print(approx_obj_value)
