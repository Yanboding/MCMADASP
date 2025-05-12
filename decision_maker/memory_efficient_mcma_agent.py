import time
from collections import defaultdict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import iter_to_tuple


class SAAdvanceFastAgent:
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

    def _acquire_grb_env(self, silent=True, wait=TOKEN_WAIT):
        """
        Try to create and start a gp.Env.  If all tokens are in use,
        wait <wait> seconds and retry indefinitely.
        """
        while True:
            try:
                grb_env = gp.Env(empty=True)  # no token yet
                if silent:
                    grb_env.setParam("OutputFlag", 0)
                grb_env.start()  # tries to grab ONE token
                print('Get one token...')
                return grb_env  # success
            except gp.GurobiError as e:
                if "All tokens currently in use" in str(e):
                    print('Waiting...')
                    time.sleep(wait)  # back‑off and try again
                else:
                    raise  # some other licence error

    def solve(self, state, t, x=None, action=None):
        N = self.env.decision_epoch
        I = self.env.num_types
        H = N - t
        gamma = self.discount_factor
        w = np.asarray(self.env.holding_cost)
        r = np.asarray(self.env.treatment_pattern)  # (l, I)
        C = self.env.regular_capacity
        O = self.env.overtime_cost
        l = self.env.num_sessions

        bookings, delta_t, future_schedule = state  # b shape = (H+1, I)
        z = convet_state_to_booked_slots(bookings, future_schedule, self.env.treatment_pattern)
        decision_variable_type = GRB.INTEGER
        # ---------- shortcuts ----------
        with gp.Model("SA_Advance", env=self.grb_env) as m:
            m.setParam("OutputFlag", 0)
            m.setParam("LogToConsole", 0)
            # ---------- 1. today’s increments ----------
            a_t = m.addVars(H+1, I, vtype=decision_variable_type, name="a_t")
            if action is not None:
                for j in range(H + 1):
                    for i in range(I):
                        a_t[j, i].lb = a_t[j, i].ub = int(action[j, i])
            # N-t+l-1 = H + l-1

            z_bar_t = m.addVars(H + l, vtype=GRB.CONTINUOUS, name='z_t')

            # overtime
            y_t = m.addVars(H + l, lb=0, vtype=GRB.CONTINUOUS, name="y")

            # ---------- 4. demand constraints ----------
            # demand for today
            m.addConstrs(
                (
                    gp.quicksum(a_t[j, i] for j in range(H + 1)) == delta_t[i]
                    for i in range(I)
                ),
                name="C1_demand_today",
            )
            # booked appointment slots at the end of today
            m.addConstrs(
                (
                    z_bar_t[j] == z[j] + gp.quicksum(gp.quicksum(a_t[min(j-k, H), i] * r[k, i] for k in range(min(l-1, j)+1)) for i in range(I))
                    for j in range(H+l)
                ),
                name="C2_booked_slot_today"
            )
            # ---------- 5. capacity (overtime) ----------
            m.addConstrs(
                (
                    y_t[j] >= z_bar_t[j] - C
                    for j in range(H + 1)
                ),
                name="C3_overtime",
            )
            # ---------- 6. objective ----------
            imm_cost = gp.quicksum(
                gp.quicksum(
                    gp.quicksum(gamma**k * w[i]
                                for k in range(j+1)) * a_t[j, i]
                    for j in range(H + 1))
                for i in range(I)) + gp.quicksum(gamma**j * O * y_t[j] for j in range(H+l))
            m.setObjective(imm_cost, GRB.MINIMIZE)
            # ---------- 7. solve ----------
            m.setParam("Presolve", 2)
            m.setParam("Threads", 0)
            m.optimize()
            # ---------- 8. return ----------
            if m.Status == GRB.OPTIMAL:
                a_now = np.zeros((H + 1, I), dtype=int)
                for (j, i), v in m.getAttr("X", a_t).items():
                    a_now[j, i] = int(round(v))
                return a_now, m.getAttr("X", y_t), m.ObjVal
            else:
                raise RuntimeError("Optimal solution not found")
    def policy(self, state, t):
        state_tuple = iter_to_tuple(state)
        if (state_tuple, t) in self.action_map:
            return self.action_map[(state_tuple, t)]
        action, overtime, obj_value = self.solve(state, t)
        self.action_map[(state_tuple, t)] = action
        return action

def convet_state_to_booked_slots(bookings, future_schedule, treatment_patterns):
    appointment_slots = future_schedule @ treatment_patterns.T
    N, P = appointment_slots.shape
    total_len = N + len(bookings) - 1
    booked_slots = np.zeros(total_len, dtype=appointment_slots.dtype)

    # 2.  Vectorised diagonal add:
    #     element (i,j) in `appointment_slots` goes to position i+j in `booked_slots`.
    idx = np.arange(P) + np.arange(N)[:, None]  # shape (N,P)
    np.add.at(booked_slots, idx.ravel(), appointment_slots.ravel())

    # 3.  Pre‑existing bookings (past days).
    booked_slots[:len(bookings)] += bookings
    return booked_slots

if __name__ == "__main__":
    from experiments import Config
    from environment import AdvanceSchedulingEnv
    config = Config.from_adjust_EJOR_case()
    env_params = config.env_params
    env = AdvanceSchedulingEnv(**env_params)
    agent = SAAdvanceFastAgent(env, discount_factor=env_params['discount_factor'])
    init_state = config.init_state
    action, overtime, obj_value = agent.solve(init_state, 1)
    print(action)
    print(overtime)
    print(obj_value)

