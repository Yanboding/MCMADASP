import time
from collections import defaultdict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import iter_to_tuple


class SAAdvanceFastAgent:
    TOKEN_WAIT = 15

    def __init__(self, env, discount_factor, is_myopic=True, V=None, Q=None):
        self.env = env
        self.discount_factor = discount_factor
        self.is_myopic = is_myopic
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
        M = self.sample_path_number
        gamma = self.discount_factor
        w = self.env.holding_cost
        r = np.asarray(self.env.treatment_pattern)  # (l, I)
        C = self.env.regular_capacity
        O = self.env.overtime_cost
        l = self.env.num_sessions

        z, delta_t = state  # b shape = (H+1, I)
        delta = self.delta  # shape = (M, H, I)
        decision_variable_type = GRB.INTEGER
        # ---------- shortcuts ----------
        with gp.Model("SA_Advance", env=self.grb_env) as m:
            m.setParam("OutputFlag", 0)
            m.setParam("LogToConsole", 0)
            m.setParam("Presolve", 2)
            m.setParam("Threads", 0)
            # ---------- 1. today’s increments ----------
            a_t = m.addVars(H+1, I, vtype=decision_variable_type, name="a_t")
            if action is not None:
                for j in range(H + 1):
                    for i in range(I):
                        a_t[j, i].lb = a_t[j, i].ub = int(action[j, i])

            # ---------- 2. triangular index sets ----------
            if self.is_myopic == False:
                idx_a_fut = [
                    (omega, tau, j, i)
                    for tau in range(1, H + 1)  # stage
                    for j in range(H - tau + 1)  # slot
                    for omega in range(M)  # scenario
                    for i in range(I)  # class
                ]
                a_fut = m.addVars(idx_a_fut, lb=0, vtype=decision_variable_type, name="a_fut")
            # N-t+l-1 = H + l-1
            z_bar_t = m.addVars(H + l, vtype=GRB.CONTINUOUS, name='z_t')
            if self.is_myopic == False:
                idx_z_bar_fut = [
                    (omega, tau, j)
                    for tau in range(1, H + 1)
                    for j in range(H + l - tau)
                    for omega in range(M)
                ]
                z_bar_fut = m.addVars(idx_z_bar_fut, vtype=GRB.CONTINUOUS, name='z_t_fut')
            # overtime
            y_t = m.addVars(H + l, lb=0, vtype=GRB.CONTINUOUS, name="y")
            if self.is_myopic == False:
                idx_y_t_fut = [
                    (omega, tau, j)
                    for tau in range(1, H + 1)
                    for j in range(H + l - tau)
                    for omega in range(M)
                ]
                y_fut = m.addVars(idx_y_t_fut, lb=0, vtype=GRB.CONTINUOUS, name='y_fut')

            # ---------- 6. objective ----------
            imm_wait_cost = gp.quicksum(gp.quicksum(gamma ** k * w(k,i) for k in range(j + 1)) * a_t[j, i]
                                        for j in range(H + 1)
                                        for i in range(I))
            imm_overtime_cost = gp.quicksum(gamma ** j * O * y_t[j] for j in range(H + l))
            imm_cost = imm_wait_cost + imm_overtime_cost
            obj_func = imm_cost
            if self.is_myopic == False:
                fut_wait_cost = gp.quicksum(
                    gp.quicksum(gamma ** (k + tau) * w(k,i) for k in range(j + 1)) * a_fut[omega, tau, j, i]
                    for omega in range(M)
                    for tau in range(1, H + 1)
                    for j in range(H + 1 - tau)
                    for i in range(I)) / M
                fut_overtime_cost = gp.quicksum(
                        gamma ** (tau+j) * O * y_fut[omega, tau, j]
                        for omega in range(M)
                        for tau in range(1, H + 1)
                        for j in range(H + l - tau)) / M
                fut_cost = fut_wait_cost + fut_overtime_cost
                obj_func += fut_cost
            m.setObjective(obj_func, GRB.MINIMIZE)
            # ---------- 4. demand constraints ----------
            # demand for today
            m.addConstrs(
                (
                    gp.quicksum(a_t[j, i] for j in range(H + 1)) == delta_t[i]
                    for i in range(I)
                ),
                name="C1_demand_today",
            )
            if self.is_myopic == False:
                # demand for future
                m.addConstrs(
                    (
                        gp.quicksum(a_fut[omega, tau, j, i] for j in range(H + 1 - tau)) == delta[omega, tau, i]
                        for tau in range(1, H + 1)
                        for omega in range(M)
                        for i in range(I)
                    ),
                    name="C1_demand_future",
                )
            # booked appointment slots at the end of today
            m.addConstrs(
                (
                    z_bar_t[j] == z[j] + gp.quicksum(
                        gp.quicksum(a_t[min(k, H), i] * r[j - k, i] for k in range(max(j - l + 1, 0), min(j, H) + 1))
                        for i in range(I))
                    for j in range(H + l)
                ),
                name="C2_booked_slot_today"
            )
            if self.is_myopic == False:
                # booked appointment slots at the end of each future period
                m.addConstrs(
                    (
                        z_bar_fut[omega, tau, j] == (
                            z_bar_t[j + 1] if tau == 1 else z_bar_fut[omega, tau - 1, j + 1]) + gp.quicksum(
                            gp.quicksum(a_fut[omega, tau, min(k, H - tau), i] * r[j - k, i]
                                        for k in range(max(j - l + 1, 0), min(j, H - tau) + 1))
                            for i in range(I))
                        for tau in range(1, H + 1)
                        for j in range(H + l - tau)
                        for omega in range(M)
                    ),
                    name="C6_booked_slot_future"
                )
            # ---------- 5. capacity (overtime) ----------
            m.addConstrs(
                (
                    y_t[j] >= z_bar_t[j] - C
                    for j in range(H + l)
                ),
                name="C3_overtime",
            )
            if self.is_myopic == False:
                m.addConstrs(
                    (
                        y_fut[omega, tau, j] >= z_bar_fut[omega, tau, j] - C
                        for tau in range(1, H + 1)
                        for omega in range(M)
                        for j in range(H + l - tau)
                    ),
                    name="C3_overtime_future",
                )
            # ---------- 7. solve ----------
            m.optimize()
            '''
            print("imm_cost:", imm_cost.getValue())
            print("imm_wait_cost:", imm_wait_cost.getValue())
            print("imm_overtime_cost:", imm_overtime_cost.getValue())
            '''
            cur_mem = m.getAttr(GRB.Attr.MemUsed)  # current RAM in GB
            peak_mem = m.getAttr(GRB.Attr.MaxMemUsed)  # peak RAM in GB
            print(f"Memory now: {cur_mem:.2f} GB  (peak {peak_mem:.2f} GB)")
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
    from experiments import ExperimentConfig

    # 54946.988268116984
    config = ExperimentConfig.from_EJOR_case()
    env = config.env
    init_state = config.init_state
    t = 1
    agent = SAAdvanceFastAgent(env=env, discount_factor=env.discount_factor)
    agent.set_sample_paths(500)
    action, owvertime, opt_val = agent.solve(init_state, t)
    print("value function lower bound:", opt_val)

