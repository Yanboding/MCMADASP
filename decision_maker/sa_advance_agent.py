import time
from collections import defaultdict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import iter_to_tuple

class SAAdvanceAgent:
    TOKEN_WAIT = 15
    def __init__(self, env, discount_factor, V=None, Q=None, sample_path_number=500, current_decision_var_type='integer', future_decision_var_type='continuous'):
        self.env = env
        self.discount_factor = discount_factor
        self.sample_path_number = sample_path_number
        self.current_decision_var_type = GRB.INTEGER if current_decision_var_type is None or current_decision_var_type =='integer' else GRB.CONTINUOUS
        self.future_decision_var_type = GRB.INTEGER if future_decision_var_type is None or future_decision_var_type =='integer' else GRB.CONTINUOUS
        delta = []
        for omega in range(self.sample_path_number):
            new_arrivals = self.env.reset_arrivals(1)
            delta.append(new_arrivals)
        self.delta = np.array(delta)
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

    def set_real_sample_paths(self, sample_paths):
        self.sample_path_number = len(sample_paths)
        self.delta = np.array(sample_paths)

    def set_sample_path(self, sample_path):
        self.sample_path_number = 1
        self.delta = np.array([sample_path])
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

    def solve(self, state, t, action=None):
        # ---------- shortcuts ----------
        N = self.env.decision_epoch
        I = self.env.num_types
        H = N - t  # remaining horizon
        M = self.sample_path_number
        gamma = self.discount_factor
        w = self.env.holding_cost
        r = np.asarray(self.env.treatment_pattern)  # (l, I)
        C = self.env.regular_capacity
        O = self.env.overtime_cost
        l = self.env.num_sessions

        z, delta_t = state  # b shape = (H+1, I)
        delta = self.delta  # shape = (M, H, I)
        # ---------- model ----------
        with (gp.Model("SA_Advance", env=self.grb_env) as m):
            #m.setParam("OutputFlag", 0)
            #m.setParam("LogToConsole", 0)
            #m.setParam("MIPFocus", 1)
            # ---------- 1. today’s increments ----------
            a_t = m.addVars(H + 1, I, vtype=self.current_decision_var_type, name="a_t")
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
            a_fut = m.addVars(idx_a_fut, lb=0, vtype=self.future_decision_var_type, name="a_fut")
            # N-t+l-1 = H + l-1
            z_bar_t  = m.addVars(H + l, vtype=GRB.CONTINUOUS, name='z_t')
            idx_z_fut = [
                (omega, tau, j)
                for tau in range(1, H + 1)
                for j in range(H + l - tau)
                for omega in range(M)
            ]
            z_bar_fut = m.addVars(idx_z_fut, lb=0, vtype=GRB.CONTINUOUS, name="z_fut")

            # overtime
            y = m.addVars(M, H + l, lb=0, vtype=GRB.CONTINUOUS, name="y")
            # ---------- 1. objective ----------
            imm_wait_cost = gp.quicksum(gp.quicksum(gamma**k * w(k, i) for k in range(j+1)) * a_t[j, i]
                                   for j in range(H+1)
                                   for i in range(I))
            imm_overtime_cost = O * y[0, 0]
            imm_cost = imm_wait_cost + imm_overtime_cost

            fut_wait_cost = gp.quicksum(gamma**tau * gp.quicksum(gamma**k * w(k, i) for k in range(j+1)) * a_fut[omega, tau, j, i]
                                        for omega in range(M)
                                        for tau in range(1, H + 1)
                                        for j in range(H + 1 - tau)
                                        for i in range(I))/M
            fut_overtime_cost = gp.quicksum(gamma ** tau * O * y[omega, tau]
                                            for omega in range(M)
                                            for tau in range(1, H + l))/M
            
            fut_cost = fut_wait_cost + fut_overtime_cost
            m.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)
            # ---------- 4. demand constraints ----------
            # demand for today
            m.addConstrs(
                (
                    gp.quicksum(a_t[j, i] for j in range(H + 1)) == delta_t[i]
                    for i in range(I)
                ),
                name="C1_demand_today",
            )
            # demand for future
            m.addConstrs(
                (
                    gp.quicksum(a_fut[omega, tau, j, i] for j in range(H - tau + 1)) == delta[omega, tau, i]
                    for tau in range(1, H + 1)
                    for omega in range(M)
                    for i in range(I)
                ),
                name="C2_demand_future",
            )
            # booked appointment slots at the end of today
            m.addConstrs(
                (
                    z_bar_t[j] == z[j] + gp.quicksum(
                        gp.quicksum(a_t[k, i] * r[j - k, i] for k in range(max(j - l + 1, 0), min(j, H) + 1))
                        for i in range(I))
                    for j in range(H + l)
                ),
                name="C3_booked_slot_today"
            )
            # booked appointment slots at the end of t+tau
            m.addConstrs(
                (
                    z_bar_fut[omega, tau, j] == (
                        z_bar_t[j + 1] if tau == 1 else z_bar_fut[omega, tau - 1, j + 1]) + gp.quicksum(
                        gp.quicksum(a_fut[omega, tau, k, i] * r[j - k, i]
                                    for k in range(max(j - l + 1, 0), min(j, H - tau) + 1))
                        for i in range(I))
                    for tau in range(1, H + 1)
                    for j in range(H + l - tau)
                    for omega in range(M)
                ),
                name="C4_booked_slot_future"
            )
            # ---------- 5. capacity (overtime) ----------
            m.addConstrs(
                (
                    y[omega, tau] >= (z_bar_t[0] if tau == 0 else z_bar_fut[omega, tau, 0]) - C
                    for tau in range(H + 1)
                    for omega in range(M)
                ),
                name="C5_overtime_main",
            )
            m.addConstrs(
                (
                    y[omega, H + tau] >= (z_bar_t[tau] if H == 0 else z_bar_fut[omega, H, tau]) - C
                    for tau in range(1, l)
                    for omega in range(M)
                ),
                name="C5_overtime_tail",
            )
            # ---------- 7. solve ----------
            m.setParam("Presolve", 2)
            m.setParam("Threads", 0)
            m.optimize()
            '''
            print("imm_cost:", imm_cost.getValue())
            print("imm_wait_cost:", imm_wait_cost.getValue())
            print("imm_overtime_cost:", imm_overtime_cost.getValue())
            print("fut_cost:", fut_cost.getValue())
            print('fut_wait_cost', fut_wait_cost.getValue())
            print('fut_overtime_cost', fut_overtime_cost.getValue())
            #print("a_t:",m.getAttr("X", a_t))
            #print('a_fut:', m.getAttr("X", a_fut))
            #print('y:', m.getAttr("X", y))
            #print('z_bar_t:', m.getAttr("X", z_bar_t))
            #print('z_bar_fut:', m.getAttr("X", z_bar_fut))
            
            cur_mem = m.getAttr(GRB.Attr.MemUsed)  # current RAM in GB
            peak_mem = m.getAttr(GRB.Attr.MaxMemUsed)  # peak RAM in GB
            print(f"Memory now: {cur_mem:.2f} GB  (peak {peak_mem:.2f} GB)")
            '''
            # ---------- 8. return ----------
            if m.Status == GRB.OPTIMAL:
                a_now = np.zeros((H + 1, I), dtype=int)
                for (j, i), v in m.getAttr("X", a_t).items():
                    a_now[j, i] = int(round(v))
                return a_now, y[0,0].X, m.ObjVal
            else:
                raise RuntimeError("Optimal solution not found")

    def policy(self, state, t):
        action, overtime, obj_value = self.solve(state, t)
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

if __name__ =="__main__":
    from experiments import get_config_by_type

    config = get_config_by_type('default', random_seed=None)
    # action_value_function_compare_experiment(config)
    agents = {
        'Hindsight Approx Policy': (SAAdvanceAgent, {'sample_path_number': 500})
    }
    env = config.env
    discount_factor = env.discount_factor
    agent = SAAdvanceAgent(env, discount_factor, **{'sample_path_number': 500})
    state = (np.array([6, 0]), np.array([1, 1, 4]))
    action, _, val = agent.solve(state, 2)
    print(action)

