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

    def solve_original(self, state, t, x=None, action=None):
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
            #m.setParam("MIPFocus", 1)
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
            # current cumulative schedule
            b_bar_t = m.addVars(
                H + 1,
                I,
                lb=0,
                vtype=decision_variable_type,
                name="b_now",
            )
            idx_b_fut = [
                (omega, tau, j, i)
                for tau in range(1, H + 1)
                for j in range(H - tau + 1)
                for omega in range(M)
                for i in range(I)
            ]
            b_bar_fut = m.addVars(
                idx_b_fut,
                lb=0,
                vtype=decision_variable_type,
                name="b_fut",
            )
            # N-t+l-1 = H + l-1
            z_bar_t  = m.addVars(range(l), vtype=GRB.CONTINUOUS, name='z_t')
            idx_z_fut = [
                (omega, tau, j)
                for tau in range(1, H + 1)
                for j in range(l)
                for omega in range(M)
            ]
            z_bar_fut = m.addVars(idx_z_fut, lb=0, vtype=GRB.CONTINUOUS, name="z_fut")

            # overtime
            y = m.addVars(M, H + l, lb=0, vtype=GRB.CONTINUOUS, name="y")

            # ---------- 3. definitions ----------
            # today
            m.addConstrs(
                (b_bar_t[j, i] == b[j, i] + a_t[j, i] for j in range(H + 1) for i in range(I)),
                name="C1_schedule_today",
            )
            # tau ≥ 1 cascade
            m.addConstrs(
                (
                    b_bar_fut[omega, tau, j, i]
                    == (b_bar_t[j+1, i] if tau == 1 else b_bar_fut[omega, tau - 1, j + 1, i]) + a_fut[omega, tau, j, i]
                    for tau in range(1, H + 1)
                    for j in range(H + 1 - tau)
                    for omega in range(M)
                    for i in range(I)
                ),
                name="C2_schedule_future",
            )

            # ---------- 4. demand constraints ----------
            # demand for today
            m.addConstrs(
                (
                    gp.quicksum(a_t[j, i] for j in range(H + 1)) == delta_t[i]
                    for i in range(I)
                ),
                name="C3_demand_today",
            )
            # demand for future
            m.addConstrs(
                (
                    gp.quicksum(a_fut[omega, tau, j, i] for j in range(H - tau + 1)) == delta[omega, tau, i]
                    for tau in range(1, H + 1)
                    for omega in range(M)
                    for i in range(I)
                ),
                name="C4_demand_future",
            )
            # booked appointment slots at the end of today
            m.addConstrs(
                (
                    z_bar_t[j] == z[j] + gp.quicksum(b_bar_t[0,i] * r[j, i] for i in range(I)) 
                    for j in range(l)
                ),
                name="C5_booked_slot_today"
            )
            # booked appointment slots at the end of t+tau
            m.addConstrs(
                (
                    z_bar_fut[omega, tau, j] == (z_bar_t[j+1] if tau==1 else z_bar_fut[omega, tau-1, j + 1]) + gp.quicksum(b_bar_fut[omega, tau, 0, i] * r[j, i] for i in range(I))
                    for tau in range(1, H+1)
                    for j in range(l - 1)
                    for omega in range(M)
                ),
                name="C6_booked_slot_future"
            )
            m.addConstrs(
                (z_bar_fut[omega, tau, l-1] == gp.quicksum(b_bar_fut[omega, tau, 0, i] * r[l-1, i] for i in range(I))
                    for tau in range(1, H+1)
                    for omega in range(M)),
                name="C7_booked_slot_future_last_day"
            )
            # ---------- 5. capacity (overtime) ----------
            m.addConstrs(
                (
                    y[omega, tau] >= (z_bar_t[0] if tau == 0 else z_bar_fut[omega, tau, 0]) - C
                    for tau in range(H + 1)
                    for omega in range(M)
                ),
                name="C8_overtime_main",
            )
            m.addConstrs(
                (
                    y[omega, H + tau] >= (z_bar_t[tau] if H == 0 else z_bar_fut[omega, H, tau]) - C
                    for tau in range(1, l)
                    for omega in range(M)
                ),
                name="C8_overtime_tail",
            )

            # ---------- 6. objective ----------
            '''
            imm_cost = gp.quicksum(w[i] * gp.quicksum(b_bar_t[j, i] for j in range(H + 1)) for i in range(I)) + O * y[0,0]
            '''
            imm_wait_cost = gp.quicksum(w[i] * gp.quicksum(b_bar_t[j, i] for j in range(H + 1)) for i in range(I))
            imm_overtime_cost = O * y[0, 0]
            imm_cost = imm_wait_cost + imm_overtime_cost
            '''
            fut_cost = gp.quicksum(gamma ** tau * gp.quicksum(
                            w[i] * gp.quicksum(b_bar_fut[omega, tau, j, i] for j in range(H - tau + 1))
                            for i in range(I)) + O * y[omega, tau]
                            for tau in range(1, H + 1)
                            for omega in range(M))/M
            tail_cost = gp.quicksum(
                gp.quicksum(gamma ** (H + tau) * O * y[omega, H + tau] for tau in range(1, l)) for omega in
                range(M)) / M
            m.setObjective(imm_cost + fut_cost + tail_cost, GRB.MINIMIZE)
            '''
            fut_cost = gp.quicksum(gp.quicksum(gamma ** tau * gp.quicksum(w[i] * gp.quicksum(b_bar_fut[omega, tau, j, i] for j in range(H - tau + 1)) for i in range(I)) for tau in range(1, H + 1)) + gp.quicksum(gamma ** tau * O * y[omega, tau] for tau in range(1, H + l)) for omega in range(M))/M
            m.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)


            # ---------- 7. solve ----------
            m.setParam("Presolve", 2)
            m.setParam("Threads", 0)
            m.optimize()
            '''
            print("imm_cost:", imm_cost.getValue())
            print("imm_wait_cost:", imm_wait_cost.getValue())
            print("imm_overtime_cost:", imm_overtime_cost.getValue())
            print("fut_cost:", fut_cost.getValue())
            #print("b_bar_t:", m.getAttr("X", b_bar_t))
            #print('a_fut:', m.getAttr("X", a_fut))
            #print('y:', m.getAttr("X", y))
            #print('z_bar_t:', m.getAttr("X", z_bar_t))
            #print('z_bar_fut:', m.getAttr("X", z_bar_fut))
            '''
            cur_mem = m.getAttr(GRB.Attr.MemUsed)  # current RAM in GB
            peak_mem = m.getAttr(GRB.Attr.MaxMemUsed)  # peak RAM in GB
            print(f"Memory now: {cur_mem:.2f} GB  (peak {peak_mem:.2f} GB)")
            # ---------- 8. return ----------
            if m.Status == GRB.OPTIMAL:
                a_now = np.zeros((H + 1, I), dtype=int)
                for (j, i), v in m.getAttr("X", a_t).items():
                    a_now[j, i] = int(round(v))
                return a_now, y[0,0].X, m.ObjVal
            else:
                raise RuntimeError("Optimal solution not found")

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

        bookings, delta_t, future_schedule = state  # b shape = (H+1, I)
        delta = self.delta  # shape = (M, H, I)
        z = convet_state_to_booked_slots(bookings, future_schedule, self.env.treatment_pattern)
        decision_variable_type = GRB.INTEGER
        # ---------- model ----------
        with (gp.Model("SA_Advance", env=self.grb_env) as m):
            m.setParam("OutputFlag", 0)
            m.setParam("LogToConsole", 0)
            m.setParam("MIPFocus", 1)
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
            '''
            imm_cost = gp.quicksum(gp.quicksum(gamma**k * w[i] for k in range(j+1)) * a_t[j, i]
                                   for j in range(H+1)
                                   for i in range(I)) + O * y[0, 0]
            '''
            imm_wait_cost = gp.quicksum(gp.quicksum(gamma**k * w[i] for k in range(j+1)) * a_t[j, i]
                                   for j in range(H+1)
                                   for i in range(I))
            imm_overtime_cost = O * y[0, 0]
            imm_cost = imm_wait_cost + imm_overtime_cost
            '''
            fut_cost = gp.quicksum(gp.quicksum(gp.quicksum(gamma**k * w[i] for k in range(j+1)) * a_fut[omega, tau, j, i]
                                               for tau in range(1, H + 1)
                                               for j in range(H + 1 - tau)
                                               for i in range(I)) +
                                   gp.quicksum(gamma ** tau * O * y[omega, tau] for tau in range(1, H + l))
                                   for omega in range(M))/M
            '''
            fut_wait_cost = gp.quicksum(gp.quicksum(gamma**(k+tau) * w[i] for k in range(j+1)) * a_fut[omega, tau, j, i]
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
                name="C3_demand_today",
            )
            # demand for future
            m.addConstrs(
                (
                    gp.quicksum(a_fut[omega, tau, j, i] for j in range(H - tau + 1)) == delta[omega, tau, i]
                    for tau in range(1, H + 1)
                    for omega in range(M)
                    for i in range(I)
                ),
                name="C4_demand_future",
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
            # booked appointment slots at the end of t+tau
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
                    y[omega, tau] >= (z_bar_t[0] if tau == 0 else z_bar_fut[omega, tau, 0]) - C
                    for tau in range(H + 1)
                    for omega in range(M)
                ),
                name="C8_overtime_main",
            )
            m.addConstrs(
                (
                    y[omega, H + tau] >= (z_bar_t[tau] if H == 0 else z_bar_fut[omega, H, tau]) - C
                    for tau in range(1, l)
                    for omega in range(M)
                ),
                name="C8_overtime_tail",
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
            '''
            cur_mem = m.getAttr(GRB.Attr.MemUsed)  # current RAM in GB
            peak_mem = m.getAttr(GRB.Attr.MaxMemUsed)  # peak RAM in GB
            print(f"Memory now: {cur_mem:.2f} GB  (peak {peak_mem:.2f} GB)")
            # ---------- 8. return ----------
            if m.Status == GRB.OPTIMAL:
                a_now = np.zeros((H + 1, I), dtype=int)
                for (j, i), v in m.getAttr("X", a_t).items():
                    a_now[j, i] = int(round(v))
                return a_now, y[0,0].X, m.ObjVal
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
if __name__ =="__main__":
    from experiments.config import Config
    from environment import AdvanceSchedulingEnv, MultiClassPoissonArrivalGenerator
    from environment.utility import get_system_dynamic
    import time
    # 54946.988268116984

    config = Config.from_adjust_EJOR_case()
    env_params = config.env_params
    env = AdvanceSchedulingEnv(**env_params)
    init_state, info = env.reset(percentage_occupied=0.5)
    #print(init_state)
    agent = SAAdvanceAgent(env, discount_factor=env_params['discount_factor'])
    agent.set_sample_paths(500)
    #print("Solver 1 path:", agent.delta)
    start = time.time()
    action, overtime, obj_value = agent.solve_original(init_state, 1)
    print("time:", time.time() - start)
    print(action)
    print(obj_value)
    # Memory now: 2.78 GB  (peak 4.47 GB)



    config = Config.from_adjust_EJOR_case()
    env_params = config.env_params
    env = AdvanceSchedulingEnv(**env_params)
    init_state, info = env.reset(percentage_occupied=0.5)
    #print(init_state)
    agent = SAAdvanceAgent(env, discount_factor=env_params['discount_factor'])
    agent.set_sample_paths(500)
    #print("Solver 2 path:", agent.delta)
    start = time.time()
    action, overtime, obj_value = agent.solve(init_state, 1)
    print("time:", time.time() - start)
    print(str(action))
    print(obj_value)
    #45.80203080177307
    #[[2 0][0 1]]
    # understand why two solvers have different result.
    # a_fut: {(0, 1, 0, 0): 2.0, (0, 1, 0, 1): 0.0, (0, 1, 1, 0): 0.0, (0, 1, 1, 1): 0.0, (0, 1, 2, 0): 0.0, (0, 1, 2, 1): 0.0, (0, 2, 0, 0): 1.0, (0, 2, 0, 1): 1.0, (0, 2, 1, 0): 0.0, (0, 2, 1, 1): 0.0, (0, 3, 0, 0): 1.0, (0, 3, 0, 1): 3.0}
    # a_fut: {(0, 1, 0, 0): 2.0, (0, 1, 0, 1): 0.0, (0, 1, 1, 0): 0.0, (0, 1, 1, 1): 0.0, (0, 1, 2, 0): 0.0, (0, 1, 2, 1): 0.0, (0, 2, 0, 0): 1.0, (0, 2, 0, 1): 1.0, (0, 2, 1, 0): 0.0, (0, 2, 1, 1): 0.0, (0, 3, 0, 0): 1.0, (0, 3, 0, 1): 3.0}
    # z_bar_fut: {(0, 1, 0): 4.0, (0, 1, 1): 2.0, (0, 1, 2): 0.0, (0, 1, 3): 0.0, (0, 2, 0): 5.0, (0, 2, 1): 0.0, (0, 2, 2): 0.0, (0, 3, 0): 5.0, (0, 3, 1): 0.0}
    # z_bar_fut: {(0, 1, 0): 4.0, (0, 1, 1): 2.0, (0, 2, 0): 5.0, (0, 2, 1): 1.0, (0, 3, 0): 6.0, (0, 3, 1): 1.0}



