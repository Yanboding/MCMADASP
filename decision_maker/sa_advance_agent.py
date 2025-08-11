import copy
import time
from collections import defaultdict

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils import iter_to_tuple, get_solution_value


class SAAdvanceAgent:
    TOKEN_WAIT = 15
    def __init__(self, env, discount_factor, V=None, Q=None, sample_path_number=500, current_decision_var_type='integer', future_decision_var_type='continuous', is_myopic=False):
        self.env = env
        self.discount_factor = discount_factor
        self.sample_path_number = sample_path_number
        self.current_decision_var_type = GRB.INTEGER if current_decision_var_type is None or current_decision_var_type =='integer' else GRB.CONTINUOUS
        self.future_decision_var_type = GRB.INTEGER if future_decision_var_type is None or future_decision_var_type =='integer' else GRB.CONTINUOUS
        self.is_myopic = is_myopic
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

    def solve_deprecate(self, state, t, action=None):
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
            if not self.is_myopic:
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
            if not self.is_myopic:
                idx_z_fut = [
                    (omega, tau, j)
                    for tau in range(1, H + 1)
                    for j in range(H + l - tau)
                    for omega in range(M)
                ]
                z_bar_fut = m.addVars(idx_z_fut, lb=0, vtype=GRB.CONTINUOUS, name="z_fut")

            # overtime
            y = m.addVar()
            y = m.addVars(M, H + l, lb=0, vtype=GRB.CONTINUOUS, name="y")
            # ---------- 1. objective ----------
            imm_wait_cost = gp.quicksum(gp.quicksum(gamma**k * w(k, i) for k in range(j+1)) * a_t[j, i]
                                   for j in range(H+1)
                                   for i in range(I))
            imm_overtime_cost = O * y[0, 0]
            obj_func = imm_wait_cost + imm_overtime_cost
            if not self.is_myopic:
                fut_wait_cost = gp.quicksum(gamma**tau * gp.quicksum(gamma**k * w(k, i) for k in range(j+1)) * a_fut[omega, tau, j, i]
                                            for omega in range(M)
                                            for tau in range(1, H + 1)
                                            for j in range(H + 1 - tau)
                                            for i in range(I))/M
                fut_overtime_cost = gp.quicksum(gamma ** tau * O * y[omega, tau]
                                                for omega in range(M)
                                                for tau in range(1, H + l))/M

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
            if not self.is_myopic:
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
            if not self.is_myopic:
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
                    y[omega, 0] >= z_bar_t[0] - C
                    for omega in range(M)
                ),
                name="C5_overtime_main",
            )
            if not self.is_myopic:
                m.addConstrs(
                    (
                        y[omega, tau] >= z_bar_fut[omega, tau, 0] - C
                        for tau in range(1, H + 1)
                        for omega in range(M)
                    ),
                    name="C5_overtime_main_future",
                )
            # H == 0
            m.addConstrs(
                (
                    y[omega, tau] >= z_bar_t[tau] - C
                    for tau in range(1, l)
                    for omega in range(M)
                ),
                name="C5_overtime_tail",
            )
            if not self.is_myopic and H != 0:
                # H != 0
                m.addConstrs(
                    (
                        y[omega, H + tau] >= z_bar_fut[omega, H, tau] - C
                        for tau in range(1, l)
                        for omega in range(M)
                    ),
                    name="C5_overtime_tail_future",
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
                return a_now, m.ObjVal
            else:
                raise RuntimeError("Optimal solution not found")

    def solve(self, state, t, action=None):
        # ---------- shortcuts ----------
        N = self.env.decision_epoch
        I = self.env.num_types
        H = N - t  # remaining horizon
        M = self.sample_path_number
        gamma = self.discount_factor

        bookings, delta_t = state  # b shape = (H+1, I)
        delta = self.delta  # shape = (M, H, I)
        # ---------- model ----------
        with (gp.Model("SA_Advance", env=self.grb_env) as m):
            # m.setParam("OutputFlag", 0)
            # m.setParam("LogToConsole", 0)
            # m.setParam("MIPFocus", 1)
            # ---------- 1. today’s increments ----------
            action_t = np.array([
                [m.addVar(vtype=self.current_decision_var_type, lb=0, name=f"A^{t},{j},{i}") for i in range(I)]
                for j in range(H + 1)
            ])
            if action is not None:
                for j in range(H + 1):
                    for i in range(I):
                        action_t[j, i].lb = action_t[j, i].ub = int(action[j, i])
            # demand for today
            m.addConstrs(
                (
                    action_t[:, i].sum() == delta_t[i]
                    for i in range(I)
                ),
                name="C1_demand_today",
            )
            # ---------- 1. objective ----------
            obj_func = self.cost_fn(m, state, action_t, t)
            if not self.is_myopic:
                prev_state_scenario = [state for _ in range(M)]
                prev_action_scenario = [action_t for _ in range(M)]
                fut_cost = 0
                for tau in range(1, H+1):
                    state_scenario = []
                    action_scenario = []
                    for omega in range(M):
                        action_var = np.array([
                                                [m.addVar(vtype=self.future_decision_var_type, lb=0, name=f"A^{t}_{omega},{j},{i}")
                                                 for i in range(I)]
                                                for j in range(H - tau + 1)
                                            ])
                        m.addConstrs(
                            (
                                action_var[:, i].sum() == delta[omega, tau, i]
                                for i in range(I)
                            ),
                            name="C2_demand_future",
                        )
                        state = self.get_next_state(prev_state_scenario[omega], prev_action_scenario[omega], delta[omega, tau])
                        fut_cost += gamma ** tau * self.cost_fn(m, state, action_var, t+tau)
                        state_scenario.append(state)
                        action_scenario.append(action_var)
                    prev_state_scenario = state_scenario
                    prev_action_scenario = action_scenario
                obj_func += fut_cost / M
            m.setObjective(obj_func, GRB.MINIMIZE)
            # ---------- 7. solve ----------
            m.setParam("Presolve", 2)
            m.setParam("Threads", 0)
            m.optimize()
            # ---------- 8. return ----------
            if m.Status == GRB.OPTIMAL:
                action = get_solution_value(action_t).astype(int)
                return action, m.ObjVal
            else:
                raise RuntimeError("Optimal solution not found")
    def cost_fn(self, model, state, action, t):
        bookings, _ = state
        waiting_cost = sum(
            sum(self.discount_factor ** k * self.env.holding_cost(k, i) for k in range(j + 1)) * action[j, i]
            for j in range(len(action))
            for i in range(len(action[0])))
        new_bookings = bookings + self.env.convert_action_to_booking_slots(action)
        overtime_hours = np.array([model.addVar(name="overtime_hours", lb=0) for _ in range(len(new_bookings))])
        overtime_cost = self.env.overtime_cost * overtime_hours[0]
        model.addConstr(overtime_hours[0] >= (new_bookings[0] * self.env.duration - self.env.regular_capacity),
                        name="overtime_0")
        if t == self.env.decision_epoch:
            # Only consider the tail overtime if we're at the last decision epoch
            for k in range(1, len(new_bookings)):
                overtime_cost += self.discount_factor ** k * self.env.overtime_cost * overtime_hours[k]
                model.addConstr(overtime_hours[k] >= (new_bookings[k] * self.env.duration - self.env.regular_capacity),
                                name=f"overtime_{k}")
        return waiting_cost + overtime_cost

    def get_next_state(self, state, action, delta):
        bookings, _ = state
        new_bookings = self.env.get_next_bookings(bookings, action)
        return (new_bookings, delta)

    def policy(self, state, t):
        action, obj_value = self.solve(state, t)
        return action

if __name__ =="__main__":
    from experiments import get_config_by_type

    config = get_config_by_type('base_case')
    env = config.env
    discount_factor = env.discount_factor
    agent = SAAdvanceAgent(env, discount_factor, **{'sample_path_number': 100, 'is_myopic':False})
    action, val = agent.solve_deprecate(config.init_state, 1)
    print(action, val)
    action1, val1 = agent.solve(config.init_state, 1)
    print(action1, val1)

