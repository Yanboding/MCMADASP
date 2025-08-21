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
        self.grb_env = self._acquire_grb_env()
    
    def set_sample_paths(self, sample_path_number):
        self.sample_path_number = sample_path_number
        delta = []
        for omega in range(self.sample_path_number):
            new_arrivals = self.env.reset_arrivals(1)
            delta.append(new_arrivals)
        self.delta = np.array(delta)
        print(self.delta)

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

    def add_action_space_constraints(self, model, state_var, action_var, t, tau):
        P = self.env.planning_horizon - t - tau
        I = self.env.num_types
        u_var, w_var = state_var
        x_var, y_var = action_var
        model.addConstrs(
            (x_var[:, i].sum() == w_var[i]
             for i in range(I)),
            name="C1_valid_advance_schedule",
        )
        booking_slots = self.env.convert_action_to_booking_slots(x_var)
        model.addConstrs(
            (
                u_var[m] + booking_slots[m] <= self.env.regular_capacity + y_var[m]
                for m in range(P + 1)
            ),
            name="C2_valid_appointment_slots",
        )
        return model

    def get_action_var(self, model, state, t, tau):
        H = self.env.decision_epoch - t - tau
        P = self.env.planning_horizon - t - tau
        I = self.env.num_types
        if tau == 0:
            decision_type = self.current_decision_var_type
        else:
            decision_type = self.future_decision_var_type
        x_var_t = np.array([
            [model.addVar(vtype=decision_type, lb=0, name=f"x^{t+tau},{j},{i}") for i in range(I)]
            for j in range(H + 1)
        ])
        y_var_t = np.array(
            [model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y^{t+tau},{j}") for j in range(P + 1)]
        )
        return (x_var_t, y_var_t)

    def get_solution(self, action_var):
        x_var, y_var = action_var
        x = get_solution_value(x_var).astype(int)
        y = get_solution_value(y_var).astype(float)
        return (x, y)

    def set_action(self, action_var, action):
        x_var, y_var = action_var
        x, y = action
        for i, row in enumerate(x_var):
            for j, var in enumerate(row):
                var.lb = var.ub = x[i][j]
        for j, var in enumerate(y_var):
            var.lb = var.ub = y[j]

    def solve(self, state, t, action=None):
        H = self.env.decision_epoch - t
        # ---------- model ----------
        with (gp.Model("SA_Advance", env=self.grb_env) as m):
            # ---------- 1. today’s increments ----------
            action_var_t = self.get_action_var(model=m, state=state, t=t, tau=0)
            if action is not None:
                self.set_action(action_var=action_var_t, action=action)
            # add action constraint
            self.add_action_space_constraints(model=m, state_var=state, action_var=action_var_t, t=t, tau=0)
            # ---------- 1. objective ----------
            obj_func = self.env.cost_fn(state, action_var_t, t)
            if not self.is_myopic:
                prev_state_scenario = [state for _ in range(self.sample_path_number)]
                prev_action_scenario = [action_var_t for _ in range(self.sample_path_number)]
                fut_cost = 0
                for tau in range(1, H+1):
                    state_scenario = []
                    action_scenario = []
                    for omega in range(self.sample_path_number):
                        state_t_tau = self.env.get_next_state(state=prev_state_scenario[omega],
                                                              action=prev_action_scenario[omega],
                                                              new_arrival=self.delta[omega, tau], is_var=True)
                        action_var_t_tau = self.get_action_var(model=m, state=state_t_tau, t=t, tau=tau)
                        self.add_action_space_constraints(model=m, state_var=state_t_tau, action_var=action_var_t_tau, t=t, tau=tau)
                        fut_cost += self.env.discount_factor ** tau * self.env.cost_fn(state_t_tau, action_var_t_tau, t+tau)
                        state_scenario.append(state_t_tau)
                        action_scenario.append(action_var_t_tau)
                    prev_state_scenario = state_scenario
                    prev_action_scenario = action_scenario
                obj_func += fut_cost / self.sample_path_number
            m.setObjective(obj_func, GRB.MINIMIZE)
            # ---------- 7. solve ----------
            m.setParam("Presolve", 2)
            m.setParam("Threads", 0)
            m.optimize()
            cur_mem = m.getAttr(GRB.Attr.MemUsed)  # current RAM in GB
            peak_mem = m.getAttr(GRB.Attr.MaxMemUsed)  # peak RAM in GB
            print(f"Memory now: {cur_mem:.2f} GB  (peak {peak_mem:.2f} GB)")
            # ---------- 8. return ----------
            if m.Status == GRB.OPTIMAL:
                action = self.get_solution(action_var_t)
                return action, m.ObjVal, {}
            else:
                m.write('not_optimal.lp')
                raise RuntimeError("Optimal solution not found")

    def policy(self, state, t):
        action, obj_value, info = self.solve(state, t)
        return action

if __name__ =="__main__":
    from experiments import get_config_by_type

    config = get_config_by_type('adv_default')
    env = config.env
    discount_factor = env.discount_factor
    agent = SAAdvanceAgent(env, discount_factor, **{'sample_path_number': 500, 'is_myopic':False})
    #action, val = agent.solve_deprecate(config.init_state, 1)
    x = np.array([[3, 2], [0,1], [0,0]])
    y = np.array([5.5,0,0])
    action = (x, y)
    action1, val1, info = agent.solve(config.init_state, 1, action=action)
    print(action1)
    print(val1)
    # 524.608121089574

