from collections import defaultdict
from gurobipy import GRB
import numpy as np

from utils import get_solution_value, clean_value, acquire_grb_env


class InfiniteRTAgent:
    TOKEN_WAIT = 15
    def __init__(self, env, discount_factor, V=None, Q=None):
        self.env = env
        self.discount_factor = discount_factor
        self.V = V
        self.Q = Q
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(float))
        if V is None:
            self.V = {}
        self.grb_env = acquire_grb_env({"Threads": 0}, verbose=False, wait=InfiniteRTAgent.TOKEN_WAIT)

        self.state_var_counter = 0
        self.action_var_counter = 0

    def get_state_var(self, model):
        self.state_var_counter += 1
        regular_booking_vars = np.array([
            model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"u^{self.state_var_counter}_{m}") for m in range(self.env.planning_horizon)
        ])
        overtime_vars = np.array([
            model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"v^{self.state_var_counter}_{m}") for m in range(self.env.planning_horizon)
        ])
        waitlist_vars = np.array([
            model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"w^{self.state_var_counter}_{i}") for i in range(self.env.num_types)
        ])
        return (regular_booking_vars, overtime_vars, waitlist_vars)

    def get_action_var(self, model, advance_scheduling_type):
        self.action_var_counter += 1
        advance_scheduling_decision_vars = np.array([
            [model.addVar(vtype=advance_scheduling_type, lb=0, name=f"x^{self.action_var_counter}_{j},{i}") for i in range(self.env.num_types)]
            for j in range(self.env.booking_window_size)
        ])
        overtime_decision_vars = np.array(
            [model.addVar(vtype=advance_scheduling_type, lb=0, name=f"y^{self.action_var_counter}_{j}") for j in range(self.env.planning_horizon)]
        )
        return (advance_scheduling_decision_vars, overtime_decision_vars)

    def get_next_state(self, model, state, action, new_arrival):
        next_state = self.env.get_next_state(state=state,
                                             action=action,
                                             new_arrival=new_arrival, is_var=True)
        # try to use get_state_var to create new state vars to link
        next_state_var = self.get_state_var(model=model)
        for i, (vars, vals) in enumerate(zip(next_state_var, next_state)):
            model.addConstrs((vars[j] == vals[j] for j in range(len(vals))), name=f"link_state_{i}")
        return next_state_var

    def get_solution(self, action_var, is_final=False):
        x_var, y_var = action_var
        if is_final:
            x = np.array([[round(var.Xn) for var in row] for row in x_var]).astype(int)
            y = np.array([round(var.Xn) for var in y_var]).astype(int)
        else:
            x = get_solution_value(x_var).astype(float)
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
    
    @staticmethod
    def set_state(state_var, state):
        u_var, v_var, w_var = state_var
        u, v, w = state
        for uj_var, uj in zip(u_var, u):
            uj_var.lb = uj_var.ub = uj
        for vj_var, vj in zip(v_var, v):
            vj_var.lb = vj_var.ub = vj
        for wi_var, wi in zip(w_var, w):
            wi_var.lb = wi_var.ub = wi

    def add_action_space_constraints(self, model, state_var, action_var):
        regular_booking_vars, overtime_booking_vars, waitlist_vars = state_var
        advance_scheduling_decision_vars, overtime_decision_vars = action_var
        model.addConstrs(
            (advance_scheduling_decision_vars[:, i].sum() <= waitlist_vars[i]
             for i in range(self.env.num_types)),
            name=f"valid_advance_scheduling",
        )
        post_action_regular_booking_vars, post_action_overtime_vars, _ = self.env.post_action_state(state_var,
                                                                                                    action_var,
                                                                                                    is_var=True)
        model.addConstrs(
            (
                post_action_regular_booking_vars[m] <= self.env.regular_capacity
                for m in range(self.env.planning_horizon)
            ),
            name=f"valid_post_action_regular_bookings",
        )
        model.addConstrs(
            (
                post_action_overtime_vars[m] <= self.env.overtime_capacity
                for m in range(self.env.planning_horizon)
            ),
            name=f"valid_post_action_overtime_bookings",
        )
        new_booking_slots = self.env.convert_action_to_booking_slots(advance_scheduling_decision_vars)
        model.addConstrs(
            (
                new_booking_slots[m] >= overtime_decision_vars[m]
                for m in range(self.env.planning_horizon)
            ),
            name="valid_new_appointment_slots",
        )
        return model

    def policy(self, state, t):
        action, obj_value, info = self.solve(state, t)
        return action