from collections import defaultdict
from gurobipy import GRB
import numpy as np

from utils import get_solution_value, clean_value, acquire_grb_env


class InfiniteRTAgent:
    TOKEN_WAIT = 15
    def __init__(self, env, discount_factor, V=None, Q=None, grb_env=None):
        self.env = env
        self.discount_factor = discount_factor
        self.V = V
        self.Q = Q
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(float))
        if V is None:
            self.V = {}
        if grb_env is None:
            self.grb_env = acquire_grb_env({"Threads": 0}, verbose=False, wait=InfiniteRTAgent.TOKEN_WAIT)
        else:
            self.grb_env = grb_env

        self.state_var_counter = 0
        self.action_var_counter = 0

    def regular_first_overtime(self, state, advance_scheduling_decision):
        """Canonical regular-first overtime split for a scheduling decision:
        overtime only for the load beyond regular capacity (the same formula
        ``env.valid_actions`` uses). Repairs optimizer vertices that book
        overtime while regular capacity remains (value-function or penalty
        terms can make the two splits tie); the repair only reduces overtime,
        so it is always feasible and never increases the realized cost."""
        new_booking_slots = self.env.convert_action_to_booking_slots(advance_scheduling_decision)
        regular_bookings = np.asarray(state[0])
        return np.maximum(
            np.round(regular_bookings + new_booking_slots - self.env.regular_capacity).astype(int),
            0,
        )

    def get_solution(self, action_var, is_final=False):
        x_var, y_var = action_var
        if is_final:
            x = np.round(x_var.Xn).astype(int)
            y = np.round(y_var.Xn).astype(int)
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
    
    def get_state_var(self, model):
        self.state_var_counter += 1
        
        regular_booking_vars = model.addMVar(
            shape=self.env.planning_horizon, 
            vtype=GRB.CONTINUOUS, 
            lb=0.0, 
            name=f"u^{self.state_var_counter}"
        )
        
        overtime_vars = model.addMVar(
            shape=self.env.planning_horizon, 
            vtype=GRB.CONTINUOUS, 
            lb=0.0, 
            name=f"v^{self.state_var_counter}"
        )
        
        waitlist_vars = model.addMVar(
            shape=self.env.num_types, 
            vtype=GRB.CONTINUOUS, 
            lb=0.0, 
            name=f"w^{self.state_var_counter}"
        )
        
        return (regular_booking_vars, overtime_vars, waitlist_vars)
    
    def get_action_var(self, model, advance_scheduling_type):
        self.action_var_counter += 1
        
        # Creates a 2D matrix variable instantly
        advance_scheduling_decision_vars = model.addMVar(
            shape=(self.env.booking_window_size, self.env.num_types),
            vtype=advance_scheduling_type,
            lb=0.0,
            name=f"x^{self.action_var_counter}"
        )
        
        # Creates a 1D array variable instantly
        overtime_decision_vars = model.addMVar(
            shape=self.env.planning_horizon,
            vtype=advance_scheduling_type,
            lb=0.0,
            name=f"y^{self.action_var_counter}"
        )
        
        return (advance_scheduling_decision_vars, overtime_decision_vars)
    
    def get_next_state(self, model, state, action, new_arrival):
        # Calculate the transition (assuming this returns arrays/MLinExprs)
        next_state = self.env.get_next_state(
            state=state,
            action=action,
            new_arrival=new_arrival, 
            is_var=True
        )
        
        # Create new state variables (this is now blazing fast thanks to previous MVar changes)
        next_state_var = self.get_state_var(model=model)
        
        # Vectorized constraint generation
        # This loop only runs 3 times (for regular bookings, overtime, waitlist)
        for i, (mvar, expr) in enumerate(zip(next_state_var, next_state)):
            # addConstr (singular) with MVar array equality does the entire block at once
            model.addConstr(mvar == expr, name=f"link_state_{i}")
            
        return next_state_var
    
    def add_action_space_constraints(self, model, state_var, action_var):
        regular_booking_vars, overtime_booking_vars, waitlist_vars = state_var
        advance_scheduling_decision_vars, overtime_decision_vars = action_var
        
        # 1. Vectorized sum across the rows (axis=0). 
        # This creates a 1D MVar array of size (num_types,) and compares it to waitlist_vars
        model.addConstr(
            advance_scheduling_decision_vars.sum(axis=0) <= waitlist_vars,
            name="valid_advance_scheduling"
        )
        
        post_action_regular_booking_vars, post_action_overtime_vars, _ = self.env.post_action_state(
            state_var, action_var, is_var=True
        )
        
        # 2. Scalar broadcasting. 
        # Gurobi applies the <= operator to every element in the 1D MVar array automatically
        model.addConstr(
            post_action_regular_booking_vars <= self.env.regular_capacity,
            name="valid_post_action_regular_bookings"
        )
        
        # 3. Scalar broadcasting again.
        model.addConstr(
            post_action_overtime_vars <= self.env.overtime_capacity,
            name="valid_post_action_overtime_bookings"
        )
        
        new_booking_slots = self.env.convert_action_to_booking_slots(advance_scheduling_decision_vars)
        
        # 4. Element-wise comparison between two 1D MVar arrays.
        model.addConstr(
            new_booking_slots >= overtime_decision_vars,
            name="valid_new_appointment_slots"
        )
        
        return model

    def policy(self, state, t):
        action, obj_value, info = self.solve(state, t)
        return action