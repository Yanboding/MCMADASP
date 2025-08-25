import time
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from utils import get_solution_value


class ALPEJORAgent:
    TOKEN_WAIT = 15
    def __init__(self, env, discount_factor, V=None, Q=None):
        self.env = env
        self.discount_factor = discount_factor
        self.V = V
        self.Q = Q
        self.is_trained = False
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(float))
        if V is None:
            self.V = {}
        self.grb_env = self._acquire_grb_env()

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

    def add_action_space_constraints(self, model, state_var, action_var):
        u_var, v_var, w_var = state_var
        x_var, y_var = action_var
        model.addConstrs(
            (x_var[:, i].sum() <= w_var[i]
             for i in range(self.env.num_types)),
            name="C1_valid_advance_schedule",
        )
        booking_slots = self.env.convert_action_to_booking_slots(x_var)
        post_decision_regular_hour_bookings = u_var + booking_slots - y_var
        model.addConstrs(
            (
                post_decision_regular_hour_bookings[m] <= self.env.regular_capacity
                for m in range(self.env.planning_horizon)
            ),
            name="C2_valid_appointment_slots",
        )
        model.addConstrs(
            (
                v_var[m] + y_var[m] <= self.env.overtime_capacity
                for m in range(self.env.planning_horizon)
            ),
            name="C3_valid_overtime_slots",
        )
        return model

    def get_action_var(self, model, state):
        x_var = np.array([
            [model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{j},{i}") for i in range(self.env.num_types)]
            for j in range(self.env.booking_window_size)
        ])
        y_var = np.array(
            [model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y_{j}") for j in range(self.env.planning_horizon)]
        )
        action_var = (x_var, y_var)
        self.add_action_space_constraints(model, state, action_var)
        return action_var

    def set_action(self, action_var, action):
        x_var, y_var = action_var
        x, y = action
        for i, row in enumerate(x_var):
            for j, var in enumerate(row):
                var.lb = var.ub = x[i][j]
        for j, var in enumerate(y_var):
            var.lb = var.ub = y[j]

    def get_state_var(self, model):
        maximum_arrival = self.env.arrival_generator.maximum_arrival
        regular_hour_booking_vars = np.array([model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.env.regular_capacity, name=f'u_{j}') for j in range(self.env.planning_horizon)])
        regular_hour_booking_vars[-1].lb = regular_hour_booking_vars[-1].ub = 0
        overtime_booking_vars = np.array(
            [model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.env.regular_capacity, name=f'v_{j}') for j in
             range(self.env.planning_horizon)])
        overtime_booking_vars[-1].lb = overtime_booking_vars[-1].ub = 0
        waitlist_vars = np.array([model.addVar(vtype=GRB.INTEGER, lb=0, ub=maximum_arrival, name=f'w_{i}') for i in range(self.env.num_types)])
        return (regular_hour_booking_vars, overtime_booking_vars, waitlist_vars)

    def get_candidate(self, state_var, action_var):
        regular_hour_booking_vars, overtime_booking_vars, waitlist_vars = state_var
        advance_scheduling_decision_vars, overtime_decision_vars = action_var
        regular_hour_bookings = get_solution_value(regular_hour_booking_vars).astype(float)
        overtime_bookings = get_solution_value(overtime_booking_vars).astype(float)
        waitlist = get_solution_value(waitlist_vars).astype(int)
        advance_scheduling_decision = get_solution_value(advance_scheduling_decision_vars).astype(int)
        overtime_decision = get_solution_value(overtime_decision_vars).astype(float)
        return ((regular_hour_bookings, overtime_bookings, waitlist), (advance_scheduling_decision, overtime_decision))

    def get_coefficients(self, solution):
        it = iter(solution)
        W_0 = float(next(it))
        U = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        V = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        W = np.array([float(next(it)) for _ in range(self.env.num_types)])
        return W_0, U, V, W

    def get_action_solution(self, action_var):
        advance_scheduling_decision_vars, overtime_decision_vars = action_var
        advance_scheduling_decision = get_solution_value(advance_scheduling_decision_vars).astype(int)
        overtime_decision = get_solution_value(overtime_decision_vars).astype(float)
        return (advance_scheduling_decision, overtime_decision)

    def get_approx_value_fn(self, state, W_0, U, V, W):
        bookings, overtimes, waitlist = state
        return W_0 + np.dot(U, bookings) + np.dot(V, overtimes) + np.dot(W, waitlist)

    def solve(self, state, action=None):
        # ---------- shortcuts ----------
        N = self.env.decision_epoch
        gamma = self.discount_factor

        mu = self.env.arrival_generator.mean_by_type
        # assume I know the
        with (gp.Model("ALP_Advance", env=self.grb_env) as m):
            # m.setParam("OutputFlag", 0)
            # m.setParam("LogToConsole", 0)
            # m.setParam("MIPFocus", 1)
            # ---------- 1. today’s increments ----------
            action_var = self.get_action_var(m, state)
            if action is not None:
                self.set_action(action_var=action_var, action=action)
            # ---------- 1. objective ----------
            imm_cost = self.env.cost_fn(state, action_var)
            new_state_var = self.env.get_next_state(state=state,
                                                    action=action_var,
                                                    new_arrival=self.env.arrival_generator.mean_by_type,
                                                    is_var=True)
            fut_cost = gamma * self.get_approx_value_fn(model=m,
                                                         state=new_state_var,
                                                         W_0=self.W_0,
                                                         U=self.U,
                                                         W=self.W)
            m.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)
            # ---------- 7. solve ----------
            m.setParam("Presolve", 2)
            m.setParam("Threads", 0)
            m.optimize()
            m.write('column_solver.lp')
            print('imm_cost:', imm_cost.getValue())
            print('future_cost:', fut_cost.getValue())
            get_val = np.vectorize(lambda e: e.getValue())
            regular_hour_bookings = get_val(new_state_var[0])
            info = {'W_0': self.W_0,
                    'new_state': regular_hour_bookings,
                    'Uu': np.dot(self.U, regular_hour_bookings),
                    'Wmu': np.dot(self.W, mu)}
            # ---------- 8. return ----------
            if m.Status == GRB.OPTIMAL:
                action = self.get_action_solution(action_var)
                return action, m.ObjVal, info
            else:
                raise RuntimeError("Optimal solution not found")

    def policy(self, state):
        action, obj_value, info = self.solve(state)
        return action