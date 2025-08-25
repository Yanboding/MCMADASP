import time
from collections import defaultdict
import gurobipy as gp
from gurobipy import GRB
import numpy as np

from utils import get_solution_value


class ALPAgent:
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
        post_decision_regular_hour_bookings = u_var + booking_slots - y_var
        model.addConstrs(
            (
                post_decision_regular_hour_bookings[m] <= self.env.regular_capacity
                for m in range(P + 1)
            ),
            name="C2_valid_appointment_slots",
        )
        model.addConstrs(
            (
                post_decision_regular_hour_bookings[m] >= 0
                for m in range(P + 1)
            ),
            name="C3_valid_appointment_slots",
        )
        return model

    def get_action_var(self, model, state, t, tau):
        H = self.env.decision_epoch - t - tau
        P = self.env.planning_horizon - t - tau
        I = self.env.num_types
        x_var_t = np.array([
            [model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x^{t+tau},{j},{i}") for i in range(I)]
            for j in range(H + 1)
        ])
        y_var_t = np.array(
            [model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"y^{t+tau},{j}") for j in range(P + 1)]
        )
        action_var = (x_var_t, y_var_t)
        self.add_action_space_constraints(model, state, action_var, t, tau)
        return (x_var_t, y_var_t)

    def set_action(self, action_var, action):
        x_var, y_var = action_var
        x, y = action
        for i, row in enumerate(x_var):
            for j, var in enumerate(row):
                var.lb = var.ub = x[i][j]
        for j, var in enumerate(y_var):
            var.lb = var.ub = y[j]

    def get_state_var(self, model, t):
        H = self.env.decision_epoch - t
        L = self.env.num_sessions
        I = self.env.num_types
        maximum_arrival = self.env.arrival_generator.maximum_arrival
        regular_hour_booking_vars = np.array([model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=self.env.regular_capacity, name=f'u^{t}_{j}') for j in range(H + L)])
        waitlist_vars = np.array([model.addVar(vtype=GRB.INTEGER, lb=0, ub=maximum_arrival, name=f'w^{t}_{i}') for i in range(I)])
        return (regular_hour_booking_vars, waitlist_vars)

    def get_candidate(self, state_var, action_var, t):
        regular_hour_booking_vars, waitlist_vars = state_var
        advance_scheduling_decision_vars, overtime_decision_vars = action_var
        regular_hour_bookings = get_solution_value(regular_hour_booking_vars).astype(float)
        waitlist = get_solution_value(waitlist_vars).astype(int)
        advance_scheduling_decision = get_solution_value(advance_scheduling_decision_vars).astype(int)
        overtime_decision = get_solution_value(overtime_decision_vars).astype(float)
        return ((regular_hour_bookings, waitlist), (advance_scheduling_decision, overtime_decision), t)

    def get_coefficients(self, solution):
        it = iter(solution)
        W_0 = {k: next(it) for k in range(1, self.env.decision_epoch + 1)}
        U = {k: [next(it) for j in range(self.env.planning_horizon - k + 1)]
             for k in range(1, self.env.decision_epoch + 1)}
        W = {k: [next(it) for i in range(self.env.num_types)] for k in range(1, self.env.decision_epoch + 1)}
        return W_0, U, W

    def get_action_solution(self, action_var):
        advance_scheduling_decision_vars, overtime_decision_vars = action_var
        advance_scheduling_decision = get_solution_value(advance_scheduling_decision_vars).astype(int)
        overtime_decision = get_solution_value(overtime_decision_vars).astype(float)
        return (advance_scheduling_decision, overtime_decision)

    def get_approx_value_fn(self, model, state, t, W_0, U, W):
        """
        Returns a gp.LinExpr for the approximate value:
            W_0[t] + sum_j U[t][j] * u[j] + sum_i W[t][i] * w[i]
        and adds simple non-negativity constraints on those contributions.
        """
        u, w = state  # arrays/MVars of gp.Var
        n_u = len(u)
        n_w = len(w)

        # Build linear expressions with Gurobi, not NumPy
        expr_u = gp.quicksum(U[t][j] * u[j] for j in range(n_u))
        expr_w = gp.quicksum(W[t][i] * w[i] for i in range(n_w))
        approximate_V = gp.LinExpr(W_0[t]) + expr_u + expr_w

        return approximate_V

    def solve(self, state, t, action=None):
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
            action_var = self.get_action_var(m, state, t, 0)
            if action is not None:
                self.set_action(action_var=action_var, action=action)
            # ---------- 1. objective ----------
            imm_cost = self.env.cost_fn(state, action_var, t)
            fut_cost = 0
            if t < N:
                new_state_var = self.env.get_next_state(state=state,
                                                       action=action_var,
                                                       new_arrival=self.env.arrival_generator.mean_by_type,
                                                       is_var=True)
                fut_cost += gamma * self.get_approx_value_fn(model=m,
                                                             state=new_state_var,
                                                             t=t+1,
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
            if t < N:
                print('future_cost:', fut_cost.getValue())
            else:
                print('future_cost:', fut_cost)
            if t < N:
                get_val = np.vectorize(lambda e: e.getValue())
                regular_hour_bookings = get_val(new_state_var[0])
                info = {'W_0': self.W_0[t + 1],
                        'new_state': regular_hour_bookings,
                        'Uu': np.dot(self.U[t+1], regular_hour_bookings),
                        'Wmu': np.dot(self.W[t], mu)}
            else:
                info = {}
            # ---------- 8. return ----------
            if m.Status == GRB.OPTIMAL:
                action = self.get_action_solution(action_var)
                return action, m.ObjVal, info
            else:
                raise RuntimeError("Optimal solution not found")

    def policy(self, state, t):
        action, obj_value, info = self.solve(state, t)
        return action