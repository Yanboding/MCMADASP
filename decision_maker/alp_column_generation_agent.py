import time
from collections import defaultdict

import numpy as np
from scipy.stats import uniform
import gurobipy as gp
from gurobipy import GRB

from utils import iter_to_tuple, get_solution_value, ColumnGenerationSolver


def make_index_counter(start=0):
    index = start
    while True:
        yield index
        index += 1

class ALPAgent:
    TOKEN_WAIT = 15
    def __init__(self, env, discount_factor, V=None, Q=None, exogenous_state_distribution_by_time=None):
        self.env = env
        self.discount_factor = discount_factor
        self.V = V
        self.Q = Q
        self.beta = exogenous_state_distribution_by_time
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(float))
        if V is None:
            self.V = {}
        self.action_map = {}
        self.booking_weights = [1] * (self.env.decision_epoch+self.env.num_sessions)
        self.waitlist_weights = [0] * self.env.num_types
        self.grb_env = self._acquire_grb_env()

        self.cg_solver = ColumnGenerationSolver(master_builder=self.master_builder,
                                                pricing_callback=self.pricing_callback,
                                                initial_columns=self.generate_initial_columns(),
                                                get_constr_coefficients=self.get_constr_coefficients,
                                                get_obj_coefficient=self.get_obj_coefficient)
        self.cg_solver.solve()

        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        final_duals = [c.Pi for c in self.cg_solver.master_model.getConstrs()]
        counter = make_index_counter(0)
        self.W_0 = {k: final_duals[next(counter)] for k in range(1, N + 1)}
        self.Z = {k: [final_duals[next(counter)] for j in range(N + l - k)] for k in range(1, N + 1)}
        self.W = {k: [final_duals[next(counter)] for i in range(I)] for k in range(1, N + 1)}


    def master_builder(self):
        '''
        Total number of constraints: N + N+l-1 + N+l -2 ... + l + N*I
        '''
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        maximum_arrival = self.env.arrival_generator.maximum_arrival
        z_max = 11
        E_z_beta = {t: [uniform(loc=0, scale=z_max).mean()]*(N+l-t) for t in
                    range(1, N + 1)}
        E_delta_beta = {t:  [uniform(loc=0, scale=maximum_arrival).mean()]* I for t in range(1, N+1)}
        master_model = gp.Model("MasterRMP")
        master_model.ModelSense = GRB.MINIMIZE
        master_model.setParam('OutputFlag', 0)
        master_model.addConstrs(
            (
                    gp.LinExpr() == 1
                    for t in range(1, N+1)
             ),
            name="constr_W_0")

        master_model.addConstrs(
            (
                gp.LinExpr() >= E_z_beta[t][j]
                for t in range(1, N + 1)
                for j in range(N + l - t)
            ),
            name="constr_Z")

        master_model.addConstrs(
            (
                gp.LinExpr() >= E_delta_beta[t][i]
                for t in range(1, N + 1)
                for i in range(I)
            ),
            name="constr_W")
        master_model.update()
        return master_model

    def pricing_callback(self, duals):
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        mu = self.env.arrival_generator.mean_by_type
        gamma = self.env.discount_factor
        counter = make_index_counter(0)
        W_0 = {k:duals[next(counter)] for k in range(1, N+1)}
        Z = {k: [duals[next(counter)] for j in range(N + l - k)] for k in range(1, N + 1)}
        W = {k: [duals[next(counter)] for i in range(I)] for k in range(1, N + 1)}
        with (gp.Model("PricingProblem", env=self.grb_env) as pricing_model):
            reduce_cost = pricing_model.addVar(name="reduce_cost")
            t_var = pricing_model.addVar(lb=1, ub=N, vtype=GRB.INTEGER, name="t")
            # 1. One-hot indicator variables
            y = np.array([pricing_model.addVar(vtype=GRB.BINARY, name=f"is_t_{k}") for k in range(1, N+1)])
            # 2. Exactly one active
            pricing_model.addConstr(y.sum() == 1)
            # 3. Link t to y
            pricing_model.addConstr(t_var == sum((k + 1) * y[k] for k in range(N)))
            costs = []
            state_action_pairs = {}
            for t in range(1, N+1):
                H = N - t
                action_var = np.array([[pricing_model.addVar(lb=0, vtype=GRB.INTEGER, name=f"A^{t}_{j},{i}")
                                     for i in range(I)] for j in range(H + 1)])
                bookings_var = np.array([pricing_model.addVar(vtype=GRB.CONTINUOUS, name=f"z^{t}_{j}") for j in range(H+l)])
                new_bookings_var = self.next_booking(bookings_var, action_var)
                waitlist_var = np.array([pricing_model.addVar(vtype=GRB.INTEGER, name=f"delta_t_{i}") for i in range(I)])
                approx_V = W_0[t] + (Z[t] * bookings_var).sum() + (W[t] * waitlist_var).sum()
                state_var = (bookings_var, waitlist_var)
                state_action_pairs[t] = (state_var, action_var)
                if t < N:
                    approx_V_old = (self.cost_fn(pricing_model, (bookings_var, waitlist_var), action_var, t) +
                            gamma * (W_0[t + 1] + (Z[t + 1] * new_bookings_var).sum() + (W[t + 1] * mu).sum()))
                else:
                    approx_V_old = self.cost_fn(pricing_model, (bookings_var, waitlist_var), np.array([waitlist_var]), N)
                costs.append(approx_V_old - approx_V)
            costs = np.array(costs)
            pricing_model.addConstr(reduce_cost == (y * costs).sum())
            # Example objective
            pricing_model.setObjective(reduce_cost, GRB.MINIMIZE)
            pricing_model.optimize()
            if pricing_model.Status == GRB.OPTIMAL:
                min_Obj = pricing_model.ObjVal
                t = int(t_var.X)
                action = get_solution_value(state_action_pairs[t][1]).astype(int)
                bookings = get_solution_value(state_action_pairs[t][0][0])
                waitlist = get_solution_value(state_action_pairs[t][0][1]).astype(int)
                state = (bookings, waitlist)
                return (state, action, t), min_Obj
        return None, None

    def next_booking(self, bookings, action):
        new_bookings = bookings + self.convert_action_to_booking_slots(action)
        return new_bookings[1:]

    def convert_action_to_booking_slots(self, action):
        appointment_slots = action @ self.env.treatment_pattern.T
        N, P = appointment_slots.shape
        total_len = len(action) + self.env.num_sessions - 1
        booked_slots = np.zeros(total_len, dtype=appointment_slots.dtype)

        # 2.  Vectorised diagonal add:
        #     element (i,j) in `appointment_slots` goes to position i+j in `booked_slots`.
        idx = np.arange(P) + np.arange(N)[:, None]  # shape (N,P)
        np.add.at(booked_slots, idx.ravel(), appointment_slots.ravel())
        return booked_slots

    def cost_fn(self, model, state, action, t):
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        bookings, _ = state
        waiting_cost = sum(sum(self.discount_factor ** k * self.env.holding_cost(k, i) for k in range(j + 1)) * action[j, i]
                           for j in range(N-t+l)
                           for i in range(I))
        new_bookings = bookings + self.convert_action_to_booking_slots(action)
        overtime_hours = np.array([model.addVar(name="overtime_hours", lb=0) for _ in range(len(new_bookings))])
        overtime_cost = self.env.overtime_cost * overtime_hours[0]
        model.addConstr(overtime_hours[0] >= (new_bookings[0] * self.env.duration - self.env.regular_capacity), name="overtime_0")
        if t == self.env.decision_epoch:
            # Only consider the tail overtime if we're at the last decision epoch
            for k in range(1, len(new_bookings)):
                overtime_cost += self.discount_factor ** k * self.env.overtime_cost * overtime_hours[k]
                model.addConstr(overtime_hours >= (new_bookings[k] * self.env.duration - self.env.regular_capacity),
                               name=f"overtime_{k}")
        return waiting_cost + overtime_cost

    def generate_initial_columns(self):
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        regular_capacity = self.env.regular_capacity
        maximum_arrival = self.env.arrival_generator.maximum_arrival
        init_columns = []
        for t in range(1, N+1):
            H = N-t
            bookings = np.array([regular_capacity] * (H+l))
            waitlist = np.array([maximum_arrival] * I)
            action = np.array([waitlist]+[[0]*I for _ in range(H)])
            state = (bookings, waitlist)
            init_columns.append((state, action, t))
        return init_columns

    def get_constr_coefficients(self, candidate):
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        mu = self.env.arrival_generator.mean_by_type
        gamma = self.env.discount_factor
        state, action, t = candidate
        bookings, waitlist = state
        # W0
        W_0 = {k:0 for k in range(1, N+1)}
        W_0[t] = 1
        if t < N:
            W_0[t+1] = -gamma
        Z = {k: [0]*(N+l-k) for k in range(1, N+1)}
        for j in range(N-t+l):
            Z[t][j] = bookings[j]
        if t < N:
            for j in range(N - t + l):
                Z[t+1][j-1] = -gamma*bookings[j]
        W_i = {k: [0]*I for k in range(1, N+1)}
        for i in range(I):
            W_i[t][i] = waitlist[i]
        if t < N:
            for i in range(I):
                W_i[t+1][i] = -gamma*mu[i]
        coefficients = list(W_0.values()) + [coefficient for l in Z.values() for coefficient in l] + [coefficient for l in W_i.values() for coefficient in l]
        return coefficients

    def get_obj_coefficient(self, candidate):
        state, action, t = candidate
        return self.env.cost_fn(state, action, t)

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
        gamma = self.discount_factor

        bookings, waitlist = state  # b shape = (H+1, I)
        mu = self.env.arrival_generator.mean_by_type
        # assume I know the
        with (gp.Model("SA_Advance", env=self.grb_env) as m):
            #m.setParam("OutputFlag", 0)
            #m.setParam("LogToConsole", 0)
            #m.setParam("MIPFocus", 1)
            # ---------- 1. today’s increments ----------
            action_var = np.array([[m.addVar(vtype=GRB.INTEGER, name="a_t") for i in range(I)] for t in range(H+1)])
            if action is not None:
                for j in range(H + 1):
                    for i in range(I):
                        action_var[j, i].lb = action_var[j, i].ub = int(action[j, i])
            # ---------- 1. objective ----------
            imm_cost = self.cost_fn(m, state, action_var, t)
            fut_cost = 0
            if t < N:
                new_bookings = self.next_booking(bookings=bookings, action=action_var)
                fut_cost += gamma * (self.W_0[t+1] + (self.Z[t+1] * new_bookings).sum() + (self.W[t+1] * mu).sum())
            m.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)
            # demand for today
            m.addConstrs(
                (
                    action_var[:, i].sum() == waitlist[i]
                    for i in range(I)
                ),
                name="C1_demand_today",
            )
            # ---------- 7. solve ----------
            m.setParam("Presolve", 2)
            m.setParam("Threads", 0)
            m.optimize()
            # ---------- 8. return ----------
            if m.Status == GRB.OPTIMAL:
                action = get_solution_value(action_var).astype(int)
                return action, m.ObjVal
            else:
                raise RuntimeError("Optimal solution not found")

    def policy(self, state, t):
        state_tuple = iter_to_tuple(state)
        if (state_tuple, t) in self.action_map:
            return self.action_map[(state_tuple, t)]
        action, obj_value = self.solve(state, t)
        self.action_map[(state_tuple, t)] = action
        return action

if __name__ =="__main__":
    from experiments import get_config_by_type
    # 54946.988268116984
    config = get_config_by_type('default', 42)
    env = config.env
    init_state = config.init_state
    t = 1
    exogenous_state_distribution_by_time = {t: uniform(loc=0, scale=100) for t in range(1, env.decision_epoch + 1)}
    agent = ALPAgent(env=env, discount_factor=env.discount_factor, exogenous_state_distribution_by_time=exogenous_state_distribution_by_time)
    '''
    duals = [1]*env.decision_epoch + [2 for t in range(1, env.decision_epoch+1) for j in range(env.decision_epoch+env.num_sessions-t)] + [3 for t in range(1, env.decision_epoch+1) for i in range(env.num_types)]
    duals = np.array(duals)
    sizes = [env.decision_epoch]+ [env.decision_epoch+env.num_sessions-t for t in range(1, env.decision_epoch+1)]+ [env.num_types for t in range(1, env.decision_epoch+1)]
    coefficient = split_by_group_sizes(duals, sizes)
    print(coefficient)
    '''
    print(agent.solve(init_state, t=1))



