import time
from collections import defaultdict
from pprint import pprint

import numpy as np
from scipy.stats import uniform
import gurobipy as gp
from gurobipy import GRB

from environment.utility import get_valid_advance_actions
from experiments import get_config_by_type
from utils import get_solution_value, ColumnGenerationSolver, generate_state_action_pairs, solve_and_handle_errors, \
    iter_to_tuple, make_index_counter


class ALPEJORAgent:
    TOKEN_WAIT = 15

    def __init__(self, env, discount_factor, V=None, Q=None, coefficients=None, pretrain=False):
        self.env = env
        self.discount_factor = discount_factor
        self.V = V
        self.Q = Q
        self.is_trained = False
        if Q is None:
            self.Q = defaultdict(lambda: defaultdict(float))
        if V is None:
            self.V = {}
        self.action_map = {}
        self.booking_weights = [1] * (self.env.decision_epoch + self.env.num_sessions)
        self.waitlist_weights = [0] * self.env.num_types
        self.N = self.env.booking_window_size
        self.M = self.env.planning_horizon
        self.I = self.env.num_types
        self.E_u_alpha = [uniform(loc=0, scale=self.env.regular_capacity).mean()*.8] * self.M
        self.E_u_alpha[-1] = 0
        #self.E_u_alpha = [0] * self.M
        self.E_v_alpha = [uniform(loc=0, scale=self.env.overtime_capacity).mean()*.8] * self.M
        self.E_v_alpha[-1] = 0
        #self.E_v_alpha = [0] * self.M
        #self.E_w_alpha = [uniform(loc=0, scale=self.env.arrival_generator.maximum_arrival).mean()] * self.I
        self.E_w_alpha = self.env.arrival_generator.mean_by_type
        self.grb_env = self._acquire_grb_env()
        if coefficients is not None:
            final_duals = coefficients
            self.is_trained = True
            self.W_0, self.U, self.V, self.W = self.convert_duals_to_coefficients(final_duals)
        if pretrain:
            self.train(False)
        print('E_u_alpha:',self.E_u_alpha)
        print('E_v_alpha:',self.E_v_alpha)
        print('E_w_alpha:',self.E_w_alpha)
        '''
        E_u_alpha: [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 0]
        E_v_alpha: [2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 2.4000000000000004, 0]
        E_w_alpha: [10.0]
        '''

    def train(self, debug=False):
        if debug == True:
            initial_columns = self.generate_all_columns()
        else:
            initial_columns = self.generate_initial_columns()
        self.cg_solver = ColumnGenerationSolver(master_builder=self.master_builder,
                                                pricing_callback=self.pricing_callback,
                                                initial_columns=initial_columns,
                                                get_constr_coefficients=self.get_constr_coefficients,
                                                get_obj_coefficient=self.get_obj_coefficient)
        self.cg_solver.solve()
        final_duals = [c.Pi for c in self.cg_solver.master_model.getConstrs()]
        self.W_0, self.U, self.V, self.W = self.convert_duals_to_coefficients(final_duals)
        self.is_trained = True
        return final_duals

    def convert_duals_to_coefficients(self, duals):
        it = iter(duals)
        W_0 = float(next(it))
        U = np.array([float(next(it)) for _ in range(self.M)])
        V = np.array([float(next(it)) for _ in range(self.M)])
        W = np.array([float(next(it)) for _ in range(self.I)])
        return W_0, U, V, W

    def master_builder(self):
        '''
        Total number of constraints: N + N+l-1 + N+l -2 ... + l + N*I
        '''
        master_model = gp.Model("MasterRMP")
        master_model.ModelSense = GRB.MINIMIZE
        master_model.setParam('OutputFlag', 0)
        master_model.addConstr(
            (
                    gp.LinExpr() == 1
            ),
            name="constr_W_0")
        master_model.addConstrs(
            (
                gp.LinExpr() >= self.E_u_alpha[j]
                for j in range(self.M)
            ),
            name="constr_U")
        master_model.addConstrs(
            (
                gp.LinExpr() >= self.E_v_alpha[j]
                for j in range(self.M)
            ),
            name="constr_V")
        master_model.addConstrs(
            (
                gp.LinExpr() >= self.E_w_alpha[i]
                for i in range(self.I)
            ),
            name="constr_W")
        master_model.update()
        return master_model

    def get_approx_value_fn(self, state_var, W_0, U, V, W):
        bookings_var, overtimes_var, waitlist_var = state_var
        return W_0 + (U * bookings_var).sum() + (V * overtimes_var).sum() + (W * waitlist_var).sum()

    def add_action_space_constraints(self, model, state_var, action_var):
        u_var, v_var, w_var = state_var
        x_var, y_var = action_var
        model.addConstrs(
            (x_var[:, i].sum() <= w_var[i]
             for i in range(self.I)),
            name="C1_valid_advance_schedule",
        )
        booking_slots = self.env.convert_action_to_booking_slots(x_var)
        model.addConstrs(
            (
                u_var[m] + booking_slots[m] <= self.env.regular_capacity + y_var[m]
                for m in range(self.M)
            ),
            name="C2_valid_appointment_slots",
        )
        model.addConstrs(
            (
                v_var[m] + y_var[m] <= self.env.overtime_capacity
                for m in range(self.M)
            ),
            name="C3_valid_overtime_slots",
        )
        return model


    def pricing_callback(self, duals):
        """
        Solves the pricing subproblem by iterating through each time period.

        This improved approach solves N smaller, independent optimization problems,
        one for each period, instead of one large MIP. This is more efficient
        and directly calculates the minimum reduced cost for each period.
        """
        # --- 1. Initial Setup (Parameters and Duals) ---
        mu = self.env.arrival_generator.mean_by_type
        gamma = self.env.discount_factor
        regular_capacity = self.env.regular_capacity
        overtime_capacity = self.env.overtime_capacity
        W_0, U, V, W = self.convert_duals_to_coefficients(duals)

        pricing_model = gp.Model(f"Pricing_Problem", env=self.grb_env)
        pricing_model.setParam('OutputFlag', 0)
        x_var = np.array([
            [pricing_model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{j},{i}") for i in range(self.I)]
            for j in range(self.N)
        ])
        y_var = np.array([pricing_model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{j}") for j in range(self.M)])
        u_var = np.array([
            pricing_model.addVar(vtype=GRB.INTEGER, lb=0, ub=regular_capacity, name=f"u_{j}") for j in range(self.M)
        ])
        v_var = np.array([
            pricing_model.addVar(vtype=GRB.INTEGER, lb=0, ub=overtime_capacity, name=f"v_{j}") for j in range(self.M)
        ])
        w_var = np.array([
            pricing_model.addVar(vtype=GRB.INTEGER, lb=0, name=f"w_{i}") for i in range(self.I)
        ])
        booked_slots = self.env.convert_action_to_booking_slots(x_var)
        # [add constraints as needed here...]
        # Action Space Constraints
        state_var = (u_var, v_var, w_var)
        action_var = (x_var, y_var)
        self.add_action_space_constraints(model=pricing_model,
                                          state_var=state_var,
                                          action_var=action_var)
        pricing_model.addConstrs(
            (booked_slots[m] >= y_var[m]
             for m in range(self.M)),
            name="C2_demand_today",
        )

        pricing_model.addConstr(u_var[-1] == 0, name="C2_regular_hour")

        pricing_model.addConstr(v_var[-1] == 0, name="C3_overtime_hour")
        state_var = (u_var, v_var, w_var)
        action_var = (x_var, y_var)
        next_state_var = self.env.next_state(state_var, action_var, mu, is_var=True)
        # --- Objective ---
        candidate_cost = self.env.cost_fn(state_var, action_var)
        approx_V = self.get_approx_value_fn(state_var, W_0, U, V, W)
        #dual_cost_sum = np.dot(np.array(self.get_constr_coefficients((state_var, action_var))), duals)
        #reduced_cost = candidate_cost - dual_cost_sum
        reduced_cost = candidate_cost + gamma * self.get_approx_value_fn(next_state_var, W_0, U, V, W) - approx_V
        pricing_model.setObjective(reduced_cost, GRB.MINIMIZE)

        pricing_model.optimize()
        print(candidate_cost.getValue())
        if solve_and_handle_errors(pricing_model):
            rc = pricing_model.ObjVal
            advance_scheduling_decision = get_solution_value(x_var).astype(int)
            overtime_decision = get_solution_value(y_var).astype(int)
            bookings = get_solution_value(u_var).astype(int)
            overtimes = get_solution_value(v_var).astype(int)
            waitlist = get_solution_value(w_var).astype(int)
            state = (bookings, overtimes, waitlist)
            action = (advance_scheduling_decision, overtime_decision)
            # If new solution, yield and exit
            return [((state, action), rc)]
        return

    def generate_initial_state_action_pairs(self):
        gamma = self.env.discount_factor
        regular_capacity = self.env.regular_capacity
        maximum_arrival = self.env.arrival_generator.maximum_arrival
        init_columns = []
        for i in range(self.I):
            with (gp.Model("init_columns", env=self.grb_env) as init_columns_model):
                u_var = np.array(
                    [init_columns_model.addVar(vtype=GRB.INTEGER, lb=0, ub=regular_capacity, name=f'init_u_{j}') for j in
                     range(self.M)])
                u_var[-1].lb = u_var[-1].ub = 0
                v_var = np.array(
                    [init_columns_model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f'init_v_{j}') for j in
                     range(self.M)]
                )
                w_var = np.array(
                    [init_columns_model.addVar(vtype=GRB.INTEGER, lb=maximum_arrival, ub=maximum_arrival, name=f'init_v_{k}') if k == i else init_columns_model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f'init_v_{k}') for k in
                     range(self.I)]
                )
                x_var = np.array([[init_columns_model.addVar(vtype=GRB.INTEGER, lb=0,
                                                                  name=f"init_x_{j},{i}") for i in range(self.I)] for
                                       j in range(self.N)])
                y_var = np.array(
                    [init_columns_model.addVar(vtype=GRB.INTEGER, lb=0, ub=0, name=f'init_v_{j}') for j in
                     range(self.M)]
                )
                state_var = (u_var, v_var, w_var)
                action_var = (x_var, y_var)
                # Action Space Constraints
                self.add_action_space_constraints(init_columns_model, state_var, action_var)
                next_bookings_var = self.env.get_next_bookings(u_var, action_var)
                bookings_diff_var = u_var - gamma * next_bookings_var
                maximum_difference_var = init_columns_model.addVar(name='maximum_difference')
                init_columns_model.addConstrs(
                    (
                        maximum_difference_var <= bookings_diff_var[j]
                        for j in range(self.M)
                    ),
                    name="C1_maximum_difference",
                )
                init_columns_model.setObjective(maximum_difference_var, GRB.MAXIMIZE)
                if solve_and_handle_errors(init_columns_model):
                    bookings = get_solution_value(u_var).astype(int)
                    overtimes = get_solution_value(v_var).astype(int)
                    waitlist = get_solution_value(w_var).astype(int)
                    advance_scheduling_decision = get_solution_value(x_var).astype(int)
                    overtime_decision = get_solution_value(y_var).astype(int)
                    column = ((bookings, overtimes, waitlist), (advance_scheduling_decision, overtime_decision))
                    init_columns.append(column)
        return init_columns

    def generate_initial_columns(self, debug=False, skip=False):
        if debug == True:
            initial_columns = self.generate_all_columns()
        else:
            initial_columns = self.generate_initial_state_action_pairs()
        self.cg_solver = ColumnGenerationSolver(master_builder=self.initial_columns_builder,
                                                pricing_callback=self.pricing_callback,
                                                initial_columns=initial_columns,
                                                get_constr_coefficients=self.get_constr_coefficients,
                                                get_obj_coefficient=None)
        initial_columns = self.cg_solver.initial_columns_solve()
        print(initial_columns)
        return initial_columns

    def initial_columns_builder(self):
        master_model = gp.Model("InitMasterRMP")
        s_var = master_model.addVar(vtype=GRB.CONTINUOUS, name=f'init_s')
        master_model.setObjective(s_var, GRB.MINIMIZE)
        master_model.setParam('OutputFlag', 0)
        master_model.addConstr(
            (
                    gp.LinExpr() == 1
            ),
            name="constr_W_0")
        master_model.addConstrs(
            (
                s_var >= self.E_u_alpha[j]
                for j in range(self.M)
            ),
            name="constr_U")
        master_model.addConstrs(
            (
                s_var >= self.E_v_alpha[j]
                for j in range(self.M)
            ),
            name="constr_V")
        master_model.addConstrs(
            (
                s_var >= self.E_w_alpha[i]
                for i in range(self.I)
            ),
            name="constr_W")
        master_model.update()
        return master_model

    def get_constr_coefficients(self, candidate):
        mu = self.env.arrival_generator.mean_by_type
        gamma = self.env.discount_factor
        state, action = candidate
        bookings, overtimes, waitlist = state
        new_bookings, new_overtimes, new_waitlist = self.env.next_state(state, action, mu, is_var=False)
        # W_0 coefficient
        W_0 = 1 - gamma
        # Z coefficients
        U = (bookings - gamma * new_bookings).tolist()
        V = (overtimes - gamma * new_overtimes).tolist()
        # W_i coefficients
        W = (waitlist - gamma * new_waitlist).tolist()
        coefficients = [W_0] + U + V + W
        return coefficients

    def get_obj_coefficient(self, candidate):
        state, action = candidate
        return self.env.cost_fn(state, action)

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

    def solve(self, state, action=None):
        # Need to Fix
        # ---------- shortcuts ----------
        gamma = self.discount_factor

        bookings, overtimes, waitlist = state  # b shape = (H+1, I)
        mu = self.env.arrival_generator.mean_by_type
        # assume I know the
        with (gp.Model("ALP_Advance", env=self.grb_env) as policy_model):
            # m.setParam("OutputFlag", 0)
            # m.setParam("LogToConsole", 0)
            # m.setParam("MIPFocus", 1)
            # ---------- 1. today’s increments ----------
            x_var = np.array([[policy_model.addVar(vtype=GRB.INTEGER, lb=0, name=f"x_{i}_{n}") for i in range(self.I)] for n in range(self.N)])
            y_var = np.array([policy_model.addVar(vtype=GRB.INTEGER, lb=0, name=f"y_{m}") for m in range(self.M)])
            if action is not None:
                x, y = action
                for n in range(self.N):
                    for i in range(self.I):
                        x_var[n, i].lb = x_var[n, i].ub = int(x[n, i])
                for m in range(self.M):
                    y_var[m].lb = y_var[m].ub = int(y[m])
            # ---------- 1. objective ----------
            action_var = (x_var, y_var)
            imm_cost = self.env.cost_fn(state, action_var)
            fut_cost = 0
            next_state = self.env.next_state(state=state, action=action_var, new_arrival=mu)
            fut_cost += gamma * self.get_approx_value_fn(next_state, self.W_0, self.U, self.V, self.W)

            policy_model.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)
            # Action Space Constraints
            self.add_action_space_constraints(model=policy_model,
                                              state_var=state,
                                              action_var=action_var)
            # ---------- 7. solve ----------
            policy_model.setParam("Presolve", 2)
            policy_model.setParam("Threads", 0)
            policy_model.optimize()
            # ---------- 8. return ----------
            if policy_model.Status == GRB.OPTIMAL:
                x = get_solution_value(x_var).astype(int)
                y = get_solution_value(y_var).astype(int)
                action = (x, y)
                return action, policy_model.ObjVal
            else:
                raise RuntimeError("Optimal solution not found")

    def coeff_C(self, i, n):
        part1 = sum(self.discount_factor ** k * self.env.holding_cost(k, i) for k in range(n + 1))
        part2 = sum(self.discount_factor * self.env.treatment_pattern[k+1-n, i] * self.U[k] for k in range(n-1,n-1+self.env.num_sessions))
        part3 = self.env.postponing_cost(i) - self.discount_factor * self.W[i]
        return part1 + part2 + part3

    def policy(self, state):
        action, obj_value = self.solve(state)
        return action

    def generate_all_columns(self):
        for column in self.env.generate_state_action_pairs():
            yield column


if "__main__" == __name__:
    config = get_config_by_type('rt_default', random_seed=1)
    env = config.env
    init_state = config.init_state
    duals = [194, 0.0, 0.0, 1.0]
    agent = ALPEJORAgent(env=env, discount_factor=env.discount_factor, pretrain=False)
    #print(agent.solve(init_state))
    #candidate = (np.array([0, 0]), np.array([0, 0]), np.array([0])), (np.array([[0],[0]]), np.array([0, 0]))
    #print(agent.get_constr_coefficients(candidate))

    #initial_columns = agent.generate_initial_state_action_pairs()
    #print(initial_columns)
    '''
    for candidate, reduce_cost in agent.pricing_callback(duals):
        candidate_cost = agent.get_obj_coefficient(candidate)
        candidate_coeffs = agent.get_constr_coefficients(candidate)
        rc = candidate_cost
        for j, coeff in enumerate(candidate_coeffs):
            rc -= duals[j] * coeff
        print('reduce_cost, rc')
        print(reduce_cost, rc)
        print('candidate')
        print(candidate)
    
    u = np.ones((agent.M,))
    u[:14] = 50
    v = np.ones((agent.M,))
    w = np.ones((agent.I,))
    w[0] = 30
    x = np.ones((agent.N, agent.I))
    y = np.ones((agent.M,))
    candidiate = ((u, v, w), (x, y))
    print(agent.get_constr_coefficients(candidiate))
    # [0.010000000000000009, -0.490000000000002, -1.4799999999999969, -2.469999999999999, -3.460000000000001, -3.460000000000001, -3.460000000000001, -3.460000000000001, -3.460000000000001, -3.460000000000001, -3.460000000000001, -3.460000000000001, -3.460000000000001, -3.460000000000001, 45.05, -3.95, -3.95, -3.95, -3.95, -3.95, -3.95, -3.95, -3.95, -3.95, -3.95, -2.96, -1.9699999999999998, -0.98, 0.010000000000000009, 1.0, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, -0.98, 1.0, 15.15]

    #all_columns = list(agent.generate_all_columns())
    #pprint(all_columns)
    #final_duals = agent.train(debug=False)
    #print(final_duals)

    #print(agent.solve(config.init_state, 1))
    '''
    duals = [1]+[2]*agent.M +[1]*agent.M +[1]*agent.I
    print(agent.pricing_callback(duals))
