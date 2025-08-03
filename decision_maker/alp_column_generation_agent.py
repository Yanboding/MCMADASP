import time
from collections import defaultdict

import numpy as np
from scipy.stats import uniform
import gurobipy as gp
from gurobipy import GRB

from environment.utility import get_valid_advance_actions
from utils import get_solution_value, ColumnGenerationSolver, generate_state_action_pairs, solve_and_handle_errors


def make_index_counter(start=0):
    index = start
    while True:
        yield index
        index += 1

class ALPAgent:
    TOKEN_WAIT = 15
    def __init__(self, env, discount_factor, V=None, Q=None, coefficients=None):
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
        self.booking_weights = [1] * (self.env.decision_epoch+self.env.num_sessions)
        self.waitlist_weights = [0] * self.env.num_types
        self.grb_env = self._acquire_grb_env()
        if coefficients is not None:
            final_duals = coefficients
            self.is_trained = True
            self.W_0, self.Z, self.W = self.convert_duals_to_coefficients(final_duals)

    def train(self,debug=False):
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
        self.is_trained = True
        final_duals = [c.Pi for c in self.cg_solver.master_model.getConstrs()]
        self.W_0, self.Z, self.W = self.convert_duals_to_coefficients(final_duals)
        return self.W_0, self.Z, self.W

    def convert_duals_to_coefficients(self, duals):
        '''
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        W_0 = {k: final_duals[next(counter) for k in range(1, N + 1)}
        Z = {k: [final_duals[next(counter)] for j in range(N + l - k)] for k in range(1, N + 1)}
        W = {k: [final_duals[next(counter)] for i in range(I)] for k in range(1, N + 1)}
        '''
        counter = make_index_counter(0)
        W_0 = duals[next(counter)]
        Z = np.array([duals[next(counter)] for k in range(self.env.decision_epoch + self.env.num_sessions - 1)])
        W = np.array([duals[next(counter)] for i in range(self.env.num_types)])
        return W_0, Z, W

    def master_builder(self):
        '''
        Total number of constraints: N + N+l-1 + N+l -2 ... + l + N*I
        '''
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        mean_by_type = self.env.arrival_generator.mean_by_type
        maximum_arrival = self.env.arrival_generator.maximum_arrival
        z_max = maximum_arrival * N * self.env.treatment_pattern.max() + self.env.regular_capacity
        E_z_beta = {t: [uniform(loc=0, scale=z_max).mean()]*(N+l-t) for t in
                    range(1, N + 1)}
        E_delta_beta = {t:  mean_by_type for t in range(1, N+1)}
        master_model = gp.Model("MasterRMP")
        master_model.ModelSense = GRB.MINIMIZE
        master_model.setParam('OutputFlag', 0)
        '''
        master_model.addConstrs(
            (
                    gp.LinExpr() == 1
                    for _ in range(1, N+1)
             ),
            name="constr_W_0")
        '''

        '''
        master_model.addConstrs(
            (
                gp.LinExpr() >= E_z_beta[t][j]
                for t in range(1, N + 1)
                for j in range(N + l - t)
            ),
            name="constr_Z")
        '''
        '''
        master_model.addConstrs(
            (
                gp.LinExpr() >= E_delta_beta[t][i]
                for t in range(1, N + 1)
                for i in range(I)
            ),
            name="constr_W")
        '''
        master_model.addConstr(
            (
                    gp.LinExpr() == N
            ),
            name="constr_W_0")
        # j = 0, t = 1, 2, 3
        # j = 1, t = 1, 2
        # j = 2, t = 1
        master_model.addConstrs(
            (
                gp.LinExpr() >= sum(E_z_beta[t][j] for t in range(1, min(N+l-j, N+1)))
                for j in range(N + l - 1)
            ),
            name="constr_Z")
        master_model.addConstrs(
            (
                gp.LinExpr() >= sum(E_delta_beta[t][i] for t in range(1, N + 1))
                for i in range(I)
            ),
            name="constr_W")
        master_model.update()
        return master_model

    def get_approx_value_fn(self, bookings_var, waitlist_var, t, W_0, Z, W):
        N = self.env.decision_epoch
        l = self.env.num_sessions
        H = N - t
        return W_0 + (Z[:H+l] * bookings_var).sum() + (W * waitlist_var).sum()

    def pricing_callback(self, duals):
        """
        Solves the pricing subproblem by iterating through each time period.

        This improved approach solves N smaller, independent optimization problems,
        one for each period, instead of one large MIP. This is more efficient
        and directly calculates the minimum reduced cost for each period.
        """
        # --- 1. Initial Setup (Parameters and Duals) ---
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        mu = self.env.arrival_generator.mean_by_type
        gamma = self.env.discount_factor
        maximum_arrival = self.env.arrival_generator.maximum_arrival
        z_max = maximum_arrival * N * self.env.treatment_pattern.max() + self.env.regular_capacity
        W_0, Z, W = self.convert_duals_to_coefficients(duals)

        with gp.Model("Pricing_Problem", env=self.grb_env) as pricing_model:
            pricing_model.setParam('OutputFlag', 0)
            # --- 3. Configure the Gurobi Solution Pool ---
            # Find and store as many solutions as possible.
            #pricing_model.setParam(GRB.Param.PoolSolutions, 300)
            # Ensure the solutions are the best ones found and are ranked.
            #pricing_model.setParam(GRB.Param.PoolSearchMode, 2)

            # --- 4. Define Variables, Constraints, and Objective (Same as before) ---
            # The model structure is identical to the previous version.
            y = pricing_model.addVars(range(1, N + 1), vtype=GRB.BINARY, name="y")
            action_vars, bookings_vars, waitlist_vars = {}, {}, {}
            # ... (variable definitions are identical to the previous version) ...
            for t in range(1, N + 1):
                H = N - t
                action_vars[t] = np.array(
                    [[pricing_model.addVar(vtype=GRB.INTEGER, lb=0, name=f"A_{t},{j},{i}") for i in range(I)] for j in
                     range(H + 1)])
                bookings_vars[t] = np.array(
                    [pricing_model.addVar(vtype=GRB.INTEGER, lb=0, ub=z_max, name=f"z_{t},{j}") for j in range(H + l)])
                waitlist_vars[t] = np.array(
                    [pricing_model.addVar(vtype=GRB.INTEGER, lb=0, ub=maximum_arrival, name=f"delta_{t},{i}") for i in
                     range(I)])
            # --- 5. Add Constraints (Same as before) ---
            pricing_model.addConstr(y.sum() == 1, name="C0_one_hot_selection")
            M = maximum_arrival
            # ... (Big-M constraints are identical to the previous version) ...
            for t in range(1, N + 1):
                for i in range(I):
                    lhs = action_vars[t][:, i].sum() - waitlist_vars[t][i]
                    pricing_model.addConstr(lhs <= M * (1 - y[t]), name=f"C1_upper_{t},{i}")
                    pricing_model.addConstr(lhs >= -M * (1 - y[t]), name=f"C1_lower_{t},{i}")

            # --- 6. Define the Unified Objective (Same as before) ---
            objective_expr = 0
            # ... (objective function definition is identical to the previous version) ...
            for t in range(1, N + 1):
                b_vars, w_vars, a_vars = bookings_vars[t], waitlist_vars[t], action_vars[t]
                approx_V_t = self.get_approx_value_fn(b_vars, w_vars, t, W_0, Z, W)
                new_bookings_t = self.next_booking(b_vars, a_vars)
                if t < N:
                    immediate_cost_t = self.cost_fn(pricing_model, (b_vars, w_vars), a_vars, t)
                    future_value_t = gamma * self.get_approx_value_fn(new_bookings_t, mu, t + 1, W_0, Z, W)
                    column_cost_t = immediate_cost_t + future_value_t
                else:
                    column_cost_t = self.cost_fn(pricing_model, (b_vars, w_vars), a_vars, N)
                reduced_cost_t = column_cost_t - approx_V_t
                objective_expr += y[t] * reduced_cost_t
            pricing_model.setObjective(objective_expr, GRB.MINIMIZE)

            # --- 2. Iteratively Solve and Cut ---
            # Loop to find up to 'max_solutions' distinct solutions
            for k in range(300):
                pricing_model.optimize()

                # If no more feasible solutions can be found, stop.
                if pricing_model.SolCount == 0:
                    break

                # --- Extract the current best solution ---
                reduced_cost = pricing_model.ObjVal
                active_t = next((t for t, var in y.items() if var.X > 0.5), -1)

                if active_t == -1:
                    break  # Should not happen if SolCount > 0

                action = get_solution_value(action_vars[active_t]).astype(int)
                bookings = get_solution_value(bookings_vars[active_t]).astype(int)
                waitlist = get_solution_value(waitlist_vars[active_t]).astype(int)
                solution = ((bookings, waitlist), action, active_t)
                # Yield the found solution
                yield solution, reduced_cost

                # --- Add "No-Good" Cut to remove this solution ---
                x_vars = list(bookings_vars[active_t]) + list(waitlist_vars[active_t]) + list(
                    action_vars[active_t].flatten())
                x_vals = list(bookings) + list(waitlist) + list(action.flatten())
                delta_list = []
                for i, (var, val) in enumerate(zip(x_vars, x_vals)):
                    delta_le = pricing_model.addVar(vtype=GRB.BINARY, name=f"delta_le_{i}")
                    delta_ge = pricing_model.addVar(vtype=GRB.BINARY, name=f"delta_ge_{i}")
                    pricing_model.addGenConstrIndicator(delta_le, True, var <= val - 1)
                    pricing_model.addGenConstrIndicator(delta_ge, True, var >= val + 1)
                    delta_list.append(delta_le)
                    delta_list.append(delta_ge)

                # At least one variable must differ:
                pricing_model.addConstr(gp.quicksum(delta_list) >= 1, name="no_good_cut")

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
        bookings, _ = state
        waiting_cost = sum(sum(self.discount_factor ** k * self.env.holding_cost(k, i) for k in range(j + 1)) * action[j, i]
                           for j in range(len(action))
                           for i in range(len(action[0])))
        new_bookings = bookings + self.convert_action_to_booking_slots(action)
        overtime_hours = np.array([model.addVar(name="overtime_hours", lb=0) for _ in range(len(new_bookings))])
        overtime_cost = self.env.overtime_cost * overtime_hours[0]
        model.addConstr(overtime_hours[0] >= (new_bookings[0] * self.env.duration - self.env.regular_capacity), name="overtime_0")
        if t == self.env.decision_epoch:
            # Only consider the tail overtime if we're at the last decision epoch
            for k in range(1, len(new_bookings)):
                overtime_cost += self.discount_factor ** k * self.env.overtime_cost * overtime_hours[k]
                model.addConstr(overtime_hours[k] >= (new_bookings[k] * self.env.duration - self.env.regular_capacity),
                               name=f"overtime_{k}")
        return waiting_cost + overtime_cost

    def generate_initial_columns(self):
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        gamma = self.env.discount_factor
        regular_capacity = self.env.regular_capacity
        maximum_arrival = self.env.arrival_generator.maximum_arrival
        init_columns = []
        z_max = maximum_arrival * (N-1) * self.env.treatment_pattern.max() + regular_capacity
        for t in range(1, N+1):
            H = N - t
            for i in range(I):
                with (gp.Model("init_columns", env=self.grb_env) as init_columns_model):
                    waitlist = np.array([0] * I)
                    waitlist[i] = maximum_arrival
                    bookings_var = np.array([init_columns_model.addVar(vtype=GRB.INTEGER, lb=0, ub=z_max, name=f'init_z^{t}_{j}') for j in range(H+l)])
                    action_var = np.array([[init_columns_model.addVar(vtype=GRB.INTEGER, lb=0, name=f"init_A^{t}_{j},{i}") for i in range(I)] for j in range(H + 1)])
                    next_bookings_var = self.next_booking(bookings_var,action_var)
                    maximum_difference_var = init_columns_model.addVar(name='maximum_difference')
                    init_columns_model.addConstrs(
                        (
                            maximum_difference_var <= (bookings_var[j] - gamma * next_bookings_var[j] if t < N + l - 1 - j else bookings_var[j])
                            for j in range(len(bookings_var))
                        ),
                        name="C1_maximum_difference",
                    )
                    init_columns_model.addConstrs(
                        (
                            action_var[:, i].sum() == waitlist[i]
                            for i in range(I)
                        ),
                        name="C2_demand_today",
                    )
                    init_columns_model.setObjective(maximum_difference_var, GRB.MAXIMIZE)
                    if solve_and_handle_errors(init_columns_model):
                        action = get_solution_value(action_var).astype(int)
                        bookings = get_solution_value(bookings_var).astype(int)
                        column = ((bookings, waitlist), action, t)
                        init_columns.append(column)
        return init_columns

    def get_constr_coefficients(self, candidate):
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        mu = self.env.arrival_generator.mean_by_type
        gamma = self.env.discount_factor
        state, action, t = candidate
        bookings, waitlist = state
        new_bookings = self.next_booking(bookings, action)
        # W_0 coefficient
        W_0 = 1 - gamma if t < N else 1
        # Z coefficients
        Z = []
        for j in range(N + l - 1):
            if t < N + l - 1 - j:
                val = bookings[j] - gamma * new_bookings[j]
            elif t == N and j < l:
                val = bookings[j]
            elif t == N + l - 1 - j and j >= l:
                val = bookings[j]
            else:
                val = 0
            Z.append(val)
        # W_i coefficients
        W_i = [(waitlist[i] - gamma * mu[i]) if t < N else waitlist[i] for i in range(I)]
        return [W_0] + Z + W_i


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
        with (gp.Model("ALP_Advance", env=self.grb_env) as m):
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
                fut_cost += gamma * self.get_approx_value_fn(new_bookings, mu, t+1, self.W_0, self.Z, self.W)
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
        action, obj_value = self.solve(state, t)
        return action

    def generate_all_columns(self):
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        regular_capacity = self.env.regular_capacity
        maximum_arrival = self.env.arrival_generator.maximum_arrival
        z_max = maximum_arrival * (N) * self.env.treatment_pattern.max() + regular_capacity
        for column in generate_state_action_pairs(maximum_slots=z_max,
                                                   maximum_num_sessions=l,
                                                   maximum_arrival=maximum_arrival,
                                                   num_type=I,
                                                   period_to_go=N):
            yield column


if __name__ =="__main__":
    from experiments import get_config_by_type
    # 54946.988268116984
    config = get_config_by_type('default')
    env = config.env
    init_state = config.init_state
    t = 1
    exogenous_state_distribution_by_time = {t: uniform(loc=0, scale=100) for t in range(1, env.decision_epoch + 1)}
    duals = [-9859748.186390756, 99.9999999999992, 6957.658727005252, 13745.750866740453, 6483.3712029391645, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4142106513626018e-14, 0.0, 0.0, 0.0, 0.0, 1.8088592518732013e-14, 1.7907706593544693e-14, 1.7728629527609246e-14, 1.7551343232333152e-14, 1.737582980000982e-14, 1.7202071502009722e-14, -7.847168073596153e-14, 0.0, 8.044804977647862e-15, 0.0, -7.776741197837839e-14, 0.0, -5.165410149195335e-14, 27180.16298871802, 166.24999999999966, 20761.625497808247, 27147.66298871802, 27147.662988718013, 54261.575977436034, 100.0000000000055, 27113.91298871803, 27113.912988718024, 20695.375497808247, 27113.912988718013, 0.0, 27075.162988718013, 54189.07597743605, 27072.662988718017, 27072.662988718028, 27072.66298871802, 0.0]
    agent = ALPAgent(env=env, discount_factor=env.discount_factor)
    columns = list(agent.generate_all_columns())
    print(columns)

    W_0, Z, W = agent.train(debug=True)
    #columns = agent.generate_initial_columns()
    #print(columns) 2767.7803639492995
    # -30.0 [15.     14.85   14.7015] [40. 20.]
    # 97.87294488252503 [41.03947446 44.91300951 48.74780922 44.00510413] [89.89858599  0.        ]
    '''
    bookings = np.array([0])
    new_arrival = np.array([0, 0, 0])
    action = np.array([[0, 0, 0]])
    t = 3
    duals = [132.20717538253967, 72.40099939898013, 58.06119176440023, 57.38614615005441, 56.859501536904766, 0.0]
    candidate = ((bookings, new_arrival), action, t)
    print(agent.pricing_callback(duals))
    '''



