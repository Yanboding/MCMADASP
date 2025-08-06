import time
from collections import defaultdict
from pprint import pprint

import numpy as np
from scipy.stats import uniform
import gurobipy as gp
from gurobipy import GRB

from environment.utility import get_valid_advance_actions
from utils import get_solution_value, ColumnGenerationSolver, generate_state_action_pairs, solve_and_handle_errors, \
    iter_to_tuple


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
        N = self.env.decision_epoch
        I = self.env.num_types
        l = self.env.num_sessions
        '''
        W_0 = {k: final_duals[next(counter) for k in range(1, N + 1)}
        Z = {k: [final_duals[next(counter)] for j in range(N + l - k)] for k in range(1, N + 1)}
        W = {k: [final_duals[next(counter)] for i in range(I)] for k in range(1, N + 1)}
        '''
        counter = make_index_counter(0)
        W_0 = duals[next(counter)]
        Z = np.array([duals[next(counter)] for k in range(N + l - 1)])
        W = np.array([duals[next(counter)] for i in range(I)])
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

    def pricing_callback(self, duals, max_attempts=10):
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
        models = []
        var_handles = []
        candidate_costs = []
        dual_costs = []

        for t in range(1, N + 1):
            m = gp.Model(f"Pricing_Problem_t{t}", env=self.grb_env)
            m.setParam('OutputFlag', 0)
            H = N - t
            a_var = np.array([
                [m.addVar(vtype=GRB.INTEGER, lb=0, name=f"A_{t},{j},{i}") for i in range(I)]
                for j in range(H + 1)
            ])
            b_var = np.array([
                m.addVar(vtype=GRB.INTEGER, lb=0, ub=z_max, name=f"z_{t},{j}") for j in range(H + l)
            ])
            w_var = np.array([
                m.addVar(vtype=GRB.INTEGER, lb=0, ub=maximum_arrival, name=f"delta_{t},{i}") for i in range(I)
            ])

            # [add constraints as needed here...]
            m.addConstrs(
                (a_var[:, i].sum() == w_var[i]
                for i in range(I)),
                name="C1_demand_today",
            )
            m.addConstr(
                w_var.sum() <= maximum_arrival,
                name="C2_maximum_arrival",
            )

            # --- Objective ---
            candidate_cost = self.cost_fn(m, (b_var, w_var), a_var, t)
            approx_V_t = self.get_approx_value_fn(b_var, w_var, t, W_0, Z, W)
            new_bookings_t = self.next_booking(b_var, a_var)
            if t < N:
                future_value_t = gamma * self.get_approx_value_fn(new_bookings_t, mu, t + 1, W_0, Z, W)
                dual_cost = approx_V_t - future_value_t
            else:
                dual_cost = approx_V_t
            candidate_costs.append(candidate_cost)
            dual_costs.append(dual_cost)
            reduced_cost_t = candidate_cost - dual_cost
            m.setObjective(reduced_cost_t, GRB.MINIMIZE)
            m.update()
            models.append(m)
            var_handles.append((b_var, w_var, a_var))
        # Now repeatedly solve until a new best solution is found
        for _ in range(max_attempts):
            best_solution = None
            best_reduced_cost = float("inf")
            best_t = None

            # Solve all models, collect their best solutions
            for idx, (m, (b_var, w_var, a_var), candidate_cost, dual_cost) in enumerate(zip(models, var_handles, candidate_costs, dual_costs)):
                t = idx + 1
                m.optimize()
                if m.Status == GRB.OPTIMAL:
                    rc = m.ObjVal
                    action = get_solution_value(a_var).astype(int)
                    bookings = get_solution_value(b_var).astype(int)
                    waitlist = get_solution_value(w_var).astype(int)
                    if rc < best_reduced_cost:
                        best_solution = ((bookings, waitlist), action, t)
                        best_reduced_cost = rc
                        best_t = t

            # If new solution, yield and exit
            yield best_solution, best_reduced_cost

            # Otherwise, add a no-good cut for this solution in its model and try again
            idx = best_t - 1
            m = models[idx]
            b_var, w_var, a_var = var_handles[idx]
            (bookings, waitlist), action, t = best_solution
            x_vars = list(b_var) + list(w_var) + list(a_var.flatten())
            x_vals = list(bookings) + list(waitlist) + list(action.flatten())
            delta_list = []
            for i, (var, val) in enumerate(zip(x_vars, x_vals)):
                delta_le = m.addVar(vtype=GRB.BINARY, name=f"delta_le_{i}")
                delta_ge = m.addVar(vtype=GRB.BINARY, name=f"delta_ge_{i}")
                m.addGenConstrIndicator(delta_le, True, var <= val - 1)
                m.addGenConstrIndicator(delta_ge, True, var >= val + 1)
                delta_list.extend([delta_le, delta_ge])
            m.addConstr(gp.quicksum(delta_list) >= 1, name=f"no_good_cut_{t}_{_}")
            m.update()

        # If we exit loop, no new improving solution exists
        return


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
        if t < N:
            W_0 = 1 - gamma
        else:
            W_0 = 1
        # Z coefficients
        Z = []
        for j in range(N + l - 1):
            val = 0
            if t <= min(N-1, N + l - 2 - j):
                val += bookings[j] - gamma * new_bookings[j]
            if t == N + l - 1 - j and j >= l:
                val += bookings[j]
            if t == N and j < l:
                val += bookings[j]
            Z.append(val)
        # W_i coefficients
        W_i = []
        for i in range(I):
            val = 0
            if t < N-1:
                val += waitlist[i] - gamma * mu[i]
            elif t == N:
                val += waitlist[i]
            W_i.append(val)
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
    config = get_config_by_type('base_case')
    env = config.env
    init_state = config.init_state
    duals = [-814342.8028237808, 99.99999999999989, 99.0, 98.00999999999992, 97.02990000000004, 96.05960100000017, 95.09900499000014, 94.14801494010004, 93.20653479069898, 92.27446944279214, 91.3517247483641, 90.43820750088041, 89.53382542587164, 88.63848717161288, 87.75210229989679, 86.87458127689774, 86.00583546412871, 85.14577710948741, 84.29431933839254, 83.45137614500865, 82.61686238355855, 81.79069375972308, 80.9727868221258, 80.16305895390457, 79.36142836436551, 78.56781408072189, 77.78213593991464, 77.00431458051533, 76.23427143471022, 75.4719287203632, 74.71720943315937, 73.97003733882761, 73.2303369654393, 72.49803359578455, 71.77305325982638, 71.05532272722792, 70.3447694999557, 69.64132180495578, 68.94490858690597, 68.25545950103658, 722.5995010000003, 332.5000000000002, 626.539899999989, 1685.422289051244, 2106.0764011371793]
    candidate = (((np.array([0,  0, 0,  0]), np.array([1, 1])),np.array([[1, 1],
       [0, 0],
       [0, 0]])),1)
    agent = ALPAgent(env=env, discount_factor=env.discount_factor, coefficients=duals)
    print(agent.solve(init_state, 1))
    '''
    ((array([11,  0, 11,  0]), array([1, 1])), array([[1, 1],
       [0, 0],
       [0, 0]]), 1)
    '''
    #agent.train(debug=False) # 696.6424199999999
    #agent.train(debug=True) # 696.6424200000007
    '''
    for candidate, reduce_cost in agent.pricing_callback(duals):
        print('candidate:', candidate)
        candidate_cost = agent.get_obj_coefficient(candidate)
        candidate_coeffs = agent.get_constr_coefficients(candidate)
        print('candidate_coeffs:', candidate_coeffs)
        # reduced cost = cost − ∑ dual[j] * coeffs[j]
        approx_V = 0
        for j, coeff in enumerate(candidate_coeffs):
            approx_V += duals[j] * coeff
        rc = candidate_cost - approx_V
        print(candidate, reduce_cost, candidate_cost, approx_V, rc)
    '''

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



