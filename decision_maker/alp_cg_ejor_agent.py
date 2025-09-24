import numpy as np
import gurobipy as gp
from gurobipy import GRB

from decision_maker import InfiniteRTAgent
from metaheuristic_algorithm import ColumnGenerationSolver
from experiments import get_config_by_type
from utils import get_solution_value, solve_and_handle_errors, clean_value


class ALPEJORColumnGenerationAgent(InfiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, coefficients=None, pretrain=False):
        super().__init__(env, discount_factor, V, Q)
        self.is_trained = False
        self.E_u_alpha = [self.env.regular_capacity * 0.99 ** (i+1) for i in range(self.env.planning_horizon)]
        self.E_u_alpha[-1] = 0
        self.E_v_alpha = [self.env.overtime_capacity * 0.99 ** (i+1) for i in range(self.env.planning_horizon)]
        self.E_v_alpha[-1] = 0
        #self.E_w_alpha = self.env.arrival_generator.mean_by_type
        self.E_w_alpha = [1 for i in range(self.env.num_types)]
        if coefficients is not None:
            final_duals = coefficients
            self.is_trained = True
            self.W_0, self.U, self.V, self.W = self.get_coefficients(final_duals)
        if pretrain:
            self.train(debug=False)

    def train(self, debug=False, verbose=False):
        if debug == True:
            initial_columns = self.generate_all_columns()
        else:
            initial_columns = self.generate_initial_columns(verbose=verbose)
        self.cg_solver = ColumnGenerationSolver(master_builder=self.master_builder,
                                                pricing_callback=self.pricing_callback,
                                                initial_columns=initial_columns,
                                                get_constr_coefficients=self.get_constr_coefficients,
                                                get_obj_coefficient=self.get_obj_coefficient)
        self.cg_solver.solve()
        final_duals = [clean_value(c.Pi, 1e-8) for c in self.cg_solver.master_model.getConstrs()]
        self.W_0, self.U, self.V, self.W = self.get_coefficients(final_duals)
        self.is_trained = True
        return final_duals

    def get_coefficients(self, solution):
        it = iter(solution)
        W_0 = float(next(it))
        U = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        V = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        W = np.array([float(next(it)) for _ in range(self.env.num_types)])
        return W_0, U, V, W

    def get_approx_value_fn(self, state, W_0, U, V, W):
        regular_bookings, overtimes, waitlist = state
        return W_0 + np.dot(U, regular_bookings) + np.dot(V, overtimes) + np.dot(W, waitlist)

    def get_candidate(self, state_var, action_var):
        regular_booking_vars, overtime_vars, waitlist_vars = state_var
        advance_scheduling_decision_vars, overtime_decision_vars = action_var
        regular_bookings = get_solution_value(regular_booking_vars).astype(float)
        overtime = get_solution_value(overtime_vars).astype(float)
        waitlist = get_solution_value(waitlist_vars).astype(int)
        advance_scheduling_decision = get_solution_value(advance_scheduling_decision_vars).astype(int)
        overtime_decision = get_solution_value(overtime_decision_vars).astype(float)
        return ((regular_bookings, overtime, waitlist), (advance_scheduling_decision, overtime_decision))

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
                for j in range(self.env.planning_horizon)
            ),
            name="constr_U")
        master_model.addConstrs(
            (
                gp.LinExpr() >= self.E_v_alpha[j]
                for j in range(self.env.planning_horizon)
            ),
            name="constr_V")
        master_model.addConstrs(
            (
                gp.LinExpr() >= self.E_w_alpha[i]
                for i in range(self.env.num_types)
            ),
            name="constr_W")
        master_model.update()
        return master_model

    def pricing_callback(self, duals):
        """
        Solves the pricing subproblem by iterating through each time period.

        This improved approach solves N smaller, independent optimization problems,
        one for each period, instead of one large MIP. This is more efficient
        and directly calculates the minimum reduced cost for each period.
        """
        # --- 1. Initial Setup (Parameters and Duals) ---
        W_0, U, V, W = self.get_coefficients(duals)

        pricing_model = gp.Model(f"Pricing_Problem", env=self.grb_env)
        pricing_model.setParam('OutputFlag', 0)

        state_var = self.get_state_var(pricing_model)
        action_var = self.get_action_var(pricing_model, advance_scheduling_type=GRB.INTEGER)
        self.add_action_space_constraints(pricing_model, state_var, action_var)

        next_state_var = self.get_next_state(pricing_model, state_var, action_var, self.env.arrival_generator.mean_by_type)

        # --- Objective ---
        candidate_cost = self.env.cost_fn(state_var, action_var)
        approx_V = self.get_approx_value_fn(state_var, W_0, U, V, W)
        reduced_cost = candidate_cost + self.env.discount_factor * self.get_approx_value_fn(next_state_var, W_0, U, V, W) - approx_V
        pricing_model.setObjective(reduced_cost, GRB.MINIMIZE)
        if solve_and_handle_errors(pricing_model):
            pricing_model.write('positive_pricing.lp')
            candidate = self.get_candidate(state_var, action_var)
            yield candidate, pricing_model.ObjVal

    def generate_initial_state_action_pairs(self):
        for i in range(self.env.num_types):
            with (gp.Model("init_columns", env=self.grb_env) as init_columns_model):
                state_var = self.get_state_var(init_columns_model)
                action_var = self.get_action_var(init_columns_model, advance_scheduling_type=GRB.INTEGER)
                self.add_action_space_constraints(init_columns_model, state_var, action_var)
                next_state_var = self.get_next_state(init_columns_model, state_var, action_var,
                                                     self.env.arrival_generator.mean_by_type)
                regular_booking_vars, overtime_booking_vars, waitlist_vars = state_var
                advance_scheduling_decision_vars, overtime_decision_vars = action_var
                next_regular_booking_vars, next_overtime_booking_vars, next_waitlist_vars = next_state_var
                bookings_diff_var = regular_booking_vars - self.env.discount_factor * next_regular_booking_vars
                maximum_difference_var = init_columns_model.addVar(lb=-GRB.INFINITY, name='maximum_difference')
                init_columns_model.setObjective(maximum_difference_var, GRB.MAXIMIZE)
                init_columns_model.addConstrs(
                    (
                        maximum_difference_var <= bookings_diff_var[j]
                        for j in range(self.env.planning_horizon)
                    ),
                    name="maximum_difference",
                )
                init_columns_model.addConstrs(
                    (
                        overtime_booking_vars[j] == 0
                        for j in range(self.env.planning_horizon)
                    ),
                    name="zero_overtime_bookings",
                )
                init_columns_model.addConstrs(
                    (
                        overtime_decision_vars[j] == 0
                        for j in range(self.env.planning_horizon)
                    ),
                    name="zero_overtime_decisions",
                )
                init_columns_model.addConstrs(
                    (
                        waitlist_vars[k] == self.env.arrival_generator.maximum_arrival if k == i else 0
                        for k in range(self.env.num_types)
                    ),
                    name="waitlist_initialization",
                )
                if not solve_and_handle_errors(init_columns_model):
                    raise ValueError("initial set of columns is infeasible.")
                candidate = self.get_candidate(state_var, action_var)
                yield candidate

    def generate_initial_columns(self, debug=False,  verbose=False):
        if debug == True:
            initial_columns = self.generate_all_columns()
        else:
            initial_columns = self.generate_initial_state_action_pairs()
        self.cg_solver = ColumnGenerationSolver(master_builder=self.initial_columns_builder,
                                                pricing_callback=self.pricing_callback,
                                                initial_columns=initial_columns,
                                                get_constr_coefficients=self.get_constr_coefficients,
                                                get_obj_coefficient=None)
        initial_columns = self.cg_solver.initial_columns_solve(verbose=verbose)
        return initial_columns

    def initial_columns_builder(self):
        master_model = gp.Model("InitMasterRMP")
        master_model.setParam('OutputFlag', 0)
        # Artificial variable for W_0 constraint
        s_W0 = master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, name='art_W0')
        # Artificial variables for U constraints (planning horizon)
        s_U = master_model.addVars(
            self.env.planning_horizon, vtype=GRB.CONTINUOUS, lb=0, name="art_U"
        )
        # Artificial variables for V constraints (planning horizon)
        s_V = master_model.addVars(
            self.env.planning_horizon, vtype=GRB.CONTINUOUS, lb=0, name="art_V"
        )
        # Artificial variables for W constraints (types)
        s_W = master_model.addVars(
            self.env.num_types, vtype=GRB.CONTINUOUS, lb=0, name="art_W"
        )
        M  = 1e4
        # Objective: minimize sum of artificials
        master_model.setObjective(
            M * (s_W0 + gp.quicksum(s_U) + gp.quicksum(s_V) + gp.quicksum(s_W)),
            GRB.MINIMIZE
        )
        # Add constraints, each backed up by its own artificial
        master_model.addConstr(
            gp.LinExpr() + s_W0 == 1,
            name="constr_W_0"
        )
        master_model.addConstrs(
            (s_U[j] >= self.E_u_alpha[j] for j in range(self.env.planning_horizon)),
            name="constr_U"
        )
        master_model.addConstrs(
            (s_V[j] >= self.E_v_alpha[j] for j in range(self.env.planning_horizon)),
            name="constr_V"
        )
        master_model.addConstrs(
            (s_W[i] >= self.E_w_alpha[i] for i in range(self.env.num_types)),
            name="constr_W"
        )
        master_model.update()
        return master_model

    def get_constr_coefficients(self, candidate):
        mu = self.env.arrival_generator.mean_by_type
        gamma = self.env.discount_factor
        state, action = candidate
        regular_bookings, overtimes, waitlist = state
        new_regular_bookings, new_overtimes, new_waitlist = self.env.get_next_state(state, action, mu, is_var=False)
        # W_0 coefficient
        W_0 = 1 - gamma
        # Z coefficients
        U = (regular_bookings - gamma * new_regular_bookings).tolist()
        V = (overtimes - gamma * new_overtimes).tolist()
        # W_i coefficients
        W = (waitlist - gamma * new_waitlist).tolist()
        coefficients = [W_0] + U + V + W
        return coefficients

    def get_obj_coefficient(self, candidate):
        state, action = candidate
        return self.env.cost_fn(state, action)

    def coeff_C(self, i, n):
        part1 = sum(self.discount_factor ** k * self.env.holding_cost(k, i) for k in range(n + 1))
        part2 = sum(self.discount_factor * self.env.treatment_pattern[k+1-n, i] * self.U[k] for k in range(n-1,n-1+self.env.num_sessions))
        part3 = self.env.postponing_cost(i) - self.discount_factor * self.W[i]
        return part1 + part2 + part3

    def generate_all_columns(self):
        for column in self.env.generate_state_action_pairs():
            yield column


if "__main__" == __name__:
    config = get_config_by_type('ejor_default')
    env = config.env
    init_state = config.init_state
    agent = ALPEJORColumnGenerationAgent(env=env, discount_factor=env.discount_factor)
    #print(list(agent.generate_initial_state_action_pairs()))
    print(agent.train(debug=False, verbose=True))
    '''
    duals = [-21.5852964, 0.0319991, 0.0316791, 0.0313623, 0.0310487, 0.0323223, 0.0319991, 0.0316791, 0.0318956, 0.0315767, 0.0312609, 0.0325143, 0.0321891, 0.0318673, 0.0305981, 0.0302921, 0.0299892, 0.0296893, 0.0293924, 0.0290985, 0.0288075, 0.0285194, 0.0282342, 0.0279519, 0.0276724, 0.0273957, 0.0271217, 0.0268505, 0.026582, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1584114]

    for candidate, reduce_cost in agent.pricing_callback(duals):
        print(candidate)
        candidate_cost = agent.get_obj_coefficient(candidate)
        candidate_coeffs = agent.get_constr_coefficients(candidate)
        # reduced cost = cost − ∑ dual[j] * coeffs[j]
        rc = candidate_cost
        for j, coeff in enumerate(candidate_coeffs):
            rc -= duals[j] * coeff
        print(rc, reduce_cost)
    '''
    # master obj: 25250


