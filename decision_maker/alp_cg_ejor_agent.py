import time
from collections import defaultdict
from pprint import pprint

import numpy as np
from scipy.stats import uniform, geom
import gurobipy as gp
from gurobipy import GRB

from decision_maker import ALPEJORAgent
from environment.utility import get_valid_advance_actions
from experiments import get_config_by_type
from utils import get_solution_value, ColumnGenerationSolver, generate_state_action_pairs, solve_and_handle_errors, \
    iter_to_tuple, clean_value


class ALPEJORColumnGenerationAgent(ALPEJORAgent):
    TOKEN_WAIT = 15

    def __init__(self, env, discount_factor, V=None, Q=None, coefficients=None, pretrain=False):
        super().__init__(env, discount_factor, V, Q)
        self.is_trained = False
        self.E_u_alpha = [uniform(loc=0, scale=self.env.regular_capacity).mean() for i in range(self.env.planning_horizon)]
        self.E_u_alpha[-1] = 0
        self.E_v_alpha = [uniform(loc=0, scale=self.env.overtime_capacity).mean() for i in range(self.env.planning_horizon)]
        self.E_v_alpha[-1] = 0
        #self.E_w_alpha = self.env.arrival_generator.mean_by_type
        self.E_w_alpha = [1 for i in range(self.env.num_types)]
        if coefficients is not None:
            final_duals = coefficients
            self.is_trained = True
            self.W_0, self.U, self.V, self.W = self.get_coefficients(final_duals)
        if pretrain:
            self.train(debug=False)

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
        print('master obj:', self.cg_solver.master_model.ObjVal)
        final_duals = [clean_value(c.Pi, 1e-8) for c in self.cg_solver.master_model.getConstrs()]
        self.W_0, self.U, self.V, self.W = self.get_coefficients(final_duals)
        self.is_trained = True
        print(final_duals)
        return final_duals

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
        action_var = (advance_scheduling_decision_vars, overtime_decision_vars) = self.get_action_var(pricing_model, state_var)

        next_state_var = self.env.get_next_state(state_var, action_var, self.env.arrival_generator.mean_by_type, is_var=True)

        booking_slots = self.env.convert_action_to_booking_slots(advance_scheduling_decision_vars)
        pricing_model.addConstrs(
            (
                booking_slots[m] >= overtime_decision_vars[m]
                for m in range(self.env.planning_horizon)
            ),
            name="C4_valid_new_appointment_slots",
        )
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
                state_var = (regular_hour_booking_vars, overtime_booking_vars, waitlist_vars) = self.get_state_var(
                    init_columns_model)
                # v_m = y_m = 0 \forall m
                for overtime_booking_var in overtime_booking_vars:
                    overtime_booking_var.lb = overtime_booking_var.ub = 0
                # w_i = 3m_i, w_j = 0 \forall j \not = i
                for k, waitlist_var in enumerate(waitlist_vars):
                    if k == i:
                        waitlist_var.lb = waitlist_var.ub = self.env.arrival_generator.maximum_arrival
                    else:
                        waitlist_var.lb = waitlist_var.ub = 0
                action_var = (advance_scheduling_decision_vars, overtime_decision_vars) = self.get_action_var(model=init_columns_model,
                                                                                                              state=state_var)
                for overtime_decision_var in overtime_decision_vars:
                    overtime_decision_var.lb = overtime_decision_var.ub = 0
                # Action Space Constraints
                next_regular_hour_next_bookings_var = self.env.get_next_bookings(regular_hour_booking_vars, action_var)
                bookings_diff_var = regular_hour_booking_vars - self.env.discount_factor * next_regular_hour_next_bookings_var
                maximum_difference_var = init_columns_model.addVar(lb=-GRB.INFINITY, name='maximum_difference')
                init_columns_model.addConstrs(
                    (
                        maximum_difference_var <= bookings_diff_var[j]
                        for j in range(self.env.planning_horizon)
                    ),
                    name="C1_maximum_difference",
                )
                init_columns_model.setObjective(maximum_difference_var, GRB.MAXIMIZE)
                if solve_and_handle_errors(init_columns_model):
                    candidate = self.get_candidate(state_var, action_var)
                    yield candidate

    def generate_initial_columns(self, debug=False):
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
        return initial_columns

    def initial_columns_builder(self):
        master_model = gp.Model("InitMasterRMP")
        s_var = master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f'init_s')
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
                for j in range(self.env.planning_horizon)
            ),
            name="constr_U")
        master_model.addConstrs(
            (
                s_var >= self.E_v_alpha[j]
                for j in range(self.env.planning_horizon)
            ),
            name="constr_V")
        master_model.addConstrs(
            (
                s_var >= self.E_w_alpha[i]
                for i in range(self.env.num_types)
            ),
            name="constr_W")
        master_model.update()
        return master_model

    def get_constr_coefficients(self, candidate):
        mu = self.env.arrival_generator.mean_by_type
        gamma = self.env.discount_factor
        state, action = candidate
        bookings, overtimes, waitlist = state
        new_bookings, new_overtimes, new_waitlist = self.env.get_next_state(state, action, mu, is_var=False)
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
    print(agent.train(debug=False))
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


