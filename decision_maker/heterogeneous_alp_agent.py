import time
from collections import defaultdict
from pprint import pprint

import numpy as np
from scipy.stats import uniform
import gurobipy as gp
from gurobipy import GRB

from decision_maker import ALPAgent
from utils import get_solution_value, ColumnGenerationSolver, solve_and_handle_errors, clean_value


class HeterogeneousALPColumnGenerationAgent(ALPAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, coefficients=None, pretrain=False):
        super().__init__(env, discount_factor, V, Q)
        self.is_trained = False
        self.E_u_beta = {
            t: [uniform(loc=0, scale=self.env.regular_capacity).mean()] * (self.env.planning_horizon - t + 1)
            for t in range(1, self.env.decision_epoch + 1)}
        self.E_w_beta = {t: self.env.arrival_generator.mean_by_type for t in range(1, self.env.decision_epoch + 1)}
        if coefficients is not None:
            self.is_trained = True
            self.W_0, self.U, self.W = self.get_coefficients(coefficients)
        if pretrain:
            self.train(False)

    def master_builder(self):
        '''
        Total number of constraints: N + N+l-1 + N+l -2 ... + l + N*I
        '''
        master_model = gp.Model("MasterRMP")
        master_model.ModelSense = GRB.MINIMIZE
        master_model.setParam('OutputFlag', 0)
        master_model.addConstrs(
            (
                    gp.LinExpr() == 1
                    for _ in range(1, self.env.decision_epoch+1)
             ),
            name="constr_W_0")

        master_model.addConstrs(
            (
                gp.LinExpr() >= self.E_u_beta[t][j]
                for t in range(1, self.env.decision_epoch + 1)
                for j in range(self.env.planning_horizon - t + 1)
            ),
            name="constr_u")
        master_model.addConstrs(
            (
                gp.LinExpr() >= self.E_w_beta[t][i]
                for t in range(1, self.env.decision_epoch + 1)
                for i in range(self.env.num_types)
            ),
            name="constr_w")
        master_model.update()
        return master_model

    def pricing_callback(self, duals, max_attempts=1):
        """
        Solves the pricing subproblem by iterating through each time period.

        This improved approach solves N smaller, independent optimization problems,
        one for each period, instead of one large MIP. This is more efficient
        and directly calculates the minimum reduced cost for each period.
        """
        # --- 1. Initial Setup (Parameters and Duals) ---
        mu = self.env.arrival_generator.mean_by_type
        gamma = self.env.discount_factor
        W_0, U, W = self.get_coefficients(duals)
        models = {}

        for t in range(1, self.env.decision_epoch + 1):
            m = gp.Model(f"Pricing_Problem_t{t}", env=self.grb_env)
            m.setParam('OutputFlag', 0)
            state_var_t = self.get_state_var(m, t)
            action_var_t = self.get_action_var(m, state_var_t, t, 0)
            # --- Objective ---
            candidate_cost = self.env.cost_fn(state_var_t, action_var_t, t)
            approx_V_t = self.get_approx_value_fn(model=m,
                                                  state=state_var_t,
                                                  t=t,
                                                  W_0=W_0,
                                                  U=U,
                                                  W=W)
            new_state_var_t = self.env.get_next_state(state_var_t, action_var_t, mu, is_var=True)
            if t < self.env.decision_epoch:
                future_value_t = gamma * self.get_approx_value_fn(model=m,
                                                                  state=new_state_var_t,
                                                                  t=t + 1,
                                                                  W_0=W_0,
                                                                  U=U,
                                                                  W=W)
                dual_cost = approx_V_t - future_value_t
            else:
                dual_cost = approx_V_t
            reduced_cost_t = candidate_cost - dual_cost
            m.setObjective(reduced_cost_t, GRB.MINIMIZE)
            m.update()
            models[t] = (m, state_var_t, action_var_t, candidate_cost, dual_cost)
        # Now repeatedly solve until a new best solution is found
        for _ in range(max_attempts):
            best_reduced_cost = float("inf")
            best_solution = None

            # Solve all models, collect their best solutions
            for t, (m, state_var_t, action_var_t, candidate_cost, dual_cost) in models.items():
                m.optimize()
                if m.Status == GRB.OPTIMAL:
                    rc = m.ObjVal
                    if rc < best_reduced_cost:
                        best_solution = (state_var_t, action_var_t, t)
                        best_reduced_cost = rc
            state_var_t, action_var_t, t = best_solution
            candidate = self.get_candidate(state_var_t, action_var_t, t)
            # If new solution, yield and exit
            yield candidate, best_reduced_cost

            # Otherwise, add a no-good cut for this solution in its model and try again
            (regular_hour_booking_vars, waitlist_vars), (advance_scheduling_decision_vars, overtime_decision_vars), t = best_solution
            (regular_hour_bookings, waitlist), (advance_scheduling_decision, overtime_decision), t = candidate
            x_vars = list(regular_hour_booking_vars) + list(waitlist_vars) + list(advance_scheduling_decision_vars.flatten()) + list(overtime_decision_vars)
            x_vals = list(regular_hour_bookings) + list(waitlist) + list(advance_scheduling_decision.flatten()) + list(overtime_decision)
            self.eliminate_one_candidate(models[t][0], x_vars, x_vals, f'no_good_cut{t}_{_}')

        # If we exit loop, no new improving solution exists
        return
    def eliminate_one_candidate(self, model, vars, vals, name):
        delta_list = []
        for i, (var, val) in enumerate(zip(vars, vals)):
            delta_le = model.addVar(vtype=GRB.BINARY, name=f"delta_le_{i}")
            delta_ge = model.addVar(vtype=GRB.BINARY, name=f"delta_ge_{i}")
            model.addGenConstrIndicator(delta_le, True, var <= val - 1)
            model.addGenConstrIndicator(delta_ge, True, var >= val + 1)
            delta_list.extend([delta_le, delta_ge])
        model.addConstr(gp.quicksum(delta_list) >= 1, name=name)
        model.update()

    def generate_initial_columns(self):
        N = self.env.decision_epoch
        I = self.env.num_types
        gamma = self.env.discount_factor
        init_columns = []
        for t in range(1, N + 1):
            for i in range(I):
                with (gp.Model("init_columns", env=self.grb_env) as init_columns_model):
                    state_var = (regular_hour_booking_vars, waitlist_vars) = self.get_state_var(init_columns_model, t)
                    for regular_hour_booking_var in regular_hour_booking_vars:
                        regular_hour_booking_var.lb = self.env.regular_capacity
                    for k, waitlist_var in enumerate(waitlist_vars):
                        if k == i:
                            waitlist_var.lb = waitlist_var.ub = self.env.arrival_generator.maximum_arrival
                        else:
                            waitlist_var.lb = waitlist_var.ub = 0
                    action_var = (x_var_t, y_var_t) = self.get_action_var(init_columns_model, state_var, t, 0)
                    next_regular_hour_booking_vars = self.env.get_next_regular_bookings(regular_hour_booking_vars, x_var_t)
                    maximum_difference_var = init_columns_model.addVar(name='maximum_difference')
                    init_columns_model.addConstrs(
                        (
                            maximum_difference_var <= (
                                regular_hour_booking_vars[j] - gamma * next_regular_hour_booking_vars[j] if t < self.env.planning_horizon - j else regular_hour_booking_vars[
                                    j])
                            for j in range(len(regular_hour_booking_vars))
                        ),
                        name="C1_maximum_difference",
                    )

                    init_columns_model.setObjective(maximum_difference_var, GRB.MAXIMIZE)
                    if solve_and_handle_errors(init_columns_model):
                        candidate = self.get_candidate(state_var, action_var, t)
                        init_columns.append(candidate)
        return init_columns

    def get_constr_coefficients(self, candidate):
        N = self.env.decision_epoch
        I = self.env.num_types
        mu = self.env.arrival_generator.mean_by_type
        gamma = self.env.discount_factor
        state, action, t = candidate
        regular_hour_bookings, waitlist = state
        new_regular_bookings, new_waitlist = self.env.get_next_state(state, action, mu)
        # W_0 coefficient
        W_0 = []
        for k in range(1, N+1):
            val = 0
            if k == t:
                val = 1
            elif k == t+1 and t < N:
                val = -gamma
            W_0.append(val)

        # U^t coefficients
        U = []
        for k in range(1, N+1):
            for j in range(self.env.planning_horizon - k + 1):
                val = 0
                if k == t:
                    val = regular_hour_bookings[j]
                elif k == t+1 and t < N:
                    val = - gamma * new_regular_bookings[j]
                U.append(val)
        # W_i^t coefficients
        W_i = []
        for k in range(1, N + 1):
            for i in range(I):
                val = 0
                if k == t:
                    val = waitlist[i]
                elif k == t + 1 and t < N:
                    val = - gamma * new_waitlist[i]
                W_i.append(val)
        return W_0 + U + W_i

    def get_obj_coefficient(self, candidate):
        state, action, t = candidate
        return self.env.cost_fn(state, action, t)

    def generate_all_columns(self):
        for column in self.env.generate_state_action_pairs():
            yield column

    def train(self, debug=False, tol=1e-4, max_iter=3000):
        if debug == True:
            initial_columns = self.generate_all_columns()
        else:
            initial_columns = self.generate_initial_columns()
        self.cg_solver = ColumnGenerationSolver(master_builder=self.master_builder,
                                                pricing_callback=self.pricing_callback,
                                                initial_columns=initial_columns,
                                                get_constr_coefficients=self.get_constr_coefficients,
                                                get_obj_coefficient=self.get_obj_coefficient)
        self.cg_solver.solve(tol=tol, max_iter=max_iter)
        self.cg_solver.master_model.write('cg.lp')
        print('Candidates:')
        pprint(self.cg_solver.candidates_list)
        print('master obj:', self.cg_solver.master_model.ObjVal)
        final_duals = [clean_value(c.Pi, tol) for c in self.cg_solver.master_model.getConstrs()]
        self.W_0, self.U, self.W = self.get_coefficients(final_duals)
        self.is_trained = True
        return final_duals
    '''
    [((array([5.]), array([9])), (array([[9]]), array([18.])), 1),
    ((array([0.]), array([0])), (array([[0]]), array([0.])), 1),
    ((array([5.]), array([0])), (array([[0]]), array([0.])), 1),
    ((array([0.]), array([9])), (array([[9]]), array([13.])), 1),
    ((array([0.]), array([3])), (array([[3]]), array([1.])), 1)]
    '''



if __name__ == "__main__":
    from experiments import get_config_by_type

    # 54946.988268116984
    config = get_config_by_type('adv_default')
    env = config.env
    agent = HeterogeneousALPColumnGenerationAgent(env=env, discount_factor=env.discount_factor)
    coefficients = agent.train(debug=False, max_iter=1000)
    print('coefficients:', coefficients)
    for candidate, reduce_cost in agent.pricing_callback(coefficients):
        print(candidate, reduce_cost)
    '''
    ((array([11,  0, 11,  0]), array([1, 1])), array([[1, 1],
       [0, 0],
       [0, 0]]), 1)
    agent.train(debug=False) # 696.6424199999999
    # agent.train(debug=True) # 696.6424200000007

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

    # columns = agent.generate_initial_columns()
    # print(columns) 2767.7803639492995
    # -30.0 [15.     14.85   14.7015] [40. 20.]
    # 97.87294488252503 [41.03947446 44.91300951 48.74780922 44.00510413] [89.89858599  0.        ]
    '''
    '''
    {1: 0.0, 2: 0.0, 3: 0.0} {1: [4.73418, 2.9402999999999992, 0.0], 2: [2.9699999999999998, 0.0], 3: [0.0]} {1: [19.468360000000004, 9.73418], 2: [15.940000000000001, 7.970000000000001], 3: [10.0, 5.0]}
    '''
    #print(agent.solve(config.init_state, 1))



