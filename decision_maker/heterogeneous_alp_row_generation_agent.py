from collections import defaultdict
from datetime import time
from pprint import pprint

import numpy as np
from scipy.stats import uniform
import gurobipy as gp
from gurobipy import GRB

from decision_maker import ALPAgent
from utils import RowGenerationSolver, get_solution_value, solve_and_handle_errors, clean_value


class HeterogeneousALPRowGenerationAgent(ALPAgent):

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

    def generate_all_candidates(self):
        for column in self.env.generate_state_action_pairs():
            yield column

    def generate_initial_candidates(self):
        for t in range(1, self.env.decision_epoch+1):
            for i in range(self.env.num_types):
                with (gp.Model("init_columns", env=self.grb_env) as init_columns_model):
                    state_var = (regular_hour_booking_vars, waitlist_vars) = self.get_state_var(init_columns_model, t)
                    for regular_hour_booking_var in regular_hour_booking_vars:
                        regular_hour_booking_var.lb = self.env.regular_capacity
                    for k, waitlist_var in enumerate(waitlist_vars):
                        if k == i:
                            waitlist_var.lb = waitlist_var.ub = self.env.arrival_generator.maximum_arrival
                        else:
                            waitlist_var.lb = waitlist_var.ub = 0
                    action_var = self.get_action_var(init_columns_model, state_var, t, 0)
                    next_regular_hour_booking_vars = self.env.get_next_regular_bookings(state_var, action_var, is_var=True)
                    maximum_difference_var = init_columns_model.addVar(name='maximum_difference')
                    init_columns_model.addConstrs(
                        (
                            maximum_difference_var <= (
                                regular_hour_booking_vars[j] - self.env.discount_factor * next_regular_hour_booking_vars[j] if t < self.env.planning_horizon - j else regular_hour_booking_vars[
                                    j])
                            for j in range(len(regular_hour_booking_vars))
                        ),
                        name="C1_maximum_difference",
                    )

                    init_columns_model.setObjective(maximum_difference_var, GRB.MAXIMIZE)
                    if solve_and_handle_errors(init_columns_model):
                        candidate = self.get_candidate(state_var, action_var, t)
                        yield candidate

    def master_builder(self):
        master_model = gp.Model('MasterRMP')
        self.W_0_vars = {k: master_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"W^{k}_0") for k in
                         range(1, self.env.decision_epoch + 1)}
        self.U_vars = {k: [master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U^{k}_{j}") for j in
                           range(self.env.planning_horizon - k + 1)]
                       for k in range(1, self.env.decision_epoch + 1)}
        self.W_vars = {k: [master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"W^{k}_{i}") for i in
                           range(1, self.env.num_types+1)] for k in range(1, self.env.decision_epoch + 1)}
        obj = 0
        for t in range(1, self.env.decision_epoch + 1):
            obj += self.W_0_vars[t]
        for j in range(self.env.planning_horizon):
            for t in range(1, min(self.env.decision_epoch, self.env.planning_horizon - j) + 1):
                obj += self.E_u_beta[t][j] * self.U_vars[t][j]
        print(self.E_w_beta.keys())
        for i in range(self.env.num_types):
            for t in range(1, self.env.decision_epoch + 1):
                obj += self.E_w_beta[t][i] * self.W_vars[t][i]
        master_model.setObjective(obj, GRB.MAXIMIZE)
        master_model.update()
        return master_model

    def separation_callback(self, solution):
        W_0, U, W = self.get_coefficients(solution)
        models = {}
        for t in range(1, self.env.decision_epoch + 1):
            m = gp.Model(f"Separation_Problem_t{t}", env=self.grb_env)
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
            new_state_var_t = self.env.get_next_state(state=state_var_t,
                                                      action=action_var_t,
                                                      new_arrival=self.env.arrival_generator.mean_by_type,
                                                      is_var=True)
            if t < self.env.decision_epoch:
                future_value_t = self.env.discount_factor * self.get_approx_value_fn(model=m,
                                                                  state=new_state_var_t,
                                                                  t=t + 1,
                                                                  W_0=W_0,
                                                                  U=U,
                                                                  W=W)
                dual_cost = approx_V_t - future_value_t
            else:
                dual_cost = approx_V_t
            violation = dual_cost - candidate_cost
            m.setObjective(violation, GRB.MAXIMIZE)
            m.update()
            models[t] = (m, state_var_t, action_var_t, candidate_cost, dual_cost)
        maximum_violation = -float("inf")
        best_solution = None
        # Solve all models, collect their best solutions
        for t, (m, state_var_t, action_var_t, candidate_cost, dual_cost) in models.items():
            m.optimize()
            if m.Status == GRB.OPTIMAL:
                violation = m.ObjVal
                if violation > maximum_violation:
                    best_solution = (state_var_t, action_var_t, t)
                    maximum_violation = violation
        state_var_t, action_var_t, t = best_solution
        candidate = self.get_candidate(state_var_t, action_var_t, t)
        # If new solution, yield and exit
        yield candidate, maximum_violation

    def get_constraint_data(self, model, candidate):
        state, action, t = candidate
        candidate_cost = self.env.cost_fn(state, action, t)
        approx_V_t = self.get_approx_value_fn(model=model,
                                              state=state,
                                              t=t,
                                              W_0=self.W_0_vars,
                                              U=self.U_vars,
                                              W=self.W_vars)
        approx_V_next = 0
        if t < self.env.decision_epoch:
            new_state = self.env.get_next_state(state, action, self.env.arrival_generator.mean_by_type,
                                                is_var=True)
            approx_V_next += self.get_approx_value_fn(model=model,
                                                      state=new_state,
                                                      t=t + 1,
                                                      W_0=self.W_0_vars,
                                                      U=self.U_vars,
                                                      W=self.W_vars)
        return approx_V_t - self.env.discount_factor * approx_V_next <= candidate_cost

    def train(self, debug, tol=1e-6, max_iter=10000):
        if debug == True:
            initial_candidates = self.generate_all_candidates()
        else:
            initial_candidates = self.generate_initial_candidates()
        self.rg_solver = RowGenerationSolver(master_builder=self.master_builder,
                                             separation_callback=self.separation_callback,
                                             get_constraint_data=self.get_constraint_data,
                                             initial_candidates=initial_candidates)
        self.rg_solver.solve(tol=tol, max_iter=max_iter)
        self.rg_solver.master_model.write('rg.lp')
        print('master obj:', self.rg_solver.master_model.ObjVal)
        final_coefficients = [clean_value(v.X, tol) for v in self.rg_solver.master_model.getVars()]
        self.W_0, self.U, self.W = self.get_coefficients(final_coefficients)
        print('W_0:', self.W_0)
        print('W:', self.W)
        print('U:', self.U)

        self.is_trained = True
        return final_coefficients


if __name__ == "__main__":
    from experiments import get_config_by_type
    # one time cost: 148.80349999999999

    # 54946.988268116984
    config = get_config_by_type('adv_default')
    env = config.env
    init_state = config.init_state
    agent = HeterogeneousALPRowGenerationAgent(env=env, discount_factor=env.discount_factor)
    coefficients = agent.train(debug=False, max_iter=1000)
    print('init_state:', init_state)
    #action = (np.array([[3, 3],[0, 0],[0, 0]]), np.array([ 4.,  0., 0.]))
    action = None
    print(agent.simplified_solve(config.init_state, 1, action=action))
    # 1344.6

    #print(coefficients)
    # [0.0, 0.0, 0.0, 4.73418, 2.9402999999999992, 0.0, 2.9699999999999998, 0.0, 0.0, 19.46836, 9.73418, 15.940000000000001, 7.970000000000001, 10.0, 5.0]