from collections import defaultdict
from datetime import time
from pprint import pprint

import numpy as np
from scipy.stats import uniform
import gurobipy as gp
from gurobipy import GRB

from decision_maker import ALPEJORAgent
from utils import RowGenerationSolver, get_solution_value, solve_and_handle_errors, clean_value

class ALPEJORRowGenerationAgent(ALPEJORAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, coefficients=None, pretrain=False):
        super().__init__(env, discount_factor, V, Q)
        self.is_trained = False
        self.E_u_alpha = [self.env.regular_capacity * 0.94 ** (i) for i in range(self.env.planning_horizon)]
        self.E_u_alpha[-1] = 0
        self.E_v_alpha = [self.env.overtime_capacity * 0.9 ** (i) for i in range(self.env.planning_horizon)]
        self.E_v_alpha[-1] = 0
        self.E_w_alpha = self.env.arrival_generator.mean_by_type
        if coefficients is not None:
            self.is_trained = True
            self.W_0, self.U, self.V, self.W = self.get_coefficients(coefficients)
        if pretrain:
            self.train(False)

    def generate_all_candidates(self):
        for column in self.env.generate_state_action_pairs():
            yield column

    def get_constraint_data(self, model, candidate):
        state, action = candidate
        candidate_cost = self.env.cost_fn(state, action)
        approx_V_t = self.get_approx_value_fn(state=state,
                                              W_0=self.W_0_var,
                                              U=self.U_vars,
                                              V=self.V_vars,
                                              W=self.W_vars)
        new_state = self.env.get_next_state(state, action, self.env.arrival_generator.mean_by_type,
                                            is_var=False)
        approx_V_next = self.get_approx_value_fn(state=new_state,
                                                 W_0=self.W_0_var,
                                                 U=self.U_vars,
                                                 V=self.V_vars,
                                                 W=self.W_vars)
        return approx_V_t - self.env.discount_factor * approx_V_next <= candidate_cost

    def generate_initial_candidates(self):
        for i in range(self.env.num_types):
            with (gp.Model("init_candidates", env=self.grb_env) as init_candidates_model):
                state_var = (regular_hour_booking_vars, overtime_booking_vars, waitlist_vars) = self.get_state_var(init_candidates_model)
                for overtime_booking_var in overtime_booking_vars:
                    overtime_booking_var.lb = overtime_booking_var.ub = 0
                for k, waitlist_var in enumerate(waitlist_vars):
                    if k == i:
                        waitlist_var.lb = waitlist_var.ub = self.env.arrival_generator.maximum_arrival
                    else:
                        waitlist_var.lb = waitlist_var.ub = 0
                action_var = (advance_scheduling_decision_vars, overtime_decision_vars) = self.get_action_var(model=init_candidates_model,
                                                                                                              state=state_var)
                for overtime_decision_var in overtime_decision_vars:
                    overtime_decision_var.lb = overtime_decision_var.ub = 0
                next_regular_hour_booking_vars = self.env.get_next_bookings(regular_hour_booking_vars, action_var)
                regular_hour_booking_delta_vars = regular_hour_booking_vars - self.env.discount_factor * next_regular_hour_booking_vars
                maximum_difference_var = init_candidates_model.addVar(name='maximum_difference')
                init_candidates_model.addConstrs(
                    (
                        maximum_difference_var <= regular_hour_booking_delta_vars[j]
                        for j in range(self.env.planning_horizon)
                    ),
                    name="C1_maximum_difference",
                )
                init_candidates_model.setObjective(maximum_difference_var, GRB.MAXIMIZE)
                if solve_and_handle_errors(init_candidates_model):
                    candidate = self.get_candidate(state_var, action_var)
                    yield candidate

    def master_builder(self):
        master_model = gp.Model('MasterRMP')
        BIGM = 1e10
        self.W_0_var = master_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=BIGM, name=f"W_0")
        self.U_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=BIGM, name=f"U_{j}") for j in range(self.env.planning_horizon)])
        self.V_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=BIGM, name=f"V_{j}") for j in range(self.env.planning_horizon)])
        self.W_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=BIGM, name=f"W_{i}") for i in range(1, self.env.num_types + 1)])
        obj = self.W_0_var + np.dot(self.U_vars, self.E_u_alpha) + np.dot(self.V_vars, self.E_v_alpha) + np.dot(self.W_vars, self.E_w_alpha)
        master_model.setObjective(obj, GRB.MAXIMIZE)
        master_model.update()
        return master_model

    def separation_callback(self, solution):
        W_0, U, V, W = self.get_coefficients(solution)
        m = gp.Model(f"Separation_Problem_t", env=self.grb_env)
        m.setParam('OutputFlag', 0)
        state_var = self.get_state_var(m)
        action_var = self.get_action_var(m, state_var)
        # --- Objective ---
        candidate_cost = self.env.cost_fn(state_var, action_var)
        approx_V = self.get_approx_value_fn(state=state_var,
                                            W_0=W_0,
                                            U=U,
                                            V=V,
                                            W=W)
        new_state_var = self.env.get_next_state(state=state_var,
                                                action=action_var,
                                                new_arrival=self.env.arrival_generator.mean_by_type,
                                                is_var=True)
        approx_V_next = self.get_approx_value_fn(state=new_state_var,
                                                 W_0=W_0,
                                                 U=U,
                                                 V=V,
                                                 W=W)
        dual_cost = approx_V - self.env.discount_factor * approx_V_next
        violation = dual_cost - candidate_cost
        m.setObjective(violation, GRB.MAXIMIZE)
        if solve_and_handle_errors(m):
            candidate = self.get_candidate(state_var, action_var)
            yield candidate, m.ObjVal

    def train(self, debug, tol=1e-6, max_iter=1000):
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
        final_coefficients = [clean_value(v.X, tol) for v in self.rg_solver.master_model.getVars()]
        self.W_0, self.U, self.V, self.W = self.get_coefficients(final_coefficients)
        self.is_trained = True
        return final_coefficients

    def coeff_C(self, i, n):
        part1 = sum(self.discount_factor ** k * self.env.holding_cost(k, i) for k in range(n + 1))
        part2 = sum(self.discount_factor * self.env.treatment_pattern[k+1-n, i] * self.U[k] for k in range(n-1,n-1+self.env.num_sessions))
        part3 = self.env.postponing_cost(i) - self.discount_factor * self.W[i]
        return part1 + part2 + part3

if "__main__" == __name__:
    from experiments import get_config_by_type
    config = get_config_by_type('ejor_default')
    env = config.env
    init_state = config.init_state
    agent = ALPEJORRowGenerationAgent(env=env, discount_factor=env.discount_factor)
    print(agent.train(debug=False))
    #duals = [84.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    #print(list(agent.separation_callback(duals)))