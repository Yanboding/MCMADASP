import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from decision_maker import FiniteRTAgent
from metaheuristic_algorithm import BenderDecompositionSolver

from utils import solve_and_handle_errors

class HindsightSAAgent(FiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, sample_path_number=500, current_decision_var_type='integer', future_decision_var_type='continuous', is_myopic=False):
        super().__init__(env=env, discount_factor=discount_factor, V=V, Q=Q)
        self.sample_path_number = sample_path_number
        self.current_decision_var_type = GRB.INTEGER if current_decision_var_type is None or current_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.future_decision_var_type = GRB.INTEGER if future_decision_var_type is None or future_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.is_myopic = is_myopic
        delta = []
        for omega in range(self.sample_path_number):
            new_arrivals = self.env.reset_arrivals(1)
            delta.append(new_arrivals)
        self.delta = np.array(delta)

    def set_sample_path(self, sample_path):
        self.sample_path_number = 1
        self.delta = np.array([sample_path])

    def direct_solve(self, state, t, action=None, verbose=False):
        H = self.env.decision_epoch - t
        # ---------- model ----------
        with (gp.Model("SA_Advance", env=self.grb_env) as m):
            m.setParam('DualReductions', 0)
            m.setParam("MultiObjPre", 0)
            m.setParam('MIPFocus', 1)
            # ---------- 1. today’s increments ----------
            action_t_var = self.get_action_var(model=m, t=t, advance_scheduling_type=GRB.INTEGER)
            if action is not None:
                self.set_action(action_var=action_t_var, action=action)
            # add action constraint
            self.add_action_space_constraints(model=m, state_var=state, action_var=action_t_var, t=t)
            # ---------- 1. objective ----------
            imm_cost = self.env.cost_fn(state, action_t_var, t)
            fut_cost = 0
            state_t_tau_vars = []
            action_t_tau_vars = []
            if not self.is_myopic:
                prev_state_scenario = [state for _ in range(self.sample_path_number)]
                prev_action_scenario = [action_t_var for _ in range(self.sample_path_number)]
                for tau in range(1, H + 1):
                    state_scenario = []
                    action_scenario = []
                    for omega in range(self.sample_path_number):
                        state_t_tau = self.get_next_state(model=m,
                                                          state=prev_state_scenario[omega],
                                                          action=prev_action_scenario[omega],
                                                          new_arrival=self.delta[omega, tau],
                                                          t=t+tau)
                        action_t_tau_var = self.get_action_var(model=m, t=t+tau,
                                                               advance_scheduling_type=GRB.CONTINUOUS)
                        self.add_action_space_constraints(model=m, state_var=state_t_tau, action_var=action_t_tau_var,
                                                          t=t+tau)
                        fut_cost += self.env.discount_factor ** tau * self.env.cost_fn(state_t_tau, action_t_tau_var,
                                                                                       t + tau)
                        state_scenario.append(state_t_tau)
                        action_scenario.append(action_t_tau_var)
                    state_t_tau_vars.append(state_scenario)
                    action_t_tau_vars.append(action_scenario)
                    prev_state_scenario = state_scenario
                    prev_action_scenario = action_scenario
                fut_cost = fut_cost / self.sample_path_number
            m.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)
            if not solve_and_handle_errors(m, verbose=verbose):
                raise RuntimeError("Master model optimal solution not found")
            '''
            cur_mem = m.getAttr(GRB.Attr.MemUsed)  # current RAM in GB
            peak_mem = m.getAttr(GRB.Attr.MaxMemUsed)  # peak RAM in GB
            print(f"Memory now: {cur_mem:.2f} GB  (peak {peak_mem:.2f} GB)")
            '''
            print('imm_cost:', imm_cost.getValue())
            if t >= self.env.decision_epoch or self.is_myopic:
                print('future_cost:', fut_cost)
            else:
                print('future_cost:', fut_cost.getValue())
            # ---------- 8. return ----------
            action = self.get_solution(action_t_var, is_final=True)
            return action, m.ObjVal, {}

    def master_builder_fn(self, state, t, action=None):
        master_model = gp.Model(f"SA_Advance_Master", env=self.grb_env)
        master_model.setParam('DualReductions', 0)
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam('MIPFocus', 1)
        # create action variables in period t
        action_t_var = self.get_action_var(model=master_model, t=t, advance_scheduling_type=GRB.INTEGER)
        # add action constraint
        self.add_action_space_constraints(model=master_model, state_var=state, action_var=action_t_var, t=t)
        if action is not None:
            self.set_action(action_var=action_t_var, action=action)
        # set imm_cost and a cost to go lb
        theta_vars = np.array(
            [master_model.addVar(vtype=GRB.CONTINUOUS, name=f"theta_{omega}") for omega in range(len(self.delta))])
        imm_cost = self.env.cost_fn(state, action_t_var, t)
        z = imm_cost + theta_vars.sum() / self.sample_path_number
        master_model.setObjective(z, GRB.MINIMIZE)
        return master_model, imm_cost, theta_vars, action_t_var

    def build_linking_constraints(self, model, action_t_var):
        x_t_var, y_t_var = action_t_var
        linking_constraints = []
        for j, row in enumerate(x_t_var):
            for i, var in enumerate(row):
                constraint = model.addConstr(var == 0.0, name=f'link_x_{j},{i}')
                linking_constraints.append(constraint)
        for j, var in enumerate(y_t_var):
            constraint = model.addConstr(var == 0.0, name=f'link_y_{j}')
            linking_constraints.append(constraint)
        return linking_constraints

    def subproblem_builder_fn(self, env, state, t, scenario_id):
        remaining_booking_window_size = self.env.decision_epoch - t + 1
        sub_model = gp.Model(f"Subproblem_SA_Advance_{scenario_id}", env=env)
        sub_model.setParam('InfUnbdInfo', 1)
        sub_model.setParam('DualReductions', 0)
        sub_model.setParam("MultiObjPre", 0)
        # get u^t+1 and linking constrs
        action_t_var = self.get_action_var(model=sub_model, t=t, advance_scheduling_type=GRB.CONTINUOUS)
        linking_constraints = self.build_linking_constraints(sub_model, action_t_var)
        # Initialize scenario state and action like in direct solution
        previous_state_var = state
        previous_action_var = action_t_var
        fut_cost = 0
        for tau in range(1, remaining_booking_window_size):
            state_t_tau_var = self.get_next_state(model=sub_model,
                                                  state=previous_state_var,
                                                  action=previous_action_var,
                                                  new_arrival=self.delta[scenario_id, tau],
                                                  t=t+tau)
            action_t_tau_var = self.get_action_var(model=sub_model, t=t+tau,
                                                   advance_scheduling_type=GRB.CONTINUOUS)
            self.add_action_space_constraints(model=sub_model, state_var=state_t_tau_var, action_var=action_t_tau_var,
                                              t=t+tau)
            fut_cost += self.env.discount_factor ** tau * self.env.cost_fn(state_t_tau_var, action_t_tau_var, t + tau)
            previous_state_var = state_t_tau_var
            previous_action_var = action_t_tau_var
        sub_model.setObjective(fut_cost, GRB.MINIMIZE)
        return sub_model, linking_constraints

    def solve(self, state, t, action=None, verbose=False):
        if self.is_myopic or self.sample_path_number <= 1:
            action, obj_value, info = self.direct_solve(state, t, action=action)
            return action, obj_value, info
        num_subproblems = self.sample_path_number
        bender_solver = BenderDecompositionSolver(master_builder_fn=self.master_builder_fn,
                                                  subproblem_builder_fn=self.subproblem_builder_fn,
                                                  get_solution=self.get_solution,
                                                  flatten_fn=None,
                                                  num_subproblems=num_subproblems)
        action_t, upper_bound, info = bender_solver.solve(master_builder_args={'state':state, 't': t, 'action': action},
                                                          subproblem_builder_args={'env': self.grb_env,
                                                                                   'state':state, 't': t},
                                                          tol=1e-6,
                                                          max_iter=15000,
                                                          verbose=verbose)
        return action_t, upper_bound, info

if __name__ == '__main__':
    from experiments import get_config_by_type

    config = get_config_by_type('finite_base_case')
    env = config.env
    discount_factor = env.discount_factor
    agent = HindsightSAAgent(env, discount_factor, **{'sample_path_number': 50, 'is_myopic': False})
    print('Init State:', config.init_state)
    print('Future arrivals:', agent.delta[0])
    start = time.time()
    # action = (np.array([[3, 1], [0, 2]]), np.array([2,0]))
    action = None
    action, obj_value, info = agent.solve(config.init_state, 1, action=action)
    print(time.time() - start)
    print('bender_decomposition:')
    print(obj_value)  # 423492.46229695214 979.8701978711838 # 127.03160285949707
    print(action)

    start = time.time()
    action = None
    action, obj_value, info = agent.direct_solve(config.init_state, 1, action=action)
    print(time.time() - start)
    print('direct solve:')
    print(obj_value) # 423493.53852545697
    print(action)
