import numpy as np
import gurobipy as gp
from gurobipy import GRB

from decision_maker import InfiniteRTAgent
from metaheuristic_algorithm import ColumnGenerationSolver
from experiments import get_config_by_type
from utils import get_solution_value, solve_and_handle_errors, clean_value

class InfiniteSAAAgent(InfiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, sample_path_number=100, current_decision_var_type='integer', future_decision_var_type='continuous', is_myopic=False):
        super().__init__(env, discount_factor, V=V, Q=Q)
        self.sample_path_number = sample_path_number
        self.current_decision_var_type = GRB.INTEGER if current_decision_var_type is None or current_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.future_decision_var_type = GRB.INTEGER if future_decision_var_type is None or future_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.is_myopic = is_myopic
        self.delta = []
        for omega in range(self.sample_path_number):
            new_arrivals = self.env.reset_arrivals()
            self.delta.append(new_arrivals)
            print(len(new_arrivals))

    def set_sample_path(self, sample_path):
        self.sample_path_number = 1
        self.delta = np.array([sample_path])

    def direct_solve(self, state, t, action=None, verbose=False):
        with (gp.Model("SA_Advance", env=self.grb_env) as m):
            m.setParam('DualReductions', 0)
            m.setParam("MultiObjPre", 0)
            m.setParam('MIPFocus', 1)
            # ---------- 1. today’s increments ----------
            action_t_var = self.get_action_var(model=m, advance_scheduling_type=GRB.INTEGER)
            if action is not None:
                self.set_action(action_var=action_t_var, action=action)
            # add action constraint
            self.add_action_space_constraints(model=m, state_var=state, action_var=action_t_var)
            # ---------- 1. objective ----------
            imm_cost = self.env.cost_fn(state, action_t_var)
            fut_cost = 0
            if not self.is_myopic:
                # for every sample path
                for omega in range(self.sample_path_number):
                    prev_state_var = state
                    prev_action_var = action_t_var
                    for tau, new_arrival in enumerate(self.delta[omega], start=1):
                        next_state_var = self.get_next_state(model=m,
                                                          state=prev_state_var,
                                                          action=prev_action_var,
                                                          new_arrival=new_arrival)
                        next_action_var = self.get_action_var(model=m, advance_scheduling_type=self.future_decision_var_type)
                        self.add_action_space_constraints(model=m, state_var=next_state_var, action_var=next_action_var)
                        fut_cost += self.env.cost_fn(next_state_var, next_action_var)
                        prev_state_var = next_state_var
                        prev_action_var = next_action_var
                fut_cost = fut_cost / self.sample_path_number
            m.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)
            if not solve_and_handle_errors(m, verbose=verbose):
                raise RuntimeError("Master model optimal solution not found")

            print('imm_cost:', imm_cost.getValue())
            if t >= self.env.decision_epoch or self.is_myopic:
                print('future_cost:', fut_cost)
            else:
                print('future_cost:', fut_cost.getValue())
            # ---------- 8. return ----------
            action = self.get_solution(action_t_var, is_final=True)
            return action, m.ObjVal, {}

if __name__ == "__main__":
    config = get_config_by_type('ejor_default')
    env = config.env
    agent = InfiniteSAAAgent(env=env, discount_factor=0.99, sample_path_number=1, is_myopic=False)
    state, info = env.reset()
    action, obj, _ = agent.direct_solve(state=state, t=1, verbose=True)

