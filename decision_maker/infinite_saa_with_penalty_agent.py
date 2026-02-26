import copy
import numpy as np
import gurobipy as gp
import time
from gurobipy import GRB

from decision_maker import InfiniteRTAgent
from metaheuristic_algorithm import BenderDecompositionSolver, ColumnGenerationSolver
from utils import solve_and_handle_errors, flatten, set_link_rhs, get_solution_value

class InfinitePenalizedSAAAgent(InfiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, 
                 sample_path_number=100, 
                 current_decision_var_type='integer', 
                 future_decision_var_type='continuous', 
                 is_myopic=False, sample_path=None, 
                 is_include_discount_factor=False,
                 sample_path_length=None, 
                 is_quasi_MC=True,
                 coefficients=1,
                 generating_function=None,
                 verbose=False):
        super().__init__(env, discount_factor, V=V, Q=Q)
        self.sample_path_number = sample_path_number
        self.current_decision_var_type = GRB.INTEGER if current_decision_var_type is None or current_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.future_decision_var_type = GRB.INTEGER if future_decision_var_type is None or future_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.is_myopic = is_myopic
        self.arrival_generator = copy.deepcopy(self.env.arrival_generator)
        self.sample_path = sample_path
        self.sample_path_length = sample_path_length
        self.coefficients = coefficients
        self.generating_function = generating_function
        # self.W_0, self.U, self.V, self.W = self.get_coefficients(coefficients)
        self.delta = []
        if self.sample_path is not None:
            self.set_sample_path(self.sample_path[1:])
        # sample path length is sample_path_length
        if not is_myopic and sample_path is None:
            if self.sample_path_length is None:
                if is_quasi_MC:
                    self.delta = self.arrival_generator.quasi_rvs(size=self.sample_path_number)
                else:
                    self.delta = self.arrival_generator.mc_rvs(size=self.sample_path_number)
        for omega in range(len(self.delta)):
            print(f'sample path {omega} length:', len(self.delta[omega]))
        self.bender_solver = None
        self.is_include_discount_factor = is_include_discount_factor
        self.direct_model, self.state_linking_constraints, self.action_t_var = None, None, None


    def set_sample_path(self, sample_path):
        self.sample_path_number = 1
        self.delta = np.array([sample_path])
    
    def direct_builder_fn(self):
        direct_model = gp.Model(f"SA_Advance_Direct_Model", env=self.grb_env)
        direct_model.setParam("MultiObjPre", 0)
        direct_model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        direct_model.setParam("MIPGap", 1e-9)
        direct_model.setParam("FeasibilityTol", 1e-9)
        direct_model.setParam("OptimalityTol", 1e-9)
        # ---------- 1. today’s increments ----------
        state_var = self.get_state_var(direct_model)
        state_linking_constraints = self.build_state_linking_constraints(direct_model, state_var)
        action_t_var = self.get_action_var(model=direct_model, advance_scheduling_type=GRB.INTEGER)
        # add action constraint
        self.add_action_space_constraints(model=direct_model, state_var=state_var, action_var=action_t_var)
        # ---------- 1. objective ----------
        imm_cost = self.env.cost_fn(state_var, action_t_var, is_var=True)
        fut_cost = 0
        costs = [[imm_cost] for _ in range(self.sample_path_number)]
        actions = [[action_t_var] for _ in range(self.sample_path_number)]
        penalties = [[] for _ in range(self.sample_path_number)]
        # for every sample path
        for omega in range(self.sample_path_number):
            prev_state_var = state_var
            prev_action_var = action_t_var
            for tau, new_arrival in enumerate(self.delta[omega], start=1):
                next_state_var = self.get_next_state(model=direct_model,
                                                    state=prev_state_var,
                                                    action=prev_action_var,
                                                    new_arrival=new_arrival)
                next_action_var = self.get_action_var(model=direct_model, advance_scheduling_type=self.future_decision_var_type)
                self.add_action_space_constraints(model=direct_model, state_var=next_state_var, action_var=next_action_var)
                
                penalty = self.coefficients * self.generating_function.calculate_penalty(prev_state_var, actions[omega][-1], new_arrival, is_var=True)
                one_time_cost = self.env.cost_fn(next_state_var, next_action_var, is_var=True)
                if self.is_include_discount_factor:
                    cost = (self.discount_factor ** tau) * (one_time_cost + penalty)
                else:
                    cost = one_time_cost + penalty
                costs[omega].append(one_time_cost)
                actions[omega].append(next_action_var)
                penalties[omega].append(penalty)
                fut_cost += cost
                prev_state_var = next_state_var
                prev_action_var = next_action_var
        fut_cost = fut_cost / self.sample_path_number
        direct_model.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)
        info = {
            'costs': costs,
            'actions': actions,
            'penalties': penalties
        }
        print(self.delta)
        return direct_model, state_linking_constraints, action_t_var, info

    def direct_solve(self, state, t=1, action=None, verbose=False):
        if self.direct_model is None:
            self.direct_model, self.state_linking_constraints, self.action_t_var, info = self.direct_builder_fn()
        flatten_state = flatten(state)
        set_link_rhs(self.state_linking_constraints, flatten_state)
        if action is not None:
            self.set_action(action_var=self.action_t_var, action=action)
        # Clean solution before resolving
        self.direct_model.reset()
        start = time.time()
        if not solve_and_handle_errors(self.direct_model, verbose=verbose):
            raise RuntimeError("Master model optimal solution not found")
        print(f"Direct model solve time: {time.time() - start} seconds")
        # ---------- 8. return ----------
        action = self.get_solution(self.action_t_var, is_final=True)
        return action, self.direct_model.ObjVal, info

    
    def master_builder_fn(self):
        master_model = gp.Model(f"SA_Advance_Master", env=self.grb_env)
        master_model.setParam('DualReductions', 0)
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam('MIPFocus', 1)
        master_model.setParam("FeasibilityTol", 1e-8)
        master_model.setParam("OptimalityTol", 1e-8)
        state_var = self.get_state_var(master_model)
        state_linking_constraints = self.build_state_linking_constraints(master_model, state_var)
        # create action variables in period t
        action_t_var = self.get_action_var(model=master_model, advance_scheduling_type=GRB.INTEGER)
        # add action constraint
        self.add_action_space_constraints(model=master_model, state_var=state_var, action_var=action_t_var)
        # set imm_cost and a cost to go lb
        theta_vars = np.array(
            [master_model.addVar(vtype=GRB.CONTINUOUS, name=f"theta_{omega}") for omega in range(len(self.delta))])
        imm_cost = self.env.cost_fn(state_var, action_t_var, is_var=True)
        z = imm_cost + theta_vars.sum() / self.sample_path_number
        master_model.setObjective(z, GRB.MINIMIZE)
        return master_model, imm_cost, theta_vars, action_t_var, state_linking_constraints
    
    def subproblem_builder_fn(self, env, scenario_id):
        sub_model = gp.Model(f"Subproblem_SA_Advance_{scenario_id}", env=env)
        sub_model.setParam('InfUnbdInfo', 1)
        # Forbidden the model to simplify the model(remove variables/constraints, tighten bounds, etc.). 
        sub_model.setParam('DualReductions', 0)
        sub_model.setParam("MultiObjPre", 0)
        sub_model.setParam("FeasibilityTol", 1e-8)
        sub_model.setParam("OptimalityTol", 1e-8)
        state_var = self.get_state_var(sub_model)
        state_linking_constraints = self.build_state_linking_constraints(sub_model, state_var)
        action_t_var = self.get_action_var(model=sub_model, advance_scheduling_type=GRB.CONTINUOUS)
        action_linking_constraints = self.build_action_linking_constraints(sub_model, action_t_var)
        # Initialize scenario state and action like in direct solution
        fut_cost = 0
        prev_state_var = state_var
        prev_action_var = action_t_var
        for tau, new_arrival in enumerate(self.delta[scenario_id], start=1):
            next_state_var = self.get_next_state(model=sub_model,
                                                 state=prev_state_var,
                                                 action=prev_action_var,
                                                 new_arrival=new_arrival)
            next_action_var = self.get_action_var(model=sub_model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=sub_model, state_var=next_state_var, action_var=next_action_var)
            # penalty for ALP
            penalty = np.dot(self.W, (self.env.arrival_generator.mean_by_type - new_arrival))
            cost = self.env.cost_fn(next_state_var, next_action_var, is_var=True) + penalty
            if self.is_include_discount_factor:
                cost = (self.discount_factor ** tau) * cost
            fut_cost += cost
            prev_state_var = next_state_var
            prev_action_var = next_action_var
        sub_model.setObjective(fut_cost, GRB.MINIMIZE)
        return sub_model, action_linking_constraints, state_linking_constraints

    def build_action_linking_constraints(self, model, action_t_var):
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
    
    def build_state_linking_constraints(self, model, state_var):
        u_var, v_var, w_var = state_var
        linking_constraints = []
        for j, uj_var in enumerate(u_var):
            constraint = model.addConstr(uj_var == 0.0, name=f'link_u_{j}')
            linking_constraints.append(constraint)
        for j, vj_var in enumerate(v_var):
            constraint = model.addConstr(vj_var == 0.0, name=f'link_v_{j}')
            linking_constraints.append(constraint)
        for i, wi_var in enumerate(w_var):
            constraint = model.addConstr(wi_var == 0.0, name=f'link_w_{i}')
            linking_constraints.append(constraint)
        return linking_constraints

    def solve(self, state, t=1, action=None, verbose=False):
        if self.is_myopic or self.sample_path_number <= 1:
            action, obj_value, info = self.direct_solve(state, t=t, action=action)
            return action, obj_value, info
        
        if self.bender_solver is None:
            self.bender_solver = BenderDecompositionSolver(master_builder_fn=self.master_builder_fn,
                                                       master_builder_args={},
                                                        subproblem_builder_fn=self.subproblem_builder_fn,
                                                        subproblem_builder_args={'env':self.grb_env},
                                                        get_solution=self.get_solution,
                                                        flatten_fn=None,
                                                        num_subproblems=self.sample_path_number)
        action_t, upper_bound, info = self.bender_solver.solve(state=state,
                                                               action=action,
                                                                tol=1e-6,
                                                                max_iter=12000,
                                                                verbose=verbose)
        return action_t, upper_bound, info
    
    def add_absolute_var(self, model, expression, name_prefix):
        abs_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"{name_prefix}_abs")
        model.addConstr(abs_var >= expression, name=f"{name_prefix}_pos_bound")
        model.addConstr(abs_var >= -expression, name=f"{name_prefix}_neg_bound")
        return abs_var
    
    def train(self, verbose=False):
        direct_model = gp.Model(f"SA_Advance_Direct_Model", env=self.grb_env)
        direct_model.setParam("MultiObjPre", 0)
        direct_model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        direct_model.setParam("MIPGap", 1e-9)
        direct_model.setParam("FeasibilityTol", 1e-9)
        direct_model.setParam("OptimalityTol", 1e-9)
        # ---------- 1. objective ----------
        coefficient_bound = 100
        total_cost = 0
        costs = [[] for _ in range(self.sample_path_number)]
        states = [[] for _ in range(self.sample_path_number)]
        actions = [[] for _ in range(self.sample_path_number)]
        post_action_states = [[] for _ in range(self.sample_path_number)]
        # for every sample path
        total_post_action_regular_bookings_vars = 0
        total_post_action_overtimes_vars = 0
        total_post_action_waitlist_vars = 0
        total_advance_scheduling_decision_vars = 0
        total_overtime_decision_vars = 0
        for omega in range(self.sample_path_number):
            state = self.env.generate_initial_state()
            state_var = self.get_state_var_fast(direct_model)
            state_linking_constraints = self.build_state_linking_constraints(direct_model, state_var)
            flatten_state = flatten(state)
            set_link_rhs(state_linking_constraints, flatten_state)
            action_var = self.get_action_var_fast(model=direct_model, advance_scheduling_type=GRB.CONTINUOUS)
            # add action constraint
            self.add_action_space_constraints_fast(model=direct_model, state_var=state_var, action_var=action_var)
            post_action_state_var = self.env.post_action_state(state_var, action_var, is_var=True)
            one_time_cost = self.env.cost_fn(state_var, action_var, is_var=True)
            prev_state_var = state_var
            prev_action_var = action_var
            states[omega].append(prev_state_var)
            actions[omega].append(prev_action_var)
            post_action_states[omega].append(post_action_state_var)
            costs[omega].append(one_time_cost)
            for tau, new_arrival in enumerate(self.delta[omega][1:], start=1):
                total_arrival_difference = np.sum(self.env.arrival_generator.mean_by_type - new_arrival)
                next_state_var = self.get_next_state_fast(model=direct_model,
                                                    state=prev_state_var,
                                                    action=prev_action_var,
                                                    new_arrival=new_arrival)
                next_action_var = self.get_action_var_fast(model=direct_model, advance_scheduling_type=self.future_decision_var_type)
                self.add_action_space_constraints_fast(model=direct_model, state_var=next_state_var, action_var=next_action_var)
                post_action_state_var = self.env.post_action_state(state=next_state_var, action=next_action_var, is_var=True)
                (post_action_regular_bookings_vars, post_action_overtimes_vars, post_action_waitlist_vars) = post_action_state_var
                total_post_action_regular_bookings_vars += total_arrival_difference * post_action_regular_bookings_vars
                total_post_action_overtimes_vars += total_arrival_difference * post_action_overtimes_vars
                total_post_action_waitlist_vars += total_arrival_difference * post_action_waitlist_vars
                advance_scheduling_decision, overtime_decision = prev_action_var
                total_advance_scheduling_decision_vars += total_arrival_difference * advance_scheduling_decision
                total_overtime_decision_vars += total_arrival_difference * overtime_decision
                
                
                one_time_cost = self.env.cost_fn(next_state_var, next_action_var, is_var=True)
                total_cost += one_time_cost
                prev_state_var = next_state_var
                prev_action_var = next_action_var
                states[omega].append(prev_state_var)
                actions[omega].append(prev_action_var)
                post_action_states[omega].append(post_action_state_var)
                costs[omega].append(one_time_cost)

        average_post_action_regular_bookings_vars = total_post_action_regular_bookings_vars / self.sample_path_number
        average_post_action_overtimes_vars = total_post_action_overtimes_vars / self.sample_path_number
        average_post_action_waitlist_vars = total_post_action_waitlist_vars / self.sample_path_number
        average_advance_scheduling_decision_vars = total_advance_scheduling_decision_vars / self.sample_path_number
        average_overtime_decision_vars = total_overtime_decision_vars / self.sample_path_number
        
        # Create absolute variables directly utilizing the helper function
        theta_u_vars = np.array([self.add_absolute_var(direct_model, average_post_action_regular_bookings_vars[j], f"theta_u_{j}") for j in range(self.env.planning_horizon)])
        theta_v_vars = np.array([self.add_absolute_var(direct_model, average_post_action_overtimes_vars[j], f"theta_v_{j}") for j in range(self.env.planning_horizon)])
        theta_w_vars = np.array([self.add_absolute_var(direct_model, average_post_action_waitlist_vars[i], f"theta_w_{i}") for i in range(self.env.num_types)])
        theta_x_vars = np.array([[self.add_absolute_var(direct_model, average_advance_scheduling_decision_vars[n][i], f"theta_x_{n}_{i}") 
                                  for i in range(self.env.num_types)] for n in range(self.env.booking_window_size)])
        theta_y_vars = np.array([self.add_absolute_var(direct_model, average_overtime_decision_vars[j], f"theta_y_{j}") for j in range(self.env.planning_horizon)])

        penalty = coefficient_bound*(theta_u_vars.sum() + theta_v_vars.sum() + theta_w_vars.sum() + theta_x_vars.sum() + theta_y_vars.sum())
        
        
        average_cost = total_cost / self.sample_path_number
        direct_model.setObjective(average_cost + penalty, GRB.MINIMIZE)
        start = time.time()
        if not solve_and_handle_errors(direct_model, verbose=verbose):
            raise RuntimeError("Master model optimal solution not found")
        print(f"Direct model solve time: {time.time() - start} seconds")

        # Extract evaluated results
        average_post_action_regular_bookings_vals = np.array([var.getValue() for var in average_post_action_regular_bookings_vars])
        average_post_action_overtimes_vals = np.array([var.getValue() for var in average_post_action_overtimes_vars])
        average_post_action_waitlist_vals = np.array([var.getValue() for var in average_post_action_waitlist_vars])
        average_advance_scheduling_decision_vals = np.array([[var.getValue() for var in row] for row in average_advance_scheduling_decision_vars])
        average_overtime_decision_vals = np.array([var.getValue() for var in average_overtime_decision_vars])
        coefficients = (
            (coefficient_bound * np.sign(average_post_action_regular_bookings_vals)).tolist() +
            (coefficient_bound * np.sign(average_post_action_overtimes_vals)).tolist() +
            (coefficient_bound * np.sign(average_post_action_waitlist_vals)).tolist() +
            (coefficient_bound * np.sign(average_advance_scheduling_decision_vals)).reshape(-1).tolist() +
            (coefficient_bound * np.sign(average_overtime_decision_vals)).tolist()
        )
        info = {
            'costs': costs,
            'actions': actions,
            'average_cost': average_cost,
            'average_post_action_regular_bookings_vals': average_post_action_regular_bookings_vals,
            'average_post_action_overtimes_vals': average_post_action_overtimes_vals,
            'average_post_action_waitlist_vals': average_post_action_waitlist_vals,
            'average_advance_scheduling_decision_vals': average_advance_scheduling_decision_vals,
            'average_overtime_decision_vals': average_overtime_decision_vals,
        }
        return coefficients, direct_model.ObjVal, info
    
if __name__ == "__main__":
    from experiments import get_config_by_type
    config = get_config_by_type('toy')
    env = config.env
    agent = InfinitePenalizedSAAAgent(env=env, discount_factor=0.99, sample_path_number=350, is_myopic=False)
    coefficients, obj, info = agent.train(verbose=True)
    print("Trained coefficients:", coefficients)


