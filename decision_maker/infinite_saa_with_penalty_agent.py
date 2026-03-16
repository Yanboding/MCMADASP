import copy
import numpy as np
import gurobipy as gp
import time
from gurobipy import GRB

from decision_maker import InfiniteRTAgent,LinearPenaltyFunction
from metaheuristic_algorithm import SubproblemWorker, BendersDecompositionSolver
from utils import solve_and_handle_errors, flatten, set_link_rhs, get_solution_value, acquire_grb_env

class InfinitePenalizedSAAAgent(InfiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, 
                 sample_path_number=100, 
                 current_decision_var_type='integer', 
                 future_decision_var_type='continuous', 
                 is_myopic=False, sample_path=None, 
                 is_include_discount_factor=False,
                 sample_path_length=None, 
                 max_periods=None,
                 geom_p=None,
                 is_quasi_MC=True,
                 penalty_ratio=1,
                 generating_function=None,
                 verbose=False):
        super().__init__(env, discount_factor, V=V, Q=Q)
        self.sample_path_number = sample_path_number
        self.current_decision_var_type = GRB.INTEGER if current_decision_var_type is None or current_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.future_decision_var_type = GRB.INTEGER if future_decision_var_type is None or future_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.is_myopic = is_myopic
        self.arrival_generator = copy.deepcopy(self.env.arrival_generator)
        if max_periods is not None:
            self.arrival_generator.set_max_periods(max_periods)
        if geom_p is not None:
            self.arrival_generator.set_geom_p(geom_p)
        self.sample_path = sample_path
        self.sample_path_length = sample_path_length
        self.penalty_ratio = penalty_ratio
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
        model = gp.Model(f"SA_Advance_Reformulated_Model", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.current_decision_var_type)
        action_t_var = action_var
        # add action constraint
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_t_var)
        # ---------- 1. objective ----------
        imm_cost = self.env.cost_fn(state_var, action_t_var, is_var=True)
        total_cost = imm_cost * self.sample_path_number
        costs = [[imm_cost] for _ in range(self.sample_path_number)]
        actions = [[action_var] for _ in range(self.sample_path_number)]
        penalties = [[] for _ in range(self.sample_path_number)]
        # for every sample path
        for omega in range(self.sample_path_number):
            for tau, new_arrival in enumerate(self.delta[omega], start=1):
                penalty = self.penalty_ratio * self.generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True)
                state_var = self.get_next_state(model=model,
                                                state=state_var,
                                                action=action_var,
                                                new_arrival=new_arrival)
                action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
                self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
                one_time_cost = self.env.cost_fn(state_var, action_var, is_var=True)
                if self.is_include_discount_factor:
                    cost = (self.discount_factor ** tau) * (one_time_cost + penalty)
                else:
                    cost = one_time_cost + penalty
                total_cost += cost
                costs[omega].append(one_time_cost)
                actions[omega].append(action_var)
                penalties[omega].append(penalty)
        average_cost = total_cost / self.sample_path_number
        model.setObjective(average_cost, GRB.MINIMIZE)
        info = {
            'costs': costs,
            'actions': actions,
            'penalties': penalties
        }
        return model, state_linking_constraints, action_t_var, info
    
    def reformulated_builder_fn(self, coefficient_bound=1000):
        # This function can be implemented to build a reformulated model that directly optimizes the coefficients without using Benders decomposition.
        model = gp.Model(f"SA_Advance_Reformulated_Model", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        action_var = self.get_action_var(model=model, advance_scheduling_type=GRB.INTEGER)
        action_t_var = action_var
        # add action constraint
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        post_action_state_var = self.env.post_action_state(state_var, action_var, is_var=True)
        # ---------- 1. objective ----------
        imm_cost = self.env.cost_fn(state_var, action_var, is_var=True)
        total_cost = imm_cost * self.sample_path_number
        costs = [[imm_cost] for _ in range(self.sample_path_number)]
        actions = [[action_var] for _ in range(self.sample_path_number)]
        penalties = [[] for _ in range(self.sample_path_number)]
        accumulated_exprs = {
            'u': 0, 'v': 0, 'w': 0, 'x': 0, 'y': 0
        }
        # for every sample path
        for omega in range(self.sample_path_number):
            for tau, new_arrival in enumerate(self.delta[omega], start=1):
                total_arrival_difference = np.sum(self.env.arrival_generator.mean_by_type - new_arrival)
                post_action_regular_bookings_vars, post_action_overtimes_vars, post_action_waitlist_vars = post_action_state_var
                advance_scheduling_decision, overtime_decision = action_var
                accumulated_exprs['u'] += total_arrival_difference * post_action_regular_bookings_vars
                accumulated_exprs['v'] += total_arrival_difference * post_action_overtimes_vars
                accumulated_exprs['w'] += total_arrival_difference * post_action_waitlist_vars
                accumulated_exprs['x'] += total_arrival_difference * advance_scheduling_decision
                accumulated_exprs['y'] += total_arrival_difference * overtime_decision
                state_var = self.get_next_state(model=model,
                                                state=state_var,
                                                action=action_var,
                                                new_arrival=new_arrival)
                action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
                self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
                post_action_state_var = self.env.post_action_state(state=state_var, action=action_var, is_var=True)
                
                
                total_cost += self.env.cost_fn(state_var, action_var, is_var=True)
        # 2. Add Absolute Value Reformulation
        # We store constraints in a dict of lists or arrays for easy Pi access
        
        average_post_action_regular_bookings_vars = accumulated_exprs['u'] / self.sample_path_number
        average_post_action_overtimes_vars = accumulated_exprs['v'] / self.sample_path_number
        average_post_action_waitlist_vars = accumulated_exprs['w'] / self.sample_path_number
        average_advance_scheduling_decision_vars = accumulated_exprs['x'] / self.sample_path_number
        average_overtime_decision_vars = accumulated_exprs['y'] / self.sample_path_number
        
        # Create absolute variables directly utilizing the helper function
        theta_u_vars = []
        theta_u_pos_constrs = []        
        theta_u_neg_constrs = []
        theta_v_vars = []
        theta_v_pos_constrs = []
        theta_v_neg_constrs = []
        theta_w_vars = []
        theta_w_pos_constrs = []
        theta_w_neg_constrs = []
        theta_x_vars = []
        theta_x_pos_constrs = []
        theta_x_neg_constrs = []
        theta_y_vars = []
        theta_y_pos_constrs = []
        theta_y_neg_constrs = []
        for j in range(self.env.planning_horizon):
            theta_u_var, theta_u_pos_constr, theta_u_neg_constr = self.add_absolute_var(model, average_post_action_regular_bookings_vars[j], f"theta_u_{j}")
            theta_u_vars.append(theta_u_var)
            theta_u_pos_constrs.append(theta_u_pos_constr)
            theta_u_neg_constrs.append(theta_u_neg_constr)
            theta_v_var, theta_v_pos_constr, theta_v_neg_constr = self.add_absolute_var(model, average_post_action_overtimes_vars[j], f"theta_v_{j}")
            theta_v_vars.append(theta_v_var)
            theta_v_pos_constrs.append(theta_v_pos_constr)
            theta_v_neg_constrs.append(theta_v_neg_constr)
            theta_y_var, theta_y_pos_constr, theta_y_neg_constr = self.add_absolute_var(model, average_overtime_decision_vars[j], f"theta_y_{j}")
            theta_y_vars.append(theta_y_var)
            theta_y_pos_constrs.append(theta_y_pos_constr)
            theta_y_neg_constrs.append(theta_y_neg_constr)
        
        for i in range(self.env.num_types):
            theta_w_var, theta_w_pos_constr, theta_w_neg_constr = self.add_absolute_var(model, average_post_action_waitlist_vars[i], f"theta_w_{i}")
            theta_w_vars.append(theta_w_var)
            theta_w_pos_constrs.append(theta_w_pos_constr)
            theta_w_neg_constrs.append(theta_w_neg_constr)
        for n in range(self.env.booking_window_size):
            theta_x_row = []
            theta_x_pos_constr_row = []
            theta_x_neg_constr_row = []
            for i in range(self.env.num_types):
                theta_x_var, theta_x_pos_constr, theta_x_neg_constr = self.add_absolute_var(model, average_advance_scheduling_decision_vars[n][i], f"theta_x_{n}_{i}")
                theta_x_row.append(theta_x_var)
                theta_x_pos_constr_row.append(theta_x_pos_constr)
                theta_x_neg_constr_row.append(theta_x_neg_constr)
            theta_x_vars.append(theta_x_row)
            theta_x_pos_constrs.append(theta_x_pos_constr_row)
            theta_x_neg_constrs.append(theta_x_neg_constr_row)
        theta_u_vars = np.array(theta_u_vars)
        theta_v_vars = np.array(theta_v_vars)
        theta_y_vars = np.array(theta_y_vars)
        theta_w_vars = np.array(theta_w_vars)
        theta_x_vars = np.array(theta_x_vars)

        penalty = coefficient_bound*(theta_u_vars.sum() + theta_v_vars.sum() + theta_w_vars.sum() + theta_x_vars.sum() + theta_y_vars.sum())
        average_cost = total_cost / self.sample_path_number
        model.setObjective(average_cost + penalty * self.penalty_ratio, GRB.MINIMIZE)
        info = {
            'costs': costs,
            'actions': actions,
            'penalties': penalties
        }
        return model, state_linking_constraints, action_t_var, info
    
    def solve(self, state, t=1, action=None, verbose=False):
        if self.direct_model is None:
            self.direct_model, self.state_linking_constraints, self.action_t_var, self.info = self.reformulated_builder_fn()
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
        return self.direct_model.ObjVal, action, self.info

    def benders_decomposition_solve(self, state, t=1, action=None, parallel=False, verbose=False):
        pass

    def evaluate(self, state, t=1, action=None, verbose=False):
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
        return self.direct_model.ObjVal, action, info

    
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

    def solve_old(self, state, t=1, action=None, verbose=False):
        if self.is_myopic or self.sample_path_number <= 1:
            action, obj_value, info = self.direct_solve(state, t=t, action=action)
            return action, obj_value, info
        '''
        if self.bender_solver is None:
            self.bender_solver = BendersDecompositionSolver(master_model=self.master_builder_fn,
                                                            workers={},
                                                            imm_cost=None,
                                                            theta_vars=theta_vars,
                                                            action_vars=coefficient_vars)
        action_t, upper_bound, info = self.bender_solver.solve(state=state,
                                                               action=action,
                                                                tol=1e-6,
                                                                max_iter=12000,
                                                                verbose=verbose)
        '''
        return action_t, upper_bound, info
    
    def add_absolute_var(self, model, expression, name_prefix):
        abs_var = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"{name_prefix}_abs")
        pos_constr = model.addConstr(abs_var >= expression, name=f"{name_prefix}_pos_bound")
        neg_constr = model.addConstr(abs_var >= -expression, name=f"{name_prefix}_neg_bound")
        return abs_var, pos_constr, neg_constr
    
    def reformulate_train(self, coefficient_bound=1000, verbose=False):
        # This function can be implemented to train the coefficients using a different approach, such as alternating optimization or gradient-based methods.
        start = time.time()
        direct_model = gp.Model(f"SA_Advance_Direct_Model", env=self.grb_env)
        direct_model.setParam("Method", 1)
        direct_model.setParam("MultiObjPre", 0)
        direct_model.setParam("FeasibilityTol", 1e-9)
        direct_model.setParam("OptimalityTol", 1e-9)
        # ---------- 1. objective ----------
        total_cost = 0
        # 1. Initialize cumulative expectation-deviation expressions
        # We use dictionaries to store the symbolic linear expressions
        accumulated_exprs = {
            'u': 0, 'v': 0, 'w': 0, 'x': 0, 'y': 0
        }
        for omega in range(self.sample_path_number):
            state = self.env.generate_initial_state()
            state_var = self.get_state_var(direct_model)

            # Link initial state
            state_linking_constraints = self.build_state_linking_constraints(direct_model, state_var)
            flatten_state = flatten(state)
            set_link_rhs(state_linking_constraints, flatten_state)
            action_var = self.get_action_var(model=direct_model, advance_scheduling_type=GRB.CONTINUOUS)
            # add action constraint
            self.add_action_space_constraints(model=direct_model, state_var=state_var, action_var=action_var)
            post_action_state_var = self.env.post_action_state(state_var, action_var, is_var=True)
            total_cost += self.env.cost_fn(state_var, action_var, is_var=True)
            for tau, new_arrival in enumerate(self.delta[omega][1:], start=1):
                total_arrival_difference = np.sum(self.env.arrival_generator.mean_by_type - new_arrival)
                (post_action_regular_bookings_vars, post_action_overtimes_vars, post_action_waitlist_vars) = post_action_state_var
                accumulated_exprs['u'] += total_arrival_difference * post_action_regular_bookings_vars
                accumulated_exprs['v'] += total_arrival_difference * post_action_overtimes_vars
                accumulated_exprs['w'] += total_arrival_difference * post_action_waitlist_vars
                advance_scheduling_decision, overtime_decision = action_var
                accumulated_exprs['x'] += total_arrival_difference * advance_scheduling_decision
                accumulated_exprs['y'] += total_arrival_difference * overtime_decision
                state_var = self.get_next_state(model=direct_model,
                                                state=state_var,
                                                action=action_var,
                                                new_arrival=new_arrival)
                action_var = self.get_action_var(model=direct_model, advance_scheduling_type=self.future_decision_var_type)
                self.add_action_space_constraints(model=direct_model, state_var=state_var, action_var=action_var)
                post_action_state_var = self.env.post_action_state(state=state_var, action=action_var, is_var=True)
                
                
                total_cost += self.env.cost_fn(state_var, action_var, is_var=True)
        # 2. Add Absolute Value Reformulation
        # We store constraints in a dict of lists or arrays for easy Pi access
        
        average_post_action_regular_bookings_vars = accumulated_exprs['u'] / self.sample_path_number
        average_post_action_overtimes_vars = accumulated_exprs['v'] / self.sample_path_number
        average_post_action_waitlist_vars = accumulated_exprs['w'] / self.sample_path_number
        average_advance_scheduling_decision_vars = accumulated_exprs['x'] / self.sample_path_number
        average_overtime_decision_vars = accumulated_exprs['y'] / self.sample_path_number
        
        # Create absolute variables directly utilizing the helper function
        theta_u_vars = []
        theta_u_pos_constrs = []        
        theta_u_neg_constrs = []
        theta_v_vars = []
        theta_v_pos_constrs = []
        theta_v_neg_constrs = []
        theta_w_vars = []
        theta_w_pos_constrs = []
        theta_w_neg_constrs = []
        theta_x_vars = []
        theta_x_pos_constrs = []
        theta_x_neg_constrs = []
        theta_y_vars = []
        theta_y_pos_constrs = []
        theta_y_neg_constrs = []
        for j in range(self.env.planning_horizon):
            theta_u_var, theta_u_pos_constr, theta_u_neg_constr = self.add_absolute_var(direct_model, average_post_action_regular_bookings_vars[j], f"theta_u_{j}")
            theta_u_vars.append(theta_u_var)
            theta_u_pos_constrs.append(theta_u_pos_constr)
            theta_u_neg_constrs.append(theta_u_neg_constr)
            theta_v_var, theta_v_pos_constr, theta_v_neg_constr = self.add_absolute_var(direct_model, average_post_action_overtimes_vars[j], f"theta_v_{j}")
            theta_v_vars.append(theta_v_var)
            theta_v_pos_constrs.append(theta_v_pos_constr)
            theta_v_neg_constrs.append(theta_v_neg_constr)
            theta_y_var, theta_y_pos_constr, theta_y_neg_constr = self.add_absolute_var(direct_model, average_overtime_decision_vars[j], f"theta_y_{j}")
            theta_y_vars.append(theta_y_var)
            theta_y_pos_constrs.append(theta_y_pos_constr)
            theta_y_neg_constrs.append(theta_y_neg_constr)
        
        for i in range(self.env.num_types):
            theta_w_var, theta_w_pos_constr, theta_w_neg_constr = self.add_absolute_var(direct_model, average_post_action_waitlist_vars[i], f"theta_w_{i}")
            theta_w_vars.append(theta_w_var)
            theta_w_pos_constrs.append(theta_w_pos_constr)
            theta_w_neg_constrs.append(theta_w_neg_constr)
        for n in range(self.env.booking_window_size):
            theta_x_row = []
            theta_x_pos_constr_row = []
            theta_x_neg_constr_row = []
            for i in range(self.env.num_types):
                theta_x_var, theta_x_pos_constr, theta_x_neg_constr = self.add_absolute_var(direct_model, average_advance_scheduling_decision_vars[n][i], f"theta_x_{n}_{i}")
                theta_x_row.append(theta_x_var)
                theta_x_pos_constr_row.append(theta_x_pos_constr)
                theta_x_neg_constr_row.append(theta_x_neg_constr)
            theta_x_vars.append(theta_x_row)
            theta_x_pos_constrs.append(theta_x_pos_constr_row)
            theta_x_neg_constrs.append(theta_x_neg_constr_row)
        theta_u_vars = np.array(theta_u_vars)
        theta_v_vars = np.array(theta_v_vars)
        theta_y_vars = np.array(theta_y_vars)
        theta_w_vars = np.array(theta_w_vars)
        theta_x_vars = np.array(theta_x_vars)

        penalty = coefficient_bound*(theta_u_vars.sum() + theta_v_vars.sum() + theta_w_vars.sum() + theta_x_vars.sum() + theta_y_vars.sum())
        
        
        average_cost = total_cost / self.sample_path_number
        direct_model.setObjective(average_cost + penalty, GRB.MINIMIZE)
        if not solve_and_handle_errors(direct_model, verbose=verbose):
            raise RuntimeError("Master model optimal solution not found")
        print(f"Direct model solve time: {time.time() - start} seconds")

        # Extract evaluated results
        average_post_action_regular_bookings_vals = np.array([var.getValue() for var in average_post_action_regular_bookings_vars])
        average_post_action_overtimes_vals = np.array([var.getValue() for var in average_post_action_overtimes_vars])
        average_post_action_waitlist_vals = np.array([var.getValue() for var in average_post_action_waitlist_vars])
        average_advance_scheduling_decision_vals = np.array([[var.getValue() for var in row] for row in average_advance_scheduling_decision_vars])
        average_overtime_decision_vals = np.array([var.getValue() for var in average_overtime_decision_vars])

        # Extract dual values (Pi) for each absolute value constraint pair
        # theta_*_vars has shape (..., 3): col 0 = abs_var, col 1 = pos_constr (>= expr), col 2 = neg_constr (>= -expr)
        theta_u_pos_duals = np.array([c.Pi for c in theta_u_pos_constrs])
        theta_u_neg_duals = np.array([c.Pi for c in theta_u_neg_constrs])
        theta_v_pos_duals = np.array([c.Pi for c in theta_v_pos_constrs])
        theta_v_neg_duals = np.array([c.Pi for c in theta_v_neg_constrs])
        theta_w_pos_duals = np.array([c.Pi for c in theta_w_pos_constrs])
        theta_w_neg_duals = np.array([c.Pi for c in theta_w_neg_constrs])
        theta_x_pos_duals = np.array([[c.Pi for c in row] for row in theta_x_pos_constrs])
        theta_x_neg_duals = np.array([[c.Pi for c in row] for row in theta_x_neg_constrs])
        theta_y_pos_duals = np.array([c.Pi for c in theta_y_pos_constrs])
        theta_y_neg_duals = np.array([c.Pi for c in theta_y_neg_constrs])

        # Recover theta coefficients from duals: theta = neg_dual - pos_dual
        # For a minimization LP with abs_var >= expr (pos) and abs_var >= -expr (neg),
        # Gurobi Pi <= 0 for >= constraints, so:
        #   expr > 0 -> pos active: Pi_pos = -coeff_bound, Pi_neg = 0 -> theta = +coeff_bound
        #   expr < 0 -> neg active: Pi_pos = 0, Pi_neg = -coeff_bound -> theta = -coeff_bound
        theta_u =  theta_u_pos_duals - theta_u_neg_duals
        theta_v = theta_v_pos_duals - theta_v_neg_duals
        theta_w = theta_w_pos_duals - theta_w_neg_duals
        theta_x = (theta_x_pos_duals - theta_x_neg_duals)
        theta_y = theta_y_pos_duals - theta_y_neg_duals
        coefficients = (
            theta_u.tolist() +
            theta_v.tolist() +
            theta_w.tolist() +
            theta_x.reshape(-1).tolist() +
            theta_y.tolist()
        )
        info = {
            'average_cost': average_cost,
            'average_post_action_regular_bookings_vals': average_post_action_regular_bookings_vals,
            'average_post_action_overtimes_vals': average_post_action_overtimes_vals,
            'average_post_action_waitlist_vals': average_post_action_waitlist_vals,
            'average_advance_scheduling_decision_vals': average_advance_scheduling_decision_vals,
            'average_overtime_decision_vals': average_overtime_decision_vals,
            'abs_duals': {
                'theta_u': {'pos': theta_u_pos_duals, 'neg': theta_u_neg_duals},
                'theta_v': {'pos': theta_v_pos_duals, 'neg': theta_v_neg_duals},
                'theta_w': {'pos': theta_w_pos_duals, 'neg': theta_w_neg_duals},
                'theta_x': {'pos': theta_x_pos_duals, 'neg': theta_x_neg_duals},
                'theta_y': {'pos': theta_y_pos_duals, 'neg': theta_y_neg_duals},
            },
        }
        direct_model.write("train_model_2.lp")
        return direct_model.ObjVal, coefficients, info
    
    def build_coefficient_linking_constraints(self, model, coefficient_vars):
        theta_u_vars, theta_v_vars, theta_w_vars, theta_x_vars, theta_y_vars = coefficient_vars
        linking_constraints = []
        for j, uj_var in enumerate(theta_u_vars):
            constraint = model.addConstr(uj_var == 0.0, name=f'link_theta_u_{j}')
            linking_constraints.append(constraint)
        for j, vj_var in enumerate(theta_v_vars):
            constraint = model.addConstr(vj_var == 0.0, name=f'link_theta_v_{j}')
            linking_constraints.append(constraint)
        for i, wi_var in enumerate(theta_w_vars):
            constraint = model.addConstr(wi_var == 0.0, name=f'link_theta_w_{i}')
            linking_constraints.append(constraint)
        for n, row in enumerate(theta_x_vars):
            for i, xi_var in enumerate(row):
                constraint = model.addConstr(xi_var == 0.0, name=f'link_theta_x_{n}_{i}')
                linking_constraints.append(constraint)
        for j, yj_var in enumerate(theta_y_vars):
            constraint = model.addConstr(yj_var == 0.0, name=f'link_theta_y_{j}')
            linking_constraints.append(constraint)
        return linking_constraints
    
    def train_master_builder_fn(self, coefficient_bound=GRB.INFINITY):
        master_model = gp.Model(f"SAA_train_Master", env=self.grb_env)
        # FORCES DUAL SIMPLEX (Crucial for Benders warm-starting)
        master_model.setParam("Method", 1)
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam("FeasibilityTol", 1e-9)
        master_model.setParam("OptimalityTol", 1e-9)
        theta_vars = np.array(
            [master_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=1e10, name=f"eta_{omega}") for omega in range(len(self.delta))])
        z = theta_vars.sum() / self.sample_path_number
        post_action_regular_bookings_coeff_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name=f"theta^u_{j}") for j in range(self.env.planning_horizon)])
        post_action_overtimes_coeff_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name=f"theta^v_{j}") for j in range(self.env.planning_horizon)])
        post_action_waitlist_coeff_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name=f"theta^w_{i}") for i in range(self.env.num_types)])
        advance_scheduling_decision_coeff_vars = np.array([[master_model.addVar(vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name=f"theta^x_{n}_{i}")  for i in range(self.env.num_types)] for n in range(self.env.booking_window_size)])
        overtime_decision_coeff_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name=f"theta^y_{j}") for j in range(self.env.planning_horizon)])
        coefficient_vars = (post_action_regular_bookings_coeff_vars.tolist() +
                            post_action_overtimes_coeff_vars.tolist() +
                            post_action_waitlist_coeff_vars.tolist() +
                            advance_scheduling_decision_coeff_vars.reshape(-1).tolist() +
                            overtime_decision_coeff_vars.tolist())
        master_model.setObjective(z, GRB.MAXIMIZE)
        return master_model, coefficient_vars, theta_vars

    def train_subproblem_builder_fn(self, env, scenario_id):
        sub_model = gp.Model(f"Subproblem_SA_Advance_{scenario_id}", env=env)
        # FORCES DUAL SIMPLEX (Crucial for Benders warm-starting)
        sub_model.setParam("Method", 1)
        sub_model.setParam('InfUnbdInfo', 1)
        # Forbidden the model to simplify the model(remove variables/constraints, tighten bounds, etc.). 
        sub_model.setParam('DualReductions', 0)
        sub_model.setParam("MultiObjPre", 0)
        sub_model.setParam("FeasibilityTol", 1e-9)
        sub_model.setParam("OptimalityTol", 1e-9)
        post_action_regular_bookings_coeff_vars = np.array([sub_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta^u_{j}") for j in range(self.env.planning_horizon)])
        post_action_overtimes_coeff_vars = np.array([sub_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta^v_{j}") for j in range(self.env.planning_horizon)])
        post_action_waitlist_coeff_vars = np.array([sub_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta^w_{i}") for i in range(self.env.num_types)])
        advance_scheduling_decision_coeff_vars = np.array([[sub_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta^x_{n}_{i}")  for i in range(self.env.num_types)] for n in range(self.env.booking_window_size)])
        overtime_decision_coeff_vars = np.array([sub_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"theta^y_{j}") for j in range(self.env.planning_horizon)])
        coefficient_vars = (post_action_regular_bookings_coeff_vars,
                            post_action_overtimes_coeff_vars, 
                            post_action_waitlist_coeff_vars, 
                            advance_scheduling_decision_coeff_vars,
                            overtime_decision_coeff_vars)
        coefficient_linking_constraints = self.build_coefficient_linking_constraints(sub_model, coefficient_vars)
        state = self.env.generate_initial_state()
        state_var = self.get_state_var(sub_model)
        state_linking_constraints = self.build_state_linking_constraints(sub_model, state_var)
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        action_var = self.get_action_var(model=sub_model, advance_scheduling_type=GRB.CONTINUOUS)
        self.add_action_space_constraints(model=sub_model, state_var=state_var, action_var=action_var)
        # Initialize scenario state and action like in direct solution
        cost = self.env.cost_fn(state_var, action_var, is_var=True)
        for tau, new_arrival in enumerate(self.delta[scenario_id][1:], start=1):
            penalty = self.generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True, coefficients=coefficient_vars)
            cost += penalty
            state_var = self.get_next_state(model=sub_model,
                                            state=state_var,
                                            action=action_var,
                                            new_arrival=new_arrival)
            action_var = self.get_action_var(model=sub_model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=sub_model, state_var=state_var, action_var=action_var)
            cost += self.env.cost_fn(state_var, action_var, is_var=True)
        sub_model.setObjective(cost, GRB.MINIMIZE)
        return sub_model, coefficient_linking_constraints
    
    def benders_decomposition_train(self, coefficient_bound=GRB.INFINITY, parallel=True, verbose=False):
        # This function can be implemented to train the coefficients using Benders decomposition, which can potentially handle larger sample sizes more efficiently.
        master_model, coefficient_vars, theta_vars = self.train_master_builder_fn(coefficient_bound)
        workers = []
        for scenario_id in range(self.sample_path_number):
            start = time.time()
            print(f'Start build {scenario_id}')
            grb_env = acquire_grb_env({"Threads": 0}, verbose=False, wait=InfiniteRTAgent.TOKEN_WAIT)
            worker_model, link_rows = self.train_subproblem_builder_fn(env=grb_env, scenario_id=scenario_id)
            workers.append(SubproblemWorker(model=worker_model,
                                            link_rows=link_rows,
                                            state_linking_constraints=None,
                                            subproblem_id=scenario_id,
                                            verbose=verbose))
            print(f'Finished build {scenario_id} in {time.time()-start} seconds')
        benders_solver = BendersDecompositionSolver(master_model=master_model,
                                                    workers=workers,
                                                    imm_cost=None,
                                                    theta_vars=theta_vars,
                                                    action_vars=coefficient_vars)
        #init_solution = [0] * len(coefficient_vars)
        init_solution = None
        upper_bound, info = benders_solver.solve(init_solution=init_solution,max_iter=1500, parallel=parallel, verbose=verbose)
        coefficients = [var.X for var in coefficient_vars]
        return upper_bound, coefficients, info
    
    def sample_mean_penalized_lowerbound(self, coefficients, verbose=False):
        coefficients = self.generating_function.get_coefficients(coefficients)
        start = time.time()
        direct_model = gp.Model(f"SA_Advance_Direct_Model", env=self.grb_env)
        direct_model.setParam("Method", 1)
        direct_model.setParam("MultiObjPre", 0)
        direct_model.setParam("FeasibilityTol", 1e-9)
        direct_model.setParam("OptimalityTol", 1e-9)
        # ---------- 1. objective ----------
        total_cost = 0
        for omega in range(self.sample_path_number):
            state = self.env.generate_initial_state()
            state_var = self.get_state_var(direct_model)
            state_linking_constraints = self.build_state_linking_constraints(direct_model, state_var)
            flatten_state = flatten(state)
            set_link_rhs(state_linking_constraints, flatten_state)
            action_var = self.get_action_var(model=direct_model, advance_scheduling_type=GRB.CONTINUOUS)
            # add action constraint
            self.add_action_space_constraints(model=direct_model, state_var=state_var, action_var=action_var)
            cost = self.env.cost_fn(state_var, action_var, is_var=True)
            for tau, new_arrival in enumerate(self.delta[omega][1:], start=1):
                penalty = self.generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True, coefficients=coefficients)
                cost += penalty
                state_var = self.get_next_state(model=direct_model,
                                                state=state_var,
                                                action=action_var,
                                                new_arrival=new_arrival)
                action_var = self.get_action_var(model=direct_model, advance_scheduling_type=self.future_decision_var_type)
                self.add_action_space_constraints(model=direct_model, state_var=state_var, action_var=action_var)
                cost += self.env.cost_fn(state_var, action_var, is_var=True)
            total_cost += cost
        average_cost = total_cost / self.sample_path_number
        direct_model.setObjective(average_cost, GRB.MINIMIZE)
        if not solve_and_handle_errors(direct_model, verbose=verbose):
            raise RuntimeError("Model optimal solution not found")
        print(f"Direct model solve time: {time.time() - start} seconds")
        info = {}
        return direct_model.ObjVal, coefficients, info
    
if __name__ == "__main__":
    from experiments import get_config_by_type
    import matplotlib.pyplot as plt
    config = get_config_by_type('toy')
    env = config.env
    init_state = env.generate_initial_state()
    generating_function = LinearPenaltyFunction(env=env)
    agent = InfinitePenalizedSAAAgent(env=env, discount_factor=0.99, sample_path_number=350, generating_function=generating_function, is_myopic=False)
    #coefficients, obj, info = agent.train(verbose=True)
    #print("Trained coefficients:", coefficients)
    # 46799.670307168795
    # [7.501425403225804, 30.591338709677363, 30.426338709677378, 2.515147177419309, 29.261338709677364, 29.766338709677367, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 86.28025, -6.792641717918127e-16, 0.33000000000000895, 0.0]
    env.reset_random_seeds()  # Reset random seeds before training again to ensure the same sample paths
    obj, direct_coefficients, info = agent.benders_decomposition_train(coefficient_bound=GRB.INFINITY, parallel=True, verbose=False)
    print("Initial state:", init_state)
    print('Obejctive from Benders decomposition training:', obj) # 46799.67030716401
    print('Coefficients from Benders decomposition training:', direct_coefficients) 
    # env.reset_random_seeds()  # Reset random seeds before training again to ensure the same sample paths
    # obj, reformulate_coefficients, info = agent.reformulate_train(coefficient_bound=GRB.INFINITY, verbose=False)
    # print('Obejctive from reformulate training:', obj)
    # print('Coefficients from reformulate training:', reformulate_coefficients)
    # # env.reset_random_seeds()
    # obj, coefficients, info = agent.sample_mean_penalized_lowerbound(coefficients=direct_coefficients, verbose=False)
    # print('Objective from sample mean penalized lower bound evaluation using original problem coefficients:', obj)
    # env.reset_random_seeds()
    # obj, coefficients, info = agent.sample_mean_penalized_lowerbound(coefficients=[0]*len(direct_coefficients), verbose=False)
    # print('Objective from sample mean zero penalized lower bound evaluation:', obj)