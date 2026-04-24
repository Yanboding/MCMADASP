import copy
import json
import numpy as np
import gurobipy as gp
import time
from gurobipy import GRB

from decision_maker import InfiniteRTAgent,LinearPenaltyFunction
from metaheuristic_algorithm import SubproblemWorker, BendersDecompositionSolver
from utils import solve_and_handle_errors, encode, flatten, set_link_rhs, get_solution_value, acquire_grb_env

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
                 grb_env=None,
                 verbose=False):
        super().__init__(env, discount_factor, V=V, Q=Q, grb_env=grb_env)
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
            self.set_sample_path(self.sample_path)
        # sample path length is sample_path_length
        if not is_myopic and sample_path is None:
            if self.sample_path_length is None:
                if is_quasi_MC:
                    self.delta = self.arrival_generator.quasi_rvs(size=self.sample_path_number, is_positive_integer_support=True)
                else:
                    self.delta = self.arrival_generator.mc_rvs(size=self.sample_path_number, is_positive_integer_support=True)
        self.benders_solver = None
        self.is_include_discount_factor = is_include_discount_factor
        self.direct_model, self.state_linking_constraints, self.action_t_var = None, None, None
        self.master_model, self.workers = None, None


    def set_sample_path(self, sample_path):
        self.sample_path_number = 1
        self.delta = np.array([sample_path])
    
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
    
    def build_action_linking_constraints(self, model, action_var):
        x_var, y_var = action_var
        linking_constraints = []
        for j, row in enumerate(x_var):
            for i, var in enumerate(row):
                constraint = model.addConstr(var == 0.0, name=f'link_x_{j},{i}')
                linking_constraints.append(constraint)
        for j, var in enumerate(y_var):
            constraint = model.addConstr(var == 0.0, name=f'link_y_{j}')
            linking_constraints.append(constraint)
        return linking_constraints
    
    def direct_builder_fn(self):
        model = gp.Model(f"Direct_Model", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        state_t_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_t_var)
        action_t_var = self.get_action_var(model=model, advance_scheduling_type=self.current_decision_var_type)
        # add action constraint
        self.add_action_space_constraints(model=model, state_var=state_t_var, action_var=action_t_var)
        # ---------- 1. objective ----------
        imm_cost = self.env.cost_fn(state_t_var, action_t_var, is_var=True)
        future_cost = 0
        costs = [[imm_cost] for _ in range(self.sample_path_number)]
        actions = [[action_t_var] for _ in range(self.sample_path_number)]
        penalties = [[] for _ in range(self.sample_path_number)]
        # for every sample path
        for omega in range(self.sample_path_number):
            state_var = state_t_var
            action_var = action_t_var
            for tau, new_arrival in enumerate(self.delta[omega], start=1):
                penalty = self.penalty_ratio * self.generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True)
                state_var = self.get_next_state(model=model,
                                                state=state_var,
                                                action=action_var,
                                                new_arrival=new_arrival)
                action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
                self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
                one_time_cost = self.env.cost_fn(state_var, action_var, is_var=True)
                cost = one_time_cost + penalty
                future_cost += cost
                costs[omega].append(one_time_cost)
                actions[omega].append(action_var)
                penalties[omega].append(penalty)
        average_future_cost = future_cost / self.sample_path_number
        model.setObjective(imm_cost + self.discount_factor * average_future_cost, GRB.MINIMIZE)
        info = {
            'costs': costs,
            'actions': actions,
            'penalties': penalties
        }
        return model, state_linking_constraints, action_t_var, info
    
    def master_builder_fn(self):
        model = gp.Model(f"Master", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.current_decision_var_type)
        # add action constraint
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        theta_vars = np.array([model.addVar(vtype=GRB.CONTINUOUS, lb=-1e10, name=f"theta_{omega}") for omega in range(len(self.delta))])
        imm_cost = self.env.cost_fn(state_var, action_var, is_var=True)
        z = imm_cost + theta_vars.sum() / self.sample_path_number
        model.setObjective(z, GRB.MINIMIZE)
        return model, imm_cost, theta_vars, action_var, state_linking_constraints
    
    def subproblem_builder_fn(self, env, scenario_id):
        model = gp.Model(f"Subproblem_{scenario_id}", env=env)
        model.setParam('InfUnbdInfo', 1)
        # FORCES DUAL SIMPLEX (Crucial for Benders warm-starting)
        model.setParam('DualReductions', 0)
        model.setParam("Method", 1)
        model.setParam("MultiObjPre", 0)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        action_linking_constraints = self.build_action_linking_constraints(model, action_var)
        # Initialize scenario state and action like in direct solution
        cost = 0
        for tau, new_arrival in enumerate(self.delta[scenario_id], start=1):
            penalty = self.penalty_ratio * self.generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True)
            cost += penalty
            state_var = self.get_next_state(model=model,
                                            state=state_var,
                                            action=action_var,
                                            new_arrival=new_arrival)
            action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
            cost += self.env.cost_fn(state_var, action_var, is_var=True)
        model.setObjective(self.discount_factor * cost, GRB.MINIMIZE)
        return model, action_linking_constraints, state_linking_constraints


    def solve(self, state, t,
              action=None,
              tol=1e-9, 
              max_iterations=1000,
              use_pareto_cuts=False,
              pareto_epsilon=1e-4,
              core_alpha=None,
              parallel=False,
              max_workers=None,
              verbose=False):
        master_model, imm_cost, theta_vars, action_vars, state_linking_constraints = self.master_builder_fn()
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        flatten_action_vars = flatten(action_vars)
        if action is not None:
            self.set_action(action_var=action_vars, action=action)
        if self.workers is None:
            self.workers = []
            for omega in range(self.sample_path_number):
                start = time.time()
                print(f'Start build {omega} with sample path length {len(self.delta[omega])}')
                if parallel:
                    env = acquire_grb_env()
                else:
                    env = self.grb_env
                sub_model, action_linking_constraints, state_linking_constraints = self.subproblem_builder_fn(env=env, scenario_id=omega)
                self.workers.append(SubproblemWorker(model=sub_model,
                                                     link_rows=action_linking_constraints,
                                                     state_linking_constraints=state_linking_constraints,
                                                     subproblem_id=omega))
            print(f'Finished build {omega} in {time.time()-start} seconds')
            
        for worker in self.workers:
            set_link_rhs(worker.state_linking_constraints, flatten_state)
            worker.model.reset()
        benders_solver = BendersDecompositionSolver(master_model=master_model,
                                                    workers=self.workers,
                                                    imm_cost=imm_cost,
                                                    theta_vars=theta_vars,
                                                    action_vars=flatten_action_vars)
        obj, info = benders_solver.solve(init_solution=None,
                                        tol=tol,
                                        max_iter=max_iterations,
                                        use_pareto_cuts=use_pareto_cuts,
                                        pareto_epsilon=pareto_epsilon,
                                        core_alpha=core_alpha,
                                        verbose=verbose,
                                        parallel=parallel,
                                        max_workers=max_workers)
        
        action_t = self.get_solution(action_vars, is_final=True)
        if 'debug_info' in info:
            debug_info = {
                'state': encode(state),
                'sample_paths': encode(self.delta)
            }
            with open('bender_error_info.json', 'w') as f:
                f.write(json.dumps(debug_info))
        return obj, action_t, info
    
    def direct_solve(self, state, t,action=None, verbose=False):
        if self.direct_model is None:
            self.direct_model, self.state_linking_constraints, self.action_t_var, self.info = self.direct_builder_fn()
        flatten_state = flatten(state)
        set_link_rhs(self.state_linking_constraints, flatten_state)
        if action is not None:
            self.set_action(action_var=self.action_t_var, action=action)
        # Clean solution before resolving
        self.direct_model.reset()
        if not solve_and_handle_errors(self.direct_model, verbose=verbose):
            raise RuntimeError("Direct model optimal solution not found")
        # ---------- 8. return ----------
        action = self.get_solution(self.action_t_var, is_final=True)
        return self.direct_model.ObjVal, action, self.info
    
    
    def train_master_builder_fn(self, coefficient_bound=GRB.INFINITY):
        master_model = gp.Model(f"SAA_train_Master", env=self.grb_env)
        # FORCES DUAL SIMPLEX (Crucial for Benders warm-starting)
        master_model.setParam("Method", 1)
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam("FeasibilityTol", 1e-9)
        master_model.setParam("OptimalityTol", 1e-9)
        # theta_vars = np.array(
        #     [master_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=1e10, name=f"eta_{omega}") for omega in range(len(self.delta))])
        theta_vars = master_model.addMVar(shape=self.sample_path_number, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=1e5, name="theta")
        z = theta_vars.sum() / self.sample_path_number
        post_action_regular_bookings_coeff_vars = master_model.addMVar(shape=self.env.planning_horizon, vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name="theta^u")
        post_action_overtimes_coeff_vars = master_model.addMVar(shape=self.env.planning_horizon, vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name="theta^v")
        post_action_waitlist_coeff_vars = master_model.addMVar(shape=self.env.num_types, vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name="theta^w")
        advance_scheduling_decision_coeff_vars = master_model.addMVar(shape=(self.env.booking_window_size, self.env.num_types), vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name="theta^x")
        overtime_decision_coeff_vars = master_model.addMVar(shape=self.env.planning_horizon, vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name="theta^y")
        # Single flat MVar view over all coefficient blocks. Keeps named sub-MVars
        # for readability in the LP while exposing an MVar to downstream code
        # (enables `coefficient_vars.X` and `duals @ coefficient_vars`).
        coefficient_vars = gp.hstack([
            post_action_regular_bookings_coeff_vars,
            post_action_overtimes_coeff_vars,
            post_action_waitlist_coeff_vars,
            advance_scheduling_decision_coeff_vars.reshape(-1),
            overtime_decision_coeff_vars,
        ])
        master_model.setObjective(z, GRB.MAXIMIZE)
        return master_model, coefficient_vars, theta_vars

    def train_subproblem_builder_fn(self, env, scenario_id, init_state = None):
        sub_model = gp.Model(f"Subproblem_SA_Advance_{scenario_id}", env=env)
        # FORCES DUAL SIMPLEX (Crucial for Benders warm-starting)
        sub_model.setParam("Method", 1)
        sub_model.setParam('InfUnbdInfo', 1)
        # Forbidden the model to simplify the model(remove variables/constraints, tighten bounds, etc.). 
        sub_model.setParam('DualReductions', 0)
        sub_model.setParam("MultiObjPre", 0)
        sub_model.setParam("FeasibilityTol", 1e-9)
        sub_model.setParam("OptimalityTol", 1e-9)
        post_action_regular_bookings_coeff_vars = sub_model.addMVar(shape=self.env.planning_horizon, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta^u")
        post_action_overtimes_coeff_vars = sub_model.addMVar(shape=self.env.planning_horizon, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta^v")
        post_action_waitlist_coeff_vars = sub_model.addMVar(shape=self.env.num_types, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta^w")
        advance_scheduling_decision_coeff_vars = sub_model.addMVar(shape=(self.env.booking_window_size, self.env.num_types), vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta^x")
        overtime_decision_coeff_vars = sub_model.addMVar(shape=self.env.planning_horizon, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="theta^y")
        coefficient_vars = (post_action_regular_bookings_coeff_vars,
                            post_action_overtimes_coeff_vars, 
                            post_action_waitlist_coeff_vars, 
                            advance_scheduling_decision_coeff_vars,
                            overtime_decision_coeff_vars)
        coefficient_linking_constraints = self.build_coefficient_linking_constraints(sub_model, coefficient_vars)
        state = self.env.generate_initial_state() if init_state is None else init_state
        state_var = self.get_state_var(sub_model)
        state_linking_constraints = self.build_state_linking_constraints(sub_model, state_var)
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        action_var = self.get_action_var(model=sub_model, advance_scheduling_type=self.future_decision_var_type)
        # action_var = self.get_action_var(model=sub_model, advance_scheduling_type=self.future_decision_var_type)
        self.add_action_space_constraints(model=sub_model, state_var=state_var, action_var=action_var)
        # Initialize scenario state and action like in direct solution
        cost = self.env.cost_fn(state_var, action_var, is_var=True)
        # Store trajectory for primal-based subgradient computation (needed for MILP subproblems
        # where Pi is unavailable). Each entry is (state_vars, action_vars, new_arrival).
        trajectory = []
        for tau, new_arrival in enumerate(self.delta[scenario_id], start=2):
            penalty = self.generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True, coefficients=coefficient_vars)
            cost += penalty
            trajectory.append((state_var, action_var, new_arrival))
            state_var = self.get_next_state(model=sub_model,
                                            state=state_var,
                                            action=action_var,
                                            new_arrival=new_arrival)
            # action_var = self.get_action_var(model=sub_model, advance_scheduling_type=self.future_decision_var_type)
            action_var = self.get_action_var(model=sub_model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=sub_model, state_var=state_var, action_var=action_var)
            cost += self.env.cost_fn(state_var, action_var, is_var=True)
        sub_model.setObjective(cost, GRB.MINIMIZE)

        return sub_model, coefficient_linking_constraints
    
    def benders_decomposition_train(self, coefficient_bound=GRB.INFINITY, init_state = None, parallel=True, verbose=False):
        # This function can be implemented to train the coefficients using Benders decomposition, which can potentially handle larger sample sizes more efficiently.
        master_model, coefficient_vars, theta_vars = self.train_master_builder_fn(coefficient_bound)
        workers = []
        for scenario_id in range(self.sample_path_number):
            start = time.time()
            print(f'Start build {scenario_id}')
            grb_env = self.grb_env
            if parallel:
                grb_env = acquire_grb_env({"Threads": 0}, verbose=False, wait=InfiniteRTAgent.TOKEN_WAIT)
            worker_model, coefficient_linking_constraints = self.train_subproblem_builder_fn(env=grb_env, scenario_id=scenario_id, init_state=init_state)
            workers.append(SubproblemWorker(model=worker_model,
                                            link_rows=coefficient_linking_constraints,
                                            state_linking_constraints=None,
                                            subproblem_id=scenario_id,
                                            objective_builder_fn=None,
                                            cut_gradient_fn=None,
                                            verbose=verbose))
            print(f'Finished build {scenario_id} with sample path length {len(self.delta[scenario_id])} in {time.time()-start} seconds')
        benders_solver = BendersDecompositionSolver(master_model=master_model,
                                                    workers=workers,
                                                    imm_cost=None,
                                                    theta_vars=theta_vars,
                                                    action_vars=coefficient_vars)
        init_solution = [0] * coefficient_vars.shape[0]
        #init_solution = None
        upper_bound, info = benders_solver.solve(init_solution=init_solution,max_iter=1500, parallel=parallel, verbose=verbose)
        coefficients = np.asarray(coefficient_vars.X).tolist()
        return upper_bound, coefficients, info
    
    def sample_mean_penalized_lowerbound(self, coefficients, ratio=1, init_state = None, verbose=False):
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
            state = self.env.generate_initial_state() if init_state is None else init_state
            state_var = self.get_state_var(direct_model)
            state_linking_constraints = self.build_state_linking_constraints(direct_model, state_var)
            flatten_state = flatten(state)
            set_link_rhs(state_linking_constraints, flatten_state)
            action_var = self.get_action_var(model=direct_model, advance_scheduling_type=self.current_decision_var_type)
            # add action constraint
            self.add_action_space_constraints(model=direct_model, state_var=state_var, action_var=action_var)
            cost = self.env.cost_fn(state_var, action_var, is_var=True)
            for tau, new_arrival in enumerate(self.delta[omega], start=2):
                penalty = ratio * self.generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True, coefficients=coefficients)
                cost += penalty
                state_var = self.get_next_state(model=direct_model,
                                                state=state_var,
                                                action=action_var,
                                                new_arrival=new_arrival)
                action_var = self.get_action_var(model=direct_model, advance_scheduling_type=self.current_decision_var_type)
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
    test_state = (np.array([5, 5, 0]), np.array([0, 0, 0]), np.array([1, 2]))
    test_state = None
    test_action = (np.array([[4, 0],
                             [1, 2],
                             [0, 0]]), np.array([1, 0, 0]))
    coefficients = [9.281778046800301, 18.406982109217235, 211.7776403012647, 1.8317253609339224, 17.076982109219387, 211.1143069679426, 454.97928707153835, 432.75496421837806, 390.1621061706687, 396.04000000002765, 391.2989818770426, 395.9409999999887, -190.9698684057874, 0.0, 1.9440832013001023e-12, 0.32999999999992724, 0.0]
    generating_function = LinearPenaltyFunction(env=env, coefficients=coefficients)
    agent = InfinitePenalizedSAAAgent(env=env, discount_factor=env.discount_factor, sample_path_number=2, generating_function=generating_function, is_myopic=False)
    # env.reset_random_seeds()
    # print('Test state:', test_state)
    # obj, action, info = agent.solve(test_state, action=None, parallel=True, verbose=False)
    # print('Objective from Benders decomposition solve with trained coefficients:', obj)
    # print('Action from Benders decomposition solve with trained coefficients:', action)

    # direct_obj, action, info = agent.direct_solve(test_state, action=None, verbose=False)
    # print('Objective from direct solve with trained coefficients:', direct_obj)
    # print('Action from direct solve with trained coefficients:', action)
    # print("gap between direct and Benders decomposition solve:", (direct_obj - obj)/direct_obj * 100)
    # coefficients, obj, info = agent.train(verbose=True)
    #print("Trained coefficients:", coefficients)
    # env.reset_random_seeds()  # Reset random seeds before training again to ensure the same sample paths
    # start = time.time()
    # obj, direct_coefficients, info = agent.benders_decomposition_train(coefficient_bound=GRB.INFINITY, parallel=True, init_state=test_state, verbose=False)
    # end = time.time()
    # print(f"Benders decomposition training time: {end - start} seconds")
    # print('Obejctive from Benders decomposition training:', obj) # 46799.67030716401
    # print('Coefficients from Benders decomposition training:', direct_coefficients)
    # env.reset_random_seeds()  # Reset random seeds before training again to ensure the same sample paths
    # start = time.time()
    # obj, reformulate_coefficients, info = agent.reformulate_train(coefficient_bound=GRB.INFINITY, verbose=False)
    # end = time.time()
    # print(f"Reformulate training time: {end - start} seconds")
    # print('Obejctive from reformulate training:', obj)
    # print('Coefficients from reformulate training:', reformulate_coefficients)
    # LP with imediate action have integer constraint.
    # direct_coefficients = [-1.3594687499522706, 17.558399553583435, 178.1039650297289, 1.1157723214212893, 15.68357812501477, 177.20034895834002, 419.8851277901139, 364.9403883928951, 328.4328333333303, 330.3090773809399, 332.4114291293804, 330.2100773809448, -153.41132459057022, 0.0, 0.028794642858840806, 0.8172321428525413, 0.0]
    # MILP with imediate action have integer constraint.
    # direct_coefficients = [9.268191043987258, 18.399990576954398, 211.77769372497636, 1.779482715174383, 17.06978530519917, 211.11833006741278, 454.88242227211236, 432.7775535054471, 390.28763575751765, 396.03190327939177, 391.3760703787603, 395.9341759642607, -190.85357865534522, 0.0, -0.0003694891595442083, 0.3287273151188725, 0.0] 
    # FULL MILP
    # direct_coefficients = [9.147308288146675, 19.002800405200663, 205.95900694128812, 3.1162106277479382, 14.888631287719091, 203.99516141290258, 497.61674182911685, 469.0295179497762, 389.4843489749785, 381.84105713537247, 383.43078908601046, 384.1750962073562, -181.40992640734066, 0.0, 0.5656574831678151, 3.221326122856308, 0.0]
    # env.reset_random_seeds()
    # penalized_obj, coefficients, info = agent.sample_mean_penalized_lowerbound(coefficients=direct_coefficients, init_state=test_state, verbose=False)
    # print('Objective from sample mean penalized lower bound evaluation using original problem coefficients:', penalized_obj)
    # env.reset_random_seeds()
    # zero_penalized_obj, coefficients, info = agent.sample_mean_penalized_lowerbound(coefficients=direct_coefficients, ratio=0,init_state=test_state, verbose=False)
    # print('Objective from sample mean zero penalized lower bound evaluation using original problem coefficients:', zero_penalized_obj)
    # print('Difference between penalized and zero-penalized objectives:', (penalized_obj - zero_penalized_obj)/zero_penalized_obj * 100)  
    # env.reset_random_seeds()
    # obj, coefficients, info = agent.sample_mean_penalized_lowerbound(coefficients=[0]*len(direct_coefficients), verbose=False)
    # print('Objective from sample mean zero penalized lower bound evaluation:', obj)
    # test_state = env.generate_initial_state()
    # valid_actions = env.valid_actions(test_state)
    # print("Test state:", test_state)
    # print('Valid actions for test state:', valid_actions)
    obj, action, info = agent.solve(test_state, t=1, verbose=True)
    print('Objective from reformulated solve:', obj)
    print('Action from reformulated solve:', action)
    # obj, action, info = agent.benders_decomposition_solve(test_state, action=None, parallel=False, verbose=False)
    # print('Objective from Benders decomposition solve with trained coefficients:', obj)
    # print('Action from Benders decomposition solve with trained coefficients:', action)
    # 0.95 discount factor, penalty coefficient:
    # coeff = [-489747.322426145, -711218.4679959532, 719784.9716079241, -346175.396042743, 102461.22802959933, -808990.8905766644, 1325723.640805767, 862252.0612548698, -144205.0131229693, 3313514.8155598124, -546073.0141164073, -594546.7608864738, -1069709.2665148135, -247532.54645759502, -1452391.584621127, -312056.15928052954, -896558.3651680413, 182408.33725924176, 885861.6038419643, -858357.20688265, 364219.60296936386, -620100.2664535658, -597168.0181206921, -443650.9600258555, 685935.6490012287, 1530358.5667221989, -1315084.5012369512, 171491.59681829266, -455134.4527753449, -480804.1855914448, 336604.36229369155, 158645.08747389718, 871082.2551786707, -302045.4461149539, 9657.923821000537, -2463156.301811835, -135484.46185770354, 546044.3882727083, 282983.83578527265, 98425.30402205292, 533326.4525604609, -543730.6568369432, 1277977.1592053517, -554831.2963686847, -319.36451376855405, 212025.94902166264, -689251.763793769, -257802.34004594426, -19160.84315628894, -407231.16302259883, 867581.2309926108, -1265158.4556900265, 155074.27008959252, 158813.69092461863, 1057668.7046582922, 638008.564133196, 337574.1370621461, -1214297.1367920388, 599934.7065020138, -1407251.407399903, 0.0, 1607041.0095909138, 3214745.518337991, -4760994.3907199865, -1292356.2503559003, -3034515.5188236963, 5955215.957214754, 2575510.6043085214, -995278.5115294635, 754594.3784337969, -6277969.566126927, -2447853.045104966, 7019657.398312768, -11847478.712698566, 11557.459752992316, 5693712.113259128, 2041911.1825693187, 1362300.0285245387, -3514242.382325819, -3331180.0389415114, 8000662.320669528, 1251574.1794287271, 5208718.1094938535, 741001.7090559595, 2816360.2160940967, -1602446.7167960433, 4886490.178662435, 2676401.6145397397, -336606.07881431415, 9785840.985333657, 2352906.585110853, 567419.674060192, 1590428.1273864245, 12180.587253554295, -3030752.4908909504, -5835466.685774844, -447109.8233191027, -2809116.13453469, -30622.357864716632, -62824.3954426463, 16863804.367515467, -12592203.859740624, -163339.30536339298, -12043155.927366259, -1379559.6580010077, 7071010.545994182, -3282030.6998342304, 26512873.197403394, -8535591.539686384, -6306995.114596074, 9229667.836569445, -300039.8395280878, 6839819.188063323, -23811955.378168013, -4751566.003276649, -19038481.602659177, 11874310.133073702, 11720479.644499252, 11133161.385026753, -5493241.675151262, 17016170.298268773, -27133194.047216203, -526098.0612809293, -5604496.882720259, 192553.5822487685, -4862347.083631181, 6088268.261546321, 112918.75808751659, 2617538.3009186047, 5770518.21101051, -6095087.717799375, 2439798.932556475, -3057381.9491510787, 5620346.467661818, 1536607.7505615554, -448049.3742719203, 4718497.035373611, 4719350.812547765, -4689917.107157717, -1639058.0644898643, 14958629.880329331, -36844018.96115995, -20316843.463248935, 3926570.578220546, 86393132.70071225, -113230024.53520028, -4331440.878572671, -1029035.3119052777, 26492856.228266485, 2958025.8790909043, 41941574.892734684, 65179533.08906259, -119970963.93884565, 0.0, 0.0, 0.0, 0.0, 0.0, -6876500.579605313, -1622436.8003096122, 30141885.141536087, -57352607.321088694, -132401765.02128424, 22998128.540401816, -14839936.684086828, -49307447.05445, 19137036.341506753, -42161678.55776894, -62519680.63970023, 41953484.51182088, -14254460.417691357, 0.0, 0.0, 0.0, 0.0, 0.0, 64583595.65112737, 13406076.345255485, -129831653.39809859, -51305116.89448609, -6633299.403740077, 50889637.564786606, -16353397.392908242, 4728957.442261873, 74524513.60320385, 14114903.46499937, 3672860.5456221937, -63395989.63284287, 9487079.870824244, 0.0, 55215884.68969526, 0.0, 0.0, 0.0, -68452369.77061923, 2685671.1566625824, -8592028.125410298, -29742900.325894974, 17841296.009100508, -1954343092.0464375, -9405758.974661654, -18778085.306856234, 25476030.01818518, -22549301.11324304, 186545275.3657411, 8979174.743190166, -42760635.60079053, 0.0, 0.0, 0.0, 0.0, 0.0, 72429963.8469732, 14607667.361279752, -102492466.52422042, 3568103.4314837833, 172423941.25083128, 1664089970.0201664, 4418833.674848343, 21448022.41272709, 31883382.70535992, 1096237.9279416539, 40283141.29294864, 16127070.466257432, 6007848.700997193, 0.0, 0.0, 0.0, 0.0, 0.0, 88835202.92366317, 15193054.826200275, -55235853.189698465, 60797375.048192345, 136038579.86719725, 456786537.7049442, -5454003.1056388235, -16266490.87978821, 34483504.076125905, 76876213.19180624, -129823863.50424701, -2077600.6526215612, 56225004.39179838, 0.0, 74448677.1462581, 0.0, 0.0, 0.0, 58281611.15445423, -26103447.422453254, 119967768.99920434, -216950904.89388728, -160158316.02545235, 0.0, -104469112.65568757, -25359959.0664207, -24577764.939726558, -53156865.92186923, -11660335.16902241, -11234444.10248559, -397240758.06269497, 0.0, 0.0, 0.0, 0.0, 0.0, 26482881.094088633, 1990168.4852494043, 131263088.41533801, 115593273.92951122, 14189892.4117598, 0.0, 26391886.24425564, 69916081.99238749, 3075389.096728769, -166566996.93068781, -33101914.915810913, 3380436.1284692744, -5625386.368192371, 0.0, 0.0, 0.0, 0.0, -328177157.41291636, 2724905.2590766726, 0.0, -59997955.86360394, 148835886.66209915, 821361994.0293069, 0.0, 27697534.281075247, -11330659.75773326, 3719916.6128258733, -4327106.359876408, -19323237.764175553, -9138736.771636564, -128310105.43552725, 0.0, 0.0, 0.0, 0.0, 0.0, -3112212.548550782, 0.0, 59638124.03168706, 153448461.36173907, -275785863.59713984, -27334722.89801229, -1435622.6212682102, 32614822.285848692, -40254024.53249082, -30424117.090019256, -22778542.38960852, 16183491.818843123, 226378751.98852193, 0.0, 0.0, 0.0, 0.0, 0.0, -47905178.214504115, 0.0, 223138954.6032202, -28445334.81842055, 122283806.79341178, -83498421.26893331, -32940501.679792315, 65719326.645221286, -16789441.31853593, 22373151.020307895, 130232446.39596483, 44514757.98312567, 192791575.29791746, 0.0, 0.0, 0.0, 0.0, 0.0, 14156178.794497712, -58958011.411265925, 78456078.70176727, -285652880.67879254, -8050202.051327966, -51090767.609103054, 119054256.93605198, -56515471.43390082, 67403825.77811804, 63418045.869291216, -6223666.663237218, 40793552.353486665, -6819361.272881651, 0.0, 0.0, 0.0, 0.0, 0.0, 95749870.71981843, -1409808260.1752121, 1866017.7002165453, -180135625.7641276, 276500144.56210405, 0.0, -19144248.84615008, 5853338.164361202, 95655914.35778868, 4925601.366996948, -138749776.19868472, 0.0, 84494391.22512001, 0.0, 0.0, 0.0, 7.123992086112836e+23, 0.0, -733710.2943551526, -50096309985.52102, 60159752.48353955, 54437783.06207424, -82324304.94223036, -70623692.47567974, 174633203.03066245, -37728798.39391569, -18856887.472575575, 10479054.992398052, 81732096.23571473, 0.0, 202613287.25510463, 0.0, -860270844.3664843, 0.0, 0.0, 0.0, 97037468.36340526, 0.0, 43426363.719161674, 85105380.43294694, 162546558.6433016, -1935553247.1072738, 253168157.32519814, 10672387.552371265, 10567590.811721569, 4880458.049717572, 52249983.73027379, 0.0, -246848255.89696813, 0.0, -34530845.248271994, 133538675.7711714, 0.0, -714791679.3893445, 74171637.42627901, 226018921.53073928, -23091893.634967364, 205958242.99333313, -36455027.060454816, -288079786.2027706, 43770548.77957391, 12437826.041244796, -227484063.1647883, -41107559.293882504, -185238028.55051082, 0.0, -80160977.14595278, 0.0, 0.0, 0.0, 0.0, 0.0, -40046822.6769048, 0.0, -92778308.21006656, -68629871.88467367, -106545818.94621688, 0.0, 7298272.19003085, -14597052.705342144, -58872150.91244401, 54130162.7680976, -54886172.01919195, 18369352.447676808, -618863.5878275352, 0.0, 102475568.53798282, 0.0, 0.0, -320173422.30719787, -109074190.1913616, -21587487686.86876, -7127138.337919671, 51326225.09462657, -199468915.2259473, -177391142.35246506, -74340316.25530471, 19764202.805126984, -25066325.90300674, -27432489.995609265, -50963046.0056363, 174574131659.9523, 338802253.1344289, 0.0, -88824823.46667118, 0.0, 0.0, 0.0, -44238647.36284854, 0.0, -51484145.63561483, 195741445.40380564, -19860954.99642953, -117904006.18275982, 29128545.21170866, 33596795.822315775, -17144410.111540657, 79203155.63187793, 121626051.72404474, 0.0, 9045141.698649693, 0.0, -125906519.8380351, 0.0, 0.0, 219040898.89313242, 2911775.451664706, 0.0, 16336926.299500776, -230520602.29161298, 40698508.61150771, 340260226.5841219, -9608705344.861834, 19339331.762660958, 61957832.07020212, -28630971.92572935, 175052160.0128615, -42295752.4233217, 469736383.222662, 0.0, 0.0, 0.0, -8986467.800891634, 0.0, -128586570.87793179, -19465825006.4603, -9885882.6871414, -69615822.19687979, 80109118.39274178, 0.0, 0.0, 25434081.813568886, 157446414.6734254, -23373905.020220093, -120636918.35818444, 0.0, 104473463.00499913, 0.0, -47477193.43160751, -346580809.0605297, -66212858.44875274, 0.0, 54532029.35673463, 102282436.635202, 39442016.13513856, 13588436.707508236, -198817136.92311776, -119653242.80407177, 31286926.149419446, -25369370.291858617, -31757718.59676295, 22150127.673137534, -752581.3601838513, 0.0, -17171918.991131876, -184425145.93652117, -26757653.06568853, 15627823.690928882, 0.0, 0.0, 4018561.3251646515, 0.0, -18656214.15905381, -46745169.07748254, 95113918.74095854, 5498385.081769293, 14655016.687713008, -62826662.80194483, -11518781.134721097, 5672406.214296923, 91234128.71771783, 117153425.2371094, 15853054.12391267, 0.0, -69068456.12668082, 13206822.802045194, 380855689.7960367, 0.0, -56331067.98954062, 0.0, 134772387.98868805, 3734061.6970202783, -137322942.77828446, 406251676.5831244, 1087962602.3690157, -34160411.068135895, -93292121.31464273, 23112565.027805977, 153267206.41919416, 0.0, -6561987.7250454975, -11354436.06361572, 419733364.4659898, 12757796.094780937, -62308850.14379802, 36959073.51883658, -94542483.20498396, -2815047.2120927824, 8021446.525005379, 3311524.151025043, -3213707.4370152387, 157493492.78185618, 59447472.71734323, -29188231.792681742, 58837693.04681155, -10473401.354003884, 17992189.287717078, 0.0, 29352533.75350182, 111750536.4157388, 5576992.753603267, 27307589.80952359, -18773252.908836614, 12799027.414501464, 39208792.91105394, 2338922.8307661116, 2158467.5218874402, 16348640.832009446, -19570155.279038563, 9904007.827536097, -3507552.3915572753, -704093.5435448391, -21010399.018653743, 7802710.377553757, -13022093.13963379, 6181766.726505094, -12158114.42832537, -21026869.804020572, 15471373.222022995, -8907068.651448589, -4290293.480364676, 16422285.085655382, 25138584.024139285, -20611962.844173715, 7951241.7857363615, 4616039.568802797, -2387455.805918985, -10987257.105565125, -5905945.467374912, -12232968.918594994, 18524551.654138934, 1493132.1437505023, -26285960.761706613, -11926680.754184006, 17212859.399804074, -24537801.842577435, 30288725.72887296, 9186539.818540458, -393755.10536774294, 28645890.48378107, 18253397.19256746, -18703353.439943086, -6725622.441208107, -21480995.81297578, 2695256.6412151796, -16177873.722586082, 11819310.097642073, 22689457.639083512, -8139020.21938418, 20063743.222299583, -33364121.15063037, 3102863.5411261776, -46797908.822431386, 35049112.18536641, -15204263.33617713, -31452482.395842556, 34985655.27118135, 18835823.39255736, 39564423.70352981, -30809431.57588613, -37470021.54838234, -24860975.69806363, -44300371.124728456, 50145205.10051838, 0.0]