import copy
import json
import numpy as np
import gurobipy as gp
import time
from gurobipy import GRB

from decision_maker import InfiniteRTAgent
from importance_sampling import GeometricLengthProposal
from importance_sampling.proposals import FixedLengthProposal
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
                 sample_path_length_proposal=None,
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
        self.sample_path_length_proposal = sample_path_length_proposal
        # self.W_0, self.U, self.V, self.W = self.get_coefficients(coefficients)
        self.delta = []
        self.period_likelihood_ratios = []
        if self.sample_path is not None:
            self.set_sample_path(self.sample_path)
        # sample path length is sample_path_length
        if not is_myopic and sample_path is None:
            self._initialize_sample_paths(is_quasi_MC=is_quasi_MC)
        self.benders_solver = None
        self.is_include_discount_factor = is_include_discount_factor
        self.direct_model, self.state_linking_constraints, self.action_t_var = None, None, None
        self.master_model, self.workers = None, None
        if verbose:
            print([len(path) for path in self.delta])


    def set_sample_path(self, sample_path):
        self.sample_path_number = 1
        self.delta = np.array([sample_path])
        self.period_likelihood_ratios = [np.ones(len(sample_path), dtype=float)]

    def _initialize_sample_paths(self, is_quasi_MC):
        if self.sample_path_length_proposal is not None:
            self._initialize_proposal_sample_paths()
            return
        if self.sample_path_length is not None:
            lengths = np.full(self.sample_path_number, int(self.sample_path_length), dtype=int)
            self.delta = self._generate_arrival_paths_with_lengths(lengths)
        else:
            if is_quasi_MC:
                self.delta = self.arrival_generator.quasi_rvs(size=self.sample_path_number, is_positive_integer_support=True)
            else:
                self.delta = self.arrival_generator.mc_rvs(size=self.sample_path_number, is_positive_integer_support=True)
        self._set_unit_likelihood_ratios()

    def _initialize_proposal_sample_paths(self):
        proposal = self.sample_path_length_proposal
        if proposal is None:
            raise ValueError("sample_path_length_proposal is required for proposal-based sampling.")
        self.delta, lengths = proposal.sample_arrival_paths(
            arrival_generator=self.arrival_generator,
            size=self.sample_path_number,
        )
        self.period_likelihood_ratios = proposal.period_likelihood_ratios(
            target_discount_factor=self.discount_factor,
            lengths=lengths,
        )

    def _generate_arrival_paths_with_lengths(self, lengths):
        return [self.arrival_generator.rvs(size=int(length)) for length in lengths]

    def _set_unit_likelihood_ratios(self):
        self.period_likelihood_ratios = [np.ones(len(path), dtype=float) for path in self.delta]

    def _get_period_likelihood_ratio(self, scenario_id, zero_based_period_index):
        if not self.period_likelihood_ratios:
            return 1.0
        return float(self.period_likelihood_ratios[scenario_id][zero_based_period_index])

    def _require_generating_function(self):
        if self.generating_function is None:
            raise ValueError("generating_function is required for penalized SAA operations.")
        return self.generating_function
    
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
        generating_function = self._require_generating_function()
        # for every sample path
        for omega in range(self.sample_path_number):
            state_var = state_t_var
            action_var = action_t_var
            for period_index, new_arrival in enumerate(self.delta[omega]):
                likelihood_ratio = self._get_period_likelihood_ratio(omega, period_index)
                penalty = self.penalty_ratio * generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True)
                state_var = self.get_next_state(model=model,
                                                state=state_var,
                                                action=action_var,
                                                new_arrival=new_arrival)
                action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
                self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
                one_time_cost = self.env.cost_fn(state_var, action_var, is_var=True)
                cost = one_time_cost + penalty
                future_cost += likelihood_ratio * cost
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
        generating_function = self._require_generating_function()
        for period_index, new_arrival in enumerate(self.delta[scenario_id]):
            likelihood_ratio = self._get_period_likelihood_ratio(scenario_id, period_index)
            penalty = self.penalty_ratio * generating_function.calculate_penalty(
                state_var, 
                action_var, 
                new_arrival, 
                is_var=True
                )
            cost += likelihood_ratio * penalty
            state_var = self.get_next_state(model=model,
                                            state=state_var,
                                            action=action_var,
                                            new_arrival=new_arrival)
            action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
            cost += likelihood_ratio * self.env.cost_fn(state_var, action_var, is_var=True)
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
        theta_vars = master_model.addMVar(shape=self.sample_path_number, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=1e8, name="theta")
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
        self.add_action_space_constraints(model=sub_model, state_var=state_var, action_var=action_var)
        # Initialize scenario state and action like in direct solution
        objective = self.env.cost_fn(state_var, action_var, is_var=True)
        generating_function = self._require_generating_function()
        for period_index, new_arrival in enumerate(self.delta[scenario_id]):
            likelihood_ratio = self._get_period_likelihood_ratio(scenario_id, period_index)
            penalty = generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True, coefficients=coefficient_vars)
            objective += likelihood_ratio * penalty
            state_var = self.get_next_state(model=sub_model,
                                            state=state_var,
                                            action=action_var,
                                            new_arrival=new_arrival)
            action_var = self.get_action_var(model=sub_model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=sub_model, state_var=state_var, action_var=action_var)
            one_time_cost = self.env.cost_fn(state_var, action_var, is_var=True)
            objective += likelihood_ratio * one_time_cost
        sub_model.setObjective(objective, GRB.MINIMIZE)

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
        upper_bound, info = benders_solver.solve(init_solution=init_solution, is_hard_bound=True, max_iter=1500, parallel=parallel, verbose=verbose)
        coefficients = np.asarray(coefficient_vars.X).tolist()
        return upper_bound, coefficients, info
    
    def sample_mean_penalized_lowerbound(self, coefficients, ratio=1, init_state = None, verbose=False):
        generating_function = self._require_generating_function()
        coefficients = generating_function.get_coefficients(coefficients)
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
            for period_index, new_arrival in enumerate(self.delta[omega]):
                likelihood_ratio = self._get_period_likelihood_ratio(omega, period_index)
                penalty = ratio * generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True, coefficients=coefficients)
                cost += likelihood_ratio * penalty
                state_var = self.get_next_state(model=direct_model,
                                                state=state_var,
                                                action=action_var,
                                                new_arrival=new_arrival)
                action_var = self.get_action_var(model=direct_model, advance_scheduling_type=self.current_decision_var_type)
                self.add_action_space_constraints(model=direct_model, state_var=state_var, action_var=action_var)
                cost += likelihood_ratio * self.env.cost_fn(state_var, action_var, is_var=True)
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
    test_state = (np.array([5, 5, 0, 0,0,0,0]), np.array([0, 0, 0,0,0,0,0]), np.array([1, 2]))
    test_state = None
    test_action = (np.array([[4, 0],
                             [1, 2],
                             [0, 0]]), np.array([1, 0, 0]))
    coefficients = [14.540626814834765, 1.5511495238658377, 1.5511495238648707, 1.551149523864899, 1.551149523864559, 1.5511495238653465, 185.41538392612895, 0.1332428878923876, 1.5511495238647601, 1.5511495238651525, 1.55114952386422, 1.551149523863338, 1.5511495238645174, 0.0, 668.562058338164, 457.8833101006897, 419.77572288855816, 367.72846880452494, 419.77572288858147, 367.72846880452687, 391.72572288863086, 367.7284688045264, 419.7757228885221, 367.7284688045268, 461.01343038856766, 367.72846880452346, 309.24753093863814, 367.7284688045271, -48.91385863630643, 0.0, 1.030851306838617e-12, 1.884686541056799e-12, -6.483213120266603e-13, 1.0746273204422448e-12, 1.3735518387560715e-12, -2.8810101527981236e-12, 185.4153839261268]
    generating_function = LinearPenaltyFunction(env=env, coefficients=coefficients)
    agent = InfinitePenalizedSAAAgent(env=env, 
                                      discount_factor=env.discount_factor, 
                                      sample_path_number=1, 
                                      generating_function=generating_function,
                                      is_myopic=False,
                                      sample_path_length_proposal=FixedLengthProposal(max_length=99),
                                      verbose=True)
    env.reset_random_seeds()
    print('Test state:', test_state)
    obj, action, info = agent.solve(test_state, t=1, action=None, parallel=True, verbose=False)
    print('Objective from Benders decomposition solve with trained coefficients:', obj)
    print('Action from Benders decomposition solve with trained coefficients:', action)

    # direct_obj, action, info = agent.direct_solve(test_state, action=None, verbose=False)
    # print('Objective from direct solve with trained coefficients:', direct_obj)
    # print('Action from direct solve with trained coefficients:', action)
    # print("gap between direct and Benders decomposition solve:", (direct_obj - obj)/direct_obj * 100)
    # env.reset_random_seeds()  # Reset random seeds before training again to ensure the same sample paths
    # start = time.time()
    # obj, direct_coefficients, info = agent.benders_decomposition_train(coefficient_bound=GRB.INFINITY, parallel=True, init_state=test_state, verbose=False)
    # end = time.time()
    # print(f"Benders decomposition training time: {end - start} seconds") # 27564.059161307774
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
    # obj, action, info = agent.benders_decomposition_solve(test_state, action=None, parallel=False, verbose=False)
    # print('Objective from Benders decomposition solve with trained coefficients:', obj)
    # print('Action from Benders decomposition solve with trained coefficients:', action)
    # 0.95 discount factor, penalty coefficient:

