import copy
import json
import numpy as np
import gurobipy as gp
import time
from gurobipy import GRB

from decision_maker import InfiniteRTAgent,LinearPenaltyFunction
from importance_sampling import GeometricLengthProposal
from importance_sampling.proposals import FixedLengthProposal
from metaheuristic_algorithm import SubproblemWorker, BendersDecompositionSolver
from utils import solve_and_handle_errors, encode, flatten, set_link_rhs, get_solution_value, acquire_grb_env

class ApproxQAgent(InfiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, 
                 sample_path_number=100, 
                 current_decision_var_type='integer', 
                 future_decision_var_type='continuous', 
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
        self.arrival_generator = copy.deepcopy(self.env.arrival_generator)
        if max_periods is not None:
            self.arrival_generator.set_max_periods(max_periods)
        if geom_p is not None:
            self.arrival_generator.set_geom_p(geom_p)
        self.sample_path_length_proposal = sample_path_length_proposal
        self.delta, self.period_likelihood_ratios = self._initialize_sample_paths(is_quasi_MC=is_quasi_MC)
        self.penalty_ratio = penalty_ratio
        self.generating_function = generating_function
        self.decision_model, self.state_linking_constraints, self.action_t_var = None, None, None
        self.is_trained = False
        self.coefficient_model, self.coefficients = None, None
        self.verbose = verbose
    
    def _require_generating_function(self):
        if self.generating_function is None:
            raise ValueError("generating_function is required for penalized SAA operations.")
        return self.generating_function
    
    def _initialize_sample_paths(self, is_quasi_MC):
        delta = []
        period_likelihood_ratios = []
        if self.sample_path_length_proposal is not None:
            delta, period_likelihood_ratios = self._initialize_proposal_sample_paths()
        else:
            if is_quasi_MC:
                delta = self.arrival_generator.quasi_rvs(size=self.sample_path_number, is_positive_integer_support=True)
            else:
                delta = self.arrival_generator.mc_rvs(size=self.sample_path_number, is_positive_integer_support=True)
            period_likelihood_ratios = [np.ones(len(path), dtype=float) for path in delta]
        return delta, period_likelihood_ratios
    
    def _initialize_proposal_sample_paths(self):
        proposal = self.sample_path_length_proposal
        if proposal is None:
            raise ValueError("sample_path_length_proposal is required for proposal-based sampling.")
        delta, lengths = proposal.sample_arrival_paths(
            arrival_generator=self.arrival_generator,
            size=self.sample_path_number,
        )
        period_likelihood_ratios = proposal.period_likelihood_ratios(
            target_discount_factor=self.discount_factor,
            lengths=lengths,
        )
        return delta, period_likelihood_ratios
    
    def _get_period_likelihood_ratio(self, scenario_id, zero_based_period_index):
        if not self.period_likelihood_ratios:
            return 1.0
        return float(self.period_likelihood_ratios[scenario_id][zero_based_period_index])
    
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

    def decision_model_builder_fn(self):
        model = gp.Model(f"Decision_Model", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        generating_function = self._require_generating_function()
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.current_decision_var_type)
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        # ---------- 1. objective ----------
        imm_cost = self.env.cost_fn(state_var, action_var, is_var=True)
        future_cost = self.penalty_ratio * generating_function.calculate_expected_continuation_value(state_var, action_var,is_var=True)
        model.setObjective(imm_cost + self.discount_factor * future_cost, GRB.MINIMIZE)
        return model, state_linking_constraints, action_var, {}
    
    def solve(self, state, t, action=None, verbose=False):
        if self.decision_model is None:
            self.decision_model, self.state_linking_constraints, self.action_var, self.info = self.decision_model_builder_fn()
        flatten_state = flatten(state)
        set_link_rhs(self.state_linking_constraints, flatten_state)
        if action is not None:
            self.set_action(action_var=self.action_var, action=action)
        # Clean solution before resolving
        self.decision_model.reset()
        if not solve_and_handle_errors(self.decision_model, verbose=verbose):
            raise RuntimeError("Direct model optimal solution not found")
        # ---------- 8. return ----------
        action = self.get_solution(self.action_var, is_final=True)
        return self.decision_model.ObjVal, action, self.info
    
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
        if self.coefficient_model is None:
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
            self.coefficient_model  = BendersDecompositionSolver(master_model=master_model,
                                                                 workers=workers,
                                                                 imm_cost=None,
                                                                 theta_vars=theta_vars,
                                                                 action_vars=coefficient_vars)
        # init_solution = [0] * coefficient_vars.shape[0]
        init_solution = None
        upper_bound, info = self.coefficient_model.solve(init_solution=init_solution, is_hard_bound=True, max_iter=1500, parallel=parallel, verbose=verbose)
        self.coefficients = np.asarray(coefficient_vars.X).tolist()
        self.is_trained = True
        self.generating_function.set_coefficients(self.coefficients)
        return upper_bound, self.coefficients, info
    

if __name__ == '__main__':
    from experiments import get_config_by_type
    config = get_config_by_type('toy')
    env = config.env
    test_state = (np.array([5, 5, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0]), np.array([1, 2]))
    coefficients = [14.540626814834765, 1.5511495238658377, 1.5511495238648707, 1.551149523864899, 1.551149523864559, 1.5511495238653465, 185.41538392612895, 0.1332428878923876, 1.5511495238647601, 1.5511495238651525, 1.55114952386422, 1.551149523863338, 1.5511495238645174, 0.0, 668.562058338164, 457.8833101006897, 419.77572288855816, 367.72846880452494, 419.77572288858147, 367.72846880452687, 391.72572288863086, 367.7284688045264, 419.7757228885221, 367.7284688045268, 461.01343038856766, 367.72846880452346, 309.24753093863814, 367.7284688045271, -48.91385863630643, 0.0, 1.030851306838617e-12, 1.884686541056799e-12, -6.483213120266603e-13, 1.0746273204422448e-12, 1.3735518387560715e-12, -2.8810101527981236e-12, 185.4153839261268]
    coefficients = None
    generating_function = LinearPenaltyFunction(env=env, coefficients=coefficients)
    agent = ApproxQAgent(env=env, 
                        discount_factor=env.discount_factor, 
                        sample_path_number=256,
                        generating_function=generating_function,
                        verbose=True)
    obj, coefficients, info = agent.benders_decomposition_train(verbose=False)
    print("Trained coefficients:", coefficients) # 28123
    obj, action, info = agent.solve(test_state, t=1, action=None, verbose=False)
    print("Objective from Decision model:", obj)
    print('Action:', action)




