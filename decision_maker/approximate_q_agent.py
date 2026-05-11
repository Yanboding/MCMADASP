import copy
import json
import time

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from decision_maker import InfiniteRTAgent
from importance_sampling.proposals import ArrivalGeneratorSamplePathProposal
from metaheuristic_algorithm import SubproblemWorker, BendersDecompositionSolver
from utils import solve_and_handle_errors, flatten, set_link_rhs, acquire_grb_env, encode

class ApproxQAgent(InfiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, 
                 sample_path_number=100, 
                 current_decision_var_type='integer', 
                 future_decision_var_type='continuous',
                 penalty_ratio=1,
                 generating_function=None,
                 sample_path_proposal=None,
                 solver_name='approx_Q',
                 is_trained=False,
                 grb_env=None,
                 verbose=False):
        super().__init__(env, discount_factor, V=V, Q=Q, grb_env=grb_env)
        self.sample_path_number = sample_path_number
        self.current_decision_var_type = GRB.INTEGER if current_decision_var_type is None or current_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.future_decision_var_type = GRB.INTEGER if future_decision_var_type is None or future_decision_var_type == 'integer' else GRB.CONTINUOUS
        self.arrival_generator = copy.deepcopy(self.env.arrival_generator)
        self.sample_path_proposal = sample_path_proposal or ArrivalGeneratorSamplePathProposal()
        self.delta, self.period_likelihood_ratios = self._initialize_sample_paths()
        self.penalty_ratio = penalty_ratio
        self.generating_function = generating_function
        self.decision_model, self.state_linking_constraints, self.action_t_var = None, None, None
        self.is_trained = is_trained
        self.coefficient_model, self.coefficients = None, None
        self.workers = None
        self.solver_name = solver_name
        self.verbose = verbose
    
    def _require_generating_function(self):
        if self.generating_function is None:
            raise ValueError("generating_function is required for penalized SAA operations.")
        return self.generating_function
    
    def _initialize_sample_paths(self):
        proposal = self.sample_path_proposal
        if proposal is None:
            raise ValueError("sample_path_proposal is required for sample-path sampling.")
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

    def decision_model_builder_fn(self):
        model = gp.Model(f"Decision_Model", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        model.setParam("NonConvex", 2)
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
    
    def approx_Q_solve(self, state, t, action=None, verbose=False):
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
        coefficient_vars = self.generating_function.get_coefficient_var(model=master_model, coefficient_bound=coefficient_bound)
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
        coefficient_vars = self.generating_function.get_coefficient_var(model=sub_model, coefficient_bound=GRB.INFINITY)
        self.generating_function.set_coefficients(solution=coefficient_vars)
        coefficient_linking_constraints = self.generating_function.build_coefficient_linking_constraints(sub_model, coefficient_vars)
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
            penalty = generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True)
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
        generating_function = self._require_generating_function()
        generating_function.set_coefficients(self.coefficients)
        return upper_bound, self.coefficients, info
    
    def calculate_information_relaxation_cost(self, state, sample_path, verbose=False):
        model = gp.Model(f"IR_Model", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.current_decision_var_type)
        # add action constraint
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        # ---------- 1. objective ----------
        imm_cost = self.env.cost_fn(state_var, action_var, is_var=True)
        future_cost = 0
        generating_function = self._require_generating_function()
        # for every sample path
        for period_index, new_arrival in enumerate(sample_path):
            penalty = self.penalty_ratio * generating_function.calculate_penalty(state_var, action_var, new_arrival, is_var=True)
            state_var = self.get_next_state(model=model,
                                            state=state_var,
                                            action=action_var,
                                            new_arrival=new_arrival)
            action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
            one_time_cost = self.env.cost_fn(state_var, action_var, is_var=True)
            cost = one_time_cost + penalty
            future_cost += cost
        model.setObjective(imm_cost + future_cost, GRB.MINIMIZE)
        if not solve_and_handle_errors(model, verbose=verbose):
            raise RuntimeError("Direct model optimal solution not found")
        return model.ObjVal
    
    def hindsight_master_builder_fn(self):
        model = gp.Model("Penalized_Hindsight_Master", env=self.grb_env)
        model.setParam("MultiObjPre", 0)
        model.setParam("MIPGapAbs", 1e-9)
        model.setParam("MIPGap", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        model.setParam("OptimalityTol", 1e-9)
        state_var = self.get_state_var(model)
        state_linking_constraints = self.build_state_linking_constraints(model, state_var)
        action_var = self.get_action_var(model=model, advance_scheduling_type=self.current_decision_var_type)
        self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
        theta_vars = np.array([
            model.addVar(vtype=GRB.CONTINUOUS, lb=-1e10, name=f"theta_{omega}")
            for omega in range(len(self.delta))
        ])
        imm_cost = self.env.cost_fn(state_var, action_var, is_var=True)
        model.setObjective(imm_cost + theta_vars.sum() / self.sample_path_number, GRB.MINIMIZE)
        return model, imm_cost, theta_vars, action_var, state_linking_constraints
    
    def hindsight_subproblem_builder_fn(self, env, scenario_id):
        model = gp.Model(f"Penalized_Hindsight_Subproblem_{scenario_id}", env=env)
        model.setParam('InfUnbdInfo', 1)
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

        cost = 0
        generating_function = self._require_generating_function()
        for period_index, new_arrival in enumerate(self.delta[scenario_id]):
            likelihood_ratio = self._get_period_likelihood_ratio(scenario_id, period_index)
            penalty = self.penalty_ratio * generating_function.calculate_penalty(
                state_var,
                action_var,
                new_arrival,
                is_var=True,
            )
            cost += likelihood_ratio * penalty
            state_var = self.get_next_state(
                model=model,
                state=state_var,
                action=action_var,
                new_arrival=new_arrival,
            )
            action_var = self.get_action_var(model=model, advance_scheduling_type=self.future_decision_var_type)
            self.add_action_space_constraints(model=model, state_var=state_var, action_var=action_var)
            cost += likelihood_ratio * self.env.cost_fn(state_var, action_var, is_var=True)
        model.setObjective(self.discount_factor * cost, GRB.MINIMIZE)
        return model, action_linking_constraints, state_linking_constraints
    
    def hindsight_solve(self, state, t,
                        action=None,
                        tol=1e-9,
                        max_iterations=1000,
                        use_pareto_cuts=False,
                        pareto_epsilon=1e-4,
                        core_alpha=None,
                        parallel=False,
                        max_workers=None,
                        verbose=False):
        self._require_generating_function()
        master_model, imm_cost, theta_vars, action_vars, state_linking_constraints = self.hindsight_master_builder_fn()
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
                worker_env = acquire_grb_env({"Threads": 0}, verbose=False, wait=InfiniteRTAgent.TOKEN_WAIT) if parallel else self.grb_env
                sub_model, action_linking_constraints, worker_state_linking_constraints = self.hindsight_subproblem_builder_fn(
                    env=worker_env,
                    scenario_id=omega,
                )
                self.workers.append(SubproblemWorker(
                    model=sub_model,
                    link_rows=action_linking_constraints,
                    state_linking_constraints=worker_state_linking_constraints,
                    subproblem_id=omega,
                ))
                print(f'Finished build {omega} in {time.time() - start} seconds')

        for worker in self.workers:
            set_link_rhs(worker.state_linking_constraints, flatten_state)
            worker.model.reset()

        benders_solver = BendersDecompositionSolver(
            master_model=master_model,
            workers=self.workers,
            imm_cost=imm_cost,
            theta_vars=theta_vars,
            action_vars=flatten_action_vars,
        )
        obj, info = benders_solver.solve(
            init_solution=None,
            tol=tol,
            max_iter=max_iterations,
            use_pareto_cuts=use_pareto_cuts,
            pareto_epsilon=pareto_epsilon,
            core_alpha=core_alpha,
            verbose=verbose,
            parallel=parallel,
            max_workers=max_workers,
        )

        action_t = self.get_solution(action_vars, is_final=True)
        if 'debug' in info or 'debug_info' in info:
            debug_info = {
                'state': encode(state),
                'sample_paths': encode(self.delta),
            }
            with open('bender_error_info.json', 'w') as f:
                f.write(json.dumps(debug_info))
        return obj, action_t, info
    
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
        if self.solver_name == 'approx_Q':
            return self.approx_Q_solve(state, t, action=action, verbose=verbose)
        elif self.solver_name == 'approx_penalized_hindsight':
            return self.hindsight_solve(state, t, action=action, tol=tol, max_iterations=max_iterations, use_pareto_cuts=use_pareto_cuts, pareto_epsilon=pareto_epsilon, core_alpha=core_alpha, parallel=parallel, max_workers=max_workers, verbose=verbose)
        else:
            raise ValueError(f"Unsupported solver_name: {self.solver_name}")
    

if __name__ == '__main__':
    from experiments import get_config_by_type
    from generating_function import MulticlassLinearPenaltyFunction, LinearPenaltyFunction
    config = get_config_by_type('toy')
    env = config.env
    test_state = (np.array([5, 5, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0]), np.array([1, 2]))
    test_action = (np.array([[0, 0],
       [0, 0],
       [1, 1],
       [0, 1],
       [0, 0],
       [0, 0],
       [0, 0]]), np.array([0, 0, 0, 0, 0, 0, 0]))
    coefficients = [-8.506673230703488, 20.803823305410205, 8.949007725191958, 7.035934826472743, 6.404644810547097, 6.152425741039451, 215.93249503160055, 1.6207846766501177, 8.822177032253531, 6.391172301334453, 5.765390772891202, 5.50902974495165, 5.495551566445247, -50.39648401261288, 0.9401940738773069, 17.37900433117047, 7.64933715141955, 6.100169675595545, 5.77057773706476, 5.652617245825845, 215.58932932336387, 0.7393353464647373, 7.139718662697062, 5.313030831788522, 5.010546356203956, 4.977777252780282, 4.9990414864448605, -50.7353596798275, -1821.3953403927717, 343.8795537537197, 37.55879181629673, 39.203762515666966, -1882.1410231036018, 420.5404093482836, -1877.8480092709772, 418.5805478627768, -1855.2227173131294, 419.05603024125645, -1856.1932945867225, 419.5402047398881, -1856.684018334899, 419.7162383810243, -1856.6242232883421, 419.7373714931838, -2485.5936935288996, 0.0, 11.917602076210022, -111.58132849100905, 12.07483450156613, -112.0223378787287, 18.392953186410587, -111.92085801812323, 20.67530043490731, -111.77122539770485, 21.239612150222747, -111.73123837761058, 20.45200112265013, -111.7523714897918, 190.27622164613697, 0.0, -5.72171663290405e-12, 1.4248189742238095, 0.2407531112679777, 0.15812277618748655, 0.1165364293994906, 0.10596987331830707, 0.0, -1.234773977816752e-12, 0.6824583695475585, 0.20696116719471824, 0.17714940526803324, 0.13121802252884512, 0.14178457863045593, 0.0]
    # coefficients = None
    # coefficients = [0]*len(coefficients)
    generating_function = MulticlassLinearPenaltyFunction(env=env, coefficients=coefficients)
    # generating_function = LinearPenaltyFunction(env=env, coefficients=coefficients)
    agent = ApproxQAgent(
            env=env,
            discount_factor=env.discount_factor,
            sample_path_number=256,
            generating_function=generating_function,
            sample_path_proposal=ArrivalGeneratorSamplePathProposal(),
            solver_name='approx_penalized_hindsight',
            is_trained=True
        )
    obj, coefficients, info = agent.benders_decomposition_train(verbose=False)
    # print("Trained coefficients:", coefficients) # 28123
    obj, action, info = agent.solve(test_state, t=1, action=test_action, verbose=False)
    print("Objective from Decision model:", obj)
    print('Action:', action)

    # information_relaxation_cost = agent.calculate_information_relaxation_cost(test_state, sample_path)
    # print("Information relaxation cost for sample path 0:", information_relaxation_cost)




