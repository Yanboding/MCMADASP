import copy
import numpy as np
import gurobipy as gp
import json
import time
from gurobipy import GRB

from decision_maker import InfiniteRTAgent
from metaheuristic_algorithm import BendersDecompositionSolver
from metaheuristic_algorithm.benders_decomposition_solver import SubproblemWorker, BendersDecompositionSolver
from utils import solve_and_handle_errors, encode, flatten, set_link_rhs, acquire_grb_env

class InfiniteSAAAgent(InfiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None,
                sample_path_number=100,
                current_decision_var_type='integer', 
                future_decision_var_type='continuous', 
                is_myopic=False, sample_path=None, 
                is_include_discount_factor=False, 
                sample_path_length=None,
                max_periods=None,
                geom_p=None,
                is_quasi_MC=True, verbose=False):
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
        print('arrival generator max periods:', self.arrival_generator.max_periods)
        self.sample_path = sample_path
        self.sample_path_length = sample_path_length
        self.delta = []
        if self.sample_path is not None:
            print('length of sample path:', len(self.sample_path))
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
                onetime_cost = self.env.cost_fn(next_state_var, next_action_var, is_var=True)
                if self.is_include_discount_factor:
                    cost = (self.discount_factor ** tau) * onetime_cost
                else:
                    cost = onetime_cost
                costs[omega].append(onetime_cost)
                actions[omega].append(next_action_var)
                fut_cost += cost
                prev_state_var = next_state_var
                prev_action_var = next_action_var
        fut_cost = fut_cost / self.sample_path_number
        direct_model.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)
        info = {
            'costs': costs,
            'actions': actions
        }
        return direct_model, state_linking_constraints, action_t_var, info
    
    def direct_solve(self, state, action=None, verbose=True):
        if self.direct_model is None:
            self.direct_model, self.state_linking_constraints, self.action_t_var, info = self.direct_builder_fn()
        flatten_state = flatten(state)
        set_link_rhs(self.state_linking_constraints, flatten_state)
        if action is not None:
            self.set_action(action_var=self.action_t_var, action=action)
        # Clean solution before resolving
        self.direct_model.reset()
        if not solve_and_handle_errors(self.direct_model, verbose=verbose):
            raise RuntimeError("Master model optimal solution not found")

        # ---------- 8. return ----------
        action = self.get_solution(self.action_t_var, is_final=True)
        return action, self.direct_model.ObjVal, info
    
    def master_builder_fn(self):
        master_model = gp.Model(f"SA_Advance_Master", env=self.grb_env)
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        master_model.setParam("MIPGap", 1e-9)
        master_model.setParam("FeasibilityTol", 1e-9)
        master_model.setParam("OptimalityTol", 1e-9)
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
        sub_model.setParam("FeasibilityTol", 1e-9)
        sub_model.setParam("OptimalityTol", 1e-9)
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
            if self.is_include_discount_factor:
                fut_cost += (self.discount_factor ** tau) * self.env.cost_fn(next_state_var, next_action_var, is_var=True)
            else:
                fut_cost += self.env.cost_fn(next_state_var, next_action_var, is_var=True)
            prev_state_var = next_state_var
            prev_action_var = next_action_var
        sub_model.setObjective(fut_cost, GRB.MINIMIZE)
        return sub_model, action_linking_constraints, state_linking_constraints

    def iterative_solve(self, state, action=None, verbose=False, use_pareto_cuts=False):
        
        if self.is_myopic or self.sample_path_number <= 1:
            action, obj_value, info = self.direct_solve(state, action=action)
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
                                                                max_iter=100,
                                                                verbose=verbose,
                                                               use_pareto_cuts=use_pareto_cuts)
        if 'debug_info' in info:
            debug_info = {
                'state': encode(state),
                'sample_paths': encode(self.delta)
            }
            with open('bender_error_info.json', 'w') as f:
                f.write(json.dumps(debug_info))
        return action_t, upper_bound, info

    def alternative_solve(self, state, action=None,
                          tol=1e-6,
                          max_iter=150,
                          use_pareto_cuts=False,
                          pareto_epsilon=1e-4,
                          core_alpha=None,
                          verbose=False):
        if self.is_myopic or self.sample_path_number <= 1:
            action, obj_value, info = self.direct_solve(state, action=action)
            return action, obj_value, info
        master_model, imm_cost, theta_vars, action_t_var, state_linking_constraints = self.master_builder_fn()
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        action_vars = flatten(action_t_var)
        if action is not None:
            self.set_action(action_var=action_t_var, action=action)
        if self.workers is None:
            self.workers = []
            for scenario_id in range(len(theta_vars)):
                worker_model, link_rows, state_linking_constraints = self.subproblem_builder_fn(env=self.grb_env,
                                                                                                scenario_id=scenario_id)
                self.workers.append(SubproblemWorker(worker_model,
                                                     link_rows,
                                                     state_linking_constraints,
                                                     scenario_id,
                                                     verbose=verbose))
        for worker in self.workers:
            set_link_rhs(worker.state_linking_constraints, flatten_state)
            worker.model.reset()
        benders_solver = BendersDecompositionSolver(master_model=master_model,
                                                    workers=self.workers,
                                                    imm_cost=imm_cost,
                                                    theta_vars=theta_vars,
                                                    action_vars=action_vars)
        upper_bound, info = benders_solver.solve(tol=tol,
                                                 max_iter=max_iter,
                                                 use_pareto_cuts=use_pareto_cuts,
                                                 pareto_epsilon=pareto_epsilon,
                                                 core_alpha=core_alpha,
                                                 verbose=verbose)
        action_t = self.get_solution(action_t_var, is_final=True)
        if 'debug_info' in info:
            debug_info = {
                'state': encode(state),
                'sample_paths': encode(self.delta)
            }
            with open('bender_error_info.json', 'w') as f:
                f.write(json.dumps(debug_info))
        return action_t, upper_bound, info

    def parallel_solve(self, state, action=None,
                          tol=1e-6,
                          max_iter=150,
                          use_pareto_cuts=False,
                          pareto_epsilon=1e-4,
                          verbose=False):
        if self.is_myopic or self.sample_path_number <= 1:
            action, obj_value, info = self.direct_solve(state, action=action)
            return action, obj_value, info
        master_model, imm_cost, theta_vars, action_t_var, state_linking_constraints = self.master_builder_fn()
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        action_vars = flatten(action_t_var)
        if action is not None:
            self.set_action(action_var=action_t_var, action=action)
        if self.workers is None:
            self.workers = []
            for scenario_id in range(len(theta_vars)):
                start = time.time()
                print(f'Start build {scenario_id}')
                grb_env = acquire_grb_env({"Threads": 0}, verbose=False, wait=InfiniteRTAgent.TOKEN_WAIT)
                worker_model, link_rows, state_linking_constraints = self.subproblem_builder_fn(env=grb_env,
                                                                                                scenario_id=scenario_id)
                self.workers.append(SubproblemWorker(worker_model,
                                                     link_rows,
                                                     state_linking_constraints,
                                                     scenario_id,
                                                     verbose=verbose))
                print(f'Finished build {scenario_id} in {time.time()-start} seconds')
        for worker in self.workers:
            set_link_rhs(worker.state_linking_constraints, flatten_state)
            worker.model.reset()
        benders_solver = BendersDecompositionSolver(master_model=master_model,
                                                    workers=self.workers,
                                                    imm_cost=imm_cost,
                                                    theta_vars=theta_vars,
                                                    action_vars=action_vars)
        upper_bound, info = benders_solver.solve_with_callback(tol=tol,
                                                               max_iter=max_iter,
                                                               use_pareto_cuts=use_pareto_cuts,
                                                               pareto_epsilon=pareto_epsilon,
                                                               max_workers=None,
                                                               verbose=verbose)
        action_t = self.get_solution(action_t_var, is_final=True)
        if 'debug_info' in info:
            debug_info = {
                'state': encode(state),
                'sample_paths': encode(self.delta)
            }
            with open('bender_error_info.json', 'w') as f:
                f.write(json.dumps(debug_info))
        return action_t, upper_bound, info
    
    def solve(self, state, t=1, action=None,
                       batch_size=32,
                       adaptive_tol=0.05,
                       gap_tol=1e-6,
                       max_iter=150,
                       use_pareto_cuts=False,
                       pareto_epsilon=1e-4,
                       core_alpha=None,
                       verbose=False):
        if self.is_myopic or self.sample_path_number <= 1:
            action, obj_value, info = self.direct_solve(state, action=action)
            return action, obj_value, info
        # Create master and subproblem models
        master_model, imm_cost, theta_vars, action_t_var, state_linking_constraints = self.adaptive_master_builder_fn()
        flatten_state = flatten(state)
        set_link_rhs(state_linking_constraints, flatten_state)
        action_vars = flatten(action_t_var)
        if action is not None:
            self.set_action(action_var=action_t_var, action=action)
        if self.workers is None:
            self.workers = []
            for scenario_id in range(self.sample_path_number):
                start = time.time()
                print(f'Start build {scenario_id}')
                grb_env = acquire_grb_env({"Threads": 0}, verbose=False, wait=InfiniteRTAgent.TOKEN_WAIT)
                worker_model, link_rows, state_linking_constraints = self.subproblem_builder_fn(env=grb_env,
                                                                                                scenario_id=scenario_id)
                self.workers.append(SubproblemWorker(model=worker_model,
                                                     link_rows=link_rows,
                                                     state_linking_constraints=state_linking_constraints,
                                                     subproblem_id=scenario_id,
                                                     verbose=verbose))
                print(f'Finished build {scenario_id} in {time.time()-start} seconds')
        for worker in self.workers:
            set_link_rhs(worker.state_linking_constraints, flatten_state)
            worker.model.reset()
        benders_solver = BendersDecompositionSolver(master_model=master_model,
                                                    workers=self.workers,
                                                    imm_cost=imm_cost,
                                                    theta_vars=theta_vars,
                                                    action_vars=action_vars)
        upper_bound, info = benders_solver.adaptive_solve(batch_size=batch_size,
                                                          adaptive_tol=adaptive_tol,
                                                          gap_tol=gap_tol,
                                                          max_iter=max_iter,
                                                          use_pareto_cuts=use_pareto_cuts,
                                                          pareto_epsilon=pareto_epsilon,
                                                          core_alpha=core_alpha,
                                                          verbose=verbose)
        action_t = self.get_solution(action_t_var, is_final=True)
        if 'debug_info' in info:
            debug_info = {
                'state': encode(state),
                'sample_paths': encode(self.delta)
            }
            with open('bender_error_info.json', 'w') as f:
                f.write(json.dumps(debug_info))
        return action_t, upper_bound, info

    def adaptive_master_builder_fn(self):
        master_model = gp.Model(f"SA_Advance_Adaptive_Master", env=self.grb_env)
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam("MIPGapAbs", 1e-9)         # Enforce extremely tight absolute gap
        master_model.setParam("MIPGap", 1e-9)
        master_model.setParam("FeasibilityTol", 1e-9)
        master_model.setParam("OptimalityTol", 1e-9)
        state_var = self.get_state_var(master_model)
        state_linking_constraints = self.build_state_linking_constraints(master_model, state_var)
        # create action variables in period t
        action_t_var = self.get_action_var(model=master_model, advance_scheduling_type=GRB.INTEGER)
        # add action constraint
        self.add_action_space_constraints(model=master_model, state_var=state_var, action_var=action_t_var)
        # set imm_cost and a cost to go lb
        theta_vars = []
        imm_cost = self.env.cost_fn(state_var, action_t_var, is_var=True)
        cost_to_go_estimation = 0 if len(theta_vars) == 0 else sum(theta_vars) / len(theta_vars)
        z = imm_cost + cost_to_go_estimation
        master_model.setObjective(z, GRB.MINIMIZE)
        return master_model, imm_cost, theta_vars, action_t_var, state_linking_constraints

if __name__ == "__main__":
    from experiments import get_config_by_type
    import time
    config = get_config_by_type('toy')
    env = config.env
    agent = InfiniteSAAAgent(env=env, discount_factor=0.99, sample_path_number=256, geom_p=0.02, is_myopic=False)
    state, info = env.reset()
    print(state)
    done = False
    #action, obj, _ = agent.solve(state=state, verbose=False, use_pareto_cuts=True)
    #print("time:", 1, "bender obj:", obj, "action:", action)
    start = time.time()
    action, obj, _ = agent.parallel_solve(state=state, verbose=False, use_pareto_cuts=False)
    print("time:", 1, "bender obj:", obj, "action:", action) # 303045.6417575597
    print(time.time() - start) # 159.18962907791138
    # start = time.time()
    # action, obj, _ = agent.solve(state=state, verbose=False, use_pareto_cuts=True)
    # print("time:", 1, "bender obj:", obj, "action:", action)  # 303045.6417575597
    # print(time.time() - start)
    # start = time.time()
    # action, obj, _ = agent.direct_solve(state=state, verbose=False)
    # print("time:", 1, "bender obj:", obj, "action:", action) # Goal: 267616.65753353486
    # print(time.time() - start)
    start = time.time()
    action, obj, _ = agent.solve(state=state, verbose=False)
    print("time:", 1, "bender obj:", obj, "action:", action) # 303045.6417575597 195.55690169334412
    print(time.time() - start)
    #print(obj) # 11451.191239064321, 11476.191239064323
    #print(action)


