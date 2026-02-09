from collections import defaultdict
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from decision_maker import InfiniteRTAgent
from metaheuristic_algorithm import RowGenerationSolver
from utils import get_solution_value, solve_and_handle_errors, clean_value

class ALPQuadraticAgent(InfiniteRTAgent):

    def __init__(self, env, discount_factor, V=None, Q=None, coefficients=None, pretrain=False, decay_factor=0.95):
        super().__init__(env, discount_factor, V, Q)
        self.is_trained = False
        self.decay_factor = decay_factor
        required_bookings = [(self.env.regular_capacity + self.env.overtime_capacity) * self.decay_factor**(j) for j in range(self.env.planning_horizon)]
        required_bookings[-1] = 0
        required_bookings = np.array(required_bookings)
        self.E_u_alpha = np.minimum(required_bookings, self.env.regular_capacity)
        self.E_v_alpha = np.minimum(np.maximum(required_bookings - self.env.regular_capacity, 0), self.env.regular_capacity)
        self.E_w_alpha = self.env.arrival_generator.mean_by_type
        self.E_w_phi_alpha = (self.E_w_alpha + sum(self.E_u_alpha+self.E_v_alpha)) ** 2
        if coefficients is not None:
            self.is_trained = True
            self.W_0, self.W = self.get_coefficients(coefficients)
        if pretrain:
            self.train(False)
    
    def get_coefficients(self, solution):
        it = iter(solution)
        W_0 = float(next(it))
        W = np.array([float(next(it)) for _ in range(self.env.num_types)])
        return W_0, W
    
    def train(self, debug):
        self.master_model, self.objective, self.coefficient_vars = self.master_builder()
        constraint_candidate_pair = defaultdict(list)
        self.candidates_list = []
        for i, candidate in enumerate(self.generate_all_candidates()):
            constraint = self.get_constraint_data(candidate=candidate)
            # Create a string representation to check for duplicates
            constraint_str = f"{constraint}"
            if constraint_str not in constraint_candidate_pair:
                #return False
                constr_name=f"init_{i + 1}"
                self.master_model.addConstr(constraint, name=constr_name)
                self.master_model.update()
                self.candidates_list.append(candidate)
            constraint_candidate_pair[constraint_str].append(candidate)
        # 1. Optimize the current relaxed master model
        if not solve_and_handle_errors(self.master_model):
            print("Master problem could not be solved to optimality. Aborting.")
        current_solution = np.array([v.X for v in self.master_model.getVars()])
        print(f"Relaxed master objective: {self.master_model.ObjVal:.6f}")
        print("current_solution:", current_solution)
        return self.master_model
    
    def problem_builder(self):
        # Define your parameters in a dictionary
        params = {
            "DualReductions": 0,
            "MultiObjPre": 0,
            "FeasibilityTol": 1e-9,
            "OptimalityTol": 1e-9,
            "OutputFlag": 0
        }

        self.model = gp.Model('QuadraticALP')

        # Apply them in a loop
        for key, value in params.items():
            self.model.setParam(key, value)
            
        self.W_0_var = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"W_0")
        self.W_vars = np.array([self.model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"W_{i}") for i in range(1, self.env.num_types + 1)])
        obj = (
            self.W_0_var
            + gp.quicksum(self.W_vars[i] * self.E_w_phi_alpha[i] for i in range(self.env.num_types))
        )
        self.model.setObjective(obj, GRB.MAXIMIZE)
        coefficient_vars = [self.W_0_var] + self.W_vars.tolist()
        candidates_list = []
        constraint_candidate_pair = defaultdict(list)
        for i, candidate in enumerate(self.env.generate_state_action_pairs()):
            constraint = self.get_constraint_data(candidate=candidate)
            # Create a string representation to check for duplicates
            constraint_str = f"{constraint}"
            if constraint_str not in constraint_candidate_pair:
                #return False
                constr_name=f"init_{i + 1}"
                self.model.addConstr(constraint, name=constr_name)
                candidates_list.append(candidate)
            constraint_candidate_pair[constraint_str].append(candidate)
        return self.model, obj, coefficient_vars


    def generate_all_candidates(self):
        for column in self.env.generate_state_action_pairs():
            yield column
    
    def get_approx_value_fn(self, state, W_0, W):
        regular_bookings, overtimes, waitlist = state
        total_bookings = regular_bookings + overtimes
        
        return W_0 + np.dot(W, (waitlist + sum(total_bookings))**2)
    
    def get_constraint_data(self, candidate):
        state, action = candidate
        candidate_cost = self.env.cost_fn(state, action)
        approx_V_t = self.get_approx_value_fn(state=state,
                                              W_0=self.W_0_var,
                                              W=self.W_vars)
        new_state = self.env.get_next_state(state, action, self.env.arrival_generator.mean_by_type, is_var=False)
        approx_V_next = self.get_approx_value_fn(state=new_state,
                                                 W_0=self.W_0_var,
                                                 W=self.W_vars)
        return approx_V_t - self.env.discount_factor * approx_V_next <= candidate_cost
    
    def master_builder(self):
        master_model = gp.Model('MasterRMP')
        master_model.setParam('DualReductions', 0)
        master_model.setParam("MultiObjPre", 0)
        master_model.setParam("FeasibilityTol", 1e-9)
        master_model.setParam("OptimalityTol", 1e-9)
        master_model.setParam('OutputFlag', 0)
        BigM = 1e4
        self.W_0_var = master_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=BigM, name=f"W_0")
        self.W_vars = np.array([master_model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=BigM, name=f"W_{i}") for i in range(1, self.env.num_types + 1)])
        obj = (
            self.W_0_var
            + gp.quicksum(self.W_vars[i] * self.E_w_phi_alpha[i] for i in range(self.env.num_types))
        )
        self.master_model.setObjective(self.objective, self.master_model.ModelSense)
        master_model.setObjective(obj, GRB.MAXIMIZE)
        master_model.update()
        coefficient_vars = [self.W_0_var] + self.W_vars.tolist()
        return master_model, obj, coefficient_vars
    
    def solve(self, state, t, action=None, verbose=False):
        with (gp.Model("ALP_policy", env=self.grb_env) as policy_model):
            policy_model.setParam("MultiObjPre", 0)
            policy_model.setParam('DualReductions', 0)
            policy_model.setParam("FeasibilityTol", 1e-9)
            policy_model.setParam("OptimalityTol", 1e-9)
            policy_model.setParam("OutputFlag", 0)
            policy_model.setParam("LogToConsole", 0)
            # m.setParam("MIPFocus", 1)
            # ---------- 1. today’s increments ----------
            action_var = self.get_action_var(policy_model, advance_scheduling_type=GRB.INTEGER)
            self.add_action_space_constraints(policy_model, state, action_var)
            if action is not None:
                self.set_action(action_var=action_var, action=action)
            imm_cost = self.env.cost_fn(state, action_var, is_var=True)
            new_state_var = self.get_next_state(model=policy_model,
                                                state=state,
                                                action=action_var,
                                                new_arrival=self.env.arrival_generator.mean_by_type)
            fut_cost = self.discount_factor * self.get_approx_value_fn(state=new_state_var,
                                                                        W_0=self.W_0,
                                                                        W=self.W)
            policy_model.setObjective(imm_cost + fut_cost, GRB.MINIMIZE)
            if not solve_and_handle_errors(policy_model, verbose=verbose):
                raise RuntimeError("Master model optimal solution not found")
            # ---------- 8. return ----------
            action = self.get_solution(action_var, is_final=True)
            return action, policy_model.ObjVal, {}
    
    def set_E_w_phi_alpha(self, E_w_phi_alpha):
        self.E_w_phi_alpha = E_w_phi_alpha
        if self.model is not None:
            obj = self.W_0_var + gp.quicksum(self.W_vars[i] * self.E_w_phi_alpha[i] for i in range(agent.env.num_types))
            self.model.setObjective(obj, GRB.MAXIMIZE)
            self.model.update()

if __name__ == "__main__":
    from experiments import get_config_by_type
    config = get_config_by_type('toy')
    env = config.env
    agent = ALPQuadraticAgent(env=env, discount_factor=env.discount_factor)
    print(agent.E_w_phi_alpha)
    model, obj, coefficient_vars = agent.problem_builder()
    if not solve_and_handle_errors(model):
        print("problem could not be solved to optimality. Aborting.")
    current_solution = np.array([v.X for v in model.getVars()])
    print(f"Relaxed master objective: {model.ObjVal:.6f}")
    print("current_solution:", current_solution)
    for i in range(1, 100):
        for j in range(1, 100):
            E_w_phi_alpha = [i, j]
            print('E_w_phi_alpha:', E_w_phi_alpha)
            agent.set_E_w_phi_alpha(E_w_phi_alpha)
            if not solve_and_handle_errors(model):
                print("problem could not be solved to optimality. Aborting.")
            current_solution = np.array([v.X for v in model.getVars()])
            print(f"Relaxed master objective: {model.ObjVal:.6f}")
            print("current_solution:", current_solution)
    '''
    agent.master_builder()
    candidate1 = ((np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), [0, 1]), 
                  (np.array([[0, 0],
                             [0, 0],
                             [0, 1]]), np.array([0., 0., 0., 0.])))
    candidate2 = ((np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), [0, 1]), 
                  (np.array([[0, 0],
                             [0, 1],
                             [0, 0]]), np.array([0., 0., 0., 0.])))
    print('candidate1')
    agent.get_constraint_data(candidate1)
    print('candidate2')
    agent.get_constraint_data(candidate2)
    '''