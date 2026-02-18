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
        scale = 200
        self.E_u_w_alpha = np.minimum(required_bookings, self.env.regular_capacity) * scale
        self.E_v_w_alpha = (required_bookings - self.E_u_w_alpha) * scale*1000
        self.E_w_phi_alpha = self.env.arrival_generator.mean_by_type * scale *10
        if coefficients is not None:
            self.is_trained = True
            self.W_0, self.U, self.V, self.W = self.get_coefficients(coefficients)
    
    def get_coefficients(self, solution):
        it = iter(solution)
        W_0 = float(next(it))
        U = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        V = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        W = np.array([float(next(it)) for _ in range(self.env.num_types)])
        return W_0, U, V, W
    
    def get_approx_value_fn(self, state, W_0, U, V, W):
        regular_bookings, overtimes, waitlist = state
        total_bookings = regular_bookings + overtimes
        #return W_0 + np.dot(U, (regular_bookings + sum(waitlist))**2) + np.dot(V, (overtimes + sum(waitlist))**2) + np.dot(W, (waitlist + sum(total_bookings))**2)
        return W_0 + np.dot(U, (regular_bookings + sum(waitlist))**2) + np.dot(V, (overtimes + sum(waitlist))**2) + np.dot(W, (np.array(waitlist))**2)

    
    def get_constraint_data(self, candidate):
        state, action = candidate
        candidate_cost = self.env.cost_fn(state, action)
        approx_V_t = self.get_approx_value_fn(state=state,
                                              W_0=self.W_0_var,
                                              U=self.U_vars, 
                                              V=self.V_vars,
                                              W=self.W_vars)
        new_state = self.env.get_next_state(state, action, self.env.arrival_generator.mean_by_type, is_var=False)
        approx_V_next = self.get_approx_value_fn(state=new_state,
                                                 W_0=self.W_0_var,
                                                 U=self.U_vars,
                                                 V=self.V_vars,
                                                 W=self.W_vars)
        return approx_V_t - self.env.discount_factor * approx_V_next <= candidate_cost
    
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
        self.U_vars = np.array([self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"U_{j}") for j in range(self.env.planning_horizon)])
        self.V_vars = np.array([self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"V_{j}") for j in range(self.env.planning_horizon)])
        self.W_vars = np.array([self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, name=f"W_{i}") for i in range(1, self.env.num_types + 1)])

        obj = (
            self.W_0_var
            + gp.quicksum(self.U_vars[j] * self.E_u_w_alpha[j] for j in range(self.env.planning_horizon))
            + gp.quicksum(self.V_vars[j] * self.E_v_w_alpha[j] for j in range(self.env.planning_horizon))
            + gp.quicksum(self.W_vars[i] * self.E_w_phi_alpha[i] for i in range(self.env.num_types))
        )
        self.model.setObjective(obj, GRB.MAXIMIZE)
        coefficient_vars = [self.W_0_var] + self.U_vars.tolist() + self.V_vars.tolist() + self.W_vars.tolist()
        candidates_list = []
        constraint_candidate_pair = defaultdict(list)
        for i, candidate in enumerate(self.env.generate_state_action_pairs()):
            # print('candidate:')
            # print(candidate)
            constraint = self.get_constraint_data(candidate=candidate)
            #print('candidate add:', candidate, i, cost)
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
    
    def set_E_w_phi_alpha(self, E_w_phi_alpha):
        self.E_w_phi_alpha = E_w_phi_alpha
        if self.model is not None:
            obj = self.W_0_var + gp.quicksum(self.W_vars[i] * self.E_w_phi_alpha[i] for i in range(agent.env.num_types))
            self.model.setObjective(obj, GRB.MAXIMIZE)
            self.model.update()
    
    def penalty_function(self, state, action, new_arrival, next_state):
        (next_regular_booking, next_overtime, next_waitlist) = next_state
        (regular_booking, overtime, waitlist) = state
        (advance_scheduling_decision, overtime_decision) = action
        arrival_difference = self.env.arrival_generator.mean_by_type - new_arrival
        outstanding_treatments = waitlist - sum(advance_scheduling_decision)
        total_booked_slots = (next_regular_booking + next_overtime).sum()
        total_outstanding_treatments = waitlist.sum() - advance_scheduling_decision.sum()
        total_arrival_difference = arrival_difference.sum()
        term_1 = 2 * sum(self.U * (next_regular_booking + total_outstanding_treatments) * total_arrival_difference)
        term_2 = 2 * sum(self.V * (next_overtime + total_outstanding_treatments) * total_arrival_difference)
        #term_3 = 2 * sum(self.W * (outstanding_treatments + total_booked_slots) * arrival_difference)
        term_3 = 2 * sum(self.W * (outstanding_treatments) * arrival_difference)
        penalty = term_1 + term_2 + term_3
        return penalty

if __name__ == "__main__":
    from experiments import get_config_by_type
    config = get_config_by_type('toy')
    env = config.env
    agent = ALPQuadraticAgent(env=env, discount_factor=env.discount_factor)
    print(agent.E_w_phi_alpha)
    model, obj, coefficient_vars = agent.problem_builder()
    if not solve_and_handle_errors(agent.model):
        print("problem could not be solved to optimality. Aborting.")
    current_solution = np.array([v.X for v in agent.model.getVars()])
    print(current_solution)
    model_name = agent.model.ModelName if agent.model.ModelName.strip() else "unnamed_model"

    # Construct filename and save the model as an LP file
    filename = f"{model_name}.lp"
    print(f"Saving model to file: {filename}")
    model.write(filename)
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