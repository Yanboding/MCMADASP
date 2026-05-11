import numpy as np
import gurobipy as gp
from gurobipy import GRB

from generating_function import GeneratingFunction

class MulticlassLinearPenaltyFunction(GeneratingFunction):
    def __init__(self, env, coefficients=None):
        super().__init__(env, coefficients)
        
    def get_coefficients(self, solution):
        if solution is None:
            if self.coefficients is None:
                return None, None, None, None, None
            solution = self.coefficients
        T, K, W = self.env.planning_horizon, self.env.num_types, self.env.booking_window_size
        sizes = [K * T, K * T, K * K, K * W * K, K * T]
        offsets = [0] + list(np.cumsum(sizes))
        if not isinstance(solution, gp.MVar):
            solution = np.array(solution, dtype=float)
        theta_u = solution[offsets[0]:offsets[1]].reshape(K, T)
        theta_v = solution[offsets[1]:offsets[2]].reshape(K, T)
        theta_w = solution[offsets[2]:offsets[3]].reshape(K, K)
        theta_x = solution[offsets[3]:offsets[4]].reshape(K, W * K)
        theta_y = solution[offsets[4]:offsets[5]].reshape(K, T)
        return theta_u, theta_v, theta_w, theta_x, theta_y

    def calculate_penalty(self, state, action, new_arrival, is_var=False, coefficients=None):
        (post_action_regular_bookings, post_action_overtimes, post_action_waitlist) = self.env.post_action_state(state, action, is_var)
        (advance_scheduling_decision, overtime_decision) = action
        arrival_difference = self.env.arrival_generator.mean_by_type - new_arrival
        theta_u, theta_v, theta_w, theta_x, theta_y = self._get_coefficient_blocks(coefficients)
        linear_approx = theta_u @ post_action_regular_bookings + theta_v @ post_action_overtimes + theta_w @ post_action_waitlist + theta_x @ advance_scheduling_decision.reshape(-1) + theta_y @ overtime_decision
        penalty_value = arrival_difference @ linear_approx
        return penalty_value
    
    def calculate_gradient(self, state, action, new_arrival, is_var=False):
        raise NotImplemented
    
    def calculate_expected_continuation_value(self, state, action, is_var=False, coefficients=None):
        (post_action_regular_bookings, post_action_overtimes, post_action_waitlist) = self.env.post_action_state(state, action, is_var)
        (advance_scheduling_decision, overtime_decision) = action
        theta_u, theta_v, theta_w, theta_x, theta_y = self._get_coefficient_blocks(coefficients)
        linear_approx = theta_u @ post_action_regular_bookings + theta_v @ post_action_overtimes + theta_w @ post_action_waitlist + theta_x @ advance_scheduling_decision.reshape(-1) + theta_y @ overtime_decision
        workload = post_action_waitlist + self.env.arrival_generator.mean_by_type
        expected_continuation_value = workload @ linear_approx
        return expected_continuation_value
    
    def get_coefficient_var(self, model, coefficient_bound):
        number_of_coefficients = (self.env.planning_horizon * 2 + self.env.num_types + self.env.booking_window_size * self.env.num_types + self.env.planning_horizon) * self.env.num_types
        return model.addMVar(shape=number_of_coefficients, vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name="coefficients")
    
    def build_coefficient_linking_constraints(self, model, coefficient_vars):
        return model.addConstr(coefficient_vars == 0.0, name="link_coefficients")


if __name__ == "__main__":
    from experiments import get_config_by_type
    config = get_config_by_type('toy')
    # Example usage
    env = config.env  # Replace with your environment instance
    state, info = env.reset(**config.reset_params)  # Replace with your state initialization logic
    coefficients = [1] * (env.planning_horizon * 2 + env.num_types + env.booking_window_size * env.num_types + env.planning_horizon) * env.num_types # Replace with your coefficients
    penalty_function = MulticlassLinearPenaltyFunction(env, coefficients=coefficients)
    action = list(env.valid_actions(state))[-1]  # Replace with your action selection logic
    print(state)
    print(action)
    print(penalty_function.theta_u.shape)
    print(penalty_function.theta_v.shape)
    print(penalty_function.theta_w.shape)
    print(penalty_function.theta_x.shape)
    print(penalty_function.theta_y.shape)
    new_arrival = np.array([1, 1])  # Replace with your new arrival information
    penalty = penalty_function.calculate_penalty(state, action, new_arrival, is_var=False)
    expected_continuation_value = penalty_function.calculate_expected_continuation_value(state, action)
    print(f"Calculated penalty: {penalty}")
    print(expected_continuation_value)