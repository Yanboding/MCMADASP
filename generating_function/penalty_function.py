import numpy as np
import gurobipy as gp
from gurobipy import GRB

from generating_function import GeneratingFunction

class LinearPenaltyFunction(GeneratingFunction):
    def __init__(self, env, coefficients=None):
        super().__init__(env, coefficients)

    @property
    def number_of_coefficients(self):
        return (self.env.planning_horizon * 2
                + self.env.num_types
                + self.env.booking_window_size * self.env.num_types
                + self.env.planning_horizon)

    def get_coefficients(self, solution):
        if solution is None:
            if self.coefficients is None:
                return None, None, None, None, None
            solution = self.coefficients
        T, K, W = self.env.planning_horizon, self.env.num_types, self.env.booking_window_size
        sizes = [T, T, K, W * K, T]
        offsets = [0] + list(np.cumsum(sizes))
        if not isinstance(solution, gp.MVar):
            solution = np.array(solution, dtype=float)
        theta_u = solution[offsets[0]:offsets[1]]
        theta_v = solution[offsets[1]:offsets[2]]
        theta_w = solution[offsets[2]:offsets[3]]
        theta_x = solution[offsets[3]:offsets[4]]
        theta_y = solution[offsets[4]:offsets[5]]
        return theta_u, theta_v, theta_w, theta_x, theta_y

    def calculate_penalty(self, state, action, new_arrival, is_var=False, coefficients=None):
        (post_action_regular_bookings, post_action_overtimes, post_action_waitlist) = self.env.post_action_state(state, action, is_var)
        (advance_scheduling_decision, overtime_decision) = action
        total_arrival_difference = np.sum(self.env.arrival_generator.mean_by_type - new_arrival)
        theta_u, theta_v, theta_w, theta_x, theta_y = self._get_coefficient_blocks(coefficients)
        linear_approx = theta_u @ post_action_regular_bookings + theta_v @ post_action_overtimes + theta_w @ post_action_waitlist + theta_x @ advance_scheduling_decision.reshape(-1) + theta_y @ overtime_decision
        penalty_value = total_arrival_difference * linear_approx
        return penalty_value
    
    def calculate_gradient(self, state, action, new_arrival, is_var=False):
        (post_action_regular_bookings, post_action_overtimes, post_action_waitlist) = self.env.post_action_state(state, action, is_var=is_var)
        (advance_scheduling_decision, overtime_decision) = action
        total_arrival_difference = np.sum(self.env.arrival_generator.mean_by_type - new_arrival)
        gradient = np.concatenate([total_arrival_difference * post_action_regular_bookings,
                                   total_arrival_difference * post_action_overtimes,
                                   total_arrival_difference * post_action_waitlist,
                                   total_arrival_difference * advance_scheduling_decision.reshape(-1),
                                   total_arrival_difference * overtime_decision])
        return gradient
    
    def calculate_expected_continuation_value(self, state, action, is_var=False, coefficients=None):
        (post_action_regular_bookings, post_action_overtimes, post_action_waitlist) = self.env.post_action_state(state, action, is_var)
        (advance_scheduling_decision, overtime_decision) = action
        theta_u, theta_v, theta_w, theta_x, theta_y = self._get_coefficient_blocks(coefficients)
        linear_approx = theta_u @ post_action_regular_bookings + theta_v @ post_action_overtimes + theta_w @ post_action_waitlist + theta_x @ advance_scheduling_decision.reshape(-1) + theta_y @ overtime_decision
        workload = (post_action_waitlist + self.env.arrival_generator.mean_by_type).sum()
        expected_continuation_value = self._as_scalar_expression(workload) * self._as_scalar_expression(linear_approx)
        return expected_continuation_value
    
    def calculate_state_value(self, state, is_var=False, coefficients=None):
        """Evaluate V_theta(s) = sum_k theta_k * phi_k(s) for a single state.

        The basis phi(s) only involves the state blocks (theta_u, theta_v,
        theta_w); the action blocks (theta_x, theta_y) have zero features and
        are therefore not identified from state-only data. For a fixed numeric
        state the expression is LINEAR in theta, so it can be used directly in
        a Gurobi least-squares fit when ``coefficients`` are decision
        variables.
        """
        regular_bookings, overtimes, waitlist = state
        theta_u, theta_v, theta_w, _, _ = self._get_coefficient_blocks(coefficients)
        state_value = self._as_scalar_expression(theta_u @ regular_bookings + theta_v @ overtimes + theta_w @ waitlist)
        return state_value
    
    def get_coefficient_var(self, model, coefficient_bound):
        number_of_coefficients = self.env.planning_horizon * 2 + self.env.num_types + self.env.booking_window_size * self.env.num_types + self.env.planning_horizon
        return model.addMVar(shape=number_of_coefficients, vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name="coefficients")
    
    def build_coefficient_linking_constraints(self, model, coefficient_vars):
        return model.addConstr(coefficient_vars == 0.0, name="link_coefficients")


if __name__ == "__main__":
    from experiments import get_config_by_type
    config = get_config_by_type('toy')
    # Example usage
    env = config.env  # Replace with your environment instance
    state, info = env.reset(**config.reset_params)  # Replace with your state initialization logic
    coefficients = [1] * (env.planning_horizon * 2 + env.num_types + env.booking_window_size * env.num_types + env.planning_horizon)  # Replace with your coefficients
    penalty_function = LinearPenaltyFunction(env, coefficients=coefficients)
    action = list(env.valid_actions(state))[-1]  # Replace with your action selection logic
    print(state)
    print(action)
    print(penalty_function.theta_u)
    print(penalty_function.theta_v)
    print(penalty_function.theta_w)
    print(penalty_function.theta_x)
    print(penalty_function.theta_y)
    new_arrival = np.array([1, 1])  # Replace with your new arrival information
    penalty = penalty_function.calculate_penalty(state, action, new_arrival, is_var=False)
    gradient = penalty_function.calculate_gradient(state, action, new_arrival)
    expected_continuation_value = penalty_function.calculate_expected_continuation_value(state, action)
    print(f"Calculated penalty: {penalty}")
    print(f"Calculated gradient: {gradient}")
    print(expected_continuation_value)