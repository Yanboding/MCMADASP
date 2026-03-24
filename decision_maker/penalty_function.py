import numpy as np

class LinearPenaltyFunction:
    def __init__(self, env, coefficients=None):
        self.env = env
        self.coefficients = coefficients
        if self.coefficients is not None:
            self.theta_u, self.theta_v, self.theta_w, self.theta_x, self.theta_y = self.get_coefficients(self.coefficients)
    
    def get_coefficients(self, solution):
        if solution is None:
            if self.coefficients is None:
                return None, None, None, None, None
            solution = self.coefficients
        it = iter(solution)
        theta_u = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        theta_v = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        theta_w = np.array([float(next(it)) for _ in range(self.env.num_types)])
        theta_x = np.array([[float(next(it)) for _ in range(self.env.num_types)] for _ in range(self.env.booking_window_size)])
        theta_y = np.array([float(next(it)) for _ in range(self.env.planning_horizon)])
        return theta_u, theta_v, theta_w, theta_x, theta_y

    def calculate_penalty(self, state, action, new_arrival, is_var=False, coefficients=None):
        (post_action_regular_bookings, post_action_overtimes, post_action_waitlist) = self.env.post_action_state(state, action, is_var)
        (advance_scheduling_decision, overtime_decision) = action
        total_arrival_difference = np.sum(self.env.arrival_generator.mean_by_type - new_arrival)
        if coefficients is not None:
            theta_u, theta_v, theta_w, theta_x, theta_y = coefficients
        else:
            theta_u, theta_v, theta_w, theta_x, theta_y = self.theta_u, self.theta_v, self.theta_w, self.theta_x, self.theta_y
        linear_approx = np.sum(theta_u * post_action_regular_bookings) + np.sum(theta_v * post_action_overtimes) + np.sum(theta_w * post_action_waitlist) + np.sum(theta_x * advance_scheduling_decision) + np.sum(theta_y * overtime_decision)
        penalty_value = total_arrival_difference * linear_approx
        return penalty_value
    
    def calculate_gradient(self, state, action, new_arrival, is_var=False):
        (post_action_regular_bookings, post_action_overtimes, post_action_waitlist) = self.env.post_action_state(state, action, is_var=is_var)
        (advance_scheduling_decision, overtime_decision) = action
        total_arrival_difference = np.sum(self.env.arrival_generator.mean_by_type - new_arrival)
        gradient = np.concatenate([total_arrival_difference * post_action_regular_bookings,
                                   total_arrival_difference * post_action_overtimes,
                                   total_arrival_difference * post_action_waitlist,
                                   total_arrival_difference * advance_scheduling_decision.flatten(),
                                   total_arrival_difference * overtime_decision])
        return gradient


if __name__ == "__main__":
    from experiments import get_config_by_type
    config = get_config_by_type('toy')
    # Example usage
    env = config.env  # Replace with your environment instance
    state, info = env.reset(**config.reset_params)  # Replace with your state initialization logic
    coefficients = [0.5] * (env.planning_horizon * 2 + env.num_types + env.booking_window_size * env.num_types + env.planning_horizon)  # Replace with your coefficients
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
    print(f"Calculated penalty: {penalty}")
    print(f"Calculated gradient: {gradient}")