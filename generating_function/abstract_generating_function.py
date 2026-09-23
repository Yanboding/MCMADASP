import numpy as np


class GeneratingFunction:
    spec_name = None
    forms = {}

    def __init__(self, env, coefficients=None):
        self.env = env
        self.coefficients = coefficients
        if self.coefficients is not None:
            self.theta_u, self.theta_v, self.theta_w, self.theta_x, self.theta_y = self.get_coefficients(self.coefficients)

    def set_coefficients(self, solution):
        self.coefficients = solution
        self.theta_u, self.theta_v, self.theta_w, self.theta_x, self.theta_y = self.get_coefficients(solution)

    def _as_scalar_expression(self, expression):
        if hasattr(expression, "item"):
            return expression.item()
        return expression

    def _get_coefficient_blocks(self, coefficients):
        if coefficients is not None:
            coefficient_blocks = coefficients
        else:
            coefficient_blocks = self.get_coefficients(self.coefficients)
        theta_u, theta_v, theta_w, theta_x, theta_y = coefficient_blocks
        if theta_u is None or theta_v is None or theta_w is None or theta_x is None or theta_y is None:
            raise ValueError("coefficients are required before evaluating the penalty function.")
        return theta_u, theta_v, theta_w, theta_x, theta_y

    def coefficient_vector(self):
        if self.coefficients is None:
            raise ValueError("coefficients are required before evaluating the penalty function.")
        return np.asarray(self.coefficients, dtype=float)

    def get_coefficients(self, solution):
        raise NotImplementedError

    def state_features(self, state):
        raise NotImplementedError(f'{self.spec_name} does not define state features')

    def approximate_value(self, state, solution=None):
        features = self.state_features(state)
        return features @ (self.coefficient_vector() if solution is None else solution)

    def features(self, state, action, arrival, is_var=False):
        raise NotImplementedError

    def expected_features(self, state, action, is_var=False):
        raise NotImplementedError

    def penalty_features(self, state, action, arrival, expected_weight, realized_weight, is_var=False):
        result = self.expected_features(state, action, is_var).scaled(expected_weight)
        if realized_weight != 0.0:
            result = result - self.features(state, action, arrival, is_var).scaled(realized_weight)
        return result

    def value(self, theta, state, action, arrival, is_var=False):
        return self.features(state, action, arrival, is_var).dot(theta)

    def expected_value(self, theta, state, action, is_var=False):
        return self.expected_features(state, action, is_var).dot(theta)

    def form(self, consumer):
        return self.forms[consumer]

    def calculate_expected_continuation_value(self, state, action, is_var=False, coefficients=None):
        raise NotImplementedError

    def get_coefficient_var(self, model, coefficient_bound):
        raise NotImplementedError
