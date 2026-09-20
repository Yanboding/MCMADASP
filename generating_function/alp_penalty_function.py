import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .abstract_generating_function import GeneratingFunction
from .features import Features
from .penalty_forms import absorption_forms


class AbsorptionALPPenaltyFunction(GeneratingFunction):
    spec_name = 'absorption_alp_penalty'
    intercept_index = 0
    forms = absorption_forms()

    def __init__(self, env, coefficients=None):
        self.env = env
        self.number_of_coefficients = 1 + 2 * env.planning_horizon + env.num_types
        self.expected_arrival = np.asarray(env.arrival_generator.mean_by_type, dtype=float)
        self.coefficients = None
        self.W_0 = self.U = self.V = self.W = None
        if coefficients is not None:
            self.set_coefficients(coefficients)

    def set_coefficients(self, solution):
        self.W_0, self.U, self.V, self.W = self.get_coefficients(solution)
        self.coefficients = solution

    def _block_offsets(self):
        T, K = self.env.planning_horizon, self.env.num_types
        return [0, 1, 1 + T, 1 + 2 * T, 1 + 2 * T + K]

    def get_coefficients(self, solution):
        if not isinstance(solution, gp.MVar):
            solution = np.asarray(solution, dtype=float)
        if tuple(solution.shape) != (self.number_of_coefficients,):
            raise ValueError(
                f"{self.spec_name} expects {self.number_of_coefficients} ALP row-generation "
                f"coefficients [W_0, U, V, W]; got shape {tuple(solution.shape)}")
        offsets = self._block_offsets()
        W_0 = solution[offsets[0]:offsets[1]]
        if not isinstance(solution, gp.MVar):
            W_0 = float(W_0[0])
        return (W_0,
                solution[offsets[1]:offsets[2]],
                solution[offsets[2]:offsets[3]],
                solution[offsets[3]:offsets[4]])

    def _next_state_features(self, state, action, arrival, is_var=False):
        post_action_state = self.env.post_action_state(state, action, is_var)
        regular, overtime, waitlist = self.env.post_action_state_to_new_state(post_action_state, arrival, is_var)
        offsets = self._block_offsets()
        return Features(self.number_of_coefficients, [
            (offsets[0], offsets[1], np.ones(1)),
            (offsets[1], offsets[2], regular),
            (offsets[2], offsets[3], overtime),
            (offsets[3], offsets[4], waitlist),
        ])

    def features(self, state, action, arrival, is_var=False):
        return self._next_state_features(state, action, np.asarray(arrival, dtype=float), is_var)

    def expected_features(self, state, action, is_var=False):
        return self._next_state_features(state, action, self.expected_arrival, is_var)

    def penalty_features(self, state, action, arrival, expected_weight, realized_weight, is_var=False):
        offsets = self._block_offsets()
        blocks = []
        post_action_weight = expected_weight - realized_weight
        if post_action_weight != 0.0:
            post_action_state = self.env.post_action_state(state, action, is_var)
            regular, overtime, waitlist = self.env.post_action_state_to_new_state(
                post_action_state, np.zeros(self.env.num_types), is_var)
            blocks += [
                (offsets[0], offsets[1], post_action_weight * np.ones(1)),
                (offsets[1], offsets[2], post_action_weight * regular),
                (offsets[2], offsets[3], post_action_weight * overtime),
                (offsets[3], offsets[4], post_action_weight * waitlist),
            ]
        arrival_term = expected_weight * self.expected_arrival
        if realized_weight != 0.0:
            arrival_term = arrival_term - realized_weight * np.asarray(arrival, dtype=float)
        if np.any(arrival_term != 0.0):
            blocks.append((offsets[3], offsets[4], arrival_term))
        return Features(self.number_of_coefficients, blocks)

    def calculate_expected_continuation_value(self, state, action, is_var=False, coefficients=None):
        if coefficients is not None:
            raise ValueError("the ALP penalty evaluates its continuation with the stored coefficients")
        return self._as_scalar_expression(self.expected_value(self.coefficient_vector(), state, action, is_var))

    def get_coefficient_var(self, model, coefficient_bound):
        return model.addMVar(shape=self.number_of_coefficients, vtype=GRB.CONTINUOUS,
                             lb=-coefficient_bound, ub=coefficient_bound, name="coefficients")
