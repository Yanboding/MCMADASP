"""The project's linear penalty: ``g(x) = (sum delta) * theta . phi_0(s+, a)``."""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from .abstract_generating_function import GeneratingFunction
from .features import Features
from .penalty_forms import legacy_forms


class LinearPenaltyFunction(GeneratingFunction):
    """Transition-scaled linear generating function.

    ``phi_0(s+, a)`` is the five-block vector [post-action regular bookings,
    post-action overtime, post-action waitlist, scheduling decision, overtime
    decision]; the feature of an arrival ``delta`` is ``phi = (sum delta) *
    phi_0`` and its expectation ``E[phi] = (sum mean) * phi_0``. Both are
    affine in the decisions, so every model that carries them stays an LP,
    and a period's penalty feature ``c_e * E[phi] - c_r * phi`` is one scalar
    times ``phi_0`` (:meth:`penalty_features`, one ``post_action_state`` call).
    """
    spec_name = 'linear_penalty'
    forms = legacy_forms()

    def __init__(self, env, coefficients=None):
        super().__init__(env, coefficients)
        self.number_of_coefficients = (self.env.planning_horizon * 2
                                       + self.env.num_types
                                       + self.env.booking_window_size * self.env.num_types
                                       + self.env.planning_horizon)
        # ``E[sum delta]``: the arrival-total scale of the expected features.
        self.expected_arrival_total = float(np.sum(self.env.arrival_generator.mean_by_type))

    def _block_offsets(self):
        T, K, W = self.env.planning_horizon, self.env.num_types, self.env.booking_window_size
        return [0] + list(np.cumsum([T, T, K, W * K, T]))

    def get_coefficients(self, solution):
        if solution is None:
            if self.coefficients is None:
                return None, None, None, None, None
            solution = self.coefficients
        offsets = self._block_offsets()
        if not isinstance(solution, gp.MVar):
            solution = np.array(solution, dtype=float)
        theta_u = solution[offsets[0]:offsets[1]]
        theta_v = solution[offsets[1]:offsets[2]]
        theta_w = solution[offsets[2]:offsets[3]]
        theta_x = solution[offsets[3]:offsets[4]]
        theta_y = solution[offsets[4]:offsets[5]]
        return theta_u, theta_v, theta_w, theta_x, theta_y

    # ---- features ------------------------------------------------------------
    def base_features(self, state, action, is_var=False):
        """``phi_0(s+, a)``: the arrival-free part shared by ``phi`` and ``E[phi]``."""
        (post_action_regular_bookings, post_action_overtimes, post_action_waitlist) = self.env.post_action_state(state, action, is_var)
        (advance_scheduling_decision, overtime_decision) = action
        offsets = self._block_offsets()
        return Features(self.number_of_coefficients, [
            (offsets[0], offsets[1], post_action_regular_bookings),
            (offsets[1], offsets[2], post_action_overtimes),
            (offsets[2], offsets[3], post_action_waitlist),
            (offsets[3], offsets[4], advance_scheduling_decision.reshape(-1)),
            (offsets[4], offsets[5], overtime_decision),
        ])

    def arrival_total(self, arrival):
        return float(np.sum(arrival))

    def features(self, state, action, arrival, is_var=False):
        return self.base_features(state, action, is_var).scaled(self.arrival_total(arrival))

    def expected_features(self, state, action, is_var=False):
        return self.base_features(state, action, is_var).scaled(self.expected_arrival_total)

    def penalty_features(self, state, action, arrival, expected_weight, realized_weight, is_var=False):
        scale = expected_weight * self.expected_arrival_total
        if realized_weight != 0.0:
            scale -= realized_weight * self.arrival_total(arrival)
        if scale == 0.0:
            return Features(self.number_of_coefficients)
        return self.base_features(state, action, is_var).scaled(scale)

    # ---- policy value function (legacy body, kept) --------------------------
    def calculate_expected_continuation_value(self, state, action, is_var=False, coefficients=None):
        (post_action_regular_bookings, post_action_overtimes, post_action_waitlist) = self.env.post_action_state(state, action, is_var)
        (advance_scheduling_decision, overtime_decision) = action
        theta_u, theta_v, theta_w, theta_x, theta_y = self._get_coefficient_blocks(coefficients)
        linear_approx = theta_u @ post_action_regular_bookings + theta_v @ post_action_overtimes + theta_w @ post_action_waitlist + theta_x @ advance_scheduling_decision.reshape(-1) + theta_y @ overtime_decision
        workload = (post_action_waitlist + self.env.arrival_generator.mean_by_type).sum()
        expected_continuation_value = self._as_scalar_expression(workload) * self._as_scalar_expression(linear_approx)
        return expected_continuation_value

    def get_coefficient_var(self, model, coefficient_bound):
        return model.addMVar(shape=self.number_of_coefficients, vtype=GRB.CONTINUOUS, lb=-coefficient_bound, ub=coefficient_bound, name="coefficients")

    def build_coefficient_linking_constraints(self, model, coefficient_vars):
        return model.addConstr(coefficient_vars == 0.0, name="link_coefficients")
