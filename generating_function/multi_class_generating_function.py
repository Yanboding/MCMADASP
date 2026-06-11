"""PO-derived quadratic value-function approximation (generating function).

Implements the pathwise-optimization (PO) derived value function

    V_kappa(s) = kappa_0
               + sum_j kappa_u[j] * u_j + sum_j kappa_v[j] * v_j
               + sum_i kappa_w[i] * w_i
               + sum_{i,j} kappa_uw[i,j] * u_j * w_i
               + sum_{i,j} kappa_vw[i,j] * v_j * w_i
               + sum_i kappa_ww[i] * w_i^2

together with the closed-form expected continuation value
E[V_kappa(f(s, a, delta))] and the reduced zero-mean penalty

    z_kappa(s, a, delta) = sum_i (E[delta_i] - delta_i) *
        ( 2 * kappa_ww[i] * (w_i - sum_n x_{n,i})
          + sum_j kappa_uw[i,j] * u^{t+1}_j
          + sum_j kappa_vw[i,j] * v^{t+1}_j )

where (u^{t+1}, v^{t+1}) are the next-period bookings (shifted post-action
bookings) and decision-independent terms have been dropped (they have zero
mean over arrivals, so the expected penalized lower bound is unchanged).

For a fixed numeric state the value V_kappa(s) is LINEAR in kappa, so the
coefficients can be fitted with ``ApproxQAgent.regression_train`` against
PO lower-bound targets. For fixed numeric kappa the expected continuation
value is QUADRATIC in the decision variables (the decision model already
sets NonConvex=2).
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from generating_function import GeneratingFunction


class MulticlassQuadraticPenaltyFunction(GeneratingFunction):

    def __init__(self, env, coefficients=None):
        self.env = env
        self.coefficients = None
        (self.kappa_0, self.kappa_u, self.kappa_v, self.kappa_w,
         self.kappa_uw, self.kappa_vw, self.kappa_ww) = (None,) * 7
        self.arrival_mean, self.arrival_second_moment = self._compute_arrival_moments(env.arrival_generator)
        if coefficients is not None:
            self.set_coefficients(coefficients)

    @property
    def number_of_coefficients(self):
        T, I = self.env.planning_horizon, self.env.num_types
        return 1 + 2 * T + 2 * I + 2 * I * T

    @staticmethod
    def _compute_arrival_moments(arrival_generator):
        """First and second moments of the per-type arrivals delta_i.

        With total arrivals N ~ truncated Poisson and a multinomial split with
        probabilities p, delta_i | N ~ Binomial(N, p_i), hence
            E[delta_i]   = p_i * E[N]
            E[delta_i^2] = p_i * (1 - p_i) * E[N] + p_i^2 * E[N^2].
        Falls back to plain Poisson moments if the truncated pmf is missing.
        """
        type_probs = np.asarray(arrival_generator.type_probs, dtype=float)
        pmf = getattr(arrival_generator, 'truncate_poisson_pmf', None)
        if pmf is not None:
            support = np.arange(len(pmf), dtype=float)
            first_moment_total = float(pmf @ support)
            second_moment_total = float(pmf @ (support ** 2))
            mean_by_type = type_probs * first_moment_total
            second_moment_by_type = (type_probs * (1.0 - type_probs) * first_moment_total
                                     + (type_probs ** 2) * second_moment_total)
        else:
            mean_by_type = np.asarray(arrival_generator.mean_by_type, dtype=float)
            second_moment_by_type = mean_by_type + mean_by_type ** 2
        return mean_by_type, second_moment_by_type

    def get_coefficients(self, solution):
        if solution is None:
            if self.coefficients is None:
                return (None,) * 7
            solution = self.coefficients
        T, I = self.env.planning_horizon, self.env.num_types
        sizes = [1, T, T, I, I * T, I * T, I]
        offsets = [0] + list(np.cumsum(sizes))
        if not isinstance(solution, gp.MVar):
            solution = np.array(solution, dtype=float)
        kappa_0 = solution[offsets[0]:offsets[1]]
        kappa_u = solution[offsets[1]:offsets[2]]
        kappa_v = solution[offsets[2]:offsets[3]]
        kappa_w = solution[offsets[3]:offsets[4]]
        kappa_uw = solution[offsets[4]:offsets[5]].reshape(I, T)
        kappa_vw = solution[offsets[5]:offsets[6]].reshape(I, T)
        kappa_ww = solution[offsets[6]:offsets[7]]
        return kappa_0, kappa_u, kappa_v, kappa_w, kappa_uw, kappa_vw, kappa_ww

    def set_coefficients(self, solution):
        self.coefficients = solution
        (self.kappa_0, self.kappa_u, self.kappa_v, self.kappa_w,
         self.kappa_uw, self.kappa_vw, self.kappa_ww) = self.get_coefficients(solution)

    def _get_coefficient_blocks(self, coefficients):
        blocks = coefficients if coefficients is not None else self.get_coefficients(self.coefficients)
        if any(block is None for block in blocks):
            raise ValueError("coefficients are required before evaluating the generating function.")
        return blocks

    def _next_period_bookings(self, state, action, is_var):
        """Next-period bookings (u^{t+1}, v^{t+1}) and post-action waitlist."""
        (post_action_regular_bookings,
         post_action_overtimes,
         post_action_waitlist) = self.env.post_action_state(state, action, is_var)
        next_regular_bookings = self.env.shift_matrix @ post_action_regular_bookings
        next_overtimes = self.env.shift_matrix @ post_action_overtimes
        return next_regular_bookings, next_overtimes, post_action_waitlist

    def calculate_state_value(self, state, is_var=False, coefficients=None):
        """Evaluate V_kappa(s); LINEAR in kappa for a fixed numeric state."""
        regular_bookings, overtimes, waitlist = state
        (kappa_0, kappa_u, kappa_v, kappa_w,
         kappa_uw, kappa_vw, kappa_ww) = self._get_coefficient_blocks(coefficients)
        value = (kappa_0.item()
                 + kappa_u @ regular_bookings
                 + kappa_v @ overtimes
                 + kappa_w @ waitlist
                 + waitlist @ (kappa_uw @ regular_bookings)
                 + waitlist @ (kappa_vw @ overtimes)
                 + kappa_ww @ (waitlist * waitlist))
        return self._as_scalar_expression(value)

    def calculate_expected_continuation_value(self, state, action, is_var=False, coefficients=None):
        """Closed-form E[V_kappa(f(s, a, delta))] over the arrival distribution.

        QUADRATIC in the decisions for fixed numeric kappa (requires
        NonConvex=2), and LINEAR in kappa for fixed numeric state/action.
        """
        (kappa_0, kappa_u, kappa_v, kappa_w,
         kappa_uw, kappa_vw, kappa_ww) = self._get_coefficient_blocks(coefficients)
        next_regular_bookings, next_overtimes, post_action_waitlist = self._next_period_bookings(state, action, is_var)
        mean = self.arrival_mean
        second_moment = self.arrival_second_moment
        expected_waitlist = post_action_waitlist + mean
        value = (kappa_0.item()
                 + kappa_u @ next_regular_bookings
                 + kappa_v @ next_overtimes
                 + kappa_w @ expected_waitlist
                 + expected_waitlist @ (kappa_uw @ next_regular_bookings)
                 + expected_waitlist @ (kappa_vw @ next_overtimes)
                 + (kappa_ww * post_action_waitlist) @ post_action_waitlist
                 + 2.0 * ((kappa_ww * mean) @ post_action_waitlist)
                 + kappa_ww @ second_moment)
        return self._as_scalar_expression(value)

    def calculate_penalty(self, state, action, new_arrival, is_var=False, coefficients=None):
        """Reduced zero-mean penalty z_kappa(s, a, delta).

        Decision-independent terms are dropped; products are arranged so the
        numeric arrival-difference vector hits the kappa blocks first, keeping
        the expression LINEAR in kappa for fixed decisions (Benders) and
        LINEAR in the decisions for fixed numeric kappa (information
        relaxation / hindsight models).
        """
        (kappa_0, kappa_u, kappa_v, kappa_w,
         kappa_uw, kappa_vw, kappa_ww) = self._get_coefficient_blocks(coefficients)
        next_regular_bookings, next_overtimes, post_action_waitlist = self._next_period_bookings(state, action, is_var)
        arrival_difference = self.arrival_mean - np.asarray(new_arrival, dtype=float)
        penalty_value = (2.0 * ((arrival_difference * kappa_ww) @ post_action_waitlist)
                         + (arrival_difference @ kappa_uw) @ next_regular_bookings
                         + (arrival_difference @ kappa_vw) @ next_overtimes)
        return self._as_scalar_expression(penalty_value)

    def calculate_gradient(self, state, action, new_arrival, is_var=False):
        raise NotImplemented

    def get_coefficient_var(self, model, coefficient_bound):
        return model.addMVar(shape=self.number_of_coefficients,
                             vtype=GRB.CONTINUOUS,
                             lb=-coefficient_bound,
                             ub=coefficient_bound,
                             name="coefficients")

    def build_coefficient_linking_constraints(self, model, coefficient_vars):
        return model.addConstr(coefficient_vars == 0.0, name="link_coefficients")


if __name__ == "__main__":
    from experiments import get_config_by_type
    config = get_config_by_type('toy')
    env = config.env
    state, info = env.reset(**config.reset_params)
    generating_function = MulticlassQuadraticPenaltyFunction(env)
    coefficients = [1] * generating_function.number_of_coefficients
    generating_function.set_coefficients(coefficients)
    action = list(env.valid_actions(state))[-1]
    print(state)
    print(action)
    print('kappa_0 :', generating_function.kappa_0.shape)
    print('kappa_u :', generating_function.kappa_u.shape)
    print('kappa_v :', generating_function.kappa_v.shape)
    print('kappa_w :', generating_function.kappa_w.shape)
    print('kappa_uw:', generating_function.kappa_uw.shape)
    print('kappa_vw:', generating_function.kappa_vw.shape)
    print('kappa_ww:', generating_function.kappa_ww.shape)
    new_arrival = np.array([1, 1])
    penalty = generating_function.calculate_penalty(state, action, new_arrival, is_var=False)
    expected_continuation_value = generating_function.calculate_expected_continuation_value(state, action)
    state_value = generating_function.calculate_state_value(state)
    print(f"Calculated penalty: {penalty}")
    print(f"Expected continuation value: {expected_continuation_value}")
    print(f"State value: {state_value}")
