"""Generating functions: linear maps ``theta -> g = theta . phi``.

A generating function only produces **features**. For a post-decision state
``(s, a)`` and the next arrival ``delta`` it returns ``phi(s, a, delta)`` and
its arrival expectation ``E[phi](s, a)``, both as block-sparse
:class:`~generating_function.features.Features`. Everything a consumer needs
is derived from those two maps:

* penalty of one period:  ``theta . (E[phi] - phi)``  (:meth:`penalty_features`
  with unit weights; the forms in :mod:`penalty_forms` supply the weights);
* Benders cut gradient:   the same feature vector without ``theta``;
* policy continuation:    ``theta . E[phi]`` (:meth:`expected_value`).

``forms`` maps a consumer name (``'training'``, ``'hindsight'``,
``'evaluation'``) to the :class:`PenaltyForm` that consumer uses.
"""
import numpy as np


class GeneratingFunction:
    spec_name = None   # name used in generating-function specs / CLI flags
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
        """The stored coefficients as a float vector; raises when untrained."""
        if self.coefficients is None:
            raise ValueError("coefficients are required before evaluating the penalty function.")
        return np.asarray(self.coefficients, dtype=float)

    def get_coefficients(self, solution):
        raise NotImplementedError

    # ---- feature interface (subclasses set ``self.number_of_coefficients``) ---
    def features(self, state, action, arrival, is_var=False):
        """``phi(s, a, delta)`` as :class:`Features`."""
        raise NotImplementedError

    def expected_features(self, state, action, is_var=False):
        """``E_delta[phi(s, a, delta)]`` in closed form, as :class:`Features`."""
        raise NotImplementedError

    def penalty_features(self, state, action, arrival, expected_weight, realized_weight, is_var=False):
        """``expected_weight * E[phi](s, a) - realized_weight * phi(s, a, delta)``.

        One period's contribution to a path's penalty feature. ``arrival`` may
        be ``None`` when ``realized_weight`` is 0 (last period of a path).
        """
        result = self.expected_features(state, action, is_var).scaled(expected_weight)
        if realized_weight != 0.0:
            result = result - self.features(state, action, arrival, is_var).scaled(realized_weight)
        return result

    def value(self, theta, state, action, arrival, is_var=False):
        """``theta . phi(s, a, delta)``."""
        return self.features(state, action, arrival, is_var).dot(theta)

    def expected_value(self, theta, state, action, is_var=False):
        """``theta . E[phi](s, a)``."""
        return self.expected_features(state, action, is_var).dot(theta)

    def form(self, consumer):
        return self.forms[consumer]

    # ---- policy value function -------------------------------------------
    def calculate_expected_continuation_value(self, state, action, is_var=False, coefficients=None):
        raise NotImplementedError

    def get_coefficient_var(self, model, coefficient_bound):
        raise NotImplementedError
