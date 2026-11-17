import numpy as np

from decision_maker.alp_rg_agent import ALPRowGenerationAgent


class ALPBookingRuleAgent(ALPRowGenerationAgent):
    """ALP policy with a start-day booking rule.

    For every booking day j, the first sessions of the restricted treatment types
    started on day j may use at most the regular capacity that is still free on
    day j before the decision, beyond a reserve of ``reserve`` slots:

        sum_{i in restricted} s_i x_{j,i} <= max(C_r - u_j - reserve, 0),

    where s_i is the number of capacity slots of the first session of type i and
    u_j the regular capacity booked on day j before the decision. Everything else
    is the ALP policy.
    """

    def __init__(self, env, discount_factor, restricted_types=(), reserve=0, **kwargs):
        super().__init__(env, discount_factor, **kwargs)
        self.restricted_types = list(restricted_types)
        self.reserve = float(reserve)
        self._rule_matrix = self._build_rule_matrix()

    def _build_rule_matrix(self):
        N, I = self.env.booking_window_size, self.env.num_types
        matrix = np.zeros((N, N * I))
        first_session = self.env.treatment_pattern[0]
        for j in range(N):
            for i in self.restricted_types:
                matrix[j, j * I + i] = first_session[i]
        return matrix

    def rule_matrix(self):
        """Row j gives the first-session slots of restricted types started on booking day j (x flattened row-major)."""
        return self._rule_matrix

    def rule_limits(self, regular_bookings):
        """max(C_r - u_j - reserve, 0) for every booking day j."""
        u = np.asarray(regular_bookings, dtype=float)[:self.env.booking_window_size]
        return np.maximum(self.env.regular_capacity - u - self.reserve, 0.0)

    def add_action_space_constraints(self, model, state_var, action_var):
        super().add_action_space_constraints(model, state_var, action_var)
        if not self.restricted_types:
            return model
        x = action_var[0].reshape(-1)
        model.addConstr(self._rule_matrix @ x <= self.rule_limits(state_var[0]), name="booking_rule")
        return model
