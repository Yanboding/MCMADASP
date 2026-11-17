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

    def add_action_space_constraints(self, model, state_var, action_var):
        super().add_action_space_constraints(model, state_var, action_var)
        if not self.restricted_types:
            return model
        regular_bookings = np.asarray(state_var[0], dtype=float)
        advance_scheduling_decision_vars = action_var[0]
        first_session = self.env.treatment_pattern[0]
        for j in range(self.env.booking_window_size):
            limit = max(self.env.regular_capacity - regular_bookings[j] - self.reserve, 0.0)
            model.addConstr(
                sum(first_session[i] * advance_scheduling_decision_vars[j, i] for i in self.restricted_types)
                <= limit,
                name=f"booking_rule_{j}",
            )
        return model
