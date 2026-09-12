from .penalty_forms import absorption_forms
from .penalty_function import LinearPenaltyFunction


class AbsorptionLinearPenaltyFunction(LinearPenaltyFunction):
    spec_name = 'absorption_linear_penalty'
    forms = absorption_forms()

    def calculate_expected_continuation_value(self, state, action, is_var=False, coefficients=None):
        if coefficients is not None:
            raise ValueError("the absorption penalty evaluates its continuation with the stored coefficients")
        return self._as_scalar_expression(self.expected_value(self.coefficient_vector(), state, action, is_var))
