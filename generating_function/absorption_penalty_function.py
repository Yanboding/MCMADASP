"""Brown-Haugh absorption-time form of the linear penalty.

Same features as :class:`LinearPenaltyFunction` (``g = (sum delta) * theta .
phi_0``, ``g(x_a) = 0``), but every consumer uses :class:`AbsorptionForm`:
survival-weighted terms, the expected term kept in the last period of an
absorbed path, stratified path weights in the hindsight master. The policy
continuation is the same ``theta . E[phi]`` as the penalty's expected term, so
the legacy ``(sum w+) * theta . phi_0`` workload term disappears.

Brown & Haugh (2017), "Information relaxation bounds for infinite horizon
Markov decision processes", Operations Research 65(5):1355-1379,
doi:10.1287/opre.2017.1631.
"""
from .penalty_forms import absorption_forms
from .penalty_function import LinearPenaltyFunction


class AbsorptionLinearPenaltyFunction(LinearPenaltyFunction):
    spec_name = 'absorption_linear_penalty'
    forms = absorption_forms()

    def calculate_expected_continuation_value(self, state, action, is_var=False, coefficients=None):
        if coefficients is not None:
            raise ValueError("the absorption penalty evaluates its continuation with the stored coefficients")
        return self._as_scalar_expression(self.expected_value(self.coefficient_vector(), state, action, is_var))
