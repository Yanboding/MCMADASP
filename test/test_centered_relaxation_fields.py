import unittest

import gurobipy as gp
import numpy as np

from experiments import get_config_by_type
from generating_function import AbsorptionALPPenaltyFunction
from importance_sampling import Terminal
from run import _centered_relaxation_fields, information_relaxation_bounds


class TestCenteredRelaxationFields(unittest.TestCase):
    def setUp(self):
        self.grb = gp.Env(empty=True)
        self.grb.setParam('OutputFlag', 0)
        self.grb.setParam('Threads', 1)
        self.grb.start()
        self.env = get_config_by_type('toy').env
        self.init_state = (np.array([2.] * (self.env.planning_horizon - 1) + [0.]),
                           np.zeros(self.env.planning_horizon), np.array([1., 2.]))
        self.path = np.array([[1., 2.], [0., 1.]])
        self.period_weights = [1., 0.99, 0.99 ** 2]
        self.anchor = np.linspace(-5.0, 5.0, AbsorptionALPPenaltyFunction(self.env).number_of_coefficients)

    def tearDown(self):
        self.grb.dispose()

    def fields(self, coefficients, uncentered_fitted):
        return _centered_relaxation_fields(
            self.env, {'name': AbsorptionALPPenaltyFunction.spec_name,
                       'coefficients': list(coefficients)},
            self.anchor, self.init_state, self.path, self.period_weights,
            Terminal.ABSORBED, uncentered_fitted, self.grb, [])

    def test_at_the_anchor_the_centered_bounds_coincide(self):
        anchor_bound = information_relaxation_bounds(
            self.env, {'name': AbsorptionALPPenaltyFunction.spec_name,
                       'coefficients': list(self.anchor)},
            [1.0], self.init_state, self.path, self.period_weights, Terminal.ABSORBED,
            self.grb, [])[1.0]
        fields = self.fields(self.anchor, anchor_bound)
        self.assertAlmostEqual(
            fields['centered_fitted_information_relaxation_cost'], anchor_bound, places=9)
        self.assertAlmostEqual(
            fields['centered_alp_information_relaxation_cost'], anchor_bound, places=9)
        self.assertAlmostEqual(fields['center_potential_gap'], 0.0, places=9)

    def test_the_gap_is_the_difference_of_the_two_centered_bounds(self):
        fields = self.fields(self.anchor + 3.0, 1000.0)
        self.assertAlmostEqual(
            fields['center_potential_gap'],
            fields['centered_fitted_information_relaxation_cost']
            - fields['centered_alp_information_relaxation_cost'], places=9)

    def test_the_centering_term_is_linear_in_the_coefficient_offset(self):
        base = self.fields(self.anchor, 0.0)['centered_fitted_information_relaxation_cost']
        single = self.fields(self.anchor + 1.0, 0.0)['centered_fitted_information_relaxation_cost']
        double = self.fields(self.anchor + 2.0, 0.0)['centered_fitted_information_relaxation_cost']
        self.assertAlmostEqual(double - base, 2.0 * (single - base), places=6)
        self.assertNotAlmostEqual(single, base, places=6)


if __name__ == '__main__':
    unittest.main()
