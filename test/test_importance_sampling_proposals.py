import unittest

import numpy as np

from importance_sampling.proposals import GeometricLengthProposal


class TestGeometricImportanceSampling(unittest.TestCase):
    def test_survival_probability_one_based_periods(self):
        proposal_gamma = 0.98
        proposal = GeometricLengthProposal(discount_factor_proposal=proposal_gamma)

        periods = np.array([1, 2, 5, 10])
        observed = proposal.survival_probability(periods)
        expected = proposal_gamma ** (periods - 1)

        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)

    def test_period_likelihood_ratios_match_closed_form_for_098_to_099(self):
        proposal_gamma = 0.98
        target_gamma = 0.99
        proposal = GeometricLengthProposal(discount_factor_proposal=proposal_gamma)

        lengths = np.array([1, 2, 5, 9])
        observed = proposal.period_likelihood_ratios(
            target_discount_factor=target_gamma,
            lengths=lengths,
        )

        # For one-based period t, ratio is:
        # gamma_target^(t-1) / gamma_proposal^(t-1)
        expected = [
            (target_gamma / proposal_gamma) ** np.arange(int(length))
            for length in lengths
        ]
        print(observed)
        for observed_ratio, expected_ratio in zip(observed, expected):
            np.testing.assert_allclose(observed_ratio, expected_ratio, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
