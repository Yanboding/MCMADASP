import unittest

import numpy as np

from importance_sampling.proposals import (
    GeometricLengthProposal,
    MixtureGeometricStratifiedQMCProposal,
)


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


class _RNGHolder:
    """Minimal stand-in exposing the ``.rng`` attribute proposals sample from."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)


class TestMixtureGeometricProposal(unittest.TestCase):
    def _proposal(self, gamma=0.99, q=0.9, lambda_0=0.5):
        return MixtureGeometricStratifiedQMCProposal(
            target_discount_factor=gamma,
            discount_factor_proposal=q,
            lambda_0=lambda_0,
        )

    def test_survival_probability_matches_mixture_closed_form(self):
        gamma, q, lambda_0 = 0.99, 0.9, 0.4
        proposal = self._proposal(gamma=gamma, q=q, lambda_0=lambda_0)
        periods = np.array([1, 2, 5, 10])
        observed = proposal.survival_probability(periods)
        expected = lambda_0 * gamma ** (periods - 1) + (1 - lambda_0) * q ** (periods - 1)
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)
        # Period 1 survival is always 1 (the mixture mass sums to one).
        self.assertAlmostEqual(observed[0], 1.0, places=12)

    def test_period_weights_match_closed_form_and_are_bounded(self):
        gamma, q, lambda_0 = 0.99, 0.9, 0.5
        proposal = self._proposal(gamma=gamma, q=q, lambda_0=lambda_0)
        length = 50
        weights = proposal.period_likelihood_ratios(
            target_discount_factor=gamma, lengths=np.array([length])
        )[0]
        periods = np.arange(1, length + 1)
        s_prop = lambda_0 * gamma ** (periods - 1) + (1 - lambda_0) * q ** (periods - 1)
        expected = gamma ** (periods - 1) / s_prop
        np.testing.assert_allclose(weights, expected, rtol=1e-12, atol=1e-12)
        # w_1 = 1, monotone increasing, bounded by 1 / lambda_0.
        self.assertAlmostEqual(weights[0], 1.0, places=12)
        self.assertTrue(np.all(np.diff(weights) >= -1e-12))
        self.assertTrue(np.all(weights <= 1.0 / lambda_0 + 1e-9))

    def test_period_likelihood_ratios_rejects_gamma_mismatch(self):
        proposal = self._proposal(gamma=0.99, q=0.9)
        with self.assertRaises(ValueError):
            proposal.period_likelihood_ratios(
                target_discount_factor=0.95, lengths=np.array([3])
            )

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):  # q must be strictly < gamma
            self._proposal(gamma=0.9, q=0.95)
        with self.assertRaises(ValueError):  # q == gamma not allowed
            self._proposal(gamma=0.9, q=0.9)
        with self.assertRaises(ValueError):  # lambda_0 must be in (0, 1]
            self._proposal(lambda_0=0.0)
        with self.assertRaises(ValueError):
            self._proposal(lambda_0=1.5)

    def test_sample_lengths_shape_positivity_and_reproducibility(self):
        proposal = self._proposal(gamma=0.99, q=0.9, lambda_0=0.5)
        size = 64
        lengths_a = proposal.sample_lengths(_RNGHolder(0), size)
        lengths_b = proposal.sample_lengths(_RNGHolder(0), size)
        self.assertEqual(lengths_a.shape, (size,))
        self.assertTrue(np.all(lengths_a >= 1))
        # Reproducible given the same RNG seed.
        np.testing.assert_array_equal(lengths_a, lengths_b)
        # Advancing the same RNG yields an independent (different) draw.
        holder = _RNGHolder(0)
        first = proposal.sample_lengths(holder, size)
        second = proposal.sample_lengths(holder, size)
        self.assertFalse(np.array_equal(first, second))

    def test_estimator_is_unbiased_for_constant_costs(self):
        # With c_t = 1 the discounted infinite-horizon sum is 1 / (1 - gamma).
        gamma, q, lambda_0 = 0.9, 0.5, 0.5
        proposal = self._proposal(gamma=gamma, q=q, lambda_0=lambda_0)
        size = 8192
        lengths = proposal.sample_lengths(_RNGHolder(12345), size)
        weights = proposal.period_likelihood_ratios(
            target_discount_factor=gamma, lengths=lengths
        )
        per_path_sums = np.array([w.sum() for w in weights])
        estimate = per_path_sums.mean()
        target = 1.0 / (1.0 - gamma)
        self.assertAlmostEqual(estimate, target, delta=0.05 * target)


if __name__ == "__main__":
    unittest.main()
