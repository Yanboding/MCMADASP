import unittest

import numpy as np

from scipy.stats import geom

from importance_sampling.proposals import (
    GeometricLengthProposal,
    MixtureGeometricStratifiedQMCProposal,
    StratifiedGeometricLengthProposal,
    build_proposal,
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
        expected = [
            (target_gamma / proposal_gamma) ** np.arange(int(length))
            for length in lengths
        ]
        print(observed)
        for observed_ratio, expected_ratio in zip(observed, expected):
            np.testing.assert_allclose(observed_ratio, expected_ratio, rtol=1e-12, atol=1e-12)


class _RNGHolder:

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
        with self.assertRaises(ValueError):
            self._proposal(gamma=0.9, q=0.95)
        with self.assertRaises(ValueError):
            self._proposal(gamma=0.9, q=0.9)
        with self.assertRaises(ValueError):
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
        np.testing.assert_array_equal(lengths_a, lengths_b)
        holder = _RNGHolder(0)
        first = proposal.sample_lengths(holder, size)
        second = proposal.sample_lengths(holder, size)
        self.assertFalse(np.array_equal(first, second))

    def test_estimator_is_unbiased_for_constant_costs(self):
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


class TestStratifiedGeometricProposal(unittest.TestCase):

    def test_every_interval_receives_its_share_of_lengths(self):
        q, num_strata, size = 0.9, 16, 64
        proposal = StratifiedGeometricLengthProposal(discount_factor_proposal=q, num_strata=num_strata)
        lengths = proposal.sample_lengths(_RNGHolder(3), size)
        strata = proposal.path_strata(size)
        self.assertEqual(lengths.shape, (size,))
        np.testing.assert_array_equal(np.bincount(strata), np.full(num_strata, size // num_strata))
        for length, stratum in zip(lengths, strata):
            lower, upper = stratum / num_strata, (stratum + 1) / num_strata
            self.assertGreaterEqual(geom.cdf(length, 1 - q), lower)
            self.assertLess(geom.cdf(length - 1, 1 - q), upper)

    def test_remainder_paths_go_to_the_first_intervals(self):
        proposal = StratifiedGeometricLengthProposal(discount_factor_proposal=0.9, num_strata=4)
        np.testing.assert_array_equal(proposal.path_strata(10), [0, 0, 0, 1, 1, 1, 2, 2, 3, 3])
        weights = proposal.path_weights(10)
        np.testing.assert_allclose(weights, [1 / 12] * 6 + [1 / 8] * 4)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertEqual(len(proposal.sample_lengths(_RNGHolder(0), 10)), 10)

    def test_mean_length_is_close_to_the_target_horizon(self):
        proposal = StratifiedGeometricLengthProposal(discount_factor_proposal=0.99, num_strata=1024)
        lengths = proposal.sample_lengths(_RNGHolder(7), 2048)
        self.assertLess(abs(lengths.mean() - 100.0), 1.0)
        iid = GeometricLengthProposal(discount_factor_proposal=0.99).sample_lengths(_RNGHolder(7), 2048)
        self.assertLess(abs(lengths.mean() - 100.0), abs(iid.mean() - 100.0))

    def test_marginal_law_and_period_weights_match_the_geometric_proposal(self):
        stratified = StratifiedGeometricLengthProposal(discount_factor_proposal=0.98, num_strata=8)
        geometric = GeometricLengthProposal(discount_factor_proposal=0.98)
        periods = np.arange(1, 50)
        np.testing.assert_allclose(stratified.survival_probability(periods), geometric.survival_probability(periods))
        for a, b in zip(stratified.period_likelihood_ratios(0.99, [1, 7, 40]), geometric.period_likelihood_ratios(0.99, [1, 7, 40])):
            np.testing.assert_allclose(a, b)
        same = StratifiedGeometricLengthProposal(discount_factor_proposal=0.99, num_strata=8)
        for ratios in same.period_likelihood_ratios(0.99, [3, 25]):
            np.testing.assert_allclose(ratios, 1.0)

    def test_reproducible_with_the_same_seed(self):
        proposal = StratifiedGeometricLengthProposal(discount_factor_proposal=0.95, num_strata=8)
        np.testing.assert_array_equal(proposal.sample_lengths(_RNGHolder(1), 32), proposal.sample_lengths(_RNGHolder(1), 32))
        self.assertFalse(np.array_equal(proposal.sample_lengths(_RNGHolder(1), 32), proposal.sample_lengths(_RNGHolder(2), 32)))

    def test_invalid_num_strata_raise(self):
        with self.assertRaises(ValueError):
            StratifiedGeometricLengthProposal(discount_factor_proposal=0.9, num_strata=0)
        proposal = StratifiedGeometricLengthProposal(discount_factor_proposal=0.9, num_strata=8)
        with self.assertRaises(ValueError):
            proposal.sample_lengths(_RNGHolder(0), 4)
        with self.assertRaises(ValueError):
            proposal.path_weights(4)

    def test_build_proposal_from_spec(self):
        proposal = build_proposal({'type': 'stratified_geometric', 'discount_factor_proposal': 0.99, 'num_strata': 1024})
        self.assertIsInstance(proposal, StratifiedGeometricLengthProposal)
        self.assertEqual(proposal.num_strata, 1024)
        self.assertEqual(proposal.discount_factor_proposal, 0.99)


if __name__ == "__main__":
    unittest.main()
