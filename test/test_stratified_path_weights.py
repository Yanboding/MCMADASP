"""Unit tests for per-path stratum weights and the stratified CI helper.

Run from the repo root:  python -m test.test_stratified_path_weights
"""
import numpy as np

from importance_sampling.proposals.arrival_generator_proposal import (
    ArrivalGeneratorSamplePathProposal,
)
from importance_sampling.proposals.mixture import (
    MixtureGeometricStratifiedQMCProposal,
)


def test_base_defaults_uniform_single_stratum():
    proposal = ArrivalGeneratorSamplePathProposal()
    weights = proposal.path_weights(10)
    strata = proposal.path_strata(10)
    assert weights.shape == (10,)
    assert np.allclose(weights, 0.1)
    assert np.isclose(weights.sum(), 1.0)
    assert strata.shape == (10,)
    assert np.all(strata == 0)


def test_mixture_weights_match_allocation():
    proposal = MixtureGeometricStratifiedQMCProposal(
        target_discount_factor=0.99, discount_factor_proposal=0.95, lambda_0=0.1)
    n_long, n_short = proposal._component_sizes(256)
    assert (n_long, n_short) == (26, 230)
    weights = proposal.path_weights(256)
    strata = proposal.path_strata(256)
    assert weights.shape == (256,)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights[:n_long], 0.1 / n_long)
    assert np.allclose(weights[n_long:], 0.9 / n_short)
    assert np.all(strata[:n_long] == 0)
    assert np.all(strata[n_long:] == 1)
    # lambda_0 == 1 degenerates to a single uniform stratum, no raise.
    pure = MixtureGeometricStratifiedQMCProposal(
        target_discount_factor=0.99, discount_factor_proposal=0.95, lambda_0=1.0)
    assert np.allclose(pure.path_weights(8), 1.0 / 8)


def test_mixture_degenerate_stratum_raises():
    proposal = MixtureGeometricStratifiedQMCProposal(
        target_discount_factor=0.99, discount_factor_proposal=0.95, lambda_0=0.1)
    try:
        proposal.path_weights(4)  # round(0.4) == 0 long paths
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for empty long stratum')


def test_ci_uniform_reduces_to_classic():
    from metaheuristic_algorithm.benders_decomposition_solver import (
        BendersDecompositionSolver as B,
    )
    values = [10.0, 12.0, 8.0, 11.0, 9.0]
    mean, half = B._objective_confidence_interval(values)
    arr = np.asarray(values)
    assert np.isclose(mean, arr.mean())
    assert np.isclose(half, 1.96 * arr.std(ddof=1) / np.sqrt(5))
    assert B._objective_confidence_interval([7.5]) == (7.5, 0.0)
    assert B._objective_confidence_interval([]) == (0.0, 0.0)


def test_ci_stratified_matches_hand_calc():
    from metaheuristic_algorithm.benders_decomposition_solver import (
        BendersDecompositionSolver as B,
    )
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    weights = np.array([0.2 / 3] * 3 + [0.8 / 2] * 2)
    strata = np.array([0, 0, 0, 1, 1])
    mean, half = B._objective_confidence_interval(values, weights, strata)
    # Weighted mean: 0.2 * 2 + 0.8 * 15 = 12.4
    assert np.isclose(mean, 12.4)
    # Var = W0^2 s0^2 / n0 + W1^2 s1^2 / n1 = 0.04 * 1 / 3 + 0.64 * 50 / 2
    expected_var = 0.04 * 1.0 / 3 + 0.64 * 50.0 / 2
    assert np.isclose(half, 1.96 * np.sqrt(expected_var))


if __name__ == '__main__':
    test_base_defaults_uniform_single_stratum()
    test_mixture_weights_match_allocation()
    test_mixture_degenerate_stratum_raises()
    test_ci_uniform_reduces_to_classic()
    test_ci_stratified_matches_hand_calc()
    print('All stratified path-weight tests passed.')
