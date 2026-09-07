"""``SamplePath`` and the proposals' survival weights / terminal outcomes.

Run from the repo root:  python -m test.test_sample_path
"""
import numpy as np

from environment.arrival_generator import MultiClassPoissonArrivalGenerator
from experiments import get_config_by_type
from importance_sampling import (
    ArrivalGeneratorSamplePathProposal,
    FixedLengthProposal,
    GeometricLengthProposal,
    MixtureGeometricStratifiedQMCProposal,
    SamplePath,
    Terminal,
    TruncatedGeometricLengthProposal,
    sample_path_from_record,
)


def generator(gamma, seed=3, max_periods=400):
    return MultiClassPoissonArrivalGenerator(
        mean_arrival_rate=3, maximum_arrival=9, type_probs=[0.5, 0.3, 0.2], random_seed=seed,
        use_qmc=False, max_periods=max_periods, geom_p=1.0 - gamma)


def test_sample_path_basics_and_record_rebuild():
    path = SamplePath(np.array([[1, 0], [0, 2]]), 'absorbed', [1.0, 0.9, 0.9], [1.0, 1.0])
    assert path.length == 2 and path.periods == 3 and path.terminal is Terminal.ABSORBED
    empty = SamplePath(np.zeros((0, 2)))
    assert empty.periods == 1 and empty.terminal is Terminal.UNSPECIFIED and empty.survival_weights is None
    record = sample_path_from_record([[1, 0]], period_weights=[1.0, 0.5, 0.25], terminal=None)
    assert record.terminal is Terminal.UNSPECIFIED
    assert np.allclose(record.survival_weights, [1.0, 0.5])  # extra entries dropped
    for bad in (dict(survival_weights=[1.0]), dict(likelihood_ratios=[1.0, 1.0, 1.0])):
        try:
            SamplePath(np.array([[1, 0], [0, 2]]), **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'expected ValueError for {bad}')
    try:
        sample_path_from_record([[1, 0], [0, 2]], period_weights=[1.0, 0.5])
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for too-short period_weights')


def test_default_proposal_weights_are_one_and_lengths_start_at_zero():
    gen = generator(0.9)
    proposal = ArrivalGeneratorSamplePathProposal()
    paths = proposal.sample_paths(gen, 200, target_discount_factor=0.9)
    lengths = np.array([p.length for p in paths])
    assert lengths.min() == 0 and lengths.max() < gen.max_periods
    for path in paths:
        assert np.all(path.survival_weights == 1.0) and path.survival_weights.shape == (path.length + 1,)
        assert np.all(path.likelihood_ratios == 1.0) and path.terminal is Terminal.ABSORBED
    # Same RNG consumption as the legacy path sampler.
    legacy, _ = proposal.sample_arrival_paths(generator(0.9), 200)
    assert all(np.array_equal(a, b.arrivals) for a, b in zip(legacy, paths))
    # Capped draws are truncated.
    capped = generator(0.9, max_periods=2)
    assert Terminal.TRUNCATED in {p.terminal for p in proposal.sample_paths(capped, 50, 0.9)}
    # Positive-integer support: first arrival certain, gamma afterwards.
    positive = ArrivalGeneratorSamplePathProposal(is_positive_integer_support=True)
    for path in positive.sample_paths(generator(0.9), 20, 0.9):
        assert path.length >= 1 and path.survival_weights[0] == 1.0
        assert np.allclose(path.survival_weights[1:], 0.9)


def test_geometric_and_mixture_weights_match_closed_forms():
    gamma, q, lambda_0 = 0.99, 0.95, 0.5
    for path in GeometricLengthProposal(gamma).sample_paths(generator(gamma), 16, gamma):
        assert path.length >= 1 and path.terminal is Terminal.ABSORBED
        assert path.survival_weights[0] == 1.0 and np.allclose(path.survival_weights[1:], gamma)
        assert np.allclose(path.likelihood_ratios, 1.0)
    mixture = MixtureGeometricStratifiedQMCProposal(gamma, q, lambda_0)
    for path in mixture.sample_paths(generator(gamma), 16, gamma):
        for index, observed in enumerate(path.survival_weights):
            s = index + 1
            survival = 1.0 if s == 1 else lambda_0 * gamma ** (s - 2) + (1 - lambda_0) * q ** (s - 2)
            assert np.isclose(observed, gamma ** (s - 1) / survival, rtol=1e-12)
        assert np.allclose(path.survival_weights[1:], gamma * path.likelihood_ratios)


def test_finite_support_proposals_report_truncation():
    fixed = FixedLengthProposal(3)
    for path in fixed.sample_paths(generator(0.9), 5, 0.9):
        assert path.length == 3 and path.terminal is Terminal.TRUNCATED
        assert np.allclose(path.survival_weights, [1.0, 0.9, 0.81, 0.729])
    truncated = TruncatedGeometricLengthProposal(0.5, max_length=2)
    paths = truncated.sample_paths(generator(0.9), 200, 0.9)
    terminals = {p.length: p.terminal for p in paths}
    assert terminals[2] is Terminal.TRUNCATED and terminals[1] is Terminal.ABSORBED
    assert all(np.all(np.isfinite(p.survival_weights)) for p in paths)


def test_discount_factor_mismatch_raises():
    for proposal in (ArrivalGeneratorSamplePathProposal(),
                     MixtureGeometricStratifiedQMCProposal(0.99, 0.95, 0.5)):
        try:
            proposal.sample_paths(generator(0.99), 4, target_discount_factor=0.9)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{type(proposal).__name__} accepted a mismatched discount factor')


def test_survival_weights_are_unbiased_for_the_target_horizon():
    """E_q[sum_s w_s 1{L >= s - 1}] = sum_s gamma ** (s - 1) = 1 / (1 - gamma)."""
    gamma, draws = 0.9, 40000
    for proposal in (ArrivalGeneratorSamplePathProposal(),
                     GeometricLengthProposal(0.8),
                     MixtureGeometricStratifiedQMCProposal(gamma, 0.7, 0.5)):
        gen = generator(gamma, seed=11)
        lengths = proposal.sample_lengths(gen, draws)
        weights = proposal.survival_weights(lengths, gamma, gen)
        estimate = np.mean([w.sum() for w in weights])
        assert abs(estimate - 1.0 / (1.0 - gamma)) < 0.03 * (1.0 / (1.0 - gamma)), (type(proposal).__name__, estimate)


def test_toy_env_default_proposal_matches_its_generator():
    config = get_config_by_type('toy')
    env = config.env
    paths = ArrivalGeneratorSamplePathProposal().sample_paths(env.arrival_generator, 8, env.discount_factor)
    assert all(np.all(p.survival_weights == 1.0) for p in paths)


if __name__ == '__main__':
    test_sample_path_basics_and_record_rebuild()
    test_default_proposal_weights_are_one_and_lengths_start_at_zero()
    test_geometric_and_mixture_weights_match_closed_forms()
    test_finite_support_proposals_report_truncation()
    test_discount_factor_mismatch_raises()
    test_survival_weights_are_unbiased_for_the_target_horizon()
    test_toy_env_default_proposal_matches_its_generator()
    print('All sample-path tests passed.')
