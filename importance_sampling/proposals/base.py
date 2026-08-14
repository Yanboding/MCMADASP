from abc import ABC, abstractmethod

import numpy as np


def _validate_discount_factor(discount_factor, name):
    if discount_factor is None or not (0 <= discount_factor < 1):
        raise ValueError(f"{name} must be in [0, 1). Got {discount_factor}.")


def _validate_max_length(max_length):
    max_length = int(max_length)
    if max_length <= 0:
        raise ValueError(f"max_length must be positive. Got {max_length}.")
    return max_length


class SamplePathLengthProposal(ABC):
    """Base class for sample-path length importance-sampling proposals.

    These proposals only change the random horizon length. The per-period
    arrival distribution remains unchanged and is sampled by the environment's
    arrival generator.

    For a target geometric horizon with discount factor gamma, the contribution
    from period t is weighted by

        P_target(L >= t) / P_proposal(L >= t)
        = gamma ** (t - 1) / survival_probability(t).

    Finite-support proposals therefore estimate the target truncated to their
    support; they do not estimate the infinite tail unless a separate tail
    correction is added.
    """

    @abstractmethod
    def sample_lengths(self, arrival_generator, size):
        """Return positive integer sample-path lengths.

        ``arrival_generator`` is the environment's arrival generator; proposals
        should draw their randomness from ``arrival_generator.rng`` so that all
        sampling shares the same RNG state.
        """
        raise NotImplementedError

    def sample_arrival_paths(self, arrival_generator, size):
        """Return ``size`` per-period arrival paths drawn under this proposal.

        Lengths are sampled via ``sample_lengths`` and the per-period arrival
        vectors are produced by ``arrival_generator.rvs``. Returns a list of
        arrays with shape ``(length_i, num_types)``.
        """
        lengths = self.sample_lengths(arrival_generator=arrival_generator, size=size)
        return [arrival_generator.rvs(size=int(length)) for length in lengths], lengths

    def path_weights(self, size):
        """Per-path aggregation weights, summing to 1. Uniform by default.

        Stratified proposals override this so the caller's average over paths
        stays unbiased for any deterministic stratum allocation. Weights are
        positionally aligned with ``sample_arrival_paths``' path order.
        """
        size = int(size)
        return np.full(size, 1.0 / size)

    def path_strata(self, size):
        """Integer stratum label per path. Single stratum by default."""
        return np.zeros(int(size), dtype=int)

    @abstractmethod
    def survival_probability(self, periods):
        """Return P_proposal(L >= t) for one-based period indices."""
        raise NotImplementedError

    def period_likelihood_ratios(self, target_discount_factor, lengths):
        _validate_discount_factor(target_discount_factor, 'target_discount_factor')

        def weights_for(length):
            periods = np.arange(1, int(length) + 1)
            proposal_survival = self.survival_probability(periods)
            if np.any(proposal_survival <= 0):
                raise ValueError(
                    "Proposal survival probability must be positive on every sampled period."
                )
            # One-based period indexing: P(L >= t) = gamma ** (t - 1).
            target_survival = target_discount_factor ** (periods - 1)
            return target_survival / proposal_survival

        return [weights_for(length) for length in lengths]
