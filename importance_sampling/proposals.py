from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    def sample_lengths(self, rng, size):
        """Return positive integer sample-path lengths."""

    @abstractmethod
    def survival_probability(self, periods):
        """Return P_proposal(L >= t) for one-based period indices."""

    def period_likelihood_ratios(self, target_discount_factor, lengths):
        _validate_discount_factor(target_discount_factor, 'target_discount_factor')
        period_weights = []
        for length in lengths:
            periods = np.arange(1, int(length) + 1)
            proposal_survival = self.survival_probability(periods)
            if np.any(proposal_survival <= 0):
                raise ValueError(
                    "Proposal survival probability must be positive on every sampled period."
                )
            target_survival = target_discount_factor ** (periods - 1)
            period_weights.append(target_survival / proposal_survival)
        return period_weights


@dataclass(frozen=True)
class GeometricLengthProposal(SamplePathLengthProposal):
    """Geometric proposal with support {1, 2, ...}.

    If discount_factor_proposal is gamma_q, then
    P(L >= t) = gamma_q ** (t - 1).
    """

    discount_factor_proposal: float

    def __post_init__(self):
        _validate_discount_factor(self.discount_factor_proposal, 'discount_factor_proposal')

    def sample_lengths(self, rng, size):
        return rng.geometric(p=1 - self.discount_factor_proposal, size=size)

    def survival_probability(self, periods):
        periods = np.asarray(periods, dtype=int)
        return self.discount_factor_proposal ** (periods - 1)


@dataclass(frozen=True)
class TruncatedGeometricLengthProposal(SamplePathLengthProposal):
    """Geometric proposal conditioned on L <= max_length."""

    discount_factor_proposal: float
    max_length: int

    def __post_init__(self):
        _validate_discount_factor(self.discount_factor_proposal, 'discount_factor_proposal')
        object.__setattr__(self, 'max_length', _validate_max_length(self.max_length))

    def sample_lengths(self, rng, size):
        if self.discount_factor_proposal == 0:
            return np.ones(size, dtype=int)
        support = np.arange(1, self.max_length + 1)
        probabilities = (1 - self.discount_factor_proposal) * self.discount_factor_proposal ** (support - 1)
        probabilities = probabilities / probabilities.sum()
        return rng.choice(support, size=size, p=probabilities)

    def survival_probability(self, periods):
        periods = np.asarray(periods, dtype=int)
        survival = np.zeros_like(periods, dtype=float)
        supported = periods <= self.max_length
        if self.discount_factor_proposal == 0:
            survival[supported] = (periods[supported] == 1).astype(float)
            return survival
        numerator = (
            self.discount_factor_proposal ** (periods[supported] - 1)
            - self.discount_factor_proposal ** self.max_length
        )
        denominator = 1 - self.discount_factor_proposal ** self.max_length
        survival[supported] = numerator / denominator
        return survival


@dataclass(frozen=True)
class FixedLengthProposal(SamplePathLengthProposal):
    """Deterministic length proposal with L = max_length."""

    max_length: int

    def __post_init__(self):
        object.__setattr__(self, 'max_length', _validate_max_length(self.max_length))

    def sample_lengths(self, rng, size):
        return np.full(size, self.max_length, dtype=int)

    def survival_probability(self, periods):
        periods = np.asarray(periods, dtype=int)
        return (periods <= self.max_length).astype(float)
