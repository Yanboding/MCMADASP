from dataclasses import dataclass

import numpy as np

from .base import (
    SamplePathLengthProposal,
    _validate_discount_factor,
    _validate_max_length,
)


@dataclass(frozen=True)
class GeometricLengthProposal(SamplePathLengthProposal):
    """Geometric proposal with support {1, 2, ...}.

    If discount_factor_proposal is gamma_q, then
    P(L >= t) = gamma_q ** (t - 1).
    """

    discount_factor_proposal: float

    def __post_init__(self):
        _validate_discount_factor(self.discount_factor_proposal, 'discount_factor_proposal')

    def sample_lengths(self, arrival_generator, size):
        return arrival_generator.rng.geometric(p=1 - self.discount_factor_proposal, size=size)

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

    def sample_lengths(self, arrival_generator, size):
        if self.discount_factor_proposal == 0:
            return np.ones(size, dtype=int)
        support = np.arange(1, self.max_length + 1)
        probabilities = (1 - self.discount_factor_proposal) * self.discount_factor_proposal ** (support - 1)
        probabilities = probabilities / probabilities.sum()
        return arrival_generator.rng.choice(support, size=size, p=probabilities)

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
