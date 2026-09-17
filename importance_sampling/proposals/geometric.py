import numpy as np
from scipy.stats import geom

from importance_sampling.sample_path import Terminal

from .base import (
    SamplePathLengthProposal,
    _validate_discount_factor,
    _validate_max_length,
)


class GeometricLengthProposal(SamplePathLengthProposal):

    def __init__(self, discount_factor_proposal: float):
        _validate_discount_factor(discount_factor_proposal, 'discount_factor_proposal')
        self.discount_factor_proposal = discount_factor_proposal

    def sample_lengths(self, arrival_generator, size):
        return arrival_generator.rng.geometric(p=1 - self.discount_factor_proposal, size=size)

    def survival_probability(self, periods):
        periods = np.asarray(periods, dtype=int)
        return self.discount_factor_proposal ** (periods - 1)


class StratifiedGeometricLengthProposal(GeometricLengthProposal):

    def __init__(self, discount_factor_proposal: float, num_strata: int):
        super().__init__(discount_factor_proposal)
        num_strata = int(num_strata)
        if num_strata <= 0:
            raise ValueError(f"num_strata must be positive. Got {num_strata}.")
        self.num_strata = num_strata

    def stratum_sizes(self, size):
        size = int(size)
        if size < self.num_strata:
            raise ValueError(
                f"size={size} is smaller than num_strata={self.num_strata}; every interval needs a sample.")
        base, remainder = divmod(size, self.num_strata)
        return np.array([base + (1 if stratum < remainder else 0) for stratum in range(self.num_strata)])

    def sample_lengths(self, arrival_generator, size):
        quantiles = np.concatenate([
            (stratum + arrival_generator.rng.uniform(size=count)) / self.num_strata
            for stratum, count in enumerate(self.stratum_sizes(size))])
        quantiles = np.clip(quantiles, 1e-12, 1.0 - 1e-12)
        return np.asarray(geom.ppf(quantiles, 1.0 - self.discount_factor_proposal)).astype(int)

    def path_weights(self, size):
        return np.concatenate([np.full(count, 1.0 / (self.num_strata * count)) for count in self.stratum_sizes(size)])

    def path_strata(self, size):
        return np.concatenate([np.full(count, stratum, dtype=int) for stratum, count in enumerate(self.stratum_sizes(size))])


class TruncatedGeometricLengthProposal(SamplePathLengthProposal):

    def __init__(self, discount_factor_proposal: float, max_length: int):
        _validate_discount_factor(discount_factor_proposal, 'discount_factor_proposal')
        self.discount_factor_proposal = discount_factor_proposal
        self.max_length = _validate_max_length(max_length)

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

    def terminal_for(self, lengths, arrival_generator=None):
        return [Terminal.TRUNCATED if int(length) >= self.max_length else Terminal.ABSORBED
                for length in lengths]
