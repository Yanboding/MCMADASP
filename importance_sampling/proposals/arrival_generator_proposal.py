from typing import Optional

import numpy as np

from .base import SamplePathLengthProposal


class ArrivalGeneratorSamplePathProposal(SamplePathLengthProposal):
    """Sample paths with the environment arrival generator's path sampler.

    This matches the agents' default non-proposal sampling branch: path lengths
    and arrivals are both produced by ``arrival_generator.mc_rvs`` or
    ``arrival_generator.quasi_rvs``, and likelihood ratios are one.
    """

    def __init__(self, use_quasi_mc: Optional[bool] = None, is_positive_integer_support: bool = False):
        self.use_quasi_mc = use_quasi_mc
        self.is_positive_integer_support = is_positive_integer_support

    def _use_quasi_mc(self, arrival_generator):
        if self.use_quasi_mc is not None:
            return bool(self.use_quasi_mc)
        return bool(getattr(arrival_generator, 'use_qmc', False))

    def sample_arrival_paths(self, arrival_generator, size):
        if self._use_quasi_mc(arrival_generator):
            paths = arrival_generator.quasi_rvs(
                size=size,
                is_positive_integer_support=self.is_positive_integer_support,
            )
        else:
            paths = arrival_generator.mc_rvs(
                size=size,
                is_positive_integer_support=self.is_positive_integer_support,
            )
        lengths = np.array([len(path) for path in paths], dtype=int)
        return paths, lengths

    def sample_lengths(self, arrival_generator, size):
        lengths = arrival_generator.rng.geometric(p=arrival_generator.geom_p, size=size)
        if not self.is_positive_integer_support:
            lengths = lengths - 1
        return np.minimum(lengths, arrival_generator.max_periods).astype(int)

    def survival_probability(self, periods):
        raise NotImplementedError(
            "ArrivalGeneratorSamplePathProposal uses the arrival generator directly; "
            "period likelihood ratios are defined by period_likelihood_ratios()."
        )

    def period_likelihood_ratios(self, target_discount_factor, lengths):
        return [np.ones(int(length), dtype=float) for length in lengths]
