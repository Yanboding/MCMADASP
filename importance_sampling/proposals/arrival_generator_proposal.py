from typing import Optional

import numpy as np

from importance_sampling.sample_path import Terminal

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

    def survival_weights(self, lengths, target_discount_factor, arrival_generator=None):
        """The generator's own law IS the target: ``L = Geom(geom_p) - 1`` on
        ``{0, 1, ...}`` has ``P(L >= k) = gamma ** k`` with ``gamma = 1 -
        geom_p``, so every survival weight is 1. With
        ``is_positive_integer_support`` (``L`` on ``{1, 2, ...}``) the first
        arrival is certain and ``w_s = gamma`` for ``s >= 2``."""
        if arrival_generator is None:
            raise ValueError("ArrivalGeneratorSamplePathProposal.survival_weights needs the arrival generator")
        implied = 1.0 - float(arrival_generator.geom_p)
        if not np.isclose(implied, target_discount_factor):
            raise ValueError(
                "target_discount_factor mismatch: the arrival generator draws horizons with "
                f"geom_p={arrival_generator.geom_p} (discount factor {implied}) but the agent "
                f"passed {target_discount_factor}.")
        tail_weight = target_discount_factor if self.is_positive_integer_support else 1.0
        return [np.concatenate(([1.0], np.full(int(length), tail_weight))) for length in lengths]

    def terminal_for(self, lengths, arrival_generator=None):
        """Paths clipped at the generator's ``max_periods`` are truncated."""
        cap = None if arrival_generator is None else int(arrival_generator.max_periods)
        return [Terminal.TRUNCATED if cap is not None and int(length) >= cap else Terminal.ABSORBED
                for length in lengths]
