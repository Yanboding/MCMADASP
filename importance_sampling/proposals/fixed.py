import numpy as np

from .base import SamplePathLengthProposal, _validate_max_length


class FixedLengthProposal(SamplePathLengthProposal):
    """Deterministic length proposal with L = max_length."""

    def __init__(self, max_length: int):
        self.max_length = _validate_max_length(max_length)

    def sample_lengths(self, arrival_generator, size):
        return np.full(size, self.max_length, dtype=int)

    def survival_probability(self, periods):
        periods = np.asarray(periods, dtype=int)
        # +1: the trailing decision period after the last sampled arrival
        # (stage cost but no arrival, see ``_evaluation_period_weights``) is
        # guaranteed to occur once the fixed-length tail has been drawn, so it
        # must not be treated as unsupported (survival probability 0).
        return (periods <= self.max_length + 1).astype(float)
