from enum import Enum

import numpy as np


class Terminal(Enum):
    ABSORBED = 'absorbed'
    TRUNCATED = 'truncated'
    UNSPECIFIED = 'unspecified'


def parse_terminal(value):
    if value is None:
        return Terminal.UNSPECIFIED
    if isinstance(value, Terminal):
        return value
    return Terminal(str(value))


class SamplePath:
    def __init__(self, arrivals, terminal=None, survival_weights=None, likelihood_ratios=None):
        self.arrivals = np.asarray(arrivals)
        self.terminal = parse_terminal(terminal)
        self.length = len(self.arrivals)
        self.periods = self.length + 1
        self.survival_weights = None
        if survival_weights is not None:
            weights = np.asarray(survival_weights, dtype=float)
            if weights.shape != (self.periods,):
                raise ValueError(
                    f"survival_weights must have length L + 1 = {self.periods}; got shape {weights.shape}")
            self.survival_weights = weights
        self.likelihood_ratios = None
        if likelihood_ratios is not None:
            ratios = np.asarray(likelihood_ratios, dtype=float)
            if ratios.shape != (self.length,):
                raise ValueError(
                    f"likelihood_ratios must have length L = {self.length}; got shape {ratios.shape}")
            self.likelihood_ratios = ratios


def sample_path_from_record(sample_path, period_weights=None, terminal=None):
    arrivals = np.asarray(sample_path)
    weights = None
    if period_weights is not None:
        weights = np.asarray(period_weights, dtype=float)
        if weights.shape[0] < len(arrivals) + 1:
            raise ValueError(
                f"period_weights has {weights.shape[0]} entries but the path visits "
                f"{len(arrivals) + 1} periods")
        weights = weights[:len(arrivals) + 1]
    return SamplePath(arrivals, terminal, weights, None)
