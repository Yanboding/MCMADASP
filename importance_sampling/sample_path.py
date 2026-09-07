"""One sampled arrival path together with what the drawing law knows about it.

A ``SamplePath`` is the unit every penalty consumer works on: ``L`` arrival
vectors visited over ``L + 1`` decision periods, plus

* ``terminal`` -- how the path ended: the process entered the absorbing state
  after the last decision period (``ABSORBED``, the generating function is 0
  there), the continuation is unknown because a finite-support proposal hit
  its bound (``TRUNCATED``), or no absorption law was attached at all
  (``UNSPECIFIED``: legacy records, fixed horizons);
* ``survival_weights`` -- ``w_1 .. w_{L+1}`` with ``w_1 = 1`` and
  ``w_s = gamma ** (s - 1) / P_q(L >= s - 1)``: the unbiased weight of
  everything revealed in period ``s`` when the path was drawn from the
  proposal ``q`` instead of the target absorption law;
* ``likelihood_ratios`` -- the proposal's per-period ratios
  ``u_t = gamma ** (t - 1) / P_q(L >= t)`` (``t = 1 .. L``) that the legacy
  penalty conventions are written in.

Both weight vectors travel with the path because ``w_{s+1} = gamma * u_s``
only holds for proposals whose length law lives on ``{1, 2, ...}``. Either may
be ``None``; only ``AbsorptionForm`` insists on ``survival_weights`` and a
known ``terminal``.
"""
from enum import Enum

import numpy as np


class Terminal(Enum):
    ABSORBED = 'absorbed'
    TRUNCATED = 'truncated'
    UNSPECIFIED = 'unspecified'


def parse_terminal(value):
    """``None`` -> ``Terminal.UNSPECIFIED``; a member or its string value -> member."""
    if value is None:
        return Terminal.UNSPECIFIED
    if isinstance(value, Terminal):
        return value
    return Terminal(str(value))


class SamplePath:
    def __init__(self, arrivals, terminal=None, survival_weights=None, likelihood_ratios=None):
        self.arrivals = np.asarray(arrivals)
        self.terminal = parse_terminal(terminal)
        # Number of sampled arrival vectors L, and the L + 1 decision periods
        # visited (the last one sees no arrival).
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
    """Rebuild the path of an evaluation record as a :class:`SamplePath`.

    ``period_weights`` (``None`` or ``L + 1`` or more entries; the extra
    entries a longer record may carry are dropped) are the record's survival
    weights over exactly this tail; ``terminal`` is the record's string or
    ``None`` for legacy records.
    """
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
