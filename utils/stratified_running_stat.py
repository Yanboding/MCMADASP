"""Weighted stratified running statistics (single-class module)."""
import numbers
from math import sqrt

import numpy as np
import scipy.stats as st

from .numpy_running_stat import RunningStats


class StratifiedRunningStats:
    """Weighted stratified statistics; one ``RunningStats`` per stratum.

    mean = sum(w_i x_i) / sum(w_i)  (self-normalized: robust to missing
    records; the plain mean when all weights are 1, which is the fallback
    for legacy records without weight fields).

    half_window = t_{n - H} * sqrt( sum_h What_h^2 * s_h^2 / n_h ) with
    ``What_h`` the stratum's share of total weight, ``s_h^2`` its
    within-stratum sample variance, and ``H`` the number of strata. With a
    single stratum and unit weights this reduces to ``RunningStats``' own
    ``t_{n-1} * s / sqrt(n)``, so legacy data reproduces legacy numbers.

    Assumes weights are equal WITHIN each stratum (true for the stratified
    mixture-geometric proposal). Derived operations (division,
    ``mean_difference``, formatting) collapse to an equivalent plain
    ``RunningStats`` via :meth:`to_running_stats` and reuse its machinery;
    the collapsed object uses df ``n - 1`` instead of ``n - H`` (negligible
    for n >> H) and drops stratum detail, so derived objects are summary
    statistics that cannot be further stratified.
    """

    def __init__(self):
        self._strata = {}        # stratum label -> RunningStats
        self._weight_sums = {}   # stratum label -> sum of weights
        self._weighted_sum = 0.0
        self._weight_total = 0.0

    def record(self, value, weight=None, stratum=None):
        weight = 1.0 if weight is None else float(weight)
        stratum = 0 if stratum is None else stratum
        self._strata.setdefault(stratum, RunningStats()).record(float(value))
        self._weight_sums[stratum] = self._weight_sums.get(stratum, 0.0) + weight
        self._weighted_sum += weight * float(value)
        self._weight_total += weight

    @property
    def n(self):
        return sum(stats.n for stats in self._strata.values())

    @property
    def mean(self):
        return self._weighted_sum / self._weight_total if self._weight_total else 0.0

    @property
    def variance_of_mean(self):
        """sum_h What_h^2 * s_h^2 / n_h — within-stratum variance only."""
        if not self._weight_total:
            return 0.0
        variance = 0.0
        for label, stats in self._strata.items():
            if stats.n >= 2:
                weight_share = self._weight_sums[label] / self._weight_total
                variance += weight_share ** 2 * stats.variance / stats.n
        return variance

    def half_window(self, confidence):
        df = self.n - len(self._strata)
        if df < 1:
            return 0.0
        t_crit = np.abs(st.t.ppf((1 - confidence) / 2, df))
        return float(t_crit * sqrt(self.variance_of_mean))

    def to_running_stats(self):
        """Collapse to a plain ``RunningStats`` reproducing this estimator's
        mean and half-window (``variance / n == variance_of_mean``)."""
        n = self.n
        m2 = self.variance_of_mean * n * max(n - 1, 0)
        return RunningStats(n=n, mean=self.mean, m2=m2)

    def __iadd__(self, other):
        if isinstance(other, numbers.Number):
            self.record(float(other))
            return self
        if not isinstance(other, StratifiedRunningStats):
            return NotImplemented
        for label, stats in other._strata.items():
            if label in self._strata:
                self._strata[label] += stats
            else:
                self._strata[label] = stats.copy()
            self._weight_sums[label] = (
                self._weight_sums.get(label, 0.0) + other._weight_sums[label])
        self._weighted_sum += other._weighted_sum
        self._weight_total += other._weight_total
        return self

    def __truediv__(self, other):
        if isinstance(other, StratifiedRunningStats):
            other = other.to_running_stats()
        return self.to_running_stats() / other

    def mean_difference(self, other, confidence):
        if isinstance(other, StratifiedRunningStats):
            other = other.to_running_stats()
        return self.to_running_stats().mean_difference(other, confidence)

    def confidence_interval(self, confidence=0.95):
        return self.to_running_stats().confidence_interval(confidence)

    def __repr__(self):
        return (f"StratifiedRunningStats(n={self.n}, strata={len(self._strata)}, "
                f"mean={self.mean:.4f}, 95% CI half={self.half_window(0.95):.4f})")
