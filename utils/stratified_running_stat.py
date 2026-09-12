import numbers
from math import sqrt

import numpy as np
import scipy.stats as st

from .numpy_running_stat import RunningStats


class StratifiedRunningStats:

    def __init__(self):
        self._strata = {}
        self._weight_sums = {}
        self._weighted_sum = 0.0
        self._weight_total = 0.0
        self.n = 0
        self.mean = 0.0

    def _refresh_mean(self):
        self.mean = self._weighted_sum / self._weight_total if self._weight_total else 0.0

    def record(self, value, weight=None, stratum=None):
        weight = 1.0 if weight is None else float(weight)
        stratum = 0 if stratum is None else stratum
        self._strata.setdefault(stratum, RunningStats()).record(float(value))
        self._weight_sums[stratum] = self._weight_sums.get(stratum, 0.0) + weight
        self._weighted_sum += weight * float(value)
        self._weight_total += weight
        self.n += 1
        self._refresh_mean()

    def variance_of_mean(self):
        if not self._weight_total:
            return 0.0
        variance = 0.0
        for label, stats in self._strata.items():
            if stats.n >= 2:
                weight_share = self._weight_sums[label] / self._weight_total
                variance += weight_share ** 2 * stats.variance() / stats.n
        return variance

    def half_window(self, confidence):
        df = self.n - len(self._strata)
        if df < 1:
            return 0.0
        t_crit = np.abs(st.t.ppf((1 - confidence) / 2, df))
        return float(t_crit * sqrt(self.variance_of_mean()))

    def to_running_stats(self):
        n = self.n
        m2 = self.variance_of_mean() * n * max(n - 1, 0)
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
        self.n += other.n
        self._refresh_mean()
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
