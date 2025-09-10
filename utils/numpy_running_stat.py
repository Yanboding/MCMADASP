import numpy as np
import numbers
from math import sqrt
import scipy.stats as st


class RunningStats:
    """
    Calculates running statistics for a data stream in a memory-efficient manner.
    """

    def __init__(self, n=0, mean=0.0, m2=0.0):
        self._n: int = n
        self._mean: float = mean
        self._m2: float = m2

    # ... (All properties: .n, .mean, .variance, .std remain the same) ...
    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean if self._n > 0 else 0.0
    
    @property
    def var_sum(self) -> float:
        return self._m2 if self._n > 0 else 0.0

    @property
    def var_sum(self) -> float:
        return self._m2 if self._n > 0 else 0.0

    @property
    def variance(self) -> float:
        if self._n < 2: return 0.0
        return self._m2 / (self._n - 1)

    @property
    def std(self) -> float:
        return sqrt(self.variance)

    def copy(self) -> 'RunningStats':
        """Creates a copy of the RunningStats instance."""
        new_stat = RunningStats()
        new_stat._n = self._n
        new_stat._mean = self._mean
        new_stat._m2 = self._m2
        return new_stat

    def record(self, value: float):
        """Adds a new sample to the running calculation."""
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._m2 += delta * delta2
    
    def record_batch(self, values, counts):
        """
        Update the running statistics with `counts[i]` copies of `values[i]`.

        Parameters
        ----------
        values : 1‑D array‑like of constants            (e.g. waiting times 0,1,2,…)
        counts : 1‑D array‑like of non‑negative integers (how many start after that wait)

        The two arrays must have equal length.
        """
        m = counts.sum()
        if m == 0:
            return
        batch_mean = (counts*values).sum()/m
        batch_var_sum = (counts * (values - batch_mean) ** 2).sum()
        # treat the batch as another RunningStat and merge once
        tmp = RunningStats()
        tmp._n = m
        tmp._mean = batch_mean
        tmp._m2 = batch_var_sum
        self += tmp

    def record_batch(self, values, counts):
        """
        Update the running statistics with `counts[i]` copies of `values[i]`.

        Parameters
        ----------
        values : 1‑D array‑like of constants            (e.g. waiting times 0,1,2,…)
        counts : 1‑D array‑like of non‑negative integers (how many start after that wait)

        The two arrays must have equal length.
        """
        m = counts.sum()
        if m == 0:
            return
        batch_mean = (counts*values).sum()/m
        batch_var_sum = (counts * (values - batch_mean) ** 2).sum()
        # treat the batch as another RunningStat and merge once
        tmp = RunningStats()
        tmp._n = m
        tmp._mean = batch_mean
        tmp._m2 = batch_var_sum
        self += tmp

    def half_window(self, confidence):
        half = 0
        if self._n > 1:
            t_crit = np.abs(st.t.ppf((1 - confidence) / 2, self._n - 1))
            half = t_crit * self.std / np.sqrt(self._n)
        return half

    # ... (confidence_interval and __repr__ remain the same) ...
    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        if self._n < 2: return self.mean, self.mean
        half_window = self.half_window(confidence)
        return self.mean - half_window, self.mean + half_window

    def __repr__(self) -> str:
        return f"RunningStats(n={self.n}, mean={self.mean:.4f}, std={self.std:.4f})"

    def __iadd__(self, other):
        """Handles in-place addition (+=)."""
        if isinstance(other, numbers.Number):
            self.record(float(other))
            return self

        if not isinstance(other, RunningStats):
            return NotImplemented

        if other.n == 0: return self

        new_n = self.n + other.n
        delta = other.mean - self.mean
        self._mean = (self.mean * self.n + other.mean * other.n) / new_n
        self._m2 += other._m2 + (delta ** 2 * self.n * other.n) / new_n
        self._n = new_n
        return self

    # --- [MODIFIED] __add__ method ---
    def __add__(self, other):
        """Handles addition (+). Returns a new instance."""
        # Case 1: Add a number (returns a new instance with the number pushed)
        if isinstance(other, numbers.Number):
            new_stat = self.copy()
            new_stat.record(float(other))
            return new_stat

        # Case 2: Add another RunningStats object
        if isinstance(other, RunningStats):
            new_stat = self.copy()
            new_stat += other  # Use the efficient in-place add for merging
            return new_stat

        # If the type is unsupported
        return NotImplemented

    # ... (__str__ and confidence_interval_diff remain the same) ...
    def __str__(self) -> str:
        if self.n == 0: return "RunningStats(empty)"
        lower, upper = self.confidence_interval()
        return f"μ = {self.mean:.4f}, 95% CI = ({lower:.4f}, {upper:.4f}), n = {self.n}"

    def mean_difference(self, other, confidence):
        meanDiff = self.mean - other.mean
        sampleVar1, sampleSize1 = self.variance, self.n
        sampleVar2, sampleSize2 = other.variance, other.n
        t_crit = np.abs(st.t.ppf((1-confidence)/2, sampleSize1 + sampleSize2 - 2))
        halfWindow = t_crit * np.sqrt(sampleVar1/sampleSize1 + sampleVar2/sampleSize2)
        return meanDiff, halfWindow

if __name__ == "__main__":
    print("--- Example 1: Basic Usage ---")
    rs = RunningStats()
    rs.record_batch(np.array([1,2,3]), np.array([10, 1, 2]))
    print(rs.mean)
    print(rs.var_sum)