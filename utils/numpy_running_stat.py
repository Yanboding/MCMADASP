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
        self += RunningStats(n=m, mean=batch_mean, m2=batch_var_sum)

    def half_window(self, confidence):
        half = 0
        if self._n > 1:
            t_crit = np.abs(st.t.ppf((1 - confidence) / 2, self._n - 1))
            half = t_crit * self.std / np.sqrt(self._n)
        return half

    # ... (confidence_interval and __repr__ remain the same) ...
    def confidence_interval(self, confidence: float = 0.95) -> str:
        if self._n < 2: return f"{self.mean} \pm 0.0"
        half_window = self.half_window(confidence)
        return f"{round(self.mean)} \pm {round(half_window,1)}"

    def __repr__(self) -> str:
        return f"RunningStats(n={self.n}, mean={self.mean:.4f}, std={self.std:.4f}, 95% CI={self.mean:.4f} \pm {self.half_window(0.8):.4f})"

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
        return f"RunningStats(n={self.n}, mean={self.mean:.4f}, std={self.std:.4f}, 95% CI={self.mean:.4f} \pm {self.half_window(0.95):.4f})"

    def mean_difference(self, other, confidence):
        meanDiff = self.mean - other.mean
        sampleVar1, sampleSize1 = self.variance, self.n
        sampleVar2, sampleSize2 = other.variance, other.n
        t_crit = np.abs(st.t.ppf((1-confidence)/2, sampleSize1 + sampleSize2 - 2))
        halfWindow = t_crit * np.sqrt(sampleVar1/sampleSize1 + sampleVar2/sampleSize2)
        return meanDiff, halfWindow
    
    def __truediv__(self, other):
        """
        Division operator: returns a RunningStats that represents
        the ratio of the means self/other, with variance estimated
        via the delta method.

        Notes
        -----
        This is an *approximation*: it does NOT reconstruct sample-wise
        ratios, but treats the ratio as a derived statistic with
        effective sample size = min(self.n, other.n).
        """
        if isinstance(other, numbers.Number):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            # scale mean, scale variance
            new_mean = self.mean / other
            new_var = self.variance / (other**2)
            eff_n = self.n
            new_m2 = new_var * (eff_n - 1)
            return RunningStats(n=eff_n, mean=new_mean, m2=new_m2)

        if isinstance(other, RunningStats):
            if other.mean == 0:
                raise ZeroDivisionError("Denominator mean is zero.")
            if self.n < 2 or other.n < 2:
                raise ValueError("Need at least 2 samples in both RunningStats.")

            ratio_mean = self.mean / other.mean
            var_ratio = (self.variance / self.n) / (other.mean**2) \
                      + (self.mean**2 / other.mean**4) * (other.variance / other.n)

            eff_n = min(self.n, other.n)  # conservative choice
            ratio_m2 = var_ratio * (eff_n - 1)

            return RunningStats(n=eff_n, mean=ratio_mean, m2=ratio_m2)

        return NotImplemented



if __name__ == "__main__":
    print("--- Example 1: Basic Usage ---")
    x_1 = RunningStats(n=10, mean=10, m2=300)
    print(x_1)
    x_2 = 2
    #x_2 = RunningStats(n=10, mean=20, m2=200)
    x_1 = x_1/x_2
    print(x_1)