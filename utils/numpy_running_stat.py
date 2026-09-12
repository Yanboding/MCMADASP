import numpy as np
import numbers
from math import sqrt
import scipy.stats as st


class RunningStats:

    def __init__(self, n=0, mean=0.0, m2=0.0):
        self.n = n
        self.mean = mean
        self.m2 = m2

    def var_sum(self):
        return self.m2 if self.n > 0 else 0.0

    def variance(self):
        if self.n < 2: return 0.0
        return self.m2 / (self.n - 1)

    def std(self):
        return sqrt(self.variance())

    def copy(self) -> 'RunningStats':
        new_stat = RunningStats()
        new_stat.n = self.n
        new_stat.mean = self.mean
        new_stat.m2 = self.m2
        return new_stat

    def record(self, value: float):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def record_batch(self, values, counts):
        m = counts.sum()
        if m == 0:
            return
        batch_mean = (counts*values).sum()/m
        batch_var_sum = (counts * (values - batch_mean) ** 2).sum()
        self += RunningStats(n=m, mean=batch_mean, m2=batch_var_sum)

    def half_window(self, confidence):
        half = 0
        if self.n > 1:
            t_crit = np.abs(st.t.ppf((1 - confidence) / 2, self.n - 1))
            half = t_crit * self.std() / np.sqrt(self.n)
        return half

    def confidence_interval(self, confidence: float = 0.95) -> str:
        if self.n < 2: return f"{self.mean} \pm 0.0"
        half_window = self.half_window(confidence)
        return f"{round(self.mean)} \pm {round(half_window,1)}"

    def __repr__(self) -> str:
        return f"RunningStats(n={self.n}, mean={self.mean:.4f}, std={self.std():.4f}, 95% CI={self.mean:.4f} \pm {self.half_window(0.95):.4f})"

    def __iadd__(self, other):
        if isinstance(other, numbers.Number):
            self.record(float(other))
            return self

        if not isinstance(other, RunningStats):
            return NotImplemented

        if other.n == 0: return self

        new_n = self.n + other.n
        delta = other.mean - self.mean
        self.mean = (self.mean * self.n + other.mean * other.n) / new_n
        self.m2 += other.m2 + (delta ** 2 * self.n * other.n) / new_n
        self.n = new_n
        return self

    def __add__(self, other):
        if isinstance(other, numbers.Number):
            new_stat = self.copy()
            new_stat.record(float(other))
            return new_stat

        if isinstance(other, RunningStats):
            new_stat = self.copy()
            new_stat += other
            return new_stat

        return NotImplemented

    def __str__(self) -> str:
        if self.n == 0: return "RunningStats(empty)"
        lower, upper = self.confidence_interval()
        return f"RunningStats(n={self.n}, mean={self.mean:.4f}, std={self.std():.4f}, 95% CI={self.mean:.4f} \pm {self.half_window(0.95):.4f})"

    def mean_difference(self, other, confidence):
        meanDiff = self.mean - other.mean
        sampleVar1, sampleSize1 = self.variance(), self.n
        sampleVar2, sampleSize2 = other.variance(), other.n
        t_crit = np.abs(st.t.ppf((1-confidence)/2, sampleSize1 + sampleSize2 - 2))
        halfWindow = t_crit * np.sqrt(sampleVar1/sampleSize1 + sampleVar2/sampleSize2)
        return meanDiff, halfWindow
    
    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            if other == 0:
                raise ZeroDivisionError("Cannot divide by zero.")
            new_mean = self.mean / other
            new_var = self.variance() / (other**2)
            eff_n = self.n
            new_m2 = new_var * (eff_n - 1)
            return RunningStats(n=eff_n, mean=new_mean, m2=new_m2)

        if isinstance(other, RunningStats):
            if other.mean == 0:
                raise ZeroDivisionError("Denominator mean is zero.")
            if self.n < 2 or other.n < 2:
                raise ValueError("Need at least 2 samples in both RunningStats.")

            ratio_mean = self.mean / other.mean
            var_ratio = (self.variance() / self.n) / (other.mean**2) \
                      + (self.mean**2 / other.mean**4) * (other.variance() / other.n)

            eff_n = min(self.n, other.n)
            ratio_m2 = var_ratio * (eff_n - 1)

            return RunningStats(n=eff_n, mean=ratio_mean, m2=ratio_m2)

        return NotImplemented


if __name__ == "__main__":
    print("--- Example 1: Basic Usage ---")
    x_1 = RunningStats(n=10, mean=10, m2=300)
    print(x_1)
    x_2 = 2
    x_1 = x_1/x_2
    print(x_1)
