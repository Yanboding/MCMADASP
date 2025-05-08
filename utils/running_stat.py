import numpy as np
import scipy.stats as st


class RunningStat:

    def __init__(self, shape):
        self.shape = shape
        self.expect = np.zeros(shape)
        self.varSum = np.zeros(shape)
        self.count = 0

    def record(self, sample):
        self.count += 1
        diff = sample - self.expect
        self.expect += diff/self.count
        self.varSum += diff * (sample - self.expect)

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
        tmp = RunningStat((1,))
        tmp.count = m
        tmp.expect = np.array([batch_mean])
        tmp.varSum = np.array([batch_var_sum])
        self.merge(tmp)

    def mean(self):
        return self.expect

    def sample_variance(self):
        sampleVariance = np.zeros(self.shape)
        if self.count > 1:
            sampleVariance = self.varSum/(self.count-1)
        return sampleVariance

    def half_window(self, confidence):
        halfWindow = np.zeros(self.shape)
        if self.count > 1:
            sampleVariance = self.varSum/(self.count-1)
            std = np.sqrt(sampleVariance)
            t_crit = np.abs(st.t.ppf((1-confidence)/2, self.count-1))
            halfWindow = t_crit * std/np.sqrt(self.count)
        return halfWindow

    def mean_difference(self, other, confidence):
        meanDiff = self.expect - other.expect
        sampleVar1, sampleSize1 = self.sample_variance(), self.sample_size()
        sampleVar2, sampleSize2 = other.sample_variance(), other.sample_size()
        t_crit = np.abs(st.t.ppf((1-confidence)/2, sampleSize1 + sampleSize2 - 2))
        halfWindow = t_crit * np.sqrt(sampleVar1/sampleSize1 + sampleVar2/sampleSize2)
        return meanDiff, halfWindow

    def sample_size(self):
        return self.count

    def std(self):
        return np.sqrt(self.sample_variance())

    def merge(self, other):
        # --- nothing to add ----------------------------------------------------
        if other.count == 0:
            return

        # --- this instance is empty: adopt other's summary --------------------
        if self.count == 0:
            self.count = other.count
            self.expect = other.expect.copy()
            self.varSum = other.varSum.copy()
            return
        total_count = self.count + other.count
        new_expected = (self.expect * self.count + other.expect * other.count) / total_count
        delta = other.expect - self.expect
        new_varSum = self.varSum + other.varSum + (self.count * other.count / total_count) * (delta ** 2)
        self.count = total_count
        self.expect = new_expected
        self.varSum = new_varSum

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    schedule1 = np.array([
        [3, 2, 0],  # start today
        [1, 0, 2],  # wait 1 day
        [0, 1, 0],  # wait 2 days
        [0, 0, 1]  # wait 3 days
    ])

    schedule2 = np.array([
        [2, 1, 1],
        [0, 2, 0],
        [1, 0, 0]
    ])

    schedule3 = np.array([
        [4, 3, 2],
        [0, 0, 1],
    ])
    schedules = [schedule1, schedule2, schedule3]

    # Build stats
    num_types = schedules[0].shape[1]
    stats = {j: RunningStat(()) for j in range(num_types)}
    for S in schedules:
        wait_times = np.arange(S.shape[0])
        for j in range(num_types):
            stats[j].record_batch(wait_times, S[:, j])

    # Extract data for plotting
    types = np.arange(num_types)
    means = [stats[j].mean() for j in types]
    ci_half = [stats[j].half_window(0.95) for j in types]

    # ------------------------------------------------------------------
    # Plot (one chart, default style)
    plt.figure(figsize=(6, 4))
    plt.errorbar(types, means, yerr=ci_half, fmt='o', capsize=5, linewidth=1.5)
    for x, y in zip(types, means):
        plt.text(x, y + 0.05, f"{y:.2f}", ha='center', va='bottom', fontsize=9)
    plt.xticks(types)
    plt.xlabel("Treatment type")
    plt.ylabel("Waiting time (days)")
    plt.title("Mean waiting time with 95% CI (demo data)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

