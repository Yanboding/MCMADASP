from collections import defaultdict
from pprint import pprint

import numpy as np
from itertools import combinations, product
from scipy.stats import poisson, multinomial


class MultiClassPoissonArrivalGenerator:

    def __init__(self, mean_arrival_rate, maximum_arrival, type_probs, random_seed1=42):
        self.mean_arrival = mean_arrival_rate
        self.maximum_arrival = maximum_arrival
        self.type_probs = np.asarray(type_probs)

        self.rng1 = np.random.default_rng(random_seed1)
        # Precompute total-poisson cdf for normalization
        total_poisson = poisson(mean_arrival_rate)
        normalizer = total_poisson.cdf(maximum_arrival)
        self.truncate_poisson_pmf = np.array([total_poisson.pmf(i) for i in range(maximum_arrival + 1)])/normalizer

        # Precompute all states + probabilities for get_system_dynamic.
        self._arrivals_with_probs = self._precompute_all_states()

    def rvs(self, size=1):
        N_values = self.rng1.choice(
            self.maximum_arrival + 1, size=size, p=self.truncate_poisson_pmf
        )
        arrivals = np.array([
            self.rng1.multinomial(N, self.type_probs) for N in N_values
        ])
        return arrivals

    def _precompute_all_states(self):
        """
        Enumerate all possible arrival vectors (x_1, ..., x_I) and their
        associated probability. Returns a list of (prob, counts_vector).
        """
        system_dynamic = []
        I = len(self.type_probs)
        # For each possible total arrival from 0..maximum_arrival
        for N in range(self.maximum_arrival+1):
            for bar_positions in combinations(range(N + I - 1), I - 1):
                # Decode those bar positions into an actual distribution.
                distribution = []
                prev_bar = -1
                for bar in bar_positions:
                    distribution.append(bar - prev_bar - 1)
                    prev_bar = bar
                # For the last bin, go until the end (N + I - 1).
                distribution.append((N + I - 1) - prev_bar - 1)
                distribution = np.array(distribution)
                prob = self.truncate_poisson_pmf[N] * multinomial(N, self.type_probs).pmf(distribution)
                system_dynamic.append([prob, distribution])

        return system_dynamic

    def get_system_dynamic(self):
        """
        Return the precomputed list of (prob, arrival_vector).
        """
        return self._arrivals_with_probs

    def get_sample_paths_with_prob(self, period_num):
        """
            Enumerate all possible sample paths of length `period_num` (i.e., sequences of arrival vectors),
            along with their probabilities.

            Returns:
                A list of tuples: [(path_probability, [arrival_vector_t0, arrival_vector_t1, ...]), ...].
            """
        # Pre-fetched single-period states: each entry is (prob, arrival_vector).
        single_period_states = self._arrivals_with_probs

        # Cartesian product to get all paths of length `period_num`.
        all_paths = []
        path_probs = []
        for path_states in product(single_period_states, repeat=period_num):
            # path_states is a tuple of ( (prob1, vec1), (prob2, vec2), ... ) of length `period_num`.
            path_prob = 1.0
            zero_holder = [0] * len(self.type_probs)
            path_vecs = [zero_holder]
            for (p, vec) in path_states:
                path_prob *= p
                path_vecs.append(vec)
            all_paths.append(path_vecs)
            path_probs.append(path_prob)
        return np.array(path_probs), np.array(all_paths)


if __name__ == "__main__":
    from utils import iter_to_tuple
    class_number = 2
    probability = 1 / class_number
    mcag = MultiClassPoissonArrivalGenerator(3, 4, [probability] * class_number, 42)
    P, delta = mcag.get_sample_paths_with_prob(4)
    print(len(P))
