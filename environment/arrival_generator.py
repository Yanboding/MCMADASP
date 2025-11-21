from collections import defaultdict
from pprint import pprint

import numpy as np
from itertools import combinations, product
from scipy.stats import poisson, multinomial
from scipy.stats import qmc


class MultiClassPoissonArrivalGenerator:

    def __init__(self, mean_arrival_rate, maximum_arrival, type_probs, random_seed=42, is_precompute_state=False, use_qmc=False, qmc_seed=123):
        self.mean_arrival = mean_arrival_rate
        self.maximum_arrival = maximum_arrival
        self.type_probs = np.asarray(type_probs)
        self.is_precompute_state = is_precompute_state

        self.rng = np.random.default_rng(random_seed)
        # Precompute total-poisson cdf for normalization
        total_poisson = poisson(mean_arrival_rate)
        normalizer = total_poisson.cdf(maximum_arrival)
        self.truncate_poisson_pmf = np.array([total_poisson.pmf(i) for i in range(maximum_arrival + 1)])/normalizer
        # Precompute the CDF for N selection (Inverse Transform Sampling)
        self.cdf_N = np.cumsum(self.truncate_poisson_pmf)

        # Precompute all states + probabilities for get_system_dynamic.
        self._arrivals_with_probs = None
        if is_precompute_state:
            self._arrivals_with_probs = self._precompute_all_states()

        self.mean_by_type = self.mean_arrival * self.type_probs

        self.use_qmc = use_qmc
        if use_qmc:
            self.qmc_dim = 1 + self.maximum_arrival
            self.qmc_sampler = qmc.Sobol(d=self.qmc_dim, scramble=True, seed=qmc_seed)
            self.qmc_engine = qmc.Sobol(d=2, seed=random_seed)


    def rvs(self, size=1):
        N_values = self.rng.choice(
            self.maximum_arrival + 1, size=size, p=self.truncate_poisson_pmf
        )
        arrivals = np.array([
            self.rng.multinomial(N, self.type_probs) for N in N_values
        ])
        return arrivals
    
    def quasi_rvs(self, size=1):
        """Quasi-Monte Carlo (QMC) generation using Sobol sequence for both N and the multinomial split."""
        
        # We need a QMC point for each sample, dimension 2 (u1 for N, u2 for multinomial)
        qmc_points = self.qmc_engine.random(size)
        u1_values = qmc_points[:, 0]
        u2_values = qmc_points[:, 1]
        
        arrivals = np.zeros((size, len(self.type_probs)), dtype=int)
        
        # Pre-calculated probabilities for the conditional sampling
        I = len(self.type_probs)
        p_remaining = np.copy(self.type_probs) # Copy for mutable updates
        p_total = 1.0
        
        for k in range(size):
            u1 = u1_values[k]
            u2 = u2_values[k]
            
            # 1. Inverse Transform Sampling for N (Total Arrivals) using u1
            N = np.searchsorted(self.cdf_N, u1)
            N_rem = N
            
            # 2. Sequential/Conditional Sampling for the arrival vector using u2
            
            # Reset remaining probability variables for the new sample
            p_remaining[:] = self.type_probs
            p_total = 1.0

            for i in range(I - 1): # We determine I-1 components
                # Stop if no items remain
                if N_rem == 0:
                    break

                # The conditional probability for type i
                p_conditional = p_remaining[i] / p_total
                
                # The count x_i follows Binomial(N_rem, p_conditional)
                binomial_pmf = binom.pmf(np.arange(N_rem + 1), N_rem, p_conditional)
                binomial_cdf = np.cumsum(binomial_pmf)
                
                # Map the single QMC value u2 to the Binomial CDF
                x_i = np.searchsorted(binomial_cdf, u2)
                
                arrivals[k, i] = x_i
                
                # Update remaining counts and probabilities
                N_rem -= x_i
                p_total -= p_remaining[i]
                
            # The last component is determined by the remainder
            arrivals[k, I - 1] = N_rem

        return arrivals

    def arrival_type_rvs(self, arrival_num, size=1):
        arrivals = self.rng.multinomial(arrival_num, self.type_probs,size=size)
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
        if self._arrivals_with_probs == None:
            self._arrivals_with_probs = self._precompute_all_states()
        return self._arrivals_with_probs

    def get_sample_paths_with_prob(self, period_num):
        """
            Enumerate all possible sample paths of length `period_num` (i.e., sequences of arrival vectors),
            along with their probabilities.

            Returns:
                A list of tuples: [(path_probability, [arrival_vector_t0, arrival_vector_t1, ...]), ...].
            """
        # Pre-fetched single-period states: each entry is (prob, arrival_vector).
        if self._arrivals_with_probs == None:
            self._arrivals_with_probs = self._precompute_all_states()
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

    print(mcag.type_probs * mcag.mean_arrival)
