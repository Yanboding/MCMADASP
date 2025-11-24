from collections import defaultdict
from pprint import pprint

import numpy as np
from itertools import combinations, product
from scipy.stats import poisson, multinomial
from scipy.stats import qmc
from scipy.stats import binom

class MultiClassPoissonArrivalGenerator:

    def __init__(self, mean_arrival_rate, maximum_arrival, type_probs, random_seed=42, is_precompute_state=False, use_qmc=True, qmc_seed=123):
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
        self.num_types = len(self.type_probs)
        self.type_cdf = np.cumsum(self.type_probs)
        self.type_cdf[-1] = 1.0  # Ensure numerical stability

        self.use_qmc = use_qmc
        if use_qmc:
            self.qmc_dim = 1 + self.maximum_arrival
            self.qmc_sampler = qmc.Sobol(d=self.qmc_dim, scramble=True, seed=qmc_seed)


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
        
        # 1. Generate Uniform QMC samples
        # Get 'size' points drawn from the d-dimensional hypercube [0,1)^d
        # Shape: (size, 1 + maximum_arrival)
        u_qmc = self.qmc_sampler.random(n=size)

        # Split dimensions:
        # Dimension 0 determines total N.
        # Dimensions 1 onwards determine types.
        u_N = u_qmc[:, 0]
        u_types_pool = u_qmc[:, 1:]

        # 2. Sample Total Arrivals (N)
        # Use Inverse Transform Sampling on the truncated Poisson CDF.
        # np.searchsorted finds indices where elements should be inserted to maintain order,
        # effectively mapping uniform draws to discrete outcomes based on CDF buckets.
        # side='right' ensures u=0 maps to index 0.
        N_values = np.searchsorted(self.cdf_N, u_N, side='right')
        # 3. Sample Types given N
        # Initialize result array (size x num_types)
        arrivals = np.zeros((size, self.num_types), dtype=int)

        for i in range(size):
            N = N_values[i]
            if N == 0:
                continue

            # We need to determine types for N arrivals.
            # We take the first N uniform variables available in the pool for this specific path 'i'.
            # The remaining (maximum_arrival - N) variables in this row are unused.
            current_path_u_types = u_types_pool[i, :N]
            # Inverse Transform Sampling for Categorical/Multinomial.
            # Map uniform draws to type indices [0, num_types-1] based on type CDF boundaries.
            type_indices = np.searchsorted(self.type_cdf, current_path_u_types, side='right')

            # Count occurrences of each type index.
            # minlength ensures the output has length 'num_types' even if some types aren't drawn.
            counts = np.bincount(type_indices, minlength=self.num_types)
            arrivals[i, :] = counts

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

# def verify_qmc_implementation():
#     print("Starting QMC Verification...")
#     # --- Setup ---
#     mean_rate = 5.0
#     max_arr = 15 # Increased slightly to see tails better
#     probs = [0.5, 0.3, 0.2] # 3 types
#     num_types = len(probs)
#     N_large = 200000 # Large sample size for accurate comparison

#     # Initialize generator
#     gen = MultiClassPoissonArrivalGenerator(
#         mean_arrival_rate=mean_rate,
#         maximum_arrival=max_arr,
#         type_probs=probs,
#         use_qmc=True,
#         qmc_seed=42,
#         random_seed=42 # Ensure PRNG baseline is reproducible
#     )

#     print(f"Generating {N_large} PRNG baseline samples...")
#     prng_samples = gen.rvs(N_large)

#     print(f"Generating {N_large} QMC samples...")
#     # Important: Reset QMC generator to ensured scrambled sequence starts fresh
#     gen.qmc_sampler.reset() 
#     qmc_samples = gen.quasi_rvs(N_large)

#     # --- Test 1: Moment Matching (Means and Covariance) ---
#     print("\n--- Moment Matching Verification ---")
    
#     # Calculate means
#     prng_means = np.mean(prng_samples, axis=0)
#     qmc_means = np.mean(qmc_samples, axis=0)
#     mean_diff = np.abs(prng_means - qmc_means)

#     print("Sample Means (Type 0, Type 1, Type 2):")
#     print(f"PRNG: {prng_means}")
#     print(f"QMC:  {qmc_means}")
#     print(f"Max Absolute Mean Difference: {np.max(mean_diff):.6f}")

#     # Calculate Covariance Matrices (to check correlations between types)
#     prng_cov = np.cov(prng_samples, rowvar=False)
#     qmc_cov = np.cov(qmc_samples, rowvar=False)
#     cov_diff = np.abs(prng_cov - qmc_cov)

#     print("\nSample Covariance Matrix (PRNG):")
#     print(np.round(prng_cov, 4))
#     print("\nSample Covariance Matrix (QMC):")
#     print(np.round(qmc_cov, 4))
#     print(f"Max Absolute Covariance Difference: {np.max(cov_diff):.6f}")

#     # Success criteria for moments at N=200k
#     if np.max(mean_diff) < 0.01 and np.max(cov_diff) < 0.01:
#         print("\n[SUCCESS] Moments match closely.")
#     else:
#         print("\n[WARNING] Moments show larger discrepancies than expected.")
#     print("Verification complete. Check plots.")
if __name__ == "__main__":
    # Parameters
     # Parameters

    mean_rate = 5.0
    max_arr = 10
    probs = [0.5, 0.3, 0.2] # 3 types


    # Initialize with QMC enabled
    generator_qmc = MultiClassPoissonArrivalGenerator(
        mean_arrival_rate=mean_rate,
        maximum_arrival=max_arr,
        type_probs=probs,
        random_seed=42,
        is_precompute_state=False,
        use_qmc=True,
        qmc_seed=888
    )

    print(generator_qmc.quasi_rvs(20)) 
