from collections import defaultdict
from pprint import pprint

import numpy as np
from itertools import combinations, product
from scipy.stats import qmc, poisson, multinomial, geom, binom

class MultiClassPoissonArrivalGenerator:

    def __init__(self, 
                 mean_arrival_rate, 
                 maximum_arrival, 
                 type_probs, 
                 random_seed=42, 
                 is_precompute_state=False, 
                 use_qmc=True, 
                 max_periods=100, 
                 geom_p=0.1,
                 qmc_seed=123):
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
        self.max_periods = max_periods
        self.geom_p = geom_p

        self.use_qmc = use_qmc
        self.qmc_seed = qmc_seed
        if use_qmc:
            self.qmc_dim = 1 + (self.max_periods * self.num_types)
            # Note: Optimization=True is slower to init but better quality
            self.qmc_sampler = qmc.Sobol(d=self.qmc_dim, scramble=True, seed=self.qmc_seed)
    
    def set_max_periods(self, max_periods):
        self.max_periods = max_periods
        if self.use_qmc:
            self.qmc_dim = 1 + (self.max_periods * self.num_types)
            self.qmc_sampler = qmc.Sobol(d=self.qmc_dim, scramble=True, seed=self.qmc_seed)
    
    def set_geom_p(self, geom_p):
        self.geom_p = geom_p

    def rvs(self, size=1):
        N_values = self.rng.choice(
            self.maximum_arrival + 1, size=size, p=self.truncate_poisson_pmf
        )
        arrivals = np.array([
            self.rng.multinomial(N, self.type_probs) for N in N_values
        ])
        return arrivals
    
    def simulate_arrival_path(self, uniform_samples,  size):
        """Simulate a single arrival path given uniform samples.
        uniform_samples: shape (1 + max_periods + max_periods * (num_types - 1),)
        0: stop time
        1:max_periods: determine total N in each period
        max_periods+1:: determine types for each period
        total dim 1 + max_periods * num_types
        """
        # --- PHASE A: Determine Path Lengths ---
        # Dimension 0 is reserved for length
        u_len = uniform_samples[:, 0]
        # Inverse Transform on Geometric Distribution
        # We clip at max_horizon to prevent array out-of-bounds
        lengths = (geom.ppf(u_len, self.geom_p)-1).astype(int)
        lengths = np.minimum(lengths, self.max_periods)
        # --- PHASE B: Determine Total Arrivals Per Period ---
        # Dimensions 1 to 1 + max_horizon
        u_totals = uniform_samples[:, 1 : 1 + self.max_periods]
        # Inverse Transform on Truncated Poisson
        # np.searchsorted acts as the PPF for the discrete distribution defined by self.cdf_N
        total_counts = np.searchsorted(self.cdf_N, u_totals, side='right').astype(int)

        # --- PHASE C: Hierarchical Split into Types ---
        # Dimensions (1 + max_horizon) to End
        # We reshape to (max_horizon, num_types - 1)
        u_splits = uniform_samples[:, 1 + self.max_periods :].reshape(size, self.max_periods, self.num_types - 1)
        
        # Placeholder for result
        paths = np.zeros((size, self.max_periods, self.num_types), dtype=int)
        
        # Recursive Binomial Splitting
        remaining_counts = total_counts.copy()
        current_prob_sum = 1.0
        
        for k in range(self.num_types - 1):
            # 1. Calculate Conditional Probability: P(Type_k | Not Type_0...Type_k-1)
            p_cond = self.type_probs[k] / current_prob_sum
            
            # 2. Get Sobol slice for this specific type decision
            u_k = u_splits[:, :, k]
            
            # 3. Binomial Inverse Transform
            # Draw how many of 'remaining_counts' belong to type k
            # scipy.stats.binom.ppf broadcasts over n (remaining_counts) and p (p_cond)
            type_k_counts = binom.ppf(u_k, n=remaining_counts, p=p_cond).astype(int)
            
            # 4. Store and Update
            paths[:, :, k] = type_k_counts
            remaining_counts -= type_k_counts
            current_prob_sum -= self.type_probs[k]
            
        # Assign remainder to the last type
        paths[:, :, -1] = remaining_counts

        # --- PHASE D: Truncation (Slicing) ---
        # Instead of masking with zeros, we slice the arrays to their actual geometric length.
        arrivals = []
        for i in range(size):
            L = lengths[i]
            # Slice the i-th path from 0 to L
            # The remaining rows (L to max_horizon) are discarded
            path_slice = paths[i, :L, :]
            arrivals.append(path_slice)
        return arrivals
    
    def mc_rvs(self, size=1):
        """Standard Monte Carlo generation using PRNG for both N and the multinomial split.
        size: sample of paths
        0: stop time
        1:max_periods: determine total N in each period
        max_periods+1:: determine types for each period
        total dim 1 + max_periods * num_types
        """
        # 1. Generate Uniform PRNG samples
        u_mc = self.rng.uniform(size=(size, 1 + self.max_periods + self.max_periods * (self.num_types - 1)))
        return self.simulate_arrival_path(u_mc, size)

    def quasi_rvs(self, size=1):
        """Quasi-Monte Carlo (QMC) generation using Sobol sequence for both N and the multinomial split.
        size: sample of paths
        0: stop time
        1:max_periods: determine total N in each period
        max_periods+1:: determine types for each period
        total dim 1 + max_periods * num_types
        """
        
        # 1. Generate Uniform QMC samples
        # Get 'size' points drawn from the d-dimensional hypercube [0,1)^d
        # Shape: (size, 1 + maximum_arrival)
        u_qmc = self.qmc_sampler.random(n=size)
        return self.simulate_arrival_path(u_qmc, size)

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
        max_periods=100,
        qmc_seed=888
    )

    print(geom.ppf(0.8, 0.05))
    print(geom.cdf(100, p=0.05))
