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
                 geom_p=0.1):
        self.mean_arrival = mean_arrival_rate
        self.maximum_arrival = maximum_arrival
        self.type_probs = np.asarray(type_probs)
        self.is_precompute_state = is_precompute_state

        self.rng = np.random.default_rng(random_seed)
        total_poisson = poisson(mean_arrival_rate)
        normalizer = total_poisson.cdf(maximum_arrival)
        self.truncate_poisson_pmf = np.array([total_poisson.pmf(i) for i in range(maximum_arrival + 1)])/normalizer
        self.cdf_N = np.cumsum(self.truncate_poisson_pmf)

        self._arrivals_with_probs = None
        if is_precompute_state:
            self._arrivals_with_probs = self._precompute_all_states()

        # Expectations must use the truncated distribution sampled by rvs().
        # mean_arrival remains the nominal (untruncated) Poisson parameter.
        support = np.arange(self.maximum_arrival + 1, dtype=float)
        self.expected_total_arrival = float(support @ self.truncate_poisson_pmf)
        self.mean_by_type = self.expected_total_arrival * self.type_probs
        self.num_types = len(self.type_probs)
        self.type_cdf = np.cumsum(self.type_probs)
        self.type_cdf[-1] = 1.0  # Ensure numerical stability
        self.max_periods = max_periods
        self.geom_p = geom_p

        self.use_qmc = use_qmc
        self.qmc_seed = random_seed
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
    
    def simulate_arrival_path(self, uniform_samples,  size, is_positive_integer_support=False):
        u_len = uniform_samples[:, 0]
        if is_positive_integer_support:
            lengths = geom.ppf(u_len, self.geom_p).astype(int)
        else:
            lengths = (geom.ppf(u_len, self.geom_p)-1).astype(int)
        lengths = np.minimum(lengths, self.max_periods)
        u_totals = uniform_samples[:, 1 : 1 + self.max_periods]
        total_counts = np.searchsorted(self.cdf_N, u_totals, side='right').astype(int)

        u_splits = uniform_samples[:, 1 + self.max_periods :].reshape(size, self.max_periods, self.num_types - 1)
        
        paths = np.zeros((size, self.max_periods, self.num_types), dtype=int)
        
        remaining_counts = total_counts.copy()
        current_prob_sum = 1.0
        
        for k in range(self.num_types - 1):
            p_cond = self.type_probs[k] / current_prob_sum
            
            u_k = u_splits[:, :, k]
            
            type_k_counts = binom.ppf(u_k, n=remaining_counts, p=p_cond).astype(int)
            
            paths[:, :, k] = type_k_counts
            remaining_counts -= type_k_counts
            current_prob_sum -= self.type_probs[k]
            
        paths[:, :, -1] = remaining_counts

        arrivals = []
        for i in range(size):
            L = lengths[i]
            path_slice = paths[i, :L, :]
            arrivals.append(path_slice)
        return arrivals
    
    def mc_rvs(self, size=1, is_positive_integer_support=False):
        u_mc = self.rng.uniform(size=(size, 1 + self.max_periods + self.max_periods * (self.num_types - 1)))
        return self.simulate_arrival_path(u_mc, size, is_positive_integer_support=is_positive_integer_support)

    def quasi_rvs(self, size=1, is_positive_integer_support=False):
        
        u_qmc = self.qmc_sampler.random(n=size)
        return self.simulate_arrival_path(u_qmc, size, is_positive_integer_support=is_positive_integer_support)

    def arrival_type_rvs(self, arrival_num, size=1):
        arrivals = self.rng.multinomial(arrival_num, self.type_probs,size=size)
        return arrivals

    def _precompute_all_states(self):
        system_dynamic = []
        I = len(self.type_probs)
        for N in range(self.maximum_arrival+1):
            for bar_positions in combinations(range(N + I - 1), I - 1):
                distribution = []
                prev_bar = -1
                for bar in bar_positions:
                    distribution.append(bar - prev_bar - 1)
                    prev_bar = bar
                distribution.append((N + I - 1) - prev_bar - 1)
                distribution = np.array(distribution)
                prob = self.truncate_poisson_pmf[N] * multinomial(N, self.type_probs).pmf(distribution)
                system_dynamic.append([prob, distribution])

        return system_dynamic

    def get_system_dynamic(self):
        if self._arrivals_with_probs == None:
            self._arrivals_with_probs = self._precompute_all_states()
        return self._arrivals_with_probs

    def get_sample_paths_with_prob(self, period_num):
        if self._arrivals_with_probs == None:
            self._arrivals_with_probs = self._precompute_all_states()
        single_period_states = self._arrivals_with_probs

        all_paths = []
        path_probs = []
        for path_states in product(single_period_states, repeat=period_num):
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


    discount_factor = 0.98
    geom_p = 1 - discount_factor
    print(geom.std(geom_p), geom.mean(geom_p))
    discount_factor_2 = 0.99
    geom_p_2 = 1 - discount_factor_2
    q1 = geom.cdf(100, geom_p_2)
    q2 = geom.cdf(200, geom_p_2)
    q3 = geom.cdf(400, geom_p_2)
    print(q1, q2, q3)
    print(geom.std(geom_p_2), geom.mean(geom_p_2))
    print(geom.ppf(q1, geom_p_2))
    print(geom.ppf(q2, geom_p_2))
    print(geom.ppf(q3, geom_p_2))
