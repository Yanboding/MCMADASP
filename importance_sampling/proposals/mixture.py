import warnings

import numpy as np
from scipy.stats import geom, qmc

from .base import SamplePathLengthProposal, _validate_discount_factor


class MixtureGeometricStratifiedQMCProposal(SamplePathLengthProposal):

    def __init__(
        self,
        target_discount_factor: float,
        discount_factor_proposal: float,
        lambda_0: float = 0.5,
    ):
        _validate_discount_factor(target_discount_factor, 'target_discount_factor')
        _validate_discount_factor(discount_factor_proposal, 'discount_factor_proposal')
        if not (discount_factor_proposal < target_discount_factor):
            raise ValueError(
                "discount_factor_proposal (q) must be strictly less than "
                f"target_discount_factor (gamma): got q={discount_factor_proposal}, "
                f"gamma={target_discount_factor}."
            )
        if not (0 < lambda_0 <= 1):
            raise ValueError(
                f"lambda_0 (mass on the long component) must be in (0, 1]. "
                f"Got {lambda_0}."
            )
        self.target_discount_factor = target_discount_factor
        self.discount_factor_proposal = discount_factor_proposal
        self.lambda_0 = lambda_0

    def _component_sizes(self, size):
        size = int(size)
        n_long = int(round(self.lambda_0 * size))
        return n_long, size - n_long

    def _qmc_geometric_lengths(self, discount_factor, size, seed):
        if size <= 0:
            return np.empty(0, dtype=int)
        with warnings.catch_warnings():
            # Sobol' warns when ``size`` is not a power of two; the draw is still
            # valid, we just lose the exact balance guarantee.
            warnings.simplefilter('ignore')
            u = qmc.Sobol(d=1, scramble=True, seed=seed).random(n=size)[:, 0]
        # geom.ppf(0) returns 0 (one below support) and geom.ppf(1) -> inf, so
        # keep the quantiles strictly inside (0, 1).
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        return np.asarray(geom.ppf(u, 1.0 - discount_factor)).astype(int)

    def sample_lengths(self, arrival_generator, size):
        size = int(size)
        if size <= 0:
            return np.empty(0, dtype=int)
        n_long, n_short = self._component_sizes(size)
        seed_long, seed_short = (
            int(s) for s in arrival_generator.rng.integers(0, 2 ** 32 - 1, size=2)
        )
        lengths_long = self._qmc_geometric_lengths(self.target_discount_factor, n_long, seed_long)
        lengths_short = self._qmc_geometric_lengths(self.discount_factor_proposal, n_short, seed_short)
        return np.concatenate([lengths_long, lengths_short]).astype(int)

    def path_weights(self, size):
        n_long, n_short = self._component_sizes(size)
        if (self.lambda_0 > 0 and n_long == 0) or (self.lambda_0 < 1 and n_short == 0):
            raise ValueError(
                f"Degenerate stratified allocation for size={size}, "
                f"lambda_0={self.lambda_0}: n_long={n_long}, n_short={n_short}.")
        parts = []
        if n_long:
            parts.append(np.full(n_long, self.lambda_0 / n_long))
        if n_short:
            parts.append(np.full(n_short, (1.0 - self.lambda_0) / n_short))
        return np.concatenate(parts)

    def path_strata(self, size):
        n_long, n_short = self._component_sizes(size)
        return np.concatenate(
            [np.zeros(n_long, dtype=int), np.ones(n_short, dtype=int)])

    def survival_probability(self, periods):
        periods = np.asarray(periods, dtype=int)
        lambda_1 = 1.0 - self.lambda_0
        return (
            self.lambda_0 * self.target_discount_factor ** (periods - 1)
            + lambda_1 * self.discount_factor_proposal ** (periods - 1)
        )

    def period_likelihood_ratios(self, target_discount_factor, lengths):
        if not np.isclose(target_discount_factor, self.target_discount_factor):
            raise ValueError(
                "target_discount_factor mismatch: the mixture proposal was built "
                f"for gamma={self.target_discount_factor} but the agent passed "
                f"{target_discount_factor}. Build the proposal with "
                "target_discount_factor equal to the environment discount factor."
            )
        return super().period_likelihood_ratios(target_discount_factor, lengths)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from environment.arrival_generator import MultiClassPoissonArrivalGenerator
    
    target_discount_factor = 0.99
    discount_factor_proposal = 0.95
    lambda_0 = 0.1
    proposal = MixtureGeometricStratifiedQMCProposal(
        target_discount_factor, discount_factor_proposal, lambda_0
    )
    generator_qmc = MultiClassPoissonArrivalGenerator(
        mean_arrival_rate=3,
        maximum_arrival=9,
        type_probs=[0.5, 0.3, 0.2],
        random_seed=42,
        is_precompute_state=False,
        use_qmc=True,
        max_periods=int(geom.ppf(0.9999, p=0.01)),
        geom_p=0.01
    )
    n_draws = 100
    lengths = proposal.sample_lengths(arrival_generator=generator_qmc, size=n_draws)
    print(lengths)
    max_length = np.max(lengths)
    empirical_survival = np.array(
        [np.mean(lengths >= t) for t in range(1, max_length + 1)]
    )
    theoretical_survival = proposal.survival_probability(np.arange(1, max_length + 1))
    plt.step(np.arange(1, max_length + 1), empirical_survival, label='Empirical')
    plt.step(np.arange(1, max_length + 1), theoretical_survival, label='Theoretical')
    plt.xlabel('Sample Path Length')
    plt.ylabel('Survival Probability')
    plt.title('Mixture Geometric Proposal Survival Function')
    plt.legend()
    plt.show()
