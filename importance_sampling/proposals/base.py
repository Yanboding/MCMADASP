import numpy as np

from importance_sampling.sample_path import SamplePath, Terminal


def _validate_discount_factor(discount_factor, name):
    if discount_factor is None or not (0 <= discount_factor < 1):
        raise ValueError(f"{name} must be in [0, 1). Got {discount_factor}.")


def _validate_max_length(max_length):
    max_length = int(max_length)
    if max_length <= 0:
        raise ValueError(f"max_length must be positive. Got {max_length}.")
    return max_length


class SamplePathLengthProposal:

    def sample_lengths(self, arrival_generator, size):
        raise NotImplementedError

    def sample_arrival_paths(self, arrival_generator, size):
        lengths = self.sample_lengths(arrival_generator=arrival_generator, size=size)
        return [arrival_generator.rvs(size=int(length)) for length in lengths], lengths

    def path_weights(self, size):
        size = int(size)
        return np.full(size, 1.0 / size)

    def path_strata(self, size):
        return np.zeros(int(size), dtype=int)

    def survival_probability(self, periods):
        raise NotImplementedError

    def period_likelihood_ratios(self, target_discount_factor, lengths):
        _validate_discount_factor(target_discount_factor, 'target_discount_factor')

        def weights_for(length):
            periods = np.arange(1, int(length) + 1)
            proposal_survival = self.survival_probability(periods)
            if np.any(proposal_survival <= 0):
                raise ValueError(
                    "Proposal survival probability must be positive on every sampled period."
                )
            # One-based period indexing: P(L >= t) = gamma ** (t - 1).
            target_survival = target_discount_factor ** (periods - 1)
            return target_survival / proposal_survival

        return [weights_for(length) for length in lengths]

    def survival_weights(self, lengths, target_discount_factor, arrival_generator=None):
        ratios = self.period_likelihood_ratios(target_discount_factor, lengths)
        return [np.concatenate(([1.0], target_discount_factor * u)) for u in ratios]

    def terminal_for(self, lengths, arrival_generator=None):
        return [Terminal.ABSORBED for _ in lengths]

    def sample_paths(self, arrival_generator, size, target_discount_factor):
        arrivals, lengths = self.sample_arrival_paths(arrival_generator=arrival_generator, size=size)
        ratios = self.period_likelihood_ratios(target_discount_factor=target_discount_factor, lengths=lengths)
        weights = self.survival_weights(lengths, target_discount_factor, arrival_generator)
        terminals = self.terminal_for(lengths, arrival_generator)
        return [SamplePath(path, terminal, weight, ratio)
                for path, terminal, weight, ratio in zip(arrivals, terminals, weights, ratios)]
