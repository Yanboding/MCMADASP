import numpy as np

from importance_sampling.sample_path import Terminal


class PenaltyForm:
    hindsight_scenario_weights = 'uniform'

    def period_weights(self, path, gamma):
        raise NotImplementedError

    def term_weights(self, s, tau, W, gamma, terminal):
        raise NotImplementedError

    def combine(self, path, gamma, expected_terms, realized_terms):
        tau = path.periods
        if len(expected_terms) != tau or len(realized_terms) != tau - 1:
            raise ValueError(
                f"expected {tau} expected terms and {tau - 1} realized terms; got "
                f"{len(expected_terms)} and {len(realized_terms)}")
        W = self.period_weights(path, gamma)
        total = 0.0
        for s in range(tau):
            expected_weight, realized_weight = self.term_weights(s, tau, W, gamma, path.terminal)
            total += expected_weight * expected_terms[s]
            if s < tau - 1:
                total -= realized_weight * realized_terms[s]
        return float(total)


class AbsorptionForm(PenaltyForm):
    hindsight_scenario_weights = 'kappa'

    def period_weights(self, path, gamma):
        if path.survival_weights is None or path.terminal is Terminal.UNSPECIFIED:
            raise ValueError(
                "the absorption penalty needs survival weights and a terminal outcome on every "
                "path: draw the paths from a length proposal (training) or generate the "
                "evaluation records with --eval-proposal")
        return np.asarray(path.survival_weights, dtype=float)

    def term_weights(self, s, tau, W, gamma, terminal):
        if s < tau - 1:
            return gamma * W[s], W[s + 1]
        if terminal is Terminal.ABSORBED:
            return gamma * W[s], 0.0
        return 0.0, 0.0


class LegacyForm(PenaltyForm):
    MODES = ('training', 'hindsight', 'evaluation')

    def __init__(self, mode):
        if mode not in self.MODES:
            raise ValueError(f"unknown legacy mode {mode!r}; use one of {self.MODES}")
        self.mode = mode

    def period_weights(self, path, gamma):
        if self.mode == 'evaluation':
            if path.survival_weights is None:
                return np.ones(path.periods)
            return np.asarray(path.survival_weights, dtype=float)
        u = np.ones(path.length) if path.likelihood_ratios is None else np.asarray(path.likelihood_ratios, dtype=float)
        return np.concatenate(([1.0], u if self.mode == 'training' else gamma * u))

    def term_weights(self, s, tau, W, gamma, terminal):
        if s == tau - 1:
            return 0.0, 0.0
        k = s if self.mode == 'evaluation' else s + 1
        return W[k], W[k]

    def __repr__(self):
        return f"LegacyForm({self.mode!r})"


def legacy_forms():
    return {mode: LegacyForm(mode) for mode in LegacyForm.MODES}


def absorption_forms():
    form = AbsorptionForm()
    return {mode: form for mode in LegacyForm.MODES}
