import warnings

import numpy as np
from scipy.stats import geom, qmc

from .base import SamplePathLengthProposal, _validate_discount_factor


class MixtureGeometricStratifiedQMCProposal(SamplePathLengthProposal):
    """Two-component geometric mixture proposal for sample-path lengths.

    Lengths are drawn from a mixture of two geometric distributions on
    ``{1, 2, ...}``:

      * a *long* component matching the target discount factor ``gamma``
        (survival ``gamma ** (t - 1)``), carrying mass ``lambda_0``;
      * a *short* component using a smaller proposal discount factor
        ``q < gamma`` (survival ``q ** (t - 1)``), carrying mass
        ``1 - lambda_0``.

    The mixture survival is therefore

        S_prop(t) = lambda_0 * gamma ** (t - 1) + (1 - lambda_0) * q ** (t - 1),

    and the per-period importance weight (computed by the base class) is

        w_t = gamma ** (t - 1) / S_prop(t),

    which is monotone increasing and bounded in ``[1, 1 / lambda_0]``. The short
    component concentrates draws on cheap short horizons while the long
    component preserves the target tail, so the weights never blow up and the
    estimator stays unbiased for the discounted infinite-horizon sum.

    ``lambda_0`` is the cost-vs-variance knob: smaller ``lambda_0`` shifts more
    mass to the cheap short paths but raises the weight ceiling ``1 / lambda_0``;
    ``lambda_0 == 1`` degenerates to a pure ``Geom(gamma)`` proposal.

    Sampling stratifies the mixture component (a deterministic ``lambda_0``
    fraction of the paths is assigned to the long component) and inverts each
    component's geometric CDF on a scrambled 1-D Sobol' sequence.
    """

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

    # -- sampling ---------------------------------------------------------
    @staticmethod
    def _qmc_geometric_lengths(discount_factor, size, seed):
        """Invert a Geom(1 - discount_factor) CDF on a scrambled Sobol' sequence."""
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
        # Stratify the mixture component: a deterministic ``lambda_0`` fraction of
        # the paths is assigned to the long (gamma) component and the remainder to
        # the short (q) component, removing the component-indicator Monte-Carlo
        # noise. Seeds are drawn from the shared RNG so the draw is reproducible
        # yet advances (independent) across calls.
        n_long = int(round(self.lambda_0 * size))
        n_short = size - n_long
        seed_long, seed_short = (
            int(s) for s in arrival_generator.rng.integers(0, 2 ** 32 - 1, size=2)
        )
        lengths_long = self._qmc_geometric_lengths(self.target_discount_factor, n_long, seed_long)
        lengths_short = self._qmc_geometric_lengths(self.discount_factor_proposal, n_short, seed_short)
        return np.concatenate([lengths_long, lengths_short]).astype(int)

    # -- weighting --------------------------------------------------------
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
