"""Empirical E[tau] / Var[tau] comparison across sample-path-length proposals.

For a target geometric horizon with discount factor ``gamma`` the importance
sampling estimators reweight each sampled path of length ``tau`` so that the
estimate targets ``E_target[tau] = sum_t gamma^(t-1) = 1 / (1 - gamma)``. This
script reports, for every proposal:

  * ``E[tau]`` / ``Var[tau]``: the distribution of the raw sampled horizon
    length (a proxy for per-sample compute cost), and
  * the per-path IS estimator ``Z = sum_t w_t`` of ``1 / (1 - gamma)`` together
    with its mean (correctness / bias check) and variance (efficiency).

Standalone script with its own sys.path bootstrap (it does not import
``test/__init__.py``).

Run:
    PYTHONPATH=. python test/proposal_tau_experiment.py
    python test/proposal_tau_experiment.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from importance_sampling.proposals import (
    ArrivalGeneratorSamplePathProposal,
    FixedLengthProposal,
    GeometricLengthProposal,
    MixtureGeometricStratifiedQMCProposal,
    TruncatedGeometricLengthProposal,
)


class _ArrivalGeneratorStub:
    """Minimal stand-in exposing only what ``sample_lengths`` reads.

    A sampled path length depends solely on ``rng`` / ``geom_p`` / ``max_periods``,
    so this faithfully exercises each proposal's real ``sample_lengths`` code
    without constructing the full environment arrival generator (which would also
    build an unused high-dimensional Sobol sampler).
    """

    def __init__(self, seed, geom_p, max_periods):
        self.rng = np.random.default_rng(seed)
        self.geom_p = geom_p
        self.max_periods = max_periods
        self.use_qmc = False


def summarize(name, proposal, target_discount_factor, n_samples, seed, geom_p, max_periods):
    stub = _ArrivalGeneratorStub(seed=seed, geom_p=geom_p, max_periods=max_periods)
    lengths = np.asarray(proposal.sample_lengths(stub, n_samples))
    # IS estimator of E_target[tau] = sum_{t=1..tau} gamma^(t-1) per path.
    weights = proposal.period_likelihood_ratios(target_discount_factor, lengths)
    z = np.array([w.sum() for w in weights])
    return {
        'name': name,
        'e_tau': float(lengths.mean()),
        'sd_tau': float(lengths.std()),
        'var_tau': float(lengths.var()),
        'est_mean': float(z.mean()),
        'est_var': float(z.var()),
    }


def main():
    target = 0.99
    target_mean = 1.0 / (1.0 - target)  # 100.0
    geom_p = 1.0 - target               # env natural length dist == target geometric
    max_periods = 10 ** 8               # effectively no truncation for the baseline
    n_samples = 100_000
    seed = 12345

    proposals = [
        ('arrival_generator (env, no IS)',
         ArrivalGeneratorSamplePathProposal(is_positive_integer_support=True)),
        ('geometric q=0.99 (= target)', GeometricLengthProposal(0.99)),
        ('geometric q=0.98', GeometricLengthProposal(0.98)),
        ('geometric q=0.95', GeometricLengthProposal(0.95)),
        ('truncated_geometric q=0.99, L<=200', TruncatedGeometricLengthProposal(0.99, 200)),
        ('fixed L=100', FixedLengthProposal(100)),
        ('mixture q=0.9, lambda0=0.5', MixtureGeometricStratifiedQMCProposal(target, 0.9, 0.5)),
        ('mixture q=0.9, lambda0=0.3', MixtureGeometricStratifiedQMCProposal(target, 0.9, 0.3)),
    ]

    rows = [
        summarize(name, proposal, target, n_samples, seed, geom_p, max_periods)
        for name, proposal in proposals
    ]

    print(
        f'Target gamma = {target}  |  target E[tau] = 1/(1-gamma) = {target_mean:.1f}  |  '
        f'N = {n_samples:,}  |  seed = {seed}'
    )
    print()
    header = (
        f"{'Proposal':36s} {'E[tau]':>9s} {'SD[tau]':>9s} {'Var[tau]':>11s} "
        f"{'IS est':>9s} {'bias%':>8s} {'Var[sum w]':>13s}"
    )
    print(header)
    print('-' * len(header))
    for r in rows:
        bias = (r['est_mean'] - target_mean) / target_mean * 100.0
        print(
            f"{r['name']:36s} {r['e_tau']:9.2f} {r['sd_tau']:9.2f} {r['var_tau']:11.1f} "
            f"{r['est_mean']:9.2f} {bias:7.2f}% {r['est_var']:13.1f}"
        )
    print()
    print('E[tau], Var[tau]: raw sampled horizon length (compute-cost proxy).')
    print('IS est: per-path estimate of E_target[tau] = 1/(1-gamma) = 100; bias% vs 100 = correctness.')
    print('Var[sum w]: per-path variance of the IS estimator (efficiency; lower is better).')


if __name__ == '__main__':
    main()
