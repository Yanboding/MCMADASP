"""Unit tests for per-path stratum weights and the stratified CI helper.

Run from the repo root:  python -m test.test_stratified_path_weights
"""
import numpy as np

from importance_sampling.proposals.arrival_generator_proposal import (
    ArrivalGeneratorSamplePathProposal,
)
from importance_sampling.proposals.mixture import (
    MixtureGeometricStratifiedQMCProposal,
)


def test_base_defaults_uniform_single_stratum():
    proposal = ArrivalGeneratorSamplePathProposal()
    weights = proposal.path_weights(10)
    strata = proposal.path_strata(10)
    assert weights.shape == (10,)
    assert np.allclose(weights, 0.1)
    assert np.isclose(weights.sum(), 1.0)
    assert strata.shape == (10,)
    assert np.all(strata == 0)


def test_mixture_weights_match_allocation():
    proposal = MixtureGeometricStratifiedQMCProposal(
        target_discount_factor=0.99, discount_factor_proposal=0.95, lambda_0=0.1)
    n_long, n_short = proposal._component_sizes(256)
    assert (n_long, n_short) == (26, 230)
    weights = proposal.path_weights(256)
    strata = proposal.path_strata(256)
    assert weights.shape == (256,)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights[:n_long], 0.1 / n_long)
    assert np.allclose(weights[n_long:], 0.9 / n_short)
    assert np.all(strata[:n_long] == 0)
    assert np.all(strata[n_long:] == 1)
    # lambda_0 == 1 degenerates to a single uniform stratum, no raise.
    pure = MixtureGeometricStratifiedQMCProposal(
        target_discount_factor=0.99, discount_factor_proposal=0.95, lambda_0=1.0)
    assert np.allclose(pure.path_weights(8), 1.0 / 8)


def test_mixture_degenerate_stratum_raises():
    proposal = MixtureGeometricStratifiedQMCProposal(
        target_discount_factor=0.99, discount_factor_proposal=0.95, lambda_0=0.1)
    try:
        proposal.path_weights(4)  # round(0.4) == 0 long paths
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for empty long stratum')


def test_ci_uniform_reduces_to_classic():
    from metaheuristic_algorithm.benders_decomposition_solver import (
        BendersDecompositionSolver as B,
    )
    values = [10.0, 12.0, 8.0, 11.0, 9.0]
    mean, half = B._objective_confidence_interval(values)
    arr = np.asarray(values)
    assert np.isclose(mean, arr.mean())
    assert np.isclose(half, 1.96 * arr.std(ddof=1) / np.sqrt(5))
    assert B._objective_confidence_interval([7.5]) == (7.5, 0.0)
    assert B._objective_confidence_interval([]) == (0.0, 0.0)


def test_ci_stratified_matches_hand_calc():
    from metaheuristic_algorithm.benders_decomposition_solver import (
        BendersDecompositionSolver as B,
    )
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    weights = np.array([0.2 / 3] * 3 + [0.8 / 2] * 2)
    strata = np.array([0, 0, 0, 1, 1])
    mean, half = B._objective_confidence_interval(values, weights, strata)
    # Weighted mean: 0.2 * 2 + 0.8 * 15 = 12.4
    assert np.isclose(mean, 12.4)
    # Var = W0^2 s0^2 / n0 + W1^2 s1^2 / n1 = 0.04 * 1 / 3 + 0.64 * 50 / 2
    expected_var = 0.04 * 1.0 / 3 + 0.64 * 50.0 / 2
    assert np.isclose(half, 1.96 * np.sqrt(expected_var))


def test_stratified_running_stats_uniform_parity():
    from utils import RunningStats, StratifiedRunningStats
    values = [10.0, 12.0, 8.0, 11.0, 9.0, 13.5]
    base, strat = RunningStats(), StratifiedRunningStats()
    for v in values:
        base += v
        strat += v
    assert strat.n == base.n
    assert np.isclose(strat.mean, base.mean, rtol=1e-12)
    assert np.isclose(strat.half_window(0.95), base.half_window(0.95), rtol=1e-12)


def test_stratified_running_stats_two_strata_hand_calc():
    import scipy.stats as st
    from utils import StratifiedRunningStats
    strat = StratifiedRunningStats()
    for v in (1.0, 2.0, 3.0):
        strat.record(v, weight=0.2 / 3, stratum=0)
    for v in (10.0, 20.0):
        strat.record(v, weight=0.8 / 2, stratum=1)
    # Weighted mean: 0.2 * 2 + 0.8 * 15 = 12.4
    assert np.isclose(strat.mean, 12.4)
    # Var = W0^2 s0^2 / n0 + W1^2 s1^2 / n1 = 0.04 * 1 / 3 + 0.64 * 50 / 2
    expected_var = 0.04 * 1.0 / 3 + 0.64 * 50.0 / 2
    assert np.isclose(strat.variance_of_mean, expected_var)
    t_crit = st.t.ppf(0.975, 5 - 2)  # df = n - strata
    assert np.isclose(strat.half_window(0.95), t_crit * np.sqrt(expected_var))


def test_stratified_running_stats_merge_collapse_percentage():
    from utils import StratifiedRunningStats
    a, b = StratifiedRunningStats(), StratifiedRunningStats()
    for v in (1.0, 2.0, 3.0):
        a.record(v, weight=0.1, stratum=0)
    for v in (10.0, 20.0):
        b.record(v, weight=0.45, stratum=1)
    a += b
    assert a.n == 5
    # mean = (0.1*6 + 0.45*30) / (0.3 + 0.9) = 14.1 / 1.2
    assert np.isclose(a.mean, 14.1 / 1.2)
    # Collapse bridge: variance/n of the collapsed object equals variance_of_mean.
    collapsed = a.to_running_stats()
    assert collapsed.n == a.n
    assert np.isclose(collapsed.mean, a.mean)
    assert np.isclose(collapsed.variance / a.n, a.variance_of_mean)
    # Percentage-conversion call chain used by the aggregation: stats / scalar / 0.01
    pct = a / a.mean / 0.01
    assert np.isclose(pct.mean, 100.0)


def test_aggregation_legacy_records_fallback():
    import json
    import os
    import tempfile
    from experiments import get_config_by_type
    from experiments.new_result_aggregration import SimulateEvaluationResult
    env = get_config_by_type('toy').env
    num_types = env.num_types
    # Shape (periods, booking days, types); 25 booking-day rows keep every
    # waiting-time-target index in range after the sum over periods.
    scheduled = [[[0] * num_types for _ in range(25)] for _ in range(2)]
    scheduled[0][0] = [1] * num_types
    record = {
        'uid': 'u1', 'policy_id': 'myopic', 'group_id': 'g', 'mutate_val': 0.5,
        'warm_up_periods': 0,
        'total_cost': 10.0, 'penalized_cost': 12.0, 'total_penalty': 2.0,
        'zero_information_relaxation_cost': 8.0,
        'penalized_information_relaxation_cost': 9.0,
        'gap_to_zero_information_relaxation': 2.0,
        'gap_to_penalized_information_relaxation': 3.0,
        'costs': [5.0, 5.0],
        'scheduled_patients': scheduled,
        'overtime': [0, 0],
    }
    record2 = {**record, 'uid': 'u2', 'total_cost': 14.0}
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, '1.jsonl'), 'w') as handle:
            handle.write(json.dumps(record) + '\n')
            handle.write(json.dumps(record2) + '\n')
        ser = SimulateEvaluationResult(tmp, '[0-9]*.jsonl', env, is_reuse=False)
    stats = ser.policy_costs[('g', 'myopic')]
    # No path_weight fields -> uniform fallback: plain mean of 10 and 14.
    assert stats.n == 2
    assert np.isclose(stats.mean, 12.0)
    assert np.isclose(ser.zero_penalized_gap[('g', 'myopic')].mean, 2.0)


if __name__ == '__main__':
    test_base_defaults_uniform_single_stratum()
    test_mixture_weights_match_allocation()
    test_mixture_degenerate_stratum_raises()
    test_ci_uniform_reduces_to_classic()
    test_ci_stratified_matches_hand_calc()
    test_stratified_running_stats_uniform_parity()
    test_stratified_running_stats_two_strata_hand_calc()
    test_stratified_running_stats_merge_collapse_percentage()
    test_aggregation_legacy_records_fallback()
    print('All stratified path-weight tests passed.')
