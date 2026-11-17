import numpy as np
import pytest


def _rec(uid, policy_id, cost, weight=1.0, stratum=0, bookings=None):
    return {'uid': uid, 'policy_id': policy_id, 'total_cost': cost, 'path_weight': weight,
            'path_stratum': stratum, 'warm_up_periods': 0,
            'scheduled_patients': bookings if bookings is not None else [[[0]]]}


def test_paired_summary_matches_hand_computed_means_and_gap_share():
    from experiments.booking_rule_report import paired_summary
    uids = ['a', 'b', 'c', 'd']
    alp = {u: _rec(u, 'row_gen_alp', c, stratum=k % 2) for k, (u, c) in enumerate(zip(uids, [100., 110., 120., 130.]))}
    ph = {u: _rec(u, 'approx_penalized_hindsight', c, stratum=k % 2) for k, (u, c) in enumerate(zip(uids, [80., 90., 100., 110.]))}
    my = {u: _rec(u, 'myopic', c, stratum=k % 2) for k, (u, c) in enumerate(zip(uids, [200., 200., 200., 200.]))}
    var = {u: _rec(u, 'v', c, stratum=k % 2) for k, (u, c) in enumerate(zip(uids, [90., 100., 110., 120.]))}
    rows = paired_summary({'row_gen_alp': alp, 'approx_penalized_hindsight': ph, 'myopic': my, 'v': var},
                          ['row_gen_alp', 'v'])
    v = rows['v']
    assert v['n'] == 4
    assert v['cost'][0] == pytest.approx(105.0)
    assert v['minus_alp'][0] == pytest.approx(-10.0)
    assert v['gap_share_pct'][0] == pytest.approx(50.0)           # (115-105)/(115-95)
    assert v['improvement_over_myopic_pct'][0] == pytest.approx(47.5)
    assert rows['row_gen_alp']['gap_share_pct'][0] == pytest.approx(0.0)


def test_paired_summary_uses_only_paths_present_for_every_policy():
    from experiments.booking_rule_report import paired_summary
    alp = {u: _rec(u, 'row_gen_alp', 10.) for u in 'abc'}
    ph = {u: _rec(u, 'approx_penalized_hindsight', 5.) for u in 'ab'}
    my = {u: _rec(u, 'myopic', 20.) for u in 'abc'}
    rows = paired_summary({'row_gen_alp': alp, 'approx_penalized_hindsight': ph, 'myopic': my}, ['row_gen_alp'])
    assert rows['row_gen_alp']['n'] == 2


def test_within_target_share_counts_start_days_up_to_the_target_day():
    from experiments.booking_rule_report import within_target_by_type
    # one type, window of 4 booking days, target day 2 (waiting cost from day 3 on)
    late = np.array([[False], [False], [True], [True]])
    bookings = [[[1], [1], [1], [1]], [[0], [2], [0], [0]]]
    record = {'uid': 'a', 'policy_id': 'p', 'path_weight': 1.0, 'path_stratum': 0, 'warm_up_periods': 0,
              'scheduled_patients': bookings}
    share = within_target_by_type({'a': record}, late)
    assert share[0] == pytest.approx(100 * 4 / 6)


def test_cost_split_separates_waiting_from_the_rest_of_the_discounted_cost():
    from experiments.booking_rule_report import cost_split
    waiting_coeffs = np.array([[0.0], [10.0]])          # booking day 2 costs 10 per treatment
    record = {'uid': 'a', 'path_weight': 1.0, 'warm_up_periods': 1,
              'scheduled_patients': [[[5], [5]], [[1], [2]], [[0], [1]]],
              'costs': [999.0, 120.0, 10.0], 'total_cost': 120.0 + 0.5 * 10.0}
    waiting, other = cost_split({'a': record}, waiting_coeffs, {'a': [1.0, 0.5]})
    assert waiting == pytest.approx(20.0 + 0.5 * 10.0)
    assert other == pytest.approx(100.0)
