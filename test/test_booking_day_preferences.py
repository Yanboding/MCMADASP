import numpy as np

from experiments.new_result_aggregration import booking_day_ranks, preference_runs


def test_ranks_order_days_by_share_and_leave_unused_days_unranked():
    shares = np.array([[50.0, 10.0],
                       [0.0, 40.0],
                       [30.0, 0.0],
                       [20.0, 50.0]])
    ranks = booking_day_ranks(shares)
    assert ranks[:, 0].tolist() == [1, 0, 2, 3]
    assert ranks[:, 1].tolist() == [3, 2, 0, 1]


def test_ranks_break_ties_by_earlier_day():
    shares = np.array([[25.0], [25.0], [50.0]])
    assert booking_day_ranks(shares)[:, 0].tolist() == [2, 3, 1]


def test_runs_follow_consecutive_ranks_in_either_direction():
    ranks = np.array([1, 5, 4, 3, 2, 6, 7, 8, 0, 0])
    assert preference_runs(ranks) == [(4, 1), (5, 7)]


def test_runs_need_at_least_three_days_and_skip_unranked_days():
    assert preference_runs(np.array([1, 2, 0, 3, 4, 5])) == [(3, 5)]
    assert preference_runs(np.array([3, 1, 2])) == []
