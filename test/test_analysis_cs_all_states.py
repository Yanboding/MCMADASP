import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.figures import cs_all_states as allfig  # noqa: E402
from analysis.figures import cs_matched_states as fig  # noqa: E402
from experiments import get_config_by_type  # noqa: E402


@pytest.fixture(scope='module')
def env():
    return get_config_by_type('ejor').env


@pytest.fixture(scope='module')
def spec(env):
    return fig.type_spec(env)


# ---------- replay of one record ----------

def test_replay_follows_bookings_overtime_and_arrivals(env):
    n_days, n_types, horizon = env.booking_window_size, env.num_types, env.planning_horizon
    w0 = np.zeros(n_types, int)
    w0[6] = 2                                    # two waiting type-7 treatments (one session of 2 slots)
    x = np.zeros((n_days, n_types), int)
    x[0, 6] = 1                                  # book one of them on booking day 1
    record = {'warm_up_periods': 1, 'costs': [0.0, 0.0, 0.0],
              'warmup_state': [[0] * horizon, [0] * horizon, w0.tolist()],
              'scheduled_patients': [np.zeros((n_days, n_types), int).tolist(), x.tolist(),
                                     np.zeros((n_days, n_types), int).tolist()]}
    arrivals = np.zeros((3, n_types), int)
    arrivals[1, 3] = 1                           # after the first post-warm-up decision a type-4 request arrives
    periods = list(allfig.replay(record, arrivals, env))
    assert len(periods) == 2
    booked, bookings, waitlist = periods[0]
    assert booked == 0
    np.testing.assert_array_equal(bookings, x)
    np.testing.assert_array_equal(waitlist, w0)
    booked, _, waitlist = periods[1]
    assert booked == 0                           # booking day 1 has passed, nothing else is booked
    expected = w0.copy()
    expected[6] -= 1
    expected[3] += 1
    np.testing.assert_array_equal(waitlist, expected)


def test_replay_counts_slots_booked_in_the_booking_window(env):
    n_days, n_types, horizon = env.booking_window_size, env.num_types, env.planning_horizon
    u = [0] * horizon
    u[1], u[n_days - 1], u[n_days] = 5, 7, 100   # the slots beyond the booking window do not count
    v = [0] * horizon
    v[2] = 3
    record = {'warm_up_periods': 0, 'costs': [0.0], 'warmup_state': [u, v, [0] * n_types],
              'scheduled_patients': [np.zeros((n_days, n_types), int).tolist()]}
    (booked, _, _), = allfig.replay(record, np.zeros((1, n_types), int), env)
    assert booked == 15


# ---------- counts and table ----------

def test_counts_are_kept_per_number_of_booked_slots_and_for_the_first_period():
    periods = [(3, np.array([[1, 0], [0, 2]]), np.array([2, 3])),
               (5, np.array([[0, 1], [0, 0]]), np.array([1, 1])),
               (3, np.array([[0, 0], [1, 0]]), np.array([1, 0]))]
    c = allfig.empty_counts(capacity=6, n_days=2, n_types=2)
    allfig.add_path(c, periods)
    assert c['periods'].tolist() == [0, 0, 0, 2, 0, 1, 0]
    np.testing.assert_array_equal(c['booked'][3], [[1, 0], [1, 2]])
    np.testing.assert_array_equal(c['waiting'][3], [3, 3])
    assert c['first_periods'].tolist() == [0, 0, 0, 1, 0, 0, 0]
    np.testing.assert_array_equal(c['first_booked'][3], [[1, 0], [0, 2]])
    np.testing.assert_array_equal(c['first_waiting'][3], [2, 3])


def test_table_groups_by_the_cut_points_and_keeps_the_part_left_on_the_waitlist():
    c = allfig.empty_counts(capacity=10, n_days=2, n_types=1)
    allfig.add_path(c, [(5, np.array([[2], [0]]), np.array([3])),      # 50%: low
                        (6, np.array([[1], [1]]), np.array([2])),      # 60%: mid
                        (7, np.array([[0], [1]]), np.array([1]))])     # 70%: high
    counts = {'ALP': c, 'PH': c}
    table, n_periods = allfig.all_states_table(counts, (60, 69), capacity=10)
    assert n_periods == {('low', 'ALP'): 1, ('mid', 'ALP'): 1, ('high', 'ALP'): 1,
                         ('low', 'PH'): 1, ('mid', 'PH'): 1, ('high', 'PH'): 1}
    np.testing.assert_allclose(table[('low', 'ALP', 1)]['d'], [2, 0])
    assert table[('low', 'ALP', 1)]['left'] == pytest.approx(1.0)
    np.testing.assert_allclose(table[('high', 'PH', 1)]['d'], [0, 1])
    assert table[('high', 'PH', 1)]['left'] == pytest.approx(0.0)


def test_figure_has_its_own_file_name_and_cache():
    assert allfig.FILE_NAME == 'cs_all_states_waiting_time_by_type_sessions.svg'
    assert allfig.FILE_NAME != fig.FILE_NAME
    assert allfig.CACHE_FILE == 'cs_all_states.pkl'


# ---------- the real results ----------

def _real_cache_is_current():
    path = os.path.join(fig.DEFAULT_RESULTS_DIR, fig.CACHE_DIR, allfig.CACHE_FILE)
    if not os.path.exists(path):
        return False
    with open(path, 'rb') as handle:
        return pickle.load(handle).get('version') == allfig.CACHE_VERSION


needs_cache = pytest.mark.skipif(not _real_cache_is_current(), reason='results are not in git; needs a current cache')


@pytest.fixture(scope='module')
def real(env):
    data, source = allfig.load_data(fig.DEFAULT_RESULTS_DIR)
    assert source == 'cache'
    capacity = env.booking_window_size * (env.regular_capacity + env.overtime_capacity)
    return data, capacity


@needs_cache
def test_replayed_costs_equal_the_recorded_costs(real):
    data, _ = real
    assert data['max_abs_cost_difference'] < 1e-6


@needs_cache
def test_number_of_periods_per_group(real):
    data, capacity = real
    _, n_periods = allfig.all_states_table(data['counts'], (60, 69), capacity)
    assert [n_periods[(g, 'ALP')] for g in fig.GROUPS] == [62431, 73141, 71138]
    assert [n_periods[(g, 'PH')] for g in fig.GROUPS] == [63193, 72943, 70574]


@needs_cache
def test_first_period_reproduces_the_first_state_figure(real, spec):
    data, capacity = real
    table, _ = allfig.all_states_table(data['counts'], (60, 69), capacity, first_only=True)
    assert fig.bar_totals(table, spec) == [
        'urgent | 98 98 2 2 0 0 0 0 | 75 75 20 20 5 5 0 0 | 16 16 23 23 44 42 12 18',
        'five | 66 57 0 26 34 17 0 0 0 0 | 20 20 1 21 74 53 5 6 0 0 | 3 7 4 8 32 21 42 38 19 26',
        'ten | 26 44 0 5 74 51 0 0 | 4 5 0 1 96 94 0 0 | 0 0 0 0 74 34 25 66',
    ]


@needs_cache
def test_all_period_bars_equal_the_replay_tables(real, spec):
    data, capacity = real
    table, _ = allfig.all_states_table(data['counts'], (60, 69), capacity)
    expected = {  # (class, group): (ALP bars, PH bars), percentages of the waiting treatments
        ('urgent', 'low'): ([99.0, 1.0, 0.0, 0.0], [99.8, 0.2, 0.0, 0.0]),
        ('urgent', 'mid'): ([78.0, 19.9, 2.2, 0.0], [93.2, 6.7, 0.1, 0.0]),
        ('urgent', 'high'): ([18.6, 24.4, 40.9, 12.3], [68.0, 29.2, 2.3, 0.5]),
        ('five', 'low'): ([67.8, 0.0, 32.2, 0.0, 0.0], [34.1, 44.1, 21.7, 0.0, 0.0]),
        ('five', 'mid'): ([21.3, 1.2, 74.1, 3.4, 0.0], [13.4, 22.9, 62.6, 1.2, 0.0]),
        ('five', 'high'): ([2.7, 4.1, 33.7, 44.2, 15.2], [15.0, 14.8, 55.0, 14.0, 1.2]),
        ('ten', 'low'): ([26.7, 0.0, 73.3, 0.0], [26.7, 9.1, 64.2, 0.0]),
        ('ten', 'mid'): ([4.2, 0.0, 95.8, 0.0], [1.5, 0.2, 97.7, 0.6]),
        ('ten', 'high'): ([0.2, 0.0, 79.6, 20.2], [0.0, 0.0, 70.5, 29.5]),
    }
    for (c, g), (alp, ph) in expected.items():
        for p, want in (('ALP', alp), ('PH', ph)):
            got = fig.shares(table, g, p, spec['classes'][c], fig.BINS[c], spec['level']).sum(1)
            np.testing.assert_allclose(got, want, atol=0.15, err_msg=f'{c} {g} {p}')
