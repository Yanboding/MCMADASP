import json
import os
import pickle
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.figures import cs_matched_states as fig  # noqa: E402
from experiments import get_config_by_type  # noqa: E402


@pytest.fixture(scope='module')
def spec():
    return fig.type_spec(get_config_by_type('ejor').env)


def test_classes_and_levels_come_from_the_case_study_parameters(spec):
    assert spec['classes'] == {'urgent': (1, 2, 3),
                               'five': (7, 8, 9, 10, 11, 12, 13, 14),
                               'ten': (4, 5, 6, 15, 16, 17, 18)}
    by_level = {lv: {t for t, l in spec['level'].items() if l == lv} for lv in (0, 1, 2)}
    assert by_level == {0: {1, 2, 3, 7, 8, 10, 12}, 1: {4, 5, 6, 9, 11, 13}, 2: {14, 15, 16, 17, 18}}


def test_row_legend_labels(spec):
    assert spec['labels'] == {
        'urgent': {0: 'Types 1–3 (1–5 sessions)'},
        'five': {0: 'Types 7, 8, 10, 12 (1–5 sessions)', 1: 'Types 9, 11, 13 (10–20 sessions)',
                 2: 'Type 14 (35 sessions)'},
        'ten': {1: 'Types 4–6 (16–20 sessions)', 2: 'Types 15–18 (33–37 sessions)'},
    }


def test_label_helper_on_small_inputs():
    assert fig.cell_label([14], [35]) == 'Type 14 (35 sessions)'
    assert fig.cell_label([1, 2, 3], [5, 1, 4]) == 'Types 1–3 (1–5 sessions)'
    assert fig.cell_label([7, 8, 10], [1, 5, 4]) == 'Types 7, 8, 10 (1–5 sessions)'
    assert fig.cell_label([2, 3], [6, 6]) == 'Types 2–3 (6 sessions)'


# ---------- loading the results folder ----------

def _record(policy, uid, warmup, decisions, warm_up_periods=1):
    # scheduled_patients holds one decision per period; index warm_up_periods is the first decision after warm-up
    scheduled = [np.zeros_like(np.asarray(decisions)).tolist()] * warm_up_periods + [np.asarray(decisions).tolist()]
    return {'policy_id': policy, 'uid': uid, 'warm_up_periods': warm_up_periods,
            'warmup_state': [list(map(int, x)) for x in warmup], 'scheduled_patients': scheduled,
            'total_cost': 1.0}


def _write(path, records):
    with open(path, 'w') as handle:
        for r in records:
            handle.write(json.dumps(r) + '\n')


STATE_A = ([1, 2, 3], [0, 1, 0], [4, 1])
STATE_B = ([5, 0, 0], [2, 0, 0], [0, 3])


def _tiny_results(tmp_path):
    _write(tmp_path / '10_1.jsonl', [
        _record('row_gen_alp', 'b', STATE_B, [[0, 1], [0, 2]]),
        _record('approx_penalized_hindsight', 'b', STATE_B, [[0, 3], [0, 0]]),
        _record('myopic', 'b', ([9, 9, 9], [9, 9, 9], [9, 9]), [[9, 9], [9, 9]]),
    ])
    _write(tmp_path / '10_2.jsonl', [
        _record('row_gen_alp', 'a', STATE_A, [[1, 0], [0, 0]]),
        _record('approx_penalized_hindsight', 'a', STATE_A, [[2, 0], [1, 1]]),
    ])
    _write(tmp_path / '11_1.jsonl', [  # rerun of one case: duplicate record, must not change the result
        _record('row_gen_alp', 'a', STATE_A, [[1, 0], [0, 0]]),
    ])
    _write(tmp_path / 'simulate.jsonl', [_record('row_gen_alp', 'z', STATE_A, [[7, 7], [7, 7]])])  # not [0-9]*


def test_loading_reads_the_two_policies_in_a_fixed_uid_order(tmp_path):
    _tiny_results(tmp_path)
    data, source = fig.load_data(str(tmp_path), processes=1)
    assert source == 'built'
    assert data['uids'] == ['a', 'b']
    np.testing.assert_array_equal(data['u'], [[1, 2, 3], [5, 0, 0]])
    np.testing.assert_array_equal(data['v'], [[0, 1, 0], [2, 0, 0]])
    np.testing.assert_array_equal(data['w'], [[4, 1], [0, 3]])
    np.testing.assert_array_equal(data['decisions']['ALP'], [[[1, 0], [0, 0]], [[0, 1], [0, 2]]])
    np.testing.assert_array_equal(data['decisions']['PH'], [[[2, 0], [1, 1]], [[0, 3], [0, 0]]])
    assert data['version'] == fig.CACHE_VERSION


def test_compact_json_records_and_blank_lines_are_read(tmp_path):
    records = [_record('row_gen_alp', 'c', STATE_A, [[1, 0], [0, 1]]),
               _record('approx_penalized_hindsight', 'c', STATE_A, [[0, 2], [0, 0]])]
    with open(tmp_path / '5_1.jsonl', 'w') as handle:
        handle.write('\n'.join(json.dumps(r, separators=(',', ':')) for r in records) + '\n\n')
    data, _ = fig.load_data(str(tmp_path), processes=1)
    assert data['uids'] == ['c']
    np.testing.assert_array_equal(data['decisions']['ALP'], [[[1, 0], [0, 1]]])
    np.testing.assert_array_equal(data['decisions']['PH'], [[[0, 2], [0, 0]]])


def test_different_warm_up_states_raise_an_error(tmp_path):
    _write(tmp_path / '1_1.jsonl', [
        _record('row_gen_alp', 'q', STATE_A, [[1, 0], [0, 0]]),
        _record('approx_penalized_hindsight', 'q', STATE_B, [[1, 0], [0, 0]]),
    ])
    with pytest.raises(ValueError, match='q'):
        fig.load_data(str(tmp_path), processes=1)


def test_a_uid_with_only_one_policy_raises_an_error(tmp_path):
    _write(tmp_path / '1_1.jsonl', [
        _record('row_gen_alp', 'p', STATE_A, [[1, 0], [0, 0]]),
        _record('row_gen_alp', 'r', STATE_B, [[0, 1], [0, 0]]),
        _record('approx_penalized_hindsight', 'r', STATE_B, [[0, 1], [0, 0]]),
    ])
    with pytest.raises(ValueError, match="'p'"):
        fig.load_data(str(tmp_path), processes=1)


def test_output_directory_inside_the_results_tree_is_refused():
    target = os.path.join(fig.REPO_ROOT, 'experiments', 'results', 'no_such_folder_for_the_figure_test')
    assert not os.path.exists(target)
    with pytest.raises(SystemExit):
        fig.main(['--out-dir', target, '--results-dir', os.path.join(fig.REPO_ROOT, 'no_such_results_folder')])
    assert not os.path.exists(target)


def test_cache_is_written_once_and_reused(tmp_path):
    _tiny_results(tmp_path)
    before = set(os.listdir(tmp_path))
    data, source = fig.load_data(str(tmp_path), processes=1)
    assert source == 'built'
    assert set(os.listdir(tmp_path)) - before == {'analysis_cache'}
    assert os.listdir(tmp_path / 'analysis_cache') == ['cs_matched_states.pkl']
    for name in before:
        os.remove(tmp_path / name)
    again, source = fig.load_data(str(tmp_path), processes=1)
    assert source == 'cache'
    assert again['uids'] == data['uids']
    np.testing.assert_array_equal(again['decisions']['PH'], data['decisions']['PH'])


def test_refresh_reads_the_records_again(tmp_path):
    _tiny_results(tmp_path)
    fig.load_data(str(tmp_path), processes=1)
    _write(tmp_path / '10_2.jsonl', [
        _record('row_gen_alp', 'a', STATE_A, [[3, 0], [0, 0]]),
        _record('approx_penalized_hindsight', 'a', STATE_A, [[2, 0], [1, 1]]),
    ])
    os.remove(tmp_path / '11_1.jsonl')
    cached, source = fig.load_data(str(tmp_path), processes=1)
    assert source == 'cache' and cached['decisions']['ALP'][0, 0, 0] == 1
    fresh, source = fig.load_data(str(tmp_path), refresh=True, processes=1)
    assert source == 'built' and fresh['decisions']['ALP'][0, 0, 0] == 3


def test_cache_with_another_version_is_rebuilt(tmp_path):
    _tiny_results(tmp_path)
    data, _ = fig.load_data(str(tmp_path), processes=1)
    path = tmp_path / 'analysis_cache' / 'cs_matched_states.pkl'
    with open(path, 'wb') as handle:
        pickle.dump({**data, 'version': fig.CACHE_VERSION - 1}, handle)
    rebuilt, source = fig.load_data(str(tmp_path), processes=1)
    assert source == 'built' and rebuilt['version'] == fig.CACHE_VERSION


# ---------- groups, table and shares ----------

def _data(u, v, w, alp, ph):
    return {'uids': [str(k) for k in range(len(u))], 'u': np.array(u), 'v': np.array(v), 'w': np.array(w),
            'decisions': {'ALP': np.array(alp), 'PH': np.array(ph)}, 'version': fig.CACHE_VERSION}


def test_table_step_on_tiny_data():
    # 2 booking days, 2 types, capacity 100 slots; ALP booked shares 0.0, 0.5, 1.0
    u = [[0, 0, 9], [30, 10, 9], [60, 20, 9]]
    v = [[0, 0, 9], [5, 5, 9], [15, 5, 9]]
    w = [[3, 1], [2, 2], [3, 4]]
    alp = [[[1, 0], [0, 1]]] * 3
    ph = [[[2, 0], [0, 0]]] * 3
    cutoffs, groups, table = fig.first_decision_table(_data(u, v, w, alp, ph), n_days=2, n_types=2, capacity=100)
    assert cutoffs == (33, 67)
    assert groups == {'low': ['0'], 'mid': ['1'], 'high': ['2']}
    row = table[('mid', 'ALP', 2)]
    assert row['n_states'] == 1
    np.testing.assert_allclose(row['d'], [0.0, 1.0])
    assert row['left'] == pytest.approx(1.0)        # waitlist 2 minus 1 booked
    row = table[('high', 'PH', 1)]
    np.testing.assert_allclose(row['d'], [2.0, 0.0])
    assert row['left'] == pytest.approx(1.0)        # waitlist 3 minus 2 booked


def test_table_step_averages_over_the_states_of_a_group():
    # 1 booking day, capacity 10 slots; booked 0, 0, 5, 5, 10, 10 gives cutoffs 33% and 67% and two states per group
    u = [[0], [0], [5], [5], [10], [10]]
    v = [[0]] * 6
    w = [[1, 0], [1, 0], [2, 0], [2, 0], [4, 0], [3, 0]]
    alp = [[[1, 0]], [[1, 0]], [[2, 0]], [[1, 0]], [[1, 0]], [[3, 0]]]
    cutoffs, groups, table = fig.first_decision_table(_data(u, v, w, alp, alp), n_days=1, n_types=2, capacity=10)
    assert cutoffs == (33, 67)
    assert groups == {'low': ['0', '1'], 'mid': ['2', '3'], 'high': ['4', '5']}
    row = table[('high', 'ALP', 1)]
    assert row['n_states'] == 2
    np.testing.assert_allclose(row['d'], [2.0])     # mean of 1 and 3
    assert row['left'] == pytest.approx(1.5)        # mean of 4-1 and 3-3
    row = table[('mid', 'ALP', 1)]
    np.testing.assert_allclose(row['d'], [1.5])
    assert row['left'] == pytest.approx(0.5)        # mean of 2-2 and 2-1


def test_shares_include_the_part_left_on_the_waitlist():
    d1 = {'n_states': 1, 'd': np.array([6.0, 2.0, 0.0, 0.0]), 'left': 2.0}
    d2 = {'n_states': 1, 'd': np.array([0.0, 4.0, 4.0, 0.0]), 'left': 2.0}
    table = {('low', 'ALP', 1): d1, ('low', 'ALP', 2): d2}
    bins = [('[0, 1)', 0, 1), ('[1, 4)', 1, 4)]
    arr = fig.shares(table, 'low', 'ALP', (1, 2), bins, {1: 0, 2: 2})
    total = 6 + 2 + 2 + 4 + 4 + 2
    np.testing.assert_allclose(arr, 100 * np.array([[6, 0, 0], [2, 0, 8]]) / total)


def test_dashed_line_positions():
    assert fig.dashed_after(fig.BINS['urgent'], 1) == 1
    assert fig.dashed_after(fig.BINS['five'], 5) == 3
    assert fig.dashed_after(fig.BINS['ten'], 10) == 3


def _synthetic_table(spec, n_days=20):
    table = {}
    for g in ('low', 'mid', 'high'):
        for p in ('ALP', 'PH'):
            for t in spec['level']:
                d = np.zeros(n_days)
                d[(t * 3) % n_days] = 1.0 + t / 10
                table[(g, p, t)] = {'n_states': 5, 'd': d, 'left': 0.1}
    return table


def test_plot_on_tiny_data_is_reproducible(spec, tmp_path):
    table = _synthetic_table(spec)
    a, b = tmp_path / 'a.svg', tmp_path / 'b.svg'
    fig.plot_figure(table, spec, (60, 69), str(a))
    fig.plot_figure(table, spec, (60, 69), str(b))
    head = a.read_text()[:600]
    assert 'width="468pt"' in head and 'height="522pt"' in head
    assert a.read_bytes() == b.read_bytes()


def _real_cache_is_current():
    path = os.path.join(fig.DEFAULT_RESULTS_DIR, fig.CACHE_DIR, fig.CACHE_FILE)
    if not os.path.exists(path):
        return False
    with open(path, 'rb') as handle:
        return pickle.load(handle).get('version') == fig.CACHE_VERSION


@pytest.mark.skipif(not _real_cache_is_current(), reason='results are not in git; needs a current cache')
def test_bar_totals_with_the_real_results(spec):
    env = get_config_by_type('ejor').env
    data, source = fig.load_data(fig.DEFAULT_RESULTS_DIR)
    assert source == 'cache'
    cutoffs, _, table = fig.first_decision_table(
        data, env.booking_window_size, env.num_types,
        env.booking_window_size * (env.regular_capacity + env.overtime_capacity))
    assert cutoffs == (60, 69)
    assert fig.bar_totals(table, spec) == [
        'urgent | 98 98 2 2 0 0 0 0 | 75 75 20 20 5 5 0 0 | 16 16 23 23 44 42 12 18',
        'five | 66 57 0 26 34 17 0 0 0 0 | 20 20 1 21 74 53 5 6 0 0 | 3 7 4 8 32 21 42 38 19 26',
        'ten | 26 44 0 5 74 51 0 0 | 4 5 0 1 96 94 0 0 | 0 0 0 0 74 34 25 66',
    ]
