import json
import os
import tempfile

import numpy as np
import pytest

import run
from experiments import get_config_by_type
from experiments.new_result_aggregration import (
    _TOY_LB_IMPROVEMENT_SERIES,
    TOY_LOWER_BOUND_SENSITIVITY_PANELS,
    partial_relaxation_costs,
    partial_relaxation_gaps,
    plot_toy_lower_bound_improvements,
)
from param_generation.command_files import read_command_file_records, write_grouped_command_file
from param_generation.partial_relaxation import (
    default_partial_relaxation_horizon,
    partial_relaxation_records,
    warm_up_states,
)

STATE = [[5] * 6 + [0], [0] * 7, [1, 2]]
SHARED = {'myopic': STATE, 'row_gen_alp': STATE}


def _toy():
    config = get_config_by_type('toy')
    return config.env, config.args


def _source(env_args, uid='a' * 32, terminal='absorbed'):
    return {'uid': uid, 'group_id': 'g', 'mutate_val': 0.5, 'env_args': env_args, 'warm_up_periods': 2,
            'sample_path': [[0, 0], [1, 1], [1, 0], [0, 2], [1, 1]], 'period_weights': [1.0, 0.99, 0.99, 0.99],
            'terminal': terminal, 'path_weight': 0.25, 'path_stratum': 3}


def test_partial_records_keep_the_known_arrivals_and_extend_them_from_the_arrival_law():
    env, env_args = _toy()
    source = _source(env_args)
    (record,) = partial_relaxation_records([source], {source['uid']: SHARED}, 'x', horizon=10)
    tail = source['sample_path'][2:]
    extension = env.arrival_generator.rvs(6, rng=np.random.default_rng(int(source['uid'][:15], 16)))
    assert record['arrival_stream'] == tail + extension.tolist()
    assert record['init_state'] == STATE and record['source_uid'] == source['uid']
    assert record['policy_ids'] == ['myopic', 'row_gen_alp']
    (short,) = partial_relaxation_records([source], {source['uid']: SHARED}, 'x', horizon=3)
    assert short['arrival_stream'] == tail[:2]
    (default,) = partial_relaxation_records([source], {source['uid']: SHARED}, 'x')
    assert len(default['arrival_stream']) == default_partial_relaxation_horizon(env.discount_factor) - 1


def test_partial_records_give_each_distinct_warm_up_state_its_own_bound_over_the_same_arrivals():
    _, env_args = _toy()
    source = _source(env_args)
    other = [[4] * 6 + [0], [0] * 7, [0, 1]]
    states = {source['uid']: {'myopic': STATE, 'row_gen_alp': other, 'alp_penalized_hindsight': STATE}}
    records = partial_relaxation_records([source], states, 'x', horizon=10)
    by_policies = {tuple(record['policy_ids']): record for record in records}
    assert set(by_policies) == {('alp_penalized_hindsight', 'myopic'), ('row_gen_alp',)}
    assert by_policies[('row_gen_alp',)]['init_state'] == other
    assert by_policies[('alp_penalized_hindsight', 'myopic')]['init_state'] == STATE
    assert len({record['uid'] for record in records}) == 2
    assert len({json.dumps(record['arrival_stream']) for record in records}) == 1


def test_partial_records_reject_bad_horizons_truncated_sources_and_missing_states():
    _, env_args = _toy()
    source = _source(env_args)
    with pytest.raises(ValueError, match='at least 2'):
        partial_relaxation_records([source], {source['uid']: SHARED}, 'x', horizon=0)
    with pytest.raises(ValueError, match='absorbed geometric horizon'):
        partial_relaxation_records([_source(env_args, terminal='truncated')], {source['uid']: SHARED}, 'x')
    with pytest.raises(ValueError, match='no post-warm-up state'):
        partial_relaxation_records([source], {}, 'x')
    with pytest.raises(ValueError, match='no evaluation records'):
        partial_relaxation_records([], {}, 'x')


def _write_results(folder, records):
    with open(os.path.join(folder, '1_1.jsonl'), 'w') as handle:
        handle.writelines(json.dumps(record) + '\n' for record in records)


def test_warm_up_states_are_read_per_policy_and_reject_conflicting_reruns():
    nested = {'agent_args': {'uid': 'f' * 32}, 'uid': 'b' * 32, 'policy_id': 'myopic', 'warmup_state': [[1]]}
    with tempfile.TemporaryDirectory() as tmp:
        _write_results(tmp, [nested, nested, {**nested, 'policy_id': 'row_gen_alp', 'warmup_state': [[3]]},
                             {'uid': 'c' * 32, 'policy_id': 'myopic', 'warmup_state': [[2]]}])
        assert warm_up_states(tmp, ['b' * 32, 'c' * 32, 'd' * 32]) == {
            'b' * 32: {'myopic': [[1]], 'row_gen_alp': [[3]]}, 'c' * 32: {'myopic': [[2]]}}
        _write_results(tmp, [nested, {**nested, 'warmup_state': [[3]]}])
        with pytest.raises(ValueError, match='two different post-warm-up states'):
            warm_up_states(tmp, ['b' * 32])


def test_command_file_records_round_trip():
    records = [{'uid': str(index), 'value': index} for index in range(5)]
    with tempfile.TemporaryDirectory() as tmp:
        dat = os.path.join(tmp, 'table.dat')
        write_grouped_command_file(results=records, num_groups=2, dat_file=dat)
        assert sorted(read_command_file_records(dat), key=lambda record: record['value']) == records


def test_partial_relaxation_costs_reject_conflicting_reruns():
    record = {'source_uid': 'a', 'group_id': 'g', 'horizon': 917, 'partial_information_relaxation_cost': 1.0}
    with tempfile.TemporaryDirectory() as tmp:
        _write_results(tmp, [record, record])
        assert partial_relaxation_costs(tmp) == {('a', None): 1.0}
        _write_results(tmp, [{**record, 'policy_ids': ['p', 'q']}, {**record, 'policy_ids': ['r'],
                                                                   'partial_information_relaxation_cost': 3.0}])
        assert partial_relaxation_costs(tmp) == {('a', 'p'): 1.0, ('a', 'q'): 1.0, ('a', 'r'): 3.0}
        _write_results(tmp, [record, {**record, 'partial_information_relaxation_cost': 2.0}])
        with pytest.raises(ValueError, match='two different'):
            partial_relaxation_costs(tmp)
        _write_results(tmp, [record, {**record, 'source_uid': 'b', 'horizon': 600}])
        with pytest.raises(ValueError, match='mixes partial-relaxation horizons'):
            partial_relaxation_costs(tmp)


class _StubResults:
    def __init__(self, costs, weights, strata):
        key = ('g', 'pi')
        self.cost_by_uid = {key: costs}
        self.weight_by_uid = {key: weights}
        self.stratum_by_uid = {key: strata}


def test_partial_relaxation_gaps_are_paired_on_source_uids():
    rng = np.random.default_rng(0)
    n = 200
    bound = rng.normal(40000, 5000, n)
    policy = bound + rng.normal(20000, 1000, n)
    uids = [f'u{i}' for i in range(n)]
    ser = _StubResults(dict(zip(uids, policy)), {uid: 1.0 / n for uid in uids}, {uid: 0 for uid in uids})
    row = partial_relaxation_gaps(ser, {(uid, None): value for uid, value in zip(uids, bound)}, 'g', 'pi')
    per_policy = partial_relaxation_gaps(ser, {(uid, 'pi'): value for uid, value in zip(uids, bound)}, 'g', 'pi')
    assert np.isclose(per_policy['gap_mean'], row['gap_mean'])
    from scipy import stats
    assert row['n'] == n
    assert np.isclose(row['partial_mean'], bound.mean())
    assert np.isclose(row['gap_mean'], (policy - bound).mean())
    assert np.isclose(row['gap_half_window'], stats.t.ppf(0.975, n - 1) * (policy - bound).std(ddof=1) / np.sqrt(n))
    assert np.isclose(row['gap_pct'], 100 * (policy - bound).mean() / bound.mean())
    with pytest.raises(ValueError, match='no partial-relaxation cost'):
        partial_relaxation_gaps(ser, {(uid, 'other'): value for uid, value in zip(uids, bound)}, 'g', 'pi')


def _toy_path(env, periods, seed=3):
    return env.arrival_generator.rvs(periods - 1, rng=np.random.default_rng(seed)), \
        (env.discount_factor ** np.arange(periods)).tolist()


def test_zero_penalty_bound_is_unchanged_by_skipping_the_penalty_features():
    env, _ = _toy()
    arrivals, weights = _toy_path(env, 8)
    state = tuple(np.array(component) for component in STATE)
    zero_coefficients = [0.0] * (1 + 2 * env.planning_horizon + env.num_types)
    skipped = run.information_relaxation_bounds(
        env, {'name': 'absorption_alp_penalty'}, [0.0], state, arrivals, weights, 'absorbed', None, [])[0.0]
    built = run.information_relaxation_bounds(
        env, {'name': 'absorption_alp_penalty', 'coefficients': zero_coefficients}, [1.0], state, arrivals, weights,
        'absorbed', None, [])[1.0]
    assert np.isclose(skipped, built)


def test_partial_bound_grows_with_the_horizon_and_is_written_once(monkeypatch):
    env, env_args = _toy()
    arrivals, _ = _toy_path(env, 12)
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.chdir(tmp)
        costs = []
        for periods in (4, 12):
            record = run.partial_information_relaxation_for_path(
                uid=f'p{periods}', source_uid='s', experiment_name='x', mutate_val=0.5, group_id='g',
                env_args=env_args, init_state=STATE, arrival_stream=arrivals[:periods - 1].tolist(), path_weight=1.0,
                path_stratum=0, grb_env=None, grb_sub_envs=[], job_id='1')
            assert record['horizon'] == periods
            costs.append(record['partial_information_relaxation_cost'])
        assert 0.0 <= costs[0] <= costs[1] + 1e-6
        assert run.partial_information_relaxation_for_path(
            uid='p12', source_uid='s', experiment_name='x', mutate_val=0.5, group_id='g', env_args=env_args,
            init_state=STATE, arrival_stream=arrivals.tolist(), path_weight=1.0, path_stratum=0, grb_env=None,
            grb_sub_envs=[], job_id='1') is None


def test_improvement_plot_tolerates_a_panel_without_values():
    rows = []
    for panel, _, _, _ in TOY_LOWER_BOUND_SENSITIVITY_PANELS:
        for value in (1.0, 2.0):
            row = {'panel': panel, 'mutate_val': value}
            for mean_key, half_window_key, _, _ in _TOY_LB_IMPROVEMENT_SERIES:
                row[mean_key] = None if panel == 'type1_share' else value
                row[half_window_key] = None if panel == 'type1_share' else 0.1
            rows.append(row)
    with tempfile.TemporaryDirectory() as tmp:
        plot_toy_lower_bound_improvements(rows, os.path.join(tmp, 'figure.svg'))
        assert os.path.exists(os.path.join(tmp, 'figure.svg'))
