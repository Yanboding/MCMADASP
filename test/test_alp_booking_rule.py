import os

import gurobipy as gp
import numpy as np
import pytest

from decision_maker import ALPRowGenerationAgent
from experiments import get_config_by_type
from param_generation.command_files import read_command_file_records

DAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'table_alp_steady_state.dat')
LONG = [5, 13, 14, 15, 16, 17]          # Types 6, 14-18 (33 or more capacity slots)
TEN_DAY = [3, 4, 5, 14, 15, 16, 17]     # Types 4-6, 15-18


@pytest.fixture(scope='module')
def grb_env():
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.setParam('Threads', 1)
        env.start()
        yield env


@pytest.fixture(scope='module')
def case():
    if not os.path.exists(DAT):
        pytest.skip('case-study command table not available')
    record = read_command_file_records(DAT)[0]
    env = get_config_by_type('infinite_custom', args=record['env_args']).env
    coefficients = next(spec['agent_args']['coefficients'] for spec in record['policy_specs']
                        if spec['agent_name'] == 'row_gen_alp')
    return env, coefficients


def _rule_agent(case, grb_env, restricted_types, reserve):
    from decision_maker.alp_booking_rule_agent import ALPBookingRuleAgent
    env, coefficients = case
    return ALPBookingRuleAgent(env, discount_factor=env.discount_factor, coefficients=coefficients,
                               restricted_types=restricted_types, reserve=reserve, grb_env=grb_env)


def _alp_agent(case, grb_env):
    env, coefficients = case
    return ALPRowGenerationAgent(env, discount_factor=env.discount_factor, coefficients=coefficients,
                                 grb_env=grb_env)


def _state(env, regular_first_days, waitlist, overrides=None):
    H = env.planning_horizon
    u = np.zeros(H)
    u[:len(regular_first_days)] = regular_first_days
    for day, value in (overrides or {}).items():
        u[day - 1] = value
    w = np.zeros(env.num_types)
    for i, n in waitlist.items():
        w[i] = n
    return u, np.zeros(H), w


def _start_days(bookings, type_index):
    bookings = np.asarray(bookings)
    return [j + 1 for j in range(bookings.shape[0]) for _ in range(int(round(bookings[j, type_index])))]


def test_empty_restricted_set_reproduces_the_alp_policy(case, grb_env):
    env, _ = case
    rng = np.random.default_rng(7)
    alp = _alp_agent(case, grb_env)
    rule = _rule_agent(case, grb_env, restricted_types=[], reserve=10)
    for _ in range(6):
        u = np.zeros(env.planning_horizon)
        u[:20] = rng.integers(95, 121, size=20)
        u[20:40] = rng.integers(0, 80, size=20)
        w = rng.integers(0, 3, size=env.num_types).astype(float)
        state = (u, np.zeros(env.planning_horizon), w)
        _, alp_action, _ = alp.solve(state, t=1)
        _, rule_action, _ = rule.solve(state, t=1)
        np.testing.assert_array_equal(np.asarray(rule_action[0]), np.asarray(alp_action[0]))
        np.testing.assert_array_equal(np.asarray(rule_action[1]), np.asarray(alp_action[1]))


def test_restricted_type_is_not_started_on_a_target_day_without_free_regular_capacity(case, grb_env):
    env, _ = case
    # Days 1-10 have no free regular capacity; day 11 onward is empty.
    state = _state(env, [env.regular_capacity] * 10, {15: 1, 6: 1})
    _, alp_action, _ = _alp_agent(case, grb_env).solve(state, t=1)
    _, rule_action, _ = _rule_agent(case, grb_env, restricted_types=LONG, reserve=0).solve(state, t=1)
    assert _start_days(alp_action[0], 15) == [10]
    rule_days = _start_days(rule_action[0], 15)
    assert len(rule_days) == 1 and rule_days[0] >= 11
    # The unrestricted five-day treatment (Type 7) is booked as under the ALP policy.
    assert _start_days(rule_action[0], 6) == _start_days(alp_action[0], 6)


@pytest.mark.parametrize('reserve', [0, 10])
def test_limit_uses_pre_decision_regular_bookings_first_session_slots_and_reserve(case, grb_env, reserve):
    env, _ = case
    first_session = int(env.treatment_pattern[0, 15])   # Type 16 starts with a 2-slot session
    full = [env.regular_capacity] * 10
    # Free regular capacity beyond the reserve on day 11 is one slot short of the first session.
    short = _state(env, full, {15: 1}, {11: env.regular_capacity - reserve - first_session + 1})
    # ... and exactly enough here.
    enough = _state(env, full, {15: 1}, {11: env.regular_capacity - reserve - first_session})
    agent = _rule_agent(case, grb_env, restricted_types=LONG, reserve=reserve)
    _, short_action, _ = agent.solve(short, t=1)
    _, enough_action, _ = agent.solve(enough, t=1)
    assert _start_days(enough_action[0], 15) == [11]
    assert _start_days(short_action[0], 15) != [11]
    assert len(_start_days(short_action[0], 15)) == 1


def test_agent_name_builds_the_variant_with_its_parameters(case, grb_env):
    from decision_maker.alp_booking_rule_agent import ALPBookingRuleAgent
    from experiments.booking_rule_eval import build_policy_agent
    env, coefficients = case
    agent = build_policy_agent(env, 'alp_booking_rule',
                               {'coefficients': coefficients, 'restricted_types': TEN_DAY, 'reserve': 20},
                               grb_env=grb_env)
    assert isinstance(agent, ALPBookingRuleAgent)
    assert list(agent.restricted_types) == TEN_DAY
    assert agent.reserve == 20
    # Other agent names still go through run.build_policy_agent.
    assert type(build_policy_agent(env, 'row_gen_alp', {'coefficients': coefficients}, grb_env=grb_env)) \
        is ALPRowGenerationAgent


def test_variant_records_keep_the_source_paths_and_add_six_variants_after_the_alp_policy():
    from experiments.booking_rule_eval import VARIANTS, variant_records
    source = [{'uid': f'u{k}', 'sample_path': [k], 'warm_up_periods': 750, 'warm_up_policy_id': 'row_gen_alp',
               'policy_specs': [{'policy_id': 'row_gen_alp', 'agent_name': 'row_gen_alp',
                                 'agent_args': {'coefficients': [1.0]}},
                                {'policy_id': 'myopic', 'agent_name': 'myopic', 'agent_args': {}}]}
              for k in range(5)]
    records = variant_records(source, experiment_name='exp', limit=3)
    assert [r['uid'] for r in records] == ['u0', 'u1', 'u2']
    for r, s in zip(records, source):
        assert r['sample_path'] == s['sample_path'] and r['warm_up_periods'] == 750
        assert r['warm_up_policy_id'] == 'row_gen_alp' and r['experiment_name'] == 'exp'
        assert r['skip_information_relaxation'] is True
        specs = r['policy_specs']
        assert specs[0] == s['policy_specs'][0]
        assert [spec['policy_id'] for spec in specs[1:]] == [v[0] for v in VARIANTS]
        for spec, (policy_id, types, reserve) in zip(specs[1:], VARIANTS):
            assert spec['agent_name'] == 'alp_booking_rule'
            assert spec['agent_args'] == {'coefficients': [1.0], 'restricted_types': types, 'reserve': reserve}
    assert len(VARIANTS) == 6
    assert {tuple(v[1]) for v in VARIANTS} == {tuple(LONG), tuple(TEN_DAY)}
    assert {v[2] for v in VARIANTS} == {0, 10, 20}


def test_stored_warm_up_records_start_from_the_saved_state_and_skip_the_warm_up():
    from experiments.booking_rule_eval import from_stored_warm_up
    record = {'uid': 'u1', 'init_state': [[0], [0], [0]], 'sample_path': [[1], [2], [3], [4]],
              'warm_up_periods': 3, 'warm_up_policy_id': 'row_gen_alp', 'period_weights': [1.0, 0.99],
              'policy_specs': [{'policy_id': 'row_gen_alp'}]}
    saved = {'u1': [[5], [6], [7]]}
    out = from_stored_warm_up([record], saved)[0]
    assert out['init_state'] == [[5], [6], [7]]
    assert out['sample_path'] == [[4]]
    assert out['warm_up_periods'] == 0
    assert out['warm_up_policy_id'] is None
    assert out['period_weights'] == [1.0, 0.99] and out['uid'] == 'u1'
    assert record['warm_up_periods'] == 3   # source record is not modified
    with pytest.raises(KeyError):
        from_stored_warm_up([{**record, 'uid': 'missing'}], saved)


def test_remaining_batches_sample_every_stratum_and_skip_finished_paths():
    from experiments.booking_rule_eval import remaining_batch
    source = [{'uid': f's{s}p{k}', 'path_stratum': s} for s in range(4) for k in range(16)]
    done = {'s0p0', 's1p3'}
    batches = [remaining_batch(source, done, n_batches=8, batch=b) for b in range(8)]
    flat = [r['uid'] for batch in batches for r in batch]
    assert len(flat) == len(set(flat)) == 64 - 2 and not (done & set(flat))
    for batch in batches:
        counts = {s: sum(r['path_stratum'] == s for r in batch) for s in range(4)}
        assert all(1 <= c <= 2 for c in counts.values())


def test_variant_records_can_restrict_variants_and_drop_the_alp_policy():
    from experiments.booking_rule_eval import variant_records
    source = [{'uid': 'u', 'policy_specs': [{'policy_id': 'row_gen_alp', 'agent_name': 'row_gen_alp',
                                             'agent_args': {'coefficients': [1.0]}}]}]
    r = variant_records(source, 'exp', variants=['alp_rule_tenday_R10', 'alp_rule_long_R10'], include_alp=False)[0]
    assert [s['policy_id'] for s in r['policy_specs']] == ['alp_rule_tenday_R10', 'alp_rule_long_R10']


def test_matrix_form_of_the_rule_matches_the_elementwise_definition(case, grb_env):
    env, _ = case
    agent = _rule_agent(case, grb_env, restricted_types=TEN_DAY, reserve=10)
    A = agent.rule_matrix()
    N, I = env.booking_window_size, env.num_types
    assert A.shape == (N, N * I)
    rng = np.random.default_rng(3)
    x = rng.integers(0, 3, size=(N, I)).astype(float)
    expected = np.array([sum(env.treatment_pattern[0, i] * x[j, i] for i in TEN_DAY) for j in range(N)])
    np.testing.assert_allclose(A @ x.reshape(-1), expected)
    u = np.zeros(env.planning_horizon)
    u[0] = env.regular_capacity            # no room: limit 0
    u[1] = env.regular_capacity - 25       # room: 25 - 10 = 15
    u[2] = env.regular_capacity - 5        # less than the reserve: limit 0
    limits = agent.rule_limits(u)
    assert limits[0] == 0 and limits[1] == 15 and limits[2] == 0
    assert limits[3] == env.regular_capacity - 10


def test_runner_accepts_the_meta_farm_job_id_and_names_output_files_after_it(monkeypatch, tmp_path):
    import experiments.booking_rule_eval as eval_module
    source = [{'uid': 'u', 'policy_specs': [{'policy_id': 'row_gen_alp', 'agent_name': 'row_gen_alp',
                                             'agent_args': {'coefficients': [1.0]}}],
               'warm_up_periods': 1, 'sample_path': [[1], [2]]}]
    monkeypatch.setattr('param_generation.command_files.read_command_file_records', lambda path: source)
    seen = []
    monkeypatch.setattr(eval_module, '_run_chunks', lambda chunks, workers: seen.extend(chunks))
    eval_module.main(['--name', 'exp', '--workers', '1', '--job_id', '123_7'])
    assert [job_id for _, job_id in seen] == ['123_7-w0']


def test_runner_reads_saved_warm_up_states_from_a_json_file(monkeypatch, tmp_path):
    import json
    import experiments.booking_rule_eval as eval_module
    source = [{'uid': 'u', 'policy_specs': [{'policy_id': 'row_gen_alp', 'agent_name': 'row_gen_alp',
                                             'agent_args': {'coefficients': [1.0]}}],
               'warm_up_periods': 1, 'sample_path': [[1], [2]], 'init_state': [[0]]}]
    states = tmp_path / 'states.json'
    states.write_text(json.dumps({'u': [[9]]}))
    monkeypatch.setattr('param_generation.command_files.read_command_file_records', lambda path: source)
    seen = []
    monkeypatch.setattr(eval_module, '_run_chunks', lambda chunks, workers: seen.extend(chunks))
    eval_module.main(['--name', 'exp', '--workers', '1', '--warm-up-states-json', str(states)])
    records = seen[0][0]
    assert all(r['init_state'] == [[9]] and r['sample_path'] == [[2]] and r['warm_up_periods'] == 0
               for r in records)
