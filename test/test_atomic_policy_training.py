"""Atomic policy training through run.py, and the commands that generate_params writes for it."""
import json
import pickle
import shlex
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from decision_maker import MyopicAgent
from experiments import get_config_by_type
from param_generation import cli, datasets
from param_generation.caching import training_uid
from param_generation.experiment_specs import build_variation_test_env
from param_generation.registry import EXPERIMENT_SPECS
import run

EXPERIMENT = 'toy64_lb_capacity'
SMALL_AGENT_ARGS = {
    'sample_path_length_proposal': {'type': 'geometric', 'discount_factor_proposal': 0.5},
    'policy_network_spec': {'hidden_sizes': [8]},
    'value_network_spec': {'hidden_sizes': [8]},
    'arrival_sample_number': 2,
    'max_iterations': 3,
    'value_epochs': 1,
    'policy_epochs': 1,
    'batch_size': 64,
    'checkpoint_every': 1,
}
RECORD_KEYS = {'uid', 'experiment_name', 'mutate_val', 'sample_path_number', 'init_state_mode', 'init_state',
               'init_state_seed', 'env_args', 'agent_args', 'atomic_policy_training'}
AGENT_ARGS_KEYS = {'sample_path_length_proposal', 'policy_network_spec', 'value_network_spec',
                   'arrival_sample_number', 'max_iterations', 'clip_parameter', 'value_epochs', 'policy_epochs',
                   'batch_size', 'value_learning_rate', 'policy_learning_rate', 'checkpoint_every', 'seed'}
STATE_KEYS = {'uid', 'record', 'policy_coefficients', 'value_coefficients', 'optimizer_state', 'iteration',
              'finished', 'log'}


def train_commands(tmp_path, *extra, agent_args=None, dat='train.dat'):
    argv = ['train', EXPERIMENT, '--method', 'atomic_ppo', '--dat', str(tmp_path / dat), *extra]
    if agent_args is not None:
        argv += ['--agent-args', json.dumps(agent_args)]
    return cli.main(argv)


def small_record(tmp_path, *extra, agent_args=SMALL_AGENT_ARGS):
    record, = train_commands(tmp_path, '--variants', '6', '--sample-path-number', '4', *extra,
                             agent_args=agent_args)
    return json.loads(json.dumps(record))


def policy_file(tmp_path, record):
    return tmp_path / 'experiments' / 'results' / record['experiment_name'] / 'atomic_policy' / f"{record['uid']}.pkl"


def read_policy_file(tmp_path, record):
    with open(policy_file(tmp_path, record), 'rb') as handle:
        return pickle.load(handle)


def run_record(record, job_id='local'):
    run.main(['--params', json.dumps(record), '--job_id', job_id])


@pytest.fixture(autouse=True)
def results_in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def forbid_gurobi(monkeypatch):
    import decision_maker.infinite_rt_agent as infinite_rt_agent

    def fail(*args, **kwargs):
        raise AssertionError('a Gurobi token was acquired')

    monkeypatch.setattr(run, 'acquire_grb_env', fail)
    monkeypatch.setattr(infinite_rt_agent, 'acquire_grb_env', fail)


# ----- generate_params train --method atomic_ppo -----

def test_training_record_has_the_keys_and_the_starting_values_of_the_spec(tmp_path):
    records = train_commands(tmp_path)
    assert [record['mutate_val'] for record in records] == [6, 7, 8]
    assert len({record['uid'] for record in records}) == 3
    for record in records:
        assert set(record) == RECORD_KEYS
        assert record['atomic_policy_training'] is True
        assert record['experiment_name'] == EXPERIMENT
        assert (record['init_state_mode'], record['init_state'], record['init_state_seed']) == ('generate', None, 12345)
        assert record['sample_path_number'] == 256
        assert (record['env_args']['regular_capacity'], record['env_args']['overtime_capacity']) == (
            record['mutate_val'], 10 - record['mutate_val'])
        assert record['agent_args']['agent_name'] == 'atomic_ppo'
        assert record['agent_args']['agent_args'] == {
            'sample_path_length_proposal': {'type': 'geometric', 'discount_factor_proposal': 0.99},
            'policy_network_spec': {'hidden_sizes': [64, 64], 'activation': 'tanh'},
            'value_network_spec': {'hidden_sizes': [64, 64], 'activation': 'tanh'},
            'arrival_sample_number': 16,
            'max_iterations': 200,
            'clip_parameter': 0.2,
            'value_epochs': 10,
            'policy_epochs': 10,
            'batch_size': 4096,
            'value_learning_rate': 0.001,
            'policy_learning_rate': 0.0003,
            'checkpoint_every': 10,
            'seed': 0,
        }
        assert set(record['agent_args']['agent_args']) == AGENT_ARGS_KEYS

    lines = (tmp_path / 'train.dat').read_text().splitlines()
    assert len(lines) == 3
    for index, (line, record) in enumerate(zip(lines, records), 1):
        tokens = shlex.split(line)
        assert tokens[:4] == [str(index), 'python', 'run.py', '--params']
        assert json.loads(tokens[4]) == json.loads(json.dumps(record))


def test_uid_covers_the_whole_training_spec(tmp_path):
    base = small_record(tmp_path)
    assert small_record(tmp_path)['uid'] == base['uid']
    changes = [
        {'value_network_spec': {'hidden_sizes': []}},
        {'policy_network_spec': {'hidden_sizes': [16]}},
        {'seed': 1},
        {'clip_parameter': 0.1},
    ]
    uids = {small_record(tmp_path, agent_args={**SMALL_AGENT_ARGS, **change})['uid'] for change in changes}
    uids.add(small_record(tmp_path, '--init-state-seed', '7')['uid'])
    assert len(uids) == 5 and base['uid'] not in uids
    record, = train_commands(tmp_path, '--variants', '6', '--sample-path-number', '8', agent_args=SMALL_AGENT_ARGS)
    assert record['uid'] != base['uid'] and record['sample_path_number'] == 8


def test_agent_args_override_only_the_named_values(tmp_path):
    record = small_record(tmp_path, agent_args={'value_network_spec': {'hidden_sizes': []}, 'seed': 3})
    inner = record['agent_args']['agent_args']
    assert inner['value_network_spec'] == {'hidden_sizes': [], 'activation': 'tanh'}
    assert inner['policy_network_spec'] == {'hidden_sizes': [64, 64], 'activation': 'tanh'}
    assert (inner['seed'], inner['max_iterations']) == (3, 200)
    with pytest.raises(ValueError, match='unknown'):
        small_record(tmp_path, agent_args={'entropy_coefficient': 0.01})


def test_fields_of_the_penalty_are_in_the_record_only_when_they_are_given(tmp_path):
    base = small_record(tmp_path)
    assert set(base['agent_args']['agent_args']) == AGENT_ARGS_KEYS
    fields = {'scenarios': 'fixed', 'penalty': True, 'arrival_expectation': 'exact'}
    record = small_record(tmp_path, agent_args={**SMALL_AGENT_ARGS, **fields})
    inner = record['agent_args']['agent_args']
    assert set(inner) == AGENT_ARGS_KEYS | set(fields)
    assert {key: inner[key] for key in fields} == fields
    assert record['uid'] != base['uid']
    for key, value in fields.items():
        one_field = small_record(tmp_path, agent_args={**SMALL_AGENT_ARGS, key: value})
        assert set(one_field['agent_args']['agent_args']) == AGENT_ARGS_KEYS | {key}
        assert one_field['uid'] not in (base['uid'], record['uid'])


def test_reset_init_state_trains_from_the_state_of_the_variant(tmp_path):
    record = small_record(tmp_path, '--reset-init-state')
    assert record['init_state_mode'] == 'shared'
    assert record['init_state'] == [[5, 5, 5, 5, 5, 5, 0], [0] * 7, [1, 2]]
    assert record['uid'] != small_record(tmp_path)['uid']


@pytest.mark.parametrize('option', [['--workers', '1'], ['--policy-centering'], ['--penalty-function', 'linear_penalty'],
                                    ['--num-init-states', '2']])
def test_atomic_training_rejects_the_options_of_penalty_training(tmp_path, option):
    with pytest.raises(ValueError, match='--method atomic_ppo'):
        train_commands(tmp_path, *option)
    assert not (tmp_path / 'train.dat').exists()


# ----- run.py: training -----

def test_training_record_runs_without_gurobi_and_writes_one_file(tmp_path, forbid_gurobi):
    record = small_record(tmp_path)
    (tmp_path / 'train.dat').unlink()
    run_record(record)
    files = [str(path.relative_to(tmp_path)) for path in tmp_path.rglob('*') if path.is_file()]
    assert files == [f"experiments/results/{EXPERIMENT}/atomic_policy/{record['uid']}.pkl"]
    saved = read_policy_file(tmp_path, record)
    assert set(saved) == STATE_KEYS
    assert saved['uid'] == record['uid']
    assert json.loads(json.dumps(saved['record'])) == record
    assert (saved['iteration'], saved['finished']) == (3, True)
    assert [row['iteration'] for row in saved['log']] == [1, 2, 3]
    for row in saved['log']:
        assert np.isfinite([row['policy_cost'], row['value_fit'], row['clipped_objective']]).all()
    assert all(isinstance(value, np.ndarray) for value in saved['policy_coefficients'].values())
    assert all(isinstance(value, np.ndarray) for value in saved['value_coefficients'].values())
    assert set(saved['optimizer_state']) == {'policy', 'value'}


def test_training_record_with_the_penalty_runs_and_is_continued(tmp_path, forbid_gurobi, monkeypatch):
    record = small_record(tmp_path, agent_args={
        **SMALL_AGENT_ARGS, 'scenarios': 'fixed', 'penalty': True, 'arrival_expectation': 'exact'})
    run_record(record)
    uninterrupted = read_policy_file(tmp_path, record)
    assert set(uninterrupted) == STATE_KEYS
    assert (uninterrupted['iteration'], uninterrupted['finished']) == (3, True)
    for row in uninterrupted['log']:
        assert np.isfinite([row['policy_cost'], row['penalized_policy_cost'], row['value_fit'],
                            row['clipped_objective']]).all()
    policy_file(tmp_path, record).unlink()

    class Interrupted(Exception):
        pass

    dump = run.atomic_pickle_dump

    def interrupted_dump(path, payload):
        dump(path, payload)
        if payload['iteration'] == 2:
            raise Interrupted

    monkeypatch.setattr(run, 'atomic_pickle_dump', interrupted_dump)
    with pytest.raises(Interrupted):
        run_record(record)
    monkeypatch.setattr(run, 'atomic_pickle_dump', dump)
    run_record(record, job_id='second')
    resumed = read_policy_file(tmp_path, record)
    assert (resumed['iteration'], resumed['finished']) == (3, True)
    for key in ('policy_coefficients', 'value_coefficients'):
        for name, value in uninterrupted[key].items():
            np.testing.assert_allclose(resumed[key][name], value, rtol=1e-12, atol=1e-12)


def test_training_starts_each_iteration_from_newly_generated_states(tmp_path, monkeypatch):
    from decision_maker.atomic_action_agent import AtomicActionAgent
    record = small_record(tmp_path)
    drawn = {}
    simulate = AtomicActionAgent.simulate

    def recording_simulate(self, iteration):
        drawn[iteration] = self.initial_state_sampler(iteration, self.sample_path_number)
        return simulate(self, iteration)

    monkeypatch.setattr(AtomicActionAgent, 'simulate', recording_simulate)
    run_record(record)
    assert sorted(drawn) == [0, 1, 2]
    # the generator and the seed that penalty training uses in the mode 'generate'
    expected = run._draw_per_scenario_init_states(
        env_args=record['env_args'], init_state_seed=record['init_state_seed'], sample_path_number=12)
    flat = [state for iteration in sorted(drawn) for state in drawn[iteration]]
    assert len(flat) == 12
    for state, expected_state in zip(flat, expected):
        for component, expected_component in zip(state, expected_state):
            np.testing.assert_array_equal(component, expected_component)


def test_shared_mode_starts_every_scenario_from_the_given_state(tmp_path, monkeypatch):
    from decision_maker.atomic_action_agent import AtomicActionAgent
    record = small_record(tmp_path, '--reset-init-state')
    batches = []
    simulate = AtomicActionAgent.simulate

    def recording_simulate(self, iteration):
        batches.append(simulate(self, iteration))
        return batches[-1]

    monkeypatch.setattr(AtomicActionAgent, 'simulate', recording_simulate)
    run_record(record)
    env = get_config_by_type('infinite_custom', args=record['env_args']).env
    agent = AtomicActionAgent(env, env.discount_factor, policy_network_spec={'hidden_sizes': [8]})
    expected = agent.state_features(tuple(np.array(component) for component in record['init_state']))
    for batch in batches:
        first_rows = np.flatnonzero(np.append(True, np.diff(batch['scenarios']) != 0))
        assert len(first_rows) == 4
        np.testing.assert_allclose(batch['features'][first_rows], np.tile(expected, (4, 1)))


def test_second_run_skips_a_finished_training(tmp_path, monkeypatch):
    from decision_maker.atomic_action_agent import AtomicActionAgent
    record = small_record(tmp_path)
    run_record(record)
    original = policy_file(tmp_path, record).read_bytes()

    def fail(self, iteration):
        raise AssertionError('a finished training was simulated again')

    monkeypatch.setattr(AtomicActionAgent, 'simulate', fail)
    run_record(record, job_id='second')
    assert policy_file(tmp_path, record).read_bytes() == original


def test_second_run_continues_an_unfinished_training(tmp_path, monkeypatch):
    record = small_record(tmp_path)
    run_record(record)
    uninterrupted = read_policy_file(tmp_path, record)
    policy_file(tmp_path, record).unlink()

    class Interrupted(Exception):
        pass

    dump = run.atomic_pickle_dump

    def interrupted_dump(path, payload):
        dump(path, payload)
        if payload['iteration'] == 1:
            raise Interrupted

    monkeypatch.setattr(run, 'atomic_pickle_dump', interrupted_dump)
    with pytest.raises(Interrupted):
        run_record(record)
    partial = read_policy_file(tmp_path, record)
    assert (partial['iteration'], partial['finished']) == (1, False)

    monkeypatch.setattr(run, 'atomic_pickle_dump', dump)
    run_record(record, job_id='second')
    resumed = read_policy_file(tmp_path, record)
    assert (resumed['iteration'], resumed['finished']) == (3, True)
    assert [row['iteration'] for row in resumed['log']] == [1, 2, 3]
    for key in ('policy_coefficients', 'value_coefficients'):
        for name, value in uninterrupted[key].items():
            np.testing.assert_allclose(resumed[key][name], value, rtol=1e-12, atol=1e-12)


def test_training_needs_a_sample_path_proposal_without_truncation(tmp_path):
    record = small_record(tmp_path, agent_args={
        **SMALL_AGENT_ARGS, 'sample_path_length_proposal': {'type': 'fixed', 'max_length': 3}})
    with pytest.raises(ValueError, match='truncated'):
        run_record(record)


def test_per_scenario_init_states_are_not_supported(tmp_path):
    record = small_record(tmp_path)
    record['init_state_mode'] = 'per_scenario'
    with pytest.raises(ValueError, match='init_state_mode'):
        run_record(record)


# ----- run.py: Gurobi tokens -----

def test_main_acquires_gurobi_tokens_before_the_first_record_that_needs_them(tmp_path, monkeypatch):
    events = []
    token = Mock()
    monkeypatch.setattr(run, 'acquire_grb_env', lambda *args, **kwargs: events.append('token') or token)
    monkeypatch.setattr(run, 'train_atomic_policy_for_env', lambda **kwargs: events.append('atomic'))
    monkeypatch.setattr(run, 'train_alp_coefficients_for_env', lambda **kwargs: events.append('alp'))
    atomic = small_record(tmp_path)
    alp = {'uid': 'u', 'experiment_name': 'unit_test_alp', 'env_args': atomic['env_args'], 'alp_training': True}

    run.main(['--params', json.dumps([atomic]), '--job_id', 'only_atomic'])
    assert events == ['atomic']
    token.dispose.assert_not_called()

    events.clear()
    run.main(['--params', json.dumps([atomic, alp, atomic, alp]), '--job_id', 'mixed'])
    assert events == ['atomic', 'token', 'alp', 'atomic', 'alp']
    token.dispose.assert_called_once()


def test_infinite_rt_agent_acquires_its_token_at_first_use(monkeypatch):
    import decision_maker.infinite_rt_agent as infinite_rt_agent
    calls = []
    token = object()
    monkeypatch.setattr(infinite_rt_agent, 'acquire_grb_env', lambda *args, **kwargs: calls.append(kwargs) or token)
    env = get_config_by_type('toy64').env
    agent = MyopicAgent(env, env.discount_factor)
    assert calls == []
    assert agent.grb_env is token
    assert agent.grb_env is token
    assert len(calls) == 1

    given = object()
    assert MyopicAgent(env, env.discount_factor, grb_env=given).grb_env is given
    assert len(calls) == 1


# ----- run.py: the trained policy as a policy of an evaluation record -----

def test_build_policy_agent_loads_the_trained_atomic_policy(tmp_path, forbid_gurobi):
    record = small_record(tmp_path)
    run_record(record)
    saved = read_policy_file(tmp_path, record)
    env = get_config_by_type('infinite_custom', args=record['env_args']).env
    agent = run.build_policy_agent(env, 'atomic_ppo', {
        'training_experiment_name': record['experiment_name'], 'training_uid': record['uid']})
    assert agent.action_selection == 'greedy'
    for name, value in saved['policy_coefficients'].items():
        np.testing.assert_array_equal(agent.policy_coefficients()[name], value)
    state = tuple(np.array(component) for component in record['env_args']['reset_params']['init_state'])
    obj, (x, y), _ = agent.solve(state, 1)
    assert any(np.array_equal(x, valid_x) and np.array_equal(y, valid_y) for valid_x, valid_y in env.valid_actions(state))
    assert obj == pytest.approx(env.cost_fn(state, (x, y)))

    sampling = run.build_policy_agent(env, 'atomic_ppo', {
        'training_experiment_name': record['experiment_name'], 'training_uid': record['uid'],
        'action_selection': 'sample', 'seed': 4})
    assert (sampling.action_selection, sampling.seed) == ('sample', 4)


def test_build_policy_agent_needs_a_finished_training(tmp_path):
    record = small_record(tmp_path)
    env = get_config_by_type('infinite_custom', args=record['env_args']).env
    agent_args = {'training_experiment_name': record['experiment_name'], 'training_uid': record['uid']}
    with pytest.raises(FileNotFoundError, match=record['uid']):
        run.build_policy_agent(env, 'atomic_ppo', agent_args)

    run_record(record)
    saved = read_policy_file(tmp_path, record)
    saved['finished'] = False
    run.atomic_pickle_dump(str(policy_file(tmp_path, record)), saved)
    with pytest.raises(ValueError, match='not finished'):
        run.build_policy_agent(env, 'atomic_ppo', agent_args)


# ----- generate_params eval with atomic policies -----

def write_alp_cache(path):
    variants = build_variation_test_env(EXPERIMENT_SPECS[EXPERIMENT])
    with path.open('w') as handle:
        for key, variant in variants.items():
            handle.write(json.dumps({'uid': training_uid(json.loads(json.dumps(variant['env_args']))), 'result': {
                'agent_name': 'row_gen_alp', 'obj_val': 1.0, 'args': {'coefficients': [float(key[2])] * 17},
                'info': {'converged': True}}}) + '\n')
    return path


def eval_commands(tmp_path, policies, *extra):
    cache = str(write_alp_cache(tmp_path / 'alp_train.jsonl'))
    return cli.main(['eval', EXPERIMENT, '--policies', policies, '--paths', '2', '--groups', '2', '--skip-ir',
                     '--eval-proposal', '{"type": "geometric", "discount_factor_proposal": 0.5}',
                     '--penalty-function', 'absorption_alp_penalty', '--penalty-coefficients', cache,
                     '--alp-policy-coefficients', cache, '--dat', str(tmp_path / 'eval.dat'), *extra])


@pytest.fixture
def forbid_local_training(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('command generation attempted local training')

    monkeypatch.setattr(datasets, 'train_alp_coefficients', fail)
    monkeypatch.setattr(datasets, 'train_penalty_coefficients', fail)


def test_eval_adds_the_atomic_policy_of_each_variant(tmp_path, forbid_local_training):
    trained = {record['mutate_val']: record for record in train_commands(tmp_path)}
    records = eval_commands(tmp_path, 'row_gen_alp,atomic_ppo')
    assert len(records) == 6
    assert sorted({record['mutate_val'] for record in records}) == [6, 7, 8]
    for record in records:
        alp, atomic = record['policy_specs']
        assert alp['policy_id'] == 'row_gen_alp'
        assert alp['agent_args']['coefficients'] == [float(record['mutate_val'])] * 17
        assert atomic == {
            'policy_id': 'atomic_ppo',
            'agent_name': 'atomic_ppo',
            'agent_args': {
                'training_experiment_name': EXPERIMENT,
                'training_uid': trained[record['mutate_val']]['uid'],
                'action_selection': 'greedy',
            },
        }
        assert record['skip_information_relaxation'] is True
        assert [list(component) for component in record['init_state']] == [[5, 5, 5, 5, 5, 5, 0], [0] * 7, [1, 2]]


def test_eval_takes_several_atomic_policies_from_the_atomic_policy_spec(tmp_path, forbid_local_training):
    affine_args = {**SMALL_AGENT_ARGS, 'value_network_spec': {'hidden_sizes': []}}
    hidden, = train_commands(tmp_path, '--variants', '7', '--sample-path-number', '4', agent_args=SMALL_AGENT_ARGS)
    affine, = train_commands(tmp_path, '--variants', '7', '--sample-path-number', '4', '--name', 'affine_run',
                             agent_args=affine_args, dat='train_affine.dat')
    assert affine['experiment_name'] == 'affine_run'
    spec = {
        'atomic_ppo': {'agent_args': SMALL_AGENT_ARGS, 'sample_path_number': 4},
        'atomic_ppo_affine': {'agent_args': affine_args, 'sample_path_number': 4, 'name': 'affine_run',
                              'action_selection': 'sample', 'seed': 2},
    }
    records = eval_commands(tmp_path, 'atomic_ppo,atomic_ppo_affine', '--variants', '7',
                            '--atomic-policy-spec', json.dumps(spec))
    assert len(records) == 2
    for record in records:
        first, second = record['policy_specs']
        assert (first['policy_id'], first['agent_name']) == ('atomic_ppo', 'atomic_ppo')
        assert first['agent_args'] == {'training_experiment_name': EXPERIMENT, 'training_uid': hidden['uid'],
                                       'action_selection': 'greedy'}
        assert (second['policy_id'], second['agent_name']) == ('atomic_ppo_affine', 'atomic_ppo')
        assert second['agent_args'] == {'training_experiment_name': 'affine_run', 'training_uid': affine['uid'],
                                        'action_selection': 'sample', 'seed': 2}


def test_atomic_policy_spec_must_name_atomic_policies_of_the_evaluation(tmp_path, forbid_local_training):
    with pytest.raises(ValueError, match='atomic_ppo_other'):
        eval_commands(tmp_path, 'row_gen_alp,atomic_ppo', '--atomic-policy-spec', '{"atomic_ppo_other": {}}')
    with pytest.raises(ValueError, match='unknown'):
        eval_commands(tmp_path, 'atomic_ppo', '--atomic-policy-spec', '{"atomic_ppo": {"learning_rate": 1}}')


def test_evaluation_record_runs_the_trained_atomic_policy(tmp_path, monkeypatch, forbid_local_training):
    record = small_record(tmp_path)
    run_record(record)
    spec = {'atomic_ppo': {'agent_args': SMALL_AGENT_ARGS, 'sample_path_number': 4}}
    records = eval_commands(tmp_path, 'atomic_ppo', '--variants', '6', '--name', 'atomic_eval',
                            '--atomic-policy-spec', json.dumps(spec))
    monkeypatch.setattr(run, 'acquire_grb_env', lambda *args, **kwargs: Mock())
    run.main(['--params', json.dumps(json.loads(json.dumps(records))), '--job_id', 'evaluation'])
    lines = (tmp_path / 'experiments' / 'results' / 'atomic_eval' / 'evaluation.jsonl').read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    assert [row['policy_id'] for row in rows] == ['atomic_ppo', 'atomic_ppo']
    for row, evaluation in zip(rows, records):
        assert row['agent_name'] == 'atomic_ppo'
        assert row['agent_args']['training_uid'] == record['uid']
        assert np.isfinite(row['total_cost']) and row['total_cost'] >= 0
        assert len(row['costs']) == len(evaluation['sample_path']) + 1
