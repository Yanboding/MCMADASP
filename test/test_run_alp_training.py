"""Verify ALP persistence and dispatch without solving an experiment."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from param_generation import training
from param_generation.caching import training_uid
from param_generation.experiment_specs import build_variation_test_env
from param_generation.registry import EXPERIMENT_SPECS
import run


@pytest.fixture
def alp_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (_, variant), = build_variation_test_env(EXPERIMENT_SPECS['base_toy_study']).items()
    env_args = variant['env_args']
    coefficients = [float(i) for i in range(17)]
    agent = SimpleNamespace(
        train=Mock(return_value=(123.0, coefficients)),
        rg_solver=SimpleNamespace(converged=True, max_violation=1e-9, iterations=4),
    )
    factory = Mock(return_value=agent)
    monkeypatch.setattr(training, 'ALPRowGenerationAgent', factory)
    return env_args, coefficients, agent, factory


def read_cache(tmp_path, name='unit_test_alp'):
    folder = tmp_path / 'experiments' / 'results' / name
    assert sorted(path.name for path in folder.iterdir()) == ['alp_train.jsonl']
    lines = (folder / 'alp_train.jsonl').read_text().splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_handler_saves_one_file_and_reuses_it_across_job_ids(alp_run, tmp_path):
    env_args, coefficients, agent, factory = alp_run
    kwargs = dict(uid=training_uid(env_args), experiment_name='unit_test_alp',
                  env_args=env_args, grb_env=None, grb_sub_envs=[])
    assert run.train_alp_coefficients_for_env(**kwargs, job_id='local_1') == (123.0, coefficients)
    record = read_cache(tmp_path)
    assert record['uid'] == training_uid(env_args)
    assert record['result']['obj_val'] == 123.0
    assert record['result']['args']['coefficients'] == coefficients
    info = record['result']['info']
    assert info['converged'] is True
    assert info['training_time_seconds'] >= 0
    assert (info['iterations'], info['tolerance']) == (4, training.ALP_BELLMAN_TOLERANCE)
    original = (tmp_path / 'experiments/results/unit_test_alp/alp_train.jsonl').read_bytes()

    assert run.train_alp_coefficients_for_env(**kwargs, job_id='server_42') == (123.0, coefficients)
    assert training.certified_alp_coefficients(env_args, 'unit_test_alp') == (123.0, coefficients, info)
    assert (tmp_path / 'experiments/results/unit_test_alp/alp_train.jsonl').read_bytes() == original
    factory.assert_called_once()
    agent.train.assert_called_once_with(debug=False, tol=training.ALP_BELLMAN_TOLERANCE, verbose=False)


def test_dispatch_routes_alp_training_records_to_one_cache(alp_run, tmp_path, monkeypatch):
    env_args, coefficients, agent, factory = alp_run
    env = Mock()
    acquire = Mock(return_value=env)
    monkeypatch.setattr(run, 'acquire_grb_env', acquire)
    params = {'uid': training_uid(env_args), 'experiment_name': 'unit_test_alp',
              'env_args': env_args, 'alp_training': True}
    run.main(['--params', json.dumps(params), '--job_id', 'dispatch'])
    record = read_cache(tmp_path)
    assert record['result']['args']['coefficients'] == coefficients
    assert record['result']['info']['training_time_seconds'] >= 0
    acquire.assert_called_once()
    env.dispose.assert_called_once()


def test_old_cache_without_timing_is_reused_without_retraining(alp_run, tmp_path):
    env_args, coefficients, agent, factory = alp_run
    folder = tmp_path / 'experiments/results/unit_test_alp'
    folder.mkdir(parents=True)
    record = {'uid': training_uid(env_args), 'result': {
        'agent_name': 'row_gen_alp', 'obj_val': 123.0,
        'args': {'coefficients': coefficients, 'agent_args': {}},
        'info': {'converged': True},
    }}
    path = folder / 'alp_train.jsonl'
    original = json.dumps(record) + '\n'
    path.write_text(original)
    assert run.train_alp_coefficients_for_env(
        uid=record['uid'], experiment_name='unit_test_alp', env_args=env_args,
        grb_env=None, grb_sub_envs=[], job_id='new_job') == (123.0, coefficients)
    assert path.read_text() == original
    read_cache(tmp_path)
    factory.assert_not_called()


def test_unconverged_fit_writes_no_cache(alp_run, tmp_path):
    env_args, _, agent, _ = alp_run
    agent.rg_solver.converged = False
    with pytest.raises(ValueError, match='did not converge'):
        run.train_alp_coefficients_for_env(
            uid=training_uid(env_args), experiment_name='unit_test_alp', env_args=env_args,
            grb_env=None, grb_sub_envs=[], job_id='failed')
    assert not list(tmp_path.rglob('*.jsonl'))
