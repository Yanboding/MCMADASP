"""ALP command generation must not fit coefficients or sample training paths."""
import json
import shlex

import gurobipy as gp
import pytest

from param_generation import cli, datasets, training
from param_generation.caching import training_uid


@pytest.fixture(autouse=True)
def forbid_local_training(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('command generation attempted local training or solver creation')

    for module in (cli, datasets, training):
        monkeypatch.setattr(module, 'train_alp_coefficients', forbidden)
    monkeypatch.setattr(cli, 'generate_penalty_coefficient_training_env', forbidden)
    monkeypatch.setattr(gp, 'Model', forbidden)
    monkeypatch.setattr(gp, 'Env', forbidden)


def read_commands(path):
    records = []
    for index, line in enumerate(path.read_text().splitlines(), 1):
        tokens = shlex.split(line)
        assert tokens[:4] == [str(index), 'python', 'run.py', '--params']
        assert len(tokens) == 5
        records.append(json.loads(tokens[4]))
    return records


def test_alp_generation_uses_runner_schema_and_env_uid(tmp_path):
    path = tmp_path / 'table_alp_lb_train.dat'
    records = cli.main(['train', 'toy_lb_type1_cost', '--method', 'alp',
                        '--variants', '100', '--name', 'alp_only', '--dat', str(path)])
    assert read_commands(path) == json.loads(json.dumps(records))
    assert len(records) == 1
    record = records[0]
    assert set(record) == {'uid', 'experiment_name', 'env_args', 'alp_training'}
    assert record['alp_training'] is True
    assert record['experiment_name'] == 'alp_only'
    assert record['uid'] == training_uid(record['env_args'])
    assert record['env_args']['holding_cost_by_day_by_type'][1] == [100, 10]


def test_alp_generation_retains_all_selected_variants(tmp_path):
    path = tmp_path / 'multi.dat'
    records = cli.main(['train', 'toy_lb_type1_cost', '--method', 'alp', '--dat', str(path)])
    assert len(records) == 3
    assert len({record['uid'] for record in records}) == 3
    assert {record['experiment_name'] for record in records} == {'toy_lb_type1_cost'}
    assert read_commands(path) == json.loads(json.dumps(records))


@pytest.mark.parametrize('option', [
    ['--sample-path-number', '512'], ['--num-path-seeds', '1'],
    ['--init-state-seed', '12345'], ['--workers', '1'], ['--policy-centering'],
    ['--min-norm-slack=0'],
])
def test_alp_rejects_explicit_po_options_even_at_default_values(tmp_path, option):
    path = tmp_path / 'invalid.dat'
    with pytest.raises(ValueError, match='--method alp'):
        cli.main(['train', 'base_toy_study', '--method', 'alp', '--dat', str(path), *option])
    assert not path.exists()


def test_alp_rejects_figure1_po_group(tmp_path):
    with pytest.raises(ValueError, match='figure1'):
        cli.main(['train', 'figure1', '--method', 'alp', '--dat', str(tmp_path / 'invalid.dat')])


def test_alp_name_requires_one_variant(tmp_path):
    with pytest.raises(ValueError, match='single variant'):
        cli.main(['train', 'toy_lb_type1_cost', '--method', 'alp',
                  '--name', 'ambiguous', '--dat', str(tmp_path / 'invalid.dat')])


def test_six_alp_commands_accumulate_once(tmp_path):
    names = ['toy_lb_treatment_pattern', 'toy_lb_arrival_rates',
             'toy_lb_pattern_arrival_rates', 'toy_lb_booking_window',
             'toy_lb_waiting_costs', 'small_lb']
    path = tmp_path / 'table_alp_lb_train.dat'
    for name in names:
        cli.main(['train', name, '--method', 'alp', '--dat', str(path)])
    cli.main(['train', names[0], '--method', 'alp', '--dat', str(path)])
    records = read_commands(path)
    assert [record['experiment_name'] for record in records] == names
    assert len({record['uid'] for record in records}) == 6
    assert all(record['uid'] == training_uid(record['env_args']) for record in records)
    assert all(set(record) == {'uid', 'experiment_name', 'env_args', 'alp_training'}
               for record in records)
