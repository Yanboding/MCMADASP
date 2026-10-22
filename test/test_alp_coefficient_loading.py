import json
from unittest import mock

import pytest

from param_generation import cli


def _write_records(tmp_path, *records):
    path = tmp_path / 'alp_train.jsonl'
    path.write_text(''.join(json.dumps(record) + '\n' for record in records))
    return path


def _cached_record(uid='alp-env', coefficients=None):
    return {
        'uid': uid,
        'result': {
            'agent_name': 'row_gen_alp',
            'args': {'coefficients': coefficients if coefficients is not None else [1, 2.5]},
            'info': {'converged': True},
        },
    }


def test_loads_nested_alp_cache(tmp_path):
    path = _write_records(tmp_path, _cached_record())
    assert cli._load_initial_coefficients(path) == (
        [1.0, 2.5], {'file': str(path), 'uid': 'alp-env'})


def test_preserves_legacy_first_flat_record_and_source(tmp_path):
    path = _write_records(tmp_path,
        {'uid': 'legacy', 'coefficients': [3, 4], 'mutate_val': 0.5},
        {'uid': 'another', 'coefficients': [5, 6]})
    assert cli._load_initial_coefficients(path) == (
        [3.0, 4.0], {'file': str(path), 'uid': 'legacy', 'mutate_val': 0.5})


@pytest.mark.parametrize('second', [
    _cached_record(uid='another-env'),
    _cached_record(coefficients=[3, 4]),
])
def test_rejects_ambiguous_nested_alp_cache(tmp_path, second):
    path = _write_records(tmp_path, _cached_record(), second)
    with pytest.raises(ValueError, match='multiple.*coefficient records'):
        cli._load_initial_coefficients(path)


def test_accepts_identical_repeated_cache_records(tmp_path):
    path = _write_records(tmp_path, _cached_record(), _cached_record())
    assert cli._load_initial_coefficients(path)[0] == [1.0, 2.5]


def test_rejects_file_without_coefficients(tmp_path):
    path = _write_records(tmp_path, {'uid': 'missing', 'result': {'args': {}}})
    with pytest.raises(ValueError, match='no record with a coefficients list'):
        cli._load_initial_coefficients(path)


def test_none_preserves_optional_coefficients():
    assert cli._load_initial_coefficients(None) == (None, None)


def test_train_warm_start_reads_nested_alp_cache(tmp_path):
    coefficients = [float(index) for index in range(17)]
    path = _write_records(tmp_path, _cached_record(coefficients=coefficients))
    with mock.patch.object(cli, 'generate_penalty_coefficient_training_env',
                           return_value=[{'stub': True}]) as generate, \
         mock.patch.object(cli, 'write_command_file'):
        cli.main(['train', 'base_toy_study', '--init-coefficients', str(path),
                  '--penalty-function', 'absorption_alp_penalty', '--dat', str(tmp_path / 'table.dat')])
    variant, = generate.call_args.args[0].values()
    inner = variant['agent_args']['agent_args']
    assert inner['initial_coefficients'] == coefficients
    assert inner['initial_coefficients_source'] == {'file': str(path), 'uid': 'alp-env'}


@pytest.mark.parametrize('flag,policy,field', [
    ('--alp-coefficients', 'alp_penalized_hindsight', 'alp_coefficients'),
    ('--alp-policy-coefficients', 'row_gen_alp', 'alp_policy_coefficients'),
])
def test_evaluation_coefficient_flags_read_nested_alp_cache(tmp_path, flag, policy, field):
    path = _write_records(tmp_path, _cached_record())
    with mock.patch.object(cli, 'generate_test_paths_and_init_state',
                           return_value=[{'uid': 'stub'}]) as generate, \
         mock.patch.object(cli, 'write_grouped_command_file'):
        cli.main(['eval', 'base_toy_study', '--policies', policy, flag, str(path),
                  '--dat', str(tmp_path / 'table.dat')])
    assert generate.call_args.kwargs[field] == [1.0, 2.5]
