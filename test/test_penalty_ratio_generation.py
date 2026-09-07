"""Tests for the ``--penalty-ratios`` evaluation-record generation.

Run from the repo root:  python -m test.test_penalty_ratio_generation
"""
import contextlib
import io
import json
import os
import tempfile
from unittest import mock

from param_generation import datasets
from param_generation.caching import (
    load_trained_coefficient_record_from_folder,
    load_trained_coefficients_from_folder,
)
from param_generation.datasets import generate_test_paths_and_init_state, normalize_penalty_ratios
from test.test_eval_proposal_generation import make_test_envs


def _write_training_record(folder, **overrides):
    record = {
        'uid': 'fake-uid', 'experiment_name': 'eval_prop_unit_test',
        'mutate_val': 0.1, 'init_state_mode': 'generate', 'init_state_seed': 7,
        'tight_penalized_lower_bound': 123.0, 'coefficients': [0.5, -0.5, 1.0],
        'training_time_seconds': 1.0,
    }
    record.update(overrides)
    path = os.path.join(folder, '1.jsonl')
    with open(path, 'a') as handle:
        handle.write(json.dumps(record) + '\n')
    return path


def test_record_loader_returns_record_with_file():
    with tempfile.TemporaryDirectory() as tmp:
        assert load_trained_coefficient_record_from_folder('x', folder_path=tmp) is None
        path = _write_training_record(tmp)
        record = load_trained_coefficient_record_from_folder(
            'x', mutate_val=0.1, folder_path=tmp)
        assert record['uid'] == 'fake-uid' and record['file'] == path
        assert record['coefficients'] == [0.5, -0.5, 1.0]
        assert load_trained_coefficients_from_folder(
            'x', mutate_val=0.1, folder_path=tmp) == [0.5, -0.5, 1.0]
        # mutate_val filter still applies
        assert load_trained_coefficient_record_from_folder(
            'x', mutate_val=0.2, folder_path=tmp) is None


def _generate(penalty_ratios, folder, random_init=False):
    return generate_test_paths_and_init_state(
        test_envs=make_test_envs(),
        test_sample_path_num=2,
        warm_up_periods=0,
        num_periods=None,
        dat_file=None,
        is_require_penalty_coefficients=True,
        is_random_initial_state=random_init,
        policy_ids=[],
        penalty_coefficients_dir=folder,
        penalty_ratios=penalty_ratios,
    )


def test_normalize_penalty_ratios_adds_endpoints():
    assert normalize_penalty_ratios([0.5, 1.5, 0.5]) == [0.0, 0.5, 1.0, 1.5]
    assert normalize_penalty_ratios([]) == [0.0, 1.0]


def test_records_carry_grid_and_coefficients_source():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_training_record(tmp, init_state_mode='shared')
        with mock.patch.object(datasets, 'train_penalty_coefficients',
                               side_effect=AssertionError('must not train')):
            records = _generate([0.5, 1.5], tmp)
    assert len(records) == 2
    for record in records:
        assert record['penalty_ratios'] == [0.0, 0.5, 1.0, 1.5]
        assert record['policy_specs'] == []
        assert record['generating_function_spec']['coefficients'] == [0.5, -0.5, 1.0]
        source = record['coefficients_source']
        assert source['uid'] == 'fake-uid' and source['file'] == path
        assert source['init_state_mode'] == 'shared'
        assert source['tight_penalized_lower_bound'] == 123.0
        assert source['init_state_seed'] == 7


def test_records_without_grid_are_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        _write_training_record(tmp)
        records = _generate(None, tmp)
    assert all('penalty_ratios' not in r and 'coefficients_source' not in r for r in records)


def test_missing_coefficients_raise_instead_of_training():
    with tempfile.TemporaryDirectory() as tmp:
        with mock.patch.object(datasets, 'train_penalty_coefficients',
                               side_effect=AssertionError('must not train')):
            try:
                _generate([0.5], tmp)
            except ValueError as exc:
                assert 'refusing to train' in str(exc)
            else:
                raise AssertionError('expected ValueError when no coefficients exist')


def test_initial_state_mismatch_warning():
    with tempfile.TemporaryDirectory() as tmp:
        _write_training_record(tmp, init_state_mode='generate')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _generate([0.5], tmp, random_init=False)
        assert 'initial-state distribution mismatch' in out.getvalue()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            _generate([0.5], tmp, random_init=True)
        assert 'initial-state distribution mismatch' not in out.getvalue()


if __name__ == '__main__':
    test_record_loader_returns_record_with_file()
    test_normalize_penalty_ratios_adds_endpoints()
    test_records_carry_grid_and_coefficients_source()
    test_records_without_grid_are_unchanged()
    test_missing_coefficients_raise_instead_of_training()
    test_initial_state_mismatch_warning()
    print('All penalty-ratio generation tests passed.')
