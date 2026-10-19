"""Exercise command routing without starting an optimization or license checkout."""
import json
from unittest.mock import Mock

import pytest
import run


@pytest.mark.parametrize('as_list', [False, True])
def test_repeated_payload_forwarding_without_legacy_environments(monkeypatch, as_list):
    backend = Mock(return_value={'training': 'saved'})
    monkeypatch.setattr(run, 'run_repeated_po', backend, raising=False)
    checkout = Mock(side_effect=AssertionError('legacy checkout must not run'))
    monkeypatch.setattr(run, 'acquire_grb_env', checkout)
    record = {'run_type': 'figure1_repeated_po', 'batch_id': 'example'}
    payload = [record, dict(record, batch_id='second')] if as_list else record
    run.main(['--params', json.dumps(payload), '--job_id', 'farm-17'])
    expected = payload if as_list else [payload]
    assert backend.call_count == len(expected)
    assert [call.args for call in backend.call_args_list] == [(p, 'farm-17') for p in expected]
    checkout.assert_not_called()


def test_repeated_params_file(monkeypatch, tmp_path):
    backend = Mock()
    monkeypatch.setattr(run, 'run_repeated_po', backend, raising=False)
    monkeypatch.setattr(run, 'acquire_grb_env', Mock(side_effect=AssertionError('checkout')))
    record = {'run_type': 'figure1_repeated_po', 'batch_id': 'file'}
    source = tmp_path / 'params.json'
    source.write_text(json.dumps(record))
    run.main(['--params_file', str(source), '--job_id', 'from-file'])
    backend.assert_called_once_with(record, 'from-file')


@pytest.mark.parametrize('payload', [
    [{'run_type': 'figure1_repeated_po'}, {'alp_training': True}],
    [17], 'invalid', [], {'run_type': 'typo'}, {'run_type': []},
])
def test_invalid_or_mixed_payload_rejected_before_checkout(monkeypatch, payload):
    checkout = Mock(side_effect=AssertionError('checkout'))
    monkeypatch.setattr(run, 'acquire_grb_env', checkout)
    with pytest.raises(ValueError):
        run.main(['--params', json.dumps(payload)])
    checkout.assert_not_called()


def test_legacy_dispatch_uses_existing_environment_pool(monkeypatch):
    environment = object()
    checkout = Mock(return_value=environment)
    training = Mock()
    monkeypatch.setattr(run, 'acquire_grb_env', checkout)
    monkeypatch.setattr(run, 'train_alp_coefficients_for_env', training)
    record = {'alp_training': True, 'uid': 'legacy', 'experiment_name': 'test', 'env_args': {}}
    run.main(['--params', json.dumps(record), '--job_id', 'old-job'])
    checkout.assert_called_once()
    training.assert_called_once_with(uid='legacy', experiment_name='test', env_args={},
                                    grb_env=environment, grb_sub_envs=[], job_id='old-job')


def test_backend_compatibility_exports():
    from experiments import figure1_po_run, repeated_po_training
    assert figure1_po_run.run is repeated_po_training.run
    assert figure1_po_run.validate_params is repeated_po_training.validate_params
    assert figure1_po_run.atomic_result is repeated_po_training.atomic_result
