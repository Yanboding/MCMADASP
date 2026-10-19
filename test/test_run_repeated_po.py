"""Audited PO must use the existing normal training route and license pool."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import run


@pytest.mark.parametrize('as_file', [False, True])
def test_audited_record_uses_existing_training_function(monkeypatch, tmp_path, as_file):
    env = Mock()
    checkout, training = Mock(return_value=env), Mock()
    monkeypatch.setattr(run, 'acquire_grb_env', checkout)
    monkeypatch.setattr(run, 'train_penalty_coefficients_for_env', training)
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '64')
    record = {'uid': 'sample', 'init_state_mode': 'shared', 'sample_path_number': 512,
              'agent_args': {'agent_args': {'workers': 32}},
              'batch_metadata': {'batch_id': 'repeat-01'}}
    if as_file:
        source = tmp_path / 'params.json'
        source.write_text(json.dumps(record))
        args = ['--params_file', str(source)]
    else:
        args = ['--params', json.dumps(record)]
    run.main([*args, '--job_id', 'farm-17'])
    assert checkout.call_count == 33  # existing main environment + capped pool
    training.assert_called_once_with(**record, grb_env=env, grb_sub_envs=[env]*32, job_id='farm-17')


@pytest.mark.parametrize('payload', [[17], 'invalid', [], {'run_type': 'figure1_repeated_po'}])
def test_invalid_old_route_rejected_before_checkout(monkeypatch, payload):
    checkout = Mock(side_effect=AssertionError('checkout'))
    monkeypatch.setattr(run, 'acquire_grb_env', checkout)
    with pytest.raises(ValueError):
        run.main(['--params', json.dumps(payload)])
    checkout.assert_not_called()


def test_training_options_reach_existing_agent_method(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    environment = SimpleNamespace(discount_factor=.99, reset_random_seeds=Mock())
    generating = SimpleNamespace(number_of_coefficients=2)
    agent = Mock()
    agent.benders_decomposition_train.return_value = (12., np.array([1., 2.]), {'in_sample': {'mean': 12.}})
    factory = Mock(return_value=agent)
    monkeypatch.setattr(run, 'get_config_by_type', lambda *a, **k: SimpleNamespace(env=environment))
    monkeypatch.setattr(run, '_build_generating_function', lambda **k: generating)
    monkeypatch.setattr(run, 'ApproxQAgent', factory)
    monkeypatch.setattr(run, 'dispose_agent_models', Mock())
    result = run.train_penalty_coefficients_for_env(
        uid='training', experiment_name='test', mutate_val=None, sample_path_number=4,
        init_state_mode='shared', init_state=[[0], [0], [1, 1]], init_state_seed=2,
        env_args={}, agent_args={'agent_args': {'workers': 2, 'max_iterations': 1500,
            'absolute_gap_tolerance': 1e-6, 'save_path_certificates': False,
            'master_time_limit_seconds': 20., 'subproblem_time_limit_seconds': 10.}},
        training_generating_function_spec='absorption_alp_penalty',
        grb_env=None, grb_sub_envs=[], job_id='j')
    kwargs = agent.benders_decomposition_train.call_args.kwargs
    assert kwargs['max_iterations'] == 1500
    assert kwargs['absolute_gap_tolerance'] == 1e-6
    assert kwargs['master_time_limit_seconds'] == 20.
    assert kwargs['subproblem_time_limit_seconds'] == 10.
    assert 'workers' not in factory.call_args.kwargs
    assert 'max_iterations' not in factory.call_args.kwargs
    assert result['coefficients'] == [1., 2.]


def test_audit_export_roundtrip_and_no_overwrite(tmp_path):
    from experiments.po_training_audit import write_training_audit
    from experiments.repeated_po_schema import match_manifest, validate_batch
    from test.test_analyze_repeated_po import batch
    original = batch()
    metadata = {k: copy.deepcopy(v) for k, v in original.items() if k not in ('runtime', 'training', 'paths')}
    metadata['output_dir'] = str(tmp_path)
    audit = dict(original['training'], paths=original['paths'])
    result = write_training_audit(metadata, [3.], {'audit': audit}, 'farm', 2.)
    saved = json.loads((tmp_path / 'batch.json').read_text())
    validate_batch(saved)
    match_manifest(saved, {'params': {'init_state_mode': 'shared', 'batch_metadata': metadata}})
    assert result['paths'] == original['paths']
    assert result['runtime']['job_id'] == 'farm'
    assert result['training']['runtime_seconds'] == 2.
    with pytest.raises(FileExistsError):
        write_training_audit(metadata, [3.], {'audit': audit}, 'farm', 2.)


def test_audit_validates_effective_proposal_alias(tmp_path):
    from experiments.po_training_audit import validate_training_metadata
    from test.test_analyze_repeated_po import batch
    metadata = batch()
    metadata['output_dir'] = str(tmp_path)
    env = copy.deepcopy(metadata['configuration'])
    env.update(env_random_seed=100, arrival_random_seed=101, stop_time_random_seed=102)
    inner = dict(current_decision_var_type='continuous', future_decision_var_type='continuous',
                 policy_centering=True, cap_policy='alp', initial_coefficients=[2.],
                 reference_coefficients=[2.], save_path_certificates=True,
                 sample_path_length_proposal=copy.deepcopy(metadata['sampling']['proposal']),
                 **metadata['solver'])
    def validate():
        validate_training_metadata(metadata, env, {'agent_args': inner}, 4, 'shared', [1], 103)
    validate()
    inner['sample_path_proposal'] = dict(inner['sample_path_length_proposal'], discount_factor_proposal=.95)
    with pytest.raises(ValueError, match='sampling'):
        validate()
