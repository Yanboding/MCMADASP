"""Persist certificates returned by the existing coefficient-training method.

This module does not construct models, allocate licenses, or optimize coefficients.
"""
import copy
import json
import os
from pathlib import Path
import subprocess

from .repeated_po_schema import validate_batch


METADATA_KEYS = ('schema_version', 'config_id', 'batch_id', 'repetition', 'configuration',
                 'initial_state', 'decision_set', 'theta_domain', 'seeds', 'reference',
                 'sampling', 'solver')


def validate_training_metadata(metadata, env_args, agent_args, sample_path_number,
                               init_state_mode, init_state, init_state_seed):
    """Ensure the audit describes the ordinary training record actually executed."""
    inner = agent_args.get('agent_args', {})
    expected_env = copy.deepcopy(metadata['configuration'])
    for key, field in (('env', 'env_random_seed'), ('arrival', 'arrival_random_seed'),
                       ('stop_time', 'stop_time_random_seed')):
        expected_env[field] = metadata['seeds'][key]
    if expected_env != env_args or init_state_seed != metadata['seeds']['initial_state']:
        raise ValueError('Training seeds/configuration disagree with batch metadata')
    if init_state_mode != 'shared' or init_state != metadata['initial_state']:
        raise ValueError('Audited training requires the recorded shared initial state')
    if metadata['decision_set'] != {'first': 'continuous', 'future': 'continuous'}:
        raise ValueError('Audited comparisons require all-period continuous decisions')
    if any(inner.get(key) != 'continuous' for key in ('current_decision_var_type', 'future_decision_var_type')):
        raise ValueError('Actual decision types disagree with audit')
    domain = dict(coefficient_bound=inner.get('coefficient_bound'),
                  alp_value_floor=bool(inner.get('alp_value_floor', False)),
                  regularization=0, fixed_coefficients=inner.get('fixed_coefficients') or {})
    if (domain != metadata['theta_domain'] or domain['coefficient_bound'] is not None
            or domain['alp_value_floor'] or domain['fixed_coefficients']
            or inner.get('coefficient_bound_overrides')):
        raise ValueError('Audit requires the unrestricted coefficient domain')
    if (not inner.get('policy_centering') or inner.get('cap_policy') != 'alp'
            or metadata['reference']['policy'] != 'alp'
            or inner.get('initial_coefficients') != metadata['reference']['coefficients']
            or inner.get('reference_coefficients') != metadata['reference']['coefficients']
            or not inner.get('save_path_certificates')):
        raise ValueError('Audit reference, fixed policy, or certificate option disagrees')
    if (sample_path_number != metadata['sampling']['sample_path_number']
            or inner.get('sample_path_proposal', inner.get('sample_path_length_proposal'))
                != metadata['sampling']['proposal']):
        raise ValueError('Actual sampling differs from batch metadata')
    for key, value in metadata['solver'].items():
        if inner.get(key) != value:
            raise ValueError(f'Actual solver option differs from audit: {key}')
    output = Path(metadata['output_dir']) / 'batch.json'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing result {output}')


def _runtime(job_id):
    import numpy as np
    import gurobipy as gp
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(git_commit=commit, job_id=job_id, numpy_version=np.__version__,
                gurobi_version='.'.join(map(str, gp.gurobi.version())))


def _write_exclusive(path, result):
    """Publish only a complete JSON file, without replacing another batch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with temporary.open('x') as stream:
            json.dump(result, stream, allow_nan=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic, fails if the destination already exists
    finally:
        temporary.unlink(missing_ok=True)


def write_training_audit(metadata, coefficients, info, job_id, elapsed, subproblem_environment_count=None):
    audit = copy.deepcopy(info['audit'])
    result = {key: copy.deepcopy(metadata[key]) for key in METADATA_KEYS}
    result['paths'] = audit.pop('paths')
    audit.update(coefficients=list(map(float, coefficients)), runtime_seconds=float(elapsed))
    result.update(training=audit, runtime=_runtime(job_id),
                  sampling_observed=info.get('importance_sampling'))
    if subproblem_environment_count is not None:
        result['runtime']['subproblem_environment_count'] = int(subproblem_environment_count)
        result['runtime']['total_environment_count'] = int(subproblem_environment_count) + 1
    validate_batch(result)
    _write_exclusive(Path(metadata['output_dir']) / 'batch.json', result)
    return result
