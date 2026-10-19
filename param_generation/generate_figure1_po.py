"""Prepare Figure 1 repeated PO inputs without fitting or initializing a solver."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shlex

from param_generation.experiment_specs import build_variation_test_env
from param_generation.registry import EXPERIMENT_SPECS
from utils import get_uid


DIMENSIONS = (
    ('occupancy', 'Initial occupancy'),
    ('type1_share', 'Type 1 arrival share'),
    ('type2_target', 'Type 2 waiting-time target'),
    ('type1_cost', 'Type 1 waiting cost'),
)


def figure1_configurations():
    configurations, display, seen = [], [], {}
    for dimension, label in DIMENSIONS:
        experiment = 'toy_lb_' + dimension
        for (_, _, value), variant in build_variation_test_env(EXPERIMENT_SPECS[experiment]).items():
            # JSON canonicalization makes tuple/list reset-state representations identical.
            env = json.loads(json.dumps(variant['env_args']))
            uid = get_uid(env)
            if uid not in seen:
                config_id = 'figure1_' + uid
                seen[uid] = config_id
                configurations.append({
                    'config_id': config_id, 'configuration': env,
                    'initial_state': env['reset_params']['init_state'],
                    'source_experiment': experiment + '_alp_alpcoef',
                })
            display.append({'label': f'{label}: {value}', 'dimension': dimension,
                            'value': value, 'config_id': seen[uid]})
    return configurations, display


def _reference(config, cache_root):
    path = Path(cache_root) / config['source_experiment'] / 'alp_train.jsonl'
    try:
        data = path.read_bytes()
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    except (OSError, ValueError) as exc:
        raise ValueError(f'invalid or missing ALP cache {path}: {exc}') from exc
    uid = get_uid(config['configuration'])
    matches = [r for r in rows if r.get('uid') == uid]
    if not matches:
        raise ValueError(f'ALP cache {path} has no matching full configuration UID {uid}')
    references = []
    for row in matches:
        result = row.get('result', {})
        info = result.get('info', {})
        violation = info.get('max_bellman_violation')
        tolerance = info.get('tolerance')
        if (result.get('agent_name') != 'row_gen_alp' or info.get('converged') is not True
                or not isinstance(violation, (float, int)) or not math.isfinite(violation)
                or not isinstance(tolerance, (float, int)) or not math.isfinite(tolerance)
                or tolerance <= 0 or violation > tolerance):
            raise ValueError(f'ALP cache lacks a valid convergence certificate: {path}, {uid}')
        coefficients = result.get('args', {}).get('coefficients')
        if (not isinstance(coefficients, list) or len(coefficients) != 17
                or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in coefficients)):
            raise ValueError(f'invalid ALP coefficients in cache {path}, {uid}')
        references.append([float(v) for v in coefficients])
    if any(values != references[0] for values in references[1:]):
        raise ValueError(f'ambiguous ALP coefficients in cache {path}, {uid}')
    return {'coefficients': references[0], 'policy': 'alp', 'source': str(path.resolve()),
            'source_sha256': hashlib.sha256(data).hexdigest(), 'source_uid': uid,
            'certificate': copy.deepcopy(matches[0]['result']['info'])}


def figure1_training_group(cache_root):
    """Load the existing variants and their certified, configuration-matched ALP fits."""
    configurations, display = figure1_configurations()
    variants = {}
    for config in configurations:
        config['reference'] = _reference(config, cache_root)
        experiment = config['source_experiment'].removesuffix('_alp_alpcoef')
        for key, variant in build_variation_test_env(EXPERIMENT_SPECS[experiment]).items():
            if get_uid(json.loads(json.dumps(variant['env_args']))) == config['config_id'][len('figure1_'):]:
                variant['env_args'] = copy.deepcopy(config['configuration'])
                variants[key] = variant
                break
    return configurations, display, variants


def build_manifest(configurations, display, records, seed, repetitions, sample_path_number):
    """Keep campaign labels and analysis metadata outside ordinary training records."""
    if len(records) != len(configurations) * repetitions:
        raise ValueError('training record count does not match configurations and repetitions')
    runs = []
    for index, record in enumerate(records):
        config = configurations[index // repetitions]
        repetition = index % repetitions + 1
        # Sampling seeds differ across repetitions; the underlying configuration does not.
        environment = copy.deepcopy(record['env_args'])
        for field in ('env_random_seed', 'arrival_random_seed', 'stop_time_random_seed'):
            environment[field] = config['configuration'][field]
        if environment != config['configuration'] or record['init_state'] != config['initial_state']:
            raise ValueError('training record order does not match the configuration manifest')
        inner = record['agent_args']['agent_args']
        command = shlex.join(['python', 'run.py', '--params',
                              json.dumps(record, sort_keys=True, allow_nan=False)])
        runs.append({
            'case_id': index + 1, 'schema_version': 1,
            'config_id': config['config_id'],
            'batch_id': f"{config['config_id']}_r{repetition:02d}", 'repetition': repetition,
            'configuration': copy.deepcopy(config['configuration']),
            'initial_state': copy.deepcopy(record['init_state']),
            'decision_set': {'first': inner['current_decision_var_type'],
                             'future': inner['future_decision_var_type']},
            'theta_domain': {'coefficient_bound': inner.get('coefficient_bound'),
                             'alp_value_floor': inner.get('alp_value_floor', False),
                             'regularization': 0, 'fixed_coefficients': inner.get('fixed_coefficients', {})},
            'reference': copy.deepcopy(config['reference']),
            'sampling': {'sample_path_number': record['sample_path_number'],
                         'proposal': copy.deepcopy(inner['sample_path_length_proposal']),
                         'tau_definition': 'arrival_length_plus_one',
                         'weight_convention': 'stratum_mass_over_count',
                         'quantile_clip': [1e-12, .999999999999]},
            'solver': {key: inner.get(key) for key in
                       ('max_iterations', 'absolute_gap_tolerance', 'workers',
                        'master_time_limit_seconds', 'subproblem_time_limit_seconds')},
            'seeds': {'env': record['env_args']['env_random_seed'],
                      'arrival': record['env_args']['arrival_random_seed'],
                      'stop_time': record['env_args']['stop_time_random_seed'],
                      'initial_state': record['init_state_seed']},
            'parameters': record, 'command': command,
        })
    return {'schema_version': 1, 'display_settings': display, 'configurations': configurations,
            'runs': runs, 'expected_batches': repetitions, 'expected_paths': sample_path_number,
            'seed_base': seed}


def write_bundle(manifest, output_bundle=None, *, dat_file=None, manifest_file=None):
    """Create new files exclusively, protecting an existing command table."""
    if output_bundle is not None:
        if dat_file is not None or manifest_file is not None:
            raise ValueError('output_bundle cannot be combined with explicit output files')
        table, metadata = Path(output_bundle) / 'table.dat', Path(output_bundle) / 'manifest.json'
    else:
        table = Path(dat_file if dat_file is not None else 'table.dat')
        metadata = Path(manifest_file) if manifest_file is not None else table.with_suffix('.manifest.json')
    if table.resolve() == metadata.resolve():
        raise ValueError('command table and manifest must have distinct paths')
    for path in (table, metadata):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'refusing to overwrite existing output: {path}')
    table.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    with metadata.open('x') as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write('\n')
    with table.open('x') as handle:
        for run in manifest['runs']:
            handle.write(f"{run['case_id']} {run['command']}\n")
