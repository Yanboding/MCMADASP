"""Prepare Figure 1 repeated PO inputs without fitting or initializing a solver."""
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
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


def batch_metadata(config, repetition, index, seed, results_root, solver):
    """Describe one ordinary training record without selecting an execution route."""
    batch_id = f"{config['config_id']}_r{repetition:02d}"
    return {
        'schema_version': 1, 'config_id': config['config_id'], 'batch_id': batch_id,
        'repetition': repetition, 'configuration': copy.deepcopy(config['configuration']),
        'initial_state': copy.deepcopy(config['initial_state']),
        'decision_set': {'first': 'continuous', 'future': 'continuous'},
        'theta_domain': {'coefficient_bound': None, 'alp_value_floor': False,
                         'regularization': 0, 'fixed_coefficients': {}},
        'seeds': dict(zip(('env', 'arrival', 'stop_time', 'initial_state'),
                          range(seed + 4 * index, seed + 4 * index + 4))),
        'reference': copy.deepcopy(config['reference']),
        'sampling': {'sample_path_number': 512, 'proposal': {
            'type': 'stratified_geometric', 'discount_factor_proposal': .99, 'num_strata': 256},
            'tau_definition': 'arrival_length_plus_one',
            'weight_convention': 'stratum_mass_over_count',
            'quantile_clip': [1e-12, .999999999999]},
        'solver': copy.deepcopy(solver),
        'output_dir': str(PurePosixPath(results_root) / batch_id),
    }


def build_manifest(configurations, display, records, server_project, results_root, seed,
                   repetitions, sample_path_number):
    """Index records already produced by the standard training generator."""
    runs = []
    for index, record in enumerate(records):
        metadata = record['batch_metadata']
        command = shlex.join(['python', 'run.py', '--params',
                              json.dumps(record, sort_keys=True, allow_nan=False)])
        runs.append({'case_id': index + 1,
                     **{key: copy.deepcopy(metadata[key]) for key in
                        ('batch_id', 'config_id', 'repetition', 'seeds', 'output_dir')},
                     'params': record, 'command': command})
    return {'schema_version': 1, 'display_settings': display, 'configurations': configurations,
            'runs': runs, 'expected_batches': repetitions, 'expected_paths': sample_path_number,
            'server_project': str(server_project), 'server_results': str(results_root),
            'seed_base': seed, 'execution_working_directory': str(server_project)}


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
