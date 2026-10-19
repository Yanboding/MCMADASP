"""Prepare Figure 1 repeated PO inputs without fitting or initializing a solver."""
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shlex
import sys

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


def build_manifest(cache_root, server_project, server_results, seed=420000,
                   max_iterations=1000, absolute_gap_tolerance=1e-5,
                   subproblem_time_limit_seconds=None, master_time_limit_seconds=None, workers=32):
    """Return a complete reproducible command bundle; no experiment runs here."""
    if seed < 0 or seed + 1200 >= 2**32:
        raise ValueError('seed range must fit unsigned 32-bit generator seeds')
    if workers < 1 or max_iterations < 1 or not absolute_gap_tolerance > 0:
        raise ValueError('workers, max_iterations and absolute_gap_tolerance must be positive')
    for limit in (subproblem_time_limit_seconds, master_time_limit_seconds):
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            raise ValueError('time limits must be positive finite values')
    if not PurePosixPath(server_project).is_absolute() or not PurePosixPath(server_results).is_absolute():
        raise ValueError('server project and result roots must be absolute paths')
    configurations, display = figure1_configurations()
    runs = []
    for config in configurations:
        config['reference'] = _reference(config, cache_root)
        for repetition in range(1, 31):
            index = len(runs)
            batch_id = f"{config['config_id']}_r{repetition:02d}"
            seeds = dict(zip(('env', 'arrival', 'stop_time', 'initial_state'),
                             range(seed + 4 * index, seed + 4 * index + 4)))
            params = {
                'run_type': 'figure1_repeated_po',
                'schema_version': 1, 'config_id': config['config_id'], 'batch_id': batch_id,
                'repetition': repetition, 'configuration': copy.deepcopy(config['configuration']),
                'initial_state': copy.deepcopy(config['initial_state']),
                'decision_set': {'first': 'continuous', 'future': 'continuous'},
                'theta_domain': {'coefficient_bound': None, 'alp_value_floor': False,
                                 'regularization': 0, 'fixed_coefficients': {}},
                'seeds': seeds, 'reference': copy.deepcopy(config['reference']),
                'sampling': {'sample_path_number': 512, 'proposal': {
                    'type': 'stratified_geometric', 'discount_factor_proposal': .99, 'num_strata': 256},
                    'tau_definition': 'arrival_length_plus_one',
                    'weight_convention': 'stratum_mass_over_count',
                    'quantile_clip': [1e-12, .999999999999]},
                'solver': {'max_iterations': max_iterations,
                           'absolute_gap_tolerance': absolute_gap_tolerance,
                           'subproblem_time_limit_seconds': subproblem_time_limit_seconds,
                           'master_time_limit_seconds': master_time_limit_seconds, 'workers': workers},
                'output_dir': str(PurePosixPath(server_results) / batch_id),
            }
            command = shlex.join(['python', 'run.py', '--params',
                                  json.dumps(params, sort_keys=True, allow_nan=False)])
            runs.append({'case_id': index + 1, 'batch_id': batch_id, 'config_id': config['config_id'],
                         'repetition': repetition, 'seeds': seeds, 'output_dir': params['output_dir'],
                         'params': params, 'command': command})
    return {'schema_version': 1, 'display_settings': display, 'configurations': configurations,
            'runs': runs, 'expected_batches': 30, 'expected_paths': 512,
            'server_project': str(server_project), 'server_results': str(server_results),
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


def main(argv=None):
    """Compatibility entrypoint; the shared CLI owns argument parsing."""
    from param_generation.cli import main as cli_main

    return cli_main(['figure1-repeated-po', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    main()
