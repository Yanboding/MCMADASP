import json
import shlex
from collections import Counter

import pytest

from param_generation import cli
from param_generation.generate_figure1_po import figure1_configurations, write_bundle
from param_generation.caching import training_uid


@pytest.fixture
def cache(tmp_path):
    configs, _ = figure1_configurations()
    for index, config in enumerate(configs):
        folder = tmp_path / 'cache' / config['source_experiment']
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'alp_train.jsonl').open('a') as handle:
            handle.write(json.dumps({'uid': training_uid(config['configuration']), 'result': {
                'agent_name': 'row_gen_alp', 'args': {'coefficients': [index + k / 10 for k in range(17)]},
                'info': {'converged': True, 'max_bellman_violation': 1e-8, 'tolerance': 1e-6}}}) + '\n')
    return tmp_path / 'cache'


def arguments(cache, tmp_path, *options):
    return ['train', 'figure1', '--alp-cache-root', str(cache),
            '--server-project', '/server/project', '--results-root', '/server/results',
            '--dat', str(tmp_path / 'table.dat'), *options]


def manifest_at(tmp_path):
    return json.loads((tmp_path / 'table.manifest.json').read_text())


def test_full_configuration_dedup():
    configs, display = figure1_configurations()
    assert len(display) == 12 and len(configs) == 10
    shared = [x['config_id'] for x in display if (x['dimension'], x['value']) in
              [('occupancy', .5), ('type2_target', 1), ('type1_cost', 100)]]
    assert len(set(shared)) == 1
    share = next(c for c in configs if c['configuration']['arrival_rates'] == [.9, 2.1])
    assert share['initial_state'][2] == [1, 2]


def test_train_group_emits_existing_training_records_and_manifest(cache, tmp_path, monkeypatch):
    import gurobipy
    from param_generation import datasets

    def forbidden(*args, **kwargs):
        raise AssertionError('generation must not initialize a solver')

    monkeypatch.setattr(gurobipy, 'Env', forbidden)
    monkeypatch.setattr(gurobipy, 'Model', forbidden)
    generated = []

    def record_generator(*args, **kwargs):
        records = datasets.generate_penalty_coefficient_training_env(*args, **kwargs)
        generated.extend(records)
        return records

    monkeypatch.setattr(cli, 'generate_penalty_coefficient_training_env', record_generator)
    records = cli.main(arguments(cache, tmp_path))
    manifest = manifest_at(tmp_path)
    assert records == generated and len(records) == 300
    assert len(manifest['display_settings']) == 12 and len(manifest['configurations']) == 10
    assert manifest['expected_batches'] == 30 and manifest['expected_paths'] == 512
    assert manifest['seed_base'] == 420000
    assert len({r['uid'] for r in records}) == 300
    counts = Counter(r['batch_metadata']['config_id'] for r in records)
    assert len(counts) == 10 and set(counts.values()) == {30}
    lines = (tmp_path / 'table.dat').read_text().splitlines()
    assert len(lines) == 300
    for index, (record, run, line) in enumerate(zip(records, manifest['runs'], lines)):
        argv = shlex.split(line)
        assert argv[:4] == [str(index + 1), 'python', 'run.py', '--params']
        assert json.loads(argv[4]) == record == run['params']
        assert shlex.split(run['command']) == argv[1:]
        assert record['init_state_mode'] == 'shared' and 'run_type' not in record
        assert record['sample_path_number'] == 512
        metadata = record['batch_metadata']
        assert 'run_type' not in metadata
        assert record['init_state'] == metadata['initial_state']
        assert metadata['seeds'] == dict(env=420000 + 4 * index, arrival=420001 + 4 * index,
                                         stop_time=420002 + 4 * index, initial_state=420003 + 4 * index)
        for stream, field in [('env', 'env_random_seed'), ('arrival', 'arrival_random_seed'),
                              ('stop_time', 'stop_time_random_seed')]:
            assert record['env_args'][field] == metadata['seeds'][stream]
        assert record['init_state_seed'] == metadata['seeds']['initial_state']
        assert metadata['output_dir'] == '/server/results/' + metadata['batch_id']
        inner = record['agent_args']['agent_args']
        assert inner['current_decision_var_type'] == inner['future_decision_var_type'] == 'continuous'
        assert inner['sample_path_length_proposal'] == metadata['sampling']['proposal']
        assert inner['initial_coefficients'] == inner['reference_coefficients'] == metadata['reference']['coefficients']
        assert inner['policy_centering'] is True and inner['cap_policy'] == 'alp'
        assert inner['save_path_certificates'] is True
        assert inner['max_iterations'] == 1500 and inner['absolute_gap_tolerance'] == 1e-6
        assert inner['workers'] == 32
        assert metadata['solver'] == dict(max_iterations=1500, absolute_gap_tolerance=1e-6, workers=32,
                                          subproblem_time_limit_seconds=None, master_time_limit_seconds=None)
        assert metadata['theta_domain'] == dict(coefficient_bound=None, alp_value_floor=False,
                                                regularization=0, fixed_coefficients={})
        assert metadata['reference']['source_uid'] == training_uid(metadata['configuration'])


def test_train_group_uses_common_options(cache, tmp_path):
    records = cli.main(arguments(cache, tmp_path, '--num-path-seeds', '2', '--sample-paths-seed', '1977',
        '--workers', '4', '--max-iterations', '50', '--absolute-gap-tolerance', '0.001',
        '--subproblem-time-limit-seconds', '60', '--master-time-limit-seconds', '120',
        '--manifest', str(tmp_path / 'custom.json')))
    manifest = json.loads((tmp_path / 'custom.json').read_text())
    assert len(records) == 20 and manifest['expected_batches'] == 2
    assert not (tmp_path / 'table.manifest.json').exists()
    assert records[-1]['batch_metadata']['seeds']['initial_state'] == 1977 + 79
    solver = records[0]['batch_metadata']['solver']
    assert solver == dict(workers=4, max_iterations=50, absolute_gap_tolerance=.001,
                         subproblem_time_limit_seconds=60., master_time_limit_seconds=120.)


@pytest.mark.parametrize('options', [
    ['--coefficient-bound', '10'], ['--alp-value-floor'], ['--fix-intercept'],
    ['--num-init-states', '2'], ['--expected-init-state'], ['--sample-path-number', '256'],
    ['--penalty-function', 'linear_penalty'], ['--cap-policy', 'penalized_hindsight'],
    ['--num-path-seeds', '0'], ['--max-iterations', '0'], ['--absolute-gap-tolerance', '0'],
    ['--sample-paths-seed', '-1'],
])
def test_group_rejects_incompatible_settings_before_writing(cache, tmp_path, options):
    with pytest.raises(ValueError):
        cli.main(arguments(cache, tmp_path, *options))
    assert not (tmp_path / 'table.dat').exists()
    assert not (tmp_path / 'table.manifest.json').exists()


@pytest.mark.parametrize('mode', ['missing', 'malformed', 'uncertified', 'nonfinite'])
def test_group_validates_real_cache_shape_before_writing(cache, tmp_path, mode):
    path = next(cache.glob('*/alp_train.jsonl'))
    if mode == 'missing':
        path.unlink()
    elif mode == 'malformed':
        path.write_text('{broken}\n')
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if mode == 'uncertified':
            rows[0]['result']['info']['converged'] = False
        else:
            rows[0]['result']['args']['coefficients'][0] = float('nan')
        path.write_text('\n'.join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError):
        cli.main(arguments(cache, tmp_path))
    assert not (tmp_path / 'table.dat').exists()


@pytest.mark.parametrize('existing', ['table.dat', 'table.manifest.json'])
def test_group_protects_existing_command_bundle(cache, tmp_path, existing):
    (tmp_path / existing).write_text('keep')
    with pytest.raises(FileExistsError):
        cli.main(arguments(cache, tmp_path))
    assert (tmp_path / existing).read_text() == 'keep'
    assert len(list(tmp_path.glob('table*'))) == 1


def test_removed_subcommand_and_helper_entrypoint():
    from param_generation import generate_figure1_po
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(['figure1-repeated-po', '--alp-cache-root', '/cache',
                                       '--server-project', '/project', '--server-results', '/results'])
    assert not hasattr(generate_figure1_po, 'main')


def test_bundle_writer_rejects_identical_output_paths_before_writing(tmp_path):
    output = tmp_path / 'same.json'
    with pytest.raises(ValueError, match='distinct'):
        write_bundle({'runs': []}, dat_file=output, manifest_file=output)
    assert not output.exists()
