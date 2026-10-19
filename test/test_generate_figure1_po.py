import json
import shlex

import pytest

from param_generation.generate_figure1_po import build_manifest, write_bundle, figure1_configurations
from param_generation.caching import training_uid


@pytest.fixture
def cache(tmp_path):
    configs, _ = figure1_configurations()
    for config in configs:
        folder = tmp_path / config['source_experiment']
        folder.mkdir(exist_ok=True)
        with (folder / 'alp_train.jsonl').open('a') as handle:
            handle.write(json.dumps({'uid': training_uid(config['configuration']), 'result': {
                'agent_name': 'row_gen_alp', 'args': {'coefficients': list(range(17))},
                'info': {'converged': True, 'max_bellman_violation': 1e-8, 'tolerance': 1e-6}}}) + '\n')
    return tmp_path


def test_full_configuration_dedup():
    configs, display = figure1_configurations()
    assert len(display) == 12 and len(configs) == 10
    shared = [x['config_id'] for x in display if (x['dimension'], x['value']) in
              [('occupancy', .5), ('type2_target', 1), ('type1_cost', 100)]]
    assert len(set(shared)) == 1
    share = next(c for c in configs if c['configuration']['arrival_rates'] == [.9, 2.1])
    assert share['initial_state'][2] == [1, 2]


def test_manifest_commands_and_model_invariants(cache, tmp_path):
    m = build_manifest(cache, '/server/project with space', '/server/results', seed=1977)
    assert len(m['runs']) == 300 and m['expected_batches'] == 30 and m['expected_paths'] == 512
    assert len({r['batch_id'] for r in m['runs']}) == 300
    assert len({r['output_dir'] for r in m['runs']}) == 300
    all_seeds = [s for r in m['runs'] for s in r['seeds'].values()]
    assert len(set(all_seeds)) == len(all_seeds)
    for r in m['runs']:
        p = r['params']
        assert p['decision_set'] == {'first': 'continuous', 'future': 'continuous'}
        assert p['sampling']['proposal']['num_strata'] == 256
        assert p['sampling']['sample_path_number'] == 512
        assert p['theta_domain'] == {'coefficient_bound': None, 'alp_value_floor': False,
                                     'regularization': 0, 'fixed_coefficients': {}}
        argv = shlex.split(r['command'])
        assert argv[1:3] == ['-m', 'experiments.figure1_po_run']
        assert json.loads(argv[argv.index('--params') + 1]) == p
        assert p['reference']['source_uid'] == training_uid(p['configuration'])
    target = tmp_path / 'bundle'
    write_bundle(m, target)
    assert len((target / 'table.dat').read_text().splitlines()) == 300
    assert json.loads((target / 'manifest.json').read_text()) == m
    with pytest.raises(FileExistsError):
        write_bundle(m, target)
    assert m == build_manifest(cache, '/server/project with space', '/server/results', seed=1977)


def test_missing_or_bad_cache_rejected(cache):
    file = next(cache.glob('*/alp_train.jsonl'))
    file.write_text('{broken}\n')
    with pytest.raises(ValueError, match='cache'):
        build_manifest(cache, '/project', '/results')
    file.unlink()
    with pytest.raises(ValueError, match='cache'):
        build_manifest(cache, '/project', '/results')


def test_noncertified_and_nonfinite_coefficients_rejected(cache):
    file = next(cache.glob('*/alp_train.jsonl'))
    rows = [json.loads(l) for l in file.read_text().splitlines()]
    rows[0]['result']['info']['converged'] = False
    file.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='certif'):
        build_manifest(cache, '/project', '/results')
    rows[0]['result']['info']['converged'] = True
    rows[0]['result']['args']['coefficients'][0] = float('nan')
    file.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='coefficients'):
        build_manifest(cache, '/project', '/results')


def test_generation_never_initializes_solver(cache, monkeypatch, tmp_path):
    import gurobipy
    from param_generation.generate_figure1_po import main

    def forbidden(*args, **kwargs):
        raise AssertionError('generation must not create Gurobi environments or models')

    monkeypatch.setattr(gurobipy, 'Env', forbidden)
    monkeypatch.setattr(gurobipy, 'Model', forbidden)
    m = main(['--alp-cache-root', str(cache), '--server-project', '/server/project',
              '--server-results', '/server/results', '--output-bundle', str(tmp_path / 'out')])
    assert len(m['runs']) == 300

