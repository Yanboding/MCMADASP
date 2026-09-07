"""``--coefficient-bound``: CLI tagging and the box constraint in training.

Run from the repo root:  python -m test.test_coefficient_bound
"""
import os
import shutil
from unittest import mock

import numpy as np

import run
from param_generation import cli


def _capture_train(argv):
    captured = []

    def fake_generate(test_envs, **kwargs):
        captured.append(test_envs)
        return [{'stub': True}]

    with mock.patch.object(cli, 'generate_penalty_coefficient_training_env', side_effect=fake_generate), \
         mock.patch.object(cli, 'write_command_file'):
        cli.main(argv)
    return captured


def test_flag_tags_variant_and_composes_with_regularization():
    base = ['train', 'case_study_099_occupancy_l01_scenario_256', '--variants', '0.5', '--reset-init-state',
            '--dat', 'unused.dat']
    for envs in _capture_train(base + ['--coefficient-bound', '1000']):
        for (uid, name, mutate_val), variant in envs.items():
            assert name == 'case_study_099_occupancy_l01_scenario_256_cb1000', name
            assert variant['agent_args']['policy_id'].endswith('_cb1000')
            assert variant['agent_args']['agent_args']['coefficient_bound'] == 1000.0
            assert mutate_val == 0.5
    names = sorted(key[1] for envs in _capture_train(
        base + ['--coefficient-bound', '1000', '--regularization', 'l1', '--regularization-lambda', '0.01,0.001'])
        for key in envs)
    assert names == ['case_study_099_occupancy_l01_scenario_256_cb1000_l1_0_001',
                     'case_study_099_occupancy_l01_scenario_256_cb1000_l1_0_01'], names
    for envs in _capture_train(base):
        for (uid, name, _), variant in envs.items():
            assert name == 'case_study_099_occupancy_l01_scenario_256'
            assert 'coefficient_bound' not in variant['agent_args']['agent_args']
    assert cli._coefficient_bound_tag(100000.0) == 'cb100000' and cli._coefficient_bound_tag(2.5) == 'cb2_5'
    try:
        cli.main(base + ['--coefficient-bound', '0'])
    except ValueError:
        pass
    else:
        raise AssertionError('a non-positive bound must be rejected')


def test_new_grid_points_are_registered():
    from param_generation.registry import EXPERIMENT_SPECS
    for name in ('case_study_099_occupancy_l01_scenario_256', 'case_study_099_occupancy_l03_scenario_256',
                 'case_study_099_occupancy_l05_scenario_256', 'case_study_099_occupancy_l01_scenario_512',
                 'case_study_099_occupancy_l01_scenario_1024'):
        assert name in EXPERIMENT_SPECS, name


def test_runner_enforces_the_bound_and_records_it():
    """Toy end-to-end: the emitted command trains with |theta| <= 5 and the
    record carries the bound; an unbounded run of the same command exceeds it."""
    with mock.patch.object(cli, 'write_command_file'):
        bounded = cli.main(['train', 'base_toy_study', '--coefficient-bound', '5', '--dat', 'unused.dat'])
        unbounded = cli.main(['train', 'base_toy_study', '--dat', 'unused.dat'])
    results = []
    for record in (bounded[0], unbounded[0]):
        record = dict(record)
        record['sample_path_number'] = 4
        # Fresh result folders: never touch experiments/results/base_toy_study.
        record['experiment_name'] = record['experiment_name'] + '_unittest'
        folder = os.path.join('experiments', 'results', record['experiment_name'])
        assert not os.path.exists(folder), folder
        try:
            out = run.train_penalty_coefficients_for_env(**record, grb_env=None, grb_sub_envs=None, job_id='cb_test')
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        results.append(out)
    bounded_out, unbounded_out = results
    assert bounded_out['experiment_name'] == 'base_toy_study_cb5_unittest' and bounded_out['coefficient_bound'] == 5.0
    assert unbounded_out['coefficient_bound'] is None
    assert np.max(np.abs(bounded_out['coefficients'])) <= 5.0 + 1e-6, np.max(np.abs(bounded_out['coefficients']))
    assert np.max(np.abs(unbounded_out['coefficients'])) > 5.0
    # A tighter feasible set cannot give a larger (maximised) lower bound.
    assert bounded_out['tight_penalized_lower_bound'] <= unbounded_out['tight_penalized_lower_bound'] + 1e-6
    assert 'training_time_seconds' in bounded_out


if __name__ == '__main__':
    test_flag_tags_variant_and_composes_with_regularization()
    test_new_grid_points_are_registered()
    test_runner_enforces_the_bound_and_records_it()
    print('All coefficient-bound tests passed.')
