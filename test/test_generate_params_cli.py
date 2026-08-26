"""CLI dispatch tests for ``python generate_params.py``.

Run from the repo root:  python -m test.test_generate_params_cli
"""
from unittest import mock

from param_generation import cli


def test_parser_accepts_all_subcommands():
    parser = cli.build_parser()
    args = parser.parse_args(['train', 'case_study_099_scenario_number'])
    assert args.command == 'train' and args.dat == 'table.dat'
    args = parser.parse_args(['eval', 'base_toy_study', '--policies', 'myopic',
                              '--paths', '8', '--warm-up', '3', '--skip-ir'])
    assert (args.command, args.paths, args.warm_up, args.skip_ir) == ('eval', 8, 3, True)
    args = parser.parse_args(['lowerbound', 'base_toy_study', '--paths', '16'])
    assert args.command == 'lowerbound'
    assert cli.build_parser().parse_args(['saure-ejor']).command == 'saure-ejor'
    assert cli.build_parser().parse_args(
        ['eval-proposal-comparison']).command == 'eval-proposal-comparison'


def test_train_resolves_per_variant_sample_path_number():
    captured = []

    def fake_generate(test_envs, **kwargs):
        captured.append(kwargs['sample_path_number'])
        return [{'stub': True}]

    with mock.patch.object(cli, 'generate_penalty_coefficient_training_env',
                           side_effect=fake_generate), \
         mock.patch.object(cli, 'write_command_file') as writer:
        records = cli.main(['train', 'case_study_099_scenario_number',
                            '--dat', 'unused.dat'])
    assert sorted(captured) == [64, 128, 256, 512]
    assert len(records) == 4
    writer.assert_called_once()


def test_lowerbound_dispatches_with_empty_policies():
    with mock.patch.object(cli, 'generate_test_paths_and_init_state',
                           return_value=[{'uid': 'x'}]) as generator, \
         mock.patch.object(cli, 'write_grouped_command_file'):
        cli.main(['lowerbound', 'base_toy_study', '--paths', '8', '--skip-ir'])
    kwargs = generator.call_args.kwargs
    assert kwargs['policy_ids'] == []
    assert kwargs['test_sample_path_num'] == 8


def test_policy_spec_overrides_variant_agent_args_and_guards_retrain():
    import json
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        spec_path = os.path.join(tmp, 'spec.json')
        json.dump({'approx_penalized_hindsight': {'sample_path_number': 512,
                                                   'penalty_ratio': 0.5}},
                  open(spec_path, 'w'))
        # Retrain guard: no cached coefficients for the overridden config -> exit.
        with mock.patch.object(cli, 'load_trained_coefficients_from_folder',
                               return_value=None):
            try:
                cli.main(['eval', 'base_toy_study', '--policies',
                          'approx_penalized_hindsight', '--policy-spec', spec_path])
            except SystemExit as exc:
                assert 'allow-retrain' in str(exc)
            else:
                raise AssertionError('expected SystemExit from the retrain guard')
        # With --allow-retrain the overrides reach the generator's test_envs.
        with mock.patch.object(cli, 'generate_test_paths_and_init_state',
                               return_value=[]) as generator, \
             mock.patch.object(cli, 'write_grouped_command_file'):
            cli.main(['eval', 'base_toy_study', '--policies',
                      'approx_penalized_hindsight', '--policy-spec', spec_path,
                      '--allow-retrain'])
        (variant,) = generator.call_args.kwargs['test_envs'].values()
        inner = variant['agent_args']['agent_args']
        assert inner['sample_path_number'] == 512 and inner['penalty_ratio'] == 0.5
        # Untouched keys survive the deep merge.
        assert inner['current_decision_var_type'] == 'integer'
        # Unknown policy id in the spec is rejected.
        json.dump({'ghost_policy': {'x': 1}}, open(spec_path, 'w'))
        try:
            cli.main(['eval', 'base_toy_study', '--policies', 'myopic',
                      '--policy-spec', spec_path, '--allow-retrain'])
        except ValueError as exc:
            assert 'ghost_policy' in str(exc)
        else:
            raise AssertionError('expected ValueError for unknown policy id')
        # Conflicting overrides across policy ids are rejected.
        json.dump({'approx_penalized_hindsight': {'penalty_ratio': 0.5},
                   'approx_hindsight': {'penalty_ratio': 0.7}}, open(spec_path, 'w'))
        try:
            cli.main(['eval', 'base_toy_study', '--policies',
                      'approx_penalized_hindsight,approx_hindsight',
                      '--policy-spec', spec_path, '--allow-retrain'])
        except ValueError as exc:
            assert 'conflict' in str(exc)
        else:
            raise AssertionError('expected ValueError for conflicting overrides')


def test_train_expands_init_states_and_path_seeds():
    import json
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        dat = os.path.join(tmp, 't.dat')
        records = cli.main(['train', 'base_toy_study', '--num-init-states', '3',
                            '--num-path-seeds', '2', '--sample-paths-seed', '42',
                            '--dat', dat])
    # 1 variant x 3 fixed initial states x 2 path seeds.
    assert len(records) == 6
    assert all(r['init_state_mode'] == 'shared' for r in records)
    assert sorted({r['env_args']['arrival_random_seed'] for r in records}) == [42, 43]
    # Three distinct fixed states, each paired with both seeds.
    states = {json.dumps(r['init_state'], sort_keys=True) for r in records}
    assert len(states) == 3
    # All uids distinct (state and seed both feed the uid).
    assert len({r['uid'] for r in records}) == 6
    # Same seed + same state -> reproducible: rerun yields identical uids.
    with tempfile.TemporaryDirectory() as tmp:
        again = cli.main(['train', 'base_toy_study', '--num-init-states', '3',
                          '--num-path-seeds', '2', '--sample-paths-seed', '42',
                          '--dat', os.path.join(tmp, 't.dat')])
    assert [r['uid'] for r in again] == [r['uid'] for r in records]


def test_variants_filter_selects_mutate_vals():
    with mock.patch.object(cli, 'generate_test_paths_and_init_state',
                           return_value=[]) as generator, \
         mock.patch.object(cli, 'write_grouped_command_file'):
        cli.main(['lowerbound', 'case_study_099_scenario_number',
                  '--variants', '512', '--paths', '4'])
    keys = list(generator.call_args.kwargs['test_envs'])
    assert [key[2] for key in keys] == [512]
    try:
        cli.main(['lowerbound', 'case_study_099_scenario_number',
                  '--variants', '999', '--paths', '4'])
    except ValueError as exc:
        assert '999' in str(exc)
    else:
        raise AssertionError('expected ValueError for unknown variant')


def test_train_pathwise_safety_records_carry_mode():
    with mock.patch.object(cli, 'write_command_file'):
        records = cli.main(['train', 'case_study_099_pathwise_safety', '--dat', 'unused.dat'])
    modes = sorted(str(r['agent_args']['agent_args']['pathwise_safety']) for r in records)
    assert modes == ['1.0', 'hard']
    assert all(r['sample_path_number'] == 256 for r in records)


if __name__ == '__main__':
    test_parser_accepts_all_subcommands()
    test_train_resolves_per_variant_sample_path_number()
    test_lowerbound_dispatches_with_empty_policies()
    test_policy_spec_overrides_variant_agent_args_and_guards_retrain()
    test_train_expands_init_states_and_path_seeds()
    test_variants_filter_selects_mutate_vals()
    test_train_pathwise_safety_records_carry_mode()
    print('All generate_params CLI tests passed.')
