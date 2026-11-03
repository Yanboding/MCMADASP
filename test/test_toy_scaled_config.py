import json

import gurobipy as gp
import pytest

from experiments import get_config_by_type
from param_generation import cli
from param_generation.experiment_specs import build_variation_test_env
from param_generation.registry import EXPERIMENT_SPECS


@pytest.fixture(autouse=True)
def forbid_solver_initialization(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('configuration generation must not initialize a solver')

    monkeypatch.setattr(gp, 'Env', fail)
    monkeypatch.setattr(gp, 'Model', fail)


def scaled_environments():
    variants = build_variation_test_env(EXPERIMENT_SPECS['toy_scaled'])
    return {key[2]: value['env_args'] for key, value in variants.items()}


def test_toy_scaled_sweeps_the_three_scales():
    assert EXPERIMENT_SPECS['toy_scaled'].val_args == [1, 5, 10]
    assert sorted(scaled_environments()) == [1, 5, 10]


@pytest.mark.parametrize('n', [1, 5, 10])
def test_toy_scaled_scales_capacity_arrivals_and_cap(n):
    env_args = scaled_environments()[n]
    assert env_args['arrival_rates'] == [n, 2 * n]
    assert env_args['regular_capacity'] == 6 * n
    assert env_args['overtime_capacity'] == 4 * n
    assert env_args['maximum_total_arrival'] == 9 * n


@pytest.mark.parametrize('n', [1, 5, 10])
def test_toy_scaled_keeps_the_other_toy_parameters(n):
    base = get_config_by_type('toy').args
    env_args = scaled_environments()[n]
    for field in ['patterns', 'booking_window_size', 'holding_cost_by_day_by_type', 'overtime_cost_by_day',
                  'postponing_cost', 'duration', 'discount_factor']:
        assert env_args[field] == base[field]


@pytest.mark.parametrize('n', [1, 5, 10])
def test_toy_scaled_reference_state_has_half_occupancy(n):
    regular, overtime, waitlist = scaled_environments()[n]['reset_params']['init_state']
    assert list(regular) == [5 * n] * 6 + [0]
    assert list(overtime) == [0] * 7
    assert list(waitlist) == [n, 2 * n]


def test_toy_scaled_builds_a_valid_environment():
    env = get_config_by_type('infinite_custom', args=scaled_environments()[10]).env
    assert (env.regular_capacity, env.overtime_capacity) == (60, 40)
    assert env.arrival_generator.maximum_arrival == 90


def write_alp_directory(directory):
    directory.mkdir()
    for n in [1, 5, 10]:
        record = {'experiment_name': 'toy_scaled', 'mutate_val': n, 'coefficients': [float(n)] * 17,
                  'tight_penalized_lower_bound': 0.0}
        (directory / f'alp_penalty_n{n}.jsonl').write_text(json.dumps(record) + '\n')
    return directory


def scaled_eval_argv(tmp_path, dat, *extra):
    alp_dir = tmp_path / 'alp' if (tmp_path / 'alp').exists() else write_alp_directory(tmp_path / 'alp')
    return ['eval', 'toy_scaled', '--policies', 'row_gen_alp,myopic,alp_penalized_hindsight',
            '--alp-coefficients', str(alp_dir), '--alp-policy-coefficients', str(alp_dir),
            '--penalty-dir', str(alp_dir), '--warm-up', '3', '--warm-up-policy', 'row_gen_alp',
            '--paths', '4', '--eval-proposal',
            '{"type": "stratified_geometric", "discount_factor_proposal": 0.99, "num_strata": 4}',
            '--penalty-function', 'absorption_alp_penalty', '--dat', str(tmp_path / dat), *extra]


def command_lines(path):
    with open(path) as handle:
        return [json.loads(line.split(" --params '", 1)[1].rsplit("'", 1)[0]) for line in handle]


def test_alp_coefficient_directory_gives_each_variant_its_own_coefficients(tmp_path):
    cli.main(scaled_eval_argv(tmp_path, 'all.dat', '--groups', '3'))
    records = [record for group in command_lines(tmp_path / 'all.dat') for record in group]
    assert sorted({record['mutate_val'] for record in records}) == [1, 5, 10]
    for record in records:
        n = float(record['mutate_val'])
        policies = {spec['policy_id']: spec['agent_args'] for spec in record['policy_specs']}
        assert policies['row_gen_alp']['coefficients'] == [n] * 17
        assert policies['alp_penalized_hindsight']['generating_function_spec']['coefficients'] == [n] * 17
        assert record['alp_coefficients'] == [n] * 17
        assert record['generating_function_spec']['coefficients'] == [n] * 17


def test_alp_coefficient_directory_needs_a_record_for_every_variant(tmp_path):
    write_alp_directory(tmp_path / 'alp')
    (tmp_path / 'alp' / 'alp_penalty_n5.jsonl').unlink()
    with pytest.raises(ValueError, match='mutate_val 5'):
        cli.main(scaled_eval_argv(tmp_path, 'all.dat'))


def test_per_variant_groups_split_each_variant_on_its_own(tmp_path):
    cli.main(scaled_eval_argv(tmp_path, 'single.dat', '--groups', '12'))
    records = [record for group in command_lines(tmp_path / 'single.dat') for record in group]
    cli.main(scaled_eval_argv(tmp_path, 'split.dat', '--groups', '1:2,5:1,10:3'))
    groups = command_lines(tmp_path / 'split.dat')
    by_variant = {n: [record for record in records if record['mutate_val'] == n] for n in [1, 5, 10]}
    expected = ([by_variant[1][i::2] for i in range(2)] + [by_variant[5]]
                + [by_variant[10][i::3] for i in range(3)])
    assert groups == expected


def test_per_variant_groups_must_name_every_variant(tmp_path):
    with pytest.raises(ValueError, match='--groups'):
        cli.main(scaled_eval_argv(tmp_path, 'split.dat', '--groups', '1:2,5:1'))


def test_eval_name_renames_the_records_and_keeps_their_uids(tmp_path):
    cli.main(scaled_eval_argv(tmp_path, 'tagged.dat', '--groups', '3'))
    cli.main(scaled_eval_argv(tmp_path, 'named.dat', '--groups', '3', '--name', 'toy_scaled'))
    tagged, named = command_lines(tmp_path / 'tagged.dat'), command_lines(tmp_path / 'named.dat')
    assert {record['experiment_name'] for group in tagged for record in group} == {'toy_scaled_alp'}
    for tagged_group, named_group in zip(tagged, named, strict=True):
        assert [{**record, 'experiment_name': 'toy_scaled'} for record in tagged_group] == named_group


def test_inline_policy_spec_matches_the_spec_file(tmp_path):
    spec = '{"alp_penalized_hindsight": {"sample_path_number": 8}}'
    (tmp_path / 'spec.json').write_text(spec)
    cli.main(scaled_eval_argv(tmp_path, 'file.dat', '--policy-spec', str(tmp_path / 'spec.json')))
    cli.main(scaled_eval_argv(tmp_path, 'inline.dat', '--policy-spec', spec))
    assert (tmp_path / 'file.dat').read_text() == (tmp_path / 'inline.dat').read_text()
