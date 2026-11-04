import json

import gurobipy as gp
import pytest

from experiments import get_config_by_type
from param_generation import cli
from param_generation.caching import training_uid
from param_generation.experiment_specs import build_variation_test_env
from param_generation.registry import EXPERIMENT_SPECS

LB_DIMENSIONS = ['occupancy', 'type1_share', 'type2_target', 'type1_cost']
APPENDIX = ['treatment_pattern', 'arrival_rates', 'pattern_arrival_rates', 'booking_window', 'waiting_costs']


@pytest.fixture
def forbid_solver(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('configuration generation must not initialize a solver')

    monkeypatch.setattr(gp, 'Env', fail)
    monkeypatch.setattr(gp, 'Model', fail)


def environments(name):
    return {key[2]: value['env_args'] for key, value in build_variation_test_env(EXPERIMENT_SPECS[name]).items()}


def test_toy64_config_only_changes_the_capacity_split(forbid_solver):
    toy, toy64 = get_config_by_type('toy').args, get_config_by_type('toy64').args
    assert (toy64['regular_capacity'], toy64['overtime_capacity']) == (6, 4)
    assert (toy['regular_capacity'], toy['overtime_capacity']) == (7, 3)
    for field in set(toy) - {'regular_capacity', 'overtime_capacity', 'reset_params'}:
        assert toy64[field] == toy[field]


@pytest.mark.parametrize('dimension', LB_DIMENSIONS + APPENDIX)
def test_toy64_specs_mirror_the_toy_specs_with_the_new_capacities(forbid_solver, dimension):
    old, new = EXPERIMENT_SPECS['toy_lb_' + dimension], EXPERIMENT_SPECS['toy64_lb_' + dimension]
    assert new.val_args == old.val_args and new.config_type == 'toy64'
    for value, env_args in environments('toy64_lb_' + dimension).items():
        reference = dict(environments('toy_lb_' + dimension)[value], regular_capacity=6, overtime_capacity=4)
        if dimension != 'occupancy':
            assert json.loads(json.dumps(env_args)) == json.loads(json.dumps(reference))


@pytest.mark.parametrize('occupancy,booked', [(0.2, (2, 0)), (0.5, (5, 0)), (0.9, (6, 3))])
def test_toy64_occupancy_states_fill_regular_capacity_first(forbid_solver, occupancy, booked):
    regular, overtime, waitlist = environments('toy64_lb_occupancy')[occupancy]['reset_params']['init_state']
    assert list(regular) == [booked[0]] * 6 + [0]
    assert list(overtime) == [booked[1]] * 6 + [0]
    assert list(waitlist) == [1, 2]


def test_toy64_base_case_is_the_smallest_scaled_instance(forbid_solver):
    base = environments('toy64_lb_occupancy')[0.5]
    scaled = environments('toy_scaled')[1]
    assert training_uid(json.loads(json.dumps(base))) == training_uid(json.loads(json.dumps(scaled)))


def write_alp_cache(path):
    with path.open('w') as handle:
        for value, env_args in environments('toy64_lb_occupancy').items():
            handle.write(json.dumps({'uid': training_uid(json.loads(json.dumps(env_args))), 'result': {
                'agent_name': 'row_gen_alp', 'obj_val': 1.0, 'args': {'coefficients': [value] * 17},
                'info': {'converged': True}}}) + '\n')
    return path


def write_po_folder(folder, values):
    folder.mkdir()
    with (folder / 'po_train.jsonl').open('w') as handle:
        for value in values:
            handle.write(json.dumps({'experiment_name': 'toy64_lb_occupancy_alp', 'mutate_val': value,
                                     'sample_path_number': 512, 'coefficients': [10 * value] * 17,
                                     'tight_penalized_lower_bound': 0.0}) + '\n')
    return folder


def bounds_argv(tmp_path, *extra):
    return ['lowerbound', 'toy64_lb_occupancy', '--paths', '4', '--groups', '3', '--eval-proposal',
            '{"type": "stratified_geometric", "discount_factor_proposal": 0.99, "num_strata": 4}',
            '--penalty-function', 'absorption_alp_penalty', '--partial-ir',
            '--alp-coefficients', str(write_alp_cache(tmp_path / 'alp_train.jsonl')),
            '--dat', str(tmp_path / 'bounds.dat'), *extra]


def test_bounds_record_carries_the_po_penalty_and_the_alp_anchor_of_its_setting(forbid_solver, tmp_path):
    folder = write_po_folder(tmp_path / 'po', [0.2, 0.5, 0.9])
    records = cli.main(bounds_argv(tmp_path, '--penalty-coefficients', str(folder)))
    assert sorted({r['mutate_val'] for r in records}) == [0.2, 0.5, 0.9]
    for record in records:
        value = record['mutate_val']
        assert record['generating_function_spec']['coefficients'] == [10 * value] * 17
        assert record['alp_coefficients'] == [value] * 17
        assert len(record['partial_information_arrival_stream']) == 916


def test_bounds_need_po_coefficients_for_every_setting(forbid_solver, tmp_path):
    folder = write_po_folder(tmp_path / 'po', [0.2, 0.5])
    with pytest.raises(ValueError, match='mutate_val 0.9'):
        cli.main(bounds_argv(tmp_path, '--penalty-coefficients', str(folder)))


@pytest.mark.parametrize('regular', [6, 7, 8])
def test_toy64_capacity_split_keeps_total_capacity_and_the_base_state(forbid_solver, regular):
    assert EXPERIMENT_SPECS['toy64_lb_capacity'].val_args == [6, 7, 8]
    env_args = environments('toy64_lb_capacity')[regular]
    assert (env_args['regular_capacity'], env_args['overtime_capacity']) == (regular, 10 - regular)
    regular_bookings, overtime_bookings, waitlist = env_args['reset_params']['init_state']
    assert list(regular_bookings) == [5] * 6 + [0] and list(overtime_bookings) == [0] * 7
    assert list(waitlist) == [1, 2]


def test_toy64_capacity_split_six_four_is_the_base_case(forbid_solver):
    base = environments('toy64_lb_occupancy')[0.5]
    assert training_uid(json.loads(json.dumps(environments('toy64_lb_capacity')[6]))) == \
        training_uid(json.loads(json.dumps(base)))


def write_capacity_alp_cache(path):
    with path.open('w') as handle:
        for value, env_args in environments('toy64_lb_capacity').items():
            handle.write(json.dumps({'uid': training_uid(json.loads(json.dumps(env_args))), 'result': {
                'agent_name': 'row_gen_alp', 'obj_val': 1.0, 'args': {'coefficients': [float(value)] * 17},
                'info': {'converged': True}}}) + '\n')
    return path


def test_train_takes_each_variants_alp_fit_and_agent_overrides(forbid_solver, tmp_path):
    cache = write_capacity_alp_cache(tmp_path / 'alp_train.jsonl')
    overrides = '{"current_decision_var_type": "continuous", "save_path_certificates": true}'
    records = cli.main(['train', 'toy64_lb_capacity', '--variants', '7,8', '--penalty-function',
                        'absorption_alp_penalty', '--reset-init-state', '--init-coefficients', str(cache),
                        '--policy-centering', '--cap-policy', 'alp', '--alp-value-floor', '--sample-path-number',
                        '512', '--agent-args', overrides, '--dat', str(tmp_path / 'train.dat')])
    assert sorted(r['mutate_val'] for r in records) == [7, 8]
    for record in records:
        inner = record['agent_args']['agent_args']
        assert inner['initial_coefficients'] == [float(record['mutate_val'])] * 17
        assert inner['alp_value_floor'] and inner['save_path_certificates']
        assert inner['current_decision_var_type'] == 'continuous'
        assert record['sample_path_number'] == 512
