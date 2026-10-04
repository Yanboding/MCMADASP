import json
import os
import tempfile

import numpy as np

from experiments.new_result_aggregration import SimulateEvaluationResult


def _policy_record(uid, scheduled_patients, postponing_decisions=None):
    periods = len(scheduled_patients)
    num_types = len(scheduled_patients[0][0])
    return {
        'uid': uid, 'group_id': 'g', 'experiment_name': 'x', 'mutate_val': 0.5,
        'warm_up_periods': 0, 'path_weight': 1.0, 'path_stratum': 0,
        'policy_id': 'row_gen_alp', 'agent_name': 'row_gen_alp',
        'total_cost': 100.0,
        'costs': [1.0] * periods,
        'overtime': [0.0] * periods,
        'scheduled_patients': scheduled_patients,
        'postponing_decisions': postponing_decisions or [0] * num_types,
        'solving_time_per_state': 0.0,
        'zero_information_relaxation_cost': None,
        'penalized_information_relaxation_cost': None,
    }


def _load(records):
    from experiments import get_config_by_type
    env = get_config_by_type('toy').env
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, '1.jsonl'), 'w') as handle:
            for record in records:
                handle.write(json.dumps(record) + '\n')
        return SimulateEvaluationResult(tmp, '[0-9]*.jsonl', env, is_reuse=False)


def _one_period(bookings):
    grid = np.zeros((7, 2))
    for day, treatment_type, count in bookings:
        grid[day, treatment_type] = count
    return [grid.tolist()]


def test_bookings_on_the_target_day_are_not_violations():
    ser = _load([_policy_record('u0', _one_period([(0, 0, 4)]))])
    assert ser.waiting_time_violation[('g', 'row_gen_alp')].mean == 0.0


def test_bookings_past_the_target_day_are_violations():
    ser = _load([_policy_record('u0', _one_period([(0, 0, 3), (2, 0, 1)]))])
    assert np.isclose(ser.waiting_time_violation[('g', 'row_gen_alp')].mean, 25.0)


def test_every_type_contributes_to_the_violation_rate():
    ser = _load([_policy_record('u0', _one_period([(0, 0, 1), (3, 0, 1), (0, 1, 1), (5, 1, 1)]))])
    assert np.isclose(ser.waiting_time_violation[('g', 'row_gen_alp')].mean, 50.0)
