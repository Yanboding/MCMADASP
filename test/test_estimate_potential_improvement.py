import json
import os

import numpy as np

import run
from experiments import get_config_by_type
from param_generation.potential_improvement import generate_potential_improvement_records
from test.toy_potential_improvement import TOY_ALP_COEFFICIENTS, TOY_ENV_ARGS
from utils import get_uid


def _toy_records(tmp_path, size=2, T=3):
    penalty_dir = tmp_path / 'alp'
    penalty_dir.mkdir()
    (penalty_dir / 'alp_penalty.jsonl').write_text(json.dumps({
        'uid': 'toy', 'mutate_val': 0.5, 'coefficients': TOY_ALP_COEFFICIENTS,
        'tight_penalized_lower_bound': 0.0}) + '\n')
    test_envs = {(get_uid(TOY_ENV_ARGS), 'toy', 0.5): {'env_args': TOY_ENV_ARGS}}
    return generate_potential_improvement_records(
        test_envs, size, {"type": "geometric", "discount_factor_proposal": 0.9}, str(penalty_dir), 4004, True, 'toy_improvement_test', prefix_periods=T)


def test_worker_records_schedule_minus_baseline_and_skips_duplicates(tmp_path, monkeypatch):
    records = _toy_records(tmp_path)
    monkeypatch.chdir(tmp_path)
    record = dict(records[0])
    record.pop('potential_improvement')
    result = run.estimate_potential_improvement(**record, grb_env=None, grb_sub_envs=[], job_id='1')
    assert set(result) == {
        'uid', 'group_id', 'penalized_information_relaxation_cost', 'partial_information_relaxation_cost',
        'potential_gap', 'prefix_partial_information_relaxation_cost', 'tail_error_bound',
        'schedule_penalized_information_relaxation_cost', 'run_time_seconds', 'path_weight', 'path_stratum'}
    assert (result['schedule_penalized_information_relaxation_cost']
            >= result['penalized_information_relaxation_cost'] - 1e-6)
    np.testing.assert_allclose(result['partial_information_relaxation_cost'],
                               result['prefix_partial_information_relaxation_cost'] + result['tail_error_bound'])
    np.testing.assert_allclose(result['potential_gap'], result['partial_information_relaxation_cost']
                               - result['penalized_information_relaxation_cost'])
    env = get_config_by_type('infinite_custom', args=TOY_ENV_ARGS).env
    stream = np.asarray(record['arrival_stream'])
    baseline = run.information_relaxation_bounds(
        env, record['generating_function_spec'], [1.0], tuple(np.array(c) for c in record['init_state']),
        stream[:record['baseline_length']], record['baseline_period_weights'], record['baseline_terminal'],
        None, [], remove_path_noise=True)[1.0]
    np.testing.assert_allclose(result['penalized_information_relaxation_cost'], baseline, rtol=1e-9)
    assert run.estimate_potential_improvement(**record, grb_env=None, grb_sub_envs=[], job_id='1') is None
    with open(os.path.join('experiments', 'results', 'toy_improvement_test', '1.jsonl')) as handle:
        lines = [json.loads(line) for line in handle]
    assert lines == [result]
