import json
import os
import shutil
import subprocess
import sys
import uuid

from experiments import get_config_by_type
from param_generation.experiment_specs import build_variation_test_env
from param_generation.registry import EXPERIMENT_SPECS
import run


def toy_env_args():
    (_, variant), = build_variation_test_env(EXPERIMENT_SPECS['base_toy_study']).items()
    return variant['env_args']


def test_handler_trains_caches_and_skips_duplicates():
    env_args = toy_env_args()
    experiment_name = f'unit_test_alp_{uuid.uuid4().hex[:8]}'
    folder = os.path.join('experiments', 'results', experiment_name)
    try:
        record = run.train_alp_coefficients_for_env(
            uid='alp-uid', experiment_name=experiment_name, env_args=env_args,
            grb_env=None, grb_sub_envs=[], job_id='job')
        env = get_config_by_type('infinite_custom', args=env_args).env
        assert len(record['coefficients']) == 1 + 2 * env.planning_horizon + env.num_types
        job_record = json.loads(open(os.path.join(folder, 'job.jsonl')).readline())
        assert job_record['uid'] == 'alp-uid'
        assert job_record['coefficients'] == record['coefficients']
        assert job_record['alp_objective'] == record['alp_objective']
        cached = json.loads(open(os.path.join(folder, 'alp_train.jsonl')).readline())
        assert cached['result']['args']['coefficients'] == record['coefficients']
        assert run.train_alp_coefficients_for_env(
            uid='alp-uid', experiment_name=experiment_name, env_args=env_args,
            grb_env=None, grb_sub_envs=[], job_id='job') is None
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_dispatch_routes_alp_training_records():
    env_args = toy_env_args()
    experiment_name = f'unit_test_alp_{uuid.uuid4().hex[:8]}'
    folder = os.path.join('experiments', 'results', experiment_name)
    params = {'uid': 'alp-uid', 'experiment_name': experiment_name, 'env_args': env_args, 'alp_training': True}
    try:
        completed = subprocess.run(
            [sys.executable, 'run.py', '--params', json.dumps(params), '--job_id', 'dispatch'],
            capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr[-2000:]
        job_record = json.loads(open(os.path.join(folder, 'dispatch.jsonl')).readline())
        assert job_record['experiment_name'] == experiment_name
        assert os.path.exists(os.path.join(folder, 'alp_train.jsonl'))
    finally:
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == '__main__':
    test_handler_trains_caches_and_skips_duplicates()
    test_dispatch_routes_alp_training_records()
    print('All ALP training command tests passed.')
