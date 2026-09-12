import copy
import glob
import json
import os
import shutil
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from param_generation.command_files import write_grouped_command_file  # noqa: E402
from param_generation.datasets import generate_test_paths_and_init_state  # noqa: E402

OUTPUT_DIR = os.path.join('outputs', 'cs_lb_2048')
TRAIN_TABLE = 'table_train_cs.dat'
DAT_FILE = 'table.dat'
EXPERIMENTS = ['cs_base', 'cs_l1_0.001', 'cs_l2_0.001', 'cs_n256', 'cs_l1_0.01', 'cs_l2_0.01']
# The four coefficient sets were trained on sample-path seed 42 (the seed
# sensitivity study reuses experiment_name cs_base with seeds 43-71).
TRAINING_SEED = 42
TEST_SAMPLE_PATH_NUM = 2048
SEED_OFFSET = 1001
EVALUATION_PROPOSAL_SPEC = {'type': 'geometric', 'discount_factor_proposal': 0.99}


def load_training_commands():
    found = {}
    with open(TRAIN_TABLE) as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = line.split(" --params '", 1)[1].rsplit("'", 1)[0]
            params = json.loads(payload)
            name = params.get('experiment_name')
            if name not in EXPERIMENTS or 'init_state_mode' not in params:
                continue
            if params['env_args'].get('env_random_seed') != TRAINING_SEED:
                continue
            if name in found:
                raise RuntimeError(f'{TRAIN_TABLE} holds two seed-{TRAINING_SEED} training commands for {name}')
            found[name] = params
    missing = [name for name in EXPERIMENTS if name not in found]
    if missing:
        raise RuntimeError(f'no seed-{TRAINING_SEED} training command in {TRAIN_TABLE} for {missing}')
    return found


def pin_coefficients(name, training_uid):
    matches = []
    for path in sorted(glob.glob(os.path.join('experiments', 'results', name, '*.jsonl'))):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get('uid') == training_uid and isinstance(record.get('coefficients'), list):
                    matches.append((path, record))
    if len(matches) != 1:
        raise RuntimeError(
            f'{name}: expected exactly one trained record with uid={training_uid} '
            f'under experiments/results/{name}, found {len(matches)}')
    source_path, record = matches[0]
    target_dir = os.path.join(OUTPUT_DIR, 'coefficients', name)
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    target_path = os.path.join(target_dir, f'{training_uid}.jsonl')
    with open(target_path, 'w') as handle:
        handle.write(json.dumps(record) + '\n')
    record = dict(record)
    record['source_file'] = source_path
    record['pinned_file'] = target_path
    return record


def main():
    training_commands = load_training_commands()

    base_env_args = training_commands[EXPERIMENTS[0]]['env_args']
    for name in EXPERIMENTS[1:]:
        if training_commands[name]['env_args'] != base_env_args:
            raise RuntimeError(f'{name}: env_args differ from {EXPERIMENTS[0]}; paths would not be shared')
    init_state = base_env_args['reset_params']['init_state']
    total_capacity = base_env_args['regular_capacity'] + base_env_args['overtime_capacity']
    booked = init_state[0][0] + init_state[1][0]
    if booked != int(round(0.5 * total_capacity)):
        raise RuntimeError(f'reset initial state books {booked} slots, not 50% of {total_capacity}')

    all_records = []
    manifest_experiments = []
    for name in EXPERIMENTS:
        training = training_commands[name]
        pinned = pin_coefficients(name, training['uid'])
        if training['init_state'] != init_state:
            raise RuntimeError(f'{name}: training init_state differs from the reset initial state')
        variant = {
            'env_args': copy.deepcopy(training['env_args']),
            'agent_args': copy.deepcopy(training['agent_args']),
        }
        inner = variant['agent_args']['agent_args']
        for key in ('generating_function_spec', 'penalized_lowerbound_generating_function_spec'):
            if inner[key]['name'] != 'absorption_linear_penalty':
                raise RuntimeError(f'{name}: {key} is {inner[key]}, not absorption_linear_penalty')
        if inner['sample_path_number'] != pinned['sample_path_number']:
            raise RuntimeError(f'{name}: sample_path_number mismatch between command and record')
        test_envs = {(training['uid'], name, training['mutate_val']): variant}
        records = generate_test_paths_and_init_state(
            test_envs=test_envs,
            test_sample_path_num=TEST_SAMPLE_PATH_NUM,
            warm_up_periods=0,
            num_periods=None,
            dat_file=None,
            is_require_penalty_coefficients=True,
            is_random_initial_state=False,
            policy_ids=[],
            evaluation_proposal_spec=EVALUATION_PROPOSAL_SPEC,
            penalty_coefficients_dir=os.path.dirname(pinned['pinned_file']),
            sample_gen_seed_offset=SEED_OFFSET,
            penalty_ratios=[],
        )
        if len(records) != TEST_SAMPLE_PATH_NUM:
            raise RuntimeError(f'{name}: generated {len(records)} records')
        for record in records:
            if record['generating_function_spec']['coefficients'] != pinned['coefficients']:
                raise RuntimeError(f'{name}: record does not carry the pinned coefficients')
            if record['coefficients_source']['uid'] != training['uid']:
                raise RuntimeError(f'{name}: coefficients_source uid mismatch')
            if [list(component) for component in record['init_state']] != init_state:
                raise RuntimeError(f'{name}: record init_state is not the 50% state')
            if record['terminal'] != 'absorbed' or record['period_weights'] is None:
                raise RuntimeError(f'{name}: record lacks absorption metadata')
        all_records.extend(records)
        manifest_experiments.append({
            'experiment_name': name,
            'training_uid': training['uid'],
            'group_id': training['uid'],
            'coefficients_file': pinned['pinned_file'],
            'coefficients_source_file': pinned['source_file'],
            'sample_path_number': pinned['sample_path_number'],
            'regularization': pinned.get('regularization'),
            'coefficient_bound': pinned.get('coefficient_bound'),
            'in_sample_tight_penalized_lower_bound': pinned['tight_penalized_lower_bound'],
            'policy_id': training['agent_args']['policy_id'],
            'results_file_pattern': os.path.join('experiments', 'results', name, '<job_id>.jsonl'),
        })

    base_records = all_records[:TEST_SAMPLE_PATH_NUM]
    for offset in range(1, len(EXPERIMENTS)):
        for index in range(TEST_SAMPLE_PATH_NUM):
            other = all_records[offset * TEST_SAMPLE_PATH_NUM + index]
            for key in ('init_state', 'sample_path', 'period_weights', 'terminal', 'path_weight', 'path_stratum'):
                if other[key] != base_records[index][key]:
                    raise RuntimeError(f'path {index} differs between experiments on {key}')
    uids = [record['uid'] for record in all_records]
    if len(set(uids)) != len(uids):
        raise RuntimeError('duplicate record uids')

    pre_existing = sum(1 for line in open(DAT_FILE) if line.strip()) if os.path.exists(DAT_FILE) else 0
    lines = write_grouped_command_file(results=all_records, num_groups=len(all_records), dat_file=DAT_FILE)
    if len(lines) != len(EXPERIMENTS) * TEST_SAMPLE_PATH_NUM:
        raise RuntimeError(f'wrote {len(lines)} commands')
    line_ranges = {}
    with open(DAT_FILE) as handle:
        for line_number, line in enumerate(handle, 1):
            payload = line.split(" --params '", 1)[1].rsplit("'", 1)[0]
            params = json.loads(payload)
            if not (isinstance(params, list) and len(params) == 1 and 'penalty_ratios' in params[0]):
                continue
            name = params[0]['experiment_name']
            low, high = line_ranges.get(name, (line_number, line_number))
            line_ranges[name] = (min(low, line_number), max(high, line_number))
    for entry in manifest_experiments:
        entry['first_line'], entry['last_line'] = line_ranges[entry['experiment_name']]
        if entry['last_line'] - entry['first_line'] + 1 != TEST_SAMPLE_PATH_NUM:
            raise RuntimeError(f"{entry['experiment_name']}: evaluation commands are not contiguous in {DAT_FILE}")

    lengths = np.array([len(record['sample_path']) for record in base_records])
    manifest = {
        'dat_file': DAT_FILE,
        'total_commands': len(lines),
        'pre_existing_commands': pre_existing,
        'commands_per_experiment': TEST_SAMPLE_PATH_NUM,
        'training_table': TRAIN_TABLE,
        'training_seed': TRAINING_SEED,
        'sample_gen_seed_offset': SEED_OFFSET,
        'sample_generation_seeds': {
            key: base_env_args[key] + SEED_OFFSET
            for key in ('env_random_seed', 'arrival_random_seed', 'stop_time_random_seed')},
        'evaluation_proposal_spec': EVALUATION_PROPOSAL_SPEC,
        'penalty_function': 'absorption_linear_penalty',
        'penalty_ratios': [0.0, 1.0],
        'warm_up_periods': 0,
        'init_state': init_state,
        'init_state_occupancy': 0.5,
        'discount_factor': base_env_args['discount_factor'],
        'shared_paths_across_experiments': True,
        'path_length': {
            'mean': float(lengths.mean()), 'min': int(lengths.min()), 'max': int(lengths.max()),
            'median': float(np.median(lengths)),
        },
        'experiments': manifest_experiments,
    }
    with open(os.path.join(OUTPUT_DIR, 'manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'init_state'}, indent=2))


if __name__ == '__main__':
    main()
