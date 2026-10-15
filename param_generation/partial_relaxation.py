import glob
import json
import os

import numpy as np
from scipy.stats import geom

from experiments import get_config_by_type
from utils import get_uid


def default_partial_relaxation_horizon(discount_factor, quantile=0.9999):
    return int(geom.ppf(quantile, 1.0 - discount_factor))


def warm_up_states(results_dir, uids):
    wanted = set(uids)
    states = {}
    for path in sorted(glob.glob(os.path.join(results_dir, '[0-9]*.jsonl'))):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                uid = record.get('uid')
                if uid not in wanted or record.get('warmup_state') is None:
                    continue
                if states.setdefault(uid, record['warmup_state']) != record['warmup_state']:
                    raise ValueError(f'path {uid} reaches different post-warm-up states across policies in '
                                     f'{results_dir}; the partial bound needs one shared warm-up trajectory')
    return states


def partial_relaxation_records(source_records, states, experiment_name, horizon=None):
    if not source_records:
        raise ValueError('the source command file holds no evaluation records')
    if horizon is not None and horizon < 2:
        raise ValueError(f'the horizon must cover at least 2 decision periods; got {horizon}')
    envs = {}
    records = {}
    for source in source_records:
        uid = source['uid']
        if source.get('terminal') != 'absorbed' or source.get('period_weights') is None:
            raise ValueError(f"path {uid} was not evaluated over an absorbed geometric horizon (terminal "
                             f"{source.get('terminal')!r}), so its policy costs and the partial bound would "
                             'target different discounted costs')
        if uid not in states:
            raise ValueError(f'path {uid} has no post-warm-up state in the source results')
        group_id = source['group_id']
        if group_id not in envs:
            envs[group_id] = get_config_by_type('infinite_custom', args=source['env_args']).env
        env = envs[group_id]
        periods = horizon if horizon is not None else default_partial_relaxation_horizon(env.discount_factor)
        generator = env.arrival_generator
        tail = np.asarray(source['sample_path'][source['warm_up_periods']:], dtype=int).reshape(-1, generator.num_types)
        missing = periods - 1 - len(tail)
        if missing > 0:
            tail = np.concatenate([tail, generator.rvs(missing, rng=np.random.default_rng(int(uid[:15], 16)))])
        record = {
            'partial_information_relaxation': True,
            'source_uid': uid,
            'experiment_name': experiment_name,
            'mutate_val': source['mutate_val'],
            'group_id': group_id,
            'env_args': source['env_args'],
            'init_state': states[uid],
            'arrival_stream': tail[:periods - 1].tolist(),
            'path_weight': source['path_weight'],
            'path_stratum': source['path_stratum'],
        }
        record['uid'] = get_uid(record)
        records[uid] = record
    return list(records.values())
