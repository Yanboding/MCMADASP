"""Read/write helpers for cached training results and regression data."""

import glob
import json
import os

import numpy as np

from utils import get_uid


def training_uid(env_args, agent_args=None):
    """Cache uid that includes agent_args (e.g. proposal spec) when present.

    When agent_args is None or empty, falls back to the env-only uid for
    backward compatibility with previously cached records.
    """
    if not agent_args:
        return get_uid(env_args)
    return get_uid({'env_args': env_args, 'agent_args': agent_args})


def load_cached_training_result(experiment_name, file_name, env_args, agent_args):
    # useful parameters for coefficient training
    file_path = os.path.join('experiments', 'results', experiment_name, file_name)
    if not os.path.exists(file_path):
        return None
    target_uid = training_uid(env_args, agent_args)
    print(f"Looking for cached training result with uid={target_uid} in {file_path}...")
    cached_record = None
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            print(record)
            if record.get('uid') == target_uid:
                cached_record = record
    print('cached_record')
    print(cached_record)
    return cached_record


def save_training_result(experiment_name, file_name, env_args, agent_name, obj_val, coefficients, info=None, agent_args=None):
    file_path = os.path.join('experiments', 'results', experiment_name, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    record = {
        'uid': training_uid(env_args, agent_args),
        'result': {
            'agent_name': agent_name,
            'obj_val': obj_val,
            'args': {'coefficients': coefficients, 'agent_args': agent_args or {}},
            'info': info or {},
        },
    }
    with open(file_path, 'a') as f:
        f.write(json.dumps(record) + '\n')
    return record


def load_trained_coefficients_from_folder(
    experiment_name,
    mutate_val=None,
    sample_path_number=None,
    folder_path=None,
):
    """Load one trained coefficient vector from JSONL training outputs.

    The expected record format is the per-init-state output written by
    ``run.py::train_lowerbound_for_init_state`` with keys like
    ``tight_penalized_lower_bound`` and ``coefficients``.
    """
    search_dir = folder_path or os.path.join('experiments', 'results', experiment_name)
    if not os.path.isdir(search_dir):
        return None

    candidates = []
    for file_path in sorted(glob.glob(os.path.join(search_dir, '*.jsonl'))):
        with open(file_path, 'r') as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                coefficients = record.get('coefficients')
                if not isinstance(coefficients, list) or len(coefficients) == 0:
                    continue
                if 'tight_penalized_lower_bound' not in record:
                    continue
                if mutate_val is not None and record.get('mutate_val') != mutate_val:
                    continue
                if (
                    sample_path_number is not None
                    and 'sample_path_number' in record
                    and record.get('sample_path_number') != sample_path_number
                ):
                    continue

                init_state_index = record.get('init_state_index')
                sort_index = init_state_index if isinstance(init_state_index, int) else 10**9
                candidates.append((sort_index, file_path, line_number, coefficients))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def load_regression_training_data(train_data_dir):
    """Load (X, Y) value-function regression data from a folder of JSONL files.

    Each record contributes one (state, target) pair: the state ``X`` is the
    record's ``init_state`` (a 3-block ``(regular_bookings, overtimes,
    waitlist)`` state) and the target ``Y`` is its
    ``tight_penalized_lower_bound``. Records are de-duplicated by ``uid`` and
    records without a usable target are skipped.

    Returns ``(X, Y)`` where ``X`` is a list of ``(np.ndarray, np.ndarray,
    np.ndarray)`` states and ``Y`` is a 1-D ``np.ndarray`` of targets.
    """
    records = {}
    for file_path in sorted(glob.glob(os.path.join(train_data_dir, '*.jsonl'))):
        with open(file_path, 'r') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get('tight_penalized_lower_bound') is None or 'init_state' not in record:
                    continue
                records[record.get('uid', len(records))] = record
    X, Y = [], []
    for record in records.values():
        state = tuple(np.array(block, dtype=float) for block in record['init_state'])
        X.append(state)
        Y.append(float(record['tight_penalized_lower_bound']))
    if not X:
        raise ValueError(
            "No training records with 'init_state' and 'tight_penalized_lower_bound' "
            f"found in {train_data_dir}."
        )
    return X, np.asarray(Y, dtype=float)
