"""Fit a value-function approximation on the toy-study training data and
evaluate the resulting greedy policy on a sample path.

Pipeline:
  1. Read every JSONL record in ``experiments/results/toy_study_train/``.
  2. ``X`` = ``init_state`` (regular bookings, overtimes, waitlist);
     ``Y`` = ``tight_penalized_lower_bound`` (the value-function target V(s)).
  3. Fit ``V_theta`` on ``(X, Y)`` via ``ApproxQAgent.regression_train``.
  4. Build the greedy policy with ``approx_Q_solve`` and read off the action
     for a given state.
  5. Roll the policy out on one sample path. The episode terminates at the
     environment's geometric stop time, so the undiscounted cost summed over
     the path is the infinite-horizon estimator (as in ``PolicyEvaluator``).
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from decision_maker import ApproxQAgent
from experiments import get_config_by_type
from generating_function import MulticlassQuadraticPenaltyFunction

TRAIN_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, 'experiments', 'results', 'toy_study_train'))


def load_training_data(directory):
    """Return ``(X, Y)``: states and their tight penalized lower bounds.

    Records are de-duplicated by ``uid`` so repeated job-array emissions of the
    same fit are not double-counted.
    """
    records = {}
    for path in sorted(glob.glob(os.path.join(directory, '*.jsonl'))):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if 'tight_penalized_lower_bound' not in record:
                    continue
                records[record.get('uid', len(records))] = record

    X, Y = [], []
    for record in records.values():
        bookings, overtimes, waitlist = record['init_state']
        X.append((np.array(bookings, dtype=float),
                  np.array(overtimes, dtype=float),
                  np.array(waitlist, dtype=float)))
        Y.append(record['tight_penalized_lower_bound'])
    return X, np.asarray(Y, dtype=float)


def main():
    config = get_config_by_type('toy')
    env = config.env

    # Steps 1-2: read the training data and build (X, Y).
    X, Y = load_training_data(TRAIN_DIR)
    print(f"Loaded {len(X)} training states from {os.path.relpath(TRAIN_DIR)}")
    print(f"target mean / std: {Y.mean():.2f} / {Y.std():.2f}")

    agent = ApproxQAgent(env,
                         discount_factor=env.discount_factor,
                         sample_path_number=1,
                         generating_function=MulticlassQuadraticPenaltyFunction(env),
                         solver_name='approx_Q')

    # Step 3: fit V_theta on (X, Y).
    coefficients, training_mse = agent.regression_train(X, Y, regularization=1e-6)
    print(f"Fitted {len(coefficients)} coefficients; train RMSE = {np.sqrt(training_mse):.4f}")

    # Step 4: greedy action for a representative state.
    demo_state = X[0]
    q_value, action, _ = agent.approx_Q_solve(demo_state, t=1)
    advance_scheduling, overtime = action
    print(f"Greedy action for the first training state (Q = {q_value:.2f}):")
    print(f"  advance scheduling decision (period x type):\n{advance_scheduling.astype(int)}")
    print(f"  overtime decision: {overtime.astype(int)}")

    # Step 5: evaluate the policy on a single sample path.
    sample_path = env.reset_arrivals()
    state, _ = env.reset(demo_state, t=1, new_arrivals=sample_path)
    total_cost, done, period = 0.0, False, 0
    while not done:
        _, action, _ = agent.approx_Q_solve(state, t=1 + period)
        state, cost, done, _ = env.step(action)
        total_cost += cost
        period += 1
    print(f"Evaluated the policy over {period} periods; total cost = {total_cost:.2f}")


if __name__ == '__main__':
    main()
