"""Summaries of the booking-rule evaluation on the same footing as the case-study table.

Costs are paired by sample path (uid) with the original records of the ALP-based
penalized hindsight, ALP and myopic policies, and confidence intervals use the same
path weights, strata and estimators as experiments/new_result_aggregration.py.
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

from experiments.new_result_aggregration import (_stratified_relative_improvement, _stratified_stats,
                                                 _uid_weights_and_strata)

PH, ALP, MYOPIC = 'approx_penalized_hindsight', 'row_gen_alp', 'myopic'


def load_records(folder, pattern, policy_ids=None):
    """{policy_id: {uid: record}} from jsonl files; the first record of a (policy, uid) pair wins."""
    records = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(folder, pattern))):
        with open(path) as handle:
            for line in handle:
                record = json.loads(line)
                if policy_ids is None or record['policy_id'] in policy_ids:
                    records[record['policy_id']].setdefault(record['uid'], record)
    return records


def paired_summary(records, policy_ids, confidence=0.95):
    """Cost, improvement over Myopic, difference to ALP and share of the ALP-to-PH gap per policy."""
    uids = sorted(set.intersection(*(set(records[p]) for p in [PH, ALP, MYOPIC, *policy_ids])))
    reference = records[ALP]
    weights, strata = _uid_weights_and_strata(uids, {u: reference[u].get('path_weight') for u in uids},
                                              {u: reference[u].get('path_stratum') for u in uids})
    cost = {p: np.array([records[p][u]['total_cost'] for u in uids], dtype=float) for p in {PH, ALP, MYOPIC, *policy_ids}}
    rows = {}
    for p in policy_ids:
        stats = _stratified_stats(cost[p], weights, strata)
        diff = _stratified_stats(cost[p] - cost[ALP], weights, strata)
        rows[p] = {
            'n': len(uids),
            'cost': (stats.mean, stats.half_window(confidence)),
            'minus_alp': (diff.mean, diff.half_window(confidence)),
            'improvement_over_myopic_pct': _stratified_relative_improvement(
                cost[MYOPIC] - cost[p], cost[MYOPIC], weights, strata, confidence),
            'gap_share_pct': _stratified_relative_improvement(
                cost[ALP] - cost[p], cost[ALP] - cost[PH], weights, strata, confidence),
        }
    return rows


def within_target_by_type(records, late):
    """Path-weighted percentage of post-warm-up bookings of each type that start on or before the target day.

    ``late`` is the boolean (booking day x type) array env.waiting_cost_coeffs > 0.
    """
    late = np.asarray(late, dtype=bool)
    on_time = np.zeros(late.shape[1])
    booked = np.zeros(late.shape[1])
    for record in records.values():
        weight = float(record.get('path_weight') or 1.0)
        bookings = np.asarray(record['scheduled_patients'][record['warm_up_periods']:], dtype=float)
        if bookings.size == 0:
            continue
        total = bookings.sum(axis=0)
        on_time += weight * (total * ~late).sum(axis=0)
        booked += weight * total.sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return 100 * on_time / booked


def cost_split(records, waiting_cost_coeffs, period_weights):
    """Path-weighted mean discounted waiting cost and remaining cost (overtime and postponement).

    The waiting cost of a period is waiting_cost_coeffs * bookings; the rest of that period's cost
    is the recorded period cost minus it. Period weights are those of the evaluation record.
    """
    coeffs = np.asarray(waiting_cost_coeffs, dtype=float)
    waiting_total = other_total = weight_total = 0.0
    for uid, record in records.items():
        weight = float(record.get('path_weight') or 1.0)
        w = record['warm_up_periods']
        bookings = np.asarray(record['scheduled_patients'][w:], dtype=float)
        costs = np.asarray(record['costs'][w:], dtype=float)
        discount = np.asarray(period_weights[uid], dtype=float)[:len(costs)]
        waiting = np.array([(coeffs * b).sum() for b in bookings[:len(costs)]])
        waiting_total += weight * float(discount @ waiting)
        other_total += weight * float(discount @ (costs - waiting))
        weight_total += weight
    return waiting_total / weight_total, other_total / weight_total
