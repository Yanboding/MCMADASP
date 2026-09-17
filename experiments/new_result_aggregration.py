import csv
import glob
import json
import os
import re
from pprint import pprint
import pickle

import numpy as np
from collections import defaultdict

from scipy.stats import geom

from utils import RunningStats, StratifiedRunningStats, load_pickle_if_exists
from visualization import approximate_value_plot_from_running_stats_dict
from visualization.line_plot import approximate_value_plot_from_running_stats

# --- 1. Top-level Factory Functions (Required for Pickling) ---

AGGREGATION_VERSION = 5
BOOKING_POLICY_ORDER = (
    ('approx_penalized_hindsight', 'Penalized Hindsight', '#2a78d6'),
    ('row_gen_alp', 'ALP', '#eb6834'),
    ('myopic', 'Myopic', '#1baf7a'),
)


def booking_day_ranks(shares):
    ranks = np.zeros(shares.shape, dtype=int)
    for column in range(shares.shape[1]):
        share = shares[:, column]
        used = np.flatnonzero(share > 0)
        order = used[np.argsort(-share[used], kind='stable')]
        ranks[order, column] = np.arange(1, len(order) + 1)
    return ranks


def preference_runs(ranks, min_length=3):
    runs = []
    start = 0
    while start < len(ranks):
        if ranks[start] == 0:
            start += 1
            continue
        end = start
        step = 0
        while end + 1 < len(ranks) and ranks[end + 1] != 0:
            delta = int(ranks[end + 1]) - int(ranks[end])
            if delta not in (1, -1) or (step and delta != step):
                break
            step = delta
            end += 1
        if end - start + 1 >= min_length:
            runs.append((start, end) if step == 1 else (end, start))
        start = end + 1 if end > start else start + 1
    return runs


def dd_int_factory():
    return defaultdict(int)

def dd_rs_factory():
    return defaultdict(RunningStats)

def dd_dd_rs_factory():
    return defaultdict(dd_rs_factory)

def dd_float_factory():
    return defaultdict(float)

def dd_dd_float_factory():
    return defaultdict(dd_float_factory)

def dd_dd_dd_float_factory():
    return defaultdict(dd_dd_float_factory)


EXPERIMENT_PLOT_CONFIGS = [
    {
        'name': 'initial_state_congestion',
        'scale': 100,
        'xlabel': 'Initial State Congestion (%)',
    },
    {
        'name': 'high_priority_proportion',
        'scale': 100,
        'xlabel': 'High Priority Proportion (%)',
    },
    {
        'name': 'low_priority_waiting_time_target',
        'scale': None,
        'xlabel': 'Low Priority Waiting Time Target (Days)',
    },
    {
        'name': 'high_priority_waiting_time_penalty',
        'scale': None,
        'xlabel': 'High Priority Waiting Time Penalty (Cost)',
    },
    {
        'name': 'total_arrival_rate',
        'scale': None,
        'xlabel': 'Total Arrival Rate (Mean)',
    },
    {
        'name': 'overtime_cost',
        'scale': None,
        'xlabel': 'Overtime Cost (Cost)',
    },
    {
        'name': 'type_1_treatment_pattern',
        'scale': None,
        'xlabel': 'Type 1 Treatment Pattern (Pattern)',
    },
]

IMPROVEMENT_YLABEL = "Lower Bound Improvement (%)"

class SimulateEvaluationResult:

    _CACHE_KEYS = (
        'policy_costs',
        'zero_penalized_gap',
        'penalized_gap',
        'zero_penalized_information_relaxation_cost',
        'penalized_information_relaxation_cost',
        'gap_to_information_relaxation',
        'one_time_cost_by_policy',
        'after_warmup_policy_costs',
        'waiting_time_target_ptc_by_day',
        'waiting_time_target_ptc_by_type_day',
        'waiting_time_violation',
        'overtime_utilization',
        'solving_time_per_state',
        'uids_by_policy',
        'after_warmup_cost_by_uid',
        'cost_by_uid',
        'weight_by_uid',
        'stratum_by_uid',
        'ir_zero_by_uid',
        'ir_penalized_by_uid',
        'ir_weight_by_uid',
        'ir_stratum_by_uid',
        'ir_cost_by_ratio_by_uid',
        'ir_coefficients_source',
        'booking_day_counts',
    )

    def __init__(self,directory_path, file_pattern, env, group_ids=None, is_reuse=False):
        self.directory_path = directory_path
        self.file_pattern = file_pattern
        self.env = env
        self.waiting_time_targets = np.array([env.holding_cost.get_waiting_target(i) for i in range(env.num_types)])
        self.group_ids = group_ids if group_ids is not None else []
        self.is_reuse = is_reuse
        self.scenario_results = dd_dd_dd_float_factory()
        
        # Built-in types (int, list) and Classes (RunningStats) are already picklable.
        self.policy_costs = defaultdict(StratifiedRunningStats)
        self.zero_penalized_information_relaxation_cost = defaultdict(StratifiedRunningStats)
        self.penalized_information_relaxation_cost = defaultdict(StratifiedRunningStats)
        self.zero_penalized_gap = defaultdict(StratifiedRunningStats)
        self.penalized_gap = defaultdict(StratifiedRunningStats)
        self.zero_penalized_improvement = defaultdict(RunningStats)
        self.penalized_improvement = defaultdict(RunningStats)

        # Equal-weight by scope decision: operational metrics below ignore the
        # stratified path weights for now (revisit if the stratified proposal
        # is used for operational reporting).
        self.after_warmup_policy_costs = defaultdict(RunningStats)
        self.after_warmup_cost_by_uid = defaultdict(dict)
        self.cost_by_uid = defaultdict(dict)
        self.weight_by_uid = defaultdict(dict)
        self.stratum_by_uid = defaultdict(dict)
        self.ir_zero_by_uid = defaultdict(dict)
        self.ir_penalized_by_uid = defaultdict(dict)
        self.ir_weight_by_uid = defaultdict(dict)
        self.ir_stratum_by_uid = defaultdict(dict)
        self.ir_cost_by_ratio_by_uid = defaultdict(dict)
        self.ir_coefficients_source = {}
        self.booking_day_counts = {}
        self.waiting_time_target_ptc_by_type_day = defaultdict(dd_dd_rs_factory)
        self.waiting_time_target_ptc_by_day = defaultdict(dd_rs_factory)
        self.waiting_time_violation = defaultdict(RunningStats)
        
        self.overtime_utilization = defaultdict(RunningStats)
        self.postponement_rate = defaultdict(RunningStats)
        
        self.gap_to_information_relaxation = defaultdict(StratifiedRunningStats)
        self.improvement = defaultdict(RunningStats)

        self.one_time_cost_by_policy = dd_dd_rs_factory()
        self.solving_time_per_state = defaultdict(RunningStats)
        self.uids_by_policy = defaultdict(set)
        self.number_of_periods = None
        self.information_relaxation_id = None

        pickle_file = os.path.join(self.directory_path, 'simulate_evaluation_result', 'scenario_results.pickle')
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        cached_data = load_pickle_if_exists(pickle_file)
        if self.is_reuse and self._has_valid_cache(cached_data):
            self._load_from_cache(cached_data)
        else:
            self._load_from_jsonl()
            self._save_cache(pickle_file)
        for (group_id, policy_id), stats in self.zero_penalized_gap.items():
            self.zero_penalized_improvement[(group_id, policy_id)] = self.zero_penalized_gap[(group_id, policy_id)] / self.zero_penalized_information_relaxation_cost[(group_id, policy_id)].mean / 0.01
        for (group_id, policy_id), stats in self.penalized_gap.items():
            self.penalized_improvement[(group_id, policy_id)] = self.penalized_gap[(group_id, policy_id)] / self.penalized_information_relaxation_cost[(group_id, policy_id)].mean / 0.01
        for (group_id, mutate_val), stats in self.gap_to_information_relaxation.items():
            self.improvement[(group_id, mutate_val)] = self.gap_to_information_relaxation[(group_id, mutate_val)] / self.zero_penalized_information_relaxation_cost[(group_id, mutate_val)].mean / 0.01
    
    def _has_valid_cache(self, data):
        if data is None:
            return False
        # Version gate: stale equal-weight caches must rebuild instead of
        # silently serving pre-stratification numbers.
        if data.get('aggregation_version') != AGGREGATION_VERSION:
            return False
        return all(key in data for key in self._CACHE_KEYS)

    def _load_from_cache(self, data):
        for key in self._CACHE_KEYS:
            setattr(self, key, data[key])

    def _load_from_jsonl(self):
        pattern = os.path.join(self.directory_path, self.file_pattern)
        jsonl_files = sorted(glob.glob(pattern))
        for file_path in jsonl_files:
            with open(file_path, 'r') as f:
                for line in f:
                    self.load(json.loads(line))

    def _save_cache(self, pickle_file):
        res = {key: getattr(self, key) for key in self._CACHE_KEYS}
        res['aggregation_version'] = AGGREGATION_VERSION
        with open(pickle_file, 'wb') as f:
            pickle.dump(res, f)

        
    def load(self, data):
        # Results folders can also hold penalty-coefficient training records
        # (``coefficients`` / ``tight_penalized_lower_bound``); those carry no
        # ``policy_id`` and are not evaluation records.
        if 'policy_id' not in data:
            return
        uid = data.get('uid')
        policy_id = data['policy_id']
        group_id = data['group_id']
        mutate_val = data['mutate_val']
        # Dedup by (uid, mutate_val): experiments that mutate only the agent
        # (e.g. the mixing-probability sweep) can produce identical sample
        # paths -- hence identical uids -- across variants, and those are
        # distinct records, not duplicates.
        if uid is not None and (uid, mutate_val) in self.uids_by_policy[policy_id]:
            return
        if uid is not None:
            self.uids_by_policy[policy_id].add((uid, mutate_val))
        path_weight = data.get('path_weight')
        path_stratum = data.get('path_stratum')
        if policy_id == 'information_relaxation_only':
            self.gap_to_information_relaxation[(group_id, mutate_val)].record(
                data['penalized_information_relaxation_cost'] - data['zero_information_relaxation_cost'],
                path_weight, path_stratum)
            self.zero_penalized_information_relaxation_cost[(group_id, mutate_val)].record(
                data['zero_information_relaxation_cost'], path_weight, path_stratum)
            self.penalized_information_relaxation_cost[(group_id, mutate_val)].record(
                data['penalized_information_relaxation_cost'], path_weight, path_stratum)
            if uid is not None:
                key = (group_id, mutate_val)
                self.ir_zero_by_uid[key][uid] = data['zero_information_relaxation_cost']
                self.ir_penalized_by_uid[key][uid] = data['penalized_information_relaxation_cost']
                self.ir_weight_by_uid[key][uid] = path_weight
                self.ir_stratum_by_uid[key][uid] = path_stratum
                by_ratio = data.get('information_relaxation_cost_by_penalty_ratio')
                if by_ratio:
                    self.ir_cost_by_ratio_by_uid[key][uid] = {
                        float(ratio): float(cost) for ratio, cost in by_ratio}
                    if data.get('coefficients_source') is not None:
                        self.ir_coefficients_source[key] = data['coefficients_source']
            return
        if self.group_ids and group_id not in self.group_ids:
            self.group_ids.append(group_id)
        self.policy_costs[(group_id, policy_id)].record(
            data['total_cost'], path_weight, path_stratum)
        if uid is not None:
            self.cost_by_uid[(group_id, policy_id)][uid] = data['total_cost']
            self.weight_by_uid[(group_id, policy_id)][uid] = path_weight
            self.stratum_by_uid[(group_id, policy_id)][uid] = path_stratum
        self.information_relaxation_id = policy_id
        # Runs launched with skip_information_relaxation=True carry None here.
        if data['zero_information_relaxation_cost'] is not None:
            self.zero_penalized_information_relaxation_cost[(group_id, policy_id)].record(
                data['zero_information_relaxation_cost'], path_weight, path_stratum)
            self.penalized_information_relaxation_cost[(group_id, policy_id)].record(
                data['penalized_information_relaxation_cost'], path_weight, path_stratum)
            self.zero_penalized_gap[(group_id, policy_id)].record(
                data['gap_to_zero_information_relaxation'], path_weight, path_stratum)
            self.penalized_gap[(group_id, policy_id)].record(
                data['gap_to_penalized_information_relaxation'], path_weight, path_stratum)
            
        if self.number_of_periods is None:
            self.number_of_periods = len(data['costs'])
        for t, cost in enumerate(data['costs']):
            self.one_time_cost_by_policy[policy_id][t] += cost
        
        warm_up_periods = data["warm_up_periods"]
        costs_after_warmup = data['costs'][warm_up_periods:] if len(data['costs']) > warm_up_periods else data['costs']

        after_warmup_cost = sum(cost for t, cost in enumerate(costs_after_warmup))
        self.after_warmup_policy_costs[(group_id, policy_id)] += after_warmup_cost
        if uid is not None:
            self.after_warmup_cost_by_uid[(group_id, policy_id)][uid] = after_warmup_cost
        
        scheduled_patients = np.array(data["scheduled_patients"])[warm_up_periods:].sum(axis=0) if len(data["scheduled_patients"]) > warm_up_periods else np.array(data["scheduled_patients"]).sum(axis=0)
        counts = self.booking_day_counts.get(policy_id)
        self.booking_day_counts[policy_id] = scheduled_patients if counts is None else counts + scheduled_patients
        
        total_scheduled_patients = scheduled_patients.sum()

        cum_scheduled_patients = scheduled_patients.cumsum(axis=0)
        total_scheduled_patients_by_type = scheduled_patients.sum(axis=0)
        scheduled_patients_ptc_by_day = np.divide(
                                                cum_scheduled_patients,
                                                total_scheduled_patients_by_type,
                                                out=np.ones_like(cum_scheduled_patients, dtype=float),
                                                where=total_scheduled_patients_by_type != 0
                                            ) * 100
        cum_total_scheduled_patients_by_day = scheduled_patients.sum(axis=1).cumsum(axis=0)
        total_scheduled_patients_ptc_by_day = (cum_total_scheduled_patients_by_day/total_scheduled_patients if total_scheduled_patients > 0 else np.ones_like(cum_total_scheduled_patients_by_day)) * 100

        for day in range(len(scheduled_patients)):
            # One sample per path and day: keep this OUT of the type loop, or the
            # Total row's n is inflated by num_types and its CI shrinks by sqrt(num_types).
            self.waiting_time_target_ptc_by_day[policy_id][day] += total_scheduled_patients_ptc_by_day[day]
            for treatment_type in range(len(scheduled_patients[day])):
                self.waiting_time_target_ptc_by_type_day[policy_id][treatment_type][day] += scheduled_patients_ptc_by_day[day][treatment_type]
        
        # Overtime utilization over the simulated evaluation days only: drop the
        # warm-up prefix and the partially-booked trailing booking window.
        num_simulated_days = len(data['costs'])
        overtime_days = (data["overtime"][warm_up_periods:num_simulated_days]
                         if num_simulated_days > warm_up_periods
                         else data["overtime"][:num_simulated_days])
        for overtime in overtime_days:
            self.overtime_utilization[(group_id, policy_id)] += overtime / self.env.overtime_capacity * 100
        patients_outside_target = sum(
            total_scheduled_patients_by_type[t] - cum_scheduled_patients[self.waiting_time_targets[t] - 1][t]
            for t in range(self.env.num_types)
            if total_scheduled_patients_by_type[t] > 0
        )
        if total_scheduled_patients > 0:
            self.waiting_time_violation[(group_id, policy_id)] += patients_outside_target / total_scheduled_patients * 100
        
        self.solving_time_per_state[(group_id, policy_id)] += data.get('solving_time_per_state', 0)
    
    def waiting_time_target_ptc_table(self, days=(1, 5, 10, 15, 20), confidence=0.95,
                                      label='tab:case_study_thresholds'):
        policy_order = [
            ('approx_penalized_hindsight', 'PH'),
            ('row_gen_alp', 'ALP'),
            ('myopic', 'M'),
        ]
        policies = [(pid, plabel) for pid, plabel in policy_order
                    if pid in self.waiting_time_target_ptc_by_type_day]
        num_policies = len(policies)
        num_types = max(len(self.waiting_time_target_ptc_by_type_day[pid])
                        for pid, _ in policies)

        def cell(stats):
            return f'{stats.mean:.0f}$\\pm${stats.half_window(confidence):.1f}'

        def total_cell(stats):
            return f'\\(\\mathbf{{{stats.mean:.0f} \\pm {stats.half_window(confidence):.1f}}}\\)'

        col_spec = f'l*{{{num_policies * len(days)}}}{{c}}'
        header_groups = '\n'.join(
            f'& \\multicolumn{{{num_policies}}}{{c}}{{{day} workday{"s" if day > 1 else ""}}}'
            for day in days
        )
        cmidrules = '\n'.join(
            f'\\cmidrule(lr){{{2 + i * num_policies}-{1 + (i + 1) * num_policies}}}'
            for i in range(len(days))
        )
        policy_header = ' '.join(f'& {plabel}' for _ in days for _, plabel in policies)

        body_lines = []
        for type in range(num_types):
            cells = ' '.join(
                f'& {cell(self.waiting_time_target_ptc_by_type_day[pid][type][day - 1])}'
                for day in days for pid, _ in policies
            )
            body_lines.append(f'{type + 1} {cells} \\\\')
        total_cells = '\n'.join(
            '& ' + ' & '.join(total_cell(self.waiting_time_target_ptc_by_day[pid][day - 1])
                              for pid, _ in policies)
            for day in days
        )
        body = '\n'.join(body_lines)

        n_paths = self.waiting_time_target_ptc_by_day[policies[0][0]][days[0] - 1].n
        abbreviations = ' and '.join(
            text for pid, text in (('approx_penalized_hindsight', 'PH denotes Penalized Hindsight'),
                                   ('myopic', 'M denotes Myopic'))
            if pid in self.waiting_time_target_ptc_by_type_day
        )

        table = f"""\\begin{{table}}[!htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Percentage of treatments initiated within selected waiting-time thresholds.}}
\\label{{{label}}}
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\renewcommand{{\\arraystretch}}{{1.08}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{col_spec}}}
\\toprule
Type
{header_groups} \\\\
{cmidrules}
{policy_header} \\\\
\\midrule
{body}
\\midrule
\\textbf{{Total}}
{total_cells} \\\\
\\bottomrule
\\end{{tabular}}%
}}
\\begin{{tablenotes}}[flushleft]
\\scriptsize
\\item \\textit{{Note.}} Entries are percentages reported as sample means \\(\\pm\\) {round(confidence * 100)}\\% confidence-interval half-widths over {n_paths:,} evaluation sample paths. {abbreviations}. Treatment inputs appear in Table~\\ref{{tab:case_study_inputs}}.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}"""
        return table
    
    def booking_day_preferences(self):
        return {policy_id: counts / counts.sum(axis=0, keepdims=True) * 100
                for policy_id, counts in self.booking_day_counts.items()}

    def booking_day_preference_table(self, policy_order=BOOKING_POLICY_ORDER):
        shares = self.booking_day_preferences()
        policies = [(pid, label) for pid, label, _ in policy_order if pid in shares]
        lines = [f"{'type':>4} | " + ' | '.join(f'{label}: booking days in order of preference' for _, label in policies)]
        for i in range(next(iter(shares.values())).shape[1]):
            cells = []
            for pid, _ in policies:
                ranks = booking_day_ranks(shares[pid][:, i:i + 1])[:, 0]
                order = [int(day) for day in np.argsort(np.where(ranks > 0, ranks, np.inf), kind='stable')[:int((ranks > 0).sum())] + 1]
                cells.append(' > '.join(map(str, order)))
            lines.append(f'{i + 1:>4} | ' + ' | '.join(cells))
        return '\n'.join(lines)

    def plot_booking_day_preferences(self, save_file, policy_order=BOOKING_POLICY_ORDER, types=None):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        shares = self.booking_day_preferences()
        policies = [(pid, label, color) for pid, label, color in policy_order if pid in shares]
        days = next(iter(shares.values())).shape[0]
        types = list(range(1, next(iter(shares.values())).shape[1] + 1)) if types is None else list(types)
        n_paths = max(stats.n for stats in self.policy_costs.values())
        stem, extension = os.path.splitext(save_file)
        os.makedirs(os.path.dirname(save_file) or '.', exist_ok=True)
        saved = []
        for pid, label, color in policies:
            fig, ax = plt.subplots(figsize=(0.7 * days + 1.6, 0.5 * len(types) + 1.6))
            ranks = booking_day_ranks(shares[pid])
            for row, type_id in enumerate(types):
                share = shares[pid][:, type_id - 1]
                rank = ranks[:, type_id - 1]
                for day in range(days):
                    if rank[day] == 0:
                        ax.text(day + 1, row, '\u00b7', ha='center', va='center', fontsize=13, color='#b5b3aa')
                        continue
                    ax.add_patch(plt.Rectangle((day + 0.5, row - 0.5), 1, 1, color=color, alpha=0.55 * share[day] / share.max(), linewidth=0))
                    circled = dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor='black', linewidth=1.2) if rank[day] == 1 else None
                    ax.text(day + 1, row, str(rank[day]), ha='center', va='center', fontsize=13,
                            fontweight='bold' if rank[day] == 1 else 'normal', bbox=circled)
                for run_start, run_end in preference_runs(rank):
                    ax.annotate('', xy=(run_end + 1, row - 0.4), xytext=(run_start + 1, row - 0.4),
                                arrowprops=dict(arrowstyle='-|>', color='black', linewidth=1.1, shrinkA=0, shrinkB=0))
                    ax.plot(run_start + 1, row - 0.4, 'o', color='black', markersize=3.5)
            ax.set_xlim(0.5, days + 0.5)
            ax.set_ylim(-0.7, len(types) - 0.4)
            ax.set_xticks(range(1, days + 1))
            ax.set_yticks(range(len(types)))
            ax.set_yticklabels(types)
            ax.tick_params(labelsize=13, length=0)
            ax.set_xlabel('booking day (first appointment)', fontsize=15)
            ax.set_ylabel('treatment type', fontsize=15)
            ax.set_title(f'{label}: booking day preferences by treatment type ({n_paths:,} evaluation paths)', fontsize=16, loc='left', pad=14)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            fig.text(0.01, 0.005, 'Cell = rank of the day by scheduling frequency (1 = most frequent, circled; \u00b7 = never used); '
                     'shading = frequency; arrows = runs of consecutive ranks.', fontsize=12, color='#555555')
            fig.tight_layout(rect=(0, 0.03, 1, 1))
            policy_file = f'{stem}_{pid}{extension}'
            fig.savefig(policy_file, dpi=150, bbox_inches='tight')
            fig.savefig(f'{stem}_{pid}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved.append(policy_file)
        return saved

    def performance_summary_table(self):
        policy_label = {
            'myopic': 'Myopic',
            'row_gen_alp': 'ALP',
            'approx_penalized_hindsight': 'Penalized Hindsight',
        }
        table = ''
        for (group_id, policy_id), stats in self.after_warmup_policy_costs.items():
            table += f"{policy_label.get(policy_id, policy_id)} & ${stats.confidence_interval(0.8)}$ & ${self.waiting_time_violation[(group_id, policy_id)].confidence_interval()}$ & ${self.overtime_utilization[(group_id, policy_id)].confidence_interval()}$\\\\\n"
        return table

    def improvement_over_baseline(self, group_id, policy_id, baseline_id='myopic', confidence=0.95,
                                  cost_by_uid=None):
        if cost_by_uid is None:
            cost_by_uid = self.after_warmup_cost_by_uid
        policy_costs = cost_by_uid[(group_id, policy_id)]
        baseline_costs = cost_by_uid[(group_id, baseline_id)]
        common_uids = policy_costs.keys() & baseline_costs.keys()
        n = len(common_uids)
        if n < 2:
            return 0.0, 0.0, n
        uids = list(common_uids)
        weights, strata = _uid_weights_and_strata(
            uids, self.weight_by_uid.get((group_id, policy_id), {}),
            self.stratum_by_uid.get((group_id, policy_id), {}))
        pc = np.array([policy_costs[uid] for uid in uids])
        bc = np.array([baseline_costs[uid] for uid in uids])
        improvement, half_width = _stratified_relative_improvement(
            bc - pc, bc, weights, strata, confidence)
        return improvement, half_width, n

    def information_relaxation_improvement(self, group_id, mutate_val, confidence=0.95):
        key = (group_id, mutate_val)
        zero_costs = self.ir_zero_by_uid.get(key, {})
        penalized_costs = self.ir_penalized_by_uid.get(key, {})
        uids = list(zero_costs.keys() & penalized_costs.keys())
        n = len(uids)
        if n < 2:
            return 0.0, 0.0, n
        weights, strata = _uid_weights_and_strata(
            uids, self.ir_weight_by_uid.get(key, {}), self.ir_stratum_by_uid.get(key, {}))
        zero = np.array([zero_costs[uid] for uid in uids])
        penalized = np.array([penalized_costs[uid] for uid in uids])
        improvement, half_width = _stratified_relative_improvement(
            penalized - zero, zero, weights, strata, confidence)
        return improvement, half_width, n

    def overall_performance_table(self, gamma='0.99', baseline_id='myopic', confidence=0.95):
        policy_order = [
            ('approx_penalized_hindsight', 'Penalized Hindsight'),
            ('row_gen_alp', 'ALP'),
            ('myopic', 'Myopic'),
        ]

        def fmt(value, decimals=0):
            return f'{value:,.{decimals}f}'.replace(',', '{,}')

        def cell(mean, half_width, mean_decimals=0, hw_decimals=1, bold=False):
            body = f'{fmt(mean, mean_decimals)} \\pm {fmt(half_width, hw_decimals)}'
            return f'\\(\\mathbf{{{body}}}\\)' if bold else f'\\({body}\\)'

        group_ids = {gid for gid, _ in self.after_warmup_policy_costs.keys()}
        rows = []
        n_paths = 0
        for group_id in sorted(group_ids, key=str):
            policies = [(pid, label) for pid, label in policy_order
                        if (group_id, pid) in self.after_warmup_policy_costs]
            costs = {pid: self.after_warmup_policy_costs[(group_id, pid)] for pid, _ in policies}
            violations = {pid: self.waiting_time_violation[(group_id, pid)] for pid, _ in policies}
            overtimes = {pid: self.overtime_utilization[(group_id, pid)] for pid, _ in policies}
            improvements = {
                pid: self.improvement_over_baseline(group_id, pid, baseline_id, confidence)
                for pid, _ in policies if pid != baseline_id
            }
            n_paths = max(n_paths, max(stats.n for stats in costs.values()))

            best_cost = min(costs, key=lambda pid: costs[pid].mean)
            best_violation = min(violations, key=lambda pid: violations[pid].mean)
            best_overtime = min(overtimes, key=lambda pid: overtimes[pid].mean)
            best_improvement = max(improvements, key=lambda pid: improvements[pid][0]) if improvements else None

            for pid, label in policies:
                cost_cell = cell(costs[pid].mean, costs[pid].half_window(confidence), bold=pid == best_cost)
                if pid == baseline_id:
                    improvement_cell = '---'
                else:
                    improvement, half_width, _ = improvements[pid]
                    improvement_cell = cell(improvement, half_width, mean_decimals=1, hw_decimals=1,
                                            bold=pid == best_improvement)
                violation_cell = cell(violations[pid].mean, violations[pid].half_window(confidence),
                                      bold=pid == best_violation)
                overtime_cell = cell(overtimes[pid].mean, overtimes[pid].half_window(confidence),
                                     bold=pid == best_overtime)
                rows.append(f'{label} & {cost_cell} & {improvement_cell} & {violation_cell} & {overtime_cell} \\\\')

        body = '\n'.join(rows)
        gamma_tag = gamma.replace('.', '')
        table = f"""\\begin{{table}}[!htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Estimated case-study policy performance with $\\gamma={gamma}$.}}
\\label{{tab:overall_performance_{gamma_tag}_case_study}}

\\small
\\setlength{{\\tabcolsep}}{{6pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}

\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} lcccc @{{}}}}
\\toprule
Policy
& \\makecell{{Discounted\\\\total cost}}
& \\makecell{{Improvement over\\\\Myopic (\\%)}}
& \\makecell{{Wait-time\\\\violations}}
& \\makecell{{Overtime\\\\utilization}}\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}

\\begin{{tablenotes}}[flushleft]
\\footnotesize
\\item \\textit{{Note.}} Values are sample means \\(\\pm\\) {round(confidence * 100)}\\% confidence-interval half-widths over \\({fmt(n_paths)}\\) evaluation sample paths. Improvement over Myopic is the paired relative reduction in discounted total cost on common sample paths, with a delta-method {round(confidence * 100)}\\% confidence interval. Bold entries are the best value in each metric. Table~\\ref{{tab:case_study_inputs}} reports the case-study inputs.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}"""
        return table

    def group_id_to_mutate_val(self):
        mapping = {}
        pattern = os.path.join(self.directory_path, self.file_pattern)
        for file_path in sorted(glob.glob(pattern)):
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if 'group_id' in record and 'mutate_val' in record:
                        mapping[record['group_id']] = record['mutate_val']
        return mapping

    def mixture_probability_table(self, policy_id='approx_penalized_hindsight', confidence=0.95):
        group_id_by_mixing_probability = sorted(
            (mutate_val, group_id)
            for group_id, mutate_val in self.group_id_to_mutate_val().items()
            if (group_id, policy_id) in self.zero_penalized_improvement
        )
        if not group_id_by_mixing_probability:
            raise ValueError(f"No records loaded for policy_id '{policy_id}'.")

        def cell(percentage_stats):
            return f"\\({round(percentage_stats.mean)} \\pm {round(percentage_stats.half_window(confidence), 1)}\\)"

        rows = []
        for mixing_probability, group_id in group_id_by_mixing_probability:
            key = (group_id, policy_id)
            rows.append(
                f"\\({mixing_probability:.2f}\\) "
                f"& {cell(self.zero_penalized_improvement[key])} "
                f"& {cell(self.penalized_improvement[key])} \\\\"
            )
        body = '\n'.join(rows)

        return f"""\\begin{{table}}[H]
\\centering
\\begin{{threeparttable}}
\\caption{{Policy-quality sensitivity to the defensive mixing probability.}}
\\label{{tab:toy_importance_sampling}}
\\small
\\setlength{{\\tabcolsep}}{{8pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular}}{{@{{}}ccc@{{}}}}
\\toprule
Defensive mixing probability \\(\\varepsilon\\)
& \\(\\%\\) gap to zero-penalty LB
& \\(\\%\\) gap to penalized LB \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}[flushleft]
\\footnotesize
\\item \\textit{{Note.}} The instance uses 50\\% initial occupancy and 4,096 evaluation sample paths; other inputs are given in Table~\\ref{{tab:treatment_pattern_and_penalty_toy_study_1}}.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}"""

    def plot_percentage_improvement(self, scale, xlabel, ylabel, file_name):
        plot_stats = {
            (mutate_val * scale if scale is not None else mutate_val): self.zero_penalized_improvement[(group_id, mutate_val)]
            for (group_id, mutate_val) in self.zero_penalized_improvement.keys()
        }
        approximate_value_plot_from_running_stats(running_stats_dict=plot_stats,
                                                  xlabel=xlabel,
                                                  ylabel=ylabel,
                                                title=None,
                                                save_file=os.path.join(self.directory_path,
                                                                        file_name))


def _uid_weights_and_strata(uids, weights_map, strata_map):
    weights = np.array([
        1.0 if weights_map.get(uid) is None else float(weights_map[uid])
        for uid in uids])
    strata = np.array([
        0 if strata_map.get(uid) is None else int(strata_map[uid])
        for uid in uids])
    return weights, strata


def _stratified_relative_improvement(diff, base, weights, strata, confidence):
    from scipy.stats import norm
    diff = np.asarray(diff, dtype=float)
    base = np.asarray(base, dtype=float)
    weight_total = weights.sum()
    dm = float(weights @ diff) / weight_total
    bm = float(weights @ base) / weight_total
    if bm == 0:
        return 0.0, 0.0
    improvement = dm / bm * 100
    cov_mean = np.zeros((2, 2))
    for label in np.unique(strata):
        mask = strata == label
        n_h = int(mask.sum())
        if n_h < 2:
            continue
        weight_share = float(weights[mask].sum()) / weight_total
        cov_h = np.cov(np.vstack([diff[mask], base[mask]]))
        cov_mean += weight_share ** 2 * cov_h / n_h
    var = (cov_mean[0, 0] / bm ** 2 - 2 * dm * cov_mean[0, 1] / bm ** 3
           + dm ** 2 * cov_mean[1, 1] / bm ** 4)
    half_width = norm.ppf((1 + confidence) / 2) * np.sqrt(max(var, 0.0)) * 100
    return improvement, half_width


def _merge_stats_by_policy(stats_dict, policy_id):
    merged = RunningStats()
    for (group_id, pid), stats in stats_dict.items():
        if pid == policy_id:
            merged += stats
    return merged


def policy_performance_comparison_table(
    env,
    base_results_dir=os.path.join('.', 'experiments', 'results'),
    conditions=(
        ('Fixed initial state', 'base_toy_study', True),
        ('Steady state', 'steady_state_toy_study', False),
        ('ALP warm-up', 'alp_steady_state_toy_study', True),
    ),
    file_pattern='[0-9]*.jsonl',
    is_reuse=True,
    confidence=0.95,
):
    policy_order = [
        ('approx_hindsight', 'Hindsight'),
        ('approx_penalized_hindsight', 'Penalized hindsight'),
        ('myopic', 'Myopic'),
        ('row_gen_alp', 'ALP'),
    ]

    results = []
    for label, folder_name, show_gap_percent in conditions:
        ser = SimulateEvaluationResult(
            os.path.join(base_results_dir, folder_name),
            file_pattern,
            env,
            is_reuse=is_reuse,
        )
        results.append((label, ser, show_gap_percent))

    loaded_policy_ids = {
        pid for _, ser, _ in results for (_, pid) in ser.policy_costs.keys()
    }
    policies = [(pid, label) for pid, label in policy_order if pid in loaded_policy_ids]

    def ci_cell(stats):
        if stats.n == 0:
            return 'NA'
        return f"\\({round(stats.mean)} \\pm {round(stats.half_window(confidence), 1)}\\)"

    def gap_percent_cell(stats, show_gap_percent):
        if not show_gap_percent or stats.n == 0:
            return 'NA'
        return f"\\({round(stats.mean)} \\pm {round(stats.half_window(confidence), 1)}\\)"

    col_spec = 'l' + 'r' * (2 * len(results))
    group_header = '\n'.join(
        f"& \\multicolumn{{2}}{{c}}{{{label}}}" for label, _, _ in results
    ) + ' \\\\'
    cmidrules = ' '.join(
        f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(results))
    )
    ci_header = '& ' + ' \n& '.join('95\\% CI & Gap \\%' for _ in results) + ' \\\\'

    blocks = []
    for pid, policy_label in policies:
        lines = [f"{policy_label} {'& ' * 2 * len(results)}\\\\"]
        cost_cells, zero_cells, max_cells = [], [], []
        for _, ser, show_gap_percent in results:
            cost_cells += [ci_cell(_merge_stats_by_policy(ser.policy_costs, pid)), '']
            zero_cells += [
                ci_cell(_merge_stats_by_policy(ser.zero_penalized_gap, pid)),
                gap_percent_cell(_merge_stats_by_policy(ser.zero_penalized_improvement, pid), show_gap_percent),
            ]
            max_cells += [
                ci_cell(_merge_stats_by_policy(ser.penalized_gap, pid)),
                gap_percent_cell(_merge_stats_by_policy(ser.penalized_improvement, pid), show_gap_percent),
            ]
        lines.append('\\quad Policy cost \n& ' + ' \n& '.join(cost_cells).rstrip() + ' \\\\')
        lines.append('\\quad Zero-penalty gap \n& ' + ' \n& '.join(zero_cells) + ' \\\\')
        lines.append('\\quad Max-penalty gap \n& ' + ' \n& '.join(max_cells) + ' \\\\')
        blocks.append('\n'.join(lines))
    body = '\n\\addlinespace[2pt]\n\n'.join(blocks)

    table = f"""\\begin{{table}}[t]
\\centering
\\begin{{threeparttable}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{3.5pt}}
\\renewcommand{{\\arraystretch}}{{0.95}}

\\caption{{Performance comparison of different policies:
$I=2$, $C_r=7$, $C_o=3$, $o=100$, $N=7$, $g_i=2000$,
$\\boldsymbol{{\\lambda}}=(1,2)$ with total arrival rate \\(3\\) capped at \\(9\\),
$\\gamma=0.99$, and $M_{{\\text{{eval}}}}=5{{,}}000$.
Treatment patterns and waiting-time penalties are given in
Tables~\\ref{{tab:treatment_pattern_toy_study_1}} and
\\ref{{tab:wait_time_penalty_toy_study_1}}.}}
\\label{{tab:policy_performance_comparision}}

\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}} {col_spec} @{{}}}}
\\toprule
Initial-state condition
{group_header}
{cmidrules}
{ci_header}
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}

\\begin{{tablenotes}}[flushleft]
\\footnotesize
\\item Notes. The zero-penalty gap is the difference between the policy cost and the zero-penalty perfect-information relaxation cost. The max-penalty gap is the difference between the policy cost and the penalized perfect-information relaxation cost obtained with the fitted penalty coefficients \\(\\hat{{\\boldsymbol{{\\theta}}}}^*\\). Gap \\% reports each gap as a percentage of the corresponding information-relaxation cost.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}"""
    return table


def _fmt_latex_number(value, decimals=0):
    return f'{value:,.{decimals}f}'.replace(',', '{,}')


def _value_cell(mean, half_width, mean_decimals=0, hw_decimals=1, bold=False):
    body = f'{_fmt_latex_number(mean, mean_decimals)} \\pm {_fmt_latex_number(half_width, hw_decimals)}'
    return f'\\(\\mathbf{{{body}}}\\)' if bold else f'\\({body}\\)'


def _metric_cell(stats, confidence, mean_decimals=0, hw_decimals=1, bold=False):
    return _value_cell(stats.mean, stats.half_window(confidence), mean_decimals, hw_decimals, bold)


def saure_ejor_steady_state_table(
    base_results_dir=os.path.join('.', 'experiments', 'results'),
    folder_name='case_study_ejor_alp_steady_state',
    file_pattern='[0-9]*.jsonl',
    is_reuse=True,
    baseline_id='myopic',
    confidence=0.95,
):
    from experiments import get_config_by_type
    env = get_config_by_type('ejor').env

    policy_order = [
        ('approx_penalized_hindsight', 'Penalized Hindsight'),
        ('row_gen_alp', 'ALP'),
        ('myopic', 'Myopic'),
    ]
    ser = SimulateEvaluationResult(
        os.path.join(base_results_dir, folder_name),
        file_pattern,
        env,
        is_reuse=is_reuse,
    )

    rows = []
    n_paths = 0
    for group_id in sorted({gid for gid, _ in ser.policy_costs.keys()}, key=str):
        policies = [(pid, label) for pid, label in policy_order
                    if (group_id, pid) in ser.policy_costs]
        costs = {pid: ser.policy_costs[(group_id, pid)] for pid, _ in policies}
        violations = {pid: ser.waiting_time_violation[(group_id, pid)] for pid, _ in policies}
        overtimes = {pid: ser.overtime_utilization[(group_id, pid)] for pid, _ in policies}
        zero_gaps = {pid: ser.zero_penalized_gap[(group_id, pid)] for pid, _ in policies}
        penalized_gaps = {pid: ser.penalized_gap[(group_id, pid)] for pid, _ in policies}
        improvements = {
            pid: ser.improvement_over_baseline(group_id, pid, baseline_id, confidence,
                                               cost_by_uid=ser.cost_by_uid)
            for pid, _ in policies if pid != baseline_id
        }
        n_paths = max(n_paths, max(stats.n for stats in costs.values()))

        best_cost = min(costs, key=lambda pid: costs[pid].mean)
        best_violation = min(violations, key=lambda pid: violations[pid].mean)
        best_overtime = min(overtimes, key=lambda pid: overtimes[pid].mean)
        best_zero_gap = min(zero_gaps, key=lambda pid: zero_gaps[pid].mean)
        best_penalized_gap = min(penalized_gaps, key=lambda pid: penalized_gaps[pid].mean)
        best_improvement = max(improvements, key=lambda pid: improvements[pid][0]) if improvements else None

        for pid, label in policies:
            if pid == baseline_id:
                improvement_cell = '---'
            else:
                improvement, half_width, _ = improvements[pid]
                improvement_cell = _value_cell(improvement, half_width, mean_decimals=1,
                                               bold=pid == best_improvement)
            rows.append(
                f'{label}'
                f' & {_metric_cell(costs[pid], confidence, bold=pid == best_cost)}'
                f' & {improvement_cell}'
                f' & {_metric_cell(violations[pid], confidence, bold=pid == best_violation)}'
                f' & {_metric_cell(overtimes[pid], confidence, bold=pid == best_overtime)}'
                f' & {_metric_cell(zero_gaps[pid], confidence, bold=pid == best_zero_gap)}'
                f' & {_metric_cell(penalized_gaps[pid], confidence, bold=pid == best_penalized_gap)} \\\\'
            )
    body = '\n'.join(rows)

    table = f"""\\begin{{table}}[!htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Estimated case-study policy performance starting from the ALP steady state with $\\gamma=0.99$.}}
\\label{{tab:saure_ejor_steady_state}}

\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}

\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccc}}
\\toprule
Policy
& \\makecell{{Discounted\\\\total cost}}
& \\makecell{{Improvement over\\\\Myopic (\\%)}}
& \\makecell{{Wait-time\\\\violations}}
& \\makecell{{Overtime\\\\utilization}}
& \\makecell{{Gap to\\\\zero-penalty LB}}
& \\makecell{{Gap to\\\\penalized LB}}\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}

\\begin{{tablenotes}}[flushleft]
\\footnotesize
\\item \\textit{{Note.}} Values are sample means \\(\\pm\\) {round(confidence * 100)}\\% confidence-interval half-widths over \\({_fmt_latex_number(n_paths)}\\) evaluation sample paths. Discounted total cost is the importance-weighted discounted cost of the post-warm-up evaluation periods. Improvement over Myopic is the paired relative reduction in discounted total cost on common sample paths, with a delta-method {round(confidence * 100)}\\% confidence interval. Wait-time violations and overtime utilization are percentages computed over the evaluation periods only. The last two columns report the policy's suboptimality gap (in cost units) to the zero-penalty and penalized perfect-information relaxation lower bounds. Bold entries are the best value in each metric.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}"""
    return table


def case_study_policy_table(
    folder_name,
    caption,
    label,
    base_results_dir=os.path.join('.', 'experiments', 'results'),
    file_pattern='[0-9]*.jsonl',
    is_reuse=False,
    baseline_id='myopic',
    confidence=0.95,
    uid_aliases=None,
):
    from experiments import get_config_by_type
    env = get_config_by_type('ejor').env
    policy_order = [
        ('approx_penalized_hindsight', 'Penalized Hindsight'),
        ('row_gen_alp', 'ALP'),
        ('myopic', 'Myopic'),
    ]
    ser = SimulateEvaluationResult(os.path.join(base_results_dir, folder_name), file_pattern, env, is_reuse=is_reuse)
    if uid_aliases:
        for by_uid in (ser.cost_by_uid, ser.weight_by_uid, ser.stratum_by_uid):
            for key, values in by_uid.items():
                by_uid[key] = {uid_aliases.get(uid, uid): value for uid, value in values.items()}
    rows = []
    n_paths = 0
    for group_id in sorted({gid for gid, _ in ser.policy_costs.keys()}, key=str):
        policies = [(pid, name) for pid, name in policy_order if (group_id, pid) in ser.policy_costs]
        costs = {pid: ser.policy_costs[(group_id, pid)] for pid, _ in policies}
        violations = {pid: ser.waiting_time_violation[(group_id, pid)] for pid, _ in policies}
        overtimes = {pid: ser.overtime_utilization[(group_id, pid)] for pid, _ in policies}
        zero_gaps = {pid: ser.zero_penalized_gap[(group_id, pid)] for pid, _ in policies}
        zero_relative = {pid: ser.zero_penalized_improvement[(group_id, pid)] for pid, _ in policies}
        penalized_gaps = {pid: ser.penalized_gap[(group_id, pid)] for pid, _ in policies}
        penalized_relative = {pid: ser.penalized_improvement[(group_id, pid)] for pid, _ in policies}
        improvements = {
            pid: ser.improvement_over_baseline(group_id, pid, baseline_id, confidence, cost_by_uid=ser.cost_by_uid)
            for pid, _ in policies if pid != baseline_id
        }
        n_paths = max(n_paths, max(stats.n for stats in costs.values()))
        best = {
            'cost': min(costs, key=lambda pid: costs[pid].mean),
            'violation': min(violations, key=lambda pid: violations[pid].mean),
            'overtime': min(overtimes, key=lambda pid: overtimes[pid].mean),
            'zero': min(zero_gaps, key=lambda pid: zero_gaps[pid].mean),
            'penalized': min(penalized_gaps, key=lambda pid: penalized_gaps[pid].mean),
            'improvement': max(improvements, key=lambda pid: improvements[pid][0]) if improvements else None,
        }
        for pid, name in policies:
            if pid == baseline_id:
                improvement_cell = '---'
            else:
                improvement, half_width, _ = improvements[pid]
                improvement_cell = _value_cell(improvement, half_width, mean_decimals=1, bold=pid == best['improvement'])
            rows.append(
                f'{name}'
                f' & {_metric_cell(costs[pid], confidence, bold=pid == best["cost"])}'
                f' & {improvement_cell}'
                f' & {_metric_cell(violations[pid], confidence, bold=pid == best["violation"])}'
                f' & {_metric_cell(overtimes[pid], confidence, bold=pid == best["overtime"])}'
                f' & {_metric_cell(zero_gaps[pid], confidence, bold=pid == best["zero"])}'
                f' & {_metric_cell(zero_relative[pid], confidence, mean_decimals=1, bold=pid == best["zero"])}'
                f' & {_metric_cell(penalized_gaps[pid], confidence, bold=pid == best["penalized"])}'
                f' & {_metric_cell(penalized_relative[pid], confidence, mean_decimals=1, bold=pid == best["penalized"])} \\\\'
            )
    body = '\n'.join(rows)
    table = f"""\\begin{{table}}[!htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{{caption}}}
\\label{{{label}}}

\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}

\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lcccccccc}}
\\toprule
Policy
& \\makecell{{Discounted\\\\total cost}}
& \\makecell{{Improvement over\\\\Myopic (\\%)}}
& \\makecell{{Wait-time\\\\violations (\\%)}}
& \\makecell{{Overtime\\\\utilization (\\%)}}
& \\makecell{{Gap to\\\\zero-penalty LB}}
& \\makecell{{Gap to zero-\\\\penalty LB (\\%)}}
& \\makecell{{Gap to\\\\penalized LB}}
& \\makecell{{Gap to\\\\penalized LB (\\%)}}\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}%
}}

\\begin{{tablenotes}}[flushleft]
\\footnotesize
\\item \\textit{{Note.}} Values are sample means \\(\\pm\\) {round(confidence * 100)}\\% confidence-interval half-widths over \\({_fmt_latex_number(n_paths)}\\) evaluation sample paths. Discounted total cost is the survival-weighted cost over the sampled absorption horizon. Improvement over Myopic is the paired relative reduction in discounted total cost on common sample paths, with a delta-method {round(confidence * 100)}\\% confidence interval. Wait-time violations and overtime utilization are percentages over the evaluation periods. The gap to the zero-penalty lower bound is the policy cost minus the zero-penalty perfect-information relaxation cost; the gap to the penalized lower bound is the penalized policy cost minus the penalized perfect-information relaxation cost. Relative gaps are expressed as a percentage of the corresponding lower bound. Bold entries are the best value in each metric.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}"""
    return table, ser


def uid_aliases_from_command_files(canonical_dat, aliased_dat):
    def uids(path):
        return [json.loads(line.split(" --params '", 1)[1].rsplit("'", 1)[0])[0]['uid'] for line in open(path)]
    canonical, aliased = uids(canonical_dat), uids(aliased_dat)
    if len(canonical) != len(aliased):
        raise ValueError(f'{canonical_dat} has {len(canonical)} commands but {aliased_dat} has {len(aliased)}')
    return dict(zip(aliased, canonical))


def case_study_booking_preferences(
    folder_name,
    save_file,
    base_results_dir=os.path.join('.', 'experiments', 'results'),
    file_pattern='[0-9]*.jsonl',
    is_reuse=False,
):
    from experiments import get_config_by_type
    config = get_config_by_type('ejor')
    ser = SimulateEvaluationResult(os.path.join(base_results_dir, folder_name), file_pattern, config.env, is_reuse=is_reuse)
    ser.plot_booking_day_preferences(save_file)
    table = ser.booking_day_preference_table()
    print(table)
    return table, ser


def saure_ejor_replication_table(
    base_results_dir=os.path.join('.', 'experiments', 'results'),
    folder_name='case_study_ejor_replication',
    file_pattern='[0-9]*.jsonl',
    is_reuse=True,
    confidence=0.95,
):
    from experiments import get_config_by_type
    env = get_config_by_type('ejor').env

    policy_order = [
        ('row_gen_alp', 'ALP'),
        ('myopic', 'Myopic'),
    ]
    ser = SimulateEvaluationResult(
        os.path.join(base_results_dir, folder_name),
        file_pattern,
        env,
        is_reuse=is_reuse,
    )

    rows = []
    n_paths = 0
    for group_id in sorted({gid for gid, _ in ser.policy_costs.keys()}, key=str):
        policies = [(pid, label) for pid, label in policy_order
                    if (group_id, pid) in ser.policy_costs]
        costs = {pid: ser.policy_costs[(group_id, pid)] for pid, _ in policies}
        violations = {pid: ser.waiting_time_violation[(group_id, pid)] for pid, _ in policies}
        overtimes = {pid: ser.overtime_utilization[(group_id, pid)] for pid, _ in policies}
        n_paths = max(n_paths, max(stats.n for stats in costs.values()))

        best_cost = min(costs, key=lambda pid: costs[pid].mean)
        best_violation = min(violations, key=lambda pid: violations[pid].mean)
        best_overtime = min(overtimes, key=lambda pid: overtimes[pid].mean)

        for pid, label in policies:
            rows.append(
                f'{label}'
                f' & {_metric_cell(costs[pid], confidence, bold=pid == best_cost)}'
                f' & {_metric_cell(violations[pid], confidence, bold=pid == best_violation)}'
                f' & {_metric_cell(overtimes[pid], confidence, bold=pid == best_overtime)} \\\\'
            )
    body = '\n'.join(rows)

    table = f"""\\begin{{table}}[!htbp]
\\centering
\\begin{{threeparttable}}
\\caption{{Estimated case-study policy performance under the EJOR replication design (empty start with a 750-day warm-up) with $\\gamma=0.99$.}}
\\label{{tab:saure_ejor_replication}}

\\small
\\setlength{{\\tabcolsep}}{{6pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}

\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}} lccc @{{}}}}
\\toprule
Policy
& \\makecell{{Discounted\\\\total cost}}
& \\makecell{{Wait-time\\\\violations}}
& \\makecell{{Overtime\\\\utilization}}\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular*}}

\\begin{{tablenotes}}[flushleft]
\\footnotesize
\\item \\textit{{Note.}} Values are sample means \\(\\pm\\) {round(confidence * 100)}\\% confidence-interval half-widths over \\({_fmt_latex_number(n_paths)}\\) evaluation sample paths. Each policy warms itself up for 750 days from an empty system; discounted total cost is the discounted cost of the 750 post-warm-up evaluation periods. Wait-time violations and overtime utilization are percentages computed over the evaluation periods only. Bold entries are the best value in each metric.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}"""
    return table


def run_saure_ejor_waiting_time_tables(is_reuse=True):
    from experiments import get_config_by_type
    env = get_config_by_type('ejor').env
    for folder, label in (('case_study_ejor_alp_steady_state', 'tab:case_study_thresholds'),
                          ('case_study_ejor_replication', 'tab:case_study_thresholds_replication')):
        ser = SimulateEvaluationResult(
            os.path.join('.', 'experiments', 'results', folder),
            '[0-9]*.jsonl',
            env,
            is_reuse=is_reuse,
        )
        print(f'% ===== {folder} =====')
        print(ser.waiting_time_target_ptc_table(label=label))
        print()


def run_improvement_plots(base_results_dir, file_pattern, env_info, group_ids, is_reuse=False):
    for config in EXPERIMENT_PLOT_CONFIGS:
        directory_path = os.path.join(base_results_dir, config['name'])
        ser = SimulateEvaluationResult(
            directory_path,
            file_pattern,
            env_info,
            group_ids=group_ids,
            is_reuse=is_reuse,
        )
        ser.plot_percentage_improvement(
            scale=config['scale'],
            xlabel=config['xlabel'],
            ylabel=IMPROVEMENT_YLABEL,
            file_name=f"{config['name']}_lower_bound_improvement.svg",
        )
def run_mixture_probability_table(is_reuse=True, policy_id='approx_penalized_hindsight'):
    from experiments import get_config_by_type
    config = get_config_by_type('toy')
    ser = SimulateEvaluationResult(
        os.path.join('.', 'experiments', 'results', 'mixture_probability_toy_study'),
        '[0-9]*.jsonl',
        config.env,
        is_reuse=is_reuse,
    )
    for mutate_val, group_id in sorted((mv, gid) for gid, mv in ser.group_id_to_mutate_val().items()):
        stats = ser.zero_penalized_gap.get((group_id, policy_id))
        if stats is not None:
            print(f"epsilon={mutate_val}, policy={policy_id}: {stats.n} sample paths")
    table = ser.mixture_probability_table(policy_id=policy_id)
    print(table)
    return table


def report_eval_proposal_policy_costs(is_reuse=True, confidence=0.95):
    from experiments import get_config_by_type
    proposals = (
        ('toy_eval_proposal_fixed_459',
         '\\makecell{Fixed horizon\\\\(\\(459\\) periods)}'),
        ('toy_eval_proposal_geometric_099',
         '\\makecell{Geometric\\\\(\\(\\gamma_{\\text{proposal}} = 0.99\\))}'),
        ('toy_eval_proposal_mixture_095_l01',
         '\\makecell{Mixture geometric\\\\(\\(\\gamma_{\\text{proposal}} = 0.95\\), \\(\\lambda_0 = 0.1\\))}'),
    )
    policy_labels = {
        'approx_penalized_hindsight': 'Penalized Hindsight',
        'row_gen_alp': 'ALP',
        'myopic': 'Myopic',
    }

    def fmt(value, decimals=0):
        return f'{value:,.{decimals}f}'.replace(',', '{,}')

    costs = {}
    for experiment_name, _ in proposals:
        ser = SimulateEvaluationResult(
            os.path.join('.', 'experiments', 'results', experiment_name),
            '[0-9]*.jsonl',
            get_config_by_type('toy').env,
            is_reuse=is_reuse,
        )
        for (group_id, policy_id), stats in sorted(ser.policy_costs.items()):
            costs.setdefault(policy_id, {})[experiment_name] = stats

    rows = []
    for policy_id, stats_by_experiment in sorted(costs.items()):
        cells = [
            f"\\({fmt(stats_by_experiment[name].mean)} \\pm {fmt(stats_by_experiment[name].half_window(confidence), 1)}\\)"
            for name, _ in proposals
        ]
        rows.append(f"{policy_labels.get(policy_id, policy_id)} & " + ' & '.join(cells) + " \\\\")
    body = '\n'.join(rows)
    header = '\n'.join(f"& {label}" for _, label in proposals)

    table = f"""\\begin{{table}}[H]
\\centering
\\begin{{threeparttable}}
\\caption{{Policy-cost estimates under the three evaluation-path proposals.}}
\\label{{tab:eval_proposal_policy_costs}}
\\small
\\setlength{{\\tabcolsep}}{{8pt}}
\\renewcommand{{\\arraystretch}}{{1.15}}
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
Policy
{header} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\begin{{tablenotes}}[flushleft]
\\footnotesize
\\item \\textit{{Note.}} Values are discounted-total-cost sample means \\(\\pm\\) {round(confidence * 100)}\\% confidence-interval half-widths under each proposal's importance-sampling period weights, over 4{{,}}096 evaluation sample paths per proposal. The instance uses 50\\% initial occupancy; other inputs are given in Table~\\ref{{tab:treatment_pattern_and_penalty_toy_study_1}}.
\\end{{tablenotes}}
\\end{{threeparttable}}
\\end{{table}}"""
    print(table)
    return table


def report_information_relaxation_lower_bounds(experiment_name, config_type='ejor',
                                               confidence=0.95, is_reuse=False):
    from experiments import get_config_by_type
    ser = SimulateEvaluationResult(
        os.path.join('.', 'experiments', 'results', experiment_name),
        '[0-9]*.jsonl',
        get_config_by_type(config_type).env,
        is_reuse=is_reuse,
    )
    rows = []
    for key in sorted(ser.gap_to_information_relaxation, key=str):
        group_id, mutate_val = key
        zero = ser.zero_penalized_information_relaxation_cost[key]
        penalized = ser.penalized_information_relaxation_cost[key]
        improvement = ser.gap_to_information_relaxation[key]
        relative_pct, relative_half_window_pct, _ = ser.information_relaxation_improvement(
            group_id, mutate_val, confidence)
        rows.append({
            'group_id': group_id, 'mutate_val': mutate_val, 'n': zero.n,
            'zero_mean': zero.mean, 'zero_half_window': zero.half_window(confidence),
            'penalized_mean': penalized.mean,
            'penalized_half_window': penalized.half_window(confidence),
            'improvement_mean': improvement.mean,
            'improvement_half_window': improvement.half_window(confidence),
            'relative_improvement_pct': relative_pct,
            'relative_improvement_half_window_pct': relative_half_window_pct,
        })
    level = round(confidence * 100)
    print(f"{experiment_name}: information-relaxation lower bounds ({level}% CI)")
    for row in rows:
        print(f"  group {row['group_id']}  mutate_val={row['mutate_val']}  n={row['n']}")
        print(f"    zero-penalty LB      : {row['zero_mean']:,.2f} +/- {row['zero_half_window']:,.2f}")
        print(f"    penalized LB         : {row['penalized_mean']:,.2f} +/- {row['penalized_half_window']:,.2f}")
        print(f"    improvement (paired) : {row['improvement_mean']:,.2f} +/- {row['improvement_half_window']:,.2f}"
              f"  ({row['relative_improvement_pct']:.2f}% +/- {row['relative_improvement_half_window_pct']:.2f}% of zero-penalty LB)")
    return rows


_LOWER_BOUND_CSV_COLUMNS = (
    ('experiment_name', 'experiment_name'),
    ('group_id', 'group_id'),
    ('mutate_val', 'mutate_val'),
    ('n', 'n'),
    ('zero_information_relaxation_cost_mean', 'zero_mean'),
    ('zero_information_relaxation_cost_half_window', 'zero_half_window'),
    ('penalized_information_relaxation_cost_mean', 'penalized_mean'),
    ('penalized_information_relaxation_cost_half_window', 'penalized_half_window'),
    ('gap_mean', 'improvement_mean'),
    ('gap_half_window', 'improvement_half_window'),
    ('relative_gap_pct', 'relative_improvement_pct'),
    ('relative_gap_half_window_pct', 'relative_improvement_half_window_pct'),
)
_TRAINING_LOG_CSV_COLUMNS = (
    ('training_penalized_lb_mean', 'mean'),
    ('training_penalized_lb_half_window_95', 'half_window'),
)
_TRAINING_LOG_UID_RE = re.compile(r'Training penalty coefficients for uid=([0-9a-f]+)')
_TRAINING_LOG_OBJECTIVE_RE = re.compile(
    r'Iteration (\d+), subproblem objective mean ([-\d.]+) \+/- ([-\d.]+) \(95% CI, N=(\d+)\)')
_TRAINING_LOG_FINISHED_RE = re.compile(r'solver=benders, obj=')


def training_lower_bounds_from_slurm_logs(output_dir):
    candidates = defaultdict(list)
    for file_path in sorted(glob.glob(os.path.join(output_dir, 'slurm-*.out'))):
        with open(file_path, errors='replace') as handle:
            text = handle.read()
        uid_match = _TRAINING_LOG_UID_RE.search(text)
        objective_matches = _TRAINING_LOG_OBJECTIVE_RE.findall(text)
        if uid_match is None or not objective_matches:
            continue
        iteration, mean, half_window, n = objective_matches[-1]
        candidates[uid_match.group(1)].append({
            'finished': _TRAINING_LOG_FINISHED_RE.search(text) is not None,
            'iteration': int(iteration), 'mean': float(mean), 'half_window': float(half_window),
            'n': int(n), 'file': os.path.basename(file_path),
        })
    return {uid: max(rows, key=lambda row: (row['finished'], row['iteration'], row['file']))
            for uid, rows in candidates.items()}


def training_records_by_uid(experiment_names):
    records = {}
    for experiment_name in experiment_names:
        pattern = os.path.join('experiments', 'results', f'{experiment_name}_train', '*.jsonl')
        for file_path in sorted(glob.glob(pattern)):
            with open(file_path) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if (not isinstance(record.get('coefficients'), list)
                            or record.get('tight_penalized_lower_bound') is None):
                        continue
                    records[record['uid']] = {
                        'experiment_name': experiment_name,
                        'training_time_seconds': record.get('training_time_seconds'),
                    }
    return records


def write_information_relaxation_lower_bounds_csv(experiment_names, csv_path, config_type='ejor',
                                                  confidence=0.95, is_reuse=False,
                                                  slurm_output_dir=None):
    training_logs = {} if slurm_output_dir is None else training_lower_bounds_from_slurm_logs(slurm_output_dir)
    training_records = training_records_by_uid(experiment_names)
    columns = list(_LOWER_BOUND_CSV_COLUMNS)
    if slurm_output_dir is not None:
        columns += [(column, f'training_{key}') for column, key in _TRAINING_LOG_CSV_COLUMNS]
    columns.append(('training_time_seconds', 'training_time_seconds'))
    rows = []
    for experiment_name in experiment_names:
        evaluation_rows = []
        if os.path.isdir(os.path.join('experiments', 'results', experiment_name)):
            evaluation_rows = report_information_relaxation_lower_bounds(
                experiment_name, config_type=config_type, confidence=confidence, is_reuse=is_reuse)
        if not evaluation_rows:
            evaluation_rows = [{'group_id': uid} for uid, record in training_records.items()
                               if record['experiment_name'] == experiment_name]
        for row in evaluation_rows:
            log = training_logs.get(row['group_id'], {})
            record = training_records.get(row['group_id'], {})
            rows.append({
                'experiment_name': experiment_name, **row,
                **{f'training_{key}': log.get(key) for _, key in _TRAINING_LOG_CSV_COLUMNS},
                'training_time_seconds': record.get('training_time_seconds'),
            })
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([column for column, _ in columns])
        for row in rows:
            writer.writerow(['' if row.get(key) is None else row[key] for _, key in columns])
    print(f"Saved {len(rows)} rows to {csv_path}")
    return rows

def penalty_shrinkage_rows(ser, key, zetas=(0.0, 0.05), confidence=0.95):
    by_uid = ser.ir_cost_by_ratio_by_uid.get(key, {})
    if not by_uid:
        return [], None
    grids = [tuple(sorted(costs)) for costs in by_uid.values()]
    grid = max(set(grids), key=grids.count)
    uids = [uid for uid, costs in by_uid.items() if tuple(sorted(costs)) == grid]
    weights, strata = _uid_weights_and_strata(
        uids, ser.ir_weight_by_uid.get(key, {}), ser.ir_stratum_by_uid.get(key, {}))
    values = {t: np.array([by_uid[uid][t] for uid in uids], dtype=float) for t in grid}
    base = values[0.0]

    def stats_of(samples):
        stats = StratifiedRunningStats()
        for value, weight, stratum in zip(samples, weights, strata):
            stats.record(value, weight, stratum)
        return stats

    rows = []
    for t in grid:
        bound = stats_of(values[t])
        diff = stats_of(values[t] - base)
        relative, relative_half_window = _stratified_relative_improvement(
            values[t] - base, base, weights, strata, confidence)
        shares = {zeta: stats_of((values[t] < (1.0 - zeta) * base).astype(float)) for zeta in zetas}
        rows.append({
            'group_id': key[0], 'mutate_val': key[1], 'penalty_ratio': t, 'n': len(uids),
            'mean': bound.mean, 'half_window': bound.half_window(confidence),
            'diff_vs_zero_mean': diff.mean, 'diff_vs_zero_half_window': diff.half_window(confidence),
            'relative_diff_pct': relative, 'relative_diff_half_window_pct': relative_half_window,
            'violation_share': {zeta: shares[zeta].mean for zeta in zetas},
            'violation_share_half_window': {zeta: shares[zeta].half_window(confidence) for zeta in zetas},
        })
    t_star = max(rows, key=lambda row: row['mean'])['penalty_ratio']
    gain = stats_of(values[t_star] - values[1.0])
    unit_row = next(row for row in rows if row['penalty_ratio'] == 1.0)
    summary = {
        'group_id': key[0], 'mutate_val': key[1], 'n': len(uids), 'grid': list(grid),
        't_star': t_star,
        'gain_over_unit_mean': gain.mean,
        'gain_over_unit_half_window': gain.half_window(confidence),
        'overfitting': bool(t_star < 1.0 and gain.mean - gain.half_window(confidence) > 0),
        'unit_below_zero': bool(unit_row['diff_vs_zero_mean'] + unit_row['diff_vs_zero_half_window'] < 0),
        'coefficients_source': ser.ir_coefficients_source.get(key),
    }
    return rows, summary


def print_penalty_shrinkage(ser, experiment_name, zetas=(0.0, 0.05), confidence=0.95):
    level = round(confidence * 100)
    all_rows, summaries = [], {}
    print(f"{experiment_name}: penalty-ratio shrinkage diagnostic ({level}% CI)")
    for key in sorted(ser.ir_cost_by_ratio_by_uid, key=str):
        rows, summary = penalty_shrinkage_rows(ser, key, zetas=zetas, confidence=confidence)
        if summary is None:
            continue
        all_rows.extend(rows)
        summaries[key] = summary
        source = summary['coefficients_source'] or {}
        print(f"  group {key[0]}  mutate_val={key[1]}  n={summary['n']}")
        print(f"    coefficients: uid={source.get('uid')}  N={source.get('sample_path_number')}  "
              f"init_state_mode={source.get('init_state_mode')}  "
              f"in-sample objective={source.get('tight_penalized_lower_bound')} "
              f"(all-LP relaxation on the training proposal; indicative only)")
        share_header = ''.join(f"  share<{1 - zeta:.2f}V0" for zeta in zetas)
        print(f"    {'t':>5}  {'LB':>16}  {'vs t=0 (paired)':>26}  {'rel %':>14}{share_header}")
        for row in rows:
            shares = ''.join(f"  {100 * row['violation_share'][zeta]:11.1f}%" for zeta in zetas)
            print(f"    {row['penalty_ratio']:5.2f}  {row['mean']:>16,.1f}"
                  f"  {row['diff_vs_zero_mean']:>+12,.1f} +/- {row['diff_vs_zero_half_window']:<9,.1f}"
                  f"  {row['relative_diff_pct']:>+6.2f} +/- {row['relative_diff_half_window_pct']:<5.2f}{shares}")
        print(f"    t* = {summary['t_star']:g};  f(t*) - f(1) = {summary['gain_over_unit_mean']:+,.1f} "
              f"+/- {summary['gain_over_unit_half_window']:,.1f}")
        if summary['overfitting']:
            print(f"    VERDICT: OVERFITTING -- the out-of-sample bound peaks at t*={summary['t_star']:g} < 1 "
                  f"and the paired gain over t=1 is significant at {level}%.")
        else:
            print("    VERDICT: no evidence of overfitting on this grid.")
        if summary['unit_below_zero']:
            print("    NOTE: the penalized bound (t=1) is significantly below the zero-penalty bound.")
    return all_rows, summaries


def report_penalty_shrinkage(experiment_name, config_type='ejor', zetas=(0.0, 0.05),
                             confidence=0.95, is_reuse=False):
    from experiments import get_config_by_type
    ser = SimulateEvaluationResult(
        os.path.join('.', 'experiments', 'results', experiment_name),
        '[0-9]*.jsonl',
        get_config_by_type(config_type).env,
        is_reuse=is_reuse,
    )
    return print_penalty_shrinkage(ser, experiment_name, zetas=zetas, confidence=confidence)


if __name__ == "__main__":
    
    print(saure_ejor_steady_state_table())
    print()
    print(saure_ejor_replication_table())
    print()
    run_saure_ejor_waiting_time_tables()
