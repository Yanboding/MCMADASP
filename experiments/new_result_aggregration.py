import glob
import json
import os
from pprint import pprint
import pickle

import numpy as np
from collections import defaultdict

from scipy.stats import geom

from utils import RunningStats, load_pickle_if_exists
from visualization import approximate_value_plot_from_running_stats_dict
from visualization.line_plot import approximate_value_plot_from_running_stats

# --- 1. Top-level Factory Functions (Required for Pickling) ---

def dd_int_factory():
    """Returns a defaultdict of integers (0)."""
    return defaultdict(int)

def dd_rs_factory():
    """Returns a defaultdict of RunningStats."""
    return defaultdict(RunningStats)

def dd_dd_rs_factory():
    """Returns a 2-level nested defaultdict of RunningStats."""
    return defaultdict(dd_rs_factory)

def dd_float_factory():
    """Returns a defaultdict of floats (0.0)."""
    return defaultdict(float)

def dd_dd_float_factory():
    """Returns a 2-level nested defaultdict of floats (0.0)."""
    return defaultdict(dd_float_factory)

def dd_dd_dd_float_factory():
    """Returns a 3-level nested defaultdict of floats (0.0)."""
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

'''
cumulateive costs: [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
algo: 
1. calculate cumulative costs after warm-up period
'''
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
    )

    def __init__(self,directory_path, file_pattern, env, group_ids=None, is_reuse=False):
        self.directory_path = directory_path
        self.file_pattern = file_pattern
        self.env = env
        self.waiting_time_targets = np.array([env.holding_cost.get_waiting_target(i) for i in range(env.num_types)])
        self.group_ids = group_ids if group_ids is not None else []
        self.is_reuse = is_reuse
        # Using named functions instead of lambdas
        self.scenario_results = dd_dd_dd_float_factory()
        
        # Built-in types (int, list) and Classes (RunningStats) are already picklable
        self.policy_costs = defaultdict(RunningStats)
        self.zero_penalized_information_relaxation_cost = defaultdict(RunningStats)
        self.penalized_information_relaxation_cost = defaultdict(RunningStats)
        self.zero_penalized_gap = defaultdict(RunningStats)
        self.penalized_gap = defaultdict(RunningStats)
        self.zero_penalized_improvement = defaultdict(RunningStats)
        self.penalized_improvement = defaultdict(RunningStats)

        # Discountred total cost gap after warmup period (if the information relaxation includes the discount factor, then we should use discounted gap.)
        self.after_warmup_policy_costs = defaultdict(RunningStats)
        # Per-sample-path after-warmup cost, keyed (group_id, policy_id) -> {uid: cost}, used for paired comparisons
        self.after_warmup_cost_by_uid = defaultdict(dict)
        # Waiting time target violation
        self.waiting_time_target_ptc_by_type_day = defaultdict(dd_dd_rs_factory)
        self.waiting_time_target_ptc_by_day = defaultdict(dd_rs_factory)
        self.waiting_time_violation = defaultdict(RunningStats)
        
        # Average ovetime utilization per day
        self.overtime_utilization = defaultdict(RunningStats)
        # Average postponement rate
        self.postponement_rate = defaultdict(RunningStats)
        
        self.gap_to_information_relaxation = defaultdict(RunningStats)
        self.improvement = defaultdict(RunningStats)

        self.one_time_cost_by_policy = dd_dd_rs_factory()
        self.solving_time_per_state = defaultdict(RunningStats)
        self.uids_by_policy = defaultdict(set)
        self.number_of_periods = None
        self.information_relaxation_id = None

        pickle_file = os.path.join(self.directory_path, 'simulate_evaluation_result', 'scenario_results.pickle')
        # Make sure the parent directories exist
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
        # # for (group_id, mutate_val), stats in self.zero_penalized_gap.items():
        #     self.zero_penalized_improvement[(group_id, mutate_val)] = self.zero_penalized_gap[(group_id, mutate_val)] / self.zero_penalized_information_relaxation_cost[(group_id, mutate_val)].mean / 0.01
    
    def _has_valid_cache(self, data):
        if data is None:
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
                    # self.lowerbound_load(json.loads(line))
                    self.load(json.loads(line))

    def _save_cache(self, pickle_file):
        res = {key: getattr(self, key) for key in self._CACHE_KEYS}
        with open(pickle_file, 'wb') as f:
            pickle.dump(res, f)

        
    def load(self, data):
        uid = data.get('uid')
        policy_id = data['policy_id']
        if uid is not None and uid in self.uids_by_policy[policy_id]:
            return
        if uid is not None:
            self.uids_by_policy[policy_id].add(uid)

        group_id = data['group_id']
        mutate_val = data['mutate_val']
        if policy_id == 'information_relaxation_only':
            self.gap_to_information_relaxation[(group_id, mutate_val)] += data['penalized_information_relaxation_cost'] - data['zero_information_relaxation_cost']
            self.zero_penalized_information_relaxation_cost[(group_id, mutate_val)] += data['zero_information_relaxation_cost']
            return
        if self.group_ids and group_id not in self.group_ids:
            self.group_ids.append(group_id)
        self.policy_costs[(group_id, policy_id)] += data['total_cost']
        # use the first loaded policy as the information relaxation benchmark
        self.information_relaxation_id = policy_id
        self.zero_penalized_information_relaxation_cost[(group_id, policy_id)] += data['zero_information_relaxation_cost']
        self.penalized_information_relaxation_cost[(group_id, policy_id)] += data['penalized_information_relaxation_cost']
        self.zero_penalized_gap[(group_id, policy_id)] += data['gap_to_zero_information_relaxation']
        self.penalized_gap[(group_id, policy_id)] += data['gap_to_penalized_information_relaxation']
            
        if self.number_of_periods is None:
            self.number_of_periods = len(data['costs'])
        for t, cost in enumerate(data['costs']):
            self.one_time_cost_by_policy[policy_id][t] += cost
        
        warm_up_periods = data["warm_up_periods"]
        # becarful abount the warm-up period.
        costs_after_warmup = data['costs'][warm_up_periods:] if len(data['costs']) > warm_up_periods else data['costs']

        after_warmup_cost = sum(cost for t, cost in enumerate(costs_after_warmup))
        self.after_warmup_policy_costs[(group_id, policy_id)] += after_warmup_cost
        if uid is not None:
            self.after_warmup_cost_by_uid[(group_id, policy_id)][uid] = after_warmup_cost
        
        scheduled_patients = np.array(data["scheduled_patients"])[warm_up_periods:].sum(axis=0) if len(data["scheduled_patients"]) > warm_up_periods else np.array(data["scheduled_patients"]).sum(axis=0)
        
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
            for treatment_type in range(len(scheduled_patients[day])):
                self.waiting_time_target_ptc_by_type_day[policy_id][treatment_type][day] += scheduled_patients_ptc_by_day[day][treatment_type]
                self.waiting_time_target_ptc_by_day[policy_id][day] += total_scheduled_patients_ptc_by_day[day]
        
        for day in range(len(data["overtime"])):
            self.overtime_utilization[(group_id, policy_id)] += data["overtime"][day] / self.env.overtime_capacity * 100
        # calculate the waiting time violation rate
        # scheduled_patients is a 2D array of shape (num_days, num_types), where each entry represents the number of patients of a certain type scheduled on a certain day. We need to calculate the percentage of patients that are scheduled outside of their waiting time target. For each treatment type, we have a waiting time target (e.g., 1 day, 5 days, etc.). We can calculate the cumulative percentage of patients scheduled by each day and compare it to the waiting time target to determine the violation rate.
        patients_outside_target = sum(
            total_scheduled_patients_by_type[t] - cum_scheduled_patients[self.waiting_time_targets[t] - 1][t]
            for t in range(self.env.num_types)
            if total_scheduled_patients_by_type[t] > 0
        )
        if total_scheduled_patients > 0:
            self.waiting_time_violation[(group_id, policy_id)] += patients_outside_target / total_scheduled_patients * 100
        
        self.solving_time_per_state[(group_id, policy_id)] += data.get('solving_time_per_state', 0)
    
    def waiting_time_target_ptc_table(self, days=(1, 5, 10, 15, 20)):
        policy_order = [
            ('approx_penalized_hindsight', 'PH'),
            ('row_gen_alp', 'ALP'),
            ('myopic', 'M'),
        ]
        # Keep only the policies present in the loaded data.
        policies = [(pid, label) for pid, label in policy_order
                    if pid in self.waiting_time_target_ptc_by_type_day]
        num_policies = len(policies)
        num_types = max(len(self.waiting_time_target_ptc_by_type_day[pid])
                        for pid, _ in policies)

        def cell(stats):
            return f'{round(stats.mean)}$\\pm${round(stats.half_window(0.95))}'

        col_spec = 'l' + 'c' * (num_policies * len(days))
        header_groups = '\n'.join(
            f'& \\multicolumn{{{num_policies}}}{{c}}{{\\textbf{{{day} workday{"s" if day > 1 else ""}}}}}'
            for day in days
        )
        cmidrules = '\n'.join(
            f'\\cmidrule(lr){{{2 + i * num_policies}-{1 + (i + 1) * num_policies}}}'
            for i in range(len(days))
        )
        policy_header = ' '.join(f'& {label}' for _ in days for _, label in policies)

        body_lines = []
        for type in range(num_types):
            cells = ' '.join(
                f'& {cell(self.waiting_time_target_ptc_by_type_day[pid][type][day - 1])}'
                for day in days for pid, _ in policies
            )
            body_lines.append(f'{type + 1} {cells} \\\\')
        total_cells = '\n'.join(
            '& ' + ' & '.join(cell(self.waiting_time_target_ptc_by_day[pid][day - 1])
                              for pid, _ in policies)
            for day in days
        )
        body = '\n'.join(body_lines)

        table = f"""\\begin{{table}}[!htbp]
\\centering
\\scriptsize
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\renewcommand{{\\arraystretch}}{{1.05}}
\\caption{{Percentage of cases initiated within given number of workdays under the Penalized Hindsight (PH), ALP, and Myopic (M) policies. $I=18$, $C_r=120$, $C_o=15$, $o=100$, $N=25$, $g_i=2000$, $\\lambda=8.25 \\text{{ with maximum }} 25$, and $\\gamma=0.95$. The waiting-time penalty, treatment patterns, and arrivals are shown in Tables~\\ref{{tab:treatment_pattern_case_study}} and~\\ref{{tab:wait_time_penalty_case_study}}, respectively.}}
\\label{{tab:policy_performance_095_case_study}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{{col_spec}}}
\\toprule
\\textbf{{Type}}
{header_groups} \\\\
{cmidrules}
{policy_header} \\\\
\\midrule
{body}
\\textbf{{Total}}
{total_cells} \\\\
\\bottomrule
\\end{{tabular}}
}}
\\end{{table}}"""
        return table
    
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

    def improvement_over_baseline(self, group_id, policy_id, baseline_id='myopic', confidence=0.95):
        """Paired relative improvement (%) of `policy_id` over `baseline_id` in
        after-warmup discounted total cost, matched by uid (common random numbers).

        Uses the ratio-of-paired-means estimator mean(C_b - C_p) / mean(C_b)
        (per-path ratios are undefined when a baseline path has zero cost),
        with a delta-method confidence-interval half-width.

        Returns (improvement_pct, half_width_pct, n_pairs).
        """
        from scipy.stats import norm
        policy_costs = self.after_warmup_cost_by_uid[(group_id, policy_id)]
        baseline_costs = self.after_warmup_cost_by_uid[(group_id, baseline_id)]
        common_uids = policy_costs.keys() & baseline_costs.keys()
        pc = np.array([policy_costs[uid] for uid in common_uids])
        bc = np.array([baseline_costs[uid] for uid in common_uids])
        n = len(common_uids)
        if n < 2 or bc.mean() == 0:
            return 0.0, 0.0, n
        diff = bc - pc
        improvement = diff.mean() / bc.mean() * 100
        cov = np.cov(np.vstack([diff, bc]))
        dm, bm = diff.mean(), bc.mean()
        var = (cov[0, 0] / bm ** 2 - 2 * dm * cov[0, 1] / bm ** 3 + dm ** 2 * cov[1, 1] / bm ** 4) / n
        half_width = norm.ppf((1 + confidence) / 2) * np.sqrt(max(var, 0.0)) * 100
        return improvement, half_width, n

    def overall_performance_table(self, gamma='0.99', baseline_id='myopic', confidence=0.95):
        """Generates the overall case-study performance LaTeX table with
        discounted total cost, relative improvement over the baseline policy,
        wait-time violations, and overtime utilization."""
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

        # Keep only policies present in the data; assume a single group_id.
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
    
    def plot_percentage_improvement(self, scale, xlabel, ylabel, file_name):
        # Plot percentage improvement of myopic and ALP over the information relaxation benchmark
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
# I want to plot discount improvement
if __name__ == "__main__":
    from experiments import get_config_by_type
    config = get_config_by_type('toy')
    env = config.env
    waiting_time_targets = [env.holding_cost.get_waiting_target(i) for i in range(env.num_types)]
    base_results_dir = os.path.join('.', 'experiments', 'results', 'base_toy_study')
    file_pattern = '[0-9]*.jsonl'
    # group_ids = ['73d11360affe39305e7716cf5c42ac04', '841708e72300000ddfd948daf08d6805', 'a3202d39ed34711b47ecebb72aabad43']
    # run_improvement_plots(
    #     base_results_dir=base_results_dir,
    #     file_pattern=file_pattern,
    #     env_info=env_info,
    #     group_ids=group_ids,
    #     is_reuse=False,
    # )
    ser = SimulateEvaluationResult(
            base_results_dir,
            file_pattern,
            env,
            is_reuse=True,
        )
    
    # print(ser.last_decision_period_distribution_table())
    print('ser.zero_penalized_gap')
    pprint(ser.zero_penalized_gap)
    print('ser.penalized_gap')
    pprint(ser.penalized_gap)
    print('ser.after_warmup_policy_costs')
    pprint(ser.after_warmup_policy_costs)
    print('ser.zero_penalized_information_relaxation_cost')
    pprint(ser.zero_penalized_information_relaxation_cost)
    print('ser.penalized_information_relaxation_cost')
    pprint(ser.penalized_information_relaxation_cost)
    print('ser.zero_penalized_improvement')
    pprint(ser.zero_penalized_improvement)
    print('ser.penalized_improvement')
    pprint(ser.penalized_improvement)
    print(ser.waiting_time_target_ptc_table())
    print('Summary table')
    print(ser.performance_summary_table())
    print('Overall performance table')
    print(ser.overall_performance_table(gamma='0.99'))
    # print('gap_to_information_relaxation')
    # pprint(ser.gap_to_information_relaxation)
    # print('improvement')
    # pprint(ser.improvement)
