import glob
import json
import os
from pprint import pprint

import numpy as np
from collections import defaultdict

from scipy.stats import geom

from utils import RunningStats
from visualization import approximate_value_plot_from_running_stats_dict


class SimulateEvaluationResult:

    def __init__(self,directory_path, file_pattern):
        self.directory_path = directory_path
        self.file_pattern = file_pattern

        self.total_cost = defaultdict(lambda: RunningStats())
        self.total_discounted_cost = defaultdict(lambda: RunningStats())
        self.total_cost_after_warmup = defaultdict(lambda: RunningStats())
        self.total_discounted_cost_after_warmup = defaultdict(lambda: RunningStats())

        self.total_penalty = defaultdict(lambda: RunningStats())
        self.total_discounted_penalty = defaultdict(lambda: RunningStats())
        self.total_penalty_after_warmup = defaultdict(lambda: RunningStats())
        self.total_discounted_penalty_after_warmup = defaultdict(lambda: RunningStats())
        self.cost_by_day = defaultdict(lambda: defaultdict(lambda: RunningStats()))

        self.average_waiting_time = defaultdict(lambda: RunningStats())
        self.average_overtime = defaultdict(lambda: RunningStats())
        self.average_postponing_decision = defaultdict(lambda: RunningStats())

        self.waiting_time_by_type = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        self.waiting_time_target_ptc_by_day = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        self.waiting_time_target_ptc_by_day_type = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: RunningStats())))

        self.overtime_by_day = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        self.overtime_ptc_by_day = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        self.total_overtime_used_ptc = defaultdict(lambda: RunningStats())

        self.postponing_decision_by_type = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        self.experiment_labels = {
            'myopic': 'Myopic Policy',
            'row_gen_alp': 'ALP',
            'hindsight_approx': 'Hindsight Approximation',
            'lowerbound': 'Lower Bound',
            # 'penalized_lowerbound': 'Penalized Lower Bound',
        }

        pattern = os.path.join(self.directory_path, self.file_pattern)
        jsonl_files = glob.glob(pattern)
        for file_path in jsonl_files:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.load(data)
        print(self.total_cost_after_warmup)
    def load(self, data):
        agent_name = str(data['agent_name'])
        #print(str(agent_name['args']))
        for day, cost in enumerate(data['costs']):
            self.cost_by_day[agent_name][day] += cost
        warm_up_periods = data["warm_up_periods"]
        costs = data['costs'][warm_up_periods:]
        cost_sum = sum(costs)
        penalties = data['penalties'][warm_up_periods:] if len(data['penalties']) > 0 else []
        penalties_sum = sum(penalties)
        scheduled_patients = np.array(data["scheduled_patients"])[warm_up_periods:].sum(axis=0)
        cum_scheduled_patients = scheduled_patients.cumsum(axis=0)
        total_scheduled_patients_by_type = scheduled_patients.sum(axis=0)
        scheduled_patients_ptc_by_day = cum_scheduled_patients/total_scheduled_patients_by_type * 100
        total_scheduled_patients = scheduled_patients.sum()
        cum_total_scheduled_patients_by_day = scheduled_patients.sum(axis=1).cumsum(axis=0)
        total_scheuled_patients_ptc_by_day = cum_total_scheduled_patients_by_day/total_scheduled_patients * 100
        maximum_waiting_days = len(total_scheuled_patients_ptc_by_day[total_scheuled_patients_ptc_by_day<100])

        self.total_cost_after_warmup[agent_name] += sum(costs)
        self.total_penalty[agent_name] += penalties_sum
        for day in range(len(scheduled_patients)):
            for type in range(len(scheduled_patients[day])):
                self.average_waiting_time[agent_name].record_batch(day, scheduled_patients[day][type])
                self.waiting_time_by_type[agent_name][type].record_batch(day, scheduled_patients[day][type])
                self.waiting_time_target_ptc_by_day_type[agent_name][type][day] += scheduled_patients_ptc_by_day[day][type]
                self.waiting_time_target_ptc_by_day[agent_name][day] += total_scheuled_patients_ptc_by_day[day]


        overtimes = data['overtime'][warm_up_periods:]
        for day, overtime in enumerate(overtimes):
            self.average_overtime[agent_name] += overtime
            self.overtime_by_day[agent_name][day] += overtime

        '''
        postponing_decisions = data['postponing_decision'][warm_up_periods:]
        for day in range(len(scheduled_patients)):
            for type in range(len(scheduled_patients[day])):
                self.average_postponing_decision[agent_name] += postponing_decisions[day][type]
                self.postponing_decision_by_type[agent_name][type] += postponing_decisions[day][type]
        '''

    def get_experiment_labels(self, stats):
        experiment_labels = {}
        for k in stats.keys():
            agent = eval(k)
            experiment_labels[k] = self.experiment_labels[agent['agent_name']]
        return experiment_labels

    def plot(self):
        '''
        Waiting time by type
        overtime by day
        overtime ptc by day
        '''
        key = list(self.waiting_time_by_type.keys())[0]
        x_values = sorted(list(self.waiting_time_by_type[key].keys()))
        experiment_labels = self.get_experiment_labels(self.waiting_time_by_type)
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.waiting_time_by_type,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel="Patient Type",
                                                       ylabel="Waiting Time (days)",
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path, f'waiting_time_by_type.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)
        x_values = sorted(list(self.overtime_by_day[key].keys()))
        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.overtime_by_day,
                                                       x_vals=x_values,
                                                       xticks=None,
                                                       xticklabels=None,
                                                       xlabel="Day",
                                                       ylabel="Number of overtime (slots)",
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'overtime_used_by_day.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)
        x_values = sorted(list(self.cost_by_day[key].keys()))

        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.cost_by_day,
                                                       x_vals=x_values,
                                                       xticks=None,
                                                       xticklabels=None,
                                                       xlabel="Day",
                                                       ylabel="One-time Cost ($)",
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              f'cost_by_day.svg'),
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)

    def format_table(self):
        kys = self.waiting_time_target_ptc_by_day_type.keys()
        experiment_labels = self.get_experiment_labels(self.waiting_time_target_ptc_by_day_type)
        for agent_name in kys:
            table = defaultdict(list)
            table2 = {}
            for day in [1, 5, 10, 15, 20, 25, 30]:
                total_mean = self.waiting_time_target_ptc_by_day[agent_name][day].mean
                total_hw = self.waiting_time_target_ptc_by_day[agent_name][day].half_window(0.95)
                table2[day] = (round(total_mean, 2), round(total_hw, 3))
                for type in range(len(self.waiting_time_target_ptc_by_day_type[agent_name])):
                    mean = self.waiting_time_target_ptc_by_day_type[agent_name][type][day].mean
                    hw = self.waiting_time_target_ptc_by_day_type[agent_name][type][day].half_window(0.95)
                    table[type+1].append((day, round(mean), round(hw)))
            print(table2)
            print(experiment_labels[agent_name])
            for type in sorted(table.keys()):
                line = f'{type} '
                for (day, ptc, hw) in table[type]:
                    line += f' & {ptc} $\pm$ {hw}'
                line += r' \\'
                print(line)
            line = r'\textbf{Total} '
            for day, (ptc, hw) in table2.items():
                line += f' & {ptc} $\pm$ {hw}'
            line += r' \\'
            print(line)

    def report(self):
        for agent_key, agent_name in self.experiment_labels.items():
            line = (
                f"{agent_name} & "
                f"{self.total_cost_after_warmup[agent_key].mean:.2f}"
                f" $\\pm$ {self.total_cost_after_warmup[agent_key].half_window(0.95):.3f} & "
                f"{self.total_penalty[agent_key].mean:.2f}"
                f" $\\pm$ {self.total_penalty[agent_key].half_window(0.95):.3f} & "
                f"{self.average_waiting_time[agent_key].mean:.2f}"
                f" $\\pm$ {self.average_waiting_time[agent_key].half_window(0.95):.3f} & "
                f"{self.average_overtime[agent_key].mean:.2f}"
                f" $\\pm$ {self.average_overtime[agent_key].half_window(0.95):.3f}"
                r" \\ "
            )
            print(line)

    def plot_value_function(self, order_by_agent, x_values, xlabel, ylabel, file_name):
        self.total_cost_after_warmup_by_x = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        experiment_labels = {}
        for name, orders in order_by_agent.items():
            experiment_labels[name] = self.experiment_labels[name]
            for agent, x in zip(orders, x_values):
                self.total_cost_after_warmup_by_x[name][x] = self.total_cost_after_warmup[str(agent)]
        print(self.total_cost_after_warmup_by_x)

        approximate_value_plot_from_running_stats_dict(running_stats_dict=self.total_cost_after_warmup_by_x,
                                                       x_vals=x_values,
                                                       xticks=x_values,
                                                       xticklabels=x_values,
                                                       xlabel=xlabel,
                                                       ylabel=ylabel,
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              file_name),
                                                       is_show_text=True,
                                                       is_set_x_color=False,
                                                       ncol=2)

if __name__=='__main__':
    directory_path = os.path.join('./results', "ejor_similar_small_case")
    # 5: 33.1989634321917 0.13896181129865617
    # 10: 36.687370600414376 0.1486390341192171
    # 20: 39.3842249382221 0.5255526412672854
    file_pattern = '[0-9]*.jsonl'
    ser = SimulateEvaluationResult(directory_path, file_pattern)
    truncations = [0.5, 0.8, 0.995]
    x_values = [int(geom.ppf(truncation, 0.05)) for truncation in truncations]
    order_by_agent = \
    {
        # "hindsight_approx":[{'agent_name': 'hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
        #                                         'future_decision_var_type': 'continuous', 'is_myopic': False,
        #                                         'sample_path_length': None, 'is_include_discount_factor': False,
        #                                         'is_quasi_MC': False, 'max_periods': int(geom.ppf(truncation, 0.05))}} for truncation in truncations],
        "myopic": [{'agent_name': 'myopic', 'args': {}} for truncation in truncations],
        #"row_gen_alp":[{"agent_name": "row_gen_alp", "obj_val": 38378.436743001745, "param_value": 0, "args": {"coefficients": [-27486.9977716474, 81.450625, 77.37809375, 77.37809374999975, 77.37809374999975, 77.37809375, 77.37809374999937, 73.50918906249917, 69.83372960937426, 66.34204312890549, 63.02494097246049, 59.873693923837244, 56.88000922764559, 54.03600876626328, 51.33420832795001, 48.767497911552496, 46.32912301597468, 44.01266686517609, 0.0, 1.761206831921552e-13, 8.901818342990879e-13, 7.134220612601481e-13, -2.944627073064474e-13, -2.982140298901816e-13, -5.17036151508308e-13, -3.538361424262668e-13, -6.940600446158161e-13, -1.0815765441959863e-13, -3.71153985504416e-13, -5.017454886012157e-13, -4.382004836578732e-13, -7.617989809721963e-13, -6.374236371400009e-13, -5.832139889236659e-13, 0.0, 0.0, 0.0, 77.37809374999983, 136.17577273827973, 179.77864412394328, 211.01772423341168]}} for truncation in truncations],
        "lowerbound":[{'agent_name': 'lowerbound', 'args': {'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'is_include_discount_factor':False}} for truncation in truncations]
        # "lowerbound": [{'agent_name': 'lowerbound_hindsight_approx', 'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
        #                                         'future_decision_var_type': 'continuous', 'is_myopic': False,
        #                                         'sample_path_length': None, 'is_include_discount_factor': False,
        #                                         'is_quasi_MC': False, 'max_periods': int(geom.ppf(truncation, 0.05))}} for truncation in truncations],
    }

    # order_by_agent = \
    #     {
    #         "hindsight_approx": [{'agent_name': 'hindsight_approx',
    #                               'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                        'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                        'sample_path_length': None, 'is_include_discount_factor': False,
    #                                        'is_quasi_MC': False, 'max_periods': int(geom.ppf(truncation, 0.1)), "geom_p": 0.1}} for
    #                              truncation in truncations],
    #         "lowerbound": [{'agent_name': 'lowerbound_hindsight_approx',
    #                         'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                  'is_quasi_MC': False, 'max_periods': int(geom.ppf(truncation, 0.1)), "geom_p": 0.1}} for
    #                        truncation in truncations],
    #     }
    #ser.plot_value_function(order_by_agent, x_values=x_values, xlabel="Truncated Bound", ylabel="Value Function", file_name="baseline_value_function.svg")
    ser.format_table()
    #ser.report()
    ser.plot()
