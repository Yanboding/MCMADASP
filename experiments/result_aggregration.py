import glob
import json
import os
from pprint import pprint

import numpy as np
from collections import defaultdict

from scipy.stats import geom
from torch.backends.cudnn import benchmark

from utils import RunningStats
from visualization import approximate_value_plot_from_running_stats_dict


'''
cumulateive costs: [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
algo: 
1. calculate cumulative costs after warm-up period
'''
class SimulateEvaluationResult:

    def __init__(self,directory_path, file_pattern, fuck):
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
        self.cumulative_cost_by_day = defaultdict(lambda: defaultdict(lambda: RunningStats()))

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
            'hindsight_approx_MC': 'Hindsight Approximation via MC',
            'hindsight_approx_QMC': 'Hindsight Approximation via QMC',
            'lowerbound_hindsight_approx_MC': 'Lower Bound',
            'lowerbound_hindsight_approx_QMC': 'QMC Lower Bound',
            # 'penalized_lowerbound': 'Penalized Lower Bound',
            'penalized_lowerbound_hindsight_approx_MC': 'Penalized Lower Bound',
            'penalized_lowerbound_hindsight_approx_QMC': 'QMC Penalized Lower Bound',
        }
        self.fuck = fuck
        pattern = os.path.join(self.directory_path, self.file_pattern)
        jsonl_files = glob.glob(pattern)
        for file_path in jsonl_files:
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.load(data)
        print(self.cost_by_day)

    def load(self, data):
        agent_name = str(data['agent_name'])
        #print(str(agent_name['args']))
        for day, cost in enumerate(data['costs']):
            self.cost_by_day[agent_name][day] += cost
        warm_up_periods = data["warm_up_periods"]
        costs = data['costs'][warm_up_periods:]
        penalties = data['penalties'] if len(data['penalties']) > 0 else []
        cumulative_costs = np.cumsum(data['costs'][self.fuck:])
        for day, cumulative_cost in enumerate(cumulative_costs):
            self.cumulative_cost_by_day[agent_name][day] += cumulative_cost
        penalties_sum = sum(penalties)
        scheduled_patients = np.array(data["scheduled_patients"])[warm_up_periods:].sum(axis=0)
        cum_scheduled_patients = scheduled_patients.cumsum(axis=0)
        total_scheduled_patients_by_type = scheduled_patients.sum(axis=0)
        scheduled_patients_ptc_by_day = cum_scheduled_patients/total_scheduled_patients_by_type * 100
        total_scheduled_patients = scheduled_patients.sum()
        cum_total_scheduled_patients_by_day = scheduled_patients.sum(axis=1).cumsum(axis=0)
        total_scheuled_patients_ptc_by_day = cum_total_scheduled_patients_by_day/total_scheduled_patients * 100

        self.total_cost_after_warmup[agent_name] += sum(costs) + penalties_sum
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
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)

    def plot_relative_error(self, order_by_agent, x_values, xlabel, ylabel, file_name):
        relative_error_by_day = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        experiment_labels = {}
        mark = {} # agent_name -> ptc -> day
        for name, orders in order_by_agent.items():
            experiment_labels[name] = self.experiment_labels[name]
            for agent, x in zip(orders, x_values):
                relative_error = self.cumulative_cost_by_day[str(agent)][x].half_window(0.9) / self.cumulative_cost_by_day[str(agent)][x].mean * 100
                relative_error_by_day[name][x + self.fuck + 1] += relative_error
                if name not in mark:
                    mark[name] = {}
                if 7 not in mark[name] and relative_error < 7:
                    mark[name][7] = x + self.fuck + 1
                if 6 not in mark[name] and relative_error < 6:
                    mark[name][6] = x + self.fuck + 1
                elif 5 not in mark[name] and relative_error < 5:
                    mark[name][5] = x + self.fuck + 1
                elif 4 not in mark[name] and relative_error < 4:
                    mark[name][4] = x + self.fuck + 1
                elif 3 not in mark[name] and relative_error < 3:
                    mark[name][3] = x + self.fuck + 1
        print('Relative Error by Day:')
        print("mark:", mark)
        approximate_value_plot_from_running_stats_dict(running_stats_dict=relative_error_by_day,
                                                       x_vals=np.array(x_values)+ self.fuck + 1,
                                                       xticks=None,
                                                       xticklabels=None,
                                                       xlabel=xlabel,
                                                       ylabel=ylabel,
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              file_name),
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)

    def plot_warmup_relative_error(self,order_by_agent, x_values, xlabel, ylabel, file_name):
        relative_error_by_day = defaultdict(lambda: defaultdict(lambda: RunningStats()))
        experiment_labels = {}
        mark = {}  # agent_name -> ptc -> day
        for name, orders in order_by_agent.items():
            experiment_labels[name] = self.experiment_labels[name]
            for agent, x in zip(orders, x_values):
                relative_error = self.cost_by_day[str(agent)][x].half_window(0.9) / \
                                 self.cost_by_day[str(agent)][x].mean * 100
                relative_error_by_day[name][x] += relative_error
                if name not in mark:
                    mark[name] = {}
                if 7 not in mark[name] and relative_error < 7:
                    mark[name][7] = x
                if 6 not in mark[name] and relative_error < 6:
                    mark[name][6] = x
                elif 5 not in mark[name] and relative_error < 5:
                    mark[name][5] = x
                elif 4 not in mark[name] and relative_error < 4:
                    mark[name][4] = x
                elif 3 not in mark[name] and relative_error < 3:
                    mark[name][3] = x
        print('Relative Error by Day:')
        print(len(self.cost_by_day[list(self.cost_by_day.keys())[0]]))
        print(relative_error_by_day)
        approximate_value_plot_from_running_stats_dict(running_stats_dict=relative_error_by_day,
                                                       x_vals=np.array(x_values),
                                                       xticks=None,
                                                       xticklabels=None,
                                                       xlabel=xlabel,
                                                       ylabel=ylabel,
                                                       plot_labels=experiment_labels,
                                                       title=None,
                                                       save_file=os.path.join(self.directory_path,
                                                                              file_name),
                                                       is_show_text=False,
                                                       is_set_x_color=False,
                                                       ncol=2)


if __name__=='__main__':
    directory_path = os.path.join('./results', "toy_problem")
    # 5: 33.1989634321917 0.13896181129865617
    # 10: 36.687370600414376 0.1486390341192171
    # 20: 39.3842249382221 0.5255526412672854
    file_pattern = '[0-9]*.jsonl'
    fuck = 250
    ser = SimulateEvaluationResult(directory_path, file_pattern, fuck)

    x_values = [i for i in range(2000-fuck)]
    xlabel = "Periods"
    order_by_agent = {
        "myopic": [{'agent_name': 'myopic','args': {}} for i in x_values],
        # "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
        #             "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
        #                              97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
        #                              93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
        #                              89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
        #                              86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
        #                              82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
        #                              78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
        #                              75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
        #                              72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
        #                              69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
        #                              66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
        #                              64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
        #                              61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
        #                              59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
        #                              4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
        #                              6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
        #                              8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
        #                              9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
        #                              1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
        #                              1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
        #                              1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
        #                              1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
        #                              7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
        #                              4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
        #                              3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
        #                              -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #                              0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
        #                              1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
        #                         for _ in x_values],
        # "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
        #                           'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
        #                                    'future_decision_var_type': 'continuous', 'is_myopic': False,
        #                                    'sample_path_length': None, 'is_include_discount_factor': False,
        #                                    'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
        #                                    "geom_p": 0.05}} for i in x_values],
        # "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
        #                          'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
        #                                   'future_decision_var_type': 'continuous', 'is_myopic': False,
        #                                   'sample_path_length': None, 'is_include_discount_factor': False,
        #                                   'is_quasi_MC': True, "max_periods": int(geom.ppf(0.995, 0.05)),
        #                                   "geom_p": 0.05}} for i in x_values]
    }

    ser.plot_relative_error(order_by_agent, x_values=x_values, xlabel=xlabel, ylabel="Cumulative Cost Relative Error (%)",
                                   file_name="relative_error_sample_path_total_cost.svg")
    x_values = [i for i in range(2000)]
    order_by_agent = {
        "myopic": [{'agent_name': 'myopic', 'args': {}} for i in x_values],
        # "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
        #             "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
        #                              97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
        #                              93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
        #                              89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
        #                              86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
        #                              82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
        #                              78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
        #                              75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
        #                              72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
        #                              69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
        #                              66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
        #                              64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
        #                              61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
        #                              59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
        #                              4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
        #                              6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
        #                              8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
        #                              9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
        #                              1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
        #                              1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
        #                              1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
        #                              1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
        #                              7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
        #                              4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
        #                              3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
        #                              -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #                              0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
        #                              1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
        #                         for _ in x_values],
        # "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
        #                           'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
        #                                    'future_decision_var_type': 'continuous', 'is_myopic': False,
        #                                    'sample_path_length': None, 'is_include_discount_factor': False,
        #                                    'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
        #                                    "geom_p": 0.05}} for i in x_values],
        # "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
        #                          'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
        #                                   'future_decision_var_type': 'continuous', 'is_myopic': False,
        #                                   'sample_path_length': None, 'is_include_discount_factor': False,
        #                                   'is_quasi_MC': True, "max_periods": int(geom.ppf(0.995, 0.05)),
        #                                   "geom_p": 0.05}} for i in x_values]
    }
    ser.plot_warmup_relative_error(order_by_agent, x_values=x_values, xlabel=xlabel, ylabel="One-Time Cost Relative Error (%)",
                                   file_name="relative_error_warmup_period_onetime_cost.svg")
    ser.format_table()

    """
    1. Number of scenarios vs. total cost
    """
    # scenarios_values = [128, 256, 512]
    # scenarios_xlabel = "Number of Scenarios"
    # order_by_agent = {
    #     "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
    #                              'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
    #                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                       'sample_path_length': None, 'is_include_discount_factor': False,
    #                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
    #                                       "geom_p": 0.05}} for n in scenarios_values],
    #     "lowerbound_hindsight_approx_MC": [{'agent_name': 'lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
    #                                                  "geom_p": 0.05}}
    #                                        for n in scenarios_values],
    #     "penalized_lowerbound_hindsight_approx_MC": [{'agent_name': 'penalized_lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
    #                                                  "geom_p": 0.05}}
    #                                        for n in scenarios_values],
    #     "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
    #                           'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
    #                                    'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                    'sample_path_length': None, 'is_include_discount_factor': False,
    #                                    'is_quasi_MC': True, "max_periods": int(geom.ppf(0.995, 0.05)), "geom_p": 0.05}}
    #                          for n in scenarios_values],
    #     "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
    #         "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
    #                          97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
    #                          93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
    #                          89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
    #                          86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
    #                          82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
    #                          78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
    #                          75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
    #                          72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
    #                          69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
    #                          66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
    #                          64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
    #                          61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
    #                          59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
    #                          4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
    #                          6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
    #                          8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
    #                          9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
    #                          1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
    #                          1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
    #                          1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
    #                          1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
    #                          7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
    #                          4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
    #                          3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
    #                          -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                          0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
    #                          1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
    #                     for _ in scenarios_values]
    # }
    # ser.plot_value_function(order_by_agent, x_values=scenarios_values, xlabel=scenarios_xlabel,
    #                         ylabel="Total Cost After Warm-up ($)",
    #                         file_name="scenario_total_cost.svg")
    #
    # """
    # 2. discount factor vs. total cost
    # """
    # discount_values = [0.9, 0.95, 0.98]
    # discount_xlabel = "Discount Factor"
    # order_by_agent = {
    #     "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
    #                              'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                       'sample_path_length': None, 'is_include_discount_factor': False,
    #                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 1 - gamma)),
    #                                       "geom_p": round(1 - gamma, 2)}}
    #                             for gamma in discount_values],
    #     "lowerbound_hindsight_approx_MC": [{'agent_name': 'lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False,
    #                                                  "max_periods": int(geom.ppf(0.995, 1 - gamma)),
    #                                                  "geom_p": round(1 - gamma, 2)}}
    #                                        for gamma in discount_values],
    #     "penalized_lowerbound_hindsight_approx_MC": [{'agent_name': 'penalized_lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False,
    #                                                  "max_periods": int(geom.ppf(0.995, 1 - gamma)),
    #                                                  "geom_p": round(1 - gamma, 2)}}
    #                                        for gamma in discount_values],
    #     "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
    #                           'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                    'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                    'sample_path_length': None, 'is_include_discount_factor': False,
    #                                    'is_quasi_MC': True, "max_periods": int(geom.ppf(0.995, 1-gamma)), "geom_p": round(1-gamma, 2)}}
    #                          for gamma in discount_values],
    #     "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
    #         "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
    #                          97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
    #                          93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
    #                          89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
    #                          86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
    #                          82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
    #                          78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
    #                          75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
    #                          72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
    #                          69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
    #                          66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
    #                          64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
    #                          61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
    #                          59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
    #                          4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
    #                          6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
    #                          8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
    #                          9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
    #                          1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
    #                          1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
    #                          1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
    #                          1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
    #                          7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
    #                          4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
    #                          3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
    #                          -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                          0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
    #                          1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
    #                     for gamma in discount_values]
    # }
    # ser.plot_value_function(order_by_agent, x_values=discount_values, xlabel=discount_xlabel,
    #                         ylabel="Total Cost After Warm-up ($)",
    #                         file_name="discount_total_cost.svg")
    # """
    # 3. truncation level vs. total cost
    # """
    # truncation_values = [50, 80, 99.5]
    # truncation_xlabel = "Truncation Level (%)"
    # order_by_agent = {
    #     "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
    #                              'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                       'sample_path_length': None, 'is_include_discount_factor': False,
    #                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(truncation / 100, 0.05)),
    #                                       "geom_p": 0.05}}
    #                             for truncation in truncation_values],
    #     "lowerbound_hindsight_approx_MC": [{'agent_name': 'lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False,
    #                                                  "max_periods": int(geom.ppf(truncation / 100, 0.05)),
    #                                                  "geom_p": 0.05}}
    #                                        for truncation in truncation_values],
    #     "penalized_lowerbound_hindsight_approx_MC": [{'agent_name': 'penalized_lowerbound_hindsight_approx',
    #                                         'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                                  'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                                  'sample_path_length': None, 'is_include_discount_factor': False,
    #                                                  'is_quasi_MC': False,
    #                                                  "max_periods": int(geom.ppf(truncation / 100, 0.05)),
    #                                                  "geom_p": 0.05}}
    #                                        for truncation in truncation_values],
    #     "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
    #                           'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
    #                                    'future_decision_var_type': 'continuous', 'is_myopic': False,
    #                                    'sample_path_length': None, 'is_include_discount_factor': False,
    #                                    'is_quasi_MC': True, "max_periods": int(geom.ppf(truncation/100, 0.05)),
    #                                    "geom_p": 0.05}}
    #                          for truncation in truncation_values],
    #     "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
    #         "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
    #                          97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
    #                          93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
    #                          89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
    #                          86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
    #                          82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
    #                          78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
    #                          75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
    #                          72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
    #                          69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
    #                          66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
    #                          64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
    #                          61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
    #                          59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
    #                          4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
    #                          6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
    #                          8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
    #                          9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
    #                          1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
    #                          1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
    #                          1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
    #                          1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
    #                          7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
    #                          4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
    #                          3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
    #                          -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #                          0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
    #                          1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
    #                     for gamma in truncation_values]
    # }
    # '''
    # Combine plots into one figure
    # '''
    # ser.plot_value_function(order_by_agent, x_values=truncation_values, xlabel=truncation_xlabel,
    #                         ylabel="Total Cost After Warm-up ($)",
    #                         file_name="truncation_total_cost.svg")


