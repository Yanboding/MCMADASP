import glob
import json
import os
import numpy as np
from collections import defaultdict
from pprint import pprint
import pickle
import matplotlib.pyplot as plt
from utils import RunningStats, load_pickle_if_exists

def set_fontsize(ax, size):
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(size)

class SimulationResult:

    def __init__(self, directory_path, file_pattern, is_reuse=False, saved_result_filename='scenario_results.pickle'):
        self.directory_path = directory_path
        self.file_pattern = file_pattern
        self.is_reuse = is_reuse
        self.saved_result_filename = saved_result_filename
        pickle_file = os.path.join(self.directory_path, 'simulate_evaluation_result', self.saved_result_filename)
        # Make sure the parent directories exist
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        self.data = load_pickle_if_exists(pickle_file)

        if self.data != None and self.is_reuse:
           self.extract_summary_from_data()
        else:
            pattern = os.path.join(self.directory_path, self.file_pattern)
            jsonl_files = glob.glob(pattern)
            for file_path in jsonl_files:
                with open(file_path, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        self.load(data)
            # calculate gap
            with open(pickle_file, 'wb') as f:
                res = self.save_summary_to_pickle()
                pickle.dump(res, f)
    

    def load(self, data):
        raise NotImplementedError("Subclasses should implement this method to load data from jsonl files.")
    
    def extract_summary_from_data(self):
        raise NotImplementedError("Subclasses should implement this method to extract summary statistics from the loaded data.")
    
    def save_summary_to_pickle(self):
        return None

class LowerBoundAggregation(SimulationResult):

    def __init__(self, directory_path, file_pattern, is_reuse=False):
        # Initialize stats before parent constructor because parent may call load().
        self.gap_between_penalized_and_zero_stats = defaultdict(RunningStats)
        self.zero_penalized_information_relaxation_cost = defaultdict(RunningStats)
        self.percentage_improvement_stats = defaultdict(RunningStats)
        self.coefficient_stats = defaultdict(lambda: np.array([RunningStats() for _ in range(17)]))
        super().__init__(directory_path, file_pattern, is_reuse)
        self._finalize_percentage_improvement()

    def _finalize_percentage_improvement(self):
        self.percentage_improvement_stats = defaultdict(RunningStats)
        for group_id in self.gap_between_penalized_and_zero_stats:
            self.percentage_improvement_stats[group_id] = self.gap_between_penalized_and_zero_stats[group_id] / self.zero_penalized_information_relaxation_cost[group_id].mean / 0.01

    def load(self, data):
        # Implement the logic to load data from jsonl files and update the summary statistics
        group_id = data.get('group_id')
        self.gap_between_penalized_and_zero_stats[group_id] += data['gap_between_penalized_and_zero']
        self.zero_penalized_information_relaxation_cost[group_id] += data['zero_penalized_lower_bound_objective']
        coefficients = data.get('coefficients')
        self.coefficient_stats[group_id] += np.array(coefficients)
    
    def extract_summary_from_data(self):
        # Implement the logic to extract summary statistics from the loaded data
        data = self.data or {}
        self.gap_between_penalized_and_zero_stats = data.get('gap_between_penalized_and_zero_stats', defaultdict(RunningStats))
        self.zero_penalized_information_relaxation_cost = data.get('zero_penalized_information_relaxation_cost', defaultdict(RunningStats))
        self.percentage_improvement_stats = data.get('percentage_improvement_stats', defaultdict(RunningStats))
        if len(self.percentage_improvement_stats) == 0:
            self._finalize_percentage_improvement()
    
    def save_summary_to_pickle(self):
        # Implement the logic to save summary statistics to a dictionary for pickling
        return {
            'gap_between_penalized_and_zero_stats': self.gap_between_penalized_and_zero_stats,
            'zero_penalized_information_relaxation_cost': self.zero_penalized_information_relaxation_cost
        }
    
    def plot_coefficients(self, save_file):
        labels = {
            'acbffa87277103d172340d09fb3d6714': f'20% occupancy',
            'b0b4f1c19b307d6a79f44e80a39b44cf': f'50% occupancy',
            'b25851a57c0a8ed02e9956116cf3c659': f'80% occupancy',
        }
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        x_vals = np.arange(17)
        for group_id, coeff_stats_array in self.coefficient_stats.items():
            means = np.array([stat.mean for stat in coeff_stats_array])
            half_windows = np.array([stat.half_window(0.95) for stat in coeff_stats_array])
            ax.plot(x_vals, means, label=labels.get(group_id, f'Group {group_id}'), marker='o')
            ax.fill_between(x_vals, means - half_windows, means + half_windows, alpha=0.2)
        set_fontsize(ax, 30)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_vals, rotation=0, fontsize=20)
        ax.set_xlabel('Coefficient Index', fontsize=33)
        ax.set_ylabel('Coefficient Value', fontsize=33)
        n_groups = len(self.coefficient_stats)
        ax.legend(fontsize=20, bbox_to_anchor=(0.5, -0.12), loc='upper center', ncol=n_groups)
        fig.tight_layout()
        plt.savefig(save_file, bbox_inches='tight', format='svg')


if __name__ == "__main__":
    directory_path = os.path.join(
        ".", "experiments", "results", "initial_state_variation_impact"
    )
    file_pattern = '[0-9]*.jsonl'
    result = LowerBoundAggregation(directory_path, file_pattern)
    pprint(result.gap_between_penalized_and_zero_stats)
    pprint(result.percentage_improvement_stats)
    print("Coefficient stats:")
    pprint(result.coefficient_stats)
    result.plot_coefficients(os.path.join(directory_path, 'coefficient_plot.svg'))