from experiments.result_aggregration import SimulateEvaluationResult
import os
from scipy.stats import geom
from pprint import pprint
import json
ALP_coefficients = [-699.9999999999966, 100.0, 99.0, 0.0, 0.0, 0.0, 0.0, 300.0, 199.99999999999997]
directory_path = os.path.join('.', 'experiments', 'results', "toy_problem")
# 5: 33.1989634321917 0.13896181129865617
# 10: 36.687370600414376 0.1486390341192171
# 20: 39.3842249382221 0.5255526412672854
file_pattern = '[0-9]*.jsonl'
is_reuse = True
ser = SimulateEvaluationResult(directory_path, file_pattern, is_reuse)
penalty_coefficients = [round(i,1) for i in range(2)]
print(penalty_coefficients)
penalty_xlabel = "Penalty Coefficient"
order_by_agent = {
    "penalized_lowerbound_row_gen_alp": [{'agent_name': 'penalized_lowerbound_row_gen_alp',
                              'args': {"coefficients": ALP_coefficients},
                              'lowerbound_args': {'current_decision_var_type': 'integer', 'future_decision_var_type': 'continuous', 'is_myopic': False, 'is_include_discount_factor':False, 'coefficients': penalty_coefficient}
                              }
                             for penalty_coefficient in penalty_coefficients],
}

ser.plot_ptc_opt_gap(order_by_agent, x_values=penalty_coefficients, xlabel=penalty_xlabel,
                            ylabel="Percentage Suboptimality Gap (%)",
                            file_name="penalty_ptc_opt_gap.svg")

ser.plot_opt_gap(order_by_agent, x_values=penalty_coefficients, xlabel=penalty_xlabel,
                            ylabel="Suboptimality Gap (absolute)",
                            file_name="penalty_opt_gap.svg")