from experiments.result_aggregration import SimulateEvaluationResult
import os
from scipy.stats import geom
from pprint import pprint
import json
ALP_coefficients = [-76678.18782338874, 99.99999999999494, 98.99999999999578, 98.00999999999564, 97.02989999999578, 96.0596009999956, 95.09900498999593, 94.14801494009595, 93.20653479069482, 92.27446944278816, 91.35172474836035, 90.43820750087656, 89.53382542586759, 88.63848717160896, 87.75210229989321, 86.87458127689426, 86.0058354641255, 85.1457771094845, 84.29431933839003, 83.45137614500648, 82.61686238355678, 81.79069375972112, 80.97278682212384, 80.16305895390298, 79.3614283643639, 78.56781408072065, 77.7821359399135, 77.00431458051499, 76.23427143471005, 75.4719287203627, 0.0, 0.0, -2.8323496447313526e-12, -4.089823424147671e-12, -3.5550949688705957e-12, -4.280202035332999e-12, -4.828961143224857e-12, -4.357336517758216e-12, -3.652951684357216e-12, -4.148418844728308e-12, -3.5462090331465734e-12, -5.871360901465957e-12, -6.3383939786662296e-12, -7.58192738900058e-12, -5.144201849460992e-12, -4.050289456806172e-12, -4.559006334326415e-12, -2.9254567853696344e-12, -4.0099091474932665e-12, -2.9441512285219246e-12, -9.913384438787362e-13, -4.3973680064506373e-13, 6.836551246690057e-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 584.1985059899752, 1564.4795430105723, 2073.9605507526003]
directory_path = os.path.join('.', 'experiments', 'results', "toy_problem")
# 5: 33.1989634321917 0.13896181129865617
# 10: 36.687370600414376 0.1486390341192171
# 20: 39.3842249382221 0.5255526412672854
file_pattern = '[0-9]*.jsonl'
is_reuse = True
ser = SimulateEvaluationResult(directory_path, file_pattern, is_reuse)
'''
CI=120338.7811 \pm 6510.3445
'''
'''
discount_values = [0.9, 0.95, 0.98]
discount_xlabel = "Discount Factor"
order_by_agent = {
    "penalized_lowerbound_hindsight_approx": [{'agent_name': 'penalized_lowerbound_hindsight_approx',
                              'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
                                       'sample_path_length': None, 'is_include_discount_factor': False,
                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 1 - gamma)),
                                       "geom_p": round(1 - gamma, 2)}}
                             for gamma in discount_values],
    "lowerbound_hindsight_approx": [{'agent_name': 'lowerbound_hindsight_approx',
                              'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
                                       'sample_path_length': None, 'is_include_discount_factor': False,
                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 1 - gamma)),
                                       "geom_p": round(1 - gamma, 2)}}
                             for gamma in discount_values],
}

ser.plot_ptc_opt_gap(order_by_agent, x_values=discount_values, xlabel="Discount Factor",
                            ylabel="Percentage Suboptimality Gap (%)",
                            file_name="discount_opt_gap.svg")

order_by_agent = {
    "hindsight_approx": [{'agent_name': 'hindsight_approx',
                              'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
                                       'sample_path_length': None, 'is_include_discount_factor': False,
                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 1 - gamma)),
                                       "geom_p": round(1 - gamma, 2)}}
                             for gamma in discount_values],
    "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0,
                'args': {"coefficients": ALP_coefficients}}
                for gamma in discount_values],
    "lowerbound_hindsight_approx_MC": [{'agent_name': 'lowerbound_hindsight_approx',
                              'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
                                       'sample_path_length': None, 'is_include_discount_factor': False,
                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 1 - gamma)),
                                       "geom_p": round(1 - gamma, 2)}}
                             for gamma in discount_values],
}

# plot overtime
ser.plot_overtime(order_by_agent, x_values=discount_values, xlabel="Discount Factor", ylabel="Number of Overtime (slots)", file_name="discount_overtime.svg")

order_by_agent = {
    "hindsight_approx": [{'agent_name': 'hindsight_approx',
                              'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
                                       'sample_path_length': None, 'is_include_discount_factor': False,
                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 1 - gamma)),
                                       "geom_p": round(1 - gamma, 2)}}
                             for gamma in discount_values],
    "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0,
                'args': {"coefficients": ALP_coefficients}}
                for gamma in discount_values],
    "lowerbound_hindsight_approx_MC": [{'agent_name': 'lowerbound_hindsight_approx',
                              'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                       'future_decision_var_type': 'continuous', 'is_myopic': False,
                                       'sample_path_length': None, 'is_include_discount_factor': False,
                                       'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 1 - gamma)),
                                       "geom_p": round(1 - gamma, 2)}}
                             for gamma in discount_values],
}

# plot average waiting time
ser.plot_average_waiting_time(order_by_agent, x_values=discount_values, xlabel="Discount Factor", ylabel="Average Waiting Time (days)", file_name="discount_average_waiting_time.svg")
'''