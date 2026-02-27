from experiments.result_aggregration import SimulateEvaluationResult
import os
from scipy.stats import geom
directory_path = os.path.join('.', 'experiments', 'results', "small_scale_problem")
# 5: 33.1989634321917 0.13896181129865617
# 10: 36.687370600414376 0.1486390341192171
# 20: 39.3842249382221 0.5255526412672854
file_pattern = '[0-9]*.jsonl'
fuck = 250
ser = SimulateEvaluationResult(directory_path, file_pattern, fuck)

"""
1. Number of scenarios vs. total cost
"""
scenarios_values = [128, 256, 512]
scenarios_xlabel = "Number of Scenarios"
order_by_agent = {
    "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
                             'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
                                      'future_decision_var_type': 'continuous', 'is_myopic': False,
                                      'sample_path_length': None, 'is_include_discount_factor': False,
                                      'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
                                      "geom_p": 0.05}} for n in scenarios_values],
    "lowerbound_hindsight_approx_MC": [{'agent_name': 'lowerbound_hindsight_approx',
                                        'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
                                                 'future_decision_var_type': 'continuous', 'is_myopic': False,
                                                 'sample_path_length': None, 'is_include_discount_factor': False,
                                                 'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
                                                 "geom_p": 0.05}}
                                       for n in scenarios_values],
    "penalized_lowerbound_hindsight_approx_MC": [{'agent_name': 'penalized_lowerbound_hindsight_approx',
                                        'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
                                                 'future_decision_var_type': 'continuous', 'is_myopic': False,
                                                 'sample_path_length': None, 'is_include_discount_factor': False,
                                                 'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 0.05)),
                                                 "geom_p": 0.05}}
                                       for n in scenarios_values],
    "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
                          'args': {'sample_path_number': n, 'current_decision_var_type': 'integer',
                                   'future_decision_var_type': 'continuous', 'is_myopic': False,
                                   'sample_path_length': None, 'is_include_discount_factor': False,
                                   'is_quasi_MC': True, "max_periods": int(geom.ppf(0.995, 0.05)), "geom_p": 0.05}}
                         for n in scenarios_values],
    "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
        "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
                         97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
                         93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
                         89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
                         86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
                         82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
                         78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
                         75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
                         72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
                         69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
                         66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
                         64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
                         61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
                         59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
                         4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
                         6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
                         8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
                         9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
                         1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
                         1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
                         1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
                         1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
                         7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
                         4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
                         3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
                         -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
                         1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
                    for _ in scenarios_values]
}
ser.plot_value_function(order_by_agent, x_values=scenarios_values, xlabel=scenarios_xlabel,
                        ylabel="Total Cost After Warm-up ($)",
                        file_name="scenario_total_cost.svg")

"""
2. discount factor vs. total cost
"""
discount_values = [0.9, 0.95, 0.98]
discount_xlabel = "Discount Factor"
order_by_agent = {
    "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
                             'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                      'future_decision_var_type': 'continuous', 'is_myopic': False,
                                      'sample_path_length': None, 'is_include_discount_factor': False,
                                      'is_quasi_MC': False, "max_periods": int(geom.ppf(0.995, 1 - gamma)),
                                      "geom_p": round(1 - gamma, 2)}}
                            for gamma in discount_values],
    "lowerbound_hindsight_approx_MC": [{'agent_name': 'lowerbound_hindsight_approx',
                                        'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                                 'future_decision_var_type': 'continuous', 'is_myopic': False,
                                                 'sample_path_length': None, 'is_include_discount_factor': False,
                                                 'is_quasi_MC': False,
                                                 "max_periods": int(geom.ppf(0.995, 1 - gamma)),
                                                 "geom_p": round(1 - gamma, 2)}}
                                       for gamma in discount_values],
    "penalized_lowerbound_hindsight_approx_MC": [{'agent_name': 'penalized_lowerbound_hindsight_approx',
                                        'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                                 'future_decision_var_type': 'continuous', 'is_myopic': False,
                                                 'sample_path_length': None, 'is_include_discount_factor': False,
                                                 'is_quasi_MC': False,
                                                 "max_periods": int(geom.ppf(0.995, 1 - gamma)),
                                                 "geom_p": round(1 - gamma, 2)}}
                                       for gamma in discount_values],
    "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
                          'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                   'future_decision_var_type': 'continuous', 'is_myopic': False,
                                   'sample_path_length': None, 'is_include_discount_factor': False,
                                   'is_quasi_MC': True, "max_periods": int(geom.ppf(0.995, 1-gamma)), "geom_p": round(1-gamma, 2)}}
                         for gamma in discount_values],
    "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
        "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
                         97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
                         93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
                         89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
                         86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
                         82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
                         78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
                         75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
                         72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
                         69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
                         66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
                         64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
                         61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
                         59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
                         4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
                         6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
                         8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
                         9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
                         1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
                         1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
                         1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
                         1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
                         7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
                         4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
                         3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
                         -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
                         1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
                    for gamma in discount_values]
}
ser.plot_value_function(order_by_agent, x_values=discount_values, xlabel=discount_xlabel,
                        ylabel="Total Cost After Warm-up ($)",
                        file_name="discount_total_cost.svg")
"""
3. truncation level vs. total cost
"""
truncation_values = [50, 80, 99.5]
truncation_xlabel = "Truncation Level (%)"
order_by_agent = {
    "hindsight_approx_MC": [{'agent_name': 'hindsight_approx',
                             'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                      'future_decision_var_type': 'continuous', 'is_myopic': False,
                                      'sample_path_length': None, 'is_include_discount_factor': False,
                                      'is_quasi_MC': False, "max_periods": int(geom.ppf(truncation / 100, 0.05)),
                                      "geom_p": 0.05}}
                            for truncation in truncation_values],
    "lowerbound_hindsight_approx_MC": [{'agent_name': 'lowerbound_hindsight_approx',
                                        'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                                 'future_decision_var_type': 'continuous', 'is_myopic': False,
                                                 'sample_path_length': None, 'is_include_discount_factor': False,
                                                 'is_quasi_MC': False,
                                                 "max_periods": int(geom.ppf(truncation / 100, 0.05)),
                                                 "geom_p": 0.05}}
                                       for truncation in truncation_values],
    "penalized_lowerbound_hindsight_approx_MC": [{'agent_name': 'penalized_lowerbound_hindsight_approx',
                                        'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                                 'future_decision_var_type': 'continuous', 'is_myopic': False,
                                                 'sample_path_length': None, 'is_include_discount_factor': False,
                                                 'is_quasi_MC': False,
                                                 "max_periods": int(geom.ppf(truncation / 100, 0.05)),
                                                 "geom_p": 0.05}}
                                       for truncation in truncation_values],
    "hindsight_approx_QMC": [{'agent_name': 'hindsight_approx',
                          'args': {'sample_path_number': 256, 'current_decision_var_type': 'integer',
                                   'future_decision_var_type': 'continuous', 'is_myopic': False,
                                   'sample_path_length': None, 'is_include_discount_factor': False,
                                   'is_quasi_MC': True, "max_periods": int(geom.ppf(truncation/100, 0.05)),
                                   "geom_p": 0.05}}
                         for truncation in truncation_values],
    "row_gen_alp": [{"agent_name": "row_gen_alp", "obj_val": 103509.14573268672, "param_value": 0, "args": {
        "coefficients": [-124439.1491735946, 100.00000000000125, 99.00000000000158, 98.01000000000204,
                         97.02990000000489, 96.05960100000534, 95.09900499000788, 94.14801494010817,
                         93.2065347907069, 92.27446944280116, 91.35172474837381, 90.43820750089014,
                         89.53382542588147, 88.63848717162263, 87.75210229990664, 86.87458127690832,
                         86.0058354641396, 85.14577710949854, 84.29431933840335, 83.45137614501931,
                         82.61686238356877, 81.790693759733, 80.9727868221357, 80.16305895391379, 79.36142836437372,
                         78.56781408072976, 77.78213593992184, 77.00431458052168, 76.23427143471592,
                         75.47192872036862, 74.71720943316355, 73.9700373388305, 73.23033696544275,
                         72.49803359578816, 71.77305325982952, 71.05532272723185, 70.34476949996036,
                         69.64132180496003, 68.94490858690946, 68.25545950103776, 67.57290490602628,
                         66.89717585696651, 66.22820409839525, 65.56592205741165, 64.91026283683598,
                         64.26116020846872, 63.61854860638285, 62.98236312031773, 62.35253948911323,
                         61.729014094221576, 61.11172395327853, 60.50060671374451, 59.89560064660675,
                         59.29664464013832, 58.70367819373709, 58.11664141179718, 57.5354749976776, 0.0,
                         4.031634944783184e-12, 5.341285752985398e-12, 6.360763785453865e-12,
                         6.9141596653040166e-12, 5.894540502503862e-12, 1.1678075579436753e-11,
                         8.91819363683057e-12, 9.729853009319378e-12, 1.0572344304116647e-11, 1.084559348502949e-11,
                         9.255748944634389e-12, 1.0651402166150002e-11, 1.1302901825256453e-11,
                         1.3446796261207095e-11, 1.3559942965527155e-11, 1.1446425572719307e-11,
                         1.3095012259894909e-11, 1.1786500266550947e-11, 1.401654496166155e-11,
                         1.1981720711284955e-11, 9.728135484477993e-12, 8.87008949164051e-12,
                         1.1577992960795368e-11, 1.14010265975899e-11, 1.407003631426617e-11, 8.983044573617862e-12,
                         7.465571127990325e-12, 6.3398021133489564e-12, 5.782679414667533e-12,
                         4.5311301594266965e-12, 7.499514657331857e-12, 1.8421770789903246e-12, 5.4625888926784e-12,
                         3.3604800426744375e-12, 0.0, 0.0, 3.497033512701823e-12, -1.2708007331954973e-12,
                         -3.5958763552241433e-12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0, 0.0, 584.19850598999, 1433.8274995375318, 1004.4159575772131,
                         1826.7859099987277, 2643.2326739051077, 2643.2326739051114]}}
                    for gamma in truncation_values]
}
'''
Combine plots into one figure
'''
ser.plot_value_function(order_by_agent, x_values=truncation_values, xlabel=truncation_xlabel,
                        ylabel="Total Cost After Warm-up ($)",
                        file_name="truncation_total_cost.svg")
