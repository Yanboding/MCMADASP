import pandas as pd

from visualization import approximate_value_plot

def case_study_plot(file_name):
    result_df = pd.read_csv(file_name, index_col=0)
    approximate_value_plot(result_df,
                           xlabel='Number of Sample Paths',
                           ylabel='Value function',
                           approx_labels=['SAAdvanceAgent', 'SAAdvanceAgent_hindsight'],
                           text_labels=['SAAdvanceAgent', 'SAAdvanceAgent_hindsight'],
                           plot_labels={'SAAdvanceAgent': "Hindsight Policy Performance", 'SAAdvanceAgent_hindsight': "Lower Bound Sample Average Approximation"},
                           save_file='case_study_value_function',
                           x_val_col='num_sample_path')
    approximate_value_plot(result_df,
                           xlabel='Number of Sample Paths',
                           ylabel='Seconds',
                           approx_labels=['SAAdvanceAgent_runtime'],
                           text_labels=['SAAdvanceAgent_runtime'],
                           plot_labels={'SAAdvanceAgent_runtime': "Advance Hindsight Policy"},
                           save_file='case_study_run_time',
                           x_val_col='num_sample_path')

if __name__ == "__main__":
    file_name = 'value_function_data.csv'
    case_study_plot(file_name)