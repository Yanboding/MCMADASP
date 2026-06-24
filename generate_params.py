"""Backward-compatible entry point for parameter/experiment generation.

The implementation now lives in the :mod:`param_generation` package. This thin
shim re-exports the public API and the CLI entry point so that existing imports
(e.g. ``from generate_params import generate_test_paths_and_init_state`` or the
private cache helpers used by ``run.py``) and ``python generate_params.py`` keep
working unchanged.
"""

from param_generation import (
    EXPERIMENT_SPECS,
    build_variation_test_env,
    generate_experiment,
    generate_policy_efficiency_data,
    generate_test_paths_and_init_state,
    generate_train_env,
    train_alp_coefficients,
    train_penalty_coefficients,
)
from param_generation.caching import (
    load_cached_training_result as _load_cached_training_result,
    save_training_result as _save_training_result,
)
from param_generation.cli import main

__all__ = [
    'EXPERIMENT_SPECS',
    'build_variation_test_env',
    'generate_experiment',
    'generate_policy_efficiency_data',
    'generate_test_paths_and_init_state',
    'generate_train_env',
    'train_alp_coefficients',
    'train_penalty_coefficients',
    'main',
]



if __name__ == '__main__':
    main()

    # The triple-quoted blocks below are inert reference logs captured from past
    # Benders solver runs; they are kept here for benchmarking only. The runnable
    # recipes now live in param_generation/cli.py.
    '''
    Iteration 65, master solved in 0.057006120681762695 seconds
    Iteration 65, master memory used: 0.0894 GB (peak 0.0894 GB)
    Iteration 65, action from master: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 25.021504743196445, 10.053358168725675, 7.063043451772355, 6.216953928591073, 5.947031007584353, 5.810820287738708, 0.0, 11.187008919361203, 7.254799818243785, 6.4588507728931335, 6.234492388031058, 6.1830370923598865, 6.152579129839666, 0.0, 19.50692893662462, 8.474659528470424, 6.056375910077659, 5.460441131475307, 5.28643882624245, 5.179953111836649, 0.0, 8.574288536055477, 6.31731618984728, 5.816184723235822, 5.704460757858072, 5.789927111145126, 5.831256131673501, 0.0, 203.11528608052782, 148.07798996576534]
    Iteration 65, subproblems solved in 0.35s                                                                                                                                                                                                                                                                                  
    Iteration 65, master memory used: 0.0894 GB (peak 0.0894 GB)
    Iteration 65, subproblem memory used: 0.4997 GB (peak 0.6339 GB across 256 workers)
    Iteration 65, adding 256 optimality cuts
    UB: 13320.421376476756, LB: 13320.421375945056, Gap: 5.316996976034716e-07, First-stage cost: 0.0, Cost-to-go estimate: 13320.421375945056
    obj=13320.421376476756, elapsed=312.2s
    '''

    '''
    Iteration 117, master solved in 0.018671035766601562 seconds
    Iteration 117, master memory used: 0.0516 GB (peak 0.0516 GB)
    Iteration 117, action from master: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.616730590447517, 10.030054928650797, 7.342823067108933, 6.654157808104911, 6.4464816885102225, 6.410584260999075, 0.0, 11.033619329902345, 7.288276331903743, 6.734505712965286, 6.53726614555766, 6.5283657524539205, 6.5284141456860345, 0.0, 22.3526664371209, 9.8860296843891, 7.1631333979262015, 6.401340865619632, 6.222369897551891, 6.3568993797326305, 0.0, 9.797455963185683, 7.086005954265959, 6.724634542591972, 6.509801728794599, 6.442347525009969, 6.517717763684011, 0.0, 185.20931236956602, -471.225770275197]
    Iteration 117, subproblems solved in 0.20s                                                                                                                                                                                                                                                                                 
    Iteration 117, master memory used: 0.0516 GB (peak 0.0516 GB)
    Iteration 117, subproblem memory used: 0.4575 GB (peak 0.4708 GB across 62 workers)
    Iteration 117, adding 62 optimality cuts
    UB: 13619.654740421676, LB: 13619.654740421674, Gap: 1.8189894035458565e-12, First-stage cost: 0.0, Cost-to-go estimate: 13619.654740421674
    obj=13619.654740421676, elapsed=675.7s
    '''

    '''
    Iteration 59, master solved in 0.03646492958068848 seconds
    Iteration 59, master memory used: 0.0791 GB (peak 0.0791 GB)
    Iteration 59, action from master: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 24.748031621453524, 9.918235566889901, 6.789947415560918, 5.7070523630518535, 5.233641515655337, 5.025104748457616, 0.0, 11.28160334967443, 7.564451997461522, 6.751135923582664, 6.467225605886129, 6.320329947364923, 6.2789287499572035, 0.0, 16.78486889821841, 7.627737793048174, 5.126671249519612, 3.987258536644841, 3.89802930953473, 3.8762243691548863, 0.0, 7.1606919986247926, 6.065562187121943, 5.362411455521592, 5.315588459573747, 5.469605558957678, 5.77783654895791, 0.0, -872.4827337558502, 169.54391212467092]
    Iteration 59, subproblems solved in 0.19s                                                                                                                                                                                                                                                                                  
    Iteration 59, master memory used: 0.0791 GB (peak 0.0791 GB)
    Iteration 59, subproblem memory used: 0.2732 GB (peak 0.3542 GB across 256 workers)
    Iteration 59, adding 256 optimality cuts
    UB: 12178.941557563008, LB: 12178.941557563005, Gap: 3.637978807091713e-12, First-stage cost: 0.0, Cost-to-go estimate: 12178.941557563005
    obj=12178.941557563008, elapsed=129.1s
    '''



