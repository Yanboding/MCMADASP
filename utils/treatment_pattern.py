import re

import numpy as np


def str2treatment_pattern(treatment_pattern_str):
    '''
    1*2 + 15*1 + 1*2 + 3*1
    '''
    term_re = re.compile(r'\s*(\d+)\s*\*\s*(\d+)\s*')
    result = []
    for term in treatment_pattern_str.split('+'):
        term = term.strip()
        m = term_re.fullmatch(term)
        if not m:
            raise ValueError(f"Invalid term: '{term}' (expected form 'x*y')")
        x, y = map(int, m.groups())
        result.extend([y] * x)
    return np.array(result)

def concat_ragged(arrays, fill_value=0, axis=1):
    """
    Convert a list of 1-D arrays with different lengths into a
    regular 2-D NumPy array, padding missing spots with `fill_value`.

    Parameters
    ----------
    arrays : list of array-like (1-D)
        The data to concatenate.  Lengths may differ.
    fill_value : scalar, default 0
        The value used to pad and to replace any existing NaNs.
    axis : {1, 0}, default 1
        1  → each *column* is an input array  (shape = [max_len, n_arrays])
        0  → each *row*    is an input array  (shape = [n_arrays, max_len])

    Returns
    -------
    ndarray
        Padded 2-D array.
    """
    # ------ 0. edge case ------------------------------------------------
    if not arrays:
        return np.empty((0, 0), dtype=float)

    # ------ 1. normalise inputs ----------------------------------------
    cleaned = []
    for arr in arrays:
        a = np.asarray(arr, dtype=float).ravel()        # 1-D view
        a = np.nan_to_num(a, nan=fill_value)            # replace NaNs
        cleaned.append(a)

    # ------ 2. work out final shape ------------------------------------
    max_len   = max(a.size for a in cleaned)
    n_arrays  = len(cleaned)

    if axis == 1:                       # columns = inputs
        out_shape = (max_len, n_arrays)
    else:                               # rows = inputs
        out_shape = (n_arrays, max_len)

    # ------ 3. create output filled with pad value ---------------------
    out = np.full(out_shape, fill_value, dtype=float)

    # ------ 4. copy each array into the right slice --------------------
    for idx, a in enumerate(cleaned):
        if axis == 1:                   # column-wise
            out[:a.size, idx] = a
        else:                           # row-wise
            out[idx, :a.size] = a

    return out

def str2treatment_patterns(treatment_pattern_strs):
    treatment_patterns = [str2treatment_pattern(s) for s in treatment_pattern_strs]
    return concat_ragged(treatment_patterns)

def wait_time(l):
    '''
    [0,0,100,100,100, 100, 100,150]
    []
    '''
    cumulative_cost_by_day = []
    for start, end, cost in l:
        cumulative_cost_by_day += [cost]*(end - start)
    cumulative_cost_by_day = np.array(cumulative_cost_by_day)
    return cumulative_cost_by_day

class Treatment:

    def __init__(self, treatment_pattern, cumulate_cost):
        self.treatment_pattern = treatment_pattern
        self.cumulate_cost = cumulate_cost

    def get_cost(self, t, i):
        return self.cumulate_cost[t, i]

if __name__ == '__main__':
    l1_3 = [(0,1,0),(1,5,100),(5,100,150)]
    l4_6 = [(0, 10, 0), (10, 20, 50), (20,40,100), (40, 100, 150)]
    l7_12 = [(0, 5, 0), (5, 10, 65), (10, 40, 100), (40, 100, 150)]
    l13_14 = [(0, 5, 0), (5, 10, 80), (10, 100, 150)]
    l15_17 = [(0, 10, 0), (10, 20, 40), (20, 30, 80), (30, 40, 100), (40, 100, 150)]
    l18 = [(0, 10, 0), (10, 20, 50), (20, 30, 90), (30, 40, 100), (40, 100, 150)]

    print('l1_3:', wait_time(l1_3))
    print('l4_6:', wait_time(l4_6))
    print('l7_12:', wait_time(l7_12))
    print('l13_14:', wait_time(l13_14))
    print('l15_17:', wait_time(l15_17))
    print('l18:', wait_time(l18))
    cumulate_cost = []
    for i in range(18):
        if 0 <= i <3:
            wait_cost_by_day = wait_time(l1_3)
        elif 3 <= i <6:
            wait_cost_by_day = wait_time(l4_6)
        elif 6 <= i <12:
            wait_cost_by_day = wait_time(l7_12)
        elif 12 <= i <14:
            wait_cost_by_day = wait_time(l13_14)
        elif 14 <= i <17:
            wait_cost_by_day = wait_time(l15_17)
        elif 17 <= i < 18:
            wait_cost_by_day = wait_time(l18)
        cumulate_cost.append(wait_cost_by_day)
    cumulate_cost = np.array(cumulate_cost)
    patterns = ['1* 2 + 4 * 1',
                '1*2',
                '1*2+3*1',
                '1* 2 + 15 * 1',
                '1*2 + 15*1 + 1*2 + 3*1',
                '1* 3 + 15 * 2',
                '1* 2',
                '1* 2 + 4 * 1',
                '1* 2 + 9 * 1',
                '1 * 2 + 3 * 1',
                '1 * 2 + 14 * 1',
                '1 * 1',
                '1 * 2 + 19 * 1',
                '1 * 3 + 34 * 2',
                '1 * 2 + 32 * 1',
                '1 * 2 + 36 * 1',
                '1 * 2 + 21 * 1 + 1 * 2 + 14 * 1',
                '1 * 2 + 32 * 1']
    treatment_pattern = str2treatment_patterns(patterns)
    treatment = Treatment(treatment_pattern, cumulate_cost)
    print(treatment.get_cost(2,1))

