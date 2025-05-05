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
    total_cost = 0
    total_days = 0
    for s, e, c in l:
        interval_days = e-s
        total_cost += interval_days*c
        total_days += interval_days
    return total_cost/total_days


if __name__ == '__main__':
    l1_3 = [(0,1,0),(1,5,100),(5,20,150)]
    l4_6 = [(0, 10, 0), (10, 20, 55)]
    l7_12 = [(0, 5, 0), (5, 10, 65), (10, 20, 100)]
    l13_14 = [(0, 5, 0), (5, 10, 80), (10, 22, 150)]
    l15_17 = [(0, 10, 0), (10, 20, 40)]
    l18 = [(0, 10, 0), (10, 20, 50)]

    print('l1_3:', wait_time(l1_3))
    print('l4_6:', wait_time(l4_6))
    print('l7_12:', wait_time(l7_12))
    print('l13_14:', wait_time(l13_14))
    print('l15_17:', wait_time(l15_17))
    print('l18:', wait_time(l18))
