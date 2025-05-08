# ---------------------------------------------------------------------
# 1.  Immutable data prepared once at import
# ---------------------------------------------------------------------
# Seven interval thresholds   [0,1] , (1,5] , (5,10] , (10,20] , (20,30] , (30,40] , (40,100]
_INTERVAL_MAX = (1, 5, 10, 20, 30, 40, 100)  # tuple → faster than list for read‑only
_LAST_COL = len(_INTERVAL_MAX) - 1  # == 6

# Row templates taken directly from Table3
_ROW_TEMPLATES = {
    "0‑2": (0, 100, 150, 150, 150, 150, 150),
    "3‑6": (0, 0, 0, 50, 100, 100, 150),
    "6‑11": (0, 0, 65, 100, 100, 100, 150),
    "12‑13": (0, 0, 80, 150, 150, 150, 150),
    "14‑16": (0, 0, 0, 40, 80, 100, 150),
    "17": (0, 0, 0, 50, 90, 100, 150),
}

# Build a dense array indexable directly by treatment_type (0‑based slot unused)
#             0       1       2       …                         18
_PENALTY_ROW = [None] * 18

for rng, row in _ROW_TEMPLATES.items():
    a, _, b = rng.replace("‑", "-").partition("-")
    lo = hi = int(a)
    if b:
        hi = int(b)
    for t in range(lo, hi + 1):
        _PENALTY_ROW[t] = row

def _interval_idx(d: float) -> int:    # ≤ 7 comparisons, no loops / search
    if d <= 1:
        return 0
    elif d <= 5:
        return 1
    elif d <= 10:
        return 2
    elif d <= 20:
        return 3
    elif d <= 30:
        return 4
    elif d <= 40:
        return 5
    return _LAST_COL                   # d > 40 → same penalty as (40,100]

def wait_time_penalty(treatment_type: int, wait_days: int, discount_factor: float) -> int:
    """
    O(1) look‑up of the daily wait‑time penalty from Table 3.

    Parameters
    ----------
    treatment_type : int   (1 ≤ type ≤ 18)
    wait_days      : float (workdays)

    Returns
    -------
    int
        Penalty cost.
    """
    penalty = 0
    for k in reversed(range(wait_days)):
        penalty = _PENALTY_ROW[treatment_type][_interval_idx(k)] + discount_factor * penalty
    return penalty

if __name__ =="__main__":
    x = (a, b, c) = (1,2,3)
    print(a, b, c)
