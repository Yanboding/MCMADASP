"""Solver-independent certificate extraction and paired-path serialization."""
import hashlib
import json
import math

# Numeric/infeasible/unbounded/error statuses do not establish a finite certificate.
BOUND_STATUSES = {2, 7, 8, 9, 10, 11, 15, 16, 17}
STATUS_NAMES = {2: 'OPTIMAL', 3: 'INFEASIBLE', 4: 'INF_OR_UNBD', 5: 'UNBOUNDED',
                7: 'ITERATION_LIMIT', 8: 'NODE_LIMIT', 9: 'TIME_LIMIT', 10: 'SOLUTION_LIMIT',
                11: 'INTERRUPTED', 12: 'NUMERIC', 13: 'SUBOPTIMAL', 15: 'USER_OBJ_LIMIT',
                16: 'WORK_LIMIT', 17: 'MEM_LIMIT'}


def _finite(model, name):
    try:
        value = float(getattr(model, name))
        return value if math.isfinite(value) and abs(value) < 1e99 else None
    except Exception:
        # Gurobi's unavailable-attribute error derives directly from Exception.
        # This guard covers only the individual numeric attribute read.
        return None


def _snapshot(model):
    status = int(model.Status)
    count = int(model.SolCount)
    objective = _finite(model, 'ObjVal') if count else None
    bound = _finite(model, 'ObjBound') if status in BOUND_STATUSES else None
    # LP ObjBound may be unavailable on interruption. An optimal incumbent is exact.
    if status == 2 and objective is not None:
        bound = objective if bound is None else bound
    return status, count, objective, bound


def min_certificate(model):
    status, count, objective, bound = _snapshot(model)
    valid = bound is not None and (objective is None or bound <= objective + 1e-7 * max(1., abs(objective)))
    return dict(objective=objective, lower_bound=bound if valid else None, upper_bound=objective,
                status=STATUS_NAMES.get(status, str(status)), solution_count=count,
                exact=status == 2 and objective is not None, bound_valid=valid,
                runtime_seconds=_finite(model, 'Runtime'))


def max_certificate(model):
    status, count, objective, bound = _snapshot(model)
    valid = bound is not None and (objective is None or bound >= objective - 1e-7 * max(1., abs(objective)))
    return dict(sense='maximize', status=STATUS_NAMES.get(status, str(status)), objective=objective,
                objective_bound=bound if valid else None, valid=valid,
                reason=None if valid else 'No valid finite primary MAX solver bound')


def centered_value(value, gradient, theta, anchor):
    return float(value - sum(g * (t - a) for g, t, a in zip(gradient, theta, anchor)))


def affine_cut(value, gradient, theta):
    return float(value - sum(g * t for g, t in zip(gradient, theta))), list(map(float, gradient))


def stable_hash(value):
    def default(item):
        return item.tolist() if hasattr(item, 'tolist') else item.value
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), default=default,
                                     allow_nan=False).encode()).hexdigest()


def path_record(path_id, path, state, stratum, weight, gradient, ir):
    def difference(other):
        a, b = ir['fitted']['objective'], ir[other]['objective']
        return None if a is None or b is None else a - b
    return dict(path_id=int(path_id), path_hash=stable_hash(dict(arrivals=path.arrivals,
                terminal=path.terminal.value, survival_weights=path.survival_weights)),
                initial_state_hash=stable_hash(state), stratum=int(stratum), weight=float(weight),
                arrival_length=int(path.length), termination_time=int(path.periods), terminal=path.terminal.value,
                policy_gradient=list(map(float, gradient)), ir=ir,
                raw_gap_fitted_minus_zero=difference('zero'), raw_gap_fitted_minus_alp=difference('alp'),
                raw_gaps_are_incumbent_differences=not all(v['exact'] for v in ir.values()))
