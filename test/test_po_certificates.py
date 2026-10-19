from types import SimpleNamespace
import pytest
from experiments.po_certificates import min_certificate, max_certificate, affine_cut, centered_value, path_record


def fake(**kwargs):
    values = dict(Status=9, SolCount=1, ObjVal=12., ObjBound=8., Runtime=2.)
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_max_uses_bound_not_incumbent():
    c = max_certificate(fake(ObjVal=8., ObjBound=12.))
    assert c['objective_bound'] == 12 and c['valid']


def test_min_interrupted_lower_certificate():
    c = min_certificate(fake())
    assert c['lower_bound'] == 8 and c['upper_bound'] == 12
    assert not c['exact'] and c['bound_valid']
    c = min_certificate(fake(SolCount=0))
    assert c['objective'] is None and c['lower_bound'] == 8


def test_bad_bounds_rejected():
    assert not min_certificate(fake(ObjBound=float('inf')))['bound_valid']
    assert not max_certificate(fake(Status=12))['valid']
    assert not min_certificate(fake(ObjBound=13.))['bound_valid']


def test_cut_and_centered_constant():
    # Feasible incumbent affine schedule is a global upper cut, even if not optimal.
    intercept, slope = affine_cut(12., [3., -2.], [2., 1.])
    assert intercept == 8 and slope == [3., -2.]
    assert centered_value(12., [3., -2.], [2., 1.], [1., 1.]) == 9


def test_path_record_preserves_costs_weights_and_identity():
    import numpy as np
    p = SimpleNamespace(arrivals=np.array([[1, 0]]), length=1, periods=2,
                        terminal=SimpleNamespace(value='absorbed'), survival_weights=np.ones(2))
    ir = {'fitted': min_certificate(fake()), 'zero': min_certificate(fake(ObjVal=10.)),
          'alp': min_certificate(fake(ObjVal=11.))}
    r = path_record(1, p, ([0], [0], [0, 0]), 2, .125, [1, 2], ir)
    assert r['weight'] == .125 and r['termination_time'] == 2
    assert r['raw_gap_fitted_minus_zero'] == 2
    assert r['raw_gaps_are_incumbent_differences']
    assert r['path_hash'] and r['initial_state_hash'] and r['ir'] == ir


def test_unavailable_gurobi_bound_is_conservative():
    # GurobiError derives directly from Exception; no solver import is needed.
    class GurobiError(Exception): pass
    class MissingBound:
        Status, SolCount, ObjVal, Runtime = 2, 1, 7., .1
        @property
        def ObjBound(self): raise GurobiError(10005, 'Unable to retrieve attribute ObjBound')
    model = MissingBound()
    assert min_certificate(model)['lower_bound'] == 7.
    assert max_certificate(model)['objective_bound'] == 7.
    model.Status = 9
    assert not min_certificate(model)['bound_valid']
    assert not max_certificate(model)['valid']
