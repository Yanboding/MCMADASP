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


def test_path_record_preserves_only_costs_bounds_weights_and_controls():
    import numpy as np
    p = SimpleNamespace(arrivals=np.array([[1, 0]]), length=1, periods=2,
                        terminal=SimpleNamespace(value='absorbed'), survival_weights=np.ones(2))
    ir = {'fitted': min_certificate(fake()), 'zero': min_certificate(fake(ObjVal=10.)),
          'alp': min_certificate(fake(ObjVal=11.))}
    r = path_record(1, p, 2, .125, 3., 4., ir)
    assert r['weight'] == .125 and r['termination_time'] == 2
    assert r['raw_gap_fitted_minus_zero'] == 2
    assert r['raw_gap_fitted_minus_alp'] == 1
    assert r['cv_adjusted_fitted_ir_cost'] == 9.
    assert r['cv_adjusted_gap_fitted_minus_zero'] == -5.
    assert r['cv_adjusted_gap_fitted_minus_alp'] == -2.
    assert r['policy_control_variate'] == 3. and r['policy_penalty_at_reference'] == 4.
    assert set(r) == {'path_id', 'stratum', 'weight', 'termination_time', 'policy_control_variate',
                      'policy_penalty_at_reference', 'ir', 'raw_gap_fitted_minus_zero',
                      'raw_gap_fitted_minus_alp', 'cv_adjusted_fitted_ir_cost',
                      'cv_adjusted_gap_fitted_minus_zero', 'cv_adjusted_gap_fitted_minus_alp'}
    for name, cert in ir.items():
        assert r['ir'][name] == {k: cert[k] for k in ('objective', 'status')}
    assert ir['fitted']['upper_bound'] == 12.  # serialization must not mutate solver certificates


@pytest.mark.parametrize('missing', ['fitted', 'alp', 'zero'])
def test_adjusted_path_values_require_their_incumbents(missing):
    ir = {name: min_certificate(fake(ObjVal=value))
          for name, value in [('fitted', 12.), ('alp', 11.), ('zero', 10.)]}
    ir[missing] = min_certificate(fake(SolCount=0))
    row = path_record(0, SimpleNamespace(periods=2), 0, 1., 3., 4., ir)
    assert row['cv_adjusted_fitted_ir_cost'] == (None if missing == 'fitted' else 9.)
    assert row['cv_adjusted_gap_fitted_minus_alp'] == (None if missing in ('fitted', 'alp') else -2.)
    assert row['cv_adjusted_gap_fitted_minus_zero'] == (None if missing in ('fitted', 'zero') else -5.)


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
