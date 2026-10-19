from types import SimpleNamespace
import pytest
from experiments.po_certificates import min_certificate, max_certificate, affine_cut, centered_value, path_record
from experiments.figure1_po_run import parse_args


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


def test_cli_is_solver_free():
    assert parse_args(['--params', '{"schema_version":1}', '--job_id', 'x']).job_id == 'x'


def test_full_loop_reuses_models_and_saves_primary_bound(tmp_path, monkeypatch):
    import sys
    import types
    import numpy as np
    from experiments.figure1_po_run import run
    created = []
    class Model:
        def __init__(self):
            self.Params = SimpleNamespace()
            self.Status, self.SolCount, self.ObjVal, self.ObjBound, self.Runtime = 2, 1, 13., 13., .01
            self.calls = []
        def update(self): pass
        def optimize(self): self.calls.append(self.ObjVal)
        def addConstr(self, expr): pass
        def dispose(self): pass
    class Env:
        def __init__(self, **kwargs): pass
        def setParam(self, *args): pass
        def start(self): pass
        def dispose(self): pass
    class Vector:
        __array_priority__ = 10000
        X = np.array([3., 4.])
        def __rmatmul__(self, a): return float(np.asarray(a).sum())
    class Xi:
        UB = None
        def __getitem__(self, i): return 0.
    path = SimpleNamespace(arrivals=np.array([[1, 0]]), length=1, periods=2,
            terminal=SimpleNamespace(value='absorbed'), survival_weights=np.ones(2))
    class Agent:
        def __init__(self, *args, **kwargs):
            assert kwargs['current_decision_var_type'] == 'continuous'
            self.sample_path_number = 2
            self.sample_paths = [path, path]
            self.sample_path_weights = [.5, .5]
            self.sample_path_strata = [0, 1]
        def policy_cost_centering(self, policy, state, anchor):
            return np.array([2e8, 2e8]), np.zeros((2, 2)), anchor
        def train_master_builder_fn(self, **kwargs):
            assert kwargs['coefficient_bound'] == float('inf')
            assert kwargs['scenario_upper_bound'] == float('inf')
            return Model(), Vector(), Xi()
        def train_subproblem_builder_fn(self, env, sid, **kwargs):
            assert kwargs['cold_solve'] is False
            model = Model()
            created.append(model)
            def objective(m, theta):
                m.ObjVal = m.ObjBound = 10. + float(np.asarray(theta).sum()) / 10
            return model, None, objective, lambda m, theta: np.zeros(2)
        def _build_importance_sampling_info(self): return {}
    monkeypatch.setitem(sys.modules, 'gurobipy', SimpleNamespace(Env=Env,
        GRB=SimpleNamespace(INFINITY=float('inf')), gurobi=SimpleNamespace(version=lambda: (12, 0, 0))))
    monkeypatch.setitem(sys.modules, 'experiments.experiment_config', SimpleNamespace(
        get_config_by_type=lambda *a, **k: SimpleNamespace(env=SimpleNamespace(discount_factor=.99))))
    monkeypatch.setitem(sys.modules, 'importance_sampling', SimpleNamespace(build_proposal=lambda x: x))
    monkeypatch.setitem(sys.modules, 'generating_function', SimpleNamespace(
        AbsorptionALPPenaltyFunction=lambda *a, **k: SimpleNamespace(number_of_coefficients=2)))
    monkeypatch.setitem(sys.modules, 'decision_maker', SimpleNamespace(ApproxQAgent=Agent,
        ALPRowGenerationAgent=lambda *a, **k: object()))
    monkeypatch.setitem(sys.modules, 'decision_maker.approximate_q_agent', SimpleNamespace(dispose_agent_models=lambda x: None))
    p = dict(schema_version=1, config_id='a', batch_id='b', repetition=1, configuration={}, initial_state=[[0], [0], [0,0]],
        decision_set=dict(first='continuous', future='continuous'),
        theta_domain=dict(coefficient_bound=None, alp_value_floor=False, regularization=0, fixed_coefficients={}),
        seeds=dict(env=1, arrival=2, stop_time=3, initial_state=4),
        reference=dict(coefficients=[1,2], policy='alp'), sampling=dict(sample_path_number=2,
            proposal=dict(type='stratified_geometric', num_strata=2, discount_factor_proposal=.99),
            weight_convention='stratum_mass_over_count', tau_definition='arrival_length_plus_one'),
        solver=dict(workers=1, max_iterations=1, absolute_gap_tolerance=.1,
                    subproblem_time_limit_seconds=None, master_time_limit_seconds=None), output_dir=str(tmp_path))
    r = run(p, 'test')
    assert len(created) == 2 and all(len(m.calls) == 3 for m in created)
    assert r['training']['objective_upper_bound'] == 13
    assert r['training']['full_objective'] == pytest.approx(10.7)
    assert len(r['paths']) == 2 and r['paths'][1]['weight'] == .5
    assert (tmp_path/'batch.json').exists()
    from experiments.repeated_po_schema import validate_batch
    validate_batch(r)
    saved_lower = sum(row['weight'] * (row['ir']['fitted']['lower_bound'] -
        sum(g * (t - a) for g, t, a in zip(row['policy_gradient'],
            r['training']['coefficients'], r['reference']['coefficients']))) for row in r['paths'])
    assert r['training']['objective_lower_bound'] == pytest.approx(saved_lower)


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


def test_generator_default_limits_validate_without_solving(monkeypatch):
    from param_generation import generate_figure1_po as generator
    from experiments.figure1_po_run import validate_params
    monkeypatch.setattr(generator, '_reference', lambda *args: dict(coefficients=[1], policy='alp'))
    manifest = generator.build_manifest('/unused', '/server/project', '/server/results')
    for item in manifest['runs']:
        validate_params(item['params'])
