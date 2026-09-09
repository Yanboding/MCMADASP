"""Cold-retry coverage without relying on a platform-specific numerical failure."""
import gurobipy as gp
import pytest
from gurobipy import GRB

from metaheuristic_algorithm import BendersDecompositionSolver


PARAMETERS = ('Method', 'Presolve', 'NumericFocus', 'DualReductions',
              'FeasibilityTol', 'OptimalityTol', 'Threads', 'ScaleFlag')


class ControlledStatusModel:
    """Inject selected statuses; None runs the real optimizer."""
    def __init__(self, model, statuses):
        self.model = model
        self.statuses = statuses
        self.calls = 0
        self.resets = 0
        self.parameters_at_solve = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    @property
    def Status(self):
        if 0 < self.calls <= len(self.statuses):
            status = self.statuses[self.calls - 1]
            if status is not None:
                return status
        return self.model.Status

    def optimize(self):
        self.parameters_at_solve.append(self.parameters())
        self.calls += 1
        if self.calls > len(self.statuses) or self.statuses[self.calls - 1] is None:
            self.model.optimize()

    def reset(self):
        self.resets += 1
        self.model.reset()
        assert self.model.SolCount == 0

    def parameters(self):
        return tuple(getattr(self.model.Params, name) for name in PARAMETERS)


@pytest.fixture
def solver(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with gp.Model('cold_retry_master') as master:
        master.Params.OutputFlag = 0
        master.Params.Method = 1
        master.Params.Threads = 1
        master.Params.Presolve = 2
        master.Params.NumericFocus = 1
        master.Params.FeasibilityTol = 1e-9
        master.Params.OptimalityTol = 1e-9
        action = master.addMVar(1, lb=0, ub=2)
        theta = master.addMVar(1, lb=-GRB.INFINITY)
        master.addConstr(theta <= action)
        master.setObjective(theta.sum(), GRB.MAXIMIZE)
        master.optimize()
        assert master.SolCount == 1
        yield BendersDecompositionSolver(master, [], None, theta, action)


def master_step(solver):
    return solver._solve_master_step(1, 1e-6, False, False)


@pytest.mark.parametrize('status', [GRB.UNBOUNDED, GRB.INF_OR_UNBD, GRB.NUMERIC])
def test_transient_numerical_status_retries_with_original_parameters(solver, status):
    model = ControlledStatusModel(solver.master_model, [status, None])
    solver.master_model = model
    original_parameters = model.parameters()
    action, objective, theta = master_step(solver)
    assert action.tolist() == pytest.approx([2])
    assert theta.tolist() == pytest.approx([2])
    assert objective == pytest.approx(2)
    assert model.calls == 2
    assert model.resets == 1
    assert model.parameters_at_solve == [original_parameters] * 2
    assert model.parameters() == original_parameters


@pytest.mark.parametrize('retry_status', [GRB.NUMERIC, GRB.UNBOUNDED,
                                         GRB.INF_OR_UNBD, GRB.INFEASIBLE])
def test_failed_retry_saves_diagnostics_and_raises_without_another_solve(
        solver, retry_status, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = ControlledStatusModel(solver.master_model, [GRB.NUMERIC, retry_status])
    solver.master_model = model
    original_parameters = model.parameters()
    with pytest.raises(RuntimeError, match='Master model optimal solution not found'):
        master_step(solver)
    assert model.calls == 2
    assert model.resets == 1
    assert model.parameters_at_solve == [original_parameters] * 2
    assert model.parameters() == original_parameters
    assert len(list(tmp_path.glob('master_failure_diagnostics/*/summary.json'))) == 1


def test_true_unbounded_master_still_raises_after_one_retry(solver, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    master = solver.master_model
    free = master.addVar(lb=-GRB.INFINITY)
    master.setObjective(master.getObjective() + free, GRB.MAXIMIZE)
    master.Params.DualReductions = 0
    model = ControlledStatusModel(master, [])
    solver.master_model = model
    original_parameters = model.parameters()
    with pytest.raises(RuntimeError, match='unbounded'):
        master_step(solver)
    assert model.calls == 2
    assert model.resets == 1
    assert model.parameters_at_solve == [original_parameters] * 2


def test_successful_master_does_not_retry(solver):
    model = ControlledStatusModel(solver.master_model, [])
    solver.master_model = model
    _, objective, _ = master_step(solver)
    assert objective == pytest.approx(2)
    assert model.calls == 1
    assert model.resets == 0
