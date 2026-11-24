import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from metaheuristic_algorithm.ppo_solver import (
    PPOSolver,
    advantage_estimates,
    clipped_objective,
    cost_to_go,
    penalties,
)


def linear(inputs, outputs, weight, bias):
    layer = torch.nn.Linear(inputs, outputs).double()
    with torch.no_grad():
        layer.weight.copy_(torch.as_tensor(weight, dtype=torch.float64).reshape(outputs, inputs))
        layer.bias.copy_(torch.as_tensor(bias, dtype=torch.float64).reshape(outputs))
    return torch.nn.Sequential(layer)


def make_solver(policy_network, value_network, simulate, **kwargs):
    options = dict(
        discount_factor=0.5, max_iterations=1, clip_parameter=0.2, value_epochs=5, policy_epochs=5,
        batch_size=64, value_learning_rate=0.01, policy_learning_rate=0.05, seed=0, verbose=False)
    options.update(kwargs)
    return PPOSolver(policy_network, value_network, simulate, **options)


def test_cost_to_go_on_a_path_of_two_periods():
    # scenario 0: period 1 with atomic costs 3, 5, 7 and period 2 with atomic costs 11, 13
    # scenario 1: one period with atomic cost 2
    # scenario 2: three periods with one atomic cost each
    costs = np.array([3.0, 5.0, 7.0, 11.0, 13.0, 2.0, 1.0, 10.0, 100.0])
    weights = np.array([1.0, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0, 0.5, 2.0])
    scenarios = np.array([0, 0, 0, 0, 0, 1, 2, 2, 2])
    expected = np.array([
        3 + 5 + 7 + 1.1 * (11 + 13),
        5 + 7 + 1.1 * (11 + 13),
        7 + 1.1 * (11 + 13),
        11 + 13,
        13,
        2,
        1 + 0.5 * 10 + 2.0 * 100,
        10 + (2.0 / 0.5) * 100,
        100,
    ])
    np.testing.assert_allclose(cost_to_go(costs, weights, scenarios), expected)


def test_advantage_estimates_in_the_two_cases():
    # rows: booking, booking, end of period, booking, end of period
    values = np.array([10.0, 8.0, 5.0, 20.0, 4.0])
    costs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    end_of_period = np.array([False, False, True, False, True])
    next_period_values = np.array([30.0, 40.0])
    advantages = advantage_estimates(values, costs, end_of_period, next_period_values, discount_factor=0.9)
    expected = np.array([
        1 + 8 - 10,
        2 + 5 - 8,
        3 + 0.9 * 30 - 5,
        4 + 4 - 20,
        5 + 0.9 * 40 - 4,
    ])
    np.testing.assert_allclose(advantages, expected)


def test_solver_estimates_advantages_with_an_affine_value_function_approximation():
    value_network = linear(2, 1, [2.0, -1.0], [0.5])
    solver = make_solver(linear(2, 3, np.zeros((3, 2)), np.zeros(3)), value_network, simulate=None,
                         discount_factor=0.9)
    batch = {
        "features": np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]),
        "costs": np.array([1.0, 2.0, 3.0]),
        "end_of_period": np.array([False, False, True]),
        # one end-of-period row with two sampled arrivals
        "next_period_features": np.array([[[1.0, 1.0], [3.0, 1.0]]]),
    }
    value = lambda x: 0.5 + 2 * x[0] - x[1]
    np.testing.assert_allclose(solver.value(batch["features"]), [2.5, -0.5, 2.5])
    expected = np.array([
        1 + value([0, 1]) - value([1, 0]),
        2 + value([2, 2]) - value([0, 1]),
        3 + 0.9 * (value([1, 1]) + value([3, 1])) / 2 - value([2, 2]),
    ])
    np.testing.assert_allclose(solver.estimate_advantages(batch), expected)
    # the exact expectation: the arrival vectors of a row have the weights of next_period_weights
    batch["next_period_weights"] = np.array([0.25, 0.75])
    expected[2] = 3 + 0.9 * (0.25 * value([1, 1]) + 0.75 * value([3, 1])) - value([2, 2])
    np.testing.assert_allclose(solver.estimate_advantages(batch), expected)


def test_clipped_objective_in_its_four_cases():
    ratios = torch.tensor([0.5, 1.5, 0.5, 1.5, 3.0], dtype=torch.float64, requires_grad=True)
    advantages = torch.tensor([2.0, 2.0, -2.0, -2.0, 0.0], dtype=torch.float64)
    weights = torch.tensor([1.0, 1.0, 2.0, 0.5, 1.0], dtype=torch.float64)
    objective = clipped_objective(ratios, advantages, weights, clip_parameter=0.2)
    # nonnegative advantage: max(ratio, 1 - 0.2). negative advantage: min(ratio, 1 + 0.2).
    expected = 1.0 * 0.8 * 2.0 + 1.0 * 1.5 * 2.0 + 2.0 * 0.5 * (-2.0) + 0.5 * 1.2 * (-2.0) + 0.0
    assert objective.item() == pytest.approx(expected)
    objective.backward()
    # a ratio that was replaced no longer changes the objective
    np.testing.assert_allclose(ratios.grad.numpy(), [0.0, 2.0, -4.0, 0.0, 0.0])


class TwoActionProblem:
    """One state and two actions of cost 0 and 1. Every scenario has one period with one atomic step."""

    def __init__(self, policy_network, scenario_number=256, seed=0):
        self.policy_network = policy_network
        self.scenario_number = scenario_number
        self.seed = seed

    def probabilities(self):
        with torch.no_grad():
            scores = self.policy_network(torch.ones(1, dtype=torch.float64))
            return torch.softmax(scores, dim=-1).numpy()

    def simulate(self, iteration):
        rng = np.random.default_rng([self.seed, iteration])
        probabilities = self.probabilities()
        size = self.scenario_number
        actions = (rng.random(size) < probabilities[1]).astype(int)
        return {
            "features": np.ones((size, 1)),
            "feasible": np.ones((size, 2), dtype=bool),
            "actions": actions,
            "probabilities": probabilities[actions],
            "costs": actions.astype(float),
            "scenarios": np.arange(size),
            "periods": np.ones(size, dtype=int),
            "weights": np.ones(size),
            "end_of_period": np.ones(size, dtype=bool),
            "next_period_features": np.ones((size, 1, 1)),
        }


def two_action_solver(seed=0, **kwargs):
    torch.manual_seed(seed)
    policy_network = torch.nn.Sequential(torch.nn.Linear(1, 2)).double()
    value_network = torch.nn.Sequential(torch.nn.Linear(1, 1)).double()
    problem = TwoActionProblem(policy_network, seed=seed)
    solver = make_solver(policy_network, value_network, problem.simulate, **kwargs)
    return problem, solver


def test_probability_of_the_cheaper_action_increases():
    problem, solver = two_action_solver(max_iterations=5)
    before = problem.probabilities()[0]
    result = solver.solve()
    after = problem.probabilities()[0]
    assert after > before + 0.05
    assert result["finished"]
    assert result["iteration"] == 5

    log = result["log"]
    assert [row["iteration"] for row in log] == [1, 2, 3, 4, 5]
    for row in log:
        assert set(row) >= {"iteration", "seconds", "policy_cost", "policy_cost_standard_error",
                            "value_fit", "clipped_objective", "visited_pairs"}
        assert row["visited_pairs"] == 256
        assert 0 <= row["policy_cost"] <= 1
        assert row["policy_cost_standard_error"] > 0
    # the policy cost of this problem is the probability of the action of cost 1
    assert log[-1]["policy_cost"] < log[0]["policy_cost"]


def test_first_log_row_reports_the_simulated_policy_cost():
    problem, solver = two_action_solver(max_iterations=1)
    costs = problem.simulate(0)["costs"]
    result = solver.solve()
    row = result["log"][0]
    assert row["policy_cost"] == pytest.approx(costs.mean())
    assert row["policy_cost_standard_error"] == pytest.approx(costs.std(ddof=1) / np.sqrt(len(costs)))


def test_affine_value_function_approximation_is_fitted_exactly():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(40, 3))
    targets = 1000.0 + features @ np.array([300.0, -200.0, 50.0])

    def simulate(iteration):
        size = len(features)
        return {
            "features": features,
            "feasible": np.ones((size, 2), dtype=bool),
            "actions": np.zeros(size, dtype=int),
            "probabilities": np.full(size, 0.5),
            # one period with one atomic step per scenario, so the policy-path cost is the atomic cost
            "costs": targets,
            "scenarios": np.arange(size),
            "periods": np.ones(size, dtype=int),
            "weights": np.ones(size),
            "end_of_period": np.ones(size, dtype=bool),
            "next_period_features": features[:, None, :],
        }

    torch.manual_seed(0)
    policy_network = torch.nn.Sequential(torch.nn.Linear(3, 2)).double()
    value_network = torch.nn.Sequential(torch.nn.Linear(3, 1)).double()
    solver = make_solver(policy_network, value_network, simulate, value_epochs=0)
    result = solver.solve()
    np.testing.assert_allclose(solver.value(features), targets, rtol=1e-9)
    assert result["log"][0]["value_fit"] == pytest.approx(0.0, abs=1e-6)


def test_value_network_with_a_hidden_layer_is_fitted_by_gradient_steps():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(256, 2))
    targets = 5000.0 + 800.0 * np.tanh(features[:, 0]) - 300.0 * features[:, 1]

    def simulate(iteration):
        size = len(features)
        return {
            "features": features,
            "feasible": np.ones((size, 2), dtype=bool),
            "actions": np.zeros(size, dtype=int),
            "probabilities": np.full(size, 0.5),
            "costs": targets,
            "scenarios": np.arange(size),
            "periods": np.ones(size, dtype=int),
            "weights": np.ones(size),
            "end_of_period": np.ones(size, dtype=bool),
            "next_period_features": features[:, None, :],
        }

    torch.manual_seed(0)
    policy_network = torch.nn.Sequential(torch.nn.Linear(2, 2)).double()
    value_network = torch.nn.Sequential(
        torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1)).double()
    solver = make_solver(policy_network, value_network, simulate, max_iterations=3, value_epochs=200,
                         batch_size=256, value_learning_rate=0.01)
    result = solver.solve()
    total = np.sum((targets - targets.mean()) ** 2)
    # the fit explains most of the variation although the targets are in the thousands
    assert result["log"][-1]["value_fit"] < 0.05 * total
    assert result["log"][-1]["value_fit"] == pytest.approx(np.sum((solver.value(features) - targets) ** 2))


def test_resumed_training_equals_uninterrupted_training():
    _, solver = two_action_solver(max_iterations=4, checkpoint_every=1)
    full = solver.solve()

    class Interrupted(Exception):
        pass

    saved = []

    def checkpoint(state):
        saved.append(state)
        if state["iteration"] == 2:
            raise Interrupted

    _, solver = two_action_solver(max_iterations=4, checkpoint_every=1, checkpoint=checkpoint)
    with pytest.raises(Interrupted):
        solver.solve()
    state = saved[-1]
    assert state["iteration"] == 2
    assert not state["finished"]

    saved.clear()
    _, solver = two_action_solver(seed=0, max_iterations=4, checkpoint_every=1,
                                  checkpoint=saved.append, state=state)
    resumed = solver.solve()
    assert [entry["iteration"] for entry in saved] == [3, 4]
    assert resumed["finished"]
    assert [row["iteration"] for row in resumed["log"]] == [1, 2, 3, 4]
    for key in ("policy_coefficients", "value_coefficients"):
        assert set(resumed[key]) == set(full[key])
        for name in full[key]:
            np.testing.assert_allclose(resumed[key][name], full[key][name], rtol=1e-12, atol=1e-12)


def test_checkpoint_is_called_every_checkpoint_every_iterations_and_at_the_end():
    saved = []
    _, solver = two_action_solver(max_iterations=5, checkpoint_every=2, checkpoint=saved.append)
    solver.solve()
    assert [state["iteration"] for state in saved] == [2, 4, 5]
    assert [state["finished"] for state in saved] == [False, False, True]
    for state in saved:
        assert set(state) >= {"policy_coefficients", "value_coefficients", "optimizer_state",
                              "iteration", "finished", "log"}
        assert all(isinstance(value, np.ndarray) for value in state["policy_coefficients"].values())
        assert all(isinstance(value, np.ndarray) for value in state["value_coefficients"].values())
    # a saved state is a copy, so later iterations do not change it
    assert len(saved[0]["log"]) == 2


def test_finished_state_is_not_trained_again():
    _, solver = two_action_solver(max_iterations=2)
    finished = solver.solve()
    calls = []

    def simulate(iteration):
        calls.append(iteration)
        raise AssertionError("a finished training was simulated again")

    torch.manual_seed(1)
    policy_network = torch.nn.Sequential(torch.nn.Linear(1, 2)).double()
    value_network = torch.nn.Sequential(torch.nn.Linear(1, 1)).double()
    solver = make_solver(policy_network, value_network, simulate, max_iterations=2, state=finished)
    result = solver.solve()
    assert calls == []
    assert result["iteration"] == 2
    for name, value in finished["policy_coefficients"].items():
        np.testing.assert_array_equal(result["policy_coefficients"][name], value)


# ----- PPO algorithm with the penalty -----

def test_penalties_on_a_path_of_two_periods():
    # scenario 0: period 1 with the rows 0, 1, 2 and period 2 with the rows 3, 4. scenario 1: one row.
    mean_action_values = np.array([50.0, 40.0, 30.0, 20.0, 10.0, 7.0])
    costs = np.array([3.0, 5.0, 7.0, 11.0, 13.0, 2.0])
    values = np.array([45.0, 38.0, 25.0, 22.0, 9.0, 6.0])
    weights = np.array([1.0, 1.0, 1.0, 1.1, 1.1, 1.0])
    scenarios = np.array([0, 0, 0, 0, 0, 1])
    expected = np.array([
        50 - 3 - 38,  # booking action: the value function approximation at the next state of the period
        40 - 5 - 25,
        30 - 7 - 1.1 * 22,  # end-of-period action, and the sample path continues
        20 - 11 - 9,
        10 - 13,  # end-of-period action, and the sample path terminates
        7 - 2,
    ])
    np.testing.assert_allclose(penalties(mean_action_values, costs, values, weights, scenarios), expected)


def test_penalized_policy_path_cost_is_the_value_plus_the_mean_advantage_estimates_from_the_row_on():
    rng = np.random.default_rng(0)
    rows_per_period = [[2, 1, 3], [1], [4, 2], [1, 1, 1, 2]]
    scenarios = np.concatenate([np.full(sum(periods), scenario) for scenario, periods in enumerate(rows_per_period)])
    periods = np.concatenate([np.repeat(np.arange(1, len(periods) + 1), periods) for periods in rows_per_period])
    weights = 1.1 ** (periods - 1.0)
    rows = len(scenarios)
    costs, values = rng.uniform(0, 10, size=rows), rng.uniform(20, 60, size=rows)
    mean_action_values = values + rng.normal(0, 5, size=rows)
    penalized = cost_to_go(costs + penalties(mean_action_values, costs, values, weights, scenarios), weights, scenarios)
    np.testing.assert_allclose(penalized, values + cost_to_go(mean_action_values - values, weights, scenarios))
    # without the penalties it is the policy-path cost
    assert not np.allclose(penalized, cost_to_go(costs, weights, scenarios))


class TwoStepProblem:
    """State 0 has a booking action of cost 1, which leads to state 1, and the end-of-period action of cost 4.
    State 1 has the end-of-period action of cost 2 only. Every scenario has one period, after which its sample
    path terminates. The feature of a state is its number. The policy network gives the scores of the two
    atomic actions, booking first."""

    def __init__(self, policy_network, scenario_number=64, seed=0):
        self.policy_network = policy_network
        self.scenario_number = scenario_number
        self.seed = seed
        self.events = []
        self.step_1 = []

    def booking_probability(self):
        with torch.no_grad():
            scores = self.policy_network(torch.zeros(1, dtype=torch.float64))
            return torch.softmax(scores, dim=-1).numpy()[0]

    def simulate(self, iteration):
        self.events.append("simulate")
        rng = np.random.default_rng([self.seed, iteration])
        booking = self.booking_probability()
        columns = {key: [] for key in ("features", "feasible", "actions", "action_probabilities", "costs", "scenarios")}

        def row(scenario, state, action, probabilities, cost):
            columns["features"].append([float(state)])
            columns["feasible"].append([state == 0, True])
            columns["actions"].append(action)
            columns["action_probabilities"].append(probabilities)
            columns["costs"].append(cost)
            columns["scenarios"].append(scenario)

        for scenario in range(self.scenario_number):
            if rng.random() < booking:
                row(scenario, 0, 0, [booking, 1 - booking], 1.0)
                row(scenario, 1, 1, [0.0, 1.0], 2.0)
            else:
                row(scenario, 0, 1, [booking, 1 - booking], 4.0)
        batch = {key: np.array(values) for key, values in columns.items()}
        size = len(batch["actions"])
        batch["probabilities"] = batch["action_probabilities"][np.arange(size), batch["actions"]]
        batch["periods"] = np.ones(size, dtype=int)
        batch["weights"] = np.ones(size)
        batch["end_of_period"] = batch["actions"] == 1
        # the sample path terminates after the period, so Step 3 of the algorithm without the penalty adds nothing
        batch["next_period_features"] = np.zeros((int(batch["end_of_period"].sum()), 1, 1))
        return batch

    def action_values(self, batch, value, rng):
        """Booking in state 0: 1 plus the value function approximation at state 1. End of period: the cost."""
        self.events.append("action_values")
        states = batch["features"][:, 0]
        action_values = np.zeros((len(states), 2))
        action_values[states == 0, 0] = 1.0 + value(np.array([[1.0]]))[0]
        action_values[:, 1] = np.where(states == 0, 4.0, 2.0)
        self.step_1.append({"batch": batch, "action_values": action_values, "values": value(batch["features"])})
        return action_values


def two_step_solver(seed=0, value_network=None, **kwargs):
    torch.manual_seed(seed)
    policy_network = torch.nn.Sequential(torch.nn.Linear(1, 2)).double()
    if value_network is None:
        value_network = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)).double()
    problem = TwoStepProblem(policy_network, seed=seed)
    if kwargs.get("penalty"):
        kwargs.setdefault("action_values", problem.action_values)
    solver = make_solver(policy_network, value_network, problem.simulate, **kwargs)
    fit = solver.fit_value_function
    problem.fitted = []

    def recording_fit(features, targets, rng):
        problem.events.append("fit")
        problem.fitted.append(np.array(targets, dtype=float))
        return fit(features, targets, rng)

    solver.fit_value_function = recording_fit
    return problem, solver


def test_penalty_generated_by_the_value_function_of_the_policy_removes_the_noise():
    # With the booking probability p, the value function is 2 at state 1 and 3p + 4(1 - p) at state 0.
    problem, solver = two_step_solver(penalty=True, value_network=linear(1, 1, [0.0], [0.0]))
    booking = problem.booking_probability()
    value_at_0 = 3 * booking + 4 * (1 - booking)
    with torch.no_grad():
        solver.value_network[0].weight.fill_(2 - value_at_0)
        solver.value_network[0].bias.fill_(value_at_0)
    batch = problem.simulate(0)
    states = batch["features"][:, 0]
    assert {0.0, 1.0} == set(states) and {0, 1} == set(batch["actions"][states == 0])
    step_1 = solver.penalized_policy_path_costs(batch, np.random.default_rng(0))
    value_function = np.where(states == 0, value_at_0, 2.0)
    np.testing.assert_allclose(step_1["values"], value_function)
    # the policy-path cost depends on the selected actions, and the penalized policy-path cost does not
    assert cost_to_go(batch["costs"], batch["weights"], batch["scenarios"])[states == 0].std() > 0.1
    np.testing.assert_allclose(step_1["penalized_costs"], value_function)
    # the advantage estimate of Step 3 is then the action value function minus the value function
    rows = np.arange(len(states))
    estimates = (step_1["action_values"][rows, batch["actions"]] - step_1["mean_action_values"]
                 + step_1["penalized_costs"] - step_1["values"])
    action_value_function = np.where(states == 1, 2.0, np.where(batch["actions"] == 0, 3.0, 4.0))
    np.testing.assert_allclose(estimates, action_value_function - value_function, atol=1e-9)


def test_objective_of_step_3_with_the_penalty():
    # The policy network gives the probabilities (1/8, 2/8, 5/8), and (1/3, 2/3) where the third action is not feasible.
    policy_network = linear(1, 3, np.zeros((3, 1)), np.log([1.0, 2.0, 5.0]))
    solver = make_solver(policy_network, linear(1, 1, [0.0], [0.0]), simulate=None, clip_parameter=0.2,
                         policy_epochs=0, penalty=True, action_values=lambda batch, value, rng: None)
    feasible = np.array([[True, True, True], [True, True, False], [True, True, True], [True, True, True]])
    old = np.array([[0.25, 0.25, 0.5], [0.5, 0.5, 0.0], [0.125, 0.5, 0.375], [0.25, 0.25, 0.5]])
    new = np.array([[1, 2, 5], [8 / 3, 16 / 3, 0], [1, 2, 5], [1, 2, 5]]) / 8
    action_values = np.array([[10.0, 4.0, 7.0], [2.0, 6.0, 0.0], [8.0, 1.0, 5.0], [10.0, 4.0, 7.0]])
    batch = {
        "features": np.ones((4, 1)),
        "feasible": feasible,
        "actions": np.array([0, 1, 1, 2]),
        "action_probabilities": old,
        "weights": np.array([1.0, 1.0, 2.0, 0.5]),
    }
    values = np.array([20.0, 20.0, 20.0, 20.0])
    step_1 = {
        "values": values,
        "action_values": action_values,
        "mean_action_values": (old * action_values).sum(axis=1),
        # the penalized policy-path cost minus the value function approximation: 2, -3, -1, 1.5
        "penalized_costs": values + np.array([2.0, -3.0, -1.0, 1.5]),
    }
    objective = solver.update_policy_with_penalty(batch, step_1, np.random.default_rng(0))

    positive = lambda number: max(number, 0.0)
    expected = 0.0
    for row in range(4):
        total = 0.0
        for action in np.flatnonzero(feasible[row]):
            ratio = new[row, action] / old[row, action]
            first_part = action_values[row, action] - step_1["mean_action_values"][row]
            total += old[row, action] * (positive(first_part) * max(ratio, 0.8) - positive(-first_part) * min(ratio, 1.2))
        selected = batch["actions"][row]
        ratio = new[row, selected] / old[row, selected]
        second_part = step_1["penalized_costs"][row] - values[row]
        total += positive(second_part) * max(ratio, 0.8) - positive(-second_part) * min(ratio, 1.2)
        expected += batch["weights"][row] * total
    assert objective == pytest.approx(expected)
    # the ratios of the selected actions are below 1 - 0.2, above 1 + 0.2, below 1 - 0.2, and above 1 + 0.2
    np.testing.assert_allclose(new[np.arange(4), batch["actions"]] / old[np.arange(4), batch["actions"]],
                               [0.5, 4 / 3, 0.5, 1.25])


def test_first_iteration_with_the_penalty_computes_the_action_values_after_the_fit():
    problem, solver = two_step_solver(max_iterations=1, penalty=True)
    solver.solve()
    assert problem.events == ["simulate", "fit", "action_values"]
    batch = problem.step_1[0]["batch"]
    # Steps 1 and 2 of the first iteration are those of the algorithm without the penalty
    np.testing.assert_allclose(problem.fitted[0], cost_to_go(batch["costs"], batch["weights"], batch["scenarios"]))
    # the action values of Step 3 come from the value function approximation of this Step 2
    np.testing.assert_allclose(problem.step_1[0]["values"], solver.value(batch["features"]))


def test_later_iteration_with_the_penalty_computes_the_action_values_once_before_the_fit():
    problem, solver = two_step_solver(max_iterations=2, penalty=True)
    result = solver.solve()
    assert problem.events == ["simulate", "fit", "action_values", "simulate", "action_values", "fit"]
    step_1 = problem.step_1[1]
    batch, action_values, values = step_1["batch"], step_1["action_values"], step_1["values"]
    # the penalized policy-path cost term by term: the atomic costs plus the penalties from the row to the end of the
    # scenario
    mean_action_values = (batch["action_probabilities"] * action_values).sum(axis=1)
    expected = np.zeros(len(values))
    for row in range(len(values)):
        later = row
        while True:
            last = later + 1 == len(values) or batch["scenarios"][later + 1] != batch["scenarios"][later]
            value_at_next_state = 0.0 if last else values[later + 1]
            penalty = mean_action_values[later] - batch["costs"][later] - value_at_next_state
            expected[row] += batch["costs"][later] + penalty
            if last:
                break
            later += 1
    np.testing.assert_allclose(problem.fitted[1], expected)
    assert not np.allclose(expected, cost_to_go(batch["costs"], batch["weights"], batch["scenarios"]))

    first_rows = np.flatnonzero(np.append(True, batch["scenarios"][1:] != batch["scenarios"][:-1]))
    row = result["log"][1]
    policy_path_costs = cost_to_go(batch["costs"], batch["weights"], batch["scenarios"])[first_rows]
    assert row["policy_cost"] == pytest.approx(policy_path_costs.mean())
    assert row["penalized_policy_cost"] == pytest.approx(expected[first_rows].mean())
    assert row["penalized_policy_cost_standard_error"] == pytest.approx(
        expected[first_rows].std(ddof=1) / np.sqrt(len(first_rows)))
    assert np.isfinite(row["clipped_objective"])


def test_booking_probability_increases_with_the_penalty():
    # booking costs 1 + 2 and the end-of-period action costs 4
    problem, solver = two_step_solver(max_iterations=5, penalty=True)
    before = problem.booking_probability()
    solver.solve()
    assert problem.booking_probability() > before + 0.05


def test_solver_without_the_penalty_computes_no_action_values_of_all_atomic_actions():
    problem, solver = two_step_solver(max_iterations=2)
    result = solver.solve()
    assert problem.events == ["simulate", "fit", "simulate", "fit"]
    assert all("penalized_policy_cost" not in row for row in result["log"])


def test_penalty_needs_the_action_values_of_all_atomic_actions():
    with pytest.raises(ValueError, match="action_values"):
        make_solver(linear(1, 2, np.zeros((2, 1)), np.zeros(2)), linear(1, 1, [0.0], [0.0]), simulate=None,
                    penalty=True)


def test_resumed_training_with_the_penalty_equals_uninterrupted_training():
    _, solver = two_step_solver(max_iterations=4, checkpoint_every=1, penalty=True)
    full = solver.solve()

    class Interrupted(Exception):
        pass

    saved = []

    def checkpoint(state):
        saved.append(state)
        if state["iteration"] == 2:
            raise Interrupted

    _, solver = two_step_solver(max_iterations=4, checkpoint_every=1, penalty=True, checkpoint=checkpoint)
    with pytest.raises(Interrupted):
        solver.solve()
    problem, solver = two_step_solver(max_iterations=4, checkpoint_every=1, penalty=True, state=saved[-1])
    resumed = solver.solve()
    # the third iteration has the value function approximation of the second, so its Step 1 computes the action
    # values before its fit
    assert problem.events[:3] == ["simulate", "action_values", "fit"]
    for key in ("policy_coefficients", "value_coefficients"):
        for name in full[key]:
            np.testing.assert_allclose(resumed[key][name], full[key][name], rtol=1e-12, atol=1e-12)
