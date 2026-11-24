import itertools
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from experiments import get_config_by_type
from importance_sampling import build_proposal
from decision_maker.atomic_action_agent import (
    AtomicActionAgent,
    build_policy_network,
    build_value_network,
)

POLICY_NETWORK_SPEC = {"hidden_sizes": [8, 8], "activation": "tanh"}
VALUE_NETWORK_SPEC = {"hidden_sizes": [8, 8], "activation": "tanh"}
AFFINE_VALUE_NETWORK_SPEC = {"hidden_sizes": []}


@pytest.fixture
def env():
    return get_config_by_type("toy64").env


def make_agent(env, **kwargs):
    kwargs.setdefault("policy_network_spec", POLICY_NETWORK_SPEC)
    return AtomicActionAgent(env, discount_factor=env.discount_factor, **kwargs)


def make_training_agent(env, sample_path_number=8, discount_factor_proposal=0.9, **kwargs):
    def initial_state_sampler(iteration, size):
        env.reset_random_seeds()
        if iteration:
            env.init_state_qmc_sampler.fast_forward(iteration * size)
        return [env.generate_initial_state() for _ in range(size)]

    kwargs.setdefault("value_network_spec", VALUE_NETWORK_SPEC)
    return make_agent(
        env,
        sample_path_number=sample_path_number,
        sample_path_proposal=build_proposal(
            {"type": "geometric", "discount_factor_proposal": discount_factor_proposal}),
        initial_state_sampler=initial_state_sampler,
        arrival_sample_number=4,
        **kwargs,
    )


def random_state(env, rng, maximum_waitlist=3):
    total = rng.integers(0, env.regular_capacity + env.overtime_capacity + 1, size=env.planning_horizon)
    total[-1] = 0
    regular_bookings = np.minimum(total, env.regular_capacity)
    overtime_bookings = total - regular_bookings
    waitlist = rng.integers(0, maximum_waitlist + 1, size=env.num_types)
    return (regular_bookings, overtime_bookings, waitlist)


def tight_state():
    # toy64 has C_r = 6 and C_o = 4. Days 2 and 5 are full, day 1 has one overtime slot left.
    return (np.array([6, 6, 5, 6, 6, 4, 0]), np.array([3, 4, 0, 2, 4, 0, 0]), np.array([2, 2]))


def end_of_period(env):
    return env.booking_window_size * env.num_types


def test_feasible_mask_follows_the_atomic_action_space(env):
    agent = make_agent(env)
    N, I = env.booking_window_size, env.num_types
    mask = agent.feasible_mask(tight_state())
    assert mask.shape == (N * I + 1,)
    assert mask.dtype == bool
    assert mask[end_of_period(env)]
    # Booking (n, i) has index n * I + i. Type 1 needs 3 slots and type 2 needs 2 slots on day n.
    expected = {
        (0, 0): False, (0, 1): False,   # day 1: u = 6, v = 3, so one more slot only
        (1, 0): False, (1, 1): False,   # day 2: full
        (2, 0): True, (2, 1): True,     # day 3: u = 5, v = 0
        (3, 0): False, (3, 1): True,    # day 4: u = 6, v = 2, so two more slots
        (4, 0): False, (4, 1): False,   # day 5: full
        (5, 0): True, (5, 1): True,
        (6, 0): True, (6, 1): True,
    }
    for (n, i), feasible in expected.items():
        assert mask[n * I + i] == feasible, (n, i)

    empty_waitlist = (tight_state()[0], tight_state()[1], np.array([0, 2]))
    mask = agent.feasible_mask(empty_waitlist)
    assert not mask[np.arange(N) * I].any()
    assert mask[end_of_period(env)]


def test_atomic_step_of_a_booking_and_of_the_end_of_period_action(env):
    agent = make_agent(env)
    I = env.num_types
    state = tight_state()
    # Booking (4, 2) of the paper: day 4 (index 3), type 2 (index 1), 2 slots, both in overtime.
    next_state, cost = agent.atomic_step(state, 3 * I + 1)
    np.testing.assert_array_equal(next_state[0], [6, 6, 5, 6, 6, 4, 0])
    np.testing.assert_array_equal(next_state[1], [3, 4, 0, 4, 4, 0, 0])
    np.testing.assert_array_equal(next_state[2], [2, 1])
    assert cost == pytest.approx(env.waiting_cost_coeffs[3, 1] + 2 * env.overtime_cost_coeffs[3])
    # Booking (3, 1): day 3 (index 2), type 1 (index 0), 3 slots, one regular and two in overtime.
    next_state, cost = agent.atomic_step(state, 2 * I + 0)
    np.testing.assert_array_equal(next_state[0], [6, 6, 6, 6, 6, 4, 0])
    np.testing.assert_array_equal(next_state[1], [3, 4, 2, 2, 4, 0, 0])
    np.testing.assert_array_equal(next_state[2], [1, 2])
    assert cost == pytest.approx(env.waiting_cost_coeffs[2, 0] + 2 * env.overtime_cost_coeffs[2])

    next_state, cost = agent.atomic_step(state, end_of_period(env))
    assert next_state is None
    assert cost == pytest.approx(env.postponing_cost_coeffs @ state[2])
    # The state passed in is not changed.
    np.testing.assert_array_equal(state[2], [2, 2])


def test_atomic_costs_and_transitions_add_up_to_the_period(env):
    agent = make_agent(env)
    rng = np.random.default_rng(1)
    N, I, H = env.booking_window_size, env.num_types, env.planning_horizon
    booked = 0
    for _ in range(200):
        state = random_state(env, rng)
        current = state
        x = np.zeros((N, I), dtype=int)
        total_cost = 0.0
        while True:
            mask = agent.feasible_mask(current)
            assert mask[end_of_period(env)]
            bookings = np.flatnonzero(mask[:-1])
            if len(bookings) and rng.random() < 0.8:
                action = rng.choice(bookings)
            else:
                action = end_of_period(env)
            next_state, cost = agent.atomic_step(current, action)
            total_cost += cost
            if action == end_of_period(env):
                break
            x[divmod(action, I)] += 1
            current = next_state
        booked += x.sum()
        y = agent.regular_first_overtime(state, x)
        np.testing.assert_array_equal(current[1] - state[1], y)
        assert total_cost == pytest.approx(env.cost_fn(state, (x, y)))
        arrival = env.arrival_generator.rvs(1, rng=rng)[0]
        expected = env.get_next_state(state, (x, y), arrival)
        actual = env.get_next_state(current, (np.zeros((N, I), dtype=int), np.zeros(H, dtype=int)), arrival)
        for actual_component, expected_component in zip(actual, expected):
            np.testing.assert_array_equal(actual_component, expected_component)
    assert booked > 100


def atomic_period_actions(agent, env, state):
    N, I = env.booking_window_size, env.num_types
    found = {}
    stack = [(state, np.zeros((N, I), dtype=int))]
    while stack:
        current, x = stack.pop()
        if x.tobytes() in found:
            continue
        mask = agent.feasible_mask(current)
        assert mask[end_of_period(env)]
        found[x.tobytes()] = x
        for action in np.flatnonzero(mask[:-1]):
            next_state, _ = agent.atomic_step(current, action)
            next_x = x.copy()
            next_x[divmod(action, I)] += 1
            stack.append((next_state, next_x))
    return set(found)


def test_atomic_sequences_form_exactly_the_valid_actions(env):
    agent = make_agent(env)
    rng = np.random.default_rng(2)
    states = [tight_state()] + [random_state(env, rng, maximum_waitlist=2) for _ in range(5)]
    for state in states:
        expected = {np.asarray(x).astype(int).tobytes() for x, _ in env.valid_actions(state)}
        assert atomic_period_actions(agent, env, state) == expected


def test_state_features_are_scaled(env):
    agent = make_agent(env)
    H, I = env.planning_horizon, env.num_types
    state = (
        np.full(H, env.regular_capacity),
        np.full(H, env.overtime_capacity),
        3 * env.arrival_generator.mean_by_type,
    )
    np.testing.assert_allclose(agent.state_features(state), np.ones(2 * H + I))
    features = agent.state_features(tight_state())
    assert features.shape == (2 * H + I,)
    np.testing.assert_allclose(features[:H], tight_state()[0] / env.regular_capacity)
    np.testing.assert_allclose(features[H:2 * H], tight_state()[1] / env.overtime_capacity)


def test_policy_network_gives_probabilities_of_the_feasible_atomic_actions(env):
    N, I, H = env.booking_window_size, env.num_types, env.planning_horizon
    network = build_policy_network(env, POLICY_NETWORK_SPEC)
    scores = network(torch.zeros(2 * H + I, dtype=torch.float64))
    assert scores.shape == (N * I + 1,)

    agent = make_agent(env)
    state = tight_state()
    probabilities = agent.action_probabilities(state)
    mask = agent.feasible_mask(state)
    assert probabilities.shape == (N * I + 1,)
    assert probabilities.sum() == pytest.approx(1.0)
    assert (probabilities[~mask] == 0).all()
    assert (probabilities[mask] > 0).all()


def test_value_network_without_hidden_layers_is_affine(env):
    H, I = env.planning_horizon, env.num_types
    network = build_value_network(env, AFFINE_VALUE_NETWORK_SPEC)
    assert sum(parameter.numel() for parameter in network.parameters()) == 1 + 2 * H + I
    hidden = build_value_network(env, VALUE_NETWORK_SPEC)
    assert sum(parameter.numel() for parameter in hidden.parameters()) > 1 + 2 * H + I

    agent = make_agent(env)

    def value(state):
        with torch.no_grad():
            return network(torch.as_tensor(agent.state_features(state))).item()

    rng = np.random.default_rng(3)
    first, second = random_state(env, rng), random_state(env, rng)
    middle = tuple((a + b) / 2 for a, b in zip(first, second))
    assert value(middle) == pytest.approx((value(first) + value(second)) / 2)


def test_solve_returns_a_valid_action_and_greedy_is_deterministic(env):
    agent = make_agent(env)
    rng = np.random.default_rng(4)
    states = [tight_state()] + [random_state(env, rng, maximum_waitlist=2) for _ in range(5)]
    for state in states:
        obj, (x, y), info = agent.solve(state, 1)
        assert any(np.array_equal(x, valid_x) and np.array_equal(y, valid_y)
                   for valid_x, valid_y in env.valid_actions(state))
        assert obj == pytest.approx(env.cost_fn(state, (x, y)))
        _, (x_again, y_again), _ = agent.solve(state, 1)
        np.testing.assert_array_equal(x, x_again)
        np.testing.assert_array_equal(y, y_again)
        # greedy takes the most probable feasible atomic action at the first step
        first_action = int(np.argmax(agent.action_probabilities(state)))
        if first_action == end_of_period(env):
            assert x.sum() == 0
        else:
            assert x[divmod(first_action, env.num_types)] >= 1


def test_sample_action_selection_is_reproducible_with_a_seed(env):
    state = (np.zeros(env.planning_horizon, dtype=int), np.zeros(env.planning_horizon, dtype=int), np.array([3, 3]))
    coefficients = make_agent(env).policy_coefficients()

    def decisions(seed):
        agent = make_agent(env, policy_coefficients=coefficients, action_selection="sample", seed=seed)
        return [agent.solve(state, 1)[1][0] for _ in range(5)]

    first, again, other = decisions(0), decisions(0), decisions(1)
    for x, x_again in zip(first, again):
        np.testing.assert_array_equal(x, x_again)
        assert any(np.array_equal(x, valid_x) for valid_x, _ in env.valid_actions(state))
    assert any(not np.array_equal(x, x_other) for x, x_other in zip(first, other))
    assert any(not np.array_equal(first[0], x) for x in first[1:])


def test_policy_coefficients_rebuild_the_same_policy(env):
    agent = make_agent(env, seed=5)
    coefficients = agent.policy_coefficients()
    assert all(isinstance(value, np.ndarray) for value in coefficients.values())
    rebuilt = make_agent(env, seed=6, policy_coefficients=coefficients)
    different = make_agent(env, seed=6)
    state = tight_state()
    np.testing.assert_allclose(rebuilt.action_probabilities(state), agent.action_probabilities(state))
    assert not np.allclose(different.action_probabilities(state), agent.action_probabilities(state))


def test_agent_acquires_no_gurobi_token(env, monkeypatch):
    import decision_maker.infinite_rt_agent as infinite_rt_agent

    def fail(*args, **kwargs):
        raise AssertionError("a Gurobi token was acquired")

    monkeypatch.setattr(infinite_rt_agent, "acquire_grb_env", fail)
    agent = make_training_agent(env)
    agent.solve(tight_state(), 1)
    agent.simulate(0)


def test_simulate_records_every_visited_pair_in_path_order(env):
    gamma = env.discount_factor
    agent = make_training_agent(env, sample_path_number=8, discount_factor_proposal=0.9)
    batch = agent.simulate(0)
    H, I, N = env.planning_horizon, env.num_types, env.booking_window_size
    rows = len(batch["actions"])
    assert batch["features"].shape == (rows, 2 * H + I)
    assert batch["feasible"].shape == (rows, N * I + 1)
    for key in ("probabilities", "costs", "scenarios", "periods", "weights", "end_of_period"):
        assert batch[key].shape == (rows,)

    scenarios, periods = batch["scenarios"], batch["periods"]
    assert sorted(set(scenarios)) == list(range(8))
    assert (np.diff(scenarios) >= 0).all()
    np.testing.assert_array_equal(batch["end_of_period"], batch["actions"] == end_of_period(env))
    assert batch["feasible"][np.arange(rows), batch["actions"]].all()
    assert ((batch["probabilities"] > 0) & (batch["probabilities"] <= 1)).all()
    # Phi^(t-1) = gamma^(t-1) / P(tau >= t) for the geometric proposal with discount factor 0.9
    np.testing.assert_allclose(batch["weights"], (gamma / 0.9) ** (periods - 1))

    scale = agent.feature_scale
    for scenario in range(8):
        index = np.flatnonzero(scenarios == scenario)
        assert periods[index[0]] == 1
        assert batch["end_of_period"][index[-1]]
        ends = batch["end_of_period"][index]
        # the period number increases by one after every end-of-period action
        np.testing.assert_array_equal(periods[index], 1 + np.concatenate(([0], np.cumsum(ends)[:-1])))
        for row in index:
            features = batch["features"][row]
            state = tuple(np.rint(component).astype(int) for component in np.split(features * scale, [H, 2 * H]))
            np.testing.assert_allclose(agent.state_features(state), features)
            np.testing.assert_array_equal(agent.feasible_mask(state), batch["feasible"][row])
            assert batch["probabilities"][row] == pytest.approx(
                agent.action_probabilities(state)[batch["actions"][row]])
            next_state, cost = agent.atomic_step(state, batch["actions"][row])
            assert batch["costs"][row] == pytest.approx(cost)
            if not batch["end_of_period"][row]:
                np.testing.assert_allclose(batch["features"][row + 1], agent.state_features(next_state))

    # states that start the next period, for the expectation over the arrivals in the action value function
    end_rows = np.flatnonzero(batch["end_of_period"])
    next_period_features = batch["next_period_features"]
    assert next_period_features.shape == (len(end_rows), 4, 2 * H + I)
    for position, row in enumerate(end_rows):
        u, v, w = np.split(batch["features"][row] * scale, [H, 2 * H])
        for sample in next_period_features[position] * scale:
            next_u, next_v, next_w = np.split(sample, [H, 2 * H])
            np.testing.assert_allclose(next_u, np.append(u[1:], 0))
            np.testing.assert_allclose(next_v, np.append(v[1:], 0))
            arrival = next_w - w
            np.testing.assert_allclose(arrival, np.rint(arrival), atol=1e-9)
            assert (arrival > -1e-9).all()
        if row + 1 == rows or scenarios[row + 1] != scenarios[row]:
            continue
        # the next visited state of the scenario starts the next period with the same bookings
        next_features = batch["features"][row + 1] * scale
        np.testing.assert_allclose(next_features[:H], np.append(u[1:], 0))
        np.testing.assert_allclose(next_features[H:2 * H], np.append(v[1:], 0))


def test_simulate_depends_on_the_iteration_and_the_seed_only(env):
    first = make_training_agent(env, seed=7).simulate(3)
    again = make_training_agent(get_config_by_type("toy64").env, seed=7).simulate(3)
    other_iteration = make_training_agent(env, seed=7).simulate(4)
    other_seed = make_training_agent(env, seed=8).simulate(3)
    for key in first:
        np.testing.assert_array_equal(first[key], again[key])
    assert len(first["actions"]) != len(other_iteration["actions"]) or not np.array_equal(
        first["actions"], other_iteration["actions"])
    assert len(first["actions"]) != len(other_seed["actions"]) or not np.array_equal(
        first["actions"], other_seed["actions"])


def test_simulate_uses_the_mean_arrival_for_an_affine_value_network(env):
    agent = make_training_agent(env, value_network_spec=AFFINE_VALUE_NETWORK_SPEC)
    batch = agent.simulate(0)
    H = env.planning_horizon
    end_rows = np.flatnonzero(batch["end_of_period"])
    assert batch["next_period_features"].shape[:2] == (len(end_rows), 1)
    waitlist = batch["features"][end_rows][:, 2 * H:] * agent.feature_scale[2 * H:]
    next_waitlist = batch["next_period_features"][:, 0, 2 * H:] * agent.feature_scale[2 * H:]
    np.testing.assert_allclose(next_waitlist - waitlist,
                               np.tile(env.arrival_generator.mean_by_type, (len(end_rows), 1)))


def test_simulate_rejects_a_truncated_sample_path(env):
    agent = make_training_agent(env)
    agent.sample_path_proposal = build_proposal({"type": "fixed", "max_length": 3})
    with pytest.raises(ValueError, match="truncated"):
        agent.simulate(0)


# ----- fields of the PPO algorithm with the penalty -----

SIMULATED_KEYS = ("features", "feasible", "actions", "probabilities", "costs", "scenarios", "periods", "weights",
                  "end_of_period")


def scenario_data(agent, batch):
    """Per scenario: its initial state, its number of periods, and the arrivals after every period but the last."""
    first_waitlist_entry = 2 * agent.env.planning_horizon
    states = np.rint(batch["features"] * agent.feature_scale)
    data = []
    for scenario in np.unique(batch["scenarios"]):
        index = np.flatnonzero(batch["scenarios"] == scenario)
        ends = index[batch["end_of_period"][index]][:-1]
        arrivals = states[ends + 1, first_waitlist_entry:] - states[ends, first_waitlist_entry:]
        data.append((states[index[0]], batch["periods"][index[-1]], arrivals))
    return data


def arrival_distribution(env):
    """Every arrival vector of one period with its probability, by direct enumeration."""
    generator = env.arrival_generator
    distribution = {}
    for vector in itertools.product(range(generator.maximum_arrival + 1), repeat=env.num_types):
        total = sum(vector)
        if total > generator.maximum_arrival:
            continue
        orders = math.factorial(total) / np.prod([math.factorial(count) for count in vector])
        distribution[vector] = (generator.truncate_poisson_pmf[total] * orders
                                * np.prod(np.asarray(generator.type_probs, dtype=float) ** vector))
    return distribution


def smooth_value(features):
    """A value function approximation that is not affine in the features."""
    weights = np.linspace(-2.0, 3.0, features.shape[1])
    return 100.0 + 50.0 * np.tanh(features @ weights) + (features ** 2) @ weights[::-1]


def state_of(agent, batch, row):
    horizon = agent.env.planning_horizon
    return tuple(np.rint(part).astype(int) for part in np.split(batch["states"][row], [horizon, 2 * horizon]))


def next_period_features(agent, state, arrival):
    regular_bookings, overtime_bookings, waitlist = state
    return agent.state_features(
        (np.append(regular_bookings[1:], 0), np.append(overtime_bookings[1:], 0), waitlist + np.asarray(arrival)))


def test_fixed_scenarios_are_those_of_the_first_iteration_in_every_iteration(env):
    agent = make_training_agent(env, seed=7, scenarios="fixed")
    first, later = agent.simulate(0), agent.simulate(3)
    new_scenarios = make_training_agent(get_config_by_type("toy64").env, seed=7)
    first_new, later_new = new_scenarios.simulate(0), new_scenarios.simulate(3)
    # the first iteration is the same with fixed and with new scenarios
    for key in first_new:
        np.testing.assert_array_equal(first[key], first_new[key])

    def same(one, other):
        return all(np.array_equal(a, b) for scenario, other_scenario in zip(one, other)
                   for a, b in zip(scenario, other_scenario))

    # a later iteration has the initial states, the horizons and the arrivals of the first
    assert same(scenario_data(agent, later), scenario_data(agent, first))
    assert not same(scenario_data(agent, later_new), scenario_data(agent, first))
    # and other atomic actions
    assert len(first["actions"]) != len(later["actions"]) or not np.array_equal(first["actions"], later["actions"])
    # a training that continues at the later iteration has the same sample paths
    continued = make_training_agent(get_config_by_type("toy64").env, seed=7, scenarios="fixed").simulate(3)
    for key in later:
        np.testing.assert_array_equal(continued[key], later[key])


def test_exact_arrival_expectation_takes_every_arrival_vector_with_its_probability(env):
    agent = make_training_agent(env, arrival_expectation="exact")
    batch = agent.simulate(0)
    sampled = make_training_agent(get_config_by_type("toy64").env).simulate(0)
    assert "next_period_weights" not in sampled
    for key in SIMULATED_KEYS:
        np.testing.assert_array_equal(batch[key], sampled[key])

    expected = arrival_distribution(env)
    assert sum(expected.values()) == pytest.approx(1.0)
    H, I = env.planning_horizon, env.num_types
    end_rows = np.flatnonzero(batch["end_of_period"])
    assert batch["next_period_features"].shape == (len(end_rows), len(expected), 2 * H + I)
    assert batch["next_period_weights"].shape == (len(expected),)
    scale = agent.feature_scale[2 * H:]
    waitlist = batch["features"][end_rows][:, None, 2 * H:] * scale
    arrivals = np.rint(batch["next_period_features"][:, :, 2 * H:] * scale - waitlist).astype(int)
    # every row that ends a period has the same arrival vectors
    assert (arrivals == arrivals[0]).all()
    found = {tuple(vector): weight for vector, weight in zip(arrivals[0], batch["next_period_weights"])}
    assert set(found) == set(expected)
    for vector, probability in expected.items():
        assert found[vector] == pytest.approx(probability)


def test_simulate_with_the_penalty_records_the_state_and_the_probability_of_every_atomic_action(env):
    agent = make_training_agent(env, penalty=True)
    batch = agent.simulate(0)
    rows = len(batch["actions"])
    assert batch["states"].shape == batch["features"].shape
    np.testing.assert_array_equal(batch["states"] / agent.feature_scale, batch["features"])
    assert batch["action_probabilities"].shape == batch["feasible"].shape
    np.testing.assert_allclose(batch["action_probabilities"].sum(axis=1), 1.0)
    assert (batch["action_probabilities"][~batch["feasible"]] == 0).all()
    np.testing.assert_array_equal(
        batch["action_probabilities"][np.arange(rows), batch["actions"]], batch["probabilities"])
    # action_values computes the action values of all atomic actions, so the simulation prepares no arrival vectors
    # for Step 3
    assert "next_period_features" not in batch
    # the visited pairs are those of the simulation without the penalty
    plain = make_training_agent(get_config_by_type("toy64").env).simulate(0)
    for key in SIMULATED_KEYS:
        np.testing.assert_array_equal(batch[key], plain[key])


def test_action_values_of_every_atomic_action_with_the_exact_expectation(env):
    gamma = env.discount_factor
    agent = make_training_agent(env, penalty=True, arrival_expectation="exact")
    batch = agent.simulate(0)
    action_values = agent.action_values(batch, smooth_value, np.random.default_rng(0))
    assert action_values.shape == batch["feasible"].shape
    assert (action_values[~batch["feasible"]] == 0).all()
    distribution = arrival_distribution(env)
    booking_actions = 0
    for row in np.random.default_rng(1).choice(len(batch["actions"]), size=25, replace=False):
        state = state_of(agent, batch, row)
        for action in np.flatnonzero(batch["feasible"][row]):
            next_state, cost = agent.atomic_step(state, action)
            if action == end_of_period(env):
                # the atomic cost plus gamma times the expectation over the arrivals at the first state of the next period
                expected = cost + gamma * sum(
                    probability * smooth_value(next_period_features(agent, state, arrival)[None, :])[0]
                    for arrival, probability in distribution.items())
            else:
                # the atomic cost plus the value function approximation at the next state of the period
                expected = cost + smooth_value(agent.state_features(next_state)[None, :])[0]
                booking_actions += 1
            assert action_values[row, action] == pytest.approx(expected)
    assert booking_actions > 0


def test_action_value_of_the_end_of_period_action_with_sampled_arrival_vectors(env, monkeypatch):
    gamma = env.discount_factor
    agent = make_training_agent(env, penalty=True)
    batch = agent.simulate(0)
    rows = len(batch["actions"])
    drawn = []
    sample_arrivals = agent._sample_arrivals

    def recording_sample_arrivals(rng, size):
        drawn.append(sample_arrivals(rng, size))
        return drawn[-1]

    monkeypatch.setattr(agent, "_sample_arrivals", recording_sample_arrivals)
    action_values = agent.action_values(batch, smooth_value, np.random.default_rng(5))
    # arrival_sample_number = 4 arrival vectors for every visited state, in row order
    arrivals = np.concatenate(drawn).reshape(rows, 4, env.num_types)
    assert len(np.unique(arrivals.reshape(rows, -1), axis=0)) > 1
    for row in np.random.default_rng(1).choice(rows, size=25, replace=False):
        state = state_of(agent, batch, row)
        _, cost = agent.atomic_step(state, end_of_period(env))
        expected = cost + gamma * np.mean([
            smooth_value(next_period_features(agent, state, arrival)[None, :])[0] for arrival in arrivals[row]])
        assert action_values[row, end_of_period(env)] == pytest.approx(expected)
    # the arrival vectors come from the given generator
    np.testing.assert_array_equal(agent.action_values(batch, smooth_value, np.random.default_rng(5)), action_values)
    assert not np.array_equal(agent.action_values(batch, smooth_value, np.random.default_rng(6)), action_values)


def test_action_values_use_the_mean_arrival_for_an_affine_value_network(env):
    agent = make_training_agent(env, penalty=True, value_network_spec=AFFINE_VALUE_NETWORK_SPEC)
    batch = agent.simulate(0)
    affine_value = lambda features: 10.0 + features @ np.linspace(1.0, 2.0, features.shape[1])
    action_values = agent.action_values(batch, affine_value, np.random.default_rng(0))
    mean_arrival = np.asarray(env.arrival_generator.mean_by_type, dtype=float)
    for row in (0, len(batch["actions"]) - 1):
        state = state_of(agent, batch, row)
        _, cost = agent.atomic_step(state, end_of_period(env))
        expected = cost + env.discount_factor * affine_value(next_period_features(agent, state, mean_arrival)[None, :])[0]
        assert action_values[row, end_of_period(env)] == pytest.approx(expected)


def test_training_with_the_penalty_computes_the_action_values_once_per_iteration(env):
    agent = make_training_agent(env, sample_path_number=4, penalty=True, scenarios="fixed",
                                arrival_expectation="exact")
    before = agent.policy_coefficients()
    calls = []
    action_values = agent.action_values

    def counting_action_values(batch, value, rng):
        calls.append(len(batch["actions"]))
        return action_values(batch, value, rng)

    agent.action_values = counting_action_values
    state = agent.train(max_iterations=3, value_epochs=2, policy_epochs=2, batch_size=64, verbose=False)
    assert len(calls) == 3
    assert state["finished"] and [row["iteration"] for row in state["log"]] == [1, 2, 3]
    for row in state["log"]:
        assert np.isfinite([row["policy_cost"], row["penalized_policy_cost"], row["value_fit"],
                            row["clipped_objective"]]).all()
    after = state["policy_coefficients"]
    assert all(np.isfinite(after[name]).all() for name in after)
    assert any(not np.array_equal(before[name], after[name]) for name in before)


@pytest.mark.parametrize("field, value", [("scenarios", "resampled"), ("arrival_expectation", "mean")])
def test_agent_rejects_an_unknown_value_of_a_field(env, field, value):
    with pytest.raises(ValueError, match=field):
        make_agent(env, **{field: value})
