import itertools
import math

import numpy as np
import torch

from decision_maker import InfiniteRTAgent
from importance_sampling import Terminal
from metaheuristic_algorithm.ppo_solver import (
    PPOSolver,
    feasible_log_probabilities,
    load_network_coefficients,
    network_coefficients,
)

_ACTIVATIONS = {
    'tanh': torch.nn.Tanh,
    'relu': torch.nn.ReLU,
}

# Streams of random numbers that one seed gives. Each is used as default_rng([seed, iteration, stream]).
_SIMULATE_STREAM = 1
_ACTION_SELECTION_STREAM = 2
_POLICY_NETWORK_STREAM = 7
_VALUE_NETWORK_STREAM = 9

# The largest number of arrival vectors of one period that the exact expectation sums over, and the
# largest number of states that one evaluation of the value function approximation takes.
_MAXIMUM_ARRIVAL_VECTORS = 100_000
_VALUE_CHUNK = 200_000


def _build_network(input_size, output_size, spec, rng=None):
    spec = dict(spec or {})
    hidden_sizes = [int(size) for size in spec.get('hidden_sizes', [])]
    activation = spec.get('activation', 'tanh')
    if activation not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {activation!r}. Available: {sorted(_ACTIVATIONS)}")
    sizes = [input_size] + hidden_sizes
    layers = []
    for inputs, outputs in zip(sizes[:-1], sizes[1:]):
        layers.append(torch.nn.Linear(inputs, outputs))
        layers.append(_ACTIVATIONS[activation]())
    layers.append(torch.nn.Linear(sizes[-1], output_size))
    network = torch.nn.Sequential(*layers).double()
    if rng is not None:
        # The distribution of the default initialization of torch.nn.Linear, drawn from rng.
        with torch.no_grad():
            for layer in network:
                if isinstance(layer, torch.nn.Linear):
                    bound = 1.0 / math.sqrt(layer.in_features)
                    layer.weight.copy_(torch.as_tensor(rng.uniform(-bound, bound, size=tuple(layer.weight.shape))))
                    layer.bias.copy_(torch.as_tensor(rng.uniform(-bound, bound, size=tuple(layer.bias.shape))))
    return network


def state_feature_size(env):
    return 2 * env.planning_horizon + env.num_types


def build_policy_network(env, spec, rng=None):
    """Input (u, v, w). Output N * I + 1 scores: booking action (n, i) at index n * I + i, and the
    end-of-period action at the last index."""
    return _build_network(state_feature_size(env), env.booking_window_size * env.num_types + 1, spec, rng)


def build_value_network(env, spec, rng=None):
    """Input (u, v, w). Output the value function approximation at the state. With hidden_sizes [] it
    is an affine function of (u, v, w) with 1 + 2H + I coefficients."""
    return _build_network(state_feature_size(env), 1, spec, rng)


class AtomicActionAgent(InfiniteRTAgent):
    """Randomized step-independent atomic policy. Its policy network gives the probabilities of the
    atomic actions, and solve() forms the action of a period from the selected atomic actions."""

    def __init__(self, env, discount_factor, policy_network_spec, value_network_spec=None,
                 policy_coefficients=None, action_selection='greedy', seed=0,
                 sample_path_number=None, sample_path_proposal=None, initial_state_sampler=None,
                 arrival_sample_number=16, V=None, Q=None, grb_env=None,
                 scenarios='new', penalty=False, arrival_expectation='sampled'):
        """scenarios: 'new' draws the scenarios of every iteration anew, and 'fixed' keeps those of the
        first iteration. penalty: the PPO algorithm with the penalty. arrival_expectation: the
        expectation over the arrival vector of the next period is the average over
        arrival_sample_number sampled vectors ('sampled') or the sum over all arrival vectors
        ('exact')."""
        super().__init__(env, discount_factor, V, Q, grb_env=grb_env)
        if action_selection not in ('greedy', 'sample'):
            raise ValueError(f"action_selection must be 'greedy' or 'sample'; got {action_selection!r}")
        if scenarios not in ('new', 'fixed'):
            raise ValueError(f"scenarios must be 'new' or 'fixed'; got {scenarios!r}")
        if arrival_expectation not in ('sampled', 'exact'):
            raise ValueError(f"arrival_expectation must be 'sampled' or 'exact'; got {arrival_expectation!r}")
        self.scenarios = scenarios
        self.penalty = bool(penalty)
        self.arrival_expectation = arrival_expectation
        self._fixed_scenarios = None
        self._arrival_distribution = None
        self.action_selection = action_selection
        self.seed = int(seed)
        self.sample_path_number = None if sample_path_number is None else int(sample_path_number)
        self.sample_path_proposal = sample_path_proposal
        self.initial_state_sampler = initial_state_sampler
        self.arrival_sample_number = int(arrival_sample_number)

        booking_window_size, num_types = env.booking_window_size, env.num_types
        self.end_of_period_action = booking_window_size * num_types
        # Column n * I + i holds the slots that booking action (n, i) adds to every day.
        self.booking_slots = np.asarray(env.booking_mapping_matrix, dtype=float)
        self.booking_costs = np.asarray(env.waiting_cost_coeffs, dtype=float).reshape(-1)
        self.booking_types = np.tile(np.arange(num_types), booking_window_size)
        feature_scale = np.concatenate([
            np.full(env.planning_horizon, env.regular_capacity, dtype=float),
            np.full(env.planning_horizon, env.overtime_capacity, dtype=float),
            3.0 * np.asarray(env.arrival_generator.mean_by_type, dtype=float),
        ])
        feature_scale[feature_scale <= 0] = 1.0
        self.feature_scale = feature_scale

        self.policy_network_spec = dict(policy_network_spec)
        self.policy_network = build_policy_network(
            env, self.policy_network_spec, np.random.default_rng([self.seed, 0, _POLICY_NETWORK_STREAM]))
        if policy_coefficients is not None:
            load_network_coefficients(self.policy_network, policy_coefficients)
        self.value_network_spec = None if value_network_spec is None else dict(value_network_spec)
        self.value_network = None
        if self.value_network_spec is not None:
            self.value_network = build_value_network(
                env, self.value_network_spec, np.random.default_rng([self.seed, 0, _VALUE_NETWORK_STREAM]))
        self.action_rng = np.random.default_rng([self.seed, 0, _ACTION_SELECTION_STREAM])

    def policy_coefficients(self):
        return network_coefficients(self.policy_network)

    # ----- atomic problem, for arrays with one row per state -----

    def _feasible(self, regular_bookings, overtime_bookings, waitlist):
        overtime = np.maximum(
            regular_bookings[:, :, None] + self.booking_slots[None, :, :] - self.env.regular_capacity, 0.0)
        within_capacity = np.all(
            overtime_bookings[:, :, None] + overtime <= self.env.overtime_capacity + 1e-9, axis=1)
        outstanding = waitlist[:, self.booking_types] >= 1 - 1e-9
        end_of_period = np.ones((len(waitlist), 1), dtype=bool)
        return np.concatenate([within_capacity & outstanding, end_of_period], axis=1)

    def _book(self, regular_bookings, overtime_bookings, waitlist, actions):
        added_slots = self.booking_slots[:, actions].T
        overtime = np.maximum(regular_bookings + added_slots - self.env.regular_capacity, 0.0)
        next_waitlist = waitlist.copy()
        next_waitlist[np.arange(len(actions)), self.booking_types[actions]] -= 1
        costs = self.booking_costs[actions] + overtime @ self.env.overtime_cost_coeffs
        return regular_bookings + added_slots - overtime, overtime_bookings + overtime, next_waitlist, costs

    def _features(self, regular_bookings, overtime_bookings, waitlist):
        return np.concatenate([regular_bookings, overtime_bookings, waitlist], axis=1) / self.feature_scale

    def _probabilities(self, features, feasible):
        with torch.no_grad():
            scores = self.policy_network(torch.as_tensor(features))
            return torch.exp(feasible_log_probabilities(scores, torch.as_tensor(feasible))).numpy()

    @staticmethod
    def _as_rows(state):
        return tuple(np.asarray(component, dtype=float)[None, :] for component in state)

    # ----- atomic problem, for one state -----

    def feasible_mask(self, state):
        """Vector of length N * I + 1. True where the atomic action is in the atomic action space."""
        return self._feasible(*self._as_rows(state))[0]

    def atomic_step(self, state, action):
        """The next state and the atomic cost of a booking action, or (None, atomic cost) for the
        end-of-period action."""
        action = int(action)
        if action == self.end_of_period_action:
            return None, float(self.env.postponing_cost_coeffs @ np.asarray(state[2], dtype=float))
        if not self.feasible_mask(state)[action]:
            raise ValueError(f"atomic action {action} is not feasible in state {state}")
        regular_bookings, overtime_bookings, waitlist, costs = self._book(
            *self._as_rows(state), np.array([action]))
        return (regular_bookings[0], overtime_bookings[0], waitlist[0]), float(costs[0])

    def state_features(self, state):
        """Network input: u divided by C_r, v divided by C_o, and each w_i divided by 3 times the mean
        of the type-i arrivals."""
        return self._features(*self._as_rows(state))[0]

    def action_probabilities(self, state):
        rows = self._as_rows(state)
        return self._probabilities(self._features(*rows), self._feasible(*rows))[0]

    # ----- action of a period -----

    def solve(self, state, t):
        num_types = self.env.num_types
        current = tuple(np.array(component, dtype=float) for component in state)
        advance_scheduling_decision = np.zeros((self.env.booking_window_size, num_types), dtype=int)
        obj_value = 0.0
        while True:
            probabilities = self.action_probabilities(current)
            if self.action_selection == 'greedy':
                action = int(np.argmax(probabilities))
            else:
                action = int(self.action_rng.choice(len(probabilities), p=probabilities))
            next_state, cost = self.atomic_step(current, action)
            obj_value += cost
            if action == self.end_of_period_action:
                break
            advance_scheduling_decision[divmod(action, num_types)] += 1
            current = next_state
        overtime_decision = self.regular_first_overtime(state, advance_scheduling_decision)
        return obj_value, (advance_scheduling_decision, overtime_decision), {}

    # ----- Step 1 of the PPO iteration -----

    def _sample_paths(self, rng):
        generator = self.env.arrival_generator
        generator_rng = generator.rng
        generator.rng = rng
        try:
            paths = self.sample_path_proposal.sample_paths(
                arrival_generator=generator,
                size=self.sample_path_number,
                target_discount_factor=self.discount_factor,
            )
        finally:
            generator.rng = generator_rng
        if any(path.terminal == Terminal.TRUNCATED for path in paths):
            raise ValueError(
                "The sample path proposal gives a truncated sample path. The cost from a visited state "
                "onward has no term for the periods after the truncation, so PPO needs a proposal "
                "without truncation.")
        if any(path.length < 1 for path in paths):
            raise ValueError("Every sample path needs at least one period.")
        return paths

    def _sample_arrivals(self, rng, size):
        generator = self.env.arrival_generator
        totals = rng.choice(generator.maximum_arrival + 1, size=size, p=generator.truncate_poisson_pmf)
        return rng.multinomial(totals, generator.type_probs)

    def _arrival_vectors(self, rng, rows):
        """The arrival vectors that the expectation over the arrival vector of the next period uses for
        `rows` states. Returns (vectors, probabilities). vectors has the shape (rows, number, I), or
        (1, number, I) when every state takes the same vectors. probabilities is None when the
        expectation is the average over the vectors."""
        generator = self.env.arrival_generator
        if self.value_network_spec is not None and not self.value_network_spec.get('hidden_sizes'):
            # The expectation of an affine function is its value at the mean arrival.
            return np.asarray(generator.mean_by_type, dtype=float)[None, None, :], None
        if self.arrival_expectation == 'sampled':
            number = self.arrival_sample_number
            return self._sample_arrivals(rng, rows * number).reshape(rows, number, self.env.num_types), None
        return self._all_arrival_vectors()

    def _arrival_vector_number(self):
        """The number of arrival vectors that _arrival_vectors returns for one state."""
        if self.value_network_spec is not None and not self.value_network_spec.get('hidden_sizes'):
            return 1
        if self.arrival_expectation == 'sampled':
            return self.arrival_sample_number
        return self._all_arrival_vectors()[0].shape[1]

    def _all_arrival_vectors(self):
        """Every arrival vector of one period, with the shape (1, number, I), and its probability."""
        generator = self.env.arrival_generator
        if self._arrival_distribution is None:
            num_types, maximum = self.env.num_types, generator.maximum_arrival
            if math.comb(maximum + num_types, num_types) > _MAXIMUM_ARRIVAL_VECTORS:
                raise ValueError(
                    f"arrival_expectation='exact' sums over {math.comb(maximum + num_types, num_types)} arrival "
                    f"vectors, more than {_MAXIMUM_ARRIVAL_VECTORS}. Use arrival_expectation='sampled'.")
            type_probabilities = np.asarray(generator.type_probs, dtype=float)
            vectors, probabilities = [], []
            for vector in itertools.product(range(maximum + 1), repeat=num_types):
                total = sum(vector)
                if total > maximum:
                    continue
                # The total is drawn from the truncated Poisson distribution and split by a multinomial.
                orders = math.factorial(total) / math.prod(math.factorial(count) for count in vector)
                vectors.append(vector)
                probabilities.append(
                    generator.truncate_poisson_pmf[total] * orders * np.prod(type_probabilities ** vector))
            self._arrival_distribution = (np.array(vectors, dtype=float)[None, :, :], np.array(probabilities))
        return self._arrival_distribution

    def _draw_scenarios(self, iteration, rng):
        """Scenario omega has tau = lengths[omega] periods. Row t - 1 of its arrivals is observed after
        period t, and factors[omega, t - 1] is the likelihood ratio factor of period t. states holds
        (u, v, w) of the initial states."""
        scenario_number = self.sample_path_number
        paths = self._sample_paths(rng)
        lengths = np.array([path.length for path in paths])
        arrivals = np.zeros((scenario_number, lengths.max(), self.env.num_types))
        factors = np.ones((scenario_number, lengths.max()))
        for scenario, path in enumerate(paths):
            arrivals[scenario, :path.length] = path.arrivals
            factors[scenario, :path.length] = path.likelihood_ratios

        initial_states = self.initial_state_sampler(int(iteration), scenario_number)
        if len(initial_states) != scenario_number:
            raise ValueError(
                f"initial_state_sampler returned {len(initial_states)} states for {scenario_number} scenarios")
        states = tuple(np.array([state[part] for state in initial_states], dtype=float) for part in range(3))
        return lengths, arrivals, factors, states

    def _scenarios(self, iteration, rng):
        """The scenarios of one iteration. With scenarios='fixed', every iteration after the first takes
        the scenarios of the first, also in a training that is continued."""
        if self.scenarios == 'fixed' and iteration > 0:
            if self._fixed_scenarios is None:
                self._fixed_scenarios = self._draw_scenarios(
                    0, np.random.default_rng([self.seed, 0, _SIMULATE_STREAM]))
            return self._fixed_scenarios
        scenarios = self._draw_scenarios(iteration, rng)
        if self.scenarios == 'fixed':
            self._fixed_scenarios = scenarios
        return scenarios

    def simulate(self, iteration):
        """Runs the scenarios of one iteration with the current policy network. Returns one row per
        visited pair, sorted by scenario and in path order, as PPOSolver expects."""
        if self.sample_path_number is None or self.sample_path_proposal is None or self.initial_state_sampler is None:
            raise ValueError(
                "simulate needs sample_path_number, sample_path_proposal and initial_state_sampler.")
        env = self.env
        horizon, num_types = env.planning_horizon, env.num_types
        scenario_number = self.sample_path_number
        iteration = int(iteration)
        rng = np.random.default_rng([self.seed, iteration, _SIMULATE_STREAM])

        lengths, arrivals, factors, initial_states = self._scenarios(iteration, rng)
        regular_bookings, overtime_bookings, waitlist = (component.copy() for component in initial_states)

        period = np.ones(scenario_number, dtype=int)
        active = np.arange(scenario_number)
        columns = {key: [] for key in (
            'features', 'feasible', 'actions', 'probabilities', 'costs', 'scenarios', 'periods', 'weights',
            'next_period_state')}
        if self.penalty:
            # The PPO algorithm with the penalty needs the state and the probability of every atomic action.
            columns['states'], columns['action_probabilities'] = [], []
        while len(active):
            u, v, w = regular_bookings[active], overtime_bookings[active], waitlist[active]
            features = self._features(u, v, w)
            feasible = self._feasible(u, v, w)
            probabilities = self._probabilities(features, feasible)
            # Draw one atomic action per scenario: the first action whose cumulative probability
            # exceeds the uniform draw. An action of probability zero is never drawn.
            cumulative = np.cumsum(probabilities, axis=1)
            draws = rng.random(len(active)) * cumulative[:, -1]
            actions = np.minimum((cumulative <= draws[:, None]).sum(axis=1), self.end_of_period_action)
            rows = np.arange(len(active))
            ends = actions == self.end_of_period_action

            costs = np.empty(len(active))
            costs[ends] = w[ends] @ env.postponing_cost_coeffs
            booking_rows = rows[~ends]
            if len(booking_rows):
                next_u, next_v, next_w, booking_costs = self._book(
                    u[booking_rows], v[booking_rows], w[booking_rows], actions[booking_rows])
                costs[booking_rows] = booking_costs
                booked = active[booking_rows]
                regular_bookings[booked], overtime_bookings[booked], waitlist[booked] = next_u, next_v, next_w

            # After the end-of-period action the bookings move one day forward. The arrivals are added
            # below for the scenario itself and in next_period_features for the action value function.
            next_period_state = np.zeros((len(active), 2 * horizon + num_types))
            next_period_state[ends, :horizon - 1] = u[ends, 1:]
            next_period_state[ends, horizon:2 * horizon - 1] = v[ends, 1:]
            next_period_state[ends, 2 * horizon:] = w[ends]

            columns['features'].append(features)
            columns['feasible'].append(feasible)
            columns['actions'].append(actions)
            columns['probabilities'].append(probabilities[rows, actions])
            columns['costs'].append(costs)
            columns['scenarios'].append(active)
            columns['periods'].append(period[active])
            columns['weights'].append(factors[active, period[active] - 1])
            columns['next_period_state'].append(next_period_state)
            if self.penalty:
                columns['states'].append(np.concatenate([u, v, w], axis=1))
                columns['action_probabilities'].append(probabilities)

            continuing = ends & (period[active] < lengths[active])
            scenarios = active[continuing]
            regular_bookings[scenarios] = next_period_state[continuing, :horizon]
            overtime_bookings[scenarios] = next_period_state[continuing, horizon:2 * horizon]
            waitlist[scenarios] = w[continuing] + arrivals[scenarios, period[scenarios] - 1]
            period[scenarios] += 1
            active = active[~ends | continuing]

        batch = {key: np.concatenate(values) for key, values in columns.items()}
        order = np.argsort(batch['scenarios'], kind='stable')
        batch = {key: values[order] for key, values in batch.items()}
        batch['end_of_period'] = batch['actions'] == self.end_of_period_action

        next_period_state = batch.pop('next_period_state')[batch['end_of_period']]
        if self.penalty:
            # With the penalty, action_values computes the action values of all atomic actions.
            return batch
        arrival_vectors, probabilities = self._arrival_vectors(rng, len(next_period_state))
        next_period_states = np.repeat(next_period_state[:, None, :], arrival_vectors.shape[1], axis=1)
        next_period_states[:, :, 2 * horizon:] += arrival_vectors
        batch['next_period_features'] = next_period_states / self.feature_scale
        if probabilities is not None:
            batch['next_period_weights'] = probabilities
        return batch

    # ----- Step 1 of the PPO iteration with the penalty -----

    def action_values(self, batch, value, rng):
        """The action value of every atomic action at every visited state, generated by `value`, a
        function of the features of states: the atomic cost plus `value` at the next state for a
        booking action, and the atomic cost plus the discount factor times the expectation of `value`
        at the first state of the next period for the end-of-period action. Zero where the atomic
        action is not feasible. rng draws the arrival vectors of arrival_expectation='sampled'."""
        env = self.env
        horizon = env.planning_horizon
        states = np.asarray(batch['states'], dtype=float)
        feasible = np.asarray(batch['feasible'], dtype=bool)
        action_values = np.zeros(feasible.shape)

        rows, actions = np.nonzero(feasible[:, :self.end_of_period_action])
        for start in range(0, len(rows), _VALUE_CHUNK):
            chunk_rows, chunk_actions = rows[start:start + _VALUE_CHUNK], actions[start:start + _VALUE_CHUNK]
            u, v, w = np.split(states[chunk_rows], [horizon, 2 * horizon], axis=1)
            next_u, next_v, next_w, costs = self._book(u, v, w, chunk_actions)
            action_values[chunk_rows, chunk_actions] = costs + value(self._features(next_u, next_v, next_w))

        # After the end-of-period action the bookings move one day forward, and the arrival vector is
        # added to the waitlist.
        next_period_state = np.zeros_like(states)
        next_period_state[:, :horizon - 1] = states[:, 1:horizon]
        next_period_state[:, horizon:2 * horizon - 1] = states[:, horizon + 1:2 * horizon]
        next_period_state[:, 2 * horizon:] = states[:, 2 * horizon:]
        vector_number = self._arrival_vector_number()
        chunk = max(1, _VALUE_CHUNK // vector_number)
        expected_values = np.empty(len(states))
        for start in range(0, len(states), chunk):
            part = next_period_state[start:start + chunk]
            arrival_vectors, probabilities = self._arrival_vectors(rng, len(part))
            next_states = np.repeat(part[:, None, :], vector_number, axis=1)
            next_states[:, :, 2 * horizon:] += arrival_vectors
            values = np.asarray(value((next_states / self.feature_scale).reshape(-1, states.shape[1])))
            values = values.reshape(len(part), vector_number)
            expected_values[start:start + chunk] = (
                values.mean(axis=1) if probabilities is None else values @ probabilities)
        action_values[:, self.end_of_period_action] = (
            states[:, 2 * horizon:] @ env.postponing_cost_coeffs + self.discount_factor * expected_values)
        return action_values

    # ----- training -----

    def train(self, max_iterations=200, clip_parameter=0.2, value_epochs=10, policy_epochs=10,
              batch_size=4096, value_learning_rate=1e-3, policy_learning_rate=3e-4,
              checkpoint=None, checkpoint_every=10, state=None, verbose=True):
        """Trains the policy network by PPO. Returns the state of the solver: the coefficients of both
        networks, the state of the optimizers, the number of finished iterations, and the log."""
        if self.value_network is None:
            raise ValueError("Training needs value_network_spec.")
        solver = PPOSolver(
            policy_network=self.policy_network,
            value_network=self.value_network,
            simulate=self.simulate,
            discount_factor=self.discount_factor,
            max_iterations=max_iterations,
            clip_parameter=clip_parameter,
            value_epochs=value_epochs,
            policy_epochs=policy_epochs,
            batch_size=batch_size,
            value_learning_rate=value_learning_rate,
            policy_learning_rate=policy_learning_rate,
            seed=self.seed,
            checkpoint=checkpoint,
            checkpoint_every=checkpoint_every,
            state=state,
            verbose=verbose,
            penalty=self.penalty,
            action_values=self.action_values,
        )
        return solver.solve()
