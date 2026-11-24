import copy
import time

import numpy as np
import torch


def cost_to_go(costs, weights, scenarios):
    """Step 1. The cost that a scenario incurs from every visited state onward.

    The rows are the visited pairs, sorted by scenario and in path order. ``weights`` holds, for every
    row, the likelihood ratio factor of its period. The cost of a later period is multiplied by the
    ratio of its factor to the factor of the period of the row.
    """
    costs = np.asarray(costs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    scenarios = np.asarray(scenarios)
    weighted_costs = costs * weights
    from_row_onward = np.cumsum(weighted_costs[::-1])[::-1]
    last_rows = np.flatnonzero(np.append(scenarios[1:] != scenarios[:-1], True))
    after_scenario = np.append(from_row_onward, 0.0)[last_rows + 1]
    rows_per_scenario = np.diff(np.append(-1, last_rows))
    return (from_row_onward - np.repeat(after_scenario, rows_per_scenario)) / weights


def advantage_estimates(values, costs, end_of_period, next_period_values, discount_factor):
    """Step 3. The action value of the selected action minus the value of the visited state.

    ``values`` holds the value function approximation at the visited states. A row that does not end
    its period leads to the state of the next row. A row that ends its period leads to the next
    period, and ``next_period_values`` holds the expected value of its first state.
    """
    values = np.asarray(values, dtype=float)
    end_of_period = np.asarray(end_of_period, dtype=bool)
    same_period_rows = np.flatnonzero(~end_of_period)
    following_values = np.empty_like(values)
    following_values[same_period_rows] = values[same_period_rows + 1]
    following_values[end_of_period] = discount_factor * np.asarray(next_period_values, dtype=float)
    return np.asarray(costs, dtype=float) + following_values - values


def penalties(mean_action_values, costs, values, weights, scenarios):
    """Step 1 with the penalty. The penalty of every row: the mean of the action value over the atomic
    actions under the policy of Step 1, minus the atomic cost of the selected action, minus the value
    function approximation at the next row of the scenario. After the last row of a scenario nothing
    is subtracted. After a row that ends its period, the value at the next row is multiplied by the
    ratio of the likelihood ratio factors of the two periods.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    scenarios = np.asarray(scenarios)
    following_values = np.append(values[1:] * weights[1:] / weights[:-1], 0.0)
    following_values[np.append(scenarios[1:] != scenarios[:-1], True)] = 0.0
    return np.asarray(mean_action_values, dtype=float) - np.asarray(costs, dtype=float) - following_values


def clipped_objective(ratios, advantages, weights, clip_parameter):
    """Step 3. The sum over the visited pairs of weight * ratio * advantage estimate, where the ratio
    is replaced by max(ratio, 1 - clip_parameter) when the advantage estimate is nonnegative and by
    min(ratio, 1 + clip_parameter) when it is negative."""
    replaced_ratios = torch.where(
        advantages >= 0,
        torch.clamp(ratios, min=1.0 - clip_parameter),
        torch.clamp(ratios, max=1.0 + clip_parameter),
    )
    return torch.sum(weights * replaced_ratios * advantages)


def feasible_log_probabilities(scores, feasible):
    """Log of the softmax of the scores over the feasible entries. Infeasible entries get -inf."""
    return torch.log_softmax(scores.masked_fill(~feasible, float('-inf')), dim=-1)


def network_coefficients(network):
    return {name: value.detach().cpu().numpy().copy() for name, value in network.state_dict().items()}


def load_network_coefficients(network, coefficients):
    state = network.state_dict()
    missing = set(state) - set(coefficients)
    if missing:
        raise ValueError(f"coefficients are missing for {sorted(missing)}")
    network.load_state_dict({
        name: torch.as_tensor(np.asarray(coefficients[name]), dtype=value.dtype).reshape(value.shape)
        for name, value in state.items()})


def _single_linear_layer(network):
    layers = [module for module in network.modules() if not list(module.children())]
    if (len(layers) == 1 and isinstance(layers[0], torch.nn.Linear)
            and layers[0].out_features == 1 and layers[0].bias is not None):
        return layers[0]
    return None


def _tensor(array):
    return torch.as_tensor(np.ascontiguousarray(array))


class PPOSolver:
    """Proximal policy optimization for a cost-minimizing randomized policy.

    ``policy_network`` maps the features of a state to one score per action, and the softmax of the
    scores over the feasible actions gives the action probabilities. ``value_network`` maps the
    features of a state to the value function approximation. ``simulate(iteration)`` generates the
    sample paths of Step 1 with the current policy network and returns a dict of arrays with one row
    per visited pair, sorted by scenario and in path order:

        features, feasible, actions, probabilities, costs, scenarios, periods, weights, end_of_period

    and ``next_period_features`` with one entry per row that ends its period, in row order, of shape
    (rows that end a period, arrival vectors, features). The arrival vectors of a row have equal
    weights, or the weights ``next_period_weights`` if the batch has them.

    ``penalty=True`` runs the PPO algorithm with the penalty. The batch then needs
    ``action_probabilities``, the probabilities of all atomic actions at every row, in place of
    ``next_period_features``, and ``action_values(batch, value, rng)`` returns the action value of
    every atomic action at every row, generated by the value function approximation ``value``, as an
    array of shape (rows, actions) with zero where the action is not feasible.
    """

    VALUE_TARGET_OFFSET = 'target_offset'
    VALUE_TARGET_SCALE = 'target_scale'

    def __init__(self, policy_network, value_network, simulate, discount_factor,
                 max_iterations=200, clip_parameter=0.2, value_epochs=10, policy_epochs=10,
                 batch_size=4096, value_learning_rate=1e-3, policy_learning_rate=3e-4, seed=0,
                 checkpoint=None, checkpoint_every=10, state=None, verbose=True,
                 penalty=False, action_values=None):
        if penalty and action_values is None:
            raise ValueError("The PPO algorithm with the penalty needs action_values.")
        self.policy_network = policy_network
        self.value_network = value_network
        self.simulate = simulate
        self.penalty = bool(penalty)
        self.action_values = action_values
        self.discount_factor = float(discount_factor)
        self.max_iterations = int(max_iterations)
        self.clip_parameter = float(clip_parameter)
        self.value_epochs = int(value_epochs)
        self.policy_epochs = int(policy_epochs)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.checkpoint = checkpoint
        self.checkpoint_every = int(checkpoint_every)
        self.verbose = verbose

        self.policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=policy_learning_rate)
        # An affine value function approximation is fitted exactly, without gradient steps.
        self.affine_value_layer = _single_linear_layer(value_network)
        self.value_optimizer = None
        if self.affine_value_layer is None:
            self.value_optimizer = torch.optim.Adam(value_network.parameters(), lr=value_learning_rate)
        # The value network is fitted to (cost to go - offset) / scale. Both are set by the first fit.
        self.value_target_offset = None
        self.value_target_scale = None
        self.iteration = 0
        self.log = []
        if state is not None:
            self.load_state(state)

    def get_state(self):
        value_coefficients = network_coefficients(self.value_network)
        if self.value_target_offset is not None:
            value_coefficients[self.VALUE_TARGET_OFFSET] = np.array(self.value_target_offset)
            value_coefficients[self.VALUE_TARGET_SCALE] = np.array(self.value_target_scale)
        return {
            'policy_coefficients': network_coefficients(self.policy_network),
            'value_coefficients': value_coefficients,
            'optimizer_state': {
                'policy': copy.deepcopy(self.policy_optimizer.state_dict()),
                'value': None if self.value_optimizer is None else copy.deepcopy(self.value_optimizer.state_dict()),
            },
            'iteration': self.iteration,
            'finished': self.iteration >= self.max_iterations,
            'log': [dict(row) for row in self.log],
        }

    def load_state(self, state):
        load_network_coefficients(self.policy_network, state['policy_coefficients'])
        value_coefficients = dict(state['value_coefficients'])
        if self.VALUE_TARGET_OFFSET in value_coefficients:
            self.value_target_offset = float(value_coefficients.pop(self.VALUE_TARGET_OFFSET))
            self.value_target_scale = float(value_coefficients.pop(self.VALUE_TARGET_SCALE))
        load_network_coefficients(self.value_network, value_coefficients)
        optimizer_state = state.get('optimizer_state') or {}
        if optimizer_state.get('policy') is not None:
            self.policy_optimizer.load_state_dict(copy.deepcopy(optimizer_state['policy']))
        if self.value_optimizer is not None and optimizer_state.get('value') is not None:
            self.value_optimizer.load_state_dict(copy.deepcopy(optimizer_state['value']))
        self.iteration = int(state['iteration'])
        self.log = [dict(row) for row in state.get('log', [])]

    def value(self, features):
        """The value function approximation at the states with these features."""
        with torch.no_grad():
            output = self.value_network(_tensor(features)).squeeze(-1).numpy()
        if self.value_target_offset is None:
            return output
        return self.value_target_offset + self.value_target_scale * output

    def _batches(self, size, rng):
        order = rng.permutation(size)
        return [order[start:start + self.batch_size] for start in range(0, size, self.batch_size)]

    def fit_value_function(self, features, targets, rng):
        """Step 2. Returns the sum of squared errors after the fit."""
        features = np.asarray(features, dtype=float)
        targets = np.asarray(targets, dtype=float)
        if self.value_target_offset is None:
            self.value_target_offset = float(targets.mean())
            scale = float(targets.std())
            self.value_target_scale = scale if scale > 0 else 1.0
        standardized = (targets - self.value_target_offset) / self.value_target_scale

        if self.affine_value_layer is not None:
            design = np.hstack([features, np.ones((len(features), 1))])
            solution = np.linalg.lstsq(design, standardized, rcond=None)[0]
            with torch.no_grad():
                self.affine_value_layer.weight.copy_(_tensor(solution[:-1]).reshape(1, -1))
                self.affine_value_layer.bias.copy_(_tensor(solution[-1:]))
        else:
            feature_tensor, target_tensor = _tensor(features), _tensor(standardized)
            for _ in range(self.value_epochs):
                for rows in self._batches(len(targets), rng):
                    rows = _tensor(rows)
                    errors = self.value_network(feature_tensor[rows]).squeeze(-1) - target_tensor[rows]
                    loss = torch.mean(errors * errors)
                    self.value_optimizer.zero_grad()
                    loss.backward()
                    self.value_optimizer.step()
        return float(np.sum((self.value(features) - targets) ** 2))

    def estimate_advantages(self, batch):
        """The advantage estimates of Step 3, with the current value function approximation."""
        next_period_features = np.asarray(batch['next_period_features'], dtype=float)
        rows, samples = next_period_features.shape[:2]
        next_period_values = self.value(next_period_features.reshape(rows * samples, -1)).reshape(rows, samples)
        if batch.get('next_period_weights') is None:
            next_period_values = next_period_values.mean(axis=1)
        else:
            next_period_values = next_period_values @ np.asarray(batch['next_period_weights'], dtype=float)
        return advantage_estimates(
            values=self.value(np.asarray(batch['features'], dtype=float)),
            costs=batch['costs'],
            end_of_period=batch['end_of_period'],
            next_period_values=next_period_values,
            discount_factor=self.discount_factor,
        )

    def _objective(self, features, feasible, actions, old_log_probabilities, advantages, weights):
        log_probabilities = feasible_log_probabilities(self.policy_network(features), feasible)
        selected = log_probabilities.gather(1, actions.unsqueeze(1)).squeeze(1)
        ratios = torch.exp(selected - old_log_probabilities)
        return clipped_objective(ratios, advantages, weights, self.clip_parameter)

    def update_policy(self, batch, advantages, rng):
        """Step 3. Returns the clipped objective after the update."""
        data = (
            _tensor(np.asarray(batch['features'], dtype=float)),
            _tensor(np.asarray(batch['feasible'], dtype=bool)),
            _tensor(np.asarray(batch['actions'], dtype=np.int64)),
            torch.log(_tensor(np.asarray(batch['probabilities'], dtype=float))),
            _tensor(np.asarray(advantages, dtype=float)),
            _tensor(np.asarray(batch['weights'], dtype=float)),
        )
        return self._minimize(self._objective, data, rng)

    def _minimize(self, objective, data, rng):
        """Gradient steps on the objective of the rows of a batch, divided by their number. Returns the
        objective of all rows after the steps."""
        for _ in range(self.policy_epochs):
            for rows in self._batches(len(data[0]), rng):
                rows = _tensor(rows)
                loss = objective(*(tensor[rows] for tensor in data)) / len(rows)
                self.policy_optimizer.zero_grad()
                loss.backward()
                self.policy_optimizer.step()
        with torch.no_grad():
            return float(objective(*data))

    def penalized_policy_path_costs(self, batch, rng):
        """Step 1 with the penalty, with the current value function approximation. Returns

            values              the value function approximation at the visited states
            action_values       the action values of all atomic actions at the visited states
            mean_action_values  their mean under the policy of Step 1
            penalized_costs     the penalized policy-path costs

        The penalized policy-path cost of a row is the sum of the atomic cost plus the penalty over the
        row and the later rows of its scenario, with the factors of the policy-path cost.
        """
        values = self.value(np.asarray(batch['features'], dtype=float))
        action_values = np.asarray(self.action_values(batch, self.value, rng), dtype=float)
        mean_action_values = np.sum(np.asarray(batch['action_probabilities'], dtype=float) * action_values, axis=1)
        costs = np.asarray(batch['costs'], dtype=float)
        step_penalties = penalties(mean_action_values, costs, values, batch['weights'], batch['scenarios'])
        return {
            'values': values,
            'action_values': action_values,
            'mean_action_values': mean_action_values,
            'penalized_costs': cost_to_go(costs + step_penalties, batch['weights'], batch['scenarios']),
        }

    def _objective_with_penalty(self, features, feasible, actions, old_probabilities, first_parts,
                                second_parts, weights):
        log_probabilities = feasible_log_probabilities(self.policy_network(features), feasible)
        # An atomic action that is not feasible has probability 0, and its ratio is set to 1. The two
        # bounds keep the ratio finite when the probability of a feasible atomic action is 0 in
        # floating point.
        old_log_probabilities = torch.log(torch.clamp(old_probabilities, min=1e-300))
        changes = torch.where(feasible, log_probabilities - old_log_probabilities,
                              torch.zeros_like(old_log_probabilities))
        ratios = torch.exp(torch.clamp(changes, max=50.0))
        selected_ratios = ratios.gather(1, actions.unsqueeze(1)).squeeze(1)
        # First part: every atomic action, with its probability under the policy of Step 1 and its own
        # ratio. Second part: the selected action alone.
        return (clipped_objective(ratios, first_parts, weights.unsqueeze(1) * old_probabilities, self.clip_parameter)
                + clipped_objective(selected_ratios, second_parts, weights, self.clip_parameter))

    def update_policy_with_penalty(self, batch, step_1, rng):
        """Step 3 with the penalty, from the values of Step 1. Returns the objective after the update.

        The advantage estimate has two parts. The first is known for every atomic action: its action
        value minus the mean action value under the policy of Step 1. The second is known for the
        selected action: the penalized policy-path cost minus the value function approximation.
        """
        feasible = np.asarray(batch['feasible'], dtype=bool)
        first_parts = np.asarray(step_1['action_values'], dtype=float) - step_1['mean_action_values'][:, None]
        data = (
            _tensor(np.asarray(batch['features'], dtype=float)),
            _tensor(feasible),
            _tensor(np.asarray(batch['actions'], dtype=np.int64)),
            _tensor(np.where(feasible, np.asarray(batch['action_probabilities'], dtype=float), 0.0)),
            _tensor(np.where(feasible, first_parts, 0.0)),
            _tensor(np.asarray(step_1['penalized_costs'] - step_1['values'], dtype=float)),
            _tensor(np.asarray(batch['weights'], dtype=float)),
        )
        return self._minimize(self._objective_with_penalty, data, rng)

    def solve(self):
        while self.iteration < self.max_iterations:
            start = time.time()
            batch = self.simulate(self.iteration)
            policy_path_costs = cost_to_go(batch['costs'], batch['weights'], batch['scenarios'])
            targets, step_1 = policy_path_costs, None
            step_1_rng = np.random.default_rng([self.seed, self.iteration, 11])
            if self.penalty and self.value_target_offset is not None:
                # Step 1 with the value function approximation that Step 2 of the previous iteration fitted.
                step_1 = self.penalized_policy_path_costs(batch, step_1_rng)
                targets = step_1['penalized_costs']
            value_fit = self.fit_value_function(
                batch['features'], targets, np.random.default_rng([self.seed, self.iteration, 3]))
            policy_rng = np.random.default_rng([self.seed, self.iteration, 5])
            if self.penalty:
                if step_1 is None:
                    # First iteration: no value function approximation of a previous iteration exists.
                    # Steps 1 and 2 are those without the penalty, and Step 3 computes the penalties and
                    # the penalized policy-path costs with the value function approximation that Step 2
                    # has just fitted.
                    step_1 = self.penalized_policy_path_costs(batch, step_1_rng)
                objective = self.update_policy_with_penalty(batch, step_1, policy_rng)
            else:
                objective = self.update_policy(batch, self.estimate_advantages(batch), policy_rng)

            scenarios = np.asarray(batch['scenarios'])
            first_rows = np.flatnonzero(np.append(True, scenarios[1:] != scenarios[:-1]))

            def mean_and_standard_error(costs):
                if len(costs) < 2:
                    return float(costs.mean()), 0.0
                return float(costs.mean()), float(costs.std(ddof=1) / np.sqrt(len(costs)))

            self.iteration += 1
            row = {'iteration': self.iteration, 'seconds': time.time() - start}
            row['policy_cost'], row['policy_cost_standard_error'] = mean_and_standard_error(
                policy_path_costs[first_rows])
            if self.penalty:
                # The penalized policy-path cost at the first state of every scenario.
                row['penalized_policy_cost'], row['penalized_policy_cost_standard_error'] = (
                    mean_and_standard_error(step_1['penalized_costs'][first_rows]))
            row.update({
                'value_fit': value_fit,
                'clipped_objective': objective,
                'visited_pairs': int(len(targets)),
            })
            self.log.append(row)
            if self.verbose:
                penalized = ''
                if self.penalty:
                    penalized = (f"with penalty {row['penalized_policy_cost']:.2f} +- "
                                 f"{row['penalized_policy_cost_standard_error']:.2f}, ")
                print(
                    f"PPO iteration {row['iteration']}/{self.max_iterations}: "
                    f"policy cost {row['policy_cost']:.2f} +- {row['policy_cost_standard_error']:.2f}, {penalized}"
                    f"value fit {row['value_fit']:.4g}, clipped objective {row['clipped_objective']:.4g}, "
                    f"{row['visited_pairs']} visited pairs, {row['seconds']:.1f}s", flush=True)
            if self.checkpoint is not None and (
                    self.iteration % self.checkpoint_every == 0 or self.iteration >= self.max_iterations):
                self.checkpoint(self.get_state())
        return self.get_state()
