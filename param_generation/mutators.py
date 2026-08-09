"""Per-experiment mutators for ``env_args`` and ``agent_args``.

Each mutator is a module-level function (no nested closures) with one of two
signatures: ``mutate(env_args, val)`` or ``agent_mutate(agent_args, val)``. Keep
them small and self-contained -- everything they need should be derivable from
``env_args``/``agent_args`` and ``val`` -- so experiments stay easy to find,
test, and extend.
"""

import numpy as np

from utils import str2treatment_patterns, wait_time


def holding_cost_from_schedule(schedule):
    """Convert a per-type list of (start, end, cost) segments to the
    `holding_cost_by_day_by_type` matrix expected by env_args."""
    return np.array([wait_time(s) for s in schedule]).T.tolist()


def mutate_initial_state_congestion(env_args, occupancy_level):
    total_capacity = env_args['regular_capacity'] + env_args['overtime_capacity']
    treatment_pattern = str2treatment_patterns(env_args['patterns'])
    planning_horizon = env_args['booking_window_size'] + treatment_pattern.shape[0] - 1
    required_slots = total_capacity * occupancy_level
    regular_booking = min(required_slots, env_args['regular_capacity'])
    overtime_booking = required_slots - regular_booking
    regular_bookings = [regular_booking] * planning_horizon
    regular_bookings[-1] = 0
    overtime_bookings = [overtime_booking] * planning_horizon
    overtime_bookings[-1] = 0
    env_args['reset_params'] = {
        'init_state': (regular_bookings, overtime_bookings, env_args['arrival_rates']),
    }


def mutate_initial_state_congestion_05_const(env_args, val):
    """Always set initial-state congestion 0.5; the swept ``val`` (lambda_0)
    is ignored so the env matches ``base_toy_study`` exactly while the agent
    mutator sweeps the mixture proposal mass."""
    mutate_initial_state_congestion(env_args, 0.5)


def mutate_high_priority_proportion(env_args, proportion):
    total = sum(env_args['arrival_rates'])
    env_args['arrival_rates'] = [total * proportion, total * (1 - proportion)]


def mutate_low_priority_waiting_time_target(env_args, waiting_time_target):
    bw = env_args['booking_window_size']
    schedule = [
        [(0, 1, 0), (1, bw, 100)],
        [(0, waiting_time_target, 0), (waiting_time_target, bw, 10)],
    ]
    env_args['holding_cost_by_day_by_type'] = holding_cost_from_schedule(schedule)


def mutate_high_priority_waiting_time_penalty(env_args, waiting_penalty):
    bw = env_args['booking_window_size']
    schedule = [
        [(0, 1, 0), (1, bw, waiting_penalty)],
        [(0, 1, 0), (1, bw, 10)],
    ]
    env_args['holding_cost_by_day_by_type'] = holding_cost_from_schedule(schedule)


def mutate_total_arrival_rate(env_args, total_arrival_rate):
    arrival_rates = np.array(env_args['arrival_rates'])
    env_args['arrival_rates'] = (arrival_rates / sum(arrival_rates) * total_arrival_rate).tolist()


def mutate_type_1_treatment_pattern(env_args, pattern):
    env_args['patterns'][0] = pattern
    treatment_pattern = str2treatment_patterns(env_args['patterns'])
    booking_window_size = env_args['booking_window_size']
    regular_capacity = env_args['regular_capacity']
    arrival_rates = env_args['arrival_rates']
    planning_horizon = booking_window_size + treatment_pattern.shape[0] - 1
    regular_bookings = [regular_capacity] * planning_horizon
    regular_bookings[-1] = 0
    overtime_bookings = [0] * planning_horizon
    initial_state = (regular_bookings, overtime_bookings, arrival_rates.copy())
    env_args['reset_params'] = {'init_state': initial_state}


def mutate_overtime_cost(env_args, overtime_cost):
    env_args['overtime_cost_by_day'] = overtime_cost


def mutate_discount_factor(env_args, discount_factor):
    env_args['discount_factor'] = discount_factor


def mutate_is_proposal_098_const(agent_args, val):
    """Always set a geometric IS proposal with discount_factor_proposal=0.98.

    Used together with a 0.99 target discount-factor env to importance-sample
    longer paths from the cheaper 0.98 geometric distribution.
    """
    mutate_discount_factor_for_sample_path_length_proposal(agent_args, 0.98)


def mutate_discount_factor_for_sample_path_length_proposal(agent_args, discount_factor):
    discount_factor_str = str(discount_factor).replace('.', '_')
    agent_args.update({
        'policy_id': 'approx_penalized_hindsight_geometric_' + discount_factor_str,
        'agent_name': 'approx_penalized_hindsight',
    })
    agent_args['agent_args'].update({
        'sample_path_length_proposal': {
            'type': 'geometric',
            'discount_factor_proposal': discount_factor,
        }
    })


def mutate_fixed_length_for_sample_path_length_proposal(agent_args, max_length):
    agent_args.update({
        'policy_id': 'approx_penalized_hindsight_truncated_horizon_' + str(max_length),
        'agent_name': 'approx_penalized_hindsight',
    })
    agent_args['agent_args'].update({
        'sample_path_length_proposal': {
            'type': 'fixed',
            'max_length': max_length - 1,
        }
    })


# Mixture-geometric IS proposal sweep for the 0.99-target case study. The long
# component matches the target discount factor; the short component uses the
# cheaper proposal discount factor. ``target_discount_factor`` must equal the
# env discount factor (the agent asserts this when computing likelihood ratios).
MIXTURE_TARGET_DISCOUNT_FACTOR = 0.99
MIXTURE_PROPOSAL_DISCOUNT_FACTOR = 0.95


def mutate_mixture_target_discount_factor(env_args, val):
    """Pin the env discount factor to the mixture-proposal target (0.99).

    The swept ``val`` (``lambda_0``) is intentionally ignored: the target
    discount factor is held fixed so it matches the proposal's
    ``target_discount_factor``.
    """
    env_args['discount_factor'] = MIXTURE_TARGET_DISCOUNT_FACTOR

def mutate_mixture_target_discount_factor_overtime_50(env_args, val):
    """Pin the env discount factor to the mixture-proposal target (0.99).

    The swept ``val`` (``lambda_0``) is intentionally ignored: the target
    discount factor is held fixed so it matches the proposal's
    ``target_discount_factor``.
    """
    env_args['discount_factor'] = MIXTURE_TARGET_DISCOUNT_FACTOR
    env_args['overtime_cost_by_day'] = 50

def mutate_mixture_target_discount_factor_overtime_5(env_args, val):
    """Pin the env discount factor to the mixture-proposal target (0.99).

    The swept ``val`` (``lambda_0``) is intentionally ignored: the target
    discount factor is held fixed so it matches the proposal's
    ``target_discount_factor``.
    """
    env_args['discount_factor'] = MIXTURE_TARGET_DISCOUNT_FACTOR
    env_args['overtime_cost_by_day'] = 5

def mutate_sample_path_number_mixture_geometric_l01(agent_args, sample_path_number):
    """Sweep the Benders scenario count (``sample_path_number``) under the
    fixed mixture-geometric IS proposal (``lambda_0 = 0.1``) of the
    0.99-target case study, to measure how the scenario count drives the
    per-iteration confidence interval of the subproblem objectives.
    """
    mutate_mixture_geometric_proposal_lambda_0(agent_args, 0.1)
    agent_args['policy_id'] += '_scenario_' + str(sample_path_number)
    agent_args['agent_args']['sample_path_number'] = sample_path_number


def mutate_mixture_geometric_proposal_lambda_0(agent_args, lambda_0):
    """Set a mixture-geometric IS proposal, sweeping the long-component mass.

    The long component matches the target discount factor (0.99); the short
    component uses proposal discount factor 0.95. ``lambda_0`` is the mixture
    mass on the long (target) component and is the swept variable; the per-period
    importance weight is bounded in ``[1, 1 / lambda_0]``.
    """
    lambda_0_str = str(lambda_0).replace('.', '_')
    agent_args.update({
        'policy_id': 'approx_penalized_hindsight_mixture_geometric_lambda_' + lambda_0_str,
        'agent_name': 'approx_penalized_hindsight',
    })
    agent_args['agent_args'].update({
        'sample_path_length_proposal': {
            'type': 'mixture_geometric',
            'target_discount_factor': MIXTURE_TARGET_DISCOUNT_FACTOR,
            'discount_factor_proposal': MIXTURE_PROPOSAL_DISCOUNT_FACTOR,
            'lambda_0': lambda_0,
        }
    })
