"""Per-experiment mutators for ``env_args`` and ``agent_args``.

Each mutator is a module-level callable (function or ``functools.partial`` of
one -- no nested closures) with one of two signatures:
``mutate(env_args, val)`` or ``agent_mutate(agent_args, val)``. Constant
mutators (the swept ``val`` is ignored) are ``partial(set_env_fields, {...})``.
"""

from functools import partial

from utils import str2treatment_patterns


def set_env_fields(updates, env_args, val):
    """Generic const env mutator: apply ``updates``; the swept ``val`` is ignored."""
    for key, value in updates.items():
        env_args[key] = value


def mutate_initial_state_congestion(env_args, occupancy_level):
    total_capacity = env_args['regular_capacity'] + env_args['overtime_capacity']
    treatment_pattern = str2treatment_patterns(env_args['patterns'])
    planning_horizon = env_args['booking_window_size'] + treatment_pattern.shape[0] - 1
    # Bookings and waitlist counts are integers: round the slot target and the
    # per-type arrival rates (a no-op for the toy cases, whose capacities and
    # rates are already integral; the EJOR case has 135 * 0.5 = 67.5 slots and
    # fractional arrival rates).
    required_slots = int(round(total_capacity * occupancy_level))
    regular_booking = min(required_slots, env_args['regular_capacity'])
    overtime_booking = required_slots - regular_booking
    regular_bookings = [regular_booking] * planning_horizon
    regular_bookings[-1] = 0
    overtime_bookings = [overtime_booking] * planning_horizon
    overtime_bookings[-1] = 0
    waitlist = [int(round(rate)) for rate in env_args['arrival_rates']]
    env_args['reset_params'] = {
        'init_state': (regular_bookings, overtime_bookings, waitlist),
    }


def mutate_initial_state_congestion_05_const(env_args, val):
    """Always set initial-state congestion 0.5; the swept ``val`` (lambda_0)
    is ignored so the env matches ``base_toy_study`` exactly while the agent
    mutator sweeps the mixture proposal mass."""
    mutate_initial_state_congestion(env_args, 0.5)


# Mixture-geometric IS proposal for the 0.99-target case study. The long
# component matches the target discount factor; the short component uses the
# cheaper proposal discount factor. ``target_discount_factor`` must equal the
# env discount factor (the agent asserts this when computing likelihood ratios).
MIXTURE_TARGET_DISCOUNT_FACTOR = 0.99
MIXTURE_PROPOSAL_DISCOUNT_FACTOR = 0.95

# Const env mutators: pin the env discount factor to the mixture target
# (0.99), optionally overriding the overtime cost. The swept ``val`` is
# ignored by design (it is the agent-side lambda_0 / sweep variable).
mutate_mixture_target_discount_factor = partial(
    set_env_fields, {'discount_factor': MIXTURE_TARGET_DISCOUNT_FACTOR})
mutate_mixture_target_discount_factor_overtime_50 = partial(
    set_env_fields,
    {'discount_factor': MIXTURE_TARGET_DISCOUNT_FACTOR, 'overtime_cost_by_day': 50})
mutate_mixture_target_discount_factor_overtime_5 = partial(
    set_env_fields,
    {'discount_factor': MIXTURE_TARGET_DISCOUNT_FACTOR, 'overtime_cost_by_day': 5})


def mutate_mixture_target_initial_state_congestion(env_args, occupancy_level):
    """0.99-target case study whose reset initial state is pinned at
    ``occupancy_level`` of total capacity (see
    ``mutate_initial_state_congestion``); the swept ``val`` IS the occupancy.
    Train from that state with ``generate_params.py train --reset-init-state``."""
    mutate_mixture_target_discount_factor(env_args, occupancy_level)
    mutate_initial_state_congestion(env_args, occupancy_level)


def mutate_mixture_geometric_scenario_const(agent_args, val, lambda_0, sample_path_number):
    """Const agent mutator: mixture-geometric proposal with ``lambda_0`` on the
    long (target) component and ``sample_path_number`` Benders scenarios. The
    swept ``val`` (an env-side variable such as occupancy) is ignored."""
    mutate_mixture_geometric_proposal_lambda_0(agent_args, lambda_0)
    agent_args['policy_id'] += '_scenario_' + str(sample_path_number)
    agent_args['agent_args']['sample_path_number'] = sample_path_number


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
