import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_maker import ALPRowGenerationAgent, ApproxQAgent
from experiments import get_config_by_type
from generating_function import LinearPenaltyFunction
from importance_sampling import build_proposal


class AlpRegularFirstTest(unittest.TestCase):
    def test_solve_returns_regular_first_overtime_split(self):
        env = get_config_by_type('toy').env
        planning_horizon = env.planning_horizon
        num_types = env.num_types
        # Inflated U (regular-booking value) and zero V make shifting slots
        # into overtime look strictly profitable to the unrepaired LP.
        coefficients = (
            [0.0] + [1000.0] * planning_horizon + [0.0] * planning_horizon + [0.0] * num_types
        )
        agent = ALPRowGenerationAgent(
            env, discount_factor=env.discount_factor, coefficients=coefficients
        )

        regular_bookings = np.zeros(planning_horizon, dtype=int)
        overtimes = np.zeros(planning_horizon, dtype=int)
        waitlist = np.zeros(num_types, dtype=int)
        waitlist[0] = 1
        state = (regular_bookings, overtimes, waitlist)

        _, (advance_scheduling_decision, overtime_decision), _ = agent.solve(state, t=1)

        new_booking_slots = env.convert_action_to_booking_slots(advance_scheduling_decision)
        expected_overtime = np.maximum(
            regular_bookings + new_booking_slots - env.regular_capacity, 0
        )
        np.testing.assert_array_equal(
            np.asarray(overtime_decision), expected_overtime,
            err_msg='overtime used while regular capacity remains',
        )
        self.assertTrue(np.all(np.asarray(overtime_decision) + overtimes <= env.overtime_capacity))
        post_regular = regular_bookings + new_booking_slots - np.asarray(overtime_decision)
        self.assertTrue(np.all(post_regular >= 0))
        self.assertTrue(np.all(post_regular <= env.regular_capacity))

    def test_wasteful_overtime_action_is_repaired(self):
        env = get_config_by_type('toy').env
        planning_horizon = env.planning_horizon
        num_types = env.num_types
        coefficients = [0.0] * (1 + 2 * planning_horizon + num_types)
        agent = ALPRowGenerationAgent(
            env, discount_factor=env.discount_factor, coefficients=coefficients
        )

        regular_bookings = np.zeros(planning_horizon, dtype=int)
        state = (
            regular_bookings,
            np.zeros(planning_horizon, dtype=int),
            np.array([1, 0]),
        )
        forced_scheduling = np.zeros((env.booking_window_size, num_types), dtype=int)
        forced_scheduling[1, 0] = 1
        forced_slots = env.convert_action_to_booking_slots(forced_scheduling)
        wasteful_action = (forced_scheduling, forced_slots)

        _, (advance_scheduling_decision, overtime_decision), _ = agent.solve(
            state, t=1, action=wasteful_action
        )

        np.testing.assert_array_equal(advance_scheduling_decision, forced_scheduling)
        expected_overtime = np.maximum(
            regular_bookings + forced_slots - env.regular_capacity, 0
        )
        np.testing.assert_array_equal(
            np.asarray(overtime_decision), expected_overtime,
            err_msg='forced wasteful overtime was not repaired to regular-first',
        )


class HindsightRegularFirstTest(unittest.TestCase):
    def test_wasteful_overtime_action_is_repaired(self):
        env = get_config_by_type('toy').env
        planning_horizon = env.planning_horizon
        num_types = env.num_types
        generating_function = LinearPenaltyFunction(env=env, coefficients=None)
        generating_function.set_coefficients(
            [0.0] * generating_function.number_of_coefficients
        )
        agent = ApproxQAgent(
            env=env,
            discount_factor=env.discount_factor,
            sample_path_number=2,
            generating_function=generating_function,
            sample_path_proposal=build_proposal({'type': 'fixed', 'max_length': 2}),
            solver_name='approx_penalized_hindsight',
            is_trained=True,
        )

        regular_bookings = np.zeros(planning_horizon, dtype=int)
        state = (
            regular_bookings,
            np.zeros(planning_horizon, dtype=int),
            np.array([1, 0]),
        )
        forced_scheduling = np.zeros((env.booking_window_size, num_types), dtype=int)
        forced_scheduling[1, 0] = 1
        forced_slots = env.convert_action_to_booking_slots(forced_scheduling)
        wasteful_action = (forced_scheduling, forced_slots)

        _, (advance_scheduling_decision, overtime_decision), _ = agent.solve(
            state, t=1, action=wasteful_action, parallel=False
        )

        np.testing.assert_array_equal(advance_scheduling_decision, forced_scheduling)
        expected_overtime = np.maximum(
            regular_bookings + forced_slots - env.regular_capacity, 0
        )
        np.testing.assert_array_equal(
            np.asarray(overtime_decision), expected_overtime,
            err_msg='forced wasteful overtime was not repaired to regular-first',
        )


if __name__ == '__main__':
    unittest.main()
