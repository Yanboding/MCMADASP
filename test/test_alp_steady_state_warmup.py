"""Tests for the shared warm-up-policy evaluation mode.

``evaluate_policy_costs_with_information_relaxation(warm_up_policy_id=...)``
must (1) evaluate the warm-up policy first over the full sample path and
capture its executed warm-up prefix, (2) seed every other policy with that
prefix so it simulates the tail only, (3) memoize the information relaxation
bounds shared by all policies, and (4) leave the legacy behaviour (no
``warm_up_policy_id``) unchanged.

``calculate_policy_costs_with_penalty(warm_up_trajectory=...)`` must produce a
record whose costs/penalties/scheduled_patients cover the FULL path (shared
warm-up prefix + own tail) while the cost totals only count the tail.

The Gurobi-backed solvers are replaced by fakes, so the tests run without a
Gurobi license.
"""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run


ALP_WARMUP_STATE = ([[7, 7], [6, 6]], [3, 3], [2, 1])


class _FakeLowerBoundAgent:
    """Stands in for the two ApproxQAgent lower-bound instances."""

    instances = []

    def __init__(self, *args, **kwargs):
        self.calls = []
        _FakeLowerBoundAgent.instances.append(self)

    def calculate_information_relaxation_cost(self, state, sample_path=None, period_weights=None):
        self.calls.append({'state': state, 'sample_path_len': len(sample_path)})
        return 100.0


def _to_lists(value):
    return value.tolist() if hasattr(value, 'tolist') else value


def _fake_calculate_policy_costs(calls_log, **kwargs):
    calls_log.append(kwargs)
    result = {
        'policy_id': kwargs['policy_id'],
        'agent_name': kwargs['agent_name'],
        'agent_args': {},
        'penalized_cost': 110.0,
        'total_cost': 105.0,
        'total_penalty': 5.0,
        'warmup_state': ALP_WARMUP_STATE,
        'costs': [1.0],
        'penalties': [1.0],
    }
    if kwargs.get('return_warm_up_trajectory'):
        result['warm_up_trajectory'] = {
            'states': [ALP_WARMUP_STATE] * kwargs['warm_up_periods'],
            'actions': [[[0, 0]]] * kwargs['warm_up_periods'],
            'costs': [1.0] * kwargs['warm_up_periods'],
            'penalties': [1.0] * kwargs['warm_up_periods'],
            'end_state': ALP_WARMUP_STATE,
        }
    return result


class AlpSteadyStateWarmupTest(unittest.TestCase):
    """Orchestration of evaluate_policy_costs_with_information_relaxation."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_cwd = os.getcwd()
        os.chdir(self._tmpdir.name)
        _FakeLowerBoundAgent.instances = []
        self.policy_calls = []
        patchers = [
            mock.patch.object(run, 'get_config_by_type',
                              return_value=SimpleNamespace(env=SimpleNamespace(discount_factor=0.99))),
            mock.patch.object(run, '_build_generating_function', return_value=mock.MagicMock()),
            mock.patch.object(run, 'ApproxQAgent', _FakeLowerBoundAgent),
            mock.patch.object(run, 'calculate_policy_costs_with_penalty',
                              side_effect=lambda **kw: _fake_calculate_policy_costs(self.policy_calls, **kw)),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmpdir.cleanup()

    def _evaluate(self, warm_up_policy_id, policy_ids=('approx_penalized_hindsight', 'row_gen_alp', 'myopic')):
        return run.evaluate_policy_costs_with_information_relaxation(
            uid='uid-test',
            experiment_name='unit_test_alp_warmup',
            mutate_val=0.5,
            init_state=([[1, 1], [1, 1]], [0, 0], [4, 3]),
            sample_path=[[1, 2]] * 6,
            warm_up_periods=2,
            env_args={},
            policy_specs=[
                {'policy_id': pid, 'agent_name': pid, 'agent_args': {}} for pid in policy_ids
            ],
            group_id='g',
            grb_env=None,
            grb_sub_envs=None,
            job_id='unittest',
            generating_function_spec={'name': 'linear_penalty'},
            warm_up_policy_id=warm_up_policy_id,
        )

    def test_warm_up_policy_runs_first_and_returns_trajectory(self):
        self._evaluate('row_gen_alp')
        first_call = self.policy_calls[0]
        self.assertEqual(first_call['policy_id'], 'row_gen_alp')
        self.assertEqual(len(first_call['sample_path']), 6)
        self.assertEqual(first_call['warm_up_periods'], 2)
        self.assertTrue(first_call['return_warm_up_trajectory'])
        self.assertIsNone(first_call['warm_up_trajectory'])

    def test_other_policies_seeded_with_alp_prefix_on_full_path(self):
        self._evaluate('row_gen_alp')
        tail_calls = self.policy_calls[1:]
        self.assertEqual(
            sorted(call['policy_id'] for call in tail_calls),
            ['approx_penalized_hindsight', 'myopic'],
        )
        for call in tail_calls:
            # Full path + full warm_up_periods: the record keeps its full-length
            # shape; the seeding trajectory makes the policy skip the warm-up.
            self.assertEqual(call['warm_up_periods'], 2)
            self.assertEqual(len(call['sample_path']), 6)
            self.assertFalse(call['return_warm_up_trajectory'])
            trajectory = call['warm_up_trajectory']
            self.assertEqual(len(trajectory['states']), 2)
            self.assertEqual(len(trajectory['costs']), 2)
            self.assertEqual(trajectory['end_state'], ALP_WARMUP_STATE)

    def test_lower_bounds_memoized_across_shared_state(self):
        self._evaluate('row_gen_alp')
        zero_instance, penalized_instance = _FakeLowerBoundAgent.instances
        # All three policies share the ALP warm-up state, so each bound is
        # solved exactly once and evaluated on the tail.
        self.assertEqual(len(zero_instance.calls), 1)
        self.assertEqual(len(penalized_instance.calls), 1)
        self.assertEqual(zero_instance.calls[0]['sample_path_len'], 4)

    def test_results_written_for_every_policy_without_trajectory(self):
        summary = self._evaluate('row_gen_alp')
        self.assertEqual(len(summary), 3)
        output_file = os.path.join('experiments', 'results', 'unit_test_alp_warmup', 'unittest.jsonl')
        with open(output_file) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(
            sorted(row['policy_id'] for row in rows),
            ['approx_penalized_hindsight', 'myopic', 'row_gen_alp'],
        )
        for row in rows:
            self.assertEqual(row['warm_up_periods'], 2)
            # The warm-up trajectory is an internal hand-off, not a record field.
            self.assertNotIn('warm_up_trajectory', row)

    def test_unknown_warm_up_policy_raises(self):
        with self.assertRaises(ValueError):
            self._evaluate('row_gen_alp', policy_ids=('myopic',))

    def test_legacy_mode_unchanged_without_warm_up_policy(self):
        self._evaluate(None)
        self.assertEqual(len(self.policy_calls), 3)
        for call in self.policy_calls:
            self.assertEqual(call['warm_up_periods'], 2)
            self.assertEqual(len(call['sample_path']), 6)
            self.assertIsNone(call['warm_up_trajectory'])
            self.assertFalse(call['return_warm_up_trajectory'])


class _FakeSchedulingEnv:
    """Deterministic env: cost of period t is 10*t, state counts periods."""

    discount_factor = 1.0
    planning_horizon = 1
    num_types = 1

    def reset(self, init_state, t, new_arrivals):
        self.reset_args = {'init_state': init_state, 't': t}
        self.t = t
        self.sample_path = new_arrivals
        self.state = tuple(np.array(component) for component in init_state)
        return self.state, {}

    def step(self, action):
        cost = 10.0 * self.t
        regular, overtime, waitlist = self.state
        next_state = (regular + 1, overtime, waitlist)
        done = self.t >= len(self.sample_path) + 1
        self.t += 1
        self.state = next_state
        return next_state, cost, done, {}


class _FakeAgent:
    def __init__(self, *args, **kwargs):
        pass

    def solve(self, state, t):
        return None, (np.zeros((1, 1)), np.zeros(1)), None


class CalculatePolicyCostsSeedingTest(unittest.TestCase):
    """Warm-up-trajectory seeding inside calculate_policy_costs_with_penalty."""

    SAMPLE_PATH = [[1]] * 5
    WARM_UP = 2

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_cwd = os.getcwd()
        os.chdir(self._tmpdir.name)
        self.env = _FakeSchedulingEnv()
        patchers = [
            mock.patch.object(run, 'get_config_by_type',
                              side_effect=lambda **kw: SimpleNamespace(env=self.env)),
            mock.patch.object(run, 'MyopicAgent', _FakeAgent),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.generating_function = SimpleNamespace(
            calculate_penalty=lambda state, action, arrivals: 0.5,
            coefficients=[0.0],
        )

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmpdir.cleanup()

    def _run_policy(self, policy_id, warm_up_trajectory=None, return_warm_up_trajectory=False):
        return run.calculate_policy_costs_with_penalty(
            uid='uid-calc',
            experiment_name='unit_test_calc_seeding',
            policy_id=policy_id,
            agent_name='myopic',
            agent_args={},
            env_args={},
            init_state=(np.zeros(1), np.zeros(1), np.zeros(1)),
            sample_path=self.SAMPLE_PATH,
            warm_up_periods=self.WARM_UP,
            generating_function=self.generating_function,
            warm_up_trajectory=warm_up_trajectory,
            return_warm_up_trajectory=return_warm_up_trajectory,
        )

    def test_full_run_returns_warm_up_prefix(self):
        result = self._run_policy('warmup_policy', return_warm_up_trajectory=True)
        # 5 arrivals + trailing period -> 6 stage costs of 10*t.
        self.assertEqual(result['costs'], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        self.assertEqual(result['total_cost'], 30.0 + 40.0 + 50.0 + 60.0)
        trajectory = result['warm_up_trajectory']
        self.assertEqual(trajectory['costs'], [10.0, 20.0])
        self.assertEqual(len(trajectory['states']), self.WARM_UP)
        self.assertEqual(trajectory['end_state'], result['warmup_state'])

    def test_seeded_run_records_full_trajectory_but_counts_tail_only(self):
        trajectory = self._run_policy('warmup_policy', return_warm_up_trajectory=True)['warm_up_trajectory']
        result = self._run_policy('seeded_policy', warm_up_trajectory=trajectory)
        # The rollout resumed at period warm_up + 1 from the prefix end state.
        self.assertEqual(self.env.reset_args['t'], self.WARM_UP + 1)
        self.assertEqual(
            [_to_lists(component) for component in self.env.reset_args['init_state']],
            list(trajectory['end_state']),
        )
        # Record covers the FULL path (prefix + own tail)...
        self.assertEqual(result['costs'], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        self.assertEqual(len(result['scheduled_patients']), 6)
        self.assertEqual(len(result['penalties']), 5)
        # ...while totals only count the post-warm-up tail.
        self.assertEqual(result['total_cost'], 30.0 + 40.0 + 50.0 + 60.0)
        self.assertEqual(result['warmup_state'], trajectory['end_state'])

    def test_prefix_length_mismatch_raises(self):
        trajectory = self._run_policy('warmup_policy', return_warm_up_trajectory=True)['warm_up_trajectory']
        short_trajectory = {key: value[:1] if isinstance(value, list) else value
                            for key, value in trajectory.items()}
        with self.assertRaises(ValueError):
            self._run_policy('other_policy', warm_up_trajectory=short_trajectory)


if __name__ == '__main__':
    unittest.main()
