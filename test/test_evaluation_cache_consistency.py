"""Evaluation caches must identify the complete validated evaluation record."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gurobipy as gp
import numpy as np

import run
from experiments import get_config_by_type
from generating_function import AbsorptionLinearPenaltyFunction, LinearPenaltyFunction
from importance_sampling.sample_path import Terminal
from param_generation import datasets
from test.test_eval_proposal_generation import make_test_envs
from utils import get_uid


class EvaluationCacheConsistencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grb_env = gp.Env(empty=True)
        cls.grb_env.setParam('OutputFlag', 0)
        cls.grb_env.start()

    @classmethod
    def tearDownClass(cls):
        cls.grb_env.dispose()

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.tmpdir.name)
        self.config = get_config_by_type('toy')
        env = self.config.env
        self.penalty = AbsorptionLinearPenaltyFunction(env)
        theta = np.zeros(self.penalty.number_of_coefficients)
        theta[0] = 1.0
        self.penalty.set_coefficients(theta)
        self.args = dict(
            uid='same-record', experiment_name='cached', policy_id='myopic',
            agent_name='myopic', agent_args={}, env_args=self.config.args,
            init_state=(np.r_[1., np.zeros(env.planning_horizon - 1)],
                        np.zeros(env.planning_horizon), np.zeros(env.num_types)),
            sample_path=np.zeros((0, env.num_types)), warm_up_periods=0,
            generating_function=self.penalty, grb_env=self.grb_env,
            period_weights=[1.0], terminal='absorbed',
        )

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmpdir.cleanup()

    def evaluate(self, **changes):
        return run.calculate_policy_costs_with_penalty(**{**self.args, **changes})

    def assert_fresh_result(self, changes):
        self.evaluate()
        reused = self.evaluate(**changes)
        fresh = self.evaluate(**{**changes, 'experiment_name': 'fresh'})
        for key in ('total_cost', 'total_penalty', 'costs', 'expected_terms', 'realized_terms'):
            self.assertEqual(reused[key], fresh[key], key)
        self.assertEqual(len(list(Path('experiments/results/cached/pickles').glob('*-result.pickle'))), 2)

    def test_changed_terminal_does_not_reuse_absorbed_penalty(self):
        self.assert_fresh_result({'terminal': 'truncated'})

    def test_changed_period_weights_do_not_reuse_totals(self):
        self.assert_fresh_result({'period_weights': [2.0]})

    def test_changed_initial_state_does_not_reuse_rollout(self):
        state = copy.deepcopy(self.args['init_state'])
        state[0][0] = 2.0
        self.assert_fresh_result({'init_state': state})

    def test_changed_arrivals_do_not_reuse_rollout(self):
        self.assert_fresh_result({'sample_path': [[1, 2]], 'period_weights': [1.0, 1.0]})

    def test_changed_env_args_do_not_reuse_rollout(self):
        self.assert_fresh_result({'env_args': {**self.config.args, 'arrival_rates': [2, 3]}})

    def test_changed_warmup_length_does_not_reuse_tail_accounting(self):
        self.args.update(sample_path=[[1, 2]], period_weights=[1.0, 1.0])
        self.assert_fresh_result({'warm_up_periods': 1})

    def test_normalized_terminal_and_trimmed_weights_share_cache(self):
        expected = self.evaluate()
        with mock.patch.object(run, 'MyopicAgent', side_effect=AssertionError('cache miss')):
            actual = self.evaluate(terminal=Terminal.ABSORBED, period_weights=np.array([1.0, 99.0]))
        self.assertEqual(expected, actual)

    def test_invalid_path_metadata_is_rejected_before_cache_read(self):
        self.evaluate()
        for changes in ({'period_weights': []}, {'period_weights': None},
                        {'terminal': None}, {'terminal': 'invalid'},
                        {'period_weights': [[1.0]]}):
            with self.subTest(changes=changes):
                with mock.patch.object(run, 'load_pickle_if_exists',
                                       side_effect=AssertionError('cache read before validation')):
                    with self.assertRaises(ValueError):
                        self.evaluate(**changes)

    def test_legacy_unversioned_result_is_not_reused(self):
        legacy = LinearPenaltyFunction(self.config.env)
        legacy.set_coefficients(np.zeros(legacy.number_of_coefficients))
        old_signature = get_uid({'agent_name': 'myopic', 'agent_args': {},
                                 'penalty_coefficients': legacy.coefficients.tolist()})
        folder = Path('experiments/results/cached/pickles')
        folder.mkdir(parents=True)
        run.atomic_pickle_dump(folder / f'same-record-myopic-{old_signature}-result.pickle',
                               {'stale_legacy_result': True})
        actual = self.evaluate(generating_function=legacy, terminal=None, period_weights=None)
        self.assertIn('total_cost', actual)
        self.assertNotIn('stale_legacy_result', actual)

    def test_changed_terminal_does_not_resume_absorbed_checkpoint(self):
        # A completed rollout checkpoint is a valid resume point; poison only
        # the old input's terms to make accidental cross-input reuse visible.
        saved_dump = run.atomic_pickle_dump
        def interrupt_before_result(path, payload):
            if str(path).endswith('-result.pickle'):
                raise RuntimeError('interrupt before result')
            saved_dump(path, {**payload, 'expected_terms': [1234.0]})
        with mock.patch.object(run, 'atomic_pickle_dump', side_effect=interrupt_before_result):
            with self.assertRaisesRegex(RuntimeError, 'interrupt'):
                self.evaluate()
        actual = self.evaluate(terminal='truncated')
        fresh = self.evaluate(terminal='truncated', experiment_name='fresh')
        self.assertEqual(actual['expected_terms'], fresh['expected_terms'])


class EvaluationRecordUidTest(unittest.TestCase):
    def generate(self, **changes):
        args = dict(test_envs=make_test_envs(), test_sample_path_num=1,
                    is_require_penalty_coefficients=False, policy_ids=['myopic'],
                    evaluation_proposal_spec={'type': 'fixed', 'max_length': 1})
        return datasets.generate_test_paths_and_init_state(**{**args, **changes})[0]

    def test_terminal_and_weights_are_part_of_uid(self):
        baseline = self.generate()
        for field, value in (('terminal', 'absorbed'), ('period_weights', [1.0, 2.0])):
            with self.subTest(field=field):
                with mock.patch.object(datasets, '_evaluation_' + field, return_value=value):
                    changed = self.generate()
                self.assertEqual(baseline['sample_path'], changed['sample_path'])
                self.assertNotEqual(baseline['uid'], changed['uid'])

    def test_warmup_policy_and_policy_settings_are_part_of_uid(self):
        baseline = self.generate()
        for changes in ({'warm_up_policy_id': 'myopic'}, {'policy_ids': []}):
            with self.subTest(changes=changes):
                changed = self.generate(**changes)
                self.assertEqual(baseline['sample_path'], changed['sample_path'])
                self.assertNotEqual(baseline['uid'], changed['uid'])


class PenaltySolverSelectionTest(unittest.TestCase):
    def test_default_benders_solver_still_trains_and_reuses_completed_record(self):
        config = get_config_by_type('toy')
        with tempfile.TemporaryDirectory() as tmp, gp.Env(empty=True) as grb_env:
            grb_env.setParam('OutputFlag', 0)
            grb_env.start()
            args = dict(uid='bounded-training', experiment_name=tmp, mutate_val=0.0,
                        sample_path_number=1, init_state_mode='shared',
                        init_state=config.reset_params['init_state'], init_state_seed=1,
                        env_args=config.args, agent_args={'agent_args': {
                            'coefficient_bound': 5.0,
                            'sample_path_length_proposal': {'type': 'fixed', 'max_length': 1}}},
                        training_generating_function_spec='linear_penalty', grb_env=grb_env,
                        grb_sub_envs=None, job_id='job')
            with mock.patch.dict(os.environ):
                os.environ.pop('PENALTY_TRAIN_SOLVER', None)
                actual = run.train_penalty_coefficients_for_env(**args)
                self.assertTrue(np.isfinite(actual['tight_penalized_lower_bound']))
                self.assertLessEqual(np.max(np.abs(actual['coefficients'])), 5.0 + 1e-8)
                with mock.patch.object(run, 'ApproxQAgent', side_effect=AssertionError('duplicate trained again')):
                    self.assertIsNone(run.train_penalty_coefficients_for_env(**args))

    def test_unsupported_solver_rejected_before_duplicate_check_or_model_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = dict(uid='cached', experiment_name=tmp, mutate_val=0.0,
                        sample_path_number=1, init_state_mode='shared', init_state=[],
                        init_state_seed=1, env_args={}, agent_args={},
                        training_generating_function_spec=None, grb_env=None,
                        grb_sub_envs=None, job_id='job')
            # A real matching output used to bypass the unsupported selection.
            Path(tmp, 'job.jsonl').write_text(json.dumps({'uid': 'cached'}) + '\n')
            for solver in ('extensive', 'extensive_form', 'ef', 'deterministic_equivalent', 'typo'):
                with self.subTest(solver=solver), mock.patch.dict(os.environ, PENALTY_TRAIN_SOLVER=solver):
                    with mock.patch.object(run, 'jsonl_uid_exists',
                                           side_effect=AssertionError('duplicate lookup before solver validation')):
                        with self.assertRaisesRegex(ValueError, 'PENALTY_TRAIN_SOLVER.*benders'):
                            run.train_penalty_coefficients_for_env(**args)
