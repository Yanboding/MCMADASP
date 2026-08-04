import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments import get_config_by_type
from param_generation.caching import (
    load_trained_coefficients_from_folder as real_load_trained_coefficients,
)
from param_generation.cli import recipe_saure_ejor_case_study
from param_generation.datasets import (
    generate_test_paths_and_init_state,
    offset_sample_generation_seeds,
)
from param_generation.experiment_specs import ExperimentSpec, build_variation_test_env
from param_generation.mutators import mutate_initial_state_congestion_05_const
from param_generation.registry import EXPERIMENT_SPECS

FIXED_SPEC = {'type': 'fixed', 'max_length': 9}
MIXTURE_SPEC = {
    'type': 'mixture_geometric',
    'target_discount_factor': 0.99,
    'discount_factor_proposal': 0.95,
    'lambda_0': 0.1,
}


def make_toy_envs():
    """Single-variant toy env; no policies are resolved so no training runs."""
    spec = ExperimentSpec(
        name='saure_ejor_unit_test',
        config_type='toy',
        val_args=[0.1],
        mutate=mutate_initial_state_congestion_05_const,
    )
    return build_variation_test_env(spec)


def make_warm_up_paths(num_paths, warm_up_periods):
    (variant,) = make_toy_envs().values()
    env = get_config_by_type(
        'infinite_custom',
        args=offset_sample_generation_seeds(variant['env_args'], 3003),
    ).env
    return [env.reset_arrivals(stop_time=warm_up_periods) for _ in range(num_paths)]


def generate(evaluation_proposal_spec, **overrides):
    kwargs = dict(
        test_envs=make_toy_envs(),
        test_sample_path_num=8,
        warm_up_periods=5,
        num_periods=None,
        dat_file=None,
        is_require_penalty_coefficients=False,
        is_random_initial_state=False,
        policy_ids=[],
        evaluation_proposal_spec=evaluation_proposal_spec,
    )
    kwargs.update(overrides)
    return generate_test_paths_and_init_state(**kwargs)


class TestOffsetSampleGenerationSeeds(unittest.TestCase):
    def test_shifts_the_three_seeds_and_copies(self):
        (variant,) = make_toy_envs().values()
        env_args = variant['env_args']
        shifted = offset_sample_generation_seeds(env_args, 7)
        self.assertEqual(shifted['env_random_seed'], env_args.get('env_random_seed', 0) + 7)
        self.assertEqual(shifted['arrival_random_seed'], env_args.get('arrival_random_seed', 42) + 7)
        self.assertEqual(shifted['stop_time_random_seed'], env_args.get('stop_time_random_seed', 1) + 7)
        self.assertIsNot(shifted, env_args)


class TestSharedWarmUpPaths(unittest.TestCase):
    def test_shared_prefixes_identical_across_proposals(self):
        warm_up_paths = make_warm_up_paths(8, 5)
        fixed_records = generate(FIXED_SPEC, warm_up_paths=warm_up_paths)
        mixture_records = generate(MIXTURE_SPEC, warm_up_paths=warm_up_paths)
        for prefix, fixed, mixture in zip(warm_up_paths, fixed_records, mixture_records):
            self.assertEqual(fixed['sample_path'][:5], np.asarray(prefix).tolist())
            self.assertEqual(fixed['sample_path'][:5], mixture['sample_path'][:5])

    def test_prefixes_differ_without_shared_warm_up_paths(self):
        fixed_records = generate(FIXED_SPEC)
        mixture_records = generate(MIXTURE_SPEC)
        self.assertNotEqual(
            [r['sample_path'][:5] for r in fixed_records],
            [r['sample_path'][:5] for r in mixture_records],
        )

    def test_tails_and_weights_unaffected_by_warm_up_paths(self):
        without = generate(FIXED_SPEC)
        with_shared = generate(FIXED_SPEC, warm_up_paths=make_warm_up_paths(8, 5))
        expected_weights = [0.99 ** t for t in range(10)]
        for a, b in zip(without, with_shared):
            self.assertEqual(a['sample_path'][5:], b['sample_path'][5:])
            self.assertEqual(len(b['sample_path']), 5 + 9)
            self.assertEqual(len(b['period_weights']), 10)
            for observed, expected in zip(b['period_weights'], expected_weights):
                self.assertAlmostEqual(observed, expected)

    def test_seed_offset_changes_tails_but_not_prefixes_or_env_args(self):
        warm_up_paths = make_warm_up_paths(8, 5)
        base = generate(MIXTURE_SPEC, warm_up_paths=warm_up_paths)
        offset = generate(
            MIXTURE_SPEC, warm_up_paths=warm_up_paths, sample_gen_seed_offset=2002
        )
        self.assertNotEqual(
            [r['sample_path'][5:] for r in base],
            [r['sample_path'][5:] for r in offset],
        )
        for a, b in zip(base, offset):
            self.assertEqual(a['sample_path'][:5], b['sample_path'][:5])
            self.assertEqual(a['env_args'], b['env_args'])

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            generate(FIXED_SPEC, warm_up_paths=make_warm_up_paths(3, 5))
        with self.assertRaises(ValueError):
            generate(FIXED_SPEC, warm_up_paths=make_warm_up_paths(8, 4))
        with self.assertRaises(ValueError):
            generate(
                FIXED_SPEC,
                warm_up_paths=make_warm_up_paths(8, 5),
                num_periods=20,
            )


class TestSaureEjorSpecs(unittest.TestCase):
    def test_env_args_identical_and_only_agent_args_differ(self):
        (rep,) = build_variation_test_env(
            EXPERIMENT_SPECS['case_study_ejor_replication']
        ).values()
        (steady,) = build_variation_test_env(
            EXPERIMENT_SPECS['case_study_ejor_alp_steady_state']
        ).values()
        self.assertEqual(rep['env_args'], steady['env_args'])
        self.assertEqual(rep['env_args']['discount_factor'], 0.99)
        self.assertEqual(rep['env_args']['overtime_cost_by_day'], 100)
        self.assertNotIn(
            'sample_path_length_proposal', rep['agent_args']['agent_args']
        )
        self.assertEqual(
            steady['agent_args']['agent_args']['sample_path_length_proposal'],
            {
                'type': 'mixture_geometric',
                'target_discount_factor': 0.99,
                'discount_factor_proposal': 0.95,
                'lambda_0': 0.1,
            },
        )


SYNTHETIC_COEFFICIENTS = [1.5, -2.0, 3.25]


def _loader_skipping_own_folders(*args, **kwargs):
    """Force the recipe's own-results-folder lookups (``folder_path=None``) to
    miss, so tests stay hermetic regardless of what currently sits in the real
    ``experiments/results/case_study_ejor_*`` folders."""
    if kwargs.get('folder_path') is None:
        return None
    return real_load_trained_coefficients(*args, **kwargs)


class TestSaureEjorRecipe(unittest.TestCase):
    def _write_synthetic_coefficients(self, tmp_dir):
        coefficients_dir = os.path.join(tmp_dir, 'trained')
        os.makedirs(coefficients_dir)
        with open(os.path.join(coefficients_dir, 'penalty_coefficients.jsonl'), 'w') as f:
            f.write(json.dumps({
                'uid': 'synthetic',
                'mutate_val': 0.1,
                'sample_path_number': 256,
                'tight_penalized_lower_bound': 1.0,
                'coefficients': SYNTHETIC_COEFFICIENTS,
            }) + '\n')
        return coefficients_dir

    def _run_recipe(self, tmp_dir):
        coefficients_dir = self._write_synthetic_coefficients(tmp_dir)
        alp_result = (0.0, [10.0, 20.0])
        with mock.patch(
            'param_generation.cli.load_trained_coefficients_from_folder',
            side_effect=_loader_skipping_own_folders,
        ), mock.patch(
            'param_generation.cli.train_alp_coefficients', return_value=alp_result
        ), mock.patch(
            'param_generation.datasets.train_alp_coefficients', return_value=alp_result
        ):
            return recipe_saure_ejor_case_study(
                test_sample_path_num=4,
                warm_up_periods=6,
                evaluation_periods=5,
                num_groups=2,
                dat_file=os.path.join(tmp_dir, 'table.dat'),
                penalty_coefficients_dir=coefficients_dir,
            )

    def test_missing_coefficients_fail_fast(self):
        # Neither the experiments' own folders nor the shared folder have
        # coefficients -> the recipe must raise before any training.
        with mock.patch(
            'param_generation.cli.load_trained_coefficients_from_folder',
            return_value=None,
        ):
            with self.assertRaises(RuntimeError):
                recipe_saure_ejor_case_study()

    def test_own_folder_coefficients_take_precedence(self):
        own_coefficients = [9.0, 8.0, 7.0]

        def loader_with_own(*args, **kwargs):
            if kwargs.get('folder_path') is None:
                return own_coefficients
            return real_load_trained_coefficients(*args, **kwargs)

        alp_result = (0.0, [10.0, 20.0])
        with tempfile.TemporaryDirectory() as tmp_dir:
            coefficients_dir = self._write_synthetic_coefficients(tmp_dir)
            with mock.patch(
                'param_generation.cli.load_trained_coefficients_from_folder',
                side_effect=loader_with_own,
            ), mock.patch(
                'param_generation.datasets._load_trained_coefficients_from_folder',
                side_effect=loader_with_own,
            ), mock.patch(
                'param_generation.cli.train_alp_coefficients', return_value=alp_result
            ), mock.patch(
                'param_generation.datasets.train_alp_coefficients', return_value=alp_result
            ):
                replication, steady_state = recipe_saure_ejor_case_study(
                    test_sample_path_num=2,
                    warm_up_periods=3,
                    evaluation_periods=2,
                    num_groups=1,
                    dat_file=os.path.join(tmp_dir, 'table.dat'),
                    penalty_coefficients_dir=coefficients_dir,
                )
        for record in replication + steady_state:
            self.assertEqual(
                record['generating_function_spec']['coefficients'], own_coefficients
            )

    def test_recipe_emits_both_experiments(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            replication, steady_state = self._run_recipe(tmp_dir)

            self.assertEqual(len(replication), 4)
            self.assertEqual(len(steady_state), 4)
            with open(os.path.join(tmp_dir, 'table.dat')) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            parsed_lines = []
            for line in lines:
                payload = line[line.index("'") + 1:line.rindex("'")]
                parsed_lines.append(json.loads(payload))
            parsed = [record for group in parsed_lines for record in group]
            self.assertEqual(len(parsed), 8)
            # Segmented layout with continuous line indices: the first
            # num_groups//2 lines hold experiment 1 only (bounds skipped), the
            # remaining lines experiment 2 only (bounds kept).
            self.assertTrue(lines[0].startswith('1 '))
            self.assertTrue(lines[1].startswith('2 '))
            self.assertTrue(all(
                record['experiment_name'] == 'case_study_ejor_replication'
                and record['skip_information_relaxation'] is True
                for record in parsed_lines[0]
            ))
            self.assertTrue(all(
                record['experiment_name'] == 'case_study_ejor_alp_steady_state'
                and 'skip_information_relaxation' not in record
                for record in parsed_lines[1]
            ))

            expected_weights = [0.99 ** t for t in range(5)]
            for rep, steady in zip(replication, steady_state):
                # Shared warm-up prefix; independent tails.
                self.assertEqual(rep['sample_path'][:6], steady['sample_path'][:6])
                self.assertNotEqual(rep['sample_path'][6:], steady['sample_path'][6:])
                # Replication: 6 warm-up + 4 tail arrivals; gamma**t weights.
                self.assertEqual(len(rep['sample_path']), 6 + 4)
                self.assertEqual(len(rep['period_weights']), 5)
                for observed, expected in zip(rep['period_weights'], expected_weights):
                    self.assertAlmostEqual(observed, expected)
                # Steady state: shared ALP warm-up policy.
                self.assertNotIn('warm_up_policy_id', rep)
                self.assertEqual(steady['warm_up_policy_id'], 'row_gen_alp')
                # Both embed the trained coefficients and the same shared ALP.
                for record in (rep, steady):
                    self.assertEqual(
                        record['generating_function_spec']['coefficients'],
                        SYNTHETIC_COEFFICIENTS,
                    )
                    alp_spec = next(
                        s for s in record['policy_specs']
                        if s['policy_id'] == 'row_gen_alp'
                    )
                    self.assertEqual(alp_spec['agent_args']['coefficients'], [10.0, 20.0])

            self.assertEqual(
                [s['policy_id'] for s in replication[0]['policy_specs']],
                ['row_gen_alp', 'myopic'],
            )
            hindsight_spec = next(
                s for s in steady_state[0]['policy_specs']
                if s['policy_id'].startswith('approx_penalized_hindsight')
            )
            self.assertEqual(
                hindsight_spec['agent_args']['sample_path_length_proposal'],
                {
                    'type': 'mixture_geometric',
                    'target_discount_factor': 0.99,
                    'discount_factor_proposal': 0.95,
                    'lambda_0': 0.1,
                },
            )


if __name__ == '__main__':
    unittest.main()
