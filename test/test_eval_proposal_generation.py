import json
import os
import tempfile
import unittest
from unittest import mock

from param_generation.cli import recipe_toy_eval_proposal_comparison
from param_generation.datasets import generate_test_paths_and_init_state
from param_generation.experiment_specs import ExperimentSpec, build_variation_test_env
from param_generation.mutators import (
    mutate_initial_state_congestion_05_const,
    mutate_mixture_geometric_proposal_lambda_0,
)
from param_generation.registry import EXPERIMENT_SPECS

MIXTURE_SPEC = {
    'type': 'mixture_geometric',
    'target_discount_factor': 0.99,
    'discount_factor_proposal': 0.95,
    'lambda_0': 0.1,
}


def make_test_envs():
    """Single-variant toy env identical to mixture_probability_toy_study val=0.1."""
    spec = ExperimentSpec(
        name='eval_prop_unit_test',
        config_type='toy',
        val_args=[0.1],
        mutate=mutate_initial_state_congestion_05_const,
        agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
    )
    return build_variation_test_env(spec)


def generate(evaluation_proposal_spec, **overrides):
    kwargs = dict(
        test_envs=make_test_envs(),
        test_sample_path_num=8,
        warm_up_periods=0,
        num_periods=None,
        dat_file=None,
        is_require_penalty_coefficients=False,
        is_random_initial_state=False,
        policy_ids=['approx_penalized_hindsight'],
        evaluation_proposal_spec=evaluation_proposal_spec,
    )
    kwargs.update(overrides)
    return generate_test_paths_and_init_state(**kwargs)


class TestEvaluationProposalDecoupling(unittest.TestCase):
    def test_fixed_eval_proposal_lengths_weights_and_untouched_policy(self):
        records = generate({'type': 'fixed', 'max_length': 458})
        self.assertEqual(len(records), 8)
        expected_weights = [0.99 ** t for t in range(459)]
        for record in records:
            self.assertEqual(len(record['sample_path']), 458)
            self.assertEqual(len(record['period_weights']), 459)
            for observed, expected in zip(record['period_weights'], expected_weights):
                self.assertAlmostEqual(observed, expected, places=12)
            # Decoupling: the embedded policy still carries the agent's own
            # mixture proposal, untouched by the evaluation proposal.
            self.assertEqual(
                record['policy_specs'][0]['agent_args']['sample_path_length_proposal'],
                MIXTURE_SPEC,
            )

    def test_geometric_target_eval_proposal_weights_discount_after_first(self):
        # The rollout visits L+1 decision periods for an L-arrival tail
        # (trailing period: stage cost, no arrival), so period s is visited iff
        # L >= s-1 and its unbiased weight divides by P(L >= s-1). Under the
        # target-geometric proposal that gives 1 for s=1 and gamma afterwards.
        records = generate({'type': 'geometric', 'discount_factor_proposal': 0.99})
        for record in records:
            self.assertEqual(
                len(record['period_weights']), len(record['sample_path']) + 1
            )
            self.assertAlmostEqual(record['period_weights'][0], 1.0, places=12)
            for weight in record['period_weights'][1:]:
                self.assertAlmostEqual(weight, 0.99, places=12)

    def test_mixture_eval_proposal_weights_bounded(self):
        # Shifted weights: w_1 = 1; for s >= 2, gamma <= w_s <= gamma / lambda_0.
        records = generate(MIXTURE_SPEC)
        for record in records:
            self.assertAlmostEqual(record['period_weights'][0], 1.0, places=12)
            for weight in record['period_weights'][1:]:
                self.assertGreaterEqual(weight, 0.99 - 1e-12)
                self.assertLessEqual(weight, 0.99 / 0.1 + 1e-9)

    def test_mixture_eval_proposal_weights_match_shifted_closed_form(self):
        gamma, q, lambda_0 = 0.99, 0.95, 0.1
        records = generate(MIXTURE_SPEC)
        for record in records:
            weights = record['period_weights']
            self.assertEqual(len(weights), len(record['sample_path']) + 1)
            for index, observed in enumerate(weights):
                s = index + 1
                survival = (
                    1.0 if s == 1
                    else lambda_0 * gamma ** (s - 2) + (1 - lambda_0) * q ** (s - 2)
                )
                self.assertAlmostEqual(observed, gamma ** (s - 1) / survival, places=12)

    def test_none_falls_back_to_agent_proposal_spec(self):
        # Regression guard: omitting evaluation_proposal_spec must reproduce the
        # coupled behavior (tails drawn from the agent's own spec). Both calls
        # rebuild the env from the same seeds, so records must match exactly.
        records_default = generate(None)
        records_explicit = generate(MIXTURE_SPEC)
        self.assertEqual(len(records_default), len(records_explicit))
        for record_default, record_explicit in zip(records_default, records_explicit):
            self.assertEqual(record_default['uid'], record_explicit['uid'])
            self.assertEqual(
                record_default['sample_path'], record_explicit['sample_path']
            )
            self.assertEqual(
                record_default['period_weights'], record_explicit['period_weights']
            )


class TestPenaltyCoefficientsDir(unittest.TestCase):
    def test_coefficients_loaded_from_override_dir(self):
        fake_coefficients = [1.0, 2.0, 3.0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            record = {
                'coefficients': fake_coefficients,
                'tight_penalized_lower_bound': 0.0,
                'mutate_val': 0.1,
                'sample_path_number': 256,
            }
            with open(os.path.join(tmp_dir, 'coefficients.jsonl'), 'w') as f:
                f.write(json.dumps(record) + '\n')
            # If the override dir is ignored, the loader misses and the code
            # falls back to retraining — patch that path so the test fails
            # fast instead of launching a real (slow, Gurobi-bound) training.
            with mock.patch(
                'param_generation.datasets.train_penalty_coefficients',
                side_effect=AssertionError(
                    'retraining triggered — coefficients were not loaded '
                    'from penalty_coefficients_dir'
                ),
            ):
                records = generate(
                    MIXTURE_SPEC,
                    is_require_penalty_coefficients=True,
                    penalty_coefficients_dir=tmp_dir,
                )
        for saved in records:
            self.assertEqual(
                saved['generating_function_spec']['coefficients'], fake_coefficients
            )
            self.assertEqual(
                saved['policy_specs'][0]['agent_args']['generating_function_spec']['coefficients'],
                fake_coefficients,
            )


ARM_NAMES = [
    'toy_eval_proposal_geometric_099',
    'toy_eval_proposal_fixed_459',
    'toy_eval_proposal_mixture_095_l01',
]


class TestEvaluationArmSpecs(unittest.TestCase):
    def test_arm_specs_materialize_identical_env_and_policy(self):
        reference = build_variation_test_env(
            EXPERIMENT_SPECS['mixture_probability_toy_study']
        )
        reference_variant = next(
            variant for key, variant in reference.items() if key[2] == 0.1
        )
        for name in ARM_NAMES:
            self.assertIn(name, EXPERIMENT_SPECS)
            spec = EXPERIMENT_SPECS[name]
            self.assertEqual(list(spec.val_args), [0.1])
            self.assertEqual(spec.config_type, 'toy')
            variants = build_variation_test_env(spec)
            self.assertEqual(len(variants), 1)
            ((group_uid, experiment_name, val), variant) = next(iter(variants.items()))
            self.assertEqual(experiment_name, name)
            self.assertEqual(val, 0.1)
            # Same env + same policy as the reference arm: only the name differs.
            self.assertEqual(variant, reference_variant)


class TestEvalProposalComparisonRecipe(unittest.TestCase):
    def test_recipe_writes_single_dat_with_all_arms(self):
        fake_coefficients = [4.0, 5.0, 6.0]
        with tempfile.TemporaryDirectory() as tmp_dir:
            record = {
                'coefficients': fake_coefficients,
                'tight_penalized_lower_bound': 0.0,
                'mutate_val': 0.1,
                'sample_path_number': 256,
            }
            with open(os.path.join(tmp_dir, 'coefficients.jsonl'), 'w') as f:
                f.write(json.dumps(record) + '\n')
            dat_file = os.path.join(tmp_dir, 'table.dat')
            records = recipe_toy_eval_proposal_comparison(
                test_sample_path_num=4,
                num_groups=2,
                dat_file=dat_file,
                penalty_coefficients_dir=tmp_dir,
            )
            self.assertEqual(len(records), 12)

            with open(dat_file) as f:
                lines = f.readlines()
        self.assertEqual(len(lines), 2)
        parsed = []
        for line in lines:
            payload = line[line.index("'") + 1:line.rindex("'")]
            parsed.extend(json.loads(payload))
        self.assertEqual(len(parsed), 12)

        by_arm = {}
        for saved in parsed:
            by_arm.setdefault(saved['experiment_name'], []).append(saved)
            for key in ('sample_path', 'period_weights', 'policy_specs',
                        'generating_function_spec', 'init_state', 'uid'):
                self.assertIn(key, saved)
            self.assertEqual(
                saved['generating_function_spec']['coefficients'], fake_coefficients
            )
        self.assertEqual(
            {name: len(items) for name, items in by_arm.items()},
            {
                'toy_eval_proposal_geometric_099': 4,
                'toy_eval_proposal_fixed_459': 4,
                'toy_eval_proposal_mixture_095_l01': 4,
            },
        )
        for saved in by_arm['toy_eval_proposal_fixed_459']:
            self.assertEqual(len(saved['sample_path']), 458)
        for saved in by_arm['toy_eval_proposal_geometric_099']:
            weights = saved['period_weights']
            self.assertAlmostEqual(weights[0], 1.0, places=12)
            for weight in weights[1:]:
                self.assertAlmostEqual(weight, 0.99, places=12)


if __name__ == '__main__':
    unittest.main()
