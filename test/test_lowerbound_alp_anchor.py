import json
import os
import tempfile
import unittest
from unittest import mock

from param_generation import datasets

from param_generation.cli import main as cli_main
from param_generation.datasets import generate_test_paths_and_init_state
from test.test_eval_proposal_generation import make_test_envs

ANCHOR = [1.5, -2.5, 3.0]


def generate(alp_coefficients=None, policy_ids=()):
    return generate_test_paths_and_init_state(
        test_envs=make_test_envs(),
        test_sample_path_num=2,
        warm_up_periods=0,
        num_periods=None,
        dat_file=None,
        is_require_penalty_coefficients=False,
        is_random_initial_state=False,
        policy_ids=list(policy_ids),
        evaluation_proposal_spec={'type': 'fixed', 'max_length': 3},
        alp_coefficients=alp_coefficients,
    )


class TestLowerBoundAlpAnchor(unittest.TestCase):
    def test_records_carry_the_anchor(self):
        records = generate(ANCHOR)
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(record['alp_coefficients'], ANCHOR)

    def test_records_without_an_anchor_omit_the_field(self):
        self.assertTrue(all('alp_coefficients' not in record for record in generate()))

    def test_the_anchor_is_part_of_the_record_uid(self):
        anchored = [record['uid'] for record in generate(ANCHOR)]
        plain = [record['uid'] for record in generate()]
        self.assertFalse(set(anchored) & set(plain))

    def test_a_requested_alp_policy_does_not_overwrite_the_anchor(self):
        with mock.patch.object(datasets, 'train_alp_coefficients',
                               return_value=(None, [99.0, -99.0])):
            records = generate(ANCHOR, policy_ids=['row_gen_alp'])
        self.assertTrue(records)
        for record in records:
            self.assertEqual(record['alp_coefficients'], ANCHOR)
            self.assertEqual(record['policy_specs'][0]['agent_args']['coefficients'], [99.0, -99.0])

    def test_several_variants_refuse_a_single_anchor(self):
        from param_generation.experiment_specs import ExperimentSpec, build_variation_test_env
        from param_generation.mutators import mutate_initial_state_congestion_05_const
        multi = build_variation_test_env(ExperimentSpec(
            name='anchor_multi_variant_test', config_type='toy', val_args=[0.1, 0.2],
            mutate=mutate_initial_state_congestion_05_const))
        with self.assertRaises(ValueError) as caught:
            generate_test_paths_and_init_state(
                test_envs=multi, test_sample_path_num=2, warm_up_periods=0, num_periods=None,
                dat_file=None, is_require_penalty_coefficients=False, is_random_initial_state=False,
                policy_ids=[], evaluation_proposal_spec={'type': 'fixed', 'max_length': 3},
                alp_coefficients=ANCHOR)
        self.assertIn('one variant', str(caught.exception))

    def test_cli_loads_the_anchor_from_a_coefficient_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchor_file = os.path.join(tmp, 'anchor.jsonl')
            with open(anchor_file, 'w', encoding='utf-8') as handle:
                handle.write(json.dumps({'uid': 'anchor-uid', 'mutate_val': 0.5,
                                         'coefficients': [7.0, 8.0]}) + '\n')
            penalty_dir = os.path.join(tmp, 'fitted')
            os.makedirs(penalty_dir)
            with open(os.path.join(penalty_dir, '1.jsonl'), 'w', encoding='utf-8') as handle:
                handle.write(json.dumps({'uid': 'fitted-uid', 'mutate_val': 0.5,
                                         'tight_penalized_lower_bound': 1.0,
                                         'coefficients': [11.0, 12.0]}) + '\n')
            records = cli_main([
                'lowerbound', 'toy_stratified_099_scenario_512', '--paths', '2',
                '--groups', '2', '--penalty-function', 'absorption_alp_penalty',
                '--eval-proposal', '{"type": "fixed", "max_length": 3}',
                '--penalty-dir', penalty_dir,
                '--alp-coefficients', anchor_file,
                '--dat', os.path.join(tmp, 'table.dat')])
        self.assertTrue(records)
        for record in records:
            self.assertEqual(record['alp_coefficients'], [7.0, 8.0])
            self.assertEqual(record['generating_function_spec']['coefficients'], [11.0, 12.0])


if __name__ == '__main__':
    unittest.main()
