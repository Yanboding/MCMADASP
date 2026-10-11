import json
import os
import tempfile
import unittest

from param_generation.cli import main as cli_main
from param_generation.datasets import generate_test_paths_and_init_state
from test.test_eval_proposal_generation import make_test_envs

ANCHOR = [11.0, 12.0, 13.0]
FITTED = [21.0, 22.0, 23.0]


def generate(policy_ids, alp_coefficients=None, penalty_dir=None):
    return generate_test_paths_and_init_state(
        test_envs=make_test_envs(),
        test_sample_path_num=2,
        warm_up_periods=0,
        num_periods=None,
        dat_file=None,
        is_require_penalty_coefficients=penalty_dir is not None,
        is_random_initial_state=False,
        policy_ids=list(policy_ids),
        penalty_coefficients_dir=penalty_dir,
        evaluation_proposal_spec={'type': 'fixed', 'max_length': 3},
        alp_coefficients=alp_coefficients,
    )


def write_fitted(folder):
    with open(os.path.join(folder, '1.jsonl'), 'w', encoding='utf-8') as handle:
        handle.write(json.dumps({'uid': 'fitted-uid', 'mutate_val': 0.1,
                                 'tight_penalized_lower_bound': 1.0,
                                 'coefficients': FITTED}) + '\n')


class TestPenalizedHindsightSplit(unittest.TestCase):
    def specs(self, records):
        return {spec['policy_id']: spec for spec in records[0]['policy_specs']}

    def test_both_variants_carry_their_own_coefficients(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fitted(tmp)
            records = generate(['alp_penalized_hindsight', 'po_penalized_hindsight'],
                               alp_coefficients=ANCHOR, penalty_dir=tmp)
        specs = self.specs(records)
        self.assertEqual(sorted(specs), ['alp_penalized_hindsight', 'po_penalized_hindsight'])
        self.assertEqual(
            specs['alp_penalized_hindsight']['agent_args']['generating_function_spec']['coefficients'],
            ANCHOR)
        self.assertEqual(
            specs['po_penalized_hindsight']['agent_args']['generating_function_spec']['coefficients'],
            FITTED)

    def test_both_variants_keep_the_existing_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_fitted(tmp)
            records = generate(['alp_penalized_hindsight', 'po_penalized_hindsight'],
                               alp_coefficients=ANCHOR, penalty_dir=tmp)
        for spec in self.specs(records).values():
            self.assertEqual(spec['agent_name'], 'approx_penalized_hindsight')
            self.assertEqual(spec['agent_args']['solver_name'], 'approx_penalized_hindsight')

    def test_the_alp_variant_needs_the_anchor(self):
        with self.assertRaises(ValueError) as caught:
            generate(['alp_penalized_hindsight'])
        self.assertIn('alp_coefficients', str(caught.exception))

    def test_cli_exposes_the_anchor_on_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            anchor_file = os.path.join(tmp, 'anchor.jsonl')
            with open(anchor_file, 'w', encoding='utf-8') as handle:
                handle.write(json.dumps({'uid': 'a', 'mutate_val': 0.5,
                                         'coefficients': ANCHOR}) + '\n')
            fitted_dir = os.path.join(tmp, 'fitted')
            os.makedirs(fitted_dir)
            with open(os.path.join(fitted_dir, '1.jsonl'), 'w', encoding='utf-8') as handle:
                handle.write(json.dumps({'uid': 'f', 'mutate_val': 0.5,
                                         'tight_penalized_lower_bound': 1.0,
                                         'coefficients': FITTED}) + '\n')
            records = cli_main([
                'eval', 'toy_stratified_099_scenario_512', '--paths', '2', '--groups', '2',
                '--policies', 'alp_penalized_hindsight,po_penalized_hindsight',
                '--penalty-function', 'absorption_alp_penalty',
                '--eval-proposal', '{"type": "fixed", "max_length": 3}',
                '--penalty-dir', fitted_dir, '--alp-coefficients', anchor_file,
                '--dat', os.path.join(tmp, 'table.dat')])
        specs = {s['policy_id']: s for s in records[0]['policy_specs']}
        self.assertEqual(
            specs['alp_penalized_hindsight']['agent_args']['generating_function_spec']['coefficients'],
            ANCHOR)
        self.assertEqual(
            specs['po_penalized_hindsight']['agent_args']['generating_function_spec']['coefficients'],
            FITTED)


if __name__ == '__main__':
    unittest.main()
