# Evaluation-Proposal Comparison Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `python generate_params.py` emit a single `table.dat` with 3 x 4096 evaluation records — the same fixed `approx_penalized_hindsight` policy evaluated on sample paths drawn from three different length proposals (geometric 0.99 / fixed 458 / mixture λ₀=0.1).

**Architecture:** Add an optional `evaluation_proposal_spec` parameter to `generate_test_paths_and_init_state` so the evaluation-path proposal is decoupled from the agent's internal SAA proposal, plus a `penalty_coefficients_dir` passthrough so new experiment names reuse already-trained coefficients. Register three byte-identical-except-name `ExperimentSpec` arms and one CLI recipe that generates all three arms into one grouped dat file.

**Tech Stack:** Python 3, numpy, scipy.stats.geom, unittest/pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-29-eval-proposal-comparison-design.md` (approved; read it first).

## Global Constraints

- All changes backward compatible: omitting the new parameters must preserve current behavior exactly (old recipes untouched).
- The evaluated policy is identical across arms: agent `approx_penalized_hindsight`, internal mixture proposal `{'type': 'mixture_geometric', 'target_discount_factor': 0.99, 'discount_factor_proposal': 0.95, 'lambda_0': 0.1}`, `sample_path_number=256`.
- Fixed arm: proposal `max_length = int(geom.ppf(0.99, 1 - 0.99)) - 1 = 458` (459 weighted decision periods). NEVER 459 — `FixedLengthProposal(459)` raises `ValueError` in `_evaluation_period_weights` (survival is 0 at period 460).
- Penalty coefficients are loaded from `experiments/results/mixture_probability_toy_study` (record filter: `mutate_val=0.1`, `sample_path_number=256`; the matching record lives in `local_2.jsonl`), never retrained.
- Match existing code style: module-level mutators, unittest-style test classes, no type annotations in `param_generation` modules.
- Run tests from the repo root: `python3 -m pytest test/<file>.py -v`.
- Surgical changes only: do not reformat or "improve" adjacent code.

---

### Task 1: Decouple the evaluation proposal (`evaluation_proposal_spec`)

**Files:**
- Modify: `param_generation/datasets.py` (signature at line 133; proposal build at lines 284-287; docstring)
- Test: `test/test_eval_proposal_generation.py` (new file)

**Interfaces:**
- Produces: `generate_test_paths_and_init_state(..., warm_up_policy_id=None, evaluation_proposal_spec=None, penalty_coefficients_dir=None)` — `evaluation_proposal_spec` is a `build_proposal`-style dict or `None`. (`penalty_coefficients_dir` is added to the signature here but wired up in Task 2 — adding both now avoids touching the signature twice.)
- Consumes: existing `build_proposal`, `_evaluation_period_weights`, `ExperimentSpec`, `build_variation_test_env`, mutators.

- [ ] **Step 1: Write the failing tests**

Create `test/test_eval_proposal_generation.py`:

```python
import unittest

from param_generation.datasets import generate_test_paths_and_init_state
from param_generation.experiment_specs import ExperimentSpec, build_variation_test_env
from param_generation.mutators import (
    mutate_initial_state_congestion_05_const,
    mutate_mixture_geometric_proposal_lambda_0,
)

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

    def test_geometric_target_eval_proposal_weights_are_all_one(self):
        records = generate({'type': 'geometric', 'discount_factor_proposal': 0.99})
        for record in records:
            self.assertEqual(
                len(record['period_weights']), len(record['sample_path']) + 1
            )
            for weight in record['period_weights']:
                self.assertEqual(weight, 1.0)

    def test_mixture_eval_proposal_weights_bounded(self):
        records = generate(MIXTURE_SPEC)
        for record in records:
            for weight in record['period_weights']:
                self.assertGreaterEqual(weight, 1.0 - 1e-12)
                self.assertLessEqual(weight, 10.0 + 1e-9)

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


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest test/test_eval_proposal_generation.py -v`
Expected: 4 ERRORS with `TypeError: generate_test_paths_and_init_state() got an unexpected keyword argument 'evaluation_proposal_spec'`

- [ ] **Step 3: Implement the decoupling in `param_generation/datasets.py`**

3a. Change the signature (line 133) from:

```python
def generate_test_paths_and_init_state(test_envs, test_sample_path_num, warm_up_periods=0, num_periods=None, dat_file=None, num_groups=None, is_require_penalty_coefficients=True, is_random_initial_state=False, policy_ids=None, train_data_dir=None, warm_up_policy_id=None):
```

to:

```python
def generate_test_paths_and_init_state(test_envs, test_sample_path_num, warm_up_periods=0, num_periods=None, dat_file=None, num_groups=None, is_require_penalty_coefficients=True, is_random_initial_state=False, policy_ids=None, train_data_dir=None, warm_up_policy_id=None, evaluation_proposal_spec=None, penalty_coefficients_dir=None):
```

3b. Append to the docstring (after the `warm_up_policy_id` paragraph, before the closing `'''`):

```python
    ``evaluation_proposal_spec`` optionally decouples the EVALUATION-path
    proposal from the agent's own IS proposal: when given (a
    ``build_proposal``-style spec dict), the post-warm-up tails and
    ``period_weights`` are drawn from it while the embedded ``policy_specs``
    keep the untouched agent proposal. When ``None`` the tails fall back to
    the agent's spec (legacy coupled behavior).

    ``penalty_coefficients_dir`` optionally points the trained-coefficient
    lookup at another experiment's results folder (passed through as
    ``folder_path``), so a new experiment name can reuse coefficients without
    copying files or retraining.
```

3c. Replace the proposal build (lines 284-287):

```python
        sample_path_proposal = build_proposal(
            inner_agent_args.get('sample_path_length_proposal')
            or inner_agent_args.get('sample_path_proposal')
        )
```

with:

```python
        eval_proposal_spec = (
            evaluation_proposal_spec
            if evaluation_proposal_spec is not None
            else inner_agent_args.get('sample_path_length_proposal')
            or inner_agent_args.get('sample_path_proposal')
        )
        sample_path_proposal = build_proposal(eval_proposal_spec)
```

Also update the comment right above it: change "Draw the post-warm-up evaluation tails from the SAME importance-sampling proposal used for penalty-coefficient training." to "Draw the post-warm-up evaluation tails from ``evaluation_proposal_spec`` when given, else from the SAME importance-sampling proposal used for penalty-coefficient training." (leave the rest of the comment block unchanged).

`penalty_coefficients_dir` is accepted but not yet used — Task 2 wires it.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test/test_eval_proposal_generation.py -v`
Expected: 4 PASSED (allow ~1-2 minutes; each test builds toy envs and rolls the samplers).

- [ ] **Step 5: Regression check on existing proposal tests**

Run: `python3 -m pytest test/test_importance_sampling_proposals.py -v`
Expected: all PASSED.

- [ ] **Step 6: Commit**

```bash
git add param_generation/datasets.py test/test_eval_proposal_generation.py
git commit -m "feat: decouple evaluation-path proposal from agent IS proposal"
```

---

### Task 2: Wire `penalty_coefficients_dir` into coefficient loading

**Files:**
- Modify: `param_generation/datasets.py` (the `_load_trained_coefficients_from_folder` call, currently lines 204-208)
- Test: `test/test_eval_proposal_generation.py` (add one test class)

**Interfaces:**
- Consumes: `load_trained_coefficients_from_folder(experiment_name, mutate_val=None, sample_path_number=None, folder_path=None)` from `param_generation/caching.py` (imported in datasets.py as `_load_trained_coefficients_from_folder`); the `folder_path` override already exists.
- Produces: `generate_test_paths_and_init_state(..., penalty_coefficients_dir=...)` loads coefficients from that folder.

- [ ] **Step 1: Write the failing test**

Append to `test/test_eval_proposal_generation.py` (add `import json`, `import os`, `import tempfile`, and `from unittest import mock` to the imports at the top):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/test_eval_proposal_generation.py::TestPenaltyCoefficientsDir -v`
Expected: FAIL fast with `AssertionError: retraining triggered — coefficients were not loaded from penalty_coefficients_dir` (the loader ignores the override dir, misses, and hits the patched retraining fallback).

- [ ] **Step 3: Wire the parameter**

In `param_generation/datasets.py`, change (currently lines 204-208):

```python
            loaded_coefficients = _load_trained_coefficients_from_folder(
                experiment_name=experiment_name,
                mutate_val=mutate_val,
                sample_path_number=sample_path_number,
            )
```

to:

```python
            loaded_coefficients = _load_trained_coefficients_from_folder(
                experiment_name=experiment_name,
                mutate_val=mutate_val,
                sample_path_number=sample_path_number,
                folder_path=penalty_coefficients_dir,
            )
```

(`folder_path=None` keeps the legacy `experiments/results/<experiment_name>` search dir, so old callers are unaffected.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test/test_eval_proposal_generation.py -v`
Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add param_generation/datasets.py test/test_eval_proposal_generation.py
git commit -m "feat: allow loading penalty coefficients from an override folder"
```

---

### Task 3: Register the three evaluation-arm ExperimentSpecs

**Files:**
- Modify: `param_generation/registry.py` (append three specs to the `EXPERIMENT_SPECS` list, after the `mixture_probability_toy_study` entry)
- Test: `test/test_eval_proposal_generation.py` (add one test class)

**Interfaces:**
- Produces: `EXPERIMENT_SPECS['toy_eval_proposal_geometric_099' | 'toy_eval_proposal_fixed_459' | 'toy_eval_proposal_mixture_095_l01']` — each materializes exactly one variant whose `env_args`/`agent_args` equal the `mixture_probability_toy_study` val=0.1 variant.
- Consumes: existing mutators `mutate_initial_state_congestion_05_const`, `mutate_mixture_geometric_proposal_lambda_0` (already imported in registry.py).

- [ ] **Step 1: Write the failing test**

Append to `test/test_eval_proposal_generation.py` (add `from param_generation.registry import EXPERIMENT_SPECS` to the imports):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/test_eval_proposal_generation.py::TestEvaluationArmSpecs -v`
Expected: FAIL with `AssertionError: 'toy_eval_proposal_geometric_099' not found in ...`

- [ ] **Step 3: Add the three specs**

In `param_generation/registry.py`, inside the `EXPERIMENT_SPECS` list, immediately after the `mixture_probability_toy_study` entry (currently lines 181-187), add:

```python
        # Evaluation-proposal comparison arms (see
        # docs/superpowers/specs/2026-07-29-eval-proposal-comparison-design.md):
        # env and policy are byte-identical to mixture_probability_toy_study
        # val=0.1; only the name (= results directory) differs. The evaluation
        # proposal itself is supplied per-arm by the CLI recipe.
        ExperimentSpec(
            name='toy_eval_proposal_geometric_099',
            config_type='toy',
            val_args=[0.1],
            mutate=mutate_initial_state_congestion_05_const,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
        ExperimentSpec(
            name='toy_eval_proposal_fixed_459',
            config_type='toy',
            val_args=[0.1],
            mutate=mutate_initial_state_congestion_05_const,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
        ExperimentSpec(
            name='toy_eval_proposal_mixture_095_l01',
            config_type='toy',
            val_args=[0.1],
            mutate=mutate_initial_state_congestion_05_const,
            agent_mutate=mutate_mixture_geometric_proposal_lambda_0,
        ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test/test_eval_proposal_generation.py -v`
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add param_generation/registry.py test/test_eval_proposal_generation.py
git commit -m "feat: register the three evaluation-proposal comparison arms"
```

---

### Task 4: CLI recipe + `main()` switch (`python generate_params.py` end-to-end)

**Files:**
- Modify: `param_generation/cli.py` (imports; new recipe before `main()`; switch `main()`)
- Test: `test/test_eval_proposal_generation.py` (add one smoke-test class)

**Interfaces:**
- Consumes: `generate_test_paths_and_init_state(..., evaluation_proposal_spec=..., penalty_coefficients_dir=...)` (Tasks 1-2), the three `EXPERIMENT_SPECS` arms (Task 3), `load_trained_coefficients_from_folder` (`param_generation/caching.py`), `write_grouped_command_file` (`param_generation/command_files.py`).
- Produces: `recipe_toy_eval_proposal_comparison(test_sample_path_num=4096, num_groups=500, dat_file='table.dat', penalty_coefficients_dir='experiments/results/mixture_probability_toy_study')` returning the list of all records; `main()` calls it with defaults.

- [ ] **Step 1: Write the failing smoke test**

Append to `test/test_eval_proposal_generation.py` (add `from param_generation.cli import recipe_toy_eval_proposal_comparison` to the imports):

```python
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
            self.assertTrue(all(w == 1.0 for w in saved['period_weights']))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest test/test_eval_proposal_generation.py::TestEvalProposalComparisonRecipe -v`
Expected: ERROR with `ImportError: cannot import name 'recipe_toy_eval_proposal_comparison'`

- [ ] **Step 3: Implement the recipe**

In `param_generation/cli.py`:

3a. Extend the imports at the top:

```python
from scipy.stats import geom

from param_generation.caching import load_trained_coefficients_from_folder
from param_generation.command_files import write_grouped_command_file
```

(place `from scipy.stats import geom` with the third-party imports above the `param_generation` imports; the two `param_generation` imports join the existing block.)

3b. Add the recipe directly above `main()`:

```python
def recipe_toy_eval_proposal_comparison(
    test_sample_path_num=4096,
    num_groups=500,
    dat_file='table.dat',
    penalty_coefficients_dir=os.path.join(
        'experiments', 'results', 'mixture_probability_toy_study'
    ),
):
    """Evaluation-proposal comparison on the toy env (spec:
    docs/superpowers/specs/2026-07-29-eval-proposal-comparison-design.md).

    ONE fixed approx_penalized_hindsight policy (internal mixture IS proposal
    lambda_0=0.1, trained coefficients reused from
    ``mixture_probability_toy_study``) is evaluated on tails drawn from three
    different length proposals; each record carries the matching
    ``period_weights`` so ``run.py`` stays unbiased per arm. All
    3 x ``test_sample_path_num`` records go into ONE grouped dat file.
    """
    # Fail fast if the trained coefficients are missing: silently retraining
    # per arm would be slow and would break the fixed-policy comparison.
    coefficients = load_trained_coefficients_from_folder(
        experiment_name='mixture_probability_toy_study',
        mutate_val=0.1,
        sample_path_number=256,
        folder_path=penalty_coefficients_dir,
    )
    if coefficients is None:
        raise RuntimeError(
            'No trained penalty coefficients (mutate_val=0.1, '
            f'sample_path_number=256) found in {penalty_coefficients_dir}.'
        )

    # 459 = 0.99-quantile of Geom(1 - 0.99). The fixed proposal samples
    # max_length = 458 arrivals so the rollout visits exactly 459 weighted
    # decision periods (see the spec: FixedLengthProposal(459) would raise in
    # _evaluation_period_weights).
    fixed_horizon = int(geom.ppf(0.99, 1 - 0.99))
    assert fixed_horizon == 459, fixed_horizon

    arms = [
        ('toy_eval_proposal_geometric_099',
         {'type': 'geometric', 'discount_factor_proposal': 0.99}),
        ('toy_eval_proposal_fixed_459',
         {'type': 'fixed', 'max_length': fixed_horizon - 1}),
        ('toy_eval_proposal_mixture_095_l01',
         {'type': 'mixture_geometric', 'target_discount_factor': 0.99,
          'discount_factor_proposal': 0.95, 'lambda_0': 0.1}),
    ]
    all_records = []
    for experiment_name, evaluation_proposal_spec in arms:
        test_envs = build_variation_test_env(EXPERIMENT_SPECS[experiment_name])
        all_records.extend(
            generate_test_paths_and_init_state(
                test_envs=test_envs,
                test_sample_path_num=test_sample_path_num,
                warm_up_periods=0,
                num_periods=None,
                dat_file=None,
                is_require_penalty_coefficients=True,
                is_random_initial_state=False,
                policy_ids=['approx_penalized_hindsight'],
                evaluation_proposal_spec=evaluation_proposal_spec,
                penalty_coefficients_dir=penalty_coefficients_dir,
            )
        )
    write_grouped_command_file(
        results=all_records, num_groups=num_groups, dat_file=dat_file
    )
    return all_records
```

3c. Switch `main()`:

```python
def main():
    """Run the currently-active dataset-generation recipe."""
    recipe_toy_eval_proposal_comparison()
    #recipe_case_study_099_mixture_geometric_proposal_095_overtime_50_policy_evaluation()
    #recipe_case_study_099_mixture_geometric_proposal_095_policy_evaluation()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest test/test_eval_proposal_generation.py -v`
Expected: 7 PASSED.

- [ ] **Step 5: Full regression sweep of fast tests**

Run: `python3 -m pytest test/test_importance_sampling_proposals.py test/test_eval_proposal_generation.py -v`
Expected: all PASSED.

- [ ] **Step 6: Commit**

```bash
git add param_generation/cli.py test/test_eval_proposal_generation.py
git commit -m "feat: add evaluation-proposal comparison recipe as active generate_params entry"
```

---

### Task 5: Production smoke run of `python generate_params.py` (reduced scale) + final verification

**Files:**
- No source changes. Verification only.

**Interfaces:**
- Consumes: everything above, plus the real `experiments/results/mixture_probability_toy_study/local_2.jsonl` coefficients.

- [ ] **Step 1: Reduced-scale run against the REAL coefficients folder**

Run from the repo root:

```bash
python3 - <<'EOF'
from param_generation.cli import recipe_toy_eval_proposal_comparison
records = recipe_toy_eval_proposal_comparison(
    test_sample_path_num=2, num_groups=2, dat_file='/tmp/eval_proposal_smoke.dat'
)
lengths = {}
for r in records:
    lengths.setdefault(r['experiment_name'], []).append(len(r['sample_path']))
print('records:', len(records))
print('lengths by arm:', lengths)
print('coefficients loaded:', len(records[0]['generating_function_spec']['coefficients']))
EOF
```

Expected output: `records: 6`; `toy_eval_proposal_fixed_459` lengths `[458, 458]`; `coefficients loaded: 37` (the real trained vector from `local_2.jsonl`). If coefficients were missing this raises `RuntimeError` instead of silently training.

- [ ] **Step 2: Confirm the entry point wiring**

Run: `python3 -c "from generate_params import main; import param_generation.cli as cli; assert cli.main.__doc__; print('entry OK')"`
Expected: `entry OK` (do NOT run `python generate_params.py` at full scale here — 3 x 4096 paths is the user's production run).

- [ ] **Step 3: Clean up the smoke dat file**

```bash
rm -f /tmp/eval_proposal_smoke.dat
```

- [ ] **Step 4: Final commit check**

```bash
git status --short
git log --oneline -6
```

Expected: clean tree (only pre-existing untracked `.DS_Store`), four feature commits on `case_study`.

---

## Post-implementation notes for the user

- Production generation: `python generate_params.py` → one `table.dat` (500 groups, 12288 records). The per-arm results land in `experiments/results/toy_eval_proposal_{geometric_099,fixed_459,mixture_095_l01}/pickles` once `run.py` consumes the groups.
- The fixed arm's records have 458 arrivals / 459 weighted decision periods by design (see spec).
