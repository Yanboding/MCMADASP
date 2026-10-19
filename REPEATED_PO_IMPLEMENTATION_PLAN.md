# Repeated PO experiment: reuse the existing training workflow

The user's latest correction supersedes the earlier dedicated-driver design. Use Superpowers TDD, independent review and verification. Do not add a separate PO optimizer or `figure1-repeated-po` command.

## Required execution chain

`generate_params.py train figure1` → existing `_run_train` → `generate_penalty_coefficient_training_env` → ordinary `run.py` record with `init_state_mode="shared"` → `train_penalty_coefficients_for_env` → `ApproxQAgent.benders_decomposition_train` → `BendersDecompositionSolver.solve`.

`figure1` is only a group of existing experiment configurations. Its helper loads configurations/certified ALP caches and records a manifest; it does not solve models. Remove the independent driver and old `run_type` dispatch. Preserve the existing Benders algorithm and minimum-norm tie breaking.

## Experiment contract

- Twelve displayed settings deduplicate to ten complete configurations.
- Thirty independent batches per configuration; M=512, 256 geometric strata, two paths per stratum.
- Fitted, zero and ALP penalties all use continuous decisions in every period, identical states, arrivals and weights.
- Fixed configuration-matched ALP coefficients and nonanticipative ALP policy.
- Unrestricted coefficient domain: no ALP floor, coefficient box, fixed intercept or regularization.
- Seed block for case i (zero based): 420000+4i through 420003+4i for environment, arrivals, stopping and initial state.
- 1500 maximum Benders iterations; absolute tolerance 1e-6; at most 32 subproblem environments plus one main environment.

## Minimal extensions

1. Forward optional iteration, tolerance, time-limit and audit settings through the existing training function. Keep ordinary training defaults unchanged.
2. Record primary MAX bounds before minimum-norm auxiliary solves. A temporarily pinned master cannot certify the unrestricted optimum. Do not use an artificial scenario cap for audited training; retain the valid fixed-policy cap.
3. Reuse existing subproblem models and final fitted evaluation; record fitted/ALP/zero MIN certificates and paired path differences. Preserve optimal-only solver behavior and fail explicitly on interrupted solves.
4. Add optional `batch_metadata` and a serialization helper to write `batch.json` alongside ordinary training results/checkpoints. Check metadata against actual execution settings.
5. Preserve the solver-free local inference code: 30 independent batch units, certified master upper bounds, correct ALP/zero reference constants, repeated-SAA intervals and optional batch-level CV. Never pool 30×512 fitted paths as independent observations.

## Validation and deployment

- Test normal train routing, generation without license checkout, complete configuration/cache matching, independent seeds, metadata, certificate signs and constants, min-norm/pinned-master safeguards, and existing Benders behavior.
- Use only a tiny bounded real-solver smoke test locally; do not run the formal M512 experiments locally.
- Obtain independent code review; fix findings and verify before committing.
- Git carries code/docs only. Generated commands, manifests, results and farm state stay out of Git.
- Apply changes to `/Users/yanboding/Desktop/PhdTopic/MCMADASP`, push the reviewed experiment branch, and sync the exact commit plus a regenerated `table.dat` to Nibi while its farm is inactive. Preserve server modifications and prior outputs.
- The user canceled all previous jobs with `kill.run`. **Do not restart jobs as part of this correction.** Leave the synchronized batch unsubmitted. Before any later submission, follow `NIBI_EXPERIMENT_WORKFLOW.md`, verify token budget and preserve then clear the four inactive farm entries.
