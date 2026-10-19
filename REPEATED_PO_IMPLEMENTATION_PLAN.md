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
- Use the existing sample-path seed input: case i (zero based) uses 420000+i. The fixed initial state is unchanged; no new seed mapping or batch input is required.
- 1500 maximum Benders iterations; absolute tolerance 1e-6; at most 32 subproblem environments plus one main environment.

## Minimal extensions

1. Forward optional iteration, tolerance and time-limit settings through the existing training function. Handle path-output settings in `train_penalty_coefficients_for_env`, outside the coefficient-fitting method.
2. Record primary MAX bounds before minimum-norm auxiliary solves. A temporarily pinned master cannot certify the unrestricted optimum. Fixed-policy runs use the valid policy cap regardless of path-output settings; unrelated runs retain their historical scenario cap.
3. After training, `train_penalty_coefficients_for_env` reuses the final fitted evaluation and existing subproblem models to calculate ALP/zero MIN certificates and paired path differences before disposing the models. Preserve optimal-only solver behavior and fail explicitly on interrupted solves.
4. Restore the original training function inputs and result paths. Save compact path costs/bounds/statuses and master bounds in the existing JSONL record. Replace duplicated full path gradients with the two scalar products needed by local statistics; retain full gradients in the policy cache. Preserve cut logs and checkpoint metadata for resumption. Do not add an audit wrapper, Git lookup, runtime metadata or parameter fingerprint. Keep batch metadata only in the external manifest and join results by UID. Reuse the existing fixed-policy anchor for the ALP reference.
5. Preserve the solver-free local inference code: 30 independent batch units, certified master upper bounds, correct ALP/zero reference constants, repeated-SAA intervals and optional batch-level CV. Never pool 30×512 fitted paths as independent observations.

## Validation and deployment

- Test normal train routing, generation without license checkout, complete configuration/cache matching, independent seeds, metadata, certificate signs and constants, min-norm/pinned-master safeguards, and existing Benders behavior.
- Use only a tiny bounded real-solver smoke test locally; do not run the formal M512 experiments locally.
- Obtain independent code review; fix findings and verify. Leave all changes uncommitted.
- Git carries code/docs only. Generated commands, manifests, results and farm state stay out of Git.
- Apply reviewed changes locally to `/Users/yanboding/Desktop/PhdTopic/MCMADASP`. **Do not commit, push, sync to the server, or submit jobs for this task.**
