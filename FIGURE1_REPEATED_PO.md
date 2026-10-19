# Figure 1 repeated PO experiment

This experiment uses **continuous decisions in every period**, as explicitly requested. Fitted, zero-penalty and ALP-based penalty IR costs use the same model, initial state and sample path. Its improvement bounds therefore concern the continuous relaxation, not Figure 1's integer-first-period policy formulation.

The twelve displayed settings give ten unique environment configurations. Each configuration has thirty independently seeded training batches of 512 paths. Each batch uses 256 equiprobable geometric strata with two paths per stratum and retains the proposal's actual weights. Coefficients are unrestricted; there is no ALP value-function floor, regularization or coefficient box. Matching ALP coefficients are read from existing certified caches, with their source hash recorded.

## Prepare commands

From the project root:

```bash
python generate_params.py train figure1 \
  --alp-cache-root /Users/yanboding/Desktop/PhdTopic/MCMADASP/experiments/results \
  --dat /tmp/figure1_repeated_po_20261018/table.dat \
  --manifest /tmp/figure1_repeated_po_20261018/manifest.json \
  --num-path-seeds 30 --sample-path-number 512 --sample-paths-seed 420000 \
  --max-iterations 1500 --absolute-gap-tolerance 1e-6 --workers 32
```

This writes 300 ordinary `python run.py --params ...` training records to `table.dat` and records all parameters and setting aliases in `manifest.json`. `figure1` is an experiment group under the existing `train` command. Generation uses `generate_penalty_coefficient_training_env`; each record has `init_state_mode="shared"` and executes the existing `run.train_penalty_coefficients_for_env → ApproxQAgent.benders_decomposition_train → BendersDecompositionSolver.solve` chain. No separate Figure 1 training driver or optimization loop exists.

The group defaults to 30 repetitions, 512 paths, base seed 420000, 32 workers, 1500 iterations and absolute gap tolerance 1e-6. The normal `train` command retains its existing defaults for other experiments. `--dat` defaults to `table.dat`; `--manifest` defaults to `<dat-stem>.manifest.json`. `--workers` caps the shared subproblem environment pool within each job. Command generation does not initialize a solver or run training. Each command uses a different existing sample-path seed (420000–420299 by default); its normal UID separates the results and checkpoints. The fixed initial state and ALP fit remain unchanged. Manifest metadata is not passed to the training function. Existing command bundles are not overwritten.

## Training and certificates

For each batch the fixed ALP policy supplies the gradient and the upper cap evaluated at the fixed reference coefficients. The existing Benders method maximizes the weighted sum of the IR value minus the policy-based control variate, preserving its minimum-norm tie breaking and continuous-decision training. Fixed-policy training uses the policy cost cap without an additional artificial scenario cap, regardless of whether path results are saved. The solver snapshots the finite **primary MAX master bound before any minimum-norm auxiliary solve**. A master with coefficients temporarily fixed to the initial vector is excluded from global certificates.

Each run writes one compact record to `experiments/results/<experiment_name>/<job_id>.jsonl`. With `save_path_certificates` enabled, it keeps the UID, experiment/setting, path count, fitted coefficients, elapsed time, raw and control-variate objective values, certified candidate lower and master upper bounds, master status and termination reason. Each path keeps its ID, stratum, weight, termination time, the fitted/ALP/zero costs as `objective` and `status` only, and both raw paired differences. The policy gradient appears only through `policy_control_variate = g·(theta_hat-anchor)` and `policy_penalty_at_reference = g·anchor`, which suffice for the existing local statistics. Configuration is joined from the command manifest by UID. Duplicate summaries, hashes and per-path runtimes are omitted.

After training, `train_penalty_coefficients_for_env` copies the final fitted certificates, evaluates ALP and zero using the same models and paths, and saves the raw differences before disposing the models. `save_path_certificates` controls these extra evaluations and outputs, not the master cap or objective. Interrupted subproblem solves still fail the batch. The solver retains full certificates internally; local analysis validates the saved bounds and statuses and can read both existing and compact results.

The existing `calculate_information_relaxation_cost` builds and disposes a model per call. Training diagnostics instead reuse the trained workers through `evaluate_action_certificates`, preserving their bases and parallel environment pool; both use `pathwise_terms` for the model formulation. For the absorption ALP penalty with continuous decisions, the restored raw worker values agree with direct evaluation on the same initial states, arrivals, survival weights and terminal outcomes. Since these workers require `OPTIMAL`, the saved objective is also the pathwise MIN lower bound. Analysis reconstructs that bound internally and rejects a nonoptimal result with no bound; older outputs containing explicit valid bounds remain supported.

Each path also saves `cv_adjusted_fitted_ir_cost = fitted - policy_control_variate`, `cv_adjusted_gap_fitted_minus_alp = fitted - alp - policy_control_variate`, and `cv_adjusted_gap_fitted_minus_zero = fitted - zero - policy_control_variate - policy_penalty_at_reference`. The zero comparison adjusts both sides to the same training objective. These fields are calculated from the existing values without additional solves; missing required incumbents produce `null`. Their weighted gap means describe the fitted candidate, while potential-improvement upper estimates continue to use the certified master upper bound and baseline MIN lower bounds. They coincide when the solves have converged. Older outputs without these fields remain readable.

Keep `benders_checkpoints/`: the policy `.npz` retains costs, full gradients and reference coefficients, while the cut log and checkpoint metadata support resuming training without starting over. These caches save computation and are not replaced by the compact result. Local repeated-SAA CI and optional variance-reduction analysis need only the result records and manifest, with no new PO or IR solves.

The coefficient fit is the unrestricted SAA solution up to its recorded solver gap. It is not automatically the population-optimal penalty. Thirty runs must use identical settings and algorithm limits; failed or missing runs cannot simply be dropped from the statistical sample.

## Server execution

For this local revision, do not commit, push, sync, or submit jobs. The steps below describe the workflow for a later explicitly requested deployment.

Follow `NIBI_EXPERIMENT_WORKFLOW.md`: Git for code; transfer the command bundle separately; preserve old outputs before clearing inactive farm state. Install the verified 300-command file as the project's `table.dat`.

Both resource scripts are configured with:

```bash
python experiments/configure_farm.py --project-dir . --profile po --jobs 128 --other-tokens 0
```

Use the actual verified other-job token count, not the example zero. PO uses 32 tasks and 128G per worker; policy evaluation uses 16 tasks and 64G via `--profile evaluation --tokens-per-job <verified-count>`. One Python process uses at most 32 subproblem environments plus one main environment: conservatively **33 tokens per PO worker**. At 128 workers this is 4,224 tokens before other jobs. Total account use must remain strictly below 5,000, including queued/automatic reruns. Do not start another overlapping farm against the same budget. `submit.run 128 -auto` submits workers, not 128 distinct cases; they consume the 300 commands sequentially.

## Local statistics after downloading results

The independent statistical units are the **30 complete batches**. Raw training-path differences are saved for diagnosis and must not be pooled as 15,360 independent fitted observations.

For each batch, the ALP comparison is the certified master upper bound minus the weighted ALP IR baseline. The zero comparison subtracts the weighted sum of the zero IR cost **plus** the policy gradient dotted with the ALP coefficients. This sign accounts for the reference constant in the same control-variate objective. When a baseline solve is interrupted, subtract its valid MIN lower bound instead of its incumbent. These batch quantities bound potential improvement **in expectation**, not for every realized sample.

Download the ordinary training JSONL files into a local results folder. Analysis matches them to the manifest by UID, then checks the decision set, seeds, certificates and weights:

```bash
python -m experiments.analyze_repeated_po baseline \
  --manifest /path/to/manifest.json \
  --results-root /path/to/downloaded/training_results \
  --output /path/to/baseline_report.json
```

This reports a Student-t interval for the expectation of the batch upper estimate and a one-sided 95% upper confidence limit on potential improvement. Optionally supply a **predeclared** `--half-width-threshold`; this reports whether the threshold was met and never starts CV automatically.

If requested after inspecting the baseline, run the already implemented post-processing:

```bash
python -m experiments.analyze_repeated_po cv \
  --manifest /path/to/manifest.json \
  --results-root /path/to/downloaded/training_results \
  --output /path/to/cv_report.json --bootstrap-count 5000 --seed 20261018
```

CV uses the batch's weighted termination time minus its analytic proposal mean. Here `tau = arrival_length + 1`; the implementation includes this shift and the sampler's numeric quantile clipping. Leave-one-batch-out regression uses an intercept to estimate a slope, but subtracts only slope times the known-zero control. Numerically constant training-fold controls get slope zero. A complete-batch bootstrap, paired across benchmarks, refits all leave-one-out slopes each time and reports basic intervals. It does not rerun PO or IR solves.

Both procedures provide approximate statistical coverage, not a finite-sample guarantee. The two-sided interval concerns the expected upper estimator, not improvement itself. Keep both reports; a CV comparison triggered by baseline width is exploratory, and its interval is not guaranteed to be narrower. No experiment results or generated command bundles should be committed to Git.
