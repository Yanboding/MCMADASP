# Figure 1 repeated PO experiment

This experiment uses **continuous decisions in every period**, as explicitly requested. Fitted, zero-penalty and ALP-based penalty IR costs use the same model, initial state and sample path. Its improvement bounds therefore concern the continuous relaxation, not Figure 1's integer-first-period policy formulation.

The twelve displayed settings give ten unique environment configurations. Each configuration has thirty independently seeded training batches of 512 paths. Each batch uses 256 equiprobable geometric strata with two paths per stratum and retains the proposal's actual weights. Coefficients are unrestricted; there is no ALP value-function floor, regularization or coefficient box. Matching ALP coefficients are read from existing certified caches, with their source hash recorded.

## Prepare commands

From the project root:

```bash
python -m param_generation.generate_figure1_po \
  --alp-cache-root /Users/yanboding/Desktop/PhdTopic/MCMADASP/experiments/results \
  --server-project /home/dingyanb/projects/def-sarhangi/dingyanb/MCMADASP \
  --server-results /home/dingyanb/projects/def-sarhangi/dingyanb/MCMADASP/experiments/results/figure1_repeated_po_20261018 \
  --output-bundle /tmp/figure1_repeated_po_20261018 \
  --max-iterations 1500 --absolute-gap-tolerance 1e-6 --workers 32
```

This writes 300 commands to `table.dat` and records all parameters and setting aliases in `manifest.json`. It does not initialize a solver or run training. The runner accepts `--params`, `--params_file` and `--job_id`, including Meta Farm's existing params-file conversion. Existing command bundles and batch results are not overwritten silently.

## Training and certificates

For each batch the fixed ALP policy supplies the gradient and the upper cap evaluated at the fixed reference coefficients. The driver maximizes the weighted sum of the IR value minus the policy-based control variate. It reuses the existing continuous subproblem builders and adds affine upper cuts from feasible subproblem solutions. It saves the best finite **primary MAX master bound**, with all objective constants included. A MIN incumbent is not treated as a lower certificate.

Each run writes `batch.json`, including coefficients, primary master certificate, candidate lower bound, solver gap, timing, version and seeds. Each path has its stratum/weight, arrival length, termination time, hashes, policy gradient, the fitted/zero/ALP raw IR costs and MIN certificates, and both paired raw differences. Saved fitted cost and bound snapshots refer to the selected coefficient vector. Interrupted MIN solves use their finite valid lower certificates for conservative baseline subtraction; missing certificates fail validation. No master or subproblem solves are required for later local analysis.

The coefficient fit is the unrestricted SAA solution up to its recorded solver gap. It is not automatically the population-optimal penalty. Thirty runs must use identical settings and algorithm limits; failed or missing runs cannot simply be dropped from the statistical sample.

## Server execution

Follow `NIBI_EXPERIMENT_WORKFLOW.md`: Git for code; transfer the command bundle separately; preserve old outputs before clearing inactive farm state. Install the verified 300-command file as the project's `table.dat`.

Both resource scripts are configured with:

```bash
python experiments/configure_farm.py --project-dir . --profile po --jobs 128 --other-tokens 0
```

Use the actual verified other-job token count, not the example zero. PO uses 32 tasks and 128G per worker; policy evaluation uses 16 tasks and 64G via `--profile evaluation --tokens-per-job <verified-count>`. One Python process uses at most 32 subproblem environments plus one main environment: conservatively **33 tokens per PO worker**. At 128 workers this is 4,224 tokens before other jobs. Total account use must remain strictly below 5,000, including queued/automatic reruns. Do not start another overlapping farm against the same budget. `submit.run 128 -auto` submits workers, not 128 distinct cases; they consume the 300 commands sequentially.

## Local statistics after downloading results

The independent statistical units are the **30 complete batches**. Raw training-path differences are saved for diagnosis and must not be pooled as 15,360 independent fitted observations.

For each batch, the ALP comparison is the certified master upper bound minus the weighted ALP IR baseline. The zero comparison subtracts the weighted sum of the zero IR cost **plus** the policy gradient dotted with the ALP coefficients. This sign accounts for the reference constant in the same control-variate objective. When a baseline solve is interrupted, subtract its valid MIN lower bound instead of its incumbent. These batch quantities bound potential improvement **in expectation**, not for every realized sample.

Run only after all downloaded batch files pass manifest, decision-set, seed, solver-certificate and weight checks:

```bash
python -m experiments.analyze_repeated_po baseline \
  --manifest /path/to/manifest.json \
  --results-root /path/to/downloaded/figure1_repeated_po_20261018 \
  --output /path/to/baseline_report.json
```

This reports a Student-t interval for the expectation of the batch upper estimate and a one-sided 95% upper confidence limit on potential improvement. Optionally supply a **predeclared** `--half-width-threshold`; this reports whether the threshold was met and never starts CV automatically.

If requested after inspecting the baseline, run the already implemented post-processing:

```bash
python -m experiments.analyze_repeated_po cv \
  --manifest /path/to/manifest.json \
  --results-root /path/to/downloaded/figure1_repeated_po_20261018 \
  --output /path/to/cv_report.json --bootstrap-count 5000 --seed 20261018
```

CV uses the batch's weighted termination time minus its analytic proposal mean. Here `tau = arrival_length + 1`; the implementation includes this shift and the sampler's numeric quantile clipping. Leave-one-batch-out regression uses an intercept to estimate a slope, but subtracts only slope times the known-zero control. Numerically constant training-fold controls get slope zero. A complete-batch bootstrap, paired across benchmarks, refits all leave-one-out slopes each time and reports basic intervals. It does not rerun PO or IR solves.

Both procedures provide approximate statistical coverage, not a finite-sample guarantee. The two-sided interval concerns the expected upper estimator, not improvement itself. Keep both reports; a CV comparison triggered by baseline width is exploratory, and its interval is not guaranteed to be narrower. No experiment results or generated command bundles should be committed to Git.
