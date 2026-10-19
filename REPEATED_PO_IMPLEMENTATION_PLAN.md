# Figure 1 Repeated PO Implementation Plan

> Use Superpowers subagent-driven-development, test-driven-development, requesting-code-review, and verification-before-completion. The attached instruction and the user's subsequent corrections are the approved requirements.

**Goal:** Generate 30 independent M=512 PO runs for each unique Figure 1 configuration, save auditable solver certificates and paired paths, provide local baseline/CV analysis, review and deploy the implementation, then submit the requested Meta Farm experiment.

**Architecture:** Reuse the existing environment, affine penalty, proposal and training-model builders. A dedicated auditable cutting-plane driver preserves continuous first-period PO training and captures valid optimization bounds without changing the legacy Benders policy-evaluation algorithm. A separate solver-free local analysis module consumes versioned batch results. A reproducible command generator supplies complete JSON parameters via the existing Meta Farm params-file conversion.

**Tech Stack:** Python 3.11+, NumPy/SciPy, Gurobi 12 on Nibi, pytest, Git, Meta Farm.

## Global constraints

- User correction: PO training keeps its existing continuous first-period formulation. Do not change the legacy model to integer first-period training.
- User confirmed: all three comparison costs use the same all-period continuous decision set. The improvement bounds refer to this continuous relaxation.
- Twelve display settings; deduplicate full model configurations (expected ten). Thirty independent repetitions each; M=512.
- Fixed matching ALP coefficients and nonanticipative ALP policy; unrestricted coefficients, no floor, box, regularization or fixed intercept.
- Preserve Figure 1 arrivals and geometric stratification. At M=512, use 256 equiprobable strata with two paths each and actual proposal weights. Save arrival length and decision-period tau=L+1 separately.
- Full centered objective and valid MAX master bound; use MIN lower certificates for conservative baseline subtraction. Same initial state, sample paths and feasible set across comparisons.
- Local baseline/CV only read results; no solver calls. Thirty batch replicates, never 15360 independent training gaps. Manual CV with analytic known-zero length control, LOO slopes and complete-batch basic bootstrap.
- User now authorizes code review/fixes, Git upload, Nibi synchronization, latest table.dat installation and job submission. No formal local experiment runs. Synthetic/unit tests and a tiny resource-bounded server validation may be used before bulk submission.
- PO scripts: ntasks-per-node=32, mem=128G. Policy-evaluation scripts: ntasks-per-node=16, mem=64G. Remove conflicting old cpus-per-task=16. Respect strict account total <5000 Gurobi tokens, including master environments and other jobs.
- No results in Git. Preserve existing server modifications and needed outputs before four-path farm cleanup. Follow NIBI_EXPERIMENT_WORKFLOW.md.

## Shared interfaces

Commands invoke `python -m experiments.figure1_po_run --params '<JSON>'`; CLI also accepts `--params_file` and `--job_id` for single_case.sh compatibility.

Parameter JSON fields: schema_version=1, config_id, batch_id, repetition (1..30), configuration (full unseeded env_args), initial_state, decision_set={first,future}, theta_domain={coefficient_bound:null,alp_value_floor:false,regularization:0,fixed_coefficients:{}}, seeds={env,arrival,stop_time,initial_state}, reference={coefficients,policy:'alp',source,source_sha256,source_uid}, sampling={sample_path_number:512,proposal:{type:'stratified_geometric',discount_factor_proposal:0.99,num_strata:256},tau_definition:'arrival_length_plus_one',weight_convention:'stratum_mass_over_count',quantile_clip:[1e-12,0.999999999999]}, solver={max_iterations,absolute_gap_tolerance,subproblem_time_limit_seconds,master_time_limit_seconds,workers:32}, output_dir.

Each run writes `batch.json` atomically under its unique output_dir. Copy the identifying/configuration/reference/sampling fields above into the result. Add training={coefficients,full_objective,objective_lower_bound,objective_upper_bound,master_certificate:{sense:'maximize',status,objective,objective_bound,valid,reason},termination_reason,absolute_gap,relative_gap,runtime_seconds}. Runtime metadata includes Git commit, job_id and observed package versions.

Each paths element: path_id, path_hash, initial_state_hash, stratum, weight, arrival_length, termination_time (=arrival_length+1), terminal, policy_gradient, ir={fitted,zero,alp}, raw_gap_fitted_minus_zero, raw_gap_fitted_minus_alp. Each IR result: objective (incumbent or null), lower_bound (finite valid MIN certificate or null), upper_bound (incumbent or null), status, solution_count, exact, bound_valid, runtime_seconds. Raw incumbent differences must be labeled if nonoptimal; certificates are used independently for inference.

Manifest fields: schema_version=1, display_settings (12 entries with label/config_id), configurations (10 entries), runs (300 entries containing full params, command, case_id, batch_id, repetition, config_id, seeds, output_dir), expected_batches=30, expected_paths=512.

## Task 1 — Generator (agent owned)

Files: param_generation/generate_figure1_po.py, test/test_generate_figure1_po.py.

- [ ] Test 12→10 configuration dedup, correct ALP UID lookup, malformed/missing cache rejection, 300 commands with unique seeds/output directories, actual CLI compatibility, complete manifests and preserved model/proposal/weights.
- [ ] Build from existing toy_lb_* registry and explicit ALP cache root. Read-only generation; never initialize licenses or solve models.
- [ ] CLI configurable server project/results paths and output bundle. Write table.dat and manifest.json. Do not overwrite existing files silently.
- [ ] Verify generation using existing local ALP caches; generated outputs stay out of Git.

## Task 2 — Certified PO runner (agent owned)

Files: experiments/figure1_po_run.py, experiments/po_certificates.py, test/test_po_certificates.py; narrow training-builder extension only if required.

- [ ] Failing fake-model tests: MAX ObjBound rather than ObjVal; valid affine incumbent cuts; centered constant once; MIN lower certificate on interruptions; complete same-path records; no artificial coefficient/scenario cap; CLI parsing without solver initialization.
- [ ] Use existing all-continuous train_subproblem_builder_fn, fixed ALP policy gradients, seed ALP and zero candidates, valid policy upper caps and upper cuts. Never add a lower objective restriction from a MIN incumbent.
- [ ] Dedicated loop records best finite primary master ObjBound and best certified candidate LB. Capture bounds before any other objective solve; no minimum-norm objective.
- [ ] Evaluate ALP, zero and fitted values on the same built models. Save all certificate/state/path identifiers, status, gap and runtimes atomically; retain invalid/incomplete outcomes as failures rather than reporting a valid bound.
- [ ] Bound environment count to one main plus at most 32 subproblem environments and dispose resources. Do not share an environment across simultaneous optimizations.

## Task 3 — Local inference (parent owned)

Files: experiments/analyze_repeated_po.py, experiments/repeated_po_schema.py, test/test_analyze_repeated_po.py.

- [ ] Synthetic failing tests with manually calculable R_ALP and R_zero, conservative MIN baseline substitution, exact retained weights, invalid certificates and missing/duplicate batches, metadata/seed/decision mismatches.
- [ ] Baseline: batch Student-t intervals, one-sided upper confidence limit, optional predeclared absolute half-width check.
- [ ] Analytic length mean from saved law including tau shift and numeric quantile clipping. Reject unknown/nonmatching weighted distributions.
- [ ] CV: training-fold-only one-variable LOO regression with intercept, subtract slope*H only; stable zero-variance fallback; bootstrap whole positions, shared indices across benchmarks, rerun LOO each replicate, basic intervals.
- [ ] Report baseline and CV side by side, CI target/approximation caveat, diagnostics, no retrospective cherry-picking. No solver or remote imports.

## Task 4 — Resources, review and delivery

Files: job_script.sh, resubmit_script.sh, experiments/configure_farm.py, test/test_configure_farm.py, FIGURE1_REPEATED_PO.md.

- [ ] Explicit PO/evaluation resource configurator, preserve requested ntasks/mem, one process per worker using correct internal CPU budget. Remove shared dependency installation on worker startup and avoid shared Gurobi config writes.
- [ ] Review each task and whole change set; fix important findings with regression tests. Run new tests plus relevant solver-free existing tests. Syntax-check submit scripts.
- [ ] Apply verified files to requested local project, reconcile Git branches without losing edits, commit code/instructions only and push.
- [ ] Login Nibi (Pixel 8 root MFA if needed), preserve server edits/results, sync exact commit and command bundle, validate env/table/resource/token limits. Run a tiny checked test only if needed; then clean inactive old farm paths and submit latest 300-case table with safe concurrency.
- [ ] Report commit, commands/jobs/token budget and job IDs. CI analysis waits for downloaded complete results; no automatic follow-up unless requested.
