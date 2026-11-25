# Roadmap

## Phase 2 — Research Organization, Training Research, and Optimized Cascades

## PR 43 — Training Research Notebook and Stage 4 Policy Research

**Objective:** Establish the train-only Training Research Notebook, register a whole-system abstention policy, and evaluate Stage 4 approaches on the frozen PR 39 training-side out-of-fold (OOF) evidence. Give the GLASS Linear Arbiter priority, diagnose input shift for retained candidates, and provide recommendations for PR 44 optimization and PR 45 adoption.

**Execution principles:**

- Use training rows only. Preserve the PR 39 artifacts, manifests, accepted configurations, and registered 2×2 crossover. Write research evidence under distinct PR 43 run identities.
- Apply the PR 42 claim boundary throughout. The frozen OOF evidence supports random-split comparisons. Target-derived features use later contacts’ labels, and the MLP early-stopping split shares encoder fitting. Forward-in-time claims require the temporal evaluation in PR 49.
- Register objectives, decision rules, budgets, and conditional experiments before inspecting candidate results. Use identical evaluation rows, folds, coverage obligations, and uncertainty methods for paired comparisons. Record each arm’s search space and budget.
- Reuse existing implementations by import. Justify substantial additions before implementation by identifying the missing capability and the alternatives inspected. Reserve repository-wide consolidation for after PR 46.
- Screen candidates on frozen inputs first. Use full-batch optimization for the small convex candidate. Reserve upstream fitting for the registered input-shift experiment.
- Run targeted tests and applicable research acceptance checks after each implementation step. Reuse accepted execution records and intermediate results where their recorded identities remain valid. Reserve the full default, slow, and comprehensive real-data regression suites, final clean-kernel notebook execution, and complete frozen-baseline verification for Commit 6. Repeat expensive experiments only when a material change invalidates their accepted evidence.
- Review the implementation footprint alongside scientific correctness. Avoid duplicating metrics, diagnostics, research summaries, plotting functionality, and test coverage. Store sufficient evidence to reproduce and defend each finding without unnecessarily duplicating large intermediate results.

**Commit order:** Commit 1 records the plan. Commit 2 establishes the training boundary and registers the evaluation protocol. Commit 3 diagnoses the voters under that protocol. Commit 4 evaluates the priority GLASS Linear Arbiter, targeted GLASS formula changes, and Meta-XGBoost on the same evidence. Commit 5 uses those results to limit upstream refitting to the candidates and questions that require it. Commit 6 consolidates recommendations, completes validation, and publishes the accepted executed notebooks.

### Commit 1 — Roadmap Planning and Deferred Engineering Updates

**Objective:** Record the PR 43 execution plan and reconcile Deferred Engineering with the completed PR 42 handoff. Documentation only.

- [x] Replace the existing PR 43 section with the agreed commit sequence. Keep its deliverable immediately before PR 44.
- [x] Add the PR 44 acceptance requirement for measuring Stage 1 and Stage 3 hyperparameter-selection optimism before adoption.
- [x] Review Deferred Engineering dispositions. Keep the resolved Windows execution and PuLP 3.3.2 items closed. Retain the separate PuLP 4.x migration, low-priority Loky shutdown issue, Optuna widget rendering, portability work, and optional development utility in their existing queues.
- [x] Keep the seven SAG/SAGA convergence warnings visible for PR 44 optimization or resolution before PR 45 regeneration.
- [x] Record the repository consolidation and documentation cleanup planned after PR 46 and before publicly announcing Phase 2 completion. Preserve scientific evidence and provenance during that cleanup.
- [x] Make no source-code, notebook, or artifact changes.

### Commit 2 — Train-Only Notebook, Policy Registration, and Nested Stage 4 Validation

**Objective:** Establish the training inputs, scoring policy, and evaluation procedure before examining Stage 4 candidate results.

#### Step 2A — Notebook Foundation and Frozen Input Boundary

- [x] Build the Training Research Notebook from `documentation/notebook_map.md` section 2.2. Inventory and reuse the existing training, calibration, cross-fitting, threshold, evaluation, statistical, plotting, and research-record utilities.
- [x] Reuse `feature_research.training_side` and extend notebook tests G5 and G7 to cover the labelled execution path and train-only boundary.
- [x] Assemble both upstream OOF tables from the PR 39 pins. Verify row order, labels, producer fold identities, route states, probability semantics, and fingerprints. Preserve the distinct Stage 1 and Stage 2/3 fold designs.
- [x] Reuse the PR 38 router, LR, and EBM boundary extractors. Supply the smallest verified train-only access needed for the MLP, Stage 3 XGBoost, and missing router evidence. Check the notebook entry point for test-field access because the production Stage 4 loaders and `build_feature_frames` construct both sides of the split.
- [x] Record the accepted configurations and threshold procedures. Reproduce the production abstention search over distinct confidence values when checking the historical reference. The research helper searches a fixed grid.

#### Step 2B — Stage 4 Objective, Abstention Policy, and Comparison Registration

- [x] Define what abstention means operationally and how abstained rows contribute to whole-system scoring. Register the primary loss, coverage grid, selective-risk definition, AURC convention, tie handling, class-retention safeguards, and rule for selecting a deployable cut.
- [x] Evaluate scoring functions at matched coverage. Separately measure the coverage achieved on outer-fold rows by an inner-selected cut. Report decisions before abstention, retained-row results, and whole-system outcomes, including effective recall, per-class retention, retained prevalence, and correct and incorrect decisions deferred.
- [x] Keep retained F2 as a descriptive historical metric. It has no true-negative term, and the selected cut reached the 50% training coverage floor in all four accepted crossover cells. Use the registered criteria for weight, threshold, and policy selection.
- [x] Register the GLASS Linear Arbiter comparisons against the GLASS Arbiter and Meta-XGBoost on each upstream, separately from the historical PR 39 2×2.
- [x] Predefine the paired uncertainty method, meaningful improvement and comparability margins, calibration and class-retention limits, interpretability requirements, and computational budget. Favor the simpler interpretable approach when performance falls within the registered comparability margin.
- [x] Record each arm’s search space, trial budget, stopping criteria, diagnostic triggers for optional variants, and decision rule for expanding the simulated holdout.

#### Step 2C — Nested Stage 4 Validation on Fixed Upstream Inputs

- [x] Use common outer folds for evaluation and inner folds for selection. Fit each Stage 4 procedure from scratch, including learned voter cuts, weights or coefficients, calibration, class thresholds, and abstention cuts. Keep raw scoring, calibration, class decisions, and deferral separate.
- [x] Refit the accepted GLASS Arbiter procedure per outer fold. Reproduce its historical operating rule as a reference and evaluate its scoring function under the common registered policy. Apply the same outer evaluation and selection safeguards to Meta-XGBoost.
- [x] Measure Stage 4 selection optimism from inner-selected and outer-evaluated results. Document that upstream OOF inputs remain fixed and that upstream fits producing one row’s inputs can include other Stage 4 outer-fold labels. Fully nested Stage 1/3 hyperparameter selection belongs to PR 44.
- [x] Verify Stage 4 fold isolation, input parity, and reference behavior with targeted checks. Record run IDs, seeds, fold identities, parameters, trial counts, elapsed time, and code identity under `research_logs/pr43/`.

### Commit 3 — Voter Diagnostics and GLASS Decision Structure

**Objective:** Determine where each voter contributes, where confidence is poorly scaled, and which arbiter changes warrant evaluation.

- [ ] On the frozen GLASS OOF rows, tabulate routing state × LR vote × EBM vote, support, and observed positive rate. Measure agreement, error correlation, double faults, unique-correct predictions, and correct lone dissents at the Stage 4 voter cuts and at matched flag budgets. Break out class and routing state where supported.
- [ ] Count Pass 1 leaked positives flagged by LR or EBM. For Pass 2 rows where both voters vote negative, count the false positives a veto would remove and the true positives it would lose.
- [ ] Measure LR and EBM discrimination and calibration on router-deferred rows. Estimate voter calibration intercepts and slopes by routing state where estimable, with support and uncertainty.
- [ ] Trace accepted-cut abstentions and per-voter contributions by class and route. Quantify how the bound on negative probability margins affects Pass 2 and router-deferred rows.
- [ ] Record which registered triggers were met for optional variants. Assess redundancy through shared errors and unique contributions. High agreement and near-uniform formula weights alone do not establish redundancy.

### Commit 4 — Stage 4 Candidate Evaluation at Matched Coverage

**Objective:** Compare the priority GLASS Linear Arbiter, targeted GLASS formula improvements, and Meta-XGBoost under the registered policy and nested Stage 4 procedure.

#### Step 4A — GLASS Linear Arbiter

- [ ] Fit regularized logistic regression with nonnegative coefficients and an intercept using full-batch log-loss optimization. Use `logit(p) − logit(t)` for Stage 1 and Stage 3 and signed `logit(r) − logit(b)` router evidence on routed rows, with zero router contribution on deferral. Specify finite clipping and direction conventions before fitting.
- [ ] Derive inputs from the Stage 1/3 probabilities, router probability, and route indicators available to Meta-XGBoost, using training-fitted constants where required. Verify each upstream’s router semantics. For GLASS, recover the rule rate using its producing model’s base rate. Keep rule identity and per-rule outcome counts outside the linear candidate’s inputs.
- [ ] Assess probability calibration before and after any inner-fitted correction. Report coefficients, bootstrap intervals, and fold-to-fold stability.
- [ ] Evaluate routing-state-specific coefficients shrunk toward the global coefficients only if the registered Commit 3 reliability trigger is met.
- [ ] Evaluate both upstreams against both reference arbiters. Apply the registered paired improvement and class-retention criteria before retaining the linear candidate for PR 45.

#### Step 4B — GLASS Formula and Meta-XGBoost Comparators

- [ ] Isolate GLASS log-odds margin normalization with the accepted formula weights first. Then evaluate sensitivity to the recall target of 0.70, weight blend α = 0.5, and abstention cut. Compare sequential fitting with nested joint selection where justified.
- [ ] Compare selected-rule in-sample evidence with cross-fitted router rule precision. Document rule support, unseen-rule handling, and what the frozen records can establish. Account for any required upstream refit in the Commit 5 budget.
- [ ] For Meta-XGBoost, compare the accepted balanced-weight, F2-at-0.5 tuning objective with registered threshold-free alternatives. Evaluate raw and calibrated confidence using the implemented `operating_space="calibrated"` path while preserving nonlinear learning.
- [ ] Produce one paired comparison per upstream, using the same rows for all arbiters. Report matched-coverage risk, AURC, achieved coverage at locked cuts, calibration where applicable, class retention, and whole-system outcomes. Record retained candidates and inconclusive comparisons.

### Commit 5 — OOF-versus-Refit Input Shift and Coverage Stability

**Objective:** Determine how much Stage 4 coverage changes with the operating rule and how much changes when upstream models are refitted.

#### Step 5A — Frozen-Table Stability Checks

- [ ] Compare Meta-XGBoost coverage under fold-nested operating thresholds and the single global threshold used for deployment. Label the global-cut result on training rows as descriptive.
- [ ] Measure abstention-cut and achieved-coverage stability across training-row resamples for retained candidates.
- [ ] Identify the remaining questions that require fresh upstream fits. Record the expected fit count and runtime before starting the simulated holdout.

#### Step 5B — Training-Only Simulated Upstream Holdout

- [ ] Start with the black-box upstream. Hold out a stratified subset of training rows and fit the upstream procedures on the remainder using accepted hyperparameters. Keep learned transformations, routing evidence, calibration, and cuts inside the fitting partition. Reuse the resulting fits and predictions across arbiters.
- [ ] Compare the fitting partition’s OOF input distribution with the simulated holdout distribution. On the same untouched holdout rows, compare full-refit and fold-averaged upstream predictions with Stage 4 held fixed.
- [ ] Isolate Meta-XGBoost’s own OOF/refit effect with upstream inputs held fixed. Report changes in route state, confidence, decisions, and achieved coverage.
- [ ] Apply the registered expansion rule to the GLASS upstream, accounting for fold-model versus full-fit rule rates and base rates. Evaluate retained candidates under the simulated conditions relevant to their proposed use. Identify any unresolved arm-specific shift as a prerequisite for PR 45 adoption.
- [ ] Record observed runtime, bottlenecks, and computational estimates for PR 45 regeneration and PR 49 repeated splits. Keep simulated fits research-local and preserve the accepted artifacts.

### Commit 6 — Recommendations, Validation, Executed Notebooks, and PR Closure

**Objective:** Record the provisional Stage 4 decision, hand actionable findings to PRs 44–45, complete reproducibility checks, and publish the accepted executed notebooks.

#### Step 6A — Stage 4 Recommendations and Research Handoff

- [ ] Consolidate the registered policy, supporting run IDs, paired comparisons, accepted and rejected variants, shift findings, and remaining uncertainties. Record the provisional GLASS Stage 4 choice or an inconclusive outcome. Specify the decision PR 45 must repeat on regenerated inputs.
- [ ] Hand PR 44 the upstream properties that materially affect Stage 4. Connect Stage 1 probability quality to the past-only encoding and MLP early-stopping work, router reliability and complementarity to the qualified `gap_placed` and `eci_warm` experiments, and deferred-row performance to shared `jed_alone` and the EBM interaction-budget comparison.
- [ ] Update the notebook map and PR 43 research log with accepted records, reproduction commands, costs, claim boundaries, and unresolved prerequisites. Reference the PR 42 handoff for feature-contract details.

#### Step 6B — Final Validation, Reproducibility, and Notebook Publication

- [ ] Run targeted research, notebook, and packaging checks, followed by the full default, slow, and comprehensive real-data regression suites. Run the applicable accepted-record checks and verify the frozen PR 39 pins, crossover identity, and PR 42 contract references.
- [ ] Execute the final Training Research Notebook in a clean Windows kernel through the established `_run_notebook` path. Verify reproduction against the registered tolerances and save the accepted executed copy with its run ID under `notebook_outputs/pr43/`.
- [ ] Reuse the PR 42 packager with explicitly accepted runs. Refresh the flat project-wide `executed_notebooks.zip` with the latest accepted copy of each notebook and repository source paths in `MANIFEST.json`. Prepare `pr43_executed_notebooks.zip` with run IDs in its filenames.
- [ ] Extract both archives and verify that the notebooks open with their saved tables and plots. Confirm tracked notebooks are output-free with code and Markdown cells intact.
- [ ] Publish both assets on the `pr43-notebooks` GitHub release. Verify the root README’s stable `releases/latest/download/executed_notebooks.zip` link and include the release-specific PR 43 snapshot link in the PR description.
- [ ] Review the complete diff for unintended changes and generated files. Record final validation results, warnings, and outstanding handoffs, then close the PR.

**Deliverable:** The Training Research Notebook, a registered Stage 4 abstention policy and comparison protocol, training-side voter diagnostics, an evaluated GLASS Linear Arbiter candidate, matched-coverage comparisons with both reference arbiters, an input-shift diagnosis, and a provisional Stage 4 recommendation for PR 45.

---

## PR 44 — Stage 1–3 Optimization and Router Research

**Objective:** Systematically evaluate training objectives and optimization strategies for the upstream stages of both cascades (Phase 2 Optimization Principle), prioritizing ranking, probability quality, and whole-system contribution, without intermediate artifact regeneration.

Stages 1–3 are trained independently: no upstream stage reads another's outputs during training, and only Stage 4 consumes them. Their optimization can proceed in any order; one regeneration follows in PR 45.

### Step 44A — Stage 1

#### GLASS — Calibrated Logistic Regression

- [ ] Evaluate regularization and penalty alternatives beyond the current search (100 trials, ROC-AUC, balanced class weights).
- [ ] Compare unweighted training with balanced weights followed by sigmoid calibration.
- [ ] Reevaluate calibration only when supported by training-side evidence.
- [ ] Preserve coefficient interpretability and the accepted feature contract.

#### Black-Box — Calibrated MLP (lower priority)

Stage 1 has three inputs, so architecture and optimization changes have limited expected headroom; each experiment must justify its computational cost.

- [ ] Evaluate architecture width and depth, regularization, and optimization settings.
- [ ] Investigate batch size, learning rate, early stopping, and validation strategies.
- [ ] Compare positive oversampling with class-weighted or alternative objectives, including their effects on calibration.
- [ ] Examine training stability, initialization variance, and convergence.

#### Shared Evaluation

- [ ] Compare ROC-AUC, PR-AUC, log-loss, Brier score, calibration, and computational efficiency.
- [ ] Treat Stage 1 decision thresholds as standalone reporting only: Stage 4 sets its own cuts, so threshold strategies are not Stage 1 optimization targets.
- [ ] Evaluate the incremental value of each change against the PR 39 baseline.
- [ ] Evaluate the PR 42 Stage 1 candidates for both arms, with a chronological comparison among the adoption criteria (PR 42 Step 6A, sections 2.5 and 5.3):
  - the past-only encodings drafted as `stage1-features` 3.0.0-candidate;
  - the MLP's encoder-isolated early stopping.

### Step 44B — Stage 2 Routers

#### GLASS Router

- [ ] Preserve the two-pass routing structure and union-aware ILP selection.
- [ ] Investigate candidate-rule generation and selection efficiency.
- [ ] Evaluate redundancy, complementarity, and marginal coverage.
- [ ] Consider alternative selector objectives only when supported by measured deficiencies.
- [ ] Test the feature-bin candidates identified in PR 42 against the router-level requirements of PR 42 Step 6A, section 3.4:
  - NSD `gap_placed`, fitted with both its registered and its fold-local reference inputs;
  - `eci_warm` required as a filter on Pass 2 partner rules.
- [ ] Preserve symbolic rule interpretability and auditable routing decisions.

#### Random Forest Router

- [ ] Start from the PR 35 objective-specialization results (the baseline configuration was retained for both passes); revisit only under a changed contract or new evidence.
- [ ] Evaluate hyperparameters, class weighting, and feature sampling where justified.
- [ ] Optimize thresholds under the same shared operating obligations.
- [ ] Compare routing coverage, constraint feasibility, and complementary predictions.

#### Shared Obligations

- [ ] Preserve the Pass 1 NPV/leakage and Pass 2 precision/support requirements.
- [ ] Preserve the lexicographic coverage objective unless a separately justified methodology revision is approved.
- [ ] Maintain fold-level cross-fitting and nested eligibility safeguards.
- [ ] Preserve G13 leakage auditing and applicable F31 diagnostics.
- [ ] Evaluate routing efficiency, stability, abstentions, and unique-correct contributions.
- [ ] Register a new protocol version before any Stage 2 comparison against revised Stage 1/3 models (F10: the PR 38 protocol pins the historical LR/EBM flag fingerprints).

### Step 44C — Stage 3

- [ ] Evaluate each Stage 3 model on the router-deferred rows it serves (`Stage3Artifact.evaluate_served`, training-side) before tuning.
- [ ] Compare the current tuning objective (F2 at the 0.5 cut with balanced weights) with threshold-free objectives (log-loss, PR-AUC) and with unweighted training. Before isotonic calibration, both accepted Stage 3 models have training OOF ECE of about 0.26–0.27.
- [ ] Evaluate the PR 42 Stage 3 candidate `jed_alone` (with its two Step 5A pairs) for both the EBM and XGBoost, against refitted baselines. Stage 3 features stay shared between the arms (PR 42 Step 6A, section 4.5).

#### GLASS — Explainable Boosting Machine

- [ ] Evaluate learning rate, boosting rounds, bagging, and regularization.
- [ ] Compare interaction budgets with the PR 42 procedural selector and the Step 5A cascade screen on Stage 2 abstentions (PR 42 Step 6A, section 4.5):
  - the three fixed pairs;
  - none;
  - one or two pairs;
  - a campaign pair fixed by construction;
  - capped automatic detection.
- [ ] Assess binning and smoothing parameters.
- [ ] Evaluate monotonic constraints only where supported by defensible hypotheses.
- [ ] Quantify the interpretability cost of additional terms, preserving transparent feature contributions and explanation fidelity.

#### Black-Box — XGBoost

- [ ] Investigate tree depth, learning rate, and early stopping in place of a tuned estimator count.
- [ ] Evaluate regularization, subsampling, and column sampling.
- [ ] Compare objectives and class-imbalance strategies.
- [ ] Examine probability calibration and threshold selection separately.
- [ ] Evaluate training efficiency and the stability of selected configurations.

### Step 44D — Optional Advanced Strategies

- [ ] Knowledge distillation and cascade compression: optional, separately identified experiments using leakage-safe (cross-fitted) teacher predictions. Report student fidelity, genuine-label performance, calibration, and interpretability separately, and report cross-family distillation as its own arm.
- [ ] Evaluate alternative proper scoring objectives where justified.
- [ ] Do not prioritize focal loss, reinforcement learning, or contextual bandits without new evidence: calibration matters to Stage 4, and the dataset is a static, fully labelled log without action-dependent outcomes.
- [ ] Promote a strategy only when inexpensive screening justifies further work.

**Acceptance:**

- [ ] Select accepted Stage 1–3 configurations using predefined criteria.
- [ ] Preserve the PR 39 configurations for comparison.
- [ ] Version candidate training contracts.
- [ ] Defer artifact regeneration to PR 45.
- [ ] Measure hyperparameter-selection optimism for shortlisted Stage 1 and Stage 3 configurations using nested selection and outer-fold evaluation before adoption. Reuse the PR 43 evaluation methodology where applicable.

**Deliverable:** Evaluated, versioned Stage 1–3 candidates for both cascades, ready for a single upstream regeneration.

---

## PR 45 — Final Upstream Regeneration, Stage 4 Training, and Comparison Registration

**Objective:** Regenerate the accepted upstream candidates once, train the Stage 4 arbiters under the policies registered in PR 43, and register the comparison protocol before optimized results are presented.

### Upstream Integration

- [ ] Lock the accepted upstream configurations and feature contracts.
- [ ] Adopt only PR 44-accepted candidates, each as a separately versioned contract that meets its leakage, G13 and regeneration requirements (PR 42 Step 6A, section 5).
- [ ] With the first adopted Stage 2 or Stage 3 revision, move the Stage 2/3 feature DAG to one production-owned implementation, keeping compatibility aliases for the PR 39 Stage 3 artifacts (PR 42 Step 6A, section 6).
- [ ] Regenerate the affected upstream artifacts once, in dependency order, with correct fold isolation.
- [ ] Reuse unchanged artifacts only where provenance and compatibility checks establish that reuse is valid.
- [ ] Verify stacking dependencies, row alignment, fingerprints, and feature provenance.

### Stage 4 Training

- [ ] Fit the GLASS Arbiter, Meta-XGBoost, and the GLASS Linear Arbiter (if retained in PR 43) under the PR 43 objectives, using training-side evidence only for candidate selection, thresholds, and coverage cutoffs.
- [ ] Select the final GLASS Stage 4 approach by the criteria predefined in PR 43, favoring the simpler interpretable model when performance is meaningfully comparable.
- [ ] Evaluate retained-only and whole-system performance together, with class-specific abstention, retained base rates, error rates, and calibration where applicable.
- [ ] Reject degenerate solutions that improve retained-row metrics by abstaining on excessive numbers of otherwise correct predictions.
- [ ] Recompute the registered 2×2 crossover (four cells) on the regenerated upstream, and run any separately registered GLASS Linear Arbiter comparison apart from it.

### Comparison Registration

- [ ] Before PR 46 runs the optimized notebooks, register the comparison questions, primary metrics, coverage grids, uncertainty methods, and analysis plan.

### Validation and Freeze

- [ ] Perform targeted and integration regression validation.
- [ ] Preserve the PR 39 reference artifacts.
- [ ] Freeze the optimized candidate artifacts under new run identifiers with distinct provenance.

**Deliverable:** Optimized, reproducible GLASS and black-box cascade candidates, trained Stage 4 arbiters, and a registered comparison protocol.

---

## PR 46 — Cascade Notebook Presentation and Comparison-Ready Freeze

**Objective:** Finalize the GLASS and black-box cascade notebooks as clear, reproducible presentations of the optimized architectures.

- [ ] Consolidate verbose notebook outputs into concise stage summaries.
- [ ] Build or reuse shared plotting and diagnostic utilities.
- [ ] Present architecture diagrams, feature and voter contributions, and performance results.
- [ ] Show routing as Pass 1 / Pass 2 / abstain rather than a binary outcome, including in the cascade trace.
- [ ] Include coverage, calibration, and stage-level diagnostics.
- [ ] Distinguish the frozen PR 39 baseline from the optimized Phase 2 candidates.
- [ ] Relocate research and comparison cells to the Training Research and Comparative Analysis notebooks, and remove duplicate training logic and unnecessary computation.
- [ ] Reproduce Phase 1 from the Phase 1 tag: its reproduction tooling depends on the PR 39 notebook cell headers.
- [ ] Standardize comparison-ready artifact manifests and loaders.
- [ ] Run both notebooks in clean kernels and confirm deterministic reproduction and preserved contracts.
- [ ] Freeze and tag the comparison-ready optimized cascade implementations.

**Deliverable:** Two finalized cascade notebooks and a versioned, reproducible comparison baseline.

---

## Phase 3 — Comparative Research and Robustness

**Objective:** Compare the finalized cascades and standalone models under the registered protocol, with robust statistical analysis and explicit claim boundaries.

## PR 47 — Comparison Framework and Cascade-to-Cascade Evaluation

### Comparison Infrastructure

- [ ] Establish the Comparative Analysis Notebook.
- [ ] Load frozen artifacts without retraining.
- [ ] Enforce feature, split, model, and provenance compatibility, and reject stale or mismatched artifacts.
- [ ] Add targeted tests for comparison loaders and metric calculations.
- [ ] Implement the comparison questions, metrics, and result schemas registered in PR 45.

### Cascade Comparison

- [ ] GLASS versus black-box end-to-end performance.
- [ ] Stage-level performance and complementarity.
- [ ] Registered 2×2 Stage 4 arbiter crossover (four cells).
- [ ] Separately registered GLASS Linear Arbiter comparisons, reported apart from the accepted 2×2 crossover.
- [ ] Matched positive-flag-budget comparisons.
- [ ] Matched abstention-coverage comparisons.
- [ ] Coverage-prescription comparison:
  - registered coverage grid (30–80%)
  - training OOF cutoffs applied to test
  - retained and whole-system metrics per coverage
  - risk–coverage curves / AURC
- [ ] Stage 4 abstention drift, per-class exclusion, and retained base rate.
- [ ] Test-side compliance with training-side obligations (the Stage 4 coverage floor, Stage 2 Pass 1 leakage and Pass 2 precision).
- [ ] Prediction agreement, disagreement, and unique-correct analysis.
- [ ] Between-arm OOF score correlation per stage.
- [ ] Paired bootstrap confidence intervals.
- [ ] Calibration-quality comparisons on appropriate probability scales (the GLASS Arbiter's weighted score is not a calibrated probability).
- [ ] Documented differences in model capacity, interpretability, and optimization budgets.
- [ ] Interpretation of performance, complexity, and transparency tradeoffs.

**Deliverable:** Reproducible primary cascade-comparison results and visual evidence.

---

## PR 48 — Standalone Model Baselines

### Glass-Box Candidates

- [ ] Logistic Regression.
- [ ] Classical GAM.
- [ ] EBM.
- [ ] Shallow Decision Tree.
- [ ] Sparse Rule List or RuleFit-style model.

### Black-Box Candidates

- [ ] XGBoost.
- [ ] Random Forest.
- [ ] MLP.
- [ ] RBF SVM.
- [ ] CatBoost or LightGBM.

### Methodology

- [ ] Define the shared, audited standalone feature population explicitly, and fit all preprocessing within folds.
- [ ] Apply common split, fold, tuning-budget, calibration, and evaluation obligations as appropriate, and record each model's budget.
- [ ] Generate leakage-safe OOF and held-out predictions.
- [ ] Compare at matched coverage: each standalone model abstains on its own confidence under the registered policy, so the cascades' abstention does not confound the comparison.
- [ ] Compare the strongest standalone alternatives against the respective cascades.
- [ ] Evaluate whether cascade complexity delivers measurable additional predictive or decision-making value.
- [ ] Optionally include a compressed-cascade baseline from PR 44.
- [ ] Record the additional dependencies (GAM, rule-list or RuleFit, CatBoost or LightGBM), and use a kernel approximation if an RBF SVM is impractical at about 33,000 training rows.

**Deliverable:** Controlled standalone baselines and evidence of the cascades' incremental value.

---

## PR 49 — Robustness, Ablations, and Scientific Validity

### Robustness

- [ ] Five-fold versus ten-fold sensitivity on both arms.
- [ ] Repeated seeds and splits for model-family claims, refitting procedural feature decisions within each split.
- [ ] Whether GLASS/black-box differences persist across protocol changes.
- [ ] Stage-removal ablations.
- [ ] Decision-threshold sensitivity.
- [ ] Abstention and coverage sensitivity.
- [ ] Tuning-budget sensitivity.
- [ ] Router-contract and feature-binning sensitivity, including NSD/ECI drift using PR 42 evidence.
- [ ] Temporal robustness using the PR 42 protocol (train on earlier months, test on later), reported as robustness evidence rather than independent validation.
- [ ] Build the past-only Stage 2–3 constructions and run them in the chronological cascade (PR 42 Step 6A, sections 5.2–5.3):
  - NSD, ECI and the day × month rate;
  - a past-only fallback for the past-only density.

  Refit bin cuts and interaction pairs inside each split, with their reference inputs stated.

### Stacking Methodology Sensitivity (F39; includes the PR 36 deferral)

- [ ] Assess indirect label dependence in the upstream OOF stacking features of the learned arbiters (Meta-XGBoost and, if retained, the GLASS Linear Arbiter).
- [ ] Evaluate the feasibility of fully nested upstream fitting within each meta-fold.
- [ ] Compare standard OOF stacking against fully nested stacking where computationally feasible.
- [ ] Measure effects on calibration, thresholds, abstention, and predictive performance.
- [ ] Document the methodological trade-offs and implications for the GLASS versus black-box comparison.

### Independent Evaluation and Claim Boundaries

- [ ] Record that no Tier 3 data exists for this dataset: every row has been used for training or in the previously exposed test split, and the other UCI Bank Marketing files come from the same campaigns.
- [ ] Identify any genuinely independent dataset or prospective evaluation, and freeze model selection and evaluation procedures before accessing new labels.
- [ ] Report independent findings separately from development and robustness results.
- [ ] Where independent evaluation is unavailable, explicitly limit generalization claims.

### Interpretability Boundaries

- [ ] Define which components the interpretability claims cover, including calibration maps, training-time guidance such as RF-guided rule search, and the GLASS Arbiter's weighted score.

**Deliverable:** Robustness evidence, quantified methodological sensitivity, and a transparent statement of scientific validity and claim boundaries.

---

## Phase 4 — Paper Readiness and Final Research Freeze

## PR 50 — Paper Exports, Reproducibility, and Final Freeze

- [ ] Finalize the research questions and accepted comparison claims.
- [ ] Export publication-quality figures and tables.
- [ ] Consolidate model, cascade, calibration, and abstention results.
- [ ] Present interpretability/performance tradeoffs.
- [ ] Document statistical uncertainty and limitations.
- [ ] Distinguish development findings, robustness evidence, and independently validated findings.
- [ ] Verify experiment provenance, dependency versions, and artifact identities.
- [ ] Run final end-to-end reproducibility and regression validation.
- [ ] Freeze the research implementation and experiment artifacts.
- [ ] Tag the research release.
- [ ] Prepare paper-ready methods, results, and supplemental evidence.

**Deliverable:** Reproducible, publication-ready research artifacts and a finalized experimental record.

---

## Deferred Engineering and Optional Research

- [x] **Windows notebook-execution warning (deferred from PR 40; resolved in PR 41).** On Windows, nbclient ran on Python's default Proactor event loop, which lacks the reader methods zmq needs, so pyzmq added a tornado selector thread and emitted one `RuntimeWarning` per process. PR 41 Commit 4 added a Windows-only path to `upstream_regeneration/vehicle.py` (`_run_notebook`): on Windows with Python 3.11 or later and no running event loop, `client.async_execute()` runs under `asyncio.Runner` on its own `SelectorEventLoop`, which is closed and unset afterward. Other platforms, older Pythons, and calls from a running loop still use `client.execute()`. No global event-loop policy changes, the deprecated `WindowsSelectorEventLoopPolicy` is not used, and the warning is not suppressed. The change is a reviewed exception to the provenance-sensitive execution code. PR 41 Commit 6 validated it with the vehicle tests (`tests/upstream_regeneration/test_execution_path.py`) and a representative Windows kernel execution, confirming the warning's elimination, subprocess compatibility, exception propagation, and cleanup.
- [x] **PuLP 3.3.2 deprecations (resolved in PR 41).** The GLASS Router's ILP code constructed `LpVariable` objects directly and solved with `PULP_CBC_CMD`, both deprecated in PuLP 3.3. PR 41 Commit 6 replaced them with `prob.add_variable(...)` (`ilp_builder.py`, `novelty_analyzer.py`) and with `COIN_CMD` on the CBC executable bundled with PuLP 3.x (`PULP_CBC_CMD.pulp_cbc_path`; CBC 2.10.3 with the recorded PuLP 3.3.2), which is what `PULP_CBC_CMD` ran. The CBC version, solver options, ILP objectives, constraints, and solver-status semantics are unchanged, and a regression test pins the solver path and options (`tests/stage2/test_ilp_rule_selector_union.py`). Both PuLP warning filters were removed from `pytest.ini`; `requirements.txt` keeps `pulp>=2.8,<4`.
- [ ] **PuLP 4.x migration (future compatibility).** PuLP 4.0 removes the bundled CBC binary and `PULP_CBC_CMD`, from which `ilp_builder.py` reads its pinned CBC path (`_BUNDLED_CBC`). CBC then comes from the `pulp[cbc]` extra or a `cbc` executable on `PATH`, possibly a different build from CBC 2.10.3, and PuLP 4.0 changes other modelling and solve APIs (see its 3.x-to-4.0 migration guide). Migrating requires a dependency change, a new solver pin, and solver-parity validation against the accepted Stage 2 selections (objectives, termination statuses, time-limit and feasible-but-not-optimal outcomes, and the CBC log parsing in `parse_cbc_log`), as a separately versioned Stage 2 change. Until then, keep `pulp>=2.8,<4`.
- [ ] **Joblib/Loky shutdown traceback (low priority).** On Windows (Python 3.14.3, Joblib 1.6.0 with vendored Loky 3.6.0, ipykernel 6.31.0), the Joblib/Loky resource tracker can print a `KeyError` traceback for a `joblib_memmapping_folder_` entry during kernel shutdown, after notebook execution has completed. Local reproduction supports a shutdown-order conflict: ipykernel terminates the tracker, and Joblib's exit-time cleanup restarts it with an empty registry. No effect on notebook results, accepted artifacts, or execution records has been observed, and no repository change is required (`research_logs/pr41/pr41_loky_resource_tracker.md`). [Joblib PR #1829](https://github.com/joblib/joblib/pull/1829) concerns temporary-folder registration and cleanup and may address it; before upgrading Joblib, repeat the isolated reproduction with the candidate release.
- [ ] **Lightweight GLASS development utility.** A small command-driven tool (`run script`, `run notebook`, `run cell`, `apply test`, `scan artifacts`) that reuses existing Python, pytest, notebook, and artifact utilities, protects accepted artifacts, never retrains unexpectedly, and leaves Git operations to Git. Evaluate when its workflow benefit justifies implementation.
- [ ] Investigate advanced training strategies that survive preliminary scientific and computational screening.
- [ ] Avoid new framework dependencies or infrastructure without demonstrated need.
- [ ] **Scikit-learn SAG/SAGA convergence warnings.** Seven convergence warnings occurred during PR 42 temporal-validity real-data validation. Investigate feature scaling, solver tolerance, iteration limits, and convergence during PR 44 training optimization or before PR 45 artifact regeneration. Do not suppress the warnings.
- [ ] **Optuna notebook widget rendering (low priority).** Investigate interactive progress-bar rendering warnings in saved executed notebooks. Preserve optimization results while improving published notebook presentation.
- [ ] **Cross-platform compatibility.** Evaluate Windows, Linux, and macOS portability, including LF/CRLF handling, filesystem paths, notebook execution, and cross-platform CI.
