## Roadmap

### Phase 1 — Finish and Validate the Cascades

## PR 40 — Test Warning Cleanup and Maintenance

**Objective:** Eliminate recurring, actionable test warnings by correcting their underlying causes rather than suppressing them. Improve test clarity and maintainability while preserving the finalized PR 39 modeling contracts, behavior, artifacts, and reproducibility.

### Commit 1 — Warning Diagnosis, Corrections, and Validation

**Objective:** Identify and resolve the recurring scikit-learn MLP batch-size warnings and any closely related, actionable warnings through minimal, reviewable changes.

#### Step 1A — Warning Inventory and Root-Cause Analysis

- [x] Start from the completed PR 39 baseline with a clean working tree.
- [x] Reproduce and inventory recurring warnings from the relevant test suites.
- [x] Identify warning sources, affected tests, triggering configurations, and underlying causes.
- [x] Distinguish harmless test-fixture issues from genuine implementation, compatibility, or deprecation concerns.
- [x] Determine the smallest safe correction for each actionable warning.

#### Step 1B — Targeted Warning Corrections

- [x] Correct invalid or unnecessary MLP batch-size configurations, prioritizing synthetic test fixtures.
- [x] Preserve the effective training behavior and numerical expectations of affected tests.
- [x] Address other recurring warnings only where the cause is verified and the correction is similarly narrow.
- [x] Do not suppress warnings through filters, blanket pytest settings, or equivalent workarounds.
- [x] Preserve production hyperparameters, model-family behavior, frozen contracts, and existing artifacts.
- [x] Complete the corrections in one focused implementation patch; use a second only if necessary.

#### Step 1C — Regression and Final Verification

- [x] Rerun affected tests and confirm their original assertions and contracts remain valid.
- [x] Verify that targeted warnings no longer occur without suppression.
- [x] Run relevant regression tests and confirm no new failures or warnings were introduced.
- [x] Verify that no model artifacts, historical evidence, or frozen research records were modified.
- [x] Review the final diff, confirm the changes remain maintenance-only, and commit the cleanup.

**Deliverable:** A cleaner test suite with the identified warning causes corrected, passing regression tests, unchanged modeling behavior, and no warning suppression. One maintenance commit, with one or at most two implementation patches.

### Commit 2 — Research Roadmap Expansion and README Update

**Objective:** Update `prototype/bank_pipeline/documentation/research_roadmap.md` to incorporate the completed PR 39 baseline, revised Phase 2 research direction, and five-notebook research plan. Update the root `README.md` to accurately reflect the completed Phase 1 implementation and research status. Finalize the forward-looking roadmap without implementing new research or restructuring the repository.

- [x] Review the existing research roadmap and reconcile completed PR 38–39 milestones.
- [x] Incorporate the accepted Phase 1 baseline, four-stage cascade architectures, and four-way Stage 4 arbiter crossover.
- [x] Integrate the revised five-notebook research plan, preserving the intended responsibilities and experimental boundaries of each notebook.
- [x] Organize Phase 2 research priorities, including:
  - Controlled GLASS versus black-box comparisons.
  - Model-family-specific optimization under shared evaluation obligations.
  - Stage 4 training-objective and hyperparameter optimization.
  - Confidence, abstention, coverage, and whole-system performance analysis.
  - Investigation of Stage 4 crossover findings and structural differences.
  - Independent evaluation, leakage safeguards, and scientific validity.
  - Notebook presentation, reusable research modules, and preparation of publication-quality evidence.
- [x] Preserve the distinction between completed experiments, accepted findings, research limitations, and future hypotheses.
- [x] Record future maintenance work, including PR 41 structural cleanup and the proposed lightweight GLASS development environment, without implementing either.
- [x] Remove obsolete or redundant planning entries where appropriate without discarding historical research evidence.
- [x] Review the finalized roadmap for completeness, consistency, and realistic execution order.
- [x] Update the root `README.md` to reflect the finalized Phase 1 architecture, GLASS Arbiter terminology, validated results, reproducibility, and current research status.
- [x] Commit both documentation changes together: `prototype/bank_pipeline/documentation/research_roadmap.md` and the root `README.md`.

**Deliverable:** A substantially finalized research roadmap reflecting the accepted Phase 1 baseline and planned Phase 2 research program, alongside an updated repository README accurately documenting the completed Phase 1 implementation. One documentation-only commit modifying `prototype/bank_pipeline/documentation/research_roadmap.md` and the root `README.md`.

---

### PR 40 — Final Completion Boundary

- [x] **Commit 1 complete:** Recurring actionable test warnings diagnosed, corrected, and validated without altering frozen modeling contracts or artifacts.
- [x] **Commit 2 complete:** Research roadmap (`prototype/bank_pipeline/documentation/research_roadmap.md`) and root `README.md` updated with the revised five-notebook plan, Phase 2 priorities, and future engineering work.
- [x] **PR 40 complete:** Both commits validated, pushed, reviewed, and merged.
- [x] **Phase 1 complete:** GLASS and black-box cascades operational, integrated, validated, and ready for Phase 2 research and presentation.

**Out of scope:** Model retraining, hyperparameter optimization, feature engineering changes, calibration revisions, Stage 2 selector changes, artifact regeneration, notebook redesign, and unrelated refactoring.

---

### Accepted Phase 1 Baseline (PR 39)

PR 39 established the validated, reproducible development baseline. All accepted artifacts, manifests, and historical records remain frozen. Subsequent research produces separately versioned results.

| Stage | GLASS Cascade | Black-Box Cascade |
|---|---|---|
| 1 | Calibrated Logistic Regression | Calibrated MLP |
| 2 | Symbolic GLASS Router | Random Forest Router |
| 3 | Explainable Boosting Machine | XGBoost |
| 4 | GLASS Arbiter | Meta-XGBoost |

- **GLASS Arbiter:** Performance-weighted, margin-based voting with confidence-threshold abstention. The Stage 3 EBM is an upstream voter, not a Stage 4 meta-model.
- **Meta-XGBoost:** Learned nonlinear Stage 4 arbiter using cross-fitted upstream predictions.
- **Crossover:** Registered 2×2 comparison of both arbiters on both upstream architectures.
- **Validation:** 3,390 fast, 110 real-data, and 11 slow tests passed. Both notebooks reproduced deterministically with maximum numerical difference 0.0.
- **Accepted runs:** Stage 2 `c3_20261112_153333`; Stages 1/3 `c4_20261115_204125`; Stage 4 `c5_20261116_153459`; reproduction `c6_20261116_194111`.
- **Limitation:** The previously examined test split supports descriptive findings, not independent validation.

### Phase 2–4 Research Principles

**Objective:** Optimize both cascades to their full legitimate potential, evaluate them fairly, and prepare reproducible scientific comparisons.

- **Mirrored evaluation:** Preserve equivalent stage responsibilities, information availability, operating obligations, and leakage safeguards. Allow architecture-specific training methods and objectives.
- **Systematic optimization:** Investigate relevant training strategies at every stage, prioritizing inexpensive experiments before expensive cross-fitting and regeneration.
- **Stage 4 research:** Evaluate improvements to the existing GLASS Arbiter, the proposed GLASS Linear Arbiter, and Meta-XGBoost under shared abstention and coverage objectives.
- **Scientific validity:** Use training-only evidence for model selection, register comparison criteria in advance, and distinguish development, robustness, and independent validation.
- **Model selection:** Consider predictive performance, calibration, coverage, interpretability, stability, and computational efficiency. Favor simpler interpretable models when performance is meaningfully comparable.
- **Provenance:** Preserve the accepted Phase 1 baseline, version methodology changes, and document differences in model capacity and optimization budgets.

### Five-Notebook Research Structure

| Notebook | Responsibility |
|---|---|
| Feature Research | Feature engineering, binning, leakage, and temporal validity |
| Training Research | Stage-wise objectives, optimization, and model selection |
| GLASS Cascade | Final interpretable architecture and presentation |
| Black-Box Cascade | Final black-box architecture and presentation |
| Comparative Analysis | Cascade comparisons, standalone baselines, robustness, and paper figures |

Existing research modules and notebook content will be reused wherever practical.

---

### Phase 2 — Research Organization, Training Research, and Optimized Cascades

#### PR 41 — Structural Cleanup and Research Organization

**Objective:** Organize the repository for Phase 2 before substantial notebook and training development, with no scientific behavior changes.

**Phase 1 protection**

- [ ] Tag the PR 40 merge commit as the Phase 1 baseline before any structural change.
- [ ] Keep provenance-sensitive paths unchanged: `upstream_regeneration/`, `stage4_regeneration/`, `phase1_reproduction/`, `stage2_crossfit/`, `stage2_binning/`, `stage1_calibration/`, and `research_logs/`. Their records and documented `python -m` commands name these paths, and Phase 1 re-execution depends on fixed file lists and notebook cell headers.
- [ ] Preserve historical compatibility identifiers, including `meta_ebm`, and every persisted artifact identifier.
- [ ] Re-run the accepted runs' record `check` commands after any import or path change.

**Structural cleanup**

- [ ] Review existing Python modules, tests, filenames, and notebook responsibilities.
- [ ] Rename files according to their actual functions rather than historical implementation steps.
- [ ] Group related tests by stage and responsibility.
- [ ] Use `git mv` and minimal import/path corrections.
- [ ] Identify reusable training, evaluation, plotting, and artifact-loading utilities.
- [ ] Resolve unnecessary duplication without changing modeling logic, test assertions, or research behavior.
- [ ] Preserve existing tests, historical provenance, and accepted research artifacts.

**Research organization**

- [ ] Map existing notebook cells and modules to the five notebooks without rebuilding notebooks yet.
- [ ] Run relevant regression tests.

**Conditional maintenance:** the Windows ZeroMQ/asyncio warning (Deferred Engineering) may be evaluated here. It is not a PR 41 requirement and must not change the accepted notebook execution pathway.

**Deliverable:** A cleaner repository structure ready for Phase 2 experimentation, a tagged Phase 1 baseline, and no scientific behavior changes.

---

#### PR 42 — Feature Research Notebook and Feature-Contract Investigation

**Objective:** Develop the Feature Research Notebook and determine, on training-side evidence, whether the accepted feature representations should be retained or separately revised before training optimization.

**Feature research**

- [ ] Inventory existing engineered features, feature contracts, and stage-specific feature populations.
- [ ] Review feature distributions, correlations, redundancy, predictive contributions, and stability.
- [ ] Investigate interpretable feature interactions and candidate transformations.
- [ ] Verify that every candidate feature is available at the intended prediction time.
- [ ] Ensure feature selection, encoding, and transformation fitting remain fold-isolated.
- [ ] Evaluate candidate features using training-side evidence and reproducible statistical analyses.
- [ ] Preserve shared feature comparability between arms.

**Stage 1 temporal validity (high priority)**

Context from repository inspection: the dataset rows are in date order (May 2008 – November 2010), and the subscription rate rises from about 3–6% to about 45–57% over that period. Two of the three Stage 1 features, `subscribe_rate_by_euribor_bin` and `subscribe_rate_by_dow_month`, are target rates keyed on period-identifying variables, and most euribor quantile bins fall within one to three consecutive months. Under the random split, these encodings are fitted partly on rows contacted later in the same period.

- [ ] Document exactly how both encodings are constructed, fitted, and applied (training split, Stage 1 fold designs, test mapping).
- [ ] Determine whether the protocol contains direct label leakage, temporal look-ahead, or both.
- [ ] Verify that every transformation is fitted within its proper training fold.
- [ ] Evaluate past-only (expanding-window) encodings: feature availability, deployment realism, and predictive effect.
- [ ] Extend the audit to the other target-derived features used by Stages 2–3, such as the neighborhood subscription density.
- [ ] Determine whether candidate revisions require separately versioned feature contracts; do not alter the accepted PR 39 baseline.
- [ ] Design the temporal robustness protocol: chronological split, past-only encodings, and a policy for prevalence shift, which affects thresholds, abstention cuts, and the Stage 2 NPV and precision obligations. Execution follows in PR 49.

**Procedural feature decisions**

- [ ] Where feasible, make the NSD/ECI bin derivation and the EBM interaction selection procedural and fold-fitted, so repeated-split and nested evaluations can refit them rather than reuse constants chosen on all training rows.

**Stage 2 NSD/ECI binning drift and sensitivity (deferred from PR 39)**

- [ ] Revisit the substantial lift drift observed during PR 39 Step 3A, particularly:
  - `eci_warm`: original 1.49–1.67× lift versus approximately 1.03× OOF.
  - `eci_hot`: original 3.53× versus approximately 2.50× OOF.
  - `nsd_elevated`: approximately 2.49× pooled OOF lift, with significant fold-level variation.
- [ ] Reproduce the original NSD/ECI feature-bin research using the corrected, fold-isolated feature engineering implementation.
- [ ] Distinguish threshold-placement problems from encoder behavior, population shifts, and in-sample label leakage.
- [ ] Reevaluate decile-lift distributions, candidate thresholds, bin counts, support, confidence intervals, and fold stability.
- [ ] Investigate whether transitional or weak bins retain meaningful predictive utility through conjunctions with other Stage 2 features.
- [ ] Evaluate alternative threshold configurations and bin structures under leakage-safe training protocols, without modifying the frozen Phase 1 baseline.
- [ ] Identify the candidates that deserve downstream router testing in PR 44 (coverage, constraint feasibility, complementarity, and computational cost).
- [ ] Compare candidate configurations against the frozen PR 39 binning baseline using predefined evaluation criteria.
- [ ] Determine whether the existing bins should remain unchanged or whether evidence warrants a separately versioned methodology revision.
- [ ] If revision is justified, document the required feature-contract changes, G13 leakage audit, downstream artifact regeneration, and follow-up work rather than silently modifying frozen artifacts.

**Acceptance**

- [ ] Record candidate configurations, supporting evidence, and rejected alternatives.
- [ ] Retain the frozen feature contracts unless evidence supports a separately versioned change.
- [ ] Establish the feature candidates and contracts for subsequent training experiments.

**Deliverable:** A reproducible Feature Research Notebook, a documented temporal-validity assessment and robustness protocol, a quantified explanation of the NSD/ECI lift drift, and evidence-based feature-contract recommendations.

---

#### PR 43 — Training Research Notebook and Stage 4 Policy Research

**Objective:** Establish the train-only training-research workflow and complete inexpensive Stage 4 methodology research, including the GLASS Linear Arbiter as a priority experimental candidate, on the frozen PR 39 training-side OOF evidence, before upstream optimization or regeneration.

Every Stage 4 candidate in this PR can be screened on the frozen training-side OOF table (Stage 1–3 predictions, router decisions, labels, and fold identities) without regenerating Stages 1–3. Small convex problems are solved with full-batch optimization; no mini-batch, online-learning, or reinforcement-learning infrastructure is introduced.

##### Step 43A — Training Experiment Framework

- [ ] Build the Training Research Notebook from existing research cells and modules; inventory tuning, calibration, cross-fitting, and training utilities before adding infrastructure.
- [ ] Enforce train-only selection with the PR 38 boundary extractors: the notebook loads no test split, test prediction, or test metric.
- [ ] Define stage-specific objectives, acceptance metrics, experiment budgets, and stopping criteria.
- [ ] Screen candidates inexpensively before full nested procedures, without weakening leakage safeguards.
- [ ] Separate raw model scoring, calibration, and decision-threshold selection.
- [ ] Record each arm's search spaces, trial budgets, and selection protocols.
- [ ] Measure hyperparameter-selection optimism. Stage 1 and Stage 3 currently tune on the same folds that produce their OOF predictions (calibration and thresholds are nested; hyperparameters are not). Define when nested selection is required for adoption.
- [ ] Build a nested Stage 4 validation harness on the frozen training-side OOF table: outer folds for evaluation, inner folds for selection, each fold fitted from scratch (no warm starts across folds), with the accepted GLASS Arbiter's procedure refitted per outer fold as the baseline.
- [ ] Track parameters, seeds, fold identities, elapsed time, and results, and support reproducible comparison against the accepted PR 39 configurations.
- [ ] Estimate the computational budget for the PR 45 regeneration and the PR 49 repeated-split experiments.

##### Step 43B — Stage 4 Objective and Abstention Policy

- [ ] Define the purpose and cost of abstention (for example, deferral to review) and a whole-system scoring policy for abstained rows; none is currently registered.
- [ ] Register coverage/risk criteria: selective risk at registered coverage levels, risk–coverage curves and AURC, or an explicit cost rule.
- [ ] Evaluate alternatives to retained F2 first. F2 has no true-negative term, so abstaining on a predicted negative never lowers retained F2; in all four accepted crossover cells the selected cut sits at the 50% training coverage floor.
- [ ] Compare Stage 4 scoring functions at matched coverage (AURC and selective risk on the registered coverage grid), separately from operating-point selection; never select weights or thresholds by retained F2.
- [ ] Apply one registered abstention/coverage policy to all Stage 4 candidates.
- [ ] Investigate excessive abstention and class-specific coverage (per-class retention, retained base rate) on training-side evidence, and reject candidates that improve retained-row metrics by abstaining on more correct predictions.

##### Step 43C — Stage 4 Voter Diagnostics

Measured overall and by routing state (Pass 1, Pass 2, router-deferred) on the frozen training-side OOF table, with voter decisions at the Stage 4 cuts and at matched flag budgets. High agreement is not treated as redundancy, and nearly uniform weights are not treated as inherently wrong.

- [ ] Vote-pattern table: counts and observed positive rates for each routing state × LR vote × EBM vote.
- [ ] LR–EBM agreement, error correlation, and double-fault rate (both voters wrong).
- [ ] Unique-correct predictions and correct lone dissents, by voter, class, and routing state.
- [ ] Router contributions: Pass 1 leaked positives flagged by LR or EBM, and the Pass 2 veto opportunity (false and true positives where both LR and EBM vote negative).
- [ ] Router-deferred rows: LR and EBM discrimination and calibration within that population.
- [ ] Segment calibration: intercept and slope of each voter's logit within each routing state.
- [ ] Abstention composition at the accepted cut, by class and routing state.

##### Step 43D — GLASS Arbiter Improvements

Context: the accepted weights (LR 0.343, EBM 0.355, router 0.302) are near-uniform by construction of the formula, and per-row influence is dominated by the confidence margins, whose negative side cannot exceed a voter's cut. The margins and the abstention objective are investigated before the weights.

- [ ] Analyze per-voter contributions and decision structure, including the structural confidence bound and its effect on Pass 2 and router-deferred rows.
- [ ] Evaluate log-odds confidence-margin normalization with the existing formula weights, isolating the effect of normalization.
- [ ] Evaluate sensitivity to the recall target (0.70), the weight blend (α = 0.5), and the abstention threshold.
- [ ] Evaluate nested joint selection of these parameters, where justified, against the current sequential fit.
- [ ] Compare in-sample rule evidence with cross-fitted router rule precision.
- [ ] Preserve the accepted configuration as the reproducible reference.

##### Step 43E — GLASS Linear Arbiter (priority experimental candidate)

- [ ] Define the candidate: regularized logistic regression with nonnegative coefficients and an intercept on log-odds voter-confidence signals: `logit(p) − logit(t)` for LR and EBM, and the signed `logit(r) − logit(b)` for the router where it routes (zero elsewhere). Its output is a probability estimate whose calibration must be assessed; any calibration correction uses training-only evidence.
- [ ] Use only the information channels Meta-XGBoost receives; give it no privileged information.
- [ ] Fit with full-batch convex optimization and a log-loss objective under the Step 43A nested harness, applying the registered abstention policy.
- [ ] Report coefficients with bootstrap intervals and fold-to-fold stability.
- [ ] Evaluate routing-state-specific coefficients, shrunk toward the global coefficients, only if Step 43C shows meaningful reliability differences between routing states.
- [ ] Compare it with the GLASS Arbiter at matched coverage; retain it only if its improvement exceeds fold-to-fold variability with no worse class-specific retention.
- [ ] Register it as a separate experimental arbiter, not a replacement; the GLASS Arbiter and Meta-XGBoost remain reference configurations.

##### Step 43F — Meta-XGBoost

- [ ] Compare its tuning objective (F2 at the 0.5 cut with balanced weights) with threshold-free alternatives.
- [ ] Compare confidence behavior in raw and calibrated probability space (`operating_space="calibrated"` is implemented).
- [ ] Evaluate its abstention policy under the registered criteria, without restricting its nonlinear learning.

##### Step 43G — Stage 4 Input Shift and Coverage Stability

- [ ] Meta-XGBoost: compare training-side coverage under its per-fold (fold-nested) thresholds and under the single global operating threshold used at test time.
- [ ] Investigate OOF-versus-refit input shift with a training-only simulated holdout: hold out a stratified part of the training split, fit the upstream stages on the remainder with the accepted hyperparameters, and compare OOF, refit, and fold-averaged inputs, confidence, and coverage. Start with the black-box upstream and pre-register the decision rule.
- [ ] Localize any shift by stage, including Meta-XGBoost's own meta-level OOF/refit asymmetry and the GLASS router evidence (fold-model rule rates and base rates on the training side, full-training values on the test side).
- [ ] Evaluate the retained Stage 4 candidates in the simulated holdout before PR 45; learned coefficients can amplify OOF-versus-refit shift.
- [ ] Measure abstention-cut and coverage stability across training-row resamples.

**Registration**

- [ ] Keep the registered PR 39 2×2 crossover unchanged.
- [ ] Register the GLASS Linear Arbiter comparisons (against the GLASS Arbiter and Meta-XGBoost, applied to both upstreams) separately, before their results are examined.

**Acceptance**

- [ ] Predefine the criteria for selecting the final GLASS Stage 4 approach: performance, calibration, coverage, interpretability, and computational efficiency, favoring the simpler interpretable model when performance is meaningfully comparable.
- [ ] Record the registered Stage 4 objectives, policies, candidates, and the provisional GLASS Stage 4 choice with their training-side evidence.
- [ ] Identify the upstream properties Stage 4 depends on (for example, probability quality on router-deferred rows) to direct PR 44.
- [ ] Regenerate no artifacts.

**Deliverable:** The Training Research Notebook, a registered Stage 4 abstention policy, training-side voter diagnostics, an evaluated GLASS Linear Arbiter candidate, a diagnosis of Stage 4 input shift, and predefined Stage 4 selection criteria for PR 45.

---

#### PR 44 — Stage 1–3 Optimization and Router Research

**Objective:** Systematically evaluate training objectives and optimization strategies for the upstream stages of both cascades (Phase 2 Optimization Principle), prioritizing ranking, probability quality, and whole-system contribution, without intermediate artifact regeneration.

Stages 1–3 are trained independently: no upstream stage reads another's outputs during training, and only Stage 4 consumes them. Their optimization can proceed in any order; one regeneration follows in PR 45.

##### Step 44A — Stage 1

**GLASS — Calibrated Logistic Regression**

- [ ] Evaluate regularization and penalty alternatives beyond the current search (100 trials, ROC-AUC, balanced class weights).
- [ ] Compare unweighted training with balanced weights followed by sigmoid calibration.
- [ ] Reevaluate calibration only when supported by training-side evidence.
- [ ] Preserve coefficient interpretability and the accepted feature contract.

**Black-box — Calibrated MLP (lower priority)**

Stage 1 has three inputs, so architecture and optimization changes have limited expected headroom; each experiment must justify its computational cost.

- [ ] Evaluate architecture width and depth, regularization, and optimization settings.
- [ ] Investigate batch size, learning rate, early stopping, and validation strategies.
- [ ] Compare positive oversampling with class-weighted or alternative objectives, including their effects on calibration.
- [ ] Examine training stability, initialization variance, and convergence.

**Shared evaluation**

- [ ] Compare ROC-AUC, PR-AUC, log-loss, Brier score, calibration, and computational efficiency.
- [ ] Treat Stage 1 decision thresholds as standalone reporting only: Stage 4 sets its own cuts, so threshold strategies are not Stage 1 optimization targets.
- [ ] Evaluate the incremental value of each change against the PR 39 baseline.

##### Step 44B — Stage 2 Routers

**GLASS Router**

- [ ] Preserve the two-pass routing structure and union-aware ILP selection.
- [ ] Investigate candidate-rule generation and selection efficiency.
- [ ] Evaluate redundancy, complementarity, and marginal coverage.
- [ ] Consider alternative selector objectives only when supported by measured deficiencies.
- [ ] Test the feature-bin candidates identified in PR 42.
- [ ] Preserve symbolic rule interpretability and auditable routing decisions.

**Random Forest Router**

- [ ] Start from the PR 35 objective-specialization results (the baseline configuration was retained for both passes); revisit only under a changed contract or new evidence.
- [ ] Evaluate hyperparameters, class weighting, and feature sampling where justified.
- [ ] Optimize thresholds under the same shared operating obligations.
- [ ] Compare routing coverage, constraint feasibility, and complementary predictions.

**Shared obligations**

- [ ] Preserve the Pass 1 NPV/leakage and Pass 2 precision/support requirements.
- [ ] Preserve the lexicographic coverage objective unless a separately justified methodology revision is approved.
- [ ] Maintain fold-level cross-fitting and nested eligibility safeguards.
- [ ] Preserve G13 leakage auditing and applicable F31 diagnostics.
- [ ] Evaluate routing efficiency, stability, abstentions, and unique-correct contributions.
- [ ] Register a new protocol version before any Stage 2 comparison against revised Stage 1/3 models (F10: the PR 38 protocol pins the historical LR/EBM flag fingerprints).

##### Step 44C — Stage 3

- [ ] Evaluate each Stage 3 model on the router-deferred rows it serves (`Stage3Artifact.evaluate_served`, training-side) before tuning.
- [ ] Compare the current tuning objective (F2 at the 0.5 cut with balanced weights) with threshold-free objectives (log-loss, PR-AUC) and with unweighted training. Before isotonic calibration, both accepted Stage 3 models have training OOF ECE of about 0.26–0.27.

**GLASS — Explainable Boosting Machine**

- [ ] Evaluate learning rate, boosting rounds, bagging, and regularization.
- [ ] Compare interaction budgets: the three fixed pairs, none, and capped automatic detection.
- [ ] Assess binning and smoothing parameters.
- [ ] Evaluate monotonic constraints only where supported by defensible hypotheses.
- [ ] Quantify the interpretability cost of additional terms, preserving transparent feature contributions and explanation fidelity.

**Black-box — XGBoost**

- [ ] Investigate tree depth, learning rate, and early stopping in place of a tuned estimator count.
- [ ] Evaluate regularization, subsampling, and column sampling.
- [ ] Compare objectives and class-imbalance strategies.
- [ ] Examine probability calibration and threshold selection separately.
- [ ] Evaluate training efficiency and the stability of selected configurations.

##### Step 44D — Optional Advanced Strategies

- [ ] Knowledge distillation and cascade compression: optional, separately identified experiments using leakage-safe (cross-fitted) teacher predictions. Report student fidelity, genuine-label performance, calibration, and interpretability separately, and report cross-family distillation as its own arm.
- [ ] Evaluate alternative proper scoring objectives where justified.
- [ ] Do not prioritize focal loss, reinforcement learning, or contextual bandits without new evidence: calibration matters to Stage 4, and the dataset is a static, fully labelled log without action-dependent outcomes.
- [ ] Promote a strategy only when inexpensive screening justifies further work.

**Acceptance**

- [ ] Select accepted Stage 1–3 configurations using predefined criteria.
- [ ] Preserve the PR 39 configurations for comparison.
- [ ] Version candidate training contracts.
- [ ] Defer artifact regeneration to PR 45.

**Deliverable:** Evaluated, versioned Stage 1–3 candidates for both cascades, ready for a single upstream regeneration.

---

#### PR 45 — Final Upstream Regeneration, Stage 4 Training, and Comparison Registration

**Objective:** Regenerate the accepted upstream candidates once, train the Stage 4 arbiters under the policies registered in PR 43, and register the comparison protocol before optimized results are presented.

**Upstream integration**

- [ ] Lock the accepted upstream configurations and feature contracts.
- [ ] Regenerate the affected upstream artifacts once, in dependency order, with correct fold isolation.
- [ ] Reuse unchanged artifacts only where provenance and compatibility checks establish that reuse is valid.
- [ ] Verify stacking dependencies, row alignment, fingerprints, and feature provenance.

**Stage 4 training**

- [ ] Fit the GLASS Arbiter, Meta-XGBoost, and the GLASS Linear Arbiter (if retained in PR 43) under the PR 43 objectives, using training-side evidence only for candidate selection, thresholds, and coverage cutoffs.
- [ ] Select the final GLASS Stage 4 approach by the criteria predefined in PR 43, favoring the simpler interpretable model when performance is meaningfully comparable.
- [ ] Evaluate retained-only and whole-system performance together, with class-specific abstention, retained base rates, error rates, and calibration where applicable.
- [ ] Reject degenerate solutions that improve retained-row metrics by abstaining on excessive numbers of otherwise correct predictions.
- [ ] Recompute the registered 2×2 crossover (four cells) on the regenerated upstream, and run any separately registered GLASS Linear Arbiter comparison apart from it.

**Comparison registration**

- [ ] Before PR 46 runs the optimized notebooks, register the comparison questions, primary metrics, coverage grids, uncertainty methods, and analysis plan.

**Validation and freeze**

- [ ] Perform targeted and integration regression validation.
- [ ] Preserve the PR 39 reference artifacts.
- [ ] Freeze the optimized candidate artifacts under new run identifiers with distinct provenance.

**Deliverable:** Optimized, reproducible GLASS and black-box cascade candidates, trained Stage 4 arbiters, and a registered comparison protocol.

---

#### PR 46 — Cascade Notebook Presentation and Comparison-Ready Freeze

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

### Phase 3 — Comparative Research and Robustness

**Objective:** Compare the finalized cascades and standalone models under the registered protocol, with robust statistical analysis and explicit claim boundaries.

#### PR 47 — Comparison Framework and Cascade-to-Cascade Evaluation

**Comparison infrastructure**

- [ ] Establish the Comparative Analysis Notebook.
- [ ] Load frozen artifacts without retraining.
- [ ] Enforce feature, split, model, and provenance compatibility, and reject stale or mismatched artifacts.
- [ ] Add targeted tests for comparison loaders and metric calculations.
- [ ] Implement the comparison questions, metrics, and result schemas registered in PR 45.

**Cascade comparison**

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

#### PR 48 — Standalone Model Baselines

**Glass-box candidates**

- [ ] Logistic Regression.
- [ ] Classical GAM.
- [ ] EBM.
- [ ] Shallow Decision Tree.
- [ ] Sparse Rule List or RuleFit-style model.

**Black-box candidates**

- [ ] XGBoost.
- [ ] Random Forest.
- [ ] MLP.
- [ ] RBF SVM.
- [ ] CatBoost or LightGBM.

**Methodology**

- [ ] Define the shared, audited standalone feature population explicitly, and fit all preprocessing within folds.
- [ ] Apply common split, fold, tuning-budget, calibration, and evaluation obligations as appropriate, and record each model's budget.
- [ ] Generate leakage-safe OOF and held-out predictions.
- [ ] Compare at matched coverage: each standalone model abstains on its own confidence under the registered policy, so the cascades' abstention does not confound the comparison.
- [ ] Compare the strongest standalone alternatives against the respective cascades.
- [ ] Evaluate whether cascade complexity delivers measurable additional predictive or decision-making value.
- [ ] Optionally include a compressed-cascade baseline from PR 44.
- [ ] Record the additional dependencies (GAM, rule-list or RuleFit, CatBoost or LightGBM), and use a kernel approximation if an RBF SVM is impractical at about 33,000 training rows.

**Deliverable:** Controlled standalone baselines and evidence of the cascades' incremental value.

#### PR 49 — Robustness, Ablations, and Scientific Validity

**Robustness**

- [ ] Five-fold versus ten-fold sensitivity on both arms.
- [ ] Repeated seeds and splits for model-family claims, refitting procedural feature decisions within each split.
- [ ] Whether GLASS/black-box differences persist across protocol changes.
- [ ] Stage-removal ablations.
- [ ] Decision-threshold sensitivity.
- [ ] Abstention and coverage sensitivity.
- [ ] Tuning-budget sensitivity.
- [ ] Router-contract and feature-binning sensitivity, including NSD/ECI drift using PR 42 evidence.
- [ ] Temporal robustness using the PR 42 protocol (train on earlier months, test on later), reported as robustness evidence rather than independent validation.

**Stacking methodology sensitivity (F39; includes the PR 36 deferral)**

- [ ] Assess indirect label dependence in the upstream OOF stacking features of the learned arbiters (Meta-XGBoost and, if retained, the GLASS Linear Arbiter).
- [ ] Evaluate the feasibility of fully nested upstream fitting within each meta-fold.
- [ ] Compare standard OOF stacking against fully nested stacking where computationally feasible.
- [ ] Measure effects on calibration, thresholds, abstention, and predictive performance.
- [ ] Document the methodological trade-offs and implications for the GLASS versus black-box comparison.

**Independent evaluation and claim boundaries**

- [ ] Record that no Tier 3 data exists for this dataset: every row has been used for training or in the previously exposed test split, and the other UCI Bank Marketing files come from the same campaigns.
- [ ] Identify any genuinely independent dataset or prospective evaluation, and freeze model selection and evaluation procedures before accessing new labels.
- [ ] Report independent findings separately from development and robustness results.
- [ ] Where independent evaluation is unavailable, explicitly limit generalization claims.

**Interpretability boundaries**

- [ ] Define which components the interpretability claims cover, including calibration maps, training-time guidance such as RF-guided rule search, and the GLASS Arbiter's weighted score.

**Deliverable:** Robustness evidence, quantified methodological sensitivity, and a transparent statement of scientific validity and claim boundaries.

### Phase 4 — Paper Readiness and Final Research Freeze

#### PR 50 — Paper Exports, Reproducibility, and Final Freeze

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

### Deferred Engineering and Optional Research

- [ ] **Windows notebook-execution warning (deferred from PR 40).** On Windows, `upstream_regeneration.vehicle.execute` runs nbclient on Python's default Proactor event loop; pyzmq adds a tornado selector thread and emits one `RuntimeWarning` per process. It is non-blocking: every PR 39 execution ran through this path and reproduced deterministically. If addressed, run `client.async_execute()` under `asyncio.Runner(loop_factory=asyncio.SelectorEventLoop)` on Windows only, as a separately versioned execution-path change validated by the vehicle tests, with a full Windows notebook execution only where that validation is already justified (for example, the PR 46 clean-kernel runs). Do not change global event-loop policies, use the deprecated `WindowsSelectorEventLoopPolicy`, or suppress the warning.
- [ ] **PuLP 4.0 compatibility.** The GLASS Router's ILP code constructs `LpVariable` objects directly and uses `PULP_CBC_CMD`, both deprecated in PuLP 3 ahead of PuLP 4.0. `requirements.txt` pins `pulp<4`, and `pytest.ini` keeps two targeted filters for these messages. Assess separately from frozen Stage 2 work.
- [ ] **Lightweight GLASS development utility.** A small command-driven tool (`run script`, `run notebook`, `run cell`, `apply test`, `scan artifacts`) that reuses existing Python, pytest, notebook, and artifact utilities, protects accepted artifacts, never retrains unexpectedly, and leaves Git operations to Git. Evaluate when its workflow benefit justifies implementation.
- [ ] Investigate advanced training strategies that survive preliminary scientific and computational screening.
- [ ] Avoid new framework dependencies or infrastructure without demonstrated need.
