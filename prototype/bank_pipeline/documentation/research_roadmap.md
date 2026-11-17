## Roadmap

### Phase 1 — Finish and Validate the Cascades

## PR 39 — Protocol Finalization, Artifact Regeneration, and Phase 1 Closure

**Objective:** Finalize Stage 1 calibration and feature naming, revalidate Stage 2 binning, standardize GLASS component naming, regenerate both cascades under their final contracts, and establish a validated pre-presentation baseline that concludes Phase 1.

**Execution principles:**

- Preserve the accepted PR 38 Stage 2 selector design, historical results, and provenance.
- Complete all methodology and naming changes before final artifact regeneration.
- Use train-only or properly cross-fitted evidence for decisions that affect modeling.
- Apply the same shared operating obligations to both cascade arms.
- Regenerate dependent artifacts in order, without repeating expensive runs unnecessarily.
- Keep final notebook presentation and restructuring in Phase 2.
- Perform comprehensive regression testing and clean-kernel notebook execution after integration stabilizes.

---

### Commit 1 — Planning, Dependency Audit, and Finalization Contract

**Objective:** Establish the PR 39 execution sequence, regeneration dependencies, evaluation requirements, and acceptance criteria before modifying the finalized cascades.

- [x] Review the PR 38 handoff, accepted run `c3_20261112_153333`, frozen Stage 2 selector, and outstanding PR 36 naming requirements.
- [x] Inventory Stage 1–4 artifacts, training scripts, shared contracts, downstream consumers, and notebook dependencies for both arms.
- [x] Identify which artifacts must be regenerated after calibration, feature naming, binning, or component naming changes.
- [x] Define the Stage 1 calibration comparison and decision criteria before examining new results.
- [x] Define the train-only Stage 2 binning evaluation, threshold-change criteria, and validation requirements.
- [x] Establish a versioned PR 39 artifact/provenance location without overwriting PR 38 evidence.
- [x] Define the Stage 1–3 regeneration sequence, Stage 4 dependencies, and 2×2 arbiter crossover protocol.
- [x] Establish testing, reproducibility, serialization, compatibility, and final notebook acceptance requirements.
- [x] Explicitly separate Phase 1 functional validation from Phase 2 notebook presentation and research preparation.
- [x] Finalize and version-control the PR 39 roadmap.

**Deliverable:** Approved PR 39 execution plan with frozen decision criteria, artifact dependencies, and acceptance requirements.

---

### Commit 2 — Stage 1 Calibration and Feature Contract Finalization

**Objective:** Resolve the remaining Stage 1 methodological and naming decisions before any downstream artifact regeneration.

#### Step 2A — Calibration Policy

- [x] Review existing sigmoid and isotonic calibration implementations, diagnostics, and historical evidence.
- [x] Compare calibration quality using leakage-safe training-side OOF evidence, including probability reliability, score resolution, and calibration stability.
- [x] Measure whether isotonic score granularity materially affects Stage 4 inputs or decision behavior using an appropriately isolated diagnostic comparison.
- [x] Evaluate the comparison against the criteria established in Commit 1.
- [x] Select one shared calibration policy for the GLASS and black-box arms.
- [x] Update calibration configuration, provenance requirements, shared contracts, and focused tests.
- [x] Freeze the Stage 1 calibration policy without regenerating the full cascade yet.

#### Step 2B — Feature Naming and Contract Migration

- [x] Rename the following Stage 1 features:
  - `cellular_crisis` → `cellular_downturn`
  - `euribor3m_local_rate` → `subscribe_rate_by_euribor_bin`
  - `dow_month_encoded` → `subscribe_rate_by_dow_month`
- [x] Update feature definitions, training code, imports, metadata, documentation, and dependent references.
- [x] Verify that naming-only changes preserve feature values and mathematical behavior.
- [x] Revalidate leakage-safe handling of label-derived feature encodings, especially within OOF folds.
- [x] Update serialization and compatibility expectations for the new versioned artifact contracts.
- [x] Run focused calibration, feature-contract, and compatibility tests.

**Deliverable:** Frozen Stage 1 calibration and feature contracts, with implementation changes validated and ready for regeneration.

---

### Commit 3 — Stage 2 Binning and GLASS Naming Finalization

**Objective:** Resolve final Stage 2 feature-bin semantics and the outstanding GLASS component naming cleanup before retraining downstream models.

#### Step 3A — NSD and ECI Binning Revalidation

- [x] Re-examine existing NSD and ECI thresholds, original design rationale, and current bin definitions.
- [x] Perform train-only, leakage-safe lift analysis using appropriate fold boundaries.
- [x] Measure per-bin support, class rates, lift, and fold-level stability.
- [x] Identify weak, unstable, redundant, or overlapping bins without assuming they should be removed.
- [x] Evaluate conjunction behavior and actual GLASS/RF router utility before merging or changing bins.
- [x] Retain existing thresholds unless the pre-established evidence criteria justify revisions.
- [x] If changes are justified, update bin definitions, encoders, feature contracts, and dependent tests.
- [x] Preserve the PR 38 selector algorithm and frozen shared routing obligations; version any changed input contract explicitly.
- [x] Freeze the final Stage 2 binning configuration.

#### Step 3B — GLASS Component Naming Standardization

- [x] Resolve naming inconsistencies between GLASS Router and GLASS Arbiter across modules, imports, interfaces, and documentation.
- [x] Update artifact paths, metadata, configuration identifiers, and downstream references where appropriate.
- [x] Preserve the distinction between GLASS and black-box model families.
- [x] Avoid changing modeling behavior during naming cleanup.
- [x] Validate compatibility and add focused regression tests only where changes require them.
- [x] Confirm the final Stage 1–2 contracts and naming structure are ready for regeneration.

**Deliverable:** Finalized Stage 2 binning and GLASS component naming, with the frozen selector design preserved.

---

### Commit 4 — Final Stage 1–3 Artifact Regeneration

**Objective:** Regenerate the affected upstream artifacts using the finalized methodology, while preserving historical baselines and validating the new training dependencies before Stage 4.

#### Step 4A — Regeneration Readiness and Upstream Execution

- [x] Verify that the Commit 2 and Commit 3 methodology/configuration freezes are complete.
- [x] Verify historical artifact identities and initialize the versioned PR 39 regeneration records.
- [x] Regenerate Stage 1 artifacts for both arms using the shared calibration and updated feature contracts.
- [x] Regenerate affected Stage 2 GLASS/RF artifacts using the final bins and frozen selector design.
- [x] Recompute Stage 2 OOF predictions where changed upstream inputs or feature contracts require it.
- [x] Preserve nested training, fold isolation, Pass 1 precedence, and Pass 2 remainder eligibility.
- [x] Regenerate affected Stage 3 artifacts and OOF predictions under the finalized upstream dependencies.
- [x] Reuse unchanged artifacts only when provenance and compatibility checks establish that reuse is valid.

#### Step 4B — Upstream Acceptance and Artifact Verification

- [x] Validate fold identities, OOF alignment, feature contracts, upstream fingerprints, and training-side isolation.
- [x] Verify both Stage 2 arms against the same frozen shared operating contract.
- [x] Record infeasible folds, solver outcomes, constraint compliance, and any deviations or limitations.
- [x] Validate serialization/deserialization, versions, metadata, artifact paths, and interface compatibility.
- [x] Verify that historical PR 38 results and model artifacts remain unchanged.
- [x] Run focused real-data and upstream regression tests.
- [x] Record an explicit Stage 1–3 acceptance decision before authorizing Stage 4 regeneration.

**Deliverable:** Regenerated and validated Stage 1–3 artifacts and OOF dependencies, accepted for final Stage 4 training.

---

### Commit 5 — Stage 4 Regeneration and Arbiter Crossover

**Objective:** Regenerate the final arbiters, execute the 2×2 crossover, and establish the complete cascade performance baseline under the finalized upstream contracts.

#### Step 5A — Final Stage 4 Integration

- [x] Verify accepted Stage 1–3 artifacts and OOF inputs from Commit 4.
- [x] Regenerate the GLASS and black-box Stage 4 arbiters using leakage-safe finalized upstream OOF evidence.
- [x] Verify input alignment, model-family ownership, routing conventions, and calibration dependencies.
- [x] Validate Stage 4 artifact provenance, metadata, versioning, and serialization.
- [x] Confirm both complete cascade arms execute under their finalized contracts.
- [x] Run focused Stage 4 integration tests before comparative evaluation.

#### Step 5B — 2×2 Crossover and Final Baseline

- [x] Re-run the registered 2×2 arbiter crossover using the finalized upstream artifacts.
- [x] Evaluate the registered combinations under consistent data splits, operating points, and comparison procedures.
- [x] Report predictive performance, calibration, abstention/coverage where applicable, and comparative uncertainty.
- [x] Examine Stage 2 contributions and GLASS/black-box differences without changing the frozen selector based on these results.
- [x] Revalidate mirrored contracts and identify any remaining integration limitations.
- [x] Record final model identities, crossover findings, and the pre-presentation cascade baseline.
- [x] Preserve earlier PR 38 comparisons as historical evidence rather than replacing them with PR 39 results.

**Stage 4 abstention metric accounting and interpretation**

- [x] Audit how Stage 4 computes accuracy, precision, recall, F1, F2, and ROC-AUC with and without abstention.
- [x] Verify whether abstained observations are excluded, counted as incorrect, or assigned another outcome in each reported metric.
- [x] Reconcile TP, FP, TN, FN, and abstention counts against the reported coverage and evaluation population.
- [x] Report retained-only metrics separately from whole-system metrics, including effective recall over all actual positives and per-class retention/abstention rates.
- [x] Investigate the observed accuracy decline alongside improvements in precision, recall, and ROC-AUC, accounting for class-prevalence differences and operating thresholds.
- [x] Correct any confirmed metric-accounting or labeling errors without altering the frozen Stage 4 training and evaluation protocol.
- [x] Document the interpretation and limitations of abstention-conditioned performance in the final cascade baseline.

**Deliverable:** Regenerated Stage 4 arbiters, completed crossover analysis, and the final pre-presentation baseline for both cascades.

---

### Commit 6 — Final Validation, Notebook Execution, and Phase 1 Freeze

**Objective:** Complete the final integrated verification, validate reproducibility from clean notebook kernels, and formally close Phase 1.

#### Step 6A — Final Regression and Reproducibility Validation

- [ ] Run the relevant targeted, regression, real-data, and end-to-end validation suites against the final implementation.
- [ ] Verify shared and arm-specific contracts across Stages 1–4.
- [ ] Validate training/test isolation, OOF provenance, model versions, serialization, fingerprints, and configuration identities.
- [ ] Verify the final GLASS/RF Stage 2 constraints and three-state routing semantics.
- [ ] Confirm the 2×2 crossover results and final baseline references remain reproducible.
- [ ] Check historical baseline preservation and confirm no unexpected artifact or model-family changes.
- [ ] Resolve material failures before freezing Phase 1; record any limitations that do not block acceptance.

#### Step 6B — Clean-Kernel Notebook Runs and Phase 1 Handoff

- [ ] Execute the relevant cascade notebooks from fresh kernels using the finalized implementation and artifact contracts.
- [ ] Verify that notebook cells run in order without relying on stale kernel state or undocumented local variables.
- [ ] Validate notebook references, artifact loading, output consistency, and integration with the finalized pipeline.
- [ ] Resolve execution-breaking notebook issues without undertaking the Phase 2 presentation redesign.
- [ ] Record final artifact identities, accepted baselines, test results, notebook execution outcomes, and known limitations.
- [ ] Freeze the Phase 1 cascade implementations and validated research baseline.
- [ ] Document the Phase 2 handoff, including notebook restructuring, baseline presentation, and remaining research deliverables.
- [ ] Reconcile the PR 39 roadmap and confirm final acceptance.

**Deliverable:** Validated, reproducible GLASS and black-box cascades, finalized artifact contracts, clean-kernel execution evidence, and an approved Phase 1 baseline.

---

### PR 39 — Final Completion Boundary

- [x] **Commit 1 complete:** PR 39 roadmap, dependency audit, evaluation rules, and regeneration sequence finalized.
- [x] **Commit 2 complete:** Shared Stage 1 calibration and feature naming frozen.
- [x] **Commit 3 complete:** Stage 2 binning and GLASS component naming finalized.
- [x] **Commit 4 complete:** Final Stage 1–3 artifacts regenerated and accepted.
- [x] **Commit 5 complete:** Stage 4 arbiters regenerated, 2×2 crossover completed, and cascade baselines established.
- [ ] **Commit 6 complete:** Final regression, provenance, reproducibility, and clean-kernel notebook validation completed.
- [ ] **PR 39 complete:** Both cascade implementations and their final Phase 1 artifact contracts frozen, with Phase 2 handoff established.


**Out of scope:** Complementarity-aware selector redesign, additional Stage 2 training-objective experiments, notebook presentation restructuring, final notebook slicing and organization, paper-level independent validation, and new research extensions. These belong to later phases.

**Execution policy:** Use the Patch Method throughout. Favor one focused implementation patch per substantial step, with additional patches only for genuine corrections. Run focused tests during development and reserve expensive complete regression/notebook execution for the final integrated state wherever the acceptance requirements permit.

## PR 40 — Test Warning Cleanup and Maintenance

**Objective:** Eliminate recurring, actionable test warnings by correcting their underlying causes rather than suppressing them. Improve test clarity and maintainability while preserving the finalized PR 39 modeling contracts, behavior, artifacts, and reproducibility.

### Commit 1 — Warning Diagnosis, Corrections, and Validation

**Objective:** Identify and resolve the recurring scikit-learn MLP batch-size warnings and any closely related, actionable warnings through minimal, reviewable changes.

#### Step 1A — Warning Inventory and Root-Cause Analysis

- [ ] Start from the completed PR 39 baseline with a clean working tree.
- [ ] Reproduce and inventory recurring warnings from the relevant test suites.
- [ ] Identify warning sources, affected tests, triggering configurations, and underlying causes.
- [ ] Distinguish harmless test-fixture issues from genuine implementation, compatibility, or deprecation concerns.
- [ ] Determine the smallest safe correction for each actionable warning.

#### Step 1B — Targeted Warning Corrections

- [ ] Correct invalid or unnecessary MLP batch-size configurations, prioritizing synthetic test fixtures.
- [ ] Preserve the effective training behavior and numerical expectations of affected tests.
- [ ] Address other recurring warnings only where the cause is verified and the correction is similarly narrow.
- [ ] Do not suppress warnings through filters, blanket pytest settings, or equivalent workarounds.
- [ ] Preserve production hyperparameters, model-family behavior, frozen contracts, and existing artifacts.
- [ ] Complete the corrections in one focused implementation patch; use a second only if necessary.

#### Step 1C — Regression and Final Verification

- [ ] Rerun affected tests and confirm their original assertions and contracts remain valid.
- [ ] Verify that targeted warnings no longer occur without suppression.
- [ ] Run relevant regression tests and confirm no new failures or warnings were introduced.
- [ ] Verify that no model artifacts, historical evidence, or frozen research records were modified.
- [ ] Review the final diff, confirm the changes remain maintenance-only, and commit the cleanup.

**Deliverable:** A cleaner test suite with the identified warning causes corrected, passing regression tests, unchanged modeling behavior, and no warning suppression. One maintenance commit, with one or at most two implementation patches.

---

### PR 40 — Final Completion Boundary

- [ ] **Commit 1 complete:** Recurring actionable warnings diagnosed, corrected, and validated without altering frozen modeling contracts or artifacts.
- [ ] **PR 40 complete:** Maintenance changes committed, regression tests passing, and warning cleanup documented.
- [ ] **Phase 1 complete:** GLASS and black-box cascades are operational, integrated, validated, and ready for final notebook organization and research presentation.

**Out of scope:** Model retraining, hyperparameter optimization, feature engineering changes, calibration revisions, Stage 2 selector changes, artifact regeneration, notebook redesign, and unrelated refactoring.

### Phase 2 — Finalize the Notebooks and Freeze the Cascades

#### PR 41 — Black-Box Notebook Presentation

- [ ] Condense verbose outputs into concise stage summaries
- [ ] Build shared plotting infrastructure
- [ ] Add stage-level and cascade-level visuals
- [ ] Rewrite the cascade Venn trace to treat router abstention as a third state
- [ ] Standardize saved cascade artifacts for downstream comparison loading
- [ ] Final clean-kernel rerun

#### PR 42 — GLASS Notebook Presentation and Cascade Freeze

- [ ] Condense and consolidate GLASS notebook outputs
- [ ] Add cascade visuals through the shared plotting module
- [ ] Export GLASS artifacts in the standardized cascade format
- [ ] Verify both cascades use the finalized methodology and contracts
- [ ] Final clean-kernel rerun of both cascade notebooks
- [ ] Freeze comparison-ready artifacts
- [ ] Tag a frozen release of both cascade implementations

### Phase 3 — Comparisons Pipeline

#### PR 43 — Comparison Scaffolding

- [ ] Load frozen artifacts without retraining
- [ ] Enforce split-fingerprint and feature-contract provenance
- [ ] Extend `pytest` coverage to comparison loaders
- [ ] Reject stale or mismatched cascade artifacts
- [ ] Freeze final comparison questions, primary metrics, and analysis plan before the final comparison rerun
- [ ] Establish shared comparison tables / result schemas

#### PR 44 — Cascade vs Cascade

**Cascade comparison**
- [ ] GLASS vs black-box comparison
- [ ] Overall cascade performance
- [ ] Stage-level comparison
- [ ] Routing and abstention comparison
- [ ] 2×2 Stage 4 arbiter crossover analysis
- [ ] Matched flag-budget comparison
- [ ] Matched abstention-coverage comparison
- [ ] Coverage-prescription comparison:
  - pre-registered coverage grid (30–80%)
  - train-OOF-set cutoffs applied to test
  - retained and whole-system metrics per coverage
  - risk–coverage curves / AURC
  - paired bootstrap confidence intervals
- [ ] Stage 4 abstention drift analysis
- [ ] Evaluate coverage-targeted abstention cutoffs
- [ ] Report retained base rate and per-class abstention
- [ ] Test Meta-XGB confidence in calibrated space
- [ ] Agreement / disagreement / unique-correct analysis
- [ ] Between-arm OOF score correlation per stage
- [ ] Paired bootstrap confidence intervals
- [ ] Compare calibrated probabilities where calibration-quality metrics are interpreted

#### PR 45 — Standalone Model Baselines

- [ ] Glass-box standalone models:
  - Logistic Regression
  - Classical GAM
  - EBM
  - Shallow Decision Tree
  - Sparse Rule List / RuleFit-style model
- [ ] Black-box standalone models:
  - XGBoost
  - Random Forest
  - MLP
  - RBF SVM
  - CatBoost or LightGBM
- [ ] Same audited feature population, split, folds, tuning budget, calibration policy, and evaluation protocol as applicable
- [ ] Generate OOF train-side predictions and held-out test predictions under the same leakage safeguards
- [ ] Test whether either cascade adds predictive value beyond its strongest standalone alternatives

#### PR 46 — Robustness and Ablations

- [ ] 5-fold vs 10-fold sensitivity on both arms
- [ ] Repeated seeds / splits for model-family claims
- [ ] Test whether GLASS/black-box differences persist across protocol changes
- [ ] Stage-removal ablations
- [ ] Threshold sensitivity
- [ ] Abstention / coverage sensitivity
- [ ] Tuning-budget sensitivity
- [ ] Temporal-split sensitivity check (train on earlier months, test on later) — macro features are monthly and identify the contact period

**Stacking methodology sensitivity (F39)**

- [ ] Assess indirect label dependence in Meta-XGBoost's upstream OOF stacking features
- [ ] Evaluate the feasibility of fully nested upstream fitting within each meta-fold
- [ ] Compare standard OOF stacking against fully nested stacking where computationally feasible
- [ ] Measure effects on calibration, thresholds, abstention, and predictive performance
- [ ] Document the methodological trade-offs and implications for the GLASS vs. black-box comparison

**Deferred from PR 36**

- [ ] Evaluate sensitivity to upstream stacking dependence through robustness experiments

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
- [ ] Measure GLASS/RF router sensitivity to candidate binning changes, including coverage, constraint feasibility, complementarity, and computational cost.
- [ ] Compare candidate configurations against the frozen PR 39 binning baseline using predefined evaluation criteria.
- [ ] Determine whether the existing bins should remain unchanged or whether evidence warrants a separately versioned methodology revision.
- [ ] If revision is justified, document the required feature-contract changes, G13 leakage audit, downstream artifact regeneration, and follow-up PR rather than silently modifying frozen artifacts.

**Deliverable:** Reproducible NSD/ECI binning sensitivity analysis, quantified explanation of the observed lift drift, and an evidence-based recommendation to retain or revisit the frozen binning configuration.

### Phase 4 — Paper Readiness

#### PR 47 — Paper Exports and Final Freeze

- [ ] Export final figures and tables
- [ ] Final end-to-end validation
- [ ] Freeze implementation
- [ ] Freeze experiment artifacts
- [ ] Prepare paper-ready result tables