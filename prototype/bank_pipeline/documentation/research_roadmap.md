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

### Commit 2 — Research Roadmap Expansion and Finalization

**Objective:** Update `documentation/research_roadmap.md` to incorporate the completed PR 39 baseline, revised Phase 2 research direction, and five-notebook research plan. Finalize the forward-looking roadmap without implementing new research or restructuring the repository.

- [ ] Review the existing research roadmap and reconcile completed PR 38–39 milestones.
- [ ] Incorporate the accepted Phase 1 baseline, four-stage cascade architectures, and four-way Stage 4 arbiter crossover.
- [ ] Integrate the revised five-notebook research plan, preserving the intended responsibilities and experimental boundaries of each notebook.
- [ ] Organize Phase 2 research priorities, including:
  - Controlled GLASS versus black-box comparisons.
  - Model-family-specific optimization under shared evaluation obligations.
  - Stage 4 training-objective and hyperparameter optimization.
  - Confidence, abstention, coverage, and whole-system performance analysis.
  - Investigation of Stage 4 crossover findings and structural differences.
  - Independent evaluation, leakage safeguards, and scientific validity.
  - Notebook presentation, reusable research modules, and preparation of publication-quality evidence.
- [ ] Preserve the distinction between completed experiments, accepted findings, research limitations, and future hypotheses.
- [ ] Record future maintenance work, including PR 41 structural cleanup and the proposed lightweight GLASS development environment, without implementing either.
- [ ] Remove obsolete or redundant planning entries where appropriate without discarding historical research evidence.
- [ ] Review the finalized roadmap for completeness, consistency, and realistic execution order.
- [ ] Commit the documentation update as one file change.

**Deliverable:** A substantially finalized research roadmap reflecting the accepted Phase 1 baseline and planned Phase 2 research program. One documentation-only commit modifying `documentation/research_roadmap.md`.

---

### PR 40 — Final Completion Boundary

- [x] **Commit 1 complete:** Recurring actionable test warnings diagnosed, corrected, and validated without altering frozen modeling contracts or artifacts.
- [ ] **Commit 2 complete:** Research roadmap updated with the revised five-notebook plan, Phase 2 priorities, and future engineering work.
- [ ] **PR 40 complete:** Both commits validated, pushed, reviewed, and merged.
- [ ] **Phase 1 complete:** GLASS and black-box cascades operational, integrated, validated, and ready for Phase 2 research and presentation.

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