## Roadmap

### Phase 1 — Finish and Validate the Cascades

## PR 38 — Stage 2 Cross-Fit and Router Complementarity

**Objective:** Execute the corrected Stage 2 cross-fitting protocol established in PR 37, evaluate router redundancy and complementarity, determine whether complementarity-aware selection is justified, and freeze the Stage 2 selector implementation for PR 39.

### Commit 1 — Planning, Roadmap, and Execution Contract

**Objective:** Establish the PR 38 execution sequence, requirements, dependencies, and acceptance criteria before implementation or corrected training.

- [x] Review the completed PR 37 handoff, frozen Stage 2 contract, F17/F31 decision records, and pre-registered evaluation protocol.

- [x] Map F17 gates G1–G13 and F31 requirements R1–R8 to their PR 38 execution and validation steps.

- [x] Identify existing PR 37 functionality that can be reused versus work requiring new implementation.

- [x] Define the corrected GLASS/RF execution sequence and required training-side inputs.

- [x] Define the preflight, primary rerun, sensitivity-analysis, and complementarity-ablation boundaries.

- [x] Establish test, reproducibility, artifact-provenance, and final acceptance requirements.

- [x] Preserve the frozen PR 37 operating contract and historical baselines without retroactive modification.

- [x] Define the PR 38/PR 39 boundary: selector design freezes in PR 38; final binning, calibration, and cascade regeneration remain PR 39.

- [x] Finalize and version-control the updated PR 38 roadmap.

**Deliverable:** Reviewed, approved, and version-controlled PR 38 roadmap.

---

### Commit 2 — Corrected-Selector Preflight and Safeguards

**Objective:** Complete all required implementation and feasibility gates before initiating the corrected five-fold primary reruns.

#### **Step 2A — Configuration, Integration, and Leakage Safeguards**

- [x] Before any real-data fit (including the G13 audit), complete handoff Phase 0:

  - Verify the archived snapshot and historical baseline (`--verify`, archive `--check`).

  - Pin `$BANK_ARTIFACT_MANIFEST`.

  - Inventory historical router artifacts.

  - Establish a separate versioned PR 38 output location.

- [x] Validate G1: frozen primary configuration, including required `nested_t1` and `fold_nested_rf` safeguards.

- [x] Validate G2: correct GLASS/RF wiring, arm-specific ownership, and Pass 2 training on each arm's own OOF Pass 1 remainder.

- [x] Validate G5: required persistence, intermediate records, and training-side provenance support, including F31 R1, R2, and R5 records that must be produced inside the cross-fit.

- [x] Validate G8: explicit rejection of infeasible fits, missing valid incumbents, insufficient remainder support, and invalid configurations.

- [x] Validate G9: independent arm configuration and identical frozen shared-contract obligations.

- [x] Complete G13: audit label-derived feature engineering for own-label and cross-fold leakage, including density, curvature, and day × month encoders.

  - [x] Correct and validate any confirmed material own-label dependency, and decide whether the primary adopts the correction, before any five-fold primary run.

- [x] Add or update targeted regression tests for any implementation changes.

#### **Step 2B — Real-Data Feasibility and Primary-Run Authorization**

- [x] Execute G3 real-data feasibility preflight under the frozen operating contract.

- [x] Verify Pass 1 and Pass 2 constraint feasibility, support requirements, and solver outcomes.

- [x] Verify training population boundaries, fold construction, and upstream OOF dependencies.

- [x] Confirm historical baselines and existing artifacts remain unchanged.

- [x] Run targeted regression and relevant integration tests.

- [x] Record preflight outcomes, limitations, and explicit pass/fail decisions.

- [x] Authorize the primary five-fold reruns only after required gates pass.

**Deliverable:** Validated corrected-selector implementation with documented primary-run readiness.

---

### Commit 3 — Corrected Stage 2 Cross-Fit and Complementarity Evaluation

**Objective:** Implement, execute, and validate corrected Stage 2 cross-fitting, complementarity analysis, and sensitivity experiments using the pre-registered PR 37 protocol. Complete all required testing, artifact verification, and structural acceptance before closing Commit 3.

#### Step 3A — Corrected Five-Fold Primary Reruns

- [x] Run corrected GLASS Router under the frozen Stage 2 operating contract.
- [x] Run corrected RF Router under the identical shared contract.
- [x] Complete the full five-fold Stage 2 outer cross-fit for both arms.
- [x] Verify G4: fold isolation, training-population boundaries, and OOF provenance.
- [x] Verify G7: deterministic reproducibility and fold identities.
- [X] Verify G10: versioned artifacts, configuration identities, and provenance pins.
- [x] Verify G11: preservation of historical baselines and previously frozen evidence.
- [x] Record mandatory F31 requirements R1–R5, R7, and R8; R6 band-level spread remains optional.
- [x] Verify routing semantics, Pass 1 precedence, Pass 2 remainder eligibility, and abstention accounting.
- [x] Record every fold's fit status, including infeasible fits, solver nonoptimality, greedy fallbacks, and unsuccessful fits.
- [x] Require all five folds accepted in both arms before any outer-OOF performance result is examined, without relaxing the contract.
- [x] Regenerate corrected Stage 2 OOF and required model artifacts.
- [x] Validate artifact serialization, fingerprints, metadata, and interface compatibility.
- [x] Pass structural acceptance V1–V8, including population and upstream identity verification, before any comparative analysis.

#### Step 3B — Corrected-Router Complementarity Analysis

- [x] Re-run the three-state router trace:
  - `PASS_1_NOT_SUBSCRIBE`
  - `PASS_2_SUBSCRIBE`
  - `ABSTAIN`
- [x] Re-measure Pass 2 unique-positive discovery at matched flag rates.
- [x] Re-measure Pass 1 false-positive suppression against frozen LR/EBM operating points.
- [x] Measure GLASS/RF routing coverage, precision, NPV, leakage, abstention, and class-conditional behavior.
- [x] Report observed outer-OOF constraint compliance and shortfalls under protocol section 11, separately from fit-time status.
- [x] Compare pre-correction versus post-correction redundancy and routed-row identities.
- [x] Measure GLASS Router versus LR/EBM decision overlap, score-conditional routing, and disagreement using the existing Step 2 diagnostic definitions.
- [x] Evaluate comparative results using the pre-registered metrics, paired comparisons, and uncertainty procedures.
- [x] Distinguish implementation correctness, contract feasibility, and empirical predictive performance.

#### Step 3C — F17 Sensitivity and Final Validation

- [x] Execute G12 sensitivity analysis S1: GLASS `full` versus `oof_remainder`, descriptive only, after the applicable primary-run gates pass.
- [x] Verify G6 Pass 1 invariance during S1; reject S1 if its required invariance fails.
- [x] Report S1 separately from the corrected primary comparison.
- [x] Complete regression, provenance, reproducibility, and artifact-integrity verification.
- [x] Document corrected primary results, sensitivity findings, remaining limitations, and implications for complementarity-aware selection.

**Deliverable:** Corrected and structurally accepted GLASS/RF five-fold cross-fitted artifacts, completed complementarity and sensitivity analyses, passing regression and validation tests, verified provenance, and documented findings ready for Commit 4.

---

### Commit 4 — Housekeeping Research Records and Report Organization

**Objective:** Organize PR 38 experimental records, JSON files, and validation reports into a consistent, easily navigable structure without changing pipeline functionality.

- [ ] Establish clear locations for versioned methodology records, generated execution evidence, test reports, and final research outputs.
- [ ] Organize records consistently by PR, run ID, and purpose.
- [ ] Ensure important research evidence is version-controlled and accessible for read-only review on GitHub.
- [ ] Keep temporary logs, redundant generated outputs, and large model artifacts out of Git where appropriate.
- [ ] Preserve existing records, provenance, fingerprints, and write-once safeguards.
- [ ] Update file paths and references only where necessary, without modifying algorithms, tests, validation criteria, or experimental behavior.
- [ ] Verify affected record-storage and retrieval operations with focused checks only.

**Deliverable:** Evidence-based complementarity-ablation decision, selected Stage 2 implementation and artifacts validated and frozen, and PR 39 regeneration requirements documented.

---

### Commit 5 — Complementarity Ablation and Final Stage 2 Freeze

**Objective:** Prepare and validate the optional complementarity-aware selection experiment and final-selector workflow.

**Execution scheduling:** Implement and test the conditional experiment and freeze workflow in Commit 4. Complete evidence-dependent evaluation and selector decisions.

#### **Step 5A — Complementarity-Aware Selection Experiment**

- [ ] Review Commit 3 evidence against the pre-registered criteria to determine whether the optional experiment is warranted.

- [ ] If warranted, implement and evaluate complementarity-aware selection using train-side OOF upstream predictions only.

- [ ] Evaluate rule objectives favoring:

  - [ ] Positive cases missed by Stage 1 LR.

  - [ ] Rows where LR and EBM disagree.

- [ ] Compare standard union-coverage selection against complementarity-aware selection.

- [ ] Measure unique-positive discovery, matched flag-budget performance, coverage, precision, leakage, uncertainty, and computational cost.

- [ ] Verify training-side isolation and preservation of the frozen operating contract.

- [ ] Treat complementarity-aware selection as a secondary ablation unless the pre-registered adoption criteria are satisfied.

- [ ] If adoption is justified, implement an equivalent RF Router objective or explicitly document the methodological asymmetry.

- [ ] Record an evidence-based adoption or rejection decision without changing the frozen criteria retroactively.

#### **Step 5B — Final Selector Decision and Artifact Freeze**

- [ ] Select the final Stage 2 selector design based on the pre-registered comparisons and acceptance hierarchy.

- [ ] If complementarity-aware selection is adopted, rerun the required GLASS/RF cross-fitting and evaluation under the frozen contract and a later, versioned evaluation protocol; preserve the Commit 3 primary results.

- [ ] If the original selector is retained, re-verify and freeze the accepted Commit 3 artifacts without unnecessary retraining.

- [ ] Regenerate final PR 38 Stage 2 OOF and required test artifacts only when the selected design requires regeneration.

- [ ] Verify final fold identities, OOF provenance, constraint compliance, routing semantics, and mirrored contracts.

- [ ] Validate artifact versions, fingerprints, serialization, upstream compatibility, and reproducibility.

- [ ] Confirm historical baseline preservation and no unintended model-artifact or notebook changes.

- [ ] Run final targeted, regression, and applicable real-data validation.

- [ ] Freeze the selected Stage 2 implementation, configuration, and PR 38 research artifacts.

- [ ] Document the final selector decision, experimental evidence, methodological limitations, and PR 39 regeneration requirements.

- [ ] Confirm all PR 38 acceptance criteria and reconcile the roadmap.

**Deliverable:** Tested optional ablation and selector-freeze implementation, ready for an evidence-based decision in Commit 5.

---

### PR 38 — Final Completion Boundary

- [x] **Commit 1 complete:** PR 38 roadmap and validation requirements reviewed, finalized, and version-controlled.

- [x] **Commit 2 complete:** Preflight, leakage safeguards, and corrected-run readiness validated.

- [x] **Commit 3 complete:** Corrected five-fold GLASS/RF cross-fitting, complementarity analysis, and sensitivity experiments executed, validated, and structurally accepted.

- [ ] **Commit 4 complete:** Research records, JSON evidence, and validation reports reorganized into a consistent, structured storage system; relevant evidence version-controlled, with provenance, reproducibility, and existing pipeline functionality preserved.

- [ ] **Commit 5 complete:** Optional complementarity ablation evaluated against pre-registered criteria, final Stage 2 selector selected and frozen, and PR 39 handoff requirements documented.

- [ ] **PR 38 complete:** Stage 2 selector research concluded, validated artifacts and provenance preserved, and PR 39 handoff established.

**Out of scope:** Stage 1 calibration changes, Stage 1 feature renaming, final Stage 2 binning revisions, Stage 4 regeneration, final cascade notebook presentation, and paper-level independent validation. These remain assigned to subsequent PRs.

#### PR 39 — Protocol Finalization and Artifact Regeneration

**Stage 1 calibration**

- [ ] Resolve sigmoid vs isotonic calibration once for both arms
- [ ] Measure whether isotonic score granularity materially affects Stage 4
- [ ] Freeze the shared Stage 1 calibration policy

**Stage 1 feature naming**

- [ ] Rename:
  - `cellular_crisis` → `cellular_downturn`
  - `euribor3m_local_rate` → `subscribe_rate_by_euribor_bin`
  - `dow_month_encoded` → `subscribe_rate_by_dow_month`
- [ ] Update feature contracts, documentation, artifacts, and downstream references

**Stage 2 binning revalidation**

- [ ] Re-evaluate NSD and ECI bin thresholds using train-only, leakage-safe lift analysis
- [ ] Compare current lift behavior against the original threshold rationale
- [ ] Measure per-bin support, lift, and fold stability
- [ ] Check conjunction / router utility before merging or removing weak bins
- [ ] Re-derive thresholds where drift materially changes the intended bin semantics
- [ ] Freeze the final Stage 2 binning contract before artifact regeneration

**Final regeneration**

- [ ] Regenerate affected Stage 1–3 artifacts under the finalized contracts
- [ ] Rerun Stage 4 with finalized upstream artifacts
- [ ] Re-run the 2×2 arbiter crossover
- [ ] Revalidate mirrored contracts after all methodology changes
- [ ] Record the final pre-presentation cascade baseline

**Deferred from PR 36**

- [ ] Standardize GLASS Router / GLASS Arbiter naming across modules, imports, artifact paths, metadata, and documentation where appropriate. Preserve model-family identities and validate compatibility before final artifact regeneration.

### Phase 2 — Finalize the Notebooks and Freeze the Cascades

#### PR 40 — Black-Box Notebook Presentation

- [ ] Condense verbose outputs into concise stage summaries
- [ ] Build shared plotting infrastructure
- [ ] Add stage-level and cascade-level visuals
- [ ] Rewrite the cascade Venn trace to treat router abstention as a third state
- [ ] Standardize saved cascade artifacts for downstream comparison loading
- [ ] Final clean-kernel rerun

#### PR 41 — GLASS Notebook Presentation and Cascade Freeze

- [ ] Condense and consolidate GLASS notebook outputs
- [ ] Add cascade visuals through the shared plotting module
- [ ] Export GLASS artifacts in the standardized cascade format
- [ ] Verify both cascades use the finalized methodology and contracts
- [ ] Final clean-kernel rerun of both cascade notebooks
- [ ] Freeze comparison-ready artifacts
- [ ] Tag a frozen release of both cascade implementations

### Phase 3 — Comparisons Pipeline

#### PR 42 — Comparison Scaffolding

- [ ] Load frozen artifacts without retraining
- [ ] Enforce split-fingerprint and feature-contract provenance
- [ ] Extend `pytest` coverage to comparison loaders
- [ ] Reject stale or mismatched cascade artifacts
- [ ] Freeze final comparison questions, primary metrics, and analysis plan before the final comparison rerun
- [ ] Establish shared comparison tables / result schemas

#### PR 43 — Cascade vs Cascade

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

#### PR 44 — Standalone Model Baselines

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

#### PR 45 — Robustness and Ablations

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

### Phase 4 — Paper Readiness

#### PR 46 — Paper Exports and Final Freeze

- [ ] Export final figures and tables
- [ ] Final end-to-end validation
- [ ] Freeze implementation
- [ ] Freeze experiment artifacts
- [ ] Prepare paper-ready result tables