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

### Commit 4 — Research Report Consolidation and Housekeeping

**Objective:** Reduce PR 38's excessive generated CSV/JSON reports into a compact, navigable research reporting structure. Preserve the scientific information, independently required acceptance/provenance evidence, and reproducibility—not every redundant file. Complete the work with one implementation patch, or at most two if compatibility fixes genuinely require a follow-up. No retraining or changes to modeling behavior.

#### Step 4A — Inventory and Owner Decisions (read-only; no patch)

- [x] Inventory both local PR 38 research-log runs, including the accepted c3_20261112_153333 and historical c3_20261112_134824 directories. Include file counts, relative paths, sizes, purpose, and references/dependencies.

- [x] Identify which CSV/JSON outputs duplicate information, can be consolidated, are temporary, or must remain independent for acceptance, checksum verification, or reproducibility.

- [x] Propose a concise per-group disposition: consolidate / retain on GitHub / retain locally / discard. Review consequential decisions with the owner before deleting or changing any evidence.

- [x] Choose compact report groupings. Aim for approximately 10 or fewer primary CSV tables and 5–8 primary JSON reports where practical; independent validation/provenance files are exempt. These are our optimistic targets, as opposed to acceptance gates.

#### Step 4B — Consolidation, Cleanup, and Focused Verification (1–2 patches total)

- [x] Patch 1 — Main consolidation: Produce consolidated, clearly named research tables and JSON reports from existing validated outputs; reduce overlapping reporting; curate GitHub-facing results; include a short index identifying the authoritative run, report contents, and where supporting evidence lives.

- [x] Where feasible within scope, adjust the reporting/writer paths so future runs do not recreate unnecessary fragmentation. Retain detailed internal evidence only where technically or scientifically necessary. Do not rewrite immutable accepted records or break checksum/provenance chains.

- [x] Carry out the owner-approved cleanup. Discard truly redundant or temporary files once their lack of required dependencies is established; retain locally only what is necessary for verification or worthwhile historical evidence. Do not create an automatic permanent archive of every original file.

- [x] Keep necessary acceptance, G7/G13, G10/F31, S1, authorization, manifest, and checksum evidence independently verifiable where required; preserve run IDs, fingerprints, frozen configurations, and write-once safeguards. Large model binaries stay Git-ignored.

- [x] Patch 2 — Only if necessary: Apply narrowly scoped path/reader/verification compatibility changes that cannot reasonably be included in Patch 1. Otherwise finish in one patch.

- [x] Run focused record-generation/retrieval tests and relevant existing integrity checks only. No primary cross-fit, complementarity rerun, S1 rerun, new modeling experiments, or broad test campaign.

- [x] Commit and push the approved consolidated reporting and necessary supporting evidence as one housekeeping commit. Keep Commit 5's selector/ablation decisions out of this commit.

**Deliverable:** A substantially smaller, coherent set of GitHub-accessible PR 38 research reports; essential evidence remains verifiable, redundant files are removed where approved, and any affected reporting/storage paths pass focused checks. Maximum two implementation patches; one Commit 4 Git commit.

---

### Commit 5 — Complementarity Ablation and Final Stage 2 Freeze

**Objective:** Decide whether the pre-registered complementarity ablation is warranted, execute it only if justified, and freeze the final Stage 2 selector and artifacts for PR 39.

**Execution scheduling:** All ablation decisions, any justified experimental implementation, and the final selector freeze belong to Commit 5—not Commit 4.

#### **Step 5A — Complementarity-Aware Selection Experiment**

- [x] Review Commit 3 evidence against the pre-registered criteria to determine whether the optional experiment is warranted.

- [x] If warranted, implement and evaluate complementarity-aware selection using train-side OOF upstream predictions only.

- [x] Evaluate rule objectives favoring:

  - [x] Positive cases missed by Stage 1 LR.

  - [x] Rows where LR and EBM disagree.

- [x] Compare standard union-coverage selection against complementarity-aware selection.

- [x] Measure unique-positive discovery, matched flag-budget performance, coverage, precision, leakage, uncertainty, and computational cost.

- [x] Verify training-side isolation and preservation of the frozen operating contract.

- [x] Treat complementarity-aware selection as a secondary ablation unless the pre-registered adoption criteria are satisfied.

- [x] If adoption is justified, implement an equivalent RF Router objective or explicitly document the methodological asymmetry.

- [x] Record an evidence-based adoption or rejection decision without changing the frozen criteria retroactively.

#### **Step 5B — Final Selector Decision and Artifact Freeze**

- [x] Select the final Stage 2 selector design based on the pre-registered comparisons and acceptance hierarchy.

- [x] If complementarity-aware selection is adopted, rerun the required GLASS/RF cross-fitting and evaluation under the frozen contract and a later, versioned evaluation protocol; preserve the Commit 3 primary results.

- [x] If the original selector is retained, re-verify and freeze the accepted Commit 3 artifacts without unnecessary retraining.

- [x] Regenerate final PR 38 Stage 2 OOF and required test artifacts only when the selected design requires regeneration.

- [x] Verify final fold identities, OOF provenance, constraint compliance, routing semantics, and mirrored contracts.

- [x] Validate artifact versions, fingerprints, serialization, upstream compatibility, and reproducibility.

- [x] Confirm historical baseline preservation and no unintended model-artifact or notebook changes.

- [x] Run final targeted, regression, and applicable real-data validation.

- [x] Freeze the selected Stage 2 implementation, configuration, and PR 38 research artifacts.

- [x] Document the final selector decision, experimental evidence, methodological limitations, and PR 39 regeneration requirements.

- [x] Confirm all PR 38 acceptance criteria and reconcile the roadmap.

**Deliverable:** Evidence-based complementarity-ablation decision; any justified ablation completed and evaluated; final Stage 2 selector, accepted artifact identities, and PR 39 regeneration/handoff requirements documented and frozen.

---

### PR 38 — Final Completion Boundary

- [x] **Commit 1 complete:** PR 38 roadmap and validation requirements reviewed, finalized, and version-controlled.

- [x] **Commit 2 complete:** Preflight, leakage safeguards, and corrected-run readiness validated.

- [x] **Commit 3 complete:** Corrected five-fold GLASS/RF cross-fitting, complementarity analysis, and sensitivity experiments executed, validated, and structurally accepted.

- [x] **Commit 4 complete:** Redundant research reporting consolidated into a compact GitHub-facing record set; owner-approved cleanup completed, required provenance preserved, and reporting compatibility verified in one or two patches.

- [x] **Commit 5 complete:** Optional complementarity ablation evaluated against pre-registered criteria, final Stage 2 selector selected and frozen, and PR 39 handoff requirements documented.

- [x] **PR 38 complete:** Stage 2 selector research concluded, validated artifacts and provenance preserved, and PR 39 handoff established.

**Out of scope:** Stage 1 calibration changes, Stage 1 feature renaming, final Stage 2 binning revisions, Stage 4 regeneration, final cascade notebook presentation, and paper-level independent validation. These remain assigned to subsequent PRs.

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

- [ ] Review the PR 38 handoff, accepted run `c3_20261112_153333`, frozen Stage 2 selector, and outstanding PR 36 naming requirements.
- [ ] Inventory Stage 1–4 artifacts, training scripts, shared contracts, downstream consumers, and notebook dependencies for both arms.
- [ ] Identify which artifacts must be regenerated after calibration, feature naming, binning, or component naming changes.
- [ ] Define the Stage 1 calibration comparison and decision criteria before examining new results.
- [ ] Define the train-only Stage 2 binning evaluation, threshold-change criteria, and validation requirements.
- [ ] Establish a versioned PR 39 artifact/provenance location without overwriting PR 38 evidence.
- [ ] Define the Stage 1–3 regeneration sequence, Stage 4 dependencies, and 2×2 arbiter crossover protocol.
- [ ] Establish testing, reproducibility, serialization, compatibility, and final notebook acceptance requirements.
- [ ] Explicitly separate Phase 1 functional validation from Phase 2 notebook presentation and research preparation.
- [ ] Finalize and version-control the PR 39 roadmap.

**Deliverable:** Approved PR 39 execution plan with frozen decision criteria, artifact dependencies, and acceptance requirements.

---

### Commit 2 — Stage 1 Calibration and Feature Contract Finalization

**Objective:** Resolve the remaining Stage 1 methodological and naming decisions before any downstream artifact regeneration.

#### Step 2A — Calibration Policy

- [ ] Review existing sigmoid and isotonic calibration implementations, diagnostics, and historical evidence.
- [ ] Compare calibration quality using leakage-safe training-side OOF evidence, including probability reliability, score resolution, and calibration stability.
- [ ] Measure whether isotonic score granularity materially affects Stage 4 inputs or decision behavior using an appropriately isolated diagnostic comparison.
- [ ] Evaluate the comparison against the criteria established in Commit 1.
- [ ] Select one shared calibration policy for the GLASS and black-box arms.
- [ ] Update calibration configuration, provenance requirements, shared contracts, and focused tests.
- [ ] Freeze the Stage 1 calibration policy without regenerating the full cascade yet.

#### Step 2B — Feature Naming and Contract Migration

- [ ] Rename the following Stage 1 features:
  - `cellular_crisis` → `cellular_downturn`
  - `euribor3m_local_rate` → `subscribe_rate_by_euribor_bin`
  - `dow_month_encoded` → `subscribe_rate_by_dow_month`
- [ ] Update feature definitions, training code, imports, metadata, documentation, and dependent references.
- [ ] Verify that naming-only changes preserve feature values and mathematical behavior.
- [ ] Revalidate leakage-safe handling of label-derived feature encodings, especially within OOF folds.
- [ ] Update serialization and compatibility expectations for the new versioned artifact contracts.
- [ ] Run focused calibration, feature-contract, and compatibility tests.

**Deliverable:** Frozen Stage 1 calibration and feature contracts, with implementation changes validated and ready for regeneration.

---

### Commit 3 — Stage 2 Binning and GLASS Naming Finalization

**Objective:** Resolve final Stage 2 feature-bin semantics and the outstanding GLASS component naming cleanup before retraining downstream models.

#### Step 3A — NSD and ECI Binning Revalidation

- [ ] Re-examine existing NSD and ECI thresholds, original design rationale, and current bin definitions.
- [ ] Perform train-only, leakage-safe lift analysis using appropriate fold boundaries.
- [ ] Measure per-bin support, class rates, lift, and fold-level stability.
- [ ] Identify weak, unstable, redundant, or overlapping bins without assuming they should be removed.
- [ ] Evaluate conjunction behavior and actual GLASS/RF router utility before merging or changing bins.
- [ ] Retain existing thresholds unless the pre-established evidence criteria justify revisions.
- [ ] If changes are justified, update bin definitions, encoders, feature contracts, and dependent tests.
- [ ] Preserve the PR 38 selector algorithm and frozen shared routing obligations; version any changed input contract explicitly.
- [ ] Freeze the final Stage 2 binning configuration.

#### Step 3B — GLASS Component Naming Standardization

- [ ] Resolve naming inconsistencies between GLASS Router and GLASS Arbiter across modules, imports, interfaces, and documentation.
- [ ] Update artifact paths, metadata, configuration identifiers, and downstream references where appropriate.
- [ ] Preserve the distinction between GLASS and black-box model families.
- [ ] Avoid changing modeling behavior during naming cleanup.
- [ ] Validate compatibility and add focused regression tests only where changes require them.
- [ ] Confirm the final Stage 1–2 contracts and naming structure are ready for regeneration.

**Deliverable:** Finalized Stage 2 binning and GLASS component naming, with the frozen selector design preserved.

---

### Commit 4 — Final Stage 1–3 Artifact Regeneration

**Objective:** Regenerate the affected upstream artifacts using the finalized methodology, while preserving historical baselines and validating the new training dependencies before Stage 4.

#### Step 4A — Regeneration Readiness and Upstream Execution

- [ ] Verify that the Commit 2 and Commit 3 methodology/configuration freezes are complete.
- [ ] Verify historical artifact identities and initialize the versioned PR 39 regeneration records.
- [ ] Regenerate Stage 1 artifacts for both arms using the shared calibration and updated feature contracts.
- [ ] Regenerate affected Stage 2 GLASS/RF artifacts using the final bins and frozen selector design.
- [ ] Recompute Stage 2 OOF predictions where changed upstream inputs or feature contracts require it.
- [ ] Preserve nested training, fold isolation, Pass 1 precedence, and Pass 2 remainder eligibility.
- [ ] Regenerate affected Stage 3 artifacts and OOF predictions under the finalized upstream dependencies.
- [ ] Reuse unchanged artifacts only when provenance and compatibility checks establish that reuse is valid.

#### Step 4B — Upstream Acceptance and Artifact Verification

- [ ] Validate fold identities, OOF alignment, feature contracts, upstream fingerprints, and training-side isolation.
- [ ] Verify both Stage 2 arms against the same frozen shared operating contract.
- [ ] Record infeasible folds, solver outcomes, constraint compliance, and any deviations or limitations.
- [ ] Validate serialization/deserialization, versions, metadata, artifact paths, and interface compatibility.
- [ ] Verify that historical PR 38 results and model artifacts remain unchanged.
- [ ] Run focused real-data and upstream regression tests.
- [ ] Record an explicit Stage 1–3 acceptance decision before authorizing Stage 4 regeneration.

**Deliverable:** Regenerated and validated Stage 1–3 artifacts and OOF dependencies, accepted for final Stage 4 training.

---

### Commit 5 — Stage 4 Regeneration and Arbiter Crossover

**Objective:** Regenerate the final arbiters, execute the 2×2 crossover, and establish the complete cascade performance baseline under the finalized upstream contracts.

#### Step 5A — Final Stage 4 Integration

- [ ] Verify accepted Stage 1–3 artifacts and OOF inputs from Commit 4.
- [ ] Regenerate the GLASS and black-box Stage 4 arbiters using leakage-safe finalized upstream OOF evidence.
- [ ] Verify input alignment, model-family ownership, routing conventions, and calibration dependencies.
- [ ] Validate Stage 4 artifact provenance, metadata, versioning, and serialization.
- [ ] Confirm both complete cascade arms execute under their finalized contracts.
- [ ] Run focused Stage 4 integration tests before comparative evaluation.

#### Step 5B — 2×2 Crossover and Final Baseline

- [ ] Re-run the registered 2×2 arbiter crossover using the finalized upstream artifacts.
- [ ] Evaluate the registered combinations under consistent data splits, operating points, and comparison procedures.
- [ ] Report predictive performance, calibration, abstention/coverage where applicable, and comparative uncertainty.
- [ ] Examine Stage 2 contributions and GLASS/black-box differences without changing the frozen selector based on these results.
- [ ] Revalidate mirrored contracts and identify any remaining integration limitations.
- [ ] Record final model identities, crossover findings, and the pre-presentation cascade baseline.
- [ ] Preserve earlier PR 38 comparisons as historical evidence rather than replacing them with PR 39 results.

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

- [ ] **Commit 1 complete:** PR 39 roadmap, dependency audit, evaluation rules, and regeneration sequence finalized.
- [ ] **Commit 2 complete:** Shared Stage 1 calibration and feature naming frozen.
- [ ] **Commit 3 complete:** Stage 2 binning and GLASS component naming finalized.
- [ ] **Commit 4 complete:** Final Stage 1–3 artifacts regenerated and accepted.
- [ ] **Commit 5 complete:** Stage 4 arbiters regenerated, 2×2 crossover completed, and cascade baselines established.
- [ ] **Commit 6 complete:** Final regression, provenance, reproducibility, and clean-kernel notebook validation completed.
- [ ] **PR 39 complete:** Both cascade implementations and their final Phase 1 artifact contracts frozen, with Phase 2 handoff established.
- [ ] **Phase 1 complete:** GLASS and black-box cascades are operational, integrated, validated, and ready for final notebook organization and research presentation.

**Out of scope:** Complementarity-aware selector redesign, additional Stage 2 training-objective experiments, notebook presentation restructuring, final notebook slicing and organization, paper-level independent validation, and new research extensions. These belong to later phases.

**Execution policy:** Use the Patch Method throughout. Favor one focused implementation patch per substantial step, with additional patches only for genuine corrections. Run focused tests during development and reserve expensive complete regression/notebook execution for the final integrated state wherever the acceptance requirements permit.

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