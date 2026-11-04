## Roadmap

### Phase 1 — Finish and Validate the Cascades

#### PR 37 — Router Baseline and Stage 2 Operating Contract

**Router redundancy diagnosis** — measure before changing Stage 2

- [x] Add a three-state router trace: Pass 1 / Pass 2 / abstain
- [x] Measure Pass 2 unique positives vs LR/EBM at a matched flag rate
- [x] Measure Pass 1 false-positive suppression on train OOF:
  - negatives routed NOT_SUBSCRIBE
  - that LR and EBM would otherwise flag at their frozen upstream operating points
- [x] Measure Pass 1 / Pass 2 overlap and redundancy
- [x] Record the current Stage 2 redundancy baseline before selector changes

**Stage 2 operating contract**

- [ ] Replace `match_glass_oof` with pre-registered Stage 2 constraints shared by both arms
- [ ] Fix NPV / leakage / precision floors before rerunning either router
- [ ] Let each router maximize coverage subject to the same constraints
- [ ] Document the chosen constraints and rationale before viewing final rerun results
- [ ] Freeze the Stage 2 operating contract for subsequent selector experiments

**Deferred from PR 36**

- [x] Decide whether GLASS should retain `full` or switch to `oof_remainder` for a more directly controlled router comparison — decided in Commit 3: both routers fit Pass 2 on their own OOF Pass 1 remainder, with both nested safeguards on; GLASS `full` is kept only as sensitivity analysis S1 ([F17 decision record](methodology/pr37_f17_decision_record.md))
- [x] **(conditional):** Reassess RF Router confidence-blending assumptions only if Stage 2 probability construction changes. Otherwise, retain the validated implementation. — reviewed in Commit 3: construction retained (Disposition A); PR 38 owes validation and provenance requirements R1–R8 ([F31 disposition record](methodology/pr37_f31_disposition_record.md))

**Methodological safeguards and cross-arm independence**

- [x] Verify and regression-protect independent fitted-estimator ownership across GLASS and the black-box arm; distinguish shared RF hyperparameters from shared fitted model state — Commit 3: no fitted estimator crosses arms; pinned by `tests/parity/test_stage2_cross_arm_independence.py` ([F17 record, section 5](methodology/pr37_f17_decision_record.md))
- [ ] Resolve whether RF Router hyperparameters should continue to be sourced from a GLASS-side artifact or an independent, versioned configuration — open Commit 4 decision D6 ([handoff](methodology/pr37_commit3_handoff.md))
- [x] Specify whether `nested_t1` and `fold_nested_rf` will be required in the PR 38 primary protocol — both required (F17 gate G1); opt-in code and synthetic tests in Commit 3 ([F17 record](methodology/pr37_f17_decision_record.md))
- [x] Define validation and explicit failure requirements for infeasible inner folds and insufficient remainder support — F17 gates G3 (feasibility preflight) and G8 (explicit rejection, no fallback), implemented as refusals in Commit 3; the uniform no-feasible-candidate rule is Commit 4 decision D4 ([F17 record](methodology/pr37_f17_decision_record.md))
- [x] Finalize and version-control the Step 3A methodological investigation and F17/F31 decision records — the three records and the Commit 3 handoff, under `documentation/methodology/` ([index](methodology/pr37_commit3_handoff.md))
- [x] Document remaining differences between fit-time rule constraints and shared outer-OOF evaluation requirements — documented in F17 L2 and F31 K2; resolving the GLASS fit-time constraint mapping remains open Commit 4 decision D3 ([handoff](methodology/pr37_commit3_handoff.md))

#### PR 38 — Stage 2 Cross-Fit and Router Complementarity

**Corrected-selector Stage 2 rerun**

- [ ] Validate the PR 37 F17/F31 obligations before treating any corrected rerun as primary evidence ([Commit 3 handoff](methodology/pr37_commit3_handoff.md)):
  - before the full 5-fold primary reruns, pass the preflight and implementation gates: G1 configuration, G2 wiring, G3 feasibility preflight, G5 persistence, G8 explicit rejection, G9 independent contract, G13 feature-engineering audit (a confirmed material own-label dependency must be corrected and validated before primary results are accepted)
  - with the primary reruns, verify G4 isolation provenance, G7 reproducibility, G10 versioned artifacts and pins, G11 baseline preservation; record the mandatory F31 requirements R1–R5, R7 and R8 (R6, band-level spread, is an optional diagnostic)
  - afterwards, G12: sensitivity analysis S1 (GLASS `full` vs `oof_remainder`, descriptive only), begun once the applicable primary-run gates other than G6 have passed; G6 Pass 1 invariance is verified during S1, and S1 is not accepted unless it passes
- [ ] Run the corrected GLASS Router under the frozen Stage 2 operating contract
- [ ] Run the RF Router under the same operating contract
- [ ] Rerun the full 5-fold Stage 2 outer cross-fit for both arms
- [ ] Regenerate Stage 2 artifacts
- [ ] Verify OOF provenance, fold identity, routing semantics, and constraint compliance

**Corrected-router complementarity**

- [ ] Re-run the three-state router trace
- [ ] Re-measure Pass 2 unique positives at matched flag rate
- [ ] Re-measure Pass 1 false-positive suppression
- [ ] Compare redundancy before vs after the selector correction
- [ ] Measure GLASS Router vs LR/EBM OOF score and decision overlap

**Complementarity experiment**

- [ ] Evaluate an optional complementarity-aware selector using train-side OOF upstream predictions only
- [ ] Test rules favoring:
  - positives Stage 1 LR misses
  - rows where LR and EBM disagree
- [ ] Compare standard union-coverage selection vs complementarity-aware selection
- [ ] Treat complementarity-aware selection as an ablation unless evidence supports adopting it
- [ ] If adopted, provide the RF Router an equivalent objective or document the asymmetry explicitly
- [ ] Freeze the final Stage 2 selector design

**Final Stage 2 freeze**

- [ ] Select the final Stage 2 selector design based on the pre-declared comparison
- [ ] If complementarity-aware selection is adopted, rerun both arms under the frozen operating contract
- [ ] Regenerate final Stage 2 OOF and test artifacts
- [ ] Verify final OOF provenance, constraint compliance, and mirrored contracts
- [ ] Freeze the Stage 2 implementation and artifacts for downstream regeneration

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