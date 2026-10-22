## Roadmap

### Phase 1 — Finish and Validate the Cascades

#### PR 36 — Automated Pipeline Test Suite

**Automated pipeline test suite (`pytest`, independent of notebooks)**

- [x] Consolidate existing stage-level validation into one `tests/` suite with shared synthetic-data fixtures
- [x] Add focused tests for the feature-research pipeline
- [ ] Add notebook-contract tests for each active research notebook
- [x] Split identity and row alignment:
  - fingerprints match across stages
  - every output index equals `X_train` / `X_test`
  - reordered or reset indices are rejected
- [x] Feature contracts:
  - missing, extra, duplicate, or reordered columns fail
  - engineered transformations are fitted on train only
- [ ] OOF integrity:
  - every training row scored exactly once by a model that never saw it
  - fold plans shared where declared
- [ ] Probability / threshold consistency:
  - probability outputs remain finite and in `[0, 1]`
  - intentional NaNs in abstention feature slots follow the declared Stage 4 contract
  - stored decisions reproduce from probabilities and thresholds where applicable
  - nested thresholds never use their own fold labels
- [ ] Routing and abstention:
  - Stage 2 state/masks reproduce stored decisions
  - Pass 1 precedence and abstention semantics are enforced
  - Pass 2 follows its declared training population (`full` or `oof_remainder`)
- [x] Calibration / threshold safeguards:
  - calibration gates use training-side OOF predictions
  - no calibrator or threshold uses held-out test labels
- [ ] Artifact provenance:
  - wrong-split artifacts fail
  - wrong-feature-contract artifacts fail
  - stale artifacts are rejected
  - unknown hyperparameters fail loudly
  - mandatory current-format provenance is enforced after artifact regeneration
- [x] Stage 4 hand-off schema:
  - required columns
  - dtypes
  - indices
  - OOF/refit provenance
- [x] Meta-level OOF integrity for Meta-XGB
- [x] Fast-running synthetic suite suitable for CI

**Additional Findings — Corrections, Safeguards & Deferred Work**

*Correctness issues resolved*

- [x] **F1:** Harden hyperparameter validation across the black-box models; reject missing, unsupported, or invalid parameter values
- [x] **F2:** Reject duplicated output indices and strengthen index-alignment regression coverage
- [x] **F11:** Validate split identity, shared fold plans, fold seeds, and cross-pipeline fold provenance
- [x] **F12 (partial):** Introduce label-aware training fingerprints; reject mismatched fingerprints and within-fold label swaps when metadata is available
- [x] **F17:** Record and validate Pass 2 training populations; enforce agreement between cross-fit and full-train router contracts
- [x] **F25/F32:** Strengthen directional router evidence, model-family identity, and upstream provenance checks
- [x] **F29:** Implement explicit 2×2 crossover support for GLASS/black-box upstreams and Meta-EBM/Meta-XGB arbiters
  - record actual upstream combinations and consumed-output fingerprints
  - reject mixed or undeclared upstream artifacts
  - verify identical upstream inputs in held-fixed arbiter comparisons
- [x] **F30:** Strengthen exact routing/abstention validation
- [x] **F33/F34:** Strengthen MLP threshold-source and label validation
- [x] **F36:** Reject invalid RF Router prediction inputs, including nonfinite and nonbinary values where prohibited
- [x] **F37/F38:** Strengthen Meta-XGB identity enforcement and intentional abstention-NaN handling

*Methodological findings documented*

- [x] **F17 / D1:** Document GLASS `full` versus RF `oof_remainder` Pass 2 training populations
- [x] **F29 / D2:** Validate RF Router band-rate evidence compatibility with the existing Meta-EBM evidence calculation
- [x] **F24:** Document the limitation of globally selected tuning parameters versus fully nested evaluation
- [x] **F39:** Document the possibility of upstream training dependence in Stage 4 stacking despite meta-level OOF separation

*Deferred follow-ups*

- [ ] **F12 — Commit 4:** Make router label fingerprints mandatory, regenerate artifacts, and reject legacy incomplete provenance
- [ ] **F17 — PR 37:** Decide whether GLASS should retain `full` or switch to `oof_remainder` for a more directly controlled router comparison
- [ ] **F21 — Commit 4:** Correct the GLASS Stage 3 notebook configuration and artifact identity; remove the expected-failure test
- [ ] **F23 — Commit 4:** Resolve stale Stage 3 notebook blocks/caches during regeneration
- [ ] **F15 — Audit follow-up:** Recover and verify the original finding definition and regression evidence
- [ ] **F31 — Future review:** Reassess RF Router confidence blending if its probability construction changes
- [ ] **F39 — PR 45:** Evaluate sensitivity to upstream stacking dependence through robustness experiments

**Cleanup**

- [ ] Remove dead or duplicated code
- [ ] Remove duplicate stale Stage 3 cells from the black-box notebook
- [ ] Remove legacy standalone modules in `blackbox_pipeline/models/xgb/` superseded by `shared_protocol.py`
- [ ] Remove superseded GLASS notebooks preserved by Git history
- [ ] Standardize GLASS Router / GLASS Arbiter naming
- [ ] Tighten package and notebook documentation

#### PR 37 — Router Baseline and Stage 2 Operating Contract

**Router redundancy diagnosis** — measure before changing Stage 2

- [ ] Add a three-state router trace: Pass 1 / Pass 2 / abstain
- [ ] Measure Pass 2 unique positives vs LR/EBM at a matched flag rate
- [ ] Measure Pass 1 false-positive suppression on train OOF:
  - negatives routed NOT_SUBSCRIBE
  - that LR and EBM would otherwise flag at their frozen upstream operating points
- [ ] Measure Pass 1 / Pass 2 overlap and redundancy
- [ ] Record the current Stage 2 redundancy baseline before selector changes

**Stage 2 operating contract**

- [ ] Replace `match_glass_oof` with pre-registered Stage 2 constraints shared by both arms
- [ ] Fix NPV / leakage / precision floors before rerunning either router
- [ ] Let each router maximize coverage subject to the same constraints
- [ ] Document the chosen constraints and rationale before viewing final rerun results
- [ ] Freeze the Stage 2 operating contract for subsequent selector experiments

#### PR 38 — Stage 2 Cross-Fit and Router Complementarity

**Corrected-selector Stage 2 rerun**

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

### Phase 4 — Paper Readiness

#### PR 46 — Paper Exports and Final Freeze

- [ ] Export final figures and tables
- [ ] Final end-to-end validation
- [ ] Freeze implementation
- [ ] Freeze experiment artifacts
- [ ] Prepare paper-ready result tables