## Roadmap

### Phase 1 — Finish and Validate the Cascades

#### PR 33 — Implement Stage 4 & Clean Cascades

- [x] Validate Stages 1–3 under a mirrored protocol on both cascades
- [x] Retire GLASS-BRW naming — modules, variables, artifact keys, notebook markdown, scanned for stragglers
- [x] Align GLASS train-side Stage 2/3 outputs with the OOF protocol
- [x] Share Stage 4 input/alignment infrastructure across both arms
- [x] Verify Stage 3 population policy across both arms
- [x] Establish OOF stacking-feature policy
- [x] Finalize matched Stage 4 input contracts
- [x] Update GLASS Stage 4 to the audited Stage 1–3 artifact contracts
- [x] Implement Meta-XGB as the black-box Stage 4 counterpart
- [x] Combine Stage 1–4 signals into complete cascade outputs
- [x] Freeze and sign off the Stage 2 baseline
- [x] Validate end-to-end black-box cascade execution
- [x] Record provisional GLASS Arbiter vs Meta-XGB Stage 4 results
- [x] Run 2×2 arbiter crossover:
  - GLASS Arbiter on GLASS upstream outputs
  - Meta-XGB on GLASS upstream outputs
  - GLASS Arbiter on black-box upstream outputs
  - Meta-XGB on black-box upstream outputs
- [x] Clean cascade pipeline infrastructure, naming, stale compatibility code, and documentation

#### PR 34 — Validation, Research-Pipeline Audit, and Cleanup

**Full pipeline validation**
- [ ] Clean-kernel top-to-bottom rerun of all three active research notebooks
- [ ] Verify all notebook outputs are reproducible from the current codebase
- [ ] Final leakage, split, feature, calibration, threshold, and OOF audits
- [ ] Verify artifact provenance, library versions, and reload guards
- [ ] Verify final cascade prediction diversity/correlation
- [ ] Rerun and record final metrics after validation

**Repository cleanup**
- [ ] Consolidate `shared/stage3/` and `shared/stage_runner.py` into one source of truth; make the other a thin re-export
- [ ] Unify the two split-fingerprint formats (`stage_io` 12-hex and router 16-hex) into one scheme
- [ ] Anchor artifact directories to their package roots instead of the notebook working directory

**Feature-research pipeline validation**
- [ ] Audit the feature-research pipeline end to end
- [ ] Verify every learned feature transformation is fitted on training data only
- [ ] Verify feature names, order, dtypes, and source-feature provenance
- [ ] Verify feature-block fingerprints / contracts where applicable
- [ ] Reject missing, extra, duplicated, or reordered feature columns
- [ ] Verify cached feature artifacts belong to the current split and feature contract
- [ ] Verify no test-derived statistics enter feature construction
- [ ] Verify feature-research outputs consumed downstream reproduce exactly after reload
- [ ] Remove stale feature-research paths, aliases, and compatibility code no longer required

**Notebook-specific validation**

- [ ] Feature-research notebook
  - clean-kernel execution
  - feature-generation contracts
  - train-only fitting
  - artifact reload/provenance
  - deterministic outputs where expected

- [ ] GLASS cascade notebook
  - Stage 1–4 contracts
  - OOF hand-offs
  - split/index/fold identity
  - calibration and threshold provenance
  - routing / abstention semantics
  - GLASS Arbiter inputs and outputs

- [ ] Black-box cascade notebook
  - Stage 1–4 mirror contracts
  - OOF hand-offs
  - split/index/fold identity
  - matched feature/population contracts
  - Meta-XGB cross-fitting
  - threshold / calibration / abstention provenance

#### PR 35 — Automated Pipeline Test Suite

**Automated pipeline test suite (`pytest`, independent of notebooks)**
- [ ] Consolidate existing stage-level validation into one `tests/` suite with shared synthetic-data fixtures
- [ ] Add focused tests for the feature-research pipeline
- [ ] Add notebook-contract tests for each active research notebook
- [ ] Split identity and row alignment:
  - fingerprints match across stages
  - every output index equals `X_train` / `X_test`
  - reordered or reset indices are rejected
- [ ] Feature contracts:
  - missing, extra, duplicate, or reordered columns fail
  - engineered transformations are fitted on train only
- [ ] OOF integrity:
  - every training row scored exactly once by a model that never saw it
  - fold plans shared where declared
- [ ] Probability / threshold consistency:
  - probabilities remain in `[0, 1]`
  - no NaN / inf
  - stored decisions reproduce from probabilities and thresholds where applicable
  - nested thresholds never use their own fold labels
- [ ] Routing and abstention:
  - Stage 2 state/masks reproduce stored decisions
  - Pass 2 remainder is built from OOF Pass 1 routing
- [ ] Calibration / threshold safeguards:
  - calibration gates use training-side OOF predictions
  - no calibrator or threshold uses held-out test labels
- [ ] Artifact provenance:
  - wrong-split artifacts fail
  - wrong-feature-contract artifacts fail
  - stale artifacts are rejected
  - unknown hyperparameters fail loudly
- [ ] Stage 4 hand-off schema:
  - required columns
  - dtypes
  - indices
  - OOF/refit provenance
- [ ] Meta-level OOF integrity for Meta-XGB
- [ ] Fast-running synthetic suite suitable for CI

**Cleanup**
- [ ] Remove dead or duplicated code
- [ ] Remove duplicate stale Stage 3 cells from the black-box notebook
- [ ] Remove legacy standalone modules in `blackbox_pipeline/models/xgb/` superseded by `shared_protocol.py`
- [ ] Remove superseded GLASS notebooks preserved by Git history
- [ ] Archive older churn notebooks that remain useful as research references
- [ ] Standardize GLASS Router / GLASS Arbiter naming
- [ ] Tighten package and notebook documentation

#### PR 36 — Router Baseline and Stage 2 Operating Contract

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


#### PR 37 — GLASS Router Selector Correction

**Union-aware ILP selector**

- [ ] Replace per-rule coverage scoring with useful-row union coverage
- [ ] Enforce union leakage as a hard set-level constraint
- [ ] Make ILP optimization lexicographic:
  1. maximize useful union coverage
  2. among coverage-equivalent solutions, optimize secondary rule quality / parsimony
- [ ] Remove or relax pairwise novelty constraints made redundant by union-aware selection
- [ ] Wire `QualityGateFilter` coverage caps through `ILPRuleSelector`
- [ ] Make selector λ defaults explicit or remove unused penalties
- [ ] Log ILP solve status and fallback usage per fold
- [ ] Update greedy fallback to select rules by marginal union-coverage gain

**Selector validation**

- [ ] Validate the corrected selector on the frozen full-train baseline
- [ ] Verify selected-set union leakage satisfies the configured cap when feasible
- [ ] Record solver/fallback behavior when the constraint set is infeasible
- [ ] Compare before/after:
  - useful union coverage
  - subscriber leakage
  - covered F1
  - rule redundancy
  - rule count
  - solve time
- [ ] Confirm no regression in rule semantics or artifact contracts


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

### Phase 4 — Paper Readiness

#### PR 46 — Paper Exports and Final Freeze

- [ ] Export final figures and tables
- [ ] Final end-to-end validation
- [ ] Freeze implementation
- [ ] Freeze experiment artifacts
- [ ] Prepare paper-ready result tables