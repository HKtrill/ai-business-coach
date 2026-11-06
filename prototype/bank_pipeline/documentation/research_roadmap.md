## Roadmap

### Phase 1 — Finish and Validate the Cascades

#### PR 37 — Router Baseline and Stage 2 Operating Contract

**Router redundancy diagnosis** — measure before changing Stage 2

- [x] Add a three-state router trace: Pass 1 / Pass 2 / abstain.
- [x] Measure Pass 2 unique positives vs LR/EBM at a matched flag rate.
- [x] Measure Pass 1 false-positive suppression on train OOF:
  - Negatives routed `NOT_SUBSCRIBE`.
  - Negatives that LR and EBM would otherwise flag at their frozen upstream operating points.
- [x] Measure Pass 1 / Pass 2 overlap and redundancy.
- [x] Record the current Stage 2 redundancy baseline before selector changes.
- [x] Preserve the pre-correction baseline as an immutable, version-controlled research record — Commit 2: 19-file manifest, pinned code identities, and baseline verification/reproduction checks.

**Stage 2 operating contract**

- [x] Define a shared, arm-neutral Stage 2 operating contract independent of either router's realized performance — Commit 4, Step 4A.
- [x] Implement and validate the working NPV / leakage / precision constraints before corrected reruns:
  - Pass 1 NPV ≥ 0.95.
  - Pass 1 positive leakage ≤ 0.15.
  - Pass 2 precision ≥ 0.30.
  - Pass 1 minimum routed-set support: 125.
  - Pass 2 minimum routed-set support: 75.
  - No absolute positive-leakage count cap.
- [x] Define lexicographic coverage optimization: maximize feasible Pass 1 routed rows, then maximize feasible Pass 2 routed rows on the resulting remainder — Steps 4A–4B.
- [x] Implement opt-in coverage maximization under the same shared constraints for GLASS and RF Router, while preserving historical selector behavior — Step 4B.
- [x] Document the provisional constraints, their rationale, historical evidence exposure, and feasibility limitations before corrected PR 38 reruns — Step 4A.
- [x] Implement versioned contract configuration, deterministic fingerprinting, and explicit working/frozen status — Step 4A.
- [x] Remove dependence on `match_glass_oof` from the new primary shared-contract pathway while retaining historical reproduction compatibility — Steps 4A–4B.
- [ ] Finalize `match_glass_oof` replacement as the frozen primary PR 38 operating policy — Step 4D.
- [ ] Freeze the final Stage 2 operating contract, numerical settings, version, fingerprint, and approved configuration profiles for subsequent selector experiments — Step 4D.

**Constrained optimization and implementation safeguards — Commit 4**

- [x] Implement a shared, arm-neutral obligation checker with exact count-based comparisons, common metric definitions, and appropriate denominators — Step 4A.
- [x] Enforce explicit validation of zero support, insufficient class support, undefined ratios, and invalid constraint configurations — Step 4A.
- [x] Distinguish fit-time constraint enforcement from observed outer-OOF performance; do not silently treat outer-OOF shortfalls as fit-time violations — Step 4A.
- [x] Implement explicit, versioned constraint provenance, including historical OOF exposure, prior test-set exposure, and separation of historical evidence from active constraint sources — Step 4A.
- [x] Implement shared-contract adapters for GLASS and RF Router without sharing fitted estimators or changing historical defaults — Steps 4A–4B.
- [x] Enforce GLASS Pass 1 constraints on the selected rule union rather than relying solely on individual-rule gates:
  - Union NPV.
  - Union positive leakage.
  - Minimum union support.
  - Exact union-membership accounting.
- [x] Enforce GLASS Pass 2 constraints on the selected rule union:
  - Union precision.
  - Minimum union support.
  - Inclusion of zero-positive atoms in precision calculations.
  - Exact union-membership accounting.
- [x] Implement opt-in Pass 2 union-row coverage maximization while preserving the historical positive-yield objective — Step 4B.
- [x] Define and test deterministic candidate ordering, selection behavior, and constraint-boundary comparisons — Step 4B.
- [x] Independently validate ILP solutions, feasible incumbents, and greedy fallback outputs against the shared obligations — Step 4B.
- [x] Improve CBC solver reporting by recording available termination, incumbent, bound, and MIP-gap evidence without changing solver tolerances or time limits — Step 4B.
- [x] Preserve explicit failure behavior for infeasible or unsupported selections; prohibit silent relaxation of the shared constraints — Step 4B.
- [x] Propagate shared constraints into the GLASS inner-fold selectors and required Pass 2 remainder selection without activating corrected notebook reruns — Step 4B.
- [x] Preserve training-side population boundaries, arm-specific estimator ownership, and held-out-label isolation in the new contract pathways — Step 4B.

**GLASS selector performance optimization — Commit 4, Step 4B**

- [x] Profile GLASS selector performance and identify significant avoidable computation.
- [x] Identify redundant rule-matching operations in pairwise overlap diagnostics.
- [x] Optimize overlap diagnostics by caching rule-matching masks and reusing them across rule pairs.
- [x] Preserve identical diagnostic values and avoid changes to rule-search or selection semantics.
- [x] Add deterministic performance-equivalence regression coverage.
- [x] Independently benchmark the optimization in the local Windows environment:
  - 124 rules, 2,000 rows, 7,626 rule pairs.
  - Historical implementation: 2.39 seconds.
  - Optimized implementation: 0.041 seconds.
  - Approximately **58× faster overlap diagnostics**, with identical values.
- [x] Preserve historical solver time limits, optimization tolerances, candidate-generation behavior, and default selector configurations.
- [x] Evaluate any further selector optimization only if justified by later profiling; avoid unnecessary algorithmic changes during PR 38.

**Pre-registered PR 38 evaluation protocol — Commit 4, Step 4C**

- [x] Define the primary GLASS-versus-RF comparison metrics and acceptance hierarchy.
- [x] Define secondary diagnostics for coverage, NPV, positive leakage, precision, recall, abstention, and class-conditional routing.
- [x] Specify comparisons against LR/EBM at frozen upstream operating points and matched flag rates.
- [x] Define paired pre-correction versus post-correction comparisons, including changes in routed-row identities and unique contributions.
- [x] Distinguish changes attributable to F17, shared constraints, Pass 2 coverage optimization, and any later selector ablation; avoid unsupported causal attribution.
- [x] Pre-register pooled OOF and fold-level reporting, uncertainty estimation, constraint shortfalls, and infeasible-candidate handling.
- [x] Define explicit reporting for solver nonoptimality, missing bounds, greedy fallback, unsuccessful folds, and insufficient support.
- [x] Define complementarity-aware selection as a secondary ablation unless pre-registered evidence justifies adoption.
- [x] Specify complementarity-aware adoption/rejection criteria for unique-positive discovery, matched flag budgets, constraint preservation, coverage, uncertainty, and computational feasibility.
- [x] Require an equivalent RF objective or documented methodological asymmetry if complementarity-aware selection is adopted.
- [x] Define structural, methodological, reproducibility, and empirical-performance requirements for final Stage 2 artifact acceptance.
- [x] Document prior test-set exposure and its implications for claims of independent validation.
- [x] Distinguish historical exploratory OOF evidence, corrected nested cross-fitted evaluation, and genuinely independent external validation.
- [x] Prohibit test-derived results from informing corrected selector choices, contract adjustments, or ablation adoption decisions.
- [x] Version-control the evaluation protocol and validate its executable invariants before PR 38 corrected reruns.

**Contract freeze, independent configuration, and final handoff — Commit 4, Step 4D**

- [x] Freeze the approved shared contract values, schema version, content fingerprint, and provenance record.
- [x] Freeze the GLASS arm-internal selection profile, including rule-quality gates, rule counts, novelty settings, and relevant selector configuration.
- [x] Move RF Router hyperparameter ownership into an independent, version-controlled configuration rather than sourcing the primary configuration dynamically from a GLASS-side artifact.
- [x] Record the original RF hyperparameter tuning provenance, including full-training-data exposure and limitations of prior model selection.
- [x] Freeze the approved F17 primary configurations and distinguish them from secondary sensitivity configurations.
- [x] Freeze the PR 38 evaluation protocol, metrics, comparison procedures, uncertainty rules, and acceptance criteria.
- [x] Record final contract and configuration identities required by PR 38.
- [x] Reconcile PR 37 roadmap checkboxes with actual implementation, documentation, and validation evidence.
- [x] Produce the final Commit 4 methodology record and PR 38 handoff requirements.
- [x] Perform final combined regression, real-data, provenance, Git integrity, and historical baseline validation.
- [x] Create and push **one Commit 4** after Steps 4A–4D have been completed and validated.

## Commit 5 — Final Validation, Documentation, and PR 38 Handoff

**Objective:** Independently validate the completed PR 37 implementation, reconcile its documentation and provenance, preserve the historical baseline, and establish explicit PR 38 readiness. No new selector architecture, corrected model training, or operating-contract changes are planned.

### Step 5A — Targeted Automated Validation

**Router diagnostics and shared contract**

- [x] Run all new router-diagnostic, shared-contract, constrained-optimization, and evaluation-protocol tests from Commits 1–4.
- [x] Verify three-state routing is mutually exclusive and exhaustive:
  - `PASS_1_NOT_SUBSCRIBE`
  - `PASS_2_SUBSCRIBE`
  - `ABSTAIN`
- [x] Verify Pass 1 precedence, Pass 2 remainder eligibility, coverage accounting, and abstention calculations.
- [x] Verify unique-positive discovery, overlap, redundancy, and Pass 1 false-positive suppression calculations.
- [x] Verify matched-flag-rate logic, frozen upstream operating points, and deterministic score-tie handling.
- [x] Verify shared NPV, leakage, precision, and support constraints using exact count-based comparisons.
- [x] Verify GLASS union-level obligations, including zero-positive atoms and exact union membership.
- [x] Verify infeasible-candidate handling, solver outcome reporting, incumbent validation, and explicit fallback behavior.
- [x] Verify GLASS and RF apply identical shared obligations without sharing fitted estimator state.
- [x] Verify contract versions, fingerprints, configuration provenance, and validation refusals.
- [x] Verify evaluation-protocol invariants, including pooled OOF denominators, paired comparisons, undefined metrics, and unsuccessful-fold reporting.

**Leakage prevention and methodological safeguards**

- [x] Test cross-arm index, row-identity, split, fold-assignment, and provenance mismatches.
- [x] Verify no new held-out test-label dependency, training/OOF contamination, or cross-fold selection leakage.
- [x] Verify primary F17 configuration requirements and training-population isolation.
- [x] Verify F31 confidence semantics and Stage 4 interface compatibility.
- [x] Confirm the primary operating policy no longer depends dynamically on `match_glass_oof`.
- [x] Confirm previously exposed test results cannot alter contract values, thresholds, selector choices, or evaluation rules.

**Regression and integration testing**

- [x] Run the complete existing PR 36/PR 37 fast regression suite.
- [x] Run the complete Stage 2 and parity regression suites.
- [x] Run the local real-data tests using the existing artifacts.
- [x] Run designated slow ILP integration tests.
- [x] Verify historical default behavior and backward compatibility.
- [x] Verify the GLASS overlap-diagnostics optimization preserves identical diagnostic outputs.
- [x] Resolve all newly introduced failures, regressions, and actionable warnings.
- [x] Record final test counts, deselections, warnings, environment information, and any known limitations.

### Step 5B — Documentation, Provenance, and Baseline Preservation

**Research methodology and decision records**

- [x] Publish and reconcile the complete router redundancy-diagnostic methodology.
- [x] Preserve the pre-correction redundancy tables, source identities, and immutable historical baseline.
- [x] Verify that the F17 training-population decision and F31 confidence disposition are accurately documented.
- [x] Publish the frozen shared operating contract, numerical constraints, objective, and rationale.
- [x] Publish the GLASS and RF arm-internal configuration records, including RF hyperparameter provenance.
- [x] Publish the pre-registered PR 38 evaluation protocol and acceptance criteria.
- [x] Explain the distinction between historical diagnostic OOF analysis, corrected cross-fitted evaluation, and independent external validation.
- [x] Document previous test-set exposure and its implications for performance claims.
- [x] Document solver limitations, uncertainty handling, infeasible-fold reporting, and methodological asymmetries.
- [x] Verify documentation matches actual schemas, metric definitions, configuration defaults, and executable behavior.

**Engineering changes and optimization records**

- [x] Document the Commit 4 GLASS union-level constrained-optimization implementation.
- [x] Document CBC solver reporting, time-limited incumbents, fallback requirements, and reproducibility limitations.
- [x] Document the selector overlap-diagnostics optimization:
  - Original repeated rule matching.
  - Cached-mask implementation.
  - Behavioral equivalence.
  - Local benchmark: 2.39 seconds → 0.041 seconds, approximately 58× faster.
  - Clarify that this improvement applies to overlap diagnostics, not overall model training.
- [x] Record any remaining performance bottlenecks without expanding PR 37 into another optimization effort.

**Reproducibility and artifact integrity**

- [x] Verify the immutable 19-file pre-correction baseline using both `--verify` and `--check`.
- [x] Verify all 15 baseline-pinned code identities.
- [x] Perform the independent external 37-artifact SHA-256 verification.
- [x] Verify historical RF two-pass metadata and sweep provenance.
- [x] Verify contract, protocol, GLASS configuration, and RF hyperparameter fingerprints.
- [x] Confirm model artifacts remain unchanged and Git-ignored where required.
- [x] Confirm all intended source code, tests, configuration records, and methodology documents are version-controlled.
- [x] Verify no unintended changes to notebooks, Stage 4 interfaces, historical baselines, or model artifacts.
- [x] Reconcile the final PR 37 roadmap against completed implementation and test evidence.

### Step 5C — PR 38 Readiness Review

**Operating contract and architectural decisions**

- [x] Confirm pre-correction baselines remain reproducible.
- [x] Confirm F17 training-population policy is resolved and frozen.
- [x] Confirm F31 disposition is resolved, with outstanding PR 38 validation requirements explicitly handed off.
- [x] Confirm `match_glass_oof` replacement is fully specified and frozen for the primary corrected reruns.
- [x] Confirm identical shared operating constraints for GLASS and RF Router.
- [x] Confirm independent fitted-estimator ownership and versioned arm-specific configurations.
- [x] Confirm the corrected-selector evaluation protocol is frozen.
- [x] Confirm complementarity-aware selection remains a secondary ablation unless its pre-registered adoption requirements are met.

**PR 38 execution gates**

- [x] Document the required G3 real-data feasibility preflight before corrected primary reruns.
- [x] Document the G13 label-derived feature-engineering leakage audit.
- [x] Confirm all required F17 nested-training safeguards are specified.
- [x] Confirm the F31 nested-versus-deployed `t1` boundary-strip review trigger and reporting requirements.
- [x] Confirm infeasible fits, missing incumbents, insufficient remainder support, and contract shortfalls have explicit handling rules.
- [x] Confirm corrected GLASS and RF rerun procedures, required training-side inputs, and fold identities.
- [x] Confirm artifact regeneration, serialization, fingerprint, and provenance requirements.
- [x] Confirm structural artifact acceptance is distinguished from comparative predictive performance.
- [x] Confirm final evaluation cannot be used retroactively to adjust frozen contracts or adoption criteria.
- [x] Confirm PR 38 has a clear implementation and validation checklist.

**Final PR 37 acceptance**

- [x] Review all changes introduced across Commits 1–5.
- [x] Confirm the final test suite passes with no unresolved newly introduced failures.
- [x] Confirm baseline and artifact integrity checks pass.
- [x] Confirm the roadmap and documentation accurately represent completed and deferred work.
- [x] Verify the final Git diff, staged files, commit history, and branch state.
- [x] Confirm no unintended generated artifacts or local files are included.
- [x] Prepare and review the final PR 37 handoff record.
- [x] Create and push Commit 5 after validation and documentation reconciliation.
- [x] Perform the final PR 37 readiness review before merging or closing the PR.

**Deliverable:** Fully validated and documented PR 37 implementation, preserved historical evidence, frozen Stage 2 methodology, and an explicit execution-ready PR 38 handoff.

---

### PR 37 — Final completion boundary

- [ ] **Commit 4 complete:** Shared constrained optimization, pre-registered PR 38 evaluation protocol, frozen contract and configuration provenance.
- [ ] **Commit 5 complete:** Final integration validation, documentation reconciliation, historical baseline verification, and PR 38 readiness certification.
- [ ] **PR 37 complete:** Historical redundancy evidence preserved, methodological decisions version-controlled, shared Stage 2 operating policy frozen, and PR 38 execution requirements formally handed off.

**Out of scope for PR 37:** Corrected five-fold Stage 2 reruns, final model artifact regeneration, notebook deployment of corrected selectors, full complementarity-aware selector experiments, and independent-validation claims based on previously exposed test data.

**Deferred from PR 36**

- [x] Decide whether GLASS should retain `full` or switch to `oof_remainder` for a more directly controlled router comparison — decided in Commit 3: both routers fit Pass 2 on their own OOF Pass 1 remainder, with both nested safeguards on; GLASS `full` is retained only as sensitivity analysis S1 ([F17 decision record](methodology/pr37_f17_decision_record.md)).
- [x] **(Conditional)** Reassess RF Router confidence-blending assumptions only if Stage 2 probability construction changes. Otherwise, retain the validated implementation — reviewed in Commit 3: construction retained (Disposition A); PR 38 still owes validation and provenance requirements R1–R8 ([F31 disposition record](methodology/pr37_f31_disposition_record.md)).

**Methodological safeguards and cross-arm independence**

- [x] Verify and regression-protect independent fitted-estimator ownership across GLASS and the black-box arm; distinguish shared RF hyperparameters from shared fitted model state — Commit 3: no fitted estimator crosses arms; protected by `tests/parity/test_stage2_cross_arm_independence.py` ([F17 record, section 5](methodology/pr37_f17_decision_record.md)).
- [ ] Finalize independent, version-controlled RF Router hyperparameter ownership — decision established; implementation and provenance freeze scheduled for Step 4D ([Commit 3 handoff](methodology/pr37_commit3_handoff.md)).
- [x] Specify whether `nested_t1` and `fold_nested_rf` will be required in the PR 38 primary protocol — both required (F17 gate G1); opt-in code and synthetic tests introduced in Commit 3 ([F17 record](methodology/pr37_f17_decision_record.md)).
- [x] Define validation and explicit failure requirements for infeasible inner folds and insufficient remainder support — F17 gates G3 and G8, strengthened by shared obligation and selector outcome checks in Steps 4A–4B ([F17 record](methodology/pr37_f17_decision_record.md)).
- [x] Finalize and version-control the Step 3A methodological investigation and F17/F31 decision records — three records and the Commit 3 handoff under `documentation/methodology/` ([index](methodology/pr37_commit3_handoff.md)).
- [x] Document differences between fit-time rule constraints and shared outer-OOF evaluation requirements — established in F17 L2 and F31 K2; GLASS union-level constraint mapping implemented in Step 4B.
- [x] Preserve historical GLASS and RF operating behavior unless the new shared contract is explicitly activated — regression-tested in Steps 4A–4B.
- [x] Preserve F31 Disposition A: RF empirical confidence bands and heuristic probability-like blending remain unchanged and must not be represented as calibrated probabilities.
- [ ] Complete PR 38 G3 real-data feasibility preflight under the frozen shared contract before corrected primary reruns.
- [ ] Complete the PR 38 G13 label-derived feature-engineering leakage audit before accepting corrected artifacts.
- [ ] Review the F31 nested-versus-deployed `t1` boundary strip when it reaches the declared 5% Pass 2 flagged-band trigger.

**Regression testing, reproducibility, and baseline preservation**

- [x] Add targeted regression coverage for the shared contract, provenance validation, exact constraint arithmetic, undefined metrics, and RF adapter mapping — Step 4A.
- [x] Add targeted regression coverage for GLASS union constraints, zero-positive atoms, deterministic selection, solver outcomes, shared adapters, and performance equivalence — Step 4B.
- [x] Independently validate the combined Steps 4A–4B implementation in the local Windows environment:
  - 2,450 full fast-suite tests passed.
  - 1,381 Stage 2/parity tests passed.
  - 87 real-data tests passed.
  - Additional slow ILP test passed.
- [x] Verify that the frozen historical 19-file baseline remains unchanged and reproducible after Steps 4A–4B — both `--verify` and `--check` passed 19/19.
- [x] Validate Step 4B patch integrity, 9/9 prerequisite identities, 12/12 patched files, and zero conflicts.
- [x] Verify clean Git whitespace checks and preservation of the intended uncommitted working-tree changes.
- [x] Confirm historical RF two-pass metadata and sweep-file membership in the external PR 36 SHA-256 snapshot.
- [ ] Complete final independent artifact-hash verification and combined regression checks after Step 4D.
- [ ] Confirm that all final Commit 4 documentation, tests, contract files, and configurations are included in the Git commit, while Git-ignored model artifacts and frozen baselines remain unchanged.

**PR 37 completion boundary**

- [ ] Commit 4 complete: shared constrained optimization, pre-registered evaluation protocol, and frozen operating/configuration contracts.
- [ ] Commit 5 complete: final PR 37 regression review, documentation cleanup, readiness checks, and PR 38 handoff.
- [ ] PR 37 closed with historical redundancy evidence preserved, methodological decisions version-controlled, and PR 38 execution gates explicitly documented.

**Out of scope for PR 37:** Corrected five-fold Stage 2 reruns, final model artifact regeneration, deployment of corrected selectors into production notebooks, full real-data complementarity-aware selector experiments, and claims of independent validation based on previously exposed test data. These remain PR 38 or later work.

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