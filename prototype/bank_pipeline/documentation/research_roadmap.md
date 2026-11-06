## Roadmap

### Phase 1 — Finish and Validate the Cascades

#### PR 37 — Router Baseline and Stage 2 Operating Contract

**Status key.** `[x]` done and locally validated. `[ ]` open, labelled **implemented, pending local validation** (in the working tree, not yet validated), **pending** (not started) or owned by PR 38. **Frozen** marks a version-controlled record whose status is `frozen`.

| Stage | State |
|---|---|
| Commit 1: router redundancy diagnostics | Committed (`8689a89`) |
| Commit 2: immutable pre-correction baseline | Committed (`e5c437c`) |
| Commit 3: F17 / F31 decisions and cross-arm independence safeguards | Committed (`ab7b063`) |
| Commit 4, Steps 4A–4D: shared contract, union-level obligations, PR 38 evaluation protocol, freeze of the contract, protocol and arm configurations | Committed and pushed (`f7a987e`); the frozen records are in effect |
| Commit 5: final integration verification, documentation and PR 38 handoff | Implemented and locally validated; commit and push pending |
| PR 37 review and merge | Pending |

**Router redundancy diagnosis** — measured before changing Stage 2 (Commits 1–2)

- [x] Add a three-state router trace: Pass 1 / Pass 2 / abstain.
- [x] Measure Pass 2 unique positives vs LR/EBM at a matched flag rate.
- [x] Measure Pass 1 false-positive suppression on train OOF:
  - negatives routed `NOT_SUBSCRIBE`;
  - that LR and EBM would otherwise flag at their frozen upstream operating points.
- [x] Measure Pass 1 / Pass 2 overlap and redundancy.
- [x] Record the Stage 2 redundancy baseline before selector changes, preserved as an immutable, version-controlled research record: 19-file manifest, pinned code identities, `--verify` and `--check` reproduction checks (Commit 2).

**Methodological decisions and cross-arm independence** — Commit 3, including the items deferred from PR 36

- [x] Decide whether GLASS should retain `full` or switch to `oof_remainder` — both routers fit Pass 2 on their own OOF Pass 1 remainder with both nested safeguards on (`nested_t1`, `fold_nested_rf`; F17 gate G1); GLASS `full` is retained only as sensitivity analysis S1 ([F17 decision record](methodology/pr37_f17_decision_record.md)).
- [x] **(Conditional)** Reassess RF Router confidence blending only if Stage 2 probability construction changes — construction retained (F31 Disposition A): the empirical bands and probability-like blending are unchanged and never presented as calibrated probabilities; PR 38 owes R1–R8 ([F31 disposition record](methodology/pr37_f31_disposition_record.md)).
- [x] Verify and regression-protect independent fitted-estimator ownership; distinguish shared RF hyperparameter values from shared fitted state — no fitted estimator crosses arms (`tests/parity/test_stage2_cross_arm_independence.py`; [F17 record, section 5](methodology/pr37_f17_decision_record.md)).
- [x] Define explicit failure requirements for infeasible inner folds and insufficient remainder support — F17 gates G3 and G8, strengthened by the shared obligations and selector outcome checks of Steps 4A–4B.
- [x] Document the differences between fit-time rule constraints and shared outer-OOF evaluation — F17 L2 and F31 K2; the GLASS union-level mapping in Step 4B.
- [x] Version-control the Step 3A investigation and the F17 / F31 decision records ([Commit 3 handoff](methodology/pr37_commit3_handoff.md)).

**Shared Stage 2 operating contract** — Commit 4, Step 4A ([contract record](methodology/pr37_commit4_contract_record.md))

- [x] Define a shared, arm-neutral contract independent of either router's realised performance: versioned schema, deterministic fingerprint, working/frozen status and structured provenance (historical OOF exposure, prior test-set exposure, evidence kept separate from rationale).
- [x] Set the operating constraints before any corrected rerun, with their rationale, exposure and feasibility limitations documented:
  - Pass 1 NPV ≥ 0.95;
  - Pass 1 positive leakage ≤ 0.15 of all positives;
  - Pass 1 routed-set support ≥ 125;
  - Pass 2 precision ≥ 0.30;
  - Pass 2 routed-set support ≥ 75;
  - no absolute positive-leakage count cap.
- [x] Define the lexicographic coverage objective: maximise feasible Pass 1 routed rows, then feasible Pass 2 routed rows on that router's own remainder.
- [x] Implement an arm-neutral obligation checker: exact count-based comparisons, common metric definitions and denominators, explicit handling of zero support, insufficient class support, undefined ratios and invalid configurations.
- [x] Keep fit-time enforcement separate from observed outer-OOF performance: an outer-OOF shortfall is never treated as a fit-time violation.
- [x] Remove `match_glass_oof` from the new shared-contract pathway (refused by name) while keeping historical reproduction intact (Steps 4A–4B).

**Constrained optimisation and selector safeguards** — Commit 4, Step 4B

- [x] Shared-contract adapters for GLASS and RF Router, with no shared fitted estimator and no change to historical defaults.
- [x] Enforce the GLASS Pass 1 obligations on the selected rule union: union NPV, union positive leakage, minimum union support, exact union-membership accounting.
- [x] Enforce the GLASS Pass 2 obligations on the selected rule union: union precision, minimum union support, zero-positive atoms included, exact union-membership accounting.
- [x] Opt-in coverage maximisation under the shared constraints for both arms (GLASS Pass 2 union-row objective `"rows"`), the historical positive-yield objective kept as the default.
- [x] Deterministic candidate ordering, selection and constraint-boundary comparisons.
- [x] Validate ILP solutions, time-limited incumbents and greedy fallbacks independently against the shared obligations; fail explicitly on infeasible or unsupported selections, never relax silently.
- [x] Record CBC termination, incumbent, bound and MIP-gap evidence where available, without changing solver tolerances or time limits.
- [x] Propagate the obligations into the GLASS inner-fold selectors and the Pass 2 remainder selection; preserve training-side population boundaries, arm-specific estimator ownership and held-out-label isolation; no notebook rerun activated.
- [x] Optimise the GLASS overlap diagnostics by caching rule-matching masks: identical values, unchanged rule search and selection, deterministic equivalence tests. Local benchmark, 124 rules × 2,000 rows (7,626 pairs): 2.39 s → 0.041 s, about 58× faster for the overlap diagnostics only, not overall training. Solver limits, tolerances, candidate generation and default configurations unchanged; remaining costs recorded; further optimisation only if later profiling justifies it.

**Pre-registered PR 38 evaluation protocol** — Commit 4, Step 4C ([protocol](methodology/pr38_evaluation_protocol.md))

- [x] Primary GLASS-versus-RF metrics and acceptance hierarchy (validity gates V1–V8, outcomes O1–O4), and secondary diagnostics for coverage, NPV, leakage, precision, recall, abstention and class-conditional routing.
- [x] Comparisons against LR/EBM at frozen upstream operating points and matched flag rates, with row-level upstream identity checks.
- [x] Paired pre- versus post-correction comparisons, including routed-row identity changes and unique contributions, with no unsupported causal attribution among F17, the shared constraints, Pass 2 coverage optimisation and any ablation.
- [x] Pooled-OOF and fold-level reporting, the registered stratified paired bootstrap, shortfall classification and infeasible-candidate handling.
- [x] Explicit reporting of solver non-optimality, missing bounds, greedy fallback, unsuccessful folds and insufficient support; a fold is accepted only from its complete selection records.
- [x] Complementarity-aware selection as a secondary ablation with pre-registered adoption criteria C1–C9: unique positives, flag budget pooled and per fold, constraint preservation, coverage, uncertainty and computational feasibility; adoption needs an equivalent RF objective or a documented asymmetry.
- [x] Structural, methodological and reproducibility requirements for artifact acceptance, kept separate from comparative performance.
- [x] Evidence tiers distinguishing historical exploratory OOF, corrected nested cross-fitted evaluation and independent external validation; prior test-set exposure documented; test-derived results prohibited from informing selector choice, contract revision, thresholds or adoption.
- [x] Version-control the protocol and validate its executable invariants.

**Contract freeze and configuration provenance** — Commit 4, Step 4D ([contract record, section 24](methodology/pr37_commit4_contract_record.md))

- [x] Freeze the shared contract as the PR 38 primary operating policy replacing `match_glass_oof`: `pr37-stage2-primary` 1.0.0, fingerprint `0817825b9eecbd11`, every value unchanged from the working profile, loaded only by `load_frozen_contract`.
- [x] Freeze the PR 38 protocol `pr38-stage2-evaluation` 1.0.0, fingerprint `245a1a68fb68cf4e`: D-C1–D-C11 resolved, registered criteria distinguished from descriptive heuristics, bound to the frozen contract and arm profiles.
- [x] D-C10: pin the audited flag fingerprints (LR `64d511516d69e73e`, EBM `907a7d2d1cad6908`) and pass the real-artifact re-verification `tests/realdata/test_stage2_freeze_artifacts.py`; a failure voids the freeze.
- [x] D-C11: accept a GLASS fold only with selection reports that correspond exactly to its required selections.
- [x] Freeze the GLASS arm profile `pr37-glass-router-primary` 1.0.0 (`198731d5cdf263f4`): the historical legacy per-rule gates, rule counts, novelty, complexity and ranker settings, union obligations only from the contract, F17 primary settings, GLASS `full` kept as the S1 sensitivity.
- [x] Give the RF Router an independent, versioned configuration `pr37-rf-router-primary` 1.0.0 (`4e3f9c07e6ecc8ad`): the historical hyperparameters preserved exactly, their tuning provenance and full-training-data exposure recorded, no run-time read of a GLASS artifact; historical artifacts and pathway unchanged.
- [x] Record the frozen identities PR 38 needs and complete the Commit 4 methodology record.
- [x] Record the owner's formal approval (Phillip Harris, 2026-11-05) of the four frozen records, including the legacy GLASS profile as the primary PR 38 configuration; strict loaders refuse any record without a confirmed approval.
- [x] Reconcile this roadmap with the Step 4D implementation and validation evidence.
- [x] Run the final combined validation: fast, Stage 2/parity, real-data (including the freeze re-verification) and slow suites, historical baseline `--verify` and `--check`, provenance and Git inspection.
- [x] Create and push **one Commit 4** with Steps 4A–4D (`f7a987e`): every source, test, record and document included; Git-ignored model artifacts and the frozen baseline unchanged.

**Local validation evidence**

- [x] Steps 4A–4B: 2,450 fast tests, 1,381 Stage 2/parity tests, 87 real-data tests and the slow ILP test passed; Step 4B patch integrity with 9/9 prerequisites, 12/12 files and zero conflicts; clean Git whitespace checks; the historical baseline unchanged (`--verify` and `--check` 19/19).
- [x] Historical RF two-pass metadata and sweep files confirmed in the external PR 36 SHA-256 snapshot.
- [x] Step 4C revision 3: 20/20 prerequisites, 6 files, zero conflicts, `CHECK OK`; 120 focused, 218 Step 4A/4B, 1,501 Stage 2/parity and 2,570 fast tests passed; 90 real-data tests collected and passed; slow suite 8 passed; baseline `--verify` and `--check` 19/19; no baseline-pinned code or model artifact changed.
- [x] D-C10 read-only real-artifact audit: artifact SHA-256 identities, row and label alignment, EBM outer folds, frozen thresholds, per-row decisions, and LR / EBM / union / intersection counts 7,727 / 6,113 / 8,001 / 5,839.
- [x] Step 4D, before Commit 4: freeze re-verification 3, Step 4D 64, Step 4C 120, Step 4A/4B 218 (1 deselected), Stage 2/parity 1,565 (8 deselected), fast 2,634 (101 deselected), real-data 93 and slow 8 tests passed; baseline `--verify` 19/19 and `--check` identical; `git diff --check` clean; the four frozen records version-controlled, not Git-ignored.
- [x] Commit 5, revisions 2–3: handoff and freeze 107, Stage 2/parity 1,608 (8 deselected), fast 2,677 (101 deselected), real-data 93 (including the freeze re-verification) and slow 8 tests passed, with only the pre-existing Dash deprecation warning; baseline `--verify` 19/19 and `--check` identical; patch `--check` and `git diff --check` clean.
- [x] Commit 5, final checks: external snapshot 37/37 archived files intact and 37/37 live artifacts identical (read-only verifier); HEAD and upstream at `f7a987e`; exactly the seven Commit 5 files changed, nothing staged; notebooks, historical baselines and Stage 4 paths unchanged; frozen fingerprints as approved; Python and ML dependency versions recorded.

##### Commit 5 — Final Validation, Documentation and PR 38 Handoff

Commits 1–4 already provide, and regression-protect, the component verification the earlier Commit 5 plan listed: three-state routing; Pass 1 precedence and remainder eligibility; unique positives, overlap, redundancy and suppression; matched rates and tie handling; exact contract arithmetic; GLASS union obligations and zero-positive atoms; solver outcomes and fallbacks; identical obligations without shared estimator state; contract and protocol identities; protocol invariants; cross-arm identity mismatches; held-out-label isolation; F17 and F31 requirements; Stage 4 compatibility; historical defaults; overlap-diagnostic equivalence. Commit 5 re-runs that evidence on the final branch rather than rebuilding it. No new selector architecture, corrected training or contract change. The audit found one material gap, closed in Commit 5: the GLASS ranker settings reach the cross-fit as separate arguments that no strict check covered (`check_glass_primary_ranker`).

**Step 5A — Final integration verification**

- [x] Audit the regression coverage against the PR 37 requirements by inspecting the tests and their assertions, not only their counts; close the material gap found: `check_glass_primary_ranker`, tested in `tests/stage2/test_stage2_pr38_handoff.py`.
- [x] Run the fast, Stage 2/parity, real-data (including `test_stage2_freeze_artifacts.py`) and slow suites on the final branch; resolve any newly introduced failure or actionable warning; record counts, deselections, warnings and environment (*Local validation evidence*).
- [x] On the final branch, verify the historical baseline (`--verify`, `--check`), the 15 pinned code identities, the external 37-artifact SHA-256 snapshot (read-only verifier, archived and live copies) and the frozen contract, protocol and arm-profile fingerprints (*Local validation evidence*).

**Step 5B — Documentation and provenance**

- [x] Make one consistency pass over the methodology records (diagnostics, F17, F31, contract record, protocol, arm profiles) against the final schemas, definitions, defaults and behaviour, correcting drift only: the status after Commit 4, the M2 and M5 resolutions at the freeze, the Commit 3 handoff's resolved decisions.
- [x] Confirm the model artifacts are unchanged and Git-ignored, and that no notebook, Stage 4 interface, historical baseline or model artifact changed unintentionally: 37/37 live artifacts identical to the snapshot; no notebook, Stage 4 interface or baseline change in PR 37 (Stage 4: one documented docstring, W2).

**Step 5C — PR 38 readiness**

- [x] Document the PR 38 execution gates: G3 fit-only preflight; G13 label-derived feature audit; F17 nested-training safeguards; F31 R1–R8 including the 5% boundary-strip trigger; explicit handling of infeasible fits, missing incumbents, insufficient support and shortfalls; rerun inputs and fold identities; artifact persistence and provenance; structural versus comparative acceptance; no retroactive change to frozen records ([Commit 3 handoff](methodology/pr37_commit3_handoff.md), [F17](methodology/pr37_f17_decision_record.md), [F31](methodology/pr37_f31_disposition_record.md), [protocol, sections 10–16](methodology/pr38_evaluation_protocol.md)).
- [x] Assemble the single PR 38 handoff record ([PR 38 handoff](methodology/pr38_handoff.md)): frozen identities; preservation; preflight gates; during-rerun requirements; post-rerun acceptance and evaluation; secondary analyses; final acceptance; the historical-versus-corrected comparison and the exploratory profiling plan.
- [x] Sign off readiness against the frozen records: baselines reproducible; F17, F31 and the `match_glass_oof` replacement frozen; identical constraints for both arms; independent estimator ownership and versioned arm configurations; protocol frozen; complementarity-aware selection a secondary ablation (Step 5C review, 2026-11-05).

**PR 37 closure** — implementation completion is Commit 5 validated, committed and pushed; closure is the review and merge

- [ ] Review the complete PR 37 diff, staged files, commit history and branch state, with no unintended generated or local files; create and push Commit 5 — **pending**.
- [ ] Perform the final PR 37 readiness review, then merge PR 37 — **pending**.

**Boundary.** PR 37 establishes and freezes the methodology, contracts, diagnostics, provenance and acceptance rules. PR 38 implements corrected-selector deployment, runs the full corrected outer cross-fits, regenerates artifacts and evaluates the corrected results under the frozen protocol. Out of scope for PR 37: corrected five-fold Stage 2 reruns, model artifact regeneration, notebook deployment of corrected selectors, full complementarity-aware selector experiments, and independent-validation claims based on previously exposed test data.

#### PR 38 — Stage 2 Cross-Fit and Router Complementarity

Runs only under the frozen PR 37 specification: contract `pr37-stage2-primary` 1.0.0, protocol `pr38-stage2-evaluation` 1.0.0 and the arm profiles `pr37-glass-router-primary` and `pr37-rf-router-primary` 1.0.0 ([identities: contract record, section 24](methodology/pr37_commit4_contract_record.md)). Changing any of them needs a new version and approval before corrected results are inspected. The execution order, the evidence each step leaves and the failure conditions are in the [PR 38 handoff record](methodology/pr38_handoff.md).

**Historical vs. Corrected Stage 2 Artifact Comparison** — the first three items before any notebook execution or corrected cross-fit; the comparisons after the corrected artifacts pass structural and provenance validation, under the frozen protocol ([handoff, sections 3 and 10](methodology/pr38_handoff.md))

- [ ] Preserve and inventory existing historical GLASS and RF Router artifacts before any new notebook execution or corrected cross-fit.
- [ ] Record artifact identities, dataset/split fingerprints, feature contracts, configuration identities, and source provenance.
- [ ] Store newly generated artifacts separately to prevent overwriting historical results.
- [ ] Verify row alignment, fold identity, target definitions, and metric comparability before conducting paired analyses.
- [ ] Compare historical versus corrected three-state routing decisions, including a row-level Pass 1 / Pass 2 / abstain transition matrix.
- [ ] Compare total coverage, per-pass coverage, abstention, NPV, positive leakage, Pass 2 precision, and class-conditional behavior.
- [ ] Compare unique-positive discoveries, upstream LR/EBM overlap, redundancy, and false-positive suppression.
- [ ] Compare selected symbolic rules, rule counts, selected-union coverage, selector outcomes, solver termination, and fallback behavior where equivalent evidence exists.
- [ ] Compare execution runtimes for candidate generation, overlap diagnostics, ILP selection, nested folds, and total Stage 2 execution where comparable timing measurements are available.
- [ ] Distinguish effects associated with F17, shared constraints, coverage optimization, and any complementarity-aware ablation. Do not claim causal attribution that the available comparisons cannot support.
- [ ] Generate reproducible before-and-after comparison tables, visualizations, and a documented research summary.
- [ ] Preserve the immutable 19-file pre-correction research baseline permanently.
- [ ] Verify archival integrity and reproducibility before retiring redundant historical working artifacts. Never automatically delete historical evidence required for research reproduction.

**Corrected-selector Stage 2 rerun**

- [ ] Validate the PR 37 F17/F31 obligations before treating any corrected rerun as primary evidence ([Commit 3 handoff](methodology/pr37_commit3_handoff.md)):
  - before the full 5-fold primary reruns, pass the preflight and implementation gates: G1 configuration, G2 wiring, G3 feasibility preflight, G5 persistence, G8 explicit rejection, G9 independent contract, G13 feature-engineering audit (a confirmed material own-label dependency must be corrected and validated before primary results are accepted)
  - with the primary reruns, verify G4 isolation provenance, G7 reproducibility, G10 versioned artifacts and pins, G11 baseline preservation; record the mandatory F31 requirements R1–R5 (including the R2 boundary-strip review at the 5% trigger), R7 and R8 (R6, band-level spread, is an optional diagnostic)
  - afterwards, G12: sensitivity analysis S1 (GLASS `full` vs `oof_remainder`, descriptive only), begun once the applicable primary-run gates other than G6 have passed; G6 Pass 1 invariance is verified during S1, and S1 is not accepted unless it passes
- [ ] Re-verify the frozen specification on the artifacts (`tests/realdata/test_stage2_freeze_artifacts.py`) before any run, with the historical Stage 4 arbiters pinned by `$BANK_ARTIFACT_MANIFEST`, and obtain it only through the strict entry points (`load_frozen_specification()`, `glass_primary_config()`, `rf_primary_config()`), checking every config used with `check_glass_primary_config` / `check_rf_primary_config`
- [ ] Run the corrected GLASS Router under the frozen Stage 2 operating contract (`glass_primary_config()`, with `ranker_settings()` checked by `check_glass_primary_ranker`; the GLASS-arm RF is not re-tuned)
- [ ] Run the RF Router under the same operating contract with its independent frozen profile (`rf_primary_config()`); no hyperparameter is read from a GLASS artifact
- [ ] Rerun the full 5-fold Stage 2 outer cross-fit for both arms
- [ ] Persist each fold's fit-time evidence: GLASS `selection_reports()` (inner folds included) and solver logs, RF selection records, reproduction reruns for unproven incumbents, fit-time reviews of greedy fallbacks and the M6 GLASS inner-OOF Pass 1 diagnostics; classify every fold from complete records (D-C11)
- [ ] Regenerate Stage 2 artifacts
- [ ] Verify OOF provenance, fold identity, routing semantics, and constraint compliance

**Corrected-router complementarity** — evaluated under the frozen protocol, in the runner order of protocol section 16; before-and-after comparisons are in the comparison section above

- [ ] Re-run the three-state router trace
- [ ] Re-measure Pass 2 unique positives at matched flag rate
- [ ] Re-measure Pass 1 false-positive suppression
- [ ] Measure GLASS Router vs LR/EBM OOF score and decision overlap
- [ ] Produce the comparative report with no gaps, with structural acceptance decided separately from performance

**Complementarity experiment**

- [ ] Evaluate an optional complementarity-aware selector using train-side OOF upstream predictions only
- [ ] Test rules favoring:
  - positives Stage 1 LR misses
  - rows where LR and EBM disagree
- [ ] Compare standard union-coverage selection vs complementarity-aware selection
- [ ] Treat complementarity-aware selection as an ablation unless the frozen adoption criteria C1–C9 pass (`evaluate_adoption`); any failure keeps the primary
- [ ] If adopted, provide the RF Router an equivalent objective or document the asymmetry explicitly

**Final Stage 2 freeze**

- [ ] Select the final Stage 2 selector design based on the pre-declared comparison
- [ ] If complementarity-aware selection is adopted, rerun both arms under the frozen operating contract; it becomes the primary only under a later, versioned protocol
- [ ] Regenerate final Stage 2 OOF and test artifacts
- [ ] Verify final OOF provenance, constraint compliance, and mirrored contracts
- [ ] Freeze the Stage 2 implementation and artifacts for downstream regeneration

**Exploratory historical-notebook profiling** — a separate activity, never PR 38 evidence ([handoff, section 11](methodology/pr38_handoff.md))

- [ ] Profile the historical notebooks in a fresh kernel in an isolated clone, every historical artifact and identity verified unchanged before and after; record cell and sub-phase timings with their provenance

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