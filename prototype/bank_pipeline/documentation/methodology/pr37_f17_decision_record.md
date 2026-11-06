# F17 Final Methodological Decision Record — Pass 2 Training Population

| | |
|---|---|
| Author | Phillip Harris |
| Record | F17 decision (Stage 2 Pass 2 training population) |
| PR / step | PR 37, Commit 3, Step 3B |
| Branch | `pr-37-router-baseline-contract` |
| Approved | 2026-11-04, with one added safeguard (gate G13) |
| Evidence | [`pr37_f17_investigation.md`](pr37_f17_investigation.md) (Step 3A) |
| Related | [`pr37_f31_disposition_record.md`](pr37_f31_disposition_record.md) (Step 3C); handoff in [`pr37_commit3_handoff.md`](pr37_commit3_handoff.md) |
| Line numbers | At `e5c437c`, before Commit 3, unless a file exists only in Commit 3 |

PR 37, Commit 3, Step 3B. Approved on 2026-11-04 with one added safeguard (gate G13). Nothing in this record has been activated, and no corrected-router reruns have been conducted.

## 1. Decision and status

**F17 is decided as follows: each router fits Pass 2 on its own out-of-fold Pass 1 remainder, with both nested safeguards on.** GLASS `full` is kept only as a single-factor sensitivity analysis. The policy is fixed now, before any corrected PR 38 result exists, and is not revisited on the basis of those results.

| Item | Decision |
| --- | --- |
| Decision ID | F17: Stage 2 Pass 2 training population |
| Status | Approved in Step 3B (2026-11-04), with the L1 feature-engineering audit added as mandatory gate G13. Not activated |
| Primary, GLASS Router | `pass2_population="oof_remainder"`, `pass1_oof_folds=5`, `random_state=42`, `remainder_min_size=500`, `inner_feature_ranker="fold_nested_rf"` |
| Primary, RF Router | The OOF Pass 1 remainder (its existing design), `eligibility="nested_t1"`, `eligibility_folds=5`, `n_oof_folds=5`, `random_state=42`, `remainder_min_size=500` |
| Remainder ownership | Each router builds its own remainder from its own Pass 1 decisions. Neither uses the other's row mask |
| Secondary analysis | S1: GLASS `full` versus GLASS `oof_remainder`, everything else identical (section 6) |
| Constraints | The frozen Commit 4 contract, applied independently to both arms. `match_glass_oof` is not the PR 38 operating policy |
| What is complete | Opt-in safeguard code and synthetic regression tests (Commit 3; validated locally before this record was added) |
| What is deferred | Activation, real-data feasibility and every corrected-router run (PR 38) |

## 2. Historical configuration and justification

Historically the arms fitted Pass 2 on different populations: GLASS on every outer-training row, RF on its OOF Pass 1 remainder. Both arms applied the same decision-time rule. The frozen baseline records this asymmetry as caveat D12.

|  | GLASS Router | RF Router |
| --- | --- | --- |
| Pass 2 fitting population | `full`: all outer-training rows (`glass_cascade.ipynb` Cell 13, profile `"legacy"`; `glass_router_pipeline.py:202–214`) | OOF Pass 1 remainder, `p1_oof >= t1` (`router/two_pass.py:115`) |
| Decision-time eligibility | Pass 2 decides only rows that Pass 1 left (`glass_router_pipeline.py:360–382`) | The same (`router/base.py:58–100`) |
| `oof_remainder` support in code | Implemented but unused (`glass_router_pipeline.py:217–305`) | Always on |

### Why this policy

1. **GLASS `full` is misaligned with its own decision-time eligibility.** Pass 2 candidates, per-rule gates and the union-positive ILP objective are computed over every training row (`ilp_rule_selector.py:337–358`). At prediction, though, Pass 2 sees only the rows Pass 1 leaves. The ILP therefore counts positives that Pass 1 claims first, and each stored rule precision describes a population the rule never serves.
2. **`full` would undermine the RF Router's two-pass design.** With `RF_PASS2_PARAMS = None`, the Pass 2 forest shares Pass 1's parameters and seed (`config.py:278–281`; `two_pass.py:93–94`). Fitted on the same rows, it would be the same forest, which collapses the cascade into the existing single-RF ablation. This follows by construction; it was not executed.
3. **`oof_remainder` gives a directly controlled comparison.** Both arms then fit Pass 2 on the population Pass 2 actually decides, and membership is decided out of fold. A common Commit 4 Pass 2 precision floor then refers to the same conditional quantity, P(y = 1 | flagged, Pass 1 remainder), in both arms. Under the historical asymmetry it would not.
4. **Each router must build its own remainder.** A router's Pass 2 serves the rows its own Pass 1 leaves. A shared mask would train one arm's Pass 2 on rows its Pass 1 routes away, and omit rows it must decide. It would also import the other arm's fitted Pass 1 decisions, which is a cross-arm dependence on fitted state.
5. **Equal obligations do not require identical algorithms.** The obligations are the protocol: the same outer folds, labels and outcome definitions, the same nesting discipline, the same frozen contract and the same evaluation rules. The learners differ by design: rule search plus ILP against forest plus thresholds. That difference is what the comparison measures.

No contradiction with the Step 3A evidence was found. One new limitation qualifies the isolation claims without reversing the decision (section 7, L1). Its audit is mandatory gate G13.

## 3. Selected protocol and nested safeguards

**Both corrected pathways are selected for the PR 38 primary.** They give the two arms the same nesting obligation: no router-level component that decides a row's Pass 2 membership was fitted or tuned using that row. Their real-data feasibility is unproven (gate G3). Complete inner-fold isolation is not claimed until the feature-engineering audit passes (gate G13; section 7, L1).

| Property | RF `eligibility="nested_t1"` | GLASS `inner_feature_ranker="fold_nested_rf"` |
| --- | --- | --- |
| Removes | The row's own label from the `t1` that decides its Pass 2 membership | An inner-fold Pass 1 search ordered by an RF that saw the inner held-out rows |
| Inner-fold label isolation | A row in inner fold j is scored by a forest fitted without fold j, and cut by `t1_j` solved only from rows outside j (`oof/eligibility.py`) | Inner fold j's ranker and Pass 1 rules are fitted only on rows outside j |
| Threshold nesting | Eligibility uses a per-fold `t1_j`. The deployed `t1` is unchanged and is applied only to rows outside P_k. `t2` is solved on the remainder's inner OOF | No threshold. Rule selection is in-sample (section 7, L2) |
| Provenance | `fit_report_.remainder`: per-fold `t1_j`, rows, eligible rows, fold plan; cross-fit `pass2_eligibility` | `fit_report_.inner_ranker_provenance`: per fold `n_train`, label fingerprint, estimator, seed, feature-order fingerprint; cross-fit `glass_inner_feature_ranker` |
| Fitted-estimator independence | Fresh forests in every fold; nothing reused | A fresh ranker per inner fold, using the fold-RF recipe. Refused if it is the outer `rf_model` or another fold's ranker |
| Determinism | Fixed folds and seeds. Bit-identical scores need `n_jobs=1`: parallel `predict_proba` sums trees in a nondeterministic order | Fixed seed; stable importance sort |
| Outer cross-fit | Unchanged loop; outer-fold label isolation tested | Unchanged loop. `crossfit_stage2` supplies the factory and requires `glass_rf_model="fold_rf"` |
| Cost | About 25 extra Pass 1 forest fits per router fit (5 × 5) | 5 extra RF fits per outer fold, on top of the 6 Pass 1 selections `oof_remainder` already needs (about 4–6× a `full` fit on synthetic data) |
| Failure behaviour | An infeasible `t1_j` raises; a remainder under 500 rows or single-class raises; `resolve_at` is refused; mixed rules across outer folds are refused | A missing factory is refused before any fit; a reused or invalid ranker raises; an infeasible inner selection propagates |
| Validation status | Synthetic regression tests only | Synthetic regression tests only |

**No fallback, by design.** A failure under the primary protocol never reverts to `global_t1`, `outer_rf` or `full`. It is recorded as a feasibility finding. Any constraint relaxation is decided uniformly for both arms, before results are produced, under Commit 4's no-feasible-candidate rule. If V9 shows the cost or feasibility is unacceptable, the decision returns to Step 3B's owners.

## 4. Rejected alternatives

| Alternative | Rejected because |
| --- | --- |
| A: `full` for both | It collapses the RF two-pass cascade (section 2, point 2), and it keeps GLASS's misalignment |
| C: architecture-specific, keeping GLASS `full` as primary | It carries a known misspecification into the frozen comparison. Fit-time Pass 2 precision would mean different things in the two arms. Kept only as sensitivity analysis S1 |
| The `"recommended"` GLASS profile as "the `oof_remainder` run" | It also changes support, rule counts, precision floors, leakage gates, the recall ceiling, the novelty floor and the RF penalty lambdas, which confounds F17 with tuning |
| An in-sample remainder (Pass 2 fitted on rows the deployed Pass 1 leaves, with Pass 1 fitted on those same rows) | A row's own label would decide its own Pass 2 membership: the failure `oof_remainder` exists to prevent (`two_pass.py:14–18`) |
| One shared remainder mask for both arms | A structural mismatch, plus a cross-arm dependence on fitted state (section 2, point 4) |
| `oof_remainder` with the historical inner pathways (`global_t1`, `outer_rf`) as primary | It leaves two known inner-level selection dependencies in place when opt-in removals exist. Not recommended. Retained only as a recorded fallback **decision**, to be taken explicitly, never automatically, if G3 fails |
| Choosing the primary after seeing the S1 results | That is outcome-driven selection, and it is prohibited |

## 5. Cross-arm independence

**Principle.** GLASS and the black-box cascade each optimise their own routing behaviour under the same externally specified experimental constraints. Neither arm is calibrated or constrained to match the other arm's realised performance.

**Traced finding (Step 3A, pinned by the Commit 3 tests).** `glass_rf_model="fold_rf"` does not share a fitted estimator with the black-box arm. `crossfit_stage2` fits a GLASS-owned fold RF from `rf_params` (`pipeline/crossfit.py:240–253`). The black-box arm reads only the hyperparameter dict from the GLASS RF artifact (`tuning/legacy.py:115–138`). Because the recipes match, GLASS's fold RF and the RF Router's refitted Pass 1 forest are probably identical by construction. That is the same computation, not shared state.

| Kind of sharing | Status | Examples |
| --- | --- | --- |
| Dataset partitions, outer folds, labels | Legitimate, and required | Fold plan `8aea082fcaf371b7`; identical inner folds (V1) |
| Algorithms and comparable hyperparameters | Legitimate | Both use RFs with the tuned parameters; the GLASS ranker uses the fold-RF recipe |
| Evaluation requirements | Legitimate, and required | One frozen Commit 4 contract; common outcome and coverage definitions |
| Fitted estimator state | Improper | None found; pinned by `tests/parity/test_stage2_cross_arm_independence.py` |
| Held-out information | Improper | The outer held-out fold never reaches its own model (tested for both arms) |
| Performance-derived operating constraints | Improper | `match_glass_oof`: RF constraints read off GLASS's pooled OOF metrics. Historical provenance only; refused under the independent pathway (`shared/stage2/contract.py`) |

**Two kinds of dependence, kept distinct.**

- **Inner-fold selection leakage** happens within one arm: a component that decides a training row's Pass 2 membership was fitted with that row. Examples are RF `global_t1` and GLASS `outer_rf`. It is addressed by the nested safeguards (section 3).
- **Cross-arm performance dependence** happens between arms: one arm's operating point is a function of the other's results, as with `match_glass_oof`. It is eliminated from the primary comparison. Commit 4 freezes independent values, and PR 38 activates them.

Step 3B chooses no numerical constraints.

## 6. Secondary sensitivity analysis S1

**S1 runs GLASS twice, once with `full` and once with `oof_remainder`, changing nothing else.** It is descriptive. It quantifies what the population policy does to GLASS, and it is never used to choose the primary.

### Held constant

| Factor | Held at |
| --- | --- |
| Outer folds and labels | Fold plan `8aea082fcaf371b7`; the same `y_train` |
| Feature engineering | The same per-fold fitted encoders and bins |
| RF-guided ranking | `glass_rf_model="fold_rf"`; `inner_feature_ranker="fold_nested_rf"` in both configs. It acts only under `oof_remainder`, whose inner fits are part of what that policy is |
| Symbolic selector and ILP | The PR 38 corrected selector, solver limits, novelty floors and lambdas, all identical |
| Rule budgets and support | Identical |
| Seeds | `random_state=42` everywhere |
| Operating contract | The same frozen Commit 4 contract |
| Evaluation | The Step 2A–2D diagnostic definitions and code |

### Required comparisons (per outer fold and pooled)

1. **Pass 1 rule invariance.** The deployed Pass 1 segments and precisions must be identical in both runs. Any difference is a defect and stops the analysis.
2. **Pass 2 fitting population:** rows, positives and prevalence (`full`: all of P_k).
3. **Pass 2 outcomes on the outer OOF:** rows, TP, FP, precision and coverage.
4. **Unique positives** against the LR/EBM union (Step 2C method).
5. **Matched-flag-rate comparisons:** per-fold matched counts, LR calibrated probabilities, EBM ranked within each fold, deterministic label-free tie-breaks (Step 2C).
6. **Fold-level variation and feasibility:** selector status, lexicographic guarantee, any infeasibility.
7. **Differences relative to `pr37_stage2_pre_correction_v1`.** These are confounded by the selector correction and the contract change, so only the within-S1 difference is attributed to the population policy.

### Interpretation rules

- The primary policy is fixed before S1 runs. S1 results are reported whatever their direction.
- The two runs are paired by fold. No significance claim is made unless a test is pre-specified, and none is specified here.
- No RF counterpart is run: RF under `full` is, in effect, the existing single-RF ablation.

## 7. Precision-estimation asymmetry and known limitations

The policy aligns what each arm fits Pass 2 on, not how honestly each arm estimates its own precision. The safeguards nest the routers, not the feature engineering that feeds them.

### L1. Outer-level target-encoded features (new in Step 3B)

- **Evidence.** `engineer_features` fits its engineer on P_k with the labels (`shared/stage2/feature_engineering.py:614–630`). `neighborhood_subscription_density` is a KDE over reference rows' labels: all of P_k, or a random `n_ref` subset on large data (`:242–268`). The derivative features fit local rates on the labels (`:292–297`). Both cross-fits fit these encoders once per outer fold, on all of P_k (`pipeline/crossfit.py`, step 1; `rf_router/crossfit.py:118`).
- **Consequence.** In both arms, and under both the historical and the nested pathways, inner-OOF Pass 1 decisions use feature values that can carry the scored row's own label, whenever that row is a KDE reference point. Inner-fold isolation therefore holds for models, rankers and thresholds conditional on the engineered features, but not for the feature engineering itself.
- **What is unaffected.** Outer-fold isolation: each held-out fold is transformed by encoders fitted only on P_k.
- **Status.** Both arms consume the same encoders, but that does not make the effects equivalent: rules over binned segments and forests may respond differently to the same leakage, and its size is unmeasured. L1 stays a documented limitation, and its audit is mandatory PR 38 preflight gate G13 (section 8). If the audit confirms a material own-label dependency, meaning that a row's own label reaches the features that decide its inner-OOF eligibility, a cross-fitted or leave-one-out correction must be specified, resolved and validated before complete inner-fold isolation is claimed. The same applies before the primary corrected-router results can be accepted. No fix is implemented in Step 3B or Step 3D. The Commit 3 synthetic label-perturbation tests use pre-binned features without target encoding, so they establish router-level isolation only.

### L2. Precision-estimation asymmetry

- **GLASS:** rule precision, which drives the per-rule gates, the ILP and the stored confidence, is estimated in-sample on the fitting population (all of P_k under `full`, the remainder under `oof_remainder`). The rules were selected on those same rows.
- **RF:** `t2` and both band precisions are estimated on inner-OOF scores, from forests that did not train on the scored rows.

`oof_remainder` makes the conditional quantity the same; it does not make the procedures equivalent. GLASS's fit-time precision is selection-optimistic, and more so for low-support rules on a smaller remainder. How the shared constraints are enforced at fit time belongs to Commit 4. Any confidence implications belong to F31 (Step 3C). Confidence blending is unchanged.

### Other limitations

| ID | Limitation |
| --- | --- |
| L3 | Global hyperparameters and configuration (F24). RF parameters were tuned once over all of `X_train`; the GLASS legacy profile was set with OOF results visible; the corrected selector's configuration will be chosen with historical evidence seen |
| L4 | The training remainder is not the runtime remainder, by design. GLASS's comes from inner-fold rule sets, the RF's from per-fold `t1_j`; at runtime each uses its deployed Pass 1. Their agreement is unmeasured (O2) |
| L5 | Under `nested_t1`, `RFRouter.predict_train_oof` cuts at the deployed `t1`, so it no longer reproduces the training remainder. It is a diagnostic only; Stage 4 consumes the outer cross-fit outputs |
| L6 | With `n_jobs=-1`, forest probabilities differ in the last bits between runs, so routes can differ at exact ties |
| L7 | The safeguards are validated on synthetic data only. GLASS `oof_remainder` has never run on real data; 3 of 5 synthetic outer folds were observed |
| L8 | The nominal test split was evaluated in PRs 33–36 and cannot be described as untouched |

## 8. PR 38 gates

PR 38 may treat F17 as active only when every mandatory gate passes. A failure stops that work and is reported; it is never worked around.

### Mandatory

| Gate | Requirement | Passes when |
| --- | --- | --- |
| G1 Configuration | GLASS: PR 38's frozen profile plus `pass2_population="oof_remainder"`, `pass1_oof_folds=5`, `random_state=42`, `remainder_min_size=500`, `inner_feature_ranker="fold_nested_rf"`, and nothing from `"recommended"`. RF: `eligibility="nested_t1"`, `eligibility_folds=5`, `n_oof_folds=5`, `random_state=42`, `remainder_min_size=500` | A config diff against the PR 38 base profile shows exactly these keys |
| G2 Wiring | The GLASS full-train fit (Cell 13) passes `feature_ranker_factory=make_rf_feature_ranker_factory(rf_result.params, RANDOM_STATE)` (without it the fit refuses, by design). The cross-fit keeps `glass_rf_model="fold_rf"`. RF full-train and cross-fit routers come from one config factory | Protocols and fit reports declare the same policy on the train and test sides, and Stage 4 accepts them |
| G3 Feasibility preflight (V9) | One outer fold, real data, both arms, before any 5-fold run | No infeasibility; each remainder has at least 500 rows and both classes; every inner selection is solved with its status recorded; wall time recorded |
| G4 Isolation provenance | For every outer fold, on real data | Remainder row keys lie within P_k and outside the held-out fold; each GLASS inner ranker's rows equal its inner-train rows; no RF eligibility forest is fitted on the fold it cuts; no held-out row appears in any fit |
| G5 Persistence | Saved artifacts, not just memory, record per fold: remainder rows, positives and prevalence; RF `t1_j` and eligible counts; GLASS inner rule counts, selector outcomes and ranker provenance | An `oof_remainder` artifact without these records is refused (V4) |
| G6 Pass 1 invariance | Across the S1 pair, verified during S1 (G12) | The deployed GLASS Pass 1 rules are identical in every fold. Mandatory: S1 cannot be accepted unless G6 passes |
| G7 Reproducibility | Rerun at least one outer fold | Identical rule sets, remainders and decisions. Thresholds are bit-identical with `n_jobs=1`; otherwise decisions are identical, thresholds agree within 1e-12, and any tie difference is investigated |
| G8 Explicit rejection | Insufficient support or an infeasible selection | It raises and is recorded. Never a fallback to `global_t1`, `outer_rf` or `full` |
| G9 Independent contract | Both arms consume the frozen Commit 4 contract | RF records `constraint_source="shared_contract"` with the contract fingerprint; `check_rf_constraints_follow_contract` passes; no GLASS artifact is read for constraint values; the GLASS mapping follows Commit 4 |
| G10 Artifacts and pins | Regenerated Stage 2 artifacts under new versioned names, validated by the Stage 4 contract. `test_the_glass_notebook_runs_the_full_population_profile_for_both_fits` and the GLASS `full` expectation in `tests/support/artifact_audit.py` are updated deliberately, in the same change | `artifact_audit.py` is one of the 15 baseline-pinned files, so its edit registers on `stage2_baseline.py --check` as a code-identity change. That is recorded explicitly, or handled by a new baseline version; v1 is never edited |
| G11 Baseline preservation | `pr37_stage2_pre_correction_v1` is untouched | `--verify` passes throughout; corrected results are new versioned outputs that cite it |
| G12 Sensitivity analysis | S1 as specified in section 6 | Begins only after the applicable primary-run gates other than G6 hold for the primary (G1–G5, G7–G11 and G13). G6 is verified during S1, and S1 cannot be accepted unless G6 passes |
| G13 Feature-engineering isolation audit (L1) | A preflight beside G3: one outer fold, real training data, both arms, before any five-fold run. It determines whether a row's own label reaches the engineered features, and their bins, that decide its inner-OOF eligibility. First, compare each row's features from the fitted encoders with features from encoders that exclude the row (leave-one-out, or refitted inside its inner fold). Then compare the resulting inner-OOF eligibility for each arm. Outer-fold boundaries are unchanged, and encoder fit rows and fingerprints are recorded. Test data is never used | Materiality is judged from the recorded audit evidence: the rows, per arm, whose binned features or inner-OOF eligibility change. That judgement is documented before any primary results. Not confirmed: no material own-label dependency affects inner-OOF eligibility, and the result is recorded per arm. Confirmed: a material own-label dependency affects inner-OOF eligibility in either arm. Then its correction must be resolved and validated before the primary corrected-router results can be accepted. A cross-fitted or leave-one-out correction is specified, which states the feature matrix each step uses. It is implemented opt-in, validated on synthetic and real data, and approved. Its effect is reported separately for each arm. Whether the primary adopts the correction before the five-fold runs is decided before any results. Primary runs made without a required correction are not accepted as primary evidence. Complete inner-fold isolation is claimed only after this |

### Optional diagnostics

| ID | Diagnostic |
| --- | --- |
| O1 | Ranker comparison (V6): `outer_rf` against `fold_nested_rf` on one fold |
| O2 | Training-versus-runtime remainder agreement (V7) |
| O3 | RF `nested_check` per outer fold (V8) |
| O4 | RF eligibility sensitivity: `global_t1` against `nested_t1`, descriptive |
| O5 | Real-data outer-fold label perturbation on one fold (expensive) |
| O6 | In-sample fit-time accounting of the GLASS `full` misalignment |
| O7 | Read-only historical diagnostics D1 and D2 |
| O8 | Promoted to mandatory gate G13 at Step 3B approval |

The G13 acceptance condition was clarified at Step 3D review (2026-11-04). If a material own-label dependency affecting inner-OOF eligibility is confirmed, its correction must be resolved and validated before the primary corrected-router results can be accepted. The assessment stays evidence-driven: the audit decides whether a correction is needed, and no feature-engineering change is made in Commit 3.

The G6/G12 ordering was clarified at the same review. G6 needs the S1 pair, so it cannot be a prerequisite of S1. S1 begins after the applicable primary-run gates other than G6 have passed. G6 is verified during S1; it stays mandatory, and S1 cannot be accepted unless it passes.

## 9. Dependencies on Commit 4 and F31

F17 decides the population and its nesting. The items below are not F17 decisions; they are listed so none is left implicit.

### Commit 4 (the operating contract)

- The numerical values: the Pass 1 NPV floor, the Pass 1 leakage-rate cap and the Pass 2 precision floor.
- Whether to keep a leakage count cap. `SharedOperatingContract` forces an explicit `max_pass1_leakage_count`.
- How set-level requirements map onto GLASS's per-rule gates, its Pass 1 union cap and its in-sample selection, and whether GLASS's fit-time enforcement must be checked on inner-OOF estimates (L2).
- The common coverage definition, and the no-feasible-candidate rule. That rule must cover inner-fold infeasibility and refused remainders.
- The RF band-support floors. These are arm-internal estimation safeguards, passed to the adapter explicitly.
- Whether RF hyperparameters should still be read from the GLASS RF artifact. Sharing hyperparameters is legitimate, but that read is a cross-arm file lookup.
- The contract record's own provenance, which must state that historical OOF results had been examined.

### F31 (Step 3C): dependencies only

- GLASS Pass 2 confidence becomes in-sample precision on the remainder, rather than on all rows.
- Under `nested_t1`, RF Pass 2 band precision is measured on the nested training remainder (per-fold `t1_j`), while runtime eligibility uses the deployed `t1` (L4).
- The probability transform and the base rate b (each fit's full training base rate) are unchanged in both arms.
- Stage 4's Pass 2 lift is measured against the global prior.
- Under `nested_t1`, `predict_train_oof` diverges from the training remainder (L5).
- The Stage 4 router-evidence read-back (`rule_positive_rate`) is structurally unaffected.

## 10. Evidentiary status, baseline preservation, Step 3D corrections

**No corrected-router reruns have been conducted, and no PR 38 results exist.**

### What the decision rests on

- **Evidence base:** the Step 3A source trace; the historical OOF diagnostics in the Commit 2 baseline; synthetic regression tests; and a partial synthetic probe.
- **Nature of the decision:** a design decision, not one chosen by comparing corrected metrics. It is not prospectively uninformed, though. The historical GLASS and RF OOF behaviour (Commit 2), the historical test-set exposure (PRs 33–36) and the synthetic probe outputs had all been seen. The Commit 4 constants will be informed the same way and must say so.
- **The Commit 3 safeguard changes** were applied and validated locally before this record was added: 71 new tests, 1,161 Stage 2 and parity tests, 2,230 fast-suite tests and 87 real-data tests passed; the baseline passed `--verify` and `--check`; all 37 artifact hashes matched. Those counts describe the safeguard changes alone. They do not cover the Step 3D documentation change, which is validated separately.

### Baseline preservation

- `pr37_stage2_pre_correction_v1` is the permanent pre-correction reference. It is never edited or regenerated in place; a material change requires a new version.
- The Commit 3 patch touched none of the package's 19 files and none of its 15 pinned code files.
- The 37 historical artifacts are unchanged. PR 38 writes new, versioned artifacts.

### Step 3D wording corrections (documentation only, from L1)

Approved in Step 3B for the Step 3D patch. The safeguards behave as tested, but four statements overclaim in the real pipeline, where features are target-encoded at the outer level:

1. **`blackbox_pipeline/models/rf_router/oof/eligibility.py`, module docstring.** "Nothing reads `y_i`" should say: within the router; the engineered features were fitted on P_k with its labels.
2. **`NESTED_SOURCE` in the same file.** "(no row's own label in its cut)" should become router-level wording that notes the outer-fold feature engineering. Only future artifacts record this string.
3. **`blackbox_pipeline/models/rf_router/config.py`, the eligibility comment.** "no row's own label reaches the cut" needs the same qualification.
4. **`glass_pipeline/glass_router/rf/feature_ranker.py` and `glass_router_pipeline.py`, the fold-nested ranker docstrings.** "Never sees the rows it routes" should read "is never trained on the rows it routes", with the same feature-engineering note.

None of these changes behaviour, and the existing tests reference `NESTED_SOURCE` by import, so they hold.

**Status: applied in Step 3D.** All four items are in the Step 3D change, docstrings and comments only. The one exception is the `NESTED_SOURCE` string itself, which future `nested_t1` artifacts record. It now reads: `"OOF pass1 probabilities >= fold-nested t1 (router-level: no row's own label in its cut; features engineered on the outer-training partition)"`. The same qualification was added to the `router/two_pass.py` module docstring and to its verbose fit message. The test docstrings that describe these pathways now say they establish router-level isolation only. No historical artifact records the old string: `nested_t1` has never been run on real data.
