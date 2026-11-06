# PR 37 — Commit 3: Methodological Records and Handoff to Commit 4 and PR 38

| | |
|---|---|
| Author | Phillip Harris |
| PR / commit | PR 37, Commit 3 (Steps 3A–3D) |
| Branch | `pr-37-router-baseline-contract` |
| Base | `e5c437c` (Commit 2: Stage 2 router diagnostics and the frozen pre-correction baseline) |
| Resolves | F17 (Pass 2 training population) and F31 (conditional RF confidence-blending review) |
| Date | 2026-11-04 |
| Status | Historical record of Commit 3. Its open Commit 4 decisions (D1–D6, P1, P2) were resolved in Commit 4 (`f7a987e`); PR 38 executes from [`pr38_handoff.md`](pr38_handoff.md) (Commit 5), which lists the resolutions |

## 1. Records

| Record | Step | Status | Contents |
| --- | --- | --- | --- |
| [`pr37_f17_investigation.md`](pr37_f17_investigation.md) | 3A | Reviewed; supporting evidence | Source trace of both routers' Pass 2 populations, risks, the A/B/C policy comparison, validation plan (D1–D2, V1–V9) |
| [`pr37_f17_decision_record.md`](pr37_f17_decision_record.md) | 3B | Approved 2026-11-04 | F17 decision, nested safeguards, cross-arm principle, sensitivity analysis S1, limitations L1–L8, PR 38 gates G1–G13 |
| [`pr37_f31_disposition_record.md`](pr37_f31_disposition_record.md) | 3C | Approved 2026-11-04 | F31 Disposition A, effects of `nested_t1`, band validity, Stage 4 compatibility, reopening conditions, PR 38 requirements R1–R8 |
| This handoff | 3D | — | What Commit 3 contains, compatibility with Commit 4, open decisions, PR 38 obligations, regression coverage |

The two decision records govern where any record differs. None of them reports a corrected-router result: no corrected rerun has been conducted.

## 2. What Commit 3 contains

**Behaviour, all opt-in.** Every historical default is unchanged. Nothing is activated in the notebooks.

| Change | Historical default | Opt-in |
| --- | --- | --- |
| RF Pass 2 eligibility (`blackbox_pipeline/models/rf_router/oof/eligibility.py`) | `eligibility="global_t1"` | `"nested_t1"`: a per-inner-fold `t1_j` solved without the fold; the deployed `t1` unchanged |
| GLASS inner-fold feature ranker (`glass_pipeline/glass_router/rf/feature_ranker.py`) | `inner_feature_ranker="outer_rf"` | `"fold_nested_rf"`: a fresh ranker per inner fold, trained on inner-train rows only |
| Independent operating contract (`shared/stage2/contract.py`, `blackbox_pipeline/models/rf_router/contract.py`) | `match_glass_oof` remains the historical pathway | `SharedOperatingContract` plus `rf_constraints_from_contract`; the independent pathway refuses `match_glass_oof`. **No contract values** |
| Realised-population provenance | — | RF `oof_fold_summary` remainder rows and positives, protocol `pass2_eligibility`; GLASS fold summaries, in-memory `fold_fit_reports`, protocol `glass_inner_feature_ranker`, inner-ranker provenance |

**Step 3D, documentation and wording only.**

- The three records above and this handoff, under `documentation/methodology/`.
- The research roadmap:
    - the F17 and F31 items ticked;
    - the six methodological-safeguard tasks added under PR 37, five complete and one left open for Commit 4;
    - the F17/F31 validation entry added under PR 38.
- The approved wording corrections in code docstrings and comments:
    - F17 items 1–4: router-level isolation, conditional on the engineered features (L1);
    - F31 W1–W4.
- The `NESTED_SOURCE` provenance string qualified the same way. Only future `nested_t1` artifacts record it.
- Two statements in the Step 3A record corrected and marked: the section 6 status line, and the timing of the `match_glass_oof` removal.
- The F17 G13 acceptance condition clarified at review: primary results are not accepted while a confirmed material own-label dependency is unresolved.
- The F17 G6/G12 ordering clarified at review. S1 begins after the applicable primary-run gates other than G6 have passed. G6 is verified during S1, and S1 cannot be accepted unless it passes.
- Two regression tests (section 8).

No behaviour changed. The code edits are docstrings, comments, the `NESTED_SOURCE` constant and one verbose print string.

## 3. Compatibility with Commit 4

Commit 4 freezes the shared Stage 2 operating contract. The requirements below come from the PR 37 roadmap ("Stage 2 operating contract"), the Step 3A record (section 8) and the F17 record (section 9). Commit 3 supplies the mechanism for each, never values.

| # | Commit 4 requirement | Source | What Commit 3 provides | Compatible? |
| --- | --- | --- | --- | --- |
| 1 | Replace `match_glass_oof` with pre-registered constraints shared by both arms | Roadmap | `SharedOperatingContract`; `validate_constraint_source` refuses `match_glass_oof` and relabelled GLASS-derived values under the independent pathway; `rf_constraints_from_contract` | Yes |
| 2 | Fix the NPV, leakage and precision floors before rerunning either router | Roadmap | Every contract field must be given explicitly; out-of-range values are refused; Commit 3 ships none | Yes; the values are Commit 4's |
| 3 | Each router maximises coverage subject to the same constraints | Roadmap | Objectives unchanged: the RF solver takes the largest feasible Pass 1 band and the smallest feasible `t2`; the GLASS ILP maximises union coverage | Yes for RF. The GLASS mapping is open decision D3 |
| 4 | Document the constraints and rationale before viewing final rerun results | Roadmap | `constraint_provenance` and `check_contract_provenance` record the contract identity, fingerprint and provenance text | Yes; the provenance content is open item P2 |
| 5 | Freeze the contract for subsequent selector experiments | Roadmap | A deterministic fingerprint covering every value | Yes |
| 6 | Fold-nested, training-only selection | Step 3A section 8; F17 sections 3 and 9 | `nested_t1` and `fold_nested_rf` remove the two inner-level selection dependencies at router level | Yes, at router level. GLASS's in-sample rule precision (L2) and the feature engineering (L1, gate G13) remain |
| 7 | A no-feasible-candidate rule covering inner-fold infeasibility and refused remainders | Step 3A section 5; F17 section 9 | No silent fallback: an infeasible `t1_j` raises, an infeasible inner GLASS selection propagates, a remainder under the floor or single-class raises | Yes; the rule's wording is open decision D4 |
| 8 | Leakage and recall on global denominators | Step 3A section 8 | The contract fixes `recall_denominator="global"`; RF already uses it; GLASS `oof_remainder` rescales to global denominators | Yes |
| 9 | One Pass 2 precision floor with one meaning in both arms | F17 section 2, point 3 | Under the F17 policy both arms fit Pass 2 on their own OOF remainder, so the floor refers to P(y = 1 \| flagged, Pass 1 remainder) in both | Yes for the population. The estimation procedures still differ (L2) |

### Open Commit 4 decisions

Commit 3 decides none of these.

| ID | Decision | Notes |
| --- | --- | --- |
| D1 | The numerical values: Pass 1 NPV floor, Pass 1 leakage-rate cap, Pass 2 precision floor | The bands sit at or just above these floors, so they largely set RF confidence values (F31 section 4) |
| D2 | Whether to keep a Pass 1 leakage count cap | `max_pass1_leakage_count` must be passed explicitly. `None` means no count cap, mapped to `NO_COUNT_CAP` for RF. The choice moves `t1` |
| D3 | How set-level requirements map onto GLASS's per-rule gates, its Pass 1 union cap and its in-sample selection | Includes whether GLASS's fit-time enforcement must be checked on inner-OOF estimates (F17 L2; F31 K2) |
| D4 | The no-feasible-candidate rule | Must cover inner-fold infeasibility and refused remainders; no fallback to the historical pathways (F17 G8) |
| D5 | The RF band-support floors | Arm-internal estimation safeguards, passed to `rf_constraints_from_contract` as explicit arguments; they feed F31 R1 |
| D6 | Whether RF hyperparameters should still be read from the GLASS RF artifact | Sharing hyperparameters is legitimate; the read is a cross-arm file lookup |

Two items must also be defined in Commit 4:

- **P1, the common coverage definition**, used to compare the arms.
- **P2, the contract record's own provenance.** It must state that historical OOF results, and the test split in PRs 33–36, had been examined. The constants are not prospectively uninformed.

Commit 4 must also state how an outer-OOF contract shortfall is evaluated. That rule decides F31 reopening condition 4.

## 4. PR 38 obligations

PR 38 may treat F17 as active only when every mandatory gate passes. A failure stops that work and is reported; it is never worked around. Full wording is in the records.

**Order:**

1. The single-fold preflights (G3, G13) and the implementation gates come before the full five-fold primary reruns.
2. The remaining gates are verified on those reruns.
3. G12, sensitivity analysis S1, begins once the applicable primary-run gates other than G6 have passed. G6 is verified during S1. It stays mandatory, and S1 cannot be accepted unless G6 passes.

**F17 mandatory gates** (decision record, section 8):

| Gate | Short form |
| --- | --- |
| G1 | Primary configuration: GLASS `oof_remainder` with `fold_nested_rf`; RF `nested_t1`; nothing from the `"recommended"` profile |
| G2 | Wiring: the GLASS full-train fit passes the ranker factory; one RF config factory for full-train and cross-fit; Stage 4 accepts both sides |
| G3 | Single-fold real-data feasibility preflight (V9), both arms |
| G4 | Isolation provenance on real data, every outer fold |
| G5 | Persistence of realised populations and selector outcomes in saved artifacts |
| G6 | GLASS Pass 1 invariance across the S1 pair, verified during S1; S1 cannot be accepted unless it passes |
| G7 | Reproducibility of at least one outer fold |
| G8 | Explicit rejection; never a fallback to `global_t1`, `outer_rf` or `full` |
| G9 | Both arms consume the frozen Commit 4 contract |
| G10 | New versioned artifacts; deliberate updates to the GLASS `full` pins, including the baseline-pinned `tests/support/artifact_audit.py` |
| G11 | `pr37_stage2_pre_correction_v1` untouched; `--verify` passes throughout |
| G12 | S1 begins after the applicable primary-run gates other than G6 (G1–G5, G7–G11, G13) hold for the primary; G6 is verified during S1 |
| G13 | Feature-engineering isolation audit (L1), a preflight beside G3. If a material own-label dependency affecting inner-OOF eligibility is confirmed, its correction must be resolved and validated before the primary corrected-router results can be accepted |

**F31 requirements** (disposition record, section 8): R1 band record; R2 boundary-strip diagnostic; R3 optimism gap; R4 Stage 4 checks; R5 eligibility provenance at Stage 4; R6 level spread (diagnostic); R7 relabel the blackbox Cell 21 diagnostic; R8 coupling to G13. R1–R5, R7 and R8 are mandatory. R2 and R3 are always recorded and reported, whatever they show.

**F31 reopening conditions** (disposition record, section 7). Any one reopens F31:

1. Strip A makes up 5% or more of any fold router's flagged Pass 2 band. This is a mandatory review trigger, not an acceptance criterion: staying below 5% does not by itself establish that the confidence is valid.
2. The feature correction adopted under G13 makes the deployed forests use different features from the OOF scores the bands were estimated on.
3. Stage 4 starts consuming raw scores or binned bands, or the probability transform changes.
4. Commit 4's evaluation rules require band values to be re-estimated after an outer-OOF contract shortfall.

**Sensitivity analysis S1:** GLASS `full` versus GLASS `oof_remainder`, everything else identical. It is descriptive, and never used to choose the primary.

## 5. Evidentiary status

- **No corrected-router reruns** have been conducted, and no PR 38 results exist.
- **The decisions are design decisions,** made with historical evidence seen:
    - the Commit 2 OOF diagnostics;
    - the test-split exposure in PRs 33–36. That split cannot be described as untouched;
    - a partial synthetic probe (3 of 5 GLASS `oof_remainder` outer folds observed).
- **Both opt-in pathways are validated on synthetic data only.** GLASS `oof_remainder` has never run on real data.
- **Isolation is router-level only,** conditional on target-encoded features fitted on the outer-training partition (L1).
    - Complete inner-fold isolation is not claimed until G13 passes.
    - If G13 confirms a material own-label dependency, the primary corrected-router results are not accepted until its correction is resolved and validated.

## 6. Preservation

- `pr37_stage2_pre_correction_v1` is not edited or regenerated. It remains the permanent pre-correction reference; a material change requires a new version.
- Commit 3, including Step 3D, modifies none of the package's 19 files and none of the 15 code files whose identities it pins.
- The 37 historical model artifacts are unchanged. PR 38 writes new, versioned artifacts.
- Historical defaults are unchanged: GLASS `pass2_population="full"` in the notebook profile, `inner_feature_ranker="outer_rf"`, RF `eligibility="global_t1"` and the `match_glass_oof` pathway.

## 7. Wording corrections applied in Step 3D

| Item | File | Change |
| --- | --- | --- |
| F17-1 | `blackbox_pipeline/models/rf_router/oof/eligibility.py` (module docstring) | "Nothing reads `y_i`" → no router component reads it. Adds a "Scope of the guarantee" section on outer-fold feature engineering (L1, G13) |
| F17-2 | `blackbox_pipeline/models/rf_router/oof/eligibility.py` (`NESTED_SOURCE`) | Router-level wording; notes the outer-training feature engineering |
| F17-3 | `blackbox_pipeline/models/rf_router/config.py` (eligibility comment) | The same qualification |
| F17-4 | `glass_pipeline/glass_router/rf/feature_ranker.py`, `glass_pipeline/glass_router/pipeline/glass_router_pipeline.py` | "Never sees the rows it routes" → "is never trained on the rows it routes", with the same note |
| F17 (consistency) | `blackbox_pipeline/models/rf_router/router/two_pass.py` (module docstring, verbose message); the two pathway test modules' docstrings | The same router-level qualification |
| W1 | `blackbox_pipeline/models/rf_router/router/base.py` (`_make_bands`) | "Stricter" → the smaller per-pass floor; a redundant check after the solver |
| W2 | `glass_pipeline/meta_ebm/router_evidence.py` (module docstring) | A rule's in-sample rate is measured on the rows the rule was selected on, under `full` and under `oof_remainder` |
| W3 | `blackbox_pipeline/models/rf_router/router/two_pass.py` (`predict_train_oof`) | A diagnostic; under `nested_t1` it does not reproduce the fitting remainder (strip A → Pass 1, strip B abstains) |
| W4 | `blackbox_pipeline/models/rf_router/bands.py` (module notes) | Selection-optimism risk; under `nested_t1` the Pass 2 band is measured on the fold-nested remainder and the Pass 1 band is unchanged |

None of these files is baseline-pinned.

## 8. Regression coverage audit

Each Commit 3 requirement maps to the tests that protect it. Test paths are relative to `prototype/bank_pipeline/`.

| Requirement | Protected by |
| --- | --- |
| RF default is the historical global cut; report unchanged; old configs load | `tests/stage2/test_rf_router_nested_eligibility.py`: `test_the_default_is_the_historical_global_cut_with_an_unchanged_report`, `test_an_explicit_global_rule_reproduces_the_default_exactly`, `test_old_configs_without_the_field_load_as_the_historical_rule` |
| The nested cut is computed as specified, with provenance | `test_the_nested_cut_matches_an_independent_recomputation` |
| No eligibility forest sees the fold it cuts | `test_no_eligibility_forest_is_fitted_on_the_fold_it_cuts` |
| The deployed `t1` and Pass 1 OOF scores are identical under both rules | `test_the_deployed_t1_is_the_same_under_both_rules` |
| A fold's own labels cannot move its eligibility (with a control) | `test_a_folds_own_labels_cannot_move_its_eligibility_under_the_nested_rule`, `test_control_the_global_rule_lets_those_labels_move_the_same_rows` |
| Determinism | `test_the_nested_rule_is_deterministic` |
| No fallback; contract refusals | `test_an_infeasible_fold_cut_raises_instead_of_falling_back`, `test_inputs_that_break_the_fold_contract_are_refused`, `test_resolve_at_is_refused_under_the_nested_rule`, `test_invalid_eligibility_settings_are_refused` |
| Outer cross-fit provenance; mixed rules refused; outer-fold label isolation (V5) | `test_the_crossfit_records_the_rule_and_each_folds_realised_remainder`, `test_fold_routers_with_different_rules_are_refused`, `test_outer_fold_k_ignores_its_own_labels_under_the_nested_rule` |
| **F31 band accounting (Step 3D):** Pass 1 band unchanged; Pass 2 band measured on the fold-nested remainder; the strips exist under `nested_t1` and not under `global_t1` | **New:** `test_the_bands_are_measured_on_the_populations_the_f31_record_names` |
| **F31 C3 / W3 (Step 3D):** `predict_train_oof` departs from the remainder only on the strips | **New:** `test_predict_train_oof_departs_from_the_remainder_only_on_the_boundary_strips` |
| GLASS inner rankers trained on inner-train rows only; search order; provenance | `tests/stage2/test_glass_inner_feature_ranker.py`: `test_each_inner_ranker_is_fitted_on_its_folds_training_rows_only`, `test_each_inner_search_is_ordered_by_its_own_ranker_and_the_rest_by_rf_model`, `test_the_nested_ranker_provenance_is_recorded` |
| GLASS default unchanged; `full` fits unaffected; validation | `test_the_default_orders_inner_searches_by_rf_model_and_calls_no_factory`, `test_a_full_fit_report_is_unchanged_whatever_the_ranker_setting`, `test_the_config_default_and_validation` |
| GLASS no fallback; ranker independence | `test_a_missing_factory_is_refused_before_any_fit`, `test_a_reused_or_invalid_ranker_is_refused`, `test_an_infeasible_inner_selection_propagates_and_never_becomes_a_full_fit` |
| Ranker recipe and cross-fit wiring | `test_the_factory_fits_the_fold_rf_recipe_on_exactly_the_rows_given`, `test_the_crossfit_supplies_a_fold_nested_factory_only_on_the_opt_in_path`, `test_the_nested_ranker_requires_the_fold_rf`, `test_glass_pipeline_accepts_the_factory_by_keyword_only_as_an_addition` |
| Identical inner folds across arms (V1) | `test_glass_inner_folds_are_the_rf_routers_inner_folds_for_the_same_rows`, `test_the_rf_router_scores_pass1_on_that_same_plan` |
| Contract: explicit, valid, value-free, independent, fingerprinted | `tests/stage2/test_stage2_operating_contract_support.py`: all 12 tests (requirement completeness, invalid values, no shipped values, refusing `match_glass_oof` and relabelled GLASS-derived constraints, historical pathway intact, provenance, RF adapter, count cap, extra floors, fingerprint, import isolation) |
| No fitted estimator crosses arms | `tests/parity/test_stage2_cross_arm_independence.py`: all 4 tests |
| Historical GLASS notebook profile is `full` | `tests/parity/test_router_operating_contract.py`: `test_the_glass_notebook_runs_the_full_population_profile_for_both_fits` (pre-existing) |
| GLASS `oof_remainder` construction | `tests/stage2/test_glass_router_pass2_remainder.py` (pre-existing, stubbed rule search and ILP) |
| Frozen baseline and artifacts unchanged | `research_logs/baselines/stage2_baseline.py --verify` and `--check`; the 37-hash snapshot check (scripts, not pytest) |

**Not protected by a Commit 3 test, by design:**

- **V3, deployed GLASS Pass 1 rules identical across `pass2_population` settings.** This holds by construction. Pass 1 deduplication and the ILP both separate the classes before selecting. A meaningful synthetic test needs the real generator and the CBC ILP, which makes it a `slow` test, so it is left to PR 38 with gate G6 on real data. The Step 3A record previously listed it as implemented; that is corrected.
- **V2, a complete GLASS `oof_remainder` cross-fit with the real ILP.** PR 38 (`slow`), then gate G3 on real data.
- **The wording corrections themselves.** `NESTED_SOURCE` is referenced by import in the tests, so a wording change cannot break them. No test pins the prose.
- **The PR 38 primary configuration (G1)** is checked by gate G1 when it is activated. The field names and allowed values it uses are covered by the validation tests above.

The two Step 3D tests reuse the module's existing fitted routers, so they add no fits. They protect the facts the F31 disposition rests on (sections 3 and 5 of that record: C2, C3, W3, W4). Before Step 3D, nothing tested those facts directly.
