# PR 37 — Commit 4: Stage 2 Operating Contract Record

| | |
|---|---|
| Author | Phillip Harris |
| Record | Pre-registered Stage 2 operating contract |
| PR / step | PR 37, Commit 4. This version records Steps 4A, 4B, 4C and 4D |
| Branch | `pr-37-router-baseline-contract` |
| Base | `ab7b063` (Commit 3) |
| Contract record | Frozen: [`shared/stage2/contracts/stage2_primary_v1.json`](../../shared/stage2/contracts/stage2_primary_v1.json), `pr37-stage2-primary` `1.0.0`, fingerprint `0817825b9eecbd11`. Working profile kept for history: [`stage2_primary_working.json`](../../shared/stage2/contracts/stage2_primary_working.json), `0.1.0-working`, fingerprint `a6ca7462d4773005` |
| Status | **Frozen in Step 4D** (section 24); formally approved by Phillip Harris on 2026-11-05. The freeze takes effect when Commit 4 is created after local validation, including the real-artifact re-verification |
| Related | [`pr38_evaluation_protocol.md`](pr38_evaluation_protocol.md) (Step 4C); [`pr37_commit3_handoff.md`](pr37_commit3_handoff.md); [`pr37_f17_decision_record.md`](pr37_f17_decision_record.md); [`pr37_f31_disposition_record.md`](pr37_f31_disposition_record.md) |
| Date | 2026-11-05 |

No corrected-router run has been conducted. Nothing here is activated in a notebook. Step 4B adds opt-in GLASS enforcement and both arm adapters; no default changes. Step 4C pre-registers the PR 38 evaluation protocol (section 23); it computes no corrected result. Step 4D freezes the contract, the protocol and both arms' configurations without changing any value (section 24).

## 1. What Steps 4A, 4B and 4C deliver

| Deliverable | Where |
|---|---|
| Schema 2 of the shared contract: versions, status, objective, metric set, support floors, structured provenance | `shared/stage2/contract.py` (`SharedOperatingContractV2`, a subclass of the Commit 3 `SharedOperatingContract`) |
| A loader that reads one JSON record and nothing else | `load_operating_contract`, `working_profile_path` |
| The working numerical profile | `shared/stage2/contracts/stage2_primary_working.json` |
| An arm-neutral, exact-count obligation checker | `shared/stage2/obligations.py` |
| The RF adapter taking every value, support floors included, from a schema-2 contract | `blackbox_pipeline/models/rf_router/contract.py` (`rf_constraints_from_contract_v2`) |
| Regression tests | `tests/stage2/test_stage2_operating_contract_v2.py`, `tests/stage2/test_stage2_obligations.py` |

Step 4B (sections 16–22) adds:

| Deliverable | Where |
|---|---|
| Opt-in GLASS union NPV, union precision and union support obligations with exact atom indicators; the opt-in Pass 2 row objective; greedy and validator updates | `glass_pipeline/glass_router/ilp_rule_selector/` (`ilp_rule_selector.py`, `ilp_builder.py`, `greedy_selector.py`, `utils.py`); `core/config.py` |
| CBC objective, bound and gap from the solver log, under obligations | `ilp_builder.parse_cbc_log`, `ILPBuilder.record_solver_log` |
| The GLASS adapter, its check, the fit-outcome mapping and the reproduction record | `glass_pipeline/glass_router/contract.py` |
| Per-pass selection reports, inner folds included | `GlassRouterPipeline.selection_reports()` |
| The RF configuration adapter and its check | `blackbox_pipeline/models/rf_router/contract.py` (`rf_config_from_contract`, `check_rf_config_follows_contract`) |
| One behaviour-preserving performance change | `rule_evaluator/overlap_analyzer.py` (section 22) |
| Regression tests | `tests/stage2/test_glass_union_obligations.py`, `tests/stage2/test_stage2_contract_adapters.py` |

Step 4D (section 24) adds:

| Deliverable | Where |
|---|---|
| The frozen contract and its pinned loader | `shared/stage2/contracts/stage2_primary_v1.json`; `load_frozen_contract`, `frozen_profile_path`, `FROZEN_PROFILE_FINGERPRINT` |
| The frozen PR 38 protocol, its freeze record (D-C1 to D-C11) and bindings | `shared/stage2/protocols/pr38_evaluation_v1.json`; `load_frozen_protocol`, `load_frozen_specification`, `is_pinned_frozen`, `check_arm_profile_binding`, `glass_reports_required` |
| Versioned arm configuration records | `shared/stage2/arm_profile.py` (`ArmProfile`, `load_arm_profile`) |
| The frozen GLASS profile | `glass_pipeline/glass_router/profiles/glass_router_primary_v1.json`; `glass_router.profile` (`glass_primary_config()`, `ranker_settings()`, `check_glass_primary_config`) |
| The frozen, independent RF profile | `blackbox_pipeline/profiles/rf_router_primary_v1.json`; `rf_router.profile` (`rf_primary_config()`, `check_rf_primary_config`) |
| GLASS selection-report evidence required for acceptance (D-C11) | `check_selection_completeness` |
| Regression tests | `tests/stage2/test_stage2_freeze.py`; `tests/realdata/test_stage2_freeze_artifacts.py` (opt-in, the freeze re-verification) |

Step 4C (section 23) adds:

| Deliverable | Where |
|---|---|
| The pre-registered PR 38 evaluation protocol, working | [`pr38_evaluation_protocol.md`](pr38_evaluation_protocol.md); record `shared/stage2/protocols/pr38_evaluation_working.json` |
| The protocol's executable invariants: record loader and freeze guard, contract binding, population identity, row-level upstream identity (artifact SHA-256, recomputed decisions, EBM fold thresholds, row-aligned flag fingerprints), paired routings and reference flags bound to row identity, pooled and per-fold metrics, the stratified paired bootstrap and its registered, digest-checked runs, selection-record completeness, fit status and fold ledger, shortfall classification, the F31 R2 trigger, structural acceptance, evidence tiers, a defense-in-depth input guard, ablation adoption gates | `shared/stage2/evaluation_protocol.py` |
| Regression tests | `tests/stage2/test_stage2_evaluation_protocol.py`; `tests/realdata/test_evaluation_protocol_artifacts.py` (opt-in) |

The Commit 3 class, adapter signature, tests and every historical default are unchanged. Values still live only in records: no contract instance exists in code.

## 2. Approved Step 4A decisions

These are recorded as approved on 2026-11-04 and are not reopened here.

| ID | Decision |
|---|---|
| M1 | Primary objective: **lexicographic coverage**. Maximise feasible Pass 1 routed rows; then, on that router's own Pass 1 remainder, maximise feasible Pass 2 routed rows. Not a joint total-coverage optimum |
| M2 | Observed constraint compliance is kept separate from uncertainty and review classification. An observed violation is a violation even when statistically small. Pooled and per-fold results, signed shortfalls, support and descriptive uncertainty are reported. The 2-standard-error materiality threshold is a Step 4C recommendation, not a frozen rule |
| M3 | Stop on proven infeasibility, on no feasible solution, and on an insufficient remainder. A feasible but unproven ILP incumbent may be accepted only if every constraint is verified independently and solver status, gap and a deterministic reproduction are documented. A greedy fallback is never automatically equivalent to an optimised result; it needs an explicit review of its objective gap and comparability |
| M4 | No automatic relaxation and no pre-authorised numerical revision. If the fit-only PR 38 preflight fails, stop and investigate. Any revision needs explicit approval, a new version, preserved provenance and disclosure before corrected outer-OOF results are inspected |
| M5 | Common routed-set support floors, in principle. 125 (Pass 1) and 75 (Pass 2) are provisional; their justification remains under review. Arm-specific support diagnostics may remain |
| M6 | Available GLASS inner-OOF Pass 1 union NPV and leakage are reported as diagnostics, never used to select the contract, with no extra fitting |
| M7 | Working candidate: NPV ≥ 0.95, leakage ≤ 0.15, precision ≥ 0.30, no count cap, support 125 / 75 |
| M8 | Independent, version-controlled ownership of the existing RF hyperparameters, values preserved exactly with their original tuning provenance; the tuning and feature-engineering dependence documented; no nested tuning in PR 37 |

## 3. The shared objective

Both arms receive the same obligations and each maximises its own feasible coverage, lexicographically:

1. **Pass 1:** the most rows routed NOT_SUBSCRIBE that meet every Pass 1 obligation on the fit's Pass 1 population.
2. **Pass 2:** on the remainder that router's own Pass 1 leaves, the most rows flagged SUBSCRIBE that meet every Pass 2 obligation.

Neither arm maximises total coverage jointly, and Pass 1 is never traded for Pass 2. Joint optimisation would need a Pass 2 refit for every Pass 1 candidate. It is undefined in the GLASS architecture, whose remainder comes from inner-fold Pass 1 selections, and it would let Pass 2 labels steer Pass 1. F17's order (Pass 1, then the OOF remainder, then Pass 2) is preserved. Total coverage is a reported outcome.

| Arm | Pass 1 | Pass 2 |
|---|---|---|
| RF | Largest feasible band (`solve_pass1`). Already the objective | Largest feasible band on its remainder (`solve_pass2`). Already the objective |
| GLASS | ILP Phase 1 maximises union rows. Already the objective | ILP Phase 1 maximises union *positives* by default. Step 4B adds the opt-in `pass2_union_objective="rows"`, valid only together with the union precision floor; the GLASS adapter sets it (section 17) |

Architectures, selected rules, thresholds and fitted estimators may differ. Sharing the obligations does not make the arms mathematically equivalent.

## 4. Metric definitions

Identifier `pr37.stage2.contract_metrics/1`. The names, numerators and denominators are the diagnostics registry's (`shared/stage2/diagnostics/metrics.py`, baseline-pinned: imported, never edited).

| Metric | Formula | Registry name | Role |
|---|---|---|---|
| Pass 1 NPV | TN_P1 / N_P1 | `pass1_npv` | Floor |
| Pass 1 positive leakage | FN_P1 / P_total | `pass1_leakage_share` | **Ceiling** |
| Pass 2 precision | TP_P2 / N_P2 | `pass2_precision` | Floor |
| Pass 1 coverage | N_P1 / N_evaluated | `pass1_rate` | Reported |
| Pass 2 coverage | N_P2 / N_evaluated | `pass2_rate` | Reported |
| Total coverage | (N_P1 + N_P2) / N_evaluated | `coverage` | Reported |
| Abstention | N_abstain / N_evaluated | `abstain_rate` | Reported |
| Pass 2 conditional rate | N_P2 / (N_evaluated − N_P1) | `pass2_remainder_rate` | Reported, not constrained |

**Populations.** At fit time each obligation is measured on the population its pass was selected on: Pass 1 on the fit's training population (P_total = its positives); Pass 2 on the remainder that router's own Pass 1 left. This holds at every nesting level: full-train fit, outer-fold fit, nested eligibility solve, inner-fold selection. At evaluation the population is one outer held-out fold, or the folds pooled. The test split is not used for any contract purpose.

**Exact arithmetic.** A limit is the decimal the record writes, read exactly (0.15 is 3/20, not the nearest binary float). Each comparison is between integers: TN·d ≥ n·N_P1 for an NPV floor n/d, and so on. The RF solver's float comparisons and GLASS's union-cap formula agree with the checker on every tested population (section 12).

**Zero support.**

- *Fit time.* An empty routed set and a set below its support floor are infeasible. An undefined ratio is never satisfied. A population without both classes is rejected.
- *Evaluation.* An undefined ratio is reported as `undefined` (value `None`), never 0 and never compliant.

**Pooling.** Numerators and denominators are summed across folds before any ratio. Fold ratios are never averaged.

## 5. Working numerical profile

| Setting | Value | Basis | Rationale |
|---|---|---|---|
| Pass 1 NPV floor | 0.95 | Protective guard | Caps contamination among Pass 1 rows at 5% (about 0.44× the 11.27% base rate). Not the primary lever: see the identity below |
| Pass 1 leakage ceiling | 0.15 | Substantive requirement | At most 15% of all actual positives routed NOT_SUBSCRIBE. Matches the OOF target the GLASS notebook already declared, which was itself set with OOF results visible |
| Leakage count cap | None | Structural argument | Fits run on 32,950, 26,360 and about 21,088 rows, so a fixed count is a different rate in each. Historically it equalled the rate cap at full train and could not bind in a fold router |
| Pass 2 precision floor | 0.30 | Substantive requirement | A meaningful lift over the 15–16% historical remainder prevalence (about 1.9×) and about 2.7× the base rate |
| Pass 1 support floor | 125 | Historical configuration anchor (provisional) | The RF band floor the notebook used, with no recorded rationale. Prevents tiny tail sets |
| Pass 2 support floor | 75 | Historical configuration anchor (provisional) | Same origin and purpose |

**The values are not independently established optima.** They were chosen after the historical OOF results and the test split had been examined.

**Identities that hold for any fit** (b = base rate, λ = realised leakage, c₁ = Pass 1 coverage, F = NPV floor, L = leakage ceiling, π = precision floor):

- NPV₁ = 1 − λb/c₁. For c₁ ≥ L·b/(1 − F) = 0.338 the leakage ceiling implies the NPV floor. Below that, the NPV floor is the tighter constraint, which is why it is kept.
- Remainder prevalence ρ = b(1 − λ)/(1 − c₁), with *realised* λ, not the ceiling.
- Pass 2 coverage ≤ (1 − c₁)·min(1, ρ/π).
- Pass 2 is degenerate (flags its whole remainder) only if π ≤ ρ. With π = 0.30 that requires Pass 1 to route at least 1 − b/π = 62% of rows.

## 6. Support requirements

Support floors are **fit-time** obligations on the rows a pass routes. At evaluation the routed counts are reported, not judged: an outer held-out fold is about a quarter of its fit population.

The floors prevent degenerate tail sets. They do **not** guarantee narrow estimates: at the floors the 95% Wilson half-widths are about ±4.0 percentage points (125 rows at NPV 0.95) and ±10.2 (75 rows at precision 0.30). Their justification remains under review (M5).

The F17 remainder floor (500 rows, both classes) is an arm setting fixed by F17, not a contract field.

## 7. Fit time versus outer OOF

| Layer | What it is | Implemented as |
|---|---|---|
| Fit-time compliance | Binding. Every fit's selected sets satisfy every obligation on their own selection populations, or the fit stops | `check_pass1_fit`, `check_pass2_fit`, `require_fit` |
| Outer-OOF observation | Observed, not certified. Every obligation reported per fold and pooled as satisfied, violated or undefined, with its exact signed shortfall and the counts behind it | `observe_routing`, `observe_folds` |

Both arms select at the edge of their feasible sets, so observed outer-OOF violations are expected about half the time per check even without optimism. They remain violations (M2). Their classification, review triggers and artifact acceptance are pre-registered in Step 4C (protocol sections 10, 11 and 14).

The F31 boundary-strip review is unchanged: strip A at 5% or more of any fold's flagged Pass 2 band reopens F31 (R2). Nothing in Step 4A re-estimates bands after a shortfall; Step 4C must keep it that way for F31 reopening condition 4 to stay closed.

## 8. Fit outcomes

| Outcome | Disposition | Meaning |
|---|---|---|
| `feasible_optimal` | Primary | Verified feasible; the search is certified optimal (RF exhaustive scan, proven ILP) |
| `feasible_unproven_incumbent` | Primary only if documented | Verified feasible; optimality unproven. Needs solver status, gap and a deterministic reproduction (M3) |
| `feasible_greedy_fallback` | Requires review | Verified feasible heuristic; objective gap and comparability must be reviewed (M3) |
| `proven_infeasible` | Stop | No candidate meets the obligations |
| `no_feasible_solution_found` | Stop | The search ended without a verified feasible set; infeasibility not proven |
| `insufficient_population_or_support` | Stop | Population without both classes, remainder under its floor, or a pass below its support floor |

`classify_fit_outcome` refuses a feasible claim the obligation check contradicts, and a stop claim alongside a report in which every obligation holds. No outcome relaxes a constraint. The GLASS mapping and its solver gap are in section 20.

## 9. Evidence and exposure

**Seen before the values were chosen:**

- the frozen pre-correction OOF baseline (both routers);
- the test split, evaluated in PRs 33–36, including the PR 35 test-split constraint results;
- the Commit 3 synthetic probes.

During the review the full RF sidecar was shared. It also stores historical test metrics, which were excluded from every analysis.

**Feasibility screening** (read-only; RF only; full-train router). Sources, with SHA-256 recorded in the contract record:

- `rf_router_two_pass_metadata.json`
- `rf_router_two_pass_pass1_sweep.csv`
- `rf_router_two_pass_pass2_sweep.csv`

Their identity was verified against the sidecar's fit report. Their membership in the snapshot `SHA256SUMS.txt` is not yet verified locally.

| | Result |
|---|---|
| Pass 1 | Feasible. Largest band 13,496 / 32,950 rows (coverage 0.4096); 555 of 3,712 positives leaked (0.1495) against an integer cap of 556. Leakage binds; NPV 0.9589 is slack; support far above 125 |
| Remainder left by these values | 19,454 rows, prevalence 0.1623, below the 0.30 floor |
| Pass 2, historical remainder only | Feasible. Largest band 8,122 / 20,263 rows; precision 0.3015 binds; support far above 75 |
| Total coverage, abstention | Not inferred. The Pass 2 sweep belongs to a different remainder; only the bound Pass 2 coverage ≤ 0.3194 holds |

The screen does not establish feasibility for the PR 38 outer-fold `nested_t1` routers, whose frontiers were never saved, nor for GLASS under union-level obligations. Both are first observed at the fit-only PR 38 preflight.

## 10. Provenance design

The contract's provenance cannot depend dynamically on any arm's results:

1. **Static source.** Values come only from a version-controlled JSON record. `load_operating_contract(path)` reads that one file. Its only other arguments are an expected fingerprint and a required status. `shared/stage2/contract.py` imports no arm, no artifact reader and no diagnostics.
2. **Structured, validated provenance.** Each setting needs a rationale whose basis is one of: substantive requirement, protective guard, structural argument, historical configuration anchor. An arm's realised performance is not an allowed basis. In addition:
   - Evidence names its kind, its split (no test-split value exists) and its role. A field name with a `test`, `holdout` or `oracle` segment is refused.
   - Exposure must be declared.
   - No setting's rationale may cite the historical GLASS-derived pathway (`match_glass_oof`). The record may, and does, describe that pathway truthfully as history in its evidence, exposure and limitations. The Commit 3 rule that the one-paragraph summary may not name it is inherited unchanged.
3. **Performance dependence: two separate statements.**
   - `derived_from_arm_realised_performance: false` (required) states that no value was computed from, copied from or mechanically set to match an arm's realised performance, which is what the historical pathway did.
   - `historical_exposure_informed_decisions` discloses the different fact that examined historical results informed the methodological choices. It must be `true` whenever historical OOF results were examined: the record may not claim that the exposure left its decisions uninformed. The working profile records `true`.
4. **Identity.** `fingerprint()` (16 hex characters) is the prefix of `content_sha256()` over the record's canonical JSON. The working profile's fingerprint is pinned in a test. Key order and layout do not affect it.
5. **Status.** A working record carries a pre-release version (`0.1.0-working`) and no approval. A frozen record carries a plain version and an approver and date. PR 38 loads with `require_status="frozen"`.
6. **Runtime.** Artifacts record `constraint_provenance(contract)`: source, contract id, fingerprint, and for schema 2 the schema version, contract version and status. `check_contract_provenance` refuses a wrong source or fingerprint for both schemas. For schema 2 it also refuses a recorded `contract_id`, `schema_version`, `contract_version` or `contract_status` that is missing or differs from the contract; schema-1 checking is unchanged. `check_rf_constraints_follow_contract` refuses RF constraints that do not follow the contract.

Numerical equality with a historical value is not a violation. Only its justification is checked.

## 11. RF adapter

| RF `OperatingConstraints` | From the schema-2 contract |
|---|---|
| `min_precision_not_subscribe` | `min_pass1_npv` |
| `max_subscriber_leakage_rate` | `max_pass1_leakage_rate` |
| `max_subscriber_leakage_absolute` | `max_pass1_leakage_count` (`None` → `NO_COUNT_CAP`) |
| `min_precision_subscribe` | `min_pass2_precision` |
| `min_band_support_pass1` / `_pass2` | `min_pass1_support` / `min_pass2_support` |
| `min_pass1_coverage`, `min_recall_subscribe` | `None` (the router maximises coverage) |
| `recall_denominator` | `"global"` |

`rf_constraints_from_contract_v2(contract)` takes every value from the contract. `rf_constraints_from_contract` keeps its Commit 3 signature; given a schema-2 contract it refuses floors that differ from the contract's. `check_rf_constraints_follow_contract` also checks the floors under schema 2. The historical constraints recorded in the baseline (D13) are refused under the working contract. RF confidence construction is unchanged.

## 12. Known GLASS / RF asymmetries

| Asymmetry | Status |
|---|---|
| **Enforcement.** RF enforces every obligation on its routed band. Historically GLASS enforced only the Pass 1 union leakage cap on its selected union; NPV, precision and support were per-rule gates, and a union of rules can fall below a per-rule floor | **Resolved in Step 4B, opt-in** (sections 16–17): every obligation is enforced on the selected union. Off by default; PR 38 activates it through the adapter |
| **Pass 2 objective.** GLASS maximises union positives; the contract specifies rows | Resolved in Step 4B, opt-in (section 17) |
| **Optimality certificate.** RF scans every threshold exhaustively; GLASS's ILP can stop on its time limit with an unproven incumbent | Documented (section 20). The gap is recorded; an unproven GLASS result needs a recorded reproduction to be primary |
| **Fit-time estimation.** RF checks inner-OOF scores; GLASS checks in-sample rule matches (F17 L2, F31 K2) | Documented. M6 diagnostics in PR 38; both arms assessed identically on outer OOF |
| **Evidence.** Only RF has a saved feasibility frontier | GLASS feasibility first observed at the PR 38 preflight |

## 13. Preserved F17, F31 and hyperparameter decisions

**F17 primary configurations** (activated only in PR 38, gate G1):

- GLASS: `pass2_population="oof_remainder"`, `pass1_oof_folds=5`, `random_state=42`, `remainder_min_size=500`, `inner_feature_ranker="fold_nested_rf"`.
- RF: `eligibility="nested_t1"`, `eligibility_folds=5`, `n_oof_folds=5`, `random_state=42`, `remainder_min_size=500`.

**F31 Disposition A** is unchanged:

- the empirical band NPV and precision;
- the probability-like transform;
- the Stage 4 interface.

RF confidence is a heuristic blend, not a calibrated probability. Because bands sit at or just above the floors, these floors largely set the RF confidence values.

**RF hyperparameters (M8).** The values used historically were read from the GLASS RF artifact `rf_result_20261023_025611.joblib` (SHA `76b198aa…`, attested in the baseline manifest):

| Setting | Value |
|---|---|
| `n_estimators` | 650 |
| `max_depth` | 6 |
| `min_samples_leaf` | 12 |
| `max_features` | 0.39989815740885837 |
| `minority_weight` | 1.3001084911026792 |
| `class_weight` | `"balanced"` |
| `pass2_rf_params` | `null` (Pass 2 reuses the Pass 1 settings) |

The notebook's `RF_PARAMS_EXPLICIT` differs on five of the six keys and must not be used as a stand-in.

How they were tuned: Optuna TPE, seed 42, 200 trials, 5-fold ROC-AUC CV on all of `X_eng_train`, with features engineered on the full training set and its labels.

Limitations, documented now:
- each outer fold's held-out labels helped choose the hyperparameters of the router that routes it;
- the tuning CV saw target-encoded features informed by its own validation labels;
- the RF Router uses these values in both forests, while GLASS uses an RF only to order its rule search, so any resulting optimism plausibly favours RF.

The version-controlled record is [`rf_router_primary_v1.json`](../../blackbox_pipeline/profiles/rf_router_primary_v1.json) (Step 4D, section 24). No nested tuning in PR 37.

## 14. Open requirements

| Step | Requirement |
|---|---|
| 4D | Done (section 24): frozen contract `1.0.0` (`0817825b9eecbd11`), frozen protocol `1.0.0` (`245a1a68fb68cf4e`) resolving D-C1 to D-C11 with the D-C10 fingerprints pinned, frozen GLASS (`198731d5cdf263f4`) and RF (`4e3f9c07e6ecc8ad`) profiles, the roadmap reconciled. The PR 38 solve time limit is unchanged; it is decided only if the preflight requires it, under M4 |
| Before Commit 4 | `tests/realdata/test_stage2_freeze_artifacts.py` passes on the pinned artifacts; the combined local validation passes |
| Commit 5 | Final integration verification on the final branch, a documentation consistency pass, the single PR 38 handoff checklist and the readiness sign-off |
| PR 38 | Re-verify the frozen specification on the artifacts; build both arms' configs with `glass_primary_config` and `rf_primary_config` from `load_frozen_contract()`; G13, then the fit-only G3 preflight for both arms; reruns; persistence of `selection_reports()` (inner folds included) and solver logs with the artifacts (D-C11); reproduction reruns for any unproven incumbent; the evaluation runner and report under the frozen protocol, in the order of protocol section 16; R1–R8; M6 diagnostics |

## 15. Regression coverage and preservation

| Requirement | Tests |
|---|---|
| Schema, ranges, vacuous and inconsistent settings, missing and unknown fields, malformed records | `test_stage2_operating_contract_v2.py`, sections 1–2 |
| Working profile values, status and pinned identity; deterministic serialisation | Same file, section 1 |
| Structured provenance; refusal of performance-derived values, of a rationale citing the GLASS-derived pathway and of test-derived evidence; the pathway described as history allowed; exposure that informed decisions disclosed; equality with historical values allowed | Same file, section 3 |
| A loader that reads one file; no artifact or arm import in the contract module | Same file, section 4 |
| Schema-2 artifact provenance: missing or mismatched id, schema version, contract version and status refused; schema 1 unchanged. RF adapter mapping, floors, refusal of the historical constraints | Same file, section 5 |
| Router defaults and the 15 baseline-pinned code files unchanged | Same file, section 6 |
| Exact boundaries, ceiling direction, support, zero support, class support | `test_stage2_obligations.py`, sections 1–2 |
| Undefined, observed and pooled evaluation; registry and `evaluate_router` parity; both arms' tokens | Same file, section 3 |
| RF solver and GLASS union cap equivalence; lexicographic-coverage choice | Same file, section 4 |
| Fit-outcome dispositions and contradictions; arm neutrality | Same file, sections 5–6 |
| GLASS union obligations: defaults off, union-level enforcement, exact indicators in both directions, exact ratios, boundaries, leakage cap together, empty unions, greedy and incumbent validation, exhaustive-search equivalence (Pass 1; Pass 2 under both objectives), configuration, determinism and ties, solver log | `test_glass_union_obligations.py`, sections 1–7 |
| Overlap-diagnostics equivalence (values, printed output, match count) and a microbenchmark | Same file, section 8 and `benchmark()` |
| One real fit under the obligations (opt-in, `-m slow`) | Same file, section 9 |
| GLASS and RF adapters, refusals, cross-arm equality, provenance | `test_stage2_contract_adapters.py`, sections 1–2 |
| Pipeline wiring: every selector under the contract, inner-fold isolation, Pass 2 on the OOF remainder, selection reports, no fallback after an inner-fold failure | Same file, section 3 |
| GLASS fit outcomes from real selections, the gap and the reproduction record | Same file, section 4 |
| Defaults unchanged | Same file, section 5 |
| Protocol record: pinned fingerprint, freeze guard, refusals, contract binding, equality with both adapters' F17 settings and the baseline manifest (artifacts included), registered procedure identities, every metric defined, arm neutrality | `test_stage2_evaluation_protocol.py`, section 1 |
| Population identity; upstream identity per row (count-preserving flag moves, a tampered view, a changed artifact file, EBM fold and per-row thresholds, pending fingerprints); paired alignment by row identity | Same file, section 2 |
| Denominators, pooled versus per-fold, undefined denominators, class-conditional reconciliation | Same file, section 3 |
| Bootstrap determinism, strata, pairing, undefined replicates; reference flags bound to the pairing (reordered and relabelled refused) and equal to `router_upstream_metrics`; Wilson | Same file, section 4 |
| Required selection keys; missing, duplicated, unexpected and unsupported records; GLASS selection-report cross-check; fit status; fold ledger refusing accepted status strings and a `FoldFit` that is incomplete, relabelled, stripped of its review or classified under other settings; pooling without silent drops; shortfall classification | Same file, section 5 |
| F31 trigger, structural versus comparative acceptance, evidence tiers, input guard; adoption gates from recomputed values on a verified reference (moved union flags, a wrong view, a changed artifact, wrong fingerprints refused; a working protocol never official), with C6 failing for a missing, edited, re-digested, foreign-procedure, other-settings, other-data, other-reference, incomplete or malformed run; the Pass 2 budget pooled and per fold | Same file, section 6 |
| Matched budgets: label-free, tie-break by row identity, bounds | Same file, section 7 |
| Pinned identities against the historical artifacts (opt-in, `-m realdata`) | `tests/realdata/test_evaluation_protocol_artifacts.py` |
| Step 4D: frozen contract identity and unchanged values; frozen protocol identity, freeze-only differences from the working record, D-C1 to D-C11, D-C10 pins, frozen-only bindings and record validation; the recorded approval and the approval gate (unconfirmed or missing approvers refused by every strict loader); strict PR 38 entry points with no injected record, primary-config checks and official decisions only under the pinned protocol; arm profiles (identities, historical values against the baseline manifest and the notebook, configs under the frozen contract, refusals, no GLASS-artifact read by RF); D-C11 report evidence (missing, unexpected, duplicated, null slots, inconsistent); the roadmap's PR 39 onward unchanged | `test_stage2_freeze.py` |
| Step 4D re-verification on the real artifacts: population and complete five-layer upstream identity under the frozen protocol, the verified reference, both profiles against the artifacts' recorded values (opt-in, `-m realdata`) | `tests/realdata/test_stage2_freeze_artifacts.py` |

**Preservation:**
- `pr37_stage2_pre_correction_v1` is not edited;
- none of its 15 pinned code files is touched;
- no notebook or model artifact is changed;
- every historical default is unchanged;
- the Commit 3, Step 4A and Step 4B tests are unchanged.

## 16. Step 4B: GLASS set-level obligations

Four optional `GlassRouterConfig` / `ILPRuleSelector` fields join the existing Pass 1 union leakage cap. Each is `None` by default, and the historical model is then built unchanged.

| Field | Obligation on the selected union | Rows it is measured on |
|---|---|---|
| `max_union_leakage_rate_pass1` (PR 34) | positives routed ≤ ⌊L·P⌋ | the Pass 1 selection population (P = its positives) |
| `min_union_npv_pass1` | negatives routed / rows routed ≥ F | the Pass 1 selection population |
| `min_union_support_pass1` | rows routed ≥ S₁ | the Pass 1 selection population |
| `min_union_precision_pass2` | positives flagged / rows flagged ≥ π | the Pass 2 selection population (the OOF remainder under `oof_remainder`) |
| `min_union_support_pass2` | rows flagged ≥ S₂ | the Pass 2 selection population |

**Formulation** (`ILPBuilder.add_union_set_obligations`). Rows are grouped into coverage atoms: maximal sets of rows matched by exactly the same candidates (`membership_atoms`). For every atom a covered by two or more candidates, a continuous y_a ∈ [0, 1] is linked to the rule binaries x_j in both directions:

    y_a ≥ x_j   for every candidate j covering a   (covered → counted)
    y_a ≤ Σ_{j covers a} x_j                      (uncovered → not counted)

so y_a equals the atom's membership in the selected union for every binary x, whatever the sign of its coefficient. An atom covered by one candidate uses x_j itself. Both directions matter: without the lower links the solver could leave a covered zero-positive atom uncounted under a precision floor, and without the upper links it could count positives no selected rule covers. The tests build both traps.

With r_a the atom's rows, g_a its "good" rows (negatives for NPV, positives for precision) and the floor read exactly as n/d:

    ratio floor:    Σ_a (d·g_a − n·r_a)·y_a ≥ 0
    support floor:  Σ_a r_a·y_a ≥ max(1, S)

Every coefficient is an integer. A limit is read as the decimal it is written as (`exact_fraction`, equal to the checker's `exact_limit`): 0.14 is 7/50, so 7 positives in 50 rows meets 0.14 although `0.14 * 50 > 7` in floating point. The union leakage cap keeps its PR 34 formula, ⌊L·P + 10⁻⁹⌋, which equals the exact cap ⌊3P/20⌋ at L = 0.15 for every P up to 500,000 (tested).

**Empty unions.** Under any set-level obligation an empty union is infeasible: the support floor is at least 1 row, and a ratio over zero rows is undefined. When the quality gates or candidate generation leave nothing, the selector raises a proven `SelectorInfeasibleError` instead of returning an empty pass.

**Per-rule gates** stay arm-internal candidate filters. They are never evidence of union compliance, and a config cannot claim the contract through them (section 17).

## 17. Step 4B: the GLASS and RF adapters

`glass_config_from_contract(base, contract)` takes values only from a schema-2 contract:

| `GlassRouterConfig` | From the contract |
|---|---|
| `max_union_leakage_rate_pass1` | `max_pass1_leakage_rate` |
| `min_union_npv_pass1` | `min_pass1_npv` |
| `min_union_support_pass1` | `min_pass1_support` |
| `min_union_precision_pass2` | `min_pass2_precision` |
| `min_union_support_pass2` | `min_pass2_support` |
| `pass2_union_objective` | `"rows"` (the lexicographic coverage objective) |
| `contract_provenance` | `constraint_provenance(contract)` |

It refuses a contract with a leakage count cap (GLASS has no union count constraint; the working profile sets none), a base without the F17 primary settings, a base that already sets a union field to another value (refused, not overwritten) and a base that already records a provenance. Everything else in the base (per-rule gates, rule counts, novelty, complexity, the RF ranker) is the arm's own profile. That profile is not a contract obligation, but it shapes GLASS coverage, so it must be fixed with the frozen record before PR 38 results (Step 4D).

`GlassRouterConfig` refuses a `contract_provenance` unless every union obligation is set and the Pass 2 objective is `"rows"`. `check_glass_config_follows_contract` refuses drift, a missing provenance and a foreign fingerprint.

`rf_config_from_contract(base, contract)` keeps every RF setting, hyperparameters included, and replaces only the constraints with `rf_constraints_from_contract_v2(contract)`. It requires the RF F17 primary settings (`two_pass`, `nested_t1`, 5 eligibility folds, 5 OOF folds, seed 42, remainder ≥ 500). `check_rf_config_follows_contract` refuses historical or drifted constraints. The RF adapter still imports nothing but the shared contract.

Under the working contract both arms carry the same five numbers (tested). The estimators differ: GLASS checks in-sample rule matches on its selection rows, RF checks inner-OOF scores (section 12).

## 18. Step 4B: populations and isolation

Every selector the GLASS pipeline builds comes from `_new_selector()`, which passes all union fields through. Under `oof_remainder`:

1. Pass 1 is selected on all fitting rows under the Pass 1 obligations.
2. Each inner fold's Pass 1 is selected, under the same obligations, on that fold's training rows only, with its own fold-nested RF ranker; it never sees the rows it routes.
3. The remainder is the rows not routed by their OOF Pass 1 decision.
4. Pass 2 is selected on exactly the remainder, under the Pass 2 obligations measured on the remainder.

An infeasible inner fold stops the fit; nothing falls back to `full`. `selection_reports()` returns copies of the Pass 1, Pass 2 and inner-fold reports. They are registered before each solve, so a failed pass stays visible. `fit_report_` is unchanged. Under `full` (the secondary sensitivity) Pass 2's obligations are measured on all fitting rows, as that pathway defines.

## 19. Step 4B: determinism and tie-breaking

- **Inputs.** Candidates enter the ILP in generation order, after the per-rule gates, segment deduplication and the constraint-equivalent merge. Twins (same rows and base-feature signature) keep the higher adjusted quality, then the earlier position. The merge is exact under union obligations, because those depend only on covered rows.
- **ILP.** Phase 1 maximises union rows (Pass 1, and Pass 2 under `"rows"`) or union positives (historical Pass 2). Phase 2 keeps union coverage at least at the Phase 1 value, minimises overlap, then maximises adjusted quality, with weights scaled so quality can never outweigh one unit of overlap. Exact ties beyond that are left to CBC.
- **CBC.** PuLP passes no thread option, so CBC runs with its default single thread. For a fixed model and CBC build, a search that completes is reproducible (tested by repeated solves). A search stopped by its time limit depends on wall-clock progress and may not reproduce. That is why an unproven incumbent needs a recorded rerun (section 20).
- **Greedy.** It picks the highest (union gain, secondary score); exact ties go to the earlier candidate. Under a ratio floor it skips any rule that would take the union below the floor. Its result is validated against every obligation, the support floor included, before use.

No solver tolerance, time limit (300 s per solve), rule-search setting or numerical constraint changed.

## 20. Step 4B: solver status, gap and fit outcomes

PuLP's CBC interface returns a status and a solution status, but no bound or MIP gap. CBC 2.10 prints them in its log. Under any set-level obligation the builder captures that log with PuLP's `logPath`, which changes only where CBC's output goes: the command line is identical. `parse_cbc_log` then reads:

- `Partial search - best objective A (best possible B)` after an early stop (bound B, basis `search`);
- `Search completed - best objective A`, or `Result - Optimal solution found` with `Objective value: A`, for a proven optimum (bound A, basis `proven_optimal`).

`gap_abs = |bound − objective|` and `gap_rel = gap_abs / |objective|`. A value the log does not state stays `None`; nothing is estimated. CBC prints the early-stop line with six significant digits, so that gap carries the same precision. The same messages appear in the CBC builds bundled with PuLP 2.9 (Linux) and PuLP 3.3 (Windows, 64-bit); another build that words them differently leaves the gap `None`. Each phase's reading is stored in its report entry (`solver_log`). Historical solves capture no log.

`glass_fit_outcome(scope, selection_report, report)` maps a pass onto the Step 4A taxonomy:

| Selection report | Outcome | Disposition |
|---|---|---|
| `source="ilp"`, both phases proven | `feasible_optimal` | Primary |
| `source="ilp"`, any phase unproven | `feasible_unproven_incumbent` | Primary only if documented: status, Phase 1 gap and `reproduced=True` |
| `source="greedy_fallback"` | `feasible_greedy_fallback` | Requires review |
| `source="infeasible"` with a stage, or Phase 1 proven infeasible | `proven_infeasible` | Stop |
| `source="infeasible"` otherwise | `no_feasible_solution_found` | Stop |
| `source="empty …"` (historical, `min_rules = 0`) | `insufficient_population_or_support` | Stop |

A feasible claim is refused unless the arm-neutral check of the selected union (`glass_pass_report`) holds. `glass_record_reproduction(outcome, first, rerun)` sets `reproduced` only after an explicit rerun of the same fit: true only if the source, the selected rule ids in order and the union counts match. When Phase 1 is proven and only Phase 2 is unproven, the coverage objective is optimal; the outcome is still classified as unproven, which is conservative.

**Observed on synthetic data** (900 training rows; 85 Pass 1 candidates in the ILP; the working ratios 0.95 / 0.15 / 0.30 with support floors scaled to 60 / 20). With 30 s per solve, Pass 1 under these obligations stopped on its time limit with a 16.9% relative gap (incumbent 449 rows, bound 524.7). The historical Pass 1 also stopped unproven at 30 s on the same data. At 2 s, CBC found no Pass 1 incumbent under the obligations and the validated greedy result was used; for Pass 2 it found none either, the greedy result fell below `min_pass2_rules`, and the fit stopped (`no_feasible_solution_found`). At 5 s the same fit completed. Pass 2 was proven optimal at 30 s. Production uses 300 s, and real-data behaviour is first observed at the PR 38 preflight.

## 21. Step 4B: what the obligations do not establish

- Fit-time compliance is in-sample for GLASS. It is a selection constraint, not an estimate of outer-OOF performance; the shortfall protocol is Step 4C.
- The working values remain provisional until the Step 4D freeze.
- Union obligations narrow GLASS's feasible set. Whether they are feasible on the PR 38 `oof_remainder` populations is unknown until the preflight, and M4 applies: no automatic relaxation.

## 22. Step 4B: GLASS selector performance audit

Profiled on one synthetic `full` fit (1,200 rows; 900 training rows; 5 s per solve), cProfile cumulative times:

| Area | Time | Finding |
|---|---|---|
| Candidate evaluation: overlap diagnostics | 51.8 s of 104 s | Printed diagnostics only. Re-matched the second rule of every pair: 40,200 of 40,600 `match_rule` calls |
| Candidate generation: beam search | 30.8 s | 29.8 s of it in `rule_metrics.compute_all_metrics` (14,012 calls, pandas Series operations). Rule search: not changed |
| ILP solves | 15.0 s | CBC at its time limit. Solver settings: not changed |
| Per-rule quality gates, Pass 1 | 1.4 s | A per-index `y_val.iloc` loop; grows with covered rows on real data. Candidate for a later change |
| Selector outside CBC, under the working obligations | 0.5 s (union indicators 0.1 s) | Measured on a separate fit with the obligations on. Not material |

**One change, behaviour-preserving.** `OverlapAnalyzer.pairwise_jaccards` matches each rule once and computes every pair's intersection from cached boolean masks. Union = |A| + |B| − |A ∩ B| in exact integers, so each Jaccard value is bitwise identical, the pair order is unchanged and the printed report is identical (tested on rules with missing features, empty matches and duplicates). It does not affect any selection: the diagnostics only print.

| Measurement | Before | After |
|---|---|---|
| Overlap diagnostics on the fit's 200 + 200 evaluated rules | 17.4 s | 0.15 s |
| Wall-clock fit, same data (one run each) | 45.0 s | 28.9 s; identical selected rules |
| Microbenchmark, 124 rules × 2,000 rows (`python -m tests.stage2.test_glass_union_obligations`) | 2.56 s | 0.04 s |

Timings are from the sandbox and only indicate relative cost. Locally (Windows), the same microbenchmark (124 rules × 2,000 rows, 7,626 pairs) ran in 2.39 s before and 0.041 s after, about 58× faster. The gain is in the overlap diagnostics only, not overall training. The remaining costs above (beam-search metrics, the Pass 1 quality-gate loop) are recorded and not changed in PR 37.

## 23. Step 4C: the pre-registered PR 38 evaluation protocol

The protocol is [`pr38_evaluation_protocol.md`](pr38_evaluation_protocol.md), with its record `shared/stage2/protocols/pr38_evaluation_working.json` (`pr38-stage2-evaluation` `0.1.0-working`, fingerprint `161c6c589696b859`). It binds this contract (`a6ca7462d4773005`). It is **working, not frozen**, and `require_frozen` stops it from governing PR 38 until Step 4D freezes it.

It was written before any corrected result existed. It reuses the diagnostics registry, the three-state trace, the Step 2B and 2C contribution and matched-rate code, and the Step 4A observation checks: all imported, none modified. It adds only:

- the record;
- identity checks: population, and upstream row by row;
- pairing and reference flags bound to row identity;
- the stratified paired bootstrap with registered runs;
- selection-record completeness;
- the fit-status and fold ledger;
- shortfall classification;
- the F31 R2 trigger;
- acceptance, evidence-tier and adoption helpers.

The first review revision closed four gaps:

1. Upstream identity checks per-row decisions, not just counts.
2. A fold's status needs every required selection exactly once.
3. Reference flags must match the pairing's rows.
4. Adoption's C6 passes only when the reported bootstrap run reproduces exactly.

The second review revision closed four more:

1. The fold ledger accepts a fold only through a `FoldFit` that re-classifies from its complete selection records. A status string never makes a fold accepted.
2. Adoption takes its reference only from upstream sources that pass `check_upstream_identity` on the paired rows (`verify_reference`). A decision is official only under a frozen protocol with a complete identity.
3. A malformed or incomplete reported bootstrap run fails C6 with its reason; it never raises.
4. The Pass 2 flag budget is held pooled in C1, and per fold in C5: a fold whose own budget grew does not count as reproduction (protocol section 13).

| Requirement | Implemented in Step 4C | PR 38 execution obligation | Needs Step 4D approval |
|---|---|---|---|
| Populations and identity | Pinned identity; `check_population_identity`; `PairedRouting` | Run on the corrected views | |
| Primary hierarchy | V1–V8, O1–O4, descriptive totals | Report in that order | Statement rule (D-C1) |
| Secondary diagnostics | Named and defined; two protocol metrics | Compute | |
| Frozen upstream points | Pinned artifact SHA-256, LR threshold record, EBM fold thresholds and counts; `check_upstream_identity` (five layers) | Verify with `require(complete=True)`; contribution tables | Pin the row-aligned flag fingerprints (D-C10) |
| Matched budgets | Step 2C method pinned to the record | Run per arm | RF-truncation diagnostic (D-C8) |
| Pre/post | `PairedRouting.pre_post`, transitions, attribution rules | Run per arm, with S1 | |
| Uncertainty | `stratified_paired_bootstrap`, `run_registered_bootstrap` (procedure `pr38.stage2.stratified_paired_bootstrap/1`), `ReferenceFlags`, `wilson_interval` | Apply | Settings (D-C3) |
| Shortfalls | `classify_observation` | Classify and report | 2-SE rule (D-C2); low-support 30 (D-C4) |
| Infeasibility and folds | `required_selection_keys`, `check_selection_completeness`, `classify_fold_fit`, `FoldLedger` (accepts a fold only through a reproducing `FoldFit`), `pooled_observation`; fit-only preflight procedure | Record every selection of every fold | All five folds required (D-C6); review record for a greedy fallback (D-C7) |
| F31 boundary | `f31_strip_review` at 1/20 | Compute R2 and R3 inside the cross-fit | |
| Complementarity ablation | Criteria C1–C9; `evaluate_adoption` on a verified upstream reference (`verify_reference`), recomputing every interval and per-fold budget with the registered run (C6: the reported run must reproduce; a malformed run fails); Pass 2 budget pooled (C1) and per fold (C5) | Only if the ablation is run; official only under the frozen protocol | Criteria, margins and the budget policy: proposed judgements (D-C5) |
| Artifact acceptance | `structural_acceptance`, `comparative_report_gaps` | Assemble gate evidence | |
| Exposure and tiers | Tiers, the source registry, `check_inputs` (defense in depth only) | The seven provenance checks (protocol section 15); label every result | |
| Freeze | Working record, fingerprint, freeze guard (a frozen record must pin the flag fingerprints) | | Freeze and re-bind (D-C9) |

## 24. Step 4D: the frozen Stage 2 specification

Step 4D freezes what PR 38 runs under. It changes no numerical value, definition, denominator, floor, ceiling or objective, and no F17 or F31 decision. Every frozen record is loaded only through a loader that checks its status, its pinned fingerprint and a confirmed owner approval.

| Record | Version | Fingerprint | Loader |
|---|---|---|---|
| Operating contract `pr37-stage2-primary` | `1.0.0`, frozen | `0817825b9eecbd11` | `shared.stage2.contract.load_frozen_contract` |
| Evaluation protocol `pr38-stage2-evaluation` | `1.0.0`, frozen | `245a1a68fb68cf4e` | `shared.stage2.evaluation_protocol.load_frozen_protocol` |
| GLASS profile `pr37-glass-router-primary` | `1.0.0`, frozen | `198731d5cdf263f4` | `glass_pipeline.glass_router.profile.load_glass_primary_profile` |
| RF profile `pr37-rf-router-primary` | `1.0.0`, frozen | `4e3f9c07e6ecc8ad` | `blackbox_pipeline.models.rf_router.profile.load_rf_primary_profile` |
| Superseded, kept for history | Contract `0.1.0-working` (`a6ca7462d4773005`); protocol `0.1.0-working` (`161c6c589696b859`) | | |

The protocol binds the contract (D-C9) and both profiles (`arm_profiles`, `check_arm_profile_binding`); a frozen protocol accepts only frozen bindings. **Approval.** Phillip Harris formally approved the Step 4D freeze on 2026-11-05, including the historical legacy GLASS Router profile as the primary PR 38 configuration with the frozen F17 population settings and the shared operating constraints. All four records carry the same approval (`approved_by`, `approved_on`); it was recorded on the owner's express instruction, never inferred from an author field. A record whose approver is missing or carries the `UNCONFIRMED` marker is refused by every strict loader and by `require_frozen` (`approval_confirmed`).

**What is frozen, unchanged.**

- **Constraints:** Pass 1 NPV ≥ 0.95; Pass 1 positive leakage ≤ 0.15 of all positives; Pass 1 routed-set support ≥ 125; Pass 2 precision ≥ 0.30; Pass 2 routed-set support ≥ 75; no leakage-count cap. The support floors were reviewed and kept as protective guards, not precision guarantees (M5).
- **Objective:** lexicographic coverage (M1).
- **F17:** GLASS Pass 2 on the OOF Pass 1 remainder with fold-nested Pass 1 eligibility (`fold_nested_rf`); RF `nested_t1`. GLASS `full` stays the S1 sensitivity only.
- **F31 Disposition A:** RF bands and blending unchanged and never presented as calibrated probabilities; the 1/20 boundary-strip trigger (R2).
- **PR 38 gates:** the G13 audit of `neighborhood_subscription_density` and the G3 fit-only preflight remain required before primary results.

**Protocol decisions (protocol section 17).**

| ID | Resolution | Kind |
|---|---|---|
| D-C1 statement rule | Approved | Reporting rule |
| D-C2 2-SE materiality | Approved | Descriptive heuristic, not a compliance test |
| D-C3 bootstrap settings | Approved | Registered procedure; intervals conditional on fitted models |
| D-C4 low support 30 | Approved | Reporting convention |
| D-C5 adoption criteria, margins, Pass 2 budget pooled and per fold | Approved | Registered criteria; the margins are judgements |
| D-C6 all five folds | Approved | Registered criterion |
| D-C7 fit-time review of a greedy fallback | Approved | Registered criterion |
| D-C8 RF Pass 2 outer scores | Not required | Optional diagnostic; never a PR 38 prerequisite |
| D-C9 bindings | Approved | Frozen contract and profiles |
| D-C10 flag fingerprints | Approved: LR `64d511516d69e73e`, EBM `907a7d2d1cad6908` | Binding, re-verified on the artifacts |
| D-C11 GLASS selection reports | Approved | Registered procedure (below) |

The resolutions above are approved with the records.

**D-C10: re-verification, not a copied value.** The fingerprints came from the read-only audit of the pinned artifacts, which also confirmed the artifact SHA-256 identities, row and label alignment, EBM outer folds, frozen thresholds, per-row decisions and the counts 7,727 / 6,113 / 8,001 / 5,839. `tests/realdata/test_stage2_freeze_artifacts.py` re-runs the complete five-layer identity under the frozen protocol with `require(complete=True)`, plus the verified reference. It must pass before Commit 4 and before any PR 38 run; a failure voids the freeze.

**D-C11: GLASS report evidence.** The Step 4C cross-check rejected a record without a report, but a report for a selection that is not required (`inner_pass1[5]`) went unnoticed, and a report's content was never compared with its record. Under the frozen protocol (`fit_status.glass_selection_reports = "required"`):

- a GLASS fold without `selection_reports()` is `fit_time_invalid`;
- the reports must correspond exactly to the required selections, slot by slot with `None` entries counted: none missing (a `None` in a required slot is missing), none unexpected (a slot beyond the required count, a trailing `None` included), no report object standing for two selections;
- each record must be its report's classification: the outcome the report's `source` and ILP guarantee map to (as `glass_fit_outcome` does) and the same selected rule ids;
- `FoldLedger.build` re-classifies with the stored reports, so stripping them afterwards does not reproduce.

The RF Router has no equivalent report API; its fit evidence remains its selection records. This asymmetry is documented, the existing `FoldFit` requirements are unchanged, and the Step 4B solver semantics are untouched. The working protocol keeps the reports optional, as Step 4C defined it, but now refuses unexpected, duplicated and inconsistent reports whenever they are supplied.

**The PR 38 pathway takes no injected record.** PR 38 obtains everything through strict loaders that check status, pinned fingerprint and confirmed approval, with no review or override switch:

- `load_frozen_specification()`: the frozen protocol and contract, bound;
- `glass_primary_config()` and `ranker_settings()`; `rf_primary_config()` (only the run-time options `verbose` and `validate_nested`);
- `check_glass_primary_config` and `check_rf_primary_config` refuse any config that is not exactly the primary one, so a configuration built by hand or from another profile (`glass_config_from_profile`, `rf_config_from_profile`, for reviews and tests) is never treated as the PR 38 primary;
- decisions are official only under the pinned protocol itself (`is_pinned_frozen`): `AdoptionDecision.official` and `StructuralAcceptance.official` are false for any frozen record built another way.

**Arm configurations.**

- **RF (M8).** Before Step 4D the black-box notebook read the hyperparameters from the newest GLASS-arm RF artifact at run time. The RF profile now owns them: the six historical values exactly (`n_estimators` 650, `max_depth` 6, `min_samples_leaf` 12, `max_features` 0.39989815740885837, `minority_weight` 1.3001084911026792, `class_weight` balanced), `pass2_rf_params` null, the F17 fold settings, and the provenance of both artifacts they were copied from (GLASS-arm RF result `76b198aa…`, RF Router `5204a2d6…`) with the tuning method and its full-training-data exposure. `rf_primary_config` imports no artifact reader. The historical pathway (`tuning.legacy`) and the historical artifacts are unchanged. `verbose` and `validate_nested` stay run-time options: they never change a selection. The record lives in `blackbox_pipeline/profiles/`, not under `models/`, whose JSON files are Git-ignored as model artifacts.
- **GLASS.** The profile is the historical legacy profile (notebook Cell 13) preserved exactly, with three changes only: the union leakage cap (legacy 0.14) is removed because every union obligation comes from the contract (0.15); the population and nesting settings are the F17 primary; the Pass 2 union objective is `"rows"` via the adapter. The ranker records `glass_rf_model = "fold_rf"` and the same six hyperparameter values as GLASS's own historical setting. The legacy gates were set with historical OOF results visible and were not tuned for the union-level contract; GLASS feasibility under them is first observed at the G3 preflight (M4 applies).

The two arms share hyperparameter values, recorded independently in each arm's profile, and never a fitted estimator.

**Limitations kept.** The bootstrap intervals are conditional on the fitted routers and frozen upstream flags. The nominal test split was evaluated in PRs 33–36 and is never Tier 3 evidence. The margins and support floors are judgements and guards, not established guarantees. No corrected result existed at the freeze.
