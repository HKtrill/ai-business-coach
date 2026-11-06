# PR 38 Stage 2 Evaluation Protocol (pre-registered, frozen)

| | |
|---|---|
| Protocol | `pr38-stage2-evaluation`, version `1.0.0`, status **frozen** (Step 4D) |
| Record | `shared/stage2/protocols/pr38_evaluation_v1.json` (fingerprint `245a1a68fb68cf4e`), loaded by `load_frozen_protocol`. The Step 4C working record `pr38_evaluation_working.json` (`0.1.0-working`, `161c6c589696b859`) is kept for history and may not govern PR 38 |
| Code | `shared/stage2/evaluation_protocol.py` |
| Tests | `tests/stage2/test_stage2_evaluation_protocol.py`, `tests/stage2/test_stage2_freeze.py`; opt-in: `tests/realdata/test_evaluation_protocol_artifacts.py`, `tests/realdata/test_stage2_freeze_artifacts.py` (the freeze re-verification) |
| Binds | Contract `pr37-stage2-primary` `1.0.0`, fingerprint `0817825b9eecbd11`; arm profiles `pr37-glass-router-primary` `1.0.0` (`198731d5cdf263f4`) and `pr37-rf-router-primary` `1.0.0` (`4e3f9c07e6ecc8ad`) |
| Written | PR 37, Commit 4, Step 4C, revised after two reviews: (1) upstream identity, fit completeness, reference alignment, C6; (2) fold-ledger evidence, verified adoption reference, malformed bootstrap runs, the Pass 2 budget policy. No corrected PR 38 result existed when it was written |
| Freeze | Step 4D, formally approved by Phillip Harris on 2026-11-05. Every pending decision is resolved (section 17). `load_frozen_protocol` and `require_frozen` refuse any record without a confirmed approval. `tests/realdata/test_stage2_freeze_artifacts.py` re-verifies the pinned values on the real artifacts and must pass before the freeze is relied on; a failure voids it. In effect from Commit 4 (`f7a987e`); the PR 38 execution checklist is [`pr38_handoff.md`](pr38_handoff.md) (Commit 5) |

This protocol fixes how the corrected Stage 2 routers will be judged. It does not choose or improve a router. Three kinds of statement appear below, and each result PR 38 reports carries one of these labels:

- **Confirmatory within Tier 2:** a pre-registered primary outcome, computed exactly as specified here, after the freeze. It is still subject to the exposure history in section 15.
- **Exploratory:** anything chosen, extended or adopted after results are seen, and every complementarity-ablation result.
- **Descriptive:** counts, rates, intervals and diagnostics reported without a decision attached.

## 1. Principle

Both arms work under the same contract obligations and are evaluated identically. Each arm optimises its own coverage independently. They may differ in architecture, thresholds, rules and fitted estimators, and those differences are reported, never hidden. A shared contract does not make the arms mathematically equivalent:

- GLASS checks its obligations on in-sample rule matches.
- RF checks them on inner-OOF scores.
- Only the outer-OOF evaluation is common to both.

## 2. Evaluation population and identity (V1)

**Primary universe.** The 32,950 training rows of the global split, each scored out of fold by the five-fold outer cross-fit. Every primary cross-arm number uses this universe with:

- identical row identities, labels and outer folds;
- identical metric formulas and denominators;
- all three exclusive states, `PASS_1_NOT_SUBSCRIBE`, `PASS_2_SUBSCRIBE` and `ABSTAIN`.

| Pinned identity | Value | Source |
|---|---|---|
| Rows / positives / negatives | 32,950 / 3,712 / 29,238 | Baseline manifest |
| Outer folds | 5, `StratifiedKFold(shuffle=True, random_state=42)` | Fold plan fingerprint `8aea082fcaf371b7` |
| Rows per fold | 6,590 × 5 | |
| Positives per fold | 742, 742, 742, 743, 743 | |
| Router split fingerprint | `fc3e73a5ef91aea4` (row indices only) | Validated |
| Train label fingerprint | `b984d2a50527` | Validated, and recomputed by the check |
| `stage_io` split fingerprint | `d6cddadc3439` | Attested, never recomputed (its hash includes test labels) |
| Baseline | `pr37_stage2_pre_correction_v1`, manifest SHA-256 `905a0b48…` | Frozen |

**Before any cross-arm metric is computed:**

1. Build each arm's corrected view with `boundary.extract_router_view`.
2. Build the trace with `trace.build_router_trace`, which runs checks T1–T7.
3. Run `check_population_identity(trace, PopulationIdentity.from_protocol(protocol))` and require it to pass. It checks the counts, the per-fold counts, that the fold ids reproduce the seeded plan, both arms' declared fold fingerprints, the router split fingerprint, and the label fingerprint (recorded and recomputed).

**Alignment rules.** Alignment is by row identity, never by position. `PairedRouting` refuses rows that differ, the same rows in a different order, labels that differ and fold ids that differ. Two eligibility populations of similar size are never treated as interchangeable.

**Other populations, always labelled with their own denominators:**

| Population | Used for | Label |
|---|---|---|
| Arm-specific remainder | Pass 2 rows / rows its own Pass 1 left (`pass2_remainder_rate`); `pass2_recall_eligible` | "arm-specific remainder" |
| Common eligible: neither arm's Pass 1 | Matched-budget comparisons (section 7) | "common eligible" |
| A fit's selection population | Fit-time obligations only | "fit-time", never an evaluation result |

**F17 primary settings** (identical to the adapters' `GLASS_F17_PRIMARY` and `RF_F17_PRIMARY`; tested):

- **GLASS:** `pass2_population="oof_remainder"`, `pass1_oof_folds=5`, `random_state=42`, `remainder_min_size=500`, `inner_feature_ranker="fold_nested_rf"`.
- **RF:** `mode="two_pass"`, `eligibility="nested_t1"`, `eligibility_folds=5`, `n_oof_folds=5`, `random_state=42`, `remainder_min_size=500`.

GLASS `pass2_population="full"` is the S1 single-factor sensitivity analysis (F17 decision record, section 6). It is descriptive and never the primary.

## 3. Hierarchy: validity gates, primary outcomes, secondary diagnostics

### 3.1 Validity gates (per arm; all before any comparison)

| Gate | Requirement | Sources |
|---|---|---|
| V1 | Population identity (section 2) | T1–T7; the pinned identities |
| V2 | Configuration: F17 primary settings, adapters applied, the bound contract (`check_contract_binding`, `check_glass_config_follows_contract`, `check_rf_config_follows_contract`) | G1, G9 |
| V3 | Fit-time compliance: every fit (each outer fold's Pass 1 and Pass 2, every GLASS inner Pass 1 selection) has an accepted fit status (section 10), and its fit-time obligation report holds | M3, G8 |
| V4 | Isolation: remainder and ranker rows inside their fold; no held-out row in any fit; the G13 feature-engineering audit resolved; no test label anywhere | G4, G13 |
| V5 | Persistence and provenance: remainder sizes, positives and prevalence; `t1_j`; selection reports, including inner folds; solver logs; band records; eligibility provenance | G5, R1, R2, R5 |
| V6 | Reproducibility of at least one outer fold | G7 |
| V7 | New versioned artifacts, Stage 4 interface checks, wiring, baseline untouched (`--verify`) | G2, G10, G11, R4 |
| V8 | F31 Disposition A semantics: RF confidence labelled a heuristic blend, not a probability; R7 relabelling; R8 coupling to G13 | F31 A, R7, R8 |

A valid fit is not good performance. The gates establish that a result can be interpreted; they say nothing about which arm is better.

### 3.2 Primary outcomes (pooled and per fold, both arms, paired difference GLASS − RF)

| ID | Metric | Formula | Denominator | Role |
|---|---|---|---|---|
| O1 | `pass1_rate` | N_P1 / N | all rows | First lexicographic objective |
| O2 | `pass2_rate` | N_P2 / N | all rows | Second lexicographic objective, read only together with O1 |
| O3a | `pass1_npv` | TN_P1 / N_P1 | Pass 1 rows | Observed contract outcome |
| O3b | `pass1_leakage_share` | FN_P1 / P | all positives | Observed contract outcome (ceiling) |
| O3c | `pass2_precision` | TP_P2 / N_P2 | Pass 2 rows | Observed contract outcome |
| O4a | `unique_tp` | P2 ∧ ¬U ∧ y=1 | count | Pass 2 positives the frozen LR-or-EBM union U misses |
| O4b | `unique_fp` | P2 ∧ ¬U ∧ y=0 | count | Their false-positive cost |
| O4c | `pass1_fp_suppression` | P1 ∧ U ∧ y=0 / U ∧ y=0 | union false positives | Pass 1 contribution |
| O4d | `pass1_tp_cost` | P1 ∧ U ∧ y=1 / U ∧ y=1 | union true positives | Its positive cost |

Total coverage `(N_P1 + N_P2) / N` and `abstain_rate` are reported as **descriptive totals**. The fitting objective is lexicographic (Pass 1 rows, then Pass 2 rows on the arm's own remainder), not total coverage. Total coverage never substitutes for it, and no retrospective aggregate score is used.

**Statement rule (D-C1, approved at the freeze).**

- Report outcomes in the order O1, O2, O3, O4.
- A paired difference is described as "higher" or "lower" only when its 95% stratified paired bootstrap interval (section 9) excludes 0. Otherwise it is "not resolved at this sample".
- There is no weighted score and no overall winner.
- O2 is read with O1. An arm with more Pass 2 rows but fewer Pass 1 rows has not done better under the objective.
- O3 qualifies O1 and O2. Coverage gained with a material outer-OOF shortfall is reported as exactly that.
- These are descriptive, per-outcome statements. The intervals are unadjusted for multiplicity and conditional on fixed fits (section 9), so the multiple primary outcomes support no unqualified global superiority claim.

### 3.3 Secondary diagnostics

All are decision-level; definitions come from `diagnostics.metrics.METRIC_DEFINITIONS` unless marked †, in which case they come from `evaluation_protocol.PROTOCOL_METRICS`.

| Group | Metrics |
|---|---|
| Pass 1 | `pass1_npv`, `pass1_leakage_share`, `pass1_negative_recall` (negatives routed / all negatives), `pass1_positive_rate` (within-band leakage) |
| Pass 2 | `pass2_precision`, `pass2_recall` (/ all positives), `pass2_recall_eligible` (/ positives not routed by Pass 1, reported separately), `pass2_remainder_rate` (arm-specific denominator), false positives (count), `pass2_negative_routing_rate`† (Pass 2 negatives / all negatives) |
| Abstention | `abstain_rate`, `abstain_positive_rate` (prevalence), `abstain_negative_share`† |
| Per state | rows, positives, negatives, `state_positive_rate`, `state_negative_rate`; class-conditional shares (`class_conditional_counts`) |
| Contribution | `incremental_positive_recall`, `incremental_yield`, `marginal_precision`, `union_precision` (hypothetical OR, not Stage 4), `pass2_upstream_redundancy`, `pass2_unique_positive_share`, `pass1_upstream_flag_rate` |
| Cross-arm | `cross_state_agreement`, `cross_routed_agreement`, `cross_opposite_rate`, `cross_one_abstains_rate`, `cross_both_abstain_rate`, the 3 × 3 crosstab |
| Support | per-fold routed rows and positives; class support of every remainder |
| M6 | GLASS inner-OOF Pass 1 union NPV and leakage, where available; diagnostics only |

RF Router confidence is reported only as the F31 Disposition A heuristic blend, never as a calibrated probability.

## 4. Metric conventions

- Every rate is a registry `Ratio` with its numerator and denominator. An empty denominator is `undefined_empty_denominator`: never 0, never compliant.
- Pooled values sum numerators and denominators across folds (`pool_counts`); fold ratios are never averaged. A test shows the two differ.
- Per-fold tables (`metric_table`) list every fold and the pool. `fold_dispersion` gives defined and undefined fold counts, minimum, maximum, range and the pooled value.
- A denominator never changes silently. The tables carry them, and the tests pin them.

## 5. Frozen upstream operating points

The Stage 1 LR and Stage 3 EBM artifacts are the ones the baseline manifest records. They are pinned by the SHA256SUMS snapshot: LR `61a6b144…`, EBM `cfdac6d2…`. They are never retuned against corrected Stage 2 results.

| Operating point | Rule | Selection | Flags |
|---|---|---|---|
| `lr.in_stage_f2_cv` | Calibrated OOF probability ≥ 0.10 | Global F2 on all training labels: in-selection, optimistic; disclosed | 7,727 |
| `ebm.nested_f2` | Stored fold-nested decisions; raw thresholds 0.56, 0.56, 0.54, 0.49, 0.49 | Fold-nested | 6,113 |
| Union (either) | | | 8,001 |
| Intersection (both) | | | 5,839 |

Counts alone do not identify a flag set: two masks can share every count, including the union and intersection totals. `check_upstream_identity(refs, protocol, lr_view=, ebm_view=, population=)` therefore checks five layers, and every one must pass before a contribution metric is computed:

| Layer | Check | Pinned now? |
|---|---|---|
| 1. Artifact | Each view was extracted from a file (`source.path`) whose SHA-256 equals the baseline manifest's record (`upstream.artifacts`): LR `61a6b144…`, EBM `cfdac6d2…`. Checked both at extraction (the extractor's `file_sha256` provenance) and at check time. A byte-identical artifact fixes every stored per-row value | Yes |
| 2. Rows | Each view covers the evaluation rows, in order, with their labels. The EBM's fold ids are the evaluation folds | Yes |
| 3. Threshold provenance | The LR record: 0.10 in `stage_output.threshold`, global selection. The EBM: fold-nested thresholds equal to the pinned (0.56, 0.56, 0.54, 0.49, 0.49) in `oof.decision / oof.threshold_per_row`, and every row's threshold equal to its fold's | Yes |
| 4. Decisions | Each supplied flag set equals the registered operating point recomputed from its view (`resolve_operating_point`), row by row. Union and intersection equal the recomputed OR and AND | Yes |
| 5. Counts and row fingerprints | The four counts; the row-aligned fingerprint of each flag set (`flag_fingerprint`, SHA-256 over each row's index label and flag, in row order) | Counts yes; fingerprints pinned at Step 4D (D-C10): LR `64d511516d69e73e`, EBM `907a7d2d1cad6908`. The working record leaves them **pending**, never passed; a frozen protocol must pin both |

The tests deliberately move flags between rows while keeping all four counts. Flags supplied that way fail layer 4. A view altered after extraction, so that recomputation agrees with it, fails layer 5 only: layers 1–4 cannot see it, which is why the pinned fingerprints are required before the freeze. A changed artifact file fails layer 1.

If PR 38 must change an upstream artifact (G13 is not expected to), that change needs a new protocol version before results. Secondary operating points (`SECONDARY_OPERATING_POINTS`) are reported as secondary only.

**Contribution against the frozen points.** Use the Step 2B and Step 2C definitions, against LR, EBM, their union and their intersection.

- **Pass 2:**
  - total TP routed `SUBSCRIBE`;
  - TP also flagged by LR, by EBM, by either and by both;
  - unique TP against the union;
  - the corresponding FP counts and each set's flag count (budget).
- **Pass 1:**
  - TN routed `NOT_SUBSCRIBE`;
  - negatives LR flags, EBM flags, either flags and both flag, among those routed;
  - FP suppression rate;
  - positives Pass 1 leaks, and those of them the references flag (TP cost).

Raw overlap is not incremental benefit. Every contribution is read with its flag budget and its denominator.

## 6. Two comparison families, kept separate

| Family | What is compared | Never mixed with |
|---|---|---|
| Frozen operating points (section 5) | Each arm against the fixed LR, EBM, union and intersection flags | Matched budgets |
| Matched budget (section 7) | Each arm against reference flags sized to that arm's own Pass 2 count | Frozen points |

## 7. Matched-budget comparisons

This is the Step 2C method, unchanged (`contribution.matched_reference_selection`, `pass2_matched_contribution`), so that pre and post results stay comparable.

| Question | Answer |
|---|---|
| Flag event matched | `PASS_2_SUBSCRIBE` |
| Universe | Common eligible: rows neither arm routed to Pass 1, per outer fold. Not either arm's fit population |
| Target K | The arm's Pass 2 rows in that fold's common eligible population (arm-specific K) |
| Ranked models | LR calibrated OOF probability; EBM raw OOF score; each ranked inside its own outer fold |
| Ties | Broken by ascending SHA-256 of the row's index label: fixed per row, not row order, label-free. The strict and inclusive cuts (`matched_rate_cut`) are reported as bounds; neither is preferred |
| Exact? | Exactly K rows after the tie-break; the bounds show the tie effect |
| Labels | Read only to count outcomes after selection. They never choose K, a cut, a tie or a comparator |
| Routers | GLASS and RF Pass 2 are never ranked or truncated: no continuous Pass 2 outer score is persisted (GLASS routes by rule membership) |
| Undefined | K = 0 in a fold gives empty sets and undefined ratios, reported as such. A score with NaN in the universe is refused |
| Pass 1 (secondary) | Each arm's Pass 1 set against the K lowest-scoring LR rows of all rows (`matched_rate_comparison`, strict and inclusive cuts) |

Results are labelled "descriptive matched-budget comparison", never a deployable threshold. Each arm is matched against references at its own budget. **Cross-arm budget equality is not achieved by this family.** An optional RF-truncated-to-GLASS-K diagnostic would require persisting RF Pass 2 outer scores (D-C8). It would be asymmetric, because GLASS has no score, and it would be descriptive.

## 8. Pre-correction versus post-correction

**Reference.** `pr37_stage2_pre_correction_v1` (aggregate tables), plus the historical views it was built from (rebuilt by `stage2_baseline.py --check`, which must match). The baseline and its code identities are never changed.

**Pairing.** For each arm, `PairedRouting.pre_post(historical_view, corrected_view)` pairs the same rows, labels and folds; it refuses anything else. Report:

- the 3 × 3 state transitions with positives and negatives (`transitions`), and the changed-row count: which rows moved, not only the rates;
- the changes in Pass 1, Pass 2 and total coverage, NPV, leakage, Pass 2 precision and both recalls, unique positives, Pass 1 FP suppression and abstention;
- pooled and per-fold values, with stratified paired bootstrap intervals on the pre/post differences.

**Attribution.** The correction changes several factors at once:

- the GLASS selector (union obligations, the row objective);
- the shared contract, which replaces `match_glass_oof` for RF;
- the F17 populations (`oof_remainder` for GLASS, `nested_t1` for RF);
- the nested rankers.

The pre/post difference is therefore reported as a **combined correction**. Only S1 isolates one factor: the GLASS population policy, with everything else held constant. No other component is attributed unless a single-factor run is declared in a new protocol version before results. A later complementarity-ablation difference is reported separately, against the corrected primary.

**Known asymmetries, stated with every pre/post table:**

- Pre GLASS fitted Pass 2 on `full`; pre RF used `match_glass_oof` constraints and `global_t1`.
- The common eligible population changes when either Pass 1 changes, so matched-budget populations differ before and after.
- The historical GLASS confidence is in-sample rule precision.

## 9. Uncertainty and fold variation

**Procedure** (`stratified_paired_bootstrap`; settings from the record, D-C3):

- Resample rows with replacement inside every (outer fold × label) stratum, keeping stratum sizes.
- 2,000 replicates, seed 42, numpy `default_rng` (PCG64).
- Both routings are scored on the same resample, so a difference carries only its own uncertainty.
- Percentile 95% interval, numpy `linear` quantiles.
- Point values are the exact counts.
- Fitted routers and reference flags are held fixed, so the interval reflects evaluation-row sampling, not refitting variability.

**Undefined replicates** (an empty denominator in a resample) are counted. Any such replicate makes the interval undefined, reported with the count, rather than computed over a biased subset.

**Registered runs.** `run_registered_bootstrap(paired, metrics, protocol, reference=)` takes its settings from the protocol, never from the caller. It returns a `BootstrapRun` holding:

- the rows and the exact per-fold point values;
- a provenance: procedure identity `pr38.stage2.stratified_paired_bootstrap/1` (also registered in the record), protocol id, version and fingerprint, settings, pairing kind and names, and the digests of the paired data and the reference flags;
- a SHA-256 over all of it, so any later edit is detectable (`verify`).

A change to the algorithm needs a new procedure identity and a new protocol version.

**Malformed or incomplete runs.** A reported run fails C6 with its reason when it is malformed or incomplete:

- no provenance, or a provenance that lacks a recorded entry (procedure identity, settings, protocol fingerprint, pairing, metrics, data or reference digest);
- rows that are not one mapping per metric naming its metric, duplicated rows, or listed metrics without rows;
- missing or malformed per-fold points;
- a digest that is not a SHA-256 hex digest.

An error raised while comparing a run with the registered run also fails C6, with the error named. `evaluate_adoption` never raises on a reported run.

**Reference flags** are passed as `ReferenceFlags`: row-indexed, built from the frozen `UpstreamFlags` (`from_upstream`). They are accepted only when their rows, order and labels are the pairing's own. A bare array, a reordered mask or relabelled rows are refused. The contribution point values equal `router_upstream_metrics` on the same trace (tested).

**Why not reuse `rf_router.evaluation.paired.paired_bootstrap`.** It lives in the RF arm, so arm-neutral code may not import it. It resamples without strata. It also scores leakage and recall as 0.0 when there are no positives, which conflicts with the registry's undefined convention.

**Single proportions.** A Wilson score 95% interval (`wilson_interval`, which agrees with `feature_research.metrics.wilson_ci` and is undefined at n = 0) is descriptive. It is never an interval on a paired difference.

**Folds.** Five outer folds are not five independent replications: they share training data. Report per-fold values, minimum, maximum and range. No fold-level standard error or test.

**Compliance.** No interval establishes hard contract compliance, and none turns an observed violation into compliance.

**What the intervals are not.**

- They are conditional on the fitted routers and the frozen upstream flags. They describe evaluation-row sampling only.
- They do not account for refitting variability, for the earlier model-selection, feature-engineering, hyperparameter and contract decisions (section 15), or for the number of outcomes reported.
- No multiplicity adjustment is pre-registered. With nine primary outcomes, some intervals can be expected to exclude 0 by chance.

## 10. Fit status, infeasibility and solver reporting

**Completeness first.** A fold's status comes from its keyed selection records (`SelectionRecord(key, FitOutcome)`). The required keys follow from the F17 primary settings (`required_selection_keys`):

| Arm | Required selections per outer fold | Source of the inner records |
|---|---|---|
| GLASS | `pass1`, `pass2`, `inner_pass1[0..4]` (`pass1_oof_folds = 5`) | `GlassRouterPipeline.selection_reports()` (Step 4B) |
| RF | `pass1`, `pass2`, `eligibility_t1[0..4]`: one nested `t1_j` solve per Pass 1 OOF fold (`n_oof_folds = 5`; `rf_router.oof.eligibility`) | The RF fit report's eligibility records |

`check_selection_completeness` reports four kinds of problem:

- **missing:** a required key has no record, or (with GLASS `selection_reports`) no report behind it;
- **duplicated:** a key appears more than once;
- **unexpected:** an unknown kind, an inner index out of range, or the other arm's inner kind;
- **unsupported:** a record that is not a `FitOutcome`, an outcome whose scope does not match its key, or a record with no selection report behind it.

**GLASS report evidence (D-C11).** The frozen protocol requires the GLASS selection reports (`fit_status.glass_selection_reports = "required"`). A GLASS fold without them is unsupported. With them, the reports must correspond exactly to the required selections, slot by slot with `None` entries counted: a `None` in a required slot is missing, a slot for a selection that is not required (`inner_pass1[5]`, a trailing `None` included) is unexpected, and one report object standing for two selections is duplicated. Each record must be its report's classification: the outcome its `source` and ILP guarantee map to (as `glass_fit_outcome` does) and the same selected rule ids. The RF Router has no equivalent report; its evidence is its selection records (a documented asymmetry).

Any problem makes the fold `fit_time_invalid`. An accepted fold status therefore cannot be manufactured from a subset of its required fits. Reviews of a greedy fallback are recorded per selection key. The Step 4B solver semantics and the outcome categories are unchanged.

Then each selection's status is classified, and the most severe wins:

| Fit status | From | Accepted? | Treatment |
|---|---|---|---|
| `fit_time_invalid` | No fit record; a feasible claim without a satisfied fit-time report | No | Stop; investigate |
| `proven_infeasible` | Proven-infeasible ILP or empty selected union; no feasible RF threshold | No | Stop (M4) |
| `no_feasible_solution` | Time limit with no validated incumbent; greedy that cannot meet obligations | No | Stop (M4) |
| `insufficient_population_or_support` | Single-class or under-500 remainder; a pass below its support floor | No | Stop (M4) |
| `fallback_unreviewed` | Validated greedy fallback with no review record | No | Review before acceptance |
| `valid_unproven_undocumented` | Validated incumbent missing status, gap or a `reproduced=True` rerun | No | Document (CBC log gap; `glass_record_reproduction`) |
| `fallback_accepted_after_review` | Greedy fallback with a recorded reviewer and rationale (D-C7) | Yes | Reported as a fallback in every table |
| `valid_unproven_documented` | Status, gap and a reproduction recorded | Yes | Reported with its gap |
| `valid_optimal` | Proven optimum (ILP) or exhaustive scan (RF) | Yes | |

- A missing CBC bound or gap is recorded as `None`, never estimated.
- An incumbent whose reproduction does not match stays undocumented.
- Fold-level solver status, gap, fallback incidence and wall time are tabulated per arm.

**Fold ledger** (`FoldLedger`):

- A fold is accepted only through its `FoldFit` from `classify_fold_fit`. The `FoldFit` carries the evidence it was classified from: the selection records, the greedy-fallback reviews and, for GLASS, the selection reports.
  - An accepted `FoldFit` is refused at construction unless its completeness report is complete, its records are exactly the required selections and its status is the most severe selection status.
  - `FoldLedger.build` re-classifies every `FoldFit` under the ledger's protocol and refuses one that does not reproduce. This catches a relabelled status, a removed review, and a fit classified under other F17 settings.
- A fit-status string can record a fold as not accepted, for example a fit that stopped before it produced records. It can never record a fold as accepted, however the ledger is constructed. The summary lists such folds as `status_only`.
- Report the attempted folds (0–4), those reported, those whose fit completed, and those accepted, per arm, with every non-accepted fold's status.
- The primary comparison requires every attempted fold accepted in both arms (D-C6).
- `pooled_observation` refuses a pool over fewer folds unless it is asked for explicitly. That pool is then labelled "accepted folds only (k of 5)", is never the primary comparison, and still lists every fold.
- An unsuccessful fold is never discarded. Any non-accepted fold stops corrected evaluation for investigation (M4) before outer-OOF results are inspected.
- No arm receives easier constraints because it cannot meet the contract. Its failure is an outcome.

**Fit-only preflight (G3 beside G13; M4).** Run before any five-fold run, on one pre-declared outer fold (fold 0), with real data, for both arms.

- **Recorded:** every fit outcome, remainder size and class support, selection reports (inner folds included), solver logs and wall time.
- **Not computed or viewed:** fold 0's outer-OOF metrics. The preflight is fit-only.
- **On infeasibility:** stop and investigate. Any revision needs a new contract version, preserved provenance, approval and disclosure before corrected outer-OOF results are inspected.

## 11. Outer-OOF shortfalls

**Two axes, never merged.**

- **Fit status** (section 10): whether the fold's fit is valid evidence.
- **Observation:** each obligation per fold and pooled, from `obligations.observe_routing`: `satisfied`, `violated` or `undefined`, with the exact signed shortfall and its counts.

`classify_observation` keeps the observed status and adds:

| Classification | Meaning |
|---|---|
| The fit status itself | The fold's fit is not accepted; the observation is shown but is not outer-OOF evidence of a valid fit |
| `undefined_evaluation` | Empty denominator |
| `valid_fit_outer_oof_shortfall` | Accepted fit; observed violation |
| `valid_fit_observed_satisfied` | Accepted fit; observed within the limit (a shortfall of exactly 0 is satisfied) |

Flags carried alongside the classification:

- **`low_support`:** the denominator is below 30, the baseline's `min_support` (D-C4). Reported, never compliant by default.
- **`material`:** violations only (D-C2, a descriptive heuristic approved at the freeze; M2). The rule is `shortfall > 2·sqrt(L(1 − L)/n)`, where L is the limit and n the metric's denominator.

**Why 2 SE (ex ante, not from data).** It is a descriptive escalation heuristic for review, not a formal compliance test. Both arms select at the edge of their feasible sets, so an outer-OOF observation of a correctly enforced operating point varies around the limit with a standard error of about sqrt(L(1 − L)/n). That gives a violation in roughly half of all checks without any optimism (record, section 7). A shortfall beyond 2 SE would arise by chance in about 2–3% of checks for a point exactly at the limit. A shortfall that large is better explained by fit-time optimism (GLASS in-sample estimates, RF inner-OOF estimates) or by a difference between the fit and evaluation populations. Illustrative tolerances:

| Limit | n | 2 SE |
|---|---|---|
| NPV 0.95 | 2,000 Pass 1 rows | 0.0097 |
| Leakage 0.15 | 742 fold positives | 0.026 |
| Leakage 0.15 | 3,712 pooled positives | 0.012 |
| Precision 0.30 | 200 Pass 2 rows | 0.065 |

**Actions.**

- **Any violation:** reported as a violation.
- **A material violation in a fold:** a recorded review note on that fold. No refit, no band re-estimation.
- **A material pooled violation:** a primary finding: the arm does not meet the contract out of sample.

None of these revises the contract for PR 38 (M4), re-estimates bands or thresholds (F31 reopening condition 4 stays closed), or relabels the observation. A later, versioned contract may cite the finding as Tier 2 evidence, with that disclosed.

## 12. F31 boundary review (R2, R3)

- **Measure:** for every RF outer-fold router under `nested_t1`, strip A rows (`t1_j ≤ p1_oof < t1`) inside its flagged outer-OOF Pass 2 band, divided by the rows of that band. Strip A and B rows, positives and prevalence are reported too.
- **Trigger:** strip A ≥ 1/20 of any fold's flagged band (`f31_strip_review`, exact) is a mandatory F31 review. It is not an acceptance criterion, and a value below 1/20 is not evidence that the confidence is valid.
- **Empty band:** undefined; reported and reviewed, since R1 requires a supported band.
- **R3 optimism gap:** band NPV and precision against realised outer-OOF NPV and precision, per fold and pooled. Always reported, never used to tune.
- **Confidence semantics are unchanged** (Disposition A): the RF confidence is a heuristic blend, not a calibrated probability.

## 13. Complementarity-aware selector: secondary ablation

- **Role:** an exploratory ablation. It never replaces the primary corrected selector because its OOF results look better. It is not implemented in PR 37.
- **Design:** the primary corrected GLASS against the complementarity-aware GLASS ablation, with the same contract, populations, frozen upstream points, matched-budget procedure, folds, seeds and solver limits. Complexity (rules), runtime and solver outcomes are reported per fold.
- **RF:** adoption requires an RF objective with the same complementarity term, or a documented methodological asymmetry (C9).

**Adoption criteria** (with their margins, D-C5, approved at the freeze; `evaluate_adoption`). The criteria were fixed before any corrected or ablation result exists, and not to favour the ablation. Differences are ablation − primary, pooled; bounds are from section 9. **All criteria must pass**; one that fails or cannot be evaluated keeps the primary.

| | Criterion | Margin | Rationale |
|---|---|---|---|
| C1 | More unique positives at a controlled budget | Lower bound of the unique-TP difference > 0; gain ≥ 1/20 of the primary's unique TP; ablation Pass 2 rows ≤ primary's, pooled over all attempted folds (exact counts, no tolerance) | A gain must exceed sampling noise and be large enough to justify an extra objective term. It may not be bought with more flags |
| C2 | Pass 1 leakage | Pass-2-only ablation: Pass 1 rules identical in every fold (as G6). Otherwise the upper bound of the leakage difference ≤ 1/200 (0.5 pp of positives, about 19) | Pass 1 has precedence. A Pass 2 change should leave it untouched |
| C3 | Obligations | Fit-time compliance in every fold; no fold with a material outer-OOF violation the primary does not have | The contract is not traded for coverage |
| C4 | Routed coverage | Lower bound of the total-coverage difference ≥ −1/100 (1 pp, about 330 rows) | No material loss of routed rows |
| C5 | Fold reproducibility | Unique-TP gain ≥ 0 in at least 4 of 5 folds, counting only folds whose own Pass 2 budget did not grow (ablation Pass 2 rows ≤ primary's in that fold) | A pooled gain driven by one fold, or by budget moved between folds, is not reproducible |
| C6 | Uncertainty | The reported uncertainty is a `BootstrapRun` that reproduces exactly: intact digest, registered procedure identity and settings, this protocol's fingerprint, the ablation-vs-primary pairing, the same data and reference digests, the adoption metrics with defined intervals. A malformed or incomplete run fails (section 9) | Demonstrable use of the registered procedure, not interval fields supplied by hand |
| C7 | Solver | No fold with a worse fit status than the primary; wall time ≤ 3× the primary's | Five-fold reruns must stay feasible and stable |
| C8 | Class-conditional routing | Upper bound of the Pass 2 negative-routing-rate difference ≤ 1/200; lower bound of the Pass 1 negative-recall difference ≥ −1/100 | No hidden shift of errors between classes |
| C9 | Complexity | Added rules, runtime and code documented; RF symmetry resolved | The added complexity must be justified on the record |

**How the evaluation is done.** `evaluate_adoption(evidence, protocol, paired=PairedRouting.ablation(primary, ablation), upstream=UpstreamSources.from_views(lr_view, ebm_view))`:

- **Reference.** It is never taken from the caller. It is the record's `complementarity.reference` (`or`: the frozen LR-or-EBM union of O4a), taken from `upstream` by `verify_reference` only after `check_upstream_identity` passes on the paired rows. Moved union flags, a wrong view, a changed artifact file or a wrong pinned fingerprint stop the evaluation.
- **Official decisions.** A decision is official only under the pinned frozen protocol itself (`is_pinned_frozen`: the record `load_frozen_protocol` returns) with a complete upstream identity, the row-aligned flag fingerprints included. A frozen record built any other way exercises the gates but never decides officially. Under the working protocol (`allow_working`) pinned values may be pending (D-C10), and the decision is marked not official, with the pending checks listed in its provenance.
- **Recomputation.** Every interval, count, per-fold gain and per-fold Pass 2 budget it uses is recomputed with the registered bootstrap. None is taken from the caller.

The caller supplies only facts the outer-OOF data cannot show:

- fit-time compliance;
- Pass 1 rule identity, which is also checked against the routings row by row;
- material violations, fit statuses and runtime;
- complexity documentation and RF symmetry.

**Pass 2 flag budget: pooled or per fold (review).** C1's budget rule exists so that a unique-TP gain cannot be bought with more Pass 2 flags. The question is where the budget is held.

- **Pooled only** (the first proposal) guards the pooled C1 claim. It cannot see budget moved between folds: a fold that routes more rows, offset by another that routes fewer, keeps the pooled count. Per-fold unique-TP yield differs between folds, so such a move can raise the pooled gain without any better selection. C5 then counts the inflated fold's gain as reproduction.
- **Every fold strictly** removes that, but it turns outer-OOF sampling noise of a few rows in one fold into a failed adoption. It would be an implicit 5-of-5 rule, stricter than C5's 4 of 5 for the same per-fold evidence.
- **A per-fold tolerance** would be a new, unregistered margin.
- **Budget matching** (truncating the ablation's Pass 2 to the primary's per-fold count) needs Pass 2 scores, which GLASS rules do not have. It would also make D-C8-like persistence a prerequisite.

**Recommended pre-registered policy (D-C5), implemented:**

1. **Pooled (C1):** ablation Pass 2 rows ≤ primary Pass 2 rows over all attempted folds. These are exact outer-OOF counts, with no tolerance and no interval: the budget is a property of the realised routing, not an estimate.
2. **Per fold (C5):** a fold counts toward the 4 of 5 only if its unique-TP gain is ≥ 0 **and** its ablation Pass 2 rows are ≤ its primary Pass 2 rows. This reuses C5's existing tolerance (one fold may fail) and adds no new margin. Budget moved into two or more folds therefore fails C5 even when the pooled count is unchanged. The record sets this with `fold_counts_only_without_pass2_budget_inflation: true`.
3. **Reporting:** every fold's Pass 2 rows for both selectors are reported. The decision's reasons list the pooled budgets, the folds with a non-negative gain and the folds whose budget grew.

**Residual:** one fold may still grow its budget, offset elsewhere, and contribute to the pooled C1 gain. It is reported and not counted in C5. Removing it entirely would need the strict every-fold rule, which the freeze considered and did not choose (D-C5).

**Evidence label:** exploratory. No Tier 3 data exists, so adoption is never presented as independent validation.

**The margins are research and engineering judgements.** They are not empirically established thresholds and guarantee nothing. They were set before any corrected or ablation result and approved under D-C5 at the freeze.

**Effect:** an adopted ablation becomes the primary only under a later, versioned protocol. PR 38's primary results are not replaced. Only Tier 1 and Tier 2 sources may inform adoption: `evaluate_adoption` refuses test-derived and previously exposed sources.

## 14. Artifact acceptance

**A. Structural and methodological validity** (`structural_acceptance`; every gate must pass with recorded evidence; frozen protocol only):

| Gate | Checklist |
|---|---|
| V1 | Split, row identities, fold assignments, fingerprints (section 2) |
| V2 | F17 settings; contract adapters; bound contract; RF `constraint_source="shared_contract"` with the fingerprint |
| V3 | Union and threshold constraints met at fit time; accepted fit statuses; solver and fallback provenance complete |
| V4 | Arm ownership and estimator isolation; G4 on real data; G13 audit resolved; no test-label influence |
| V5 | G5 persistence; R1 band records; R2 strip records; R5 eligibility provenance; selection reports and solver logs |
| V6 | G7 reproduction of at least one fold |
| V7 | Valid serialization and reload; new versioned artifact names; Stage 4 handoff and interface checks (R4); `artifact_audit.py` change recorded (G10); baseline `--verify` (G11) |
| V8 | F31 Disposition A preserved; R7 relabelled diagnostic; R8 bands from the G13-passing state |

**B. Comparative empirical performance** (`comparative_report_gaps`; completeness only):

- primary outcomes; secondary diagnostics; fold variation; shortfalls;
- unique contribution; matched budgets; pre/post;
- uncertainty; fold ledger; solver outcomes;
- declared sensitivities (S1) and ablations.

Passing A is not evidence of better predictive performance. A disappointing B result does not make an artifact structurally invalid. The two are reported separately, and `structural_acceptance` takes no performance input.

## 15. Historical exposure and evidence tiers

| Tier | Definition | Members |
|---|---|---|
| Tier 1 | Exploratory historical OOF evidence that informed research decisions or the contract | `pr37_stage2_pre_correction_v1`; historical OOF of PRs 33–36 |
| Tier 2 | The corrected nested cross-fitted outer-OOF evaluation, after the contract and protocol freeze | `pr38_corrected_outer_oof` |
| Tier 3 | Data never used for model selection, feature engineering, contract revision, thresholds or protocol decisions | None available |
| Previously exposed | The Bank Marketing held-out test split, evaluated in PRs 33–36 (artifacts store test metrics and test-oracle thresholds) | `bank_marketing_test_split`; never Tier 3 |

**What PR 38 can claim without new external data:** under a contract and protocol frozen before the corrected results, with nested cross-fitting, the arms achieved the reported Tier 2 outcomes on the training population's outer folds. Those outcomes remain conditional on earlier exposure: hyperparameters, features and contract values were chosen with historical OOF and test results visible.

**What it cannot claim:**

- independent or external validation;
- that corrected cross-fitting removes earlier research-selection bias;
- that test-split numbers are untouched confirmation.

**Prohibited:** test-derived results informing selector choice, contract revision, threshold choice or adoption.

**Provenance checks the PR 38 runner must perform** to exclude test-derived data:

1. Read routers and upstream models only through the boundary extractors. They read allowlisted training-side fields and refuse forbidden tokens (`FORBIDDEN_FIELD_TOKENS`), and their tests prove that test-side fields are never read (`trapped`).
2. Open every artifact with `expected_sha256`: the baseline sources come from the SHA256SUMS snapshot, new PR 38 artifacts from their versioned manifests.
3. Build the `TrainReference` from the global split's training rows. The test index is used only for the row-only split fingerprint, and the test labels never at all.
4. Run V1 (`check_population_identity`, including the recomputed label fingerprint) and the upstream identity (section 5).
5. Bind reference flags to the pairing (`ReferenceFlags`), and pairs to row identity (`PairedRouting`). For adoption, take the reference only from `verify_reference` (section 13), never as flags supplied by the caller.
6. Take fit-time evidence only from selection reports, solver logs and fold records produced inside the fits.
7. Register every evidence source used in a decision (`evidence_tier`).

The runner loads no test split, test prediction or test metric.

**The name guard is defense in depth only.** `check_inputs` refuses input names with a `test`, `holdout` or `oracle` segment. A name says nothing about where data came from, and a renamed test-derived value would pass it. Provenance rests on checks 1–7.

## 16. Governance

- **Owner:** this record and its JSON. **Frozen in Step 4D:**
  - `status` is `frozen` under `1.0.0`;
  - bound to the frozen contract (D-C9) and both frozen arm profiles;
  - every pending approval resolved (section 17) and the owner's approval recorded in the record's `freeze` block (Phillip Harris, 2026-11-05);
  - the fingerprint pinned in code (`FROZEN_PROTOCOL_FINGERPRINT`) and in the tests.
- After the freeze, any change, including a margin, a seed or a metric, needs a new version, a recorded rationale and approval before the results it would affect are inspected. Changes made after results are labelled exploratory.
- This protocol does not rewrite the historical F17, F31 or Commit 3 records.

**PR 38 runner obligations** (documented here, not built in PR 37), in order:

1. Re-verify the frozen specification on the artifacts (`tests/realdata/test_stage2_freeze_artifacts.py`); then `load_frozen_specification()` (the strictly loaded protocol and contract, bound). No record is taken from the caller.
2. `check_arm_profile_binding` for both strictly loaded arm profiles.
3. Build both configs with `glass_primary_config()` and `rf_primary_config()`; check every config used with `check_glass_primary_config` / `check_rf_primary_config`, and the GLASS ranker settings with `check_glass_primary_ranker` (Commit 5), and run the adapter checks (V2).
4. G13 and the fit-only preflight (section 10).
5. Five-fold runs.
6. `extract_router_view`, then `build_router_trace`.
7. `check_population_identity` and `check_upstream_identity`, with `require(complete=True)`.
8. Classify every fold from complete selection records (`classify_fold_fit`, with the GLASS `selection_reports()`: D-C11), then build the fold ledger from those `FoldFit`s. An accepted status is never given as a string.
9. Validity gates V1–V8.
10. The primary outcomes with `run_registered_bootstrap`.
11. Secondary diagnostics, matched budgets, pre/post, S1, F31 R2 and R3.
12. A comparative report with no gaps.
13. If the ablation is run: `evaluate_adoption` with the `UpstreamSources` of the frozen views. It is official only under the frozen protocol.

New results go under new versioned outputs that cite the baseline.

## 17. Decisions resolved at the freeze (Step 4D)

The record's `freeze.decisions` holds each decision, its resolution, its kind and its rationale. Registered criteria and procedures govern PR 38; descriptive heuristics and reporting conventions label results and never decide compliance.

| ID | Decision | Resolution | Kind |
|---|---|---|---|
| D-C1 | Statement rule for cross-arm differences (section 3.2) | Approved | Reporting rule |
| D-C2 | Materiality of an outer-OOF violation: 2 binomial SE at the limit (M2) | Approved | Descriptive heuristic |
| D-C3 | Bootstrap: 2,000 replicates, seed 42, fold × label strata, percentile 95% | Approved | Registered procedure |
| D-C4 | Low-support reporting threshold 30 (baseline `min_support`) | Approved | Reporting convention |
| D-C5 | Complementarity adoption criteria and margins; Pass 2 budget pooled in C1 and per fold in C5 (section 13) | Approved | Registered criterion |
| D-C6 | Primary comparison requires all five folds accepted in both arms | Approved | Registered criterion |
| D-C7 | A greedy fallback is accepted only with a review recorded at fit time | Approved | Registered criterion |
| D-C8 | Persist RF Pass 2 outer scores for an optional RF-truncation diagnostic | Not required | Optional diagnostic |
| D-C9 | Bind the frozen contract `1.0.0` (`0817825b9eecbd11`) and the frozen arm profiles | Approved | Binding |
| D-C10 | Pin the row-aligned flag fingerprints: LR `64d511516d69e73e`, EBM `907a7d2d1cad6908` | Approved; re-verified on the artifacts | Binding |
| D-C11 | GLASS fold acceptance requires exactly corresponding selection reports (section 10) | Approved | Registered procedure |

**Methodological review (the rationale recorded at the freeze).**

- **D-C1 (statement rule):** a per-outcome, descriptive rule.
  - Nine primary outcomes support no unqualified global superiority claim. A favourable interval on one outcome is a statement about that outcome only.
  - The intervals are not adjusted for multiplicity.
  - The lexicographic reading still applies: O2 never offsets O1, and O3 qualifies both.
- **D-C2 (2 SE):** a descriptive escalation heuristic, not a formal compliance test.
  - An observed violation stays a violation whatever its size.
  - The normal approximation is poor for small n and near the bounds (NPV 0.95), hence the low-support flag.
  - Across about 30 checks (3 obligations × 5 folds × 2 arms), roughly 0.7 material flags would be expected by chance even for operating points exactly at their limits.
- **D-C3 (bootstrap):**
  - 2,000 replicates make the Monte Carlo error of the percentile bounds small relative to their width.
  - The intervals are conditional on fixed fitted models. They do not account for earlier model-selection decisions or refitting variability, so they understate the uncertainty of rerunning the whole procedure.
- **D-C4 (30):** a reporting convention reused from the baseline, not a statistical threshold.
- **D-C5 (margins):** engineering and research judgements, not empirically established guarantees. The Pass 2 budget is held pooled in C1 and per fold in C5 (section 13). The strict every-fold rule was the alternative and was not chosen. The 5% relative gain, the budget non-inflation rule, the 0.5 pp and 1 pp margins, 4 of 5 folds and 3× runtime could each reasonably be set differently. They were approved at the freeze, before any ablation result existed.
- **D-C6 (all five folds):** strict by design. If one arm cannot fit a fold after M4 investigation, the primary comparison is unavailable and that failure is the reported outcome; it is not a reason to grant easier constraints.
- **D-C7 (greedy review):** record the review at fit time, before that fold's outer-OOF results are examined.
- **D-C8 (RF Pass 2 outer scores):** optional. The primary comparison and every gate are defined without it, and it must not become an unplanned prerequisite for PR 38. If it is not done, the RF-truncation diagnostic is simply not reported.
- **D-C9:** freezing the contract changed its fingerprint. The protocol binding, both arm profiles and the pinned fingerprints were frozen in the same Step 4D change, so PR 38 runs under one consistent specification.
- **D-C10:** under the working record, upstream identity rested on layers 1–4.
  - They catch count-preserving moves of the supplied flags, a changed artifact file and changed thresholds.
  - They do not catch a view edited after extraction so that recomputation agrees with it; only layer 5 does. The frozen record pins both fingerprints, and the real-data freeze test re-runs all five layers with `require(complete=True)`: a copied value alone is never the proof.
  - Adoption decisions can now be official, under the frozen protocol only.
- **D-C11:** a GLASS fit outcome is derived from its selection report, so without the report it cannot be verified. Reports that are missing, unexpected, duplicated or inconsistent with their records make the fold `fit_time_invalid`. The RF Router has no equivalent report: its evidence remains its selection records.

## 18. Limitations

- Tier 2 is still conditioned on earlier exposure (section 15).
- Intervals hold fits fixed (section 9).
- GLASS fit-time compliance is in-sample.
- Matched budgets are arm-specific (section 7).
- Pre/post differences are a combined correction (section 8).
- The 2 SE rule is an approximate escalation heuristic, not a compliance test, and is poor for small n. Hence the low-support flag.
- The upstream row fingerprints are pinned (D-C10); the freeze holds only while the real-data re-verification passes.
- GLASS fit evidence includes its selection reports; RF fit evidence is its selection records only (D-C11).
- The input-name guard is defense in depth only (section 15).
- Under the pooled-and-per-fold budget policy, one fold may grow its Pass 2 budget, offset elsewhere, and still contribute to the pooled C1 gain (section 13).
