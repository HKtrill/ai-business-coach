# Pre-Correction Stage 2 Baseline — `pr37_stage2_pre_correction_v1`

Format `pr37.stage2_router_baseline` v1 · designation **pre_correction** · status **frozen** · PR 37, Commit 2, Step 2E

This package is the frozen historical reference for the GLASS Router and the RF Router **before any corrected-selector change**. It records what the historical artifacts show on the outer out-of-fold outputs of the 32,950 training rows. PR 38's corrected-selector results are compared with it; they never replace it. A later baseline is a new version with a new identifier and directory: this package is never edited or regenerated in place, and later interpretation is added as a separate research log that cites it.

It is **not**: corrected-selector performance; a held-out or test-set evaluation; a measurement of Stage 3, Stage 4 or the integrated cascade; a ranking of the routers; or a significance test. Every number is a historical reference anchor, not a target to optimise.

## 1. Identity and provenance

| Item | Value | Status |
| --- | --- | --- |
| Training rows / positives / outer folds | 32,950 / 3,712 / 5 | validated |
| Rows per outer fold | 6,590, 6,590, 6,590, 6,590, 6,590 | validated |
| Router split fingerprint | `fc3e73a5ef91aea4` | validated |
| Training-label fingerprint | `b984d2a50527` | validated |
| Outer fold plan | seed 42, fingerprint `8aea082fcaf371b7` | validated |
| stage_io split fingerprint | `d6cddadc3439` | attested (never recomputed: its hash includes test labels) |
| Snapshot SHA256SUMS | pr36_regen_2026-10-23/SHA256SUMS.txt (37 entries), `e82c6a845b4754e6…` | attested |

Source artifacts (SHA-256 prefix; full values in `MANIFEST.json`):

| Role | File (relative to bank_pipeline/) | SHA-256 | Snapshot |
| --- | --- | --- | --- |
| stage4_arbiter.meta_ebm | `models/meta_ebm/meta_ebm_20261023_070055_573687.joblib` | `82a893e134e5a8ae…` | attested |
| stage4_arbiter.meta_ebm_on_blackbox | `blackbox_pipeline/models/stage4_crossover/arbiter_on_blackbox/meta_ebm_20261023_095719_949475.joblib` | `b4b78c589ec27743…` | attested |
| stage1.lr | `models/lr/lr_calibrated_20261023_003319.joblib` | `61a6b144b625e7a5…` | attested |
| stage2.glass_router | `models/glass_router/glass_router_20261023_025611.joblib` | `2c70879bfe04f2ad…` | attested |
| stage3.ebm | `models/ebm/ebm_stage3_20261023_070053_780762.joblib` | `cfdac6d206d73790…` | attested |
| stage2.rf_router | `models/rf_router/rf_router_two_pass.joblib` | `5204a2d6732352ab…` | attested |
| glass_arm.rf_result | `models/rf/rf_result_20261023_025611.joblib` | `76b198aa5eff376e…` | attested |

Diagnostic code identity: LF-normalised SHA-256 of 15 files (`MANIFEST.json` → `code`). The artifacts do not record the code that produced them; router identity is the artifacts' own record (`MANIFEST.json` → `routers`).

## 2. Frozen reference operating points and the RF constraint source

| Reference | Operating point | Threshold | Flags | TP | Precision |
| --- | --- | --- | ---: | ---: | ---: |
| `lr` | `lr.in_stage_f2_cv` | 0.1 | 7,727 | 2,421 | 2,421/7,727 = 31.33% |
| `ebm` | `ebm.nested_f2` | by fold: 0.56, 0.56, 0.54, 0.49, 0.49 | 6,113 | 2,307 | 2,307/6,113 = 37.74% |
| `union` | `(lr.in_stage_f2_cv or ebm.nested_f2)` | - | 8,001 | 2,477 | 2,477/8,001 = 30.96% |
| `intersection` | `(lr.in_stage_f2_cv and ebm.nested_f2)` | - | 5,839 | 2,251 | 2,251/5,839 = 38.55% |

LR's threshold was selected on all training labels (in-selection; its rates are optimistic). EBM's decisions are fold-nested: each fold's raw-score threshold was selected on inner OOF scores of that fold's training rows.

RF Router `constraint_source = 'match_glass_oof'`. Recorded constraints against the GLASS Router's pooled outer-OOF values in this baseline:

| Constraint | Recorded | GLASS realised | Status |
| --- | ---: | ---: | --- |
| `min_precision_not_subscribe` | 0.9553485943075986 | 10,976/11,489 | equal |
| `max_subscriber_leakage_rate` | 0.13820043103448276 | 513/3,712 | equal |
| `max_subscriber_leakage_absolute` | 513 | 513 | equal |
| `min_precision_subscribe` | 0.2759801606046292 | 2,337/8,468 | equal |

## 3. Principal results (outer OOF, pooled; per-fold values in `tables/`)

### 3.1 Router states

| State | GLASS rows (share) | GLASS positives (rate) | RF rows (share) | RF positives (rate) |
| --- | ---: | ---: | ---: | ---: |
| Pass 1 (NOT_SUBSCRIBE) | 11,489 (34.87%) | 513 (4.47%) | 13,074 (39.68%) | 529 (4.05%) |
| Pass 2 (SUBSCRIBE) | 8,468 (25.70%) | 2,337 (27.60%) | 9,427 (28.61%) | 2,560 (27.16%) |
| Abstain | 12,993 (39.43%) | 862 (6.63%) | 10,449 (31.71%) | 623 (5.96%) |
| Coverage (Pass 1 + Pass 2) | 19,957/32,950 = 60.57% |  | 22,501/32,950 = 68.29% |  |
| Abstention rate | 12,993/32,950 = 39.43% |  | 10,449/32,950 = 31.71% |  |

### 3.2 Pass 1 (NOT_SUBSCRIBE): NPV, leakage and suppression of the reference flags

|  | GLASS Router | RF Router |
| --- | ---: | ---: |
| Pass 1 NPV | 10,976/11,489 = 95.53% | 12,545/13,074 = 95.95% |
| Positive leakage (Pass 1 positives / all positives) | 513/3,712 = 13.82% | 529/3,712 = 14.25% |
| Union-flagged rows routed to Pass 1 | 644 | 14 |
| ... of which actual positives (suppressed TP) | 82 | 5 |
| Union FP suppression (intercepted FP / union FP) | 562/5,524 = 10.17% | 9/5,524 = 0.16% |
| Union TP cost (suppressed TP / union TP) | 82/2,477 = 3.31% | 5/2,477 = 0.20% |
| LR-flagged rows routed to Pass 1 | 586 | 5 |
| ... of which actual positives | 74 | 0 |
| EBM-flagged rows routed to Pass 1 | 250 | 9 |
| ... of which actual positives | 53 | 5 |

### 3.3 Pass 2 (SUBSCRIBE) against the frozen references (all rows)

|  | GLASS Router | RF Router |
| --- | ---: | ---: |
| Pass 2 rows | 8,468 | 9,427 |
| Pass 2 TP | 2,337 | 2,560 |
| Pass 2 FP | 6,131 | 6,867 |
| Pass 2 precision | 2,337/8,468 = 27.60% | 2,560/9,427 = 27.16% |
| Pass 2 recall (all positives) | 2,337/3,712 = 62.96% | 2,560/3,712 = 68.97% |
| Unique vs union: TP | 131 | 117 |
| Unique vs union: FP | 1,862 | 1,669 |
| Marginal precision vs union | 131/1,993 = 6.57% | 117/1,786 = 6.55% |
| Unique vs LR: TP | 152 | 158 |
| Unique vs EBM: TP | 248 | 268 |

### 3.4 Common eligible population and matched flag rate

Common eligible population: 16,189 rows (2,931 positives) that neither router routed to Pass 1. It excludes 74 GLASS Pass 2 rows that RF routed to Pass 1 and 813 RF Pass 2 rows that GLASS routed to Pass 1. Within each router fold, LR / EBM are re-cut to the arm's K highest-scoring rows (label-free; ties: ascending SHA-256 of the row's index label (label-free, fixed per row; not row order)).

| Router | Matched reference | K (= Pass 2 rows) | Pass 2 TP | Reference TP | Pass 2 − reference TP | Unique TP [tie bounds] | Tie block | Tie-broken rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GLASS Router | `lr_matched` | 8,394 | 2,329 | 2,409 | -80 | 85 [93 – 78] | 554 | 300 |
| GLASS Router | `ebm_matched` | 8,394 | 2,329 | 2,445 | -116 | 104 [104 – 104] | 5 | 0 |
| RF Router | `lr_matched` | 8,614 | 2,466 | 2,466 | 0 | 58 [78 – 43] | 605 | 303 |
| RF Router | `ebm_matched` | 8,614 | 2,466 | 2,494 | -28 | 76 [76 – 76] | 8 | 2 |

K differs between the arms, so each arm is compared with its own matched references; this is not an equal-count comparison of the two routers.

### 3.5 Agreement with the references (union of LR and EBM flags)

| Metric (numerator / denominator) | GLASS Router | RF Router |
| --- | ---: | ---: |
| `coverage` (rows routed Pass 1 or Pass 2 / all rows) | 19,957/32,950 = 60.57% | 22,501/32,950 = 68.29% |
| `pass1_upstream_flag_rate` (Pass 1 rows the upstream flags / Pass 1 rows) | 644/11,489 = 5.61% | 14/13,074 = 0.11% |
| `pass2_upstream_redundancy` (Pass 2 rows the upstream also flags / Pass 2 rows) | 6,475/8,468 = 76.46% | 7,641/9,427 = 81.05% |
| `pass2_upstream_disagreement` (Pass 2 rows the upstream does not flag / Pass 2 rows) | 1,993/8,468 = 23.54% | 1,786/9,427 = 18.95% |
| `covered_agreement` (covered rows where (Pass 2) == (upstream flag) / covered rows) | 17,320/19,957 = 86.79% | 20,701/22,501 = 92.00% |
| `abstain_upstream_flag_rate` (abstained rows the upstream flags / abstained rows) | 882/12,993 = 6.79% | 346/10,449 = 3.31% |
| `upstream_flags_in_pass1_share` (upstream flags routed by Pass 1 / upstream flags) | 644/8,001 = 8.05% | 14/8,001 = 0.17% |
| `upstream_flags_in_pass2_share` (upstream flags routed by Pass 2 / upstream flags) | 6,475/8,001 = 80.93% | 7,641/8,001 = 95.50% |
| `upstream_flags_in_abstain_share` (upstream flags on abstained rows / upstream flags) | 882/8,001 = 11.02% | 346/8,001 = 4.32% |

### 3.6 GLASS state × RF state (rows / actual positives)

|  | RF Pass 1 (NOT_SUBSCRIBE) | RF Pass 2 (SUBSCRIBE) | RF Abstain |
| --- | ---: | ---: | ---: |
| GLASS Pass 1 (NOT_SUBSCRIBE) | 7,802 / 261 | 813 / 94 | 2,874 / 158 |
| GLASS Pass 2 (SUBSCRIBE) | 74 / 8 | 7,423 / 2,267 | 971 / 62 |
| GLASS Abstain | 5,198 / 260 | 1,191 / 199 | 6,604 / 403 |

| Metric | Numerator / denominator | Value |
| --- | --- | ---: |
| `cross_state_agreement` | rows with the same GLASS and RF state / rows | 21,829/32,950 = 66.25% |
| `cross_both_routed_rate` | rows both routers route / rows | 16,112/32,950 = 48.90% |
| `cross_routed_agreement` | rows both routers route to the same pass / rows both routers route (routed only) | 15,225/16,112 = 94.49% |
| `cross_opposite_rate` | rows one router routes Pass 1 and the other Pass 2 / rows | 887/32,950 = 2.69% |
| `cross_one_abstains_rate` | rows exactly one router abstains on / rows | 10,234/32,950 = 31.06% |
| `cross_both_abstain_rate` | rows both routers abstain on / rows | 6,604/32,950 = 20.04% |

### 3.7 Cross-router Pass 2 discoveries (unique relative to the LR / EBM union)

|  | Both routers | GLASS only | RF only |
| --- | ---: | ---: | ---: |
| Actual positives | 74 | 57 | 43 |
| Actual negatives | 1,002 | 860 | 667 |

GLASS-only positives that RF routed to Pass 1 / abstained on: 8 / 49; RF-only positives that GLASS routed to Pass 1 / abstained on: 16 / 27.

### 3.8 Observed vs expected positives (descriptive, not tests)

| Group | Rows | Observed | LR-calibrated | EBM-calibrated | EBM-decile stratified | Joint-grid stratified |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `glass_pass1` | 11,489 | 513 | 582.4 | 528.6 | 551.8 | 687.2 |
| `rf_pass1` | 13,074 | 529 | 560.7 | 514.1 | 530.5 | 539.9 |
| `glass_pass2` | 8,468 | 2,337 | 2,311.9 | 2,340.9 | 2,275.6 | 2,056.5 |
| `rf_pass2` | 9,427 | 2,560 | 2,522.7 | 2,578.6 | 2,539.4 | 2,412.3 |
| `glass_abstain` | 12,993 | 862 | 814.0 | 852.8 | 884.7 | 968.3 |
| `rf_abstain` | 10,449 | 623 | 624.8 | 629.5 | 642.1 | 759.7 |
| `glass_pass1_and_lr_flag` † | 586 | 74 | 111.3 | 75.0 | 84.9 | 134.1 |
| `glass_pass1_and_ebm_flag` † | 250 | 53 | 52.7 | 54.7 | 66.0 | 65.8 |
| `glass_pass1_and_union_flag` † | 644 | 82 | 115.1 | 85.0 | 98.6 | 145.1 |
| `rf_pass1_and_union_flag` † | 14 | 5 | 1.2 | 2.3 | 2.6 | 0.9 |
| `glass_pass2_unique_vs_lr` † | 2,092 | 152 | 155.0 | 149.7 | 151.2 | 293.4 |
| `rf_pass2_unique_vs_lr` † | 1,989 | 158 | 153.0 | 169.7 | 173.4 | 353.8 |
| `glass_pass2_unique_vs_union` † | 1,993 | 131 | 147.1 | 125.5 | 128.8 | 267.5 |
| `rf_pass2_unique_vs_union` † | 1,786 | 117 | 137.2 | 120.6 | 123.6 | 303.7 |
| `glass_pass1_and_rf_pass2` | 813 | 94 | 119.6 | 96.4 | 108.0 | 181.1 |
| `glass_pass2_and_rf_pass1` | 74 | 8 | 4.2 | 4.0 | 4.3 | 4.8 |
| `glass_pass1_and_rf_abstain` | 2,874 | 158 | 178.6 | 160.5 | 164.1 | 222.2 |
| `glass_abstain_and_rf_pass1` | 5,198 | 260 | 272.4 | 238.4 | 246.6 | 251.2 |
| `glass_pass2_and_rf_abstain` | 971 | 62 | 77.4 | 61.0 | 64.5 | 115.2 |
| `glass_abstain_and_rf_pass2` | 1,191 | 199 | 172.8 | 206.4 | 224.6 | 294.7 |
| `both_abstain` | 6,604 | 403 | 368.8 | 408.0 | 413.5 | 422.4 |

† Defined by the reference thresholds: the joint-grid expectation is not interpretable for these groups (section 5).

## 4. Step 2D findings (validated summary)

- Coverage: GLASS 19,957/32,950 = 60.57%; RF 22,501/32,950 = 68.29%.
- Same three-state decision: 21,829/32,950 = 66.25%. Agreement when both route (routed only): 15,225/16,112 = 94.49%. Exactly one router abstains: 10,234/32,950 = 31.06%. Opposite routing decisions: 887/32,950 = 2.69%.
- GLASS Pass 1 / RF Pass 2: 813 rows, 94 actual positives. GLASS Pass 2 / RF Pass 1: 74 rows, 8 actual positives.
- Pass 1 interceptions of the LR / EBM union flags: GLASS 644 rows (82 actual positives); RF 14 rows (5 actual positives).
- Pass 2 positives the union misses: GLASS 131, RF 117; of these 74 are found by both routers, 57 by GLASS only and 43 by RF only, with 1,002 / 860 / 667 accompanying negatives.
- Router states against EBM's calibrated expectation: the largest difference is -18.6 positives (`rf_pass2`); every state is within 2.9% of its expectation.

Interpretation (as of this version): the routers show different functional behaviour, especially in abstention and deferral rather than direct contradiction. Useful incremental predictive information beyond the LR and EBM references, and any Stage 4 benefit, remain unproven. Calibrated EBM expectations explain several observed disagreement populations reasonably well. The larger deviations in the coarse joint LR × EBM strata do not establish predictive independence, because within-bin score variation remains possible (and, for threshold-defined groups, the grid is not applicable: section 5). Descriptive expectations, chance arithmetic and empirical stratification are not statistical significance tests. This baseline describes the historical routers; it is not, and does not predict, corrected-selector performance.

## 5. Known limitations of the exported Step 2D tables

- **D17** Known limitation (identified after Step 2D; definitions unchanged): the 3 x 3 joint grid splits each score at within-fold tertiles, so its top cell contains rows on both sides of both reference thresholds. For populations defined by those thresholds (reference-flag groups and 'unique vs' groups), expected_stratified_lr_x_ebm compares them with rows across the threshold and is not interpretable; use the calibrated expectations. Decile expectations share the bias to a smaller degree.
- **D18** Known limitation (identified after Step 2D; definitions unchanged): pooled ('all') stratum rows sum bin k of every fold, and where LR isotonic ties merge decile edges the bin indices shift, so pooled LR bin k covers different score ranges in different folds (LR occupied bins by fold: fold 0: 9, fold 1: 8, fold 2: 8, fold 3: 10, fold 4: 9; merged in folds [0, 1, 2, 4]). Pooled LR strata and LR-conditioned pooled contrasts are approximate; per-fold cells and all expectations are unaffected; the EBM strata are aligned.
- Reference thresholds by within-fold percentile: LR 77.3, 76.3, 76.6, 75.3, 77.3; EBM 82.8, 83.1, 81.9, 79.3, 80.1 (folds 0-4); the joint grid's edges are the within-fold 1/3 and 2/3 quantiles.
- Low-support contrasts (either state under 30 rows): 534, left empty in `tables/stratum_contrasts.csv`.

## 6. Definitions

- **Router states.** Each training row has exactly one state per router: Pass 1 (NOT_SUBSCRIBE), Pass 2 (SUBSCRIBE) or Abstain. Pass 1 claims first; Pass 2 only claims rows Pass 1 left.
- **Coverage** = (Pass 1 + Pass 2 rows) / all rows. **Pass 1 NPV** = Pass 1 negatives / Pass 1 rows. **Positive leakage** = Pass 1 positives / all positives. **Pass 2 precision** = Pass 2 positives / Pass 2 rows.
- **Suppression.** An intercepted reference flag is a flagged row routed to Pass 1: an intercepted false positive is a potential benefit (pass1_fp_suppression = intercepted FP / reference FP); a suppressed true positive is an error (pass1_tp_cost = suppressed TP / reference TP). They are never netted.
- **Unique positives** are Pass 2 positives the stated reference does not flag. The hypothetical OR (Pass 2 or the reference) is not Stage 4's behaviour.
- **Denominators.** Rates over all rows of the fold / pool unless the label says otherwise; 'routed only' rates exclude abstained rows. Undefined ratios are left empty with status undefined_empty_denominator, never 0.
- **Folds.** 'all' rows pool the five router outer folds; per-fold values are descriptive, not independent replicates.
- Every metric, table and column: `definitions.json`.

## 7. Research disclosures

- **D01** Metric formulas and denominators: every ratio has its numerator, denominator, value and status in the table that reports it; ratio names are defined in metrics.METRIC_DEFINITIONS (definitions.json 'metrics'); wide-table columns are defined in definitions.json 'tables'. Undefined ratios stay undefined (empty value, status undefined_empty_denominator), never 0.
- **D02** Pass 1 precedence and Pass 2 eligibility: Pass 1 claims rows first; Pass 2 only claims rows Pass 1 left; everything else abstains. Final Pass 1 and Pass 2 never overlap by construction. For the RF Router this is verified from the artifacts (Pass 1 score equivalence); for GLASS it is code-level only (no per-fold rule sets).
- **D03** Positive leakage: Pass 1 actual positives (routed NOT_SUBSCRIBE) / all positives (pass1_leakage_share); within-band leakage is Pass 1 positives / Pass 1 rows (pass1_positive_rate); Pass 1 NPV is Pass 1 negatives / Pass 1 rows.
- **D04** Frozen reference flags: lr.in_stage_f2_cv (LR calibrated probability >= its stored threshold, selected on ALL training labels: in-selection, rates optimistic) and ebm.nested_f2 (stored fold-nested EBM raw-score decisions). Both come from the GLASS arm and are the common reference for both routers; the RF Router's own black-box cascade uses other upstream models.
- **D05** Overall (frozen) comparisons keep each side's own flag rate on all rows; matched-rate comparisons re-cut LR / EBM at each arm's Pass 2 count K per fold on the common eligible population, label-free. K is the router's observed count (a descriptive control, not a pre-registration); the two arms have different K.
- **D06** EBM raw scores are never pooled or ranked across folds (their fold-specific nested thresholds show the scales differ): EBM strata, ranks and threshold positions are within the router outer fold.
- **D07** LR calibrated probabilities are isotonic, so large tied blocks occur. Strata keep tied scores in one bin (merged edges). Matched selections break ties at the K-th score by a predetermined label-free rule (ascending SHA-256 of the row's index label (label-free, fixed per row; not row order)) and report strict / inclusive bounds.
- **D08** Two kinds of expected positives: calibrated model expectations (sums of LR or EBM calibrated OOF probabilities; assume the calibration is right) and internal standardisation (expected_stratified_*: the group's rows' (fold, stratum) positive rates on the same labels). Neither is a null distribution or a test.
- **D09** Cells with fewer than 30 rows are flagged low_support and excluded from within-stratum contrasts (difference left empty).
- **D10** Final-state Pass 1 / Pass 2 overlap is available and zero by construction (a routing invariant, not evidence of independent signals). Pre-routing signal overlap is unavailable: GLASS candidate-rule matches and per-fold rule sets, and RF Pass 2 outer scores, were never persisted. The full-train GLASS rule set is not applied to training rows (in-sample).
- **D11** Shared component, code-level only: crossfit_stage2 defaults to glass_rf_model='fold_rf', passing each outer fold's RF to the GLASS fit (segment prioritisation by its feature_importances_); that fold forest's oof_proba is the RF Router's Pass 1 score. The artifacts' own record of glass_rf_model is not read.
- **D12** Pass 2 fit-time populations differ historically: GLASS 'full', RF 'oof_remainder' (F17, Commit 3). Pass 2 differences are not attributable to the selector alone.
- **D13** RF constraint_source='match_glass_oof' with recorded constraints {'min_precision_not_subscribe': 0.9553485943075986, 'max_subscriber_leakage_rate': 0.13820043103448276, 'max_subscriber_leakage_absolute': 513, 'min_pass1_coverage': None, 'min_precision_subscribe': 0.2759801606046292, 'min_recall_subscribe': None, 'recall_denominator': 'global', 'min_band_support_pass1': 125, 'min_band_support_pass2': 75, 'require_band_ordering': True}: derived from GLASS's realised OOF metrics (MANIFEST.json rf_constraint_provenance), so the arms are dependent and not exactly rate-matched. Commit 4 defines the successor; PR 38 applies it.
- **D14** The GLASS configuration is historical and not optimised; this is a baseline, not its performance potential.
- **D15** Historical test-set exposure: the held-out test split was evaluated in PRs 33-36, and the October 23 artifacts store test metrics and test-oracle thresholds. PR 37 diagnostics read only allowlisted training-side fields and prevent further test-derived influence; they do not make the test split untouched. No test label, test prediction or test-derived threshold is read or exported here.
- **D16** No claim of Stage 3 / Stage 4 or integrated-cascade benefit, no independent external validation, no ranking of the routers, no significance test. Fold values are descriptive, not independent replicates; no standard errors are given.
- **D19** The artifacts do not record the code that produced them. MANIFEST.json 'code' identifies the diagnostic and export code at generation (LF-normalised SHA-256), not the router implementations; router identity is the artifacts' own record (MANIFEST.json 'routers').

## 8. Files, verification and versioning

| File | Rows | Contents |
| --- | ---: | --- |
| `definitions.json` | - | metric registry, table / column definitions and conventions |
| `tables/states.csv` | 36 | Three-state counts and class composition per router and fold. |
| `tables/ratios.csv` | 6,159 | Every count and ratio of the long-format diagnostics, with explicit numerator, denominator, value and status. |
| `tables/policy_audit.csv` | 16 | Routing-policy claims per router: what the artifacts verify, what is code-level only, declared or not verifiable. |
| `tables/reference_flags.csv` | 4 | The frozen LR / EBM reference flags (and their union / intersection) used by every router-to-reference comparison. |
| `tables/pass2_breakdown.csv` | 12 | Pass 2 TP / FP split by which frozen references flag them. |
| `tables/common_population.csv` | 6 | The common eligible population (rows neither router routed to Pass 1) and what it excludes. |
| `tables/matched_selection.csv` | 20 | The label-free rate-matched reference selections: K, cut, tie block and tie-broken rows per arm, model and fold. |
| `tables/contingency.csv` | 288 | Router state x reference flag: rows, positives and negatives. |
| `tables/crosstab.csv` | 54 | GLASS state x RF state cells with label composition and reference flags. |
| `tables/pass2_cross_router.csv` | 30 | Pass 2 positives and negatives both routers / one router flag, overall and unique relative to each reference. |
| `tables/signal_availability.csv` | 18 | Which pre-routing signals the artifacts persist, and which overlap analyses are therefore unavailable. |
| `tables/score_conditional.csv` | 1,002 | Router state x label-free score stratum: label composition, calibrated expectations and reference flags. |
| `tables/stratum_contrasts.csv` | 1,002 | Positive-rate differences between router states within a stratum (defined only where both states have minimum support). |
| `tables/observed_expected.csv` | 126 | Observed vs expected positives for the router states and the named disagreement populations (descriptive; not tests). |
| `tables/chance_baselines.csv` | 96 | Arithmetic chance expectations for unique positives and Pass 1 interceptions (not statistical tests). |
| `tables/model_disagreement.csv` | 18 | How much LR and EBM already disagree (frozen, and at each arm's matched K), next to the routers' unique positives. |
| `tables/validation.csv` | 127 | Every trace check (T1-T7) and boundary extractor check, with its result. |
| `REPORT.md` | - | this report |
| `MANIFEST.json` / `MANIFEST.json.sha256` | - | identity, provenance, settings, output hashes / the manifest's SHA-256 |

From `prototype/bank_pipeline/`:

```
python research_logs/baselines/stage2_baseline.py --verify   # integrity only; no artifacts; writes nothing
python research_logs/baselines/stage2_baseline.py --check    # rebuild from the local historical artifacts; compare; writes nothing
```

`--verify` detects any change to the package's files. `--check` needs the historical artifacts recorded in `MANIFEST.json` and the snapshot SHA256SUMS (`PR37_BASELINE_SHA256SUMS`) and also detects changed source artifacts, fingerprints or results. A new baseline is a new version (new identifier, new directory); this one is never overwritten.
