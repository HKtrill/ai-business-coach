# PR 42 Commit 4, Step 4A — Stage 2 NSD/ECI binning drift: reproduction and diagnosis

**Question.** PR 39 Step 3A retained both bin families but left a drift unexplained. The lifts quoted when the
thresholds were locked (`eci_warm` 1.49–1.67×, `eci_hot` 3.53×, `nsd_elevated` 1.04–1.43×) differ from the
fold-isolated out-of-fold lifts (1.031×, 2.501× and 2.486×), and `nsd_elevated`'s folds range from 0.62× to 3.36×.
Which of threshold placement, encoder behaviour, population and in-sample label dependence does the
training-side evidence support?

**In brief.**

- **`eci_warm`: there is no drift in the bin.** The quoted 1.49–1.67× are two historical *deciles* inside the
  bin. The bin's own lift, printed beside it when the cut points were locked, was 0.92×. Its lower cut, 0.0758,
  admits the 0.35× decile (0.0758, 0.0777], the bin's largest. On the training rows the bin is 1.038× in sample and
  1.031× out of fold. Its rows come from contact periods between 0.26× (May 2008) and 5.5× (2010).
- **`eci_hot`: the drop lies between the historical fit and the training-side fits, not in fold isolation.** The
  quoted 3.53× was the historical top decile, which was the bin. On the training rows the bin is already 2.457× in
  sample, and 2.501× out of fold. In sample it consists of two row sets, checked on row masks:
  - the 2,533 rows of periods at or above the base rate, which are exactly the in-sample top decile, at 3.48×;
  - all rows of four May 2009 macro states (1,456 rows) at 0.68×.

  All six training-side encoders assign those four states to the bin, so the same 1,456 rows dilute it in every
  fold.
- **`nsd_elevated`: one fold's encoder moved whole macro states across both cuts.** Fold 0's encoder assigns the
  bin seven May 2009 states (prevalence 8.8%), which are `nsd_warm` in sample and under every other fold's
  encoder. It also assigns `nsd_hot` to all 50 states of the in-sample bin. Every row that enters or leaves the
  bin between the in-sample and out-of-fold encoders is a fold-0 row. The result: fold 0 is at 0.62× on 374 rows,
  the other folds at 2.82–3.36×, and the pool without fold 0 at 3.15×.
- **Across both families.** NSD and ECI are functions of the macro state (euribor3m, nr.employed, emp.var.rate):
  346 states over 26 periods. Period mix reproduces every bin's aggregate subscriber count to within 2%.
  Within-period deviations, which that ratio can cancel, total 0.7–4.3% of expected out of fold. Commit 3's
  later-day label shares are similar in sample and out of fold. That is provenance: their effect on any lift gap
  is unresolved.

**What the comparisons describe.** With the frozen cut points, three evaluations are compared, each against the
base rate of the rows it evaluates:

1. the all-rows engineer on its rows (in sample);
2. each fold's engineer on its own fit rows (fold fit; every row counts four times);
3. each fold's engineer on its held-out rows (out of fold, PR 39).

The two contrasts are named for what they compute. `fold_fit_minus_in_sample` changes the encoders and how rows
are weighted. `heldout_minus_fold_fit` evaluates the same five encoders on different rows. Neither isolates a
cause. In particular, `heldout_minus_fold_fit` is not the effect of a row's own label. Within a fold the
held-out and fit rows differ in composition, and per-fold differences of opposite sign cancel in the pool. What
an encoder does is shown directly by the macro states it assigns to each bin, and by which rows change bin.

The historical figures cannot be put through these comparisons. They were computed on all 41,188 rows, including
the rows now in the test split, with an encoder fitted on those rows. They are transcribed and Git-verified,
never recomputed (G7).

**Scope.** Training side only, diagnosis only. No threshold is proposed, no binning procedure is defined and no
configuration is compared for selection (Step 4B). The frozen binning (`shared.stage2.binning`, identity
`9cdd79af…`), `stage2_binning` and its accepted record `research_logs/pr39/c3_20261114_231604/` are read, never
changed.

**Evidence.**

- *Where it is produced.* `feature_research/feature_research.ipynb`, Cells 31–39 (module
  `feature_research.binning_drift`). The notebook writes `research_logs/pr42/<run_id>/binning_drift.json` beside
  the run's three earlier records.
- *Which run.* The numbers below come from this revision's sandbox run on the 32,950 training rows (Linux, Python
  3.13). Every count is an integer, and the PR 39 E-B section reproduces exactly on Linux. Cell 38 requires each
  claim below to hold in the run that writes the record. The Windows run is the accepted one.
- *Reuse, not new fitting.* The module reuses existing code throughout:
  - out-of-fold values: `stage2_binning.runner.oof_sources`;
  - in-sample values: Cell 7's `TRAINING_FRAMES["stage2_sources"]`;
  - statistics and triggers: `stage2_binning.evidence.bin_table` and `evidence.triggers`;
  - label provenance: `encoding_audit.audit_paths`.

  It imports no feature-engineering module, so the Commit 2 caller scan is unchanged and Cell 30's Step 3A
  reproduction still applies (`test_pr42_binning_drift.py` keeps it so).
- *Accepted printed outputs.* The historical outputs (commit `1ce1649`) and the exploratory notebook's Cells
  15A–15B are read from Git: the exploratory notebook as blob `266860ba…`, committed in PR 36 and pinned by
  SHA-256. They are never read from a working copy that Commit 6 may clear. Without the blob, the working copy is
  used only if its SHA-256 is the pinned one. Otherwise the comparison is reported as not checked, and a mismatch
  stops the run.
- *Tests.* `tests/feature_research/test_pr42_binning_drift.py` (synthetic, including a cleared working copy, mask
  identities and ratio cancellation) and `tests/realdata/test_pr42_binning_drift.py` (the real rows:
  reproduction, flips, reconciliation, the focal figures and identities, the exploratory cells with the working
  copy cleared, and every claim).

## 1. What each figure measures

| Bin | Figure | Source | Estimand | Rows | Encoder fit | Reproduced here |
|---|---|---|---|---|---|---|
| `eci_warm` | 1.49–1.67× | `1ce1649` Cell 13C (the frozen code comment) | in-sample **decile** lift: (0.0777, 0.0797] 1.67× on 2,019 rows; (0.0797, 0.0948] 1.49× on 4,991 | 41,188 (full population) | one fit on the same rows | no (G7); Git-verified transcription |
| | 0.92× | `1ce1649` Cell 13D | in-sample **bin** lift, 14,630 rows | 41,188 | the same fit | no |
| | 1.038× | PR 39 E-B `in_sample`; exploratory Cell 15B | in-sample bin lift, 12,491 rows | 32,950 training rows | one fit on them | yes, exactly |
| | 1.031× | PR 39 E-B `oof` | out-of-fold bin lift, 11,549 rows | 32,950 | five fits on 26,360 rows each | yes, exactly |
| `eci_hot` | 3.53× | Cell 13C | in-sample top decile (0.0948, 0.153], 3,178 rows | 41,188 | one fit | no |
| | 3.53× | Cell 13D | in-sample bin lift, the same 3,178 rows | 41,188 | the same fit | no |
| | 2.457× / 2.501× | PR 39 E-B | in sample (3,989 rows) / out of fold (4,074) | 32,950 | as above | yes, exactly |
| `nsd_elevated` | 1.04–1.43× | Cell 13C (the frozen comment's "warm 0.19–0.23") | deciles (0.190835, 0.192003] 1.04× and (0.192003, 0.230462] 1.43× | 41,188 | one fit | no |
| | 1.25× | Cell 13D | in-sample bin lift, 4,394 rows (10.7%) | 41,188 | the same fit | no |
| | 3.126× / 2.486× | PR 39 E-B | in sample (1,326 rows, 4.0%) / out of fold (1,421) | 32,950 | as above | yes, exactly |

All lifts are a rate divided by the base rate of the same rows (0.1127 historically; 3,712 / 32,950 on the
training rows). The historical encoder was the same NSD and ECI construction (KDE bandwidth 1.0 over 5,000
reference rows; curvature of 20 quantile-bin rates), but it drew its references with `np.random.seed(42)` and
`np.random.choice`, where the production engineer uses `np.random.default_rng(42)`. In `eci_hot`'s Cell 13C/13D
rows, the decile and the bin are both upper intervals of the same values, and both hold 3,178 rows, so they are
the same rows.

**Where the frozen cuts came from.** Each cut is a printed decile edge. Pandas prints interval edges rounded while
assigning rows by the exact edges, and three NSD edges were rounded further. The printed counts place rows between
each cut and its edge (`implied_cut_rows`):

| Family | Cut | Printed edge | Rows between at lock time |
|---|---|---|---|
| NSD | 0.0524 | 0.0524146 | 610 |
| NSD | 0.192 | 0.192003 | 520 |
| NSD | 0.23 | 0.230462 | 129 |
| ECI | 0.0291 | 0.0291 | 0 |
| ECI | 0.0758 | 0.0758 | 2,112 |
| ECI | 0.0948 | 0.0948 | 0 |

**Discrepancies with the accepted figures.** The out-of-fold figures in the roadmap and the PR 39 record reproduce
exactly. The "historical" figures are described differently than they were measured. 1.49–1.67× and 1.04–1.43×
are decile lifts, not bin lifts. The 1.04× decile lies in `nsd_warm`, not `nsd_elevated`, and the 0.35× decile
inside `eci_warm`'s range is left out of its quoted range. The bins' own historical lifts are 0.92× and 1.25×.

## 2. Reproduction and isolation

- **PR 39 Step 3A by import.** `eb_evidence` on the training rows reproduces the record's E-B section exactly:
  out-of-fold and in-sample tables, moved rows, NSD × ECI overlap and fold fits. It does so on the PR 39
  population and fold plan (`8aea082fcaf371b7`). T1–T4 and the retain decision for both families reproduce too.
- **Row-level sources.** The out-of-fold values (`oof_sources`) and the in-sample values (Cell 7) rebuild E-B's
  tables exactly. The local cut assignment agrees with the production binning path.
- **Held-out labels.** For each of the five folds, flipping all 6,590 held-out labels leaves that fold's NSD and
  ECI unchanged. In every fold the control moves all 26,360 rows of the other folds.
- **Macro states.** The 32,950 rows hold 346 states (at most three periods each). Under each of the six
  training-side encoders, NSD and ECI are constant within a state (largest spread 3e-17). Each fold engineer's
  values on its fit rows are obtained through `oof_sources`. That engineer is fitted on exactly that fold's fit
  rows (label fingerprint equal to E-B's), and its values agree with its held-out values for every shared state
  (within 1.1e-16).

## 3. The encoder comparison

Frozen cuts, each lift against the base rate of the evaluated rows. Fold fit pools (fold, fit row) pairs.

| Bin | In sample | Fold fit | Out of fold | `fold_fit_minus_in_sample` | `heldout_minus_fold_fit` | `oof_minus_in_sample` |
|---|---|---|---|---|---|---|
| `nsd_cold` | 0.437 | 0.428 | 0.436 | −0.009 | +0.008 | −0.001 |
| `nsd_warm` | 1.120 | 1.003 | 1.000 | −0.117 | −0.003 | −0.120 |
| **`nsd_elevated`** | 3.126 | 2.509 | 2.486 | −0.618 | −0.022 | −0.640 |
| `nsd_hot` | 4.303 | 4.201 | 4.182 | −0.103 | −0.019 | −0.121 |
| `eci_cold` | 0.520 | 0.521 | 0.519 | +0.001 | −0.002 | −0.001 |
| `eci_mid` | 0.698 | 0.719 | 0.733 | +0.020 | +0.014 | +0.035 |
| **`eci_warm`** | 1.038 | 1.037 | 1.031 | −0.001 | −0.006 | −0.008 |
| **`eci_hot`** | 2.457 | 2.508 | 2.501 | +0.051 | −0.006 | +0.045 |

These are descriptive differences. The pooled `heldout_minus_fold_fit` hides per-fold differences of both signs:

| Bin | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| `nsd_elevated`: fit-row lift / held-out lift | 1.006 / 0.617 | 3.091 / 3.262 | 3.199 / 2.821 | 3.117 / 3.163 | 3.070 / 3.356 |
| `nsd_elevated`: held out − fit | −0.388 | +0.171 | −0.378 | +0.045 | +0.286 |
| `eci_warm`: held out − fit | +0.010 | −0.024 | −0.016 | −0.003 | +0.008 |
| `eci_hot`: held out − fit | +0.006 | +0.030 | +0.033 | −0.075 | −0.037 |

The pooled −0.022 for `nsd_elevated` is therefore not a measure of a row's own label. What the table does show
is that fold 0's low lift appears on its encoder's fit rows as well (1.006×). It goes with the states that
encoder assigns to the bin, and the direct evidence of those assignments is in section 7. Rows that change bin
between the in-sample and the out-of-fold encoders:

| Family | In sample → out of fold | Rows | Lift | Folds |
|---|---|---|---|---|
| NSD | `nsd_cold` → `nsd_warm` | 1,756 | 0.45 | 1, 2, 4 |
| NSD | `nsd_warm` → `nsd_elevated` | 374 | 0.62 | 0 only |
| NSD | `nsd_elevated` → `nsd_hot` | 279 | 3.02 | 0 only |
| ECI | `eci_mid` → `eci_cold` | 2,124 | 0.54 | 0, 3, 4 |
| ECI | `eci_cold` → `eci_mid` | 910 | 0.58 | 1 |
| ECI | `eci_warm` → `eci_mid` | 857 | 0.79 | 1, 3, 4 |
| ECI | `eci_warm` → `eci_hot` | 88 | 4.64 | 1, 2 |
| ECI | `eci_hot` → `eci_warm` | 3 | 5.92 | 3 |

## 4. Support, uncertainty and fold stability

Pooled out of fold (`stage2_binning.evidence.bin_table`). Lift is the bin's rate over the base rate of the same
rows. The interval is the Wilson 95% interval of the rate divided by that base rate. It covers binomial sampling
of the evaluated rows with the encoders, cut points and base rate held fixed. It does not cover fold-to-fold
encoder variation, which the fold columns show.

| Bin | Rows | Subscribers | Lift [Wilson 95%] | Fold lifts (rows) | T1 min expected |
|---|---|---|---|---|---|
| `nsd_elevated` | 1,421 | 398 | 2.486 [2.284, 2.698] | 0.62 (374), 3.26 (275), 2.82 (255), 3.16 (258), 3.36 (259) | 28.7 |
| `eci_warm` | 11,549 | 1,341 | 1.031 [0.980, 1.084] | 1.05, 1.12, 0.89, 0.96, 1.15 (2,128–2,461) | 239.9 |
| `eci_hot` | 4,074 | 1,148 | 2.501 [2.381, 2.626] | 2.46, 2.49, 2.71, 2.39, 2.43 (754–887) | 85.0 |

Fold 0's `nsd_elevated` interval, [0.43, 0.89], excludes the other folds' values. The variation is not sparse
support: T1's minimum expected positives is 28.7, against a floor of 10. The fold-to-fold range is 2.74 for
`nsd_elevated`, 0.33 for `eci_hot` and 0.26 for `eci_warm`. No trigger fires, as in PR 39.

## 5. Decile lift and the historical thresholds

Deciles are `qcut(10, duplicates="drop")` on the evaluated values (`BD.DECILE_DEFINITION`). Equal values always
share a decile, so counts are unequal. Out-of-fold deciles use the pooled out-of-fold edges in every fold. The
in-sample deciles reproduce the exploratory notebook's accepted printed Cell 15A exactly (from the pinned blob),
its bin lifts and shares reproduce Cell 15B, and both kinds equal `compute_lift`'s.

| ECI decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| historical (41,188) | 0.52 | 0.49 | 0.44 | 0.50 | 1.33 | 0.66 | 0.35 | 1.67 | 1.49 | 3.53 |
| in sample | 0.57 | 0.47 | 0.49 | 0.49 | 0.82 | 0.97 | 0.50 | 1.69 | 1.49 | 3.48 |
| out of fold | 0.56 | 0.49 | 0.50 | 0.50 | 1.14 | 0.61 | 0.72 | 0.84 | 1.78 | 2.98 |

| NSD decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| historical (41,188) | 0.49 | 0.47 | 0.45 | 0.48 | 0.27 | 0.43 | 0.91 | 1.04 | 1.43 | 4.08 |
| in sample | 0.28 | 0.46 | 0.51 | 0.47 | 0.45 | 0.46 | 1.04 | 1.05 | 1.38 | 4.06 |
| out of fold | 0.41 | 0.33 | 0.52 | 0.40 | 0.42 | 0.47 | 0.76 | 1.13 | 1.51 | 4.07 |

The shape survives: a flat low region, then a rising upper tail, with ECI non-monotonic in its middle. ECI is a
curvature of rate curves, not a rate. The in-sample ECI decile 9, (0.0829, 0.0981], holds 2,585 `eci_warm` rows
and the 1,456 `eci_hot` rows of the four May 2009 states. The 0.0948 cut runs through it, and the decile above it
is exactly the bin's other rows. NSD's 0.192 cut lies inside decile 9, and its 0.23 cut inside decile 10, in
sample and out of fold.

**The historical candidate thresholds** are the nine printed decile edges per family, applied unchanged to the
training-side encodings (`historical_thresholds`). They are described here, not compared as candidates.

| Interval | Frozen bin | Historical | In sample | Out of fold [Wilson 95%] | Fold range |
|---|---|---|---|---|---|
| ECI (0.0758, 0.0777] | `eci_warm` | 0.35× (5,508) | 0.33× (1,679) | 0.77× [0.67, 0.88] (2,221) | 0.30–2.31 |
| ECI (0.0777, 0.0797] | `eci_warm` | 1.67× (2,019) | 0.96× (4,362) | 0.68× [0.60, 0.76] (3,378) | 0.44–1.03 |
| ECI (0.0797, 0.0948] | `eci_warm` | 1.49× (4,991) | 1.28× (6,450) | 1.33× [1.25, 1.41] (5,950) | 0.92–1.90 |
| ECI (0.0948, 0.153] | `eci_hot` | 3.53× (3,178) | 2.46× (3,989) | 2.50× [2.38, 2.63] (4,074) | 2.39–2.71 |
| NSD (0.190835, 0.192003] | `nsd_warm` | 1.04× (4,192) | — (0) | 0.94× (545, fold 0 only) | — |
| NSD (0.192003, 0.230462] | `nsd_elevated` | 1.43× (4,003) | 3.13× (1,326) | 2.52× [2.32, 2.73] (1,462) | 0.92–3.36 |
| NSD (0.230462, 0.350394] | `nsd_hot` | 4.08× (4,113) | 4.30× (2,665) | 4.19× [4.03, 4.35] (2,903) | 3.89–4.46 |

On the training-side encodings the narrow historical ECI intervals do not keep their lifts. (0.0777, 0.0797] falls
to 0.68×, and its neighbours' fold ranges span more than 1.0. Four of the ten historical NSD intervals hold no
rows out of fold, and four in sample, because the encoders' value clusters moved.

## 6. Contact periods

**Period mix and aggregate counts.** The period-standardized ratio divides a bin's observed subscribers by
expected ones: its rows at their periods' rates. Near 1, the period mix reproduces the bin's aggregate subscriber
count. Within-period differences of opposite sign can cancel in that ratio, so two measures are shown beside it:
the absolute within-period deviation (as a share of expected) and the within-period ratio range (periods with at
least 100 rows in the bin).

| Bin (out of fold) | Observed / expected | Absolute deviation | Within-period ratio range | Periods (shared with other bins) |
|---|---|---|---|---|
| `nsd_elevated` | 0.991 | 4.3% | 0.79–1.05 | 4 (4) |
| `eci_warm` | 1.013 | 2.2% | 0.97–1.06 | 9 (6) |
| `eci_hot` | 0.981 | 1.9% | 0.86–1.00 | 11 (3) |
| all eight bins | 0.981–1.015 | 0.7–4.3% | | |

In sample, the four NSD bins are unions of whole periods: no period has rows in two NSD bins, so their ratios
are exactly 1 by construction. For the focal bins out of fold, period mix accounts for each aggregate count to
within 2%, and the within-period deviations total at most 4.3% of expected. The bins still split some periods
between them. Among the focal bins out of fold, the lowest within-period ratio is fold 0's May 2009 rows in
`nsd_elevated`, at 0.79. This is consistent with Commit 3's finding that Stage 1's pooled ROC-AUC largely ranks
periods.

| Bin (out of fold) | Periods: rows at lift |
|---|---|
| `nsd_elevated` | 2009-05: 374 at 0.62 (fold 0 only); 2009-06: 429 at 3.12; 2009-07: 116 at 3.29; 2009-08: 502 at 3.15 (folds 1–4) |
| `eci_warm` | 2008-05: 5,524 at 0.26; 2009-05: 3,205 at 0.83; 2009-04: 1,795 at 1.61; 2009-03 and 2010-03..08: 1,025 at 3.9–5.5 |
| `eci_hot` | 2009-05: 1,456 at 0.68 (every fold); 2009-06 to 2010-08: 2,618 at 3.1–4.9 |

**Straddling states.** These are macro states whose frozen bin differs between the six training-side encoders.
All 50 of `nsd_elevated`'s in-sample states (June–August 2009) straddle: each crosses 0.23 under at least one
encoder. Seven May 2009 states (2,104 rows; NSD 0.1715–0.1925 across encoders) cross 0.192, and 29 November 2008
states (2,941 rows) cross 0.0524. Ten 2008 ECI states (10,360 rows) cross 0.0291, and 35 May 2008 and April 2009
states (2,621 rows) cross 0.0758. The 29 states that cross 0.0948 hold only 424 rows, and only one of them (9 rows)
is in `eci_hot` in sample, so that cut is not near a large cluster under any training-side encoder.

**Later-day label shares (provenance).** Commit 3's attribution (`audit_paths`, Stage 2 final fit and outer
folds) gives each row's mean share of label mass from later contact days. For ECI that share measures influence.

| Bin | Out of fold | In sample |
|---|---|---|
| `nsd_elevated` | 0.262 | 0.237 |
| `eci_warm` | 0.600 | 0.602 |
| `eci_hot` | 0.460 | 0.471 |
| all NSD rows | 0.469 | 0.464 |
| all ECI rows | 0.516 | 0.517 |

The means differ by at most 0.025 for the focal bins (0.070 for `nsd_warm`). Similar means can come from
different contributing labels and give different encoded values. So these shares do not show that look-ahead
leaves the in-sample to out-of-fold gaps unchanged, and they say nothing about its effect on the historical
figures. That effect needs a past-only NSD and ECI. Commit 3 did not build them (Step 3B §5), and they are not
built here.

## 7. Findings

`focal_assignments` establishes the row- and state-level facts below, on row masks and on each state's bin under
each of the six encoders.

| Fact | Checked as |
|---|---|
| `eci_hot` in sample = 2,533 rows of at-or-above-base periods + 1,456 rows of below-base periods | row masks |
| the 2,533 rows are exactly the in-sample top decile (0.0981, 0.141], at 3.48× | row-mask equality |
| the 1,456 rows are exactly all rows of four May 2009 states, at 0.68× | row-mask equality against those states' rows |
| all six encoders assign those four states to `eci_hot`; the out-of-fold bin's below-base rows are the same 1,456 rows (266–308 per fold) | per-state, per-encoder bins; row-mask equality |
| fold 0's encoder assigns `nsd_elevated` to seven May 2009 states, which are `nsd_warm` in sample and under folds 1–4 | per-state, per-encoder bins |
| fold 0's encoder assigns `nsd_hot` to all 50 in-sample `nsd_elevated` states | per-state, per-encoder bins |
| all 653 rows entering or leaving `nsd_elevated` (in sample → out of fold) are fold-0 rows | row masks |

| Cause | `eci_warm` | `eci_hot` | `nsd_elevated` |
|---|---|---|---|
| Threshold placement | **main**: the quoted figures describe a narrower interval; the lower cut admits the 0.35× decile | the cut 0.0948 lies just below four low-rate May 2009 states, which every training-side encoder puts in the bin | **with the encoder**: 0.192 and 0.23 sit inside the ranges that whole May 2009 and June–August 2009 states take across encoders |
| Encoder behaviour | no measurable change (contrasts ≤ 0.006) | the historical-to-training change is not separable; training-side encoders agree on the cluster | **main**: fold 0's encoder moves seven states into the bin and all 50 in-sample states out of it |
| Population (period mix) | rows from periods at 0.26–5.5×; period mix reproduces the aggregate count (1.01) | an above-base tail at 3.48× and one below-base cluster | fold 0's bin holds May 2009 states (below base), the other folds' June–August 2009 states (above); period mix reproduces the aggregate count (0.99) |
| In-sample label dependence | not isolated (pooled held-out − fit −0.006; per fold −0.024 to +0.010) | not isolated (−0.006; per fold −0.075 to +0.033) | not isolated (−0.022; per fold −0.39 to +0.29) |
| Look-ahead | mean shares 0.600 / 0.602; effect unresolved | 0.460 / 0.471; effect unresolved | 0.262 / 0.237; effect unresolved |
| Entangled | historical 0.92× → training 1.04× (population, fit, reference draw) | whether the four states fell below 0.0948 in the historical fit | whether a fit's labels (a row's own included) or its 5,000-row reference draw move the states |

- **`eci_warm` (1.49–1.67× → 1.03×).** This is an estimand and threshold-placement artefact, not drift in the bin.
  The bin was 0.92× when its cuts were locked. On the training rows it is 1.04× in sample and 1.03× out of fold
  (Wilson [0.98, 1.08]), and its folds range only 0.89–1.15. Both descriptive contrasts are below 0.01. It sits
  near the population rate because its macro states come from periods far below (May 2008, May 2009) and far
  above (2010) the base rate.
- **`eci_hot` (3.53× → 2.50×).** The drop lies between the historical full-population fit and the training-side
  fits, not in fold isolation (in sample 2.457×, out of fold 2.501×). On the training rows the bin is exactly the
  in-sample top decile (3.48×) plus the rows of four May 2009 states (0.68×). Every training-side encoder agrees,
  so the upper tail keeps its lift (3.52× out of fold) and every fold is diluted alike (2.39–2.71×). Whether those
  states fell below 0.0948 in the historical fit remains open: that fit used labels of rows now in the test split.
- **`nsd_elevated` (pooled 2.49×, folds 0.62–3.36×).** One fold's encoder moves whole macro states across both
  cuts. Fold 0's encoder places seven May 2009 states in the bin, which no other encoder does, and moves all 50
  in-sample bin states into `nsd_hot`. Every row entering or leaving the bin is a fold-0 row, and the low lift
  appears on that encoder's fit rows too (1.006×). Support is adequate (T1 margin 28.7). How much of the shift
  comes from the fit's labels, a row's own included, and how much from its reference draw is not separated. The
  historical 1.25× bin held a different, larger set of states (10.7% of rows against 4.0%), under a fit that
  cannot be rebuilt here.

## 8. What Step 4B inherits (questions, not decisions)

- Any cut on these encodings partitions **macro states**, so all rows of a state move together. A candidate
  procedure will need to say how it treats a cut that lands inside the range one state takes across fits (NSD
  0.192 and 0.23; ECI 0.0291 and 0.0758).
- Under random folds, period mix reproduces these bins' aggregate counts closely. A candidate's routing value
  under a chronological split is PR 49's question, not one a random-fold comparison can answer.
- Whether the **reference draw** or the **fit labels** move NSD states is open. Both differ between the six
  encoders here.
- The **historical figures** cannot serve as a baseline. They are full-population decile lifts. The PR 39
  out-of-fold tables are the baseline (roadmap, Step 4B).

## 9. Limits

- The historical figures were computed on all 41,188 rows with an encoder fitted on them. They are transcribed and
  Git-verified, never recomputed, so the historical-to-training difference combines population, fit size and
  reference draw, which the training side cannot separate.
- Every comparison uses random folds. All encoders use labels of later contacts (about 47% of the NSD and 52% of
  the ECI label mass or influence). Similar mean shares do not imply similar contributing labels or encoded values.
  A past-only NSD or ECI is not built, so the effect of look-ahead on these lifts and their gaps is unresolved.
- The two encoder contrasts are descriptive. `fold_fit_minus_in_sample` changes encoders and row weights at once.
  `heldout_minus_fold_fit` compares the same encoders on different rows, so fold composition and cancellation
  affect it. Neither isolates a row's own label.
- Period-standardized ratios near 1 show that period mix reproduces a bin's aggregate count. Within-period
  differences can cancel in them; the absolute deviation and the within-period ratios are reported beside them.
- The six encoders differ in their fit labels and in their KDE reference draw at once.
- The frozen cut points are constants chosen with all labels (the PR 39 limitation). No threshold is evaluated
  here as a candidate.
- The Wilson intervals cover binomial sampling of the evaluated rows only. Fold-fit tables count each row four
  times, so their pooled intervals are omitted.

## 10. Where the evidence is

| Evidence | Notebook cell | Record `binning_drift.json` |
|---|---|---|
| the PR 39 record (pinned SHA-256), the historical transcription and its Git check, implied cut rows, the exploratory notebook's accepted printed outputs (pinned blob, source and status), the figure table | 31 | `provenance` |
| E-B, T1–T4 and the decision reproduced by import (checks, E-B digest, triggers, decision) | 32 | `reproduction` |
| row-level sources rebuilding E-B; the out-of-fold fits | 32 | `sources` |
| held-out label flips with controls | 33 | `isolation` |
| macro states, their values under each encoder, the fold-fit checks | 34 | `macro_states` |
| in-sample / fold-fit / out-of-fold lifts, the two descriptive contrasts, per fold, migration, fold-fit tables | 34 | `encoder_comparison` |
| frozen-bin support, subscribers, lift and Wilson intervals (estimand), fold stability | 35 | `support` |
| deciles (definition, intervals, frozen bins and cuts inside, per fold) and their reproduction checks | 36 | `deciles` |
| historical candidate thresholds on the training-side encodings | 36 | `historical_thresholds` |
| period rates, bin × period composition, period-standardized ratios with absolute deviations, within-period ranges and shared periods | 37 | `periods` |
| straddling states and their share of each bin | 37 | `straddling` |
| Commit 3 later-day shares by bin | 37 | `label_provenance` |
| row- and state-level facts behind the findings | 38 | `focal_assignments` |
| findings, claims (each with its evidence and `holds`), limits | 38 | `findings`, `claims`, `limits` |
