# PR 42 Commit 3, Step 3A — Encoding construction and leakage audit

**Question.** For a row being encoded, exactly which rows and labels can influence its feature value, and which
of those labels could have been known at the time of contact?

**Scope.** The accepted target-derived encodings of both cascades — Stage 1 `subscribe_rate_by_euribor_bin` and
`subscribe_rate_by_dow_month` (GLASS LR and black-box MLP), and the Stage 2–3 sources
`neighborhood_subscription_density` (NSD), `economic_curvature_intensity` (ECI) and `dow_month_encoded`, with the
features that inherit them — on every path that fits them. Training side only: the test split is never read; how
it is mapped is traced from the code. Past-only alternatives, predictive comparisons, contract recommendations and
the PR 49 robustness protocol are Step 3B.

**Evidence.** `feature_research/feature_research.ipynb`, Cells 14–21 (module `feature_research.encoding_audit`),
write `research_logs/pr42/<run_id>/encoding_audit.json` beside the run's `feature_inventory.json`. The numbers
below are from this revision's run on the 32,950 training rows (Linux sandbox). Counts are exact integers; shares
are means over rows, rounded to three decimals. `tests/realdata/test_pr42_encoding_audit.py` re-derives the time
index, the accepted Stage 2 fold 0 plan and the Stage 1 out-of-fold counts on the local data.

Three mechanisms are kept apart throughout:

| | Mechanism | Finding |
|---|---|---|
| (a) | a fit row's own label in its own training representation | **present** in every fit, with small weight: every fit row for the bin, cell and curvature encodings (except a few euribor rows at rounded bin edges); only the reference rows for NSD |
| (b) | a held-out row's label in its evaluation representation | **absent**: measured 0 on every fold path; the test mapping fits no test label |
| (c) | labels of later contacts in the encoding of an earlier row | **present** on every path, for every family, including the folds |

## 1. The PR 42 time index

**Source order, verified** (Cell 14, `verify_source_order`). `build_database` inserts the CSV's data rows in file
order (blank lines skipped; there are none), so `raw_bank` rowids run 1..41,188; `transform.sql` reads
`ORDER BY rowid` into a RangeIndex; the split keeps those labels. On the training rows, each index label is its
0-based CSV data line: every pass-through column the index and the encodings read, and the label, equal the source
line's, and every coded column corresponds one to one with its source values. Only the training rows' lines are
kept; 2,244 training rows have an identical twin, and swapping twins changes no time key and no label.

**Periods and contact days** (`build_time_index`, `time_index_checks`). The file has no year or day of month. The
year is 2008 at the first row and increases wherever the month decreases in source order. This gives 26 periods,
2008-05 to 2010-11, with no training rows in 2008-09, January–February 2009 or January–February 2010. Each period has a
single value of each monthly or quarterly indicator, and the consumer price index differs between every two
periods, so same-month periods are different years. A **contact day** is a run of rows sharing year, month,
weekday and the daily `euribor3m`: 485 days. No period has more contact days than its calendar month has weekdays,
and within each period the weekday advances except at no more than four week changes (Mon–Fri cycles). Eighteen
day keys recur later in their month, always after a week change; the index treats them as separate days.

**Resolution** (`EA.RESOLUTION`). The finest defensible unit is the contact day. A contributing contact is
classified relative to the encoded row as an earlier day, the same day, a later day of the same month, or a later
month.

- *Between periods*: calendar order of (year, month).
- *Between days of a period*: source order. This relies on the documented date order; the weekday checks are
  consistent with it but do not prove it.
- *Same day*: unresolved. No time of day is recorded, so same-day contacts are treated as simultaneous.
- *Outcome timing*: unknown. A label records whether the client subscribed, not when that became known. A row is
  the client's last contact of the campaign (`campaign` counts the contacts, including this one), and no client
  identifier links rows. A later-day label was therefore not yet known at contact; an earlier-day label was at
  most plausibly known.
- *Training rows only*: a day whose contacts all fell in the test split is absent. At most, this merges two days
  with the same key into one, which is classified same-day and never later.

The subscription rate is 3.0% in 2008-05, 4–6% for most of 2008 and 8.8% in 2009-05, but 34–60% in every period
from 2009-06 onwards (`PERIODS`). Two periods are tiny: 2008-10 has 55 rows and 2008-12 has 6.

## 2. How the encodings are built

(Cell 15, `EA.CONSTRUCTION`, traced from the production code.)

| Encoding | Fit | Applied | Fallback | Own label in a fit row's value |
|---|---|---|---|---|
| `subscribe_rate_by_euribor_bin` | `qcut(euribor3m, 20)` of the fit rows; mean label per bin | `pd.cut` with the stored, rounded `IntervalIndex` | median of the bin rates (the middle bin's rate, or the mean of the two middle bins' rates) | 1/n_bin |
| `subscribe_rate_by_dow_month`, `dow_month_encoded` | `(n_c·mean_c + 100·global) / (n_c + 100)` per weekday × month cell (no year) | cell → rate | global rate of the fit rows | 1/(n_c+100) through the cell, plus 100/((n_c+100)·N) through the global rate |
| NSD | scaler on all fit rows; reference = 5,000 fit rows (`default_rng(42)`) when the fit has more than 10,000 rows; Gaussian kernel, bandwidth 1.0 | kernel average of the reference labels | none | only for the 5,000 reference rows |
| ECI | for euribor3m, nr.employed and emp.var.rate: `qcut(20)`, rate per bin, `abs(gradient(gradient(rates)))` | bin → curvature (Stage 2: rounded categories; Stage 3: exact edges); mean of three | 0.0 outside the bins | yes, through the bin's rate and its neighbours' |

On the full training split, 15 of the 20 euribor bins fall within one to three periods. Of the five wider bins,
bins 0–2 cover the low-euribor regime from 2009-05 to 2010-11, bin 6 spans the late-2008 rate cut (2008-11 to
2009-04) and bin 19 is the top edge. Of the 50 day × month cells, 49 pool more than
one year: May 2008, May 2009 and May 2010 contacts on a Monday share one rate.

## 3. Fitting paths

(Cell 16, `fit_paths`, `path_table`, `plan_checks`, `router_plan_reference`.)

| Stage | Path | Fits | Label rows | Held-out rows | Used for |
|---|---|---|---|---|---|
| 1 (both arms) | final | 1 | 32,950 | none (in-sample) | final base model and scaler; maps the test split (the final calibrator uses OOF scores) |
| 1 | outer | 10 | 29,655 | 3,295 | the Optuna tuning folds (`tune_cv_folds = cv_folds = 10`: the same designs) and the OOF base scores, threshold and Stage 4 column |
| 1 | inner | 100 | 26,689–26,690 | 2,965–2,966 | nested calibration (F19) |
| 2 (both routers) | final | 1 | 32,950 | none | full-train routers; maps the test split |
| 2 | final_inner_label_excluded | 5 | 26,360 (structure 32,950) | 6,590 | the full-train routers' Pass 1 OOF / eligibility folds |
| 2 | outer | 5 | 26,360 | 6,590 | outer cross-fit: Stage 4 train side |
| 2 | inner_label_excluded | 25 | 21,088 (structure 26,360) | 5,272 | each outer fit's Pass 1 OOF / eligibility folds |
| 3 (both arms) | final | 1 | 32,950 | none | refit model; maps the test split (calibrator and thresholds use OOF scores) |
| 3 | tuning / reporting / inner | 5 / 10 / 25 | 26,360 / 29,655 / 21,088 | 6,590 / 3,295 / 5,272 | tuning, OOF, calibration and thresholds; reporting; F22 inner folds |

Several paths coincide, so their evidence is shared:

- Both Stage 1 arms build their folds with the same function from the same plan (`fold_assignment(y, 10, 42)`),
  and their tuning folds are their OOF designs.
- Both routers use one engineer and one plan.
- The final Stage 2 fit's inner folds hold out exactly the outer folds' rows.
- The Stage 3 tuning folds are the Stage 2 outer folds.
- Outer fold 0 and its inner plan reproduce the accepted Stage 2 run's record (PR 38 `c3_20261112_153333`, fold 0):
  26,360 fit rows, label fingerprint `85c06667c5ec`, plan `413b8241fbe25e9a`, 5,000 reference rows, of which
  3,994 / 4,016 / 3,956 / 4,003 / 4,031 are kept per inner fold.

In a label-excluded inner fold, only labels are excluded. The scaler, the reference sample and the quantile edges
are those of the outer (or final) fit, so the excluded rows' covariates, but never their labels, shape their own
encodings.

## 4. Findings by fitting path

(Cells 18 and 20: `audit_paths`, `findings_table`; record `findings.by_path` and `attribution.summary`.)

Every encoded value on every path is written as a combination of fit-row labels and reconstructed. Each
reconstruction is compared with the production output before any aggregation, and a nonfinite value on either side
fails the comparison. Cell 18 stops unless the completed comparisons match the expected ones (`COMPARISON_POLICY`,
`comparison_coverage`, record `attribution.coverage`):

- **Expected and completed.** 801 comparisons were expected and 801 passed, with 0 failed and 0 nonfinite values.
  They cover all held-out rows, every final fit (including its final mapping) and every Stage 1 and Stage 3 fit's
  own rows. The largest difference is 3.3e-16 against a tolerance of 1e-12.
- **Intentionally skipped.** 105 comparisons were expected to be skipped, and 105 were: the Stage 2 fold fits' own
  rows. Production does not re-encode those rows; the same blocks are compared on each fit's held-out rows.
- **Product check.** All 81 expected `decay_x_density` = `joint_economic_decay` × NSD checks passed.

That accounts for all 906 attributions. Timing is by contact day (§1): *later month* and *later day* are
demonstrably later contacts; *same day* is unresolved.

How to read the columns:

- **(a)** Fit-row encodings that carry the row's own label, over all fit-row encodings of that path, summed over
  its fits.
- **(b)** Held-out encodings that carry the row's own label, over all held-out encodings. This is measured, not
  assumed: each held-out row is located among the fit's source rows, so a label that entered its own encoding would
  be counted. A row outside the source, or in it but excluded from the label statistics (Stage 2 label-excluded
  folds), measures zero. A final fit has no held-out rows: it maps the test split, and no test label is fitted.
- **(c)** Computed on the held-out rows (fold paths) or on the final mapping at each training row's contact day
  (final fits, marked †):
  - *later month*: rows with any direct contributor (a bin, cell, kernel or curvature member) from a later month;
  - *any later*: rows with a direct or smoothing-prior contributor from any later day;
  - *share*: the mean share of the encoding's label mass that is later;
  - *majority*: rows whose label mass is more than half later;
  - *same day*: the mean unresolved same-day share.

| Feature | Stage, path (fits) | (a) own label, fit rows | (b) own label, held-out rows | (c) later month | (c) any later | (c) share | (c) majority | same day |
|---|---|---|---|---|---|---|---|---|
| S1 euribor bin | 1 final (1) | 32,950/32,950 | — | †20,787/32,950 | †29,062/32,950 | †0.413 | †13,481/32,950 | †0.174 |
| S1 euribor bin | 1 outer (10) | 296,550/296,550 | 0/32,950 | 20,680/32,950 | 29,228/32,950 | 0.414 | 13,311/32,950 | 0.172 |
| S1 euribor bin | 1 inner (100) | 2,668,869/2,668,950 | 0/296,550 | 186,153/296,550 | 263,631/296,550 | 0.414 | 120,554/296,550 | 0.172 |
| S1 day × month | 1 final (1) | 32,950/32,950 | — | †31,142/32,950 | †32,941/32,950 | †0.369 | †11,489/32,950 | †0.238 |
| S1 day × month | 1 outer (10) | 296,550/296,550 | 0/32,950 | 31,142/32,950 | 32,941/32,950 | 0.370 | 11,446/32,950 | 0.234 |
| S1 day × month | 1 inner (100) | 2,668,950/2,668,950 | 0/296,550 | 280,278/296,550 | 296,469/296,550 | 0.371 | 103,111/296,550 | 0.231 |
| NSD | 2 final (1) | 5,000/32,950 | — | †32,851/32,950 | †32,941/32,950 | †0.464 | †14,654/32,950 | †0.021 |
| NSD | 2 final inner, label-excluded (5) | 20,000/131,800 | 0/32,950 | 32,851/32,950 | 32,940/32,950 | 0.464 | 14,704/32,950 | 0.020 |
| NSD | 2 outer (5) | 25,000/131,800 | 0/32,950 | 32,851/32,950 | 32,928/32,950 | 0.469 | 14,590/32,950 | 0.020 |
| NSD | 2 inner, label-excluded (25) | 100,000/527,200 | 0/131,800 | 131,404/131,800 | 131,709/131,800 | 0.469 | 58,353/131,800 | 0.020 |
| NSD | 3 final (1) | 5,000/32,950 | — | †32,851/32,950 | †32,941/32,950 | †0.464 | †14,654/32,950 | †0.021 |
| NSD | 3 tuning (5) | 25,000/131,800 | 0/32,950 | 32,851/32,950 | 32,928/32,950 | 0.469 | 14,590/32,950 | 0.020 |
| NSD | 3 reporting (10) | 50,000/296,550 | 0/32,950 | 32,851/32,950 | 32,930/32,950 | 0.465 | 14,511/32,950 | 0.020 |
| NSD | 3 inner (25) | 125,000/527,200 | 0/131,800 | 131,404/131,800 | 131,760/131,800 | 0.466 | 58,342/131,800 | 0.020 |
| ECI | 2 final (1) | 32,950/32,950 | — | †32,851/32,950 | †32,941/32,950 | †0.517 | †19,140/32,950 | †0.037 |
| ECI | 2 final inner, label-excluded (5) | 131,800/131,800 | 0/32,950 | 32,851/32,950 | 32,941/32,950 | 0.517 | 19,154/32,950 | 0.037 |
| ECI | 2 outer (5) | 131,800/131,800 | 0/32,950 | 32,851/32,950 | 32,941/32,950 | 0.516 | 18,870/32,950 | 0.037 |
| ECI | 2 inner, label-excluded (25) | 527,200/527,200 | 0/131,800 | 131,404/131,800 | 131,764/131,800 | 0.517 | 75,396/131,800 | 0.037 |
| ECI | 3 final (1) | 32,950/32,950 | — | †32,851/32,950 | †32,941/32,950 | †0.517 | †19,140/32,950 | †0.037 |
| ECI | 3 tuning (5) | 131,800/131,800 | 0/32,950 | 32,851/32,950 | 32,941/32,950 | 0.516 | 18,862/32,950 | 0.037 |
| ECI | 3 reporting (10) | 296,550/296,550 | 0/32,950 | 32,851/32,950 | 32,941/32,950 | 0.517 | 18,904/32,950 | 0.037 |
| ECI | 3 inner (25) | 527,200/527,200 | 0/131,800 | 131,404/131,800 | 131,764/131,800 | 0.516 | 75,547/131,800 | 0.037 |
| `dow_month_encoded` | 2 final (1) | 32,950/32,950 | — | †31,142/32,950 | †32,941/32,950 | †0.369 | †11,489/32,950 | †0.238 |
| `dow_month_encoded` | 2 final inner, label-excluded (5) | 131,800/131,800 | 0/32,950 | 31,142/32,950 | 32,941/32,950 | 0.371 | 11,459/32,950 | 0.231 |
| `dow_month_encoded` | 2 outer (5) | 131,800/131,800 | 0/32,950 | 31,142/32,950 | 32,941/32,950 | 0.371 | 11,459/32,950 | 0.231 |
| `dow_month_encoded` | 2 inner, label-excluded (25) | 527,200/527,200 | 0/131,800 | 124,568/131,800 | 131,764/131,800 | 0.373 | 46,238/131,800 | 0.224 |
| `dow_month_encoded` | 3 final (1) | 32,950/32,950 | — | †31,142/32,950 | †32,941/32,950 | †0.369 | †11,489/32,950 | †0.238 |
| `dow_month_encoded` | 3 tuning (5) | 131,800/131,800 | 0/32,950 | 31,142/32,950 | 32,941/32,950 | 0.371 | 11,459/32,950 | 0.231 |
| `dow_month_encoded` | 3 reporting (10) | 296,550/296,550 | 0/32,950 | 31,142/32,950 | 32,941/32,950 | 0.370 | 11,446/32,950 | 0.234 |
| `dow_month_encoded` | 3 inner (25) | 527,200/527,200 | 0/131,800 | 124,568/131,800 | 131,764/131,800 | 0.373 | 46,238/131,800 | 0.224 |

The Stage 1 rows cover both arms: the GLASS LR and the black-box MLP fit the same designs. The Stage 2 rows
cover both routers. The NSD (a) counts are the reference rows: only their labels enter, about 4,000 per fit on a
label-excluded path. The few rows with no later contributor (for example 9/32,950 out of fold) are the last
contact days.

### (a) A fit row's own label: present in every fit, small

Which fit rows carry their own label depends on the encoding:

- **Bin and cell encodings** (Stage 1 euribor bin and day × month, `dow_month_encoded`, and the ECI curvature bins).
  Each fit row's own label enters its own value, through its bin's or cell's rate and, for the cell rates, the
  global prior. The one exception is a euribor row near a bin edge: the rounded categories can map it to the bin next
  to the one its own fit assigned it, and then its own label is absent. That happens in 81 of 2,668,950 Stage 1
  inner fit-row encodings, and in none of the final or outer fits.
- **NSD.** Only the reference rows' own labels enter: the 5,000 rows each fit samples (about 4,000 on a
  label-excluded path). Every other fit row's NSD carries no label of its own: 5,000/32,950 in the final fits.

The models are therefore trained on encodings that mostly include each row's own label, and scored on encodings
that do not.

- **Size.** The mean own-label share in the final fits is 6.1e-4 (euribor bin), 1.1e-3 (day × month and
  `dow_month_encoded`), 8.3e-5 (NSD, averaged over all rows) and 1.3e-4 (ECI). The maximum is 8.4e-3, for the
  smallest day × month cell.
- **Verification.** Flipping one training row's label moves its own full-train encodings by exactly the attributed
  self weight (20 rows × 2 features, deviation 5.6e-17; `isolation.checks`).

### Black-box MLP early stopping

(Cell 17, `mlp_early_stopping`; record `isolation.mlp_early_stopping` and `findings.mlp_early_stopping`;
`EA.MLP_EARLY_STOPPING_ORDER`, `EA.MLP_ISOLATED_SPLIT`, `EA.MLP_OUTER_ISOLATION`.)

**Order.** Every black-box MLP fit runs in the same order:

1. The fit's Stage 1 encoders and scaler are fitted on all of its rows. This applies to each Optuna trial on each
   of the 10 outer designs, each calibration refit on an outer or inner design (`oof_calibration._fit_on`) and the
   final model.
2. `Stage1MLPClassifier.fit` then draws its early-stopping subset from those already-encoded rows:
   `train_test_split(test_size=0.15, stratify=y, random_state=42)`, with oversampling after the split. The rest of
   the rows are the optimization rows the network is trained on.

The audit reproduces that split on all 111 Stage 1 fits. A one-epoch fit confirms it on the final and outer fold 0
rows: the estimator's recorded validation ROC-AUC equals the AUC on the reproduced rows (0.786709548334 and
0.811366122871).

**Which labels reach which features.** The encoders were fitted with the subset's labels, so those labels enter
the encodings of both sides of the split:

- the optimization rows' encodings, which the network is trained on;
- the validation rows' encodings, which the stopping criterion scores: each validation row's own label and the
  other validation rows' labels.

| MLP fits (count) | Optimization row's label mass from the subset | Optimization-row encodings carrying subset labels | Validation row's label mass from the subset (own + other validation) | Validation-row encodings carrying their own label |
|---|---|---|---|---|
| final (1) | 15.0% | 28,007/28,007 | 15.03% (0.061% + 14.97%) euribor bin; 15.07% (0.106% + 14.96%) day × month | 4,943/4,943 |
| outer = tuning and calibration-outer designs (10) | 15.0% | 252,060/252,060 | 15.06% (0.068% + 14.99%); 15.09% (0.117% + 14.97%) | 44,490/44,490 |
| inner calibration designs (100) | 15.0% | 2,268,550/2,268,550 | 15.06% (0.075% + 14.99%); 15.11% (0.126% + 14.98%) | 400,385/400,400 (euribor bin); 400,400/400,400 |

(Encoding counts are per feature: both features give the same counts except the inner euribor bin.)

**What this means for early stopping.** The subset is not isolated from the encoders. A properly isolated internal
validation split excludes the entire subset from encoder fitting: the encoders and scaler would be fitted on the
optimization rows only, and no subset label would enter any encoding, of optimization or validation rows. In the
accepted fits, about 15% of both sides' label mass comes from the subset. The epoch is therefore chosen by a
validation ROC-AUC computed on features built partly from the validation labels themselves, for a network trained on
features that also carry them. Mechanism (c) applies to these rows as to any. The effects on the stopping epoch,
the predictions and the AUC are not quantified here. The GLASS LR has no internal validation split.

**Outer OOF isolation, assessed separately.** The outer held-out rows are outside all of the above:

- they are excluded from the encoders, the scaler, the network fit and its early-stopping subset;
- Optuna's fold scores and the calibration's OOF base scores are computed on `design.X_val`;
- the tuning objective and `_fit_on` are passed only the design's training rows (`label_argument_checks`);
- the flips in (b) leave the outer and inner designs unchanged;
- the accepted MLP fit records per-fold encoders (`isolation.accepted_fits`).

The tuning-fold scores, the OOF scores and the Stage 4 column are therefore scored on encoder-held-out rows.

### (b) Held-out labels: absent on every fold path

- **Attribution, measured.** The table's (b) column is 0 on every fold path: 0 of 1,845,200 held-out encodings
  (445 feature × fold-fit combinations), across all 22 feature × path groups (`attribution.held_out_own_labels`).
  The check can fail. With one held-out label deliberately included, the audit measures exactly its known positive
  weight: 1/(n+k) + k/((n+k)N) through the cell, on both a Stage 1 fold and a Stage 2 label-excluded fold
  (`test_a_deliberately_*`).
- **Label flips** (Cell 17, `isolation_checks`), through the production builders, each with a control that must
  change:
  - Stage 1 `build_fold_designs`: with a fold's held-out labels flipped for every encoder fit, its design matrices
    are unchanged in 10/10 outer folds; flipping its training labels changes them in 10/10.
  - `inner_fold_designs`: the same holds in 10/10 inner folds (plus controls), and flipping an outer fold's
    held-out labels leaves all ten of its inner designs unchanged (F19).
  - Stage 2's label-excluded encoder: flipping an inner fold's labels leaves its features unchanged in 25/25 folds
    (plus controls).
- **Call sites** (AST, `label_argument_checks`). The Stage 1 designs, both Stage 2 routers' outer fits, Stage 3's
  `fold_frames` and the MLP's tuning and calibration fits are passed the fold's training labels only.
- **Accepted fits** (`accepted_fit_scopes`, record `isolation.accepted_fits`). The verified Commit 2 record's
  boundary-extractor fold checks passed for the LR, both routers and the EBM, including the EBM's
  `feature_fit_scope is 'per_fold'`. The accepted Stage 1 fits of both arms record per-fold encoders in tuning, OOF
  and calibration folds (PR 39 Commit 2: `oof_preprocessing`, `oof_provenance`).
- **Test mapping.** The final fits use training labels only (`engineer_features`, `Stage3Block.build`).

### (c) Later contacts: present on every path

About 41% (euribor bin), 37% (day × month), 47% (NSD) and 52% (ECI) of an encoding's label mass comes from later
contact days. This holds on every path, out of fold and in the final mappings alike. Random folds remove (b) but not
(c), because every fold's fit rows span the whole timeline.

- **Out of fold.** 20,680/32,950 rows (euribor bin) and more than 94% of rows for every other family (31,142 to
  32,851 of 32,950) receive labels from a later month.
- **Through the smoothing prior.** The day × month encodings draw about 0.04 of their mass from later months
  through the global rate alone. The cell key pools years, so May 2010 contacts inform May 2008.
- **Medians and upper deciles.** Out of fold, the median (p90) later share is 0.419 (0.787) for the euribor bin,
  0.312 (0.751) for day × month, 0.440 (0.869) for NSD, 0.518 (0.714) for ECI and 0.312 (0.748) for
  `dow_month_encoded`.

Look-ahead is largest early in the timeline (Cell 19, record `attribution.final_mapping_by_period`). These are the
final mappings' later shares in a few periods:

| Period | Euribor bin | Day × month | NSD | ECI |
|---|---|---|---|---|
| 2008-05 | 0.471 | 0.655 | 0.828 | 0.710 |
| 2009-03 | 0.320 | 0.485 | 0.938 | 0.693 |
| 2009-09 | 0.830 | 0.359 | 0.335 | 0.474 |
| 2010-11 | 0.011 | 0.009 | 0.016 | 0.007 |

The prevalence shift coincides with this: an early, low-rate row's encoding is built partly from later contacts in
other regimes.

### Stage 2–3 feature families

(Cell 15 `ENCODINGS`; Cells 16–18; record `construction.descendants`.) The models consume the following members.
Each member inherits its family's label inputs, with no label input of its own.

| Family: labels enter through | Members the models consume | Fit on each path | Evidence |
|---|---|---|---|
| **NSD**: the kernel over the reference rows' labels (5,000 sampled from the fit rows; scaler on all fit rows) | Stage 2, both routers: `nsd_cold`, `nsd_warm`, `nsd_elevated`, `nsd_hot` (fixed cut points on NSD). Stage 3, both arms: `decay_x_density` = `joint_economic_decay` × NSD (label-free factor; product verified, max diff 0) | Stage 2: refit on each outer fold's 26,360 rows. Label-excluded inner folds (5 × 5, and the final fit's 5) keep the outer (final) fit's scaler and sample and drop fold j's ~1,000 reference labels. Stage 3: a new `EBMFeaturePipeline` per tuning, reporting and inner fold. Final fits on all 32,950 rows map the test split | table rows NSD; `fitting_paths`; `isolation.checks` (Stage 2 flips); `router_plan_reference` (fold 0 kept references); `test_the_label_excluded_inner_fold_uses_no_label_of_its_fold` |
| **ECI**: the second difference of 20-bin label rates of euribor3m, nr.employed and emp.var.rate (up to five neighbouring bins), averaged | Stage 2: `eci_cold`, `eci_mid`, `eci_warm`, `eci_hot`. Stage 3: `economic_curvature_intensity`. The three curvature intermediates are not consumed | as NSD (label-excluded: fold j's rows leave the bin rates; the quantile edges stay) | table rows ECI; reconstruction through the curvature operator (`test_the_curvature_operator_is_numpys_double_gradient`) |
| **Day × month**: the smoothed cell rate (same mechanism as Stage 1's) | Stage 2: `dow_cold`, `dow_low`, `dow_mid`, `dow_hot`. Stage 3: `dow_month_encoded` | as NSD (label-excluded: fold j's rows leave the cells and the global rate) | table rows `dow_month_encoded`; the Stage 1 day × month evidence applies (same formula) |

`euribor3m_local_rate` is computed in both implementations but consumed by no stage. Stage 2's and Stage 3's
implementations agree on these values in-sample (Commit 2 parity). Each implementation is audited on its own paths.

**Frozen thresholds** (Cell 20, `provenance_table`). The nine cut points of the 12 Stage 2 bins are fixed constants:
when a row is transformed, they add no label input. Their provenance is separate. They were chosen in the Cell 13C
lift analysis with all training labels (PR 39 Step 3A: NSD and ECI revalidated, outcome *retain*; DOW retained
without revalidation), so every fold reuses cut points selected with its held-out labels. That is a selection-level
dependence, not a per-row label input. Stage 1's `n_bins = 20` and `k = 100` ("tuned in Cell 10D") have no recorded
rows.

## 5. What the evidence establishes, and its limits

**Established, on the training side, with the accepted code and plans:**

- (b) is absent on every fold path of both arms (measured 0). That includes the black-box MLP's outer OOF folds.
- (a) is present in every fit, at most 0.84% of an encoding's label mass. For NSD it applies only to the reference
  rows; a few euribor rows at rounded bin edges are exceptions.
- The black-box MLP's early-stopping subset is not isolated from the encoders: its labels make up about 15% of
  both the optimization rows' and the validation rows' label mass.
- (c) is pervasive on every path and for every family, including the mappings that score the test split.

**Limits.**

- **What the contact order can resolve.** It orders periods by calendar, and days within a period by the
  documented source order. The weekday checks are consistent with that order but do not prove it. It cannot order
  contacts within a day: the same-day share (0.02–0.24) is unresolved.
- **Outcome timing is unknown.** The later share is therefore a lower bound on the label mass that was not
  available at contact. Same-day labels, and any earlier-day label whose outcome came later, may also have been
  unavailable.
- **ECI.** The share is over the absolute coefficients of the curvature operator: a measure of influence, not a
  convex weight.
- **What is not measured.** The audit establishes which labels enter each value. It does not measure their effect
  on predictions, metrics, thresholds, or the MLP's stopping epoch and AUC; that is Step 3B.
- **The test mapping** is characterised at the training rows' contact days. The test rows are a random 20% of the
  same timeline, but they are not read.
- **Stage 4** inherits these encodings through the upstream stage outputs and is not audited here.

**For Step 3B to act on:**

- Every family needs a past-only definition at the contact-day unit, with an explicit rule for same-day contacts
  and an outcome-lag parameter.
- The day × month key pools years: a past-only version must choose whether to keep pooling.
- The euribor bins and the NSD / ECI kernels act as period proxies, so the prevalence shift (3% to about 50%) will
  interact with any past-only encoding.
- An isolated MLP early-stopping split would exclude the whole subset from encoder (and scaler) fitting.
- The nine Stage 2 cut points were chosen on all training labels (with Commit 4).

## Evidence locations

| Claim | Notebook | Record key (`encoding_audit.json`) | Code | Tests |
|---|---|---|---|---|
| Source order, periods, days | Cell 14 | `time_index` | `verify_source_order`, `build_time_index`, `time_index_checks` | `test_pr42_encoding_audit.py` (time index, source check); realdata |
| Construction, families | Cell 15 | `construction` | `CONSTRUCTION`, `DESCENDANTS`, `stage1_construction` | — |
| Paths and coincidences | Cell 16 | `fitting_paths` | `fit_paths`, `plan_checks`, `router_plan_reference` | plan test; realdata fold 0 |
| (b) flips, call sites, accepted fits | Cell 17 | `isolation.checks`, `isolation.accepted_fits` | `isolation_checks`, `label_argument_checks`, `accepted_fit_scopes` | pass and leaky-builder tests; call-site test |
| MLP early stopping | Cell 17 | `isolation.mlp_early_stopping`, `findings.mlp_early_stopping` | `mlp_early_stopping`, `MLP_EARLY_STOPPING_ORDER`, `MLP_ISOLATED_SPLIT`, `MLP_OUTER_ISOLATION` | split-reproduction test |
| (a), (b) measured, (c) by path and role | Cells 18–19 | `attribution` (`summary`, `fits`, `held_out_own_labels`, `final_mapping_by_period`) | `attribute`, `audit_paths`, `path_summary`, `held_out_own_labels` | exact relation counts; deliberate-leak regressions (Stage 1, Stage 2 label-excluded); realdata Stage 1 counts |
| Reconstruction coverage | Cell 18 | `attribution.coverage` | `COMPARISON_POLICY`, `comparison_coverage`, `_compare` | nonfinite, missing-comparison and product regressions |
| Thresholds and constants | Cell 20 | `provenance` | `provenance_table` | provenance test |
| Per-path and per-family findings | Cell 20 | `findings.by_path`, `findings.by_family` | `findings_table`, `family_findings` | findings test |
| Commit 2 inventory reproduced | Cell 21 | `commit2_inventory` | `research_records.compare_records` | record tests |
