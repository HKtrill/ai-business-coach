# PR 42 Commit 5, Step 5A — Feature analysis and Stage 3 simplification

**Question.** Can the frozen PR 39 Stage 3 EBM (13 features, three pairs) be simplified so that its contributions are
easier to explain and more stable across folds and contact periods? What does each simplification gain or cost in
out-of-fold predictive performance, calibration, and its ability to correct Stage 1 mistakes, especially on rows
where Stage 2 abstains?

**In brief.** All four registered candidates stay inside the registered performance and calibration tolerances.
Two of them also show a documented interpretability gain, and those two are recommended for Step 5B reselection.

| Configuration | Features / pairs | Pair lost | ROC-AUC Δ [95% CI] | PR-AUC Δ | Nested F2 Δ | Calibrated Brier / ECE | Stability of the changed terms (fold sd, log-odds) | Step 5B |
|---|---|---|---|---|---|---|---|---|
| baseline (refitted) | 13 / 3 | — | 0.7933 | 0.4170 | 0.5508 | 0.0796 / 0.0019 | `decay_x_density` 0.513; period-level regime step sd 0.632 | reference |
| `drop_first`: drop `euribor3m_sigmoid_slope` | 12 / 2 | `euribor3m_sigmoid_slope × economic_curvature_intensity` | −0.0004 [−0.0009, 0.0002] | −0.0018 | +0.0004 | 0.0797 / 0.0018 | kept slope 0.051 (step sd 0.073); regime step sd 0.755 | **recommended** |
| `drop_second`: drop `emp_var_rate_sigmoid_slope` | 12 / 2 | `campaign × emp_var_rate_sigmoid_slope` (TIER 1) | −0.0013 [−0.0023, −0.0003] | −0.0020 | +0.0007 | 0.0797 / 0.0021 | kept slope 0.107 (step sd 0.111) | not recommended |
| `jed_alone`: `decay_x_density` → `joint_economic_decay` | 13 / 2 | `prior_x_stress × decay_x_density` | +0.0001 [−0.0004, 0.0005] | +0.0006 | +0.0010 | 0.0796 / 0.0027 | `joint_economic_decay` 0.085; regime step sd 0.064 | **recommended** |
| `past_density`: `decay_x_density` → past-only variant | 13 / 2 | `prior_x_stress × decay_x_density` | +0.0005 [−0.0005, 0.0014] | +0.0017 | −0.0011 | 0.0796 / 0.0025 | `decay_x_density_past` 0.172; regime step sd 0.071 | not recommended (PR 49) |

- **`decay_x_density` is the least stable term, because of one fold.** In the baseline, fold 0's model gives
  `decay_x_density` a step about three times the size the other folds give it: −2.2 log-odds at zero density
  against −0.7 to −0.2, and +1.1 to +2.0 above it. That fold also puts 2.75 log-odds of 2009–10 vs 2008 regime step
  on the period-level terms, against 1.1–1.6 in the other folds. Without `decay_x_density`, the baseline's other
  period-level terms agree across folds (sd 0.034), so the excursion is that one term's. The whole additive score
  shows the same thing on both platforms: fold sd 0.604 in the sandbox and 0.603 in the accepted Windows run, with
  fold 0 at 3.32 in both. `joint_economic_decay`, the label-free macro factor of `decay_x_density` (r = 0.9935),
  removes this. The slot's fold sd falls from 0.513 to 0.085, and the period-level regime step agrees across folds
  (sd 0.064). The past-only variant does almost as well (0.172 and 0.071).
- **The fold offsets of the two correlated slopes are real shape differences, not centring.** Re-centring each fold
  curve over well-supported values moves it by at most 0.0085 log-odds (the five shifts span 0.0135) and leaves the
  fold sd unchanged (0.097 and 0.0465). The folds disagree about how large each slope's regime step is: euribor
  slope 0.06–0.32, emp.var.rate slope 0.17 in four folds and 0.34 in fold 0. Across the five folds their steps correlate at −0.47. With five
  points that is descriptive, not a test, but it is consistent with the two correlated slopes sharing one regime
  signal that each fold divides differently (section 6.2).
- **Only one of the three accepted pairs matters inside the joint support.** On support-defined cells (identical
  on both platforms), `campaign × emp_var_rate_sigmoid_slope` spans 0.81–0.87 log-odds across its well-supported
  cells, with fold sd at most about 0.06. Repeated contacts (campaign 6 and above) at emp.var.rate slopes 0.487–0.559
  score down to −0.62 in the Windows fit and −0.67 in the sandbox fit. The other two pairs carry at most about 0.05
  and 0.07 log-odds in any well-supported cell. For `euribor3m_sigmoid_slope × economic_curvature_intensity` the fold
  sd reaches about 0.2, four times its largest effect, so that surface is noise-level. The exact sizes come from
  each platform's fits and differ by up to 0.05, about the pair's own fold spread. The large values on the
  registered regular grid (up to 1.76) sat in cells with few or no rows (section 7).
- **The past-only variant is past-only for 97.2% of held-out rows, not all.** 32,033 held-out rows get a raw
  density built from earlier contacts only. The 917 held-out rows contacted on days 0–2 (2.8%) have no history and
  take the production cleaner's fallback instead: the median of `decay_x_density_past` over their fold's whole fit
  side, which depends on labels of later contacts.
  - Flipping a tenth of the fit labels from day 3 on raises that fallback 3.1–3.6×. With the fitted fold models held
    fixed, those rows' term score rises by 0.07–0.16 log-odds.
  - The day-3 raw values, the control, do not move.
  - The candidate is kept as registered and fitted, and the prediction-time claim is qualified accordingly
    (section 4).
- **On Stage 2 abstentions, `jed_alone` keeps enough of the baseline's Stage 1 corrections; `drop_first` does not.**
  The accepted Windows run computed complementarity (Cell 57); the sandbox has no accepted artifacts. On the 14,628
  abstentions the refitted baseline's net corrections of Stage 1 are 530, and the registered cascade rule needs at
  least 456.86.
  - `jed_alone` nets 486 and is nominated.
  - `drop_first` nets 441 and is not: it corrects 568 Stage 1 errors there against the baseline's 665.
  - `drop_second` (633) and `past_density` (689) clear the rule but are not recommended for Step 5B.
  - The accepted EBM itself nets 447, so the comparisons are against the refitted baseline, as registered (section
    8).
- **Recommendation for Step 5B.**
  - Carry `jed_alone` and `drop_first` into reselection under the three-pair budget (the registered rule; the
    Windows results do not change it).
  - Only `jed_alone` is also nominated for the cascade comparisons of PRs 44–45. `drop_first` goes into Step 5B
    with its abstention shortfall stated, and its reselection would have to recover at least 16 net corrections.
  - Both keep `campaign × emp_var_rate_sigmoid_slope` as the anchor pair and free one slot. The other pair each
    keeps is noise-level inside the joint support.
  - Their combination (12 features, one pair, two open slots) is suggested by these results and is **exploratory**
    (not fitted here).
  - `drop_second` loses the one substantive pair. `past_density` is more stable than the baseline but not simpler;
    its case is temporal validity, which PR 49 tests (section 9).

**Scope.** This is training-side work on the 32,950 training rows, and the test split is never bound (G7). The frozen
PR 39 artifacts and the accepted records are read, never changed, and accepted artifacts are opened only through the
PR 38 boundary extractors. No pair is reselected; that is Step 5B. No cascade is compared; that is PRs 44–45. No
chronological split is used; that is PR 49.

**Evidence.**

- *Where it is produced.* `feature_research/feature_research.ipynb`, Cells 49–59 (module
  `feature_research.stage3_simplification`), after the Commit 2–4 cells in the same run. The notebook writes
  `research_logs/pr42/<run_id>/stage3_simplification.json` beside the run's five earlier records.
- *Which run.* Unless marked otherwise, the numbers below come from this revision's sandbox run (Linux, Python
  3.13; a Commit 5 run, `c5_…`).
  - Complementarity, row-aligned parity and the values marked Windows come from the accepted Windows run of the
    same code (`c5_20261123_220648`, stopped at Cell 58 by the three statements this revision corrects).
  - Cell 58 requires every figure and verdict in this document to hold in the run that writes the record
    (`SS.STATED`: 90 checks in the sandbox; 100 in the Windows run, where 11 exact complementarity statements replace
    the one marked not checkable).
  - Data-determined quantities (rows, folds, cells, support, fallback rows, Windows counts) must hold exactly.
    Fit-dependent values must hold within the tolerance stated beside each, because fits differ slightly between
    platforms. Where that difference exceeds a precise statement (the pair sizes), the document gives both
    platforms' values and checks the claim they support.
- *Reuse, not new code paths.* Every configuration is fitted by the PR 39 Stage 3 path:
  - `Stage3Block` folds and per-fold features;
  - `OOFGenerator` outer and inner out-of-fold scores;
  - `Stage3Calibrator` fold-nested isotonic calibration;
  - `F2ThresholdSelector` global and fold-nested thresholds;
  - `EBMFactory` with the accepted settings.

  The past-only density reuses the production kernel. The module imports neither feature-engineering module named
  in the Commit 2 caller scan, so that scan and Cell 30's Step 3A reproduction are unchanged.
- *Tests.* `tests/feature_research/test_pr42_stage3_simplification.py` (32, synthetic) covers:
  - imports, the registration and the baseline's sources;
  - pair mapping after removal and replacement, and identical configurations fitted once;
  - the past-only kernel, same-day and later-label exclusion, and cold start;
  - the fallback: a later fit-side label moves an earlier row's fallback but never an earlier raw value, and the
    trace finds both forms in the EBM frames;
  - the accepted views: missing (the sandbox path, and an error when required), a present artifact failing its pin
    or its boundary check (an error naming the role), the requirement switch, and Cell 57's requirement checks;
  - whole-fit fold isolation with a failing control, and fitted terms;
  - the paired metrics, shapes, pair cells and centring check;
  - complementarity reconciliation, the nomination rules and the notebook layout.

  `tests/realdata/test_pr42_stage3_simplification.py` checks the real rows: the registration, block, past-density,
  correlation and configuration checks, the fallback trace and its stated counts, and the baseline's parity with the
  records (row by row, and required with `PR42_REQUIRE_ACCEPTED_VIEWS=1`).

## 1. The registration

`SS.REGISTRATION` (id `pr42-step5a-registration`, version 1.0.0, digest `08b3a744…09bc`) was fixed before any
candidate was fitted, and Cell 49 checks the digest first. It records:

- the baseline and where each part comes from:
  - features from `EBM_FEATURES`, checked against the PR 39 Commit 4 record;
  - pairs from `define_ebm_interactions`;
  - hyperparameters transcribed from `glass_cascade` Cell 18 at commit `0062625f`, blob `a366cc40`;
  - the fold plan (fingerprint `8aea082fcaf371b7`) and the calibration and threshold protocol;
- the correlated-pair rule and the four candidates, with the pairs each must lose;
- the past-only density's definition, outcome-availability assumption and cold start;
- the evaluation (same rows, folds, seeds, settings and calibration; no retuning);
- the metrics and paired differences;
- the shape and interaction conventions;
- the complementarity counts;
- the Step 5B and cascade nomination rules.

Only one thing was known beforehand: the baseline refit alone had been run once, for parity (section 2). Any
simplification suggested by the results is labelled exploratory.

The Step 5B rule recommends a candidate when all of the following hold:

- ΔROC-AUC ≥ −0.005, with a 95% lower bound ≥ −0.010;
- ΔPR-AUC ≥ −0.010;
- Δnested F2 ≥ −0.010;
- calibrated ECE ≤ 0.05;
- calibrated Brier ≤ the baseline's + 0.002;
- its shapes give a documented interpretability gain.

The cascade rule then requires, on Stage 2 abstentions, a net count of at least the baseline's net minus 0.5% of
the abstention rows.

Four additions were made after the registration and the first results. None changes a fit, metric, threshold or
ranking, and all are listed in the record (`post_registration`):

- the accepted views are checked before any fit, failing artifacts stop the run, and the acceptance run can require
  all three (section 8);
- the past-only fallback is traced to the EBM's frames and its dependence on later labels measured (section 4);
- pair surfaces are drawn on support-defined cells (section 7);
- the slope offsets are re-examined by a centring check (section 6.2).

The registered centring remains the reported one.

## 2. The baseline and its parity

The refitted baseline reproduces the accepted Stage 3 EBM closely, but not exactly:

| Quantity | Accepted (PR 39 records) | Refitted (sandbox) | Refitted (Windows) |
|---|---|---|---|
| out-of-fold ECE, raw → calibrated | 0.27109 → 0.00232 | 0.27111 → 0.00194 | 0.27110 → 0.00212 |
| global raw F2 threshold | 0.53 | 0.53 | 0.53 |
| global calibrated threshold | 0.11 | 0.12 | 0.11 |
| fold-nested thresholds | 0.56, 0.56, 0.54, 0.49, 0.49 | 0.55, 0.56, 0.54, 0.50, 0.51 | 0.55, 0.56, 0.54, 0.51, 0.51 |
| ROC-AUC / F2 at 0.53 / nested F2 | 0.7933 / 0.5535 / 0.5503 | 0.7933 / 0.5533 / 0.5508 | 0.7933 / 0.5528 / 0.5509 |

The differences come from the learning rate, which is transcribed from the only tracked source where it is printed
to six decimals, and from the platform. Where the accepted EBM artifact exists (the Windows run), Cell 52 also
compares the out-of-fold scores row by row:
- the raw scores correlate at 0.999991 with the accepted EBM's, with a largest difference of 0.022;
- the calibrated probabilities differ by up to 0.11 on some rows, where an isotonic step moves.

The refitted baseline's decisions therefore differ somewhat from the accepted EBM's (section 8). Every candidate is compared with
the refitted baseline, which uses the same code, platform and settings, never with the artifact's vectors.

The refitted baseline's term importances (mean |term score|, all-rows fit) put `decay_x_density` first (0.413),
then:

- `dow_month_encoded` 0.307;
- `cpi_high_cellular` 0.264;
- the two slopes, 0.183 and 0.165;
- the three pairs, 0.047, 0.011 and 0.010.

The two slopes vary most across folds (coefficient of variation 0.24 and 0.37).

## 3. The remaining correlated pair

The frozen inventory's pruning note names `euribor3m_sigmoid_slope` and `emp_var_rate_sigmoid_slope`. On the
training rows their Pearson r is 0.725 (0.722–0.728 on the five outer fit sides). It is the only pair of the 13
features at |r| ≥ 0.7, as the registration requires; the next is 0.675.

Both slopes are period-level: each contact day has one value, and most values belong to one contact period. Both
encode the 2008 versus 2009–10 macro regime. The two candidates `drop_first` and `drop_second` remove one member
each.

## 4. Candidates and the pairs each loses

Pairs are resolved by feature name. A candidate keeps a baseline pair only if it keeps both features, a lost pair's
slot stays empty, and a replacement feature does not inherit its predecessor's pair. Each fitted EBM's terms were
checked to be exactly its features plus its pairs. interpret orders pair terms itself, so pairs are compared as a
set.

| Candidate | Removes | Adds | Features | Pairs kept | Pair lost |
|---|---|---|---|---|---|
| `drop_first` | `euribor3m_sigmoid_slope` | — | 12 | `campaign × emp_var_rate_sigmoid_slope`, `prior_x_stress × decay_x_density` | `euribor3m_sigmoid_slope × economic_curvature_intensity` |
| `drop_second` | `emp_var_rate_sigmoid_slope` | — | 12 | `euribor3m_sigmoid_slope × economic_curvature_intensity`, `prior_x_stress × decay_x_density` | `campaign × emp_var_rate_sigmoid_slope` |
| `jed_alone` | `decay_x_density` | `joint_economic_decay` | 13 | `campaign × emp_var_rate_sigmoid_slope`, `euribor3m_sigmoid_slope × economic_curvature_intensity` | `prior_x_stress × decay_x_density` |
| `past_density` | `decay_x_density` | `decay_x_density_past` | 13 | the same two | `prior_x_stress × decay_x_density` |

`joint_economic_decay` is not among the 13 features, so `jed_alone` adds it once and keeps 13 features. No two
candidates coincide, so each is fitted once. Every difference below measures the whole simplification. Removing a
feature also removes its pair, and replacing `decay_x_density` also removes `prior_x_stress × decay_x_density`.

**The past-only density.** `decay_x_density_past = joint_economic_decay × nsd_past`. `nsd_past` is the accepted NSD
kernel with one change: for a row contacted on day *d*, it averages the labels of the context's rows contacted
before *d*. The scaler is fitted on those rows and every one of them is kept in the reference, so the history is
grown, not redrawn. Same-day labels are never used. The rest of the definition:

- *Outcome availability.* It assumes an outcome is known by the next contact day (Step 3B's lag-0 rule). The
  outcome time is not recorded.
- *Cold start.* A row has no raw value before day 3, or while its history holds fewer than 500 rows or 10
  positives. Over all rows that is the 917 rows contacted on days 0–2, and the first valued day is 3.
- *Contexts.* Each outer fit side and each inner fit side is a context.
- *Checks (Cell 50).* The production `IntegralFeatureEngineer` fitted on exactly the available rows gives the same
  values on sampled days (maximum difference 1e-14). Flipping held-out labels, or labels of later days, changes no
  raw value, and each flip has a control that does change.

**Raw density and fallback.** The final feature the EBM receives, `decay_x_density_past`, takes one of two forms
(`SS.fallback_trace`, Cell 50). Both are verified in the frames the EBM is fitted on and scores.

| Form | Value | Labels it uses | Held-out rows | Fit rows per fold |
|---|---|---|---|---|
| earlier-day raw density | `joint_economic_decay` × the kernel average of its context's rows contacted before its day | earlier contacts only | 32,033 (97.2%) | 25,620–25,634 |
| full-fit-fold median fallback | the production cleaner's median of `decay_x_density_past` over the fold's whole fit side (about 2.8e-06 in every fold) | fit-side contacts through day 483 (the latest valued fit row is on day 484) | 917 (2.8%), contacted on days 0–2 (177–191 per fold) | 726–740 |

The inner contexts add 4,374 held-out fallback rows, which feed the fold-nested calibration and thresholds.

The fallback is the value at the middle of the fit-side ordering. That middle falls on rows contacted on days
10–14, whose values use labels from days after the fallback rows', and the ordering depends on every valued fit
row. The trace flips fit labels from the first held-out day with a raw value (day 3) onwards. No raw value up to
that day can use those labels, and all of them come after every fallback row's day. Results:

| Perturbation, per fold | Fit labels flipped | Fallback moves | Term score change of the fallback rows (fold model held fixed) | Day-3 raw values (339 held-out rows) |
|---|---|---|---|---|
| a tenth of the labels from day 3 on | about 2,560 | in every fold, 3.1–3.6× | +0.07 to +0.16 log-odds | unchanged |
| every label from day 3 on | about 25,600 | in every fold, 25.7–26.6× | +0.09 to +0.28 log-odds | unchanged |

**What this means for the prediction-time claim.** The raw density is feasible at prediction time under the lag-0
assumption: a row's value uses outcomes of earlier contacts only. The fallback is not. For the 917 held-out rows it
fills, a later fit-side label can change the value, and the change reaches the fitted model's score. The registered
candidate is evaluated as fitted, so its out-of-fold metrics include those rows with that fallback. A deployable
version would need a fallback fixed from earlier data only, such as the median of values available by that day or a
constant set before the first contact. PR 49 would test that version, not this one.

## 5. Paired out-of-fold comparison and calibration

All configurations share the baseline's outer folds, inner folds, rows, seeds, settings and calibration protocol.
Each candidate is compared with the baseline on the same 32,950 out-of-fold rows.

The ROC-AUC difference uses DeLong's paired standard error. The other differences are pooled, with per-fold means
and spreads in Cell 53. Brier and ECE use the same 10-bin rule and are labelled raw or calibrated; the calibration
is fold-nested isotonic.

| Configuration | ROC-AUC | Δ [95% CI], p | PR-AUC (Δ) | Nested F2 (Δ) | Brier raw → calibrated | ECE raw → calibrated | Decisions ≠ baseline |
|---|---|---|---|---|---|---|---|
| baseline | 0.7933 | — | 0.4170 | 0.5508 | 0.1622 → 0.0796 | 0.2711 → 0.0019 | — |
| `drop_first` | 0.7930 | −0.0004 [−0.0009, 0.0002], p 0.21 | 0.4152 (−0.0018) | 0.5513 (+0.0004) | 0.1625 → 0.0797 | 0.2718 → 0.0018 | 420 |
| `drop_second` | 0.7920 | −0.0013 [−0.0023, −0.0003], p 0.011 | 0.4149 (−0.0020) | 0.5515 (+0.0007) | 0.1625 → 0.0797 | 0.2716 → 0.0021 | 428 |
| `jed_alone` | 0.7934 | +0.0001 [−0.0004, 0.0005], p 0.73 | 0.4176 (+0.0006) | 0.5518 (+0.0010) | 0.1623 → 0.0796 | 0.2712 → 0.0027 | 162 |
| `past_density` | 0.7938 | +0.0005 [−0.0005, 0.0014], p 0.31 | 0.4186 (+0.0017) | 0.5497 (−0.0011) | 0.1620 → 0.0796 | 0.2707 → 0.0025 | 499 |

How the candidates compare:

- Every candidate is inside the registered tolerances.
- `drop_second` is the only one whose ROC-AUC interval excludes zero.
- `past_density` has the widest per-fold nested-F2 spread (sd 0.0087; per fold −0.0096 to +0.0109).
- `jed_alone` moves the fewest decisions: 117 rows newly flagged and 45 no longer flagged.
- Calibrated ECE rises slightly for `jed_alone` (0.0019 → 0.0027) and `past_density` (0.0025), well under the 0.05
  gate.

The global raw and calibrated thresholds are chosen on the rows they are applied to (in-selection). The fold-nested
thresholds are the primary operating point.

## 6. Shapes, folds and contact periods

### 6.1 How to read the shape figures

Each figure in Cell 55 shows one feature across every configuration that has it, on matched axes.

- *Units.* The vertical axis is the EBM's additive term score in log-odds. The calibrated probability is a separate,
  monotone map of the summed score and is never plotted as a term.
- *Curves.* Thin lines are the five outer-fold models and the thick line the all-rows fit. Each is evaluated on a
  common grid and shifted so its mean over the 32,950 training rows is 0 (the registered centring). The EBM centres
  each term on its own fit rows; the shift is recorded.
- *Support.* Grey bands mark sparse bins (under 1% of the rows). The panel below stacks training rows by contact
  year, and hatches bins where one contact period holds more than half the rows.

For a period-level feature a shape point is evidence about the period that holds it. The 2008 versus 2009–10 step
compares periods.

### 6.2 The fold offsets of the correlated slopes (post-registration centring check)

The fold curves of `euribor3m_sigmoid_slope` and `emp_var_rate_sigmoid_slope` look vertically offset from one
another. `SS.centring_check` (Cell 55) asks whether that is an artefact of centring over all rows. It changes no
fit, metric or ranking.

- **Re-centring does not remove it.** Each fold curve is re-centred over the rows in well-supported bins only.
  - *Euribor slope.* The largest absolute shift is 0.0085 log-odds (fold 0), and the five shifts span 0.0135 across
    folds.
  - *Emp.var.rate slope.* The largest shift is 0.0002 (span 0.0001).
  - *Spread.* The row-level fold sd is unchanged: 0.097 and 0.0465 under either centring. (No figure plots the
    shifts; they are in Cell 55's table and the record.)
- **It is a difference in step size.** The step is the mean contribution of the 2009–10 rows minus that of the 2008
  rows, and adding a constant to a curve does not change it. In the baseline the euribor slope's step is 0.12, 0.29,
  0.32, 0.06 and 0.29 across folds (sd 0.119). The emp.var.rate slope's is 0.34 in fold 0 and 0.17 in the others
  (sd 0.077). Removing the euribor slope does not fix fold 0's emp.var.rate step (`drop_first`: 0.37 against
  0.20).
- **The two slopes appear to share one regime signal.** Across the five folds their steps correlate at −0.47, and
  their row-level fold deviations at −0.33. With five fold steps this is descriptive evidence, not a test. It is
  consistent with each fold dividing one regime signal between the two correlated slopes differently.
- **The larger disagreement is in the decay slot.** Summed over every period-level main term, the regime step's fold
  sd is:
  - 0.632 in the baseline, 0.755 in `drop_first` and 0.638 in `drop_second`, because fold 0 puts about 2.7–3.2
    log-odds on it against 1.0–1.6 in the other folds;
  - 0.064 in `jed_alone` and 0.071 in `past_density`.

  Without the decay slot, the baseline's period-level terms agree across folds (sd 0.034). Fold 0 gives
  `decay_x_density` itself a step of 2.19 against 0.61–1.00 in the other folds, and the remaining terms 0.51–0.59 in
  every fold. The whole additive score behaves the same way (0.604 against 0.063). Folds disagree about the regime
  effect only when `decay_x_density` is in the model.
- **Which terms are period-level, and the same result on Windows.** A term is period-level when its feature is
  constant within every contact day up to floating-point noise (a range of at most 1e-9 relative, the rule the pair
  cells use for duplicate values; Cell 55 lists every feature's largest within-day range). The first Windows run
  required exact equality. Within a day every row of `decay_x_density` has the same inputs and the same reference,
  but its NSD factor is a kernel row sum and matrix-vector product whose last bits can differ by a row's position.
  So Windows left `decay_x_density` out, and the Windows "period-level" numbers were in fact the without-the-decay-slot
  ones (0.032, 0.125, 0.046). Recomputed here with that membership they come out at 0.034, 0.125 and 0.046. The
  whole additive score, which needs no membership, reproduces on Windows: fold sd 0.603 against 0.604, fold 0 at
  3.317 against 3.322, and every fold within 0.02. The real spreads separate cleanly: period-level features have a
  within-day range of 0 here, and the smallest range among the others is 0.7.

### 6.3 Each changed feature

- **`euribor3m_sigmoid_slope`.** A two-regime step. It scores −0.12 to −0.26 for the 2008 rows (slope below 0.04,
  21,971 rows), rises from −0.20 at 0.53 to about +0.1 between 0.73 and 0.93, and reaches +0.33 at the top value
  (2009–10). The fold models agree on direction, not on the step's size, and disagree most at the two ends: −0.35 to
  +0.05 at the lowest value and +0.11 to +0.56 at the top. 8 of its 12 well-supported grid points are
  single-period. Dense fold sd: 0.119 (baseline), 0.107
  (`drop_second`), 0.111 (`jed_alone`), 0.135 (`past_density`).
- **`emp_var_rate_sigmoid_slope`.** Values up to 0.058 score −0.1 to −0.2 and the rest +0.1 to +0.3. Fold 0 gives
  a larger step in the baseline and in `drop_first`. In `jed_alone` the top value (0.99, 519 rows, late 2010) turns
  negative: −0.61 in the all-rows fit, −0.13 to −0.44 in the fold models. It takes over part of the late-2010
  decline that `decay_x_density` carried. It is most stable in `past_density` (fold sd 0.016).
- **`decay_x_density` (baseline).** Rises from about −0.1 at zero density (2008, 21,991 rows in the lowest grid
  bin) to a plateau of +0.6 to +1.3, and falls below 0 above 0.13 (sparse, 2010). Fold 0 departs from the others
  everywhere. Its fold sd over well-supported values, 0.513, is the largest of any term. Its out-of-fold
  contribution is about average through 2008 (0.8–1.1 times its mean), below it in March–May 2009 (0.4–0.7 times),
  and above it from June 2009 on (1.2–3.0 times) (Cell 54).
- **`joint_economic_decay` (`jed_alone`).** A monotone rise in three levels:
  - −0.13 to −0.04 for the 2008 rows (values up to 0.036);
  - +0.47 to +0.66 for 0.07 to 0.09;
  - +0.75 to +1.09 for 0.32 to 0.39.

  The fold curves stay close (fold sd 0.085), and the slope changes direction twice over well-supported values. It
  uses no labels; its only inputs are the three macro series.
- **`decay_x_density_past` (`past_density`).** Not monotone. It scores −0.58 at zero past density and +0.76 just
  above it, falls to +0.18 at 0.008–0.010, then rises to +1.25 at 0.116 before falling to +0.92. That is 5
  direction changes over well-supported values. The folds disagree most at low values (0.01 to 0.82), and
  cold-start rows carry the imputed median. Fold sd 0.172.

For the other main effects, the fold sd over well-supported values changes by at most 0.021 between
configurations (Cell 55's summary table). Their ranges move where they share the regime effect with a changed term.
`cons.conf.idx` goes from 0.92 to 1.16 in `drop_first` and to 0.79 in `jed_alone`, and
`economic_curvature_intensity` from 1.12 to 0.69 in `past_density`.

## 7. Retained and lost interactions

Each pair is drawn on support-defined cells (Cell 56), the same cells in every panel.

- *Cells.* Every distinct value, or the distinct 5% quantiles for a feature with more than 30 values. A cell's value
  is the mean pair-term contribution of its training rows, in log-odds, centred to mean 0 over the training rows.
- *Empty and sparse cells.* Empty cells stay empty, so nothing is drawn outside the joint support. A cross marks
  cells with under 0.1% of the rows.
- *Panels.* The top row shows joint support, then each model's all-rows score on one symmetric scale per pair. The
  bottom row shows the largest contact period's share per cell, then each model's per-cell sd across its five fold
  models.

The registered regular grid had left almost every cell empty, so its colour scale was set by extrapolation into
cells without rows.

The cells and their support are fixed by the data and are identical on both platforms, cell by cell. The cell
means come from each platform's fits. Values are sandbox / Windows:

| Pair | Well-supported cells (rows in sparse cells) | Largest effect (baseline) | Range (baseline) | Largest fold sd | In candidates (largest effect) |
|---|---|---|---|---|---|
| `campaign × emp_var_rate_sigmoid_slope` (TIER 1) | 39 of 56 (236) | 0.671 / 0.620 | 0.871 / 0.813 | 0.049 / 0.057 | kept by `drop_first` 0.58 / 0.59, `jed_alone` 0.56 / 0.56, `past_density` 0.59 / 0.64; lost by `drop_second` |
| `euribor3m_sigmoid_slope × economic_curvature_intensity` | 44 of 360, 50 with rows (72) | 0.048 / 0.051 | 0.078 / 0.083 | 0.202 / 0.206 | kept by `drop_second`, `jed_alone`, `past_density`; lost by `drop_first` |
| `prior_x_stress × decay_x_density` | 30 of 80, 31 with rows (20) | 0.068 / 0.072 | 0.086 / 0.089 | 0.106 / 0.100 | kept by `drop_first`, `drop_second`; lost by `jed_alone`, `past_density` |

The two platforms' fits differ by up to 0.05 in a pair's largest cell (0.671 against 0.620 for the TIER 1 pair),
about that pair's own fold-to-fold spread in its supported cells. So the third decimal of a pair's size is a
property of the platform's fit, not of the data. The statements Cell 58 checks are therefore the claims both
platforms support:
- the cells hold exactly;
- the TIER 1 pair's largest supported effect is at least 0.5 and its range at least 0.7, with fold sd at most 0.1;
- the other two pairs' largest effects are under 0.1;
- the euribor × curvature pair's fold sd is at least twice its largest effect (about four times on both).

- **`campaign × emp_var_rate_sigmoid_slope`** is the only substantive pair. Repeated contacts (campaign 6 and above)
  score from about −0.4 down to −0.67 (sandbox) or −0.62 (Windows) log-odds beyond the main effects at emp.var.rate
  slopes of 0.487 and 0.559. The strongest cell, campaign 8 and above at slope 0.559, has 146 rows on both
  platforms. The 0.487 slope is
  October–November 2008, and 0.559 is March–May 2009 and March–May 2010. At higher slopes such cells are sparse. The
  pattern is the same in every model that keeps the pair. Its cells are mostly single-period, because each slope
  value covers few periods, but the pattern appears in two separate regimes.
- **`euribor3m_sigmoid_slope × economic_curvature_intensity`** has rows only along a diagonal of macro states. Its
  well-supported effect is at most about 0.05 log-odds while its fold sd reaches about 0.2, so it is noise-level.
  `drop_first` loses little by dropping it.
- **`prior_x_stress × decay_x_density`** is at most about 0.07 log-odds in a well-supported cell. In the sandbox fit
  the largest effect is −0.07 at `prior_x_stress` 0.42–0.67 with `decay_x_density` 0.0074–0.0075 (577 rows). `jed_alone` and
  `past_density` lose little by dropping it.

## 8. Complementarity with Stage 1

Cell 57 computes this only where the accepted Stage 1 LR, GLASS router and EBM views all pass. They do in the
accepted Windows run, not in the sandbox. Cell 49 reads the views before any fit, through the PR 38 boundary
extractors at the PR 39 pins.

- An artifact whose file is absent is reported as not available locally. That is the sandbox path, where
  complementarity is not computed.
- An artifact that is present but fails its pin, its boundary extraction or the training rows' labels stops the run
  with its role and the reason. A present artifact is never skipped.
- With `PR42_REQUIRE_ACCEPTED_VIEWS=1` (the acceptance run), all three views must pass. Cell 57 then also requires
  (`SS.complementarity_requirement`) that complementarity was computed for the accepted EBM and every
  configuration, each with an all-rows and a Stage 2 abstention table, reconciled. A completed acceptance run
  therefore has both tables.

Every EBM is aligned with Stage 1 by row identity on the 32,950 out-of-fold rows, each with a stated decision
source:

- Stage 1: `lr.in_stage_f2_cv`, the calibrated LR probability at its stored threshold.
- The accepted EBM: its stored `ebm.nested_f2` decisions.
- Each refitted EBM: its raw out-of-fold score at its own fold-nested thresholds.
- Stage 2 abstentions: the accepted router's out-of-fold uncertain rows.

For all rows and for abstentions, each EBM gets the four paired counts (both correct, Stage 1 only correct, EBM
only correct, both wrong), each with its denominator and reconciled to its population. Each candidate is then
compared with the refitted baseline:

- Stage 1 errors it corrects;
- correct Stage 1 decisions it overturns;
- the net, and the registered cascade rule's verdict.

A disagreement count does not establish a better cascade decision. PRs 44–45 compare cascades at comparable
coverage.

**The accepted Windows run (Cell 57).** All three views passed with `PR42_REQUIRE_ACCEPTED_VIEWS=1`. Every table
reconciles to its population: 32,950 rows with 3,712 positives, and 14,628 Stage 2 abstentions with 1,120
positives. On the abstentions, "corrects" counts the Stage 1 errors an EBM gets right, "overturns" the correct
Stage 1 decisions it gets wrong, and net is corrects minus overturns:

| EBM | Net, all rows | Corrects, abstentions | Overturns, abstentions | Net, abstentions | Δ net vs refitted baseline | Cascade (net ≥ 456.86) |
|---|---|---|---|---|---|---|
| accepted EBM (stored decisions) | 1,317 | 589 | 142 | 447 | −83 | — |
| refitted baseline | 1,459 | 665 | 135 | 530 | — | — |
| `drop_first` | 1,292 | 568 | 127 | 441 | −89 | **not nominated** |
| `drop_second` | 1,567 | 781 | 148 | 633 | +103 | nominated |
| `jed_alone` | 1,370 | 618 | 132 | 486 | −44 | **nominated** |
| `past_density` | 1,618 | 852 | 163 | 689 | +159 | nominated |

The threshold is the refitted baseline's net minus 0.5% of the 14,628 abstentions (530 − 73.14).

- The accepted EBM's own stored decisions net 83 less than the refitted baseline on the abstentions. That is the
  decision-level counterpart of the parity differences in section 2, and why every candidate is compared with the
  refitted baseline, as registered.
- `drop_second` and `past_density` correct more Stage 1 errors than the baseline does. Neither is recommended for
  Step 5B, and a higher net is not a cascade improvement until PRs 44–45 measure it at comparable coverage.

In the sandbox the cascade column reads "not evaluated". There, each candidate changes 162 (`jed_alone`) to 499
(`past_density`) of the baseline's 32,950 fold-nested decisions.

## 9. Recommendation for Step 5B

The registered rule (section 1) recommends **`jed_alone`** and **`drop_first`**. Both are inside the tolerances,
and both have a documented interpretability gain (`SS.INTERPRETATION`). Step 5B owns reselection under the existing
three-pair budget.

- **`jed_alone` (primary).**
  - *Gain:* it replaces the least stable, label-derived term with a label-free macro factor, and the decay slot
    becomes fold-stable (0.513 → 0.085). The regime effect is then identified consistently across folds (step sd
    0.632 → 0.064). It loses a pair worth at most about 0.07 log-odds.
  - *Cost:* none measurable in ROC-AUC, PR-AUC or nested F2. Calibrated ECE rises 0.0019 → 0.0027, and the
    emp.var.rate slope's top value takes a negative score (−0.61) with fold disagreement. On Stage 2 abstentions
    its net Stage 1 corrections fall from 530 to 486 (Windows), within the cascade rule's allowance, so it is
    nominated.
  - *For 5B:* `campaign × emp_var_rate_sigmoid_slope` is the anchor. The euribor × curvature pair is noise-level,
    and one slot is empty.
- **`drop_first`.**
  - *Gain:* one fewer term in a correlated, period-concentrated pair, so the regime signal the two slopes appear
    to share is carried by one. It drops a noise-level pair.
  - *Cost:* ROC-AUC −0.0004 (interval includes 0), PR-AUC −0.0018. It does not fix `decay_x_density`'s fold-0
    excursion, and the regime step sd rises to 0.755.
  - *Cascade:* not nominated. On Stage 2 abstentions its net Stage 1 corrections are 441 (Windows), short of the
    456.86 the rule needs: it corrects 568 Stage 1 errors there against the baseline's 665.
  - *For 5B:* the TIER 1 pair is kept and `prior_x_stress × decay_x_density` is small, and one slot is empty. It
    stays recommended for reselection under the registered rule. A reselected `drop_first` would need its own
    complementarity check before any cascade comparison.
- **Exploratory, not fitted: `jed_alone` + `drop_first`.** This would have 12 features, keep only
  `campaign × emp_var_rate_sigmoid_slope`, and leave two slots open. The two gains act on different terms, and both
  pairs it would lose are noise-level inside the joint support. Step 5B may register it. Since `drop_first` alone
  falls short on the abstentions, the combination would need its own complementarity check.
- **Not recommended.**
  - `drop_second` removes the more stable slope and the only substantive pair, and its ROC-AUC interval excludes
    zero.
  - `past_density` meets the tolerances and is more stable than the baseline. It is not easier to explain: it has a
    non-monotone shape, a label-dependent history state and the lag-0 assumption. It is also not fully past-only:
    its 917 held-out fallback rows depend on later labels (section 4). `joint_economic_decay` gets the same
    stability without labels. Its value is temporal, which PR 49 would test with a past-only fallback.

Cascade nomination follows the registered rule on the Windows run's abstention counts (Cell 57). Only a candidate
recommended here and nominated there goes on to the cascade comparisons of PRs 44–45: **`jed_alone`**.
`drop_second` and `past_density` are nominated by the counts but not recommended, and `drop_first` is recommended
but not nominated.

## 10. Figures to inspect

In the executed notebook:

- Cell 55, the decay-slot figure: `decay_x_density` (fold 0 apart), `joint_economic_decay` and
  `decay_x_density_past` on one vertical axis.
- Cell 55, `euribor3m_sigmoid_slope` and `emp_var_rate_sigmoid_slope`: the fold offsets. Check the −0.61 top value in
  the `jed_alone` column.
- Cell 55, the last figure: each fold's regime step for the two slopes, the period-level terms with and without the
  decay slot, and the whole score.
- Cell 56, the three pair figures: support, scores on matched cells, the period share and the fold sd.
- Cell 54, the period figure: where each term's out-of-fold contribution is concentrated.
- Cell 57 (Windows only): the four paired counts, the abstention comparison and the cascade column.

## 11. Limits

- *Training side, random outer folds.* A simplification's value under a chronological split is PR 49's question.
- *Parity and platforms.* The baseline is refitted from a learning rate transcribed to six decimals. Candidates are
  compared with the refitted baseline, never with the artifact; on the abstentions the artifact's own decisions net
  83 fewer Stage 1 corrections.
  - Fit-dependent values differ slightly between the sandbox (Linux) and the accepted Windows run. Pooled metrics
    differ in the last digits; shape and pair summaries differ by up to about 0.05 log-odds.
  - Data-determined quantities are identical on both platforms.
  - Pair sizes are therefore given for both platforms and checked as claims.
- *Period-level membership.* "Constant within a contact day" allows floating-point noise (1e-9 relative), because
  a feature built by a numerical reduction (the NSD kernel) can differ in its last bits between rows of the same day
  on some platforms.
- *Attribution.* Each difference is the whole simplification's, feature and pair together. No pair is reselected,
  and no candidate is retuned.
- *The past-only density.* It assumes outcomes are known by the next contact day, and its history is grown, not
  redrawn. Only its raw values are past-only. The 917 held-out rows that take the full-fit-fold median fallback (and
  726–740 fit rows per fold, 4,374 inner held-out rows) depend on later labels. The candidate is evaluated as
  registered and fitted, not repaired.
- *Shapes.*
  - Shapes are evaluated on a common grid with sparse and single-period regions marked, not removed.
  - The grid keeps every distinct value as registered, so two `emp_var_rate_sigmoid_slope` values that are equal up
    to floating-point noise appear twice.
  - The pair figures' support-defined cells, the centring check, the fallback trace and the views requirement were
    added after the registration. They change no fit or verdict.
  - The −0.47 correlation of the slopes' steps rests on five fold steps and is descriptive only.
- *Judgements.* The interpretability judgements were written after the results. The registered rule asks for a
  documented gain, not a threshold.
- *Complementarity.* It needs the accepted artifacts (the Windows run), and its counts are that run's. Disagreement
  counts do not establish a better cascade.
- *Encoders.* Every target-dependent encoder (NSD, ECI, day × month) uses labels of later contacts under the
  accepted protocol (Commit 3). Only `past_density` removes that, and only for one feature.

## 12. Where the evidence is

| Evidence | Notebook cell | Record `stage3_simplification.json` |
|---|---|---|
| the registration, its digest and the baseline-source checks | 49 | `registration`, `sources` |
| the accepted views: status and whether they were required (checked before any fit) | 49 | `complementarity` (`views`) |
| post-registration additions | — | `post_registration` |
| rows, outer and inner fold identities, block checks | 50 | `rows_and_folds` |
| past-only density contexts and checks | 50 | `past_only_density` |
| the fallback trace: both forms in the EBM frames, fallback rows, later-label sensitivity, term-score change | 50, 52 | `past_only_density` (`fallback_trace`) |
| distributions, correlations, the correlated pair, term importance, period concentration | 51, 54 | `review` |
| configurations, pairs by name and their checks | 51 | `configurations` |
| fits: thresholds, calibration, fitted terms, out-of-fold digests; fold isolation | 52 | `fits` |
| parity (records; row by row on Windows) | 52 | `parity` |
| operating points, pooled and per-fold paired differences, decisions | 53 | `metrics` |
| shape curves, support, shape summary, the centring check (with each feature's within-day range) | 55 | `shapes` (`curves`, `support`, `summary`, `centring_check`) |
| pair cells and table | 56 | `shapes` (`pairs`, `pair_table`) |
| complementarity (Windows) and its requirement checks, nominations, interpretation, feature notes | 57 | `complementarity`, `nominations`, `interpretation`, `feature_notes` |
| this document's statements; limits | 58 | `stated`, `limits` |
