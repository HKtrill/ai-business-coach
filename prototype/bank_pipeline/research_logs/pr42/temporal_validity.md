# PR 42 Commit 3, Step 3B — Past-only encodings and the temporal robustness protocol

**Question.** Step 3A measured which labels reach each encoded value. Step 3B asks how Stage 1 predicts when
its target encodings use only labels known when a row is scored. How do the accepted and past-only protocols
compare, under the random split and under a chronological one? Which candidates need their own feature
contracts? And how should PR 49 test temporal robustness?

**What a comparison here measures.** Every comparison is a difference between two whole protocols on identical
rows. The past-only protocol changes several things at once:

- it removes same-day and later labels (the Step 3A look-ahead);
- it removes each training row's own label from its encoding;
- it fits edges, rates and fallbacks on fewer rows;
- it leaves the cold-start rows out of its fits.

No comparison separates these. A difference is therefore consistent with a contribution from look-ahead, not a
measurement of it. A 95% interval that includes 0 leaves a difference unresolved; it does not show that the
difference is absent.

**Scope.** Stage 1's two target encodings, `subscribe_rate_by_euribor_bin` and `subscribe_rate_by_dow_month`.
Both arms are covered: the GLASS LR and the black-box MLP, refitted in memory with the accepted PR 39 settings.
The Stage 2–3 families are extended by construction and by the Step 3A attribution, not by experiment (§5).
Training side only. No artifact is opened and no model is saved. The frozen PR 39 contracts, artifacts and
records are not changed.

**Evidence.**

- *Where it is produced.* `feature_research/feature_research.ipynb`, Cells 22–30 (module
  `feature_research.temporal_validity`). The notebook writes `research_logs/pr42/<run_id>/temporal_validity.json`
  beside the run's two Step 3A records.
- *Which run.* The numbers below come from this revision's run on the 32,950 training rows (Linux sandbox).
- *Tests.* `tests/feature_research/test_pr42_temporal_validity.py` (synthetic) and
  `tests/realdata/test_pr42_temporal_validity.py` (the real rows, LR only) re-derive the windows, the state
  verification, the availability counts, the invariance and the LR comparisons.
- *Platform dependence.* The LR is bit-reproducible: on the accepted designs it reproduces the PR 39 Commit 2 best
  CV ROC-AUC exactly (0.776336915456). The MLP's floating-point path depends on the BLAS build. In the sandbox its
  mean fold ROC-AUC is 2.3e-7 above the recorded value, so MLP figures from another platform can differ in the
  last digits.
- *Step 3A records.* This run's Step 3A records are identical, by content digest, to a separate sandbox run of the
  accepted Step 3A notebook, so Step 3B changes nothing that Step 3A records. Cell 30 checks this before it
  writes the Step 3B record. If the accepted run's directory (`c3_20261122_201830`) is absent, the comparison is
  reported as unavailable. If it exists, it must hold both records and their sidecars; a missing file fails the
  integrity check in any environment. Each record's sidecar and content digest are checked, and that the digest
  is the accepted one. Exact reproduction is required only where the run matches the accepted run's environment
  (operating system, platform, Python, interpreter, packages) and had the same accepted artifacts available. On
  another checkout, such as Linux after the accepted run is committed, reproduction is reported with the reason
  and not required.
- *Registration.* `TV.REGISTRATION` 1.0.0 was fixed before any result was computed. Version 1.1.0 was made after
  review of those results and is recorded in its `revisions`. It adds the lag-consistent chronological arm and
  its regression (§3, §4.3), names every contrast as a difference between protocols, and changes the Step 3A
  record check. No 1.0.0 comparison's protocols, rows or computation changed.

## The answers in brief

| | Finding |
|---|---|
| What a past-only encoding knows | Only the covariates and labels of rows contacted on earlier days (less the outcome lag): its quantile edges, bin rates, median fallback, cell rates and global prior are all refitted from those rows at every update. Never its own label, a same-day label or a later label. Verified: every state equals the production `LRFeatureEngineer` fitted on that state's rows (13 contexts, 0.0 difference) |
| Availability | The primary rule (lag 0) encodes 32,033 of 32,950 training rows (97.2%); the first three contact days are the cold start. Of those, 12,636 (39.4%) get the euribor **median fallback**, because euribor falls to new lows through 2008–2010 and a new value has no past bin. 3,929 (12.3%) meet an unseen day × month cell |
| Invariance | Flipping any unavailable label never changes a past-only encoding (39 of 39 checks over 40 days and three contexts, each with a control that does change). The accepted rule's encodings do change when same-day and later labels flip (contrast) |
| Past-only vs accepted protocol, random split | LR on the same 32,033 rows: accepted 0.7713 versus past-only 0.7648 ROC-AUC, **−0.0065** [−0.0105, −0.0025] pooled. Within windows the difference is −0.046 [−0.064, −0.029]. Consistent with a contribution from look-ahead; not isolated from the other changes the protocol makes |
| Chronological vs random split (2 × 2, windows' rows) | Within windows the chronological split ranks lower by 0.068 [0.051, 0.085] under the accepted protocol. Under the past-only protocol the difference is −0.0005 [−0.0093, 0.0084]: **not resolved**. That bounds it at about ±0.009 here; it does not show the split no longer matters. The interaction is +0.068 [0.049, 0.087] |
| Chronological split | Past-only updating vs the accepted window fit: **+0.014** [0.001, 0.027] pooled, +0.021 [0.006, 0.036] within windows. Pooled and within-window results can differ in sign: frozen past-only is −0.056 pooled but +0.011 [0.006, 0.016] within windows |
| Outcome lag | The lag-consistent arm (model, scaler and encodings all under the delay) measures operating under the delay. Within windows: lag 5 −0.017 [−0.037, 0.004] and lag 20 −0.009 [−0.024, 0.006], neither resolved; monthly −0.047 [−0.063, −0.031]. Pooled, lag 20 is −0.151, which mixes score levels across windows. The 1.0.0 lag arm is a lag-0 model fed slower encodings, a different question: lag 20 there is −0.050 pooled and +0.012 [−0.002, 0.027] within windows |
| MLP early stopping | Isolating the early-stopping subset moves the median stopping epoch from 17 to 32.5, and the validation ROC-AUC it stops on from 0.7969 to 0.7942. Under the random split, the held-out ROC-AUC difference is not resolved: −0.0017 [−0.0037, 0.0004] pooled |
| Contracts | The lag 0 rule is a new feature definition. It needs its own versioned contract with distinct names (proposed, not registered) and goes to PR 44. The lag and monthly rules are PR 49 sensitivities. The isolated early stopping is a training-procedure candidate. The accepted contracts are unchanged |
| PR 49 | The protocol is designed (§7). Under a pure prior shift, the frozen GLASS routing meets all three Stage 2 obligations only for prevalence between 10.9% and 12.0%, while prevalence across the burn-in and the windows ranges from 4.7% to 49%. Missed obligations are expected by construction and must be reported, not removed |

## 1. What each encoding knows when a row is scored

The time unit is the Step 3A **contact day**: 485 days in 26 periods, from 2008-05 to 2010-11, training rows
only. Rows contacted on the same day are simultaneous, because no time of day is recorded, so a same-day label is
never available. When an outcome became known is not recorded either: each row is a client's last contact and its
label is the campaign outcome. The rules therefore carry an **outcome lag** L in contact days. Lag 0 (an outcome is
known by the next contact day) is the primary candidate. Lags 5 and 20 (about a week and about a month) and a
monthly refit are the sensitivities. All of this was fixed in `TV.REGISTRATION` (digest `88c004c3…`) before any
result was computed.

| Quantity | Accepted, random fold | Accepted, final fit (and the test mapping) | Past-only, lag L (day d) | Past-only, monthly |
|---|---|---|---|---|
| euribor bin edges | `qcut(20)` of the fold's training rows, all periods | `qcut(20)` of every training row | `qcut(20)` of the context's rows with day < d − L | rows of earlier periods, refitted at each period start |
| bin rates | mean label per bin of those rows | the same, all rows | the same rows | the same rows |
| median fallback (value outside the bins, or an empty bin) | median of those bin rates | the same | median of the available bin rates, refitted with them | the same |
| day × month cell rates | `(n·mean + 100·global)/(n + 100)`, years pooled | the same | the same formula on the available rows | the same |
| global prior (smoothing; unseen cell) | mean label of the fold's training rows | of every training row | of the available rows | of the earlier periods' rows |
| the row's own label | in a fit row's own value (a); never in a held-out row's value (b absent) | in every training row's value | never | never |
| same-day labels | yes | yes | never | never (the whole period is unavailable) |
| later labels | yes: 37–41% of the label mass on average (Step 3A, c) | yes | never | never |

- **Cold start.** Rows on the first three contact days (plus the lag), or in the first period under the monthly
  rule, have no past-only encoding. They are left out of that rule's fits and of every comparison that includes
  it: 917 rows for lag 0. Every state a rule uses must hold at least 500 rows and 10 positives. The smallest holds
  917 rows and 15 positives with the full history, and 812 rows and 11 positives within a random fold.
- **Unseen values.** The production fallbacks are kept: the median bin rate for a euribor value outside the
  available bins, and the global rate for an unseen cell. Cells pool years, as in production, so a cell's rate can
  be a year old when its month recurs.
- **Within a random fold.** The context is the fold's training rows only. A held-out row is encoded from fold
  rows contacted before it, so the encoding is both fold-isolated and past-only, and mechanisms (a) and (c) are
  removed together.
- **Verification** (Cell 24). Each state is computed from arrays, for speed. `TV.verification` then fits the
  production `LRFeatureEngineer` on exactly that state's rows and transforms the rows that use it. This covers
  every state of every rule in the full-history context, the primary rule's 482 states in two fold contexts, and
  a sample of days for the lagged rules and for four isolated early-stopping contexts. There are 13 contexts and
  every difference is 0.0, including the 16,565 fallback values of the lag 0 rule.

## 2. Availability and fallbacks

| Scope | Rule | Encoded | Cold start | euribor fallback | Unseen cell | Smallest state (rows / positives) |
|---|---|---:|---:|---:|---:|---|
| every training row, full history | lag 0 | 32,033 (97.2%) | 917 | 12,636 | 3,929 | 917 / 15 |
| | lag 5 | 30,507 (92.6%) | 2,443 | 16,325 | 7,114 | 917 / 15 |
| | lag 20 | 25,714 (78.0%) | 7,236 | 16,513 | 17,304 | 917 / 15 |
| | monthly | 26,775 (81.3%) | 6,175 | 15,001 | 18,365 | 6,175 / 185 |
| random held-out rows (fold history) | lag 0 | 32,033 (97.2%) | 917 | 12,638 | 3,932 | 812 / 11 |
| chronological predicted rows (W1–W5) | lag 0, updating | 13,771 | 0 | 10,172 | 435 | 19,179 / 895 |
| | frozen cutoff state | 13,771 | 0 | 11,978 | 5,361 | 19,179 / 895 |

**The euribor bin rate cannot extrapolate.** The 3-month euribor runs from 4.86–4.97 in 2008-05..08 down to
0.63 in 2010-03. Each contact day carries one value, which is usually a new minimum, so it falls outside every bin
of the rows seen so far and takes the median rate. The share of predicted rows that get the fallback, by window:

| Encoding | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|
| updating state | 98.8% | 75.2% | 93.1% | 58.7% | 15.8% |
| frozen cutoff state | 98.9% | 100% | 100% | 66.6% | 49.1% |

Under the random split the accepted encoder never meets this, because every fold spans every period. In time,
the feature is mostly a constant until the rate stops falling. The day × month rate fails similarly for months not
yet contacted. With the frozen W1 and W2 states every predicted cell is unseen: no October–December, March or
April contact precedes them. Early states also have few bins, because a state over k contact days holds at most k
distinct euribor values: the minimum is 1 bin and the median is 15.

## 3. No unavailable label reaches an earlier encoding

Cell 25 runs the checks in three contexts: the full history, random fold 0's training rows, and fold 0's isolated
early-stopping context. Each rule is checked at 40 contact days spread over the timeline:

- flipping every label unavailable to that day leaves the day's encodings unchanged;
- flipping only the same day's labels leaves them unchanged;
- flipping the available labels changes them, so the check can fail.

All 39 checks pass. The unavailable labels are the same day's, later days', those inside the lag, and the same
period's under the monthly rule. As a contrast, the production encoder fitted on the whole context (the accepted
rule) changes the day's encodings in every case when same-day and later labels flip.

Cell 23 adds checks at model level:

- each window's training rows are exactly the rows contacted before its first day;
- the frozen arm's predicted encodings equal the accepted window fit's;
- for the deployed past-only LR in W1, flipping the labels of contact day 94 and every later day leaves day 94's
  encodings and scores unchanged, while flipping earlier window labels changes them;
- in every window, the lag-consistent model at lag 0 is the past-only updating model (same rows, encodings,
  outcomes and scores).

**The lag-consistent regression.** In W1 and W5, under lag 5 and lag 20, Cell 23 flips the pre-cutoff outcomes
that are still unavailable at the cutoff: rows contacted within the lag before it, 1,023 and 4,535 rows in W1.
That leaves the lag-consistent model unchanged (scaler means and scales, coefficients, intercept). It also leaves
the deployed scores unchanged for every predicted row whose encoding was fixed before those outcomes arrive.
Flipping outcomes known at the cutoff changes the model (control). The same flip changes the lag-0 model of the
1.0.0 lag arm, which trains on every pre-cutoff outcome (contrast). Under the monthly rule no pre-cutoff outcome
is unavailable at a period-start cutoff, so the flip is of the window's first-period outcomes.

All 26 checks pass. Two synthetic regressions show that these checks can fail:
`test_a_rule_that_admits_same_day_labels_is_caught` (a rule admitting same-day labels) and
`test_a_lag_arm_that_trains_on_unavailable_outcomes_is_caught` (a lag arm that treats every pre-cutoff outcome as
known).

## 4. Predictive comparisons

**Settings.** Both models use the accepted PR 39 Commit 2 settings, read from
`research_logs/pr39/c2_20261114_184433/stage1_calibration_evidence.json`:

- *LR:* saga, elastic net with C = 0.00176 and l1_ratio = 0.0306, class-balanced, scored by P(y = 1);
- *MLP:* (16, 8), ReLU, up to 200 epochs with patience 15, a 0.15 validation split and seed 42.

**Folds and windows.** Every fit refits its scaler on its own fit rows. The random split is the accepted plan
(`build_fold_designs`, 10 folds, seed 42).

| Segment | Periods | Rows | Positives | Prevalence |
|---|---|---:|---:|---:|
| burn-in (training only) | 2008-05..2008-08 | 19,179 | 895 | 4.7% |
| W1 | 2008-10..2008-11 | 2,941 | 195 | 6.6% |
| W2 | 2008-12..2009-04 | 2,178 | 452 | 20.8% |
| W3 | 2009-05 | 4,661 | 411 | 8.8% |
| W4 | 2009-06..2009-09 | 1,543 | 551 | 35.7% |
| W5 | 2009-10..2010-11 | 2,448 | 1,208 | 49.3% |

Each window is predicted by models fitted on every earlier row and frozen at its first day.

**Comparisons.** Each comparison is a paired DeLong test on identical rows, reported as b − a with a 95% interval.
Results are given pooled over the rows and as the unweighted mean over the five windows. That mean uses
independent-window standard errors and is restricted to the windows' rows, also for the random split. Pooled
chronological scores come from five models whose score levels differ across windows of 7% to 49% prevalence. The
mean over windows measures ranking within them. The two can differ in sign, and each is labelled.

**Reproduction.** The accepted protocols reproduce PR 39:

- LR mean fold ROC-AUC: 0.776336915456, exact;
- MLP: 0.781170055 against the recorded 0.781169825;
- all 30 isolated MLP fits early-stopped on the reproduced subset.

### 4.1 LR, random split (the accepted fold plan): past-only vs accepted protocol

The random split has no cutoff, so here the lag and monthly rules change only the encodings. Every fold
training row's outcome enters the model.

| Past-only rule | Rows | Accepted ROC-AUC | Past-only ROC-AUC | Pooled difference [95%] | Mean within windows [95%] |
|---|---:|---:|---:|---|---|
| lag 0 | 32,033 | 0.7713 | 0.7648 | −0.0065 [−0.0105, −0.0025] | −0.0464 [−0.0641, −0.0287] |
| lag 5 | 30,507 | 0.7674 | 0.7552 | −0.0122 [−0.0167, −0.0078] | −0.0500 [−0.0686, −0.0314] |
| lag 20 | 25,714 | 0.7625 | 0.7529 | −0.0096 [−0.0141, −0.0051] | −0.0501 [−0.0682, −0.0321] |
| monthly | 26,775 | 0.7613 | 0.7557 | −0.0056 [−0.0105, −0.0006] | −0.0735 [−0.0930, −0.0539] |

### 4.2 Protocols and splits: the 2 × 2 on the windows' rows (LR)

The four protocols are compared on the same 13,771 rows (2,817 positives), the rows of W1–W5:

- **A:** random split, accepted protocol (the accepted OOF scores);
- **B:** random split, past-only protocol;
- **C:** chronological split, accepted window fit;
- **D:** chronological split, past-only protocol, updating.

| Cell | Pooled ROC-AUC | Mean within windows | W1 | W2 | W3 | W4 | W5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A random / accepted | 0.7619 | 0.5734 | 0.579 | 0.672 | 0.597 | 0.491 | 0.527 |
| B random / past-only | 0.7433 | 0.5270 | 0.446 | 0.568 | 0.543 | 0.527 | 0.552 |
| C chronological / accepted | 0.6733 | 0.5053 | 0.520 | 0.500 | 0.467 | 0.529 | 0.511 |
| D chronological / past-only updating | 0.6874 | 0.5265 | 0.502 | 0.534 | 0.541 | 0.524 | 0.531 |

| Protocol difference | Pooled [95%] | Mean within windows [95%] |
|---|---|---|
| past-only vs accepted, random split (B − A) | −0.0187 [−0.0243, −0.0130] | −0.0464 [−0.0641, −0.0287] |
| past-only vs accepted, chronological (D − C) | +0.0141 [0.0007, 0.0274] | +0.0213 [0.0061, 0.0364] |
| chronological vs random, accepted (C − A) | −0.0886 [−0.0999, −0.0773] | −0.0681 [−0.0850, −0.0513] |
| chronological vs random, past-only (D − B) | −0.0559 [−0.0635, −0.0482] | −0.0005 [−0.0093, 0.0084] (not resolved) |
| interaction ((D − C) − (B − A)) | +0.0327 [0.0197, 0.0457] | +0.0677 [0.0486, 0.0867] |

What the 2 × 2 shows. Each line is a difference of whole protocols.

- **Within windows, under the random split, the past-only protocol ranks lower by 0.046** (B − A). This is
  consistent with a contribution from the look-ahead Step 3A found in the accepted encodings. The past-only
  protocol also removes each training row's own label from its encoding, fits its edges, rates and fallbacks on
  fewer rows (the euribor fallback alone covers 39% of its rows) and leaves the cold-start rows out of its fits.
  This experiment does not separate these contributions. PR 44 could, for example with an encoding that leaves
  only the own label out and keeps every other label.
- **Within windows, under the past-only protocol, the two splits do not differ resolvably**: D − B = −0.0005
  [−0.0093, 0.0084]. The experiment bounds the difference at about ±0.009 at this sample; it does not show that
  the split no longer matters.
- **Within windows, under the accepted protocol, the chronological split ranks lower by 0.068** (C − A). The
  interaction (+0.068 [0.049, 0.087]) says the two protocols respond differently to the change of split, again
  without saying which of the protocol's changes is responsible.
- **Pooled, both protocols rank lower under the chronological split** (C − A = −0.089, D − B = −0.056). A model
  frozen at a cutoff cannot place a new regime's rows relative to earlier ones. A model trained on every period
  can, because the encodings and `cellular_downturn` carry the period's rate. Most of Stage 1's pooled ROC-AUC
  (0.76–0.78) ranks periods: within a window it is 0.49–0.67 even for the accepted random-split scores.
- **Within the chronological split, the past-only updating protocol ranks higher than the accepted window fit**
  (D − C: +0.014 pooled, +0.021 within windows). One plausible reason is that it learns the new regime's rates
  as outcomes arrive; the experiment does not test that reason.
  - *Frozen at the cutoff.* The past-only protocol is −0.056 [−0.069, −0.043] pooled against C but +0.011
    [0.006, 0.016] within windows. The signs differ, so the pooled figure reflects score levels across windows
    rather than ranking within them.
  - *Updating versus frozen.* +0.070 [0.061, 0.078] pooled, +0.010 [−0.004, 0.024] within windows (not
    resolved).
  - *W2 under C and `past_frozen`.* Every predicted row has the same features (euribor fallback, unseen cells,
    and a `cellular_downturn` that is constant in the training rows), so the scores are constant and the
    ROC-AUC is 0.500.

### 4.3 Outcome lag in the chronological windows (LR)

The 1.0.0 lag arm (`past_updating` under lag 5, lag 20 or monthly) keeps the lag-0 model and slows only its
prediction-time encodings. That model is trained on lag-0 encodings and on every outcome before the cutoff, some
of which a lagged deployment would not yet have. The arm answers one question: what does a lag-0 model lose when
its incoming features update more slowly?

The **lag-consistent** arm, added in registration 1.1.0, operates under the delay throughout. Its scaler and
model are fitted only on training rows whose outcome is known at the cutoff (`TV.known_at_cutoff`: contacted
before cutoff − L; before the cutoff for the monthly rule) and that the rule encodes, with the rule's
encodings. Predicted rows are encoded by the rule's state at their own contact day. In W1, lag 20 leaves 14,644
of the 19,179 pre-cutoff outcomes known, and 7,408 of those rows have an encoding. Lag 5 leaves 18,156 and
15,713. All against lag 0 (b − a):

| Arm | Lag | Pooled [95%] | Mean within windows [95%] |
|---|---|---|---|
| lag-consistent (operating under the delay) | 5 | −0.0613 [−0.0694, −0.0531] | −0.0167 [−0.0373, 0.0038] (not resolved) |
| | 20 | −0.1509 [−0.1622, −0.1395] | −0.0090 [−0.0236, 0.0056] (not resolved) |
| | monthly | +0.0280 [0.0210, 0.0349] | −0.0472 [−0.0633, −0.0311] |
| lag-0 model fed slower encodings (1.0.0 arm) | 5 | −0.0197 [−0.0251, −0.0143] | +0.0072 [−0.0023, 0.0168] (not resolved) |
| | 20 | −0.0501 [−0.0579, −0.0424] | +0.0122 [−0.0022, 0.0267] (not resolved) |
| | monthly | −0.0514 [−0.0590, −0.0438] | −0.0350 [−0.0489, −0.0211] |

At lag 20, lag-consistent against the lag-0 model fed lag-20 encodings is −0.1008 [−0.1105, −0.0910] pooled and
−0.0213 [−0.0300, −0.0125] within windows.

What this shows:

- **The 1.0.0 figure does not measure operating under a delay.** The pooled −0.050 for lag 20 belongs to a lag-0
  model fed slower encodings.
- **Within windows, operating under a 5- or 20-day delay is not resolved** at this sample (−0.017 and −0.009);
  the monthly rule is resolved, at −0.047.
- **Pooled, the lag-consistent costs are larger** (−0.061 and −0.151), and the monthly sign reverses. The pooled
  figures mix the five models' score levels across windows, so they are not ranking costs.
- **Outcome timing is unknown**, so lag 0 remains an optimistic bound. The lag-consistent arm is the form PR 49
  runs at cascade level.

### 4.4 MLP, with the accepted and the isolated early stopping

**Accepted versus isolated early stopping.** The accepted early stopping draws its validation subset from rows
whose encodings and scaler were fitted with the subset's labels (Step 3A). The isolated split draws the
estimator's own split first and fits the encoders and scaler without the subset. The fits:

| Fits | Early stopping | Median best epoch (min–max) | Mean best validation ROC-AUC |
|---|---|---|---:|
| random, accepted encoding (10) | accepted | 17 (1–56) | 0.7969 |
| | isolated | 32.5 (3–77) | 0.7942 |
| random, past-only (10) | accepted | 27 (1–72) | 0.7718 |
| | isolated | 26.5 (1–69) | 0.7707 |
| chronological, accepted window (5) | accepted | 11 (1–21) | 0.6611 |
| | isolated | 26 (1–62) | 0.6522 |
| chronological, past-only (5) | accepted | 10 (5–75) | 0.6500 |
| | isolated | 35 (3–57) | 0.6547 |

The difference on held-out rows (isolated − accepted early stopping, same protocol and rows):

| Arm | Pooled [95%] | Mean within windows [95%] | Rows (pooled) |
|---|---|---|---:|
| random split, accepted protocol | −0.0017 [−0.0037, 0.0004] (not resolved) | −0.0024 [−0.0096, 0.0048] (not resolved) | 32,950 |
| random split, past-only protocol | −0.0008 [−0.0031, 0.0014] (not resolved) | −0.0111 [−0.0217, −0.0005] | 32,033 |
| chronological, accepted window | +0.0309 [0.0253, 0.0364] | +0.0084 [−0.0012, 0.0181] (not resolved) | 13,771 |
| chronological, past-only updating | −0.0274 [−0.0406, −0.0142] | +0.0298 [0.0104, 0.0491] | 13,771 |

Under the random split, then, isolating the subset moves the stopping epoch. It also lowers the validation
ROC-AUC the network stops on, by about 0.003 with the accepted protocol. The pooled held-out ROC-AUC difference
is not resolved under either protocol, which does not show that it is absent. Within windows it is resolved only
under the past-only protocol, at −0.011. In the chronological windows each difference comes from five fits.
Pooled and within-window figures differ in sign for the past-only protocol (−0.027 pooled, +0.030 within
windows), so the pooled figures reflect score levels across windows (below).

**The 2 × 2 with isolated early stopping.** Within windows the MLP's protocol differences have the LR's signs:

| Protocol difference | Mean within windows [95%] |
|---|---|
| past-only vs accepted, random split (B − A) | −0.021 [−0.039, −0.003] |
| past-only vs accepted, chronological (D − C) | +0.019 [0.002, 0.035] |
| chronological vs random, past-only (D − B) | −0.020 [−0.041, 0.002] (not resolved) |
| chronological vs random, accepted (C − A) | −0.059 [−0.077, −0.041] |
| interaction | +0.039 [0.013, 0.066] |

**Pooled MLP figures.** Pooled, the MLP's chronological past-only cell is much worse: ROC-AUC 0.564 and AP 0.210,
against 0.618 and 0.366 for the accepted window fit. Its within-window ROC-AUC is nonetheless no lower (0.541
against 0.523). The pooled figures measure whether score levels follow the prevalence across windows. That is a
calibration-under-prevalence-shift question, which is PR 49's (§7), not a ranking result.

### 4.5 Limitations

- **What a comparison isolates.** Each comparison is between whole protocols. The past-only protocol changes the
  look-ahead, the own-label contribution, the rows its edges, rates and fallbacks are fitted on, and the
  cold-start rows together, and nothing here attributes a difference to one of them.
- **Settings.** The accepted hyperparameters were tuned under the accepted encoding. Nothing was retuned for the
  past-only encodings, which may understate them. PR 44 tunes candidates under its own protocol.
- **Inference.**
  - One seed, one fold plan, five windows of unequal size and prevalence.
  - The DeLong intervals treat rows as independent.
  - The mean-window intervals treat windows as independent.
  - Pooled chronological scores come from five different models, so pooled and within-window results can differ
    in sign; both are reported.
  - The lag-consistent arm was added after review of the 1.0.0 results (registration 1.1.0).
- **Rows.**
  - The cold-start rows (917, or 2.8%) are left out of every past-only comparison. On the accepted side, the
    same rows are scored but the model is trained on all rows.
  - The timeline holds training rows only. Contact days whose contacts all fell in the test split are absent,
    and every state's history is about 20% thinner than the full data's.
- **Outcomes.** The outcome lag is assumed, not observed. Each row is a client's last contact; earlier contacts of
  the same client are not in the data.
- **Coverage.** Stage 1 only. No Stage 2–4 component was refitted, so these numbers bound nothing about the
  cascade.
- **Not independent validation.** These are training-side research comparisons. The test split was not read.

## 5. What extends to the Stage 2–3 families

| Family (descendants) | Step 3A held-out later-label share (mean) | Past-only construction | Direct evidence here | Inferred only |
|---|---:|---|---|---|
| `dow_month_encoded` (dow_* bins; Stage 3 feature) | 0.371 (Stage 2 outer), 0.371 (Stage 3 tuning) | the primary rule unchanged: Stages 2 and 3 use the same smoothed day × month formula (k = 100, global prior, unseen cell → global; Step 3A construction and reconstruction) | the rule's realizability, invariance, availability and unseen-cell frequency (the Stage 1 day × month results apply to its values) | its effect through the dow_* bins on the Stage 2 routing and obligations, and in Stage 3 |
| `neighborhood_subscription_density` (nsd_* bins, `decay_x_density`) | 0.469 / 0.469 | a kernel average over reference rows contacted before day d, with the scaler fitted on rows before d; the 5,000-row reference sample must be redrawn or grown as rows arrive (a design choice not made here) | none for a past-only version (Step 3A attribution only) | realizable in principle; its look-ahead share is larger than Stage 1's (about 0.37–0.41); its predictive change is not estimated |
| `economic_curvature_intensity` (eci_* bins; Stage 3 ECI) | 0.516 / 0.516 | per macro variable, bin rates from rows before day d, then the curvature over the non-empty bins | none for ECI itself; the euribor bin-rate rule tested here is its first step for one of its three variables | the qcut-on-the-past behaviour (few early bins, the fallback when a value has no past bin) applies to each of its three variables. Its fallback is 0.0, so in a falling-rate regime ECI would often be zero |

The conclusions that carry over directly are:

- the availability rule and same-day exclusion;
- the cold-start rule;
- the verification method (production fit on the state's rows);
- the invariance checks;
- the finding that a past-only bin rate falls back for most rows of a regime it has not seen (euribor here; also
  `nr.employed` and `emp.var.rate`, which trend over the same years).

The predictive comparisons (§4) do not carry over. The Stage 2–3 families feed rule learning, routing and Stage 3
models that were not refitted; PR 45 builds and tests their past-only versions, and PR 49 runs them in the
cascade.

## 6. Contract decisions

The registered criteria for a candidate:

- **realizable** (§3);
- **differing** from the accepted encoding on at least 1% of rows;
- **available** for at least 95% of training rows.

A past-only rule is a new feature definition. The accepted Stage 1 contract (`stage1-features` 2.0.0, fingerprint
`d0d8b07f2ca5e81d`) identifies features by their names. Reusing those names could not tell the two definitions
apart, so any adopted rule needs its own versioned contract with distinct names.

| Candidate | Role | Realizable | Available | Values differ | Versioned contract | Destination |
|---|---|---|---:|---:|---|---|
| `past_daily_lag0` | primary | yes | 97.2% | 32,033 rows | yes | PR 44 Stage 1 candidate |
| `past_daily_lag5` | lag sensitivity | yes | 92.6% | 30,507 | yes, if adopted | PR 49 sensitivity |
| `past_daily_lag20` | lag sensitivity | yes | 78.0% | 25,714 | yes, if adopted | PR 49 sensitivity |
| `past_monthly` | lag sensitivity | yes | 81.3% | 26,775 | yes, if adopted | PR 49 sensitivity |
| MLP isolated early stopping | training procedure | yes | all rows | none (features unchanged) | no | PR 44 training-procedure candidate |
| accepted (PR 39) | frozen baseline | no (looks ahead) | all rows | — | no | unchanged; the baseline PR 43 reads |

**The proposed contract** (`TV.candidate_contract()`, recorded, not registered):

- `stage1-features` 3.0.0-candidate, features `cellular_downturn`, `subscribe_rate_by_euribor_bin_past` and
  `subscribe_rate_by_dow_month_past` (names fingerprint `1d48dea8b294ff74`);
- a definition fingerprint (`fe5d54e195b7a604`) over the rule, the lag, same-day exclusion, the cold start, the
  history floor and what is refitted at each update;
- the contact-day time index as a new pipeline input.

`stage1_feature_contract` does not recognise these names, so no existing artifact check can confuse them with the
accepted ones. The accepted contracts, the PR 39 artifacts and records, and the frozen Stage 2 operating contract
(`pr37-stage2-primary` 1.0.0, `0817825b9eecbd11`) are unchanged.

## 7. The PR 49 temporal robustness protocol (design; `TV.PR49_PROTOCOL`)

- **Split.** Forward chaining over contact periods on the windows registered here. 2008-05..2008-08 trains only;
  each window is predicted by a cascade fitted on every earlier row and frozen at the window's first day. A
  one-period gap is the outcome-lag sensitivity. Whether the previously exposed test rows join is PR 49's
  decision, disclosed with that exposure.
- **Fitted at each cutoff.**
  - *Stage 1:* encoders (past-only rule; frozen and updating arms for predicted rows), scaler, model with the
    accepted hyperparameters, calibrator by nested OOF, the F2 threshold.
  - *Under an outcome lag, at every stage:* each of these is fitted only on training rows whose outcomes are known
    at the cutoff (`TV.known_at_cutoff`), with lag-consistent encodings, as in Step 3B's lag-consistent arm.
  - *Stage 2:* the NSD, ECI and day × month sources, the bins (frozen PR 39 cut points, or Commit 4's procedural
    bins), the GLASS rule ILP and the RF router, under the frozen operating contract.
  - *Stage 3:* models, calibrators (ECE gate) and nested thresholds.
  - *Stage 4:* arbiters, weights and the abstention cut at the registered coverage floor.
- **Updated after the cutoff.** Only the label statistics of past-only encodings, as outcomes arrive, under the
  registered rule and lag. Models, calibrators, thresholds, bins, routes, rules and the abstention cut stay
  frozen within a window. A separately labelled recalibration arm may refit calibrators and thresholds on labels
  known before each window; it is reported beside the frozen arm and never replaces it.
- **Prevalence shift.** The windows' prevalence runs from 6.6% to 49.3% against 4.7% in the burn-in. Each of the
  following is reported per window and arm:
  - *Calibration:* calibration in the large, reliability, Brier and ECE with the cutoff calibrator, and a
    prior-shift-corrected curve as a labelled diagnostic.
  - *Thresholds:* F2, recall, precision and confusion counts at the frozen threshold, beside the window's own
    F2-optimal threshold as a diagnostic only.
  - *Abstention:* realised coverage and covered-row accuracy against the coverage floor. The cut is not moved.
  - *Stage 2 obligations:* observed with `observe_routing`, never certified. Support floors are fit-time only.
- **Prior-shift attribution.** Each window's observed obligation value is shown beside its pure-prior-shift
  projection. The projection holds the training OOF class-conditional routing rates fixed and moves only the
  prevalence; the difference between the two is the change in routing rates. Step 3B's projection for the
  frozen routing uses the PR 38 report's pooled training OOF counts (`TV.stage2_prevalence_projection`; no
  artifact read). In the table below, a value with an asterisk misses its limit.

  | Router | Training OOF (11.3%) | NPV holds up to | Precision holds from | Leakage (prevalence-invariant) | W1 6.6% | W2 20.8% | W3 8.8% | W4 35.7% | W5 49.3% |
  |---|---|---:|---:|---|---|---|---|---|---|
  | GLASS | NPV 0.9534, leak 0.1463, prec 0.3075 (all met) | 12.0% | 10.9% | 0.1463 (met) | prec 0.199* | NPV 0.909* | prec 0.253* | NPV 0.824* | NPV 0.727* |
  | RF | NPV 0.9594, leak 0.1514*, prec 0.2980* | 13.6% | 11.4% | 0.1514 (0.0014 over) | prec 0.192* | NPV 0.920* | prec 0.244* | NPV 0.844* | NPV 0.755* |

  - *What the projection shows.* With frozen rules, a higher prevalence lowers Pass 1 NPV and raises Pass 2
    precision, while the leakage share (a share of the positives) does not move. Every window is therefore
    expected to miss at least one obligation from prevalence alone.
  - *RF router at training prevalence.* The PR 38 report's training OOF observation already sits 0.0014 above
    the leakage ceiling and 0.0020 below the precision floor. That is observed, read from that report, and not
    re-derived here.
- **Missed-obligation report.**
  - *What identifies it:* the obligation, window, arm and stage.
  - *What it reports:* numerator and denominator, the observed value, the limit and the exact signed shortfall
    (`observe_routing`), the routed rows and positives, the window prevalence and the prior-shift projection.
  - *Status:* satisfied, violated or undefined. Undefined is never compliant, and any violation is reported
    however small.
  - *Fit-time comparison:* whether the same obligation held on the training rows' OOF routing at that cutoff.
  - *What it bounds:* a miss limits the temporal claim for that window and arm. It is a robustness finding, not a
    pipeline failure. It is never averaged away across windows, and nothing is refitted or re-thresholded to
    remove it.
- **Comparisons.** Each is a difference of protocols on the same rows, reported pooled and within windows.
  - accepted versus past-only encoding protocols, frozen versus updating, both arms;
  - random-split versus chronological results;
  - outcome lags 0, 5 and 20 and the monthly rule, lag-consistent throughout, with the lag-0 model fed slower
    encodings as a secondary arm;
  - MLP accepted versus isolated early stopping.

  All are fixed by this design before PR 49 examines any chronological cascade result.

## 8. Interim temporal-validity finding

- **The finding.**
  - *The accepted encodings.* The accepted Stage 1 target encodings are fold-isolated but look ahead in time.
  - *Their past-only versions:*
    - are realizable;
    - are verified equal to the production encoder on the available rows;
    - leave out only the cold start.
  - *Past-only versus accepted protocol, under the random split (LR):*
    - −0.0065 ROC-AUC pooled;
    - −0.046 within windows.

    These are differences of whole protocols, consistent with a contribution from look-ahead but not isolating
    it.
  - *Chronological versus random split, under the past-only protocol.* Within windows the difference is not
    resolved (−0.0005 [−0.0093, 0.0084]). It is bounded, not shown to be zero.
  - *Under the chronological split.* The updating past-only protocol ranks higher than the accepted window fit,
    pooled and within windows.
  - *Operating under an outcome delay.* With the model, scaler and encodings all lag-consistent, the cost within
    windows is not resolved for 5 or 20 contact days. The monthly rule is resolved, at −0.047. The 1.0.0 figure
    (−0.050 pooled for lag 20) measured a lag-0 model fed slower encodings.
  - *The euribor bin rate.* It mostly falls back to a constant in the periods that matter, because it cannot
    extrapolate.
  - *Stage 1's ranking.* Its pooled discrimination largely ranks periods, which carry very different
    subscription rates.
- **PR 43** may use the frozen PR 39 training-side OOF evidence as random-split evidence. It is fold-isolated
  (mechanism b absent), but every target-derived feature looks ahead in time and the MLP's early stopping is not
  encoder-isolated. PR 43 may not present that evidence, or Stage 4 conclusions built on it, as forward-in-time
  performance. Its comparisons stay valid as comparisons under the random split.
- **PRs 44–45** need:
  - the versioned candidate contract (distinct names, the contact-day time index as an input, the registered cold
    start);
  - the isolated early stopping as an MLP training candidate;
  - past-only Stage 2–3 constructions, not built here;
  - adoption criteria that include a chronological comparison, not the random split alone.
- **Open for PR 49:**
  - the chronological cascade across Stages 1–4;
  - the prevalence-shift effects on calibration, thresholds, abstention and the Stage 2 obligations, including
    the expected misses above;
  - past-only NSD and ECI;
  - the outcome-lag sensitivity at cascade level, lag-consistent throughout;
  - which of the past-only protocol's changes accounts for its differences (look-ahead, own label, fit rows and
    fallbacks, cold start);
  - whether exposed test rows join.

## 9. Where the evidence is

| Evidence | Notebook cell | Record `temporal_validity.json` |
|---|---|---|
| registration (1.1.0 with its revisions), digest, windows | 22 | `registration` |
| accepted settings read (source and SHA-256) | 23 | `settings` |
| states against production | 24 | `verification` |
| availability per scope and rule, history floors, by period | 24 | `availability` |
| invariance flips; window, deployed-score and lag-consistent regression checks | 25, 23 | `invariance` |
| reproduction, protocol table, comparisons (pooled, per segment, mean over windows), factorials, per-fold ROC-AUC, lag-consistent fit rows, MLP fits | 26, 27 | `predictive` |
| contract decisions, candidate contract | 28 | `contracts` |
| Stage 2–3 extension | 28 | `stage23_extension` |
| PR 49 protocol, prior-shift projection | 29 | `pr49_protocol`, `stage2_prior_shift` |
| interim finding and its interpretation (what each contrast does and does not isolate; resolution per contrast) | 29 | `interim_finding` |
| the accepted Step 3A run: unavailable if its directory is absent; otherwise both records and sidecars required, integrity checked, and exact reproduction where the environment matches (all before the record is written) | 30 | `step3a_reproduction` |
