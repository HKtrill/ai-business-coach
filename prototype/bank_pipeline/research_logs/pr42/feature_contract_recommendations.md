# PR 42 Commit 6, Step 6A — Feature-contract recommendations and research handoff

**Question.** Commits 2–5 produced seven accepted research records. On that training-side evidence, which frozen
feature contracts stay as they are? Which candidates go on for further evaluation, and to which later step? Which
methodology revisions should become separately versioned changes, and what would each need before adoption? Where
does the Stage 3 feature DAG belong, and can the copies in `shared/stage2/feature_engineering.py` be retired?

**In brief.**

| Area | Frozen reference (kept) | Goes on | Destination | Closed or not advanced |
|---|---|---|---|---|
| Stage 1 encodings | `stage1-features` 2.0.0 (`d0d8b07f2ca5e81d`) | `past_daily_lag0` as the drafted `stage1-features` 3.0.0-candidate; MLP encoder-isolated early stopping | Step 44A; lag and monthly rules to PR 49 | none |
| Stage 2 bins | binning identity `9cdd79af…` (29 bins) | NSD `gap_placed`, **qualified**; `eci_warm` required as a filter on Pass 2 partner rules | Step 44B | ECI `share_matched`, `gap_placed`, `tail_selected`; NSD `share_matched` |
| Stage 3 features | 13 features (`8f1df13b…`), DAG `3a8d1957…` | `jed_alone` with its Step 5A pairs (the only candidate both recommended and nominated) | Step 44C | `drop_first`, `drop_second`, `past_density` (to PR 49 as a temporal question) |
| EBM interactions | the three accepted pairs | the procedural selector (interpret FAST, fitted inside every fit) and the budget question | Step 44C | reselection under the three-pair budget for `jed_alone` and `drop_first` |
| Stage 3 DAG ownership | `feature_research/feature_engineering/` (Stage 3) and `shared/stage2/feature_engineering.py` (Stage 2), both unchanged | one production-owned implementation, with compatibility aliases | the adopting contract version (PR 45) | any move in PR 42 |

- **Every frozen contract is retained.** No candidate is adopted. Adoption is PR 45's, and only for a candidate
  that Step 44A, 44B or 44C accepts under predefined criteria.
- **Three methodology revisions are recommended** as separately versioned changes (section 5.2):
  - past-only encodings for any claim about prediction time;
  - encoder-isolated MLP early stopping;
  - fold-fitted derivation of every label-chosen constant in nested or repeated evaluations, with the scope of
    its reference inputs stated.
- **The frozen random-split benchmark stays valid as a random-split benchmark.** No held-out label reaches its own
  evaluation encoding. Every target-derived feature still uses labels of later contacts, so the benchmark is not
  evidence of forward-in-time performance (section 2.4).
- **One accepted result is stated in no earlier review document.** In the accepted Windows run of Step 5B,
  reselection under the three-pair budget makes `jed_alone` miss Step 5A's cascade screen: its net Stage 1
  corrections on Stage 2 abstentions fall from 486 to 389, against the 456.86 needed. `drop_first` falls from 441 to
  397. Step 44C therefore receives `jed_alone` with its fixed Step 5A pairs, not the reselected fit (section 4.4).

**Scope.** Consolidation only. Nothing is fitted, rerun or recorded, and no test row is read. The accepted records
are read through `research_records.read_run_record`, which checks each record's sidecar and content digest. The
production feature code, the frozen contracts, artifacts and recorded identities, and the accepted records and
their registered verdicts are unchanged. Where a later sensitivity qualifies a registered verdict, the verdict is
kept and the handoff qualified.

**Four dispositions.** Each item below has one. A methodology revision (3) states a method; the candidates that
implement it are advanced (2) and evaluated before any adoption.

1. **Retain** the frozen baseline: the accepted PR 39 contract, artifact and identity stay what PRs 43–49 read.
2. **Advance** a research candidate: a later step evaluates it under its own registered criteria. Nothing is adopted.
3. **Recommend a separately versioned methodology revision**: the evidence supports changing a method, under a
   new version with its own identity, if and when a stage is refitted. The adopting PR implements it.
4. **Adopt** a revised production contract: new version, regenerated artifacts, validation. **PR 42 adopts
   nothing**; adoption is PR 45's.

**Evidence.** The accepted records (section 1). The review documents of Commits 2–5 quote their sandbox runs; this
document quotes the accepted Windows records and says so where they differ (section 8.1). The figures and verdicts
the dispositions rest on are cited as [S…] and listed in the evidence index (section 10). Other figures are quoted
from the review documents named beside them. `tests/feature_research/test_pr42_contract_recommendations.py`
resolves each index row in its named record and requires the stated value.

## 1. The accepted evidence

<!-- BEGIN ACCEPTED RUNS -->
| Step | Run | Record | Content SHA-256 | Registration | Review document |
|---|---|---|---|---|---|
| Commit 2 | `c2_20261122_124809` | `feature_inventory` | `b824bd4bdc65…` | — | [`feature_inventory.md`](feature_inventory.md) |
| Step 3A | `c3_20261122_201830` | `encoding_audit` | `9751f4affcf6…` | — | [`encoding_audit.md`](encoding_audit.md) |
| Step 3B | `c3_20261122_232809` | `temporal_validity` | `26de64937438…` | `pr42-step3b-registration` 1.1.0 `0ccc9fa34900…` | [`temporal_validity.md`](temporal_validity.md) |
| Step 4A | `c4_20261123_081749` | `binning_drift` | `46c26889ec06…` | — | [`binning_drift.md`](binning_drift.md) |
| Step 4B | `c4_20261123_151515` | `binning_candidates` | `f223bf72e34d…` | `pr42-step4b-registration` 1.0.0 `066195f04c96…` | [`binning_candidates.md`](binning_candidates.md) |
| Step 5A | `c5_20261124_002645` | `stage3_simplification` | `d70330e1d8fa…` | `pr42-step5a-registration` 1.0.0 `08b3a74474a3…` | [`stage3_simplification.md`](stage3_simplification.md) |
| Step 5B | `c5_20261124_122029` | `interaction_selection` | `0d999bc8c58a…` | `pr42-step5b-registration` 1.0.0 `9d18391acf8a…` | [`interaction_selection.md`](interaction_selection.md) |
<!-- END ACCEPTED RUNS -->

- **Platform.** Every accepted run is a Windows run (Python 3.14.3), the first run that wrote its step's record.
- **Reproduction.** The latest run, `c5_20261124_122029`, holds all seven records.
  - Its Step 3A–5A records have the accepted runs' content digests.
  - Its inventory differs from the Commit 2 record only in `duplicates.callers`: the PR 42 research modules added
    after Commit 2 import the feature-engineering modules. It has the content digest of the Step 3A run's inventory.
  - Its recorded code identity equals commit `3a6a08b` (the Commit 5 head) for all 70 files it traces. Step 6A
    changes one of them, the notebook, in two Markdown cells.
- **Production feature code.** Every accepted run recorded the same LF SHA-256 for every production feature module
  (Stage 1–3 feature engineering, the Stage 2 binning, the Stage 3 DAG modules and the EBM interaction definition).
  PR 42 changed none of them, and Step 6A changes none (the test checks both).
- **Superseded runs.** `c5_20261123_220648` stopped at Step 5A's statement check (Cell 58) and wrote no Step 5A
  record. Its earlier records equal the accepted ones by content.

## 2. Stage 1 — temporal validity and the encodings (Steps 3A–3B)

### 2.1 Five questions kept apart

| Question | Finding | What follows |
|---|---|---|
| **Own label in training-side construction** (Step 3A mechanism a) | Present in every fit, with small weight: a fit row's own label enters its bin or cell rate. The mean own-label share in the final fit is 0.0006 of the euribor bin rate's label mass [S1.1], at most 0.84% for the smallest day × month cell | The models are trained on encodings that include each row's label and scored on encodings that do not. This is a training/scoring asymmetry, not a held-out leak. A past-only encoding removes it |
| **Held-out-label contamination** (mechanism b) | Absent. Measured 0 on all 22 feature × path groups of Stages 1–3 [S1.2]. Label flips through the production builders leave held-out designs unchanged, and the final fits map the test split with training labels only | Every cross-validated output of the accepted cascades is fold-isolated. This is what makes the random-split benchmark valid as a random-split comparison |
| **Temporal look-ahead under random splitting** (mechanism c) | Present on every path, including every fold. Out of fold, 41% of the euribor bin rate's label mass and 37% of the day × month rate's come from later contact days [S1.3, S1.4], 47% for NSD and 52% (influence) for ECI [S1.5, S1.6] | Random folds remove (b), not (c). The absence of (b) is not temporal validity |
| **MLP early-stopping boundary** | The black-box MLP draws its 15% early-stopping subset [S1.7] from rows whose Stage 1 encoders and scaler were fitted with the subset's labels. The outer folds are outside all of this | Isolating the subset moves the median stopping epoch from 17 to 32.5 [S1.8, S1.9]. Under the random split the held-out ROC-AUC difference is not resolved: −0.0017 [−0.0037, 0.0004] pooled [S1.10] |
| **Availability at prediction time** | The accepted encodings, as fitted, use labels of later contacts, which do not exist when a row is scored. Past-only versions under an outcome lag of 0 are realizable: they encode 97.2% of training rows [S1.11], and the first three contact days are the cold start | Outcome timing is not recorded, so lag 0 is an optimistic bound. The past-only euribor rate falls back to the median bin rate for 12,636 of the 32,033 encoded rows [S1.12], because the rate keeps reaching new lows |

### 2.2 The past-only experiments

All LR comparisons are paired DeLong differences between whole protocols on identical rows (Step 3B). The past-only
protocol removes look-ahead, the own-label share, part of the fitting rows and the cold start together, so no
difference here isolates look-ahead.

| Comparison (LR) | Pooled [95%] | Mean within windows [95%] |
|---|---|---|
| past-only vs accepted protocol, random split, 32,033 rows | −0.0065 [−0.0105, −0.0025] [S1.13] | — |
| the same on the windows' rows (B − A) | — | −0.0464 [−0.0641, −0.0287] [S1.14] |
| past-only vs accepted protocol, chronological split (D − C) | +0.0141 [S1.15] | +0.0213 [S1.16] |
| chronological vs random split, accepted protocol (C − A) | — | −0.0681 [S1.17] |
| chronological vs random split, past-only protocol (D − B) | — | −0.0005 [−0.0093, 0.0084], not resolved [S1.18] |
| operating under a 20-day outcome lag, lag-consistent | — | −0.0090, not resolved [S1.19] |
| operating under the monthly rule, lag-consistent | — | −0.0472 [S1.20] |

- Under the random split the past-only protocol costs a little pooled and more within windows.
- Under the chronological split it ranks better than the accepted window fit.
- Within windows, a past-only Stage 1 is not resolvably worse chronologically than under the random split; the
  difference is bounded at about ±0.009, not shown to be zero.
- The MLP comparisons have the LR's signs within windows (Step 3B section 4.4). Under the random split, isolated
  early stopping changes no resolved pooled ROC-AUC.

### 2.3 The three-feature macroeconomic representation

**Keep it at the head of both cascades** (the same three features and model classes) and correct its methodology:

- **What the evidence shows.**
  - It is simple and shared by both arms.
  - Making its two target encodings past-only is realizable.
  - The past-only protocol ranks 0.0065 ROC-AUC lower pooled under the random split, and higher under the
    chronological split.
- **What the evidence does not show.** It gives no reason to add Stage 1 features or replace the representation.
  Most of Stage 1's pooled discrimination ranks contact periods; within a window it is 0.49–0.67 even for the
  accepted random-split scores. That is a property of the representation and the data, which no encoding
  definition here changes.
- **What to correct.**
  - The temporal definition of the two encodings: past-only, under a versioned contract (section 5, R1).
  - The MLP's early-stopping isolation (R2).
- **What stays open for Step 44A.** The past-only euribor bin rate cannot extrapolate: in time it is mostly its
  fallback constant until the rate stops falling (98.8% of W1's predicted rows under the updating state). Whether
  the euribor encoding needs a different past-only form within the same three-feature representation is not
  established here.

### 2.4 The random-split benchmark and a prospective claim

- **Valid as a random-split benchmark.** Mechanism (b) is absent on every path of both arms, so the frozen PR 39
  training-side OOF evidence compares models fairly under the random split. PR 43 may use it as such.
- **Not a prospective claim.** Every target-derived feature of Stages 1–3 uses labels of later contacts (c), and the
  MLP's early stopping scores features built partly from its validation labels. Neither the frozen evidence nor any
  Stage 4 result built on it may be presented as forward-in-time performance.
- **Prevalence.** Under a pure prior shift, the frozen GLASS routing meets all three Stage 2 obligations only for
  prevalence between 10.9% and 12.0% [S1.21, S1.22]. The windows run from 6.6% to 49.3%, so a chronological
  evaluation is expected to miss obligations by construction (PR 49).

### 2.5 Stage 1 dispositions

| Item | Disposition | Destination | Evidence |
|---|---|---|---|
| `stage1-features` 2.0.0, fingerprint `d0d8b07f2ca5e81d` | 1, retain | the baseline PR 43 reads | S1.23 |
| `past_daily_lag0`, drafted as `stage1-features` 3.0.0-candidate (names `1d48dea8b294ff74`, definition `fe5d54e195b7a604`) | 2, advance; it implements revision M1 | Step 44A, both arms | S1.24–S1.27 |
| `past_daily_lag5`, `past_daily_lag20`, `past_monthly` | 2, advance as sensitivities only | PR 49 | S1.28 |
| MLP encoder-isolated early stopping | 3, recommend (revision M2) | Step 44A evaluates it with the other MLP changes; adopted only with the next MLP fit (PR 45) | S1.29 |
| An own-label-only (leave-one-out) encoding that keeps every other label | not built; suggested by Step 3B to separate the mechanisms | Step 44A or PR 49, optional | — |

## 3. Stage 2 — NSD/ECI binning (Steps 4A–4B)

### 3.1 The drift, explained

| Bin | Quoted in PR 39 | What it was | Training side: in sample → out of fold | Main cause (Step 4A) |
|---|---|---|---|---|
| `eci_warm` | 1.49–1.67× | two historical deciles inside the bin; the bin itself was 0.92× | 1.038× → 1.031× [S2.1, S2.2] | estimand and threshold placement: no drift in the bin |
| `eci_hot` | 3.53× | the historical top decile, which was the bin | 2.457× → 2.501× [S2.3, S2.4] | the drop lies between the historical fit and the training-side fits: four low-rate May 2009 macro states dilute the bin under every encoder |
| `nsd_elevated` | ~2.49× pooled out of fold, with fold variation | the pooled out-of-fold lift itself | 3.126× → 2.486× [S2.5, S2.6]; fold 0 at 0.617× [S2.7] | one fold's encoder moved whole macro states across both cuts |

All 17 of Step 4A's claims hold in the accepted run [S2.8].

**Encoder refitting versus held-out population.** NSD and ECI are functions of the macro state (346 states over 26
periods), so a cut moves whole states. The two descriptive contrasts do not isolate a cause:

- held-out minus fold-fit pools per-fold differences of both signs;
- fold-fit minus in-sample changes encoders and row weights at once.

What the record does establish:

- **Encoder behaviour.** `nsd_elevated`'s fold-0 collapse is encoder behaviour. Fold 0's encoder places seven
  May 2009 states in the bin and all 50 in-sample states above it, and its own fit rows show the low lift too.
- **Population.** Period mix reproduces every bin's aggregate subscriber count to within 2%.
- **Look-ahead.** Its effect on these lifts is unresolved: no past-only NSD or ECI exists.

### 3.2 The registered candidates and their verdicts

Registration `pr42-step4b-registration` 1.0.0 was fixed before any candidate result [S2.9]. The comparator is the
frozen bins on identical held-out rows.

| Family | Candidate | Focal bin, out of fold | Verdict (registered) | Evidence |
|---|---|---|---|---|
| NSD | frozen (comparator) | `nsd_elevated` 2.486× [2.284, 2.698], 1,421 rows, refit migration 653 | — | S2.6, S2.10 |
| NSD | `gap_placed` | 3.126× [2.902, 3.358], worst fold 2.821×, 1,326 rows, migration 0; held-out bin identical to the frozen in-sample bin | **advance to PR 44** | S2.11–S2.13, S2.12b |
| NSD | `share_matched` | meets A0–A4 | close out: outranked by `gap_placed` | S2.14 |
| ECI | `share_matched` | — | close out: fails A3a, A4 | S2.15 |
| ECI | `gap_placed` | — | close out: fails A1 (T2 fires for `eci_warm`) | S2.16 |
| ECI | `tail_selected` | `eci_hot` 3.481×, migration 735 | close out: fails A4 (depends on May 2009) | S2.17, S2.18 |
| filter | `eci_warm` required on partner rules | effect in 3 of 5 slots; precision 1.194× the partner's in slot 3 | **advance to PR 44 as a filter** (`warm_required`) | S2.19–S2.21 |

**The post-registration sensitivity qualifies `gap_placed`; it does not change its verdict.**

- **What changed.** Taking `gap_placed`'s targets and windows from each fold's own fit rows, instead of the
  full-training reference shares, removes its registered advantage.
- **Result.** Fold 0 stays at 0.644×, 555 rows migrate on refit, and the pooled lift is 2.630× [S2.22–S2.24]. The
  record classifies this as a material change [S2.25].
- **Why.** The registered fold-0 repair and zero migration come from reference inputs that each fold's held-out rows
  and labels helped form.
- **Reading.** The registered 3.126× is not a nested estimate.

### 3.3 What this evidence is, and is not

- **Bin-level, on held-out training rows.** The filter population is every held-out row, not Pass 2's, so the
  Pass 2 obligations were applied as a screen, not as compliance. Required `eci_warm` keeps 25–48% of a partner
  rule's rows as a standalone conjunction. Its effect on router coverage can be smaller, larger or of either sign.
- **Not router-level or cascade-level.** No router was fitted. Under `gap_placed`, every held-out `nsd_elevated`
  row is also in the frozen `eci_hot`, so its marginal value to a router that already uses `eci_hot` is bounded.
- **Random folds only.** Whether any of it holds chronologically is PR 49's question.

### 3.4 Stage 2 dispositions and the Step 44B requirements

| Item | Disposition | Destination |
|---|---|---|
| Frozen binning, identity `9cdd79af…`, 29 bins `7b0d4f50…` | 1, retain | the routers PRs 43–49 read |
| NSD `gap_placed` | 2, advance, qualified | Step 44B; a versioned binning revision only if 44B adopts it (the record leaves it "pending PR 44" [S2.26]) |
| `eci_warm` required on GLASS Pass 2 partner rules | 2, advance | Step 44B |
| ECI candidates, NSD `share_matched` | closed out by the registered criteria | — |
| Fold-fitted bin derivation with stated reference inputs | recommended as revision M3 | Step 44B and PR 49 |

What Step 44B must still measure before any adoption:

| Requirement | What PR 42 established | What Step 44B measures |
|---|---|---|
| Coverage | standalone support only (`gap_placed` keeps 75% of the frozen bin's support in its weakest fold; the filter keeps 25–48% of a partner's rows) | router coverage under the lexicographic-coverage objective, Pass 1 and Pass 2 separately |
| Constraint feasibility | a screen on all held-out rows | the shared obligations (Pass 1 NPV and leakage; Pass 2 precision and support) on the actual Pass 2 population, at fit time and out of fold |
| Complementarity | `gap_placed`'s `nsd_elevated` lies inside the frozen `eci_hot` | marginal routed rows and unique-correct contributions against the frozen router, both arms |
| Fold stability | registered: no fold-0 collapse, zero migration; fold-local: neither | both reference variants (the registered constants and the fold-local ones) fitted inside every Stage 2 cross-fit fold; stated which one is used |
| Computational cost | `gap_placed` needs a sort of each fold's macro states; no extra engineer fit | the ILP's solve time and candidate count with the filter in the rule space |
| Leakage-safe procedural refitting | held-out flips leave every fold fit unchanged, conditional on the fixed reference inputs | `stage2_crossfit.g13` extended to the cut fit, required inside every cross-fit fold, with the reference inputs' scope stated |

A Stage 2 comparison against revised Stage 1 or Stage 3 models also needs a new protocol version first (F10: the
PR 38 protocol pins the historical LR and EBM flag fingerprints).

## 4. Stage 3 — simplification and interaction selection (Steps 5A–5B)

### 4.1 The frozen inventory

- **Features.** 13 features (ordered-names SHA-256 `8f1df13b…`), the same block for the GLASS EBM and the
  black-box XGBoost (block fingerprint `9cc10fa1…`).
- **Label-dependent features.** Three: `decay_x_density` through NSD, `economic_curvature_intensity` and
  `dow_month_encoded`. All three look ahead (Step 3A).
- **Pairs.** Three EBM pairs, GLASS only. The TIER 1 pair came from a FAST ranking; the two TIER 2 pairs were chosen
  by judgement (Step 5B).
- **The one correlated pair.** At |r| ≥ 0.7 it is `euribor3m_sigmoid_slope` and `emp_var_rate_sigmoid_slope`
  (r 0.725). Both slopes are period-level.

### 4.2 The candidates (accepted Windows run of Step 5A)

Every candidate is compared with the refitted baseline on the same 32,950 out-of-fold rows, folds and settings.

| Configuration | Features / pairs | ΔROC-AUC [95%] | ΔPR-AUC | ΔNested F2 | Calibrated ECE | Changed term's fold sd | Regime-step fold sd | Net Stage 1 corrections on 14,628 abstentions | Step 5B | Cascade |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline (refitted) | 13 / 3 | 0.7933 | 0.4171 | 0.5509 | 0.0021 | `decay_x_density` 0.512 | 0.631 | 530 | — | — |
| `jed_alone` | 13 / 2 | +0.0001 [−0.0004, 0.0006] | +0.0007 | +0.0008 | 0.0029 | `joint_economic_decay` 0.085 | 0.065 | 486 | recommended | **nominated** |
| `drop_first` | 12 / 2 | −0.0003 [−0.0009, 0.0002] | −0.0019 | +0.0001 | 0.0017 | kept slope 0.050 | 0.746 | 441 | recommended | not nominated |
| `drop_second` | 12 / 2 | −0.0013 [−0.0022, −0.0003] | −0.0022 | +0.0014 | 0.0019 | kept slope 0.109 | 0.641 | 633 | not recommended | nominated |
| `past_density` | 13 / 2 | +0.0005 [−0.0005, 0.0014] | +0.0015 | −0.0010 | 0.0025 | `decay_x_density_past` 0.171 | 0.071 | 689 | not recommended | nominated |

Evidence: S3.1–S3.16 and their lettered rows. The cascade threshold is 456.86, the refitted baseline's 530 less
0.5% of the abstentions.

- **`jed_alone` is the clearest simplification.**
  - *What it changes.* It replaces the least stable, label-derived term with its label-free macro factor
    (r 0.9935).
  - *Stability.* The decay slot's fold sd falls from 0.512 to 0.085, and the regime effect is identified
    consistently across folds (0.631 → 0.065).
  - *Pair lost.* `prior_x_stress × decay_x_density`, at most about 0.07 log-odds in a well-supported cell.
  - *Label dependence.* It removes NSD from Stage 3 altogether, so Stage 3 keeps two label-dependent features.
  - *Costs.* Calibrated ECE rises to 0.0029, still far under the 0.05 gate. `emp_var_rate_sigmoid_slope`'s top value
    (late 2010) turns negative with fold disagreement.
- **Only one accepted pair matters inside the joint support.** `campaign × emp_var_rate_sigmoid_slope` reaches 0.62
  log-odds in the Windows fit (range 0.81), with fold sd at most 0.06. The two TIER 2 pairs carry at most 0.051 and
  0.072 log-odds, and the euribor × curvature pair's fold sd is about four times its effect [S3.17–S3.19].
- **`past_density` is not fully past-only.** Its 917 held-out rows on the first three contact days take a fallback
  that depends on later fit labels [S3.20]. Its case is temporal, which PR 49 would test with a past-only fallback.
- **Comparisons use the refitted baseline.** The accepted EBM's own stored decisions net 447 corrections on the
  abstentions against the refitted baseline's 530 [S3.21, S3.22]. Step 5A attributes the difference to a learning
  rate transcribed to six decimals and to the platform.

### 4.3 What Stage 3 numbers are and are not

- **Standalone Stage 3 out of fold.** Every metric above is a Stage 3 metric on the training side under random outer
  folds.
- **Not cascade decisions.** The complementarity counts are disagreements with Stage 1 under a registered screen. A
  higher net is not a better cascade until PRs 44–45 compare cascades at comparable coverage under the same
  operating obligations.
- **Only the GLASS EBM was fitted.** The black-box XGBoost consumes the same 13 features and was not fitted on any
  candidate. A Stage 3 feature-set revision changes both arms' input; to keep the arms comparable it must be
  evaluated, and adopted or rejected, for both.

### 4.4 Procedural interaction selection (accepted Windows run of Step 5B)

The selector is interpret 0.7.8's FAST ranking inside the accepted EBM (`interactions=3`, mean rank over 14 bags),
run inside every fit. Acceptance passed [S3.23].

- **Recovery.** On the frozen 13 features it recovers only the TIER 1 pair: first on all rows, and selected by all
  five outer fits [S3.24, S3.25]. The TIER 2 pairs rank 30th and 39th of 78 on all rows (38th in the sandbox), and
  no outer fit selects either [S3.26–S3.28].
- **Recurrence.** What recurs is `campaign` crossed with the economic regime: `decay_x_density × campaign` (4 of 5)
  and `euribor3m_sigmoid_slope × campaign` (3 of 5). Read as one campaign-by-regime effect divided among correlated
  macro features: an interpretation of the surfaces, not a tested decomposition.
- **Reselection under the three-pair budget.** Neither candidate changes materially against its own Step 5A fit
  (pooled ROC-AUC −0.0006 for `jed_alone`, +0.0004 for `drop_first`; every interval includes 0) [S3.29–S3.32].
  Both flag more rows, and on Stage 2 abstentions both lose Stage 1 corrections:

  | Configuration | Net on abstentions | Screen (≥ 456.86) |
  |---|---|---|
  | `jed_alone`, Step 5A pairs | 486 | meets |
  | `jed_alone`, reselected | 389 | **misses** |
  | `drop_first`, Step 5A pairs | 441 | misses |
  | `drop_first`, reselected | 397 | misses |

  Evidence: S3.33–S3.37. Reselection changes `jed_alone`'s screen verdict. The record flags this for review; it is
  not evidence about cascades in either direction.
- **Sandbox and Windows.** The Windows run flags 9 items against the sandbox figures of the Step 5B review document
  [S3.38]: one fold's pair order, five changes of direction below the materiality thresholds, two statements that
  no longer hold (section 8.1), and the screen verdict above. All acceptance checks pass.

### 4.5 Stage 3 dispositions and the Step 44C handoff

| Item | Disposition | Destination |
|---|---|---|
| The 13 features, the block and the three pairs | 1, retain | PRs 43–49 read them |
| `jed_alone` with its two Step 5A pairs and one empty slot | 2, advance | Step 44C, both arms, then the cascade comparisons of PRs 44–45 |
| `drop_first`, and the exploratory `jed_alone` + `drop_first` (12 features, one pair; not fitted) | not advanced to the cascade path; may be registered as exploratory in Step 44C, each with its own complementarity check | Step 44C, optional |
| `drop_second` | closed (loses the only substantive pair) | — |
| `past_density` | not advanced for Stage 3; a temporal question | PR 49, with a past-only fallback |
| Procedural interaction selection | 2, advance as a procedure; reselection at three pairs not advanced for either candidate | Step 44C |

**Step 44C receives:**

- `jed_alone` as nominated, with its two fixed pairs and one empty slot. It must be evaluated for XGBoost on the
  same feature set before any adoption (section 4.3).
- The procedural selector, reproducible and fitted inside every fit, with the evidence that it supports the TIER 1
  pair and neither TIER 2 pair.
- The budget question that Step 5B leaves open. Step 44C should compare:
  - the three fixed pairs;
  - none;
  - one or two pairs;
  - one campaign pair fixed by construction, against a ranking that divides the campaign-by-regime effect among
    correlated macro features;
  - capped automatic detection.

  Each needs the cascade screen on Stage 2 abstentions, since reselection lowered both candidates' corrections
  there.
- The comparator: the refitted baseline, never the accepted artifact's stored decisions.

## 5. Feature-contract versioning and adoption requirements

### 5.1 The frozen contracts and identities, all retained

| Contract or identity | Value | Read by |
|---|---|---|
| Stage 1 feature contract | `stage1-features` 2.0.0, fingerprint `d0d8b07f2ca5e81d` | both Stage 1 arms |
| Stage 2 binning | identity `9cdd79aff9d61101…`, 29 ordered bins `7b0d4f507f3becd4…` | both routers |
| Stage 2 inner-fold pathway | `fold_label_excluded`, `stage2-inner-fold-feature-isolation` 1.0.0 | both routers' cross-fits |
| Stage 2 operating contract | `pr37-stage2-primary` 1.0.0, `0817825b9eecbd11` | both routers |
| Stage 3 features | 13 ordered names `8f1df13bf14ca472…`; block fingerprint `9cc10fa1f5cf8d85…` | the EBM and XGBoost |
| Stage 3 DAG implementation | LF SHA-256 `3a8d195700ce3462…` over 11 modules | the block-cache identity, PR 42 runs |
| EBM interaction pairs | `campaign × emp_var_rate_sigmoid_slope`, `euribor3m_sigmoid_slope × economic_curvature_intensity`, `prior_x_stress × decay_x_density` | the GLASS EBM |
| Stage 4 Meta-XGB inputs | 6 channels `6e2be2c81f71c82e…` | Meta-XGBoost |

The test checks every code-computable one against the accepted Step 5B run's frozen references.

### 5.2 Recommended methodology revisions (disposition 3)

- **M1, past-only target encodings for any claim about prediction time.**
  - *Rule.* A deployment or forward-in-time claim may rest only on encodings that use labels known when a row is
    scored, under a stated outcome lag, with same-day labels excluded and a stated cold start.
  - *Stage 1.* Drafted as `stage1-features` 3.0.0-candidate.
  - *Stages 2–3.* Not built (R7).
  - *Scope.* The random-split benchmark keeps the accepted contracts.
- **M2, encoder-isolated early stopping for the black-box MLP.**
  - *Rule.* The estimator's validation subset is drawn first, and the encoders and scaler are fitted without it.
  - *Why.* The change removes a validation-label path and has no resolved random-split cost.
  - *Scope.* It changes a training procedure, not the feature contract.
- **M3, fold-fitted derivation of every label-chosen constant in nested or repeated evaluations.**
  - *Applies to.* Bin cuts and interaction pairs, refitted inside each fold or split, with their reference inputs
    and the scope of those inputs stated.
  - *Why.* Step 4B showed that full-training reference inputs can carry a fold's held-out labels into its fit.
  - *Destination.* PR 49's repeated splits use this rule. Step 44B and Step 44C state it for their candidates.

### 5.3 Requirements per revision

Each revision below would be a new version beside the frozen one, never an edit of it.

**R1. Stage 1 past-only encodings (`stage1-features` 3.0.0-candidate).**
- *Frozen contract:* `stage1-features` 2.0.0.
- *Revision:* `subscribe_rate_by_euribor_bin_past` and `subscribe_rate_by_dow_month_past` beside
  `cellular_downturn`; rule `past_daily_lag0`, refitted at each contact day.
- *Evidence and uncertainty:* section 2.2. Outcome timing is unknown, the euribor fallback covers 39% of encoded rows,
  and nothing was retuned for the past-only encodings.
- *Changes:* feature values, semantics (new names), fitting procedure (expanding window) and identities (new
  fingerprints).
- *At prediction time:* the contact-day time index (`encoding_audit.build_time_index`) becomes a pipeline input; a
  rule for cold-start rows is needed at scoring time.
- *Audits:* Step 3B's invariance checks (no unavailable label reaches an earlier encoding) inside the Stage 1
  cross-fit; the F3 and F19 fold-isolation audits for the new encoder.
- *Dependencies:* both Stage 1 arms; the Stage 4 inputs of both arms. A Stage 2 comparison against a revised Stage 1
  needs a new protocol version (F10).
- *Before adoption:* Step 44A's criteria, including a chronological comparison and not the random split alone; then
  PR 45 regenerates both Stage 1 arms and their calibrators, then Stage 4, beside the PR 39 artifacts.

**R2. MLP encoder-isolated early stopping.**
- *Frozen contract:* the black-box Stage 1 training procedure; the feature contract is unchanged.
- *Revision:* the isolated split Step 3A defines (`encoding_audit.MLP_ISOLATED_SPLIT`) and Step 3B ran: the
  estimator's split drawn first.
- *Evidence and uncertainty:* section 2.1. No resolved random-split effect; one seed.
- *Changes:* fitting procedure and the black-box Stage 1 artifact identity; no feature value.
- *At prediction time:* no change.
- *Audits:* a check that no early-stopping-subset label enters any encoding or the scaler.
- *Dependencies:* the black-box Stage 1 artifact, Meta-XGBoost's inputs, the 2×2 crossover.
- *Before adoption:* Step 44A (with its other MLP changes); PR 45 refits the MLP once.

**R3. Stage 2 NSD `gap_placed` binning.**
- *Frozen contract:* binning identity `9cdd79af…`.
- *Revision:* a fitted NSD cut procedure in place of constants; the 29 bin names and order unchanged.
- *Evidence and uncertainty:* section 3.2, qualified by the fold-local sensitivity.
- *Changes:* feature values (bin memberships), fitting procedure, and a new binning version and identity.
- *At prediction time:* the cuts become fitted state of the Stage 2 engineer, stored with the routers.
- *Audits:* G13 covering the cut fit inside every cross-fit fold, with the reference inputs' scope stated.
- *Dependencies:* both routers (Stage 2 features are shared by the two arms), the Stage 4 inputs of both arms, and
  any Stage 3 evaluation on router-deferred rows. The Step 4B record lists the Stage 3–4 artifacts; PR 45's reuse
  checks decide which must be regenerated.
- *Before adoption:* Step 44B (section 3.4); PR 45 regenerates the Stage 2 frames, GLASS rules and RF forests.

**R4. `eci_warm` required as a filter on GLASS Pass 2 partner rules.**
- *Frozen contract:* the GLASS rule space and profile (the bins are unchanged).
- *Revision:* conjunctions with `eci_warm` admitted as filters on Pass 2 partner rules.
- *Evidence and uncertainty:* section 3.2; support loss as a standalone conjunction; router effect unmeasured.
- *Changes:* the GLASS selection procedure and the GLASS router identity; no feature value. The RF Router is
  unaffected, and the two arms keep the same 29 bins.
- *At prediction time:* no change.
- *Audits:* the existing G13 and Pass 2 obligation checks.
- *Dependencies:* the GLASS router, the GLASS Stage 4 inputs.
- *Before adoption:* Step 44B (coverage and obligations on the Pass 2 population).

**R5. Stage 3 `jed_alone`.**
- *Frozen contract:* the 13 features, `8f1df13b…`.
- *Revision:* `decay_x_density` replaced by `joint_economic_decay`; the EBM keeps two pairs and one empty slot.
- *Evidence and uncertainty:* section 4.2. EBM only (no XGBoost fit); random folds; reselection lowers its
  abstention corrections.
- *Changes:* the feature list (and its SHA-256), semantics (a label-free factor), the block identity and
  fingerprint; NSD leaves Stage 3.
- *At prediction time:* improved. `joint_economic_decay` is a function of the row's three macro values.
- *Audits:* a G2/G3 inventory update (two label-dependent Stage 3 features); the block's fold-isolation checks.
- *Dependencies:* the GLASS EBM, the black-box XGBoost, both Stage 4 arbiters, the F10 protocol version.
- *Before adoption:* Step 44C for both arms; PR 45 rebuilds the block and refits both Stage 3 models, then Stage 4.

**R6. Procedural EBM interaction selection (and any change of budget).**
- *Frozen contract:* the three pairs from `define_ebm_interactions`.
- *Revision:* `IS.ProceduralEBMFactory` (`interactions=k`), selection inside every fit.
- *Evidence and uncertainty:* section 4.4. Selection near ties can differ by platform and interpret version.
- *Changes:* the selection procedure and the GLASS EBM identity; no feature value.
- *At prediction time:* no change.
- *Audits:* Step 5B's ranking-to-fitted-term agreement and fit-frame checks in every fit.
- *Dependencies:* the GLASS EBM, the GLASS Stage 4 inputs.
- *Before adoption:* Step 44C's budget comparison with the cascade screen.

**R7. Past-only Stage 2–3 constructions (NSD, ECI, day × month, the past-only density's fallback).**
- *Frozen contract:* the Stage 2 sources and the Stage 3 features that inherit them.
- *Revision:* defined by Step 3B section 5 and Step 5A section 4, not built. Design choices remain open: NSD's
  reference set grown or redrawn; ECI's 0.0 fallback; a past-only fallback for the density.
- *Evidence and uncertainty:* construction and attribution only; no predictive estimate.
- *Changes:* feature values, semantics and identities in Stages 2 and 3.
- *At prediction time:* the contact-day index and the outcome lag become inputs.
- *Audits:* the Step 3B invariance checks per family; G13 for the Stage 2 sources.
- *Dependencies:* both arms at Stages 2–4.
- *Before adoption:* PR 49 runs them in the chronological cascade. Any adoption is a later, separately versioned
  decision.

### 5.4 Adoption order (PR 45)

1. Register each accepted candidate as its own contract version before regeneration. PR 42 drafted only the Stage 1
   one.
2. Stages 1–3 train independently, so their regenerations do not depend on one another. Register a new Stage 2
   protocol version before any Stage 2 comparison against revised Stage 1 or Stage 3 models (F10).
3. Move the Stage 2/3 feature implementation (section 6) with the first adopted Stage 2 or Stage 3 revision.
4. Stage 4 last, on the regenerated OOF tables.
5. Preserve the PR 39 artifacts and records, give every regenerated artifact a new run id, run the PR 39 records'
   `check` commands on the frozen artifacts, and update the tests that pin identities, this step's included.

## 6. The Stage 3 feature DAG and the duplicated implementation

### 6.1 What the repository shows

| Question | Finding | Evidence |
|---|---|---|
| Which implementation does the accepted Stage 3 pipeline run? | `feature_research/feature_engineering/` (`crisis`, `integrals`, `derivatives`, `temporal`, `prior`, `overlap`, `_checks`) with `feature_research/config.py`, orchestrated by `shared.stage3.feature_engineering.EBMFeaturePipeline`. Both Stage 3 arms use it. Its only production importer is `shared/stage3/feature_engineering.py` | Commit 2 G7; S4.1 |
| And the accepted Stage 2 pipeline? | `shared/stage2/feature_engineering.py`: self-contained by design ("no imports from feature_research"), with its own copies of crisis, integrals, derivatives, temporal, prior and overlap | module docstring; Commit 2 G6 |
| Are they equivalent? | In sample, equal on all 30 compared columns except `dow_month_encoded`, which differs only by floating-point rounding (5.6e-17) [S4.2]. On the five shared folds' held-out rows they diverge: the copy re-cuts quantile bins with rounded `IntervalIndex` categories, so `euribor3m_local_rate` differs on 8 rows (up to 0.243), and `economic_curvature_intensity` on the same 8 rows (up to 0.0013) [S4.3, S4.4]. The copy imputes remaining NaN or inf with the transformed frame's median, where the research modules raise | Commit 2 record `duplicates.parity` |
| Other consumers | Research modules: PR 42's inventory, frozen-reference and encoding-audit modules, tests, and the exploratory notebook (E 11, E 12G: `FeaturePipeline` and the research registries). Stage 2 copy: both routers' cross-fit and full-train fits, `stage2_binning`, `stage2_crossfit` (G13 imports its private constants `_KDE_KEY_FEATURES`, `_KDE_LARGE_DS_THRESH`, `_apply_binning_thresholds`), `upstream_regeneration.checks`, G 10 (legacy) and B 16 | Commit 2 G7 |
| Compatibility obligations | Every Stage 3 artifact stores its fitted `EBMFeaturePipeline` (`refit["feature_pipeline"]`), whose engineers are instances of `feature_research.feature_engineering.{integrals,derivatives,temporal}` classes. The accepted EBM and XGBoost artifacts are unpickled by these import paths, by the PR 38 boundary extractors among others. No Stage 2 save path stores the engineer: the routers are saved, not it | `shared/stage3/runner.py`, `shared/stage3/artifact.py` |
| Recorded identities a move would change | The block-cache identity hashes each reachable module's raw bytes under its module name, so any move or rename rebuilds and renames the cache. The PR 42 DAG digest `3a8d1957…` covers 11 module paths [S4.5]. `research_records.CODE_FILES` names each file. PR 39 Step 3A's `CODE_FILES` names `shared/stage2/feature_engineering.py` | `shared/stage3/block_cache.py`, `feature_research/frozen_references.py` |
| Dependency boundary | Importing `shared.stage3.feature_engineering` also imports the research package root (`feature_research.config`, `interactions`, `metrics`, `separation`, `validation`, `visualization`), the research `FeaturePipeline`, `optuna` and `matplotlib.pyplot`. Importing the Stage 2 copy imports nothing from `feature_research` | checked on this branch |

### 6.2 Recommendation

- **Where the DAG belongs.**
  - *Owner.* One production-owned implementation of the Stage 2/3 feature DAG under `shared/` (for example a
    `shared/features/` package; the name is the adopting version's choice), imported by Stage 2 and Stage 3 alike.
  - *Dependencies.* Research code imports it; production never imports `feature_research`. Its seed is a
    production constant rather than `feature_research.config.RANDOM_SEED`.
  - *Numerical conventions.* Chosen once and recorded in the adopting contract versions: quantile edges exact or
    rounded, and non-finite values raised or imputed.
  - *What stays in research.* The research `FeaturePipeline` and the historical registries in
    `feature_research.feature_engineering.__init__`.
- **Whether the copies can be retired.** Yes, but not on their own.
  - *Values.* Pointing Stage 2 at the research modules, or Stage 3 at the copy, changes accepted values: ECI on 8
    held-out rows of the shared folds, and the non-finite handling. That changes Stage 2 bin memberships or Stage 3
    features.
  - *Identities.* Even a byte-identical move changes the block-cache identity and file name, the PR 42 DAG digest,
    the code-identity digests and the pickled class paths.
  - *When.* Retire the copies in the same adopting version as the first Stage 2 or Stage 3 contract revision that
    PR 45 adopts, regenerating once. If PR 45 adopts none, keep both implementations, with the parity measured here.
- **What any move must keep.**
  - `feature_research.feature_engineering.<module>` as compatibility aliases that resolve to the production modules
    (the `glass_pipeline.meta_ebm` pattern), so the PR 39 Stage 3 artifacts still unpickle.
  - The PR 39 records' `check` commands, run on the frozen artifacts.
- **Deferred.** Everything above. Step 6A moves, deletes and consolidates nothing.

## 7. Downstream handoff matrix

| Destination | Receives | Supporting records | Follow-up evaluations | Unresolved | Before adoption |
|---|---|---|---|---|---|
| **PR 43** | The frozen PR 39 training-side OOF evidence as random-split evidence; the claim boundary | Step 3A `encoding_audit`; Step 3B `temporal_validity` (`interim_finding.pr43`) | none from PR 42 | how look-ahead shapes Stage 4 voter behaviour | nothing to adopt; label every Stage 4 result as random-split, never forward-in-time |
| **Step 44A** | `past_daily_lag0` as `stage1-features` 3.0.0-candidate (both arms); MLP encoder-isolated early stopping | `temporal_validity` (`contracts`, `predictive`) | tuning under each candidate; a chronological comparison in the adoption criteria; the euribor encoding's fallback behaviour | which of the past-only protocol's changes accounts for its differences; outcome timing | R1/R2 requirements (section 5.3); F10 protocol version for Stage 2 comparisons |
| **Step 44B** | NSD `gap_placed` (qualified, both reference variants); `eci_warm` required on Pass 2 partner rules | Step 4A `binning_drift`; Step 4B `binning_candidates` (`decision`, `sensitivity`, `filter`) | the six router-level requirements (section 3.4) under the shared obligations | whether a reference formed without each fold's held-out rows, but not by that fold's own engineer, repairs fold 0 | R3/R4 requirements; G13 extension |
| **Step 44C** | `jed_alone` with its Step 5A pairs; the procedural selector; the budget comparison | Step 5A `stage3_simplification`; Step 5B `interaction_selection` (`complementarity`, `review`) | XGBoost on the same feature set; budgets with the cascade screen; the exploratory 12-feature combination if registered | why reselection lowers abstention corrections; the selector's platform and version sensitivity | R5/R6 requirements; refitted-baseline comparator |
| **PR 45** | adoption of accepted candidates; dependency-ordered regeneration; the DAG consolidation | this document, sections 5–6 | parity of every regenerated artifact with its contract; the PR 39 `check` commands | — | section 5.4 |
| **PR 49** | the chronological protocol (`TV.PR49_PROTOCOL`); past-only Stage 1 rules and lags; past-only Stage 2–3 constructions (R7); procedural refits of bins and pairs in every split (M3); prevalence-shift reporting | `temporal_validity` (`pr49_protocol`, `stage2_prior_shift`); `binning_candidates`; `stage3_simplification` | cascade-level chronological evaluation of both arms; `tail_selected` and `past_density` as temporal questions; whether exposed test rows join | which part of the past-only protocol explains its differences; outcome lag at cascade level | missed obligations reported, never refitted away; robustness evidence, not independent validation |

## 8. Inconsistencies and uncertainties found

### 8.1 Accepted records against the review documents

None of these changes a registered verdict.

- **Step 5B.** `interaction_selection.md` quotes the sandbox run and leaves the cascade screen to the Windows record.
  That record reports what section 4.4 gives:
  - `jed_alone` reselected misses the screen (389), and `drop_first` reselected misses it (397);
  - `drop_first`'s reselected Δnested F2 is +0.0005, not the document's −0.0031 [S3.32];
  - its PR-AUC is better than its Step 5A fit's in 4 of 5 folds, not 5 [S3.32b];
  - `jed_alone`'s reselected Δnested F2 is −0.0006 against the document's +0.0001, within the document's tolerance
    [S3.29b];
  - `prior_x_stress × decay_x_density` ranks 39th, not 38th;
  - the materiality verdicts are unchanged.
- **Step 3B.** Three pooled lag-consistent LR figures in `temporal_validity.md` section 4.3 differ from the accepted
  record. The within-window figures, on which the findings rest, are identical.

  | Figure | Document | Accepted record |
  |---|---|---|
  | lag 20 | −0.1509 | −0.1512 [S1.30] |
  | monthly | +0.0280 | +0.0083 [S1.31] |
  | lag 20 against the lag-0 model fed lag-20 encodings | −0.1008 | −0.1011 [S1.32] |

  The monthly sign is the same, so the document's statement that it reverses still holds.

### 8.2 Documentation and code comments (not changed here)

- **The "importance=0.205" comment.** The TIER 1 comment in `glass_pipeline/ebm/interactions.py`, and the pruning
  note in `shared/stage3/feature_engineering.py`, cite it. No printed output of the pinned exploratory notebook gives
  it (Step 5B). Both files are production code; correct them with the adopting version.
- **A stale module name.** `shared/stage3/feature_engineering.py`'s docstring is headed
  `glass_pipeline.ebm.feature_engineering`.
- **The roadmap's Step 4A item** still calls 1.49–1.67× and 3.53× the original lifts. Step 4A showed that they are
  decile lifts (section 3.1). The item is completed and is kept as written.
- **Notebook map section 5** named three duplicated modules; the copy also inlines prior, overlap and its own
  day × month encoder (Commit 2). Step 6A updates that row.

### 8.3 Remaining uncertainties

- **Outcome timing.** It is not recorded, so lag 0 is an optimistic bound, and same-day order is unresolved.
- **Inference.**
  - One seed and one fold plan per step.
  - DeLong and Wilson intervals treat rows as independent.
  - Five chronological windows of unequal size and prevalence.
- **Not built.** Past-only NSD, ECI and day × month constructions. The effect of look-ahead on the Stage 2–3 families
  is unestimated.
- **Not fitted.** No router, no XGBoost on a Stage 3 candidate and no cascade. Every PR 42 result is feature-, bin-
  or stage-level.
- **Stage 3 parity.** The refitted baseline reproduces the accepted EBM closely but not exactly, so cascade
  comparisons need refitted baselines.
- **Selector stability.** The interaction selector's near ties can move with the platform or the interpret version.
- **Dependent constants.** Several reference inputs and constants (the frozen cuts, `frozen_shares()`, Stage 1's
  `n_bins = 20` and `k = 100`) were chosen with all training labels or have no recorded rows.

## 9. What Step 6A does not do

- It adopts no contract, moves no code and regenerates nothing.
- It runs no experiment and writes no research record.
- It changes no registered verdict or accepted record.
- It reads no test row.
- It makes no cascade-level claim from a feature-, bin- or stage-level result.

## 10. Evidence index

Each row names an accepted record, a key path in it and the value stated above. How paths are read:

- `/` steps into a key or a list index.
- `[k=v;…]` selects the one list element whose fields match.
- `[?k=v;…]` selects every match.
- `*` maps over a list.
- `@all`, `@sum`, `@max`, `@len` and `@unique` aggregate.
- A number holds at the precision written, and a value ending in `…` is a prefix.
- `\|` is a literal `|`.

<!-- BEGIN EVIDENCE INDEX -->
| Id | Claim | Record | Key | Value |
|---|---|---|---|---|
| S1.1 | own-label share, euribor bin rate, final fit | `c3_20261122_201830/encoding_audit` | `findings/by_path/[stage=stage1;feature=subscribe_rate_by_euribor_bin;path=final]/a_self_share_mean` | `0.0006` |
| S1.2 | held-out encodings carrying their own label: none, 22 groups | `c3_20261122_201830/encoding_audit` | `attribution/held_out_own_labels/*/passed/@all` | `true` |
| S1.2b | the 22 groups | `c3_20261122_201830/encoding_audit` | `attribution/held_out_own_labels/@len` | `22` |
| S1.3 | later-day share, euribor bin rate, out of fold | `c3_20261122_201830/encoding_audit` | `findings/by_path/[stage=stage1;feature=subscribe_rate_by_euribor_bin;path=outer]/c_later_share_mean` | `0.41` |
| S1.4 | later-day share, day × month rate, out of fold | `c3_20261122_201830/encoding_audit` | `findings/by_path/[stage=stage1;feature=subscribe_rate_by_dow_month;path=outer]/c_later_share_mean` | `0.37` |
| S1.5 | later-day share, NSD, Stage 2 outer | `c3_20261122_201830/encoding_audit` | `findings/by_path/[stage=stage2;feature=neighborhood_subscription_density;path=outer]/c_later_share_mean` | `0.47` |
| S1.6 | later-day share, ECI, Stage 2 outer | `c3_20261122_201830/encoding_audit` | `findings/by_path/[stage=stage2;feature=economic_curvature_intensity;path=outer]/c_later_share_mean` | `0.52` |
| S1.7 | MLP early-stopping subset | `c3_20261122_201830/encoding_audit` | `isolation/mlp_early_stopping/val_fraction` | `0.15` |
| S1.8 | median stopping epoch, accepted early stopping | `c3_20261122_232809/temporal_validity` | `predictive/mlp_early_stopping/[split=random;encoding=accepted;early_stopping=accepted]/best_epoch_median` | `17` |
| S1.9 | median stopping epoch, isolated early stopping | `c3_20261122_232809/temporal_validity` | `predictive/mlp_early_stopping/[split=random;encoding=accepted;early_stopping=isolated]/best_epoch_median` | `32.5` |
| S1.10 | MLP isolated vs accepted early stopping, random split, pooled | `c3_20261122_232809/temporal_validity` | `predictive/comparisons/[arm=mlp;a=mlp\|random\|accepted\|es_accepted;b=mlp\|random\|accepted\|es_isolated;scope=pooled]/diff` | `-0.0017` |
| S1.10b | its upper bound | `c3_20261122_232809/temporal_validity` | `predictive/comparisons/[arm=mlp;a=mlp\|random\|accepted\|es_accepted;b=mlp\|random\|accepted\|es_isolated;scope=pooled]/ci_high` | `0.0004` |
| S1.11 | lag 0 availability | `c3_20261122_232809/temporal_validity` | `contracts/decisions/[candidate=past_daily_lag0]/available_share` | `0.972` |
| S1.12 | lag 0 euribor median fallback rows | `c3_20261122_232809/temporal_validity` | `availability/tables/[rule=past_daily_lag0;scope=training rows, full history]/euribor_fallback_rows` | `12636` |
| S1.13 | past-only vs accepted, random split, pooled (LR) | `c3_20261122_232809/temporal_validity` | `contracts/decisions/[candidate=past_daily_lag0]/lr_random_roc_auc_diff` | `-0.0065` |
| S1.14 | B − A within windows | `c3_20261122_232809/temporal_validity` | `predictive/factorial/lr/effects/[effect=past-only vs accepted protocol, random split (B - A)]/mean_window_diff` | `-0.0464` |
| S1.15 | D − C pooled | `c3_20261122_232809/temporal_validity` | `predictive/factorial/lr/effects/[effect=past-only vs accepted protocol, chronological (D - C)]/pooled_diff` | `0.0141` |
| S1.16 | D − C within windows | `c3_20261122_232809/temporal_validity` | `predictive/factorial/lr/effects/[effect=past-only vs accepted protocol, chronological (D - C)]/mean_window_diff` | `0.0213` |
| S1.17 | C − A within windows | `c3_20261122_232809/temporal_validity` | `predictive/factorial/lr/effects/[effect=chronological vs random split, accepted protocol (C - A)]/mean_window_diff` | `-0.0681` |
| S1.18 | D − B within windows | `c3_20261122_232809/temporal_validity` | `predictive/factorial/lr/effects/[effect=chronological vs random split, past-only protocol (D - B)]/mean_window_diff` | `-0.0005` |
| S1.18b | its upper bound | `c3_20261122_232809/temporal_validity` | `predictive/factorial/lr/effects/[effect=chronological vs random split, past-only protocol (D - B)]/mean_window_ci_high` | `0.0084` |
| S1.19 | lag-consistent lag 20, within windows | `c3_20261122_232809/temporal_validity` | `predictive/comparisons/[arm=lr;a=lr\|chrono\|lag_consistent\|past_daily_lag0;b=lr\|chrono\|lag_consistent\|past_daily_lag20;scope=mean over windows]/diff` | `-0.0090` |
| S1.20 | lag-consistent monthly, within windows | `c3_20261122_232809/temporal_validity` | `predictive/comparisons/[arm=lr;a=lr\|chrono\|lag_consistent\|past_daily_lag0;b=lr\|chrono\|lag_consistent\|past_monthly;scope=mean over windows]/diff` | `-0.0472` |
| S1.21 | frozen GLASS routing: NPV holds up to this prevalence | `c3_20261122_232809/temporal_validity` | `stage2_prior_shift/bands/[arm=glass]/npv_holds_up_to_prevalence` | `0.120` |
| S1.22 | frozen GLASS routing: precision holds from this prevalence | `c3_20261122_232809/temporal_validity` | `stage2_prior_shift/bands/[arm=glass]/precision_holds_from_prevalence` | `0.109` |
| S1.23 | the accepted Stage 1 contract is unchanged | `c3_20261122_232809/temporal_validity` | `contracts/candidate/accepted_contract/fingerprint` | `d0d8b07f2ca5e81d` |
| S1.24 | candidate contract version | `c3_20261122_232809/temporal_validity` | `contracts/candidate/version` | `3.0.0-candidate` |
| S1.25 | candidate names fingerprint | `c3_20261122_232809/temporal_validity` | `contracts/candidate/fingerprint` | `1d48dea8b294ff74` |
| S1.26 | candidate definition fingerprint | `c3_20261122_232809/temporal_validity` | `contracts/candidate/definition_fingerprint` | `fe5d54e195b7a604` |
| S1.27 | lag 0 destination | `c3_20261122_232809/temporal_validity` | `contracts/decisions/[candidate=past_daily_lag0]/destination` | `PR 44 Stage 1 candidate (versioned contract)` |
| S1.28 | lag 20 destination | `c3_20261122_232809/temporal_validity` | `contracts/decisions/[candidate=past_daily_lag20]/destination` | `PR 49 outcome-lag sensitivity (versioned contract only if adopted)` |
| S1.29 | MLP isolated early stopping destination | `c3_20261122_232809/temporal_validity` | `contracts/decisions/[candidate=MLP isolated early stopping]/destination` | `PR 44 training-procedure candidate (feature contract unchanged)` |
| S1.30 | lag-consistent lag 20, pooled | `c3_20261122_232809/temporal_validity` | `predictive/comparisons/[arm=lr;a=lr\|chrono\|lag_consistent\|past_daily_lag0;b=lr\|chrono\|lag_consistent\|past_daily_lag20;scope=pooled]/diff` | `-0.1512` |
| S1.31 | lag-consistent monthly, pooled | `c3_20261122_232809/temporal_validity` | `predictive/comparisons/[arm=lr;a=lr\|chrono\|lag_consistent\|past_daily_lag0;b=lr\|chrono\|lag_consistent\|past_monthly;scope=pooled]/diff` | `0.0083` |
| S1.32 | lag-consistent lag 20 against the lag-0 model fed lag-20 encodings, pooled | `c3_20261122_232809/temporal_validity` | `predictive/comparisons/[arm=lr;a=lr\|chrono\|past_updating\|past_daily_lag20;b=lr\|chrono\|lag_consistent\|past_daily_lag20;scope=pooled]/diff` | `-0.1011` |
| S2.1 | `eci_warm` in sample | `c4_20261123_081749/binning_drift` | `support/table/[bin=eci_warm;rows=pooled;scope=in_sample]/lift` | `1.038` |
| S2.2 | `eci_warm` out of fold | `c4_20261123_081749/binning_drift` | `support/table/[bin=eci_warm;rows=pooled;scope=oof]/lift` | `1.031` |
| S2.3 | `eci_hot` in sample | `c4_20261123_081749/binning_drift` | `support/table/[bin=eci_hot;rows=pooled;scope=in_sample]/lift` | `2.457` |
| S2.4 | `eci_hot` out of fold | `c4_20261123_081749/binning_drift` | `support/table/[bin=eci_hot;rows=pooled;scope=oof]/lift` | `2.501` |
| S2.5 | `nsd_elevated` in sample | `c4_20261123_081749/binning_drift` | `support/table/[bin=nsd_elevated;rows=pooled;scope=in_sample]/lift` | `3.126` |
| S2.6 | `nsd_elevated` out of fold | `c4_20261123_081749/binning_drift` | `support/table/[bin=nsd_elevated;rows=pooled;scope=oof]/lift` | `2.486` |
| S2.7 | `nsd_elevated` fold 0 | `c4_20261123_081749/binning_drift` | `support/table/[bin=nsd_elevated;rows=fold_0;scope=oof]/lift` | `0.617` |
| S2.8 | every Step 4A claim holds | `c4_20261123_081749/binning_drift` | `claims/*/holds/@unique` | `[true]` |
| S2.9 | Step 4B registration | `c4_20261123_151515/binning_candidates` | `registration/sha256` | `066195f04c96…` |
| S2.10 | frozen `nsd_elevated` migration | `c4_20261123_151515/binning_candidates` | `comparison/focal/[family=NSD;procedure=frozen]/migration_rows` | `653` |
| S2.11 | `gap_placed` lift | `c4_20261123_151515/binning_candidates` | `comparison/focal/[family=NSD;procedure=gap_placed]/lift` | `3.126` |
| S2.12 | `gap_placed` migration | `c4_20261123_151515/binning_candidates` | `comparison/focal/[family=NSD;procedure=gap_placed]/migration_rows` | `0` |
| S2.12b | `gap_placed`'s held-out bin is the frozen in-sample bin | `c4_20261123_151515/binning_candidates` | `comparison/focal/[family=NSD;procedure=gap_placed]/same_rows_as_frozen_in_sample` | `true` |
| S2.13 | `gap_placed` verdict | `c4_20261123_151515/binning_candidates` | `decision/verdicts/[family=NSD;procedure=gap_placed]/verdict` | `advance to PR 44` |
| S2.14 | NSD `share_matched` reason | `c4_20261123_151515/binning_candidates` | `decision/verdicts/[family=NSD;procedure=share_matched]/reason` | `meets A0-A4 but is outranked by gap_placed` |
| S2.15 | ECI `share_matched` reason | `c4_20261123_151515/binning_candidates` | `decision/verdicts/[family=ECI;procedure=share_matched]/reason` | `fails A3a, A4` |
| S2.16 | ECI `gap_placed` reason | `c4_20261123_151515/binning_candidates` | `decision/verdicts/[family=ECI;procedure=gap_placed]/reason` | `fails A1` |
| S2.17 | ECI `tail_selected` reason | `c4_20261123_151515/binning_candidates` | `decision/verdicts/[family=ECI;procedure=tail_selected]/reason` | `fails A4` |
| S2.18 | `tail_selected` lift | `c4_20261123_151515/binning_candidates` | `comparison/focal/[family=ECI;procedure=tail_selected]/lift` | `3.481` |
| S2.19 | `eci_warm` filter verdict | `c4_20261123_151515/binning_candidates` | `decision/eci_warm_filter/verdict` | `advance to PR 44 as a filter` |
| S2.20 | slots with a required effect | `c4_20261123_151515/binning_candidates` | `decision/eci_warm_filter/slots_required_effect` | `3` |
| S2.21 | slot 3 precision ratio | `c4_20261123_151515/binning_candidates` | `filter/by_slot/[slot=3]/required_ratio` | `1.194` |
| S2.22 | fold-local `gap_placed` lift | `c4_20261123_151515/binning_candidates` | `sensitivity/summary/[procedure=gap_placed_fold_local]/lift` | `2.630` |
| S2.23 | fold-local fold 0 | `c4_20261123_151515/binning_candidates` | `sensitivity/summary/[procedure=gap_placed_fold_local]/fold0_lift` | `0.644` |
| S2.24 | fold-local migration | `c4_20261123_151515/binning_candidates` | `sensitivity/summary/[procedure=gap_placed_fold_local]/migration_rows` | `555` |
| S2.25 | the sensitivity is a material change | `c4_20261123_151515/binning_candidates` | `sensitivity/persists/material_change` | `true` |
| S2.26 | a revision is pending PR 44 | `c4_20261123_151515/binning_candidates` | `decision/revision/warranted` | `pending PR 44` |
| S3.1 | baseline ROC-AUC | `c5_20261124_002645/stage3_simplification` | `metrics/pooled/[configuration=baseline]/roc_auc` | `0.7933` |
| S3.2 | `jed_alone` ΔROC-AUC | `c5_20261124_002645/stage3_simplification` | `metrics/pooled/[configuration=jed_alone]/d_roc_auc` | `0.0001` |
| S3.3 | `jed_alone` ΔROC-AUC lower bound | `c5_20261124_002645/stage3_simplification` | `metrics/pooled/[configuration=jed_alone]/d_roc_auc_low` | `-0.0004` |
| S3.4 | `jed_alone` ΔPR-AUC | `c5_20261124_002645/stage3_simplification` | `metrics/pooled/[configuration=jed_alone]/d_pr_auc` | `0.0007` |
| S3.5 | `jed_alone` Δnested F2 | `c5_20261124_002645/stage3_simplification` | `metrics/pooled/[configuration=jed_alone]/d_nested_f2` | `0.0008` |
| S3.6 | `jed_alone` calibrated ECE | `c5_20261124_002645/stage3_simplification` | `metrics/pooled/[configuration=jed_alone]/ece_calibrated` | `0.0029` |
| S3.7 | `drop_first` ΔROC-AUC | `c5_20261124_002645/stage3_simplification` | `metrics/pooled/[configuration=drop_first]/d_roc_auc` | `-0.0003` |
| S3.8 | `drop_second` ΔROC-AUC upper bound | `c5_20261124_002645/stage3_simplification` | `metrics/pooled/[configuration=drop_second]/d_roc_auc_high` | `-0.0003` |
| S3.9 | decay slot fold sd, baseline | `c5_20261124_002645/stage3_simplification` | `shapes/summary/[configuration=baseline;feature=decay_x_density]/fold_sd_dense` | `0.512` |
| S3.10 | decay slot fold sd, `jed_alone` | `c5_20261124_002645/stage3_simplification` | `shapes/summary/[configuration=jed_alone;feature=joint_economic_decay]/fold_sd_dense` | `0.085` |
| S3.11 | regime-step fold sd, baseline | `c5_20261124_002645/stage3_simplification` | `shapes/centring_check/step_spread/[configuration=baseline;statistic=sd across folds]/period-level terms` | `0.631` |
| S3.12 | regime-step fold sd, `jed_alone` | `c5_20261124_002645/stage3_simplification` | `shapes/centring_check/step_spread/[configuration=jed_alone;statistic=sd across folds]/period-level terms` | `0.065` |
| S3.10b | kept slope's fold sd, `drop_first` | `c5_20261124_002645/stage3_simplification` | `shapes/summary/[configuration=drop_first;feature=emp_var_rate_sigmoid_slope]/fold_sd_dense` | `0.050` |
| S3.10c | kept slope's fold sd, `drop_second` | `c5_20261124_002645/stage3_simplification` | `shapes/summary/[configuration=drop_second;feature=euribor3m_sigmoid_slope]/fold_sd_dense` | `0.109` |
| S3.10d | decay slot fold sd, `past_density` | `c5_20261124_002645/stage3_simplification` | `shapes/summary/[configuration=past_density;feature=decay_x_density_past]/fold_sd_dense` | `0.171` |
| S3.12b | regime-step fold sd, `drop_first` | `c5_20261124_002645/stage3_simplification` | `shapes/centring_check/step_spread/[configuration=drop_first;statistic=sd across folds]/period-level terms` | `0.746` |
| S3.12c | regime-step fold sd, `drop_second` | `c5_20261124_002645/stage3_simplification` | `shapes/centring_check/step_spread/[configuration=drop_second;statistic=sd across folds]/period-level terms` | `0.641` |
| S3.12d | regime-step fold sd, `past_density` | `c5_20261124_002645/stage3_simplification` | `shapes/centring_check/step_spread/[configuration=past_density;statistic=sd across folds]/period-level terms` | `0.071` |
| S3.13 | `jed_alone`: recommended for Step 5B | `c5_20261124_002645/stage3_simplification` | `nominations/[configuration=jed_alone]/step5b` | `recommended` |
| S3.14 | `jed_alone`: nominated, net 486 | `c5_20261124_002645/stage3_simplification` | `nominations/[configuration=jed_alone]/abstention_net` | `486` |
| S3.15 | `drop_first`: not nominated | `c5_20261124_002645/stage3_simplification` | `nominations/[configuration=drop_first]/cascade` | `not nominated` |
| S3.15b | `drop_second`: not recommended, net 633 | `c5_20261124_002645/stage3_simplification` | `nominations/[configuration=drop_second]/abstention_net` | `633` |
| S3.15c | `past_density`: not recommended, net 689 | `c5_20261124_002645/stage3_simplification` | `nominations/[configuration=past_density]/abstention_net` | `689` |
| S3.15d | `past_density` is not recommended for Step 5B | `c5_20261124_002645/stage3_simplification` | `nominations/[configuration=past_density]/step5b` | `not recommended` |
| S3.16 | the screen's threshold | `c5_20261124_002645/stage3_simplification` | `nominations/[configuration=drop_first]/abstention_net_needed` | `456.86` |
| S3.17 | TIER 1 pair, largest supported effect (Windows) | `c5_20261124_002645/stage3_simplification` | `shapes/pair_table/[configuration=baseline;pair=campaign & emp_var_rate_sigmoid_slope]/max_abs` | `0.62` |
| S3.18 | euribor × curvature pair, largest supported effect | `c5_20261124_002645/stage3_simplification` | `shapes/pair_table/[configuration=baseline;pair=euribor3m_sigmoid_slope & economic_curvature_intensity]/max_abs` | `0.051` |
| S3.19 | `prior_x_stress` × decay pair, largest supported effect | `c5_20261124_002645/stage3_simplification` | `shapes/pair_table/[configuration=baseline;pair=prior_x_stress & decay_x_density]/max_abs` | `0.072` |
| S3.20 | `past_density` held-out fallback rows | `c5_20261124_002645/stage3_simplification` | `past_only_density/fallback_trace/totals/held_out_fallback` | `917` |
| S3.21 | accepted EBM, net on abstentions | `c5_20261124_002645/stage3_simplification` | `complementarity/against_baseline/[ebm=accepted;population=Stage 2 abstentions]/net` | `447` |
| S3.22 | refitted baseline, net on abstentions | `c5_20261124_002645/stage3_simplification` | `complementarity/against_baseline/[ebm=accepted;population=Stage 2 abstentions]/reference_net` | `530` |
| S3.23 | Step 5B acceptance | `c5_20261124_122029/interaction_selection` | `acceptance/passed` | `true` |
| S3.24 | the accepted pairs recovered on all rows | `c5_20261124_122029/interaction_selection` | `recovery/summary/accepted_recovered_all_rows` | `["campaign x emp_var_rate_sigmoid_slope"]` |
| S3.25 | TIER 1 pair, outer fits selecting | `c5_20261124_122029/interaction_selection` | `recovery/summary/accepted_outer_counts/campaign x emp_var_rate_sigmoid_slope` | `5` |
| S3.26 | euribor × curvature pair, all-rows rank | `c5_20261124_122029/interaction_selection` | `recovery/summary/accepted_all_rows_positions/euribor3m_sigmoid_slope x economic_curvature_intensity` | `30` |
| S3.27 | `prior_x_stress` × decay pair, all-rows rank | `c5_20261124_122029/interaction_selection` | `recovery/summary/accepted_all_rows_positions/prior_x_stress x decay_x_density` | `39` |
| S3.28 | TIER 2 pairs, outer fits selecting | `c5_20261124_122029/interaction_selection` | `recovery/summary/accepted_outer_counts/prior_x_stress x decay_x_density` | `0` |
| S3.29 | `jed_alone` reselected vs 5A, ΔROC-AUC | `c5_20261124_122029/interaction_selection` | `against_5a/table/[configuration=jed_alone_selected]/d_roc_auc` | `-0.0006` |
| S3.29b | `jed_alone` reselected vs 5A, Δnested F2 | `c5_20261124_122029/interaction_selection` | `against_5a/table/[configuration=jed_alone_selected]/d_nested_f2` | `-0.0006` |
| S3.30 | `drop_first` reselected vs 5A, ΔROC-AUC | `c5_20261124_122029/interaction_selection` | `against_5a/table/[configuration=drop_first_selected]/d_roc_auc` | `0.0004` |
| S3.31 | no material change, either candidate | `c5_20261124_122029/interaction_selection` | `against_5a/table/*/material_change/@any` | `false` |
| S3.32 | `drop_first` reselected vs 5A, Δnested F2 | `c5_20261124_122029/interaction_selection` | `against_5a/table/[configuration=drop_first_selected]/d_nested_f2` | `0.0005` |
| S3.32b | `drop_first` reselected: folds with a better PR-AUC than its 5A fit | `c5_20261124_122029/interaction_selection` | `against_5a/fold_differences/drop_first_selected/[metric=pr_auc]/folds_better` | `4` |
| S3.33 | `jed_alone` reselected, net on abstentions | `c5_20261124_122029/interaction_selection` | `complementarity/screen/[configuration=jed_alone_selected]/abstention_net` | `389` |
| S3.34 | `jed_alone` reselected misses the screen | `c5_20261124_122029/interaction_selection` | `complementarity/screen/[configuration=jed_alone_selected]/screen` | `misses` |
| S3.35 | `drop_first` reselected, net on abstentions | `c5_20261124_122029/interaction_selection` | `complementarity/screen/[configuration=drop_first_selected]/abstention_net` | `397` |
| S3.36 | `jed_alone` with its 5A pairs meets the screen | `c5_20261124_122029/interaction_selection` | `complementarity/screen/[configuration=jed_alone]/screen` | `meets` |
| S3.37 | reselection changes `jed_alone`'s verdict | `c5_20261124_122029/interaction_selection` | `complementarity/change/[candidate=jed_alone]/conclusion_changes` | `true` |
| S3.38 | items flagged for review | `c5_20261124_122029/interaction_selection` | `review/flagged` | `9` |
| S4.1 | the Stage 3 DAG's modules | `c5_20261124_122029/feature_inventory` | `stage3_dag/modules/@len` | `11` |
| S4.2 | in sample: rows that differ between the copy and the research modules (all `dow_month_encoded`) | `c2_20261122_124809/feature_inventory` | `duplicates/parity/[?scope=in-sample (all training rows): stage2 copy vs research]/*/rows_differ/@sum` | `4832` |
| S4.2b | in sample: largest difference | `c2_20261122_124809/feature_inventory` | `duplicates/parity/[?scope=in-sample (all training rows): stage2 copy vs research]/*/max_abs_diff/@max` | `5.6e-17` |
| S4.3 | held out: `euribor3m_local_rate` rows that differ | `c2_20261122_124809/feature_inventory` | `duplicates/parity/[scope=held-out (5 shared folds): stage2 copy vs research;column=euribor3m_local_rate]/rows_differ` | `8` |
| S4.3b | held out: its largest difference | `c2_20261122_124809/feature_inventory` | `duplicates/parity/[scope=held-out (5 shared folds): stage2 copy vs research;column=euribor3m_local_rate]/max_abs_diff` | `0.243` |
| S4.4 | held out: ECI's largest difference | `c2_20261122_124809/feature_inventory` | `duplicates/parity/[scope=held-out (5 shared folds): stage2 copy vs research;column=economic_curvature_intensity]/max_abs_diff` | `0.0013` |
| S4.5 | the Stage 3 DAG digest | `c5_20261124_122029/feature_inventory` | `stage3_dag/sha256_lf_combined` | `3a8d195700ce3462…` |
<!-- END EVIDENCE INDEX -->
