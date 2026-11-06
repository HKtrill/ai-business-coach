# F31 Methodological Disposition Record — RF Router Confidence

| | |
|---|---|
| Author | Phillip Harris |
| Record | F31 disposition (conditional RF Router confidence-blending review) |
| PR / step | PR 37, Commit 3, Step 3C |
| Branch | `pr-37-router-baseline-contract` |
| Approved | 2026-11-04: Disposition A; F31 resolved |
| Related | [`pr37_f17_decision_record.md`](pr37_f17_decision_record.md) (F17); [`pr37_f17_investigation.md`](pr37_f17_investigation.md) (Step 3A); handoff in [`pr37_commit3_handoff.md`](pr37_commit3_handoff.md) |
| Line numbers | The final Commit 3 tree, including the Step 3D docstring edits. Files Commit 3 does not touch match `e5c437c` |

PR 37, Commit 3, Step 3C. Approved on 2026-11-04: Disposition A; F31 resolved. The review changed no code. Its four wording items (W1–W4) were applied in Step 3D as docstring edits only, and its line numbers were refreshed to the final tree.

## 1. Disposition

**F31 disposition: A, retained. No change to how RF confidence is constructed.** The approved F17 policy, `nested_t1` and the Commit 4 contract change the band values and slightly sharpen what the Pass 2 band describes. None of them invalidates the construction, the probability transform or the Stage 4 interface. PR 38 owes eight validation and provenance requirements (section 8); none of them is a correction.

| Question | Answer |
| --- | --- |
| Does F17 change RF construction? | No. The RF Router already fits Pass 2 on its OOF Pass 1 remainder |
| Does `nested_t1` change the Pass 1 band? | No. The Pass 1 OOF scores and the deployed `t1` are identical under both rules |
| Does `nested_t1` change the Pass 2 band? | Its value and its estimation population, but not its construction. It is still an inner-OOF empirical precision, now on a remainder cut without each row's own label |
| Is there a material training-versus-runtime mismatch? | Not by construction. The two populations differ only in a per-fold strip between `t1_j` and `t1`; its size is unmeasured, and PR 38 measures it (R2) |
| Does the Commit 4 contract change construction? | No. It changes the operating points, and through them the band values |
| Is Stage 4 compatible? | Yes. The transform, base rates, route encoding and abstain semantics are unchanged; the read-back identity holds by construction |
| Corrections required? | None. Four documentation wording items (W1–W4), approved for the Step 3D patch (section 8) |

## 2. Historical RF confidence and source evidence

RF "confidence" is one empirical number per route, measured on training OOF scores and attached unchanged to every row that takes that route. Nothing in the chain is a calibrated probability.

### The quantities, kept distinct

| Quantity | What it is | Where | Reaches Stage 4? |
| --- | --- | --- | --- |
| Raw scores `p1`, `p2` | Forest P(y = 1 \| x) from class-balanced RFs. `p2` exists only for rows Pass 1 leaves. Used only for cutting | `two_pass.py:299–319`; `forest.py:68`, `:138–139` | No (`last_scores_` stays in memory) |
| Thresholds `t1`, `t2` | Cut points solved on inner-OOF scores: the largest feasible Pass 1 band, the smallest feasible Pass 2 cut | `threshhold/solver.py:46–142`, `:147–262` | No (recorded in artifacts) |
| Band NPV / precision | `pass1_confidence` = P(y = 0 \| `p1_oof` < `t1`); `pass2_confidence` = P(y = 1 \| `p2_oof` > `t2`, remainder). Empirical OOF rates | `bands.py:86–124` | As `confidence`, and inside the probability |
| Discrete routing state | `pass1` / `pass2` / `uncertain`, from `apply_cuts` with Pass 1 precedence | `router/base.py:58–100` | Yes (`train_decisions`, `test_decisions`) |
| Probability-like output | Pass 1: b(1 − conf); Pass 2: b + (1 − b)·conf; abstain: b. A heuristic blend, explicitly not calibrated | `router/base.py:298–330` (note at `:323–326`), `:348–368` | Yes (`train_proba`, `test_proba`) |
| Directional confidence | Arbiter-side: r − b (Pass 2), b − r (Pass 1), with r read back from the probability | `meta_ebm/router_evidence.py:77–103` | Computed inside Stage 4 (GLASS arbiter, F29 crossover) |
| Calibrated probability | None exists for the RF Router | — | — |

### Construction, step by step

1. **Fit population and b.** The router is fitted on one outer-training partition P_k; b is P_k's positive rate (`two_pass.py:106`).
2. **Pass 1 band.** Inner-OOF `p1` over all of P_k (`:122`). `t1` is solved on those scores and labels (`:128`). The Pass 1 band is P(y = 0) among rows with `p1_oof < t1` (`:178` → `bands.py:117–124`).
3. **Pass 2 band.** Inner-OOF `p2`, with folds built inside the remainder (`two_pass.py:157`; `oof/scorer.py:57–147`). `t2` is solved on those scores (`:164`). The Pass 2 band is P(y = 1) among remainder rows with `p2_oof > t2`.
4. **Assignment.** At prediction, the refitted forests and the frozen `t1`, `t2` give each row a route; `bands_.assign` gives every Pass 1 row the Pass 1 NPV, every Pass 2 row the Pass 2 precision, and every abstained row 0.0 (`two_pass.py:319`; `bands.py:126–147`). The two bands are never combined; each route carries its own number.
5. **Transform.** `_proba_from_triple` maps (route, confidence, b) to the probability (`router/base.py:348–368`).
6. **Train-side outputs.** Each outer fold's held-out rows are routed by that fold's router (`rf_router/crossfit.py:123`). So a training row carries its own fold router's band value and b = its P_k rate, giving five values per route. Test rows carry the full-train router's bands, with b = the global training rate.
7. **Abstention.** Confidence 0.0 and probability b in the router output. The black-box arbiter replaces abstain probabilities with NaN and adds a one-hot route (`meta_xgb/inputs.py:76–85`).

`BinnedBandConfidence` exists but is unwired (`bands.py:38–45`); every router uses the flat `BandConfidence`.

## 3. Effects of `nested_t1` and the F17 policy

`nested_t1` leaves the Pass 1 band exactly as it was. It moves the Pass 2 estimation population closer to deployment in the property that matters: the cut that decides a row's eligibility was chosen without that row. The cost is a small per-fold boundary strip, and the two bands' estimation populations stop being a partition.

|  | `global_t1` (historical) | `nested_t1` (approved) | Deployment |
| --- | --- | --- | --- |
| Pass 1 band population | `p1_oof < t1` over P_k | The same rows, the same `t1` | Refitted `p1 < t1` on rows outside P_k |
| Pass 2 training and band population | `p1_oof >= t1` | For each inner fold j: `p1_oof >= t1_j` | Refitted `p1 >= t1` |
| Was the eligibility cut chosen without the row's label? | No: `t1` used it | Yes, at router level (L1 caveat, section 6) | Yes: `t1` never saw the row |

1. **The Pass 1 band is unchanged.** `p1_oof` and the deployed `t1` are computed before eligibility and are identical under both rules (pinned by `test_the_deployed_t1_is_the_same_under_both_rules`).
2. **The Pass 2 band keeps its construction.** It is still an inner-OOF empirical precision at a threshold solved on the same OOF scores. What changes is its population: rows eligible under a cut not informed by their own labels. `t1_j` differs from `t1` by the cut's sampling variation (`t1_j` comes from about 80% of P_k), which is the same kind of variation the deployed `t1` has relative to new rows. The global remainder matches the deployed cut's value but not its independence. Neither is identical to deployment, and the nested one is the honest analogue.
3. **A boundary strip appears.** In fold j there are two kinds of strip row. Strip A holds rows with `t1_j <= p1_oof < t1`: they are in the Pass 2 training population, but the deployed rule would route them to Pass 1. Strip B holds rows with `t1 <= p1_oof < t1_j`: the deployed rule makes them eligible, but they are excluded from training. The strip sits at the low-risk edge of the remainder, while the Pass 2 band is the top tail of `p2`, so strip rows plausibly rarely enter the flagged band. That is unmeasured (R2).
4. **The two estimation populations stop being a partition.** Under `global_t1`, the Pass 1 band rows and the Pass 2 remainder split P_k exactly. Under `nested_t1`, strip A rows feed both estimates and strip B rows feed neither. Each band is still a well-defined conditional estimate, so this is a change in accounting, not a defect. It does mean the router's stored training OOF no longer reproduces one consistent three-state partition (section 5, `predict_train_oof`).
5. **Support and prevalence move.** The remainder's size and positives change, and `t2` is re-solved on it. `remainder_min_size` and `min_band_support_pass2` are still enforced by the solver (`solver.py:202`); band supports are recorded (R1).
6. **The F17 population policy itself changes nothing on the RF side.** The router already used its OOF Pass 1 remainder.

## 4. Band-estimation validity

Both bands are valid empirical estimates for the populations their routes cover. Choosing each cut on the very OOF scores its band is measured on creates a risk that both band estimates are optimistic. That risk is historical, `nested_t1` does not change it, and it is documented and measured here rather than corrected.

| Check | Finding |
| --- | --- |
| Pass 1 NPV tied to the right population | Yes: the rows Pass 1 routes on training OOF, analogous to the refitted `p1 < t1` at deployment |
| Pass 2 precision out of fold | Yes, with respect to the forests: folds are built inside the remainder (`oof/scorer.py:57–147`) |
| `nested_t1` keeps the estimation boundaries | Yes. It nests the eligibility cut. `t2` and the band still share one set of OOF scores, as before |
| `t2` and its support constraint fit the new remainder | Yes: re-solved on it, with `min_band_support_pass2` enforced by the solver (`solver.py:202`) |
| Band-support safeguards still meaningful | Yes. The solver enforces each pass's own floor (`solver.py:82`, `:202`). The redundant check in `_make_bands` applies the smaller of the two floors to both bands, although its docstring said "stricter" (`router/base.py:430–445`). That was a wording item, with no behavioural effect (W1, applied in Step 3D) |
| Reliance on small support | Not historically: inner bands held thousands of rows, and 9,427 rows were routed to Pass 2 on the outer OOF. A higher Commit 4 precision floor could shrink the band, so support floors belong in the contract, and supports are recorded (R1) |
| Optimism from selecting and estimating on the same OOF | A risk created by construction. `t2` is the smallest cut whose OOF precision meets the floor, so each band sits at or just above the floor, and its estimate may be optimistic for held-out rows. The training-side record is consistent with this concern: every RF fold router's Pass 2 band met 0.27598 on its inner OOF, while the pooled realised outer-OOF precision was 27.16% (frozen baseline). That discrepancy neither establishes the size of any optimism nor shows it in every fold. PR 38 measures the actual per-fold gaps (R3) |
| Effect of the Commit 4 values | Operating points only. Because the bands sit just above the floors, RF Pass 2 confidence will largely reflect the contract's precision floor rather than the forest's discrimination. That is an interpretation note, not a defect |
| L1 (target-encoded features) | Inner-OOF scores are computed on features fitted on P_k, so band values may carry feature-level optimism. G13 decides. If a feature correction is adopted, the bands are re-estimated under it with no construction change |

## 5. Stage 4 compatibility

The Stage 4 interface is unchanged and stays valid. The approved policy changes values reaching Stage 4, not their encoding or meaning. Stage 4 does not yet record which eligibility rule produced an RF artifact, and that gap is closed by provenance requirement R5.

| Interface element | Finding |
| --- | --- |
| Routing-state encoding | Unchanged: `pass1` / `pass2` / `uncertain`, normalised by `router_stream` (`shared/stage4/adapters.py:296–330`) |
| Fields consumed | `train_proba`, `train_decisions`, `test_proba`, `test_decisions`. The meta-XGB arbiter also uses a one-hot route, with NaN on abstain (`meta_xgb/inputs.py:76–85`) |
| Confidence direction and magnitude | Pass 1 carries an NPV (probability below b); Pass 2 carries a precision (probability above b). The `router_stream` semantic checks require Pass 1's training positive rate below b and Pass 2's above it (`adapters.py:420–431`); they still apply |
| Probability-like transform | Unchanged; the read-back identity r = conf holds by construction |
| Base rate | The full-train router's `training_base_rate` must equal the global training rate (`adapters.py:411–414`). Training-side rows use their fold router's P_k rate, read back with `fold_base_rates` (`meta_ebm/router_evidence.py:58–67`). Unchanged |
| Pass 2 lift against the global prior | `directional_confidence` is r − b with the global (or fold) prior (`router_evidence.py:95–103`). For RF, part of that lift is the prevalence enrichment Pass 1 leaves in the remainder, as before |
| Abstention | Unchanged: confidence 0, probability b; meta-XGB uses NaN plus a route flag |
| Training-side vs test-side consistency | Unchanged: fold-router bands at P_k rates on training rows, full-train bands at the global rate on test rows. The F29 crossover checks this row by row (`meta_ebm/loader.py:125–160`) |
| Provenance | Stage 4 records `pass2_population` for both sides, but not `pass2_eligibility` (R5) |

### GLASS evidence under F17

Under `oof_remainder`, a GLASS Pass 2 rule's stored precision, and so its read-back r, is its in-sample positive rate on the remainder rows it was selected on, rather than on all rows. The interface and the read-back identity are unchanged, and `glass_blend_mismatch` still verifies them row by row. The router-evidence docstring ("in-sample rate on the router's training rows") remains true if "training rows" means the population each rule was selected on; a clarifying wording item was approved (W2) and applied in Step 3D. Pass 1 r is unchanged, given Pass 1 invariance (gate G6 in the F17 record). Pass 2 r values will plausibly rise with the remainder's higher prevalence, which widens the r − b lift. That is a change in evidence values, not in its construction. This review does not extend to redesigning GLASS confidence.

### `predict_train_oof` under `nested_t1`

It cuts the stored OOF scores at the deployed `t1`. So strip A rows come out as Pass 1, and strip B rows abstain (their `p2` is NaN). It no longer reproduces the training remainder. It is used only by a blackbox Cell 21 diagnostic ("decision agreement, outer cross-fit vs predict_train_oof") and by tests. The Stage 4 training inputs are the outer cross-fit outputs (`extra.train_*`, Cell 24). **No Stage 4 input is affected.** Under `nested_t1`, the Cell 21 diagnostic changes meaning and must be relabelled or replaced (R7).

## 6. Confirmed issues, unverified concerns, known limitations

Nothing here invalidates the construction. Every confirmed item is either historical and documented, an accounting or diagnostic consequence of `nested_t1`, or a wording or provenance gap.

### Confirmed from the code

| ID | Item | Impact | Handling |
| --- | --- | --- | --- |
| C1 | Choosing each cut on the OOF scores its band is measured on creates a risk of optimistic band estimates. By construction, the Pass 2 band sits at or just above its floor | Historical, not caused by F17. The size of any optimism is unknown: the historical discrepancy (section 4) is consistent with the risk, but does not establish its magnitude or show it in every fold | Documented; measured per fold (R3) |
| C2 | Under `nested_t1`, the two bands' estimation populations stop partitioning P_k (strip A counts in both, strip B in neither) | Accounting only | Strip sizes measured (R2) |
| C3 | Under `nested_t1`, `predict_train_oof` no longer reproduces the training remainder | Diagnostic only; no Stage 4 input | Cell 21 diagnostic relabelled or replaced (R7) |
| C4 | `_make_bands` docstring said "stricter" floor; the code applies the smaller one, redundantly, after the solver has enforced each pass's own floor | None on behaviour | Wording item W1 (applied in Step 3D) |
| C5 | Stage 4 records `pass2_population` but not `pass2_eligibility` | Two RF artifacts with different eligibility rules look identical downstream | Provenance requirement R5 |

### Unverified (needs measurement)

| ID | Concern | Handling |
| --- | --- | --- |
| U1 | The boundary strip's size, and how much of it enters the flagged Pass 2 band | R2 |
| U2 | The training side carries five band levels per route, one per fold router, while the test side carries one. Meta-XGB could split within a route on fold-specific levels, which encode fold identity rather than row evidence. This is a pre-existing property; F17 and Commit 4 change the values and possibly the spread | R6 |
| U3 | Band support under the Commit 4 floors | R1, and the Commit 4 support floors |
| U4 | The size of the optimism gap per fold | R3 |

### Known limitations (cross-referenced)

| ID | Limitation |
| --- | --- |
| K1 | F17 L1: inner-OOF scores, and so the band values, use target-encoded features fitted on P_k. Gate G13 decides whether a feature correction is needed |
| K2 | F17 L2: GLASS confidence is an in-sample precision; RF's is an inner-OOF precision whose threshold selection carries an optimism risk. In the F29 crossover, the GLASS arbiter reads both through one read-back, so their magnitudes are not guaranteed comparable across arms. Recorded, not corrected here; fit-time enforcement belongs to Commit 4 |
| K3 | No output is calibrated. That is deliberate: both arms use one heuristic transform so their outputs are comparable objects (`router/base.py:323–326`) |
| K4 | The nominal test split was exposed in PRs 33–36. This review used no test-derived information |

## 7. Why A and not B

B needs a demonstrated defect in construction, estimation, interpretation, calibration or downstream compatibility that the future methodology causes. None was found.

- **Construction and transform are unchanged,** and the Stage 4 read-back identity holds by construction under the approved policy.
- **The new Pass 2 estimation population is a closer deployment analogue,** in that its cut is chosen without the row's own label. The strip is a bounded, measurable difference, not an error in the estimator.
- **The selection-optimism risk (C1) is historical, not F17's.** Correcting it for RF alone, for example by re-estimating bands on a held-out slice of P_k, would add a new estimator to one arm, change Stage 4 comparability, and answer a problem not yet measured. Measure it first (R3).
- **U2 is a pre-existing Stage 4 modelling question,** not a confidence-construction defect.
- **Everything else is provenance, diagnostics or wording.**

### Conditions that would reopen F31

These are fixed now, before any PR 38 result.

1. **R2:** strip A rows make up 5% or more of any fold router's flagged Pass 2 band. This was approved in Step 3C as a mandatory review trigger, not an acceptance criterion. Crossing it reopens F31 for investigation; staying below it does not by itself establish that the confidence is valid.
2. **G13:** the adopted feature correction makes the deployed forests use different features from the OOF scores the bands were estimated on.
3. **Stage 4:** it starts consuming raw scores or binned bands, or the probability transform changes (the PR 36 roadmap condition).
4. **Commit 4:** its evaluation rules require band values to be re-estimated after an outer-OOF contract shortfall.

Whatever the triggers show, R2 and R3 are always recorded and reported. If no trigger fires, F31 stays closed through PR 38.

## 8. PR 38 requirements and Step 3D wording items

**PR 38 makes no confidence-construction change.** `BandConfidence`, `_make_bands`, `_proba_from_triple` and the assignment of bands to routes stay as they are, and `BinnedBandConfidence` stays unwired. The requirements below are validation and provenance only. R1–R5, R7 and R8 are mandatory; R6 is a diagnostic.

| ID | Requirement | Passes when |
| --- | --- | --- |
| R1 Band record | Persist `pass1_confidence`, `pass2_confidence` and both supports for every outer fold router (in `oof_fold_summary`) as well as the full-train router (already in `bands`) | Present for every fold; each support at or above its pass's floor |
| R2 Strip diagnostic | For each outer fold router under `nested_t1`: rows, positives and prevalence in strip A and in strip B, plus the strip A rows and positives inside the flagged OOF Pass 2 band (strip B has no `p2`). Computed inside the cross-fit, because fold routers are discarded (`rf_router/crossfit.py`) | Always recorded and reported. Strip A at 5% or more of any fold's flagged band is a mandatory trigger to reopen F31 (section 7). Below 5% is not, by itself, evidence that the confidence is valid |
| R3 Optimism gap | For each fold router: its band NPV and precision against the realised NPV and precision of its routes on its own outer held-out rows, using training labels only; also pooled | Always recorded and reported, per fold and pooled, whatever R2 shows. Descriptive only; never used to tune |
| R4 Stage 4 checks | Regenerated RF artifacts pass `router_stream`'s semantic checks, meta-XGB's NaN-on-abstain and route checks, and the F29 crossover blend check at fold base rates (training side) and the global rate (test side) | All pass |
| R5 Eligibility provenance | RF artifacts carry `eligibility` in the config, `fit_report_.remainder` (source, rule, per-fold `t1_j`) and protocol `pass2_eligibility`. Stage 4 records `pass2_eligibility` beside `pass2_population` and refuses an artifact whose cross-fit and full-train rules disagree (a metadata-only adapter change) | Recorded at Stage 4; a mismatch is refused |
| R6 Level spread (diagnostic) | Report the distinct training-side `stage2_proba` levels per route and their spread, and whether meta-XGB splits within a route on them | Reported. Any response is a Stage 4 decision, outside F31 |
| R7 Cell 21 diagnostic | Under `nested_t1`, relabel "decision agreement, outer cross-fit vs predict_train_oof" (it no longer reproduces the training remainder), or replace it with R2 | Not presented as a training-remainder check |
| R8 L1 coupling | Band values are reported only from the pipeline state that passed G13 | Provenance links the bands to the G13 outcome |

### Step 3D wording items (approved in Step 3C, documentation only)

1. **W1, `router/base.py` `_make_bands` docstring.** "the stricter of the two per-pass minimums" should say the smaller one, and note that this is a redundant check after the solver has enforced each pass's own floor.
2. **W2, `meta_ebm/router_evidence.py` docstring.** A rule's in-sample rate is measured on the rows the rule was selected on: all training rows under `full`, and the Pass 1 remainder for Pass 2 rules under `oof_remainder`.
3. **W3, `router/two_pass.py`, `predict_train_oof` docstring.** Under `nested_t1` it cuts at the deployed `t1`, so it does not reproduce the training remainder. It is a diagnostic only.
4. **W4, `bands.py` module notes.** Under `nested_t1`, the Pass 2 band is measured on the fold-nested remainder.

None of these four files is among the 15 baseline-pinned code files.

**Status: applied in Step 3D, docstrings only.** W4 also records, in the same notes, the selection-optimism risk (C1) and that the Pass 1 band is unchanged under `nested_t1`. W2 also states that the same read-back gives the RF band precision in the F29 crossover. Two Step 3D regression tests pin the behaviour that W3 and W4 describe, and the band accounting in section 3 (C2, C3), in `tests/stage2/test_rf_router_nested_eligibility.py`:

- `test_the_bands_are_measured_on_the_populations_the_f31_record_names`
- `test_predict_train_oof_departs_from_the_remainder_only_on_the_boundary_strips`

## 9. Dependencies on Commit 4 and the F17 record

### Commit 4

- **The precision and NPV floors** set the lower bounds that the bands sit on, and so largely set RF confidence values (section 4).
- **The RF band-support floors** are adapter arguments to the contract; they feed R1.
- **The leakage count-cap choice** moves `t1`, and through it the Pass 1 band.
- **The outer-OOF contract evaluation rule** decides what an outer-OOF shortfall means (reopening condition 4).
- **How GLASS's fit-time enforcement is made honest (K2)** bears on whether the two arms' Stage 4 evidence is comparable in magnitude.

### F17 record

| F17 gate or limitation | Link to F31 |
| --- | --- |
| G3 preflight | Records band supports (R1) |
| G5 persistence | Extended by R1, R2 and R5 |
| G7 reproducibility | Band values must be identical across reruns (bit-identical with `n_jobs=1`) |
| G13 L1 audit | R8, and reopening condition 2 |
| L4 / L5 | Their F31 consequences are the strip (R2) and `predict_train_oof` (R7, W3) |

### The F17 record's F31 dependencies, resolved

| Dependency listed in F17, section 9 | Resolution |
| --- | --- |
| GLASS Pass 2 confidence becomes in-sample precision on the remainder | The interface is unchanged; the values shift. Section 5; W2 |
| RF Pass 2 band measured on the nested remainder while runtime uses `t1` | The construction stays valid; the strip is measured. Sections 3–4; R2 |
| Transform and base rate unchanged | Confirmed. Sections 2 and 5 |
| Stage 4 lift measured against the global prior | Retained as the intended semantics. Section 5 |
| `predict_train_oof` diverges | A diagnostic only. Section 5; R7, W3 |
| Router-evidence read-back unaffected | Confirmed by construction; verified on regenerated artifacts (R4) |
