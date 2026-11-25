# PR 42 Commit 4, Step 4B — Fold-fitted NSD/ECI binning and `eci_warm` as a filter

**Question.** Step 4A diagnosed the PR 39 NSD/ECI lift drift and left three questions. On identical held-out
training rows, does a fold-fitted derivation of the bins improve on the frozen PR 39 bins enough, at usable support
and with fold stability, to earn a PR 44 router comparison? Does the frozen `eci_warm`, used as a filter on partner
rules, help identify subscribers or exclude false positives at usable support?

**In brief.**

| Family / bin | Comparator (frozen, out of fold) | Best candidate | Verdict | Why (registered criteria) |
|---|---|---|---|---|
| NSD, `nsd_elevated` | 2.486×, folds 0.62–3.36×, 1,421 rows, refit migration 653 rows | `gap_placed`: 3.126× [2.902, 3.358], folds 2.82–3.36×, 1,326 rows, migration 0 | **Advance `gap_placed` to PR 44**, qualified (sections 7–8); frozen bins kept | Meets A0–A4 and ranks first. `share_matched` (3.131×, folds 2.81–3.36×, migration 27) also meets them and is outranked. Post-registration: with fold-local reference shares the fold-0 repair and zero migration do not persist |
| ECI, `eci_hot` | 2.501×, folds 2.39–2.71×, 4,074 rows | `tail_selected`: 3.481× [3.314, 3.651], folds 3.40–3.57×, 2,532 rows | **No candidate advances**; frozen bins kept | `tail_selected` fails A4: its gain depends on one contact period (May 2009). `gap_placed` fails A1 (T2 fires for `eci_warm`). `share_matched` fails A3a and A4 |
| `eci_warm` as a filter | partner rules alone | `eci_warm` required | **Advance to PR 44 as a filter (`warm_required`)** | 3 of 5 partner slots meet the filter criterion; `warm_excluded` never does. As a standalone conjunction it keeps 25–48% of a partner's rows; its router-level coverage effect is PR 44's to measure |

- **`nsd_elevated`: cuts placed in the widest gap between macro states remove the fold-0 anomaly.** In every fold
  `gap_placed` puts the second and third cuts in the widest gap among fit-side states near the frozen shares. Its
  held-out `nsd_elevated` rows are exactly the rows the frozen bins assign that bin in sample, so no held-out row
  changes membership between its fold's fit and the all-rows fit. Folds 1–4 are unchanged. Fold 0 goes from 0.62×
  to 3.02×: its cuts no longer admit the seven May 2009 states that fold 0's encoder places above 0.192. The gain is
  stability, not a better bin outside May 2009. With that period removed, the frozen bin is 3.045× and `gap_placed`
  3.018×.
- **The repair comes from the full-training reference shares.** Every registered procedure reads fixed reference
  inputs (`frozen_shares()`, `window_halfwidths()`, `_tail_range()`) formed on all 32,950 training rows, each fold's
  held-out rows and labels included. In a separately labelled post-registration check, `gap_placed` takes its targets
  and windows from each fold's own fit rows instead. Fold 0 then stays at 0.64× (276 rows), 555 rows change
  membership on refit, and the pooled lift is 2.630×. Folds 1–4 are unchanged. The registered verdict stands; the
  PR 44 handoff is qualified (sections 7–8).
- **`eci_hot`: a label-using tail drops the four May 2009 states, and its gain rests on them.** `tail_selected`
  chooses an upper tail on the fit side by the Wilson lower bound of its subscriber rate. Its held-out bin, 2,532
  rows, is a subset of the frozen bin, with folds 3.40–3.57×. Of the 1,542 frozen rows it leaves out, 1,456 are the
  four May 2009 states. With May 2009 removed, the frozen bin (3.395×) is slightly better than `tail_selected`
  (3.361×), so A4 closes it out. Its all-rows fit also chooses a narrower tail than the folds do (1,797 rows), so 735
  held-out rows change membership on refit.
- **`eci_warm` as a filter: higher precision, much less support.** Within the partner rules chosen on the fit side
  (built around `nsd_hot`, `behav_hot`, `jed_hot`, `cci_hot1` and `cci_hot2`), requiring `eci_warm` raises held-out
  precision by 1.13–1.27× in slots 3–5 and by about 1.07× in slots 1–2. As a standalone conjunction it keeps 25–48%
  of each partner's rows; in slot 3 it keeps 286 of 843 subscribers. Excluding `eci_warm` always lowers precision.
  The population is every held-out training row, not Pass 2's, so the obligations are a screen, not compliance. How
  much router coverage the filter costs is a different quantity, which PR 44 measures (section 5).

**Scope.** Training side only. The frozen binning (`shared.stage2.binning`, identity `9cdd79af…`), `stage2_binning`
and the accepted PR 39 and Step 4A records are read, never changed. No router is fitted. PR 44 owns the router-level
comparison and any adoption. The registered procedures are fold-fitted given fixed full-training reference inputs,
like the frozen comparator, so no arm of the comparison is fully nested (section 1).

**Evidence.**

- *Where it is produced.* `feature_research/feature_research.ipynb`, Cells 40–48 (module
  `feature_research.binning_candidates`), after Step 4A's Cells 31–39 in the same run. The notebook writes
  `research_logs/pr42/<run_id>/binning_candidates.json` beside the run's four earlier records.
- *Which run.* The numbers below come from this revision's sandbox run on the 32,950 training rows (Linux, Python
  3.13). Cell 46 requires every registered verdict and figure in this document to hold in the run that writes the
  record (`BC.STATED`, 70 checks), and Cell 47 every sensitivity figure (`BC.STATED_SENSITIVITY`, 36 checks). The
  Windows run is the accepted one.
- *Reuse, not new fitting.* Every engineer is PR 39's (`stage2_binning.runner.oof_sources`: the outer fold
  engineers, and the inner engineers on each fit side). Bin statistics, Wilson intervals and T1–T4 come from
  `stage2_binning.evidence`, and the macro states and per-state values from Step 4A. The module imports no
  feature-engineering module, so the Commit 2 caller scan and the Step 3A digests are unchanged.
- *Tests.* `tests/feature_research/test_pr42_binning_candidates.py` (synthetic: registration, fit and apply, ties,
  empty windows, infeasible fits, fold isolation with a failing control, comparator parity, partner counts,
  partition reconciliation, the decision rules, the reference-input trace, the fold-local reference and the
  sensitivity beside the registered result, the notebook cells) and `tests/realdata/test_pr42_binning_candidates.py`
  (the real rows: registration inputs, isolation, parity, reconciliation, every stated figure, the sensitivity).

## 1. The registration

`BC.REGISTRATION` 1.0.0, SHA-256 `066195f04c96c5873952c02d4fd80707e32ef51af8a7a6926153ecd49ddf35f5`, was fixed and
pinned (`BC.REGISTERED_SHA256`) before any candidate cut, partner rule, filter variant or held-out statistic of one
had been computed. Cell 40 requires the digest before Cell 41 computes anything. Its design used only the Step 4A
findings, the PR 39 record's in-sample bin supports, and the frozen GLASS profile and shared contract. A later
change would add a revision with its reason, as Step 3B's 1.1.0 did.

**Known before registration (Step 4A), and the hypothesis each motivated.**

| Step 4A established | Motivated |
|---|---|
| `nsd_elevated`: fold 0's encoder moved whole macro states across both fixed cuts, and all 653 rows entering or leaving the bin are fold-0 rows | `share_matched`: if the shift is a change of scale, cuts at the frozen row shares of each fold engineer's own values restore membership |
| Both families: NSD 0.192 / 0.23 and ECI 0.0291 / 0.0758 lie inside the range some state takes across the six encoders | `gap_placed`: a cut in the widest gap between fit-side states near the frozen share leaves the most room for encoder change |
| `eci_hot`: every encoder places four low-rate May 2009 states (1,456 rows, 0.68× in sample) above 0.0948 | `tail_selected`: no label-free re-placement can remove them; a label-using upper-tail choice on the fit side can try, at a cost in support |
| `eci_warm`: the bin is about 1.03×; the quoted 1.49–1.67× were deciles inside it | judged as a filter on partner rules, with its frozen definition |

**What it fixed.**

- *Comparator.* The frozen bins on each held-out row's out-of-fold value, i.e. the PR 39 E-B out-of-fold tables.
- *Rows and folds.* The 32,950 training rows on the shared plan `make_folds(y, 5, 42)` (fingerprint
  `8aea082fcaf371b7`). Each held-out row is scored once, with its fold's engineer and its fold's cuts. Inner plan
  `make_folds(y_fit, 5, 42)` on each fit side.
- *Values.* Label-free placements use fold k's engineer value of each fit-side macro state, weighted by its fit
  rows: the scale the held-out rows are scored on. Label-using choices use inner cross-fitted values, each fit row
  valued by an engineer that never saw its label (a state's value is the mean over its fit rows).
- *States and ties.* States within 1e-12 form one block. A cut lies at the midpoint between neighbouring blocks, so
  it never splits a state on the fit side. A held-out value equal to a cut goes to the lower bin.
- *Procedures.* `share_matched` (NSD, ECI): cut j at the block boundary nearest the frozen cumulative in-sample share
  q_j (`frozen_shares()`). `gap_placed` (NSD, ECI): the widest gap among boundaries within q_j ± h_j, with h_j a
  quarter of the smaller adjacent frozen bin share (`window_halfwidths()`); an empty window falls back to
  `share_matched`. `tail_selected` (ECI only): cuts 1–2 as `share_matched`; the upper tail with the largest fit-side
  Wilson lower bound among tails holding 0.5–2× the frozen `eci_hot` share (`_tail_range()`), mapped onto the scoring
  values by share.
- *Criteria.* A0 feasible in all folds; A1 no PR 39 trigger; A2 focal support ≥ half the frozen bin's in every fold;
  A3 the family's question (NSD: higher smallest fold lift, pooled lift ≥ the frozen bin's Wilson lower bound, refit
  migration ≤ the frozen bin's; ECI: pooled lift ≥ 1.10× the frozen bin's, smallest fold lift ≥ the frozen bin's);
  A4 the A3 comparison survives removing any one fold or any one contact period of at least 100 rows. NSD compares
  against the frozen bin's Wilson lower bound, ECI against its lift. Filter: a slot shows an effect if a variant's
  pooled precision is ≥ 1.10× the partner's, exceeds it in ≥ 4 folds and passes the union screen. `eci_warm`
  advances in a direction if ≥ 3 of 5 slots do.

**The reference inputs and their provenance** (`BC.reference_inputs`, record `reference_inputs`).

| Input | Values | Used by |
|---|---|---|
| `frozen_shares()` | NSD 0.67132 / 0.878877 / 0.91912; ECI 0.22434 / 0.499848 / 0.878938 | `share_matched` (targets), `gap_placed` (targets), `tail_selected` (cuts 1–2) |
| `window_halfwidths()` | NSD 0.051889 / 0.010061 / 0.010061; ECI 0.056085 / 0.068877 / 0.030266 | `gap_placed` (windows) |
| `_tail_range()` | ECI 0.060531–0.242124 | `tail_selected` (candidate tails) |

All three are functions of `BC.FROZEN_SUPPORTS`, the frozen bins' in-sample supports on all 32,950 training rows
(PR 39 E-B). Those supports come from the all-rows engineer, fitted with every training label, binned with the
frozen cuts. The frozen cuts were in turn read from in-sample deciles on all 41,188 rows (Step 4A). Each fold's fit
therefore depends on its own held-out rows and labels through these constants. The limitation is the frozen
comparator's own, and it applies to all three procedures, not only `share_matched`. The comparison with the frozen
bins is like for like in this respect, and no arm of it is fully nested. Section 7 measures what it does to the
advancing procedure.

## 2. The learned cuts

All 35 fold fits and 7 all-rows fits are feasible. Cell 41 lists each fit's cuts, fit-row shares, how each cut was
chosen, and fit-side rows and states per bin.

| Family | Procedure | Cut 1 across folds | Cut 2 | Cut 3 | Frozen cuts |
|---|---|---|---|---|---|
| NSD | `share_matched` | 0.052–0.187 | 0.202–0.216 | 0.228–0.276 | 0.0524 / 0.192 / 0.23 |
| NSD | `gap_placed` | 0.114–0.124 | 0.191–0.211 | 0.245–0.265 | |
| ECI | `share_matched` | 0.027–0.033 | 0.041–0.063 | 0.093–0.098 | 0.0291 / 0.0758 / 0.0948 |
| ECI | `tail_selected` | as `share_matched` | as `share_matched` | 0.100–0.104 | |

The cut values move because the fit-side states are sparse. No fit-side state has an NSD value between about 0.06
and 0.17. The boundary nearest the first target share lies below that stretch in folds 1–3 (0.052–0.056) and above it
in folds 0 and 4 (0.170–0.187), so `share_matched`'s first cut jumps between folds; `gap_placed` puts it inside the
stretch. `tail_selected` chooses fit-side
tails of 1,995–2,076 rows (shares 0.076–0.079, subscriber rate about 0.39) among 105–113 candidate tails. Its all-rows
fit chooses a narrower tail (2,038 rows of the selection values, 1,797 of the scoring values).

Isolation (Cell 42): flipping every held-out label of a fold leaves its inner values, cuts, tail choice, partners and
sensitivity fit identical; the control (flipping every tenth fit-side label) moves the inner values in every fold.
These checks are **conditional on the fixed reference inputs**. A flip leaves the frozen cuts, `frozen_shares()`,
`window_halfwidths()` and `_tail_range()` as they are, so the checks show that no fit-side computation reads a fold's
held-out labels. They do not show that those constants are independent of the fold's rows and labels, and the
constants are not (section 1).

## 3. The comparison on identical held-out rows

The comparator reproduces E-B's out-of-fold binning row by row and table by table, and the frozen cuts give the
production bins through the candidates' apply path (Cell 43).

| Bin | Procedure | Rows | Lift [Wilson 95%] | Fold lifts | Smallest fold support vs frozen | Refit migration | Triggers |
|---|---|---|---|---|---|---|---|
| `nsd_elevated` | frozen | 1,421 | 2.486 [2.284, 2.698] | 0.62, 3.26, 2.82, 3.16, 3.36 | 1.00 | 653 | none |
| | `share_matched` | 1,321 | 3.131 [2.907, 3.364] | 3.02, 3.26, 2.81, 3.20, 3.36 | 0.75 | 27 | none |
| | `gap_placed` | 1,326 | 3.126 [2.902, 3.358] | 3.02, 3.26, 2.82, 3.16, 3.36 | 0.75 | 0 | none |
| `eci_hot` | frozen | 4,074 | 2.501 [2.381, 2.626] | 2.46, 2.49, 2.71, 2.39, 2.43 | 1.00 | 91 | none |
| | `share_matched` | 3,988 | 2.457 [2.336, 2.582] | 2.46, 2.49, 2.52, 2.39, 2.43 | 0.90 | 5 | none |
| | `gap_placed` | 4,614 | 2.807 [2.689, 2.927] | 2.95, 2.91, 2.71, 2.39, 2.97 | 1.00 | 631 | T2 (`eci_warm`) |
| | `tail_selected` | 2,532 | 3.481 [3.314, 3.651] | 3.45, 3.57, 3.50, 3.40, 3.48 | 0.56 | 735 | none |

- **NSD.** Both label-free procedures leave folds 1–4 essentially where they were and repair fold 0 (279 rows at
  3.02×, against 374 at 0.62×). `gap_placed`'s held-out `nsd_elevated` is identical, row for row, to the frozen
  in-sample bin, and its `nsd_hot` keeps 2,665 rows at 4.303×. The cost is 95 rows of `nsd_elevated` (1,326 against
  1,421).
- **ECI.** `share_matched` reproduces the frozen bins within a few states, as Step 4A predicted: every encoder
  agrees on the states above the cut. `gap_placed` enlarges `eci_hot` in folds 0, 1 and 4: its third cut passes
  states the frozen cut keeps in `eci_warm`. That leaves `eci_warm` at 0.827× with its interval wholly below 1, so T2
  fires. `tail_selected` raises the cut instead: its bin lies wholly inside the frozen one.
- **Complementarity.** Under `gap_placed`, every `nsd_elevated` row is also in the frozen `eci_hot`, and the 1,456
  `nsd_warm` × `eci_hot` rows are the four May 2009 states. A router conjunction of `nsd_elevated` with `eci_hot`
  selects the same rows as `nsd_elevated` alone.

## 4. Dependence on one fold, one period or a small bin

Cell 44 removes each outer fold, and each of the 23 contact periods (of 26) with at least 100 rows, and recomputes
every procedure's focal lift on the remaining rows.

- **`nsd_elevated`.** `gap_placed` stays at 2.67–3.31× under every removal, at or above the frozen bin's Wilson
  lower bound on the same rows (A4 holds). It exceeds the frozen bin's lift under every removal but two. With fold
  0 removed the two are equal (3.153×); with May 2009 removed the frozen bin is 3.045× and `gap_placed` 3.018×. The
  gain is the repair of fold 0's May 2009 states.
- **`eci_hot`.** `tail_selected` is 2.98–3.76× under every removal and above the frozen bin under every removal
  but one. With May 2009 removed, the frozen bin (2,618 rows) is 3.395× and `tail_selected` (2,532 rows) 3.361×, so
  A4 fails on that period alone. `share_matched` is at or below the frozen bin under every removal.
- **Bin size.** `tail_selected` keeps 56% of the frozen bin's support in its weakest fold (A2 needs 50%), and the
  NSD candidates keep 75%. No gain depends on a bin below the registered floor.

## 5. `eci_warm` as a filter

**Population.** Every held-out training row of each fold (6,590 per fold). The Pass 2 obligations (shared contract:
union precision ≥ 0.30 and support ≥ 75 on the Pass 2 population at fit time) cannot be applied at their defined
scope. No per-row Pass 1 route is tracked, and rebuilding one needs a GLASS Pass 1 fit (PR 44). They are applied as a
screen to one conjunction, pooled over the five folds (the full training side's row scale). The historical Pass 2
remainder's prevalence was about 0.158, against this population's 0.113. Passing the screen is not compliance.

**Partners.** Chosen on each fit side from 288 conjunctions (25 non-ECI singles, 263 cross-family pairs; `bin = 1`
literals), among the 34–39 that pass GLASS's per-rule precision and recall gates and the scaled support floors.
Ranked by the Wilson lower bound of precision, with fit-row Jaccard overlap at most 0.5. Every partner AND
`eci_warm` has at most three conditions. Slot 3 is `nsd_hot & behav_hot` in four folds and `cci_hot1 & behav_hot` in
one.

| Slot | Partner alone: rows, subscribers, precision | Warm required: rows, precision (×partner), folds above | Required removes (subscribers / non-subscribers) | Warm excluded: precision (×partner) | Effect |
|---|---|---|---|---|---|
| 1 | 908, 514, 0.566 | 426, 0.606 (1.070×), 4 | 256 / 226 | 0.531 (0.938×) | none (below 1.10×) |
| 2 | 578, 341, 0.590 | 277, 0.628 (1.065×), 4 | 167 / 134 | 0.555 (0.940×) | none (below 1.10×) |
| 3 | 1,643, 843, 0.513 | 467, 0.612 (1.194×), 5 | 557 / 619 | 0.474 (0.923×) | required |
| 4 | 936, 458, 0.489 | 308, 0.555 (1.135×), 4 | 287 / 341 | 0.457 (0.934×) | required |
| 5 | 1,292, 566, 0.438 | 318, 0.557 (1.271×), 5 | 389 / 585 | 0.399 (0.912×) | required |

In every fold and slot, the required and excluded rows partition the partner's rows, and their rows, subscribers and
non-subscribers add up to the partner's (25 checks). Every variant passes the union screen pooled. The effect
appears where the partner's precision is lowest.

**Standalone support loss is not router coverage.** As a standalone conjunction, required `eci_warm` keeps 25–48% of
a partner's rows and 31–51% of its subscribers. In slot 3 it removes 557 subscribers and 619 non-subscribers from
that one rule. In a router, rows a filter drops can still be routed by other Pass 2 rules of the selected union, and
Pass 1 changes the population the rules see. A precision gain can also change which rules the union-aware selection
admits. The router-level coverage effect can therefore be smaller, larger or of either sign, and it is not measured
here. PR 44 measures it under the shared obligations and Stage 2's lexicographic-coverage objective.

## 6. Decisions

| Candidate | A0 | A1 | A2 | A3 | A4 | Verdict |
|---|---|---|---|---|---|---|
| NSD `share_matched` | ✓ | ✓ | ✓ | ✓ | ✓ | close out: outranked by `gap_placed` (smallest fold lift 2.808 against 2.821) |
| NSD `gap_placed` | ✓ | ✓ | ✓ | ✓ | ✓ | **advance to PR 44** |
| ECI `share_matched` | ✓ | ✓ | ✓ | ✗ (2.457 < 1.10 × 2.501) | ✗ | close out |
| ECI `gap_placed` | ✓ | ✗ (T2: `eci_warm`) | ✓ | ✓ | ✓ | close out |
| ECI `tail_selected` | ✓ | ✓ | ✓ | ✓ | ✗ (period 2009-05) | close out |

- **NSD.** The frozen bins are kept in PR 42. `gap_placed` advances to PR 44 under the registered criteria. Its case
  is stability: zero refit migration and no fold-0 collapse, at 93% of the frozen bin's support. It is not a higher
  lift outside May 2009, and section 7 shows the stability depends on the full-training reference shares.
- **ECI.** The frozen bins are kept, and no candidate advances. `tail_selected`'s 3.48× is real on these rows, but it
  comes from excluding one period's states. Under the registered A4 that is a dependence, and its refit instability
  (735 rows) points the same way. Whether such a tail holds under a chronological split is PR 49's question, which a
  random-fold comparison cannot answer. Advancing it would need a revised registration, which would be post hoc.
- **`eci_warm` as a filter.** Advance to PR 44 as a `warm_required` filter on partner rules: 3 of 5 slots meet the
  criterion. Its standalone support loss is large (section 5); its router-level coverage effect, and its value on the
  actual Pass 2 population under the shared obligations, are PR 44's to measure.

The asymmetry between NSD and ECI comes from the registered questions. NSD's A3 and A4 ask for stability against the
frozen bin's lower bound, ECI's for a lift gain over the frozen bin. Both successful procedures act on May 2009
macro states.

## 7. Post-registration sensitivity: fold-local reference inputs for NSD `gap_placed`

**Not registered.** This check was added at review, after the registered results had been seen. It is outside
registration 1.0.0 (whose digest is unchanged), outside the ranking and outside every verdict above (`BC.SENSITIVITY`,
Cell 47, record `sensitivity`).

**Rule.** Within each outer fold, the frozen comparator cuts are applied to that fold's own fit-side NSD values (its
engineer's values of its fit rows). The fit-row share below each cut is the fold's target; a quarter of the smaller of
the two adjacent frozen-bin shares on those fit rows is the window half-width. Only the targets and widths change.
The placement (`fit_gap_placed`), the outer folds and scoring values, the comparator and the held-out evaluation rows
are the registered ones. The frozen cuts keep their documented historical provenance: they were read from in-sample
deciles on all 41,188 rows. The all-rows reference fit applies the same rule to all training rows; its targets are the
registered ones, and its cuts equal the registered all-rows fit.

| Fold | Fold-local targets | Fold-local half-widths | Cuts, fold-local | Cuts, registered |
|---|---|---|---|---|
| 0 | 0.669 / 0.814 / 0.879 | 0.036 / 0.016 / 0.016 | 0.124 / 0.192 / 0.211 | 0.124 / 0.211 / 0.265 |
| 1 | 0.583 / 0.880 / 0.920 | 0.074 / 0.010 / 0.010 | 0.052 / 0.202 / 0.260 | 0.121 / 0.202 / 0.260 |
| 2 | 0.587 / 0.879 / 0.919 | 0.073 / 0.010 / 0.010 | 0.052 / 0.196 / 0.252 | 0.117 / 0.196 / 0.252 |
| 3 | 0.672 / 0.878 / 0.919 | 0.052 / 0.010 / 0.010 | 0.114 / 0.191 / 0.245 | 0.114 / 0.191 / 0.245 |
| 4 | 0.583 / 0.878 / 0.918 | 0.074 / 0.010 / 0.010 | 0.051 / 0.195 / 0.247 | 0.116 / 0.195 / 0.247 |
| registered, every fold | 0.671 / 0.879 / 0.919 | 0.052 / 0.010 / 0.010 | | |

| `nsd_elevated`, same held-out rows | Frozen (comparator) | `gap_placed`, registered | `gap_placed`, fold-local (sensitivity) |
|---|---|---|---|
| Rows | 1,421 | 1,326 | 1,323 |
| Lift [Wilson 95%] | 2.486 [2.284, 2.698] | 3.126 [2.902, 3.358] | 2.630 [2.417, 2.854] |
| Fold lifts | 0.62, 3.26, 2.82, 3.16, 3.36 | 3.02, 3.26, 2.82, 3.16, 3.36 | 0.64, 3.26, 2.82, 3.16, 3.36 |
| Worst fold | 0.617 (fold 0) | 2.821 (fold 2) | 0.644 (fold 0) |
| Fold-0 rows | 374 | 279 | 276 |
| Smallest fold support vs frozen | 1.00 | 0.75 | 0.74 |
| Refit migration | 653 (all fold 0) | 0 | 555 (all fold 0) |
| Triggers | none | none | none |

- **The fold-0 repair does not persist.** Under the frozen cuts, fold 0's own fit rows put seven states in
  `nsd_elevated` and 231 in `nsd_hot`; its targets therefore leave 18.6% of fit rows above the second cut, against
  12.1% registered. With those targets `gap_placed` places fold 0's cuts at 0.192 and 0.211, close to the fold's
  frozen-cut placement (six fit-side states in `nsd_elevated`, 231 in `nsd_hot`). Its 276 held-out fold-0 rows
  (0.64×) share no row with the registered fit's 279 (3.02×): the 555 fold-0 rows that differ from the registered fit
  are these two sets.
- **The zero migration does not persist.** 555 held-out rows, all in fold 0, change `nsd_elevated` membership
  between their fold's fit and the all-rows fit, against 0 registered and 653 frozen.
- **Folds 1–4 are unchanged.** Their `nsd_elevated` rows are the registered fit's. In folds 1, 2 and 4 only the first
  cut moves, which changes `nsd_cold` and `nsd_warm`; fold 3's cuts are the registered ones.
- **Read descriptively, the registered criteria would still hold** (worst fold 0.644 against 0.617, migration 555
  against 653, pooled 2.630 above the frozen bin's 2.284 lower bound). The margins are thin, though, and the stability
  case on which `gap_placed` advanced (a repaired fold 0 and no migration) comes from the full-training reference
  shares. By the definition stated with the check (`persists.material_change`: fold 0's membership differs from the
  registered fit's, or some row migrates, or the worst fold falls below the frozen bin's lower bound), the result
  changes materially, so section 8 qualifies the handoff. Whether a reference formed without each fold's held-out rows,
  but not by that fold's own engineer, would repair fold 0 is untested.

## 8. What PR 44 receives

- **`gap_placed` for NSD** (registration `pr42-step4b-registration` 1.0.0, `066195f0…`), **qualified.** It advances
  under the registered criteria, and its registered advantage comes from reference shares and windows formed on all
  training rows (`frozen_shares()`, `window_halfwidths()`). With fold-local ones (section 7) fold 0 is not repaired
  and 555 rows migrate. The gain over the frozen bins shrinks to a pooled 2.630× against 2.486×, fold 0 at 0.64×
  against 0.62×, and migration 555 against 653. PR 44 should:
  - state which reference inputs it uses;
  - fit both variants (the registered constants and the fold-local ones) inside every Stage 2 cross-fit fold;
  - not read the registered 3.126× as a nested estimate.

  If PR 44 accepts only fully fold-local inputs, little gain should be expected. The procedure needs no engineer fit
  beyond the fold's own, so its cost is a sort of the fold's macro states. It fires no trigger, keeps 75% of the
  frozen bin's support in its weakest fold (74% fold-local), and its `nsd_elevated` lies wholly inside the frozen
  `eci_hot` (section 3), which bounds what it can add to a router that already uses `eci_hot`.
- **`eci_warm` required, as a filter** on Pass 2 partner rules. Its standalone support loss is section 5's. PR 44
  measures what it does to router coverage and whether it helps the selected union meet the shared obligations on the
  actual Pass 2 population.

A versioned revision would be warranted only if PR 44 adopts `gap_placed`. It would need:

- **Contract.** `shared.stage2.binning` would carry a fitted NSD cut procedure instead of constants: a new binning
  version and identity, the 29 names and order unchanged, and the reference inputs stated.
- **G13.** `stage2_crossfit.g13`'s audit would cover the cut fit and require it inside every Stage 2 cross-fit fold.
  It would also state the scope of the reference inputs: as registered, `gap_placed`'s targets and windows are
  full-training constants that each fold's held-out rows and labels helped form. Beyond them, `gap_placed` uses no
  label but the engineer's own fit.
- **Regeneration.** The Stage 2 binary frames, GLASS rules, RF Router forests and downstream Stage 3–4 artifacts
  would be regenerated beside the frozen PR 38–39 ones.

PR 42 changes no frozen bin, contract, artifact or accepted record.

## 9. Limits

- Random outer folds only. A candidate's value under a chronological split is PR 49's question.
- Every registered procedure reads full-training reference inputs: `share_matched` its targets (`frozen_shares()`),
  `gap_placed` its targets and windows (`frozen_shares()`, `window_halfwidths()`), `tail_selected` its first two
  targets and its tail range (`frozen_shares()`, `_tail_range()`). All come from the frozen bins' in-sample supports on
  all 32,950 training rows, so each fold's held-out rows and labels helped form them. The frozen cuts have the same
  historical provenance, and no arm of the comparison is fully nested.
- The held-out label flips are conditional on those fixed inputs. They show that no fit-side computation reads a
  fold's held-out labels, not that the constants are independent of them.
- The fold-local sensitivity was added after the results. It replaces only `gap_placed`'s targets and widths, keeps
  the frozen cuts and their historical provenance, and decides nothing.
- The filter population is every held-out training row, not the Pass 2 population. One conjunction is screened
  against obligations defined for a selected union, and GLASS checks them in sample at fit time. Its support figures
  are a standalone conjunction's, not a router's coverage.
- Inner engineers are fitted on about 21,088 rows, the scoring engineer on 26,360. `tail_selected` maps the chosen
  share onto the scoring values, where the tail can hold other states.
- The partner pool uses `bin = 1` literals of up to two non-ECI bins. GLASS's full rule space and union selection
  are PR 44's.
- Refit migration compares each fold's fit with the all-rows fit, which uses every training label. It measures
  stability, not accuracy.
- Wilson intervals cover binomial sampling of the held-out rows with every fit fixed. The fold and period removals
  overlap.
- Five candidates and five partner slots are judged. The criteria were fixed first, but a chance pass remains
  possible; PR 44's router comparison is the confirmatory step.
- Every encoder uses labels of later contacts (Commit 3), for every procedure alike.

## 10. Where the evidence is

| Evidence | Notebook cell | Record `binning_candidates.json` |
|---|---|---|
| the registration, its digest and its checks (PR 39 supports, GLASS profile, shared contract, frozen binning, partner pool) | 40 | `registration` |
| the reference inputs (`frozen_shares()`, `window_halfwidths()`, `_tail_range()`), their values, source and users | 41 | `reference_inputs` |
| held-out source checks, inner cross-fit provenance and checks | 41 | `inputs` |
| every fit's cuts, shares, gaps, bin rows and states; the tail selection; the all-rows fits | 41 | `fits` |
| held-out label flips with controls, and their scope (conditional on the reference inputs) | 42 | `isolation` |
| comparator parity, focal summaries, every bin table and T1–T4, migration, NSD × ECI joint tables | 43 | `comparison` |
| focal lift with each fold or period removed; subscriber sources | 44 | `dependence` |
| partners per fold (eligible table, chosen), the three variants per fold and slot, pooled per slot and partner, partition checks, the support scope | 45 | `filter` |
| A0–A4 per candidate, verdicts, family decisions, the `eci_warm` verdict, the revision needs | 46 | `decision` |
| the registered statements checked in the run; limits | 46 | `stated`, `limits` |
| the post-registration sensitivity: definition, fold-local targets and widths, cuts, the three-way summary, removals, descriptive criteria, persistence, its statements | 47 | `sensitivity` |
