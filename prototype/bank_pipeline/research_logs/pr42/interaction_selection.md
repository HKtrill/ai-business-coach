# PR 42 Commit 5, Step 5B — Procedural EBM interaction selection

**Questions.**
1. On the frozen PR 39 Stage 3 inventory of 13 features, does a reproducible selection procedure recover the three
   accepted interaction pairs when fitted on all training rows? Which pairs recur when selection is repeated
   independently inside each outer fold?
2. For the two candidates carried forward from Step 5A, `jed_alone` and `drop_first`, what does reselection under
   the existing three-pair budget recover? How does each reselected fit compare with its own 5A fixed-pair fit on
   identical held-out rows?

**In brief.** The procedure is interpret's own FAST ranking inside the accepted Stage 3 EBM (`interactions=3`).

| Configuration | Pairs (count) | Selected on all rows | Outer fits selecting (of 5) | ΔROC-AUC vs 5A [95% CI] | ΔPR-AUC | ΔNested F2 | Calibrated ECE (5A) | Material change |
|---|---|---|---|---|---|---|---|---|
| frozen 13 features | accepted: 3 | `campaign × emp_var_rate_sigmoid_slope`, `euribor3m_sigmoid_slope × campaign`, `decay_x_density × campaign` | 5, 3, 4; both TIER 2 pairs 0 | — | — | — | — | — |
| `jed_alone`, reselected | 5A: 2 → 3 | `campaign × emp_var_rate_sigmoid_slope`, `euribor3m_sigmoid_slope × campaign`, `joint_economic_decay × campaign` | 5, 3, 4; its 5A pair `euribor3m_sigmoid_slope × economic_curvature_intensity` 0 | −0.0006 [−0.0015, 0.0003] | −0.0012 | +0.0001 | 0.0026 (0.0027) | no |
| `drop_first`, reselected | 5A: 2 → 3 | `campaign × emp_var_rate_sigmoid_slope`, `decay_x_density × campaign`, `cons.conf.idx × campaign` | 5, 4, 1; its 5A pair `prior_x_stress × decay_x_density` 0 | +0.0005 [−0.0005, 0.0015] | +0.0013 | −0.0031 | 0.0028 (0.0018) | no |

- **The procedure recovers only the TIER 1 pair.** On all training rows it selects
  `campaign × emp_var_rate_sigmoid_slope` first. Every outer fit of every configuration selects it.
  - The two TIER 2 pairs rank 30th and 38th of 78 on all rows, and no fit selects them.
  - That matches the history: PR 39 took the TIER 1 pair from its research round's FAST ranking, and chose the TIER 2
    pairs by judgement.
- **What recurs is campaign crossed with the economic regime.** The other recurring pairs pair `campaign` with a
  correlated macro feature: `decay_x_density` or `joint_economic_decay` (4 of 5 outer fits) and
  `euribor3m_sigmoid_slope` (3 of 5).
  - With them in the model, the TIER 1 pair's surface keeps its pattern but shrinks from 0.56–0.67 to 0.23–0.28
    log-odds at its largest well-supported cell.
  - *Interpretation of the observed surfaces, not a tested decomposition:* the three campaign pairs read as one
    campaign-by-regime effect that the data do not assign to a single macro feature, as Step 5A found for the
    correlated slopes' main effects.
- **The third slot is not stable everywhere.** In `drop_first` (no euribor slope) no third pair recurs: none is
  selected by more than 2 of 5 outer fits. Its decay × campaign count is 4 of 5, and one fold's choice between it and
  `decay_x_density × dow_month_encoded` is a one-bag near-tie.
- **Against their own 5A fits, neither reselected candidate changes materially in the sandbox.** Every pooled
  difference is under the registered descriptive thresholds, and every ROC-AUC interval includes zero. Both reselected
  fits flag more rows than their 5A fits (+350 and +400), which changes 702 and 684 decisions. The Windows record gives
  its own verdicts, recomputed by the registered rule, and flags any that differ.
- **Complementarity is computed only in the Windows run.** Cell 66 applies Step 5A's cascade screen unchanged to the
  5A and reselected fits on Stage 2 abstentions. In 5A, `jed_alone` met the screen (486) and `drop_first` missed it
  (441, needed 456.86). Whether reselection changes either verdict is for the accepted Windows run to report (section
  8); the sandbox has no accepted artifacts.
- **How a run is judged (section 9).** Cell 67 keeps two questions apart.
  - *Acceptance* must hold in every run: fit isolation, ranking-to-fitted-term agreement, paired row identity,
    accepted Step 5A reproduction, and truthful materiality and cascade verdicts.
  - *Same-sandbox reproduction* of the figures quoted here is required only in the sandbox's environment. On Windows,
    differences are recorded beside the sandbox values. Changed selections or counts, changes of direction, changed
    verdicts and statements that no longer hold are flagged for review, not failed. The record is written before
    either is enforced.

**Scope.** This is training-side work on the 32,950 training rows only (G7). The frozen PR 39 artifacts and the
accepted 5A results are preserved: 5A is rerun unchanged in the same notebook run, and every comparison uses its fits.
Budgets other than three belong to PR 44, Step 44C, cascade comparisons to PRs 44–45, and chronological evaluation to
PR 49.

**Evidence.**

- *Where it is produced.* `feature_research/feature_research.ipynb`, Cells 60–68 (module
  `feature_research.interaction_selection`), after Step 5A's Cells 49–59 in the same run. The notebook writes
  `research_logs/pr42/<run_id>/interaction_selection.json` beside the run's six earlier records.
- *Which run.* The numbers below are the sandbox run `c5_20261124_015820`'s (Linux, Python 3.13).
  - `feature_research/interaction_selection_sandbox.json` (`IS.SANDBOX_REFERENCE`, pinned by its LF SHA-256) holds
    them as 888 items, read from that run's record: every selection and margin, every pair's counts and positions,
    and every estimate, difference and verdict.
  - This revision's sandbox run `c5_20261124_090104` reproduces all 888 items exactly and passes all 15
    acceptance checks and the 49 statements.
  - In the Windows run, the statements become 54: five complementarity statements replace the one marked not
    checkable.
  - Selections are stated exactly where their margin allows. A margin is the gap between the third and fourth pairs'
    mean ranks. Every outer and all-rows margin is at least 0.64 mean-rank units, about nine bag-level rank swaps,
    except `drop_first`'s outer fold 3 (0.071, one swap), whose count is stated as a bound.
  - Inner selections include near-ties, so their counts carry a tolerance. Fit-dependent differences carry the
    tolerance stated beside each.
- *Reuse.* The selector is interpret's own. `IS.ProceduralEBMFactory` subclasses the production `EBMFactory` and
  changes only `interactions`, from an explicit list to 3. Every fit goes through Step 5A's
  `SS.fit_configuration`, which now takes the factory as an argument; the protocol is otherwise unchanged:
  - `Stage3Block` folds;
  - `OOFGenerator` outer and inner scores;
  - `Stage3Calibrator`;
  - `F2ThresholdSelector`.

  A recorder observes interpret's per-bag rankings without changing the fit.
- *Tests.* `tests/feature_research/test_pr42_interaction_selection.py` (21, synthetic) covers:
  - the registration and the history, read from the pinned blob even when the working copy's outputs are cleared,
    and reported as not checked without Git or a pinned copy;
  - the accepted Step 5A record rule (integrity; reproduction where the environment matches; required in the
    acceptance run);
  - the recorder against the fitted pairs, every eligible pair ranked in every bag, and interpret's tie rule;
  - a held-out perturbation: flipping fold 0's held-out labels in every frame where they appear (they are fit rows
    of folds 1–4) leaves fold 0's outer and five inner selections unchanged, while a planted interaction in fold 0's
    own fit labels changes its selection;
  - every selection matched to its own fit frame;
  - the budget-aware terms check;
  - recurrence and change classes;
  - the paired comparison and the surfaces, including a pair selected by one fold (fold sd unavailable, in the
    table and the figure);
  - acceptance, which fails on a misreported verdict, a shifted fold plan or a selection that disagrees with its
    fitted terms;
  - reproduction: required and failing on a difference in the sandbox's environment, and elsewhere passing while
    it flags a changed direction, verdict or selection;
  - a statement about a surface that is absent (the all-rows fit does not hold the pair) or a statistic that is
    unavailable, missing or not finite: it does not hold, with its reason, rather than stopping Cell 67. It is recorded
    and flagged for review, and it still fails the same-sandbox reproduction;
  - the notebook layout.

## 1. The registration and the history

`IS.REGISTRATION` (id `pr42-step5b-registration`, version 1.0.0, digest `9d18391a…21ba`) was fixed at 09:35 UTC on
24 November, before any Step 5B selection ran on the training rows. Cell 60 checks the digest first. The
registration records:
- the selector, the eligible pairs, the ranking, ties and the budget;
- where selection runs;
- every comparison and its descriptive thresholds;
- the screen;
- what was known beforehand: Step 5A's results, the Windows 5A complementarity, and PR 39's research round.

**How PR 39 chose the accepted pairs** (read back from the tracked sources in Cell 60):

| Pair | PR 39 label | How it was chosen | Reproducible here? |
|---|---|---|---|
| `campaign × emp_var_rate_sigmoid_slope` | TIER 1, "research-validated, importance=0.205" | The research round (exploratory notebook Cells 16A/16B) used interpret's FAST with `interactions 4`, tuned by Optuna between 0 and 5, and selected it with 0.0529 | The mechanism, yes. The round itself (14 main effects, an earlier pipeline, other hyperparameters) no. The 0.205 is in the comment of `interactions.py`; none of the pinned notebook's printed outputs gives it (below) |
| `euribor3m_sigmoid_slope × economic_curvature_intensity` | TIER 2, "exploratory; consistent with research derivative analysis" | Judgement: not among the research round's four pairs | No; there is no rule to reproduce |
| `prior_x_stress × decay_x_density` | TIER 2, "prior contact quality under stress …" | Judgement: not among the research round's four | No |

**Where the history is read from.** Cell 60 reads the exploratory notebook as Step 4A pinned it (`BD.EXPLORATORY`),
never from the working copy:
- the source is Git blob `266860ba…`, committed in `42753db`, with LF SHA-256 `136c8fb7…`;
- the working copy is used only where Git lacks the blob, and only if it is byte-for-byte the pinned notebook;
- otherwise the history is reported as not checked, and a mismatch stops the run;
- the record keeps the identity read: commit, blob, digest, and whether Git or the working copy supplied it.

So clearing the working copy's outputs, as Commit 6 may, changes nothing; a test clears them and gets the same
statements. The blob holds the cited outputs: Cell 16A prints `interactions 4`, and Cell 16B prints the four pairs
below with their importances. Cells are found by their header lines, not by position.

**What is said about "importance=0.205", and on what evidence.** The number is in the TIER 1 comment of
`glass_pipeline/ebm/interactions.py`. Cell 60 searched every printed output of the pinned notebook: the 30 code cells
with output, Cells 16A and 16B among them. None contains "0.205", and Cell 16B gives that pair 0.0529. Other notebooks,
logs and earlier commits were not inspected. The registration (1.0.0, unchanged) says "no tracked output gives it",
which is broader than what was inspected; this document makes only the scoped statement.

The research round's four pairs were:
- `decay_x_density × economic_curvature_intensity` (0.0579);
- `campaign × emp_var_rate_sigmoid_slope` (0.0529);
- `economic_curvature_intensity × dow_month_encoded` (0.0448);
- `decay_x_density × emp_var_rate_sigmoid_slope` (0.0095).

**The accepted Step 5A record.** Before any Step 5B fit, Cell 60 compares this run's Step 5A record (Cell 59) with
the accepted Windows run `c5_20261124_002645` (content `d70330e1…`), by Step 3B's rule:
- the accepted record's sidecar and content digest are checked wherever its directory exists, and the acceptance run
  needs the directory;
- exact reproduction is required where this run has that run's environment and accepted views.

The content digest leaves out run, seeds, code and environment, so the Step 5B code does not move it. The sandbox,
which has no accepted directory, reports the check as not required.

**The judgement calls this step turns into rules:**
- the budget is the existing three pairs;
- the ranking is interpret's own, with no tiers, manual exclusions or hand-picked pairs;
- every fit selects for itself, with no consensus step across folds.

## 2. The selector

Inside one fit of the accepted Stage 3 EBM, with Step 5A's settings, `interactions=3`, random_state 42 and n_jobs 1:
1. Each of the 14 outer bags boosts the main effects.
2. Each bag then ranks every eligible pair by FAST strength on its training rows, with the fit's balanced weights.
3. A pair's score is its mean rank across the 14 bags.
4. The three smallest mean ranks are boosted; equal mean ranks go to the smaller column indices (interpret's rule).

The eligible pairs are every pair of the fit's features: 78 for the frozen 13 and for `jed_alone`, 66 for
`drop_first`. `campaign × emp_var_rate_sigmoid_slope` is among them wherever both features are present.

Selection runs in every fit, on that fit's rows only:
- the frozen 13: once on all 32,950 rows (descriptive) and inside each outer fold's 26,360 fit rows;
- each candidate: the 5 outer fits, the 25 inner fits whose scores feed calibration and the fold-nested thresholds,
  and the all-rows fit for the surfaces.

Cells 61–62 check every selection (6 for the recovery, 31 for each candidate):
- it ran on exactly one fit frame's rows, matched by index fingerprint;
- each of its 14 bags ranked all eligible pairs;
- its recorded top three equal the fitted model's pair terms.

Step 5A's fold-isolation check of calibration and thresholds also passes for both reselected fits.

## 3. Recovery on the frozen 13 features

| Fit | Pair 1 | Pair 2 | Pair 3 | Margin |
|---|---|---|---|---|
| outer 0 | campaign × emp_var slope | euribor slope × campaign | decay_x_density × campaign | 2.57 |
| outer 1 | campaign × emp_var slope | euribor slope × campaign | decay_x_density × campaign | 2.64 |
| outer 2 | euribor slope × campaign | decay_x_density × campaign | campaign × emp_var slope | 4.21 |
| outer 3 | decay_x_density × cons.conf.idx | campaign × emp_var slope | euribor slope × cons.conf.idx | 0.86 |
| outer 4 | campaign × emp_var slope | ECI × behavioral_favorability | decay_x_density × campaign | 0.86 |
| all rows | campaign × emp_var slope | euribor slope × campaign | decay_x_density × campaign | 1.43 |

Of the three accepted pairs, the all-rows selection recovers one: `campaign × emp_var_rate_sigmoid_slope`, ranked
first, and selected by all five outer fits. `euribor3m_sigmoid_slope × economic_curvature_intensity` ranks 30th and
`prior_x_stress × decay_x_density` 38th of 78 on all rows, and no outer fit selects either.

Recurrence across the outer fits:

| Pair | Outer fits (of 5) | Class |
|---|---|---|
| `campaign × emp_var_rate_sigmoid_slope` | 5 | stable |
| `decay_x_density × campaign` | 4 | recurring |
| `euribor3m_sigmoid_slope × campaign` | 3 | recurring |
| `economic_curvature_intensity × behavioral_favorability`, `decay_x_density × cons.conf.idx`, `euribor3m_sigmoid_slope × cons.conf.idx` | 1 each | occasional |
| both TIER 2 pairs | 0 | never |

Of the research round's four pairs, only the TIER 1 pair is selected here. That comparison is only indicative: the
round had another feature set, other settings and another pipeline.

## 4. The candidates, reselected

Both 5A candidates kept two of the accepted pairs after their feature change. Reselection gives each three.

**`jed_alone`.** 5A pairs: `campaign × emp_var_rate_sigmoid_slope` and
`euribor3m_sigmoid_slope × economic_curvature_intensity`.

| Pair | Outer (of 5) | Inner (of 25) | All rows | Change against 5A |
|---|---|---|---|---|
| `campaign × emp_var_rate_sigmoid_slope` | 5 | 17 | 1st | persists |
| `joint_economic_decay × campaign` | 4 | 20 | 3rd | enters |
| `euribor3m_sigmoid_slope × campaign` | 3 | 15 | 2nd | enters |
| `euribor3m_sigmoid_slope × economic_curvature_intensity` | 0 | 0 | 32nd | disappears |
| three others | 1 each | 2–6 | — | occasional entrants |

**`drop_first`.** 5A pairs: `campaign × emp_var_rate_sigmoid_slope` and `prior_x_stress × decay_x_density`.

| Pair | Outer (of 5) | Inner (of 25) | All rows | Change against 5A |
|---|---|---|---|---|
| `campaign × emp_var_rate_sigmoid_slope` | 5 | 19 | 1st | persists |
| `decay_x_density × campaign` | 4 | 22 | 2nd | enters |
| `decay_x_density × dow_month_encoded` | 2 | 3 | — | occasional entrant |
| `cons.conf.idx × campaign` | 1 | 2 | 3rd | occasional entrant |
| `prior_x_stress × decay_x_density` | 0 | 0 | 29th | disappears |
| three others | 1 each | 1–6 | — | occasional entrants |

`jed_alone` reproduces the frozen-13 pattern, with `joint_economic_decay` in the decay slot. `drop_first` has no
euribor slope, so its third slot has no stable occupant.
- In its outer fold 3, `decay_x_density × dow_month_encoded` (mean rank 4.214) leads `decay_x_density × campaign`
  (4.286) by one bag swap. On another platform that fold could make the count 5 of 5 for decay × campaign and 1 for
  decay × dow_month.
- Its all-rows third pair, `cons.conf.idx × campaign`, is selected by only one outer fit.

The inner counts follow the outer ones, but the inner selections include near-ties (one exact tie, broken by column
order); they are supporting evidence.

## 5. Each reselected candidate against its own 5A fit

Comparisons are on the same 32,950 out-of-fold rows, with the same folds, inner folds, settings, calibration and
operating points. Only the pairs differ, both in identity and in count (2 → 3), so a difference belongs to the
procedure as a whole.

| Quantity | `jed_alone` 5A → reselected | `drop_first` 5A → reselected |
|---|---|---|
| ROC-AUC | 0.7934 → 0.7928; Δ −0.0006 [−0.0015, 0.0003], p 0.22 | 0.7930 → 0.7935; Δ +0.0005 [−0.0005, 0.0015], p 0.31 |
| PR-AUC | 0.4176 → 0.4164 (−0.0012) | 0.4152 → 0.4165 (+0.0013) |
| nested F2 | 0.5518 → 0.5519 (+0.0001) | 0.5513 → 0.5482 (−0.0031) |
| Brier, calibrated | 0.0796 → 0.0797 (+0.0001) | 0.0797 → 0.0795 (−0.0001) |
| ECE, raw → calibrated | 0.2712 → 0.0027 against 0.2714 → 0.0026 | 0.2718 → 0.0018 against 0.2713 → 0.0028 |
| decisions changed (flagged, 5A → reselected) | 702 (6,065 → 6,415) | 684 (6,195 → 6,595) |

Per fold (mean, sd and folds better of 5):
- `jed_alone`: ROC-AUC −0.0007 (sd 0.0011), better in 1; PR-AUC −0.0012 (0.0040), better in 2; nested F2 +0.0001
  (0.0026), better in 2.
- `drop_first`: ROC-AUC +0.0005 (0.0008), better in 4; PR-AUC +0.0013 (0.0009), better in all 5; nested F2 −0.0029
  (0.0083, from −0.0168 to +0.0051), better in 2.

No pooled difference reaches a registered descriptive threshold: |ΔROC-AUC| ≥ 0.002 or a CI excluding 0,
|ΔPR-AUC| ≥ 0.005, |Δnested F2| ≥ 0.005, |Δcalibrated Brier| ≥ 0.0005. Calibrated ECE stays far under the 0.05 gate.
The per-fold findings are directional only: `drop_first`'s PR-AUC is higher in every fold, and `jed_alone`'s ROC-AUC
is lower in four. Both reselected fits flag more rows at their fold-nested thresholds, which matters for
complementarity (section 8).

## 6. Learned surfaces of the recurring and changed pairs

Cell 65 draws every pair that recurs (≥ 3 outer fits) in any configuration and every 5A pair that disappears, six in
all. Each figure uses Step 5A's support-defined cells, with matched axes and one symmetric colour scale across the
models that hold the pair. In the table:
- "share of all terms" is the pair term's mean |contribution| over all terms', all-rows fit;
- "sign agreement" is the share of well-supported cells with an effect of at least 0.05 log-odds where every fold
  model holding the pair agrees with the all-rows sign;
- "selecting folds" counts the outer folds whose model holds the pair. A fold sd and a sign agreement need at least
  two. With fewer, the table and the figure say "unavailable", never 0. Every pair shown here has at least three
  in the sandbox.

| Model | Pair | Pair, mean \|contribution\| | Main effects (a, b) | Share of all terms | Largest supported effect | Selecting folds (of 5) | Sign agreement (cells) |
|---|---|---|---|---|---|---|---|
| baseline 5A | campaign × emp_var slope | 0.052 | 0.096, 0.155 | 2.4% | 0.671 | 5 | 1.00 (14) |
| `jed_alone` 5A | campaign × emp_var slope | 0.043 | 0.095, 0.164 | 2.0% | 0.561 | 5 | 1.00 (9) |
| `jed_alone` reselected | campaign × emp_var slope | 0.016 | 0.095, 0.164 | 0.75% | 0.228 | 5 | 1.00 (5) |
| `jed_alone` reselected | joint_economic_decay × campaign | 0.012 | 0.236, 0.095 | 0.56% | 0.220 | 4 | 1.00 (8) |
| `jed_alone` reselected | euribor slope × campaign | 0.014 | 0.166, 0.095 | 0.64% | 0.164 | 3 | 0.90 (10) |
| `drop_first` 5A | campaign × emp_var slope | 0.046 | 0.097, 0.173 | 2.1% | 0.582 | 5 | 1.00 (10) |
| `drop_first` reselected | campaign × emp_var slope | 0.016 | 0.097, 0.173 | 0.75% | 0.283 | 5 | 1.00 (6) |
| `drop_first` reselected | decay_x_density × campaign | 0.021 | 0.321, 0.097 | 1.0% | 0.283 | 4 | 0.625 (16) |
| baseline / `jed_alone` 5A | euribor slope × ECI | 0.010 / 0.013 | — | 0.5% / 0.6% | 0.048 / 0.054 | 5 | — (0) / 1.00 (1) |
| baseline / `drop_first` 5A | prior_x_stress × decay_x_density | 0.007 | — | 0.3% | 0.068 / 0.066 | 5 | 0.00 (1) |

**What the interactions add over the main effects.**
- **campaign × emp_var slope.** In every model that holds it, its pattern is Step 5A's: repeated contacts (campaign 6
  and above) at the emp.var.rate slopes of late 2008 and spring 2009–2010 score lower than the main effects predict.
  Every fold model agrees on the sign wherever the effect is at least 0.05. Next to the other campaign pairs it is
  about a third the size (0.23–0.28 against 0.56–0.67) and carries 0.75% of the model's attribution against 2.0–2.4%.
- **The entering campaign pairs** (`joint_economic_decay`, `decay_x_density` or `euribor3m_sigmoid_slope` crossed
  with `campaign`) reach 0.16–0.28 log-odds in their strongest well-supported cell. Their mean contribution, 0.012–0.021,
  is a twelfth or less of their macro feature's main effect (0.17–0.32).
  - `joint_economic_decay × campaign` and `euribor3m_sigmoid_slope × campaign` keep their sign in the fold models (1.00
    and 0.90).
  - `decay_x_density × campaign` does in only 10 of its 16 strong cells, with a largest fold sd of 0.19 against a
    largest effect of 0.28: it persists where the rows are many, but its pattern is not settled.
- **Together** the campaign pairs carry about as much attribution as the 5A model's TIER 1 pair did alone. In
  `jed_alone` the three carry 1.95% against the 5A TIER 1 pair's 2.0%.
- **Interpretation (of the observed surfaces, not a tested decomposition).** We read this as one
  campaign-by-economic-regime effect, redistributed across correlated macro features once more than one campaign
  pair is allowed. Step 5A found the same for the correlated slopes' main effects. The evidence is:
  - the TIER 1 surface keeps its pattern but shrinks;
  - the entering pairs pair `campaign` with features that move with the same regime;
  - the total attribution stays about the same.

  Nothing here tests that reading. It would take a decomposition or ablation of the correlated pairs, which this
  step does not run.
- **The disappearing 5A pairs** stay what Step 5A found them to be: at most 0.05–0.07 log-odds inside the support,
  with fold spread as large or larger.

## 7. What changes, and what does not

- **Recurs.** Campaign crossed with the economic regime, in every configuration. The TIER 1 pair is selected by all
  15 outer fits across the three configurations.
- **Enters.** Campaign crossed with the other regime features available: the decay slot, and the euribor slope where
  it exists.
- **Disappears.** Both TIER 2 pairs, in every configuration.
- **Unchanged.** Pooled predictive performance and calibration against each candidate's 5A fit.
- **Changed.** What the surfaces show: one strong campaign surface in 5A, three weaker ones after reselection; read
  as one effect redistributed (section 6, an interpretation). The decisions also change: 684–702, with more rows
  flagged.

## 8. Complementarity under Step 5A's cascade screen

Cell 66 computes this only where the accepted views passed (Cell 49); with `PR42_REQUIRE_ACCEPTED_VIEWS=1` it is
required. Step 5A's comparison is reused with the reselected fits added beside the 5A ones. For all rows and the
14,628 Stage 2 abstentions it gives:
- the four paired counts;
- the Stage 1 errors corrected and the correct Stage 1 decisions overturned;
- the net against the 5A baseline.

Step 5A's screen is applied unchanged to each fit: within the 5A tolerances against the baseline, and an abstention net
of at least the baseline's net minus 0.5% of the abstention rows.

**What the Windows run establishes** (section 9):
- *Acceptance:* every screen verdict reported is Step 5A's rule applied to that run's numbers. The 5A candidates'
  verdicts equal that run's Step 5A verdicts, and the `change` table reports each 5A and reselected verdict
  truthfully.
- *The 5A anchors* (baseline net 530, `jed_alone` 486 meets, `drop_first` 441 misses) are statements. Where the
  accepted 5A record must be reproduced exactly (same environment and views), they cannot move without failing
  acceptance. Otherwise a change is flagged for review.
- *Whether reselection changes either verdict* is the finding. A changed verdict is flagged for review and recorded.

A changed verdict says that the reselected fit corrects more or fewer Stage 1 errors on abstentions under the same
rule. It is not evidence of a better cascade; PRs 44–45 compare cascades at comparable coverage. Both reselected fits
flag more rows than their 5A fits, so their abstention counts can move in either direction; the sandbox cannot tell
which.

## 9. How a run is judged: acceptance and same-sandbox reproduction

Cell 67 asks two different questions, and Cell 68 writes the record before it enforces either. A run that stops
there still leaves its evidence (selections, comparisons, surfaces, the reproduction table and the review flags) to
inspect without repeating the fits.

**Acceptance (every run).** `IS.acceptance_checks` holds the run to its invariants, 15 checks in the sandbox
and 17 with the accepted views:
- *Fit isolation.* Step 5A's fitted-terms, fit-size and fold-isolation checks pass for every reselected fit. All 68
  selections ran on one fit frame's rows, and each outer selection on exactly the rows outside its held-out fold.
- *Ranking-to-fitted-term agreement.*
  - In every selection the recorded top three by mean rank are the fitted pairs.
  - Every eligible pair was ranked in all 14 bags.
  - The models that scored the held-out rows, and the all-rows models, hold exactly their selection's pairs.
- *Paired row identity.*
  - Labels, the fold plan and the frames are the same 32,950 rows in the same order.
  - Each outer frame's held-out rows are its fold in that plan.
  - Each reselected fit and its 5A fit score the same rows from the same fit and inner frames.
- *Accepted Step 5A reproduction.* Cell 60's rule (section 1) and Cell 58's statements.
- *Truthful verdicts.*
  - The registered thresholds are intact (registration digest; materiality thresholds; Step 5A's screen
    tolerances).
  - Each materiality verdict is recomputed from the pooled differences.
  - Each cascade verdict is recomputed from the pooled differences and abstention nets.

**Same-sandbox reproduction (required only in the sandbox's environment).** `IS.sandbox_reproduction` sets this run
beside `c5_20261124_015820`'s 888 items and evaluates this document's statements (`IS.STATED`, 49 in the sandbox).
Every item has the sandbox value, this run's value and the difference.
- Where this run has the sandbox's environment (`TV.ENVIRONMENT_KEYS`) and view availability, every item must equal
  and every statement must hold. This revision's sandbox run does.
- Elsewhere, including the Windows acceptance run, nothing in it fails. These are flagged for review in the record's
  `review` section and in Cell 67:
  - a changed outer or all-rows selection, or a changed count;
  - a change of direction in a registered comparison, marked when it is below the materiality threshold;
  - a changed verdict;
  - an inner count beyond the stated ±3;
  - a statement that no longer holds;
  - a cascade verdict that reselection changes.
- Other differences (margins, inner selections, estimates within their stated tolerances) are recorded without a
  flag.
- The registered thresholds are not adjusted to make a run pass.

## 10. Handoff

- **To Step 6A.**
  - The procedure (interpret FAST, mean rank over 14 bags, three pairs, selected inside every fit) is reproducible,
    fold-fitted and recorded.
  - It supports the TIER 1 pair and none of the TIER 2 pairs.
  - For the two candidates, reselection changes which campaign pairs carry the observed surface (read in section 6 as
    one effect redistributed). The pooled differences against the 5A fits stay below the registered thresholds in
    the sandbox; the Windows record says whether that holds there.
  - The Windows run decides whether it changes the cascade screen.
- **To PR 44, Step 44C (budgets).** Two questions follow from the shared campaign-by-regime effect:
  - whether one or two pairs keep performance with a more concentrated surface;
  - whether a single campaign pair should be fixed by construction instead of left to a ranking that divides the
    effect among correlated macro features.

  This step compares no budgets.

## 11. Figures to inspect

In the executed notebook:
- **Cell 63:** the selection map, showing which pairs each outer fit and the all-rows fit selected, for the frozen 13
  features and both reselected candidates.
- **Cell 65:**
  - `campaign × emp_var_rate_sigmoid_slope` across the baseline, both 5A fits and both reselected fits on one colour
    scale: the same pattern, weaker after reselection;
  - `decay_x_density × campaign`, `joint_economic_decay × campaign` and `euribor3m_sigmoid_slope × campaign`, each
    fold-sd panel titled with its selecting folds ("n of 5"; "unavailable" below two);
  - the two disappearing TIER 2 pairs (as in Step 5A).
- **Cell 66 (Windows only):** the complementarity tables, the screen and the `change` table.
- **Cell 67:** the acceptance checks, the reproduction summary by kind, every item that differs from the sandbox, and
  the review flags.

## 12. Limits

- *Platform and version.* The selector is interpret 0.7.8's FAST ranking, and another version may rank differently.
  Fit-dependent values differ slightly between platforms, and a pair near the selection boundary can change.
  Selections are stated exactly only where their margin allows. The sandbox figures are required to reproduce only in
  the sandbox; elsewhere differences are recorded and flagged (section 9).
- *Interpretation.* That the correlated campaign pairs share one campaign-by-regime effect is a reading of the
  observed surfaces, not a tested decomposition of attribution.
- *Fold sd.* A pair selected by fewer than two outer folds has none; it is reported as unavailable.
- *History.* The 0.205 statement covers the pinned exploratory notebook's printed outputs only.
- *Attribution.* The comparison with 5A changes the pair count as well as the pairs (2 → 3), so it measures the
  procedure. Budgets are Step 44C's.
- *Recovery.* The all-rows selection is descriptive; the outer-fold selections are the evidence of recurrence, and
  inner selections are supporting counts.
- *What a rank means.* FAST strengths rank pairs on each bag's main-effect residuals. They are not a test of
  interaction, and their ranks are not comparable across feature sets.
- *The recorder.* It observes interpret's internal ranking function. Its agreement with the fitted pairs is checked in
  every fit.
- *Complementarity.* It needs the accepted views (Windows). A changed screen verdict is not a better cascade.
- *Split.* Training side, random outer folds. Chronological evaluation is PR 49's.

## 13. Where the evidence is

| Evidence | Notebook cell | Record `interaction_selection.json` |
|---|---|---|
| the registration, its digest and the history read back, with its pinned source | 60 | `registration` |
| the accepted Step 5A record: integrity and reproduction where required | 60 | `accepted_5a` |
| recovery: every selection, its rows, ranking and margin; the summary | 61 | `recovery` |
| the reselected fits: selections, thresholds, calibration, terms, out-of-fold digests; checks | 62 | `candidates`, `fit_checks` |
| recurrence and change | 63 | `recurrence`, `change` |
| against each 5A fit: pooled, per fold, decisions | 64 | `against_5a` |
| surfaces and what each pair adds (selecting folds; fold sd where available) | 65 | `surfaces` |
| complementarity and the screen (Windows), requirement checks | 66 | `complementarity` |
| acceptance checks and their verdict | 67–68 | `acceptance` |
| the sandbox reproduction: 888 items, this document's statements, checks | 67–68 | `reproduction` |
| everything flagged for review | 67–68 | `review` |
| limits | 68 | `limits` |
