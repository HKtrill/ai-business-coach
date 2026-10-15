# Research Questions: Pass-Specific Tuning Objectives for the Two-Pass RF Router

_Recorded before the held-out evaluation (Cell 28) was inspected._

## Primary question

**RQ0.** At the shared Stage 2 operating constraints, under a common tuning budget and
evaluation protocol, does selecting each pass's hyperparameters for its own routing objective
improve held-out routing performance relative to ROC-AUC selection, without violating any hard
operating constraint?

- **Pass 1 objective.** Maximize NOT_SUBSCRIBE coverage subject to the configured NPV,
  subscriber-leakage, and support constraints. Coverage is the fraction of training rows in
  the largest feasible band, as solved by the router's own threshold solver on out-of-fold
  scores; a configuration with no feasible band scores zero. NPV enters as a constraint, not
  as the quantity maximized.
- **Pass 2 objective.** Maximize SUBSCRIBE recall (global denominator) on the Pass 1
  remainder, subject to the configured precision floor and support constraint (and the recall
  floor, when configured); a configuration with no feasible band scores zero.
- **Baseline objective.** Mean per-fold ROC-AUC on the population the pass is trained on.
- *Effectiveness:* Pass 1 share (rows routed NOT_SUBSCRIBE) and Pass 2 recall (global denominator).
- *Reliability:* satisfaction of every configured hard constraint on the held-out split.
- *Scope:* RQ0 concerns the effect of the objective against matched ROC-AUC controls. Whether
  a specialized configuration should replace the existing baseline (A0) is a separate
  question, addressed in RQ6b.
- *Prior expectation:* little or no effect. Thresholds are already solved separately for each
  pass at that pass's constraints, so the tuning objective can only change ranking quality
  near a single operating point. A preliminary baseline run also showed closely aligned
  Pass 1 and Pass 2 rankings on the remainder (Spearman ≈ 0.97), which is consistent with,
  but does not establish, limited additional ranking information in the remainder.

**Arms.**

| Arm | Pass 1 | Pass 2 |
|---|---|---|
| A0 | Existing baseline parameters (ROC-AUC tuned outside this harness) | Inherits Pass 1 |
| B | ROC-AUC, full training split | Inherits Pass 1 |
| C | B's Pass 1 | ROC-AUC on B's remainder (matched control for D) |
| D | B's Pass 1 | Recall-at-precision on B's remainder |
| E | Coverage-at-constraints | Recall-at-precision on E's remainder |

Arms B–E share one tuning budget: search space, trial count, sampler seed, tuning folds, and
n_estimators = 500. A0 was tuned outside this harness and is not budget-matched.

## Supporting questions

**RQ1 — Objective effect, per pass.**
- **RQ1a (Pass 1).** With the tuning budget held fixed, does coverage-at-constraints selection
  increase held-out Pass 1 share relative to ROC-AUC selection? *Contrast:* E − B on
  `pass1_share`. Pass 1 routing depends only on Pass 1 hyperparameters, so this contrast is
  unaffected by E's Pass 2 changes. Coverage and NPV are not interchangeable: with
  NPV = 1 − L/k, both the routed band size k and the number of leaked subscribers L can
  differ between configurations. Only when two configurations leak the same number of
  subscribers does greater coverage imply higher NPV. RQ1a therefore makes no NPV claim;
  NPV is assessed as a constraint (RQ3).
- **RQ1b (Pass 2).** On the same frozen remainder, with the same budget and trial count, does
  recall-at-precision selection increase held-out Pass 2 recall relative to ROC-AUC selection?
  *Contrast:* D − C on `pass2_recall`. Both arms share B's Pass 1, so the difference arises in
  Pass 2 alone. Both are thresholded to the same precision floor on training OOF; held-out
  precision is not matched, so recall differences are read together with RQ3 eligibility.

**RQ2 — Attribution.** How much of any difference is due to the objective, rather than to
re-tuning, the tuning population, or coupling between the passes?

| Contrast | Metric | Isolates | Role |
|---|---|---|---|
| B − A0 | pass1_share, pass2_recall | Sensitivity of the AUC baseline to re-tuning (budget, n_estimators = 500, seeds) | Descriptive |
| C − B | pass2_recall | Tuning Pass 2 on its own remainder (objective fixed at AUC) | Decision |
| D − C | pass2_recall | Pass 2 objective (population, budget and trials fixed) | Decision |
| E − B | pass1_share only | Pass 1 objective | Decision |
| E − D | pass2_recall | Effect of Pass 1's objective on Pass 2 through the remainder | Descriptive |

E − B is not interpreted on Pass 2 or whole-router metrics, because E changes both passes.
*Limitation:* each arm is a single tuning run, so every contrast also contains tuning-run
variability (which configuration the search selects). The noise floor does not measure this.
B − A0 gives one indicative sample of how much AUC re-tuning alone moves the metrics.

**RQ3 — Feasibility.** On the held-out split, does each arm satisfy every configured hard
constraint: the NPV floor, leakage rate and absolute cap, both band-support floors, and the
Pass 2 precision floor? The coverage and recall floors are disabled in this configuration
and are reported as such. *Descriptive:* differences in resolved coverage, abstention rate
and abstain-region base rate (the Stage 3 population). Feasibility is assessed at the
held-out point estimate; coverage and abstention differences are reported without intervals.
Feasibility is evaluated separately from statistical improvement.

**RQ4 — Strength of evidence.**
- Are held-out differences distinguishable from zero (paired bootstrap, 2,000 resamples,
  95% interval) and at least max(0.01, 2 × training noise SD)?
- How optimistic were tuned training scores (best trial vs fresh-fold re-evaluation; a lower bound)?
- How variable is each objective across fresh fold and forest seeds?

Bootstrap intervals describe evaluation-sample uncertainty; fresh-fold seeds describe part of
training variability, measured for fixed, already-selected configurations. The complete
hyperparameter search was not repeated across seeds or independent training samples, so
neither source captures re-tuning variability or full retraining stability. The noise-scaled
threshold is a pre-registered convention, not a significance test. The three decision
contrasts use unadjusted 95% intervals.

**RQ5 — Mechanism (interpretive).** What does the evidence suggest about why specialization
did or did not change routing? For each arm we report:
- Spearman rank correlation between Pass 1 and Pass 2 scores on the remainder (global
  ranking similarity);
- agreement and overlap between the rows Pass 2 actually flags at its operating threshold and
  the same number of top-ranked rows by Pass 1, with the range attainable over tied Pass 1
  scores (similarity where routing decisions are made);
- each arm's operating point (band sizes and realised precision and recall) on training OOF
  and held-out data;
- Gate 0 degeneracy status and the Gate 1 headroom screen.

These are read jointly and reported as evidence consistent with candidate mechanisms, such
as limited additional ranking information in the remainder or limited headroom for any
objective near the operating point. They are not causal explanations. High global rank
correlation alone does not show that Pass 2 adds no useful information: two rankings can
agree on most rows yet differ on those near the threshold, which is where routing decisions
change.

**RQ6 — Design decision.**

**RQ6a — Objective effect.** Does a specialized objective outperform its matched ROC-AUC
control under the frozen decision rule? (Pass 1: E − B on Pass 1 share; Pass 2: D − C on
Pass 2 recall.) The rule assesses two things separately: a statistical improvement (paired
95% interval excluding zero and an effect of at least the pre-registered threshold), and the
specialized arm's satisfaction of every applicable hard constraint on the held-out split.

| Verdict | Interpretation |
|---|---|
| EFFECT OBSERVED (arm eligible) | Objective effect supported for that pass; proceed to RQ6b. |
| EFFECT OBSERVED BUT NOT ELIGIBLE | The improvement comes with a hard-constraint violation; not adoptable. |
| NO EFFECT DETECTED | No effect larger than the comparison could resolve (report the CI half-width); specialization not supported. |
| WORSE | Specialization underperforms its matched control. |
| INVALID EVALUATION / INSUFFICIENT BOOTSTRAP SUPPORT | Inconclusive. |

**RQ6b — Practical adoption.** Does the specialized configuration provide a meaningful benefit
over the existing baseline, A0? An objective effect is necessary but not sufficient: D can
outperform C while both fall short of A0, which would support an objective-specific effect
without justifying a change of baseline. For the pass concerned, adoption is considered only when:
1. RQ6a shows an objective effect in an eligible arm;
2. the specialized arm's held-out metrics are at least as good as A0's on both passes'
   primary metrics and better on its target metric, while satisfying every hard constraint;
3. the uncertainty of that comparison is stated: the frozen rule contains no paired interval
   against A0, so the comparison rests on held-out point estimates (a paired interval may be
   reported as an exploratory analysis outside the frozen rule);
4. the cost is acceptable, including nested tuning inside the Stage 2 cross-fit before Stage 4
   inputs are regenerated.

Arm − A0 differences mix the objective with tuning budget, n_estimators and tuning population.
They answer the adoption question, not the attribution question (RQ2).

*Stability.* Neither the paired bootstrap (evaluation-sample uncertainty) nor fresh-fold
re-evaluation (partial training variability) establishes retraining stability, because the
complete search was not repeated across seeds or independent training samples. A favourable
result therefore justifies considering adoption; it does not show that a specialized
configuration is as stable as, or more stable than, A0.

*Per-pass adoption.* Either pass may be specialized independently:
- **Pass 2 alone** corresponds to arm D. Adopting D means adopting its full configuration
  (B's re-tuned ROC-AUC Pass 1 with the recall-tuned Pass 2). Pairing the recall objective
  with A0's Pass 1 would require tuning Pass 2 on A0's remainder, which was not tested.
- **Pass 1 alone** has no dedicated arm, because E also specializes Pass 2. Adopting only the
  Pass 1 objective would require additional validation.

| Outcome | Decision |
|---|---|
| Objective effect supported and the specialized arm improves on A0 within constraints | Consider adopting that arm's configuration for the pass concerned; regenerate Stage 4 only after nested tuning (follow-up commit). |
| Objective effect supported, but the specialized arm does not improve on A0 | Retain A0; report the objective effect as a finding. |
| Gains appear only in the re-tuning or population contrasts (B − A0, C − B) | Not evidence for objective specialization; retain A0. Baseline re-tuning may be revisited as future work. |
| No supported objective effect | Retain A0 with ROC-AUC selection: the simpler, experimentally supported design. |

## Methodological boundaries

- The test split was examined in earlier research: results are held-out, not confirmatory.
- Training-OOF comparisons between tuned arms are diagnostic, since tuning used those labels.
- Statistical improvement and hard-constraint feasibility are assessed separately; neither
  substitutes for the other.
- Each arm is a single tuning run; contrasts include tuning-run variability.
- The three decision contrasts use unadjusted 95% intervals.
- Comparisons against A0 are not part of the pre-registered rule.
- One dataset, one split, one operating point (constraints matched to GLASS training OOF).
  Effects may differ at other operating points, such as a stricter precision floor.
- Stage 4 currently uses hyperparameters selected once on all of X_train; adopting specialized
  parameters requires nested tuning before the cascade is regenerated.
- The result is evidence for a design decision under this dataset and protocol, not a general
  claim about either tuning strategy.

## Future work (not required for PR #35)

- Repeated tuning runs (≥ 3 sampler and fold seeds per arm), to separate objective effects
  from search variability and to assess retraining stability.
- A Pass 1-only specialization arm (coverage-tuned Pass 1, ROC-AUC-tuned Pass 2).
- Recall-at-precision Pass 2 tuned on A0's remainder, to test Pass 2 specialization on top
  of the existing baseline.
- Paired held-out intervals for specialized arms against A0 (exploratory, outside the frozen rule).
- The distribution of constraint satisfaction across bootstrap resamples or retraining seeds.
- Other operating points.
- An untouched confirmatory split or external dataset.
- Nested tuning for Stage 4 (commit C, if specialization is adopted).

---

## Results — answers to RQ0–RQ6

**Outcome: no pass-specific objective met the pre-registered rule. The existing ROC-AUC
configuration (A0) is retained for both passes; Stage 4 is unchanged.**

_Single held-out evaluation (first logged evaluation of this split; decision rule frozen in
`decision_rule.json` before results were produced). Run provenance: tuning contract
`pr35-stage2-tuning-v2`, split `fc3e73a5ef91aea4`, data `a178bc418854e698`, budget `84934dbbb4b2200f`._

**Held-out metrics (test split, n = 8,238).**

| Metric | A0 | B | C | D | E |
|---|---|---|---|---|---|
| Pass 1 share | 36.38% | 37.18% | 37.18% | 37.18% | 38.14% |
| Pass 2 recall | 71.01% | 71.55% | 71.34% | 71.34% | 71.55% |
| Pass 2 precision | 29.20% | 28.35% | 29.21% | 29.20% | 27.98% |
| Subscribers lost by Pass 1 | 106 | 117 | 117 | 117 | 120 |
| Coverage (resolved) | 63.78% | 65.61% | 64.69% | 64.70% | 66.95% |
| Abstain-region base rate | 5.46% | 5.19% | 5.12% | 5.12% | 5.29% |
| Resolved F1 | 0.4361 | 0.4252 | 0.4348 | 0.4347 | 0.4207 |
| All hard constraints satisfied | yes | yes | yes | yes | yes |

**Paired contrasts (2,000 bootstrap resamples, 95% CI, percentage points).**

| Contrast | Metric | Δ | 95% CI | Threshold | Outcome |
|---|---|---|---|---|---|
| B − A0 (re-tuning) | Pass 1 share | +0.80 | [+0.33, +1.27] | 1.13 | Distinguishable, below threshold (descriptive) |
| B − A0 (re-tuning) | Pass 2 recall | +0.54 | [0.00, +1.15] | 1.00 | Not distinguishable (descriptive) |
| E − B (Pass 1 objective) | Pass 1 share | +0.96 | [+0.44, +1.48] | 1.13 | Distinguishable, below threshold |
| C − B (Pass 2 population) | Pass 2 recall | −0.22 | [−0.67, +0.20] | 1.00 | Not distinguishable |
| D − C (Pass 2 objective) | Pass 2 recall | 0.00 | [0.00, 0.00] | 1.00 | No difference |
| E − D (cascade interaction) | Pass 2 recall | +0.22 | [−0.52, +0.97] | 1.00 | Not distinguishable (descriptive) |

The verdict table labels sub-threshold contrasts whose interval excludes zero as "no effect
detected"; they are reported here as "distinguishable, below threshold". This does not change
any decision: the frozen rule requires both an interval excluding zero and an effect of at
least the threshold.

**RQ0 — Objective selection.**
*Answer:* No. Under matched budgets and identical constraints, neither pass-specific objective
improved held-out routing relative to its matched ROC-AUC control by the pre-registered margin.
*Evidence:* E − B and D − C above; all arms feasible.
*Caveat:* held-out, not confirmatory; one tuning run per arm; one operating point.

**RQ1a — Pass 1 objective.**
*Answer:* Coverage-at-constraints selection increased held-out Pass 1 share by a small,
statistically distinguishable amount that is below the pre-registered threshold.
*Evidence:* +0.96 pp [+0.44, +1.48] vs a 1.13 pp threshold. The arm routed 3 more subscribers
away (117 → 120; leakage 12.6% → 12.9%) at an unchanged NPV of 96.18%. On training data the
gain was not supported: re-evaluated on fresh folds, the coverage-tuned configuration's
coverage was 38.7%, against a tuned value of 40.4% (fold optimism 1.7 pp, about 3× the noise
SD), and no higher than the baseline configuration's 39.1% at the same tree count.
*Caveat:* the effect is of the same size as the shift produced by AUC re-tuning alone
(B − A0: +0.80 pp), consistent with tuning-run variability.

**RQ1b — Pass 2 objective.**
*Answer:* No effect. On the same frozen remainder, recall-at-precision selection produced the
same held-out Pass 2 recall as ROC-AUC selection.
*Evidence:* Δ = 0.00 pp in every bootstrap replicate. The two arms routed 8,237 of 8,238
test rows identically; the single difference was one non-subscriber flagged by D only, so
both flagged exactly the same subscribers. On training OOF, D exceeded C by 0.16 pp
(68.67% vs 68.51%), within training noise (SD 0.17 pp).
*Caveat:* the held-out precision of the two arms is matched only through the shared floor.

**RQ2 — Attribution.**
*Answer:* Pass 1 objective specialization produced a small, statistically distinguishable
coverage improvement, but the effect fell below the pre-registered practical threshold. With
one tuning run per arm, its reproducibility across tuning seeds remains unknown. Pass 2
specialization produced no observable recall improvement. Tuning Pass 2 on its own
remainder (C − B) and Pass 1's influence on Pass 2 through the remainder (E − D) produced no
distinguishable change.
*Caveat:* single tuning run per arm; B − A0 also differs in budget and n_estimators.

**RQ3 — Feasibility.**
*Answer:* All five RF arms satisfied every applicable hard constraint on the held-out split.
The coverage and recall floors are disabled in this configuration.
*Evidence:* tightest margin is E's Pass 2 precision (0.38 pp above the floor). For context,
the GLASS Router fell short of its NPV floor by 0.06 pp and exceeded its leakage cap by
1.05 pp on the same split. Descriptively, the re-tuned arms resolved more rows than A0
(coverage 64.7–67.0% vs 63.8%) with a similar abstain-region base rate (5.1–5.5%).
*Caveat:* feasibility is assessed at the point estimate; coverage differences have no intervals.

**RQ4 — Strength of evidence.**
*Answer:* The observed confidence intervals had half-widths of approximately 0.44–0.74
percentage points, providing useful precision for the evaluated contrasts. However, these
intervals do not establish a general minimum detectable effect. The only objective-related
shift (Pass 1) is detectable but sub-threshold, and it is not reproduced on fresh training
folds.
*Evidence:* training noise SD: Pass 1 coverage 0.57 pp, Pass 2 recall 0.17 pp, AUC 0.06 pp.
Fold optimism: coverage objective 1.71 pp; all others ≤ 0.16 pp.
*Caveat:* bootstrap intervals cover evaluation-sample uncertainty only; the search was not
repeated across seeds; the three decision contrasts are unadjusted.

**RQ5 — Mechanism (interpretive).**
The evidence is consistent with two mechanisms:
1. *Pass 1 coverage may be a noise-sensitive selection criterion.* Across 40 random
   configurations, coverage varied over only about 3.6 noise SDs, compared with about 13
   for AUC, and the coverage-tuned configuration lost most of its training advantage on
   fresh folds. These findings are consistent with noise-sensitive hyperparameter selection
   under the coverage objective, although repeated tuning runs would be necessary to
   establish how much of the observed variability comes from the search procedure.
2. *Pass 2 decisions near the threshold are insensitive to the objective.* The specialized
   Pass 2 forests rank the remainder differently from Pass 1 overall (Spearman 0.87–0.95, vs
   0.97 for A0), yet overlap with Pass 1 at the operating threshold stays at 90–97%, and C and
   D made the same held-out decisions on all subscribers. Global ranking differences did not
   translate into different routing at the cut.

Gate 0: Pass 2 non-degenerate (precision floor 11.8 pp above the remainder base rate);
Pass 1 bound by the leakage caps. Gate 1: limited observed headroom on all three questions
(regret 1.1–1.8 noise SDs; Spearman between AUC and the pass objective 0.54–0.84 across
the probe).

**RQ6 — Design decision.**
*RQ6a:* No decision contrast met the frozen rule. Pass 1: distinguishable but below threshold.
Pass 2: no difference. Population: not distinguishable. All arms were eligible.
*RQ6b:* Not reached. Descriptively, the re-tuned arms routed more customers than A0 but lost
more subscribers (117–120 vs 106) and had lower resolved F1 (0.421–0.435 vs 0.436): a
coverage-for-quality trade-off observed among the re-tuned configurations. The current
experiment does not fully separate the contributions of the tuning procedure, hyperparameter
selection, and objective specialization to this overall trade-off.
*Decision:* retain A0 with ROC-AUC selection for both passes. Stage 4 is unchanged and no
nested-tuning follow-up is required. Re-tuning the AUC baseline itself is noted as future work.
