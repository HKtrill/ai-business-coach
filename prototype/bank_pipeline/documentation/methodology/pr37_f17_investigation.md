# PR 37 — Commit 3, Step 3A: F17 Pass 2 Training Population Investigation

| | |
|---|---|
| Author | Phillip Harris |
| Record | F17 investigation (supporting evidence) |
| PR / step | PR 37, Commit 3, Step 3A |
| Branch | `pr-37-router-baseline-contract` |
| Investigated at | `e5c437c` (read-only) |
| Date | 2026-11-04 |
| Status | Reviewed. The decisions it led to are recorded in [`pr37_f17_decision_record.md`](pr37_f17_decision_record.md) (F17, approved in Step 3B) and [`pr37_f31_disposition_record.md`](pr37_f31_disposition_record.md) (F31, approved in Step 3C) |
| Line numbers | At `e5c437c`, unless a file exists only in Commit 3 |

> **Reading note.** This is the Step 3A investigation as reviewed, preserved as
> evidence. Three later developments are not repeated in its body:
>
> - The Commit 3 safeguard patch it describes as "prepared" is now applied and
>   validated locally.
> - Step 3B approved activating both opt-in pathways, and added the
>   feature-engineering limitation L1, with mandatory gate G13.
> - Step 3C retained the RF confidence construction (F31 Disposition A).
>
> Step 3D corrected two statements, and each correction is marked:
>
> - **The status line at the end of section 6.**
> - **The timing of the `match_glass_oof` removal**, in the summary (item 5)
>   and in section 3 (item 5). Commit 3 supplies the infrastructure,
>   Commit 4 freezes the values, and PR 38 activates them.
>
> Sections 6 and 8 were reworded to match. V5 in section 7 carries the
> correction already made in Commit 3. Where this record and the decision
> records differ, the decision records govern.

## Summary

**Recommendation for Step 3B review:** adopt `oof_remainder` for both routers as the PR 38 primary policy (Option B). Run GLASS `full` as a single-factor secondary sensitivity analysis under the same frozen Commit 4 contract.

The main findings behind it:

1. **The asymmetry is a configuration choice, not a missing capability.** GLASS already implements `oof_remainder` (`glass_router_pipeline.py:217–305`), using the same construction as the RF Router. The historical runs used `full` only because the GLASS notebook's active profile is `"legacy"`.
2. **GLASS `full` has no selection leakage, because it selects nothing.** Its defect is misalignment: Pass 2 rules are gated, scored and ILP-selected on every training row, but they only fire on rows Pass 1 leaves. The Pass 2 ILP counts positives that Pass 1 claims first at prediction time.
3. **`full` cannot be the common policy.** With the historical `RF_PASS2_PARAMS = None`, an RF Pass 2 forest fitted on the full population is the same forest as Pass 1. The two-pass RF Router would then collapse into the existing single-RF ablation.
4. **Both `oof_remainder` implementations decide a row's remainder membership out of fold, and stay inside the outer-training partition.** Each has one weak, documented global dependence. For RF it is the scalar `t1`, solved on all outer-training labels. For GLASS it is the outer-fold RF's feature ordering, which affects only tie-breaking. The Commit 3 patch adds opt-in corrections for both.
5. **A separate confirmed outer-fold breach.** `match_glass_oof` derives the RF constraints from GLASS's pooled OOF over all 32,950 rows. Through four scalars, RF outer fold k's `t1`, `t2` and Pass 2 training population therefore depend on fold k's own labels. *Corrected in Step 3D:* removing it takes three steps. Commit 3 supplies the independent-contract infrastructure (`SharedOperatingContract`; the independent pathway refuses `match_glass_oof`). Commit 4 freezes the actual shared constraint values. PR 38 activates them in place of `match_glass_oof`. It is not a reason to keep or change the F17 policy.

**Evidence base.** Source tracing at `e5c437c`; the frozen baseline's tables (read only); 42 existing synthetic tests run in a scratch clone; and a scratch probe of the real GLASS pipeline in both modes on synthetic data. No historical model artifacts were available or inferred, and no test-set information was used. Quantities that need local artifacts are listed in section 7 as read-only diagnostics.

## 1. Implementation evidence

The two routers differ in what Pass 2 is fitted on, how it is selected, and how its confidence is measured. Both give Pass 1 precedence at decision time. Paths below are relative to `prototype/bank_pipeline/`; line numbers are at `e5c437c`.

| Aspect | GLASS Router | RF Router |
| --- | --- | --- |
| Where the policy is declared | `GlassRouterConfig.pass2_population`, default `"full"` (`glass_pipeline/glass_router/core/config.py:114`); validated to `full` / `oof_remainder` (`:188–191`) | No parameter: the remainder is hard-wired (`blackbox_pipeline/models/rf_router/router/two_pass.py:115`); declared as a string in `crossfit.py:165` |
| Historical value | `"full"` via `CONFIG_PROFILE = "legacy"` (`glass_cascade.ipynb`, Cell 13); the same `config` object drives the full-train fit and the cross-fit (Cell 14). Pinned by `tests/parity/test_router_operating_contract.py:149–159` | `"OOF Pass 1 remainder (RF router design)"` |
| Inner OOF folds | Only in `oof_remainder`: `pass1_oof_folds = 5`, `StratifiedKFold(shuffle=True, random_state=42)` over the outer-training rows (`glass_router_pipeline.py:245–248`) | `n_oof_folds = N_STAGE_FOLDS = 5` (blackbox notebook Cell 20); the same splitter (`oof/folds.py:32–33`) |
| Pass 2 fitting rows | `full`: every outer-training row (`glass_router_pipeline.py:202–214`). `oof_remainder`: rows not routed by their inner-OOF Pass 1 rules (`:243–277`) | Rows whose inner-OOF Pass 1 probability is `>= t1` (`two_pass.py:115`); the Pass 2 forest is refitted on exactly those rows (`:150–154`) |
| What selects Pass 2 | Candidate rules generated and evaluated in-sample on the fitting rows (`X_val` defaults to `X_train`, `:177–178`), then ILP-selected on the same rows | `t2` solved on inner-OOF Pass 2 scores within the remainder (`two_pass.py:121–135`; `oof/scorer.py:57–147`) |
| Pass 2 recall denominator | Global (all training positives) in both modes; `oof_remainder` rescales recall and coverage (`glass_router_pipeline.py:271–284`) | Global: `recall_denominator="global"` with `n_positives_global` (`threshhold/solver.py:178–186`) |
| Pass 2 confidence | The first matching rule's `precision`, measured in-sample on the fitting population (`glass_router_pipeline.py:381`) | The band's OOF precision on the remainder (`bands.py:108–116`; `router/base.py:430–443`) |
| Sample weights | None; rules count rows | `class_weight` (`"balanced"` by default) plus `minority_weight`, recomputed on each population fitted, so the Pass 2 forest rebalances to remainder prevalence (`forest.py:68`, `:138–139`) |
| Decision-time eligibility | Pass 2 rules claim only rows still `"uncertain"` after every Pass 1 rule (`glass_router_pipeline.py:360–382`) | Pass 2 scored only where the refitted Pass 1 gives `p1 >= t1` (`two_pass.py:268–281`; `base.py:58–100`) |

### GLASS: what `full` actually does

- `_fit_full` (`glass_router_pipeline.py:202–214`) runs one candidate generation, one evaluation and one `select_rules` call over the outer-training rows, for both passes.
- Inside `select_rules`, Pass 1 and Pass 2 are optimised independently, on the same `y_val` / `X_val` (`ilp_rule_selector.py:337–358`). Pass 2's ILP objective is the union of positives covered (`ilp_rule_selector/utils.py:71–100`). Its per-rule gates use precision and recall measured over every matched row (`quality_gate_filter.py:104–139`).
- No step removes the rows the selected Pass 1 rules will claim. At prediction, those rows never reach Pass 2 (`:360–382`).

### GLASS: what `oof_remainder` does (implemented, not used historically)

1. Pass 1 rules are selected on all outer-training rows (`:236–239`). These are the deployed Pass 1 rules.
2. For each of 5 inner folds, a fresh generator, evaluator and selector fit Pass 1 only, on the other inner folds, then route the held-out inner fold (`:243–257`).
3. The remainder is the rows their inner-OOF Pass 1 did not route. It is refused below 500 rows or if it is single-class (`:260–269`).
4. Pass 2 candidates are generated, evaluated and ILP-selected on the remainder only. Recall and coverage are rescaled to global denominators before the gates (`:271–289`).
5. `fit_report_` records remainder rows, positives, base rate, inner Pass 1 rule counts and the mask (`:291–305`). `training_base_rate` is still the full outer-training rate, set before the branch (`:180–182`).

### Where the shared RF enters GLASS

- In the cross-fit, each outer fold's GLASS fit receives that fold's tuned RF classifier (`glass_rf_model="fold_rf"`, `pipeline/crossfit.py:248–253`). In `oof_remainder` the same object is passed to every inner-fold generator (`glass_router_pipeline.py:114`, `:252`).
- The RF reaches GLASS only through `feature_importances_`, which sets the expansion order (`rule_generator.py:233–250`). With the default `feature_importance_threshold = 0.00` (`:95`), no feature is excluded. Depth-1 seeds ignore the RF order (`:268`), and beam pruning is a stable sort on beam score (`beam_search.py:72`). The RF order therefore decides only ties and duplicate retention.
- The evaluator is never given the RF (`glass_router_pipeline.py:205`, `:280`, `:311`). Every rule therefore gets `rf_confidence = 0.5` and `rf_alignment = 0.0` (`rule_evaluator.py:140–141`), and the ILP's RF penalty terms are constants.

### RF Router fit within one outer fold (`two_pass.py:77–196`)

1. Pass 1 inner OOF over all outer-training rows (`:100–105`).
2. `t1` = the largest cut meeting the NPV, leakage and support constraints, solved on those OOF scores with every outer-training label (`:110–112`; `solver.py:46–142`).
3. Remainder = inner-OOF `p1 >= t1`; refused below 500 rows or if single-class (`:115–116`, `:199–219`).
4. Pass 2 inner OOF, with folds built within the remainder (`:121–127`).
5. `t2` = the smallest cut meeting the precision and support constraints on the remainder OOF (`:132–135`; `solver.py:147–262`).
6. Band confidences come from those OOF arrays. Pass 1 is then refitted on all outer-training rows, and Pass 2 on the remainder (`:146–154`).

The outer loop (`blackbox_pipeline/models/rf_router/crossfit.py:114–147`) re-engineers features on the outer-training rows, fits a fresh router and routes only the held-out fold. `validate_nested` is off in the outer loop (notebook Cell 21) and on only for the full-train router (Cell 20).

## 2. Population construction trace

Inside each outer fold, neither router lets the held-out fold's labels or rows shape Pass 2 directly. The one outer-fold breach comes through the RF constraint values, not through the population policy. The table separates the four phases. "Outer-train" means the 26,360 rows of the four training folds.

| Phase | GLASS `full` (historical) | GLASS `oof_remainder` (implemented, unused) | RF Router (historical) |
| --- | --- | --- | --- |
| 1. Pass 2 fitting population | All outer-train rows; no selection | Outer-train rows not routed by Pass 1 rules fitted on the other 4 inner folds | Outer-train rows with inner-OOF `p1 >= t1` |
| What decides membership | Nothing | Inner-fold rule selection on other inner folds' labels. Expansion order comes from the outer-fold RF, which saw every outer-train row | An inner-OOF forest that never saw the row, plus `t1`, solved with every outer-train label including the row's own |
| 2. Pass 2 selection / threshold | Rule gates and ILP evaluated in-sample on all outer-train rows | Same, in-sample on the remainder | `t2` from inner-OOF Pass 2 scores within the remainder |
| Constraint values | Global config constants (`"legacy"` profile) | Same | `match_glass_oof`: GLASS's pooled OOF metrics over all 32,950 rows |
| 3. Pass 2 evaluation | Outer held-out fold, by rules fitted on outer-train only (`pipeline/crossfit.py:229–260`) | Same | Outer held-out fold, by a router fitted on outer-train only (`rf_router/crossfit.py:114–125`) |
| Confidence estimate | Rule precision, in-sample on all outer-train rows | Rule precision, in-sample on the remainder | Band precision, inner-OOF on the remainder |
| 4. Routing and abstention | Pass 1 first; Pass 2 only on rows the deployed Pass 1 rules leave; else abstain (confidence 0, P = base rate) | Same | Pass 1 first (`p1 < t1` from the refitted forest); Pass 2 only where `p1 >= t1`; else abstain |

### Boundary audit, question by question

1. **Is the Pass 1 prediction used for selection out of fold?** RF: yes. `OOFScorer` refuses any train/score overlap (`oof/scorer.py:113–117`); pinned by `test_the_remainder_is_decided_out_of_fold_not_in_sample_f17`. GLASS `oof_remainder`: yes (`glass_router_pipeline.py:248–254`); pinned by `test_inner_pass1_routing_is_out_of_fold`. GLASS `full`: not applicable.
2. **Has any Pass 1 component seen the row being selected?** RF: the inner forest has not, but `t1` has, through the row's label. GLASS `oof_remainder`: the inner rules have not, but the outer-fold RF that orders their search has.
3. **Are labels used directly or indirectly?** RF: indirectly, through the `t1` constraint optimisation (one scalar). GLASS `oof_remainder`: only the other inner folds' labels, through rule selection. There is no global scalar, so GLASS is at least as nested as RF here.
4. **Is eligibility generated entirely within the outer-training partition?** GLASS: yes, in both modes. RF: yes for scores and rows. The constraint values that set `t1` and `t2`, however, were computed from GLASS outer-OOF routes and labels on all 32,950 rows, including fold k's.
5. **Does an outer held-out fold influence its own model or training population?** RF: yes, weakly, via those four constraint scalars (blackbox notebook Cell 19). Both arms also inherit globally chosen settings. These are the RF hyperparameters (`rf_router/crossfit.py:16–27`) and the GLASS legacy profile, whose Pass 1 union cap was set by looking at OOF leakage (Cell 13: held-out folds run about 1 point higher, so the cap is set slightly below). That is the F24 class of dependence, not an F17 one.
6. **Do the two routers nest differently?** Yes, in two ways. RF chooses its cuts on inner-OOF scores, while GLASS chooses rules and measures confidence in-sample on its fitting rows. RF has a global `t1`; GLASS `oof_remainder` has no equivalent.
7. **Is the selected Pass 2 population evaluated with independent predictions?** At the outer level, yes for both: every training row's stored route comes from models that never saw it. Within each fit, only RF has independent (inner-OOF) Pass 2 estimates. GLASS's are in-sample.
8. **Is the training remainder the same as the runtime remainder?** No, for either router, and that is intended. The training remainder comes from OOF Pass 1 decisions; at runtime, the deployed Pass 1 (refitted forest, or full outer-train rules) applies. On held-out rows that is itself out of sample, so the OOF remainder is the right analogue. How closely the two agree is an empirical question (see section 7).

## 3. Methodological risks

The asymmetry's main cost is that GLASS `full` fits Pass 2 for a population it never serves. Leakage is a smaller concern. Items are separated by how much the code alone establishes.

### Confirmed from the code

1. **GLASS `full` optimises Pass 2 for the wrong population.** The Pass 2 ILP's union-positive objective and its per-rule precision and recall gates include rows that the selected Pass 1 rules claim first at prediction (`ilp_rule_selector.py:337–358`; `glass_router_pipeline.py:360–382`). Positives counted there are never realised by Pass 2.
2. **GLASS `full` stores a confidence for a population the rule does not serve.** A Pass 2 rule's `precision` is measured over every matched outer-train row, including rows Pass 1 takes. The value is then attached to the rows it actually routes (`:381`). The direction of the error is unknown: removing Pass 1 rows plausibly raises precision, while in-sample optimism lowers it.
3. **Fit-time Pass 2 precision means different things historically.** For GLASS it is P(y=1 \| segment), over all rows. For RF it is P(y=1 \| p2 > t2), over the remainder. A shared Pass 2 precision floor in Commit 4 would therefore bind on different conditional quantities at fit time, unless GLASS moves to the remainder. Outer-OOF evaluation stays comparable: both are measured on the held-out rows each router actually routes.
4. **`full` is degenerate for the RF Router.** With `RF_PASS2_PARAMS = None` (blackbox notebook), `effective_pass2_params` equals `rf_params` (`config.py:278–281`), and both trainers share `random_state` (`two_pass.py:93–94`). A Pass 2 forest fitted on the full population would therefore be the Pass 1 forest. This follows by construction; it was not executed.
5. **RF outer folds depend on their own labels through `match_glass_oof`.** See section 2, item 5. The dependence is scalar and weak, but real. *Corrected in Step 3D:* Commit 3 supplies the independent-contract infrastructure, Commit 4 freezes the shared constraint values, and PR 38 activates them in place of `match_glass_oof`; only then is the dependence removed.
6. **RF remainder membership depends on the row's own label through `t1`.** This is an inner-level selection dependency: it shapes the Pass 2 fitting population inside each outer-training partition, not the outer evaluation. It is documented (`two_pass.py:20–32`), and a nested check exists (`oof/nested.py`). That check ran only on the full-train router (`VALIDATE_NESTED = True`, Cell 20), never in the outer cross-fit. The Commit 3 patch adds an opt-in correction, `eligibility="nested_t1"`; the historical default is unchanged.
7. **Realised populations were never persisted.** GLASS cross-fit summaries hold no remainder fields (`pipeline/crossfit.py:264–277`). The per-fold `fit_report_` is discarded, and RF records only `remainder_frac` per outer fold. The frozen baseline therefore marks `fit.pass2_population_realised` as not verifiable for both arms (`tables/policy_audit.csv`).
8. **Code comments assert parity that holds only for `full`.** `two_pass.py:24–27` says GLASS carries the same order of dependence, so the arms are matched. `:270–271` says that in GLASS the remainder is never a training-time subset. Both need restating once the policy changes.
9. **The notebook's `"recommended"` profile is not a pure F17 switch.** Besides `oof_remainder`, it changes Pass 1 and Pass 2 support, rule counts, precision floors, per-rule leakage gates, the recall ceiling, the Pass 2 novelty floor and the RF penalty lambdas (Cell 13). Using it as "the `oof_remainder` run" would confound F17 with tuning.
10. **GLASS `oof_remainder` inner-fold searches are ordered by a ranker that saw their held-out rows.** With `glass_rf_model="fold_rf"`, every inner-fold Pass 1 search reuses the outer fold's RF. That RF was fitted on all of P_k, including the inner held-out rows those rules route (`glass_router_pipeline.py:114`, `:252`). With `feature_importance_threshold = 0.00`, it affects only search order: tie-breaking and which duplicate is kept. This RF belongs to GLASS, not to the black-box arm (Appendix). The Commit 3 patch adds an opt-in fold-nested ranker, `inner_feature_ranker="fold_nested_rf"`.
11. **The arms' fit-time Pass 2 estimates differ in honesty.** GLASS rule precision, used by its gates, its ILP and its stored confidence, is measured in-sample, on the rows the rules were selected on, in both modes. RF `t2` and band precision come from inner-OOF scores, from forests that did not train on those rows. Option B does not change this. It is a Commit 4 question (fold-nested selection) and an F31 dependency, not an F17 one.

### Plausible, magnitude unknown

- **Rule-budget crowding.** Under `full`, Pass 2 slots (at most 8, novelty floor 0.25) may go to rules whose positives sit mostly in Pass 1 territory. This is one candidate mechanism for the historical matched-rate shortfall (−80 TP vs matched LR, −116 vs matched EBM). It has not been demonstrated.
- **Discrete Pass 1 instability.** Inner-fold GLASS Pass 1 rule sets (2–8 rules each) can differ from the deployed set more than an RF score with one shared `t1` would. GLASS's training remainder may then be a weaker proxy for its runtime remainder.
- **In-sample optimism on a smaller population.** GLASS rule precision is in-sample in both modes. The evaluator floor is `min(120, 25) // 2 = 12` matched rows (`glass_router_pipeline.py:138`; `rule_evaluator.py`). With fewer rows, low-support rules carry more optimism.
- **Inner-fold Pass 1 infeasibility and cost.** `oof_remainder` runs 6 Pass 1 selections per outer fold instead of 1, each with a 300 s CBC limit per phase (`ilp_rule_selector.py:85`). Each inner fit sees about 21,088 rows. With `min_pass1_rules = 2` and the 0.14 union-leakage cap, an inner fold could raise `SelectorInfeasibleError` and abort the run. Historical real-data fits only ever ran on 26,360 rows.

### Needs a measurement or a controlled experiment

- Actual per-fold remainder rows, positives and prevalence for each router (section 7, D1, and a PR 38 preflight).
- The effect of the population policy on GLASS Pass 2 precision, positives captured and coverage, everything else fixed (the sensitivity analysis in section 5).
- How closely each router's training remainder agrees with its runtime remainder.
- Whether a fully nested `t1` changes the RF operating point materially (the recorded `nested_check` verdict).

### Sample support and class balance

| Quantity per outer fold | GLASS `full` | GLASS `oof_remainder` | RF Router |
| --- | --- | --- | --- |
| Pass 2 fitting rows | 26,360 (fold plan: 32,950 × 4/5) | Not measured; never run on real data | `remainder_frac` × 26,360, recorded in the local RF artifact (D1) |
| Positives | 2,970 (held-out folds 0–2 hold 742) or 2,969 (folds 3–4 hold 743); `tables/common_population.csv` | Not measured | Not persisted per outer fold; full-train `fit_report.remainder.n_positives` only |
| Prevalence | 11.26–11.27% | Not measured; higher by construction | Not persisted per fold; higher by construction |
| Fold-to-fold variation | At most 1 positive | Unknown; likely larger (discrete rule sets) | Fraction only |

The support floors are small relative to any plausible remainder. As an order-of-magnitude check only, suppose inner-OOF Pass 1 routed like GLASS's historical outer-OOF Pass 1 (34.9% of rows, 13.8% of positives). A remainder would then hold about 17,000 rows and 2,500 positives. That is not a fitting-population measurement, but it suggests positive scarcity is unlikely to bind for Pass 2 rule support (floor 12–25 rows). Pass 1 inner-fold feasibility is the likelier constraint. For RF, Option B changes nothing: it already fits Pass 2 on the remainder, rebalanced by `class_weight`.

## 4. Policy comparison

Option B is the only option under which both routers train Pass 2 for the rows Pass 2 actually decides. It is also already implemented in both code paths. The key distinction: the population defines the task, while the rule search or the forest is the architecture. Giving two learners the same task is not a structural restriction on either.

| Criterion | A: `full` for both | B: `oof_remainder` for both | C: architecture-specific (historical) |
| --- | --- | --- | --- |
| Pass 2 trained for the rows it serves | Neither | Both | RF only |
| Leakage from population selection | None: nothing is selected | One weak global dependence each (RF `t1`; GLASS RF feature ordering) | As historical |
| Effect on the RF Router | Degenerates: the Pass 2 forest equals the Pass 1 forest (section 3, item 4) | None: unchanged | None |
| Effect on GLASS | None | Pass 2 re-selected on the remainder; about 6× Pass 1 selection cost; inner-fold feasibility risk | None |
| Fit-time Pass 2 precision | Full-population for GLASS; undefined for RF without a redesign | Remainder-conditional for both (GLASS in-sample, RF inner-OOF) | Differs between arms |
| Compatible with one Commit 4 Pass 2 precision floor | Not meaningfully | Yes | Only at outer-OOF evaluation, not at fit time |
| Code readiness | Needs a new RF path | Exists in both; GLASS needs a config switch plus persistence and validation | Exists |
| What a GLASS-vs-RF difference can be attributed to | Confounded by the RF collapse | Learner and Pass 1 behaviour, under one task definition | Learner, Pass 1 behaviour and population policy together (baseline caveat D12 persists) |

### Option A: `full` for both

Its one advantage is that population construction involves no selection, so there is no selection bias. It also preserves the historical GLASS fit. Its weakness is decisive, though: the RF two-pass design exists only because Pass 2 learns on what Pass 1 leaves. Making that work under `full` would need separately tuned Pass 2 parameters, chosen on the remainder, and that reintroduces a remainder anyway. **Rejected as primary.**

### Option B: `oof_remainder` for both

B is supported by the current code, but its implementation readiness is not verified. With the real rule generator and ILP on synthetic data, three of five GLASS `oof_remainder` outer-fold fits completed without error before the probe was stopped. The five-fold run was not observed to completion, and the path has never run on the real data (Appendix). Specific properties:

- **Inner folds can be identical across arms.** Both use `StratifiedKFold(5, shuffle=True, random_state=42)`, split on the outer-training labels only (`glass_router_pipeline.py:245–248`; `oof/folds.py:32–40`). Both cross-fits slice the same outer-fold rows in the same order, so they should produce the same inner partitions. PR 38 should assert this rather than assume it.
- **No new leakage.** GLASS inner-fold rules never see the inner held-out rows. The residual dependences (RF `t1`, the GLASS fold RF ordering) are of the same weak, global kind. Both can be measured, and the Commit 3 patch adds opt-in pathways that remove them.
- **Costs.** About 6× GLASS Pass 1 selection time, and a real chance an inner fold is infeasible under the legacy Pass 1 constraints. PR 38 needs a preflight before the full reruns (section 6).

The requirements are listed in sections 6 and 7.

### Option C: architecture-specific populations

C would be defensible only if `full` were an intended GLASS design property: for example, if rules were meant to be valid across the whole population and Pass 1 overlap were deliberate. The code shows otherwise. `full` is labelled the "original behaviour" (`config.py:104–114`), `oof_remainder` was added later to mirror the RF construction and sits in the "recommended" profile, and `full` trains Pass 2 on rows it is barred from deciding. Keeping C as primary would carry a known misspecification into the frozen comparison. **Rejected as primary, retained as the secondary sensitivity analysis** (GLASS `full`), because it is the historical GLASS behaviour and links PR 38 back to the frozen baseline.

### Variants considered and rejected

- **In-sample remainder.** Pass 2 fitted on rows not routed by the deployed Pass 1, fitted on those same rows. A row's own label would then decide its Pass 2 membership, which is the exact failure `oof_remainder` exists to prevent (`two_pass.py:14–18`).
- **One shared remainder for both routers.** Each Pass 2 must learn on its own Pass 1's leftovers. A shared remainder would train one router's Pass 2 on rows its own Pass 1 routes. That would be an inappropriate structural restriction.

## 5. F17 recommendation

**Primary policy for PR 38: one eligibility rule, `oof_remainder`, applied to both routers.** GLASS changes only `pass2_population` (and its three companion settings). The RF Router is unchanged. The primary is fixed before any PR 38 result is seen, whatever the sensitivity analysis later shows.

### Readiness status: proposed policy versus verified implementation

| Item | Status |
| --- | --- |
| Option B as the PR 38 primary | Proposed; decided in Step 3B |
| GLASS `oof_remainder` code path | Implemented. Synthetic only: 3 of 5 outer-fold fits completed with the real generator and ILP. Never run on real data |
| RF `oof_remainder` (`global_t1`) | The historical path; unchanged |
| RF `eligibility="nested_t1"` | Commit 3 patch: opt-in, synthetic tests only |
| GLASS `inner_feature_ranker="fold_nested_rf"` | Commit 3 patch: opt-in, synthetic tests only |
| Independent contract support | Commit 3 patch: validation and RF adapter only; values in Commit 4 |
| Realised-population records | Commit 3 patch: cross-fit summaries and protocol. Artifact persistence through ModelSaver is PR 38 |
| Inner-fold feasibility and cost on real data | Unverified; PR 38 preflight (V9) |

### Eligibility, as the decision record should state it

Within each outer-training partition P_k (outer folds `8aea082fcaf371b7`):

1. Split P_k into 5 stratified inner folds with `StratifiedKFold(5, shuffle=True, random_state=42)`, on P_k's labels in P_k's row order.
2. For each inner fold j, fit the router's Pass 1 on P_k minus j, and route the rows of j.
3. The remainder R_k is the rows of P_k that their inner-OOF Pass 1 decision did not route. RF: inner-OOF `p1 >= t1`, with `t1` solved on P_k's inner-OOF scores. GLASS: matched by no inner-fold Pass 1 rule.
4. Refuse R_k if it has fewer than 500 rows or a single class. Never fall back silently to another population.
5. Fit and select Pass 2 on R_k only. Recall denominators are global (all positives in P_k).
6. Refit the deployed Pass 1 on all of P_k. At prediction, Pass 2 sees only rows the deployed Pass 1 leaves.

### Settings that must be identical across arms

| Setting | Value | GLASS field | RF field |
| --- | --- | --- | --- |
| Outer fold plan | `8aea082fcaf371b7` | shared `STAGE_FOLDS` | shared `STAGE_FOLDS` |
| Inner folds | 5, stratified, shuffled, seed 42 | `pass1_oof_folds`, `random_state` | `n_oof_folds`, `random_state`, `stratify_oof` |
| Remainder floor | 500 rows, two classes | `remainder_min_size` | `remainder_min_size` |
| Recall denominator | Global | built in (rescaling) | `recall_denominator="global"` |
| Labels used | P_k only | — | — |

### Structural differences that stay, and why

- **How Pass 1 defines "routed".** For GLASS it is a discrete rule set; for RF, a score below `t1`. That is the architecture under comparison.
- **How Pass 2 is selected and how confidence is estimated.** GLASS works in-sample on R_k; RF uses inner OOF within R_k. This is a selection-honesty question for Commit 4, not a population one, and it should be decided there.
- **Accepted residual dependences.** Historically these were recorded, not removed. The Commit 3 patch adds opt-in removals for the first two; whether the PR 38 primary activates them is a Step 3B decision: RF `t1` (via `nested_check`), the GLASS fold RF ordering (feature-order stability), and the global hyperparameters and config (F24).

### Secondary sensitivity analysis

**S1, required: GLASS `full` versus GLASS `oof_remainder`, with exactly one factor changed.** Everything else stays identical: the corrected selector, the Commit 4 contract, folds, seeds, feature engineering, and the RF given to GLASS. Report it descriptively against the primary and against the frozen baseline. Do not use it to re-select the primary.

What S1 should report, per outer fold and pooled:

- Pass 2 rows, TP, FP and precision
- Unique positives versus the LR/EBM union
- Matched-rate comparisons (Step 2C method)
- The fit-time remainder size
- An invariance check that the deployed Pass 1 rules are identical in both runs

That last check should hold by construction. Pass 1 candidates and selection never read Pass 2 rules: deduplication keys on `(predicted_class, segment)` (`rule_deduplicator.py:28`; `beam_search.py:35–50`), and the ILP splits candidates by class first (`ilp_rule_selector.py:328–358`). Any Pass 1 difference would mean a defect.

**No RF sensitivity run is needed for the population.** RF under fully shared Pass 1/Pass 2 parameters and `full` is, in effect, the existing single-RF ablation (Experiment 1, blackbox Cell 20).

### If B is infeasible on the real data

An inner-fold `SelectorInfeasibleError` or a refused remainder is a contract-feasibility finding. It is not a reason to revert to `full`. Any constraint relaxation must be decided uniformly and before results are produced, under Commit 4's "no feasible candidate" rule.

## 6. PR 38 implementation implications

Option B needs no new algorithm. It needs one configuration change for GLASS, persistence of each fold's realised population for both arms, and the Commit 4 contract in place before any rerun. Nothing below is implemented in Commit 3.

### GLASS Router

1. **Switch only the population.** Set `pass2_population="oof_remainder"`, `pass1_oof_folds=5`, `random_state=42`, `remainder_min_size=500` on whatever profile PR 38 freezes, as a named single-factor variant. Do not adopt the `"recommended"` profile wholesale (section 3, item 9).
2. **Persist each outer fold's fit report.** `crossfit_stage2` currently discards it. Add to the fold summaries and the artifact: remainder rows, positives and base rate, inner Pass 1 rule counts, `recall_scale` and `coverage_scale`. Add the remainder as row keys or a fold-local mask. That makes `fit.pass2_population_realised` verifiable.
3. **Persist each outer fold's deployed rule sets.** That means Pass 1 and Pass 2 segments and precisions. This also closes the baseline's not-verifiable `decision.pass2_condition` and `decision.abstain_not_pass2` claims.
4. **Record inner-fold selector outcomes.** Keep each inner fold's `selection_report_` (source, lexicographic guarantee, infeasibility), and fail the run loudly on infeasibility.
5. **Keep `glass_rf_model="fold_rf"`** as the GLASS-owned ranker for the full-population Pass 1 and the Pass 2 search. Whether inner fits use the Commit 3 opt-in fold-nested ranker in the primary run is a Step 3B decision; V6 measures the difference.
6. **Budget compute.** That is 6 Pass 1 selections per outer fold. Run a single-fold preflight before the 5-fold run.

### RF Router

1. **No population change.**
2. **Extend `oof_fold_summary`.** Add each outer fold's `fit_report_.remainder` fields (`n_rows`, `n_positives`, `base_rate`), not only `remainder_frac`.
3. **Replace `match_glass_oof` with the frozen Commit 4 constants,** through the Commit 3 contract pathway. Activating them removes the outer-fold label dependence. It also stops a GLASS F17 change from moving the RF remainder through derived constraints, so Commit 4 must land before any PR 38 rerun.
4. **Optionally record `nested_check` per outer fold.** It currently runs only for the full-train router.

### Shared and documentation

- **Restate the parity claims** in `two_pass.py:20–32` and `:270–271`.
- **Update the decision pins deliberately.** `test_the_glass_notebook_runs_the_full_population_profile_for_both_fits` (`tests/parity/test_router_operating_contract.py:149–159`) and the GLASS `full` expectation in `tests/support/artifact_audit.py:87` should change in the same PR that switches the profile and regenerates Stage 2.
- **Leave Stage 4 alone.** It already accepts GLASS `oof_remainder` and requires the cross-fit and full-train declarations to agree (`shared/stage4/adapters.py:239–293`).
- **Never touch the frozen baseline.** `pr37_stage2_pre_correction_v1` stays as it is. PR 38 comparisons cite it and add new versioned outputs.

### Candidates for Commit 3 regression protection (non-behavioural, for Step 3D)

- A pinned policy declaration: historical = GLASS `full` / RF `oof_remainder`; PR 38 target = `oof_remainder` for both. Asserted against the baseline manifest and the decision record.
- A synthetic test that both arms' inner-fold constructors produce identical partitions for the same P_k (row keys compared).
- A synthetic test, using the real generator and ILP, that switching `pass2_population` leaves the deployed GLASS Pass 1 rules identical. It may need the `slow` marker.

Status, corrected in Step 3D: the Step 3A draft said all three were implemented in Commit 3. That overstated it.

- **The second is implemented:** `test_glass_inner_folds_are_the_rf_routers_inner_folds_for_the_same_rows`.
- **The first is covered for the historical side only.** The tests pin `full` in the GLASS notebook (`test_the_glass_notebook_runs_the_full_population_profile_for_both_fits`), and pin the `outer_rf` and `global_t1` defaults. The PR 38 target lives in the decision record and is checked by gate G1.
- **The third has no dedicated test.** It holds by construction and is checked on real data by gate G6. A meaningful synthetic version needs the real generator and the CBC ILP, so it stays a PR 38 `slow` test (V3).

See the coverage audit in [`pr37_commit3_handoff.md`](pr37_commit3_handoff.md).

## 7. Validation requirements

Two local diagnostics can run now and are read-only. Everything else is a test or measurement attached to PR 38. Neither diagnostic fits a model or writes an artifact, and neither may read any field whose name carries a `FORBIDDEN_FIELD_TOKENS` token (`shared/stage2/diagnostics/boundary.py:102`).

### Local read-only diagnostics (available now)

| ID | Reads | Answers | Notes |
| --- | --- | --- | --- |
| D1 | The RF two-pass artifact's metadata sidecar (`rf_router_two_pass_metadata.json`): `extra.oof_fold_summary[k]` (`remainder_frac`, `t1`, `t2`); `fit_report.remainder`; `fit_report.nested_check`; `fit_report.bands`; `config.n_oof_folds`, `config.remainder_min_size`, `config.pass2_rf_params` | The RF Pass 2 fitting rows per outer fold (`remainder_frac` × 26,360, which should be an integer); full-train remainder positives and prevalence; the nested-`t1` verdict; confirmation that `pass2_rf_params` is `None` (section 3, item 4) | Select keys explicitly: the sidecar also holds `test_metrics` and `extra.test_*`. Check the file's hash against the 37-entry snapshot first. No unpickling needed |
| D2 | The GLASS artifact: `fit_report`, `config_full.pass2_population`, `oof_protocol.glass_pass2_population`, `oof_protocol.glass_rf_model` | Confirms the historical `full` / `full` / `fold_rf` declarations | Largely re-asserts the Commit 2 boundary checks; a one-line confirmation |

The per-outer-fold RF remainder positives cannot be recovered from any persisted field. The same goes for any GLASS remainder quantity, since GLASS never ran `oof_remainder` on real data. Those need PR 38 runs.

One further diagnostic is possible but in-sample by design, so it is the maintainer's call. Apply the full-train GLASS rules to the training rows they were selected on, and count how many Pass 2 union positives fall inside Pass 1 territory, rule by rule. That would size the misalignment in section 3, item 1, as fit-time accounting only. It needs the full-train engineered training features. The boundary forbids any `glass_router_data` path, so those would have to be re-derived from training rows only. The result says nothing about out-of-fold performance.

### Tests and measurements for PR 38

| ID | Check | Kind |
| --- | --- | --- |
| V1 | For the same P_k, both arms build identical inner-fold partitions (row keys) | Synthetic, fast |
| V2 | GLASS `oof_remainder` runs end to end with the real generator and ILP inside `crossfit_stage2`; each outer fold's remainder is a subset of P_k and disjoint from the held-out fold | Synthetic, `slow` |
| V3 | Switching `pass2_population` leaves the deployed GLASS Pass 1 rules identical | Synthetic, `slow`; also checked on the real S1 run |
| V4 | Artifacts carry each outer fold's remainder rows, positives and base rate for both arms. A router declaring `oof_remainder` without them is refused | Synthetic contract test |
| V5 | Corrected: changing outer fold k's labels must not change the model that evaluates fold k, so fold k's stored outputs stay fixed. Other folds may change, because fold k's rows are in their training partitions. Pinned for RF by `test_crossfit_outputs_ignore_a_held_out_folds_own_labels`, for GLASS by `test_fold_labels_never_reach_their_own_oof_outputs`, and for the nested-eligibility pathway by `test_outer_fold_k_ignores_its_own_labels_under_the_nested_rule` (Commit 3 patch) | Synthetic |
| V6 | Fold-RF ordering dependence: per outer fold, compare the outer RF's feature order with inner-fold RFs fitted on the inner-train rows; report the rank agreement and whether the inner remainder masks change | Real, one fold suffices |
| V7 | Training-vs-runtime remainder agreement per fold: inner-OOF routing versus deployed Pass 1 routing on P_k. A structural-stability measure, in-sample by definition | Real, recorded |
| V8 | RF `nested_check` verdict recorded for the cross-fit (or per outer fold) | Real, recorded |
| V9 | Single-fold preflight: inner Pass 1 feasibility, solve status and wall time before the 5-fold run | Real, gate |

In addition, the standing PR 37 checks stay in force throughout: 37/37 artifact hashes unchanged, and `stage2_baseline.py --verify` and `--check` passing on `pr37_stage2_pre_correction_v1`.

## 8. F31 and Commit 4 dependencies

Option B leaves the RF Router's confidence and probability construction unchanged. On the GLASS side it changes what a stored Pass 2 confidence means: from full-population precision to remainder precision. That GLASS change is the item Step 3C has to rule on. This section identifies dependencies only; it does not conduct the F31 review.

| Dependency | Current behaviour | Changed by Option B? | For Step 3C to examine |
| --- | --- | --- | --- |
| Probability transform | Pass 1: b(1 − conf); Pass 2: b + (1 − b)·conf; abstain: b, where b is the producing fit's full training base rate (`glass_router_pipeline.py:390–418`; `router/base.py:348–368`) | No. GLASS sets b before the population branch (`:180–182`) | Confirm only |
| GLASS Pass 2 confidence | Rule precision, in-sample over all matched outer-train rows | **Yes**: it becomes in-sample precision on the remainder, the same conditional quantity as RF's band | Whether the GLASS arbiter's router evidence (F25/F29) keeps its meaning. The exact read-back `rule_positive_rate` is unaffected (`meta_ebm/router_evidence.py:73–88`, `:102–125`) |
| RF band confidence | Inner-OOF band precision: Pass 1 NPV over the full population, Pass 2 precision over the remainder (`bands.py:108–116`) | No. Values will move with the Commit 4 constants; the construction stays | Whether a constants-only change counts as "construction changes" for F31 |
| Lift semantics | `directional_confidence` for Pass 2 is r − b, against the global prior (`router_evidence.py:91–99`). Part of the lift is prevalence enrichment from Pass 1 removal | Aligns GLASS with RF's existing semantics | Whether lift against the global prior is the intended Stage 4 signal |
| Estimation honesty | GLASS confidence is in-sample; RF's is inner-OOF | No, but optimism may grow on GLASS's smaller population | Whether this belongs to F31 or to Commit 4 |
| Calibration and score scale | Neither output is a calibrated probability (`router/base.py:323–326`). RF Pass 2 scores come from a class-balanced remainder forest and are used only through `t2` | No | Confirm only |
| Abstention | Confidence 0, probability = b | No | Confirm only |
| Stage 4 interface | Accepts `full` and `oof_remainder`; train-side and test-side declarations must agree (`shared/stage4/adapters.py:239–293`) | No code change. A GLASS `oof_remainder` artifact must be regenerated on both sides together | Confirm only |
| `BinnedBandConfidence` | Unwired (`bands.py:30–36`) | No | Should stay out of scope |

### Commit 4 dependencies

- **Freeze the constants independently of both arms' realised OOF metrics.** Once PR 38 activates them, this removes the `match_glass_oof` outer-fold dependence. It also means a GLASS policy change cannot move the RF remainder.
- **Define the Pass 2 precision floor once:** on rows each router routes to Pass 2, measured on outer-OOF. Option B makes the fit-time analogue the same conditional quantity for both.
- **Rule on fit-time enforcement.** GLASS enforces its gates in-sample; RF enforces its gates on inner-OOF scores. Commit 4's fold-nested, training-only selection principle has to say whether GLASS's in-sample enforcement is acceptable, or must be checked on inner-OOF.
- **Make the "no feasible candidate" rule cover two new failure points:** inner-fold Pass 1 infeasibility and a refused remainder (section 5).
- **Keep leakage and recall on global denominators** (all positives in P_k). Both arms already do.

## Appendix: synthetic execution check, Commit 3 corrections and limits

Nothing here ran in the maintainer's checkout or on real data. Commit 3 adds opt-in corrections for the three dependencies found in Step 3A. At the time of this investigation they were packaged as a patch; that patch has since been applied and validated locally (see the reading note at the top).

### Synthetic execution check

This ran in a scratch clone of `e5c437c`.

- **Existing tests:** 42 targeted tests passed (`tests/stage2/test_glass_router_pass2_remainder.py`, `test_rf_router_oof.py`, `test_glass_router_crossfit.py`).
- **Real-pipeline probe.** The real `GlassRouterPipeline` (generator, evaluator, ILP) ran inside the real `crossfit_stage2`, on the synthetic bank fixture (1,875 training rows, 375 positives). The forest was tiny, and the CBC limit was cut to 20 s per solve, in the scratch copy only.
    - `full`: all 5 folds completed in 285 s. Rules per fold were 4+4, 4+3, 5+4, 4+4 and 4+4.
    - `oof_remainder`: 3 of the 5 outer-fold fits completed without error, in 283 s, 248 s and 314 s (about 4–6× the `full` fits), before the probe was stopped. No remainder sizes or Pass 1 invariance result were collected.
    - Discarded: an earlier 6,000-row run with the default 300 s limit hit its own 25-minute timeout, and an earlier probe stubbed out the router.

### Commit 3 corrections

| Dependency | Historical default (unchanged) | Opt-in pathway | Regression tests |
| --- | --- | --- | --- |
| RF `t1` decides the Pass 2 remainder with every outer-training label | `eligibility="global_t1"` | `eligibility="nested_t1"`: each inner fold's rows are cut by a `t1` solved without them; the deployed `t1` is unchanged. About `n_oof_folds × eligibility_folds` extra Pass 1 fits (25 at 5 × 5) | `tests/stage2/test_rf_router_nested_eligibility.py` |
| GLASS inner-fold searches ordered by the outer-fold RF | `inner_feature_ranker="outer_rf"` | `"fold_nested_rf"`: a fresh ranker per inner fold, with the fold-RF recipe, on inner-train rows only. Requires `glass_rf_model="fold_rf"`. 5 extra RF fits per outer fold | `tests/stage2/test_glass_inner_feature_ranker.py` |
| RF constraints derived from GLASS's realised OOF metrics | `match_glass_oof` (notebook Cell 19) | `SharedOperatingContract` plus `rf_constraints_from_contract`. The independent pathway refuses `match_glass_oof`. No values | `tests/stage2/test_stage2_operating_contract_support.py` |
| Realised Pass 2 populations not recorded | — | Recorded: RF `oof_fold_summary` gains remainder rows and positives, and the protocol gains `pass2_eligibility`; GLASS fold summaries and in-memory `fold_fit_reports`, and the protocol gains `glass_inner_feature_ranker`. ModelSaver persistence stays in PR 38 | Covered by the two pathway suites |
| Fitted estimator shared across arms | None found | Boundaries pinned | `tests/parity/test_stage2_cross_arm_independence.py` |

### Cross-arm trace

`glass_rf_model="fold_rf"` does not reuse the black-box arm's RF. `crossfit_stage2` fits a GLASS-owned fold RF from `rf_params` (`pipeline/crossfit.py:240–253`). The black-box arm takes only the hyperparameter dict from the GLASS RF artifact (`tuning/legacy.py:115–138`). The arms share hyperparameters and training rows, which is allowed; no fitted estimator crosses between them. Because the recipes match, GLASS's fold RF and the RF Router's refitted Pass 1 forest are likely identical by construction. That is the same computation, not shared state.

### Still unverified

1. That the router code at `e5c437c` matches what produced the October 23 artifacts (`git diff a036759 e5c437c` over both arms).
2. A complete GLASS `oof_remainder` run: 5 folds on synthetic data, and any run on real data.
3. Inner-fold Pass 1 feasibility on about 21,088 rows, and the real-data compute cost (V9).
4. That the deployed Pass 1 rules are identical across modes (by construction only; V3).
5. Both opt-in pathways on real data: they are tested on synthetic fixtures only.
6. Real-data remainder sizes for both arms. D1 gives RF rows per fold, not positives.
7. The RF collapse under Option A (by construction; D1 confirms `pass2_rf_params`).
8. The size and direction of the GLASS `full` misalignment and confidence bias.
9. The RF `nested_check` verdict in the local artifact.

### Bounded decisions handed on

- **Step 3B:** whether the PR 38 primary activates `eligibility="nested_t1"` and `inner_feature_ranker="fold_nested_rf"` together with `oof_remainder`. Activating both gives the arms the same nesting obligation, at the V9 cost. The alternative is to record the historical dependences as accepted. *(Resolved in Step 3B: both are activated; see the F17 decision record.)*
- **Commit 4:** the contract values; whether to keep a count cap (`max_pass1_leakage_count`); how set-level requirements map onto GLASS's per-rule gates, union cap and in-sample selection; the RF band-support floors; and whether RF hyperparameters should still be read from the GLASS RF artifact. Sharing hyperparameters is allowed, but that read is a cross-arm file lookup.
