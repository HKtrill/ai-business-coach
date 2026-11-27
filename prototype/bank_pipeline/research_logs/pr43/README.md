# PR 43 — Stage 4 recommendation and research handoff

**PR 45 candidate.** Carry the normalized log-odds margin formula forward on GLASS inputs. Keep the original GLASS formula and raw Meta-XGBoost as references on both upstreams. Normalization changes how far a vote lies from its cut, using the same fitted voter cuts, weights and routing gate. The secondary frozen-input comparison supports this isolated change. Regeneration and a prospective final comparison remain prerequisites for adoption.

## Accepted evidence

Paths are relative to this directory. Digest prefixes identify canonical content; each JSON has a separate file SHA-256 sidecar. The historical PR 39 2×2 crossover remains frozen.

| Evidence | Accepted run and record | Content SHA-256 prefix |
|---|---|---|
| Frozen inputs | [c2_20261125_111609/frozen_inputs](c2_20261125_111609/frozen_inputs.json) | `0bd27b22411bd6ba` |
| Policy receipt | [c2_20261125_112854/stage4_registration](c2_20261125_112854/stage4_registration.json) | `fd8a254bc199b2da` |
| Nested references | [c2_20261125_142638/nested_references](c2_20261125_142638/nested_references.json) | `85db12425eb5ddbd` |
| Voter diagnostics | [c3_20261125_163120/voter_diagnostics](c3_20261125_163120/voter_diagnostics.json) | `b5ab3779cb5f7455` |
| Corrected Linear Arbiter | [c4_20261125_181030/linear_arbiter](c4_20261125_181030/linear_arbiter.json) | `e0a464b259da2aac` |
| Secondary comparators | [c4_20261125_220626/stage4_comparators](c4_20261125_220626/stage4_comparators.json) | `509270be711b616f` |
| Frozen stability | [c5_20261125_224854/frozen_stability](c5_20261125_224854/frozen_stability.json) | `0969cbc371b1edb2` |
| Black-box holdout | [c5_20261126_104038/holdout_blackbox](c5_20261126_104038/holdout_blackbox.json) | `416ab0a324d07fc0` |
| Expansion decision | [c5_20261126_104317/simulated_decision](c5_20261126_104317/simulated_decision.json) | `0b5d0ef0a0613775` |
| GLASS holdout | [c5_20261126_184203/holdout_glass](c5_20261126_184203/holdout_glass.json) | `971f8a374ad2ae4b` |

The registration binds Step 2A to [policy v1](stage4_policy_v1.json), canonical SHA-256 `42f457df53baa0b52739f99097ae7da2d9d1bc32794351c8ef5dd8b7328051ba`. The comparators cite the corrected Linear run. Preserve `c4_20261125_174230` as the superseded execution with a black-box sigmoid optimization failure. The subsequent frozen and holdout records cite the accepted comparator, Linear, diagnostics and nested records. GLASS links through the expansion decision to the completed black-box record.

## Policy and candidate decisions

Abstention sends a contact to review at an assumed cost of 0.25 for either true class. Whole-system loss is (retained FP + 2 × retained FN + 0.25 × deferred rows) / all rows. Review costs 0.10 and 0.50 rescore locked actions. Retained F2 is descriptive. All-row scores, probability calibration, class decisions, confidence and the operating cut are distinct. The locked cut must retain at least 50% overall and within each class, with at least 20 retained per class in each outer fold.

Stage 4 shares five outer folds and three inner selection folds across candidates. Learned transformations use fitting-local predictions, including subfold calibration. Lock thresholds before scoring outer rows. The upstream ten-fold Stage 1 and five-fold Stage 2/3 producers predate this partition; their fits can include other outer evaluation labels. These are conditional random-split comparisons on 32,950 training rows.

For the four primary Linear contrasts, compare paired row losses with 4,000 fold-by-class stratified resamples, seed 4302, and Bonferroni 98.75% intervals. Secondary intervals are descriptive 95%. Improvement requires an upper bound below −0.005 and a gain exceeding the sample SD of five fold differences. Comparability requires the entire interval inside ±0.005. Require at most 0.02 pooled retention loss per class (Linear versus formula requires no drop), partial AURC increase ≤0.005, and, for probability models, ECE ≤0.05 and Brier within +0.005 of Stage 1 and any probability comparator. Formula scores rank decisions. Report effective recall as retained TP divided by all positives. The policy specifies no separate numerical effective-recall margin.

| Procedure | Disposition and reason |
|---|---|
| Original formula, recall 0.70 and blend 0.50 | Fold-refitted reference on both upstreams. Preserve the historical retained-F2 operating rule separately. |
| Normalized formula | Only additional retained candidate, on GLASS. Black-box loss also fell, but the registered safeguards rejected it there. |
| Recall 0.60 / 0.80 | Neither advances on either upstream. Lowering the target improved loss but failed class retention. Raising it worsened loss and failed safeguards. Inner-selected formula variants also failed safeguards. |
| Blend 0.25 / 0.75 | Comparable on both upstreams. Keep the reference blend. |
| Nonnegative Linear Arbiter | Reject on both upstreams. Loss versus original formula fell by 0.13816 on GLASS and 0.16288 on black-box, but absolute fold floors and pooled class retention failed. GLASS used global coefficients; black-box qualified for shrunk routing-state slopes in all five partitions. Inner selection chose the route form in one fold and global in four. The route slopes’ independent advantage over global slopes remains unresolved. |
| Raw Meta-XGBoost | Retain the historical F2-at-0.5 tuning procedure in raw operating space as a reference. The globally accepted fixed parameters are a separate descriptive reference and the fixed-parameter holdout configuration. |
| Alternative Meta tuning/calibration | On both upstreams the raw log-loss-tuned alternative failed safeguards. Fold-gated isotonic variants of both objectives reduced loss but failed retention safeguards; the inner-selected Meta alternative also failed. Both objectives ranked the same twelve proposals. |

**GLASS evidence at locked cuts.** Nested rows are pooled outer OOF actions (partial AURC is fold-weighted). Holdout rows use full-refit upstreams and, for Meta, the full Meta refit. Columns for retention and recall are percentages.

| Evaluation | Arbiter | Loss | Coverage | Positive retention | Negative retention | Effective recall | Partial AURC |
|---|---|---:|---:|---:|---:|---:|---:|
| Nested | Original formula | 0.27347 | 98.25 | 98.90 | 98.17 | 68.99 | 0.31590 |
| Holdout | Original formula | 0.27075 | 94.95 | 96.50 | 94.75 | 68.46 | 0.30911 |
| Nested | Normalized formula | 0.26730 | 98.26 | 99.00 | 98.16 | 68.48 | 0.28359 |
| Holdout | Normalized formula | 0.26597 | 95.89 | 97.04 | 95.74 | 68.46 | 0.28556 |
| Nested | Raw Meta | 0.19934 | 91.58 | 92.89 | 91.42 | 60.45 | 0.19970 |
| Holdout | Raw Meta | 0.18801 | 91.44 | 91.64 | 91.42 | 59.43 | 0.19271 |

On black-box inputs the nested reference losses were 0.29534 (original formula) and 0.18576 (raw searched Meta). Outer loss minus inner-selected loss was +0.00527 and −0.00122 respectively; on GLASS it was +0.00067 and +0.00517. These differences include sampling and fitting-set-size effects.

Nested normalization reduced loss by 0.00617 (95% paired interval −0.00723 to −0.00511; fold SD 0.00278). Coverage rose by one row; positive retention rose and negative retention fell slightly within tolerance. Effective recall fell 0.51 percentage points. On the holdout both class retentions rose and effective recall was equal. The holdout loss gain of 0.00478 is descriptive and below the registered improvement margin. Raw Meta achieved lower loss at a different retention/recall trade-off. These results support carrying an interpretable candidate into PR 45 without establishing a cascade-wide winner.

Exact risk comparisons require both arms to retain exactly ceil(coverage × N) rows with intact confidence ties. Across all comparator arms only 100% was jointly attainable in each outer fold. Normalized versus original on GLASS also attained 50% in fold 0; neither attained a joint 70% or 85% comparison in any fold. Read the recorded bracketing counts for missing contrasts. Partial AURC integrates the whole-tie minimum-coverage step curve over [0.50, 1], divided by 0.50.

## Where coverage changes

**Operating policy on fixed scores.** Switching historical black-box Meta from fold-specific class thresholds to its global deployment threshold changed 55 class decisions and 10,973 retention assignments. With the abstention cut held fixed, coverage fell from 50.15% to 20.81%. Under the registered policy a global class threshold alone changed zero assignments; changing both cuts moved coverage from 98.49% to 97.14%. Across 4,000 resamples of each of five arms, normalized GLASS had median out-of-bag coverage 98.21%. Locked cuts had zero recorded safeguard violations. With cut reselection there was one fold-floor failure and 43 relative-retention failures in 4,000 out-of-bag evaluations. These counts concern different safeguards.

**Refitting on one simulated split.** Seed 4305 assigned 26,360 rows (2,970 positives) to fit and 6,590 (742 positives) to evaluation. Fitting OOF versus holdout distributions compare different populations. Below, each full-minus-fold-average contrast pairs the same holdout rows with locked operating cuts. “Upstream” fixes the full Stage 4 model; “Meta” fixes full upstream inputs; “combined” changes both. Retention differences are percentage points.

| Upstream | Arbiter | Changed component | Δ coverage | Δ positive retention | Δ negative retention | Δ loss | Class flips | Retention flips |
|---|---|---|---:|---:|---:|---:|---:|---:|
| blackbox | glass_formula | Upstream | -1.14 | -1.48 | -1.09 | +0.02758 | 305 | 513 |
| blackbox | meta_xgb | Upstream | +0.11 | -0.54 | +0.19 | +0.00125 | 39 | 87 |
| blackbox | meta_xgb | Meta | -0.88 | -1.35 | -0.82 | -0.00083 | 0 | 58 |
| blackbox | meta_xgb | Combined | -0.58 | -1.48 | -0.46 | -0.00250 | 41 | 102 |
| glass | glass_formula | Upstream | -1.35 | -0.40 | -1.47 | +0.00353 | 178 | 235 |
| glass | logit_margin | Upstream | -0.50 | +0.00 | -0.56 | +0.00247 | 193 | 245 |
| glass | meta_xgb | Upstream | -0.35 | -0.67 | -0.31 | -0.00156 | 92 | 117 |
| glass | meta_xgb | Meta | -0.33 | -0.27 | -0.34 | +0.00023 | 32 | 22 |
| glass | meta_xgb | Combined | -0.71 | -1.08 | -0.67 | -0.00201 | 160 | 123 |

The original formula on black-box inputs showed the largest upstream-only loss shift (+0.02758; paired 95% interval +0.02242 to +0.03289). This exceeded the 0.005 expansion threshold. The unresolved independent rule evidence for retained normalized GLASS separately justified expansion. Normalized GLASS changed 245 retention assignments despite a much smaller net loss change. Combined shifts mix both mechanisms and need not equal the sum of contrasts measured at the full-fit conditions.

Stage 1/3 voter flips numbered 309/190 on black-box and 74/125 on GLASS. Across routing states, 558 black-box and 1,595 GLASS rows changed route. Full → fold-average transitions into deferral were 90 from Pass 1 and 108 from Pass 2 for black-box, and 446 and 296 for GLASS. Reverse transitions from deferral to Pass 1/2 were 264/95 and 553/232 respectively. The remaining changes switched directly between passes. These are simultaneous changes to the arbiter inputs; stage-specific attribution would require holding the other stages fixed.

## Independent symbolic rule evidence

Two Pass 2 rules crossed the activation threshold. In each applicable inner fold, the selected-rule fitting precision exceeded independent first-issued precision by ≥0.05 with ≥100 routed observations, in at least two of three folds. Both rules had zero issues in the first inner fold. The two supported folds give the following counts and weighted precision estimates.

| Rule conditions | Independent correct / issued | Fitting precision | First-issued precision |
|---|---|---|---|
| campaign_heavy=0, cci_dead_mid=0, dow_hot=1 | 45/156; 93/326 | 39.6%, 35.7% | 28.8%, 28.5% |
| behav_hot=1, cci_dead_mid=0, eci_mid=0 | 15/103; 52/223 | 32.7%, 32.7% | 14.6%, 23.3% |


Fitting precision covers matches on the applicable pass population, including the Pass 2 remainder. Independent precision counts only decisions actually issued after higher-priority overlapping rules. Selection, overlap and ordering can all contribute to the discrepancy. Transfer required each rule itself to have ≥100 independent issues in each of two inner folds, regardless of its own discrepancy. Five firing full-fit rules lacked that support. The record identifies their canonical conditions in `rule_evidence.unsupported`: three Pass 1 rules issued 865, 748 and 480 rows; two Pass 2 rules issued 576 and 850. The locked ablation was unavailable. Pooling other rules' outcomes would answer a different question.

## Cost and recovery provenance

| Simulated upstream | Stage 1 seconds | Router seconds | Stage 3 seconds | Meta seconds | Execution wall seconds |
|---|---:|---:|---:|---:|---:|
| Black-box | 252.22 | 377.45 | 173.23 | 2.43 | 808.90 |
| GLASS recovered execution | 12.96 reused | 4,945.19 | 355.59 | 4.00 | 5,309.60 |


Each architecture used 111 Stage 1 estimator fits, 31 Stage 3 fits and six Meta fits. The six black-box routers contained 222 forest fits. Stage 2 dominated both runs. GLASS symbolic selection includes nested ranking and ILP work; router elapsed time covers more than the solver alone. Nested references took 240.79 s. The recorded budget per upstream was 20 formula fits, 185 searched Meta fits and 20 fixed-parameter Meta fits. Voter diagnostics took 4.09 s. The corrected Linear run took 23.11 s, with 245 ridge and 65 sigmoid fits across GLASS outer partitions and 490 and 130 on black-box. Neither disposition qualified for coefficient bootstrapping. Comparator research reused reference streams and added nine outer tree refits and 171 calibration subfits (161.03 s); frozen stability used zero fits (155.51 s).

Original GLASS router producer 0 stopped on an undocumented feasible Pass 2 incumbent because the holdout path omitted M3. The recovery authenticated the frozen black-box decision, original binding and eleven GLASS Stage 1 receipts (ten producers and full refit). The original common identity was `a100c76b30d372e7…`, the corrected common identity `49883f78d9f7b977…`, and the corrected execution binding `b5fbf79af1970696…`. The recovery record links all three to the original traceback and Stage 1 receipts. Six accepted primary routers include the replacement for producer 0. One original failed fit plus three conditional M3 repetitions brought the attempt total to ten. Pass 2 on producers 0, 4 and the full refit was feasible, with solver status/gap documented and ordered rules and union counts reproduced. The other selections were proven optimal. The failed original's traceback cannot reconstruct its solver gap.

The GLASS wall clock covers the corrected invocation. Producer totals include historical Stage 1 timing and successful M3 work; the failed original attempt and time between runs are outside that wall clock. From the accepted producer timings, PR 45 linear scaling to 32,950 rows gives about 1,007 s for black-box and 6,647 s for GLASS. For PR 49, budget approximately 805 and 5,318 producer-seconds per comparable split before multiplying by the split count. ILP stopping behavior, hardware and altered contracts can change those estimates. They exclude new searches, archive I/O and temporal redesign.

## PR 44 upstream research

Use the [PR 42 feature-contract recommendations and evidence index](../pr42/feature_contract_recommendations.md) for candidate definitions, registered outcomes and versioning requirements. Keep closed candidates closed absent a new scientific justification. Measure Stage 1/3 hyperparameter-selection optimism on shortlisted configurations with inner selection and untouched outer evaluation.

| Stage | Evidence and unresolved question | Required evaluation and Stage 4 consequence |
|---|---|---|
| 1 — macro probabilities | PR 42 found later contacts' labels in target-derived encodings and MLP early-stopping rows participating in encoder fitting. Keep the compact macro design as a candidate. PR 43 observed Stage 1 flips after refitting on both architectures. | Compare past-only encodings and isolate preprocessing for early stopping. Measure discrimination, calibration, signed margins, full-versus-fold-average shift and unique correct votes by route. Determine whether the confidence scale still transfers. |
| 2 — router reliability | The qualified NSD `gap_placed` candidate lost its fold-0 repair and zero migration under fold-local references. `eci_warm` qualified as a Pass 2 partner-rule filter. GLASS changed routes on 1,595 holdout rows; independent precision could not transfer for five full-fit rules. | Test both reference constructions and the filter under the existing two-pass, union-aware constraints. Record matching and first-issued counts, independent positives, overlap, support and producing-model base rates in each fold. Measure route transitions, marginal correct decisions, confidence and ILP time. Register any contract revision before evaluating it. |
| 3 — deferred rows | On 14,628 frozen GLASS router deferrals, Stage 1/3 average precision was 0.168/0.214 and Brier 0.0677/0.0657. PR 42 advanced shared `jed_alone` with its fixed Step 5A interactions (two surviving pairs). Its net abstention corrections fell from 486 to 389 under fold-fitted three-pair reselection, missing the screen. | Compare EBM and XGBoost on the same candidate features. Compare the three fixed pairs, no interactions, one or two pairs, the fixed campaign pair and capped fold-fitted detection. Assess deferred-row calibration, discrimination, shape stability, stage disagreement and incremental correct decisions before deciding on a feature or interaction contract. |

## PR 45 integration decision

1. Freeze the justified PR 44 contracts and accepted parameters with versions, seeds, training identities and artifact hashes. Regenerate both upstreams under their specified fold designs. Verify ordered rows and labels, producer folds, direction and calibration of probabilities, route states and producing-model base rates. Preserve the PR 39 crossover identity separately.
2. On each regenerated upstream, refit the original formula and the raw Meta reference as common-policy controls. On GLASS add normalized margins with identical fitted voter cuts and weights. Compare normalized versus original and versus raw Meta on identical rows, inner-selected locked cuts and common outer folds. Distinguish the searched Meta reference from the fixed-parameter deployment check. Repeat exact attainable coverage comparisons and full-refit transfer checks.
3. Apply the registered loss margin, fold variability, uncertainty, support, class retention, calibration (where applicable), partial AURC and interpretability safeguards. Report effective recall and sensitivity to review cost. Give PR 45's final contrasts a prospective inferential role before examination; the four primary PR 43 contrasts belonged to Linear. Prefer a qualifying simpler model under demonstrated comparability. Inconclusive evidence leaves the references in place.
4. Recount independent rule evidence for the regenerated ordered rules. Resolve unsupported transfers or explicitly carry their scope and uncertainty into the decision. Exclude an unavailable ablation from the evidence for adoption. The rejected Linear, recall and calibrated Meta configurations have historical dispositions only.

All findings use the original training population. Historical features and parameter choices predate the simulated split. The conditional bootstrap holds fitted streams and class composition fixed; one stratified simulated holdout cannot estimate full upstream-refitting variability. PR 49 owns repeated independent splits and forward-in-time robustness. PR 43 authorizes a shortlist, with production selection reserved for PR 45.

## Verification, reproduction and Step 6B preservation

From `prototype/bank_pipeline`, these commands read the accepted records. The last three also require their original local numerical archives.

```powershell
python -m feature_research.stage4_stability --verify c5_20261125_224854
python -m feature_research.stage4_comparators --verify c4_20261125_220626
python -m feature_research.stage4_holdout verify c5_20261126_104038 --arm blackbox --archives
python -m feature_research.stage4_holdout verify c5_20261126_184203 --arm glass --archives
```

For a **fresh experiment**, the module sequence is `python -m feature_research.stage4_holdout blackbox`, then `decide <new-blackbox-run>`, then `glass <new-decision-run>` on expansion, all from one compatible code/environment identity. **Historical recovery** used `glass c5_20261126_104317 --recover-router0 --check-only` followed by the same arguments without `--check-only`, on the verified pre-commit working tree with original receipts and traceback. These are fitting/recovery procedures, not verification commands for the current branch. Earlier records carry their own reproduction commands.

The committed JSON stores solver dispositions, receipt and audit hashes. The underlying `holdout_cache/` is Git-ignored. Before publication, durably preserve and verify the original traceback and failed-attempt evidence, `glass/ilp_recovery.json`, eleven reused Stage 1 receipts and archives, and each producer's `solver_audits/router_<k>/` evidence. Accepted primary fits have attempt, primary and decision JSON; producers 0, 4 and −1 also have the M3 start and repeated selection reports. Preserve the score archives and receipts needed by both holdout verifiers and the nested/comparator prediction stores needed for replay. Index checksums and prove the archived files verify after restoration. A tracked hash alone cannot restore a missing audit.

Step 6B owns comprehensive regression and frozen-baseline verification, clean-kernel execution, and executed-notebook archiving and release checks. Keep the tracked notebook output-free. Use the accepted run IDs above for the record-reading cells; the GLASS notebook input is the recovered result, and its decision input is the frozen expansion decision.

## Step 6B historical compatibility and warning disposition

The Windows regression pass reported targeted 582 passed / 3 failed, default 3,886 / 4, slow 11 / 0 and comprehensive real-data 144 / 0. The four distinct failures compared source identities. These LF-normalized SHA-256 values distinguish accepted historical evidence from the two reviewed source transitions.

| Identity | Prior | PR 43 |
|---|---|---|
| Stage 3 block module | `5ce9e669838c5692a53cad6c74e3f7ed207a16f45448ebbef963d156d2b04c16` | `522340ef58cb10705752db6212398315e86fe03d4df58f7d74cf6987ba48d743` |
| Combined Stage 3 DAG (same eleven modules) | `3a8d195700ce34627f7d10d59294212ba967bcd69fe14a1c5741e87e25720017` | `a38638e9499a40d59db3bbe5386e0c0ac3ed65581948680a31e4a5c26ff9a83c` |
| Stage 2 diagnostic boundary | PR 37 manifest `0c653b69924b810e731cb3d212adb79fb12d9f13d17039d06f0f437b2448c724`; PR 39 exception `a8340178ce641fb38b951b9889fe640c6b1501595f9cbe533788398726ca4ce8` | `eb40c7507ea55e861748dd75425140120179b0f9e0196c9c272efea12a922e66` |

Commit `cd6a22d` added one default-true keyword and conditioned only the reporting-frame construction on it. Production still calls the same fold builder with the same arguments. The holdout explicitly skips those unused frames. The PR 42 record and generated inventory retain the historical identity; their tests compare only the reviewed successor field and still verify the module list, other fields and full generated block. Commit `fa690cd` added MLP and XGBoost entry points to the existing Stage 1/3 training-side checks, retaining the LR feature contracts, EBM family and fold/score checks, and router extractors. The Stage 2 manifest stays frozen; its existing deliberate-change check pins the exact successor.

**Seven real-data convergence warnings.** Windows reported scikit-learn `_sag.py:348` under `test_availability_is_the_reported_one`. The [PR 42 closeout](../pr42/README.md#documented-warnings-and-deferred-engineering) records seven at the same test. The temporal evaluator and test date to `1cd13b9`, after PR 40 warning cleanup. This is a recurrence with unresolved numerical impact.

The first test requesting the module-scoped `lr_run` fixture runs `temporal_validity.evaluate(..., mlp=False)` on the training rows. Its research LR uses the PR 39 settings: scikit-learn LogisticRegression, SAGA, elastic net, C=0.001756869502269086, l1_ratio=0.030576263600358287, balanced classes, seed 42, max_iter=5000 and sklearn's default tol=0.0001. The three inputs are cellular_downturn and the two target-derived macro rates. Random folds and chronological windows scale from their applicable fitting rows; past-only and lag-consistent arms restrict rows by availability. The fixture fits LR repeatedly across accepted and past-only encodings and label-flip controls. Pytest attributes the warnings from fixture setup to this test. Seven emissions establish that the iteration limit was reached repeatedly; the summary lacks fold/arm tags, iteration traces and coefficient deltas. It cannot identify seven particular partitions or establish the same fits warned in both PRs.

Availability counts are computed from encoding history. The reported real-data checks, including LR reproduction and temporal comparisons, passed at their specified tolerances. Coefficients and probabilities at the cap may still differ from a converged solution; ranking checks cannot quantify that difference or certify cross-platform reproducibility. This path uses the historical split default, Stage 1 feature designs and research LR; it does not call the PR 43 Stage 4 solvers, training-only MLP/XGB extractors or Stage 3 reporting switch. There is no demonstrated PR 43 cause or measured environment effect. PR 44 should capture arm, fold/window, row support, scaler statistics, package version and n_iter_ per fit before comparing coefficient, probability, calibration and temporal-contrast sensitivity under a separately justified convergence diagnostic. Preserve the seven warnings and accepted settings throughout PR 43.

Test edits change the whole-tree code identity used for new executions. Historical records and their prediction, checkpoint, solver-audit and recovery digests remain tied to the recorded trees. Verify accepted archives with the read-only commands above. A new notebook execution records its own code provenance; historical recovery authorization applies only to the reviewed recovery tree.

## Final validation and reproducibility

### Regression validation

| Suite | Passed | Failed | Runtime |
|---|---:|---:|---:|
| Focused fingerprint correction | 124 | 0 | 9.67s |
| Corrected targeted | 586 | 0 | 363.73s |
| Corrected default | 3891 | 0 | 1540.07s |
| Slow | 11 | 0 | 1737.48s |
| Comprehensive real-data | 144 | 0 | 863.52s |

All four historical fingerprint failures were resolved. The slow and
real-data results precede the test-and-documentation correction.
The seven recurring SAGA convergence warnings remain deferred to PR 44.
The post-registration packaging test rerun was interrupted after 41
passing tests; both final archives independently passed verification.

### Baseline and evidence verification

- All eight frozen-baseline verification commands passed.
- PR 39 frozen pins and crossover identity verified.
- All seven accepted PR 42 contract references verified.
- All 40 PR 42 research records remain intact.
- Accepted PR 43 research chain, prediction archives, checkpoint
  receipts and GLASS solver recovery evidence verified.

### Accepted Training Research Notebook

Final execution: `c2_20261126_231759`.

| Record | Reproduced run | Content digest |
|---|---|---|
| Frozen inputs | c2_20261126_231336 | Identical |
| Stage 4 registration | c2_20261126_231338 | Identical |
| Nested references | c2_20261126_231759 | Identical |

The final clean Windows kernel execution completed in 274.12 seconds.
All 11 code cells executed successfully, preserving 89 outputs,
41 tables and 24 plots, with zero execution errors.

All three scientific content digests reproduced exactly. The nested
prediction archive passed verification. Inline plotting initialization
and a temporary delay preventing a same-second run-ID collision are
documented in the executed notebook metadata. Tracked cell sources
remain unchanged.

The earlier reproduction without embedded plots and the interrupted
timestamp-collision attempt are preserved in the external evidence
archive. Their redundant run directories were removed from the
working tree before the final commit.

### Evidence preservation

The external archive
`pr43_research_evidence_final_20261127_103006.zip`
contains 192 verified files (31.32 MB), including numerical archives,
checkpoint receipts, solver audits, recovery records and final
notebook reproduction evidence.

Every archived file passed SHA-256 verification against the evidence
manifest. The archive passed integrity verification and has a separate
SHA-256 sidecar. The evidence archive remains outside Git.

### Executed-notebook publication

| Archive | Notebooks | Tables | Plots |
|---|---:|---:|---:|
| executed_notebooks.zip | 5 | 112 | 64 |
| pr43_executed_notebooks.zip | 1 | 41 | 24 |

Both archives passed safe extraction, member integrity, SHA-256,
manifest consistency, notebook readability, execution completeness
and accepted-source equality checks.

The project-wide archive is flat and retains repository source paths
in MANIFEST.json. The PR 43 snapshot identifies the accepted execution
by run ID. All five tracked bank-pipeline notebooks remain output-free,
with zero execution counts.

Release publication and download-link verification complete the
remaining PR closeout requirements.

