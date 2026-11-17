# PR 39 Execution Contract: Dependency Audit, Regeneration and Decision Criteria

| | |
|---|---|
| Author | Phillip Harris |
| Record | The PR 39 execution contract: dependency audit, regeneration matrix, frozen decision criteria and acceptance gates |
| PR / commit | PR 39, Commit 1 |
| Branch | `pr-39-protocol-finalization` |
| Base | `a4f66fa` (PR 38: Stage 2 selector frozen) |
| Governs | How PR 39 Commits 2–6 decide, regenerate, verify and record |
| Related | [`research_roadmap.md`](../research_roadmap.md) (PR 39); [`pr38_commit5_stage2_freeze.md`](pr38_commit5_stage2_freeze.md) (sections 3, 5, 6); [`pr38_handoff.md`](pr38_handoff.md) (sections 3, 11); [`pr38_evaluation_protocol.md`](pr38_evaluation_protocol.md) (sections 5, 14, 15); [`research_logs/pr38/README.md`](../../research_logs/pr38/README.md) |
| Status | **Frozen**, version `1.0.0`. Approved by the owner (Phillip Harris, 2026-11-14) at the Commit 1 review, with the section 10 decisions recorded. It takes effect when Commit 1 is created after local validation. Any change needs a new version of this record before the results it would affect are inspected |
| Date | 2026-11-14 |

No PR 39 step has been executed. No PR 39 artifact, record or result exists. Every number quoted here is historical (committed notebook outputs or PR 38 records) and is labelled as such.

**Authority.** The frozen PR 37/38 JSON records govern, then the PR 38 protocol, handoff and freeze record, then this contract, then the roadmap. This contract adds PR 39 criteria and changes no frozen record. Where it would amend a PR 38 handoff obligation (D3), it says so, and the amendment takes effect only with the owner's recorded approval.

**Evidence labels.** **[R]** verified from the repository at `a4f66fa`. **[L]** needs inspection on the owner's machine (Git-ignored artifacts, the snapshot, the external manifest); never claimed here.

## 1. Baseline carried into PR 39

| Item | Identity [R] |
|---|---|
| Contract | `pr37-stage2-primary` `1.0.0`, `0817825b9eecbd11`: Pass 1 NPV ≥ 0.95, Pass 1 leakage ≤ 0.15 of all positives, Pass 2 precision ≥ 0.30, routed-set support 125 / 75, lexicographic coverage objective |
| Protocol | `pr38-stage2-evaluation` `1.0.0`, `245a1a68fb68cf4e`; upstream references LR `61a6b144…`, EBM `cfdac6d2…`; D-C10 flags LR `64d511516d69e73e`, EBM `907a7d2d1cad6908` |
| Arm profiles | GLASS `pr37-glass-router-primary` `1.0.0`, `198731d5cdf263f4`; RF `pr37-rf-router-primary` `1.0.0`, `4e3f9c07e6ecc8ad` |
| G13 pathway | `fold_label_excluded`: `stage2-inner-fold-feature-isolation` `1.0.0`, `0d4953ef34d1d3dc` |
| Accepted Stage 2 | Run `c3_20261112_153333`: GLASS `oof_remainder` + `fold_nested_rf`, RF `nested_t1`; V1–V8 232/232; G7 passed; `primary_MANIFEST.json` `f70f12dc03802215…`. Artifacts under `models/pr38/c3_20261112_153333/primary/` [L] |
| Population | 32,950 training rows (3,712 positives), 8,238 test rows; router split `fc3e73a5ef91aea4`; stage I/O split `d6cddadc3439`; training labels `b984d2a50527`; shared outer folds `8aea082fcaf371b7` (5, seed 42) |
| Historical evidence | Baseline `pr37_stage2_pre_correction_v1`; PR 38 records and `report_v1`; snapshot `pr36_regen_2026-10-23`, 37 entries [L] |

PR 39 does not change the selector algorithm, objective, constraints, arm profiles, G13 pathway, fold plans, population, or any PR 37/38 record or result.

## 2. Outstanding requirements located

| Source [R] | Requirement | Lands in |
|---|---|---|
| PR 33 roadmap (`6506a06`), carried through PR 38 | Resolve sigmoid vs isotonic once for both arms; measure whether isotonic granularity materially affects Stage 4; freeze | Commit 2A, §6.1 |
| Same | Rename `cellular_crisis` → `cellular_downturn`, `euribor3m_local_rate` → `subscribe_rate_by_euribor_bin`, `dow_month_encoded` → `subscribe_rate_by_dow_month`; update contracts, documentation, artifacts and downstream references | Commit 2B, §6.6, D2 |
| PR 36 deferred follow-up (roadmap at `a036759`) | "Naming — PR 39": standardize GLASS Router / GLASS Arbiter naming across modules, imports, artifact paths, metadata and documentation where appropriate; preserve model-family identities; validate compatibility before final regeneration | Commit 3B, §6.6, D5 |
| GLASS notebook Cell 19 | The Stage 4 arbiter is "currently implemented as `meta_ebm` / Meta-EBM", a weighted-confidence arbiter to be renamed with no behavioural change | Commit 3B |
| Feature-research notebook Cell 15A; `feature_research/model_training/rf/lift_analysis.py` | The locked thresholds came from in-sample lift on label-fitted encoders; NSD and ECI drift noted; re-derive and freeze under the finalized protocol in PR 39 | Commit 3A, §6.2 |
| PR 38 freeze record, section 6 | Regenerate Stage 1, Stage 3 as affected, Stage 2 (frozen design, after the binning freeze, new run id, full gates), the Stage 2 test-side outputs, Stage 4 and the 2×2; version before running where a frozen value changes; replace the historical notebook pins (G10); standardize GLASS naming before final regeneration | §4, §5, D3 |
| G10 clarification (`g10_pin_scope_clarification_v1.json`) | The two pins (`test_the_glass_notebook_runs_the_full_population_profile_for_both_fits`; the GLASS `full` expectation in `tests/support/artifact_audit.py`) change in the change where an audited Stage 2 artifact becomes a corrected one, with the `stage2_baseline.py --check` code-identity change recorded | Commit 4A, §5.3 |

## 3. Dependency audit

### 3.1 What each stage reads [R]

| Stage | Producer | Label-derived inputs | Reads earlier stages? | Consumers |
|---|---|---|---|---|
| 1 GLASS LR | GLASS Cells 5, 8 (`glass_pipeline/lr/lr_stage_calibrated.py`) | `euribor3m_local_rate`, `dow_month_encoded`, refit in every outer and inner fold (F3, F19) | No | Stage 4 (both GLASS-upstream arbiters); black-box Cell 14A; the protocol's LR reference (historical file only) |
| 1 MLP | Black-box Cells 7–14A (`blackbox_pipeline/models/mlp/`) | The same three features | No | Stage 4 (both black-box-upstream arbiters) |
| 2 GLASS Router, RF Router | PR 38 primary: `stage2_crossfit` only. The notebooks (GLASS Cells 10–16, black-box Cells 15–25) still fit the **legacy** pathway: `CONFIG_PROFILE="legacy"`, `pass2_population="full"`, `CONSTRAINT_SOURCE="match_glass_oof"` | NSD, ECI, `dow_month_encoded` → `nsd_*`, `eci_*`, `dow_*` | **No.** Raw split, `shared/stage2/feature_engineering.py`, `BINNING_STRATEGY`, frozen records | Stage 4; black-box Cell 18 |
| 3 GLASS EBM, XGBoost | GLASS Cells 17–18; black-box Cells 29–33 (`shared/stage3`) | `dow_month_encoded`, `decay_x_density` (NSD), ECI, refit per fold in the block | **No** | Stage 4; black-box Cell 33; the protocol's EBM reference (historical file only) |
| 4 GLASS arbiter (`meta_ebm`), Meta-XGB, two crossovers | GLASS Cell 19; black-box Cells 35–38 | Train-side OOF of Stages 1–3 only | Yes: Stage 1 probabilities (70 % recall thresholds, Brier trust weights), Stage 2 routes and evidence, Stage 3 probabilities | The 2×2 comparison; `tests/support/artifact_audit.py` through `$BANK_ARTIFACT_MANIFEST` |

### 3.2 Findings that shape the plan

- **F1 — Stages 2 and 3 do not depend on Stage 1** [R]. Neither reads a Stage 1 output. A calibration change moves Stage 1 and Stage 4 only. Stage 2 changes only if its own inputs change (bins, Stage 2 feature definitions or names, frozen records); Stage 3 only if its block changes.
- **F2 — The notebooks' Stage 2 is not the frozen primary** [R]. A clean-kernel run as they stand would refit legacy routers and feed them to Stage 4, and it would replace the RF Router latest alias and four Git-tracked sweep CSVs. The notebook Stage 2 source must switch before any PR 39 notebook regeneration (§5.3).
- **F3 — `stage2_crossfit` is rooted in PR 38** [R]. `phase0.RECORD_ROOT` is `research_logs/pr38` and `OUTPUT_ROOT` is `models/pr38`; the CLI has no root option (`acceptance.require_accepted` already takes `output_root` / `record_dir`). A PR 39 Stage 2 run would otherwise write into the PR 38 folders.
- **F4 — A Stage 2 rerun need not reproduce PR 38** [R]. GLASS selections are feasible but not proven optimal in every fold (time-limited CBC; documented; G7 reproduced on fold 0 only). Reuse keeps the Tier 2 evidence describing exactly the artifacts that feed Stage 4 (D3).
- **F5 — The three renamed names are not Stage 1-only** [R]. `dow_month_encoded` is the DOW bin source in Stage 2 (`BINNING_STRATEGY["DOW"]`) and an EBM/XGBoost feature in Stage 3 (`EBM_FEATURES`). `cellular_crisis` and `euribor3m_local_rate` are Stage 2/3 intermediates, not consumed. Each stage computes them with its own encoder. The names also appear in frozen evidence: `inner_feature_isolation_v1.json`, both G13 records, and `stage2_crossfit/g13.py` (D2).
- **F6 — The bins have no versioned identity** [R]. `BINNING_STRATEGY` is a code constant. No frozen record or configuration fingerprint covers it, and `inner_feature_isolation_v1.json` lists "the 29 bins and their BINNING_STRATEGY thresholds" as unchanged. A versioned record of a bin change therefore needs a binning identity first (§6.2, §7.3).
- **F7 — The calibration evidence is partly exposed and not persisted** [R]. In the committed outputs, the GLASS LR's OOF Brier is 0.084192 (sigmoid) and 0.082922 (isotonic), and isotonic was chosen in 10/10 outer folds and for the final model. The MLP's final model is sigmoid, which the procedure picks only when its pooled OOF Brier is not higher. Neither artifact saves the per-family OOF vectors (`calibration_oof_`), only scalar Briers (`calibration_metrics`). Commit 2 must recompute paired evidence.
- **F8 — Arbiter naming touches persisted identifiers** [R]. These include schema `meta_ebm/2` (`load_meta_ebm_for_comparison` refuses any other), `models/meta_ebm/` and `stage4_crossover/arbiter_on_blackbox/`. They also include the `artifact_audit.ROOTS` keys (a baseline-pinned file), `phase0.ARBITER_KEYS` and the external `$BANK_ARTIFACT_MANIFEST` keys. Renaming without read compatibility breaks re-verification of PR 36–38 evidence.
- **F9 — Notebooks read across arms by newest file** [R]. Black-box Cell 14A reads `lr_calibrated_*`, Cell 18 `glass_router_*`, Cell 33 `ebm_stage3_*` and Cell 35 `meta_ebm_*`; `artifact_audit.select_roots` does the same without a manifest. PR 39 reads must resolve to pinned files.
- **F10 — The frozen protocol's upstream references are historical** [R]. Any Stage 2 *comparative* evaluation against PR 39 Stage 1/3 models needs a new protocol version before results (freeze record section 6). None is planned. The PR 38 complementarity results stay tied to the historical upstream.
- **F11 — `report --check` depends on the platform** [R, run on Linux]. Re-rendering `report_v1` on Linux differs only in the order of record keys in `MANIFEST.json` and `run_summary.json`, and every hash and all 64 historical table checksums reproduce. The cause: `reporting._records` sorts `Path` objects, case-insensitively on Windows and case-sensitively on POSIX. Verification stays on Windows unless that sort is made platform-independent (a code change, not in Commit 1).
- **F12 — Some mutable aliases are Git-tracked** [R]. `models/mlp/stage1_mlp_{threshold_sweep,training_history,tuning_trials}.csv` and `models/rf_router/rf_router_{single,two_pass}_pass{1,2}_sweep.csv` change whenever their producer runs (D7).

## 4. Regeneration matrix

| Artifact | Calibration (2A) | Stage 1 names (2B) | NSD/ECI bins (3A) | GLASS naming (3B) | Disposition |
|---|---|---|---|---|---|
| Stage 1 GLASS LR and its StageOutput | changes | changes | — | — | **Regenerate** (Commit 4) |
| Stage 1 MLP and its StageOutput | changes | changes | — | — | **Regenerate** (Commit 4) |
| Stage 2 GLASS Router and RF Router (OOF and test side, Stage 4 layouts) | — | only under D2 (b) | changes | only if `glass_router` identifiers change (not proposed) | **Reuse with verification** when no column changes them (§5.2, D3); otherwise **regenerate** with `stage2_crossfit` |
| Legacy notebook Stage 2 files (`models/glass_router`, `models/rf`, `models/rf_router`) | — | — | — | — | **Untouched**; not consumed in PR 39 |
| Stage 3 block, GLASS EBM, XGBoost | — | only under D2 (b) | — | — | No PR 39 change requires it. Under D4 the notebook run refits it like for like and verifies it against the Stage 3 protocol, block fingerprint and features |
| Stage 4: GLASS arbiter (native), Meta-XGB (native), GLASS arbiter on black-box, Meta-XGB on GLASS | inputs change | — | inputs change if Stage 2 does | names, schema, paths | **Regenerate** all four (Commit 5) |
| Frozen JSON records (contract, protocol, profiles, inner-feature record, clarifications) | — | — | a new *version* of the inner-feature record only if bins change | — | v1 unchanged |
| PR 38 records and `report_v1`, baseline v1, snapshot, `models/pr38/**`, every historical file in `models/` | — | — | — | — | **Never modified**; re-verified (§7.5) |
| Notebooks | the calibration argument | Stage 1 names and assertions | — | arbiter imports and labels | Functional edits only (§9) |

## 5. Regeneration sequence

### 5.1 Order

1. **Commit 2:** Stage 1 evidence, decision and implementation. No artifact is promoted.
2. **Commit 3:** Stage 2 binning evidence and decision; naming.
3. **Commit 4A:** preservation checks (§7.5) → Stage 2 (reuse verified, or regenerated) → GLASS notebook (LR, EBM) → black-box notebook (MLP, XGBoost; reads the GLASS LR, GLASS Router bundle and EBM). Stage 2 must precede the notebooks, which load it. No Stage 4 cell runs.
4. **Commit 4B:** the Stage 1–3 acceptance record and the PR 39 upstream manifest.
5. **Commit 5A:** Stage 4 from the artifact manifest: the GLASS arbiter (GLASS Cell 19), then Meta-XGB and both crossovers (black-box Cells 35–37), which read GLASS artifacts. **5B:** the comparisons (Cell 38).
6. **Commit 6:** full clean-kernel reproduction and final validation.

### 5.2 Stage 2: reuse or regenerate (D3)

**Rule.** Stage 2 is regenerated if, after Commit 3, any of the following holds:

- an NSD/ECI revision is adopted (§6.2);
- a Stage 2 feature definition, name or binning identity differs from the accepted run's;
- a frozen Stage 2 record changes version;
- a Stage 4 reader no longer accepts the accepted artifacts unchanged.

Otherwise `c3_20261112_153333` is reused, provided Commit 4A shows all of the following on the final code:

- `acceptance.require_accepted("c3_20261112_153333")` passes;
- `report --check` passes (§7.5);
- the Stage 4 interface checks of V7 (R4) pass for both upstreams, train and test side;
- the artifacts' SHA-256 match `primary_MANIFEST.json`.

**Amendment.** Reuse amends freeze record section 6 ("new run id"). That section's stated reason is that the Stage 2 builders read the renamed features and that the bins may change; it does not apply under D2 (a) with the bins retained. The owner approved reuse at the Commit 1 review (D3, section 10).

**Regeneration**, when required:

- `stage2_crossfit` with `primary_settings("fold_label_excluded")` and a new run id;
- the full PR 38 sequence: Phase 0, configs, G13, G3 preflight, authorization, primary, G7, accept V1–V8, report;
- records in `research_logs/pr39/<run_id>/` and models in `models/pr39/<run_id>/`, through a root option added in Commit 4A (F3), with defaults unchanged so PR 38 runs still verify;
- for a bin revision, the binning identity and the new inner-feature record version exist before G13 (§6.2).

### 5.3 The notebooks' Stage 2 source (D4)

- **Before** any PR 39 notebook execution, both notebooks load the Stage 2 artifacts of §5.2 by pinned path and SHA-256 and do not fit Stage 2.
- The legacy Stage 2 cells stay behind an explicit switch that is off by default, like the PR 35 `SPEC_*` cells.
- The two G10 pins change in the same change, and the `stage2_baseline.py --check` code-identity note is recorded.
- Every cross-notebook read (F9) resolves to a pinned file.

### 5.4 The regeneration vehicle (D4)

**Approved (D4):**

- **Commit 4A:** one fresh-kernel execution of each notebook (`jupyter nbconvert --execute`, the executed copy written outside the repository), with the Stage 4 cells excluded. They are excluded either by a switch that is off by default, or by a truncated copy whose remaining cell sources are byte-identical and hashed (precedent: handoff section 11 B).
- **Commit 5A:** the Stage 4 cells run in a fresh kernel that rebuilds only the split and reads the upstream paths from the PR 39 artifact manifest. The Stage 4 entry points already take paths (`train_meta_stage`, `train_meta_xgb_stage`). Stages 1–3 are not retrained.
- **Commit 6B:** one full fresh-kernel execution of each notebook on the final code, as a reproduction (§7.4). Its artifacts are new immutable versions, compared with the accepted ones; they never replace them.

**Alternative considered, not adopted:** one full execution per notebook in Commit 4, Stage 4 included, with the Stage 4 outputs left unread until 4B passes; 6B as above. This is simpler, but the ordering then rests on discipline rather than mechanism.

## 6. Frozen decision criteria

### 6.1 Stage 1 calibration (Commit 2A)

**Scope.** Stage 1 only: the GLASS LR and the MLP. Stage 3 calibration (fold-nested isotonic, gated at OOF ECE > 0.05) and Stage 4 are unchanged.

**Evidence E-C**, one run per arm:

- **The run.** The stage's own `fit()` on the training rows with `calibration_method="auto"`: current tuning, 10 outer × 10 inner nested calibration, per-fold encoders and scalers, seed 42. Both families are calibrated from the same base scores.
- **Persisted per family.** The OOF vector, the per-fold inner Brier and family, and the base scores, each with its SHA-256 in the record.
- **Excluded.** The test split is not read, and nothing is promoted to a Stage 1 artifact.

**Reported** per arm and family, pooled and per outer fold:

- Brier;
- ECE (`shared.metrics.calculate_ece`, 10 bins) and its Murphy reliability and resolution terms (same bins);
- ROC-AUC and PR-AUC;
- distinct OOF values against distinct base scores, and the largest tied block's share of rows.

**Stage 4 input diagnostic.** Isolated: only the Stage 1 vector differs, nothing is fitted on test and no Stage 4 performance is used. On each family's OOF, at the Stage 4 target recall of 0.70 (`glass_pipeline.meta_ebm.meta_stage.find_recall_threshold`):

- the threshold, achieved recall and flag count;
- the rows whose Stage 4 vote differs between the families;
- the number of distinct threshold-relative confidences |p − t|;
- the Spearman correlation between the families.

Optional and descriptive: the GLASS arbiter's train-side trust weights, with the pinned historical Stage 2/3 inputs held fixed.

**Rule (registered):**

- **C1.** One family for both arms and every outer fold. Per-fold and per-arm `auto` selection leaves the primary path; it stays available for diagnostics.
- **C2.** The default is `sigmoid`, and the burden of proof is on `isotonic` (see "Nature of the rule" below).
- **C3.** `isotonic` is selected only if all three hold in **both** arms:
  - (a) the pooled OOF ΔBrier = Brier(isotonic) − Brier(sigmoid) is < 0, with its paired bootstrap 95 % percentile interval below 0 (2,000 replicates, seed 42, rows resampled within outer fold × label strata);
  - (b) ΔBrier < 0 in at least 8 of the 10 outer folds;
  - (c) the isotonic OOF reaches the Stage 4 target with achieved recall ≤ 0.71.
- **C4.** Otherwise `sigmoid`. A failed evidence run is reported and stops the decision; it never defaults.
- **C5.** Reported-only metrics never override C3. A problem they show is recorded as a limitation. Changing a criterion needs a new version of this record before E-C is inspected.

**Nature of the rule.** C1–C5 is a conservative decision rule, not a neutral comparison of the two families. `sigmoid` is kept unless the evidence for `isotonic` is consistent in both arms, so a tie, an inconclusive result or a gain in one arm only all give `sigmoid`. The Commit 2 record states the outcome as the rule's selection, not as the better family.

- **Rationale.** One family must serve both arms (C1), so a change of family should hold in both. `sigmoid` is the procedure's existing tie rule. Platt scaling is a two-parameter, strictly monotone map: it keeps each fold's ranking and adds no ties. Isotonic regression is a nonparametric step function that pools scores into tied blocks. That pooling is the granularity this PR must check at Stage 4, where thresholds and abstention cuts are chosen among observed values.
- **Tradeoffs.**
  - The rule can forgo a real isotonic gain that is modest, appears in one arm only, or misses C3 (b) or (c).
  - `sigmoid` can leave miscalibration where the score–probability relation is not logistic.
  - Given F7, the outcome is predictable. If the MLP's exposed result reproduces, C3 (a) fails and `sigmoid` is selected. The GLASS LR then changes family and gives up its historical isotonic Brier advantage (0.00127).
- **Neutral alternative (D1).** This selects the family with the lower mean relative ΔBrier across the arms, with `sigmoid` on ties. The Commit 2 record also states, descriptively, what it would select; that statement does not change the decision.

**Freeze.**

- Both arms' constructors default to the chosen family, and the notebooks pass it.
- `calibration_requested` equals `calibration_method`, and `oof_family_by_fold` is uniform.
- Focused tests: `tests/stage1/`.

### 6.2 Stage 2 NSD and ECI binning (Commit 3A)

**Scope.** `nsd_*` and `eci_*`, four bins each. JED, CCI, DOW, BEHAV, CAMPAIGN and CPI are retained without revalidation.

**Intended semantics**, from the rationale in `shared/stage2/binning.py`:

- both families rise monotonically in subscription rate;
- `*_cold` lies below baseline;
- `nsd_elevated` and `eci_warm` lie above it, and `*_hot` well above;
- `nsd_warm` and `eci_mid` are transitional and claim no direction.

**Evidence E-B** (training rows only; shared outer folds `8aea082fcaf371b7`):

- **Features.** For each fold k, the Stage 2 engineer is fitted on the training rows outside k, transforms the rows of k, and the result is binned with the thresholds under test. No row's NSD or ECI comes from an engineer that saw its fold's labels.
- **Per bin**, per fold and pooled: support, positives, rate, lift (pooled against the pooled OOF base rate; per fold against that fold's) and a Wilson 95 % interval.
- **In-sample table.** The historical method (one engineer fitted on all training rows), to measure own-label inflation.
- **E-B2** [L]. In the accepted PR 38 run: the selected GLASS Pass 1 and Pass 2 rules per fold that contain each bin, and each bin's RF importance where recorded.

**Triggers.** A family is flagged if any one fires for any of its bins:

| Trigger | Fires when |
|---|---|
| T1 sparse bin | In any outer fold, the bin's held-out support times that fold's base rate is < 10 (fewer than ten expected positives) |
| T2 reversal | The pooled lift interval lies wholly on the wrong side of 1.0 for the bin's intended direction |
| T3 ordering | An adjacent pair is inverted in pooled OOF lift and in at least 4 of 5 folds |
| T4 hot collapse | A `*_hot` bin's pooled OOF lift is < 1.5 (the binning report's own hot marker) |

Lift drift that leaves T1–T4 clear, such as a lower but still strong hot lift, is not a trigger.

**T1 is a binning criterion, not a routing obligation.**

- **Its value** is set here on its own grounds (D6). An expected count of at least ten is a conventional adequacy condition for estimating a proportion. It is applied per fold, so that the per-fold lifts that T3 and the stability report use can be estimated. At the training base rate of 0.113, that is about 89 rows.
- **The contract's routed-set floors** (Pass 1 125 rows, Pass 2 75 rows) are a different obligation. They bound each pass's selected rule union, are enforced by the selector at fit time and are verified in V3.
- **Kept apart.** The floors are not applied to bins, and nothing here changes them. A candidate revision meets them only through the fold-0 preflight.

**Decision.**

- **No trigger.** The thresholds are retained unchanged.
- **A trigger.** A candidate is derived for that family only, from E-B, by the method the original thresholds used: decile lift of the source feature, now out of fold.
  - It keeps the bin count, the names and the `RF_FEATURES_BINARY` order. The only exception is a merge forced by T1, into the adjacent bin with the closer lift.
  - The derivation is scripted and recorded before any router fit.
  - It is adopted only if it clears T1–T4 on E-B and both arms pass a fit-only preflight on outer fold 0 under the frozen settings. The G3 conditions apply: every selection feasible and accepted, every remainder ≥ 500 rows with both classes, and no outer-OOF metric.
  - Otherwise the thresholds are retained and the limitation recorded.
- **Router utility.** E-B2 and the fold-0 fit-time objective values are evaluated and reported. They veto only through the preflight. No decision uses outer-OOF router, Stage 4 or test results.
- **Adoption** requires:
  - a new binning identity (§7.3);
  - a new inner-feature record version naming the bins;
  - a G13 re-audit;
  - Stage 2 regeneration (§5.2).
  The GLASS ranker hyperparameters are not re-tuned: the profile stays frozen, and this is disclosed.

**Limitation.** Every threshold, historical or revised, is a fixed constant chosen with all training labels. Outer-fold cross-fitting does not remove that dependence.

### 6.3 Leakage safety and fold isolation (all commits)

- **L1.** No test label, prediction or metric informs any PR 39 decision: calibration, bins, names, selector, thresholds or Stage 4 configuration. The test split is previously exposed (protocol section 15), so PR 39 test results are descriptive.
- **L2. Stage 1.** Target encodings are refit inside every outer and inner fold from raw rows (F3, F19). Calibration evidence comes only from the nested pathway.
- **L3. Stage 2.** Label-fitted encoders (NSD, ECI, DOW): binning evidence uses outer-fold refits (§6.2), and router fits use the frozen `fold_label_excluded` pathway (G13).
- **L4. Stage 3.** The block is refit per fold, and its calibration and thresholds come from OOF. Unchanged.
- **L5. Stage 4.** Arbiters train only on train-side OOF streams, which `build_stage4_inputs` checks. Thresholds and the abstention cut come from training rows, and the test split is scored once.
- **L6. Fold plans.** Stage 1 uses 10 folds with seed 42, its fingerprint recorded. Stages 2 and 3 use the shared 5-fold plan `8aea082fcaf371b7`. Stage 4 requires streams with equal fold counts to share one partition.
- **L7. Disclosed dependences.** The binning thresholds (§6.2). The Stage 1 F2 threshold is a global in-selection choice (protocol section 5). Stage 4 stacking keeps an indirect upstream dependence (F39, PR 45).

### 6.4 Shared routing obligations and the frozen selector

- **Frozen records.** The contract, protocol, both profiles, the G13 pathway and the configuration record of §1 are used as frozen: no relaxation, re-tune or hand edit (M4). Both arms share the population and the fold plan, share no fitted estimator and are configured independently (G9).
- **Any PR 39 Stage 2 run** follows the full PR 38 gate sequence (§5.2). All five folds are accepted in both arms (D-C6), and stop outcomes never fall back (G8).
- **Routing semantics.**
  - Three states: `PASS_1_NOT_SUBSCRIBE`, `PASS_2_SUBSCRIBE`, `ABSTAIN`.
  - Pass 1 precedence, and Pass 2 on each arm's own OOF Pass 1 remainder.
  - Abstention accounted for.
  - F31 Disposition A: RF band blends are never called calibrated probabilities.
- **Stage 4** reads the routers only through the existing adapters and never re-thresholds them.
- **No Stage 2 comparative re-evaluation** in PR 39 (F10). Stage 2's role in the final cascade is examined descriptively in Commit 5 and never changes the selector.

### 6.5 The 2×2 arbiter crossover (Commit 5)

| Arbiter ↓ / upstream → | GLASS (LR, GLASS Router, GLASS EBM) | Black-box (MLP, RF Router, XGBoost) |
|---|---|---|
| GLASS arbiter (`meta_ebm` today) | Native: GLASS Cell 19 | Crossover: black-box Cell 37 |
| Meta-XGB | Crossover: black-box Cell 37 | Native: black-box Cell 36 |

- **Ownership and upstream.** Each arbiter keeps its own code, configuration and slot names. The upstream is declared (`upstream=`), and a mixed or undeclared upstream is refused (F29). The record names what filled each slot.
- **Alignment.**
  - The same `GLOBAL_SPLIT` test rows, with index equality and both split-fingerprint schemes (`check_split_identity`).
  - Each arbiter at its own training-side configuration. The thresholds and the abstention cut come from training rows (`tune_min_confidence`: F2 on retained rows, coverage ≥ 0.50, exact search).
  - The test split is evaluated once.
- **Registered comparisons, as implemented:**
  - **The two columns**: the arbiter effect with the upstream held fixed, from `compare_with_meta_ebm(…, require_same_upstream=True)`. Metrics without and with abstention: coverage, accuracy, precision, recall, F1, F2, ROC-AUC, PR-AUC. Also agreement with an exact McNemar test, retention overlap, and a paired bootstrap of every Δ (2,000, seed 42).
  - **Cascade vs cascade** (native against native), reported separately: the upstreams differ by design.
- **Not registered in PR 39:** the rows (upstream effect, arbiter fixed), coverage grids and confidence tests in calibrated space. These belong to PR 43.
- **Calibration.** Meta-XGB's probability quality is reported as recorded. The GLASS arbiter's weighted score is not a calibrated probability, as the comparison already notes.
- **Status.** The results are descriptive, on a previously exposed split. They never inform calibration, bins, names, the selector or thresholds. The PR 36–38 comparisons remain as history.

### 6.6 Naming (Commits 2B, 3B)

**Stage 1 features:** the three renames of section 2, with scope D2.

- The values are identical: old and new columns are equal on train and test, for the full-train and per-fold engineers, in both arms.
- The encoders are unchanged.
- New Stage 1 artifacts record the new names under a new artifact-contract version.
- Historical Stage 1 artifacts still load through the boundary extractors and the Stage 4 adapters: PR 38 V1 and baseline `--check` read them.
- Frozen records and historical outputs keep the old names.

**GLASS components** (D5):

| Now | Adopted | Compatibility |
|---|---|---|
| `glass_pipeline/meta_ebm/`, "Meta-EBM", `train_meta_stage` | `glass_pipeline/glass_arbiter/`, "GLASS Arbiter" | Matches `feature_research/model_training/glass_arbiter` and the notebook labels |
| Schema `meta_ebm/2`; `models/meta_ebm/meta_ebm_*` | A new schema (for example `glass_arbiter/1`); `models/glass_arbiter/glass_arbiter_*` | Readers keep accepting `meta_ebm/2` for historical artifacts |
| `stage4_crossover/arbiter_on_blackbox/`; audit keys `meta_ebm`, `meta_ebm_on_blackbox` | `glass_arbiter_on_blackbox/`; keys `glass_arbiter`, `glass_arbiter_on_blackbox` | `phase0.ARBITER_KEYS` and existing manifests keep the historical keys; the `artifact_audit.py` change is recorded on `--check` |
| `compare_with_meta_ebm`, `find_meta_ebm_artifact`, `meta_ebm_path=` | GLASS-arbiter names | The old names may stay as aliases through PR 39 |
| GLASS Router (`glass_router`, `model_family="glass_router"`) | Unchanged | Keeps the PR 38 artifacts readable (D3) |
| Documentation slips (`shared/stage2/binning.py` header `glass_router.rf.binning`; a "GLASS-BRW" comment) | Corrected | No behaviour |

Also unchanged: the black-box names (MLP, RF Router, XGBoost, Meta-XGB), every `arm` and `model_family` value, and the Stage 4 slot names. The naming changes alter no behaviour, and the synthetic Stage 4 and parity tests pass with only their names updated.

## 7. Artifact and provenance contract

### 7.1 Locations

| What | Where | Git |
|---|---|---|
| PR 39 records: decision evidence, run records, manifests, acceptance | `research_logs/pr39/<run_id>/` | Committed, byte-exact: `.gitattributes` `*/** -text`, created with the first record |
| PR 39 index | `research_logs/pr39/README.md` (Commit 2) | Committed |
| Stage 2 regeneration (only if §5.2 requires it) | `models/pr39/<run_id>/` | Ignored |
| Notebook-produced Stage 1, 3 and 4 artifacts | Their producers' folders, under immutable versioned names | Ignored; pinned by the PR 39 artifact manifest |
| PR 39 artifact manifest: every consumed or produced artifact, with role, path, size and SHA-256 | `research_logs/pr39/<run_id>/artifact_manifest.json` with a `.sha256` sidecar | Committed |
| PR 39 arbiter manifest: exactly the four `artifact_audit.ROOTS` keys, which `select_roots` requires; the file `$BANK_ARTIFACT_MANIFEST` names | `research_logs/pr39/<run_id>/arbiter_manifest.json` with a `.sha256` sidecar | Committed |
| Executed notebook copies | Outside the repository; SHA-256 and cell-source hashes in the run record | Not committed |
| Failed or stopped attempts | Recorded locally, never read by code (as `research_logs/pr38/.gitignore`) | Ignored |

Never modified: `research_logs/pr38/**`, `research_logs/baselines/**`, the v1 frozen JSON records, the PR 37/38 methodology records, `models/pr38/**`, the snapshot, and every existing file in `models/`.

### 7.2 Run identities

- Run ids are `c<commit>_<YYYYMMDD_HHMMSS>`, as in PR 38, one folder per run.
- Records are written once: an existing record is never replaced, and a new attempt gets a new run id.
- An alias (`<stem>.joblib`, `*_latest`) is never evidence. Records name immutable versions.

### 7.3 Identities every PR 39 record carries

- **Population.** The stage I/O and router split fingerprints, the training-label fingerprint, the shared fold plan, and the Stage 1 fold plan (10 folds, seed 42, fingerprint).
- **Feature contract per stage.** The ordered names and the SHA-256 of that list.
- **Stage 2 binning identity.** The SHA-256 of the canonical JSON of `BINNING_STRATEGY` and `RF_FEATURES_BINARY`. It is recorded by the Commit 3A record and by any PR 39 Stage 2 run, whether or not the bins change (F6).
- **Configuration.** The calibration family and the identities of the frozen records used.
- **Code and environment.** The code commit; Python, numpy, pandas, scikit-learn, interpret, xgboost, optuna and pulp versions; the operating system.
- **Files.** The SHA-256 and size of every input and output artifact. Each manifest has its own `.sha256` sidecar.

### 7.4 Serialization and reproducibility

- **Reload.** Every new artifact class is reloaded in a fresh interpreter. `build_stage4_inputs` runs on reloaded upstream artifacts. The real-data audit passes with `$BANK_ARTIFACT_MANIFEST` naming the PR 39 arbiter manifest. Its GLASS `full` expectation (`PASS2_POPULATION`) is one of the two G10 pins (§5.3).
- **Seeds and Stage 2.** Seeds stay 42. Any Stage 2 run passes G7 as in PR 38.
- **Commit 6B reproduction** (D8). The `1e-9` value is the target for a deterministic reproduction in the accepted run's recorded environment (§7.3). It is not a requirement across environments. Each reproduction is compared with the accepted artifacts and classified:

| Class | Condition |
|---|---|
| Deterministic | Same recorded environment. Identical selected configurations (hyperparameters, calibration family), fold ids, Stage 2 inputs (SHA-256), thresholds and decisions. Probability vectors within a maximum \|Δ\| of `1e-9` |
| Numerically equivalent | Identical selected configurations, fold ids and Stage 2 inputs. Every threshold selects the same candidate, or moves by no more than the maximum \|Δ\| of the vector it is chosen on. Every route or decision that differs is on a row whose compared score lies within its drift of the threshold, allowing for the threshold's own movement. No registered comparison statement (§6.5) changes |
| Not reproduced | Anything else |

- **Evaluated and documented** in the 6B record, whatever the class:
  - environment differences (the §7.3 fields);
  - **numerical drift** per artifact: the maximum and 99.9th-percentile \|Δ\| of the OOF and test vectors, and their rank correlation;
  - **threshold stability** for every operating threshold (the Stage 1 F2 thresholds, the Stage 3 OOF thresholds, the Stage 4 recall thresholds and abstention cuts): the accepted and reproduced values, \|Δt\|, whether the same candidate is selected, and the change in flag count;
  - **routing and decision equivalence**: Stage 2 three-state routes, Stage 1 and 3 decisions at their operating thresholds, and each arbiter's decisions and abstentions. For each, the number of rows that differ, and those rows with their distance to the threshold;
  - each registered comparison statement, unchanged or changed.
- **Acceptance (A6.2).** A6.2 passes by one of two routes only:
  - **Reproduced:** the result is Deterministic or Numerically equivalent.
  - **Environmental exception:** the result is Not reproduced, and both of the following are recorded:
    - its root cause was investigated and found to be an environment difference, not a code or data difference;
    - the owner approved the exception in a recorded disposition.

    The exception is recorded as a limitation. It is never granted before that investigation, nor for a code or data cause.
  - Any other Not reproduced result blocks the Phase 1 freeze.
  - Either way, the accepted Commit 4–5 artifacts remain the baseline.

### 7.5 Historical preservation

These checks run before Commit 4A writes anything, and again after Commits 4A, 5A and 6B:

1. `phase0.verify_snapshot(SUMS)` passes, and the 37 live counterparts are unchanged or recorded as expected [L].
2. `phase0.run_baseline_checks(SUMS)` passes: `stage2_baseline.py --verify` and `stage2_baseline_archive.py --check`.
3. `python -m stage2_crossfit report --run-id c3_20261112_153333 --check` passes (on Windows; F11).
4. `acceptance.require_accepted("c3_20261112_153333")` passes [L].
5. A SHA-256 inventory of `models/` before and after shows only additions, plus the documented mutable aliases (F12, D7).
6. `git status` shows only the commit's intended changes.

Each `stage2_baseline.py --check` code-identity note is recorded in that commit's run record (G10/G11 practice). Baseline v1 is never edited.

## 8. Acceptance gates

A gate passes only with its evidence recorded. A failed gate stops the commit, and the failure is reported, never relaxed.

**Commit 2 — Stage 1 calibration and names**

| Gate | Requirement |
|---|---|
| A2.1 | E-C complete for both arms (§6.1), persisted with digests, the test split unread |
| A2.2 | Calibration decision by C1–C5, recorded before the default is changed in code |
| A2.3 | Both arms on the chosen family; StageOutput records it; `tests/stage1/` passes |
| A2.4 | Renames per D2: value identity, per-fold isolation (`test_stage1_fold_isolation`) and no remaining old name in scope, except historical records and compatibility readers |
| A2.5 | New Stage 1 artifact-contract version; the Stage 4 adapters read both the new and the historical Stage 1 artifacts (synthetic tests; `test_stage2_freeze_artifacts.py` [L]) |

**Commit 3 — binning and naming**

| Gate | Requirement |
|---|---|
| A3.1 | E-B and E-B2 recorded with T1–T4 evaluated (§6.2) |
| A3.2 | Retain, or adopt a candidate that passed the fold-0 preflight, with its binning identity and record version |
| A3.3 | Frozen contract, protocol and profile fingerprints unchanged; only the records §6.2 names are versioned |
| A3.4 | Naming per D5: compatibility readers for historical identifiers; no behaviour change; the fast suite passes |
| A3.5 | Readiness record: D2–D5 resolved, the §5.2 outcome determined, the vehicle fixed |

**Commit 4 — Stage 1–3 regeneration**

| Gate | Requirement |
|---|---|
| A4.1 | §7.5 checks before any write; the run id and output layout recorded |
| A4.2 | Stage 2 reused (the §5.2 checks) or regenerated and accepted (V1–V8, G7) |
| A4.3 | Stage 1, both arms: frozen family and names, nested OOF provenance, fold ids, split fingerprint, threshold source, fresh-interpreter reload |
| A4.4 | Stage 3, both arms: protocol, block fingerprint, features, OOF provenance, calibration gate, reload |
| A4.5 | `build_stage4_inputs` passes for both upstreams, train and test side, with no Stage 4 fit |
| A4.6 | §7.5 checks after the run; notebook executions recorded (§5.4) |
| A4.7 | Upstream manifest committed; an explicit Stage 1–3 acceptance decision |

**Commit 5 — Stage 4 and the crossover**

| Gate | Requirement |
|---|---|
| A5.1 | Every Stage 4 input resolves to the artifact manifest by SHA-256; no newest-file selection |
| A5.2 | Four arbiters built; upstream records verified; `same_upstream` for both columns |
| A5.3 | Schema, serialization, reload; the real-data audit under the PR 39 arbiter manifest |
| A5.4 | The §6.5 comparisons recorded as descriptive test-side results; no upstream change |
| A5.5 | Baseline record: final model identities and results; the PR 38 results untouched |

**Commit 6 — final validation and Phase 1 freeze**

| Gate | Requirement |
|---|---|
| A6.1 | Fast default, `slow`, `notebook` and `realdata` suites (the last under the PR 39 arbiter manifest), with saved reports |
| A6.2 | The 6B reproduction is Deterministic or Numerically equivalent. Otherwise it is Not reproduced with an environmental root cause investigated and recorded, and the owner's exception approved in a recorded disposition (§7.4). The §7.4 evaluation is recorded in every case |
| A6.3 | §7.5 checks; no unexpected file or model-family change |
| A6.4 | Notebooks run top to bottom from fresh kernels (`test_each_active_notebook_is_one_labelled_top_to_bottom_path`), loading artifacts by manifest. Execution-breaking issues fixed minimally |
| A6.5 | Phase 1 freeze record, limitations and the Phase 2 handoff |

## 9. Phase 1 and Phase 2

| PR 39 (Phase 1: functional) | PRs 40–41 (Phase 2: presentation) |
|---|---|
| The Stage 2 source switch and pinned reads (§5.3); the Stage 4 manifest entry (§5.4) | Condensed outputs and stage summaries |
| Calibration arguments, renamed features and assertions, arbiter imports | Shared plotting module and visuals |
| Name updates in existing cell text where a name changes | Three-state rewrite of the Venn trace |
| Fixes that stop a notebook from executing | Standardized export format for comparisons |
| Fresh-kernel execution evidence | Slicing, reordering and rewritten prose |

A PR 39 notebook edit that is not needed for correct execution, correct names or correct artifact consumption belongs to Phase 2.

## 10. Owner decisions

Approved by the owner (Phillip Harris) at the Commit 1 review on 2026-11-14. Each decision is the recommendation this section documented.

| ID | Decision | Approved | Needed before |
|---|---|---|---|
| D1 | The calibration rule: the conservative rule (`sigmoid` unless C3 holds in both arms: 8 of 10 folds, recall ≤ 0.71), or the neutral alternative (§6.1) | C1–C5 as written, with the rationale and tradeoffs recorded | Commit 2 |
| D2 | Rename scope: (a) Stage 1 only, the Stage 2/3 columns keep their names and the divergence is documented; or (b) pipeline-wide (Stages 1–3, feature research, G13 mapping; forces Stage 2 regeneration and Stage 3 feature-contract changes) | (a) Stage 1 only | Commit 2B |
| D3 | Stage 2 reuse under §5.2, amending freeze record section 6 | Approved | Commit 4A |
| D4 | The vehicle and the notebook Stage 2 source (§5.3–5.4) | As recommended | Commit 4A |
| D5 | GLASS arbiter identifiers and compatibility (§6.6) | As proposed | Commit 3B |
| D6 | Binning trigger values: T1 at least 10 expected positives per fold (a binning criterion, separate from the contract's routed-set floors); Wilson 95 %; T3 4 of 5 folds; T4 1.5 | As written | Commit 3A |
| D7 | Tracked alias CSVs: commit their updated content as the new latest, or stop tracking them | Commit the updates in the producing commit | Commit 4 |
| D8 | Reproduction classes and acceptance (§7.4): `1e-9` as the deterministic target in the recorded environment; numerical equivalence evaluated and documented otherwise | As written | Commit 6 |

## 11. Local verification items [L]

| Item | Why | Needed by |
|---|---|---|
| `require_accepted("c3_20261112_153333")` and `report --check` | Confirms the frozen baseline on this branch | Before Commit 2 (recommended); Commit 4A |
| Snapshot and baseline checks (§7.5, steps 1–2) | G11 continuity | Commit 4A |
| Inventory of `models/`: which Stage 1, 3 and 4 files are newest, including the `2fc030e` regeneration | F9; the pre-run inventory | Commit 4A |
| MLP `calibration_metrics` (per-family Brier, `oof_family_by_fold`) | Completes the F7 disclosure | Commit 2A |
| The accepted candidates' test-side Stage 4 outputs under the final Stage 4 code | §5.2 reuse | Commit 4A |
| The accepted run's selected rules per fold, and RF importances | E-B2 | Commit 3A |
| Determinism of Stage 1 and 3 training locally (Optuna seeds, `n_jobs`) | The reproduction class to expect (D8) | Before Commit 6 |
| The PR 38 `$BANK_ARTIFACT_MANIFEST` file and its keys | Compatibility under D5; Phase 0 | Commit 3B |

## 12. What PR 39 may and may not claim

**It may claim:**

- under the finalized and pre-registered contracts, both cascades were regenerated, verified and evaluated on the previously exposed test split;
- the calibration and binning decisions followed criteria frozen before their evidence was inspected, conditional on the disclosed historical exposure (F7, §6.2).

**It may not claim:**

- independent validation, or Tier 3 evidence;
- that the PR 39 Stage 4 or crossover results confirm anything out of sample;
- that PR 39 results revise or replace the PR 38 Tier 2 results;
- that any Stage 2 comparison against the new upstream was made (F10).
