# PR 38 Handoff: Corrected Stage 2 Cross-Fit, Evaluation and Historical Comparison

| | |
|---|---|
| Author | Phillip Harris |
| Record | The single PR 38 readiness and handoff checklist |
| PR / commit | PR 37, Commit 5 (Steps 5A–5C) |
| Branch | `pr-37-router-baseline-contract` |
| Base | `f7a987e` (Commit 4: the frozen Stage 2 specification) |
| Governs | The order in which PR 38 runs the frozen specification, and the evidence each step must leave |
| Related | [`pr38_evaluation_protocol.md`](pr38_evaluation_protocol.md); [`pr37_commit4_contract_record.md`](pr37_commit4_contract_record.md) (section 24); [`pr37_f17_decision_record.md`](pr37_f17_decision_record.md) (section 8); [`pr37_f31_disposition_record.md`](pr37_f31_disposition_record.md) (section 8); [`pr37_commit3_handoff.md`](pr37_commit3_handoff.md) |
| Date | 2026-11-05 |

No PR 38 step has been executed. No corrected router has been fitted on real data, no corrected artifact exists and no corrected result has been seen. Nothing here establishes that the corrected routers are feasible, that either arm performs better, or that any result is independently validated.

**Authority.** This record consolidates; it decides nothing new. The frozen JSON records govern, then the protocol and the contract record, then the F17 and F31 records. Any discrepancy found while executing this checklist is reported and resolved by a new version and approval, never by following whichever text is more convenient.

## 1. Frozen identities

The four frozen records are in effect from Commit 4 (`f7a987e`). That commit was created after the local validation the records name, including the real-artifact re-verification, so the condition in their approval notes ("takes effect when Commit 4 is created after local validation") is met.

| Record | Version | Fingerprint | Strict loader |
|---|---|---|---|
| Contract `pr37-stage2-primary` | `1.0.0` | `0817825b9eecbd11` | `shared.stage2.contract.load_frozen_contract` |
| Protocol `pr38-stage2-evaluation` | `1.0.0` | `245a1a68fb68cf4e` | `shared.stage2.evaluation_protocol.load_frozen_protocol` |
| GLASS profile `pr37-glass-router-primary` | `1.0.0` | `198731d5cdf263f4` | `glass_pipeline.glass_router.profile.load_glass_primary_profile` |
| RF profile `pr37-rf-router-primary` | `1.0.0` | `4e3f9c07e6ecc8ad` | `blackbox_pipeline.models.rf_router.profile.load_rf_primary_profile` |

All four carry the approval of Phillip Harris, 2026-11-05. D-C10 pins the row-aligned flag fingerprints LR `64d511516d69e73e` and EBM `907a7d2d1cad6908` (counts LR 7,727, EBM 6,113, union 8,001, intersection 5,839).

**Historical identities PR 38 depends on:**

- **Population and split:**
  - 32,950 training rows, 3,712 positives;
  - router split `fc3e73a5ef91aea4`, training labels `b984d2a50527`;
  - stage I/O split `d6cddadc3439` (attested, never recomputed);
  - shared outer fold plan: 5 folds, seed 42, `8aea082fcaf371b7`.
- **Baseline:** `pr37_stage2_pre_correction_v1`, 19 files (manifest SHA-256 `905a0b48…`), with 15 pinned code files.
- **Snapshot:** the external `pr36_regen_2026-10-23` snapshot, 37 entries (SHA256SUMS `e82c6a84…`).
- **Source artifacts:** the seven snapshot-attested files below.

| Role | Path (under `bank_pipeline/`) | SHA-256 |
|---|---|---|
| GLASS Router | `models/glass_router/glass_router_20261023_025611.joblib` | `2c70879b…` |
| RF Router | `models/rf_router/rf_router_two_pass.joblib` | `5204a2d6…` |
| GLASS-arm RF result | `models/rf/rf_result_20261023_025611.joblib` | `76b198aa…` |
| LR | `models/lr/lr_calibrated_20261023_003319.joblib` | `61a6b144…` |
| EBM | `models/ebm/ebm_stage3_20261023_070053_780762.joblib` | `cfdac6d2…` |
| Meta-EBM (GLASS) | `models/meta_ebm/meta_ebm_20261023_070055_573687.joblib` | `82a893e1…` |
| Meta-EBM (black-box) | `blackbox_pipeline/models/stage4_crossover/arbiter_on_blackbox/meta_ebm_20261023_095719_949475.joblib` | `b4b78c58…` |

The full digests are in the baseline `MANIFEST.json` and the snapshot's `SHA256SUMS.txt`. The model artifacts and the snapshot exist only on the owner's machine.

## 2. Carried forward from PR 37

**Commit 3's open Commit 4 decisions, resolved:**

| ID | Resolution | Where |
|---|---|---|
| D1 | NPV ≥ 0.95, leakage ≤ 0.15 of all positives, precision ≥ 0.30 | Frozen contract |
| D2 | No positive-leakage count cap | Frozen contract |
| D3 | Every obligation enforced on the selected GLASS union, inner folds and remainder included; legacy per-rule gates kept as candidate filters | Contract record, sections 16–21 and 24 |
| D4 | Six fit outcomes; stop outcomes never fall back | Contract record, section 8; protocol, section 10 |
| D5 | The RF band-support floors are the contract's routed-set support floors, 125 / 75, kept at the freeze as guards | Contract record, sections 6, 11 and 24 |
| D6 | The RF Router owns its hyperparameters; no GLASS artifact is read | RF profile (M8) |
| P1 | Coverage definitions and the lexicographic objective | Contract record, sections 3–4 |
| P2 | Exposure-disclosing provenance | Every frozen record |

**F17 gate G2, as wired under the freeze.**

- The F17 record names `rf_result.params` for the GLASS ranker factory because it was written before the frozen profile existed.
- Under the freeze, the ranker settings are `ranker_settings()`. Its six values equal those historical `rf_result.params`, which `test_stage2_freeze_artifacts.py` re-verifies on the GLASS-arm RF result.
- The runner passes those same settings to the cross-fit (`rf_params`, `glass_rf_model`), to the full-train ranker factory and to the fit of the full-train ordering RF, and checks them with `check_glass_primary_ranker` before use.
- GLASS notebook Cell 12 re-tunes the RF with Optuna. That is the historical pathway, not the primary: a re-tuned value is refused, and a deliberate change needs a new profile version.

## 3. Phase 0: preservation and inventory

These steps come **before any notebook execution or corrected cross-fit**.

| Step | Requirement | Evidence | Failure |
|---|---|---|---|
| 0.1 | Verify the 37-entry snapshot and its 37 live counterparts with the read-only snapshot verifier | `PASSED: snapshot intact; live baseline files unchanged`, saved | Stop. Never regenerate the snapshot; investigate the changed file |
| 0.2 | `stage2_baseline.py --verify` and `--check` on `pr37_stage2_pre_correction_v1` | 19/19 verified; rebuild identical | Stop (G11) |
| 0.3 | Pin the four historical Stage 4 arbiters in a manifest named by `$BANK_ARTIFACT_MANIFEST`. Without it, `artifact_audit.select_roots` takes the newest file, so a later run would silently change what every real-data check reads | The manifest, kept outside the repository, with its SHA-256 | No real-data check is run unpinned once new artifacts can exist |
| 0.4 | Inventory the historical GLASS and RF Router artifacts: paths, sizes, SHA-256, split, label and fold fingerprints, feature contracts, configuration identities (GLASS `config_full`; RF `rf_params`, constraint source `match_glass_oof`, eligibility) and source provenance. Most of this is already recorded in the baseline `MANIFEST.json` (`sources`, `identities`, `routers`) and the snapshot's `MANIFEST.csv` / `PROVENANCE.json`; the inventory cites them and adds only what is missing | A versioned inventory record citing the baseline | Stop if an identity cannot be established |
| 0.5 | Write every new output separately: new versioned names and directories (G10, V7), never the historical fixed-name files (section 11, table A) | Output layout recorded before the first run | A run that would overwrite a historical file is not started |

## 4. Phase 1: preflight gates

These steps come **before any five-fold primary run**, in this order (protocol section 16, steps 1–4).

| Step | Requirement | Evidence | Failure |
|---|---|---|---|
| 1.1 | Re-verify the frozen specification: `pytest -m realdata tests/realdata/test_stage2_freeze_artifacts.py`, with the snapshot sums and the pinned manifest | 3 passed, output saved | The freeze is void; stop |
| 1.2 | Load strictly: `load_frozen_specification()`; both profiles through their strict loaders; `check_arm_profile_binding`; `is_pinned_frozen` | Identities logged as in section 1 | Stop; never inject a record |
| 1.3 | Build and check the configs (G1, G2, G9, V2). GLASS: `glass_primary_config()` with `check_glass_primary_config`; `ranker_settings()` with `check_glass_primary_ranker`. RF: `rf_primary_config()` with `check_rf_primary_config`. Both: the contract adapter checks; RF `constraint_source="shared_contract"` with the contract fingerprint; one RF config factory for full-train and cross-fit | Config dicts and check results persisted | Stop (M4: no relaxation, no hand-edited config) |
| 1.4 | Implementation gates G5 and G8. The runner persists the fold records G5 requires, and an infeasible fold raises rather than falling back to `global_t1`, `outer_rf` or `full` | The runner's tests; the G3 preflight's saved records | Stop |
| 1.5 | G13: audit `neighborhood_subscription_density` and its derivatives. One outer fold, real training data, both arms (F17 record, G13) | Per-arm rows whose binned features or inner-OOF eligibility change; a materiality judgement documented before any primary result | Confirmed material dependency: specify, implement opt-in, validate and approve a correction, and decide adoption before any result. Primary runs without a required correction are not primary evidence. Feature changes reopen F31 (condition 2; R8) |
| 1.6 | G3: the fit-only feasibility preflight. One outer fold, real data, both arms | No infeasibility; each remainder ≥ 500 rows with both classes; every selection, inner folds included, solved with its status recorded; wall time; R1 band supports. No outer-OOF result is inspected | Stop and investigate (M4). Any revision needs a new version, approval and disclosure before corrected outer-OOF results are inspected. The solve time limit changes only under M4 |

## 5. Phase 2: during the primary reruns

**The runs** (protocol section 16, step 5). A full five-fold outer cross-fit is run for both arms:

- on the same population and shared fold plan;
- under the identical frozen constraints and lexicographic objective;
- with no shared fitted estimator.

**Recorded for every fold:**

| Requirement | Source |
|---|---|
| GLASS `selection_reports()`, inner folds included, corresponding exactly slot by slot; CBC status, incumbent, bound and gap where available; the selected rule ids | D-C11; protocol section 10 |
| RF selection records (`t1`, `t1_j`, eligible counts, `t2`) | G5; R5 |
| Remainder rows, positives and prevalence; GLASS inner rule counts, selector outcomes and ranker provenance | G5 |
| Isolation provenance: remainder and ranker rows inside the fold; no held-out row in any fit | G4 |
| Band records and the strip diagnostic, computed inside the cross-fit. Strip A at 5% or more of a fold's flagged Pass 2 band reopens F31 | R1; R2 |
| M6 GLASS inner-OOF Pass 1 union NPV and leakage, as diagnostics | Contract M6 |

**Fold classification.** Each fold is classified from its complete records (`classify_fold_fit`, then `FoldLedger.build`). An accepted status is never given as a string.

- A `feasible_unproven_incumbent` needs a documented deterministic reproduction (M3).
- A `feasible_greedy_fallback` needs a fit-time review recorded before that fold's outer-OOF results are examined (D-C7).
- A stop outcome stops the run and is reported (G8).
- The primary comparison needs all five folds accepted in both arms (D-C6). Otherwise that failure is the outcome reported, and it never justifies easier constraints.

**Reproducibility (G7).** At least one outer fold is rerun. It must give identical rule sets, remainders and decisions. Thresholds must be bit-identical with `n_jobs=1`; otherwise they agree within 1e-12, and any tie difference is investigated.

**Artifacts (G10, G11).**

- Stage 2 artifacts are written under new versioned names, with the fold evidence above.
- Both GLASS `full` pins are updated deliberately in the same change: `test_the_glass_notebook_runs_the_full_population_profile_for_both_fits` (`tests/parity/test_router_operating_contract.py`) and the GLASS `full` expectation in the baseline-pinned `tests/support/artifact_audit.py`. The `artifact_audit.py` edit registers as a code-identity change on `stage2_baseline.py --check`: it is recorded, or handled by a new baseline version. Baseline v1 is never edited.
- The section 3 checks are repeated after the runs (0.1, 0.2), so the historical artifacts are shown to be unchanged.

## 6. Phase 3: structural and provenance acceptance

Phase 3 comes after the reruns and before any comparative statement (protocol section 16, steps 6–9).

- **Traces:** `extract_router_view`, then `build_router_trace`, for both arms.
- **Identity:** `check_population_identity` and `check_upstream_identity` with `require(complete=True)`: the five-layer identity and the D-C10 fingerprints.
- **Gates:** V1–V8 through `structural_acceptance`, under the pinned frozen protocol only (`StructuralAcceptance.official`). V8 includes F31 Disposition A semantics, the R7 relabelling of the black-box Cell 21 diagnostic and the R8 coupling of band values to the G13-passing state.
- **Stage 4 and persistence:**
  - R4 Stage 4 semantic checks;
  - R5 eligibility provenance at Stage 4;
  - serialization and a fresh-interpreter reload;
  - OOF provenance (fold ids, row identities, label and split fingerprints) and routing semantics.

A failed gate means the artifact is not accepted and no comparative claim is made; the failure is reported. Passing establishes interpretability, not performance (protocol section 14).

## 7. Phase 4: the pre-registered evaluation

Phase 4 runs only on artifacts that passed Phase 3, under the frozen protocol, in its runner order (section 16, steps 10–12; S1, part of step 11, follows G12 in Phase 5).

- **Primary outcomes.** O1–O4, pooled and per fold, as paired GLASS − RF differences, with `run_registered_bootstrap`: 2,000 replicates, seed 42, fold × label strata, percentile 95%. They are stated by the D-C1 rule.
- **Secondary diagnostics.** These include:
  - coverage, NPV, leakage, precision, recall, abstention and class-conditional routing;
  - comparisons against LR and EBM at their frozen operating points;
  - matched-budget comparisons (protocol section 7): label-free, with ties broken by row identity;
  - shortfall classification (protocol section 11; D-C2 is descriptive only);
  - F31 R2 and R3, and the M6 diagnostics.
- **Pre/post against the baseline** (protocol section 8), reported as a combined correction.
- **A comparative report with no gaps** (`comparative_report_gaps`).

**Uncertainty.** The intervals are conditional on the fitted routers and frozen upstream flags. They do not account for model-selection history. The multiple primary outcomes support no unqualified global superiority claim.

## 8. Phase 5: secondary analyses

**S1 (G12).** S1 compares GLASS `full` with GLASS `oof_remainder`, everything else held constant.

- It begins once the applicable primary gates other than G6 hold.
- G6 (Pass 1 invariance) is verified during S1, and S1 is not accepted unless G6 passes.
- It is descriptive only, and never used to choose the primary.

**The complementarity-aware ablation** (protocol section 13).

- It uses train-side OOF upstream predictions only.
- It is judged by `evaluate_adoption`, with `UpstreamSources` and `verify_reference`, against C1–C9.
- The Pass 2 budget is held both pooled (C1) and per fold (C5).
- It is exploratory, and official only under the frozen protocol. Any criterion failure keeps the primary.
- Adoption needs an equivalent RF objective or a documented asymmetry. The ablation becomes primary only under a later, versioned protocol.

**The historical versus corrected comparison** (section 10 of this record) is descriptive.

**Optional diagnostics.** R6 and D-C8 are optional. Neither may become a prerequisite.

## 9. Phase 6: final acceptance and Stage 2 freeze

- **Design.** The final Stage 2 selector design is selected by the pre-declared comparison. The primary stays the primary unless the adoption criteria pass.
- **Adoption.** If complementarity-aware selection is adopted, both arms are rerun under the frozen contract and a later, versioned protocol.
- **Artifacts.** The final Stage 2 OOF and test-side artifacts are regenerated. Their OOF provenance, constraint compliance and mirrored contracts are verified.
- **Freeze.** The Stage 2 implementation and artifacts are frozen for downstream regeneration (PR 39).
- **Test-side outputs.** Test outputs are produced for the downstream stages only. Test-derived results never inform selector choice, contract revision, thresholds or adoption (protocol section 15).

**Resolved at PR 38 Commit 5** ([`pr38_commit5_stage2_freeze.md`](pr38_commit5_stage2_freeze.md)): the primary is retained and frozen; no ablation was run. The Commit 3 cross-fit is the Stage 2 OOF regeneration, verified in the official acceptance. Test-side outputs, whose only consumer is Stage 4, are produced in PR 39's integrated regeneration (record, section 5).

## 10. Historical versus corrected comparison

The comparison runs **after** the corrected artifacts pass Phase 3, under the frozen protocol's definitions. It has three kinds of content:

| Kind | Content | Status |
|---|---|---|
| Registered | Protocol section 8: per-arm 3 × 3 transitions (Pass 1 / Pass 2 / abstain, with positives and negatives) and changed rows; Pass 1, Pass 2 and total coverage, abstention, NPV, leakage, Pass 2 precision, both recalls, unique positives, Pass 1 FP suppression; pooled and per fold with paired bootstrap intervals | Part of the comparative report. A combined correction |
| Descriptive extension | Further class-conditional shares; upstream LR/EBM overlap and redundancy; selected rules and counts, selected-union coverage, selector outcomes, solver termination and fallbacks; runtimes | Not registered. Reported as descriptive, with its limits |
| Exploratory | Historical-notebook profiling (section 11) | Never evidence for PR 38 decisions |

**Preconditions, verified before any paired analysis.** `PairedRouting.pre_post` refuses misalignment, and the comparison records each check:

- the same 32,950 rows by index identity;
- the same fold assignment (`8aea082fcaf371b7`);
- the same training labels (`b984d2a50527`);
- the same metric definitions and global denominators;
- the same frozen upstream operating points.

**Where equivalent evidence exists.** The historical GLASS fits persisted their selected rules and fit reports. They did not persist selection reports or CBC logs, because `selection_reports()` came in Step 4B. So for the historical arm:

- solver termination and fallback can be compared only where the historical artifact records them;
- otherwise they are reported as **not recorded**, never inferred.

The historical RF Router used `match_glass_oof` constraints and `global_t1` eligibility. Its selection records are the RF artifact's thresholds, bands, sweeps and fit report.

**Historical runtimes.** The committed notebook outputs record one run on the owner's machine (Python 3.14.3, Windows 11):

- GLASS (output recorded at commit `aa51194`):
  - full-train fit and test routing, Cell 13: 170.0 s;
  - five-fold cross-fit, Cell 14: 1,192.6 s; per fold 215.8, 92.1, 366.2, 363.2 and 155.2 s;
  - fold plan `8aea082fcaf371b7`.
- Black-box: RF outer cross-fit, Cell 21: 118 s. The fold table prints cumulative seconds.

Before these serve as the historical reference, check that they belong to the pinned artifacts. The artifact timestamps fit one sequential run, but the correspondence is not proven.

**Limits on runtime comparisons:**

- Runtimes are comparable only on the same machine and environment, with recorded load.
- The historical runs used the pre-Commit 4 overlap diagnostics.
- The corrected runs add inner folds, union obligations and the row objective.
- Single runs are observations, not estimates.
- The 58× overlap-diagnostics speedup does not translate into a wall-clock or training-time claim.

**Attribution.**

- F17, the shared constraints, the coverage objective and the nested rankers change together. Only S1 isolates one factor: the GLASS population policy.
- No other component is attributed without a single-factor run declared in a new protocol version before results.
- The ablation is compared against the corrected primary, not against history.

**Outputs.** One versioned comparison package, generated by a script from recorded inputs:

- tables, figures and a research summary;
- a manifest with the SHA-256 of every input and the code commit.

Regenerating it must give identical bytes, as `stage2_baseline.py --check` does. It cites the baseline and never edits it.

**Preservation and retirement.**

- Baseline v1 and the 37-entry snapshot are kept permanently.
- A redundant historical working artifact in the repository's model folders may be retired only after:
  - archival integrity is verified (0.1);
  - reproducibility is verified (0.2);
  - the owner approves.
- Historical evidence needed for research reproduction is never deleted automatically.

## 11. Exploratory historical-notebook profiling

This is a separate activity after PR 37. It is exploratory, it is not PR 38 evaluation, and it changes no frozen record, notebook, model artifact or code path. Its results never inform PR 38 configuration, constraints or adoption.

**A. Write paths and overwrite risks** (read-only audit at `f7a987e`; paths under `bank_pipeline/`, the working directory both notebooks set).

| Writer | Files | Risk in the live checkout |
|---|---|---|
| Black-box Cell 24, `save_rf_router` | `models/rf_router/rf_router_two_pass.joblib`, `rf_router_two_pass_metadata.json`, `rf_router_two_pass_pass1_sweep.csv` and `_pass2_sweep.csv`; the same four for `rf_router_single` | **Fixed names, overwritten.** Destroys the pinned RF Router (`5204a2d6…`) and its metadata; changes four Git-tracked sweep CSVs |
| Black-box Cell 14A (Stage 1 MLP) | Fixed-name MLP outputs in `models/mlp/`, including three Git-tracked CSVs | Fixed names, overwritten |
| Stage 3 artifact stores (GLASS Cell 18, black-box Cell 31) | Timestamped files plus `ebm_stage3_latest.joblib` / `xgb_stage3_latest.joblib` | The `_latest` copies are overwritten |
| GLASS Cell 17 | `models/stage3/stage3_block_<split fp>.joblib` | Rewritten if the cache identity differs (for example, other library versions) |
| GLASS Cells 6, 8, 16, 19; black-box Cells 36–37 | Timestamped LR, RF, GLASS Router, Meta-EBM, Meta-XGB and crossover artifacts | Nothing overwritten, but they become the **newest** files, which change later selections (below) |
| Black-box Cells 26–28 | `models/rf_router/specialization/` (Git-tracked) | Off by default (`SPEC_*` flags `False`). Keep them off: Cell 28 evaluates and logs the test split |

**Newest-file readers.**

- Black-box Cell 14A: the newest `lr_calibrated_*`.
- `RF_PARAM_SOURCE="artifact"`: the newest split-matching `rf_result_*`.
- Black-box Cell 18: the GLASS Router bundle.
- The EBM and Meta-EBM lookups.
- `artifact_audit.select_roots`, which the real-data tests use without `$BANK_ARTIFACT_MANIFEST`.

So a run in the live checkout overwrites historical evidence and also redirects later reads. With the snapshot sums set, the real-data tests then fail closed.

**B. Safe execution plan.**

1. Before anything runs, the live checkout must pass checks 0.1 and 0.2, and `git status` must be clean.
2. Profile in a **separate clone** checked out at a recorded commit: never in the live checkout, and not in a `git worktree` of it, which registers itself in the live repository. The clone holds no Git-ignored artifacts, so every artifact it writes is new and isolated. If inputs from the historical artifacts are wanted, copy them (never move) from the read-only snapshot with their timestamps, and verify their hashes against `SHA256SUMS.txt` inside the clone.
3. Execute an unmodified copy of the notebook in a fresh kernel with `nbconvert --execute`. The executed copy goes to an output directory outside both repositories; per-cell timing is recorded in `metadata.execution`. A copy truncated after the Stage 2 cells is allowed only if the remaining cells' sources are byte-identical; their hashes are recorded.
4. Keep the provenance with the outputs:
   - the commit and the environment (Cell 0);
   - the machine and its load;
   - start and end times;
   - the cell-source hashes;
   - the SHA-256 inventory of the clone's model folders before and after.
5. Repeat checks 0.1 and 0.2 on the live checkout afterwards. The live 37/37 must be unchanged.

**C. What to time.**

| Measurement | Where | Notes |
|---|---|---|
| GLASS rule generation, candidate evaluation with overlap diagnostics, ILP selection | Inside GLASS Cells 13 and 14 (`RuleGenerator.generate_candidates`, `RuleEvaluator.evaluate_candidates` → `OverlapAnalyzer.compute_overlap_diagnostics`, `ILPRuleSelector.select_rules`) | Needs a timing wrapper loaded from a profiling-only IPython startup directory, so no repository file or notebook changes; the wrapper is supplied with the activity |
| Complete GLASS Stage 2 | Cells 10–16; cell timing for 13 and 14 | Cell 12 re-tunes the RF (200 Optuna trials): check `rf_result.params` against the six frozen values. If they differ, the downstream GLASS timings are not comparable with history |
| RF Router Stage 2 | Black-box Cells 15–25; the outer cross-fit (Cell 21) | `n_jobs=-1`: sensitive to load |
| Nested folds | Not available for the F17 primary | GLASS runs `full`, so no inner GLASS fold runs. The RF Router runs its inner OOF Pass 1 folds, but under `global_t1`, without the per-fold `t1_j` solves of `nested_t1`. Enabling either would change notebook behaviour. The nested cost is first measured at G3 (wall time) and in the PR 38 reruns |
| Effect of the overlap optimisation on the real legacy fit | The same cells at `ab7b063` (before it; the notebooks are unchanged in PR 37) and at the current head, in two clones | Identical selected rules are expected (behaviour-preserving). Only this paired measurement shows the wall-clock effect |

**D. Test exposure.**

- The notebooks print held-out test metrics (for example GLASS Cells 8, 12B, 13 and 15, and Stages 3–4). These are the previously exposed split, never Tier 3.
- The profiling record discloses the exposure.
- No test-side number from a profiling run informs any PR 38 choice.

## 12. Artifact and evidence dependencies

| Needed by | Item | Location |
|---|---|---|
| Phases 0–1 | The 37-entry snapshot, its `SHA256SUMS.txt`, `MANIFEST.csv`, `PROVENANCE.json`; the read-only verifier | Owner's machine, outside the repository |
| Phases 0–1, 3–4, section 10 | The seven source artifacts (section 1) and the Stage 4 meta-XGB arbiters in the snapshot | Owner's machine, `bank_pipeline/models/…` |
| Phases 0, 4, section 10 | `pr37_stage2_pre_correction_v1` and `stage2_baseline.py` | Repository |
| Phases 1–6 | The four frozen records and their loaders and checks | Repository |
| Phase 3 | `tests/support/artifact_audit.py` (baseline-pinned; G10) | Repository |
| Section 10 | Committed notebook outputs (historical timings); the corrected artifacts accepted in Phase 3 | Repository; PR 38 outputs |

## 13. What PR 38 may and may not claim

**It may claim, conditionally:** under a contract and protocol frozen before the corrected results, with nested cross-fitting, the arms achieved the reported Tier 2 outcomes on the training population's outer folds. The claim is conditional on earlier exposure: hyperparameters, features and contract values were chosen with historical OOF and test results visible.

**It may not claim:**

- independent or external validation;
- that cross-fitting removes earlier research-selection bias;
- that test-split numbers are untouched confirmation;
- that the GLASS fit-time compliance is out-of-sample, since it is in-sample (L2);
- any causal attribution beyond S1.

## 14. Status at handoff

PR 37 hands over:

- the frozen specification (section 1), with its strict loaders and checks;
- the historical baseline and its verification tools;
- the diagnostics and the evaluation machinery;
- this checklist.

PR 38 owns every phase above. None of its gates has been run.

Commit 5's own validation, commit and push, the final PR 37 review and the merge are tracked in the roadmap. They are not recorded here as done.
