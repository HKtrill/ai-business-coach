# PR 38 Commit 5: Stage 2 Selector Decision and Freeze

| | |
|---|---|
| Scope | PR 38, Commit 5 (Steps 5A and 5B) and the PR 38 completion boundary |
| Accepted run | `c3_20261112_153333` (Commit 3, `02cde1d`); consolidated report `report_v1` (Commit 4, `d78781f`) |
| Governs | [`pr38_evaluation_protocol.md`](pr38_evaluation_protocol.md) `1.0.0` (`245a1a68fb68cf4e`), frozen; [`pr38_handoff.md`](pr38_handoff.md) |
| Decision | The original corrected Stage 2 selector stays primary and is frozen. No complementarity-aware ablation is run in PR 38 |
| Changes | Records only. No code, configuration, frozen record, model artifact, notebook or accepted result is changed or regenerated |

## 1. Decision

The final Stage 2 selector is the **corrected primary accepted in Commit 3**: the corrected GLASS Router and the corrected RF Router, each under the frozen contract `pr37-stage2-primary` `1.0.0` and its frozen arm profile, with the G13 correction (`fold_label_excluded`) adopted at authorization. Its accepted artifacts are frozen as they are (section 3). Nothing is retrained.

The design is chosen by the protocol's own rule (handoff section 9; protocol section 13, `failure`): the primary stays primary unless an ablation passes every adoption criterion. No ablation was evaluated, so none can pass. This is a consequence of the registered procedure, not a performance judgement about the ablation.

The Commit 3 primary comparison and the Commit 4 report are the PR 38 results and are unchanged. Every result statement remains as labelled there (confirmatory within Tier 2, descriptive or exploratory; protocol header and section 15).

## 2. Step 5A: the complementarity-aware ablation

**Disposition: INSUFFICIENT PRE-REGISTERED BASIS. The ablation is not run in PR 38.**

**What is registered.** The frozen protocol registers how an ablation is *judged*, not *whether* it is run:

- Section 13 and the frozen record's `complementarity` block (`shared/stage2/protocols/pr38_evaluation_v1.json`) give its role (an exploratory, secondary ablation that never replaces the primary on exploratory results), its design (primary corrected GLASS against a complementarity-aware GLASS, everything else held equal) and the **adoption criteria C1–C9** with their margins (D-C5, section 17), evaluated by `evaluate_adoption` (`shared/stage2/evaluation_protocol.py`).
- C1–C9 are **adoption criteria, not an experiment trigger**. All must pass; a criterion that fails or cannot be evaluated keeps the primary (`failure`). An adopted ablation would become primary only under a later, versioned protocol with both arms rerun (`adoption_effect`; handoff section 9).
- The runner obligations name the ablation only conditionally (section 16, item 13: "If the ablation is run"). No registered record states when the evidence warrants running it.

**What is not registered.**

- **A trigger.** No criterion, threshold or condition for running the ablation exists in the protocol, its JSON record, the handoff or the contract record.
- **An objective.** Section 13 names a "complementarity-aware GLASS ablation" without an objective form, weight, lexicographic position or reference. The two candidate objectives (positives Stage 1 LR misses; rows where LR and EBM disagree) appear only in the research roadmap, where they have stood since before any corrected result; they were not adopted into the frozen protocol.
- **Multiplicity.** C1–C9 judge one ablation against the primary. Two candidate objectives would be two comparisons with no registered handling.

**Alignment with C1.** C1 counts unique true positives against the frozen LR-or-EBM union (`complementarity.reference` = `or`, O4a). Rows where LR and EBM disagree are flagged by one of them and are therefore inside the union: a true positive there cannot count toward C1 by construction. Positives LR misses count toward C1 only where EBM also misses them.

Writing a trigger, choosing an objective, or setting either after the Commit 3 results would be the post-hoc choice the protocol labels exploratory and the roadmap forbids ("without changing the frozen criteria retroactively"). No criterion is introduced here.

**The accepted evidence** (Commit 3, `report_v1`; descriptive unless stated). Observed routing differences and complementary predictive value are kept apart:

- *Routing differences between the arms* (registered primary outcomes, GLASS − RF, `01_registered_outcomes.csv`): GLASS routes fewer rows (coverage −0.115) with higher Pass 2 precision (+0.0095) and more unique true positives (O4a: +26, 95% interval [5.975, 47]) and unique false positives (O4b: +201). These compare two Stage 2 routers; they do not measure value beyond the upstream models.
- *Complementarity with LR and EBM* (`04_unique_contribution.csv`; no registered test exists): of GLASS's 2,049 Pass 2 true positives, 1,874 are flagged by both LR and EBM and 86 by neither. GLASS's 1,226 Pass 2 flags outside the union found 86 positives against 69.8 expected by chance from the same pool; RF's 999 found 60 against 61.0. The two routers' Pass 2 true positives overlap with Jaccard 0.76.
- Stage 2's contribution to the integrated cascade (Stage 4) is not measured in PR 38.

**Missing for a defensible "warranted":** a registered trigger, a registered objective aligned with the C1 reference, and registered handling of more than one objective.

**Future work.** A complementarity-aware objective is separately versioned research after PR 38, not unfinished PR 38 work. It would register one objective aligned with its evaluation reference, and its trigger and adoption criteria, in a new protocol version approved before any run. It belongs with the later Stage 2 training-strategy research (sub-population training, error-focused specialization, controlled oversampling), not with this freeze.

## 3. What is frozen

**Selector design and configuration** (as accepted; re-verified against the code at this commit by section 4, check 1):

| Item | Identity |
|---|---|
| Contract | `pr37-stage2-primary` `1.0.0`, `0817825b9eecbd11` (frozen) |
| Protocol | `pr38-stage2-evaluation` `1.0.0`, `245a1a68fb68cf4e` (frozen) |
| GLASS profile | `pr37-glass-router-primary` `1.0.0`, `198731d5cdf263f4` |
| RF profile | `pr37-rf-router-primary` `1.0.0`, `4e3f9c07e6ecc8ad` |
| Inner-fold features (G13) | `fold_label_excluded`: `stage2-inner-fold-feature-isolation` `1.0.0`, `0d4953ef34d1d3dc` |
| Clarifications | G10 pin scope `pr38-stage2-f17-g10-pin-scope-clarification` `1.0.0`, `11ecf7c53546c6fe`; F31 R2 `pr38-stage2-evaluation-f31-r2-clarification` `1.0.0`, `d55e285570ce47d4`; F31 Disposition A retained |
| Configuration record | `configs.json` (SHA-256 `7b7f803155dae6bf…`): GLASS `pass2_population` `oof_remainder`, `inner_feature_ranker` `fold_nested_rf`; RF `eligibility` `nested_t1`; both bound to the shared contract |
| Population | router split `fc3e73a5ef91aea4`, training labels `b984d2a50527`, outer folds `8aea082fcaf371b7` (5 folds, seed 42) |

**Accepted artifacts** (local only, `models/pr38/c3_20261112_153333/primary/`; pinned by the committed `primary_MANIFEST.json`, SHA-256 `f70f12dc03802215…`):

| Artifact | SHA-256 |
|---|---|
| GLASS bundle `glass_router/glass_router_pr38_primary_c3_20261112_153333.joblib` | `889c8bd572047bebe83a9e60d4de564194d142ae9e91f2faf8750127d05bb753` |
| RF Router, immutable version `rf_router/rf_router_pr38_primary_c3_20261112_153333_20261113_035911_620368.joblib` (the latest alias is byte-identical) | `aadde1ea93d0e300ca238f5b9f8844fa26646e131931e6c0e7d79714d043632e` |
| Fold evidence `evidence.json` (the outer-OOF routes and fit-time evidence) | `e30b17ca7ba1380646f1d7362df0af46b17fa98bc4a0d4a2c68299ad80a3af6a` |

The S1 candidate (`s1_full/`, GLASS bundle `2f2e1ae40694…`) is descriptive sensitivity evidence, not a selector, and is kept as such.

**Acceptance evidence** (committed in `research_logs/pr38/c3_20261112_153333/`):

- `structural_acceptance.json` (`91612167dd258278…`, digest `e101fb62…`): V1–V8 official, 232 of 232 checks. Fit status: GLASS `valid_unproven_documented` and RF `valid_optimal` in all five folds; complete fold ledgers.
- V7 covers serialization and reload in a fresh interpreter, versioned names outside every newest-file folder (G10), and the Stage 4 interface checks (R4): router streams, meta-EBM router evidence and the meta-XGB feature blocks, train and test side, for both arms. V2 covers the mirrored contracts; V1 and V4 fold identities, isolation and G13.
- `g7_reproduction_fold0.json`: G7 passed (fold 0, both arms within tolerance, identical held-out decisions).
- `acceptance_dispositions.json`: G10 `clarification_approved`, F31 condition 2 `disposition_a_retained`.
- `s1_acceptance.json`: S1 accepted, G6 passed (descriptive only).
- `report_v1/` (MANIFEST SHA-256 `5efed74cd66d65b7…`): the consolidated results, verifiable without the v1 packages (`report --check`).

**Verified from the repository for this record:** the identities above, read from the committed records; and that PR 38's own commits (`636f9a4`…`d78781f`) changed no notebook, no tracked model artifact, no baseline file, no frozen contract, protocol or arm profile. The local model files are not inspected here; section 4 re-verifies them.

## 4. Freeze verification (local)

From `prototype/bank_pipeline`. Required before the Commit 5 commit:

1. **The accepted run re-verifies in full against the code at this commit** (`acceptance.require_accepted`; reads records and the candidate only, no train side, no data, nothing written): every candidate file's SHA-256 and size, the manifest copy, no stop record, the official verdict recomputed from the gate records, every bound input's SHA-256, the authorization re-verified with the configuration built *now* from the frozen records, G7, the dispositions and a complete fold ledger.
2. **The report verifies** (`report --check`): every report file, the accepted records, and every historical v1 table reproduced from the report.
3. **The fast default suite**, once, as the roadmap's final regression check (this commit changes no code).

Already accepted and not rerun: structural acceptance V1–V8, G7, the S1 acceptance, the Commit 4 reporting tests. The v1 packages' own `--check` is no longer runnable: their tables were removed after the Commit 4 migration, and `report --check` replaces it.

Optional: the S1 candidate's integrity, and the PR 37 frozen-specification re-verification on the archived artifacts (`tests/realdata/test_stage2_freeze_artifacts.py`; Phase 0 and G11 already re-verified the snapshot and baseline after the run).

## 5. The regeneration boundary

Handoff section 9 says the final Stage 2 OOF and test-side artifacts are regenerated and verified. The roadmap's Step 5B regenerates only when the selected design requires it. They are reconciled as follows; neither text is overridden.

- **Stage 2 OOF artifacts.** The Commit 3 corrected five-fold cross-fit *is* the regeneration of the Stage 2 OOF and model artifacts under the frozen contract (roadmap Step 3A). Their OOF provenance (V1, V4), constraint compliance (V3; observed outer-OOF shortfalls reported apart, protocol section 11) and mirrored contracts (V2) were verified in the official acceptance. The retained design is that design, so no further PR 38 regeneration is required.
- **Test-side outputs.** Section 9 produces them "for the downstream stages only". Their only consumer is Stage 4, whose regeneration is out of PR 38 scope and assigned to PR 39. V7 (R4) already showed that the accepted artifacts produce valid test-side Stage 4 inputs for both arms. PR 39 changes Stage 1 calibration and feature names and may revise the Stage 2 binning; the Stage 2 feature builders read all three features PR 39 renames (`shared/stage2/feature_engineering.py`; `dow_month_encoded` also in `shared/stage2/binning.py`). Persisting test-side outputs now would produce files nothing in PR 38 reads and PR 39 would supersede. They are produced in PR 39's integrated regeneration.
- **Notebooks.** The approved G10 clarification keeps the historical notebook pins unchanged until downstream regeneration; PR 38 artifacts stay isolated in `models/pr38/` and are never read as newest.

**No PR 38 regeneration requirement remains.** None is mandatory under the retained design.

## 6. PR 39 handoff

**Regenerate**, under the PR 39 final contracts:

- Stage 1 under the resolved calibration policy and the renamed features; Stage 3 as affected.
- Stage 2 with the frozen selector design of section 3, after the Stage 2 binning contract is frozen: both arms, five-fold cross-fit, new run id, through Phase 0, authorization, primary, G7 and structural acceptance V1–V8 (`stage2_crossfit`), and its report (`report`; packages may be written to a temporary folder with `--packages-dir`).
- The Stage 2 test-side outputs for Stage 4, from the newly accepted Stage 2 artifacts.
- Stage 4 and the 2×2 arbiter crossover with the finalized upstream artifacts.

**Version before running, where a frozen value changes:**

- A Stage 1 change moves the LR flags the frozen protocol pins (D-C10: LR `64d511516d69e73e`, EBM `907a7d2d1cad6908`, union 8,001) and the frozen upstream operating points (protocol section 5). Any Stage 2 evaluation against new upstream models needs a new protocol version, approved before results.
- A change to the operating contract, an arm profile, the inner-fold feature record or the Stage 2 binning that the frozen configuration reads needs a new version of that record and a new configuration identity. The PR 38 identities above stay the record of the PR 38 results.

**Integrate:** replace the historical notebook pins (G10) with the regenerated artifacts; standardize the GLASS naming (deferred from PR 36) before final regeneration.

**Validate:** structural acceptance V1–V8 of the regenerated Stage 2 run (V7 R4 for the Stage 4 interface), mirrored contracts after all methodology changes, and the final pre-presentation cascade baseline.

**Carry forward unchanged:** the PR 38 results (`report_v1`) remain Tier 2 evidence for the selector design; PR 39 results do not replace them.

## 7. Limitations

These restate the protocol (sections 9, 15 and 18) and the accepted results; they add no criterion.

- Tier 2 only, conditional on earlier exposure: hyperparameters, features and contract values were chosen with historical OOF and test results visible. No Tier 3 data exists; nothing here is independent validation.
- Intervals hold the fitted routers fixed; they do not cover refitting or earlier model-selection variability. Nine primary outcomes support no global superiority claim (D-C1).
- GLASS fit-time compliance is in-sample. Observed outer-OOF shortfalls occur in both arms (one material, RF fold 0 leakage); they are reported, never relabelled (protocol section 11).
- GLASS selections are feasible but not proven optimal in every fold (documented, reproduced; M3).
- The unique contribution beyond LR and EBM is small in absolute terms and its value to the integrated cascade is not measured in PR 38.
- No complementarity-aware objective was evaluated (section 2): PR 38 shows nothing about one, for or against.
