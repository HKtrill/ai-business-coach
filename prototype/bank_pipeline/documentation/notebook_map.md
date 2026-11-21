# Five-Notebook Research Map

PR 41, Commit 3 (Step 3A). This map assigns the existing notebook content and modules to the
five planned Phase 2 notebooks. It is documentation only: no cell, notebook, or module has moved.
The Phase 1 reproduction reference is the tag `phase1-baseline-2026-11-17`.

Paths are relative to `prototype/bank_pipeline/`. Cell references use the notebooks' own
`# CELL <id>:` headers: **G** = `glass_pipeline/glass_cascade.ipynb`, **B** =
`blackbox_pipeline/blackbox_cascade.ipynb`, **E** = `feature_research/exploratory_feature_research.ipynb`.
An asterisk marks a legacy cell (section 1).

---

## 1. Current notebooks

| Notebook | Phase 1 role | Bound by |
|---|---|---|
| G | GLASS cascade, Stages 1–4, and the cascade trace | Phase 1 execution contract (section 4); notebook tests |
| B | Black-box cascade, Stages 1–4, and the 2×2 arbiter crossover | Phase 1 execution contract (section 4); notebook tests |
| E | Feature research, research stage trainers, and the research GLASS Arbiter | Notebook tests only (`tests/notebooks/`: G5, G7) |

**Legacy cells.** G 9–14 and 15–16 (not 14B) and B 19–28B start with `%%legacy_stage2` and run only when
`RUN_LEGACY_STAGE2 = True` (`shared/notebook_switches.py`, off by default). In the accepted path,
Stage 2 is read from the pinned PR 38 routers (G 8B, B 18). B 26–28B (the concluded PR 35
objective-specialization experiment) are also gated by `SPEC_RUN_PROBE`, `SPEC_RUN_TUNING`, and
`SPEC_EVALUATE_TEST`, all `False`.

---

## 2. Planned notebooks

### 2.1 Feature Research (PR 42)

**Responsibilities:** feature exploration, engineering, binning, leakage analysis, temporal
validity, and feature-selection evidence.

| Source | Cells | Content |
|---|---|---|
| E | 1–6 | Setup, database, sanity checks, the cascades' split, feature typing |
| E | 7–10 | Separation metrics, rankings, plots, interaction discovery |
| E | 11, 12A–12G, 13 | Feature builds (crisis, integral, derivative, temporal, prior-contact, overlap), stage feature sets, pre-training validation |
| E | 15A–15B | Lift-driven binning strategy and multi-bin encoding |
| G | 5, 7, 9\*, 10\* | Stage 1 features, correlation pruning, RF lift analysis, GLASS Router features |
| B | 7, 16–17 | Stage 1 data, Stage 2 binary features |

- **Research modules:** `feature_research/{separation,metrics,interactions,visualization,validation}.py`,
  `feature_research/model_training/rf/{binning,lift_analysis}.py`.
- **Research location, production use:** `feature_research/feature_engineering/` (with
  `feature_research/config.py`) is imported by the production Stage 3 pipeline,
  `shared/stage3/feature_engineering.py`, for both arms. Treat it as production code (section 3).
- **Production references (read-only):** `shared/stage1/feature_engineering.py`,
  `shared/stage2/{feature_engineering,binning}.py`, `shared/stage3/feature_engineering.py`,
  `glass_pipeline/lr/correlation_analysis.py`, `glass_pipeline/glass_router/rf/lift_analysis.py`.
- **Evidence:** the Stage 2 binning record `research_logs/pr39/c3_20261114_231604/` (`stage2_binning`).
- **Boundaries:** produces feature evidence and contract recommendations. It does not edit
  production feature code in place, including `feature_research/feature_engineering/`: a revised
  contract is a new version, adopted through Training Research and the PR 45 regeneration. It
  never binds a test row (G7).

### 2.2 Training Research (PR 43, PR 44)

**Responsibilities:** stage-wise training objectives, architecture-specific optimization,
calibration, tuning, and model selection.

| Source | Cells | Content |
|---|---|---|
| E | 14A–14B, 15C–15D, 16A–16B | Research LR, RF, and EBM training and diagnostics |
| E | 17A–17C, 18 | Research GLASS Arbiter (cross-fit, threshold rules, sample trace), run validation |
| G | 6, 8, 11\*–14\*, 15\*–16\* | Baseline LR, LR tuning and calibration; the legacy Stage 2 path (RF baseline and Optuna, GLASS Router fit, cross-fit, report, save), superseded by G 8B |
| B | 8–9, 19\*–22\*, 24\*–25\* | MLP tuning; the legacy Stage 2 path (configuration, single-RF ablation, cross-fit, training-OOF diagnostics, save, Stage 3 hand-off), superseded by B 18 |
| B | 26\*–28B\* | PR 35 objective specialization (concluded; gated) |

- **Research modules:** `feature_research/model_training/{lr,rf,ebm,glass_arbiter,shared}/`.
- **Production tuning and calibration (reuse by import):** `shared/stage1/oof_calibration.py`,
  `shared/thresholds.py`, `shared/stage3/{runner,tuning,calibration,thresholding}.py`,
  `glass_pipeline/lr/{tuning,calibration,penalty}.py`, `glass_pipeline/ebm/tuning.py`,
  `blackbox_pipeline/models/mlp/{tuning,calibration,thresholds}.py`,
  `blackbox_pipeline/models/rf_router/tuning/`, `glass_pipeline/glass_arbiter/{tuning,weighting}.py`,
  `shared/stage4/protocol.py` (`tune_min_confidence`), `blackbox_pipeline/models/meta_xgb/{config,estimator}.py`.
- **Frozen training-side inputs for Stage 4 research:** `shared/stage4/contract.py`
  (`build_stage4_inputs`), `glass_pipeline/glass_arbiter/loader.py` (`load_stage_outputs`),
  `blackbox_pipeline/models/meta_xgb/inputs.py` (`load_blackbox_stage4_inputs`).
- **Boundaries:** training rows only. Selection rules are registered before any test evaluation.
  It produces candidate configurations; regeneration happens in PR 45, not here. G 8 and B 8–9
  also stay in the cascades: they are the accepted training path, and Training Research studies
  alternatives to them.

### 2.3 GLASS Cascade (PR 46)

**Responsibilities:** the final interpretable cascade architecture, stage responsibilities,
inference behavior, diagnostics, and presentation.

| Source | Cells | Content |
|---|---|---|
| G | 0–4 | Environment and commit provenance, imports, database, validation, split |
| G | 5, 8 | Stage 1 calibrated LR |
| G | 8B | Stage 2 source: accepted PR 38 GLASS Router by pinned path and SHA-256 |
| G | 14B | GLASS Router selector regression checks (a test inside the notebook) |
| G | 17–18 | Stage 3 feature block and EBM |
| G | 19–20 | Stage 4 GLASS Arbiter and cascade trace |

- **Modules:** `glass_pipeline/{lr,glass_router,ebm,glass_arbiter}/`, `shared/stage1/`–`shared/stage4/`,
  `shared/{stage_io,validation,pinned_artifacts,notebook_switches}.py`, `data/`.
- **Boundaries:** presents the final architecture. Research and comparison cells move to
  Training Research and Comparative Analysis in PR 46, subject to section 4.

### 2.4 Black-Box Cascade (PR 46)

**Responsibilities:** the final black-box cascade architecture, stage responsibilities,
inference behavior, diagnostics, and presentation.

| Source | Cells | Content |
|---|---|---|
| B | 1–6 | Paths and environment, database, raw data, GLASS imports, module checks, split |
| B | 7–14A | Stage 1 MLP: data, parameters, fit, package, evaluate, plots, threshold, hand-off |
| B | 15–18 | Stage 2 constants, features, the pinned routers |
| B | 29–32 | Stage 3 XGBoost and the Stage 4 hand-off (OOF vs refit) |
| B | 35–36 | Stage 4 Meta-XGB inputs and Meta-XGB |

- **Modules:** `blackbox_pipeline/models/{mlp,rf_router,xgb,meta_xgb}/`, `shared/`, `data/`.
- **Boundaries:** as 2.3.

### 2.5 Comparative Analysis (PR 47–50)

**Responsibilities:** cascade comparisons, standalone baselines, matched operating conditions,
robustness analysis, statistical evaluation, and paper figures.

| Source | Cells | Content |
|---|---|---|
| B | 23\* | Stage 2 comparison, all arms, same metric code |
| B | 33–34 | Stage 3 EBM vs XGBoost and statistical comparison |
| B | 37–38 | Stage 4 2×2 arbiter crossover and results |

- **Comparison utilities:** `blackbox_pipeline/models/meta_xgb/comparison.py`,
  `blackbox_pipeline/models/rf_router/evaluation/{router_metrics,paired}.py`,
  `shared/stage3/comparison.py`, `shared/metrics.py`.
- **Accepted evaluation code (reuse read-only):** `stage4_regeneration/crossover.py` (the registered
  2×2 and its three-population abstention accounting), `shared/stage2/diagnostics/`,
  `stage2_crossfit/{complementarity,reporting}.py`.
- **Artifact access:** `shared/pinned_artifacts.py`, `tests/support/artifact_audit.py` (arbiter
  roots under `$BANK_ARTIFACT_MANIFEST`).
- **Records:** `research_logs/pr39/c5_20261116_153459/crossover_baseline.json`,
  `research_logs/pr38/c3_20261112_153333/report_v1/`.
- **Boundaries:** loads frozen artifacts by manifest and fits nothing. It reuses the accepted
  accounting instead of re-implementing it, and it does not modify the protected packages.

### 2.6 Shared protocol and dependencies

Every notebook, current and planned, draws the same split: `build_database()` →
`BankPreprocessor(drop_leaky=True)` → `GlobalSplitManager(test_size=0.20, random_state=42)
.create_split(..., target_col="y")`. Fold plans: 10 folds for Stage 1; 5 folds for Stages 2–3
and Meta-XGB's Stage 4 cross-fit; seed 42 (`tests/parity/test_notebook_protocol_parity.py`).

Feature Research → Training Research (feature contract versions) → GLASS and Black-Box Cascades
(registered configurations, regenerated in PR 45) → Comparative Analysis (frozen manifests from
the PR 46 freeze).

---

## 3. Utilities

| Class | Area | Modules | Notes |
|---|---|---|---|
| **Shared** | Metrics | `shared/metrics.py` (`calculate_ece`, `compute_metrics`, `binary_metrics`, `metrics_table`) | One metric code for both arms (PR 33) |
| | Thresholds | `shared/thresholds.py` (`sweep_f_beta`, `optimize_threshold_cv`) | In-stage F-beta sweep for both arms |
| | Validation ledger | `shared/validation.py` (`ValidationReport`) | |
| | Stage I/O, fingerprints | `shared/stage_io.py` (`label_fingerprint`, `split_fingerprint`) | Baseline-pinned: import only |
| | Split and data | `data/global_splits/split_manager.py`, `data/bank_database.py`, `data/preprocessing/bank_preprocessing.py` | |
| | Artifact pins and versions | `shared/pinned_artifacts.py` (`pinned_path`, `load_pinned_stage2`), `shared/artifact_versions.py` (`save_versioned`, `alias_target`) | |
| | Notebook switches | `shared/notebook_switches.py` (`register_switch`) | |
| **Research-specific** | Feature analysis | `feature_research/{separation,metrics,interactions,validation}.py` | |
| | Research trainers | `feature_research/model_training/` | Includes a separate research `tune_min_confidence` |
| | Reporting | `feature_research/visualization.py`, `feature_research/model_training/shared/report.py`, `feature_research/model_training/*/diagnostics.py` | |
| **Production** | Stage contracts | `shared/stage1/`–`shared/stage4/` (including `shared/stage2/{contract,evaluation_protocol,obligations,arm_profile}.py`) | Frozen and fingerprinted; reuse, don't edit |
| | Stage 3 feature DAG | `feature_research/feature_engineering/` (`crisis`, `integrals`, `derivatives`, `temporal`, `prior`, `overlap`, `_checks`), `feature_research/config.py` | Imported by `shared/stage3/feature_engineering.py` for both arms; part of the Stage 3 block-cache implementation identity |
| | GLASS stages | `glass_pipeline/{lr,glass_router,ebm,glass_arbiter}/` | |
| | Black-box stages | `blackbox_pipeline/models/{mlp,rf_router,xgb,meta_xgb}/` | |
| | Per-stage loaders | `glass_pipeline/lr/artifacts.py`, `blackbox_pipeline/models/mlp/artifacts.py`, `blackbox_pipeline/models/rf_router/artifacts.py`, `shared/stage3/artifact.py`, `glass_pipeline/glass_arbiter/loader.py`, `blackbox_pipeline/models/meta_xgb/{artifact,inputs}.py` | Split-fingerprint checks are stage-specific |
| | Plotting | `glass_pipeline/lr/evaluation.py`, `glass_pipeline/glass_router/rf/{evaluation,lift_analysis}.py`, `glass_pipeline/glass_arbiter/tracer.py`, `blackbox_pipeline/models/rf_router/reporting.py` | |
| **Provenance-sensitive** | Execution and records | `upstream_regeneration/`, `stage4_regeneration/`, `phase1_reproduction/`, `stage2_crossfit/`, `stage2_binning/`, `stage1_calibration/`, `research_logs/` | Protected |
| | Baseline-pinned code | `shared/stage2/diagnostics/`, `shared/stage_io.py`, `tests/support/{router_artifacts,artifact_audit}.py`, `research_logs/baselines/stage2_baseline.py` | Hash-checked at fixed paths |
| | Archive access | `tests/support/archived_artifacts.py` | Imported by protected code |
| | Compatibility aliases | `glass_pipeline/meta_ebm/`, `blackbox_pipeline/models/rf_router/threshhold/`, `blackbox_pipeline/models/mlp/stage1_mlp.py`, `blackbox_pipeline/models/mlp/evaluation/metrics.py` | Keep: historical imports and pickles |

---

## 4. Phase 1 notebook contracts

Future restructuring must respect these until Phase 1 is reproduced only from the tag. Once G or
B changes in a way that breaks them, Phase 1 re-execution uses `phase1-baseline-2026-11-17`.

| Contract | Enforced by | What must hold |
|---|---|---|
| Notebook paths | `upstream_regeneration/vehicle.py` (`NOTEBOOKS`), `stage4_regeneration/stage4_copy.py`, `upstream_regeneration/records.py` (`CODE_FILES`) | G and B stay at their current paths |
| Entry points | `python -m upstream_regeneration execute`, `python -m stage4_regeneration execute`, `python -m phase1_reproduction execute` (`--notebook glass\|blackbox`) | Unchanged module names |
| Cell headers | `vehicle._HEADER` and `tests/notebooks/test_notebook_contracts.py` (G5) | Every code cell carries exactly one unique `# CELL <id>:` header; no copied cells; no name read before the cell that binds it |
| Stage 4 cut | `vehicle.truncated_copy` | G 19–20 and B 35–38 are the last code cells, in order; no earlier cell calls a Stage 4 entry point (`vehicle.STAGE4_CALLS`) |
| Split cells | `stage4_copy.NOTEBOOKS` | The code cells before the first Stage 1 cell are exactly G 0, 1, 2, 2B, 3, 4 (first Stage 1: G 5) and B 1–6 (first Stage 1: B 7); G 4 and B 6 bind `GLOBAL_SPLIT`. The Stage 4 cells (G 19; B 35–37) call exactly the registered arbiter fits; G 20 and B 38 fit none |
| Bound variables | `stage4_copy.NOTEBOOKS["bind"]` | G: `lr_calibrated_path`, `glass_path`, `ebm_path`. B: `MLP_S1_PATH`, `GLASS_S1_PATH`, `RF_ROUTER_PATH`, `GLASS_BUNDLE_PATH`, `XGB_STAGE3_PATH`, `_ebm_path` |
| Cross-notebook reads | `$BANK_UPSTREAM_PINS`, `shared/pinned_artifacts.py`, `upstream_regeneration/runner.py` (`CONSUMERS`) | G 8B and B 14A, 18, 33, and 35 read artifacts by pinned path and SHA-256 |
| Tracked writes | `phase1_reproduction/runner.py` (`TRACKED_WRITES`) | G 20 writes `research_logs/figures/venn_cascade_trace.png`; B 14A writes the three `models/mlp/stage1_mlp_*.csv` aliases |
| Switches | `shared/notebook_switches.py`; G6 in `tests/notebooks/test_notebook_contracts.py` | `RUN_LEGACY_STAGE2` and the three `SPEC_*` flags stay `False`; gated steps stay under their flags |
| Code identity | `upstream_regeneration/records.py` (`code_identity`) | Any notebook edit changes the recorded identity; accepted `check` commands are unaffected |

New Phase 2 notebooks are new files. They do not inherit these contracts unless a later PR
registers them.

---

## 5. Deferred utility work

| Item | Location | Why deferred | Revisit |
|---|---|---|---|
| Feature-engineering copies | `shared/stage2/feature_engineering.py` keeps self-contained copies of functions in `feature_research/feature_engineering/{crisis,integrals,derivatives}.py`, which Stage 3 imports directly | PR 42 may revise these features. Any edit to the `feature_research/feature_engineering/` modules changes the accepted Stage 3 features | PR 42: decide where the Stage 3 DAG lives before removing copies |
| Research threshold and abstention helpers | `feature_research/model_training/glass_arbiter/{thresholds,arbiter}.py` vs `glass_pipeline/glass_arbiter/meta_stage.py` and `shared/stage4/protocol.py` | `find_recall_threshold` is an identical copy. The research `tune_min_confidence` applies the same F2-under-coverage rule over a fixed grid, while production searches every distinct confidence value. Stage 4 policy research may change both | PR 43 |
| Training helper overlap | `feature_research/model_training/rf/trainer.py` vs `glass_pipeline/glass_router/rf/rf_training.py` (`_build_pipe`, `_normalize_params`) | Small private helpers; research trainers are the Training Research starting point | PR 43 |
| Comparison-ready artifact loading | Per-stage loaders, `shared/pinned_artifacts.py`, `tests/support/artifact_audit.py` | Needs the PR 45 manifests; build on `pinned_path` and the existing loaders | PR 46 |
| Shared plotting | Production and research plotting modules (section 3) | Presentation needs are not yet defined | PR 46 |
