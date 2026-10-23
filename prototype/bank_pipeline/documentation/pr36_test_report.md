# PR 36 — Automated Pipeline Test Suite: Test Report

| | |
|---|---|
| Author | Phillip Harris |
| Branch | `pr-36-automated-testing` |
| Scope | `prototype/bank_pipeline` (`churn_pipeline` excluded; no changes made) |
| Commits | 1–4 (`36d178c`, `4f77da0`, `aa51194`, `42753db`); Commit 5 (this report) |
| Report date | 2026-10-23 |

## 1. Outcome

PR 36 consolidates the pipeline's stage-level validation into one `pytest`
suite that runs without the notebooks or saved models, adds static contracts
for the three research notebooks, and adds an opt-in audit of the artifacts
the notebooks save. After the Commit 4 regeneration every category passed:
**1,773 fast, 7 slow and 16 real-artifact tests — 1,796 in total.**
All 1,796 also passed together in one Windows invocation (387.42 s),
with no failures, skips, expected failures or warnings. The final source audit (P7) found no blocking implementation issues.

## 2. Results

| Tier | Selection | Tests | Windows, maintainer's tree (regenerated notebooks and artifacts) | Linux, rebuilt tree, Python 3.13.16 | Linux, rebuilt tree, Python 3.14.6 |
|---|---|---:|---|---|---|
| Fast | `pytest` or `pytest -m "not slow"` | 1,773 | 1,773 passed | 1,773 passed with each selection (428 s; 421 s) | 1,773 passed with each selection (431 s; 401 s) |
| Slow | `pytest -m slow` | 7 | 7 passed | 7 passed (65 s) | 7 passed (60 s) |
| Real-artifact | `pytest -m realdata` | 16 | 16 passed (16.1 s) | collected, not executed | collected, not executed |
| **Total** | `python -m pytest -q -m "slow or realdata or not slow"` | **1,796** | **1,796 passed (one combined run, 387.42 s; also verified by tier)** | 1,780 passed, 16 collected | 1,780 passed, 16 collected |

How to read the table:

* **The Windows suite was verified both by tier and in one combined run.**
  The combined invocation completed with `1796 passed in 387.42s` and no
  warning summary. The real-artifact-only run also reported
  `collected 1796 items / 1780 deselected / 16 selected`.
* **Windows** is the maintainer's machine, where the notebooks were re-run in
  clean kernels and the artifacts regenerated. These runs were executed and
  reported by the maintainer; console output for the real-artifact and
  combined runs was shared. Only this environment holds the regenerated artifacts, so it is the
  only one in which the real-artifact tests were executed.
* **Linux** is an independent rebuild: a fresh clone of the branch at
  `aa51194` with the approved Commit 4 patches applied in order, byte-identical
  to the tree the patches were developed and tested on. Its notebooks carry the pre-regeneration outputs; the
  suite reads notebook source only, so the outputs do not affect these
  results. The 16 real-artifact tests were collected there to confirm they are
  discoverable, not run, because the artifacts are not in the repository.
* Every selected test passed in every run listed: no failures, skips,
  expected failures or warnings.

During Commit 4, each patch was also validated on Windows (and corrected
where that validation found a problem) before the next was written; those
intermediate runs are not repeated here.

## 3. Test architecture

### 3.1 Layout and discovery

All tests live under `prototype/bank_pipeline/tests/` and are configured by
one file, `prototype/bank_pipeline/pytest.ini`:

* `testpaths = tests` and `pythonpath = .` — the project packages import
  without installation, from `bank_pipeline/`.
* `--strict-markers`, `--strict-config` and `xfail_strict = true`.
* A default selection of `-m "not slow and not notebook and not realdata"`.
* Two `filterwarnings` entries ignore PuLP deprecation messages. Six modules
  that fit the MLP on tiny synthetic folds also ignore one scikit-learn
  message (`batch_size` larger than the sample), which cannot occur at
  production scale.

| Directory | Tests | What it holds |
|---|---:|---|
| `data/` | 73 | Global split, split and fold fingerprints, the `StageOutput` hand-off |
| `preprocessing/` | 37 | `BankPreprocessor` and the shared feature contracts |
| `feature_research/` | 54 | Feature DAG, research metrics, research trainers and folds |
| `stage1/` | 269 | GLASS LR and black-box MLP: estimators, hyperparameters, calibration, fold isolation |
| `stage2/` | 434 | GLASS Router (cross-fit, routing, ILP selector, Pass 2 population) and RF Router (contracts, OOF, hyperparameters, hand-off, PR 35 specialization) |
| `stage3/` | 284 | Shared Stage 3 protocol, EBM and XGBoost arms, calibration, thresholds, provenance, block cache |
| `stage4/` | 244 | Stage 4 contract, Meta-EBM, Meta-XGB, abstention search |
| `parity/` | 277 | Cross-pipeline parity: fingerprints, folds, label fingerprints, router contracts, hyperparameters, 2×2 crossover |
| `cascade/` | 28 | GLASS cascade integration, Stages 1–4, on the production producers |
| `notebooks/` | 17 | Notebook contracts and import resolution |
| `support/` | 63 | Self-tests of the test helpers and of the real-artifact audit |
| `realdata/` | 16 | Opt-in audit of the saved artifacts |
| **Total** | **1,796** | 1,773 fast, 7 slow, 16 real-artifact |

### 3.2 Synthetic fixtures

The fast and slow tiers need no data files and no saved models. Their
builders live in `tests/support/` as plain functions:

* `synthetic` — small, deterministic bank-like frames and splits;
* `stage2_synthetic`, `rf_router_world`, `meta_xgb_world` — stage worlds in
  each arm's formats;
* `stage4_world` — synthetic Stage 1–3 artifacts in the cascade's real saved
  formats, for the Stage 4 and crossover tests;
* `recorders` and `checks` — record which rows every fit, transform and
  predict call saw, and turn those records into leakage verdicts;
* `mutations` — deliberate corruptions for negative tests;
* `notebooks` — static notebook inspection (read, never execute);
* `artifact_audit` — the real-artifact audit (§3.5).

Many leakage tests use these recorded calls, so they fail if a model ever
saw a row it should not have, not only when outputs differ.

### 3.3 Tiers and opt-in

* **Fast** (1,773): the default selection.
* **Slow** (7, `-m slow`): one real end-to-end GLASS Router fit
  (`stage2/test_glass_router_fit.py`). It takes about 30 s, and its ILP solves
  are capped at 2 s, so the tests check invariants of whatever rule set is
  returned rather than pinning one.
* **Notebook** (`-m notebook`): the marker is declared, but no test uses it.
  The notebooks are never executed by pytest.
* **Real-artifact** (16, `-m realdata`): `tests/conftest.py` deselects these
  tests unless the `-m` expression names `realdata`. A plain `pytest` and
  `pytest -m "not slow"` both leave them out, because a command-line `-m`
  replaces the default selection in `pytest.ini`. Once enabled, a missing
  database, artifact or recorded upstream file fails with its path; nothing
  is skipped.

### 3.4 Notebook contracts

`glass_pipeline/glass_cascade.ipynb`,
`blackbox_pipeline/blackbox_cascade.ipynb` and
`feature_research/exploratory_feature_research.ipynb` are parsed as source,
never executed. Every project import in them must resolve, and every
production module must import in a clean interpreter. The contracts check:

* stage wiring and producer arguments (GLASS Stage 4, black-box Cell 18);
* that artifact discovery uses only the folders the current producers write;
* one labelled, duplicate-free, top-to-bottom execution order with no package
  installs;
* that the concluded PR 35 specialization experiment is off by default;
* that the feature-research notebook uses the cascades' split protocol and
  never binds test rows;
* that both cascade notebooks follow one experimental protocol (split, folds,
  seeds).

The execution-order check gives the same findings on Python 3.11, 3.12, 3.13
and 3.14.

### 3.5 Real-artifact audit

`tests/realdata/test_saved_artifacts.py` uses `tests/support/artifact_audit.py`.
It only reads: nothing is trained, refitted, backfilled or deleted.

* **Roots.** The audit starts from the four Stage 4 arbiters of the 2×2. These
  are the native Meta-EBM (GLASS Cell 19), the native Meta-XGB (black-box
  Cell 36) and the two crossovers (black-box Cell 37). Each is either the
  newest file in its producer's folder or the file named in a JSON manifest
  given by `$BANK_ARTIFACT_MANIFEST`. The audit never falls back to an older
  file.
* **Upstream files.** The six Stage 1–3 files are the ones each arbiter
  recorded consuming, by path and content fingerprint. They are never searched
  for.
* **Checks.** The audit re-runs each arbiter's own Stage 4 loader. It also
  reloads every file in a clean interpreter, reproduces the saved test
  probabilities from the saved Stage 3 and Meta-XGB models, recomputes both
  2×2 comparison columns with the upstream held fixed, and rejects XGBoost
  files that need modules removed in Commit 4.
* **Self-tests.** The audit is itself tested on synthetic artifacts in the
  saved formats (`support/test_artifact_audit.py`, 21 tests, fast tier).

## 4. Coverage by area

Modules are named relative to `tests/`. A module can serve several areas.
These are behavioural contracts; no line-coverage measurement was made (§8).

**Split identity, fingerprints and row alignment.**
`data/test_split_manager.py`, `data/test_fingerprints.py`,
`data/test_stage_output.py`, `parity/test_split_fingerprint_parity.py`,
`parity/test_stage4_split_fold_provenance.py`,
`parity/test_router_label_fingerprint.py`.
* The single split is stratified, disjoint, exhaustive and deterministic.
* Both fingerprint schemes are pinned with golden values: index and labels
  (12 hex), and rows only (16 hex).
* Every hand-off rejects reordered, reset, foreign or duplicated indices
  (F2), and stale or tampered fingerprints.
* Stage 4 requires every stream's fold ids to be the shared stratified plan
  at its declared seed (F11).
* Routers must record label fingerprints for both their train and test
  outputs, and a missing one is refused, never backfilled (F12, mandatory
  since Commit 4).

**Feature contracts and leakage safeguards.**
`preprocessing/test_bank_preprocessing.py`,
`preprocessing/test_feature_contracts.py`,
`feature_research/test_feature_pipeline.py`,
`stage3/test_glass_ebm_features.py`, `stage1/test_stage1_fold_isolation.py`.
* Leaky columns are dropped.
* Feature contracts enforce the column set and duplicates and realign order
  by name. They do not check dtypes.
* Imputation is learned on the rows it is fitted on.
* The Stage 3 feature pipeline is fitted inside every fold, including inner
  folds (F22).
* Stage 1 target encoders and scalers are refitted per fold (F3, F4).

**OOF integrity and cross-fitting.**
`stage1/test_stage1_fold_isolation.py`,
`stage2/test_glass_router_crossfit.py`, `stage2/test_rf_router_oof.py`,
`stage3/test_oof_integrity.py`, `stage3/test_fold_label_isolation.py`,
`stage4/test_meta_xgb_stage.py`, `feature_research/test_fold_engineering.py`.
* Every training row is scored exactly once, by a model that never saw it,
  and nothing is fitted on test rows. This is checked from recorded calls.
* Shuffling labels inside one fold leaves that fold's stored outputs
  unchanged (F22).
* The RF Router's Pass 2 uses the out-of-fold Pass 1 remainder.
* Meta-XGB's meta-level OOF stays inside the meta folds.

**Calibration and threshold validation.**
`stage1/test_stage1_calibration.py`, `stage1/test_glass_lr.py`,
`stage1/test_blackbox_mlp_stage.py`, `stage3/test_calibration.py`,
`stage3/test_thresholds.py`, `stage4/test_abstention_search.py`.
* Stage 1 chooses the calibration family by Brier score on nested
  out-of-fold probabilities (F16), with preprocessing refitted inside the
  calibration folds (F19).
* Stage 3 calibration gates and maps are fitted on out-of-fold training
  predictions.
* Fold-nested thresholds never use their own fold's labels (F22).
* Nothing learned depends on test labels.
* The MLP's threshold source and labels are validated (F33, F34).
* The Stage 4 abstention cut is an exact search over distinct training
  confidences, matched against a brute-force reference (F30).

**Stage 2 routing and abstention.**
`stage2/test_glass_router_routing.py`, `stage2/test_glass_router_fit.py`,
`stage2/test_ilp_rule_selector_union.py`,
`stage2/test_glass_router_pass2_remainder.py`,
`stage2/test_rf_router_contracts.py`, `stage2/test_rf_router_handoff.py`,
`stage2/test_rf_router_specialization.py`,
`parity/test_router_operating_contract.py`.
* Routing has three states, and Pass 1 takes precedence: rows it routes are
  never re-routed by Pass 2.
* One probability transform is shared by both routers.
* Stored decisions are reproduced from the saved artifacts.
* Invalid router inputs are rejected (F36).
* Pass 2 populations are recorded and must agree between the cross-fit and
  the full-train router: GLASS `full`, RF `oof_remainder` (F17).

**Stage 3 configuration and artifact provenance.**
`stage3/test_artifact_provenance.py`, `stage3/test_glass_ebm_estimator.py`,
`stage3/test_glass_ebm_stage.py`, `stage3/test_blackbox_xgb_estimator.py`,
`stage3/test_blackbox_xgb_stage.py`,
`parity/test_hyperparameter_validation.py`, `stage1/test_blackbox_mlp_params.py`,
`stage2/test_rf_router_params.py`.
* Unknown, missing or invalid hyperparameters are rejected loudly (F1).
* Artifacts round-trip, and lookup matches split, block and features without
  falling back to a stale file.
* Parameters are reused only from a compatible, tuned artifact, and the two
  arms must run one protocol.
* The configuration fixes each artifact's identity: EBM `ebm`/`glass` (F21)
  and XGBoost `xgboost`/`blackbox` (F37).

**Stage 4 schema, stacking and 2×2 crossover.**
`stage4/test_stage4_contract.py`, `stage4/test_meta_ebm_stage4_inputs.py`,
`stage4/test_meta_ebm_stage.py`, `stage4/test_meta_ebm_arbiter.py`,
`stage4/test_meta_xgb_estimator.py`, `stage4/test_meta_xgb_stage.py`,
`stage4/test_meta_xgb_handoff.py`, `parity/test_stage4_crossover.py`,
`cascade/test_glass_cascade.py`.
* The Stage 4 feature block's names, order, dtypes and rows are checked.
* Training cells come from upstream OOF columns and test cells from
  full-train outputs; in-sample substitution is refused.
* The Meta-EBM is configured from training data only, and its router evidence
  is read at each row's own base rate (F25).
* Model-family and upstream identity are checked (F32, F37).
* Abstention NaNs are intentional and verified (F38).
* The 2×2 crossover is declared explicitly. Each arbiter records its upstream
  combination and consumed-output fingerprints, and held-fixed comparisons
  refuse arbiters that read different upstream outputs (F29).

**Cache integrity and notebook validation.**
`stage3/test_block_cache.py`, `notebooks/test_notebook_contracts.py`,
`notebooks/test_notebook_imports.py`, `parity/test_notebook_protocol_parity.py`.
* The GLASS Stage 3 block cache is reused only when its identity matches.
  The identity covers raw data, split, build settings, pipeline, the source of
  every reachable project module, and library versions.
* A reused block must also pass the block's own contracts.
* Missing, malformed, pre-F22 or pre-F23 caches are rebuilt automatically,
  and writes are atomic (F23).
* Notebook contracts are as described in §3.4.

**Regenerated artifact loading and inference checks.**
`realdata/test_saved_artifacts.py` (opt-in) and `support/test_artifact_audit.py`.
* Each arbiter is checked for its declared cell and complete evidence.
* Recorded upstream files pass the contract and are the outputs that were
  consumed.
* Every file reloads in a clean interpreter.
* Saved Stage 3 and Meta-XGB models reproduce their saved test probabilities.
* Both crossover columns hold the upstream fixed, and the arms stay apart.
* No XGBoost artifact needs code that has been removed.

## 5. Commit history

| Commit | Content |
|---|---|
| 1 — `36d178c` | Consolidated stage-level tests into `tests/` with centralized `pytest.ini` and shared synthetic fixtures; data, preprocessing, feature-research and support suites |
| 2 — `4f77da0` | GLASS cascade integration and shared Stage 1–4 contracts |
| 3 — `aa51194` | Black-box arm (MLP, RF Router, XGBoost, Meta-XGB) and cross-pipeline parity |
| 4 — `42753db` | Final testing, notebook contracts and cleanup, as seven reviewed patches and a regeneration (P1–P7, listed below) |
| 5 | This report (documentation only) |

**F15 regression coverage:** Written in the preceding PR; existing regression
tests were retained and consolidated in PR 36.

Commit 4 consists of:

* **P1:** F21 EBM identity guard and pandas 3 compatibility.
* **P2:** mandatory F12 label fingerprints and F17 Pass 2 declarations.
* **P3:** validated Stage 3 block cache (F23), notebook hygiene and contracts,
  and a Python 3.14 portability fix to the execution-order check.
* **P4:** removal of the legacy standalone XGBoost package (15 modules) and
  dead code.
* **P5:** the real-artifact audit.
* **P6:** clean-kernel regeneration of all three notebooks and their
  artifacts.
* **P7:** final audit.

## 6. Environments

| | Windows (maintainer) | Linux (independent rebuild) |
|---|---|---|
| Source tree | Local tree with regenerated notebooks and artifacts | Fresh clone at `aa51194` plus the approved Commit 4 patches |
| Python | 3.14 (the interpreter on which Commit 4's Python 3.14 portability issue appeared) | 3.13.16 and 3.14.6 |
| Libraries | Project virtual environment used for the regeneration | The same 114 pinned packages on both versions, including pytest 9.1.1, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, scipy 1.18.1, xgboost 3.4.1, interpret 0.7.8, optuna 4.9.0 and joblib 1.6.0 |
| Tiers run | Fast, slow, real-artifact, plus combined full suite | Fast, slow; real-artifact collected only |

The pandas 3 compatibility fix (P1) was also checked against pandas 3.0.6.
The notebook execution-order helper was also checked standalone on Python
3.11.17 and 3.12.3. No CI workflow is configured in the repository; the fast
tier needs no data files or saved models, so it can be run in CI as is.

## 7. Reproduction

From `prototype/bank_pipeline/`:

```bash
pytest                          # fast tier: 1,773
pytest -m "not slow"            # the same selection
pytest -m slow                  # slow tier: 7
pytest -m realdata --collect-only -q   # lists the 16 real-artifact tests
pytest -m realdata -v           # real-artifact tier: 16 (needs regenerated artifacts)
python -m pytest -q -m "slow or realdata or not slow"  # combined full suite: 1,796
```

To audit specific arbiter files instead of the newest in each folder, point
`$BANK_ARTIFACT_MANIFEST` at a JSON file. It must contain the keys `meta_ebm`,
`meta_xgb`, `meta_ebm_on_blackbox` and `meta_xgb_on_glass`, with paths that
are absolute or relative to `bank_pipeline/`.

The real-artifact tier needs artifacts regenerated by the current code, in
this order:

1. Move any pre-shared-protocol `xgb_stage3_*.joblib` files out of
   `blackbox_pipeline/models/xgb`.
2. Run `glass_cascade.ipynb` end to end in a clean kernel.
3. Then run `blackbox_cascade.ipynb` end to end in a clean kernel.
4. Run the audit in the same environment.

Re-running GLASS cells, or black-box Stages 1–2, after the arbiters are
trained breaks their recorded upstream fingerprints, and the audit then fails.

## 8. Limitations, known gaps and deferred work

**Methodological** (from `research_roadmap.md`):

* **F24:** tuning parameters are selected globally. Cross-fitted outputs are
  therefore out of fold conditional on fixed hyperparameters; this is not a
  fully nested evaluation.
* **F39:** Stage 4 stacking may retain upstream training dependence despite
  meta-level OOF separation. The sensitivity experiments are planned for
  PR 45.
* **F17 / D1:** GLASS Pass 2 is fitted on the `full` population and the RF
  Router's on the `oof_remainder` by design. The tests verify each declared
  population, not which is preferable; the decision is deferred to PR 37.
* **F29 / D2:** reading the RF Router's band rate as the Meta-EBM's router
  evidence is validated for the current probability construction. F31 (future
  review) reassesses it if that construction changes.

**Testing gaps:**

* **Synthetic scale.** The fast and slow tiers use small synthetic data. They
  establish contracts and invariants, not behaviour at production scale or on
  real-data pathologies.
* **No coverage or CI.** No line or branch coverage was measured, and no
  coverage figure is claimed. No CI workflow is configured.
* **Notebooks are not executed by the suite.** Their clean-kernel runs (P6)
  were manual. Results quoted in notebook markdown (for example in GLASS
  Cell 8) are not checked against the cell outputs.
* **The real-artifact audit covers one regeneration on one machine.** It must
  be re-run after any regeneration or move, because the arbiters record
  absolute paths. It runs only on request and depends on the library versions
  used to train the models.
* **Saved-artifact coverage is incomplete.** The black-box Cell 38 crossover
  comparisons are recomputed by the audit, not saved. The GLASS RF artifact
  (baseline parameters for black-box Cell 19), the single-pass RF ablation and
  the Stage 3 block cache are outside the real-artifact audit; the cache is
  covered by the synthetic tests.
* **Execution-order check.** On Python 3.12 and later, a comprehension's loop
  variable counts as a binding, so a later cell that reads it would not be
  flagged. No current notebook does this.
* **The `notebook` marker** is unused. Like `realdata` before its opt-in
  guard, it would be selected by `-m "not slow"`.

## 9. What the testing establishes, and what it does not

**It establishes**, for the code and artifacts tested:

* the pipelines enforce their leakage, alignment, out-of-fold and provenance
  contracts, demonstrated by recorded calls and deliberate corruptions;
* artifacts produced by the current code pass the shared Stage 4 contract in
  both arms and in all four cells of the 2×2;
* the regenerated artifacts are complete and consistent: identities,
  F12/F17 evidence, recorded upstream files, clean reloads, reproduced test
  probabilities, and held-fixed crossover comparisons;
* the notebooks' source agrees with the protocol: imports, execution order,
  artifact locations and defaults.

**It does not establish:**

* how well any model performs, or that differences between the GLASS and
  black-box cascades are real or statistically significant;
* that tuned hyperparameters are optimal, or that the evaluation is fully
  nested (F24), or that Stage 4 is free of upstream dependence (F39);
* behaviour on other datasets, splits or time periods;
* that notebook narrative and markdown figures match their outputs;
* the validity of artifacts on any other machine or after a later
  regeneration — the real-artifact audit must be re-run there;
* the absence of defects outside the stated contracts.

## 10. Final audit

The Commit 4 final audit (P7) rebuilt the source from the branch base
`aa51194` and the approved patches, and checked the following:

* the rebuilt tree was byte-identical to the tested tree, with no change
  outside `prototype/bank_pipeline`;
* `git diff --check` was clean, and no new lint findings were introduced;
* every production change was a contract guard, the new cache, an import
  cleanup, a docstring or the pandas-compatible column drop, with no change to
  thresholds, tuning, calibration, populations or features;
* GLASS Pass 2 `full` and RF `oof_remainder` were unchanged;
* the specialization flags were off;
* the deferred findings were untouched.

The audit found no blocking implementation defects in the reviewed changes.
