# PR 42 Commit 2 — Feature inventory and frozen references

The accepted features, feature contracts and stage feature populations of both cascades, traced to their
definitions and callers on `pr-42-feature-research-contracts`, and the frozen references the research commits
compare against. Training side only: how a feature is applied to the test split is stated from the production
code, never by reading the test split.

**Sources of truth.** The declarations live in `feature_research/feature_inventory.py` (`NODES`,
`POPULATIONS`, `DUPLICATE_PAIRS`) and the references in `feature_research/frozen_references.py`. The tables
between the `GENERATED` markers below are their output (`python -m feature_research.feature_inventory`);
`tests/feature_research/test_pr42_feature_inventory.py` fails when this file and the code disagree. Every run
of `feature_research/feature_research.ipynb` writes the same content, with the training-row checks, to
`research_logs/pr42/<run_id>/feature_inventory.json`.

## How each claim is established

| Claim | How it is checked | Where |
|---|---|---|
| Each population is what the stage consumes | equal to the production registry it names; the cascade cells that build or read it use what it says (AST of G and B; nothing executed) | `check_registries`, `check_call_sites` (notebook Cell 5) |
| Each node exists and the selections hold | every declared node is produced by each implementation that claims it; the training-path bins equal the test-path binning on the same rows; every bin family is exclusive and complete | `check_dag_columns` (Cell 7) |
| How NSD encodes its own fit rows | every production NSD fit has more than 10,000 rows, so it samples 5,000 reference rows; the full-train fits of both implementations keep 5,000 | `check_nsd_reference` (Cell 7) |
| Target dependence (`none` / `fitted` / `inherited`) | each implementation fitted on the training rows with true and with permuted labels; a seeded sample of those rows transformed by both; a column that changes depends on the labels (a bin through its family or its source) | `probe_target_dependence` (Cell 8) |
| Fitting scope, fold refits, calibration | the production defaults and configurations (`preprocessing_record`, `CalibratedLRStage`, `Stage1MLPConfig`, `STAGE1_CALIBRATION_FAMILY`, `EBMStage3Config`, `XGBStage3Config`, the Stage 2 pathway record) and the sigmoid family G 8 and B 9 pass | `check_fit_protocols`, `check_call_sites` (Cells 5, 9) |
| Fit rows and the rule outside the fit | read from the production code (G3 columns); not executed on test rows | this document |
| Frozen references | recomputed from code (and from training rows for the population) and compared with the PR 39 records, read after their `.sha256` sidecars | `frozen_references` (Cell 10) |
| Accepted artifacts carry the frozen contracts | read only through the PR 38 boundary extractors at the PR 39 Commit 4 manifest's SHA-256; what the extractors do not expose is not read | `accepted_artifact_views` (Cell 11) |
| Duplicated implementations | function pairs read from the code; importers found by AST; agreement measured on training rows with `compare_values` (one-sided missing values count as differences; maximum over comparable finite pairs) | `DUPLICATE_PAIRS`, `duplicate_callers`, `duplicate_parity` (Cell 12) |

## Facts recorded for the later commits

These are observations of the code and the records, not assessments; Commits 3–6 own the assessments.

1. **Target-derived features consumed.** Stage 1: two of three (`subscribe_rate_by_euribor_bin`,
   `subscribe_rate_by_dow_month`). Stage 2: 12 of the 29 bins (the NSD, ECI and DOW families), from three
   label-dependent sources (`neighborhood_subscription_density`, `economic_curvature_intensity`,
   `dow_month_encoded`). Stage 3: three of thirteen (`decay_x_density` through NSD,
   `economic_curvature_intensity`, `dow_month_encoded`). Every other consumed feature is a function of the row.
2. **Final fits versus cross-validation.** Every label-dependent statistic is refit on each fold's training rows
   in every Stage 1–3 cross-validation (G2). In Stage 1, both arms fit the final base model (LR or MLP, with its
   scaler) on the full-train feature frame, but fit the final calibrator on the out-of-fold base scores. In
   Stage 3, the refit model is fitted on the full-train frame, while its calibrator and operating thresholds are
   fitted on out-of-fold scores.
3. **How a fit encodes its own rows** (G3, "Fit rows"). For the bin and cell rates (Stage 1's two encodings,
   `euribor3m_local_rate`, the three curvature intermediates, `dow_month_encoded`), every fit row is counted in
   the rate it is mapped to, so its own label enters its value. For `neighborhood_subscription_density`, only
   the reference rows' labels contribute: every production fit has more than 10,000 rows, so the reference set
   is 5,000 fit rows sampled with seed 42. A sampled row's own label enters its density; every other fit row is
   transformed against the reference set, as a held-out or test row is. `decay_x_density` and the NSD bins
   follow NSD.
4. **Fixed constants are not refit.** The crisis thresholds and weights, the high-conversion months, the CPI
   cut, the behavioural weights and the 29 bin thresholds are constants chosen during research (cited in the
   code to Cells 10A–10F and 13C); the code does not record the rows they were chosen on. The bin thresholds were
   chosen on all training labels (the PR 39 Step 3A limitation).
5. **Two implementations of the Stage 2/3 DAG.** Stage 2 runs the self-contained copies in
   `shared/stage2/feature_engineering.py`; Stage 3 runs `feature_research/feature_engineering/` through
   `EBMFeaturePipeline`. Besides the three modules notebook map section 5 names (crisis, integrals,
   derivatives), the copy also inlines prior, overlap and its own day × month encoder. The copy's quantile bins
   re-cut with pandas' rounded `IntervalIndex` categories where the research module uses the exact edges, and it
   imputes leftover NaN/inf with the transformed frame's median where the research module raises. Stage 1's
   `subscribe_rate_by_euribor_bin` uses the same `IntervalIndex` approach. Cell 12 measures where the two
   implementations disagree on training rows (in-sample and on the shared folds' held-out rows) and records it
   under `duplicates.parity`.
6. **Research registries are not the consumed populations.** `feature_research.feature_engineering.LR_FEATURES`
   holds the historical Stage 1 names (contract 1.0.0) and `EBM_FEATURES` 14 features (the production 13 plus
   `overlap_behavioral_score`); only `RF_FEATURES` equals its production counterpart.
7. **What the PR 38 extractors do not expose is not read.** The black-box MLP and XGBoost have no extractor and
   are not opened; their feature populations are the code's, and the PR 39 Commit 4 record holds the XGBoost
   features and block fingerprint. The EBM extractor does not expose the stored interaction pairs, so they are
   stated from `define_ebm_interactions` (G5). The Stage 3 block cache holds the test split, so it is identified
   by size and SHA-256 and not opened.
8. **The block-cache key depends on line endings.** `shared.stage3.block_cache` hashes module source bytes
   without normalising line endings, so the same code gives a different cache key on an LF and a CRLF checkout
   (a rebuild, not a different block). The LF-normalised DAG digest below does not.

## Local verification

Items marked `local verification` are held by Git-ignored files. `tests/realdata/test_pr42_frozen_references.py`
(`pytest -m realdata`) checks them without unpickling anything outside the PR 38 extractors:

- the four extractable Stage 1–3 artifacts (GLASS LR, both routers, GLASS EBM) pass their boundary extractors
  at their pinned SHA-256, and the two black-box models are not opened;
- through the extractors, the LR's feature-name check and the EBM's feature, `per_fold` fit-scope, fold-plan and
  per-fold calibration-gate checks pass, and the EBM's block fingerprint is the recorded one;
- the accepted Stage 3 block cache matches its recorded size and SHA-256.

## Generated tables

<!-- BEGIN GENERATED: python -m feature_research.feature_inventory -->

### G1. Stage feature populations

| Stage | Arm | Model | Consumed features | Registry | Contract | Notebook cells |
|---|---|---|---|---|---|---|
| 1 | glass | Calibrated Logistic Regression | 3: `cellular_downturn`, `subscribe_rate_by_euribor_bin`, `subscribe_rate_by_dow_month` | `shared.stage1.feature_engineering.LR_FEATURES` | stage1-features 2.0.0 (STAGE1_FEATURE_CONTRACT) | G 5 `engineer_features`; G 8 `train_calibrated_lr_stage` |
| 1 | blackbox | Calibrated MLP | 3: `cellular_downturn`, `subscribe_rate_by_euribor_bin`, `subscribe_rate_by_dow_month` | `blackbox_pipeline.models.mlp.features.STAGE1_FEATURES` | stage1-features 2.0.0 (assert_matches_glass) | B 7 `engineer_features`; B 9 `fit` |
| 2 | glass | GLASS Router (two-pass rules, union-aware ILP) | 29: `nsd_cold`, `nsd_warm`, `nsd_elevated`, `nsd_hot`, `jed_cold`, `jed_transition`, `jed_warm`, `jed_hot`, `cci_low`, `cci_dead_mid`, `cci_hot1`, `cci_valley`, `cci_hot2`, `eci_cold`, `eci_mid`, `eci_warm`, `eci_hot`, `dow_cold`, `dow_low`, `dow_mid`, `dow_hot`, `behav_cold`, `behav_baseline`, `behav_warm`, `behav_hot`, `campaign_fresh`, `campaign_moderate`, `campaign_heavy`, `cpi_cellular` | `shared.stage2.binning.RF_FEATURES_BINARY` | stage2-binning identity (binning_identity) | G 8B `load_pinned_stage2` |
| 2 | blackbox | Two-pass Random Forest Router | 29: `nsd_cold`, `nsd_warm`, `nsd_elevated`, `nsd_hot`, `jed_cold`, `jed_transition`, `jed_warm`, `jed_hot`, `cci_low`, `cci_dead_mid`, `cci_hot1`, `cci_valley`, `cci_hot2`, `eci_cold`, `eci_mid`, `eci_warm`, `eci_hot`, `dow_cold`, `dow_low`, `dow_mid`, `dow_hot`, `behav_cold`, `behav_baseline`, `behav_warm`, `behav_hot`, `campaign_fresh`, `campaign_moderate`, `campaign_heavy`, `cpi_cellular` | `shared.stage2.binning.RF_FEATURES_BINARY` | stage2-binning identity (binning_identity) | B 16 `engineer_features`; B 17 `engineer_rf_features`; B 18 `load_pinned_stage2` |
| 3 | glass | Explainable Boosting Machine | 13: `decay_x_density`, `euribor3m_sigmoid_slope`, `economic_curvature_intensity`, `cons.conf.idx`, `age`, `dow_month_encoded`, `default`, `prior_x_stress`, `campaign`, `overlap_default_clean`, `cpi_high_cellular`, `behavioral_favorability`, `emp_var_rate_sigmoid_slope` | `shared.stage3.feature_engineering.EBM_FEATURES` | EBM_FEATURES, 13 features; interactions define_ebm_interactions | G 17 `load_or_build_block`; G 18 `train_ebm_stage` |
| 3 | blackbox | XGBoost | 13: `decay_x_density`, `euribor3m_sigmoid_slope`, `economic_curvature_intensity`, `cons.conf.idx`, `age`, `dow_month_encoded`, `default`, `prior_x_stress`, `campaign`, `overlap_default_clean`, `cpi_high_cellular`, `behavioral_favorability`, `emp_var_rate_sigmoid_slope` | `shared.stage3.feature_engineering.EBM_FEATURES` | EBM_FEATURES, 13 features (the same block as the EBM) | B 30 `from_global_split`; B 31 `train_xgb_stage` |
| 4 | glass | GLASS Arbiter (weighted confidence) | 3: `lr`, `glass`, `ebm` | `glass_pipeline.glass_arbiter.loader.load_stage_outputs` | Stage 4 contract (shared.stage4); no engineered or raw features | G 19 `train_glass_arbiter` |
| 4 | blackbox | Meta-XGBoost | 6: `stage1_proba`, `stage2_proba`, `stage2_pass1`, `stage2_pass2`, `stage2_abstain`, `stage3_proba` | `blackbox_pipeline.models.meta_xgb.config.STAGE4_FEATURES` | STAGE4_FEATURES (6 channels; FEATURE_CONTRACT) | B 35 `MetaXGBConfig`; B 36 `train_meta_xgb_stage` |

Stage 2 sources (both routers, before binning): `campaign`, `neighborhood_subscription_density`, `cons.conf.idx`, `economic_curvature_intensity`, `joint_economic_decay`, `dow_month_encoded`, `behavioral_favorability`, `cpi_high_cellular`.

EBM interaction pairs (Stage 3, GLASS arm only): `campaign` x `emp_var_rate_sigmoid_slope`; `euribor3m_sigmoid_slope` x `economic_curvature_intensity`; `prior_x_stress` x `decay_x_density`.

### G2. Fitting scope by stage

| Populations | Final fit (full training split) | Cross-validation | Fold refit | Test-time application | Where CV and final fitting differ |
|---|---|---|---|---|---|
| stage1.glass, stage1.blackbox | engineer_features(X_train, X_test, y_train) fits LRFeatureEngineer once on all training rows; the final base model (LR / MLP with the tuned hyperparameters) and its StandardScaler are fitted on that full-train feature frame. The final calibrator (sigmoid in both arms) is fitted on the out-of-fold base scores of all training rows, not on scores of the full-train frame, and is composed with the final base model (shared.stage1.oof_calibration, step 4) | 10-fold Optuna tuning, the 10 outer folds that produce the out-of-fold base scores, and the nested calibration (10 outer x 10 inner folds) each refit LRFeatureEngineer and the StandardScaler on the fold's training rows from the raw rows (shared.stage1.crossfit build_fold_designs / inner_fold_designs; audits F3, F4, F19). The out-of-fold calibrated probabilities (threshold selection, the Stage 4 train column) come from a calibrator per outer fold fitted on that fold's inner out-of-fold scores; seed 42 | yes (per fold, from X_train_raw) | the final base model scores the full-train engineer's transform of X_test (with the full-train scaler); the final calibrator maps those scores | the final base model is fitted on full-train encodings, in which each training row's own label enters both target rates (G3, 'Fit rows'); the final calibrator is fitted on out-of-fold base scores; every cross-validated output uses per-fold encoders and scalers |
| stage2.glass, stage2.blackbox | PR 38 primary run c3_20261112_153333 (stage2_crossfit.runner.fit_full_train): engineer_features(X_train, X_test, y_train) fits RFFeatureEngineer on all training rows; the 8 sources are binned with BINNING_STRATEGY; the routers' own inner folds read FoldLabelExcludedFeatures bins | 5-fold outer cross-fit (shared plan, seed 42): engineer_features(X_tr, X_va, y_tr) refits RFFeatureEngineer on each outer fold's training rows and transforms the held-out fold; inside each outer fit, inner fold j's bins come from FoldLabelExcludedFeatures (pathway fold_label_excluded: label statistics recounted without fold j; scaler, KDE reference sample and quantile edges kept from the outer fit) | yes (outer folds refit; inner folds label-excluded); bin thresholds never refit | the full-train engineer transforms X_test and the same thresholds bin it (_apply_binning_thresholds); the full-train routers score it | cross-fit outputs (Stage 4 train side) come from outer-fold fits; test-side outputs from the full-train fit. The notebooks fit no Stage 2: G 8B and B 18 read both routers by pinned path and SHA-256; B 17 builds full-train bins for the split checks only |
| stage3.glass, stage3.blackbox | Stage3Block.build fits EBMFeaturePipeline on all training rows (drop leaky -> crisis -> integrals -> derivatives -> temporal -> prior -> overlap -> select), then FeatureCleaner (inf -> NaN -> training median); the refit model (EBM / XGBoost with the tuned hyperparameters) is fitted on that full-train frame. The final calibrator (isotonic, applied when the out-of-fold ECE exceeds 0.05) and the operating thresholds are fitted on the out-of-fold scores of all training rows (shared.stage3.runner) | feature_fit_scope 'per_fold': a fresh pipeline and cleaner on each fold's training rows for the 5 tuning folds (tuning, out-of-fold scores), the 10 reporting folds and the 5 x 5 inner folds (F22). The training-side calibrated scores, nested thresholds and decisions Stage 4 reads come from each outer fold's gate, map and threshold fitted on its inner out-of-fold scores; seed 42 | yes (per fold) | the refit model scores the full-train pipeline's and cleaner's transform of X_test; the final calibrator and operating thresholds apply. Each tuning fold's pipeline also transforms X_test (FoldFrames.X_test) | the refit model is fitted on full-train features (G3, 'Fit rows', says how each label-dependent column encodes its own fit rows); its calibrator and thresholds are fitted on out-of-fold scores; the training-side outputs are fold-nested. GLASS G 17 reuses a cached block only when shared.stage3.block_cache.block_identity matches; black-box B 30 rebuilds it; both recorded block fingerprint 9cc10fa1... in PR 39 Commit 4 |
| stage4.glass, stage4.blackbox | the arbiter is fitted on the training-side OOF outputs of Stages 1-3 (shared.stage4 contract checks split, index, labels, folds, router semantics) | inputs are upstream OOF predictions; Meta-XGB is cross-fitted by the shared Stage 3 runner (5 folds); the GLASS Arbiter's weights, recall cuts and abstention cut are selected on the training OOF table | n/a (no engineered features) | the upstream full-train / refit outputs on X_test | train side = OOF predictions, test side = full-train / refit predictions (the OOF-versus-refit input shift is PR 43 Step 43G) |

### G3. Per-feature inventory

Arms: `G` GLASS, `B` black-box. Target dependence: `none` (a function of the row), `fitted` (a statistic of the fit's training labels), `inherited` (computed from a fitted input), `stage_output` (an upstream prediction). Fitting scope and fold refits are the stage's (G2). 'Fit rows' states how a label-dependent column encodes the rows of the fit that computed it (the final fit's training rows, or a fold's training rows); every other row follows the rule outside the fit.

**Stage 1**

| Feature | Arms | Kind | Definition | Source variables | Target | Fitted ancestors | Fit rows (the fit's own rows) | Rule outside the fit (held-out and test rows) | Fixed constants |
|---|---|---|---|---|---|---|---|---|---|
| `cellular_downturn` | G/B | engineered | (contact == 0) and crisis score >= 3 (historically cellular_crisis) | `contact`, `emp.var.rate`, `euribor3m`, `nr.employed` | none |  |  | computed directly from the row | crisis thresholds and weights (euribor3m < 1.5: 3, emp.var.rate < -1.0: 2, nr.employed < 5100: 1; score >= 3) from the Cell 10A importance analysis |
| `subscribe_rate_by_euribor_bin` | G/B | engineered | training subscription rate of the row's euribor3m quantile bin (20 qcut bins, duplicates dropped; equal-width fallback); historically euribor3m_local_rate | `euribor3m` | fitted | `subscribe_rate_by_euribor_bin` | every fit row is counted in the rate of its own bin, so its own label enters its value | pd.cut with the fit's IntervalIndex categories; outside them -> median of the fitted bin rates |  |
| `subscribe_rate_by_dow_month` | G/B | engineered | (n\*p + 100\*p_global) / (n + 100) of the row's day_of_week x month cell (historically dow_month_encoded) | `month`, `day_of_week` | fitted | `subscribe_rate_by_dow_month` | every fit row is counted in its own day x month cell, so its own label enters its value (directly with weight 1 / (n_cell + 100), and through the fit's global rate) | lookup of the fit's cell (string key); unseen cell -> the fit's global rate |  |

**Stage 2**

| Feature | Arms | Kind | Definition | Source variables | Target | Fitted ancestors | Fit rows (the fit's own rows) | Rule outside the fit (held-out and test rows) | Fixed constants |
|---|---|---|---|---|---|---|---|---|---|
| `nsd_cold` | G/B | bin | NSD: neighborhood_subscription_density <= 0.0524 | `emp.var.rate`, `euribor3m`, `nr.employed` | inherited | `neighborhood_subscription_density` | as neighborhood_subscription_density | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `nsd_warm` | G/B | bin | NSD: 0.0524 < neighborhood_subscription_density <= 0.192 | `emp.var.rate`, `euribor3m`, `nr.employed` | inherited | `neighborhood_subscription_density` | as neighborhood_subscription_density | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `nsd_elevated` | G/B | bin | NSD: 0.192 < neighborhood_subscription_density <= 0.23 | `emp.var.rate`, `euribor3m`, `nr.employed` | inherited | `neighborhood_subscription_density` | as neighborhood_subscription_density | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `nsd_hot` | G/B | bin | NSD: neighborhood_subscription_density > 0.23 | `emp.var.rate`, `euribor3m`, `nr.employed` | inherited | `neighborhood_subscription_density` | as neighborhood_subscription_density | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `jed_cold` | G/B | bin | JED: joint_economic_decay <= 0.000352 | `emp.var.rate`, `euribor3m`, `nr.employed` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `jed_transition` | G/B | bin | JED: 0.000352 < joint_economic_decay <= 0.038 | `emp.var.rate`, `euribor3m`, `nr.employed` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `jed_warm` | G/B | bin | JED: 0.038 < joint_economic_decay <= 0.0856 | `emp.var.rate`, `euribor3m`, `nr.employed` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `jed_hot` | G/B | bin | JED: joint_economic_decay > 0.0856 | `emp.var.rate`, `euribor3m`, `nr.employed` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `cci_low` | G/B | bin | CCI: cons.conf.idx <= -46.2 | `cons.conf.idx` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `cci_dead_mid` | G/B | bin | CCI: -46.2 < cons.conf.idx <= -41.8 | `cons.conf.idx` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `cci_hot1` | G/B | bin | CCI: -41.8 < cons.conf.idx <= -40.0 | `cons.conf.idx` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `cci_valley` | G/B | bin | CCI: -40.0 < cons.conf.idx <= -36.1 | `cons.conf.idx` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `cci_hot2` | G/B | bin | CCI: cons.conf.idx > -36.1 | `cons.conf.idx` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `eci_cold` | G/B | bin | ECI: economic_curvature_intensity <= 0.0291 | `emp.var.rate`, `euribor3m`, `nr.employed` | inherited | `emp_var_rate_abs_curvature`, `euribor3m_abs_curvature`, `nr_employed_abs_curvature` | as economic_curvature_intensity | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `eci_mid` | G/B | bin | ECI: 0.0291 < economic_curvature_intensity <= 0.0758 | `emp.var.rate`, `euribor3m`, `nr.employed` | inherited | `emp_var_rate_abs_curvature`, `euribor3m_abs_curvature`, `nr_employed_abs_curvature` | as economic_curvature_intensity | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `eci_warm` | G/B | bin | ECI: 0.0758 < economic_curvature_intensity <= 0.0948 | `emp.var.rate`, `euribor3m`, `nr.employed` | inherited | `emp_var_rate_abs_curvature`, `euribor3m_abs_curvature`, `nr_employed_abs_curvature` | as economic_curvature_intensity | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `eci_hot` | G/B | bin | ECI: economic_curvature_intensity > 0.0948 | `emp.var.rate`, `euribor3m`, `nr.employed` | inherited | `emp_var_rate_abs_curvature`, `euribor3m_abs_curvature`, `nr_employed_abs_curvature` | as economic_curvature_intensity | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `dow_cold` | G/B | bin | DOW: dow_month_encoded <= 0.0653 | `month`, `day_of_week` | inherited | `dow_month_encoded` | as dow_month_encoded | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `dow_low` | G/B | bin | DOW: 0.0653 < dow_month_encoded <= 0.0895 | `month`, `day_of_week` | inherited | `dow_month_encoded` | as dow_month_encoded | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `dow_mid` | G/B | bin | DOW: 0.0895 < dow_month_encoded <= 0.127 | `month`, `day_of_week` | inherited | `dow_month_encoded` | as dow_month_encoded | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `dow_hot` | G/B | bin | DOW: dow_month_encoded > 0.127 | `month`, `day_of_week` | inherited | `dow_month_encoded` | as dow_month_encoded | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `behav_cold` | G/B | bin | BEHAV: behavioral_favorability <= 0.3 | `age`, `default`, `contact`, `month`, `previous` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `behav_baseline` | G/B | bin | BEHAV: 0.3 < behavioral_favorability <= 0.4 | `age`, `default`, `contact`, `month`, `previous` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `behav_warm` | G/B | bin | BEHAV: 0.4 < behavioral_favorability <= 0.5 | `age`, `default`, `contact`, `month`, `previous` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `behav_hot` | G/B | bin | BEHAV: behavioral_favorability > 0.5 | `age`, `default`, `contact`, `month`, `previous` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `campaign_fresh` | G/B | bin | CAMPAIGN: campaign <= 2 | `campaign` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `campaign_moderate` | G/B | bin | CAMPAIGN: 2 < campaign <= 5 | `campaign` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `campaign_heavy` | G/B | bin | CAMPAIGN: campaign > 5 | `campaign` | none |  |  | locked thresholds on the source value as the fit transforms the row | BINNING_STRATEGY thresholds, fixed constants chosen on all training labels (Cell 13C lift analysis); never refit |
| `cpi_cellular` | G/B | bin | cpi_high_cellular (passthrough, int8) | `contact`, `cons.price.idx` | none |  |  | the source value, cast to int8 |  |

**Stage 3**

| Feature | Arms | Kind | Definition | Source variables | Target | Fitted ancestors | Fit rows (the fit's own rows) | Rule outside the fit (held-out and test rows) | Fixed constants |
|---|---|---|---|---|---|---|---|---|---|
| `decay_x_density` | G/B | engineered | joint_economic_decay \* neighborhood_subscription_density | `emp.var.rate`, `euribor3m`, `nr.employed` | inherited | `neighborhood_subscription_density` | as neighborhood_subscription_density, its only label-dependent input | the row's joint_economic_decay times its density from the fit's reference rows |  |
| `euribor3m_sigmoid_slope` | G/B | engineered | euribor3m_sigmoid \* (1 - euribor3m_sigmoid) / 0.25 | `euribor3m` | none |  |  | computed directly from the row |  |
| `economic_curvature_intensity` | G/B | engineered | mean of the three \*_abs_curvature columns | `emp.var.rate`, `euribor3m`, `nr.employed` | inherited | `emp_var_rate_abs_curvature`, `euribor3m_abs_curvature`, `nr_employed_abs_curvature` | as its three \*_abs_curvature inputs | the mean of the row's three curvatures from the fit's bins |  |
| `cons.conf.idx` | G/B | raw | preprocessed column | `cons.conf.idx` | none |  |  | the row's value (no transformation) |  |
| `age` | G/B | raw | preprocessed column | `age` | none |  |  | the row's value (no transformation) |  |
| `dow_month_encoded` | G/B | engineered | (n\*p + 100\*p_global) / (n + 100) of the row's day_of_week x month cell | `month`, `day_of_week` | fitted | `dow_month_encoded` | every fit row is counted in its own day x month cell, so its own label enters its value (directly with weight 1 / (n_cell + 100), and through the fit's global rate) | lookup of the fit's cell; unseen cell -> the fit's global rate |  |
| `default` | G/B | raw | preprocessed column | `default` | none |  |  | the row's value (no transformation) |  |
| `prior_x_stress` | G/B | engineered | has_prior_contact \* economic_stress_integral | `previous`, `emp.var.rate`, `euribor3m`, `nr.employed` | none |  |  | computed directly from the row |  |
| `campaign` | G/B | raw | preprocessed column | `campaign` | none |  |  | the row's value (no transformation) |  |
| `overlap_default_clean` | G/B | engineered | low_stress_zone \* default_clean | `default`, `emp.var.rate`, `euribor3m`, `nr.employed` | none |  |  | computed directly from the row |  |
| `cpi_high_cellular` | G/B | engineered | (cons.price.idx >= 93.5) and (contact == 0) | `contact`, `cons.price.idx` | none |  |  | computed directly from the row | 93.5 from the Cell 10F MI analysis |
| `behavioral_favorability` | G/B | engineered | (3\*cellular_contact + 3\*high_conversion_month + 2\*has_prior_contact + 1\*default_clean + 1\*[age >= 50]) / 10 | `age`, `default`, `contact`, `month`, `previous` | none |  |  | computed directly from the row | weights 3:3:2:1:1 and age 50 from Cell 10F |
| `emp_var_rate_sigmoid_slope` | G/B | engineered | emp_var_rate_sigmoid \* (1 - emp_var_rate_sigmoid) / 0.25 | `emp.var.rate` | none |  |  | computed directly from the row |  |

**Stage 4**

| Feature | Arms | Kind | Definition | Source variables | Target | Fitted ancestors | Fit rows (the fit's own rows) | Rule outside the fit (held-out and test rows) | Fixed constants |
|---|---|---|---|---|---|---|---|---|---|
| `lr` | G | stage_output | GLASS Stage 1 calibrated LR probability (StageOutput) |  | stage_output |  |  | the upstream full-train / refit outputs on X_test |  |
| `glass` | G | stage_output | GLASS Router route (Pass 1 / Pass 2 / uncertain) and probability |  | stage_output |  |  | the upstream full-train / refit outputs on X_test |  |
| `ebm` | G | stage_output | GLASS Stage 3 EBM probability (Stage3Artifact) |  | stage_output |  |  | the upstream full-train / refit outputs on X_test |  |
| `stage1_proba` | B | stage_output | Stage 1 P(y=1): MLP StageOutput (calibrated) |  | stage_output |  |  | the upstream full-train / refit outputs on X_test |  |
| `stage2_proba` | B | stage_output | RF Router P(y=1); NaN where the router abstains |  | stage_output |  |  | the upstream full-train / refit outputs on X_test |  |
| `stage2_pass1` | B | stage_output | RF Router routed Pass 1 (NOT_SUBSCRIBE) |  | stage_output |  |  | the upstream full-train / refit outputs on X_test |  |
| `stage2_pass2` | B | stage_output | RF Router routed Pass 2 (SUBSCRIBE) |  | stage_output |  |  | the upstream full-train / refit outputs on X_test |  |
| `stage2_abstain` | B | stage_output | RF Router abstained |  | stage_output |  |  | the upstream full-train / refit outputs on X_test |  |
| `stage3_proba` | B | stage_output | XGBoost Stage3Artifact P(y=1), matched calibration space |  | stage_output |  |  | the upstream full-train / refit outputs on X_test |  |

### G4. Intermediate DAG nodes (not consumed directly)

| Node | Step | Definition | Inputs | Target | Fit rows | Used by |
|---|---|---|---|---|---|---|
| `economic_crisis_score` | crisis | 3\*[euribor3m < 1.5] + 2\*[emp.var.rate < -1.0] + 1\*[nr.employed < 5100] | `euribor3m`, `emp.var.rate`, `nr.employed` | none |  | input to `cellular_crisis` |
| `cellular_crisis` | crisis | (contact == 0) and economic_crisis_score >= 3 | `contact`, `economic_crisis_score` | none |  | unused by the cascades |
| `high_conversion_month` | crisis | month in {3, 9, 10, 12} | `month` | none |  | input to `behavioral_favorability` |
| `cellular_contact` | crisis | contact == 0 | `contact` | none |  | input to `behavioral_favorability` |
| `default_clean` | crisis | default == 0 | `default` | none |  | input to `behavioral_favorability`, `overlap_default_clean` |
| `economic_stress_integral` | integrals | (3\*max(0, (1.5 - euribor3m)/1.5) + 2\*max(0, -emp.var.rate/3) + 1\*max(0, (5200 - nr.employed)/100)) / 6 | `euribor3m`, `emp.var.rate`, `nr.employed` | none |  | input to `prior_x_stress`, `low_stress_zone` |
| `neighborhood_subscription_density` | integrals | Gaussian-kernel (bandwidth 1.0) weighted mean of reference rows' labels in standardized (euribor3m, nr.employed, emp.var.rate) space | `euribor3m`, `nr.employed`, `emp.var.rate` | fitted | only the reference rows' labels contribute. Every production fit has more than 10,000 rows (check_nsd_reference), so the reference set is 5,000 fit rows sampled with seed 42: a sampled row's own label enters its density (kernel weight 1 at distance 0); every other fit row is transformed against the reference set, as a held-out or test row is. Stage 2's label-excluded inner folds also drop the reference rows that fall in inner fold j | stage2.glass (source); stage2.blackbox (source); input to `decay_x_density`, `nsd_cold`, `nsd_warm`, `nsd_elevated`, `nsd_hot` |
| `euribor3m_sigmoid` | derivatives | expit(-(euribor3m - 1.5) / 0.5) | `euribor3m` | none |  | input to `euribor3m_sigmoid_slope` |
| `emp_var_rate_sigmoid` | derivatives | expit(-(emp.var.rate + 1.0) / 0.5) | `emp.var.rate` | none |  | input to `emp_var_rate_sigmoid_slope` |
| `euribor_decay` | derivatives | exp(-0.50 \* max(0, euribor3m - 0.60)) | `euribor3m` | none |  | input to `joint_economic_decay` |
| `nr_employed_decay` | derivatives | exp(-0.02 \* max(0, nr.employed - 4964)) | `nr.employed` | none |  | input to `joint_economic_decay` |
| `emp_var_decay` | derivatives | exp(-0.80 \* max(0, emp.var.rate + 2.0)) | `emp.var.rate` | none |  | input to `joint_economic_decay` |
| `euribor3m_local_rate` | derivatives | training subscription rate of the row's euribor3m quantile bin (20 bins) | `euribor3m` | fitted | every fit row is counted in the rate of its own bin, so its own label enters its value | unused by the cascades |
| `euribor3m_abs_curvature` | derivatives | \|second difference\| of the training subscription rates over 20 euribor3m quantile bins | `euribor3m` | fitted | every fit row is counted in its bin's rate; that rate enters the curvature of its bin and of the neighbouring bins (two np.gradient passes), so its own label enters its value | input to `economic_curvature_intensity` |
| `nr_employed_abs_curvature` | derivatives | \|second difference\| of the training subscription rates over 20 nr.employed quantile bins | `nr.employed` | fitted | every fit row is counted in its bin's rate; that rate enters the curvature of its bin and of the neighbouring bins (two np.gradient passes), so its own label enters its value | input to `economic_curvature_intensity` |
| `emp_var_rate_abs_curvature` | derivatives | \|second difference\| of the training subscription rates over 20 emp.var.rate quantile bins | `emp.var.rate` | fitted | every fit row is counted in its bin's rate; that rate enters the curvature of its bin and of the neighbouring bins (two np.gradient passes), so its own label enters its value | input to `economic_curvature_intensity` |
| `joint_economic_decay` | derivatives | euribor_decay \* emp_var_decay \* nr_employed_decay | `euribor_decay`, `emp_var_decay`, `nr_employed_decay` | none |  | stage2.glass (source); stage2.blackbox (source); input to `decay_x_density`, `jed_cold`, `jed_transition`, `jed_warm`, `jed_hot` |
| `has_prior_contact` | prior | previous >= 1 | `previous` | none |  | input to `prior_x_stress`, `behavioral_favorability` |
| `low_stress_zone` | overlap | economic_stress_integral < 0.2 | `economic_stress_integral` | none |  | input to `overlap_default_clean`, `overlap_behavioral_score` |
| `overlap_behavioral_score` | overlap | low_stress_zone \* behavioral_favorability | `low_stress_zone`, `behavioral_favorability` | none |  | unused by the cascades |

### G5. Frozen references

| Reference | Value on this branch | Recorded value | Recorded in | Status | Note |
|---|---|---|---|---|---|
| Stage 1 feature contract 2.0.0 | stage1-features 2.0.0 ['cellular_downturn', 'subscribe_rate_by_euribor_bin', 'subscribe_rate_by_dow_month'] fingerprint d0d8b07f2ca5e81d | stage1-features 2.0.0 ['cellular_downturn', 'subscribe_rate_by_euribor_bin', 'subscribe_rate_by_dow_month'] fingerprint d0d8b07f2ca5e81d | research_logs/pr39/c4_20261115_204125/run_record.json identities.feature_contracts.stage1.contract | verified |  |
| Stage 1 ordered names SHA-256 | d0d8b07f2ca5e81dad32ba0bdb33e8ddf2671f7a764505ebc76ae11f5d4be513 | d0d8b07f2ca5e81dad32ba0bdb33e8ddf2671f7a764505ebc76ae11f5d4be513 | research_logs/pr39/c4_20261115_204125/run_record.json identities.feature_contracts.stage1.sha256 | verified |  |
| Stage 1 historical contract 1.0.0 (read-only) | ['cellular_crisis', 'euribor3m_local_rate', 'dow_month_encoded'] fingerprint dae2c0b25f683f3a; renames {'cellular_crisis': 'cellular_downturn', 'euribor3m_local_rate': 'subscribe_rate_by_euribor_bin', 'dow_month_encoded': 'subscribe_rate_by_dow_month'} | - | shared/stage1/feature_engineering.py | code only | names every pre-PR 39 artifact carries; read as saved, never written |
| Stage 2 binning identity (SHA-256) | 9cdd79aff9d61101949a7e761c6837e13e81036d865c37b5213dca2867912376 | 9cdd79aff9d61101949a7e761c6837e13e81036d865c37b5213dca2867912376 | research_logs/pr39/c4_20261115_204125/run_record.json identities.stage2_binning | verified |  |
| Stage 2 binning identity, PR 39 Step 3A decision | 9cdd79aff9d61101949a7e761c6837e13e81036d865c37b5213dca2867912376 | 9cdd79aff9d61101949a7e761c6837e13e81036d865c37b5213dca2867912376 | research_logs/pr39/c3_20261114_231604/stage2_binning_evidence.json decision.binning_identity (no sidecar) | verified | outcome 'retain' for NSD and ECI |
| Stage 2 ordered 29 bins SHA-256 | 7b0d4f507f3becd46be3aee95e9590e94cdbdf2a5a218b82cce5c45afb2e85f5 | 7b0d4f507f3becd46be3aee95e9590e94cdbdf2a5a218b82cce5c45afb2e85f5 | research_logs/pr39/c4_20261115_204125/run_record.json identities.feature_contracts.stage2.sha256 | verified |  |
| Stage 2 sources (8, ordered) SHA-256 | 9b8043ff6a5977a298d6ae08d59df5f7b41f3cc87f3ea51ee6b065357d6f4c92 | - | shared/stage2/feature_engineering.py RF_SOURCE_FEATURES | code only |  |
| Stage 2 inner-fold feature pathway | {'pathway': 'fold_label_excluded', 'record': ['stage2-inner-fold-feature-isolation', '1.0.0', '0d4953ef34d1d3dc']} | {'pathway': 'fold_label_excluded', 'record': ['stage2-inner-fold-feature-isolation', '1.0.0', '0d4953ef34d1d3dc']} | research_logs/pr39/c4_20261115_204125/run_record.json identities.configuration.feature_pathway | verified |  |
| Stage 3 ordered 13 features SHA-256 | 8f1df13bf14ca4722665ca60e0d2b792c6c2148791452c5e9303f1b39c309a83 | 8f1df13bf14ca4722665ca60e0d2b792c6c2148791452c5e9303f1b39c309a83 | research_logs/pr39/c4_20261115_204125/run_record.json identities.feature_contracts.stage3.sha256 | verified |  |
| Stage 3 features of both accepted Stage 3 artifacts | decay_x_density, euribor3m_sigmoid_slope, economic_curvature_intensity, cons.conf.idx, age, dow_month_encoded, default, prior_x_stress, campaign, overlap_default_clean, cpi_high_cellular, behavioral_favorability, emp_var_rate_sigmoid_slope | decay_x_density, euribor3m_sigmoid_slope, economic_curvature_intensity, cons.conf.idx, age, dow_month_encoded, default, prior_x_stress, campaign, overlap_default_clean, cpi_high_cellular, behavioral_favorability, emp_var_rate_sigmoid_slope | research_logs/pr39/c4_20261115_204125/run_record.json evidence.stage3.stage3.ebm / stage3.xgb .features | verified |  |
| Stage 3 feature DAG implementation (LF SHA-256 over its modules) | 3a8d195700ce34627f7d10d59294212ba967bcd69fe14a1c5741e87e25720017 (11 modules: feature_research/config.py, feature_research/feature_engineering/_checks.py, feature_research/feature_engineering/crisis.py, feature_research/feature_engineering/derivatives.py, feature_research/feature_engineering/integrals.py, feature_research/feature_engineering/overlap.py, feature_research/feature_engineering/prior.py, feature_research/feature_engineering/temporal.py, shared/stage3/data.py, shared/stage3/feature_engineering.py, shared/stage_io.py) | - | no PR 39 record holds per-module digests | code only | the accepted block cache stores the block_cache identity but is not opened (it holds the test split); these LF digests, written by every PR 42 run, are the reference later commits compare with |
| Stage 3 block content fingerprint | not recomputed: the block holds the test split | 9cc10fa1f5cf8d858a9168a2169372758102d06283592bed024fb06e1040318c | research_logs/pr39/c4_20261115_204125/run_record.json evidence.stage3 (stage3.ebm and stage3.xgb) | recorded only | the accepted EBM's own block_fingerprint is compared through extract_ebm_view when the artifact is local |
| Stage 3 block-cache file | models/stage3/stage3_block_d6cddadc3439_c950583f1d15.joblib | 185921179 bytes, SHA-256 8755876ce20ad3d6595cae2b294438207d4f01e0d631e13c4be7e9dd24917039 | research_logs/pr39/c4_20261115_204125/artifact_manifest.json inventory.files | local verification | identified by size and SHA-256 only, never unpickled (it holds the test split); name = stage3_block_<stage_io split fingerprint>_<block_identity digest[:12]> |
| EBM interaction pairs (3) | campaign x emp_var_rate_sigmoid_slope; euribor3m_sigmoid_slope x economic_curvature_intensity; prior_x_stress x decay_x_density | - | glass_pipeline/ebm/interactions.py define_ebm_interactions | code only | verified: the production definition, which train_ebm_stage passes to the EBM factory and stores as extras['interactions'] (tests/stage3/test_glass_ebm_stage.py), yields these pairs for the 13 EBM_FEATURES. Not verified: the pairs stored in the accepted EBM; the PR 38 EBM extractor does not expose extras, and the model is not opened outside it |
| Stage 4 Meta-XGB inputs SHA-256 | 6e2be2c81f71c82e03cdbe1e78513d04f61e601863b5109a2018103a61c7a777 | 6e2be2c81f71c82e03cdbe1e78513d04f61e601863b5109a2018103a61c7a777 | research_logs/pr39/c5_20261116_153459/run_record.json identities.feature_contracts.stage4_meta_xgb.sha256 | verified |  |
| Train-side population | verified on training rows by the notebook | rows 32950, positives 3712, label b984d2a50527, shared 5-fold 8aea082fcaf371b7, Stage 1 10-fold e4da0b4442f71ecf, router split fc3e73a5ef91aea4, stage_io split d6cddadc3439 (attested) | research_logs/pr39/c4_20261115_204125/run_record.json identities.population | recorded only |  |

### G6. Duplicated feature-engineering implementations

| `shared/stage2/feature_engineering.py` | `feature_research/feature_engineering/` | Difference (from code) |
|---|---|---|
| `_add_crisis_features` | `crisis.add_crisis_features` | same constants and outputs |
| `_IntegralFeatureEngineer` | `integrals.IntegralFeatureEngineer` | same scaler, KDE and reference sampling; research checks X/y index alignment (check_xy_aligned); seed literal 42 vs config.RANDOM_SEED (42) |
| `_add_prior_features` | `prior.add_prior_features` | same outputs; research also requires the 'previous' column explicitly |
| `_DerivativeFeatureEngineer` | `derivatives.DerivativeFeatureEngineer` | bins: the copy re-cuts with pandas IntervalIndex categories (edges rounded to 3 decimals), research with the exact numeric edges; the copy's _fill_nans imputes any remaining NaN/inf with the median of the frame being transformed, research raises (assert_finite); research checks X/y alignment |
| `_TemporalFeatureEngineer` | `temporal.TemporalFeatureEngineer` | (sum + k\*g)/(count + k) on integer keys vs (count\*mean + k\*g)/(count + k) on string keys: equal up to floating-point rounding; research requires integer-encoded day_of_week and month |
| `_add_overlap_features` | `overlap.add_overlap_features` | same constants and outputs |
| `_economic_stress_integral, _apply_kde, _add_sigmoid_features, _add_decay_features, _add_composites` | `the same-named helpers in integrals.py and derivatives.py` | same formulas and constants |

| Related copy | Of | Note |
|---|---|---|
| `shared.stage1.feature_engineering.LRFeatureEngineer` | crisis.cellular_crisis (cellular_downturn), derivatives euribor3m_local_rate (subscribe_rate_by_euribor_bin, IntervalIndex variant), temporal.dow_month_encoded (subscribe_rate_by_dow_month, string-key variant) | Stage 1 production code; its own constants are mirrored, not imported |
| `feature_research.model_training.rf.binning` | shared.stage2.binning (BINNING_STRATEGY) | research port; checked equal by duplicate_parity |

### G7. Current callers

**`shared.stage2.feature_engineering`** (re-exported as `glass_pipeline.glass_router.rf.RFFeatureEngineer`, `glass_pipeline.glass_router.rf.engineer_features`, `shared.stage2.RFFeatureEngineer`, `shared.stage2.RF_SOURCE_FEATURES`, `shared.stage2.engineer_features`)

| Group | Import | File | Names |
|---|---|---|---|
| production | direct | `glass_pipeline/glass_router/pipeline/crossfit.py` | `engineer_features` |
| production | direct | `glass_pipeline/glass_router/rf/__init__.py` | `RFFeatureEngineer`, `engineer_features` |
| production | direct | `shared/stage2/__init__.py` | `RFFeatureEngineer`, `RF_SOURCE_FEATURES`, `engineer_features` |
| research | direct | `feature_research/encoding_audit.py` | `RFFeatureEngineer` |
| research | direct | `feature_research/feature_inventory.py` | `RFFeatureEngineer`, `RF_SOURCE_FEATURES`, `_apply_binning_thresholds`, `shared.stage2.feature_engineering` |
| research | direct | `feature_research/frozen_references.py` | `RF_SOURCE_FEATURES` |
| evidence | direct | `stage2_binning/runner.py` | `RFFeatureEngineer` |
| evidence | direct | `stage2_crossfit/acceptance.py` | `engineer_features` |
| evidence | direct | `stage2_crossfit/g13.py` | `RFFeatureEngineer`, `RF_SOURCE_FEATURES`, `_KDE_KEY_FEATURES`, `_KDE_LARGE_DS_THRESH`, `_apply_binning_thresholds` |
| evidence | direct | `stage2_crossfit/inner_features.py` | `RFFeatureEngineer` |
| evidence | direct | `stage2_crossfit/runner.py` | `engineer_features` |
| evidence | direct | `upstream_regeneration/checks.py` | `RF_SOURCE_FEATURES` |
| tests | direct | `tests/feature_research/test_pr42_encoding_audit.py` | `shared.stage2.feature_engineering` |
| tests | direct | `tests/stage2/conftest.py` | `engineer_features` |
| tests | direct | `tests/stage2/test_glass_router_fit.py` | `engineer_features` |
| tests | direct | `tests/stage2/test_glass_union_obligations.py` | `engineer_features` |
| tests | direct | `tests/stage2_crossfit/test_pr38_g13.py` | `RFFeatureEngineer`, `RF_SOURCE_FEATURES` |
| tests | direct | `tests/stage2_crossfit/test_step2b_isolation.py` | `engineer_features` |
| tests | direct | `tests/support/rf_router_world.py` | `engineer_features` |
| tests | via reexport | `tests/cascade/conftest.py` | `glass_pipeline.glass_router.rf.engineer_features` |
| notebook | via reexport | `glass_pipeline/glass_cascade.ipynb` cell 10 | `glass_pipeline.glass_router.rf.engineer_features` |
| notebook | direct | `blackbox_pipeline/blackbox_cascade.ipynb` cell 16 | `feature_engineering` |

**`feature_research.feature_engineering`** (re-exported as nothing)

| Group | Import | File | Names |
|---|---|---|---|
| production | direct | `shared/stage3/feature_engineering.py` | `DerivativeFeatureEngineer`, `IntegralFeatureEngineer`, `TemporalFeatureEngineer`, `add_crisis_features`, `add_overlap_features`, `add_prior_features` |
| research | direct | `feature_research/encoding_audit.py` | `_KDE_LARGE_DS_THRESH` |
| research | direct | `feature_research/feature_inventory.py` | `feature_engineering`, `integrals` |
| tests | direct | `tests/feature_research/conftest.py` | `FeaturePipeline` |
| tests | direct | `tests/feature_research/test_feature_pipeline.py` | `LIVE_FEATURES`, `finalize_features`, `get_stage_df` |
| tests | direct | `tests/feature_research/test_fold_engineering.py` | `EBM_FEATURES` |
| tests | direct | `tests/feature_research/test_pr42_encoding_audit.py` | `feature_research.feature_engineering.integrals` |
| tests | direct | `tests/feature_research/test_stage_trainers.py` | `EBM_FEATURES`, `LR_FEATURES` |
| notebook | direct | `feature_research/exploratory_feature_research.ipynb` cell 11 | `FeaturePipeline` |
| notebook | direct | `feature_research/exploratory_feature_research.ipynb` cell 12G | `LIVE_FEATURES`, `finalize_features`, `get_stage_df` |

<!-- END GENERATED -->
