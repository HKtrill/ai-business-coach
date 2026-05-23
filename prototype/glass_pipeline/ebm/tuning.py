"""
glass_pipeline.ebm.tuning
==========================
Recall-biased EBM trainer for Stage 3 — Optuna strategy.

Transfers the validated Optuna approach from
feature_research/model_training/ebm/trainer.py into the deployable cascade.

Key differences from research trainer
--------------------------------------
interactions    Fixed validated pairs from interactions.py are passed in
                directly — EBM models exactly those pairs. Optuna does not
                search over interaction count; that decision was made in research.
                max_interaction_bins is still tuned (resolution of the 2D surfaces).

n_trials        Default 150 — matches research for fair comparison.
                MedianPruner eliminates bad configurations early; the TPE
                sampler front-loads promising regions.

max_rounds      Ceiling 5000 — matches research search space.

return          (model, best_params, best_cv_score) — compatible with
                ebm_stage.py call site.

Preserved from research
-----------------------
- F2 objective (β=2, recall weighted 4× over precision)
- Balanced sample weights — _make_sample_weights() called identically in
  Optuna objective, 10-fold eval, and full-data refit. No drift between phases.
- TPESampler + MedianPruner (n_startup_trials=20, n_warmup_steps=2)
- _normalize_params / _build_model patterns
- 5-fold stratified CV inside Optuna objective
- 10-fold final evaluation after Optuna (matches research output format)
- n_jobs=1 throughout — no nested joblib parallelism
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    fbeta_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Internal helpers  (mirrors research trainer.py patterns)
# ---------------------------------------------------------------------------

def _make_sample_weights(y: pd.Series) -> np.ndarray:
    """Balanced sample weights — called identically in objective, eval, refit."""
    return compute_sample_weight("balanced", y)


def _normalize_params(trial_params: dict[str, Any]) -> dict[str, Any]:
    """Collapse Optuna trial params into canonical EBM constructor params."""
    return {
        "learning_rate":        trial_params["learning_rate"],
        "max_rounds":           trial_params["max_rounds"],
        "max_bins":             trial_params["max_bins"],
        "max_interaction_bins": trial_params["max_interaction_bins"],
    }


def _build_model(
    params: dict[str, Any],
    interactions: list[tuple[str, str]],
    random_state: int = 42,
) -> ExplainableBoostingClassifier:
    """Construct an EBM from canonical params + validated interaction pairs.

    n_jobs=1 throughout — no nested joblib parallelism warnings.
    """
    return ExplainableBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_rounds=params["max_rounds"],
        max_bins=params["max_bins"],
        max_interaction_bins=params["max_interaction_bins"],
        interactions=interactions,
        random_state=random_state,
        n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tune_ebm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    interactions: list[tuple[str, str]],
    n_trials: int = 150,
    n_folds: int = 5,
    n_folds_eval: int = 10,
    random_state: int = 42,
    study_name: str = "ebm_stage3_recall_biased",
) -> tuple[ExplainableBoostingClassifier, dict[str, Any], float]:
    """
    Optuna-tuned EBM with recall-biased F2 objective.

    Parameters
    ----------
    X_train       : Feature matrix (post select_ebm_features).
    y_train       : Binary target.
    interactions  : Validated interaction pairs from define_ebm_interactions().
                    Passed directly to EBM — not searched over by Optuna.
    n_trials      : Optuna trials. Default 150 (matches research).
    n_folds       : Stratified CV folds inside Optuna objective. Default 5.
    n_folds_eval  : Folds for final evaluation after Optuna. Default 10.
    random_state  : Global random seed.
    study_name    : Optuna study identifier.

    Returns
    -------
    model       : ExplainableBoostingClassifier fitted on full X_train.
    best_params : Canonical hyperparameters (post _normalize_params).
    best_cv_f2  : Best Optuna CV F2 score.
    """

    print("\n" + "=" * 80)
    print("🔮 EBM STAGE 3 TUNING — RECALL-BIASED (F2 Objective, β=2)")
    print("=" * 80)
    print(f"\n📊 Dataset: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"   Class distribution: {y_train.value_counts().to_dict()}")
    print(f"   Interaction pairs:  {interactions if interactions else 'none (additive only)'}")

    # ------------------------------------------------------------------
    # Data quality guard
    # ------------------------------------------------------------------
    print("\n🔍 Pre-tuning data quality check...")
    X_train = X_train.copy()
    if np.isinf(X_train).any().any():
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
    if X_train.isnull().any().any():
        nan_cols = X_train.columns[X_train.isnull().any()].tolist()
        print(f"   ⚠️  NaN in: {nan_cols} — imputing with median")
        for col in nan_cols:
            X_train[col] = X_train[col].fillna(X_train[col].median())
    assert not np.isinf(X_train).any().any(), "Infinity values remain after cleaning"
    assert not X_train.isnull().any().any(),   "NaN values remain after cleaning"
    print("   ✅ Data clean")

    # ------------------------------------------------------------------
    # Sample weights
    # ------------------------------------------------------------------
    sample_weights = _make_sample_weights(y_train)
    pos_weight = sample_weights[y_train == 1].mean()
    neg_weight = sample_weights[y_train == 0].mean()
    print(f"\n   ⚖️  Balanced sample weights: "
          f"class 0 → {neg_weight:.4f}, class 1 → {pos_weight:.4f}")
    print(f"      (positive class upweighted {pos_weight / neg_weight:.1f}×)")
    print(f"\n   🎯 Objective: F2 (β=2) — recall weighted 4× over precision")
    print(f"      Topology diversity vs LR/RF: those optimise AUC (neutral), "
          f"EBM optimises recall-biased F2")

    # ------------------------------------------------------------------
    # Optuna CV
    # ------------------------------------------------------------------
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    def _objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate":        trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_rounds":           trial.suggest_int("max_rounds", 500, 5000, step=100),
            "max_bins":             trial.suggest_int("max_bins", 128, 512, step=32),
            "max_interaction_bins": trial.suggest_int("max_interaction_bins", 16, 128, step=16),
        }

        fold_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
            w_tr = sample_weights[tr_idx]

            try:
                model = _build_model(params, interactions, random_state=random_state)
                model.fit(X_tr, y_tr, sample_weight=w_tr)
                y_pred = model.predict(X_val)
                score = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
            except Exception:
                return float("-inf")

            fold_scores.append(score)
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    # ------------------------------------------------------------------
    # Run Optuna
    # ------------------------------------------------------------------
    print("\n" + "─" * 80)
    print(f"🧠 Bayesian Optimization — {n_trials} trials, {n_folds}-fold Stratified CV")
    print("   ⚖️  Balanced sample weights in all fit() calls")
    print("─" * 80)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=2),
        study_name=study_name,
    )
    t0 = time.perf_counter()
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)
    runtime_s = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    best_params = _normalize_params(study.best_params)
    n_complete = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )

    print(f"\n   ✅ Completed in {runtime_s:.1f}s")
    print(f"   Trials: {n_complete} completed | {n_pruned} pruned")
    print(f"   Best CV F2: {study.best_value:.6f}")
    print(f"\n   Best hyperparameters:")
    for k, v in sorted(best_params.items()):
        fmt = f"{v:.6f}" if k == "learning_rate" else str(v)
        print(f"      {k:>25s} = {fmt}")

    # ------------------------------------------------------------------
    # Interaction diagnostic
    # ------------------------------------------------------------------
    n_pairs = len(interactions)
    if n_pairs > 0:
        print(f"\n   🔬 Interaction pairs modelled ({n_pairs}): {interactions}")
    else:
        print(f"\n   🔬 No interaction pairs — pure additive model")

    # ------------------------------------------------------------------
    # 10-fold final evaluation (matches research output format)
    # ------------------------------------------------------------------
    print(f"\n{'─' * 80}")
    print(f"📊 {n_folds_eval}-Fold Cross-Validation (Final Evaluation — Recall-Biased)...")
    print(f"{'─' * 80}")

    cv_eval = StratifiedKFold(n_splits=n_folds_eval, shuffle=True, random_state=random_state)
    auc_scores, recall_scores_, precision_scores_, f1_scores, f2_scores = [], [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(cv_eval.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        w_tr = sample_weights[tr_idx]

        fold_model = _build_model(best_params, interactions, random_state=random_state)
        fold_model.fit(X_tr, y_tr, sample_weight=w_tr)

        y_proba = fold_model.predict_proba(X_val)[:, 1]
        y_pred  = fold_model.predict(X_val)

        auc_scores.append(roc_auc_score(y_val, y_proba))
        recall_scores_.append(recall_score(y_val, y_pred))
        precision_scores_.append(precision_score(y_val, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
        f2_scores.append(fbeta_score(y_val, y_pred, beta=2, zero_division=0))

        print(
            f"   Fold {fold:2d}: AUC={auc_scores[-1]:.4f}  "
            f"Recall={recall_scores_[-1]:.4f}  "
            f"Precision={precision_scores_[-1]:.4f}  "
            f"F1={f1_scores[-1]:.4f}  "
            f"F2={f2_scores[-1]:.4f}"
        )

    print(f"\n{'─' * 80}")
    print(f"   AUC:       {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"   Recall:    {np.mean(recall_scores_):.4f} ± {np.std(recall_scores_):.4f}")
    print(f"   Precision: {np.mean(precision_scores_):.4f} ± {np.std(precision_scores_):.4f}")
    print(f"   F1:        {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
    print(f"   F2:        {np.mean(f2_scores):.4f} ± {np.std(f2_scores):.4f}")

    # ------------------------------------------------------------------
    # Summary block
    # ------------------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("📊 EBM STAGE 3 — TUNING SUMMARY")
    print(f"{'=' * 80}")
    print(f"┌──────────────────────────────────────────────────────────────┐")
    print(f"│  Bayesian Optimization (Optuna) — EBM RECALL-BIASED          │")
    print(f"├──────────────────────────────────────────────────────────────┤")
    print(f"│  Trials:              {n_trials} ({n_complete} completed, {n_pruned} pruned)")
    print(f"│  Runtime:             {runtime_s:>8.1f}s")
    print(f"│  Best CV F2:          {study.best_value:.6f}")
    print(f"│  {n_folds_eval}-Fold AUC:         {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"│  {n_folds_eval}-Fold Recall:      {np.mean(recall_scores_):.4f} ± {np.std(recall_scores_):.4f}")
    print(f"│  {n_folds_eval}-Fold F2:          {np.mean(f2_scores):.4f} ± {np.std(f2_scores):.4f}")
    print(f"│  Sample weighting:    balanced (class 1 @ {pos_weight:.2f}×)")
    print(f"│  Interaction pairs:   {n_pairs}")
    print(f"├──────────────────────────────────────────────────────────────┤")
    print(f"│  learning_rate:       {best_params['learning_rate']:<12.6f}")
    print(f"│  max_rounds:          {best_params['max_rounds']:<12}")
    print(f"│  max_bins:            {best_params['max_bins']:<12}")
    print(f"│  max_interaction_bins:{best_params['max_interaction_bins']:<12}")
    print(f"└──────────────────────────────────────────────────────────────┘")

    # ------------------------------------------------------------------
    # Full-data refit on best params
    # ------------------------------------------------------------------
    print("\n🔁 Refitting on full training data with best params...")
    final_model = _build_model(best_params, interactions, random_state=random_state)
    final_model.fit(X_train, y_train, sample_weight=sample_weights)
    print("   ✅ Full-data refit complete")
    print("=" * 80)

    return final_model, best_params, float(study.best_value)