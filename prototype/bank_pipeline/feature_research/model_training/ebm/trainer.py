"""
feature_research/model_training/ebm/trainer.py
===============================================
Recall-biased EBM trainer for the Glass Cascade pipeline.

Topology
--------
Smooth shape functions + pairwise interactions (capped at 5).
The balanced-experiment interaction delta (+0.0019 AUC for 15 interactions)
did not justify the interpretability cost. Cap enforced in the Optuna search
space — downstream GLASS-BRW rule generation stays cleaner as a result.

Bias
----
Balanced sample weights throughout (mirrors LR and RF).
F2 objective (β=2) weights recall 4× over precision — provides directional
diversity against LR/RF which both optimise neutral ROC-AUC.

Design patterns (mirrors rf/trainer.py)
---------------------------------------
- _normalize_params()    collapses Optuna internals into canonical EBM params
- _make_sample_weights() called identically in objective, eval, and refit
- _build_model()         n_jobs=1 inside Optuna, n_jobs=-1 everywhere else
- EBMResult dataclass    pipe, params, metrics, study, runtime_s

Provides
--------
- EBMResult   — result dataclass
- train_ebm() — Optuna-tuned recall-biased EBM
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EBMResult:
    """Container for a trained EBM experiment.

    Attributes
    ----------
    pipe      : ExplainableBoostingClassifier fitted on full (X, y).
    params    : Canonical hyperparameters (post _normalize_params).
    metrics   : CV mean ± std for AUC, Recall, Precision, F1, F2.
    study     : Optuna study object (full trial history).
    runtime_s : Wall-clock training time in seconds.
    """

    pipe: ExplainableBoostingClassifier
    params: Dict[str, Any]
    metrics: Dict[str, float]
    study: optuna.Study
    runtime_s: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_sample_weights(y: pd.Series) -> np.ndarray:
    """Balanced sample weights.

    Called identically in the Optuna objective, fold evaluation, and full
    refit — no drift between tuning and evaluation.
    """
    return compute_sample_weight("balanced", y)


def _normalize_params(trial_params: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse Optuna trial params into canonical EBM constructor params.

    Nothing downstream sees internal Optuna search-space keys.
    """
    return {
        "learning_rate":       trial_params["learning_rate"],
        "max_rounds":          trial_params["max_rounds"],
        "max_bins":            trial_params["max_bins"],
        "max_interaction_bins": trial_params["max_interaction_bins"],
        "interactions":        trial_params["interactions"],
    }


def _build_model(
    params: Dict[str, Any],
    random_state: int = 42,
    n_jobs: int = -1,
) -> ExplainableBoostingClassifier:
    """Construct an EBM from canonical params.

    n_jobs=1 inside the Optuna objective to avoid nested joblib parallelism
    warnings. n_jobs=-1 for final evaluation and refit.
    """
    return ExplainableBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_rounds=params["max_rounds"],
        max_bins=params["max_bins"],
        max_interaction_bins=params["max_interaction_bins"],
        interactions=params["interactions"],
        random_state=random_state,
        n_jobs=n_jobs,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_ebm(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 150,
    n_folds_tuning: int = 5,
    n_folds_eval: int = 10,
    random_state: int = 42,
    study_name: str = "ebm_bayesian_opt_recall",
) -> EBMResult:
    """Optuna-tuned EBM with recall-biased F2 objective.

    Parameters
    ----------
    X             : Feature matrix (EBM_FEATURES columns).
    y             : Binary target (0/1).
    n_trials      : Optuna trials. Default 150.
    n_folds_tuning: CV folds inside Optuna objective. Default 5.
    n_folds_eval  : CV folds for final evaluation. Default 10.
    random_state  : Global random seed.
    study_name    : Optuna study identifier.

    Returns
    -------
    EBMResult
        Fitted model (full data), canonical params, CV metrics, study,
        runtime. Threshold optimisation is NOT performed here — that is a
        downstream concern.
    """

    # ── Header ────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🔮 EBM TRAINER — RECALL-BIASED (F2 Objective, β=2)")
    print("=" * 80)
    print(f"\n📊 Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"   Class distribution: {y.value_counts().to_dict()}")

    # ── Data quality check ────────────────────────────────────────────────
    print("\n🔍 Pre-modeling data quality check...")
    X = X.copy()
    if np.isinf(X).any().any():
        X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        nan_cols = X.columns[X.isnull().any()].tolist()
        print(f"   ⚠️  NaN in: {nan_cols} — imputing with median")
        for col in nan_cols:
            X[col] = X[col].fillna(X[col].median())
    assert not np.isinf(X).any().any(), "Infinity values remain after cleaning"
    assert not X.isnull().any().any(), "NaN values remain after cleaning"
    print("   ✅ Data clean")

    # ── Sample weights ────────────────────────────────────────────────────
    sample_weights = _make_sample_weights(y)
    pos_weight = sample_weights[y == 1].mean()
    neg_weight = sample_weights[y == 0].mean()
    print(f"\n   ⚖️  Balanced sample weights: "
          f"class 0 → {neg_weight:.4f}, class 1 → {pos_weight:.4f}")
    print(f"      (positive class upweighted {pos_weight / neg_weight:.1f}×)")
    print(f"\n   🎯 Objective: F2 score (β=2, recall weighted 4× over precision)")
    print(f"      Topology diversity: LR/RF optimise AUC (neutral), "
          f"EBM optimises F2 (recall-biased)")

    # ── Interaction tracker ───────────────────────────────────────────────
    interaction_tracker: dict[str, list[float]] = {"with": [], "without": []}

    # ── Optuna CV strategy ────────────────────────────────────────────────
    cv_tune = StratifiedKFold(
        n_splits=n_folds_tuning, shuffle=True, random_state=random_state
    )

    def _objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate":        trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_rounds":           trial.suggest_int("max_rounds", 500, 5000, step=100),
            "max_bins":             trial.suggest_int("max_bins", 128, 512, step=32),
            "max_interaction_bins": trial.suggest_int("max_interaction_bins", 16, 128, step=16),
            "interactions":         trial.suggest_int("interactions", 0, 5),  # capped: interpretability
        }

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_tune.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_train = sample_weights[train_idx]

            try:
                model = _build_model(params, random_state=random_state, n_jobs=1)
                model.fit(X_train, y_train, sample_weight=w_train)
                y_pred = model.predict(X_val)
                score = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
            except Exception:
                return float("-inf")

            fold_scores.append(score)
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_score = float(np.mean(fold_scores))
        interaction_tracker["with" if params["interactions"] > 0 else "without"].append(mean_score)
        return mean_score

    # ── Run Optuna ────────────────────────────────────────────────────────
    print("\n" + "─" * 80)
    print(f"🧠 Bayesian Optimization (Optuna) — {n_trials} trials, "
          f"{n_folds_tuning}-fold Stratified CV")
    print("   ⚖️  Using balanced sample weights in all fit() calls")
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

    # ── Extract results ───────────────────────────────────────────────────
    best_params = _normalize_params(study.best_params)
    n_complete = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )

    print(f"\n   ✅ Completed in {runtime_s:.1f}s")
    print(f"   Trials completed: {n_complete} | Pruned: {n_pruned}")
    print(f"   Best CV F2:  {study.best_value:.6f}")
    print(f"\n   Best hyperparameters:")
    for k, v in sorted(best_params.items()):
        fmt = f"{v:.6f}" if k == "learning_rate" else str(v)
        print(f"      {k:>25s} = {fmt}")

    # ── Interaction analysis ──────────────────────────────────────────────
    print("\n" + "─" * 80)
    print("🔬 Interaction Analysis")
    print("─" * 80)

    mean_with    = np.mean(interaction_tracker["with"])    if interaction_tracker["with"]    else 0.0
    mean_without = np.mean(interaction_tracker["without"]) if interaction_tracker["without"] else 0.0
    best_with    = max(interaction_tracker["with"])        if interaction_tracker["with"]    else 0.0
    best_without = max(interaction_tracker["without"])     if interaction_tracker["without"] else 0.0
    interaction_delta = best_with - best_without

    print(f"   Trials with interactions>0:   {len(interaction_tracker['with']):>4d}  "
          f"(mean F2: {mean_with:.6f}, best: {best_with:.6f})")
    print(f"   Trials with interactions=0:   {len(interaction_tracker['without']):>4d}  "
          f"(mean F2: {mean_without:.6f}, best: {best_without:.6f})")
    print(f"   Best interaction Δ (with - without): {interaction_delta:+.6f}")

    if interaction_delta > 0.005:
        interaction_verdict = (
            f"Interactions materially improved F2 (+{interaction_delta:.4f}). "
            f"Best config uses {best_params['interactions']} interaction(s)."
        )
    elif interaction_delta > 0.001:
        interaction_verdict = (
            f"Interactions provided marginal F2 gain (+{interaction_delta:.4f}). "
            f"Useful but not critical — monitor for overfitting on new data."
        )
    else:
        interaction_verdict = (
            f"Interactions did NOT materially improve F2 "
            f"(Δ={interaction_delta:+.4f}). Consider setting interactions=0 "
            f"for maximum interpretability."
        )
    print(f"\n   💡 {interaction_verdict}")

    # ── 10-fold final evaluation ──────────────────────────────────────────
    print("\n" + "─" * 80)
    print(f"📊 {n_folds_eval}-Fold Cross-Validation (Final Evaluation — Recall-Biased)...")
    print("─" * 80)

    cv_eval = StratifiedKFold(
        n_splits=n_folds_eval, shuffle=True, random_state=random_state
    )
    auc_scores, recall_scores_, precision_scores_, f1_scores, f2_scores = (
        [], [], [], [], []
    )

    for fold, (train_idx, val_idx) in enumerate(cv_eval.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        w_train = sample_weights[train_idx]

        fold_model = _build_model(best_params, random_state=random_state, n_jobs=1)
        fold_model.fit(X_train, y_train, sample_weight=w_train)

        y_proba = fold_model.predict_proba(X_val)[:, 1]
        y_pred  = fold_model.predict(X_val)

        auc_scores.append(roc_auc_score(y_val, y_proba))
        recall_scores_.append(recall_score(y_val, y_pred))
        precision_scores_.append(precision_score(y_val, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
        f2_scores.append(fbeta_score(y_val, y_pred, beta=2, zero_division=0))

        print(
            f"   Fold {fold:2d}: AUC={auc_scores[-1]:.4f}, "
            f"Recall={recall_scores_[-1]:.4f}, "
            f"Precision={precision_scores_[-1]:.4f}, "
            f"F1={f1_scores[-1]:.4f}, "
            f"F2={f2_scores[-1]:.4f}"
        )

    metrics = {
        "auc_mean":       float(np.mean(auc_scores)),
        "auc_std":        float(np.std(auc_scores)),
        "recall_mean":    float(np.mean(recall_scores_)),
        "recall_std":     float(np.std(recall_scores_)),
        "precision_mean": float(np.mean(precision_scores_)),
        "precision_std":  float(np.std(precision_scores_)),
        "f1_mean":        float(np.mean(f1_scores)),
        "f1_std":         float(np.std(f1_scores)),
        "f2_mean":        float(np.mean(f2_scores)),
        "f2_std":         float(np.std(f2_scores)),
    }

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "-" * 80)
    print("🔮 EBM SUMMARY (RECALL-BIASED)")
    print("-" * 80)
    print(f"AUC:       {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
    print(f"Recall:    {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
    print(f"Precision: {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
    print(f"F1:        {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"F2:        {metrics['f2_mean']:.4f} ± {metrics['f2_std']:.4f}")

    # ── Full-data refit ───────────────────────────────────────────────────
    print("\n🔁 Refitting on full data with best params...")
    final_model = _build_model(best_params, random_state=random_state, n_jobs=-1)
    final_model.fit(X, y, sample_weight=sample_weights)
    print("   ✅ Full-data refit complete")

    # ── Structured summary block ──────────────────────────────────────────
    cv_std = metrics["auc_std"]
    stability = (
        "Low variance — stable model"              if cv_std < 0.015 else
        "Moderate variance — acceptable for GAMs"  if cv_std < 0.030 else
        "High variance — consider reducing complexity"
    )
    int_count = best_params["interactions"]
    interp_note = (
        "✅ Pure additive model (interactions=0) — fully interpretable shape functions"
        if int_count == 0
        else f"⚠️  {int_count} interaction term(s) added — interpretable via 2D heatmaps"
    )

    print(f"\n{'=' * 80}")
    print("📊 EBM — TUNING SUMMARY (RECALL-BIASED)")
    print(f"{'=' * 80}")
    print(f"┌──────────────────────────────────────────────────────────────┐")
    print(f"│  Bayesian Optimization (Optuna) — EBM RECALL-BIASED          │")
    print(f"├──────────────────────────────────────────────────────────────┤")
    print(f"│  Trials:              {n_trials} ({n_complete} completed, {n_pruned} pruned)")
    print(f"│  Runtime:             {runtime_s:>8.1f}s")
    print(f"│  Best CV F2:          {study.best_value:.6f}")
    print(f"│  10-Fold AUC:         {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
    print(f"│  10-Fold Recall:      {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
    print(f"│  10-Fold F2:          {metrics['f2_mean']:.4f} ± {metrics['f2_std']:.4f}")
    print(f"│  Sample weighting:    balanced (class 1 @ {pos_weight:.2f}×)")
    print(f"├──────────────────────────────────────────────────────────────┤")
    print(f"│  learning_rate:       {best_params['learning_rate']:<12.6f}")
    print(f"│  max_rounds:          {best_params['max_rounds']:<12}")
    print(f"│  max_bins:            {best_params['max_bins']:<12}")
    print(f"│  max_interaction_bins:{best_params['max_interaction_bins']:<12}")
    print(f"│  interactions:        {best_params['interactions']:<12}")
    print(f"└──────────────────────────────────────────────────────────────┘")
    print(f"💡 Stability:    CV AUC std = {cv_std:.4f} ({stability})")
    print(f"🔬 Interactions: {interaction_verdict}")
    print(f"📝 Interpretability: {interp_note}")
    print(f"\n   💾 EBMResult: pipe fitted on full data with balanced weights")
    print("=" * 80)

    return EBMResult(
        pipe=final_model,
        params=best_params,
        metrics=metrics,
        study=study,
        runtime_s=runtime_s,
    )