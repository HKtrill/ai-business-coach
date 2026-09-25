"""
model_training.lr.trainer
=========================
Canonical LR training path for the Glass Cascade gate stage.

Decision: Optuna + class_weight='balanced', ROC-AUC objective.
Rationale: GridSearchCV (Cell 12A) and recall-biased Optuna (Cell 12C)
produce identical Recall@10%FPR (≈0.527–0.532). 12C is 3× slower and
adds a minority_weight hyperparameter that buys nothing at the ranking
level. Threshold calibration is a separate post-fit concern.

Public API
----------
train_lr(df, features, target_col, ...)  → LRResult
tune_lr(X, y, ...)                        → (params, study, runtime_s)
evaluate_lr(pipe, X, y, ...)              → metrics dict
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Module-level defaults ─────────────────────────────────────────────────────
_N_TUNE_FOLDS: int = 5
_N_EVAL_FOLDS: int = 10
_N_TRIALS: int = 100
_RANDOM_STATE: int = 42

# Coefficient magnitude thresholds for the feature report
_ACTIVE_THRESHOLD: float = 0.15
_WEAK_THRESHOLD: float = 0.10


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class LRResult:
    """
    All artifacts produced by a single LR training run.

    Downstream cells (threshold optimisation, cascade hand-off) consume:
        .pipe       — fitted Pipeline(StandardScaler + LogisticRegression)
        .params     — best Optuna hyperparameters
        .metrics    — mean/std dict for auc, recall, precision, f1,
                      plus scalar recall_at_10fpr
        .feature_importance — DataFrame sorted by |coefficient|
        .study      — raw Optuna study (for diagnostics / reuse)
        .runtime_s  — wall-clock seconds for the Optuna search
    """
    pipe: Pipeline
    params: dict
    metrics: dict
    feature_importance: pd.DataFrame
    study: optuna.Study
    runtime_s: float


# ── Private helpers ───────────────────────────────────────────────────────────
def _clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Replace inf → NaN, impute NaN → column median.
    Returns a cleaned copy; raises AssertionError if residual issues remain.
    Mirrors the safety-check pattern from Cell 12B.
    """
    X = X.copy()
    if np.isinf(X).any().any():
        inf_cols = X.columns[np.isinf(X).any()].tolist()
        print(f"   ⚠️  Infinity in: {inf_cols} → replacing with NaN")
        X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        nan_cols = X.columns[X.isnull().any()].tolist()
        print(f"   ⚠️  NaN in: {nan_cols} → imputing with median")
        for col in nan_cols:
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"      {col}: filled with {median_val:.4f}")
        print("   ✅ Data cleaned")
    else:
        print("   ✅ No inf/NaN values detected")

    assert not np.isinf(X).any().any(), "Infinity values still present after cleaning!"
    assert not X.isnull().any().any(), "NaN values still present after cleaning!"
    return X


def _build_pipe(params: dict, random_state: int) -> Pipeline:
    """Reconstruct a Pipeline from a tuned params dict."""
    lr_kwargs: dict = dict(
        C=params["C"],
        penalty=params["penalty"],
        solver="saga",
        max_iter=5000,
        random_state=random_state,
        class_weight="balanced",
    )
    if params["penalty"] == "elasticnet":
        lr_kwargs["l1_ratio"] = params["l1_ratio"]
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**lr_kwargs)),
    ])


def _make_objective(
    X: pd.DataFrame,
    y: pd.Series,
    cv: StratifiedKFold,
    random_state: int,
):
    """
    Return a closure over (X, y, cv) for use as an Optuna objective.
    Optimises ROC-AUC with balanced class weight fixed.
    Uses intermediate-value pruning to kill poor trials early.
    """
    def objective(trial: optuna.Trial) -> float:
        C = trial.suggest_float("C", 1e-4, 1e3, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        l1_ratio = (
            trial.suggest_float("l1_ratio", 0.0, 1.0)
            if penalty == "elasticnet"
            else None
        )

        lr_kwargs: dict = dict(
            C=C, penalty=penalty, solver="saga", max_iter=5000,
            random_state=random_state, class_weight="balanced",
        )
        if l1_ratio is not None:
            lr_kwargs["l1_ratio"] = l1_ratio

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**lr_kwargs)),
        ])

        fold_scores: list[float] = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            try:
                pipe.fit(X_tr, y_tr)
                auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
            except Exception:
                return float("-inf")
            fold_scores.append(auc)
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    return objective


def _print_tuning_summary(
    params: dict,
    study: optuna.Study,
    runtime_s: float,
    metrics: dict,
) -> None:
    n_complete = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )
    cv_std = metrics["auc_std"]
    stability = (
        "Low variance — stable" if cv_std < 0.015
        else "Moderate variance" if cv_std < 0.03
        else "High variance"
    )
    penalty_str = params["penalty"]
    c_str = f"{params['C']:.6f}"

    print(f"\n{'=' * 80}")
    print("📊 LOGISTIC REGRESSION — TUNING SUMMARY (BALANCED)")
    print(f"{'=' * 80}")
    print(
        f"┌────────────────────────────────────────────────────────────┐\n"
        f"│  Bayesian Optimization (Optuna) — LR BALANCED              │\n"
        f"├────────────────────────────────────────────────────────────┤\n"
        f"│  Trials:      100 ({n_complete} completed, {n_pruned} pruned)                │\n"
        f"│  Runtime:          {runtime_s:>7.1f}s                                  │\n"
        f"│  Best ROC-AUC:     {study.best_value:.6f}                                │\n"
        f"│  10-Fold Mean±Std: {metrics['auc_mean']:.4f} ± {cv_std:.4f}                          │\n"
        f"├────────────────────────────────────────────────────────────┤\n"
        f"│  C:                {c_str:<10}                              │\n"
        f"│  penalty:          {penalty_str:<10}                              │\n"
        f"│  class_weight:     balanced                                │\n"
        f"└────────────────────────────────────────────────────────────┘\n"
        f"💡 Stability: CV std = {cv_std:.4f} ({stability})"
    )
    print(f"\n   💾 params: {params}")
    print("=" * 80)


# ── Public API ────────────────────────────────────────────────────────────────
def tune_lr(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = _N_TRIALS,
    n_folds: int = _N_TUNE_FOLDS,
    random_state: int = _RANDOM_STATE,
) -> tuple[dict, optuna.Study, float]:
    """
    Bayesian hyperparameter search for LR with balanced class weight.

    Parameters
    ----------
    X, y        : Feature matrix and target (already cleaned, unscaled).
    n_trials    : Optuna trial budget (default 100).
    n_folds     : Folds for the inner CV used during tuning (default 5).
    random_state: Seed for reproducibility.

    Returns
    -------
    params    : Best hyperparameter dict (C, penalty, [l1_ratio])
    study     : Raw optuna.Study
    runtime_s : Wall-clock seconds
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    print(f"\n{'─' * 80}")
    print(f"🧠 Bayesian Optimization (Optuna) — {n_trials} trials, {n_folds}-fold Stratified CV")
    print("   ⚖️  class_weight = 'balanced' (fixed)")
    print(f"{'─' * 80}")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        study_name="lr_bayesian_opt_balanced",
    )

    t0 = time.perf_counter()
    study.optimize(
        _make_objective(X, y, cv, random_state),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    runtime_s = time.perf_counter() - t0

    params = study.best_params.copy()

    # Verification pass with the best params on the same CV split
    verify_pipe = _build_pipe(params, random_state)
    verify_scores = cross_val_score(
        verify_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    n_complete = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )

    print(f"\n   ✅ Completed in {runtime_s:.1f}s")
    print(f"   Trials completed: {n_complete} | Pruned: {n_pruned}")
    print(f"   Best CV ROC-AUC:  {study.best_value:.6f}")
    print(f"   Verified Mean±Std: {np.mean(verify_scores):.6f} ± {np.std(verify_scores):.6f}")
    print(f"\n   Best hyperparameters:")
    print(f"      {'C':>15s} = {params['C']:.6f}")
    print(f"      {'penalty':>15s} = {params['penalty']}")
    if params["penalty"] == "elasticnet":
        print(f"      {'l1_ratio':>15s} = {params['l1_ratio']:.4f}")
    print(f"      {'class_weight':>15s} = balanced")
    print(f"\n   Per-fold ROC-AUC: {['%.4f' % s for s in verify_scores]}")

    return params, study, runtime_s


def evaluate_lr(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = _N_EVAL_FOLDS,
    random_state: int = _RANDOM_STATE,
) -> dict:
    """
    Full n-fold cross-validation evaluation of a fitted LR pipeline.

    Fits the pipe fresh on each fold — does not assume prior fitting.
    After CV, refits on the full (X, y) to compute Recall@10%FPR and
    leaves the pipe in a usable state for downstream threshold work.

    Returns
    -------
    metrics dict with keys:
        auc_mean, auc_std
        recall_mean, recall_std
        precision_mean, precision_std
        f1_mean, f1_std
        recall_at_10fpr   ← scalar, computed on full-data predictions
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    print(f"\n{'─' * 80}")
    print(f"📊 {n_folds}-Fold Cross-Validation Results (Final Evaluation)...")
    print(f"{'─' * 80}")

    auc_s, recall_s, precision_s, f1_s = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe.fit(X_tr, y_tr)
        y_proba = pipe.predict_proba(X_val)[:, 1]
        y_pred = pipe.predict(X_val)

        auc_s.append(roc_auc_score(y_val, y_proba))
        recall_s.append(recall_score(y_val, y_pred))
        precision_s.append(precision_score(y_val, y_pred))
        f1_s.append(f1_score(y_val, y_pred))

        print(
            f"   Fold {fold:2d}: AUC={auc_s[-1]:.4f}, Recall={recall_s[-1]:.4f}, "
            f"Precision={precision_s[-1]:.4f}, F1={f1_s[-1]:.4f}"
        )

    # Refit on full data — leaves pipe ready for threshold pass and serialisation
    pipe.fit(X, y)
    fpr, tpr, _ = roc_curve(y, pipe.predict_proba(X)[:, 1])
    recall_at_10fpr = float(tpr[np.argmin(np.abs(fpr - 0.10))])

    return {
        "auc_mean": float(np.mean(auc_s)),
        "auc_std": float(np.std(auc_s)),
        "recall_mean": float(np.mean(recall_s)),
        "recall_std": float(np.std(recall_s)),
        "precision_mean": float(np.mean(precision_s)),
        "precision_std": float(np.std(precision_s)),
        "f1_mean": float(np.mean(f1_s)),
        "f1_std": float(np.std(f1_s)),
        "recall_at_10fpr": recall_at_10fpr,
    }


def train_lr(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_trials: int = _N_TRIALS,
    n_tune_folds: int = _N_TUNE_FOLDS,
    n_eval_folds: int = _N_EVAL_FOLDS,
    random_state: int = _RANDOM_STATE,
    baseline_auc: Optional[float] = None,
) -> LRResult:
    """
    Full LR training pipeline: clean → tune (Optuna) → evaluate → summarise.

    This is the only function the notebook cell needs to call.
    Threshold optimisation is NOT done here — it lives in a dedicated
    post-fit cell that receives lr_result.pipe and sets the operating point
    before the cascade hand-off to RF.

    Parameters
    ----------
    df           : df_engineered (must contain features + target_col)
    features     : LR_FEATURES list  (e.g. from feature_engineering.__init__)
    target_col   : TARGET_COL string
    baseline_auc : Optional float — prints HOLD / BELOW TARGET / REGRESSED
                   comparison if provided.

    Returns
    -------
    LRResult dataclass — pipe, params, metrics, feature_importance, study
    """
    print(f"\n{'=' * 80}")
    print("📊 LOGISTIC REGRESSION TRAINING — BALANCED")
    print(f"{'=' * 80}")

    X = df[features].copy()
    y = df[target_col].copy()

    print(f"\n📊 Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"   Features: {features}")
    print(f"   Class distribution: {y.value_counts().to_dict()}")
    print("\n🔍 Pre-scaling data quality check...")
    X = _clean_features(X)

    # ── Tune ──────────────────────────────────────────────────────────────────
    params, study, runtime_s = tune_lr(X, y, n_trials, n_tune_folds, random_state)

    # ── Build best pipe and evaluate ──────────────────────────────────────────
    pipe = _build_pipe(params, random_state)
    metrics = evaluate_lr(pipe, X, y, n_eval_folds, random_state)
    # pipe is now fit on full (X, y) — ready for coefficient extraction

    # ── Feature importance ────────────────────────────────────────────────────
    clf = pipe.named_steps["clf"]
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "coefficient": clf.coef_[0],
        "abs_coef": np.abs(clf.coef_[0]),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n{'-' * 80}")
    print("📊 LOGISTIC REGRESSION SUMMARY — BALANCED")
    print(f"{'-' * 80}")
    print(f"AUC:       {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
    print(f"Recall:    {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
    print(f"Precision: {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
    print(f"F1:        {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")

    print(f"\n🏆 Features by |Coefficient| ({len(features)} features):")
    for _, row in feature_importance.iterrows():
        if row["abs_coef"] >= _ACTIVE_THRESHOLD:
            status = "🔥 ACTIVE"
        elif row["abs_coef"] >= _WEAK_THRESHOLD:
            status = "⚪ WEAK"
        else:
            status = "💀 DEAD"
        print(f"   {row['feature']:30s} {row['coefficient']:+.4f}  {status}")

    print(f"\n📊 Recall @ 10% FPR: {metrics['recall_at_10fpr']:.4f}")

    if baseline_auc is not None:
        delta = metrics["auc_mean"] - baseline_auc
        if metrics["auc_mean"] >= 0.79:
            status = "✅ HOLD"
        elif metrics["auc_mean"] >= 0.785:
            status = "⚠️  BELOW TARGET"
        else:
            status = "❌ REGRESSED"
        print(f"\n📊 vs Baseline ({baseline_auc:.4f}):")
        print(f"   Current AUC: {metrics['auc_mean']:.4f} (Δ = {delta:+.4f}) → {status}")
        print(f"   Features: {len(features)}")

    _print_tuning_summary(params, study, runtime_s, metrics)

    return LRResult(
        pipe=pipe,
        params=params,
        metrics=metrics,
        feature_importance=feature_importance,
        study=study,
        runtime_s=runtime_s,
    )