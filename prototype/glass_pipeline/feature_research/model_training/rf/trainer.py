"""
model_training/rf/trainer.py

Canonical Random Forest training path — binary feature space.
Mirrors lr/trainer.py: tune → evaluate → refit on full (X, y).

Public API
----------
    train_rf(df, features, target_col, ...) -> RFResult
    tune_rf(X, y, n_trials, n_folds, random_state) -> (params, study, runtime_s)
    evaluate_rf(params, X, y, n_folds, random_state) -> metrics dict
    RFResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class RFResult:
    """
    Canonical output of train_rf().

    Attributes
    ----------
    pipe
        Pipeline(clf=RandomForestClassifier) refit on full (X, y).
        Ready for threshold pass or Venn trace — do not refit downstream.
    params
        Canonical hyperparams usable directly by RandomForestClassifier.
        No Optuna internals (use_max_depth, max_features_type, etc.).
    metrics
        10-fold CV metrics dict. Keys:
        auc_mean, auc_std, recall_mean, recall_std,
        precision_mean, precision_std, f1_mean, f1_std, recall_at_10fpr
    feature_importance
        DataFrame [feature, importance] sorted descending.
        Gini importance from the final full-data fit.
    study
        optuna.Study for trial inspection / convergence plots.
    runtime_s
        Wall-clock seconds for the tune step only.
    """
    pipe:               Pipeline
    params:             dict
    metrics:            dict
    feature_importance: pd.DataFrame
    study:              optuna.Study
    runtime_s:          float


# ── Private helpers ───────────────────────────────────────────────────────────

def _make_sample_weights(y: pd.Series, minority_weight: float) -> np.ndarray:
    """
    Minority overweight applied on top of class_weight='balanced'.
    Called identically in tune_rf objective, evaluate_rf, and the final refit
    so there is no divergence between tuning, evaluation, and inference.
    """
    return y.map({0: 1.0, 1: minority_weight}).values


def _build_pipe(params: dict, random_state: int, n_jobs: int = -1) -> Pipeline:
    """
    Construct a fresh Pipeline from canonical params.
    params must already be normalised — no Optuna internals.

    n_jobs : -1 (default) for evaluation and final refit.
             Pass 1 when calling from inside the Optuna objective to avoid
             nested joblib parallelism warnings.
    """
    return Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        ))
    ])


def _normalize_params(raw: dict) -> dict:
    """
    Resolve Optuna's conditional scaffolding into canonical model params.

    Optuna needs branching suggest calls (use_max_depth, max_features_type,
    max_features_fraction) that are tuning internals, not model params.
    This function collapses them once so nothing downstream ever sees them.
    """
    max_depth = (
        raw.get("max_depth")
        if raw.get("use_max_depth", False)
        else None
    )

    mf_type = raw.get("max_features_type", "sqrt")
    max_features = (
        raw.get("max_features_fraction")
        if mf_type == "fraction"
        else mf_type
    )

    return {
        "n_estimators":     raw["n_estimators"],
        "max_depth":        max_depth,
        "min_samples_leaf": raw["min_samples_leaf"],
        "max_features":     max_features,
        "minority_weight":  raw.get("minority_weight", 1.0),
        "class_weight":     "balanced",
    }


# ── Core functions ────────────────────────────────────────────────────────────

def tune_rf(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    n_folds: int,
    random_state: int,
) -> tuple[dict, optuna.Study, float]:
    """
    Bayesian hyperparameter search over the binary feature space.

    Returns canonical params dict, the study object, and wall-clock runtime.

    Search space
    ------------
    n_estimators      200-1200 step 50
    max_depth         None or 2-12   (binary splits are shallow by nature;
                                      cap guards against depth overfitting
                                      on pre-discretised features)
    min_samples_leaf  5-50           (coarser leaf patterns expected)
    max_features      sqrt | log2 | fraction[0.3, 1.0]
    minority_weight   1.0-15.0       (sample_weight on top of balanced)
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        n_estimators     = trial.suggest_int("n_estimators", 200, 1200, step=50)
        use_max_depth    = trial.suggest_categorical("use_max_depth", [True, False])
        max_depth        = trial.suggest_int("max_depth", 2, 12) if use_max_depth else None
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 5, 50)
        mf_type          = trial.suggest_categorical("max_features_type", ["sqrt", "log2", "fraction"])
        max_features     = (
            trial.suggest_float("max_features_fraction", 0.3, 1.0)
            if mf_type == "fraction"
            else mf_type
        )
        minority_weight  = trial.suggest_float("minority_weight", 1.0, 15.0)

        # n_jobs=1 — avoids nested joblib parallelism inside the Optuna objective.
        # The RF still parallelises tree building; the outer loop is sequential.
        pipe = Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ))
        ])

        fold_scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            sw = _make_sample_weights(y_train, minority_weight)
            try:
                pipe.fit(X_train, y_train, clf__sample_weight=sw)
                auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
            except Exception:
                return float("-inf")
            fold_scores.append(auc)
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(fold_scores)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=15, n_warmup_steps=1),
        study_name="rf_binary_opt",
    )

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    runtime_s = time.perf_counter() - t0

    return _normalize_params(study.best_params), study, runtime_s


def evaluate_rf(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int,
    random_state: int,
) -> dict:
    """
    n-fold stratified CV evaluation using canonical params.

    Applies the same dual-weighting (balanced + minority_weight) as tune_rf —
    no drift between tuning and evaluation.

    recall_at_10fpr is computed via a temporary full-data refit at the end.
    The caller's final refit in train_rf is separate — this one exists only
    to produce the threshold-independent ranking metric.

    Returns
    -------
    dict with keys:
        auc_mean, auc_std,
        recall_mean, recall_std,
        precision_mean, precision_std,
        f1_mean, f1_std,
        recall_at_10fpr
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    minority_weight = params.get("minority_weight", 1.0)

    auc_scores, recall_scores, precision_scores, f1_scores = [], [], [], []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = _build_pipe(params, random_state, n_jobs=-1)
        sw   = _make_sample_weights(y_train, minority_weight)
        pipe.fit(X_train, y_train, clf__sample_weight=sw)

        y_proba = pipe.predict_proba(X_val)[:, 1]
        y_pred  = pipe.predict(X_val)

        auc_scores.append(roc_auc_score(y_val, y_proba))
        recall_scores.append(recall_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred))
        f1_scores.append(f1_score(y_val, y_pred))

    # recall_at_10fpr — temporary full-data refit for ranking metric only
    pipe_tmp = _build_pipe(params, random_state, n_jobs=-1)
    pipe_tmp.fit(X, y, clf__sample_weight=_make_sample_weights(y, minority_weight))
    fpr, tpr, _ = roc_curve(y, pipe_tmp.predict_proba(X)[:, 1])
    recall_at_10fpr = float(tpr[np.argmin(np.abs(fpr - 0.10))])

    return {
        "auc_mean":        float(np.mean(auc_scores)),
        "auc_std":         float(np.std(auc_scores)),
        "recall_mean":     float(np.mean(recall_scores)),
        "recall_std":      float(np.std(recall_scores)),
        "precision_mean":  float(np.mean(precision_scores)),
        "precision_std":   float(np.std(precision_scores)),
        "f1_mean":         float(np.mean(f1_scores)),
        "f1_std":          float(np.std(f1_scores)),
        "recall_at_10fpr": recall_at_10fpr,
    }


def train_rf(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_trials: int = 200,
    n_tune_folds: int = 5,
    n_eval_folds: int = 10,
    random_state: int = 42,
    baseline_auc: float | None = None,
) -> RFResult:
    """
    Canonical RF training entry point for binary feature space.

    Steps
    -----
    1. Validate  — binary-only, no NaNs, hard stop on failure
    2. tune_rf   — Optuna over n_tune_folds CV
    3. evaluate_rf — n_eval_folds CV with best params
    4. Final refit on full (X, y) — pipe ready for threshold pass / Venn trace

    Parameters
    ----------
    df           : DataFrame with features + target_col. Features must be binary (0/1).
    features     : RF_FEATURES_BINARY
    target_col   : Binary target column name
    n_trials     : Optuna trials (default 200)
    n_tune_folds : CV folds during tuning (default 5)
    n_eval_folds : CV folds for final evaluation (default 10)
    random_state : Global seed for reproducibility
    baseline_auc : Optional float. Prints AUC delta in the summary.
                   Pass the continuous RF AUC to track the binary trade-off.

    Returns
    -------
    RFResult
    """
    X = df[features].copy()
    y = df[target_col].copy()

    # ── 1. Validate ───────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("🌲 RANDOM FOREST — BAYESIAN OPTIMIZATION (BINARY FEATURES)")
    print("=" * 80)
    print(f"\n📊 Dataset     : {X.shape[0]:,} samples | {X.shape[1]} binary features")
    print(f"   Feature space: 2^{X.shape[1]} = {2 ** X.shape[1]:,} possible patterns")
    print(f"   Observed     : {X.drop_duplicates().shape[0]:,} patterns")
    print(f"   Class dist   : {y.value_counts().to_dict()}")

    print("\n🔍 Pre-flight validation...")
    assert X.isin([0, 1]).all().all(), "Non-binary values detected in feature matrix."
    assert not X.isnull().any().any(),  "NaN values detected in feature matrix."
    print("   ✅ All values binary, no missing data")

    # ── 2. Tune ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"🧠 Bayesian Optimization — {n_trials} trials, {n_tune_folds}-fold CV")
    print(f"   class_weight=balanced (fixed) | minority_weight tuned by Optuna")
    print(f"{'─' * 80}")

    params, study, runtime_s = tune_rf(X, y, n_trials, n_tune_folds, random_state)

    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)

    print(f"\n   ✅ Tuning complete in {runtime_s:.1f}s")
    print(f"   Trials : {n_complete} completed | {n_pruned} pruned")
    print(f"   Best Optuna AUC : {study.best_value:.6f}")
    print(f"\n   Best params:")
    for k, v in params.items():
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"      {k:>20s} = {v_str}")

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"📊 {n_eval_folds}-Fold Evaluation with best params...")
    print(f"{'─' * 80}")

    metrics = evaluate_rf(params, X, y, n_eval_folds, random_state)

    print(f"   AUC       : {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
    print(f"   Recall    : {metrics['recall_mean']:.4f} ± {metrics['recall_std']:.4f}")
    print(f"   Precision : {metrics['precision_mean']:.4f} ± {metrics['precision_std']:.4f}")
    print(f"   F1        : {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
    print(f"   R@10%FPR  : {metrics['recall_at_10fpr']:.4f}")

    if baseline_auc is not None:
        delta = metrics["auc_mean"] - baseline_auc
        arrow = "↑" if delta > 0 else "↓"
        print(f"\n   vs baseline : {baseline_auc:.4f} → {delta:+.4f} {arrow}")

    # ── 4. Final refit ────────────────────────────────────────────────────────
    print(f"\n   Refitting on full dataset...")
    pipe = _build_pipe(params, random_state, n_jobs=-1)
    sw   = _make_sample_weights(y, params["minority_weight"])
    pipe.fit(X, y, clf__sample_weight=sw)

    clf = pipe.named_steps["clf"]
    feature_importance = (
        pd.DataFrame({"feature": features, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    stability = (
        "Low variance — stable"         if metrics["auc_std"] < 0.015 else
        "Moderate variance"              if metrics["auc_std"] < 0.030 else
        "High variance — check overfit"
    )

    print(f"\n{'=' * 80}")
    print("📊 RF BINARY — TRAINING SUMMARY")
    print(f"{'=' * 80}")
    print(f"   {n_eval_folds}-Fold AUC     : {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
    print(f"   Recall@10%FPR : {metrics['recall_at_10fpr']:.4f}")
    print(f"   Stability     : {stability}")
    print(f"   Runtime       : {runtime_s:.1f}s")
    print(f"{'=' * 80}")

    return RFResult(
        pipe=pipe,
        params=params,
        metrics=metrics,
        feature_importance=feature_importance,
        study=study,
        runtime_s=runtime_s,
    )