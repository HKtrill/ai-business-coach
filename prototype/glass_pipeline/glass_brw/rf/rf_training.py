"""
glass_pipeline.glass_brw.rf_training
======================================
RF Stage 2 training for the Glass Cascade.

Mirrors feature_research/model_training/rf/trainer.py exactly:
    tune → evaluate (10-fold CV) → refit on full X_train → evaluate on X_test

Public API
----------
train_rf_stage(X_train_eng, X_test_eng, y_train, y_test) → RFResult
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
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RFResult:
    """
    Cascade-compatible result object for RF Stage 2.

        .pipe               — Pipeline(clf=RandomForestClassifier) refit on full X_train
        .params             — canonical hyperparams (no Optuna internals)
        .metrics_cv         — 10-fold CV metrics dict
        .metrics_test       — test-set metrics dict
        .feature_importance — pd.DataFrame sorted by Gini importance
        .study              — raw Optuna Study for diagnostics
        .runtime_s          — Optuna wall-clock seconds

    Properties
        .model              — RandomForestClassifier (for GLASS_BRW)
        .metrics            — alias for metrics_cv (mirrors research trainer.py)
    """
    pipe:               Pipeline
    params:             dict
    metrics_cv:         dict
    metrics_test:       dict
    feature_importance: pd.DataFrame
    study:              optuna.Study
    runtime_s:          float

    @property
    def model(self) -> RandomForestClassifier:
        """Convenience accessor — GLASS_BRW(rf_model=result.model)."""
        return self.pipe.named_steps["clf"]

    @property
    def metrics(self) -> dict:
        """Alias for metrics_cv — mirrors research trainer.py RFResult.metrics."""
        return self.metrics_cv


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _make_sample_weights(y: pd.Series, minority_weight: float) -> np.ndarray:
    return y.map({0: 1.0, 1: minority_weight}).values


def _build_pipe(params: dict, random_state: int, n_jobs: int = -1) -> Pipeline:
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


# ---------------------------------------------------------------------------
# Tune
# ---------------------------------------------------------------------------

def _tune_rf(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    n_folds: int,
    random_state: int,
) -> tuple[dict, optuna.Study, float]:
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
        minority_weight  = trial.suggest_float("minority_weight", 1.0, 3.0)

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
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            sw = _make_sample_weights(y_tr, minority_weight)
            try:
                pipe.fit(X_tr, y_tr, clf__sample_weight=sw)
                auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
            except Exception:
                return float("-inf")
            fold_scores.append(auc)
            trial.report(np.mean(fold_scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=15, n_warmup_steps=1),
        study_name="rf_stage2_opt",
    )

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    runtime_s = time.perf_counter() - t0

    return _normalize_params(study.best_params), study, runtime_s


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def _evaluate_rf(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int,
    random_state: int,
) -> dict:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    minority_weight = params.get("minority_weight", 1.0)

    auc_scores, recall_scores, precision_scores, f1_scores = [], [], [], []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipe = _build_pipe(params, random_state, n_jobs=-1)
        sw   = _make_sample_weights(y_tr, minority_weight)
        pipe.fit(X_tr, y_tr, clf__sample_weight=sw)

        y_proba = pipe.predict_proba(X_val)[:, 1]
        y_pred  = pipe.predict(X_val)

        auc_scores.append(roc_auc_score(y_val, y_proba))
        recall_scores.append(recall_score(y_val, y_pred))
        precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred))

    # recall_at_10fpr — temporary full-data refit, ranking metric only
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def train_rf_stage(
    X_train_eng:  pd.DataFrame,
    X_test_eng:   pd.DataFrame,
    y_train:      pd.Series,
    y_test:       pd.Series,
    n_trials:     int = 200,
    n_tune_folds: int = 5,
    n_eval_folds: int = 10,
    random_state: int = 42,
    baseline_auc: float | None = None,
) -> RFResult:
    """
    Pipeline entry point — tune → evaluate → refit on full X_train → test eval.

    Parameters
    ----------
    X_train_eng  : binary feature matrix (N_train, 29)
    X_test_eng   : binary feature matrix (N_test, 29)
    y_train      : binary target for train
    y_test       : binary target for test
    n_trials     : Optuna trials (default 200)
    n_tune_folds : CV folds during tuning (default 5)
    n_eval_folds : CV folds for evaluation (default 10)
    random_state : global seed
    baseline_auc : optional float, prints AUC delta if provided
    """
    print("\n" + "=" * 80)
    print("🌲 RF STAGE 2 — BAYESIAN OPTIMIZATION (BINARY FEATURES)")
    print("=" * 80)
    print(f"\n📊 Dataset     : {X_train_eng.shape[0]:,} train | {X_test_eng.shape[0]:,} test | {X_train_eng.shape[1]} binary features")
    print(f"   Feature space: 2^{X_train_eng.shape[1]} = {2 ** X_train_eng.shape[1]:,} possible patterns")
    print(f"   Observed     : {X_train_eng.drop_duplicates().shape[0]:,} patterns")
    print(f"   Class dist   : {y_train.value_counts().to_dict()}")

    print("\n🔍 Pre-flight validation...")
    assert X_train_eng.isin([0, 1]).all().all(), "Non-binary values in X_train_eng."
    assert not X_train_eng.isnull().any().any(),  "NaN values in X_train_eng."
    assert X_test_eng.isin([0, 1]).all().all(),   "Non-binary values in X_test_eng."
    assert not X_test_eng.isnull().any().any(),    "NaN values in X_test_eng."
    print("   ✅ All values binary, no missing data")

    # ── 1. Tune ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"🧠 Bayesian Optimization — {n_trials} trials, {n_tune_folds}-fold CV")
    print(f"   class_weight=balanced (fixed) | minority_weight in [1.0, 3.0]")
    print(f"{'─' * 80}")

    params, study, runtime_s = _tune_rf(
        X_train_eng, y_train, n_trials, n_tune_folds, random_state
    )

    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned   = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)

    print(f"\n   ✅ Tuning complete in {runtime_s:.1f}s")
    print(f"   Trials : {n_complete} completed | {n_pruned} pruned")
    print(f"   Best Optuna AUC : {study.best_value:.6f}")
    print(f"\n   Best params:")
    for k, v in params.items():
        v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"      {k:>20s} = {v_str}")

    # ── 2. Evaluate ───────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"📊 {n_eval_folds}-Fold CV Evaluation with best params...")
    print(f"{'─' * 80}")

    metrics_cv = _evaluate_rf(params, X_train_eng, y_train, n_eval_folds, random_state)

    print(f"   AUC       : {metrics_cv['auc_mean']:.4f} ± {metrics_cv['auc_std']:.4f}")
    print(f"   Recall    : {metrics_cv['recall_mean']:.4f} ± {metrics_cv['recall_std']:.4f}")
    print(f"   Precision : {metrics_cv['precision_mean']:.4f} ± {metrics_cv['precision_std']:.4f}")
    print(f"   F1        : {metrics_cv['f1_mean']:.4f} ± {metrics_cv['f1_std']:.4f}")
    print(f"   R@10%FPR  : {metrics_cv['recall_at_10fpr']:.4f}")

    if baseline_auc is not None:
        delta = metrics_cv["auc_mean"] - baseline_auc
        arrow = "↑" if delta > 0 else "↓"
        print(f"\n   vs baseline : {baseline_auc:.4f} → {delta:+.4f} {arrow}")

    # ── 3. Final refit on full X_train ────────────────────────────────────────
    print(f"\n   Refitting on full X_train ({X_train_eng.shape[0]:,} rows)...")
    pipe = _build_pipe(params, random_state, n_jobs=-1)
    sw   = _make_sample_weights(y_train, params["minority_weight"])
    pipe.fit(X_train_eng, y_train, clf__sample_weight=sw)

    # ── 4. Test evaluation ────────────────────────────────────────────────────
    print(f"\n   Test evaluation on holdout set ({X_test_eng.shape[0]:,} rows)...")
    test_proba = pipe.predict_proba(X_test_eng)[:, 1]
    test_pred  = pipe.predict(X_test_eng)

    metrics_test: dict = {
        "auc":       float(roc_auc_score(y_test, test_proba)),
        "recall":    float(recall_score(y_test, test_pred)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "f1":        float(f1_score(y_test, test_pred)),
        "brier":     float(brier_score_loss(y_test, test_proba)),
    }

    # ── Feature importance ────────────────────────────────────────────────────
    clf = pipe.named_steps["clf"]
    feat_imp = (
        pd.DataFrame({"feature": X_train_eng.columns, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    stability = (
        "Low variance — stable"        if metrics_cv["auc_std"] < 0.015 else
        "Moderate variance"             if metrics_cv["auc_std"] < 0.030 else
        "High variance — check overfit"
    )

    print(f"\n{'=' * 80}")
    print("📊 RF STAGE 2 — TRAINING SUMMARY")
    print(f"{'=' * 80}")
    print(f"   {n_eval_folds}-Fold CV AUC  : {metrics_cv['auc_mean']:.4f} ± {metrics_cv['auc_std']:.4f}")
    print(f"   R@10%FPR      : {metrics_cv['recall_at_10fpr']:.4f}")
    print(f"   Test AUC      : {metrics_test['auc']:.4f}")
    print(f"   Test Recall   : {metrics_test['recall']:.4f}")
    print(f"   Test F1       : {metrics_test['f1']:.4f}")
    print(f"   Stability     : {stability}")
    print(f"   Runtime       : {runtime_s:.1f}s")
    print(f"{'=' * 80}")

    return RFResult(
        pipe=pipe,
        params=params,
        metrics_cv=metrics_cv,
        metrics_test=metrics_test,
        feature_importance=feat_imp,
        study=study,
        runtime_s=runtime_s,
    )