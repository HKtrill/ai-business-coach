"""
model_training.rf.trainer
=========================
Random Forest training for the Glass Cascade, on the 29-column binary space.

Method
------
RandomForestClassifier(class_weight='balanced') plus a tuned minority
sample_weight on top, tuned with Optuna (TPE) on mean ROC-AUC. The same
weighting is used in tuning, evaluation and the final fit.

Search space
------------
n_estimators      200–1200, step 50
max_depth         None, or 2–12 (cap guards against depth overfitting on
                  pre-discretised features)
min_samples_leaf  5–50
max_features      sqrt | log2 | fraction in [0.3, 1.0]
minority_weight   1.0–15.0 (sample_weight for y=1, on top of balanced)

Leakage contract
----------------
- Callers pass training rows only. Every reported score is out-of-fold.
- With `feature_factory` (feature pipeline + binning), features are refit on
  each fold's training rows, so target-derived sources never see validation
  labels. Fold matrices are built once per CV and reused across trials.
- Features must be 0/1 with no NaN; anything else raises.

Public API
----------
train_rf(df, features, target_col, ...)   → RFResult
tune_rf(X, y, ...)                         → (params, study, runtime_s)
evaluate_rf(params, X, y, ...)             → metrics dict
RFResult
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.pipeline import Pipeline

from feature_research.model_training.shared.folds import (
    FeatureFactory,
    Fold,
    build_folds,
    engineer_full,
    factory_name,
    recall_at_fpr,
)
from feature_research.model_training.shared.report import training_report

# ── Module-level defaults ─────────────────────────────────────────────────────
_N_TUNE_FOLDS: int = 5
_N_EVAL_FOLDS: int = 10
_N_TRIALS: int = 200
_RANDOM_STATE: int = 42
_TARGET_FPR: float = 0.10

# Verdict tolerance relative to baseline_auc (≈ 1 standard error of the
# 10-fold mean AUC). Δ ≥ −tol → HOLD; Δ ≥ −2·tol → BELOW TARGET; else REGRESSED.
_AUC_TOLERANCE: float = 0.005


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class RFResult:
    """
    Output of train_rf().

    pipe               Pipeline(clf=RandomForestClassifier) fit on all input
                       rows; predicts single-threaded for bitwise-stable scores.
                       Input: binary features in `features` order, i.e.
                       df_rf_binary[RF_FEATURES_BINARY].
    params             Canonical model params (no Optuna scaffolding):
                       n_estimators, max_depth, min_samples_leaf, max_features,
                       minority_weight, class_weight.
    metrics            Evaluation-CV metrics, out-of-fold:
                         auc_mean/std                     per-fold ROC-AUC
                         recall/precision/f1 _mean/std    per fold, at predict()'s
                                                          0.5 threshold
                         recall_at_10fpr                  pooled OOF, FPR ≤ 10%
    feature_importance feature, importance (Gini, final fit), sorted descending.
    study              Optuna study.
    runtime_s          Wall-clock seconds for the Optuna search.
    oof_proba          Out-of-fold P(y=1) from the evaluation CV, indexed like y.
                       Use for threshold selection on training rows; never use
                       pipe.predict_proba on its own training rows.
    """
    pipe: Pipeline
    params: dict
    metrics: dict
    feature_importance: pd.DataFrame
    study: optuna.Study
    runtime_s: float
    oof_proba: pd.Series


# ── Private helpers ───────────────────────────────────────────────────────────
def _require_binary(X: pd.DataFrame, where: str) -> None:
    """Raise ValueError if any column holds values other than 0/1 (including NaN)."""
    bad = [c for c in X.columns if not X[c].isin([0, 1]).all()]
    if bad:
        raise ValueError(f"{where}: non-binary (or NaN) values in {bad}")


def _make_sample_weights(y: pd.Series, minority_weight: float) -> np.ndarray:
    """Per-row sample_weight: 1.0 for y=0, minority_weight for y=1 (on top of class_weight='balanced')."""
    return y.map({0: 1.0, 1: minority_weight}).to_numpy(dtype=float)


def _build_pipe(params: dict, random_state: int, n_jobs: int = -1) -> Pipeline:
    """
    Unfitted pipeline from canonical params; the single builder for tuning,
    evaluation, the final fit and diagnostics. With random_state fixed, the
    fitted trees do not depend on n_jobs. Use _fit() to train.
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
    """Collapse Optuna's conditional scaffolding (use_max_depth, max_features_type/fraction) into model params."""
    max_depth = raw.get("max_depth") if raw.get("use_max_depth", False) else None
    mf_type = raw.get("max_features_type", "sqrt")
    max_features = raw.get("max_features_fraction") if mf_type == "fraction" else mf_type
    return {
        "n_estimators":     raw["n_estimators"],
        "max_depth":        max_depth,
        "min_samples_leaf": raw["min_samples_leaf"],
        "max_features":     max_features,
        "minority_weight":  raw.get("minority_weight", 1.0),
        "class_weight":     "balanced",
    }


def _suggest_params(trial: optuna.Trial) -> dict:
    """Sample one point from the search space (see module docstring); returns canonical params."""
    raw = {"n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=50)}
    raw["use_max_depth"] = trial.suggest_categorical("use_max_depth", [True, False])
    if raw["use_max_depth"]:
        raw["max_depth"] = trial.suggest_int("max_depth", 2, 12)
    raw["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 50)
    raw["max_features_type"] = trial.suggest_categorical(
        "max_features_type", ["sqrt", "log2", "fraction"]
    )
    if raw["max_features_type"] == "fraction":
        raw["max_features_fraction"] = trial.suggest_float("max_features_fraction", 0.3, 1.0)
    raw["minority_weight"] = trial.suggest_float("minority_weight", 1.0, 15.0)
    return _normalize_params(raw)


def _fit(params: dict, X: pd.DataFrame, y: pd.Series, random_state: int) -> Pipeline:
    """
    Fresh pipeline fit with the class-balanced + minority sample weighting.

    Trees are built in parallel (identical to a serial fit), then the forest is
    set to n_jobs=1: parallel predict_proba sums trees in a varying order, so
    scores would differ by ~1e-16 between runs and could flip tied ranks.
    """
    pipe = _build_pipe(params, random_state)
    pipe.fit(X, y, clf__sample_weight=_make_sample_weights(y, params["minority_weight"]))
    pipe.set_params(clf__n_jobs=1)
    return pipe


def _fold_auc(params: dict, fold: Fold, random_state: int) -> float:
    """Validation ROC-AUC of a fresh pipeline fit on the fold's training rows."""
    model = _fit(params, fold.X_tr, fold.y_tr, random_state)
    return float(roc_auc_score(fold.y_va, model.predict_proba(fold.X_va)[:, 1]))


def _make_objective(folds: list[Fold], random_state: int):
    """Optuna objective: mean fold AUC; reports the running mean after each fold so the pruner can stop early."""
    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        scores: list[float] = []
        for i, fold in enumerate(folds):
            scores.append(_fold_auc(params, fold, random_state))
            trial.report(float(np.mean(scores)), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))
    return objective


def _tune(
    folds: list[Fold],
    n_trials: int,
    random_state: int,
) -> tuple[dict, optuna.Study, float, list[float]]:
    """
    Run the seeded, sequential Optuna search over prebuilt folds, then refit the
    best params on the same folds and raise unless the score reproduces exactly.
    Returns (params, study, runtime_s, best-trial fold AUCs).
    """
    print(f"Tuning RF: {n_trials} trials × {len(folds)} folds")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=15, n_warmup_steps=1),
        study_name="rf_binary_opt",
    )

    t0 = time.perf_counter()
    study.optimize(
        _make_objective(folds, random_state),
        n_trials=n_trials,
        n_jobs=1,                      # parallel trials are not reproducible
        show_progress_bar=True,
    )
    runtime_s = time.perf_counter() - t0

    params = _normalize_params(study.best_params)

    # Determinism check: the best trial must reproduce exactly on the same folds
    verify_scores = [_fold_auc(params, f, random_state) for f in folds]
    if not np.isclose(np.mean(verify_scores), study.best_value, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Best-trial CV AUC did not reproduce: study={study.best_value:.12f}, "
            f"rerun={np.mean(verify_scores):.12f}"
        )

    return params, study, runtime_s, verify_scores


def _evaluate(
    params: dict,
    folds: list[Fold],
    y: pd.Series,
    random_state: int,
) -> tuple[dict, pd.Series]:
    """Fit a fresh pipeline per fold and score its validation rows. Returns (metrics, oof_proba)."""
    oof = np.full(len(y), np.nan)
    auc_s, recall_s, precision_s, f1_s = [], [], [], []

    for fold in folds:
        model = _fit(params, fold.X_tr, fold.y_tr, random_state)
        y_proba = model.predict_proba(fold.X_va)[:, 1]
        y_pred = model.predict(fold.X_va)
        oof[fold.va_idx] = y_proba

        auc_s.append(roc_auc_score(fold.y_va, y_proba))
        recall_s.append(recall_score(fold.y_va, y_pred, zero_division=0))
        precision_s.append(precision_score(fold.y_va, y_pred, zero_division=0))
        f1_s.append(f1_score(fold.y_va, y_pred, zero_division=0))

    if np.isnan(oof).any():
        raise RuntimeError("CV did not produce an out-of-fold score for every row.")

    metrics = {
        "auc_mean": float(np.mean(auc_s)),
        "auc_std": float(np.std(auc_s)),
        "recall_mean": float(np.mean(recall_s)),
        "recall_std": float(np.std(recall_s)),
        "precision_mean": float(np.mean(precision_s)),
        "precision_std": float(np.std(precision_s)),
        "f1_mean": float(np.mean(f1_s)),
        "f1_std": float(np.std(f1_s)),
        "recall_at_10fpr": recall_at_fpr(y, oof, _TARGET_FPR),   # pooled out-of-fold
    }
    return metrics, pd.Series(oof, index=y.index, name="rf_oof_proba")


def _default_cv(n_folds: int, random_state: int) -> StratifiedKFold:
    """Shuffled StratifiedKFold used when no splitter is passed."""
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)


# ── Public API ────────────────────────────────────────────────────────────────
def tune_rf(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = _N_TRIALS,
    n_folds: int = _N_TUNE_FOLDS,
    random_state: int = _RANDOM_STATE,
    *,
    cv: Optional[BaseCrossValidator] = None,
    features: Optional[list[str]] = None,
    feature_factory: Optional[FeatureFactory] = None,
) -> tuple[dict, optuna.Study, float]:
    """
    Optuna search on CV ROC-AUC (training rows only).

    X, y            : Binary features — or raw features if feature_factory is
                      given — and the target. Indices must match.
    features        : Columns to model, in order (default: all columns of X).
    cv              : Splitter. Default: shuffled StratifiedKFold(n_folds,
                      random_state); n_folds is ignored when cv is given.
    feature_factory : Refit inside every fold.

    Returns (canonical params, study, runtime_s).
    """
    features = list(X.columns) if features is None else list(features)
    cv = cv if cv is not None else _default_cv(n_folds, random_state)
    folds = build_folds(X, y, features, cv, feature_factory, "tune", check=_require_binary)
    params, study, runtime_s, _ = _tune(folds, n_trials, random_state)
    return params, study, runtime_s


def evaluate_rf(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = _N_EVAL_FOLDS,
    random_state: int = _RANDOM_STATE,
    *,
    cv: Optional[BaseCrossValidator] = None,
    features: Optional[list[str]] = None,
    feature_factory: Optional[FeatureFactory] = None,
) -> dict:
    """
    Out-of-fold metrics for canonical RF params (keys as RFResult.metrics).
    Other arguments as in tune_rf().
    """
    features = list(X.columns) if features is None else list(features)
    cv = cv if cv is not None else _default_cv(n_folds, random_state)
    folds = build_folds(X, y, features, cv, feature_factory, "eval", check=_require_binary)
    metrics, _ = _evaluate(params, folds, y, random_state)
    return metrics


def train_rf(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_trials: int = _N_TRIALS,
    n_tune_folds: int = _N_TUNE_FOLDS,
    n_eval_folds: int = _N_EVAL_FOLDS,
    random_state: int = _RANDOM_STATE,
    baseline_auc: Optional[float] = None,
    *,
    cv: Optional[BaseCrossValidator] = None,
    feature_factory: Optional[FeatureFactory] = None,
) -> RFResult:
    """
    Tune (Optuna) → out-of-fold evaluation → final fit on all rows.

    Parameters
    ----------
    df              : Training rows only, including target_col.
                      With feature_factory: raw frame (df_train).
                      Without: df_rf_binary.
    features        : Model columns (RF_FEATURES_BINARY). Must exist in df, or
                      in the factory's output when feature_factory is given.
    target_col      : Target column; every other column goes to the factory
                      (or to feature selection).
    n_trials        : Optuna trial budget.
    n_tune_folds    : Tuning folds when cv is None.
    n_eval_folds    : Evaluation folds. Evaluation always uses its own shuffled
                      StratifiedKFold(n_eval_folds, random_state), separate from
                      the tuning folds.
    random_state    : Seeds the sampler, the default splitters and the forest.
    baseline_auc    : If given, prints Δ = OOF AUC − baseline_auc and a verdict
                      (see _AUC_TOLERANCE). Must come from the same protocol.
    cv              : Tuning splitter (pass the notebook's CV).
    feature_factory : e.g. partial(BinaryFeaturePipeline, make_feature_pipeline).
                      Refit inside every tuning and evaluation fold, and once on
                      all rows for the final fit.

    Returns
    -------
    RFResult. Threshold selection is not done here; use .oof_proba.
    """
    if target_col not in df.columns:
        raise KeyError(f"target column {target_col!r} not in df")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    features = list(features)

    tune_cv = cv if cv is not None else _default_cv(n_tune_folds, random_state)
    eval_cv = _default_cv(n_eval_folds, random_state)

    # ── Tune ──────────────────────────────────────────────────────────────────
    tune_folds = build_folds(X, y, features, tune_cv, feature_factory, "tune", check=_require_binary)
    params, study, runtime_s, tune_scores = _tune(tune_folds, n_trials, random_state)

    # ── Out-of-fold evaluation ────────────────────────────────────────────────
    eval_folds = build_folds(X, y, features, eval_cv, feature_factory, "eval", check=_require_binary)
    metrics, oof_proba = _evaluate(params, eval_folds, y, random_state)

    # ── Final fit on all input rows ───────────────────────────────────────────
    X_full = engineer_full(X, y, features, feature_factory, check=_require_binary)
    pipe = _fit(params, X_full, y, random_state)

    clf = pipe.named_steps["clf"]
    feature_importance = (
        pd.DataFrame({"feature": features, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # ── Report ────────────────────────────────────────────────────────────────
    depth = params["max_depth"] if params["max_depth"] is not None else "None"
    mf = params["max_features"]
    params_line = (
        f"trees={params['n_estimators']} · depth={depth} · leaf={params['min_samples_leaf']} · "
        f"max_features={mf if isinstance(mf, str) else f'{mf:.2f}'} · "
        f"minority_weight={params['minority_weight']:.2f} · class_weight=balanced"
    )
    training_report(
        f"RANDOM FOREST ({len(features)} binary features, "
        f"{X_full.drop_duplicates().shape[0]:,} observed patterns)",
        y=y, n_features=len(features), feature_step=factory_name(feature_factory),
        study=study, runtime_s=runtime_s, tune_fold_scores=tune_scores,
        params_line=params_line, metrics=metrics, n_eval_folds=n_eval_folds,
        baseline_auc=baseline_auc, tolerance=_AUC_TOLERANCE,
    )
    top = feature_importance.head(5)
    print("Top Gini   " + " · ".join(f"{f} {v:.3f}" for f, v in zip(top.feature, top.importance))
          + "   (full ranking: rf_diagnostics)")

    return RFResult(
        pipe=pipe,
        params=params,
        metrics=metrics,
        feature_importance=feature_importance,
        study=study,
        runtime_s=runtime_s,
        oof_proba=oof_proba,
    )