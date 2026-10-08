"""
model_training.lr.trainer
=========================
Logistic-regression training for the Glass Cascade gate stage.

Method
------
StandardScaler + LogisticRegression (saga, class_weight='balanced'), tuned with
Optuna (TPE) on mean ROC-AUC. Search space: C ∈ [1e-4, 1e3] (log), penalty ∈
{l1, l2, elasticnet}, l1_ratio ∈ [0, 1] for elasticnet. Balanced Optuna was kept
over GridSearchCV and a recall-weighted objective because all three reached the
same Recall@10%FPR. Threshold selection is out of scope; use `oof_proba`.

Leakage contract
----------------
- Callers pass training rows only. Every reported score is out-of-fold.
- With `feature_factory`, the feature pipeline is fit on each fold's training
  rows only, so target-derived features never see validation labels. Fold
  matrices are built once per CV and reused across Optuna trials.
- Non-numeric or non-finite features raise; nothing is imputed.

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
import sklearn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_research.model_training.shared.folds import (
    FeatureFactory,
    Fold,
    build_folds,
    engineer_full,
    factory_name,
    recall_at_fpr,
)
from feature_research.model_training.shared.report import show_table, training_report

# ── Module-level defaults ─────────────────────────────────────────────────────
_N_TUNE_FOLDS: int = 5
_N_EVAL_FOLDS: int = 10
_N_TRIALS: int = 100
_RANDOM_STATE: int = 42
_TARGET_FPR: float = 0.10

# |coefficient| bands for the feature report. Coefficients are on standardized
# features, so they read as log-odds change per 1 SD.
_ACTIVE_THRESHOLD: float = 0.15
_WEAK_THRESHOLD: float = 0.10

# Verdict tolerance relative to baseline_auc (≈ 1 standard error of the
# 10-fold mean AUC). Δ ≥ −tol → HOLD; Δ ≥ −2·tol → BELOW TARGET; else REGRESSED.
_AUC_TOLERANCE: float = 0.005

# scikit-learn ≥ 1.8 deprecates LogisticRegression(penalty=...); see _penalty_kwargs.
_SKLEARN_PENALTY_DEPRECATED: bool = (
    tuple(int(p) for p in sklearn.__version__.split(".")[:2]) >= (1, 8)
)


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class LRResult:
    """
    Output of train_lr().

    pipe               Pipeline(StandardScaler + LogisticRegression) fit on all
                       input rows. Input: engineered features in `features`
                       order, i.e. df_engineered[LR_FEATURES].
    params             Best Optuna params: C, penalty, and l1_ratio if elasticnet.
    metrics            Evaluation-CV metrics, out-of-fold:
                         auc_mean/std                     per-fold ROC-AUC
                         recall/precision/f1 _mean/std    per fold, at predict()'s
                                                          0.5 threshold
                         recall_at_10fpr                  pooled OOF, FPR ≤ 10%
    feature_importance feature, coefficient, abs_coef from the final fit,
                       sorted by abs_coef.
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
def _penalty_kwargs(params: dict) -> dict:
    """
    LogisticRegression penalty kwargs for a params dict.
    sklearn ≥ 1.8: l1_ratio only (l2 → 0, l1 → 1, elasticnet → its l1_ratio).
    Older: penalty, plus l1_ratio for elasticnet. Same models either way.
    """
    penalty = params["penalty"]
    if _SKLEARN_PENALTY_DEPRECATED:
        return {"l1_ratio": {"l2": 0.0, "l1": 1.0}.get(penalty, params.get("l1_ratio"))}
    kw = {"penalty": penalty}
    if penalty == "elasticnet":
        kw["l1_ratio"] = params["l1_ratio"]
    return kw


def _build_pipe(params: dict, random_state: int) -> Pipeline:
    """Unfitted pipeline for a params dict; the single builder for tuning, evaluation and the final fit."""
    lr_kwargs: dict = dict(
        C=params["C"],
        solver="saga",
        max_iter=5000,
        random_state=random_state,
        class_weight="balanced",
        **_penalty_kwargs(params),
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**lr_kwargs)),
    ])


def _suggest_params(trial: optuna.Trial) -> dict:
    """Sample one point from the search space (see module docstring)."""
    params = {
        "C": trial.suggest_float("C", 1e-4, 1e3, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
    }
    if params["penalty"] == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    return params


def _fold_auc(params: dict, fold: Fold, random_state: int) -> float:
    """Validation ROC-AUC of a fresh pipeline fit on the fold's training rows."""
    model = _build_pipe(params, random_state).fit(fold.X_tr, fold.y_tr)
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
    print(f"Tuning LR: {n_trials} trials × {len(folds)} folds")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        study_name="lr_bayesian_opt_balanced",
    )

    t0 = time.perf_counter()
    study.optimize(
        _make_objective(folds, random_state),
        n_trials=n_trials,
        n_jobs=1,                      # parallel trials are not reproducible
        show_progress_bar=True,
    )
    runtime_s = time.perf_counter() - t0

    params = study.best_params.copy()

    # Determinism check: the best trial must reproduce exactly on the same folds
    verify_scores = [_fold_auc(params, f, random_state) for f in folds]
    if not np.isclose(np.mean(verify_scores), study.best_value, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"Best-trial CV AUC did not reproduce: study={study.best_value:.12f}, "
            f"rerun={np.mean(verify_scores):.12f}"
        )

    return params, study, runtime_s, verify_scores


def _evaluate(
    template: Pipeline,
    folds: list[Fold],
    y: pd.Series,
) -> tuple[dict, pd.Series]:
    """
    Fit a clone of `template` per fold and score its validation rows.
    Returns (metrics, oof_proba); `template` is never fitted.
    """
    oof = np.full(len(y), np.nan)
    auc_s, recall_s, precision_s, f1_s = [], [], [], []

    for fold in folds:
        model = clone(template).fit(fold.X_tr, fold.y_tr)
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
    return metrics, pd.Series(oof, index=y.index, name="lr_oof_proba")


def _default_cv(n_folds: int, random_state: int) -> StratifiedKFold:
    """Shuffled StratifiedKFold used when no splitter is passed."""
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)


# ── Public API ────────────────────────────────────────────────────────────────
def tune_lr(
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

    X, y            : Engineered features — or raw features if feature_factory
                      is given — and the target. Indices must match.
    features        : Columns to model, in order (default: all columns of X).
    cv              : Splitter. Default: shuffled StratifiedKFold(n_folds,
                      random_state); n_folds is ignored when cv is given.
    feature_factory : Refit inside every fold (e.g. FeaturePipeline).

    Returns (params, study, runtime_s).
    """
    features = list(X.columns) if features is None else list(features)
    cv = cv if cv is not None else _default_cv(n_folds, random_state)
    folds = build_folds(X, y, features, cv, feature_factory, "tune")
    params, study, runtime_s, _ = _tune(folds, n_trials, random_state)
    return params, study, runtime_s


def evaluate_lr(
    pipe: Pipeline,
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
    Out-of-fold metrics for an LR pipeline (keys as LRResult.metrics).

    `pipe` is a template: a clone is fit per fold and `pipe` itself is left
    unfitted. Other arguments as in tune_lr().
    """
    features = list(X.columns) if features is None else list(features)
    cv = cv if cv is not None else _default_cv(n_folds, random_state)
    folds = build_folds(X, y, features, cv, feature_factory, "eval")
    metrics, _ = _evaluate(pipe, folds, y)
    return metrics


def train_lr(
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
) -> LRResult:
    """
    Tune (Optuna) → out-of-fold evaluation → final fit on all rows.

    Parameters
    ----------
    df              : Training rows only, including target_col.
                      With feature_factory: raw frame (df_train).
                      Without: df_engineered.
    features        : Model columns (LR_FEATURES). Must exist in df, or in the
                      factory's output when feature_factory is given.
    target_col      : Target column; every other column goes to the factory
                      (or to feature selection).
    n_trials        : Optuna trial budget.
    n_tune_folds    : Tuning folds when cv is None.
    n_eval_folds    : Evaluation folds. Evaluation always uses its own shuffled
                      StratifiedKFold(n_eval_folds, random_state), separate from
                      the tuning folds.
    random_state    : Seeds the sampler, the default splitters and the solver.
    baseline_auc    : If given, prints Δ = OOF AUC − baseline_auc and a verdict
                      (see _AUC_TOLERANCE). Must come from the same protocol.
    cv              : Tuning splitter (pass the notebook's CV).
    feature_factory : e.g. FeaturePipeline (the class). Refit inside every
                      tuning and evaluation fold, and once on all rows for the
                      final fit.

    Returns
    -------
    LRResult. Threshold selection is not done here; use .oof_proba.
    """
    if target_col not in df.columns:
        raise KeyError(f"target column {target_col!r} not in df")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    features = list(features)

    tune_cv = cv if cv is not None else _default_cv(n_tune_folds, random_state)
    eval_cv = _default_cv(n_eval_folds, random_state)

    # ── Tune ──────────────────────────────────────────────────────────────────
    tune_folds = build_folds(X, y, features, tune_cv, feature_factory, "tune")
    params, study, runtime_s, tune_scores = _tune(tune_folds, n_trials, random_state)

    # ── Out-of-fold evaluation ────────────────────────────────────────────────
    eval_folds = build_folds(X, y, features, eval_cv, feature_factory, "eval")
    metrics, oof_proba = _evaluate(_build_pipe(params, random_state), eval_folds, y)

    # ── Final fit on all input rows ───────────────────────────────────────────
    X_full = engineer_full(X, y, features, feature_factory)
    pipe = _build_pipe(params, random_state).fit(X_full, y)

    clf = pipe.named_steps["clf"]
    feature_importance = pd.DataFrame({
        "feature": features,
        "coefficient": clf.coef_[0],
        "abs_coef": np.abs(clf.coef_[0]),
    }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

    # ── Report ────────────────────────────────────────────────────────────────
    params_line = f"C={params['C']:.6g} · penalty={params['penalty']}"
    if params["penalty"] == "elasticnet":
        params_line += f" · l1_ratio={params['l1_ratio']:.3f}"
    training_report(
        "LOGISTIC REGRESSION (class_weight=balanced)",
        y=y, n_features=len(features), feature_step=factory_name(feature_factory),
        study=study, runtime_s=runtime_s, tune_fold_scores=tune_scores,
        params_line=params_line + " · class_weight=balanced",
        metrics=metrics, n_eval_folds=n_eval_folds,
        baseline_auc=baseline_auc, tolerance=_AUC_TOLERANCE,
    )
    print("Coefficients (final fit, log-odds per 1 SD)")
    show_table(
        feature_importance.assign(status=lambda d: np.select(
            [d.abs_coef >= _ACTIVE_THRESHOLD, d.abs_coef >= _WEAK_THRESHOLD],
            ["active", "weak"], "dead",
        )).set_index("feature")[["coefficient", "status"]].round(4)
    )

    return LRResult(
        pipe=pipe,
        params=params,
        metrics=metrics,
        feature_importance=feature_importance,
        study=study,
        runtime_s=runtime_s,
        oof_proba=oof_proba,
    )