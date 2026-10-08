"""
feature_research/model_training/ebm/trainer.py
===============================================
Stage 3 EBM trainer — same protocol as train_lr / train_rf.

Protocol
--------
- Input is the raw training rows plus a feature factory; the feature pipeline
  is refit inside every tuning and evaluation fold (target-derived features
  never see validation labels).
- Optuna tunes on `cv`; the best trial is refit on the same folds and must
  reproduce its fold scores exactly.
- Evaluation uses its own 10-fold split; every metric is out-of-fold and
  `oof_proba` holds one score per training row (input to threshold cells).
- The final model is fit on all training rows (diagnostics only).
- Nothing is imputed: missing or non-finite features raise.

Objective (picks hyperparameters only; the EBM loss stays log-loss and the
operating threshold is re-selected downstream on OOF scores)
---------
"auc_recall" (default) ½ AUC + ½ standardised partial AUC over TPR ≥ 0.70.
             Threshold-free; leans toward ranking the hardest positives
             above negatives, i.e. the high-recall operating range.
"auc"        plain AUC, same criterion as LR/RF.
"f2"         F2 at the 0.5 cut (previous default; rewards shifting scores
             above 0.5 more than ranking).

Determinism
-----------
Every EBM uses the same random_state (outer bags + internal early-stopping
split); TPE sampler seeded. Outer bags run in parallel (n_jobs=-1), which
changes speed only; the reproduction check fails loudly if it does not.

Public API
----------
EBMResult   result dataclass
train_ebm   tune → reproduce → out-of-fold evaluation → full fit
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

from feature_research.model_training.ebm.model import (
    HIGH_RECALL_TPR,
    OBJECTIVES,
    ScoreFn,
    fit_model,
    hard_labels,
    high_recall_pauc,
    normalize_params,
    positive_proba,
    suggest_params,
)
from feature_research.model_training.shared.folds import (
    FeatureFactory,
    Fold,
    build_folds,
    engineer_full,
    factory_name,
    recall_at_fpr,
)
from feature_research.model_training.shared.report import training_report

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class EBMResult:
    """
    pipe             EBM fitted on all training rows (diagnostics only)
    params           canonical hyperparameters
    metrics          out-of-fold means/stds + pooled recall_at_10fpr
    oof_proba        out-of-fold P(y=1), indexed like the training rows
    features         model columns, in order
    feature_step     how features were built (refit per fold — …)
    objective        tuning objective key ("f2" / "auc")
    study            Optuna study (None when params were fixed)
    tune_fold_scores best trial's fold scores (reproduced exactly)
    runtime_s        total wall time
    """
    pipe: ExplainableBoostingClassifier
    params: dict[str, Any]
    metrics: dict[str, float]
    oof_proba: pd.Series
    features: list[str]
    feature_step: str
    objective: str
    study: Optional[optuna.Study]
    tune_fold_scores: list[float]
    runtime_s: float


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def _fold_scores(
    params: dict[str, Any],
    folds: list[Fold],
    score_fn: ScoreFn,
    random_state: int,
    n_jobs: int,
    trial: Optional[optuna.Trial] = None,
) -> list[float]:
    """One EBM per fold, scored on its validation rows; reports to `trial` for pruning."""
    scores: list[float] = []
    for k, f in enumerate(folds):
        model = fit_model(params, f.X_tr, f.y_tr, random_state, n_jobs)
        scores.append(score_fn(f.y_va, positive_proba(model, f.X_va)))
        if trial is not None:
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return scores


def _tune(
    folds: list[Fold],
    score_fn: ScoreFn,
    n_trials: int,
    random_state: int,
    n_jobs: int,
    study_name: str,
    show_progress_bar: bool,
) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        scores = _fold_scores(suggest_params(trial), folds, score_fn, random_state, n_jobs, trial)
        trial.set_user_attr("fold_scores", scores)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
        pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=2),
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=show_progress_bar)
    return study


def _reproduce_best(
    study: optuna.Study,
    folds: list[Fold],
    score_fn: ScoreFn,
    random_state: int,
    n_jobs: int,
) -> tuple[dict[str, Any], list[float]]:
    """Refit the best trial on the tuning folds; scores must match exactly."""
    params = normalize_params(study.best_params)
    expected = list(study.best_trial.user_attrs["fold_scores"])
    got = _fold_scores(params, folds, score_fn, random_state, n_jobs)
    if got != expected:
        raise RuntimeError(f"EBM best trial did not reproduce: tuned {expected} vs refit {got}")
    return params, got


def _interaction_line(study: optuna.Study) -> str:
    """Best completed trial with vs without interaction terms."""
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    with_ix = [t.value for t in done if t.params["interactions"] > 0]
    without = [t.value for t in done if t.params["interactions"] == 0]
    if not with_ix or not without:
        return f"only one side sampled ({len(with_ix)} with, {len(without)} without)"
    delta = max(with_ix) - max(without)
    return (
        f"best with {max(with_ix):.4f} ({len(with_ix)} trials) · "
        f"without {max(without):.4f} ({len(without)}) · Δ {delta:+.4f}"
    )


# ---------------------------------------------------------------------------
# Out-of-fold evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    params: dict[str, Any],
    folds: list[Fold],
    index: pd.Index,
    random_state: int,
    n_jobs: int,
) -> tuple[pd.Series, dict[str, list[float]]]:
    oof = np.full(len(index), np.nan)
    per_fold: dict[str, list[float]] = {
        k: [] for k in ("auc", "pauc_hi_recall", "recall", "precision", "f1", "f2")
    }
    for f in folds:
        proba = positive_proba(fit_model(params, f.X_tr, f.y_tr, random_state, n_jobs), f.X_va)
        oof[f.va_idx] = proba
        labels = hard_labels(proba)
        per_fold["auc"].append(roc_auc_score(f.y_va, proba))
        per_fold["pauc_hi_recall"].append(high_recall_pauc(f.y_va, proba))
        per_fold["recall"].append(recall_score(f.y_va, labels, zero_division=0))
        per_fold["precision"].append(precision_score(f.y_va, labels, zero_division=0))
        per_fold["f1"].append(f1_score(f.y_va, labels, zero_division=0))
        per_fold["f2"].append(fbeta_score(f.y_va, labels, beta=2, zero_division=0))
    if np.isnan(oof).any():
        raise RuntimeError(f"OOF scores missing for {int(np.isnan(oof).sum())} rows")
    return pd.Series(oof, index=index, name="ebm_oof"), per_fold


def _params_line(p: dict[str, Any]) -> str:
    return (
        f"learning_rate {p['learning_rate']:.4g} · max_rounds {p['max_rounds']} · "
        f"max_bins {p['max_bins']} · max_interaction_bins {p['max_interaction_bins']} · "
        f"interactions {p['interactions']} · balanced weights"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_ebm(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    *,
    cv: BaseCrossValidator,
    feature_factory: Optional[FeatureFactory] = None,
    objective: str = "auc_recall",
    n_trials: int = 150,
    n_folds_eval: int = 10,
    random_state: int = 42,
    params: Optional[dict[str, Any]] = None,
    n_jobs: int = -1,
    baseline_auc: Optional[float] = None,
    tolerance: float = 0.005,
    study_name: str = "ebm_feature_research",
    show_progress_bar: bool = True,
) -> EBMResult:
    """
    X, y            training rows only (df_train / y_train)
    features        model columns (EBM_FEATURES), produced by `feature_factory`
    cv              tuning splitter (CV from Cell 5)
    feature_factory refit per fold; None = X is already model-ready
    objective       "auc_recall" (default), "auc" or "f2" — see module docstring
    params          fixed hyperparameters → skip tuning (reruns, determinism checks)
    n_jobs          EBM outer-bag parallelism; does not change results
    baseline_auc    relative verdict on OOF AUC (± tolerance)
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {list(OBJECTIVES)}, got {objective!r}")
    metric_label, score_fn = OBJECTIVES[objective]
    features = list(features)
    if len(set(features)) != len(features):
        raise ValueError("duplicate names in features")

    t_start = time.perf_counter()

    # Tuning (skipped when params are given)
    study: Optional[optuna.Study] = None
    tune_scores: list[float] = []
    tune_runtime = 0.0
    if params is None:
        tune_folds = build_folds(X, y, features, cv, feature_factory, label="tune")
        t0 = time.perf_counter()
        study = _tune(tune_folds, score_fn, n_trials, random_state, n_jobs, study_name, show_progress_bar)
        tune_runtime = time.perf_counter() - t0
        params, tune_scores = _reproduce_best(study, tune_folds, score_fn, random_state, n_jobs)
        del tune_folds
    else:
        params = normalize_params(params)

    # Out-of-fold evaluation on its own split
    eval_cv = StratifiedKFold(n_splits=n_folds_eval, shuffle=True, random_state=random_state)
    eval_folds = build_folds(X, y, features, eval_cv, feature_factory, label="eval")
    oof, per_fold = _evaluate(params, eval_folds, X.index, random_state, n_jobs)
    del eval_folds

    metrics: dict[str, float] = {}
    for name, values in per_fold.items():
        metrics[f"{name}_mean"] = float(np.mean(values))
        metrics[f"{name}_std"] = float(np.std(values))
    metrics["recall_at_10fpr"] = recall_at_fpr(y, oof.to_numpy(), 0.10)  # pooled OOF scores

    # Final fit on all training rows
    X_full = engineer_full(X, y, features, feature_factory)
    model = fit_model(params, X_full, y, random_state, n_jobs)

    feature_step = factory_name(feature_factory)
    training_report(
        "Stage 3 · EBM",
        y=y,
        n_features=len(features),
        feature_step=feature_step,
        study=study,
        runtime_s=tune_runtime,
        tune_fold_scores=tune_scores,
        params_line=_params_line(params),
        metrics=metrics,
        n_eval_folds=n_folds_eval,
        baseline_auc=baseline_auc,
        tolerance=tolerance,
        tune_metric=metric_label,
    )
    print(
        f"Recall     high-recall pAUC (TPR≥{HIGH_RECALL_TPR:.2f}) "
        f"{metrics['pauc_hi_recall_mean']:.4f} ± {metrics['pauc_hi_recall_std']:.4f}"
    )
    if study is not None:
        print(f"Pairs      {_interaction_line(study)}")
    if objective != "auc" and baseline_auc is not None:
        print(f"           baseline compares OOF AUC; tuning maximised {metric_label}")

    return EBMResult(
        pipe=model,
        params=params,
        metrics=metrics,
        oof_proba=oof,
        features=features,
        feature_step=feature_step,
        objective=objective,
        study=study,
        tune_fold_scores=tune_scores,
        runtime_s=time.perf_counter() - t_start,
    )
