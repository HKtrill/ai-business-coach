"""
feature_research/model_training/ebm/model.py
=============================================
EBM construction shared by tuning, evaluation and the final fit, so the three
can never drift: search space, canonical params, per-fit balanced weights,
scoring.

Public API
----------
PARAM_KEYS                       canonical EBM hyperparameters
suggest_params(trial)            Optuna search space
normalize_params(params)         canonical dict (raises on missing keys)
fit_model(params, X, y, seed)    balanced-weight EBM fitted on (X, y)
positive_proba(model, X)         P(y=1)
hard_labels(proba)               0/1 at the 0.5 cut (same as predict())
high_recall_pauc(y, proba)       standardised partial AUC for TPR ≥ HIGH_RECALL_TPR
OBJECTIVES                       "auc_recall" (default) · "auc" · "f2" → (report label, fold score fn)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import optuna
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import fbeta_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_sample_weight

PARAM_KEYS = ("learning_rate", "max_rounds", "max_bins", "max_interaction_bins", "interactions")

HIGH_RECALL_TPR = 0.70  # high-recall band of the ROC curve used by "auc_recall"

ScoreFn = Callable[[pd.Series, np.ndarray], float]


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate":        trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_rounds":           trial.suggest_int("max_rounds", 500, 5000, step=100),
        "max_bins":             trial.suggest_int("max_bins", 128, 512, step=32),
        "max_interaction_bins": trial.suggest_int("max_interaction_bins", 16, 128, step=16),
        "interactions":         trial.suggest_int("interactions", 0, 5),  # cap: interpretability
    }


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Canonical params in PARAM_KEYS order; extra keys dropped, missing keys raise."""
    missing = [k for k in PARAM_KEYS if k not in params]
    if missing:
        raise KeyError(f"EBM params missing {missing}")
    return {k: params[k] for k in PARAM_KEYS}


def fit_model(
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    n_jobs: int = -1,
) -> ExplainableBoostingClassifier:
    """
    Balanced-weight EBM. Weights come from this fit's own labels (same as
    class_weight="balanced" in LR/RF). random_state seeds the outer bags and
    the internal early-stopping split, so n_jobs only changes speed.
    """
    model = ExplainableBoostingClassifier(
        **normalize_params(params), random_state=random_state, n_jobs=n_jobs,
    )
    model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return model


def positive_proba(model: ExplainableBoostingClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def hard_labels(proba: np.ndarray) -> np.ndarray:
    """0/1 at the 0.5 cut; ties go to 0, as in predict()."""
    return (proba > 0.5).astype(int)


def _f2_at_half(y: pd.Series, proba: np.ndarray) -> float:
    return float(fbeta_score(y, hard_labels(proba), beta=2, zero_division=0))


def _auc(y: pd.Series, proba: np.ndarray) -> float:
    return float(roc_auc_score(y, proba))


def high_recall_pauc(y, proba, min_tpr: float = HIGH_RECALL_TPR) -> float:
    """
    Standardised partial AUC over the high-recall band of the ROC curve
    (TPR ≥ min_tpr): how few negatives score above the hardest-to-catch
    positives. Same scale as AUC: random = 0.5, perfect = 1.0.
    """
    fpr, tpr, _ = roc_curve(y, proba)
    k = int(np.searchsorted(tpr, min_tpr, side="left"))  # first point with tpr ≥ min_tpr
    if k == 0:
        f0 = fpr[0]
    else:  # interpolate where the curve crosses tpr = min_tpr
        f0 = fpr[k - 1] + (fpr[k] - fpr[k - 1]) * (min_tpr - tpr[k - 1]) / (tpr[k] - tpr[k - 1])
    t = np.r_[min_tpr, tpr[k:]]
    keep = 1.0 - np.r_[f0, fpr[k:]]                       # specificity along the band
    area = float(np.sum(np.diff(t) * (keep[:-1] + keep[1:]) / 2))
    max_area = 1.0 - min_tpr                              # perfect ranking
    min_area = (1.0 - min_tpr) ** 2 / 2                   # random ranking
    return 0.5 * (1.0 + (area - min_area) / (max_area - min_area))


def _auc_recall(y: pd.Series, proba: np.ndarray) -> float:
    """Half full AUC, half high-recall partial AUC: ranking-based, leans to recall."""
    return 0.5 * _auc(y, proba) + 0.5 * high_recall_pauc(y, proba)


# objective name → (label for reports, fold score function)
OBJECTIVES: dict[str, tuple[str, ScoreFn]] = {
    "auc_recall": (f"½AUC+½pAUC(TPR≥{HIGH_RECALL_TPR:.2f})", _auc_recall),
    "auc":        ("AUC", _auc),
    "f2":         ("F2@0.5", _f2_at_half),
}
