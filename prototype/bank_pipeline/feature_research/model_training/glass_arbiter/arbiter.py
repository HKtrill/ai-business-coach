"""
feature_research/model_training/glass_arbiter/arbiter.py
========================================================
The GLASS Arbiter on the research stages (LR, RF, EBM): the same
weighted-confidence vote, trust weights, abstention rule and metrics as the
cascade's Stage 4 arbiter (glass_pipeline.meta_ebm). Pure functions, no I/O.

Difference from the cascade: all three research stages score every row, so
there are no router coverage masks; the cascade's GLASS Router votes only on
the rows it routes.

Rule (per row)
--------------
vote_m = score_m ≥ threshold_m
conf_m = |score_m − threshold_m| × weight_m          (weights normalised)
conf1 = Σ conf over models voting 1,  conf0 = Σ over models voting 0
abstain (−1) if max(conf1, conf0) < min_weighted_confidence
else 1 if conf1 > conf0 else 0
prob = Σ weight_m × score_m

Configuration (all from fit rows; see crossfit.py)
--------------------------------------------------
thresholds  find_recall_threshold at target_recall (cascade rule)
weights     ½ normalised inverse Brier + ½ normalised accuracy at threshold
min_conf    F2 on retained rows over ABSTENTION_GRID, coverage ≥ 0.50,
            strictly-greater F2 wins; nothing qualifies → fallback 0.07

Public API
----------
MODELS, ABSTENTION_GRID, ABSTENTION_MIN_COVERAGE, MIN_CONF_FALLBACK
check_scores · compute_calibration · compute_hybrid_weights · combine ·
explain · compute_metrics · evaluate_with_abstention · analyze_disagreements ·
tune_min_confidence · ArbiterConfig · fit_arbiter · apply_arbiter
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from feature_research.model_training.glass_arbiter.thresholds import pick_threshold

MODELS = ("lr", "rf", "ebm")

# Same constants as shared.stage4.protocol / glass_pipeline.meta_ebm
ABSTENTION_GRID = np.arange(0.03, 0.55, 0.02)
ABSTENTION_MIN_COVERAGE = 0.50
MIN_CONF_FALLBACK = 0.07
WEIGHT_ALPHA = 0.5


def check_scores(P: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
    """Raise unless P has columns MODELS (in order) with finite scores in [0, 1], aligned with y."""
    if tuple(P.columns) != MODELS:
        raise ValueError(f"score columns must be {MODELS}, got {tuple(P.columns)}")
    values = P.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("scores must be finite probabilities in [0, 1]")
    if y is not None and not P.index.equals(y.index):
        raise ValueError("scores and labels are not aligned on the same index")


def compute_calibration(y_true, y_prob, n_bins: int = 10) -> dict:
    """Brier + ECE; first bin closed on the left so scores of exactly 0 count (cascade convention)."""
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = (p >= bins[i]) if i == 0 else (p > bins[i])
        mask = lo & (p <= bins[i + 1])
        if mask.any():
            ece += abs(y[mask].mean() - p[mask].mean()) * mask.mean()
    return {"brier": float(brier_score_loss(y, p)), "ece": float(ece)}


def compute_hybrid_weights(y_true, P: pd.DataFrame, thresholds: dict, alpha: float = WEIGHT_ALPHA) -> dict:
    """α · normalised inverse Brier + (1 − α) · normalised accuracy at threshold; sums to 1."""
    y = np.asarray(y_true)
    X = P[list(MODELS)].to_numpy(dtype=float)
    briers = np.array([brier_score_loss(y, X[:, k]) for k in range(len(MODELS))])
    inv_b = 1.0 / (briers + 1e-6)
    inv_b /= inv_b.sum()
    accs_raw = np.array([accuracy_score(y, (X[:, k] >= thresholds[m]).astype(int)) for k, m in enumerate(MODELS)])
    accs = accs_raw / accs_raw.sum()
    hybrid = alpha * inv_b + (1 - alpha) * accs
    hybrid /= hybrid.sum()
    return {
        **dict(zip(MODELS, map(float, hybrid))),
        "details": {"brier": briers.tolist(), "inv_brier": inv_b.tolist(),
                    "accuracy": accs.tolist(), "accuracy_raw": accs_raw.tolist(), "alpha": float(alpha)},
    }


def _weighted_conf(P: pd.DataFrame, thresholds: dict, weights: dict):
    X = P[list(MODELS)].to_numpy(dtype=float)
    T = np.array([thresholds[m] for m in MODELS], dtype=float)
    w = np.array([weights[m] for m in MODELS], dtype=float)
    w = w / w.sum()
    votes = X >= T
    conf = np.abs(X - T) * w
    return (conf * votes).sum(axis=1), (conf * ~votes).sum(axis=1), X @ w


def combine(
    P: pd.DataFrame,
    thresholds: dict,
    weights: dict,
    min_conf: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Arbiter decision for every row (vectorised cascade meta_arbiter). min_conf=None → no abstention."""
    conf1, conf0, prob = _weighted_conf(P, thresholds, weights)
    pred = (conf1 > conf0).astype(int)  # ties → 0
    if min_conf is not None:
        pred[np.maximum(conf1, conf0) < min_conf] = -1
    return pred, prob


def explain(pred: np.ndarray) -> np.ndarray:
    """Per-row decision text, as in the cascade."""
    return np.where(pred == -1, "ABSTAIN: low weighted confidence",
                    np.where(pred == 1, "PREDICT 1: weighted confidence", "PREDICT 0: weighted confidence"))


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Cascade compute_metrics: roc_auc = 0.5 when scores are missing or one class only."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, np.asarray(y_prob)))
    else:
        out["roc_auc"] = 0.5
    return out


def evaluate_with_abstention(y_true, y_pred, y_prob) -> dict:
    """coverage, abstain_rate, and metrics on retained rows (None if nothing retained)."""
    y_true, y_pred, y_prob = np.asarray(y_true), np.asarray(y_pred), np.asarray(y_prob)
    covered = y_pred != -1
    return {
        "coverage": float(covered.mean()),
        "abstain_rate": float((~covered).mean()),
        "metrics": compute_metrics(y_true[covered], y_pred[covered], y_prob[covered]) if covered.any() else None,
    }


def analyze_disagreements(y_true, model_preds: pd.DataFrame) -> dict:
    """Per pair: disagreement rate and each model's accuracy where the pair disagrees."""
    y = np.asarray(y_true)
    report = {}
    for a, b in combinations(MODELS, 2):
        pa, pb = model_preds[a].to_numpy(), model_preds[b].to_numpy()
        dis = pa != pb
        report[f"{a}_{b}_disagree_rate"] = float(dis.mean())
        report[f"{a}_correct_on_{a}_{b}_disagree"] = float((pa[dis] == y[dis]).mean()) if dis.any() else float("nan")
        report[f"{b}_correct_on_{a}_{b}_disagree"] = float((pb[dis] == y[dis]).mean()) if dis.any() else float("nan")
    return report


def tune_min_confidence(
    P: pd.DataFrame,
    y_true,
    thresholds: dict,
    weights: dict,
    grid: np.ndarray = ABSTENTION_GRID,
    min_coverage: float = ABSTENTION_MIN_COVERAGE,
) -> Optional[dict]:
    """
    Cascade rule: best F2 on retained rows over `grid`, coverage ≥ min_coverage,
    strictly-greater F2 wins (ties → smaller value). None if nothing qualifies.
    """
    y = np.asarray(y_true)
    conf1, conf0, _ = _weighted_conf(P, thresholds, weights)
    pred = (conf1 > conf0).astype(int)
    confidence = np.maximum(conf1, conf0)
    best_score, best, sweep = 0.0, None, []
    for mc in grid:
        covered = confidence >= mc       # abstain where max(conf1, conf0) < mc
        cov = float(covered.mean())
        f2 = float(fbeta_score(y[covered], pred[covered], beta=2, zero_division=0)) if covered.any() else 0.0
        sweep.append({"min_conf": float(mc), "coverage": cov, "f2": f2, "eligible": cov >= min_coverage})
        if cov >= min_coverage and f2 > best_score:
            best_score = f2
            best = {"min_weighted_confidence": float(mc), "fit_f2": f2, "fit_coverage": cov}
    if best is not None:
        best["sweep"] = pd.DataFrame(sweep).set_index("min_conf")
    return best


@dataclass
class ArbiterConfig:
    """Everything the arbiter learned from its fit rows (frozen before it is applied)."""
    rule: str
    target_recall: float
    thresholds: dict
    weights: dict
    min_conf: float
    selected_by: str
    fit_f2: Optional[float]
    fit_coverage: Optional[float]
    weights_details: dict = field(default_factory=dict, repr=False)
    sweep: Optional[pd.DataFrame] = field(default=None, repr=False)


def fit_arbiter(
    P: pd.DataFrame,
    y_true,
    rule: str = "recall",
    target_recall: float = 0.70,
    grid: np.ndarray = ABSTENTION_GRID,
    min_coverage: float = ABSTENTION_MIN_COVERAGE,
) -> ArbiterConfig:
    """Thresholds → hybrid weights → min_weighted_confidence, all from (P, y_true)."""
    thresholds = {m: pick_threshold(rule, y_true, P[m].to_numpy(), target_recall) for m in MODELS}
    w = compute_hybrid_weights(y_true, P, thresholds)
    weights = {m: w[m] for m in MODELS}
    best = tune_min_confidence(P, y_true, thresholds, weights, grid, min_coverage)
    if best is None:
        return ArbiterConfig(rule, target_recall, thresholds, weights, MIN_CONF_FALLBACK,
                             "fallback default (no sweep value met coverage floor)", None, None, w["details"])
    return ArbiterConfig(rule, target_recall, thresholds, weights, best["min_weighted_confidence"],
                         f"F2 sweep on fit rows, coverage ≥ {min_coverage:.0%}", best["fit_f2"],
                         best["fit_coverage"], w["details"], best["sweep"])


def apply_arbiter(P: pd.DataFrame, config: ArbiterConfig, abstain: bool = True) -> tuple[np.ndarray, np.ndarray]:
    return combine(P, config.thresholds, config.weights, config.min_conf if abstain else None)
