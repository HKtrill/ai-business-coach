"""
feature_research/model_training/glass_arbiter/thresholds.py
===========================================================
Decision-threshold rules. Always called on fit rows only (cross-fit), never
on the rows being evaluated.

find_recall_threshold is the cascade's rule (glass_pipeline.meta_ebm.
meta_stage.find_recall_threshold), copied exactly.

Public API
----------
find_recall_threshold(y, p, target_recall) → highest observed score with recall ≥ target
find_youden_threshold(y, p)                → (threshold, tpr, fpr) at max TPR − FPR
find_f2_threshold(y, p)                    → (threshold, recall, precision, f2) at max F2
pick_threshold(rule, y, p, target_recall)  → float, rule ∈ RULES
RULES                                      ("recall", "youden", "f2")
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

RULES = ("recall", "youden", "f2")


def find_recall_threshold(y_true, y_prob, target_recall: float = 0.70) -> float:
    """
    HIGHEST threshold t (among the observed scores) with recall(y_prob ≥ t) ≥
    target_recall. Falls back to the minimum score (recall = 1.0) if none
    qualify; 0.5 if there are no positives. Same as the cascade's rule.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    pos = np.sort(y_prob[y_true == 1])
    n_pos = len(pos)
    if n_pos == 0:
        return 0.5
    thresholds = np.unique(y_prob)[::-1]                                  # descending
    n_hit = n_pos - np.searchsorted(pos, thresholds, side="left")        # positives with p ≥ t
    ok = n_hit / n_pos >= target_recall
    return float(thresholds[np.argmax(ok)]) if ok.any() else float(thresholds[-1])


def find_youden_threshold(y_true, y_prob) -> tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best = int(np.argmax(tpr - fpr))
    return float(thresholds[best]), float(tpr[best]), float(fpr[best])


def find_f2_threshold(y_true, y_prob) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    p, r = precision[:-1], recall[:-1]
    f2 = np.divide(5 * p * r, 4 * p + r, out=np.zeros_like(p), where=(4 * p + r) > 0)
    best = int(np.argmax(f2))
    return float(thresholds[best]), float(r[best]), float(p[best]), float(f2[best])


def pick_threshold(rule: str, y_true, y_prob, target_recall: float = 0.70) -> float:
    if rule == "recall":
        return find_recall_threshold(y_true, y_prob, target_recall)
    if rule == "youden":
        return find_youden_threshold(y_true, y_prob)[0]
    if rule == "f2":
        return find_f2_threshold(y_true, y_prob)[0]
    raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
