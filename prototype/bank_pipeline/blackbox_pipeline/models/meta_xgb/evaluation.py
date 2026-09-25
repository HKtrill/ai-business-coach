"""
blackbox_pipeline.models.meta_xgb.evaluation
============================================
Metric blocks for Stage 4, used identically for Meta-XGB and (in the
comparison) Meta-EBM: no-abstention metrics on every row, and with-abstention
coverage + metrics on the retained rows.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, fbeta_score,
    precision_score, recall_score, roc_auc_score,
)


def classification_metrics(y, pred, score: Optional[np.ndarray] = None) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    out = {
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "positive_prediction_rate": float(pred.mean()) if len(y) else float("nan"),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else float("nan"),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
    }
    if score is not None and len(np.unique(y)) > 1:
        s = np.asarray(score, dtype=float)
        out["roc_auc"] = float(roc_auc_score(y, s))
        out["pr_auc"] = float(average_precision_score(y, s))
    return out


def abstention_metrics(y, pred, score, retained) -> dict:
    """Coverage + metrics on retained rows (pred / score for every row)."""
    r = np.asarray(retained, dtype=bool)
    y, pred = np.asarray(y), np.asarray(pred)
    s = None if score is None else np.asarray(score, dtype=float)
    return {
        "coverage": float(r.mean()),
        "abstain_rate": float((~r).mean()),
        "n_retained": int(r.sum()),
        "metrics": (classification_metrics(y[r], pred[r], None if s is None else s[r])
                    if r.any() else None),
    }
