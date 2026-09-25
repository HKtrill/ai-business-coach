"""
blackbox_pipeline.models.xgb.evaluation.metrics
================================================
The Stage 3 metric block.

Superset of what ``glass_pipeline.ebm.evaluation.evaluate_ebm`` returns —
``to_glass_keys()`` emits exactly the EBM's seven keys, under the EBM's spelling,
so the two arms drop into one table without a translation layer. The extra
fields (Brier, ECE, confusion matrix, coverage, positive prediction rate) are
the ones PR 3 asks for on top.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..calibration import calculate_ece

__all__ = ["Stage3Metrics"]


@dataclass
class Stage3Metrics:
    """Scalar metrics at one operating point. Probabilities are not stored."""

    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    roc_auc: float
    brier: float
    ece: float
    tn: int
    fp: int
    fn: int
    tp: int
    n: int
    base_rate: float
    positive_prediction_rate: float
    coverage: float
    split: str = ""
    probability_space: str = "raw"
    ece_full: float = float("nan")   # every-row ECE; ``ece`` is GLASS-parity

    # ------------------------------------------------------------------
    @classmethod
    def compute(
        cls,
        y_true,
        proba: np.ndarray,
        threshold: float,
        *,
        split: str = "",
        probability_space: str = "raw",
        ece_bins: int = 10,
        n_population: Optional[int] = None,
        decision: Optional[np.ndarray] = None,
    ) -> "Stage3Metrics":
        """
        Parameters
        ----------
        n_population
            Rows that existed before Stage 3 selected its population. Stage 3
            currently runs on the full split, so coverage is 1.0; the parameter
            exists so the number stays meaningful if Stage 3 is ever moved
            behind the Stage 2 router.
        decision
            Precomputed 0/1 decisions (e.g. from per-fold thresholds). When
            given, it replaces ``proba >= threshold`` and ``threshold`` is
            recorded as a summary value only.
        """
        y = np.asarray(y_true).astype(int)
        p = np.asarray(proba, dtype=float)
        if len(y) != len(p):
            raise ValueError(f"length mismatch: y {len(y)}, proba {len(p)}")
        if len(y) == 0:
            raise ValueError("empty population")

        if decision is None:
            pred = (p >= float(threshold)).astype(int)
        else:
            pred = np.asarray(decision).astype(int)
            if len(pred) != len(y):
                raise ValueError(
                    f"length mismatch: y {len(y)}, decision {len(pred)}"
                )
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        n = len(y)

        # roc_auc is undefined on a single-class population
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")

        return cls(
            threshold=float(threshold),
            accuracy=float(accuracy_score(y, pred)),
            precision=float(precision_score(y, pred, zero_division=0)),
            recall=float(recall_score(y, pred, zero_division=0)),
            f1=float(f1_score(y, pred, zero_division=0)),
            f2=float(fbeta_score(y, pred, beta=2, zero_division=0)),
            roc_auc=auc,
            brier=float(brier_score_loss(y, p)),
            ece=calculate_ece(y, p, ece_bins),
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
            n=int(n),
            base_rate=float(y.mean()),
            positive_prediction_rate=float(pred.mean()),
            coverage=float(n / n_population) if n_population else 1.0,
            split=split,
            probability_space=probability_space,
            ece_full=calculate_ece(y, p, ece_bins, include_zero=True),
        )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_glass_keys(self) -> dict:
        """The EBM's ``evaluate_ebm`` dict, key for key."""
        return {
            "Threshold": self.threshold,
            "Accuracy": self.accuracy,
            "Precision": self.precision,
            "Recall": self.recall,
            "F1": self.f1,
            "F2": self.f2,
            "ROC-AUC": self.roc_auc,
        }

    def confusion(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[self.tn, self.fp], [self.fn, self.tp]],
            index=["actual 0", "actual 1"],
            columns=["pred 0", "pred 1"],
        )

    def describe(self) -> str:
        label = f"{self.split} " if self.split else ""
        return (
            f"{label}@ {self.threshold:.4f} ({self.probability_space})  "
            f"acc={self.accuracy:.4f}  prec={self.precision:.4f}  "
            f"rec={self.recall:.4f}  F1={self.f1:.4f}  F2={self.f2:.4f}  "
            f"AUC={self.roc_auc:.4f}  Brier={self.brier:.4f}  "
            f"ECE={self.ece:.4f}  PPR={self.positive_prediction_rate:.4f}"
        )

    @staticmethod
    def to_frame(metrics: list["Stage3Metrics"]) -> pd.DataFrame:
        return pd.DataFrame([m.to_dict() for m in metrics])