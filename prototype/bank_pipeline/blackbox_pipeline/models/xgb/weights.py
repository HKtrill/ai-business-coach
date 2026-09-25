"""
blackbox_pipeline.models.xgb.weights
=====================================
Class-imbalance treatment.

The EBM calls ``compute_sample_weight("balanced", y_train)`` once over the full
training split and slices that vector per fold, using the identical array in the
Optuna objective, the 10-fold evaluation block and the full-data refit.

XGBoost is given the SAME vector through ``fit(sample_weight=...)``. The obvious
alternative, ``scale_pos_weight``, is deliberately not used: it is a scalar
re-derivation of the same idea, and any drift between the two derivations would
show up as a model-family effect in the results. Passing the vector removes the
translation step entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

__all__ = ["BalancedWeights"]


@dataclass
class BalancedWeights:
    """Per-row sample weights, computed once and sliced positionally."""

    vector: np.ndarray
    strategy: str
    positive_weight: float
    negative_weight: float
    ratio: float
    n_rows: int

    # ------------------------------------------------------------------
    @classmethod
    def balanced(cls, y, strategy: str = "balanced") -> "BalancedWeights":
        y_arr = np.asarray(y).astype(int)
        if not np.isin(y_arr, (0, 1)).all():
            raise ValueError("y must be binary 0/1")
        if y_arr.min() == y_arr.max():
            raise ValueError("y is single-class; cannot balance")

        vector = np.asarray(compute_sample_weight(strategy, y_arr), dtype=float)
        pos = float(vector[y_arr == 1].mean())
        neg = float(vector[y_arr == 0].mean())
        return cls(
            vector=vector,
            strategy=strategy,
            positive_weight=pos,
            negative_weight=neg,
            ratio=float(pos / neg) if neg else float("nan"),
            n_rows=len(y_arr),
        )

    # ------------------------------------------------------------------
    def for_rows(self, idx: np.ndarray) -> np.ndarray:
        """Slice by positional index — the EBM's ``sample_weights[tr_idx]``."""
        return self.vector[np.asarray(idx, dtype=int)]

    def assert_matches(self, n_rows: int, context: str = "") -> None:
        if n_rows != self.n_rows:
            where = f" ({context})" if context else ""
            raise ValueError(
                f"Weights cover {self.n_rows} rows but were handed "
                f"{n_rows}{where}."
            )

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "positive_weight": self.positive_weight,
            "negative_weight": self.negative_weight,
            "ratio": self.ratio,
            "n_rows": self.n_rows,
        }

    def describe(self) -> str:
        return (
            f"balanced sample weights: class 0 → {self.negative_weight:.4f}, "
            f"class 1 → {self.positive_weight:.4f} "
            f"(positive class upweighted {self.ratio:.1f}×)"
        )