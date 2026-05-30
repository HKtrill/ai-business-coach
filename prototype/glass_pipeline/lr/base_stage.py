"""
lr.base_stage
=============
Abstract base class for LR pipeline stages.

Owns everything that LRStage (baseline) and CalibratedLRStage (production)
share identically: scaling, predict/predict_proba, evaluate, and
get_feature_importance.  Each subclass only defines what is genuinely
different about it — training logic, save format, and any extra attributes.

Not part of the public notebook API — import the concrete stages instead.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from typing import Dict


class BaseLRStage(ABC):
    """
    Shared scaffolding for LR stages.

    Subclasses must implement
    -------------------------
    fit(X_train, y_train)  → self
    save(output_dir)       → dict
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state  = random_state

        # Populated by subclass fit()
        self.model         = None
        self.scaler        = None
        self.feature_names = None
        self.metrics: Dict = {}
        self.fitted        = False

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "BaseLRStage":
        """Fit the stage on training data."""

    @abstractmethod
    def save(self, output_dir: str) -> dict:
        """Persist the fitted stage to disk."""

    # ── Shared predict ────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return positive-class probabilities from the primary model."""
        self._check_fitted()
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return hard predictions at the given threshold (default 0.5)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Shared evaluate ───────────────────────────────────────────────────────

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate on test data and cache metrics.

        Returns
        -------
        metrics : dict
            accuracy, precision, recall, f1, roc_auc
        """
        self._check_fitted()

        print("\n" + "=" * 80)
        print("📊 EVALUATING LR STAGE ON TEST SET")
        print("=" * 80)

        y_pred  = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        self.metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_proba),
        }

        print("\n📊 Test Performance:")
        for k, v in self.metrics.items():
            print(f"  {k.capitalize()}: {v:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
        print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")

        return self.metrics

    # ── Shared feature importance ─────────────────────────────────────────────

    def get_feature_importance(self) -> pd.DataFrame:
        """Return a DataFrame of coefficients sorted by absolute magnitude."""
        self._check_fitted()
        return (
            pd.DataFrame({
                "feature":         self.feature_names,
                "coefficient":     self.model.coef_[0],
                "abs_coefficient": np.abs(self.model.coef_[0]),
            })
            .sort_values("abs_coefficient", ascending=False)
            .reset_index(drop=True)
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self.fitted:
            raise ValueError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )

    def _scale_fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit scaler on X_train and return scaled array."""
        print("\n📏 Scaling features...")
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X_train)