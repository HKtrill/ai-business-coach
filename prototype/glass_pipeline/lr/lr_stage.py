"""
lr.lr_stage
===========
Baseline Logistic Regression stage — no Optuna, no calibration.

Used in the cascade as a comparison reference against CalibratedLRStage.
Extends BaseLRStage; only defines what is genuinely different:
  - __init__ hyperparameters (C, penalty, solver, class_weight, max_iter)
  - fit() — plain sklearn LogisticRegression, no tuning
  - save() / load() — baseline artifact schema (no predictions cached)
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Tuple

from .base_stage import BaseLRStage

warnings.filterwarnings("ignore")


class LRStage(BaseLRStage):
    """
    Baseline LR stage for cascade comparison.

    Trains a plain LogisticRegression (no Optuna, no calibration).
    Saves model + hyperparameters only — does not cache predictions,
    as this stage is not consumed by downstream artifact loaders.
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "liblinear",
        class_weight: str = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        super().__init__(random_state=random_state)
        self.C            = C
        self.penalty      = penalty
        self.solver       = solver
        self.class_weight = class_weight
        self.max_iter     = max_iter

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> "LRStage":
        print("\n" + "=" * 80)
        print("🤖 LOGISTIC REGRESSION STAGE 1: Baseline")
        print("=" * 80)
        print(f"\nTrain: {X_train.shape} | Positive rate: {y_train.mean():.4f}")
        print(f"Features: {list(X_train.columns)}")

        self.feature_names = list(X_train.columns)
        X_scaled = self._scale_fit_transform(X_train)

        print("\n🤖 Training Logistic Regression...")
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(X_scaled, y_train)

        # Training-set metrics (informational only)
        y_train_pred  = self.model.predict(X_scaled)
        y_train_proba = self.model.predict_proba(X_scaled)[:, 1]
        train_metrics = {
            "accuracy":  accuracy_score(y_train, y_train_pred),
            "precision": precision_score(y_train, y_train_pred),
            "recall":    recall_score(y_train, y_train_pred),
            "f1":        f1_score(y_train, y_train_pred),
            "roc_auc":   roc_auc_score(y_train, y_train_proba),
        }
        print("\n📊 Training Performance:")
        for k, v in train_metrics.items():
            print(f"  {k}: {v:.4f}")

        self.fitted = True
        return self

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, output_dir: str = "./models/lr") -> Dict[str, str]:
        """
        Persist baseline model + metadata.

        Baseline does not store predictions — it is not consumed by
        downstream artifact loaders (only the calibrated artifact is).
        """
        self._check_fitted()
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        artifact = {
            "model":      self.model,
            "model_name": "logistic_regression_baseline",
            "scaler":     self.scaler,
            "feature_names": self.feature_names,
            "hyperparameters": {
                "C":            self.C,
                "penalty":      self.penalty,
                "solver":       self.solver,
                "class_weight": self.class_weight,
            },
            "performance_metrics": self.metrics,
            "training_date":       timestamp,
        }

        path = os.path.join(output_dir, f"lr_baseline_{timestamp}.joblib")
        joblib.dump(artifact, path)
        print(f"\n💾 Saved baseline LR: {path}")
        return {"path": path}

    @staticmethod
    def load(path: str) -> "LRStage":
        """Restore a saved baseline LR stage from disk."""
        artifact = joblib.load(path)
        hp = artifact["hyperparameters"]

        stage = LRStage(
            C=hp["C"],
            penalty=hp["penalty"],
            solver=hp["solver"],
            class_weight=hp["class_weight"],
        )
        stage.model         = artifact["model"]
        stage.scaler        = artifact["scaler"]
        stage.feature_names = artifact["feature_names"]
        stage.metrics       = artifact["performance_metrics"]
        stage.fitted        = True

        print(f"📂 Loaded baseline LR from: {path}")
        return stage


# ── Public entry point ────────────────────────────────────────────────────────

def train_baseline_lr_stage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    save: bool = True,
    output_dir: str = "./models/lr",
) -> Tuple[LRStage, Dict[str, float]]:
    """
    Train and evaluate the baseline LR stage.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    save            : whether to persist the model
    output_dir      : where to write the joblib file

    Returns
    -------
    stage   : fitted LRStage
    metrics : test-set metric dict
    """
    print("\n" + "=" * 80)
    print("🚀 TRAINING BASELINE LR STAGE")
    print("=" * 80)
    print(f"   X_train: {X_train.shape}")
    print(f"   X_test:  {X_test.shape}")

    stage = LRStage()
    stage.fit(X_train, y_train)
    metrics = stage.evaluate(X_test, y_test)

    print("\n📊 Feature Importance (Top 10):")
    print(stage.get_feature_importance().head(10).to_string(index=False))

    if save:
        stage.save(output_dir)

    print("\n🎯 Baseline LR Stage complete")
    return stage, metrics