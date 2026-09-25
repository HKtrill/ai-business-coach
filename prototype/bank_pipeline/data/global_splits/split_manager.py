"""
Global Train/Test Split Manager
================================
Creates and manages the canonical train/test split for the Bank Marketing
dataset, ensuring consistency across all pipeline stages.

Usage
-----
    manager = GlobalSplitManager(test_size=0.20, random_state=42)
    GLOBAL_SPLIT = manager.create_split(df_processed, target_col='y')

Author: Glass Pipeline Team
"""

from __future__ import annotations

import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


class GlobalSplitManager:
    """
    Manages the global train/test split for the pipeline.

    Ensures all stages use the same split to prevent data leakage
    and enable fair comparison across models.

    Parameters
    ----------
    test_size : float, default 0.20
        Proportion of dataset reserved for test.
    random_state : int, default 42
        Seed for reproducibility.
    n_cv_folds : int, default 10
        Number of stratified CV folds for hyperparameter tuning.
    """

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
        n_cv_folds: int = 10,
    ) -> None:
        self.test_size    = test_size
        self.random_state = random_state
        self.n_cv_folds   = n_cv_folds
        self._split_dict: Dict | None = None

    # ------------------------------------------------------------------
    # Core split
    # ------------------------------------------------------------------

    def create_split(self, df: pd.DataFrame, target_col: str = "y") -> Dict:
        """
        Create a stratified train/test split and attach a CV object.

        Parameters
        ----------
        df : preprocessed DataFrame (must contain target_col)
        target_col : name of the binary target column

        Returns
        -------
        split_dict : dict with keys
            X_train, X_test, y_train, y_test,
            train_idx, test_idx, cv,
            test_size, random_state, n_cv_folds,
            feature_names, n_features
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        print("\n" + "=" * 70)
        print("📐  CREATING GLOBAL TRAIN / TEST SPLIT")
        print("=" * 70)
        print(f"    total samples : {len(df)}")
        print(f"    positive rate : {y.mean():.4f}")
        print(f"    features      : {X.shape[1]}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )

        train_idx = X_train.index.values
        test_idx  = X_test.index.values
        assert len(set(train_idx) & set(test_idx)) == 0, "Train/test index overlap."

        cv = StratifiedKFold(
            n_splits=self.n_cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        print(f"\n✅  train : {X_train.shape}  |  positive rate: {y_train.mean():.4f}")
        print(f"    test  : {X_test.shape}  |  positive rate: {y_test.mean():.4f}")
        print("🔒  no train/test leakage")

        self._split_dict = {
            "X_train":      X_train,
            "X_test":       X_test,
            "y_train":      y_train,
            "y_test":       y_test,
            "train_idx":    train_idx,
            "test_idx":     test_idx,
            "cv":           cv,
            "test_size":    self.test_size,
            "random_state": self.random_state,
            "n_cv_folds":   self.n_cv_folds,
            "feature_names": list(X_train.columns),
            "n_features":   X_train.shape[1],
        }

        print("\n📦  GLOBAL_SPLIT ready")
        return self._split_dict

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_split(self, filepath: str) -> None:
        """Pickle the split dict to filepath (directories created if needed)."""
        self._require_split()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self._split_dict, f)
        print(f"💾  split saved → {filepath}")

    @staticmethod
    def load_split(filepath: str) -> Dict:
        """Load and return a previously saved split dict."""
        with open(filepath, "rb") as f:
            split_dict = pickle.load(f)
        print(f"📂  split loaded ← {filepath}")
        print(f"    train: {split_dict['X_train'].shape}")
        print(f"    test : {split_dict['X_test'].shape}")
        return split_dict

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_split_summary(self) -> pd.DataFrame:
        """Return a tidy DataFrame summarising the split."""
        self._require_split()
        s = self._split_dict
        n_train, n_test = len(s["y_train"]), len(s["y_test"])
        p_train, p_test = s["y_train"].sum(), s["y_test"].sum()
        return pd.DataFrame({
            "split":         ["train", "test", "total"],
            "samples":       [n_train, n_test, n_train + n_test],
            "positives":     [p_train, p_test, p_train + p_test],
            "positive_rate": [
                s["y_train"].mean(),
                s["y_test"].mean(),
                (p_train + p_test) / (n_train + n_test),
            ],
        })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_split(self) -> None:
        if self._split_dict is None:
            raise RuntimeError("No split found. Call create_split() first.")