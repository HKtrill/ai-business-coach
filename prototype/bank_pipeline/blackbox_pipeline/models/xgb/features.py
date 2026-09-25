"""
blackbox_pipeline.models.xgb.features
======================================
Stage 3 input contract and data hygiene.

This module does NOT engineer features. The research DAG lives in
``feature_research`` and is orchestrated by ``glass_pipeline.ebm``; the notebook
runs it once and hands the engineered frames to both arms, exactly as Stage 2
receives ``X_eng_train`` / ``X_eng_test`` through ``unpack_global_split``.

What happens here is the guard: XGBoost must receive the same columns the EBM
received, in the same order, and see the same values after cleaning. If the
GLASS feature list ever changes, ``Stage3FeatureContract`` fails loudly rather
than quietly changing what the comparison means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

__all__ = ["Stage3FeatureContract", "FeatureCleaner"]


class Stage3FeatureContract:
    """
    Asserts that the Stage 3 input matches the EBM's feature block.

    Parameters
    ----------
    expected_features
        The column list the EBM consumed — pass
        ``glass_pipeline.ebm.feature_engineering.EBM_FEATURES``. When ``None``
        the contract only checks train/test agreement, which is weaker; the
        stage warns in that case.
    """

    def __init__(self, expected_features: Optional[list[str]] = None):
        self.expected_features = (
            None if expected_features is None else list(expected_features)
        )

    # ------------------------------------------------------------------
    def validate(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[str]:
        """
        Return the agreed column order, or raise explaining the mismatch.

        Order is ``expected_features`` when set, else ``X_train``'s order.
        Raises on non-DataFrame input, duplicate columns, train/test set
        mismatch, or any missing / extra column vs. the contract.
        """
        for name, frame in (("X_train", X_train), ("X_test", X_test)):
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(
                    f"{name} must be a DataFrame, got {type(frame).__name__}"
                )

        train_cols, test_cols = list(X_train.columns), list(X_test.columns)

        for name, cols in (("X_train", train_cols), ("X_test", test_cols)):
            dupes = sorted({c for c in cols if cols.count(c) > 1})
            if dupes:
                raise ValueError(f"{name} has duplicate columns: {dupes}")

        if set(train_cols) != set(test_cols):
            only_train = sorted(set(train_cols) - set(test_cols))
            only_test = sorted(set(test_cols) - set(train_cols))
            raise ValueError(
                "Stage 3 train/test columns differ.\n"
                f"  only in train: {only_train}\n"
                f"  only in test : {only_test}"
            )

        if self.expected_features is None:
            return train_cols

        expected = self.expected_features
        missing = [c for c in expected if c not in train_cols]
        extra = [c for c in train_cols if c not in expected]
        if missing or extra:
            raise ValueError(
                "Stage 3 input does not match the EBM feature contract — the "
                "two arms would not be seeing the same information.\n"
                f"  expected {len(expected)} columns from EBM_FEATURES\n"
                f"  missing  : {missing}\n"
                f"  unexpected: {extra}\n"
                "Re-run the GLASS engineering DAG (drop_leaky_features -> "
                "engineer_ebm_features -> select_ebm_features) and pass its "
                "output, or update config.expected_features deliberately."
            )
        return list(expected)

    # ------------------------------------------------------------------
    def align(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """Validate, then return both frames in the contracted column order."""
        order = self.validate(X_train, X_test)
        return X_train[order], X_test[order], order


@dataclass
class FeatureCleaner:
    """
    inf -> NaN -> median impute, with medians learned on the training split.

    Mirrors the guard inside ``tune_ebm``. XGBoost treats NaN as a first-class
    branch direction and would not need this, but imputing keeps both arms on
    identical inputs — the point of the experiment is the model family, not the
    missing-value policy.

    Two repairs over the EBM version, both noted in the artifact:
      * the EBM cleans only ``X_train``, and only inside ``tune_ebm``, so its
        reported ``train_predictions`` are scored on an UNCLEANED frame;
      * the EBM never cleans ``X_test`` at all.
    Here the medians are fitted once on train and applied to both splits.

    Assumes numeric columns. Raises if inf remains (e.g. ``clean_inf=False``)
    or if NaN remains after imputation (e.g. a column all-NaN in train).
    ``report_`` lists only columns with NaN in TRAIN.
    """

    clean_inf: bool = True
    impute_missing: bool = True
    medians_: Optional[pd.Series] = field(default=None, repr=False)
    columns_: Optional[list[str]] = field(default=None, repr=False)
    report_: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame) -> "FeatureCleaner":
        frame = self._replace_inf(X)
        self.columns_ = list(frame.columns)
        self.medians_ = frame.median(numeric_only=False)
        n_inf = self._count_inf(X)
        nan_cols = frame.columns[frame.isna().any()].tolist()
        self.report_ = {
            "n_inf_replaced": n_inf,
            "columns_imputed": nan_cols,
            "medians": {c: float(self.medians_[c]) for c in nan_cols},
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise ValueError("Call fit() first")
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Frame is missing fitted columns: {missing}")

        frame = self._replace_inf(X[self.columns_])
        if self.impute_missing and frame.isna().any().any():
            frame = frame.fillna(self.medians_)

        if self._count_inf(frame):
            raise AssertionError("Infinity values remain after cleaning")
        if self.impute_missing and frame.isna().any().any():
            raise AssertionError("NaN values remain after cleaning")
        return frame

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    def _replace_inf(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.clean_inf:
            return X.copy()
        return X.replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def _count_inf(X: pd.DataFrame) -> int:
        """Count ±inf across numeric columns."""
        numeric = X.select_dtypes(include=["number"])
        if numeric.empty:
            return 0
        return int(np.isinf(numeric.to_numpy(dtype=float)).sum())