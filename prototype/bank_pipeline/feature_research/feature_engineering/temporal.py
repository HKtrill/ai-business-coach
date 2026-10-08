"""
feature_research.feature_engineering.temporal
==============================================
Day-of-week interaction features extracted from the original Cell 10D.

Survivors after dead-feature audit (5 -> 1):
  - dow_month_encoded    LIVE (LR Stage 1, RF Stage 2, EBM Stage 3)

Pruned:
  - dow_target_encoded   dead
  - dow_x_stress         dead
  - good_contact_day     dead
  - dow_x_contact        dead

Background: day_of_week ranked #15 (Composite=0.0163, AUC=0.509) — negligible
as a main effect. Its value is entirely in the month interaction (10.6% MI lift),
which dow_month_encoded captures through smoothed target encoding of the
day x month cross.

Target handling:
    dow_month_encoded is target-derived; TemporalFeatureEngineer fits the
    smoothed cell rates and global mean on training data only. Unseen
    day x month cells fall back to the training global mean.
    Use it via FeaturePipeline.
"""

import numpy as np
import pandas as pd

from feature_research.feature_engineering._checks import check_xy_aligned

__all__ = ["TemporalFeatureEngineer"]

_SMOOTHING_FACTOR: int = 100   # Laplace-style smoothing; tuned in Cell 10D


def _dm_key(X: pd.DataFrame) -> pd.Series:
    """Build the day x month key, requiring integer-encoded inputs.

    The key is built from string forms, so a dtype change between splits
    (e.g. 5 vs 5.0) would silently turn every lookup into a miss and send
    all rows to the global-mean fallback. Fail loudly instead.
    """
    for col in ('day_of_week', 'month'):
        if col not in X.columns:
            raise ValueError(f"dow_month encoding requires '{col}' column.")
        if not pd.api.types.is_integer_dtype(X[col]):
            raise TypeError(
                f"dow_month encoding requires integer-encoded '{col}'; "
                f"got dtype {X[col].dtype}."
            )
    return X['day_of_week'].astype(str) + '_' + X['month'].astype(str)


# ---------------------------------------------------------------------------
# Production class (fit/transform — leakage-free)
# ---------------------------------------------------------------------------
class TemporalFeatureEngineer:
    """
    Fit-on-train / transform implementation of dow_month_encoded.

    Enforces the correct train-only fit so that test labels never
    influence the encoding. Use this in the glass_cascade pipeline.

    Parameters
    ----------
    smoothing_factor : int
        Laplace smoothing weight toward the global mean. Higher = more
        regularisation for sparse day-month combinations. Default 100.

    Usage
    -----
        eng = TemporalFeatureEngineer()
        X_train = eng.fit_transform(X_train, y_train)
        X_test  = eng.transform(X_test)
    """

    def __init__(self, smoothing_factor: int = _SMOOTHING_FACTOR) -> None:
        self.smoothing_factor = smoothing_factor
        self._global_mean = None    # float: train subscribe rate
        self._dm_smoothed = None    # Series: "dow_month" key -> smoothed rate
        self.fitted       = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> 'TemporalFeatureEngineer':
        """
        Fit smoothed encoding on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain day_of_week and month columns.
        y : pd.Series
            Binary target aligned with X.
        """
        check_xy_aligned(X, y, "TemporalFeatureEngineer.fit")
        self._global_mean = float(np.mean(y))

        dm_key = _dm_key(X)
        df_tmp = pd.DataFrame({'key': dm_key.values, 'y': np.asarray(y)})
        stats  = df_tmp.groupby('key')['y'].agg(['mean', 'count'])

        self._dm_smoothed = (
            stats['count'] * stats['mean']
            + self.smoothing_factor * self._global_mean
        ) / (stats['count'] + self.smoothing_factor)

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map day x month combinations to smoothed rates.

        Unseen combinations (present in test but not train) fall back
        to the training global mean.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain day_of_week and month columns.

        Returns
        -------
        pd.DataFrame
            Copy of X with dow_month_encoded appended.
        """
        if not self.fitted:
            raise ValueError(
                "TemporalFeatureEngineer not fitted. Call fit() first."
            )
        X = X.copy()
        dm_key = _dm_key(X)
        X['dow_month_encoded'] = (
            dm_key.map(self._dm_smoothed).fillna(self._global_mean)
        )
        return X

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fit on X + y, then transform X."""
        return self.fit(X, y).transform(X)
