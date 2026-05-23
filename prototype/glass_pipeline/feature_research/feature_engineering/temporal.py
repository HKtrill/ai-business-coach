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

Leakage note (RESOLVED):
    The functional add_temporal_features() computes target encoding on whatever
    df is passed in. In the research notebook this is the full dataset — a known
    leakage point documented here for transparency.

    In the production pipeline (glass_cascade), use TemporalFeatureEngineer
    which enforces a fit-on-train / transform pattern:
        eng = TemporalFeatureEngineer()
        X_train = eng.fit_transform(X_train, y_train)
        X_test  = eng.transform(X_test)

    Unseen day-month combinations in the test set fall back to the training
    global mean rather than the full-dataset mean.
"""

import numpy as np
import pandas as pd

__all__ = ["add_temporal_features", "TemporalFeatureEngineer"]

_SMOOTHING_FACTOR: int = 100   # Laplace-style smoothing; tuned in Cell 10D


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
        if 'day_of_week' not in X.columns or 'month' not in X.columns:
            raise ValueError(
                "TemporalFeatureEngineer.fit() requires "
                "'day_of_week' and 'month' columns."
            )

        self._global_mean = float(np.mean(y))

        dm_key = X['day_of_week'].astype(str) + '_' + X['month'].astype(str)
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
        if 'day_of_week' not in X.columns or 'month' not in X.columns:
            raise ValueError(
                "TemporalFeatureEngineer.transform() requires "
                "'day_of_week' and 'month' columns."
            )

        X = X.copy()
        dm_key = X['day_of_week'].astype(str) + '_' + X['month'].astype(str)
        X['dow_month_encoded'] = (
            dm_key.map(self._dm_smoothed).fillna(self._global_mean)
        )
        return X

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fit on X + y, then transform X."""
        return self.fit(X, y).transform(X)


# ---------------------------------------------------------------------------
# Research / exploratory functional API (kept for notebook compatibility)
# ---------------------------------------------------------------------------
def add_temporal_features(
    df: pd.DataFrame,
    target_col: str = 'y',
    smoothing_factor: int = _SMOOTHING_FACTOR,
) -> pd.DataFrame:
    """
    Add day-of-week x month smoothed target encoding.

    Requires columns: day_of_week, month, target_col.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe after add_derivative_features().
    target_col : str
        Binary target column name.
    smoothing_factor : int
        Laplace smoothing weight toward the global mean. Higher = more
        regularisation for sparse day-month combinations.

    Returns
    -------
    pd.DataFrame
        Copy of df with temporal feature appended.

    New columns
    -----------
    dow_month_encoded    float
        Smoothed P(subscribe | day_of_week, month). Captures campaign
        timing sweet spots (day + month combos).
        LIVE — LR Stage 1, RF Stage 2, EBM Stage 3.

        Formula per (day, month) cell:
            encoded = (n_cell * P_cell + lambda * P_global) / (n_cell + lambda)
        where lambda = smoothing_factor.

        Unseen combinations fall back to P_global.

    NOTE: This function uses target labels at compute time over whatever df
    is passed in. In the research notebook this is the full dataset before
    the train/test split — a known leakage point kept intentionally for
    exploratory use. In the production pipeline use TemporalFeatureEngineer
    which resolves this by fitting on training data only.
    """
    if 'day_of_week' not in df.columns or 'month' not in df.columns:
        raise ValueError(
            "add_temporal_features requires 'day_of_week' and 'month' columns."
        )

    df = df.copy()
    global_mean: float = df[target_col].mean()

    dm_key   = df['day_of_week'].astype(str) + '_' + df['month'].astype(str)
    dm_stats = df.groupby(dm_key)[target_col].agg(['mean', 'count'])

    dm_smoothed = (
        dm_stats['count'] * dm_stats['mean']
        + smoothing_factor * global_mean
    ) / (dm_stats['count'] + smoothing_factor)

    df['dow_month_encoded'] = dm_key.map(dm_smoothed).fillna(global_mean)

    return df