"""
feature_research.feature_engineering.temporal
==============================================
Day-of-week interaction features extracted from the original Cell 10D.

Survivors after dead-feature audit (5 → 1):
  - dow_month_encoded    LIVE (LR Stage 1, RF Stage 2, EBM Stage 3)

Pruned:
  - dow_target_encoded   dead — feeds only dead downstream features,
                         does NOT feed dow_month_encoded (independent computation)
  - dow_x_stress         dead
  - good_contact_day     dead
  - dow_x_contact        dead

Background: day_of_week ranked #15 (Composite=0.0163, AUC=0.509) — negligible
as a main effect.  Its value is entirely in the month interaction (10.6% MI lift),
which dow_month_encoded captures through smoothed target encoding of the
day × month cross.

Design note: dow_month_encoded uses target labels during computation (smoothed
target encoding over the training set).  Fit on training data only in production;
map unseen day-month combos to the global mean.
"""

import pandas as pd

__all__ = ["add_temporal_features"]

_SMOOTHING_FACTOR: int = 100   # Laplace-style smoothing; tuned in Cell 10D


def add_temporal_features(
    df: pd.DataFrame,
    target_col: str = 'y',
    smoothing_factor: int = _SMOOTHING_FACTOR,
) -> pd.DataFrame:
    """
    Add day-of-week × month smoothed target encoding.

    Requires columns: day_of_week, month, target_col.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe after add_derivative_features().
    target_col : str
        Binary target column name.
    smoothing_factor : int
        Laplace smoothing weight toward the global mean.  Higher = more
        regularisation for sparse day-month combinations.

    Returns
    -------
    pd.DataFrame
        Copy of df with temporal feature appended.

    New columns
    -----------
    dow_month_encoded    float
        Smoothed P(subscribe | day_of_week, month).  Captures campaign
        timing sweet spots (day + month combos).
        LIVE — LR Stage 1, RF Stage 2, EBM Stage 3.

        Formula per (day, month) cell:
            encoded = (n_cell · P_cell + λ · P_global) / (n_cell + λ)
        where λ = smoothing_factor.

        Unseen combinations fall back to P_global.

        NOTE: Uses target at compute time — train split only in production.
    """
    if 'day_of_week' not in df.columns or 'month' not in df.columns:
        raise ValueError(
            "add_temporal_features requires 'day_of_week' and 'month' columns."
        )

    df = df.copy()
    global_mean: float = df[target_col].mean()

    # Composite key: "2_3" = Tuesday in March, etc.
    dm_key = df['day_of_week'].astype(str) + '_' + df['month'].astype(str)

    dm_stats = df.groupby(dm_key)[target_col].agg(['mean', 'count'])

    dm_smoothed = (
        dm_stats['count'] * dm_stats['mean']
        + smoothing_factor * global_mean
    ) / (dm_stats['count'] + smoothing_factor)

    df['dow_month_encoded'] = dm_key.map(dm_smoothed).fillna(global_mean)

    return df