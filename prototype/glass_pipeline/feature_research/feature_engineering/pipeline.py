"""
feature_research.feature_engineering.pipeline
==============================================
Orchestrates all feature engineering modules in dependency order.

Full dependency DAG (Cells 10A–10F, complete):

    add_crisis_features()               [crisis.py]
        produces: economic_crisis_score, cellular_crisis (LR live),
                  high_conversion_month, cellular_contact, default_clean

    add_integral_features()             [integrals.py]
        produces: economic_stress_integral,
                  neighborhood_subscription_density (RF live)

    add_derivative_features()           [derivatives.py]
        requires: neighborhood_subscription_density
        produces: euribor3m_sigmoid_slope (EBM live),
                  emp_var_rate_sigmoid_slope (EBM live),
                  euribor3m_local_rate (LR live),
                  economic_curvature_intensity (RF + EBM live),
                  joint_economic_decay (RF live),
                  decay_x_density (EBM live)

    add_temporal_features()             [temporal.py]
        produces: dow_month_encoded (LR + RF + EBM live)

    add_prior_features()                [prior.py]
        requires: economic_stress_integral
        produces: has_prior_contact,
                  prior_x_stress (EBM live)

    add_overlap_features()              [overlap.py]
        requires: economic_stress_integral, cellular_contact,
                  high_conversion_month, default_clean, has_prior_contact
        produces: low_stress_zone,
                  cpi_high_cellular (RF + EBM live),
                  behavioral_favorability (RF + EBM live),
                  overlap_default_clean (EBM live),
                  overlap_behavioral_score (EBM live)
"""

import pandas as pd

from feature_research.config import RANDOM_SEED
from feature_research.feature_engineering.crisis import add_crisis_features
from feature_research.feature_engineering.integrals import add_integral_features
from feature_research.feature_engineering.derivatives import add_derivative_features
from feature_research.feature_engineering.temporal import add_temporal_features
from feature_research.feature_engineering.prior import add_prior_features
from feature_research.feature_engineering.overlap import add_overlap_features

__all__ = ["build_features", "finalize_features", "get_stage_df"]


def build_features(
    df: pd.DataFrame,
    target_col: str = 'y',
    random_state: int = RANDOM_SEED,
    n_bins: int = 20,
    smoothing_factor: int = 100,
) -> pd.DataFrame:
    """
    Apply all feature engineering modules in dependency order.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_and_preprocess() — preprocessed, no leaky features.
    target_col : str
        Binary target column name.
    random_state : int
        Seed for KDE reference-point sampling (integrals).
    n_bins : int
        Quantile bins for local-rate and curvature estimation (derivatives).
    smoothing_factor : int
        Laplace smoothing weight for day × month encoding (temporal).

    Returns
    -------
    pd.DataFrame
        df with all surviving engineered features appended.
        Live features ready for model ingestion:
          LR:  cellular_crisis, euribor3m_local_rate, dow_month_encoded
          RF:  campaign*, neighborhood_subscription_density, cons.conf.idx*,
               economic_curvature_intensity, joint_economic_decay,
               dow_month_encoded, behavioral_favorability, cpi_high_cellular
          EBM: decay_x_density, euribor3m_sigmoid_slope, economic_curvature_intensity,
               cons.conf.idx*, age*, dow_month_encoded, default*,
               prior_x_stress, campaign*, overlap_default_clean,
               overlap_behavioral_score, cpi_high_cellular,
               behavioral_favorability, emp_var_rate_sigmoid_slope
          (* raw UCI features, no engineering needed)
    """
    df = add_crisis_features(df)
    df = add_integral_features(df, target_col=target_col, random_state=random_state)
    df = add_derivative_features(df, target_col=target_col, n_bins=n_bins)
    df = add_temporal_features(df, target_col=target_col, smoothing_factor=smoothing_factor)
    df = add_prior_features(df, target_col=target_col)
    df = add_overlap_features(df)
    return df



# ---------------------------------------------------------------------------
# Helpers for model-ready feature selection
# ---------------------------------------------------------------------------

def finalize_features(
    df: pd.DataFrame,
    live_features: dict,
    target_col: str = 'y',
) -> pd.DataFrame:
    """
    Positive-select to only model-consumed columns: union of all stage
    features + target.  Everything else (original UCI columns not used
    by any model, all intermediate scaffolding) is dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_features().
    live_features : dict[str, list[str]]
        LIVE_FEATURES registry, e.g. from feature_engineering.__init__.
    target_col : str
        Target column to retain alongside features.

    Returns
    -------
    pd.DataFrame
        Shape: (n_rows, n_unique_model_features + 1)
        Columns ordered as they appear in df.
    """
    all_model_cols: set = set()
    for features in live_features.values():
        all_model_cols.update(features)
    if target_col in df.columns:
        all_model_cols.add(target_col)
    # Preserve df column order
    keep = [c for c in df.columns if c in all_model_cols]
    return df[keep]


def get_stage_df(
    df: pd.DataFrame,
    stage: str,
    live_features: dict,
    target_col: str = 'y',
) -> pd.DataFrame:
    """
    Return a model-ready dataframe for a single cascade stage.

    Parameters
    ----------
    df : pd.DataFrame
        Output of finalize_features() or build_features().
    stage : str
        One of 'lr', 'rf', 'ebm'.
    live_features : dict[str, list[str]]
        LIVE_FEATURES registry.
    target_col : str
        Target column to include.

    Returns
    -------
    pd.DataFrame
        Contains only that stage's feature columns + target.
        LR → 3+1, RF → 8+1, EBM → 14+1 columns.
    """
    if stage not in live_features:
        raise ValueError(f"stage must be one of {list(live_features)}; got {stage!r}")
    cols = live_features[stage] + (
        [target_col] if target_col in df.columns else []
    )
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"get_stage_df: columns missing from df: {missing}")
    return df[cols]