"""
feature_research.feature_engineering.derivatives
=================================================
Derivative / slope / decay features extracted from the original Cell 10C.

Survivors after dead-feature audit (~30 → 12):

  LIVE features:
    euribor3m_sigmoid_slope       LIVE (EBM Stage 3)
    emp_var_rate_sigmoid_slope    LIVE (EBM Stage 3)
    euribor3m_local_rate          LIVE (LR Stage 1)
    economic_curvature_intensity  LIVE (RF Stage 2 + EBM Stage 3)
    joint_economic_decay          LIVE (RF Stage 2) + intermediate for decay_x_density
    decay_x_density               LIVE (EBM Stage 3)

  Intermediates (kept because they feed live features):
    euribor3m_sigmoid             intermediate for euribor3m_sigmoid_slope
    emp_var_rate_sigmoid          intermediate for emp_var_rate_sigmoid_slope
    euribor_decay                 ┐
    nr_employed_decay             ├─ intermediates for joint_economic_decay
    emp_var_decay                 ┘
    euribor3m_abs_curvature       ┐
    nr_employed_abs_curvature     ├─ intermediates for economic_curvature_intensity
    emp_var_rate_abs_curvature    ┘

Pruned (dead):
  nr_employed_sigmoid, nr_employed_sigmoid_slope
  *_gradient (5), *_abs_gradient (5)          — all piecewise gradient cols
  nr_employed_local_rate, emp_var_rate_local_rate,
    cons_price_idx_local_rate, cons_conf_idx_local_rate
  *_curvature (signed, 3 cols)
  economic_slope_intensity, slope_x_stress
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from typing import Tuple

__all__ = ["add_derivative_features"]

# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------
# Sigmoid parameters (center, scale) tuned from distribution analysis.
# Only euribor3m and emp.var.rate survive the audit; nr.employed dropped.
_SIGMOID_PARAMS: dict = {
    'euribor3m':    (1.5,  0.5),   # cliff at 1.5 %
    'emp.var.rate': (-1.0, 0.5),   # cliff at -1.0
}

# Exponential decay parameters (lambda, x_min) for each economic feature
_DECAY_PARAMS: dict = {
    'euribor3m':    (0.50, 0.60),   # exp(-0.5 · max(0, x - 0.6))
    'nr.employed':  (0.02, 4964.0), # exp(-0.02 · max(0, x - 4964))
    'emp.var.rate': (0.80, -2.0),   # exp(-0.8 · max(0, x - (-2.0)))
}

_N_BINS_DEFAULT: int = 20
_SIGMOID_DERIV_MAX: float = 0.25   # normalisation: σ'(0) = 0.25


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def add_derivative_features(
    df: pd.DataFrame,
    target_col: str = 'y',
    n_bins: int = _N_BINS_DEFAULT,
) -> pd.DataFrame:
    """
    Add sigmoid-slope, exponential-decay, local-rate, curvature, and
    composite derivative features.

    Call *after* add_integral_features() so that neighborhood_subscription_density
    is present (required for decay_x_density).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe after add_integral_features().
    target_col : str
        Binary target column name.
    n_bins : int
        Number of quantile bins used for local-rate and curvature estimation.

    Returns
    -------
    pd.DataFrame
        Copy of df with derivative features appended.

    New columns (live only listed — intermediates also added, see module docstring)
    -------------------------------------------------------------------------------
    euribor3m_sigmoid_slope      float  [0, 1]   Normalised σ'(z); peaks at cliff.
    emp_var_rate_sigmoid_slope   float  [0, 1]   Normalised σ'(z); peaks at cliff.
    euribor3m_local_rate         float  [0, 1]   Empirical P(subscribe) per quantile bin.
    economic_curvature_intensity float  ≥0        mean(|∂²P/∂x²|) over three economic
                                                   features — regime-transition detector.
    joint_economic_decay         float  [0, 1]   Π of three exponential decays —
                                                   high when ALL indicators signal crisis.
    decay_x_density              float  ≥0        joint_economic_decay ×
                                                   neighborhood_subscription_density.
    """
    df = df.copy()

    _add_sigmoid_features(df)
    _add_decay_features(df)
    _add_local_rate(df, target_col=target_col, n_bins=n_bins)
    _add_abs_curvature(df, target_col=target_col, n_bins=n_bins)
    _add_composites(df)
    _fill_nans(df)

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _add_sigmoid_features(df: pd.DataFrame) -> None:
    """Part A: sigmoid value (intermediate) + normalised slope (live)."""
    for feat, (center, scale) in _SIGMOID_PARAMS.items():
        safe = feat.replace('.', '_')
        z = -(df[feat] - center) / scale
        sig_val = expit(z)                                  # σ(z) ∈ (0, 1)
        sig_slope = sig_val * (1.0 - sig_val) / _SIGMOID_DERIV_MAX  # normalised σ'

        df[f'{safe}_sigmoid'] = sig_val          # intermediate
        df[f'{safe}_sigmoid_slope'] = sig_slope  # LIVE


def _add_decay_features(df: pd.DataFrame) -> None:
    """Part B: per-feature exponential decays (all intermediates for joint_decay)."""
    lam, x0 = _DECAY_PARAMS['euribor3m']
    df['euribor_decay'] = np.exp(-lam * np.maximum(0.0, df['euribor3m'] - x0))

    lam, x0 = _DECAY_PARAMS['nr.employed']
    df['nr_employed_decay'] = np.exp(-lam * np.maximum(0.0, df['nr.employed'] - x0))

    lam, x0 = _DECAY_PARAMS['emp.var.rate']
    df['emp_var_decay'] = np.exp(-lam * np.maximum(0.0, df['emp.var.rate'] - x0))


def _add_local_rate(
    df: pd.DataFrame,
    target_col: str,
    n_bins: int,
) -> None:
    """
    Part C (survivors only): empirical P(subscribe) per quantile bin.
    Only euribor3m_local_rate survives the dead-feature audit.
    """
    df['euribor3m_local_rate'] = _compute_local_rate(
        df, feat='euribor3m', target_col=target_col, n_bins=n_bins
    )


def _add_abs_curvature(
    df: pd.DataFrame,
    target_col: str,
    n_bins: int,
) -> None:
    """
    Part D: |∂²P/∂x²| per economic feature (intermediates for
    economic_curvature_intensity).
    """
    for feat in ('euribor3m', 'nr.employed', 'emp.var.rate'):
        safe = feat.replace('.', '_')
        df[f'{safe}_abs_curvature'] = _compute_abs_curvature(
            df, feat=feat, target_col=target_col, n_bins=n_bins
        )


def _add_composites(df: pd.DataFrame) -> None:
    """Part E: composite features assembled from the survivors above."""
    # mean(|∂²P/∂x²|) across three economic dimensions
    curv_cols = [
        'euribor3m_abs_curvature',
        'nr_employed_abs_curvature',
        'emp_var_rate_abs_curvature',
    ]
    df['economic_curvature_intensity'] = df[curv_cols].mean(axis=1)

    # Product of three exponential decays — high when ALL indicators signal crisis
    df['joint_economic_decay'] = (
        df['euribor_decay'] * df['emp_var_decay'] * df['nr_employed_decay']
    )

    # Interaction with spatial density (requires add_integral_features first)
    if 'neighborhood_subscription_density' in df.columns:
        df['decay_x_density'] = (
            df['joint_economic_decay'] * df['neighborhood_subscription_density']
        )


def _fill_nans(df: pd.DataFrame) -> None:
    """Replace inf/NaN introduced by binning edge cases with column median."""
    # Avoid inplace= to stay CoW-safe under pandas 2.x
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        if np.isinf(df[col]).any():
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())


# ---------------------------------------------------------------------------
# Shared bin-statistics helpers
# ---------------------------------------------------------------------------
def _bin_target_rates(
    df: pd.DataFrame,
    feat: str,
    target_col: str,
    n_bins: int,
) -> Tuple[pd.Series, pd.Series]:
    """Return (bins, per-bin target rate) for a feature column."""
    try:
        bins = pd.qcut(df[feat], q=n_bins, duplicates='drop')
    except ValueError:
        bins = pd.cut(df[feat], bins=n_bins, duplicates='drop')
    bin_rates = df.groupby(bins, observed=True)[target_col].mean()
    return bins, bin_rates


def _compute_local_rate(
    df: pd.DataFrame,
    feat: str,
    target_col: str,
    n_bins: int,
) -> pd.Series:
    """Empirical P(subscribe) mapped to each observation via quantile bin."""
    bins, bin_rates = _bin_target_rates(df, feat, target_col, n_bins)
    local_rate = bins.map(bin_rates).astype(float)
    return local_rate.fillna(local_rate.median())


def _compute_abs_curvature(
    df: pd.DataFrame,
    feat: str,
    target_col: str,
    n_bins: int,
) -> pd.Series:
    """
    |∂²P/∂x²| per observation via discrete second-difference of per-bin rates.
    High values indicate inflection zones — where prediction uncertainty peaks.
    """
    bins, bin_rates = _bin_target_rates(df, feat, target_col, n_bins)
    rate_values = bin_rates.values
    if len(rate_values) >= 3:
        curvature = np.abs(np.gradient(np.gradient(rate_values)))
    else:
        curvature = np.zeros_like(rate_values)

    bin_to_curv = dict(zip(bin_rates.index, curvature))
    result = bins.map(bin_to_curv).astype(float)
    return result.fillna(0.0)