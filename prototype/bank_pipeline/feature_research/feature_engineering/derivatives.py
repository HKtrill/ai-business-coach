"""
feature_research.feature_engineering.derivatives
=================================================
Derivative / slope / decay features extracted from the original Cell 10C.

Survivors after dead-feature audit (~30 -> 12):

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
    euribor_decay                 |
    nr_employed_decay             +-- intermediates for joint_economic_decay
    emp_var_decay                 |
    euribor3m_abs_curvature       |
    nr_employed_abs_curvature     +-- intermediates for economic_curvature_intensity
    emp_var_rate_abs_curvature    |

Pruned (dead):
  nr_employed_sigmoid, nr_employed_sigmoid_slope
  *_gradient (5), *_abs_gradient (5)
  nr_employed_local_rate, emp_var_rate_local_rate,
    cons_price_idx_local_rate, cons_conf_idx_local_rate
  *_curvature (signed, 3 cols)
  economic_slope_intensity, slope_x_stress

Leakage note (RESOLVED in DerivativeFeatureEngineer):
    The functional add_derivative_features() computes euribor3m_local_rate
    and abs_curvature features using target labels on whatever df is passed
    in. In the research notebook this is the full dataset — a known leakage
    point kept intentionally for exploratory use.

    In the production pipeline (glass_cascade), use DerivativeFeatureEngineer
    which enforces a fit-on-train / transform pattern:
        eng = DerivativeFeatureEngineer()
        X_train = eng.fit_transform(X_train, y_train)
        X_test  = eng.transform(X_test)

    Sigmoid and decay features are computed directly on any split (no target).
    Only local rate and abs curvature require the fitted state.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from typing import Tuple

__all__ = ["add_derivative_features", "DerivativeFeatureEngineer"]

# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------
_SIGMOID_PARAMS: dict = {
    'euribor3m':    (1.5,  0.5),
    'emp.var.rate': (-1.0, 0.5),
}

_DECAY_PARAMS: dict = {
    'euribor3m':    (0.50, 0.60),
    'nr.employed':  (0.02, 4964.0),
    'emp.var.rate': (0.80, -2.0),
}

_N_BINS_DEFAULT:    int   = 20
_SIGMOID_DERIV_MAX: float = 0.25

_CURVATURE_FEATS: tuple = ('euribor3m', 'nr.employed', 'emp.var.rate')


# ---------------------------------------------------------------------------
# Production class (fit/transform — leakage-free)
# ---------------------------------------------------------------------------
class DerivativeFeatureEngineer:
    """
    Fit-on-train / transform implementation for target-dependent derivative
    features: euribor3m_local_rate and abs_curvature intermediates.

    Sigmoid and decay features contain no target signal and are computed
    directly on any split without fitting.

    Parameters
    ----------
    n_bins : int
        Number of quantile bins for local-rate and curvature estimation.

    Usage
    -----
        eng = DerivativeFeatureEngineer()
        X_train = eng.fit_transform(X_train, y_train)
        X_test  = eng.transform(X_test)

    Note: X must already contain neighborhood_subscription_density
    (from IntegralFeatureEngineer) before transform() is called, as
    decay_x_density depends on it.
    """

    def __init__(self, n_bins: int = _N_BINS_DEFAULT) -> None:
        self.n_bins = n_bins
        self._local_rate_store: dict = {}   # feat -> (IntervalIndex, Series[rate])
        self._curvature_store:  dict = {}   # feat -> (IntervalIndex, dict[interval->curv])
        self.fitted = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> 'DerivativeFeatureEngineer':
        """
        Fit bin rates and curvature maps on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain: euribor3m, nr.employed, emp.var.rate.
        y : pd.Series
            Binary target aligned with X.
        """
        self._fit_local_rate(X, y, 'euribor3m')
        for feat in _CURVATURE_FEATS:
            self._fit_abs_curvature(X, y, feat)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a split using fitted state.

        Non-leaky features (sigmoid, decay) are computed directly.
        Local rate and abs curvature use the fitted bins.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain all economic feature columns plus
            neighborhood_subscription_density (from IntegralFeatureEngineer).

        Returns
        -------
        pd.DataFrame
            Copy of X with all derivative features appended.
        """
        if not self.fitted:
            raise ValueError(
                "DerivativeFeatureEngineer not fitted. Call fit() first."
            )
        X = X.copy()

        # Non-leaky: computed directly on any split
        _add_sigmoid_features(X)
        _add_decay_features(X)

        # Leaky (target-dependent): use fitted bins
        self._transform_local_rate(X)
        self._transform_abs_curvature(X)

        # Composites: no target needed
        _add_composites(X)
        _fill_nans(X)

        return X

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Fit on X + y, then transform X."""
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------
    # Private fit helpers
    # ------------------------------------------------------------------
    def _fit_local_rate(
        self, X: pd.DataFrame, y: pd.Series, feat: str
    ) -> None:
        df_tmp = pd.DataFrame({'feat': X[feat].values, 'y': np.asarray(y)})
        try:
            bins = pd.qcut(df_tmp['feat'], q=self.n_bins, duplicates='drop')
        except ValueError:
            bins = pd.cut(df_tmp['feat'], bins=self.n_bins, duplicates='drop')
        bin_rates = df_tmp.groupby(bins, observed=True)['y'].mean()
        self._local_rate_store[feat] = (bins.cat.categories, bin_rates)

    def _fit_abs_curvature(
        self, X: pd.DataFrame, y: pd.Series, feat: str
    ) -> None:
        df_tmp = pd.DataFrame({'feat': X[feat].values, 'y': np.asarray(y)})
        try:
            bins = pd.qcut(df_tmp['feat'], q=self.n_bins, duplicates='drop')
        except ValueError:
            bins = pd.cut(df_tmp['feat'], bins=self.n_bins, duplicates='drop')
        bin_rates = df_tmp.groupby(bins, observed=True)['y'].mean()
        rate_values = bin_rates.values
        curvature = (
            np.abs(np.gradient(np.gradient(rate_values)))
            if len(rate_values) >= 3
            else np.zeros_like(rate_values)
        )
        self._curvature_store[feat] = (
            bins.cat.categories,
            dict(zip(bin_rates.index, curvature)),
        )

    # ------------------------------------------------------------------
    # Private transform helpers
    # ------------------------------------------------------------------
    def _transform_local_rate(self, X: pd.DataFrame) -> None:
        intervals, bin_rates = self._local_rate_store['euribor3m']
        mapped = (
            pd.cut(X['euribor3m'], bins=intervals, include_lowest=True)
            .map(bin_rates)
            .astype(float)
        )
        fallback = float(bin_rates.median())
        X['euribor3m_local_rate'] = mapped.fillna(fallback)

    def _transform_abs_curvature(self, X: pd.DataFrame) -> None:
        for feat in _CURVATURE_FEATS:
            safe = feat.replace('.', '_')
            intervals, curv_map = self._curvature_store[feat]
            mapped = (
                pd.cut(X[feat], bins=intervals, include_lowest=True)
                .map(curv_map)
                .astype(float)
            )
            X[f'{safe}_abs_curvature'] = mapped.fillna(0.0)


# ---------------------------------------------------------------------------
# Research / exploratory functional API (kept for notebook compatibility)
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

    NOTE: Uses target labels at compute time over the full df passed in.
    In the research notebook this is pre-split — a known leakage point kept
    intentionally for exploratory use. In production use DerivativeFeatureEngineer.
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
# Private helpers (shared between functional API and class)
# ---------------------------------------------------------------------------
def _add_sigmoid_features(df: pd.DataFrame) -> None:
    """Part A: sigmoid value (intermediate) + normalised slope (live)."""
    for feat, (center, scale) in _SIGMOID_PARAMS.items():
        safe = feat.replace('.', '_')
        z = -(df[feat] - center) / scale
        sig_val   = expit(z)
        sig_slope = sig_val * (1.0 - sig_val) / _SIGMOID_DERIV_MAX

        df[f'{safe}_sigmoid']       = sig_val    # intermediate
        df[f'{safe}_sigmoid_slope'] = sig_slope  # LIVE


def _add_decay_features(df: pd.DataFrame) -> None:
    """Part B: per-feature exponential decays (intermediates for joint_decay)."""
    lam, x0 = _DECAY_PARAMS['euribor3m']
    df['euribor_decay'] = np.exp(-lam * np.maximum(0.0, df['euribor3m'] - x0))

    lam, x0 = _DECAY_PARAMS['nr.employed']
    df['nr_employed_decay'] = np.exp(-lam * np.maximum(0.0, df['nr.employed'] - x0))

    lam, x0 = _DECAY_PARAMS['emp.var.rate']
    df['emp_var_decay'] = np.exp(-lam * np.maximum(0.0, df['emp.var.rate'] - x0))


def _add_local_rate(
    df: pd.DataFrame, target_col: str, n_bins: int
) -> None:
    """Part C: empirical P(subscribe) per quantile bin. Only euribor3m survives audit."""
    df['euribor3m_local_rate'] = _compute_local_rate(
        df, feat='euribor3m', target_col=target_col, n_bins=n_bins
    )


def _add_abs_curvature(
    df: pd.DataFrame, target_col: str, n_bins: int
) -> None:
    """Part D: |d2P/dx2| per economic feature (intermediates for curvature_intensity)."""
    for feat in _CURVATURE_FEATS:
        safe = feat.replace('.', '_')
        df[f'{safe}_abs_curvature'] = _compute_abs_curvature(
            df, feat=feat, target_col=target_col, n_bins=n_bins
        )


def _add_composites(df: pd.DataFrame) -> None:
    """Part E: composite features from survivors above."""
    curv_cols = [
        'euribor3m_abs_curvature',
        'nr_employed_abs_curvature',
        'emp_var_rate_abs_curvature',
    ]
    df['economic_curvature_intensity'] = df[curv_cols].mean(axis=1)

    df['joint_economic_decay'] = (
        df['euribor_decay'] * df['emp_var_decay'] * df['nr_employed_decay']
    )

    # Hard guard: neighborhood_subscription_density must be present.
    # add_integral_features() must run before add_derivative_features().
    if 'neighborhood_subscription_density' not in df.columns:
        raise RuntimeError(
            "decay_x_density requires 'neighborhood_subscription_density'. "
            "Ensure add_integral_features() (or IntegralFeatureEngineer) "
            "runs before add_derivative_features() / DerivativeFeatureEngineer. "
            "Check DAG order: crisis -> integrals -> derivatives."
        )
    df['decay_x_density'] = (
        df['joint_economic_decay'] * df['neighborhood_subscription_density']
    )


def _fill_nans(df: pd.DataFrame) -> None:
    """Replace inf/NaN introduced by binning edge cases with column median."""
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
    df: pd.DataFrame, feat: str, target_col: str, n_bins: int
) -> pd.Series:
    """Empirical P(subscribe) mapped to each observation via quantile bin."""
    bins, bin_rates = _bin_target_rates(df, feat, target_col, n_bins)
    local_rate = bins.map(bin_rates).astype(float)
    return local_rate.fillna(local_rate.median())


def _compute_abs_curvature(
    df: pd.DataFrame, feat: str, target_col: str, n_bins: int
) -> pd.Series:
    """
    |d2P/dx2| per observation via discrete second-difference of per-bin rates.
    High values indicate inflection zones where prediction uncertainty peaks.
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