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

Target handling:
    Sigmoid, decay and composite features are deterministic.
    euribor3m_local_rate and the *_abs_curvature intermediates are
    target-derived; DerivativeFeatureEngineer fits their bin edges and bin
    rates on training data only. Use it via FeaturePipeline.
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from typing import Tuple

from feature_research.feature_engineering._checks import assert_finite, check_xy_aligned

__all__ = ["DerivativeFeatureEngineer", "DERIVATIVE_OUTPUT_COLS"]

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

#: Every column this module adds, in creation order.
DERIVATIVE_OUTPUT_COLS: tuple = (
    'euribor3m_sigmoid', 'euribor3m_sigmoid_slope',
    'emp_var_rate_sigmoid', 'emp_var_rate_sigmoid_slope',
    'euribor_decay', 'nr_employed_decay', 'emp_var_decay',
    'euribor3m_local_rate',
    'euribor3m_abs_curvature', 'nr_employed_abs_curvature', 'emp_var_rate_abs_curvature',
    'economic_curvature_intensity', 'joint_economic_decay', 'decay_x_density',
)


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
        self._local_rate_store: dict = {}   # feat -> (edges ndarray, per-bin rate ndarray)
        self._curvature_store:  dict = {}   # feat -> (edges ndarray, per-bin |curvature| ndarray)
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
            Binary target aligned with X (same index if a Series).
        """
        check_xy_aligned(X, y, "DerivativeFeatureEngineer.fit")
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
        assert_finite(X, DERIVATIVE_OUTPUT_COLS, "DerivativeFeatureEngineer.transform")

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
        self._local_rate_store[feat] = _fit_bin_rates(X[feat], y, self.n_bins)

    def _fit_abs_curvature(
        self, X: pd.DataFrame, y: pd.Series, feat: str
    ) -> None:
        edges, rates = _fit_bin_rates(X[feat], y, self.n_bins)
        self._curvature_store[feat] = (edges, _abs_curvature_from_rates(rates))

    # ------------------------------------------------------------------
    # Private transform helpers
    # ------------------------------------------------------------------
    def _transform_local_rate(self, X: pd.DataFrame) -> None:
        edges, rates = self._local_rate_store['euribor3m']
        fallback = float(np.nanmedian(rates))   # train-derived; out-of-range rows
        X['euribor3m_local_rate'] = (
            _lookup_bins(X['euribor3m'], edges, rates).fillna(fallback)
        )

    def _transform_abs_curvature(self, X: pd.DataFrame) -> None:
        for feat in _CURVATURE_FEATS:
            safe = feat.replace('.', '_')
            edges, curv = self._curvature_store[feat]
            X[f'{safe}_abs_curvature'] = (
                _lookup_bins(X[feat], edges, curv).fillna(0.0)
            )


# ---------------------------------------------------------------------------
# Private helpers
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

    # Hard guard: neighborhood_subscription_density must be present
    # (IntegralFeatureEngineer runs before DerivativeFeatureEngineer).
    if 'neighborhood_subscription_density' not in df.columns:
        raise RuntimeError(
            "decay_x_density requires 'neighborhood_subscription_density'. "
            "Ensure IntegralFeatureEngineer runs before DerivativeFeatureEngineer "
            "(FeaturePipeline does this). "
            "Check DAG order: crisis -> integrals -> derivatives."
        )
    df['decay_x_density'] = (
        df['joint_economic_decay'] * df['neighborhood_subscription_density']
    )


# ---------------------------------------------------------------------------
# Bin-statistics helpers
# ---------------------------------------------------------------------------
def _fit_bin_rates(x, y, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantile-bin x (equal-width fallback) and return (edges, rates).

    edges : the exact numeric bin edges used to assign rows. NOT the
            IntervalIndex labels — pandas rounds those to 3 decimals, so
            re-cutting with them reassigns rows that sit between the true
            and the rounded edge.
    rates : per-bin mean of y, length len(edges) - 1, NaN for empty bins.
    """
    xs = pd.Series(np.asarray(x, dtype=float))
    try:
        codes, edges = pd.qcut(xs, q=n_bins, labels=False, retbins=True, duplicates='drop')
    except ValueError:
        codes, edges = pd.cut(xs, bins=n_bins, labels=False, retbins=True, duplicates='drop')
    observed = pd.Series(np.asarray(y, dtype=float)).groupby(codes).mean()
    rates = np.full(len(edges) - 1, np.nan)
    rates[observed.index.astype(int)] = observed.to_numpy()
    return edges, rates


def _abs_curvature_from_rates(rates: np.ndarray) -> np.ndarray:
    """|discrete second difference| over the non-empty bins; NaN for empty bins."""
    out = np.full_like(rates, np.nan)
    obs = ~np.isnan(rates)
    r = rates[obs]
    out[obs] = np.abs(np.gradient(np.gradient(r))) if len(r) >= 3 else np.zeros_like(r)
    return out


def _lookup_bins(x: pd.Series, edges: np.ndarray, values: np.ndarray) -> pd.Series:
    """Map each row to values[bin] using the stored edges; NaN outside the edges."""
    codes = pd.cut(x.astype(float), bins=edges, labels=False, include_lowest=True)
    valid = codes.notna().to_numpy()
    out = np.full(len(x), np.nan)
    out[valid] = values[codes.to_numpy()[valid].astype(int)]
    return pd.Series(out, index=x.index)
