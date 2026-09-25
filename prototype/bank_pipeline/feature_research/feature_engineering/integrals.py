"""
feature_research.feature_engineering.integrals
===============================================
Calculus-based integral features extracted from the original Cell 10B.

Survivors after dead-feature audit (4 -> 2):
  - economic_stress_integral           INTERMEDIATE -> prior_x_stress (Cell 10E)
  - neighborhood_subscription_density  LIVE (RF Stage 2) + intermediate
                                        for decay_x_density (EBM Stage 3)

Pruned:
  - cumulative_campaign_pressure     dead
  - pressure_density_synergy         dead

Leakage note (RESOLVED in IntegralFeatureEngineer):
    The functional add_integral_features() fits the KDE scaler and draws
    reference points from the full df passed in, including test labels.
    In the research notebook this is pre-split — a known leakage point
    kept intentionally for exploratory use.

    economic_stress_integral has no target dependency and is leakage-free
    in both the functional API and the class.

    In the production pipeline (glass_cascade), use IntegralFeatureEngineer
    which enforces a fit-on-train / transform pattern:
        eng = IntegralFeatureEngineer()
        X_train = eng.fit_transform(X_train, y_train)
        X_test  = eng.transform(X_test)

    The scaler and KDE reference points are fitted on training data only
    and re-used for the test transform.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

__all__ = ["add_integral_features", "IntegralFeatureEngineer"]

# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------
_KDE_KEY_FEATURES:    tuple = ('euribor3m', 'nr.employed', 'emp.var.rate')
_KDE_BANDWIDTH:       float = 1.0
_KDE_N_REF:           int   = 5_000
_KDE_LARGE_DS_THRESH: int   = 10_000

_ESI_EURIBOR_CLIFF: float = 1.5
_ESI_NR_EMP_CLIFF:  float = 5_200.0
_ESI_WEIGHTS:       tuple = (3, 2, 1)   # euribor, emp_var, nr_employed
_ESI_NORM:          float = float(sum(_ESI_WEIGHTS))


# ---------------------------------------------------------------------------
# Production class (fit/transform — leakage-free)
# ---------------------------------------------------------------------------
class IntegralFeatureEngineer:
    """
    Fit-on-train / transform implementation of integral features.

    economic_stress_integral is computed analytically — no target needed,
    safe on any split directly.

    neighborhood_subscription_density uses a Gaussian KDE over the economic
    feature space weighted by training labels. The StandardScaler and KDE
    reference points are fitted on training data only and re-applied at
    transform time to prevent test label leakage.

    Parameters
    ----------
    bandwidth : float
        KDE bandwidth (sigma). Default 1.0.
    n_ref : int
        Maximum number of reference points for large-dataset approximation.
    random_state : int
        Seed for reference-point sampling.

    Usage
    -----
        eng = IntegralFeatureEngineer()
        X_train = eng.fit_transform(X_train, y_train)
        X_test  = eng.transform(X_test)
    """

    def __init__(
        self,
        bandwidth:    float = _KDE_BANDWIDTH,
        n_ref:        int   = _KDE_N_REF,
        random_state: int   = 42,
    ) -> None:
        self.bandwidth    = bandwidth
        self.n_ref        = n_ref
        self.random_state = random_state

        self._scaler: StandardScaler = None
        self._X_ref:  np.ndarray    = None
        self._y_ref:  np.ndarray    = None
        self.fitted = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series
    ) -> 'IntegralFeatureEngineer':
        """
        Fit KDE scaler and store reference points from training data.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain: euribor3m, nr.employed, emp.var.rate.
        y : pd.Series
            Binary target aligned with X.
        """
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X[list(_KDE_KEY_FEATURES)])
        y_arr    = np.asarray(y)

        if len(X) > _KDE_LARGE_DS_THRESH:
            rng     = np.random.default_rng(self.random_state)
            ref_idx = rng.choice(len(X), size=min(self.n_ref, len(X)), replace=False)
            self._X_ref = X_scaled[ref_idx]
            self._y_ref = y_arr[ref_idx]
        else:
            self._X_ref = X_scaled
            self._y_ref = y_arr

        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a split using fitted scaler and reference points.

        economic_stress_integral is computed directly (no target needed).
        neighborhood_subscription_density uses the fitted KDE state.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain: euribor3m, nr.employed, emp.var.rate.

        Returns
        -------
        pd.DataFrame
            Copy of X with both integral features appended.
        """
        if not self.fitted:
            raise ValueError(
                "IntegralFeatureEngineer not fitted. Call fit() first."
            )
        X = X.copy()

        # economic_stress_integral — no target, computed directly
        X['economic_stress_integral'] = _economic_stress_integral(X)

        # neighborhood_subscription_density — use fitted scaler + ref points
        X_scaled = self._scaler.transform(X[list(_KDE_KEY_FEATURES)])
        X['neighborhood_subscription_density'] = _apply_kde(
            X_scaled, self._X_ref, self._y_ref, self.bandwidth
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
def add_integral_features(
    df: pd.DataFrame,
    target_col: str = 'y',
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Add calculus-based integral features for overlap-zone disambiguation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe after add_crisis_features(). Must contain: euribor3m,
        emp.var.rate, nr.employed, and target_col.
    target_col : str
        Binary target column name.
    random_state : int
        Seed for KDE reference-point sampling.

    Returns
    -------
    pd.DataFrame
        Copy of df with integral features appended.

    NOTE: neighborhood_subscription_density uses target labels at compute
    time over the full df passed in. In the research notebook this is
    pre-split — a known leakage point kept intentionally for exploratory
    use. In production use IntegralFeatureEngineer.
    """
    df = df.copy()

    df['economic_stress_integral'] = _economic_stress_integral(df)
    df['neighborhood_subscription_density'] = _kde_density(
        df, target_col=target_col, random_state=random_state
    )

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------
def _economic_stress_integral(df: pd.DataFrame) -> pd.Series:
    """
    Weighted discrete approximation of multi-dimensional economic stress.
    No target used — leakage-free on any split.

      euribor_stress  = max(0, (cliff - euribor3m) / cliff)
      emp_var_stress  = max(0, -emp.var.rate / 3)
      nr_emp_stress   = max(0, (5200 - nr.employed) / 100)

      S = (w1*f1 + w2*f2 + w3*f3) / (w1+w2+w3)
    """
    w_euribor, w_emp_var, w_nr_emp = _ESI_WEIGHTS

    euribor_stress = np.maximum(
        0, (_ESI_EURIBOR_CLIFF - df['euribor3m']) / _ESI_EURIBOR_CLIFF
    )
    emp_var_stress = np.maximum(0, (-df['emp.var.rate']) / 3.0)
    nr_emp_stress  = np.maximum(
        0, (_ESI_NR_EMP_CLIFF - df['nr.employed']) / 100.0
    )

    return (
        w_euribor * euribor_stress
        + w_emp_var * emp_var_stress
        + w_nr_emp  * nr_emp_stress
    ) / _ESI_NORM


def _apply_kde(
    X_scaled: np.ndarray,
    X_ref:    np.ndarray,
    y_ref:    np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """
    Gaussian KDE: rho(x) = sum_j y_j * w_j(x)
    where w_j(x) = exp(-||x-x_j||^2 / 2s^2) / sum_k exp(-||x-x_k||^2 / 2s^2)

    Separated from _kde_density so IntegralFeatureEngineer.transform()
    can call it with pre-scaled data and pre-fitted reference points.
    """
    dists   = cdist(X_scaled, X_ref, metric='euclidean')
    weights = np.exp(-(dists ** 2) / (2 * bandwidth ** 2))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-10
    return weights @ y_ref


def _kde_density(
    df:           pd.DataFrame,
    target_col:   str,
    random_state: int,
    key_features: tuple = _KDE_KEY_FEATURES,
    bandwidth:    float = _KDE_BANDWIDTH,
    n_ref:        int   = _KDE_N_REF,
) -> np.ndarray:
    """
    Functional KDE wrapper — fits scaler and reference points on the full df.
    Used by the research-notebook functional API only.
    """
    scaler  = StandardScaler()
    X       = scaler.fit_transform(df[list(key_features)])
    y       = df[target_col].values

    if len(df) > _KDE_LARGE_DS_THRESH:
        rng     = np.random.default_rng(random_state)
        ref_idx = rng.choice(len(df), size=min(n_ref, len(df)), replace=False)
        X_ref   = X[ref_idx]
        y_ref   = y[ref_idx]
    else:
        X_ref, y_ref = X, y

    return _apply_kde(X, X_ref, y_ref, bandwidth)