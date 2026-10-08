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

Target handling:
    economic_stress_integral is deterministic (no target).
    neighborhood_subscription_density is target-derived; IntegralFeatureEngineer
    fits the scaler and KDE reference rows on training data only and reuses
    them at transform time. Use it via FeaturePipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from feature_research.config import RANDOM_SEED
from feature_research.feature_engineering._checks import check_xy_aligned

__all__ = ["IntegralFeatureEngineer"]

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
        random_state: int   = RANDOM_SEED,
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
            Binary target aligned with X (same index if a Series).
        """
        check_xy_aligned(X, y, "IntegralFeatureEngineer.fit")
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

    """
    dists   = cdist(X_scaled, X_ref, metric='euclidean')
    weights = np.exp(-(dists ** 2) / (2 * bandwidth ** 2))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-10
    return weights @ y_ref
