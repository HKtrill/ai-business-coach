"""
feature_research.feature_engineering.integrals
===============================================
Calculus-based integral features extracted from the original Cell 10B.

Survivors after dead-feature audit (4 → 2):
  - economic_stress_integral         INTERMEDIATE → prior_x_stress (Cell 10E)
  - neighborhood_subscription_density  LIVE (RF Stage 2) + intermediate
                                        for decay_x_density (EBM Stage 3)

Pruned:
  - cumulative_campaign_pressure     dead — not in any model config
  - pressure_density_synergy         dead — not in any model config

Design note: neighborhood_subscription_density uses target labels at
compute time (Gaussian KDE over reference points drawn from the full
dataset).  This is intentional for *exploratory research*.  In a
production pipeline, fit the reference set on training data only and
transform train + test separately.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

__all__ = ["add_integral_features"]

# ---------------------------------------------------------------------------
# Private constants
# ---------------------------------------------------------------------------
_KDE_KEY_FEATURES: tuple = ('euribor3m', 'nr.employed', 'emp.var.rate')
_KDE_BANDWIDTH: float = 1.0
_KDE_N_REF: int = 5_000          # reference-point cap for large datasets
_KDE_LARGE_DS_THRESH: int = 10_000

# Economic stress integral component thresholds / weights
_ESI_EURIBOR_CLIFF: float = 1.5
_ESI_NR_EMP_CLIFF: float = 5_200
_ESI_WEIGHTS: tuple = (3, 2, 1)  # euribor, emp_var, nr_employed
_ESI_NORM: float = float(sum(_ESI_WEIGHTS))


# ---------------------------------------------------------------------------
# Public API
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
        Dataframe after add_crisis_features().  Must contain: euribor3m,
        emp.var.rate, nr.employed, and target_col.
    target_col : str
        Binary target column name.
    random_state : int
        Seed for KDE reference-point sampling.

    Returns
    -------
    pd.DataFrame
        Copy of df with integral features appended.

    New columns
    -----------
    economic_stress_integral         float  [0, ~1+]
        Weighted multi-dimensional accumulation of economic stress.
        INTERMEDIATE — feeds prior_x_stress (Cell 10E).
        Formula: (3·euribor_stress + 2·emp_var_stress + 1·nr_emp_stress) / 6

    neighborhood_subscription_density  float  [0, 1]
        Gaussian KDE estimate of P(subscribe) over the three-dimensional
        economic feature space (euribor3m × nr.employed × emp.var.rate).
        LIVE (RF Stage 2) + INTERMEDIATE for decay_x_density (EBM Stage 3).

        NOTE: Computed on the full dataset passed in — fit on train only
        in production.
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

    Each component is a normalised proximity measure (0 = no stress):
      euribor_stress  = max(0, (cliff - euribor3m) / cliff)
      emp_var_stress  = max(0, -emp.var.rate / 3)
      nr_emp_stress   = max(0, (5200 - nr.employed) / 100)

    Combined as a weighted Riemann-sum approximation:
      S = (w1·f1 + w2·f2 + w3·f3) / (w1+w2+w3)
    """
    w_euribor, w_emp_var, w_nr_emp = _ESI_WEIGHTS

    euribor_stress = np.maximum(
        0, (_ESI_EURIBOR_CLIFF - df['euribor3m']) / _ESI_EURIBOR_CLIFF
    )
    emp_var_stress = np.maximum(0, (-df['emp.var.rate']) / 3.0)
    nr_emp_stress = np.maximum(
        0, (_ESI_NR_EMP_CLIFF - df['nr.employed']) / 100.0
    )

    return (
        w_euribor * euribor_stress
        + w_emp_var * emp_var_stress
        + w_nr_emp * nr_emp_stress
    ) / _ESI_NORM


def _kde_density(
    df: pd.DataFrame,
    target_col: str,
    random_state: int,
    key_features: tuple = _KDE_KEY_FEATURES,
    bandwidth: float = _KDE_BANDWIDTH,
    n_ref: int = _KDE_N_REF,
) -> np.ndarray:
    """
    Gaussian KDE estimate of P(subscribe) in economic feature space.

        ρ(x) ≈ Σⱼ yⱼ · wⱼ(x)
        wⱼ(x) = exp(-‖x - xⱼ‖² / 2σ²) / Σₖ exp(-‖x - xₖ‖² / 2σ²)

    For datasets > _KDE_LARGE_DS_THRESH uses a random reference-point
    approximation (O(N·n_ref) instead of O(N²)).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[list(key_features)])
    y = df[target_col].values

    if len(df) > _KDE_LARGE_DS_THRESH:
        rng = np.random.default_rng(random_state)
        ref_idx = rng.choice(len(df), size=min(n_ref, len(df)), replace=False)
        X_ref = X[ref_idx]
        y_ref = y[ref_idx]
    else:
        X_ref, y_ref = X, y

    dists = cdist(X, X_ref, metric='euclidean')
    weights = np.exp(-(dists ** 2) / (2 * bandwidth ** 2))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-10

    return weights @ y_ref