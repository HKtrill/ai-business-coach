"""
glass_pipeline.glass_brw.feature_engineering
=============================================
RF Stage 2 feature engineering for the Glass Cascade.

Produces 8 continuous source features from preprocessed UCI bank data,
then converts them to 29 mutually-exclusive binary bins via binning.py.

No imports from feature_research — all logic is self-contained so the
production pipeline has zero dependency on the research package.

DAG (strict ordering required):
    crisis      (stateless) → economic crisis indicators + behavioural intermediates
    integrals   (fitted)    → economic_stress_integral, neighborhood_subscription_density
    prior       (stateless) → has_prior_contact  [needs economic_stress_integral]
    derivatives (fitted)    → economic_curvature_intensity, joint_economic_decay
                              [needs neighborhood_subscription_density in df]
    temporal    (fitted)    → dow_month_encoded
    overlap     (stateless) → behavioral_favorability, cpi_high_cellular
    select                  → RF_SOURCE_FEATURES (8 columns)

Public API
----------
RFFeatureEngineer
    Fit-on-train / transform class wrapping the full DAG.

engineer_features(X_train, X_test, y_train) → (X_train_eng, X_test_eng)
    Cell 10 entry point. Returns 29-column binary DataFrames.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# ---------------------------------------------------------------------------
# RF source features (8 continuous, pre-binning)
# ---------------------------------------------------------------------------
RF_SOURCE_FEATURES: list[str] = [
    "campaign",
    "neighborhood_subscription_density",
    "cons.conf.idx",
    "economic_curvature_intensity",
    "joint_economic_decay",
    "dow_month_encoded",
    "behavioral_favorability",
    "cpi_high_cellular",
]

# ===========================================================================
# Private constants  (inlined from feature_research — no runtime dependency)
# ===========================================================================

# ── crisis.py ──────────────────────────────────────────────────────────────
_CRISIS_EURIBOR_THRESH:   float = 1.5
_CRISIS_EMP_VAR_THRESH:   float = -1.0
_CRISIS_NR_EMP_THRESH:    float = 5100.0
_CRISIS_SCORE_CUTOFF:     int   = 3
_HIGH_CONVERSION_MONTHS:  tuple = (3, 9, 10, 12)

# ── integrals.py ───────────────────────────────────────────────────────────
_KDE_KEY_FEATURES:    tuple = ("euribor3m", "nr.employed", "emp.var.rate")
_KDE_BANDWIDTH:       float = 1.0
_KDE_N_REF:           int   = 5_000
_KDE_LARGE_DS_THRESH: int   = 10_000
_ESI_EURIBOR_CLIFF:   float = 1.5
_ESI_NR_EMP_CLIFF:    float = 5_200.0
_ESI_WEIGHTS:         tuple = (3, 2, 1)   # euribor, emp_var, nr_employed
_ESI_NORM:            float = float(sum(_ESI_WEIGHTS))

# ── derivatives.py ─────────────────────────────────────────────────────────
_SIGMOID_PARAMS: dict = {
    "euribor3m":    (1.5,  0.5),
    "emp.var.rate": (-1.0, 0.5),
}
_DECAY_PARAMS: dict = {
    "euribor3m":    (0.50, 0.60),
    "nr.employed":  (0.02, 4964.0),
    "emp.var.rate": (0.80, -2.0),
}
_N_BINS:            int   = 20
_SIGMOID_DERIV_MAX: float = 0.25
_CURVATURE_FEATS:   tuple = ("euribor3m", "nr.employed", "emp.var.rate")

# ── temporal (Laplace encoding) ────────────────────────────────────────────
_LAPLACE_SMOOTHING: int = 100

# ── overlap.py ─────────────────────────────────────────────────────────────
_LOW_STRESS_THRESH:  float = 0.2
_CPI_HIGH_THRESH:    float = 93.5
_BF_WEIGHTS: dict = {
    "cellular":      3,
    "good_month":    3,
    "prior_contact": 2,
    "default_clean": 1,
    "age_senior":    1,
}
_BF_TOTAL_WEIGHT:   float = float(sum(_BF_WEIGHTS.values()))
_AGE_SENIOR_THRESH: int   = 50

_OVERLAP_REQUIRED: tuple = (
    "economic_stress_integral",
    "cons.price.idx",
    "contact",
    "cellular_contact",
    "high_conversion_month",
    "default_clean",
    "has_prior_contact",
    "age",
)

# ===========================================================================
# Private stateless functions  (inlined from crisis.py, prior.py, overlap.py)
# ===========================================================================

def _add_crisis_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add economic crisis indicators and behavioural intermediates.
    Inlined from feature_research.feature_engineering.crisis.
    No target dependency — safe on any split.

    Adds
    ----
    economic_crisis_score  int [0–6]   weighted sum of three crisis flags
    cellular_crisis        int {0,1}   LIVE (LR Stage 1) — not consumed by RF
    high_conversion_month  int {0,1}   INTERMEDIATE → behavioral_favorability
    cellular_contact       int {0,1}   INTERMEDIATE → behavioral_favorability
    default_clean          int {0,1}   INTERMEDIATE → behavioral_favorability
    """
    df = df.copy()
    df["economic_crisis_score"] = (
          (df["euribor3m"]    < _CRISIS_EURIBOR_THRESH).astype(int) * 3
        + (df["emp.var.rate"] < _CRISIS_EMP_VAR_THRESH).astype(int) * 2
        + (df["nr.employed"]  < _CRISIS_NR_EMP_THRESH).astype(int)  * 1
    )
    df["cellular_crisis"] = (
        (df["contact"] == 0)
        & (df["economic_crisis_score"] >= _CRISIS_SCORE_CUTOFF)
    ).astype(int)
    df["high_conversion_month"] = df["month"].isin(_HIGH_CONVERSION_MONTHS).astype(int)
    df["cellular_contact"]      = (df["contact"] == 0).astype(int)
    df["default_clean"]         = (df["default"] == 0).astype(int)
    return df


def _add_prior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add prior-contact binary flag and its economic stress interaction.
    Inlined from feature_research.feature_engineering.prior.
    Stateless but requires economic_stress_integral from IntegralFeatureEngineer.

    Adds
    ----
    has_prior_contact  int   {0,1}     INTERMEDIATE → behavioral_favorability
    prior_x_stress     float [0, ~1+]  LIVE (EBM Stage 3) — not consumed by RF
    """
    if "economic_stress_integral" not in df.columns:
        raise ValueError(
            "_add_prior_features requires 'economic_stress_integral'. "
            "Ensure _IntegralFeatureEngineer.transform() runs first in the DAG."
        )
    df = df.copy()
    df["has_prior_contact"] = (df["previous"] >= 1).astype(int)
    df["prior_x_stress"]    = df["has_prior_contact"] * df["economic_stress_integral"]
    return df


def _add_overlap_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add overlap-zone interaction and behavioral composite features.
    Inlined from feature_research.feature_engineering.overlap.
    Must run last in the DAG — requires all prior intermediates.

    Adds
    ----
    low_stress_zone          int   {0,1}   INTERMEDIATE (~60–65% of data)
    cpi_high_cellular        int   {0,1}   LIVE (RF Stage 2, EBM Stage 3)
    behavioral_favorability  float [0, 1]  LIVE (RF Stage 2, EBM Stage 3)
    overlap_default_clean    int   {0,1}   LIVE (EBM Stage 3)
    overlap_behavioral_score float [0, 1]  LIVE (EBM Stage 3)
    """
    missing = [c for c in _OVERLAP_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"_add_overlap_features missing required columns: {missing}\n"
            "DAG order: crisis → integrals → prior → derivatives → temporal → overlap."
        )
    df = df.copy()

    df["low_stress_zone"] = (
        df["economic_stress_integral"] < _LOW_STRESS_THRESH
    ).astype(int)

    df["cpi_high_cellular"] = (
        (df["cons.price.idx"] >= _CPI_HIGH_THRESH) & (df["contact"] == 0)
    ).astype(int)

    df["behavioral_favorability"] = (
          _BF_WEIGHTS["cellular"]      * df["cellular_contact"]
        + _BF_WEIGHTS["good_month"]    * df["high_conversion_month"]
        + _BF_WEIGHTS["prior_contact"] * df["has_prior_contact"]
        + _BF_WEIGHTS["default_clean"] * df["default_clean"]
        + _BF_WEIGHTS["age_senior"]    * (df["age"] >= _AGE_SENIOR_THRESH).astype(int)
    ) / _BF_TOTAL_WEIGHT

    df["overlap_default_clean"]    = df["low_stress_zone"] * df["default_clean"]
    df["overlap_behavioral_score"] = df["low_stress_zone"] * df["behavioral_favorability"]
    return df


# ===========================================================================
# Private sub-engineer classes  (inlined from integrals.py, derivatives.py)
# ===========================================================================

class _IntegralFeatureEngineer:
    """
    Inlined from feature_research.feature_engineering.integrals.IntegralFeatureEngineer.

    economic_stress_integral  — analytic formula, no target dep, leakage-free.
    neighborhood_subscription_density — Gaussian KDE weighted by training labels;
        StandardScaler and reference points fitted on training data only.
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
        self._scaler: StandardScaler | None = None
        self._X_ref:  np.ndarray | None     = None
        self._y_ref:  np.ndarray | None     = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_IntegralFeatureEngineer":
        """Fit KDE scaler and store reference points from training data."""
        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(X[list(_KDE_KEY_FEATURES)])
        y_arr        = np.asarray(y)
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
        """Apply fitted scaler; compute both integral features."""
        if not self.fitted:
            raise ValueError("_IntegralFeatureEngineer not fitted. Call fit() first.")
        X = X.copy()
        X["economic_stress_integral"] = _economic_stress_integral(X)
        X_scaled = self._scaler.transform(X[list(_KDE_KEY_FEATURES)])
        X["neighborhood_subscription_density"] = _apply_kde(
            X_scaled, self._X_ref, self._y_ref, self.bandwidth
        )
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


class _DerivativeFeatureEngineer:
    """
    Inlined from feature_research.feature_engineering.derivatives.DerivativeFeatureEngineer.

    Sigmoid and decay features: no target dep, computed directly.
    euribor3m_local_rate, abs_curvature intermediates: fitted on training data.

    transform() requires neighborhood_subscription_density to be present in X
    (added by _IntegralFeatureEngineer.transform) for decay_x_density composite.
    RF consumes: economic_curvature_intensity, joint_economic_decay.
    """

    def __init__(self, n_bins: int = _N_BINS) -> None:
        self.n_bins = n_bins
        self._local_rate_store: dict = {}
        self._curvature_store:  dict = {}
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_DerivativeFeatureEngineer":
        """
        Fit bin rates and curvature maps on training data.
        Reads only raw economic columns — no prior DAG step needed.
        """
        self._fit_local_rate(X, y, "euribor3m")
        for feat in _CURVATURE_FEATS:
            self._fit_abs_curvature(X, y, feat)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Add all derivative features using fitted state + stateless decays.
        Requires neighborhood_subscription_density in X (from integrals step).
        """
        if not self.fitted:
            raise ValueError("_DerivativeFeatureEngineer not fitted. Call fit() first.")
        X = X.copy()
        _add_sigmoid_features(X)
        _add_decay_features(X)
        self._transform_local_rate(X)
        self._transform_abs_curvature(X)
        _add_composites(X)   # RuntimeError if NSD missing — confirms DAG order
        _fill_nans(X)
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ── private fit helpers ──────────────────────────────────────────────
    def _fit_local_rate(self, X: pd.DataFrame, y: pd.Series, feat: str) -> None:
        df_tmp = pd.DataFrame({"feat": X[feat].values, "y": np.asarray(y)})
        try:
            bins = pd.qcut(df_tmp["feat"], q=self.n_bins, duplicates="drop")
        except ValueError:
            bins = pd.cut(df_tmp["feat"], bins=self.n_bins, duplicates="drop")
        bin_rates = df_tmp.groupby(bins, observed=True)["y"].mean()
        self._local_rate_store[feat] = (bins.cat.categories, bin_rates)

    def _fit_abs_curvature(self, X: pd.DataFrame, y: pd.Series, feat: str) -> None:
        df_tmp = pd.DataFrame({"feat": X[feat].values, "y": np.asarray(y)})
        try:
            bins = pd.qcut(df_tmp["feat"], q=self.n_bins, duplicates="drop")
        except ValueError:
            bins = pd.cut(df_tmp["feat"], bins=self.n_bins, duplicates="drop")
        bin_rates   = df_tmp.groupby(bins, observed=True)["y"].mean()
        rate_values = bin_rates.values
        curvature   = (
            np.abs(np.gradient(np.gradient(rate_values)))
            if len(rate_values) >= 3
            else np.zeros_like(rate_values)
        )
        self._curvature_store[feat] = (
            bins.cat.categories,
            dict(zip(bin_rates.index, curvature)),
        )

    # ── private transform helpers ────────────────────────────────────────
    def _transform_local_rate(self, X: pd.DataFrame) -> None:
        intervals, bin_rates = self._local_rate_store["euribor3m"]
        mapped = (
            pd.cut(X["euribor3m"], bins=intervals, include_lowest=True)
            .map(bin_rates)
            .astype(float)
        )
        X["euribor3m_local_rate"] = mapped.fillna(float(bin_rates.median()))

    def _transform_abs_curvature(self, X: pd.DataFrame) -> None:
        for feat in _CURVATURE_FEATS:
            safe = feat.replace(".", "_")
            intervals, curv_map = self._curvature_store[feat]
            mapped = (
                pd.cut(X[feat], bins=intervals, include_lowest=True)
                .map(curv_map)
                .astype(float)
            )
            X[f"{safe}_abs_curvature"] = mapped.fillna(0.0)


class _TemporalFeatureEngineer:
    """
    Laplace-smoothed target encoding of day_of_week × month combinations.
    Produces dow_month_encoded (float, [0, 1] approximate range).

    Unseen day×month combinations at transform time fall back to global_rate.
    """

    def __init__(self, smoothing_factor: int = _LAPLACE_SMOOTHING) -> None:
        self.smoothing_factor = smoothing_factor
        self._rates:       dict  = {}
        self._global_rate: float = 0.0
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "_TemporalFeatureEngineer":
        """Fit smoothed day×month rates on training data."""
        df_tmp = pd.DataFrame({
            "dow":   X["day_of_week"].values,
            "month": X["month"].values,
            "y":     np.asarray(y),
        })
        self._global_rate = float(df_tmp["y"].mean())
        k     = self.smoothing_factor
        stats = df_tmp.groupby(["dow", "month"])["y"].agg(["sum", "count"])
        self._rates = {
            (int(dow), int(month)): (
                (row["sum"] + k * self._global_rate) / (row["count"] + k)
            )
            for (dow, month), row in stats.iterrows()
        }
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map each row to its smoothed rate; unseen combos → global rate."""
        if not self.fitted:
            raise ValueError("_TemporalFeatureEngineer not fitted. Call fit() first.")
        X = X.copy()
        X["dow_month_encoded"] = [
            self._rates.get((int(d), int(m)), self._global_rate)
            for d, m in zip(X["day_of_week"], X["month"])
        ]
        return X

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ===========================================================================
# Private shared helpers  (inlined from derivatives.py, integrals.py)
# ===========================================================================

def _economic_stress_integral(df: pd.DataFrame) -> pd.Series:
    """
    Weighted normalized economic stress. No target — leakage-free on any split.

        euribor_stress  = max(0, (cliff - euribor3m)  / cliff)
        emp_var_stress  = max(0, -emp.var.rate / 3)
        nr_emp_stress   = max(0, (5200 - nr.employed) / 100)
        S = (3*f1 + 2*f2 + 1*f3) / 6
    """
    w_euribor, w_emp_var, w_nr_emp = _ESI_WEIGHTS
    euribor_stress = np.maximum(
        0, (_ESI_EURIBOR_CLIFF - df["euribor3m"]) / _ESI_EURIBOR_CLIFF
    )
    emp_var_stress = np.maximum(0, -df["emp.var.rate"] / 3.0)
    nr_emp_stress  = np.maximum(
        0, (_ESI_NR_EMP_CLIFF - df["nr.employed"]) / 100.0
    )
    return (
        w_euribor * euribor_stress
        + w_emp_var * emp_var_stress
        + w_nr_emp  * nr_emp_stress
    ) / _ESI_NORM


def _apply_kde(
    X_scaled:  np.ndarray,
    X_ref:     np.ndarray,
    y_ref:     np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """
    Gaussian KDE: rho(x) = sum_j y_j * w_j(x), weights row-normalised.
    w_j(x) = exp(-||x - x_j||^2 / 2s^2) / sum_k exp(-||x - x_k||^2 / 2s^2)
    """
    dists   = cdist(X_scaled, X_ref, metric="euclidean")
    weights = np.exp(-(dists ** 2) / (2 * bandwidth ** 2))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-10
    return weights @ y_ref


def _add_sigmoid_features(df: pd.DataFrame) -> None:
    """Sigmoid value (intermediate) + normalised slope. Mutates df in-place."""
    for feat, (center, scale) in _SIGMOID_PARAMS.items():
        safe      = feat.replace(".", "_")
        z         = -(df[feat] - center) / scale
        sig_val   = expit(z)
        sig_slope = sig_val * (1.0 - sig_val) / _SIGMOID_DERIV_MAX
        df[f"{safe}_sigmoid"]       = sig_val
        df[f"{safe}_sigmoid_slope"] = sig_slope


def _add_decay_features(df: pd.DataFrame) -> None:
    """Per-feature exponential decay intermediates. Mutates df in-place."""
    lam, x0 = _DECAY_PARAMS["euribor3m"]
    df["euribor_decay"]     = np.exp(-lam * np.maximum(0.0, df["euribor3m"]    - x0))
    lam, x0 = _DECAY_PARAMS["nr.employed"]
    df["nr_employed_decay"] = np.exp(-lam * np.maximum(0.0, df["nr.employed"]  - x0))
    lam, x0 = _DECAY_PARAMS["emp.var.rate"]
    df["emp_var_decay"]     = np.exp(-lam * np.maximum(0.0, df["emp.var.rate"] - x0))


def _add_composites(df: pd.DataFrame) -> None:
    """
    Composite features from derivative intermediates. Mutates df in-place.
    Hard RuntimeError if neighborhood_subscription_density is absent —
    enforces DAG order: integrals must run before derivatives.
    """
    df["economic_curvature_intensity"] = df[[
        "euribor3m_abs_curvature",
        "nr_employed_abs_curvature",
        "emp_var_rate_abs_curvature",
    ]].mean(axis=1)

    df["joint_economic_decay"] = (
        df["euribor_decay"] * df["emp_var_decay"] * df["nr_employed_decay"]
    )

    if "neighborhood_subscription_density" not in df.columns:
        raise RuntimeError(
            "decay_x_density requires 'neighborhood_subscription_density'. "
            "Ensure _IntegralFeatureEngineer.transform() runs before "
            "_DerivativeFeatureEngineer.transform(). "
            "DAG order: crisis → integrals → prior → derivatives."
        )
    df["decay_x_density"] = (
        df["joint_economic_decay"] * df["neighborhood_subscription_density"]
    )


def _fill_nans(df: pd.DataFrame) -> None:
    """Replace inf/NaN introduced by binning edge cases with column median."""
    for col in df.select_dtypes(include="number").columns:
        if np.isinf(df[col]).any():
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())


# ===========================================================================
# Public API
# ===========================================================================

class RFFeatureEngineer:
    """
    Full DAG feature engineer for RF Stage 2.

    Three fitted sub-engineers, all fittable directly on raw X_train
    (no DAG pre-processing needed during fit — each reads only the raw UCI
    columns it needs):
        _IntegralFeatureEngineer   — KDE scaler + reference points from y_train
        _DerivativeFeatureEngineer — curvature bins + local rate bins from y_train
        _TemporalFeatureEngineer   — Laplace day×month encoding from y_train

    transform() runs the full DAG in strict order. Each step's outputs are
    consumed by later steps; the RuntimeError in _add_composites acts as a
    hard guardrail against misordering.

    Usage
    -----
        eng = RFFeatureEngineer()
        X_train_8 = eng.fit_transform(X_train, y_train)
        X_test_8  = eng.transform(X_test)
    """

    def __init__(self) -> None:
        self._integral_eng   = _IntegralFeatureEngineer()
        self._derivative_eng = _DerivativeFeatureEngineer()
        self._temporal_eng   = _TemporalFeatureEngineer()
        self.fitted = False

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> "RFFeatureEngineer":
        """
        Fit all three sub-engineers on raw training features.

        All three read only raw UCI columns present in X_train directly
        (euribor3m, nr.employed, emp.var.rate, day_of_week, month).
        No DAG pre-processing is needed during the fit phase.

        Parameters
        ----------
        X_train : pd.DataFrame
            Raw preprocessed UCI features (output of BankPreprocessor).
        y_train : pd.Series
            Binary target aligned with X_train.
        """
        self._integral_eng.fit(X_train, y_train)
        self._derivative_eng.fit(X_train, y_train)
        self._temporal_eng.fit(X_train, y_train)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the full feature engineering DAG and return 8 RF source features.

        DAG order
        ---------
        1. _add_crisis_features        (stateless)
        2. _integral_eng.transform     (fitted KDE; adds ESI + NSD)
        3. _add_prior_features         (stateless; needs ESI)
        4. _derivative_eng.transform   (fitted; needs NSD for composites)
        5. _temporal_eng.transform     (fitted Laplace encoding)
        6. _add_overlap_features       (stateless; needs all intermediates)
        7. select RF_SOURCE_FEATURES

        Returns
        -------
        pd.DataFrame   Shape (N, 8), columns == RF_SOURCE_FEATURES.
        """
        if not self.fitted:
            raise ValueError(
                "RFFeatureEngineer not fitted. Call fit() first."
            )
        X = _add_crisis_features(X)
        X = self._integral_eng.transform(X)
        X = _add_prior_features(X)
        X = self._derivative_eng.transform(X)
        X = self._temporal_eng.transform(X)
        X = _add_overlap_features(X)
        return X[RF_SOURCE_FEATURES].copy()

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> pd.DataFrame:
        """Fit on X_train + y_train, then transform X_train."""
        return self.fit(X_train, y_train).transform(X_train)


def engineer_features(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cell 10 entry point — engineer and bin RF features for both splits.

    Fits RFFeatureEngineer on X_train only (no leakage), transforms both
    splits to 8 continuous source features, applies the locked BINNING_STRATEGY
    to produce 29 binary bins, and validates mutual exclusivity on train.

    Parameters
    ----------
    X_train : pd.DataFrame   Raw preprocessed features, shape (N_train, F)
    X_test  : pd.DataFrame   Raw preprocessed features, shape (N_test, F)
    y_train : pd.Series      Binary target aligned with X_train

    Returns
    -------
    X_train_eng : pd.DataFrame   Shape (N_train, 29)  29 mutually-exclusive bins
    X_test_eng  : pd.DataFrame   Shape (N_test, 29)   29 mutually-exclusive bins
    """
    from .binning import (
        create_binary_features,
        validate_binary_features,
        RF_FEATURES_BINARY,
    )

    print("=" * 80)
    print("🔧 CELL 10: RF STAGE 2 FEATURE ENGINEERING")
    print("=" * 80)

    # ── 1. Fit and produce 8 continuous source features ───────────────────
    print("\n[1/3] Building 8 continuous source features...")
    eng       = RFFeatureEngineer()
    X_train_8 = eng.fit_transform(X_train, y_train)
    X_test_8  = eng.transform(X_test)

    print(f"      Train: {X_train_8.shape} | NaNs: {X_train_8.isna().sum().sum()}")
    print(f"      Test:  {X_test_8.shape}  | NaNs: {X_test_8.isna().sum().sum()}")
    _validate_source_features(X_train_8, X_test_8)

    # ── 2. Bin train → 29 binary features (with lift display) ─────────────
    print("\n[2/3] Binning to 29 binary features...")
    X_train_with_y      = X_train_8.copy()
    X_train_with_y["y"] = y_train.values
    X_train_binned, _   = create_binary_features(X_train_with_y, target_col="y")
    X_train_eng         = X_train_binned[RF_FEATURES_BINARY].copy()

    # Test: same stateless thresholds — no target needed for computation.
    X_test_eng = _apply_binning_thresholds(X_test_8, RF_FEATURES_BINARY)

    # ── 3. Validate mutual exclusivity on train (hard stop) ───────────────
    print("\n[3/3] Validating binary features...")
    validate_binary_features(X_train_binned, target_col="y", verbose=True)

    print(f"\n✅ RF features ready")
    print(f"   Train: {X_train_eng.shape} | Test: {X_test_eng.shape}")
    print("=" * 80)

    return X_train_eng, X_test_eng


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_source_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> None:
    """Assert all 8 source features are present and NaN/inf-free in both splits."""
    for split_name, df in [("train", X_train), ("test", X_test)]:
        for feat in RF_SOURCE_FEATURES:
            assert feat in df.columns, (
                f"Source feature '{feat}' missing from {split_name}."
            )
            n_nan = int(df[feat].isna().sum())
            n_inf = int(np.isinf(df[feat].values).sum())
            if n_nan > 0 or n_inf > 0:
                raise ValueError(
                    f"Source feature '{feat}' in {split_name} "
                    f"has {n_nan} NaNs and {n_inf} infs."
                )


def _apply_binning_thresholds(
    df: pd.DataFrame,
    rf_features_binary: list[str],
) -> pd.DataFrame:
    """
    Apply BINNING_STRATEGY thresholds without the lift display report.
    Used for test-set binning where target labels are not available.
    Mirrors the threshold logic in create_binary_features exactly.
    """
    from .binning import BINNING_STRATEGY

    df = df.copy()
    for group_def in BINNING_STRATEGY.values():
        source = group_def["source"]
        if "passthrough" in group_def:
            df[group_def["passthrough"]] = df[source].astype("int8")
            continue
        col = df[source]
        for bin_name, lo, hi in group_def["bins"]:
            if lo is None:
                mask = col <= hi
            elif hi is None:
                mask = col > lo
            else:
                mask = (col > lo) & (col <= hi)
            df[bin_name] = mask.astype("int8")

    return df[rf_features_binary].copy()