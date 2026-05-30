"""
Feature Engineering for Logistic Regression Stage
==================================================

Produces the three validated LR Stage 1 features:
    cellular_crisis        — cellular contact during economic crisis
    euribor3m_local_rate   — empirical P(subscribe) per euribor quantile bin
    dow_month_encoded      — smoothed P(subscribe | day_of_week x month)

Replaces the original EconomicFeatureEngineer (PCA / composite approach)
with the validated feature set ported from the research notebook.

Leakage fixes vs research notebook:
    euribor3m_local_rate — bin rates fitted on X_train only;
                           test mapped via stored IntervalIndex.
    dow_month_encoded    — target encoding fitted on X_train + y_train;
                           unseen day-month cells fall back to global mean.

Author: Glass Pipeline Team
Date: 2026-05
"""

import pandas as pd
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants (mirrored from feature_research modules)
# ---------------------------------------------------------------------------
_CRISIS_EURIBOR_THRESH: float = 1.5
_CRISIS_EMP_VAR_THRESH: float = -1.0
_CRISIS_NR_EMP_THRESH: float = 5100.0
_CRISIS_SCORE_CUTOFF:  int   = 3

_N_BINS_DEFAULT:    int = 20
_SMOOTHING_FACTOR:  int = 100

LR_FEATURES = ['cellular_crisis', 'euribor3m_local_rate', 'dow_month_encoded']


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class LRFeatureEngineer:
    """
    Feature engineer for LR Stage 1.

    Produces exactly three features:
        cellular_crisis       — no target used; safe to compute on any split.
        euribor3m_local_rate  — fitted on train only; leakage-free on test.
        dow_month_encoded     — fitted on train only; leakage-free on test.

    Usage
    -----
        engineer = LRFeatureEngineer()
        X_train_eng = engineer.fit_transform(X_train, y_train)
        X_test_eng  = engineer.transform(X_test)
    """

    def __init__(
        self,
        n_bins: int = _N_BINS_DEFAULT,
        smoothing_factor: int = _SMOOTHING_FACTOR,
    ) -> None:
        self.n_bins = n_bins
        self.smoothing_factor = smoothing_factor

        # Fitted state — populated in fit()
        self._euribor_intervals = None   # IntervalIndex from training qcut
        self._euribor_bin_rates = None   # Series: interval -> mean(y)
        self._global_mean       = None   # float: overall train subscribe rate
        self._dm_smoothed       = None   # Series: "dow_month" key -> smoothed rate
        self.fitted             = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'LRFeatureEngineer':
        """
        Fit target-dependent encodings on training data only.

        Parameters
        ----------
        X_train : pd.DataFrame
            Must contain: euribor3m, emp.var.rate, nr.employed,
                          contact, day_of_week, month.
        y_train : pd.Series
            Binary target aligned with X_train.
        """
        print("\n" + "="*70)
        print("🔧 FITTING LR FEATURE ENGINEER")
        print("="*70)

        self._fit_euribor_local_rate(X_train, y_train)
        self._fit_dow_month_encoded(X_train, y_train)

        self.fitted = True
        print("\n✅ LR feature engineer fitted.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a split using fitted state.

        Parameters
        ----------
        X : pd.DataFrame
            Train or test split with raw preprocessed columns.

        Returns
        -------
        pd.DataFrame
            Shape (n, 3) with exactly LR_FEATURES columns.
        """
        if not self.fitted:
            raise ValueError("Engineer not fitted. Call fit() first.")

        X_eng = X.copy()
        X_eng = self._add_cellular_crisis(X_eng)
        X_eng = self._transform_euribor_local_rate(X_eng)
        X_eng = self._transform_dow_month_encoded(X_eng)

        return X_eng[LR_FEATURES]

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> pd.DataFrame:
        """Fit on X_train + y_train, then transform X_train."""
        return self.fit(X_train, y_train).transform(X_train)

    # ------------------------------------------------------------------
    # cellular_crisis
    # ------------------------------------------------------------------
    def _add_cellular_crisis(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cellular contact during economic crisis.

        crisis_score = (euribor3m < 1.5)*3
                     + (emp.var.rate < -1.0)*2
                     + (nr.employed < 5100)*1

        cellular_crisis = (contact == 0) AND (crisis_score >= 3)

        No target used — safe to compute directly on any split.
        """
        crisis_score = (
            (X['euribor3m']    < _CRISIS_EURIBOR_THRESH).astype(int) * 3
            + (X['emp.var.rate'] < _CRISIS_EMP_VAR_THRESH).astype(int) * 2
            + (X['nr.employed']  < _CRISIS_NR_EMP_THRESH).astype(int) * 1
        )
        X['cellular_crisis'] = (
            (X['contact'] == 0) & (crisis_score >= _CRISIS_SCORE_CUTOFF)
        ).astype(int)

        print(f"\n✅ cellular_crisis: {X['cellular_crisis'].sum():,} positives "
              f"({X['cellular_crisis'].mean():.1%})")
        return X

    # ------------------------------------------------------------------
    # euribor3m_local_rate
    # ------------------------------------------------------------------
    def _fit_euribor_local_rate(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> None:
        """
        Fit quantile bins and per-bin subscribe rates on training data.
        Stores IntervalIndex and rate Series for leakage-free test transform.
        """
        df_tmp = pd.DataFrame({
            'feat': X_train['euribor3m'].values,
            'y':    y_train.values,
        })
        try:
            bins = pd.qcut(df_tmp['feat'], q=self.n_bins, duplicates='drop')
        except ValueError:
            bins = pd.cut(df_tmp['feat'], bins=self.n_bins, duplicates='drop')

        self._euribor_intervals  = bins.cat.categories   # IntervalIndex
        self._euribor_bin_rates  = (
            df_tmp.groupby(bins, observed=True)['y'].mean()
        )

        print(f"\n📐 euribor3m_local_rate:")
        print(f"   Bins fitted : {len(self._euribor_intervals)}")
        print(f"   Rate range  : [{self._euribor_bin_rates.min():.4f}, "
              f"{self._euribor_bin_rates.max():.4f}]")

    def _transform_euribor_local_rate(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map euribor3m values to fitted bin rates.
        Out-of-training-range values fall back to the training median rate.
        """
        mapped = pd.cut(
            X['euribor3m'],
            bins=self._euribor_intervals,
            include_lowest=True,
        ).map(self._euribor_bin_rates).astype(float)

        fallback = float(self._euribor_bin_rates.median())
        n_oob    = mapped.isna().sum()
        X['euribor3m_local_rate'] = mapped.fillna(fallback)

        print(f"✅ euribor3m_local_rate: range [{X['euribor3m_local_rate'].min():.4f}, "
              f"{X['euribor3m_local_rate'].max():.4f}]"
              + (f"  ({n_oob} OOB → median fallback)" if n_oob else ""))
        return X

    # ------------------------------------------------------------------
    # dow_month_encoded
    # ------------------------------------------------------------------
    def _fit_dow_month_encoded(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> None:
        """
        Fit Laplace-smoothed P(subscribe | day_of_week x month) on training data.

        Formula per (day, month) cell:
            encoded = (n_cell * P_cell + lambda * P_global) / (n_cell + lambda)
        where lambda = smoothing_factor.
        """
        self._global_mean = float(y_train.mean())

        dm_key = (
            X_train['day_of_week'].astype(str) + '_'
            + X_train['month'].astype(str)
        )
        df_tmp = pd.DataFrame({'key': dm_key.values, 'y': y_train.values})
        stats  = df_tmp.groupby('key')['y'].agg(['mean', 'count'])

        self._dm_smoothed = (
            stats['count'] * stats['mean']
            + self.smoothing_factor * self._global_mean
        ) / (stats['count'] + self.smoothing_factor)

        print(f"\n📐 dow_month_encoded:")
        print(f"   Global mean  : {self._global_mean:.4f}")
        print(f"   Cells fitted : {len(self._dm_smoothed)}")
        print(f"   Encoded range: [{self._dm_smoothed.min():.4f}, "
              f"{self._dm_smoothed.max():.4f}]")

    def _transform_dow_month_encoded(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map day x month combos to smoothed rates.
        Unseen combinations (e.g. rare test combos) fall back to global mean.
        """
        dm_key = (
            X['day_of_week'].astype(str) + '_'
            + X['month'].astype(str)
        )
        n_unseen = dm_key.map(self._dm_smoothed).isna().sum()
        X['dow_month_encoded'] = (
            dm_key.map(self._dm_smoothed).fillna(self._global_mean)
        )

        print(f"✅ dow_month_encoded"
              + (f": {n_unseen} unseen cells → global mean fallback" if n_unseen
                 else ": all cells mapped"))
        return X


# ---------------------------------------------------------------------------
# Convenience function (called by Cell 6)
# ---------------------------------------------------------------------------
def engineer_features(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_train: pd.Series,
    n_bins:          int = _N_BINS_DEFAULT,
    smoothing_factor: int = _SMOOTHING_FACTOR,
) -> Tuple[pd.DataFrame, pd.DataFrame, 'LRFeatureEngineer']:
    """
    Fit on X_train + y_train, transform both splits.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test  : pd.DataFrame
    y_train : pd.Series
        Training labels — required for leakage-free target encodings.
    n_bins : int
        Quantile bins for euribor3m_local_rate. Default 20.
    smoothing_factor : int
        Laplace smoothing weight for dow_month_encoded. Default 100.

    Returns
    -------
    X_train_eng : pd.DataFrame   shape (n_train, 3)
    X_test_eng  : pd.DataFrame   shape (n_test,  3)
    engineer    : LRFeatureEngineer
    """
    engineer = LRFeatureEngineer(
        n_bins=n_bins,
        smoothing_factor=smoothing_factor,
    )

    X_train_eng = engineer.fit_transform(X_train, y_train)
    X_test_eng  = engineer.transform(X_test)

    print("\n" + "="*70)
    print("📋 LR FEATURE ENGINEERING SUMMARY")
    print("="*70)
    print(f"  Features : {LR_FEATURES}")
    print(f"  X_train  : {X_train_eng.shape}  |  X_test: {X_test_eng.shape}")
    print(f"\n  Train feature stats:")
    print(X_train_eng.describe().round(4).to_string())
    print("="*70)

    return X_train_eng, X_test_eng, engineer