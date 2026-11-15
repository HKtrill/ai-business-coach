"""
Feature Engineering for Logistic Regression Stage
=================================================

Produces the three validated LR Stage 1 features:
    cellular_downturn             — cellular contact during an economic downturn
    subscribe_rate_by_euribor_bin — empirical P(subscribe) per euribor quantile bin
    subscribe_rate_by_dow_month   — smoothed P(subscribe | day_of_week x month)

PR 39 (Commit 2, Step 2B) renamed them; values and definitions are unchanged.
Before it they were ``cellular_crisis``, ``euribor3m_local_rate`` and
``dow_month_encoded`` (``STAGE1_FEATURE_RENAMES``), the names every artifact
saved earlier carries (feature contract 1.0.0). Stages 2 and 3 compute their
own features and keep those names (PR 39 decision D2 (a)).

Replaces the original EconomicFeatureEngineer (PCA / composite approach)
with the validated feature set ported from the research notebook.

Leakage fixes vs research notebook:
    subscribe_rate_by_euribor_bin — bin rates fitted on X_train only;
                                    test mapped via stored IntervalIndex.
    subscribe_rate_by_dow_month   — target encoding fitted on X_train + y_train;
                                    unseen day-month cells fall back to global mean.

Author: Glass Pipeline Team
Date: 2026-05
"""

import hashlib
import json
import pandas as pd
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Optional, Tuple
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

LR_FEATURES = ['cellular_downturn', 'subscribe_rate_by_euribor_bin', 'subscribe_rate_by_dow_month']

#: The LR features learned from the training LABELS. Fitted once on the whole
#: training split they are fine for scoring the test split, but inside any
#: cross-validation they must be refit per fold (audit F3) — see
#: ``shared.stage1.crossfit``.
TARGET_ENCODED_FEATURES = ('subscribe_rate_by_euribor_bin', 'subscribe_rate_by_dow_month')

# ---------------------------------------------------------------------------
# Feature contracts (PR 39, Commit 2, Step 2B)
# ---------------------------------------------------------------------------
#: Historical name -> final name, in ``LR_FEATURES`` order. Names only: every
#: value and definition is unchanged.
STAGE1_FEATURE_RENAMES = MappingProxyType({
    'cellular_crisis':      'cellular_downturn',
    'euribor3m_local_rate': 'subscribe_rate_by_euribor_bin',
    'dow_month_encoded':    'subscribe_rate_by_dow_month',
})

#: The names, in the same order, that every Stage 1 artifact saved before the
#: rename carries. They are read as they are, never translated.
HISTORICAL_LR_FEATURES = tuple(STAGE1_FEATURE_RENAMES)
HISTORICAL_TARGET_ENCODED_FEATURES = ('euribor3m_local_rate', 'dow_month_encoded')


def _feature_contract(version: str, features: Sequence[str]) -> Mapping:
    names = [str(f) for f in features]
    return MappingProxyType({
        "id": "stage1-features",
        "version": version,
        "features": tuple(names),
        "fingerprint": hashlib.sha256(json.dumps(names).encode("utf-8")).hexdigest()[:16],
    })


#: The current Stage 1 feature contract: names and order. Saved in every
#: Stage 1 artifact (LR and MLP artifact version 3.0).
STAGE1_FEATURE_CONTRACT = _feature_contract("2.0.0", LR_FEATURES)
#: What every earlier artifact (version 2.0) holds, readable for verification.
HISTORICAL_STAGE1_FEATURE_CONTRACT = _feature_contract("1.0.0", HISTORICAL_LR_FEATURES)


def stage1_feature_contract(feature_names) -> Optional[Mapping]:
    """
    The Stage 1 feature contract whose ordered names these are, or ``None``.

    Only the two registered lists match, each in its own order: the current
    ``LR_FEATURES`` (2.0.0) and the historical names (1.0.0).
    """
    if feature_names is None:
        return None
    names = tuple(str(f) for f in feature_names)
    for contract in (STAGE1_FEATURE_CONTRACT, HISTORICAL_STAGE1_FEATURE_CONTRACT):
        if names == contract["features"]:
            return contract
    return None


def feature_contract_record(contract: Mapping) -> dict:
    """A contract as the plain dict an artifact stores (pickle- and JSON-safe)."""
    return {"id": contract["id"], "version": contract["version"],
            "features": list(contract["features"]), "fingerprint": contract["fingerprint"]}


def current_feature_contract_record(feature_names, where: str = "Stage 1 artifact") -> dict:
    """
    The record a new Stage 1 artifact saves: the current contract, which
    ``feature_names`` must be exactly (same names, same order).

    Raises
    ------
    ValueError
        For any other list. The historical names are readable, never written:
        a stage fitted on them predates the rename and must be refit.
    """
    contract = stage1_feature_contract(feature_names)
    if contract is not STAGE1_FEATURE_CONTRACT:
        found = "historical contract 1.0.0" if contract is not None else "no registered contract"
        raise ValueError(
            f"{where}: feature_names {list(feature_names) if feature_names is not None else None} "
            f"match {found}; a new Stage 1 artifact must carry Stage 1 feature contract "
            f"{STAGE1_FEATURE_CONTRACT['version']} {list(LR_FEATURES)} (in this order)."
        )
    return feature_contract_record(contract)


def artifact_feature_contract(artifact) -> Mapping:
    """
    The registered Stage 1 feature contract a saved artifact was written with.

    Reads the artifact as saved, never translating its names: a pre-rename
    artifact (LR / MLP artifact version 2.0) is identified from its
    ``feature_names`` as contract 1.0.0; a current one (3.0) carries a
    ``feature_contract`` record, which must agree with its ``feature_names``.
    A bare ``StageOutput`` dict is identified from its own ``feature_names``.

    Raises
    ------
    ValueError
        If the names match no registered contract, or the recorded contract
        disagrees with them.
    """
    if not isinstance(artifact, Mapping):
        raise ValueError(f"expected an artifact dict, got {type(artifact).__name__}")
    names = artifact.get("feature_names")
    so = artifact.get("stage_output")
    if names is None and isinstance(so, Mapping):
        names = so.get("feature_names")
    contract = stage1_feature_contract(names)
    if contract is None:
        raise ValueError(f"feature_names {names!r} match no registered Stage 1 feature contract")
    if isinstance(so, Mapping) and so.get("feature_names") is not None and \
            stage1_feature_contract(so["feature_names"]) is not contract:
        raise ValueError("artifact feature_names and stage_output.feature_names disagree")
    recorded = artifact.get("feature_contract")
    if recorded is not None and dict(recorded) != feature_contract_record(contract):
        raise ValueError(
            f"recorded feature_contract {dict(recorded)!r} does not match the artifact's "
            f"feature_names (contract {contract['version']})")
    return contract


if [STAGE1_FEATURE_RENAMES[f] for f in HISTORICAL_LR_FEATURES] != LR_FEATURES or \
        tuple(STAGE1_FEATURE_RENAMES[f] for f in HISTORICAL_TARGET_ENCODED_FEATURES) != \
        TARGET_ENCODED_FEATURES:                                  # pragma: no cover
    raise AssertionError("STAGE1_FEATURE_RENAMES does not map the historical names in order")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class LRFeatureEngineer:
    """
    Feature engineer for LR Stage 1.

    Produces exactly three features:
        cellular_downturn             — no target used; safe on any split.
        subscribe_rate_by_euribor_bin — fitted on train only; leakage-free on test.
        subscribe_rate_by_dow_month   — fitted on train only; leakage-free on test.

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

        self._fit_subscribe_rate_by_euribor_bin(X_train, y_train)
        self._fit_subscribe_rate_by_dow_month(X_train, y_train)

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
        X_eng = self._add_cellular_downturn(X_eng)
        X_eng = self._transform_subscribe_rate_by_euribor_bin(X_eng)
        X_eng = self._transform_subscribe_rate_by_dow_month(X_eng)

        return X_eng[LR_FEATURES]

    def fit_transform(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> pd.DataFrame:
        """Fit on X_train + y_train, then transform X_train."""
        return self.fit(X_train, y_train).transform(X_train)

    # ------------------------------------------------------------------
    # cellular_downturn (historically cellular_crisis)
    # ------------------------------------------------------------------
    def _add_cellular_downturn(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Cellular contact during an economic downturn.

        crisis_score = (euribor3m < 1.5)*3
                     + (emp.var.rate < -1.0)*2
                     + (nr.employed < 5100)*1

        cellular_downturn = (contact == 0) AND (crisis_score >= 3)

        No target used — safe to compute directly on any split.
        """
        crisis_score = (
            (X['euribor3m']    < _CRISIS_EURIBOR_THRESH).astype(int) * 3
            + (X['emp.var.rate'] < _CRISIS_EMP_VAR_THRESH).astype(int) * 2
            + (X['nr.employed']  < _CRISIS_NR_EMP_THRESH).astype(int) * 1
        )
        X['cellular_downturn'] = (
            (X['contact'] == 0) & (crisis_score >= _CRISIS_SCORE_CUTOFF)
        ).astype(int)

        print(f"\n✅ cellular_downturn: {X['cellular_downturn'].sum():,} positives "
              f"({X['cellular_downturn'].mean():.1%})")
        return X

    # ------------------------------------------------------------------
    # subscribe_rate_by_euribor_bin (historically euribor3m_local_rate)
    # ------------------------------------------------------------------
    def _fit_subscribe_rate_by_euribor_bin(
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

        print(f"\n📐 subscribe_rate_by_euribor_bin:")
        print(f"   Bins fitted : {len(self._euribor_intervals)}")
        print(f"   Rate range  : [{self._euribor_bin_rates.min():.4f}, "
              f"{self._euribor_bin_rates.max():.4f}]")

    def _transform_subscribe_rate_by_euribor_bin(self, X: pd.DataFrame) -> pd.DataFrame:
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
        X['subscribe_rate_by_euribor_bin'] = mapped.fillna(fallback)

        print(f"✅ subscribe_rate_by_euribor_bin: range "
              f"[{X['subscribe_rate_by_euribor_bin'].min():.4f}, "
              f"{X['subscribe_rate_by_euribor_bin'].max():.4f}]"
              + (f"  ({n_oob} OOB → median fallback)" if n_oob else ""))
        return X

    # ------------------------------------------------------------------
    # subscribe_rate_by_dow_month (historically dow_month_encoded)
    # ------------------------------------------------------------------
    def _fit_subscribe_rate_by_dow_month(
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

        print(f"\n📐 subscribe_rate_by_dow_month:")
        print(f"   Global mean  : {self._global_mean:.4f}")
        print(f"   Cells fitted : {len(self._dm_smoothed)}")
        print(f"   Encoded range: [{self._dm_smoothed.min():.4f}, "
              f"{self._dm_smoothed.max():.4f}]")

    def _transform_subscribe_rate_by_dow_month(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Map day x month combos to smoothed rates.
        Unseen combinations (e.g. rare test combos) fall back to global mean.
        """
        dm_key = (
            X['day_of_week'].astype(str) + '_'
            + X['month'].astype(str)
        )
        n_unseen = dm_key.map(self._dm_smoothed).isna().sum()
        X['subscribe_rate_by_dow_month'] = (
            dm_key.map(self._dm_smoothed).fillna(self._global_mean)
        )

        print(f"✅ subscribe_rate_by_dow_month"
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
        Quantile bins for subscribe_rate_by_euribor_bin. Default 20.
    smoothing_factor : int
        Laplace smoothing weight for subscribe_rate_by_dow_month. Default 100.

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