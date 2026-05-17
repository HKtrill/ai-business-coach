"""
feature_research/separation.py
================================
Feature-separation analysis for the Glass Cascade pipeline.

Computes statistical divergence between the two target classes for every
feature, storing results in :class:`~feature_research.metrics.SeparationMetrics`
instances and returning a ranked :class:`pandas.DataFrame`.

Provides
--------
- :func:`compute_numeric_separation`    — Cohen's d, KS, point-biserial, MI, probe AUC
- :func:`compute_categorical_separation` — Cramér's V, MI, target-rate range, probe AUC
- :func:`compute_all_separations`       — orchestrates both; returns sorted DataFrame
- :func:`display_feature_rankings`      — formatted console ranking tables
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

from feature_research.config import OUTPUT_DIR
from feature_research.metrics import SeparationMetrics, cohens_d, cramers_v, wilson_ci


# ---------------------------------------------------------------------------
# Per-feature computation
# ---------------------------------------------------------------------------

def compute_numeric_separation(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
) -> SeparationMetrics:
    """Compute separation metrics for a single numeric feature.

    Metrics
    -------
    - **Cohen's d** — standardised mean difference (effect size).
    - **KS statistic** — maximum CDF distance between the two classes.
    - **Point-biserial r** — correlation between feature and binary target.
    - **Mutual information** — computed on a quantile-discretised version of
      the feature to keep MI estimates stable across different scales.
    - **Probe AUC** — single-feature logistic regression AUC (in-sample;
      used only as a relative ranking signal, not an out-of-sample estimate).
    - **Composite score** — unweighted mean of five normalised sub-scores.

    Parameters
    ----------
    df:
        Fully preprocessed DataFrame.
    feature:
        Column name of the numeric feature to analyse.
    target_col:
        Binary 0/1 target column name.

    Returns
    -------
    SeparationMetrics
        Populated metrics container for this feature.
    """
    X = df[feature].values.reshape(-1, 1)
    y = df[target_col].values

    class_0 = df.loc[df[target_col] == 0, feature].values
    class_1 = df.loc[df[target_col] == 1, feature].values

    # Effect size
    d = cohens_d(class_1, class_0)

    # Distribution divergence
    ks_stat, ks_pval = ks_2samp(class_0, class_1)

    # Correlation
    pb_corr, _ = pointbiserialr(y, df[feature].values)

    # Mutual information (requires discretisation)
    n_bins = max(2, min(10, df[feature].nunique()))
    discretizer = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy="quantile"
    )
    X_binned = discretizer.fit_transform(X).ravel()
    mi = mutual_info_score(y, X_binned)

    # Probe AUC
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])

    # Composite: normalise each sub-score to [0, 1] then average
    composite = float(
        np.mean([
            min(abs(d) / 2.0, 1.0),           # |d| > 2 → saturates at 1
            ks_stat,                            # already in [0, 1]
            abs(pb_corr),                       # already in [0, 1]
            mi / np.log(2) if mi > 0 else 0.0, # normalise by 1-bit ceiling
            2.0 * abs(auc - 0.5),              # distance from random
        ])
    )

    return SeparationMetrics(
        feature=feature,
        feature_type="numeric",
        cohens_d=d,
        ks_stat=ks_stat,
        ks_pvalue=ks_pval,
        point_biserial=pb_corr,
        mutual_info=mi,
        auc_probe=auc,
        composite_score=composite,
    )


def compute_categorical_separation(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
) -> SeparationMetrics:
    """Compute separation metrics for a single categorical feature.

    Metrics
    -------
    - **Cramér's V** — χ²-derived association strength in ``[0, 1]``.
    - **Mutual information** — normalised by ``log(n_categories)``.
    - **Target-rate range** — ``(min, max)`` subscription rate across categories.
    - **Probe AUC** — single-feature ordinal logistic regression AUC.
    - **Composite score** — unweighted mean of four normalised sub-scores.

    Parameters
    ----------
    df:
        Fully preprocessed DataFrame.
    feature:
        Column name of the categorical feature to analyse.
    target_col:
        Binary 0/1 target column name.

    Returns
    -------
    SeparationMetrics
        Populated metrics container for this feature.
    """
    X = df[feature].values.reshape(-1, 1)
    y = df[target_col].values

    cv = cramers_v(df[feature], df[target_col])

    mi = mutual_info_score(y, df[feature].values)

    target_rates = df.groupby(feature)[target_col].mean()
    tr_range = (float(target_rates.min()), float(target_rates.max()))

    n_cats = int(df[feature].nunique())

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X, y)
    auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])

    mi_norm = mi / np.log(n_cats) if n_cats > 1 else 0.0
    composite = float(
        np.mean([
            cv,
            mi_norm,
            tr_range[1] - tr_range[0],  # target-rate spread, already in [0, 1]
            2.0 * abs(auc - 0.5),
        ])
    )

    return SeparationMetrics(
        feature=feature,
        feature_type="categorical",
        cramers_v=cv,
        mutual_info=mi,
        auc_probe=auc,
        target_rate_range=tr_range,
        n_categories=n_cats,
        composite_score=composite,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_all_separations(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target_col: str,
) -> pd.DataFrame:
    """Compute separation metrics for every feature and return a ranked table.

    Iterates over all numeric then all categorical features, calling the
    appropriate compute function, and assembles results into a single
    :class:`pandas.DataFrame` sorted by ``composite_score`` descending.

    Parameters
    ----------
    df:
        Fully preprocessed DataFrame.
    numeric_features:
        List of numeric column names (from :func:`~feature_research.validation.classify_features`).
    categorical_features:
        List of categorical column names.
    target_col:
        Binary 0/1 target column name.

    Returns
    -------
    pd.DataFrame
        One row per feature, all metric columns present, sorted by
        ``composite_score`` descending with a reset integer index.
    """
    print("\n" + "=" * 80)
    print("📊 COMPUTING SEPARATION METRICS")
    print("=" * 80)

    all_metrics = []

    print(f"\n📈 Numeric features ({len(numeric_features)}):")
    for i, feat in enumerate(numeric_features, 1):
        print(f"  [{i:2d}/{len(numeric_features)}] {feat}...", end="\r")
        all_metrics.append(compute_numeric_separation(df, feat, target_col).to_dict())
    print(f"\n✅ Numeric features complete")

    print(f"\n📊 Categorical features ({len(categorical_features)}):")
    for i, feat in enumerate(categorical_features, 1):
        print(f"  [{i:2d}/{len(categorical_features)}] {feat}...", end="\r")
        all_metrics.append(compute_categorical_separation(df, feat, target_col).to_dict())
    print(f"\n✅ Categorical features complete")

    df_metrics = (
        pd.DataFrame(all_metrics)
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n✅ Separation metrics computed for {len(df_metrics)} features")
    return df_metrics


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_feature_rankings(
    df_metrics: pd.DataFrame,
    top_n: int = 20,
    save_csv: bool = True,
) -> None:
    """Print formatted ranking tables and optionally persist to CSV.

    Outputs three sections:
    1. **Overall** top-``top_n`` features by composite score.
    2. **Numeric-only** top-10 with Cohen's d / KS / AUC columns.
    3. **Categorical-only** top-10 with Cramér's V / n_categories / AUC.

    Parameters
    ----------
    df_metrics:
        Output of :func:`compute_all_separations`.
    top_n:
        Number of features shown in the overall ranking section.
    save_csv:
        When ``True`` (default), writes the full metrics table to
        ``research_logs/feature_rankings.csv``.
    """
    SEP = "=" * 80
    LINE = "-" * 80

    print(f"\n{SEP}")
    print(f"🏆 FEATURE RANKINGS  (Top {top_n})")
    print(SEP)

    # -- Overall --------------------------------------------------------------
    print("\n📊 OVERALL RANKING (by composite score)")
    print(LINE)

    for rank, (_, row) in enumerate(df_metrics.head(top_n).iterrows(), start=1):
        print(f"\n{rank:2d}. {row['feature']:20s} ({row['type']})")
        print(
            f"    Composite : {row['composite_score']:.4f}  |  "
            f"AUC : {row['auc_probe']:.4f}  |  "
            f"MI : {row['mutual_info']:.4f}"
        )
        if row["type"] == "numeric":
            print(
                f"    Cohen's d : {row['cohens_d']:+.3f}  |  "
                f"KS : {row['ks_stat']:.3f}  |  "
                f"r_pb : {row['point_biserial']:+.3f}"
            )
        else:
            lo, hi = row["target_rate_range"]
            print(
                f"    Cramér's V : {row['cramers_v']:.3f}  |  "
                f"Categories : {row['n_categories']}  |  "
                f"TR range : [{lo:.3f}, {hi:.3f}]"
            )

    # -- Numeric --------------------------------------------------------------
    print("\n\n📈 TOP NUMERIC FEATURES")
    print(LINE)
    df_num = df_metrics[df_metrics["type"] == "numeric"].head(10)
    for rank, (_, row) in enumerate(df_num.iterrows(), start=1):
        print(
            f"{rank:2d}. {row['feature']:20s}  |  "
            f"d={row['cohens_d']:+.3f}  |  "
            f"KS={row['ks_stat']:.3f}  |  "
            f"AUC={row['auc_probe']:.3f}"
        )

    # -- Categorical ----------------------------------------------------------
    print("\n\n📊 TOP CATEGORICAL FEATURES")
    print(LINE)
    df_cat = df_metrics[df_metrics["type"] == "categorical"].head(10)
    if df_cat.empty:
        print("(No categorical features)")
    else:
        for rank, (_, row) in enumerate(df_cat.iterrows(), start=1):
            print(
                f"{rank:2d}. {row['feature']:20s}  |  "
                f"V={row['cramers_v']:.3f}  |  "
                f"n_cat={row['n_categories']}  |  "
                f"AUC={row['auc_probe']:.3f}"
            )

    # -- Persist --------------------------------------------------------------
    if save_csv:
        csv_path = OUTPUT_DIR / "feature_rankings.csv"
        df_metrics.to_csv(csv_path, index=False)
        print(f"\n💾 Saved rankings to: {csv_path}")