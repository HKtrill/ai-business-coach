"""
model_training.lr.diagnostics
==============================
Post-training diagnostics for the LR gate stage.

Reuses feature_research modules for class-separation and visualization;
adds LR-specific checks (VIF, correlation, coefficient analysis) that
don't belong in the general feature_research package.

Public API
----------
compute_vif(X)                      → pd.DataFrame
compute_correlation(X, threshold)   → (corr_matrix, high_pairs)
lr_diagnostics(lr_result, df, ...)  → dict   (orchestrator)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

from feature_research.separation import compute_numeric_separation
from feature_research.visualization import plot_numeric_feature
from feature_research.model_training.lr.trainer import LRResult


# ── Constants ─────────────────────────────────────────────────────────────────
_HIGH_CORR_THRESHOLD: float = 0.70
_CRITICAL_CORR_THRESHOLD: float = 0.90
_HIGH_VIF_THRESHOLD: float = 10.0
_MOD_VIF_THRESHOLD: float = 5.0
_STRONG_SEP_THRESHOLD: float = 0.20
_MOD_SEP_THRESHOLD: float = 0.05
_ACTIVE_COEF_THRESHOLD: float = 0.15
_WEAK_COEF_THRESHOLD: float = 0.10


# ── Public helpers (independently testable) ───────────────────────────────────

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each feature in X.

    Scales X with StandardScaler before computing VIF — necessary for
    features on different scales to produce meaningful estimates.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (unscaled, no NaN/inf).

    Returns
    -------
    pd.DataFrame
        Columns: feature, VIF. Sorted by VIF descending.
    """
    X_scaled = pd.DataFrame(
        StandardScaler().fit_transform(X),
        columns=X.columns,
    )
    records = []
    for i, col in enumerate(X_scaled.columns):
        try:
            vif_val = variance_inflation_factor(X_scaled.values, i)
        except Exception:
            vif_val = float("inf")
        records.append({"feature": col, "VIF": vif_val})

    return pd.DataFrame(records).sort_values("VIF", ascending=False).reset_index(drop=True)


def compute_correlation(
    X: pd.DataFrame,
    threshold: float = _HIGH_CORR_THRESHOLD,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Compute the Pearson correlation matrix and extract high-correlation pairs.

    Parameters
    ----------
    X         : Feature matrix (unscaled, no NaN/inf).
    threshold : |r| above which a pair is flagged (default 0.70).

    Returns
    -------
    corr_matrix : pd.DataFrame — full correlation matrix.
    high_pairs  : list[dict]  — dicts with Feature1, Feature2, Correlation.
                  Empty list when no pairs exceed threshold.
    """
    corr_matrix = X.corr()
    high_pairs: list[dict] = []

    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > threshold:
                high_pairs.append({
                    "Feature1": cols[i],
                    "Feature2": cols[j],
                    "Correlation": r,
                })

    high_pairs.sort(key=lambda x: abs(x["Correlation"]), reverse=True)
    return corr_matrix, high_pairs


# ── Private display helpers ───────────────────────────────────────────────────

def _print_correlation(high_pairs: list[dict], threshold: float) -> None:
    if not high_pairs:
        print(f"\n✅ No high-correlation pairs (|r| > {threshold}) — multicollinearity resolved!")
        return
    print(f"\n📊 High-Correlation Pairs (|r| > {threshold}): {len(high_pairs)} found")
    for p in high_pairs:
        flag = "🔴" if abs(p["Correlation"]) > _CRITICAL_CORR_THRESHOLD else "🟡"
        print(
            f"   {flag} {p['Feature1']:35s} ↔ {p['Feature2']:35s} "
            f"r={p['Correlation']:+.3f}"
        )


def _print_vif(vif_df: pd.DataFrame) -> int:
    """Print VIF table; returns count of HIGH VIF features."""
    high_count = 0
    for _, row in vif_df.iterrows():
        if row["VIF"] > _HIGH_VIF_THRESHOLD:
            flag, high_count = "❌ HIGH", high_count + 1
        elif row["VIF"] > _MOD_VIF_THRESHOLD:
            flag = "⚠️  MOD"
        else:
            flag = "✅ LOW"
        print(f"   {row['feature']:40s} VIF={row['VIF']:>8.2f}  {flag}")
    if high_count == 0:
        print("\n✅ All VIF < 10 — multicollinearity resolved!")
    else:
        print(f"\n⚠️  {high_count} feature(s) with VIF > 10")
    return high_count


def _print_separation(sep_records: list[dict]) -> tuple[list[str], pd.DataFrame]:
    """Print class separation table; returns (weak_feature_names, sep_df)."""
    sep_df = (
        pd.DataFrame(sep_records)
        .assign(abs_r=lambda d: d["r_pb"].abs())
        .sort_values("abs_r", ascending=False)
        .reset_index(drop=True)
    )
    for _, row in sep_df.iterrows():
        if row["abs_r"] >= _STRONG_SEP_THRESHOLD:
            flag = "✅ STRONG"
        elif row["abs_r"] >= _MOD_SEP_THRESHOLD:
            flag = "⚠️  MOD"
        else:
            flag = "❌ WEAK"
        print(
            f"   {row['feature']:40s} "
            f"d={row['cohens_d']:+.3f}  "
            f"KS={row['ks_stat']:.3f}  "
            f"r={row['r_pb']:+.3f}  {flag}"
        )
    weak = sep_df.loc[sep_df["abs_r"] < _MOD_SEP_THRESHOLD, "feature"].tolist()
    if weak:
        print(f"\n⚠️  Weak separators (|r| < {_MOD_SEP_THRESHOLD}): {weak}")
    return weak, sep_df


def _print_coefficients(feature_importance: pd.DataFrame) -> list[str]:
    """Print coefficient table; returns dead feature names."""
    dead: list[str] = []
    for _, row in feature_importance.iterrows():
        if row["abs_coef"] >= _ACTIVE_COEF_THRESHOLD:
            flag = "🔥 ACTIVE"
        elif row["abs_coef"] >= _WEAK_COEF_THRESHOLD:
            flag = "⚪ WEAK"
        else:
            flag = "💀 DEAD"
            dead.append(row["feature"])
        bar = "█" * int(row["abs_coef"] * 5)
        print(
            f"   {row['feature']:40s} {row['coefficient']:+7.4f}  {bar}  {flag}"
        )
    if dead:
        print(f"\n💀 Near-zero coefficients (candidates to drop): {dead}")
    return dead


def _plot_heatmap(corr_matrix: pd.DataFrame, n_features: int) -> None:
    fig, ax = plt.subplots(figsize=(max(6, n_features * 1.5), max(5, n_features * 1.2)))
    # Lower-triangle only — mask the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(
        f"LR Features — Correlation Matrix ({n_features} features)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.show()


# ── Orchestrator ──────────────────────────────────────────────────────────────

def lr_diagnostics(
    lr_result: LRResult,
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    corr_threshold: float = _HIGH_CORR_THRESHOLD,
    top_violin: int = 3,
    show_plots: bool = True,
) -> dict:
    """
    Full LR feature diagnostic suite.

    Runs five checks and optionally renders visualizations. Designed to be
    the only call a notebook cell needs to make.

    Sections
    --------
    1. Correlation analysis          — flags multicollinear pairs
    2. VIF analysis                  — quantifies variance inflation
    3. Class separation              — reuses compute_numeric_separation()
    4. Model coefficients            — displays lr_result.feature_importance
    5. Summary & recommendations     — assembles findings

    Visualizations (when show_plots=True)
    ------
    - Correlation heatmap
    - Violin plots for top-N separators (reuses plot_numeric_feature())

    Parameters
    ----------
    lr_result     : Output of train_lr() — provides pipe + feature_importance.
    df            : df_engineered (must contain features + target_col).
    features      : LR_FEATURES list.
    target_col    : TARGET_COL string.
    corr_threshold: |r| threshold for flagging pairs (default 0.70).
    top_violin    : Number of top separators to render violin plots for.
    show_plots    : Set False to suppress all matplotlib output.

    Returns
    -------
    dict with keys: corr_matrix, high_pairs, vif_df, sep_df,
                    high_vif_count, weak_sep_features, dead_coef_features.
    """
    SEP = "=" * 80
    LINE = "-" * 80

    X = df[features].copy()

    print(f"\n{SEP}")
    print("🔬 LR FEATURE DIAGNOSTICS")
    print(f"   Feature set: {len(features)} features → {features}")
    print(SEP)

    # ── 1. Correlation ────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("1️⃣  CORRELATION ANALYSIS")
    print(LINE)
    corr_matrix, high_pairs = compute_correlation(X, corr_threshold)
    _print_correlation(high_pairs, corr_threshold)

    # ── 2. VIF ────────────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("2️⃣  VARIANCE INFLATION FACTOR (VIF)")
    print(LINE)
    vif_df = compute_vif(X)
    high_vif_count = _print_vif(vif_df)

    # ── 3. Class separation — reuses feature_research.separation ──────────────
    print(f"\n{LINE}")
    print("3️⃣  CLASS SEPARATION")
    print(LINE)
    sep_records = []
    for feat in features:
        m = compute_numeric_separation(df, feat, target_col)
        sep_records.append({
            "feature":   m.feature,
            "cohens_d":  m.cohens_d,
            "ks_stat":   m.ks_stat,
            "r_pb":      m.point_biserial,
            "auc_probe": m.auc_probe,
        })
    weak_sep, sep_df = _print_separation(sep_records)

    # ── 4. Coefficients ───────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("4️⃣  MODEL COEFFICIENTS")
    print(LINE)
    dead_coefs = _print_coefficients(lr_result.feature_importance)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n{LINE}")
    print("5️⃣  SUMMARY")
    print(LINE)
    print(f"\n   Features:          {len(features)}")
    print(f"   High VIF (>10):    {high_vif_count}")
    print(f"   High corr (>{corr_threshold}): {len(high_pairs)} pairs")
    print(f"   Dead coefs (<0.1): {len(dead_coefs)}")
    print(f"   Weak sep (<0.05):  {len(weak_sep)}")

    if high_vif_count == 0 and not dead_coefs and not weak_sep:
        print("\n   ✅ Feature set is clean — ready for cascade hand-off.")
    else:
        if high_vif_count > 0:
            print(f"\n   ⚠️  Multicollinearity present — consider dropping one from correlated pairs.")
        if dead_coefs:
            print(f"\n   ⚠️  Dead coefficients: {dead_coefs}")
        if weak_sep:
            print(f"\n   ⚠️  Weak separators: {weak_sep}")

    # ── Visualizations ────────────────────────────────────────────────────────
    if show_plots:
        print(f"\n📊 Correlation heatmap...")
        _plot_heatmap(corr_matrix, len(features))

        top_feats = sep_df.head(top_violin)["feature"].tolist()
        if top_feats:
            print(f"\n📊 Violin plots — top {len(top_feats)} separator(s)...")
            for feat in top_feats:
                # Reuses feature_research.visualization — save_path=None → inline
                plot_numeric_feature(df, feat, target_col, save_path=None)

    print(f"\n{SEP}")
    print("✅ LR diagnostics complete")
    print(SEP)

    return {
        "corr_matrix":       corr_matrix,
        "high_pairs":        high_pairs,
        "vif_df":            vif_df,
        "sep_df":            sep_df,
        "high_vif_count":    high_vif_count,
        "weak_sep_features": weak_sep,
        "dead_coef_features": dead_coefs,
    }