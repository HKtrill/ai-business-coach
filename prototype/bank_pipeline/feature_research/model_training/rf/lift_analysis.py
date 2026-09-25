"""
model_training.rf.lift_analysis
================================
Lift-driven feature analysis for binary binning strategy discovery.

Used to identify natural breakpoints in continuous RF features before
encoding them as binary bins for GLASS-BRW input. Thresholds discovered
here are locked into binning.py — this module is diagnostic/reproducible,
not part of the training hot path.

Public API
----------
compute_lift(df, features, target_col)          → dict[str, pd.DataFrame]
compute_gini_importance(rf_pipe, X)             → pd.DataFrame
compute_permutation_importance(rf_pipe, X, y)   → pd.DataFrame
rf_lift_analysis(df, features, target_col, ...) → dict  (orchestrator)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance as _sklearn_perm_imp
from sklearn.pipeline import Pipeline

from feature_research.config import FIG_DIR


# ── Public helpers ────────────────────────────────────────────────────────────

def compute_gini_importance(
    rf_pipe: Pipeline,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract Gini (impurity) feature importances from a fitted RF pipeline.

    Parameters
    ----------
    rf_pipe : Fitted Pipeline containing a 'clf' RandomForestClassifier step.
    X       : Feature matrix used for training (column names preserved).

    Returns
    -------
    pd.DataFrame — columns: feature, gini. Sorted by gini descending.
    """
    clf = rf_pipe.named_steps["clf"]
    return (
        pd.DataFrame({
            "feature": X.columns,
            "gini": clf.feature_importances_,
        })
        .sort_values("gini", ascending=False)
        .reset_index(drop=True)
    )


def compute_permutation_importance(
    rf_pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
    holdout_frac: float = 0.2,
) -> pd.DataFrame:
    """
    Compute permutation importance on a held-out slice of the data.

    Uses the last `holdout_frac` of rows as the evaluation set — avoids
    re-running expensive CV while still giving an out-of-bag-style estimate.

    Parameters
    ----------
    rf_pipe      : Fitted Pipeline.
    X, y         : Full feature matrix and target (holdout sliced internally).
    n_repeats    : Permutation repeats (default 10, ~30s for 29 features).
    holdout_frac : Fraction of data held out for evaluation (default 0.20).

    Returns
    -------
    pd.DataFrame — columns: feature, perm (mean AUC drop), std. Sorted desc.
    """
    split = int(len(X) * (1 - holdout_frac))
    X_eval = X.iloc[split:]
    y_eval = y.iloc[split:]

    result = _sklearn_perm_imp(
        rf_pipe, X_eval, y_eval,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="roc_auc",
    )
    return (
        pd.DataFrame({
            "feature": X.columns,
            "perm": result.importances_mean,
            "std": result.importances_std,
        })
        .sort_values("perm", ascending=False)
        .reset_index(drop=True)
    )


def compute_lift(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_deciles: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Compute conversion-rate lift by decile for each continuous feature.

    Features with fewer than `n_deciles` unique values are binned by value
    rather than quantile. Features that cannot be binned are skipped silently.

    Parameters
    ----------
    df         : DataFrame containing features + target_col.
    features   : List of continuous feature names to analyse.
    target_col : Binary 0/1 target column.
    n_deciles  : Number of quantile bins (default 10).

    Returns
    -------
    dict mapping feature name → pd.DataFrame with columns:
        bin, count, conversions, conv_rate, lift.
    """
    overall_rate = df[target_col].mean()
    results: dict[str, pd.DataFrame] = {}

    for feat in features:
        tmp = pd.DataFrame({"val": df[feat], "y": df[target_col]})
        n_unique = tmp["val"].nunique()
        try:
            if n_unique < n_deciles:
                tmp["bin"] = tmp["val"]
            else:
                tmp["bin"] = pd.qcut(tmp["val"], q=n_deciles, duplicates="drop")
        except Exception:
            continue

        lift_df = (
            tmp.groupby("bin", observed=True)
            .agg(count=("y", "count"), conversions=("y", "sum"), conv_rate=("y", "mean"))
            .reset_index()
        )
        lift_df["lift"] = lift_df["conv_rate"] / overall_rate
        lift_df["feature"] = feat
        results[feat] = lift_df

    return results


# ── Private display / plot helpers ────────────────────────────────────────────

def _print_importance(gini_df: pd.DataFrame, perm_df: Optional[pd.DataFrame], top_n: int = 20) -> None:
    print(f"\n{'─' * 80}")
    print("📊 FEATURE IMPORTANCE")
    print(f"{'─' * 80}")

    print(f"\n   🔝 Gini Importance (Top {top_n}):")
    print(f"   {'Rank':<6} {'Feature':<40} {'Gini':>8}")
    print(f"   {'-' * 57}")
    for i, row in gini_df.head(top_n).iterrows():
        bar = "█" * int(row["gini"] * 200)
        print(f"   {i + 1:<6} {row['feature']:<40} {row['gini']:>8.4f}  {bar}")

    if perm_df is not None:
        print(f"\n   {'Rank':<6} {'Feature':<40} {'Perm AUC Drop':>14} {'Std':>8}")
        print(f"   {'-' * 70}")
        for i, row in perm_df.head(top_n).iterrows():
            flag = "🔥" if row["perm"] > 0.01 else "⚪" if row["perm"] > 0 else "💀"
            print(f"   {i + 1:<6} {row['feature']:<40} {row['perm']:>14.4f} {row['std']:>8.4f}  {flag}")


def _print_lift(lift_results: dict[str, pd.DataFrame], overall_rate: float) -> None:
    print(f"\n{'─' * 80}")
    print("📈 CONTINUOUS FEATURE LIFT BY DECILE")
    print(f"{'─' * 80}")
    for feat, ldf in lift_results.items():
        print(f"\n   🔹 {feat}")
        print(f"   {'Bin':<35} {'Count':>7} {'ConvRate':>10} {'Lift':>7}")
        print(f"   {'-' * 62}")
        for _, row in ldf.iterrows():
            flag = "🔥" if row["lift"] > 1.5 else "🟢" if row["lift"] > 1.0 else "🔴"
            print(
                f"   {str(row['bin']):<35} {row['count']:>7,} "
                f"{row['conv_rate']:>10.4f} {row['lift']:>7.2f}x  {flag}"
            )


def _print_recommendations(lift_results: dict[str, pd.DataFrame], overall_rate: float) -> None:
    print(f"\n{'=' * 80}")
    print("🎯 BINNING RECOMMENDATIONS")
    print(f"{'=' * 80}")
    print(f"\n   Overall conversion rate: {overall_rate:.4f}\n")
    for feat, ldf in lift_results.items():
        high = ldf[ldf["lift"] > 1.5]
        low = ldf[ldf["lift"] < 0.7]
        print(f"   {feat}:")
        if len(high):
            print(f"      🔥 HIGH LIFT bins (>1.5x): {list(high['bin'].astype(str))}")
        if len(low):
            print(f"      🔴 LOW LIFT bins  (<0.7x): {list(low['bin'].astype(str))}")
        if not len(high) and not len(low):
            print(f"      ⚪ Flat lift — low binning value")


def _plot_importance(
    gini_df: pd.DataFrame,
    perm_df: Optional[pd.DataFrame],
    top_n: int = 8,
) -> None:
    n_panels = 2 if perm_df is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(9 * n_panels, 8))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle(
        "RF Feature Importance — Binning Strategy",
        fontsize=13, fontweight="bold",
    )
    top_gini = gini_df.head(top_n)
    axes[0].barh(top_gini["feature"], top_gini["gini"], color="forestgreen")
    axes[0].set_xlabel("Gini Importance")
    axes[0].set_title("Gini Importance", fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)

    if perm_df is not None:
        top_perm = perm_df.head(top_n)
        axes[1].barh(
            top_perm["feature"], top_perm["perm"],
            xerr=top_perm["std"], color="steelblue",
        )
        axes[1].set_xlabel("Permutation Importance (AUC drop)")
        axes[1].set_title("Permutation Importance", fontweight="bold")
        axes[1].invert_yaxis()
        axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    save_path = FIG_DIR / "rf_feature_importance.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n   ✅ Importance plot saved → {save_path}")


def _plot_lift_curves(
    lift_results: dict[str, pd.DataFrame],
    overall_rate: float,
) -> None:
    if not lift_results:
        return

    n_feats = len(lift_results)
    n_cols = 3
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten() if n_feats > 1 else [axes]
    fig.suptitle(
        "Conversion Rate by Decile — Lift Analysis\n"
        "(Green=High Lift >1.5x, Red=Low Lift <0.7x)",
        fontsize=13, fontweight="bold",
    )

    for idx, (feat, ldf) in enumerate(lift_results.items()):
        ax = axes[idx]
        bars = ax.bar(range(len(ldf)), ldf["conv_rate"], color="steelblue", alpha=0.7)
        for bar, lift in zip(bars, ldf["lift"]):
            bar.set_color("darkgreen" if lift > 1.5 else "darkred" if lift < 0.7 else "steelblue")

        ax.axhline(overall_rate, color="red", linestyle="--", linewidth=2,
                   label=f"Overall ({overall_rate:.3f})")
        ax.set_xticks(range(len(ldf)))
        ax.set_xticklabels([str(b)[:15] for b in ldf["bin"]], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Conversion Rate")
        ax.set_title(
            f"{feat}\n(Max: {ldf['lift'].max():.2f}x, Min: {ldf['lift'].min():.2f}x)",
            fontweight="bold", fontsize=9,
        )
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(lift_results), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    save_path = FIG_DIR / "rf_lift_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✅ Lift curves saved → {save_path}")


# ── Orchestrator ──────────────────────────────────────────────────────────────

def rf_lift_analysis(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    rf_pipe: Optional[Pipeline] = None,
    n_deciles: int = 10,
    top_n_importance: int = 20,
) -> dict:
    """
    Full lift analysis for binary binning strategy discovery.

    Computes decile lift for all continuous features, and optionally
    Gini + permutation importances if a fitted RF pipe is provided.
    Saves importance and lift-curve figures to FIG_DIR.

    Parameters
    ----------
    df           : df_engineered (must contain features + target_col).
    features     : RF_FEATURES — continuous feature list from 10G.
    target_col   : TARGET_COL string.
    rf_pipe      : Optional fitted Pipeline — enables importance analysis.
                   Pass None to run lift-only (no trained model required).
    n_deciles    : Quantile bins for decile lift (default 10).
    top_n_importance : Rows shown in importance tables (default 20).

    Returns
    -------
    dict with keys:
        overall_rate, lift_results, gini_df (or None), perm_df (or None).
    """
    print(f"\n{'=' * 80}")
    print("📈 RF LIFT ANALYSIS — BINNING STRATEGY")
    print(f"{'=' * 80}")

    X = df[features]
    y = df[target_col]
    overall_rate = float(y.mean())

    print(f"\n   Overall conversion rate : {overall_rate:.4f} ({overall_rate:.1%})")
    print(f"   Features in analysis    : {len(features)}")

    # ── Importance (optional — needs fitted pipe) ─────────────────────────────
    gini_df: Optional[pd.DataFrame] = None
    perm_df: Optional[pd.DataFrame] = None

    if rf_pipe is not None:
        rf_pipe.fit(X, y)
        gini_df = compute_gini_importance(rf_pipe, X)
        print("\n   🔀 Computing Permutation Importance (~30s)...")
        perm_df = compute_permutation_importance(rf_pipe, X, y)
        _print_importance(gini_df, perm_df, top_n_importance)
        _plot_importance(gini_df, perm_df)
    else:
        print("\n   ℹ️  No rf_pipe provided — skipping importance analysis.")
        print("      Pass rf_result.pipe to enable Gini + permutation importances.")

    # ── Lift by decile ────────────────────────────────────────────────────────
    lift_results = compute_lift(df, features, target_col, n_deciles)
    _print_lift(lift_results, overall_rate)
    _print_recommendations(lift_results, overall_rate)
    _plot_lift_curves(lift_results, overall_rate)

    print(f"\n{'=' * 80}")
    print("✅ Lift analysis complete — review figures then confirm binning.py thresholds")
    print(f"{'=' * 80}")

    return {
        "overall_rate": overall_rate,
        "lift_results": lift_results,
        "gini_df": gini_df,
        "perm_df": perm_df,
    }