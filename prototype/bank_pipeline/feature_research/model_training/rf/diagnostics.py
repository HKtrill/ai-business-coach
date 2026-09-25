"""
model_training/rf/diagnostics.py

RF feature diagnostics for the binary feature space.
RF-native only: Gini importance, permutation importance, rank agreement,
composite ranking, and cut batch recommendations.

Public API
----------
    rf_diagnostics(rf_result, df, features, target_col, ...) -> dict

Return dict keys
----------------
    gini_df        DataFrame [feature, importance, gini_rank, gini_cumulative]
    perm_df        DataFrame [feature, perm_importance, perm_std, perm_rank]
    agreement_df   DataFrame — merged gini+perm ranks, rank_diff, avg_rank
    spearman_rho   float
    spearman_p     float
    composite_df   DataFrame — composite scores and tiers
    batch1         list — TIER 4 features, safe to cut
    batch2         list — TIER 3 features, test removal
    safe_cut_list  list — low on BOTH gini and perm
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from feature_research.model_training.rf.trainer import RFResult, _build_pipe, _make_sample_weights


# ── Sub-computations ──────────────────────────────────────────────────────────

def _compute_gini(rf_result: RFResult, features: list[str]) -> pd.DataFrame:
    """Extract Gini importance from the already-fitted pipe in RFResult."""
    clf = rf_result.pipe.named_steps["clf"]
    df = (
        pd.DataFrame({"feature": features, "importance": clf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    df["gini_rank"]       = range(1, len(df) + 1)
    df["gini_cumulative"] = df["importance"].cumsum()
    return df


def _compute_permutation_cv(
    rf_result: RFResult,
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    n_folds: int = 5,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    CV-averaged permutation importance — less biased than single-pass.

    Refits the model on each fold's train split, evaluates on the val split.
    Distinct from lift_analysis.compute_permutation_importance which is a
    single-pass exploratory version.
    """
    params = rf_result.params
    minority_weight = params.get("minority_weight", 1.0)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_means = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train        = y.iloc[train_idx]
        y_val          = y.iloc[val_idx]

        pipe = _build_pipe(params, random_state)
        sw   = _make_sample_weights(y_train, minority_weight)
        pipe.fit(X_train, y_train, clf__sample_weight=sw)

        result = permutation_importance(
            pipe, X_val, y_val,
            n_repeats=n_repeats,
            scoring="roc_auc",
            random_state=random_state,
            n_jobs=-1,
        )
        fold_means.append(result.importances_mean)

    perm_mean = np.mean(fold_means, axis=0)
    perm_std  = np.std(fold_means, axis=0)

    df = (
        pd.DataFrame({"feature": features, "perm_importance": perm_mean, "perm_std": perm_std})
        .sort_values("perm_importance", ascending=False)
        .reset_index(drop=True)
    )
    df["perm_rank"] = range(1, len(df) + 1)
    return df


def _compute_rank_agreement(
    gini_df: pd.DataFrame,
    perm_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """
    Merge gini and perm rankings, compute Spearman rho and rank_diff.
    Returns (agreement_df, rho, p_value).
    """
    merged = (
        gini_df[["feature", "importance", "gini_rank"]]
        .merge(perm_df[["feature", "perm_importance", "perm_std", "perm_rank"]], on="feature")
    )
    merged["rank_diff"] = (merged["gini_rank"] - merged["perm_rank"]).abs()
    merged["avg_rank"]  = (merged["gini_rank"] + merged["perm_rank"]) / 2

    rho, p_val = spearmanr(merged["gini_rank"], merged["perm_rank"])
    return merged, float(rho), float(p_val)


def _compute_composite(
    agreement_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Composite score = 0.40×norm_gini + 0.60×norm_perm

    Permutation importance weighted higher — least biased for trees.
    Both components normalised to [0, 1] before combining.

    Tiers
    -----
    TIER 1 : top 10   → keep
    TIER 2 : 11–20    → likely keep
    TIER 3 : 21–30    → test removal
    TIER 4 : 31+      → safe to cut
    """
    df = agreement_df[["feature", "importance", "gini_rank",
                        "perm_importance", "perm_rank", "avg_rank"]].copy()

    scaler = MinMaxScaler()
    df[["norm_gini", "norm_perm"]] = scaler.fit_transform(df[["importance", "perm_importance"]])

    df["composite_score"] = 0.40 * df["norm_gini"] + 0.60 * df["norm_perm"]
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["composite_rank"] = range(1, len(df) + 1)

    def _tier(rank: int) -> str:
        if rank <= 10: return "🟢 TIER 1 (keep)"
        if rank <= 20: return "🟡 TIER 2 (likely keep)"
        if rank <= 30: return "🟠 TIER 3 (test removal)"
        return               "🔴 TIER 4 (safe to cut)"

    df["tier"] = df["composite_rank"].apply(_tier)
    return df


# ── Orchestrator ──────────────────────────────────────────────────────────────

def rf_diagnostics(
    rf_result: RFResult,
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    top_n: int = 10,
    show_plots: bool = True,
) -> dict:
    """
    RF-native feature diagnostics for the binary feature space.

    Parameters
    ----------
    rf_result   : Output of train_rf() — pipe already fitted on full data.
    df          : DataFrame with features + target_col.
    features    : RF_FEATURES_BINARY
    target_col  : Binary target column name
    top_n       : Number of top features to annotate in plots (default 10)
    show_plots  : If False, skips all matplotlib output (useful in batch runs)

    Returns
    -------
    dict — see module docstring for full key listing
    """
    import matplotlib.pyplot as plt

    X = df[features].copy()
    y = df[target_col].copy()

    print("\n" + "=" * 80)
    print("🔬 RF FEATURE DIAGNOSTICS  (binary feature space)")
    print(f"   {len(features)} features  |  2^{len(features)} = {2**len(features):,} possible patterns")
    print("=" * 80)

    # ── 1. Gini importance ────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("1️⃣  GINI IMPORTANCE (Mean Decrease in Impurity)")
    print(f"{'─' * 80}")

    gini_df  = _compute_gini(rf_result, features)
    n_for_90 = int((gini_df["gini_cumulative"] < 0.90).sum()) + 1
    gini_dead = gini_df[gini_df["importance"] < 0.01]["feature"].tolist()

    for _, row in gini_df.iterrows():
        bar  = "█" * int(row["importance"] * 100)
        flag = "🔥 TOP" if row["importance"] >= 0.03 else "⚪ MID" if row["importance"] >= 0.01 else "💀 LOW"
        print(f"   {int(row['gini_rank']):2d}. {row['feature']:35s} {row['importance']:.4f}  {bar}  {flag}")

    print(f"\n   📊 Top {n_for_90} features capture 90% of Gini importance")
    print(f"   💀 Gini < 0.01 : {len(gini_dead)} features")

    # ── 2. Permutation importance (CV-averaged) ───────────────────────────────
    print(f"\n{'─' * 80}")
    print("2️⃣  PERMUTATION IMPORTANCE (5-fold CV, 10 repeats — unbiased)")
    print(f"{'─' * 80}")
    print("   Computing... (~30-60s)")

    perm_t0 = time.perf_counter()
    perm_df = _compute_permutation_cv(rf_result, X, y, features)
    print(f"   ✅ Done in {time.perf_counter() - perm_t0:.1f}s\n")

    perm_dead     = perm_df[perm_df["perm_importance"] <= 0]["feature"].tolist()
    perm_marginal = perm_df[
        (perm_df["perm_importance"] > 0) & (perm_df["perm_importance"] < 0.001)
    ]["feature"].tolist()

    for _, row in perm_df.iterrows():
        bar  = "█" * max(0, int(row["perm_importance"] * 200))
        flag = (
            "🔥 TOP"       if row["perm_importance"] >= 0.005 else
            "⚪ MID"       if row["perm_importance"] >= 0.001 else
            "⚠️  MARGINAL"  if row["perm_importance"] >  0    else
            "💀 ZERO/NEG"
        )
        print(f"   {int(row['perm_rank']):2d}. {row['feature']:35s} "
              f"{row['perm_importance']:+.5f} ±{row['perm_std']:.5f}  {bar}  {flag}")

    print(f"\n   💀 Zero/negative : {len(perm_dead)}")
    print(f"   ⚠️  Marginal      : {len(perm_marginal)}")

    # ── 3. Rank agreement ─────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("3️⃣  GINI vs PERMUTATION AGREEMENT")
    print(f"{'─' * 80}")

    agreement_df, rho, p_val = _compute_rank_agreement(gini_df, perm_df)

    print(f"\n   Spearman ρ = {rho:.3f}  (p = {p_val:.4f})")
    if rho > 0.8:
        print("   ✅ Strong agreement — rankings are consistent")
    elif rho > 0.5:
        print("   ⚠️  Moderate agreement — some Gini bias possible")
    else:
        print("   ❌ Weak agreement — Gini likely biased")

    gini_top10    = set(gini_df.head(10)["feature"])
    perm_top10    = set(perm_df.head(10)["feature"])
    top10_overlap = gini_top10 & perm_top10

    print(f"\n   Top-10 overlap : {len(top10_overlap)}/10")
    print(f"   Both           : {sorted(top10_overlap)}")
    gini_only = gini_top10 - perm_top10
    perm_only  = perm_top10 - gini_top10
    if gini_only: print(f"   Gini only      : {sorted(gini_only)}  ← may be Gini-biased")
    if perm_only:  print(f"   Perm only      : {sorted(perm_only)}  ← genuinely predictive")

    safe_cut_mask = (
        (agreement_df["importance"] < 0.01) &
        (agreement_df["perm_importance"] < 0.001)
    )
    safe_cut_df   = agreement_df[safe_cut_mask].sort_values("avg_rank", ascending=False)
    safe_cut_list = safe_cut_df["feature"].tolist()

    print(f"\n   🎯 Safe removal candidates (low on BOTH): {len(safe_cut_list)}")
    for _, row in safe_cut_df.iterrows():
        print(f"      {row['feature']:35s}  Gini={row['importance']:.4f} (#{int(row['gini_rank'])})"
              f"  Perm={row['perm_importance']:.5f} (#{int(row['perm_rank'])})")

    # ── 4. Composite ranking & tiers ──────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("4️⃣  COMPOSITE RANKING & FEATURE TIERS")
    print(f"{'─' * 80}")
    print("\n   Score = 0.40×norm_gini + 0.60×norm_perm\n")

    composite_df = _compute_composite(agreement_df)

    for _, row in composite_df.iterrows():
        print(f"   {int(row['composite_rank']):2d}. {row['feature']:35s} "
              f"Score={row['composite_score']:.3f}  "
              f"Gini=#{int(row['gini_rank']):2d}  Perm=#{int(row['perm_rank']):2d}  "
              f"{row['tier']}")

    tier_counts = composite_df["tier"].value_counts().sort_index()
    print("\n   Tier distribution:")
    for tier, count in tier_counts.items():
        print(f"      {tier}: {count} features")

    # ── 5. Cut batches ────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("5️⃣  RECOMMENDED CUT BATCHES")
    print(f"{'─' * 80}")

    batch1 = composite_df[composite_df["tier"].str.contains("TIER 4")]["feature"].tolist()
    batch2 = composite_df[composite_df["tier"].str.contains("TIER 3")]["feature"].tolist()
    n_keep = len(composite_df[composite_df["tier"].str.contains("TIER 1|TIER 2")])

    print(f"\n   BATCH 1 — cut first (TIER 4, {len(batch1)} features):")
    if batch1:
        for f in batch1:
            score = composite_df.loc[composite_df["feature"] == f, "composite_score"].values[0]
            print(f"      ❌ {f:35s} Score={score:.3f}")
    else:
        print("      (No TIER 4 features — set may already be lean)")

    print(f"\n   BATCH 2 — test removal (TIER 3, {len(batch2)} features):")
    if batch2:
        for f in batch2:
            score = composite_df.loc[composite_df["feature"] == f, "composite_score"].values[0]
            print(f"      ⚠️  {f:35s} Score={score:.3f}")
    else:
        print("      (No TIER 3 features)")

    print(f"\n   🎯 Target : ~{n_keep} features (TIER 1 + TIER 2)")
    print(f"      {len(features)} → remove ~{len(features) - n_keep} across batches")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    if show_plots:
        print(f"\n{'─' * 80}")
        print("📊 Generating diagnostic plots...")

        tier_color_map = {
            "🟢 TIER 1 (keep)":         "#2ecc71",
            "🟡 TIER 2 (likely keep)":  "#f1c40f",
            "🟠 TIER 3 (test removal)": "#e67e22",
            "🔴 TIER 4 (safe to cut)":  "#e74c3c",
        }

        plot_data = composite_df.merge(
            gini_df[["feature", "importance"]].rename(columns={"importance": "gini_imp"}),
            on="feature",
        )

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Gini vs Permutation scatter
        ax = axes[0]
        colors = [tier_color_map.get(t, "#95a5a6") for t in plot_data["tier"]]
        ax.scatter(plot_data["gini_imp"], plot_data["perm_importance"],
                   c=colors, s=60, alpha=0.8, edgecolors="black", linewidth=0.5)
        for _, row in plot_data.head(top_n).iterrows():
            ax.annotate(row["feature"], (row["gini_imp"], row["perm_importance"]),
                        fontsize=7, alpha=0.8, rotation=15)
        ax.axhline(y=0.001, color="red",  linestyle="--", alpha=0.5, label="Perm threshold (0.001)")
        ax.axvline(x=0.01,  color="blue", linestyle="--", alpha=0.5, label="Gini threshold (0.01)")
        ax.set_xlabel("Gini Importance")
        ax.set_ylabel("Permutation Importance")
        ax.set_title(f"Gini vs Permutation  (Spearman ρ={rho:.3f})")
        ax.legend(fontsize=8)

        # Plot 2: Composite score bar chart
        ax = axes[1]
        bar_colors = [tier_color_map.get(t, "#95a5a6") for t in composite_df["tier"]]
        ax.barh(range(len(composite_df)), composite_df["composite_score"],
                color=bar_colors, edgecolor="black", linewidth=0.3)
        ax.set_yticks(range(len(composite_df)))
        ax.set_yticklabels(composite_df["feature"], fontsize=7)
        ax.set_xlabel("Composite Score  (0.40×Gini + 0.60×Perm)")
        ax.set_title("Feature Composite Ranking")
        ax.invert_yaxis()

        # Plot 3: Rank comparison (Gini rank vs Perm rank)
        ax = axes[2]
        for _, row in composite_df.iterrows():
            color  = tier_color_map.get(row["tier"], "#95a5a6")
            g_rank = agreement_df.loc[agreement_df["feature"] == row["feature"], "gini_rank"].values[0]
            p_rank = agreement_df.loc[agreement_df["feature"] == row["feature"], "perm_rank"].values[0]
            ax.plot([0, 1], [g_rank, p_rank], color=color, alpha=0.6, linewidth=1.5)
            ax.scatter([0], [g_rank], color=color, s=30, zorder=5)
            ax.scatter([1], [p_rank], color=color, s=30, zorder=5)
        big_movers = agreement_df[agreement_df["rank_diff"] > 10]
        for _, row in big_movers.iterrows():
            ax.annotate(row["feature"], (1.02, row["perm_rank"]), fontsize=6, alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Gini Rank", "Permutation Rank"])
        ax.set_ylabel("Rank (1 = best)")
        ax.set_title("Rank Agreement  (red lines = big movers)")
        ax.set_ylim(len(features) + 1, 0)

        plt.tight_layout()
        plt.show()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("6️⃣  SUMMARY")
    print(f"{'─' * 80}")
    print(f"\n   Features              : {len(features)}")
    print(f"   Gini dead (<0.01)     : {len(gini_dead)}")
    print(f"   Perm dead (≤0)        : {len(perm_dead)}")
    print(f"   Perm marginal (<0.001): {len(perm_marginal)}")
    print(f"   Top-10 agree          : {len(top10_overlap)}/10")
    print(f"   Spearman ρ            : {rho:.3f}")
    print(f"\n   🎯 REDUCTION ROADMAP:")
    print(f"      Step 1 : Remove Batch 1 ({len(batch1)} features) → re-run Cell 13C")
    print(f"      Step 2 : Remove Batch 2 in groups ({len(batch2)} features) → re-run Cell 13C each time")
    print(f"      Step 3 : Final set → re-run Cell 13D to confirm clean diagnostics")
    print(f"\n{'=' * 80}")
    print("✅ RF Diagnostics complete")
    print(f"{'=' * 80}")

    return {
        "gini_df":       gini_df,
        "perm_df":       perm_df,
        "agreement_df":  agreement_df,
        "spearman_rho":  rho,
        "spearman_p":    p_val,
        "composite_df":  composite_df,
        "batch1":        batch1,
        "batch2":        batch2,
        "safe_cut_list": safe_cut_list,
    }