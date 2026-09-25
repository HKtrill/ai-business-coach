"""
feature_research/model_training/ebm/diagnostics.py
====================================================
EBM-native diagnostics for the Glass Cascade pipeline.

Interrogates a fitted EBMResult — no refitting occurs here.
Trainer owns fitting; diagnostics owns interrogation.

Sections
--------
1. Global feature importance (main effects + interaction terms)
2. Shape function linearity  (R² of EBM shape vs linear fit)
3. Effect magnitude          (score range per feature)
4. Interaction term analysis (share of total importance)
5. Redundancy check          (linear + already upstream → flag)
6. Correlation analysis      (|r| > 0.7 pairs)
7. Composite ranking         (weighted score → TIER 1–4)
8. Visualisations            (6-panel diagnostic plot)

abs_r derivation
----------------
sep_df arrives as the raw output of compute_all_separations() which
provides ``point_biserial`` (not ``abs_r``). The absolute value is
derived internally before the composite merge — callers pass raw sep_df
unchanged.

Return dict keys
----------------
importance_df, linearity_df, magnitude_df, interaction_df,
redundancy_df, corr_df, composite_df, batch1, safe_cut_list

Provides
--------
- ebm_diagnostics()
"""

from __future__ import annotations

from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from feature_research.model_training.ebm.trainer import EBMResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ebm_diagnostics(
    result: EBMResult,
    X: pd.DataFrame,
    y: pd.Series,
    sep_df: pd.DataFrame,
    lr_features: List[str],
    rf_features: List[str],
) -> Dict[str, Any]:
    """Run EBM-native diagnostic suite on a fitted EBMResult.

    Parameters
    ----------
    result      : EBMResult from train_ebm() — pipe already fitted on full data.
    X           : Feature matrix used for training (EBM_FEATURES columns).
    y           : Binary target (0/1).
    sep_df      : Raw output of compute_all_separations() — point_biserial and
                  ks_stat columns used in composite ranking. abs_r is derived
                  internally; do not pre-compute it.
    lr_features : LR_FEATURES list — for redundancy cross-check.
    rf_features : RF_FEATURES list — for redundancy cross-check.

    Returns
    -------
    dict with keys:
        importance_df, linearity_df, magnitude_df, interaction_df,
        redundancy_df, corr_df, composite_df, batch1, safe_cut_list
    """

    model = result.pipe

    print("\n" + "=" * 80)
    print("🔬 EBM FEATURE DIAGNOSTICS")
    print(f"   Feature set: {X.shape[1]} features")
    print(f"   Model config: interactions={model.interactions}, "
          f"learning_rate={model.learning_rate:.6f}, "
          f"max_bins={model.max_bins}, "
          f"max_rounds={model.max_rounds}")
    print("=" * 80)

    ebm_global = model.explain_global()

    # =========================================================================
    # 1. GLOBAL FEATURE IMPORTANCE
    # =========================================================================
    print("\n" + "-" * 80)
    print("1️⃣  GLOBAL FEATURE IMPORTANCE (Main Effects + Interactions)")
    print("-" * 80)

    main_names: List[str] = []
    main_scores: List[float] = []
    interaction_names_raw: List[tuple] = []
    interaction_scores_raw: List[float] = []

    for i, feat in enumerate(ebm_global.data()["names"]):
        score = float(ebm_global.data()["scores"][i])
        if isinstance(feat, tuple):
            interaction_names_raw.append(feat)
            interaction_scores_raw.append(score)
        elif isinstance(feat, str) and " & " in feat:
            parts = tuple(feat.split(" & "))
            interaction_names_raw.append(parts)
            interaction_scores_raw.append(score)
        else:
            main_names.append(feat)
            main_scores.append(score)

    all_feat_names = main_names + [
        " & ".join(p) if isinstance(p, tuple) else p
        for p in interaction_names_raw
    ]
    all_feat_scores = main_scores + interaction_scores_raw

    importance_df = (
        pd.DataFrame({"feature": all_feat_names, "importance": all_feat_scores})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["imp_rank"] = range(1, len(importance_df) + 1)
    importance_df["imp_cumulative"] = importance_df["importance"].cumsum()
    total_importance = importance_df["importance"].sum()
    importance_df["imp_cumulative_pct"] = (
        importance_df["imp_cumulative"] / total_importance
    )

    for _, row in importance_df.iterrows():
        bar = "█" * int(
            row["importance"] / importance_df["importance"].max() * 30
        )
        if row["imp_cumulative_pct"] <= 0.80:
            flag = "🔥 TOP-80%"
        elif row["imp_cumulative_pct"] <= 0.90:
            flag = "⚪ TOP-90%"
        elif row["imp_cumulative_pct"] <= 0.95:
            flag = "⚠️  TOP-95%"
        else:
            flag = "💀 TAIL"
        print(
            f"   {int(row['imp_rank']):2d}. {row['feature']:40s} "
            f"{row['importance']:.4f}  cum={row['imp_cumulative_pct']:.1%}  "
            f"{bar}  {flag}"
        )

    n_for_80 = int((importance_df["imp_cumulative_pct"] < 0.80).sum()) + 1
    n_for_90 = int((importance_df["imp_cumulative_pct"] < 0.90).sum()) + 1
    n_for_95 = int((importance_df["imp_cumulative_pct"] < 0.95).sum()) + 1
    tail_features = importance_df[
        importance_df["imp_cumulative_pct"] > 0.95
    ]["feature"].tolist()

    print(f"\n   📊 Cumulative Importance Milestones:")
    print(f"      80% importance: top {n_for_80} features")
    print(f"      90% importance: top {n_for_90} features")
    print(f"      95% importance: top {n_for_95} features")
    print(f"      Tail (>95%):   {len(tail_features)} features — candidates for removal")

    # =========================================================================
    # 2. SHAPE FUNCTION LINEARITY
    # =========================================================================
    print("\n" + "-" * 80)
    print("2️⃣  SHAPE FUNCTION LINEARITY (R² of shape vs linear fit)")
    print("   → High R²: EBM learned ~linear shape → LR can handle it")
    print("   → Low R²:  EBM exploits non-linearity → EBM adds unique value")
    print("-" * 80)

    linearity_data = []
    all_ebm_names = ebm_global.data()["names"]

    for feat_idx in range(len(all_ebm_names)):
        raw_name = all_ebm_names[feat_idx]
        display_name = (
            " & ".join(raw_name) if isinstance(raw_name, tuple) else raw_name
        )

        try:
            feat_data = ebm_global.data(feat_idx)
            shape_scores = np.array(feat_data["scores"])

            if shape_scores.ndim > 1:
                shape_scores = shape_scores.flatten()

            x_vals = np.arange(len(shape_scores)).reshape(-1, 1)

            if np.std(shape_scores) < 1e-10:
                linearity_data.append({
                    "feature":      display_name,
                    "linearity_r2": 1.0,
                    "score_range":  0.0,
                    "score_std":    0.0,
                    "n_bins":       len(shape_scores),
                    "note":         "constant shape",
                })
                continue

            lr_fit = LinearRegression()
            lr_fit.fit(x_vals, shape_scores)
            r2 = float(lr_fit.score(x_vals, shape_scores))

            linearity_data.append({
                "feature":      display_name,
                "linearity_r2": max(0.0, r2),
                "score_range":  float(np.max(shape_scores) - np.min(shape_scores)),
                "score_std":    float(np.std(shape_scores)),
                "n_bins":       len(shape_scores),
                "note":         "",
            })

        except Exception as e:
            linearity_data.append({
                "feature":      display_name,
                "linearity_r2": np.nan,
                "score_range":  np.nan,
                "score_std":    np.nan,
                "n_bins":       0,
                "note":         f"error: {str(e)[:40]}",
            })

    linearity_df = (
        pd.DataFrame(linearity_data)
        .sort_values("linearity_r2", ascending=False)
        .reset_index(drop=True)
    )

    for _, row in linearity_df.iterrows():
        if pd.isna(row["linearity_r2"]):
            flag = "❓ ERROR"
        elif row["linearity_r2"] > 0.90:
            flag = "📏 LINEAR"
        elif row["linearity_r2"] > 0.70:
            flag = "〰️  MODERATE"
        elif row["linearity_r2"] > 0.40:
            flag = "🌊 NON-LINEAR"
        else:
            flag = "🔀 COMPLEX"
        print(
            f"   {row['feature']:40s} R²={row['linearity_r2']:.3f}  "
            f"range={row['score_range']:.4f}  {flag}  {row['note']}"
        )

    highly_linear = linearity_df[
        linearity_df["linearity_r2"] > 0.90
    ]["feature"].tolist()
    complex_shapes = linearity_df[
        linearity_df["linearity_r2"] < 0.40
    ]["feature"].tolist()
    highly_linear_main = [f for f in highly_linear if " & " not in f]

    print(f"\n   📏 Highly linear shapes (R² > 0.90): {len(highly_linear)} features")
    print(f"      → EBM adds no value over LR for these; check upstream coverage")
    if highly_linear_main:
        for f in highly_linear_main:
            in_lr = "✅ in LR" if f in lr_features else "❌ not in LR"
            in_rf = "✅ in RF" if f in rf_features else "❌ not in RF"
            print(f"      {f:40s} {in_lr}  {in_rf}")

    print(f"\n   🔀 Complex shapes (R² < 0.40): {len(complex_shapes)} features")
    print(f"      → These are where EBM adds unique value")

    # =========================================================================
    # 3. EFFECT MAGNITUDE
    # =========================================================================
    print("\n" + "-" * 80)
    print("3️⃣  EFFECT MAGNITUDE (Score Range per Feature)")
    print("   → Features with tiny score range have negligible prediction impact")
    print("-" * 80)

    magnitude_df = (
        linearity_df[["feature", "score_range"]]
        .copy()
        .sort_values("score_range", ascending=False)
        .reset_index(drop=True)
    )
    magnitude_df["mag_rank"] = range(1, len(magnitude_df) + 1)

    for _, row in magnitude_df.iterrows():
        bar = "█" * int(
            row["score_range"] / magnitude_df["score_range"].max() * 30
        ) if magnitude_df["score_range"].max() > 0 else ""
        if row["score_range"] >= 0.10:
            flag = "🔥 STRONG"
        elif row["score_range"] >= 0.03:
            flag = "⚪ MODERATE"
        elif row["score_range"] >= 0.01:
            flag = "⚠️  WEAK"
        else:
            flag = "💀 NEGLIGIBLE"
        print(
            f"   {int(row['mag_rank']):2d}. {row['feature']:40s} "
            f"range={row['score_range']:.4f}  {bar}  {flag}"
        )

    negligible_effect = magnitude_df[
        magnitude_df["score_range"] < 0.01
    ]["feature"].tolist()
    weak_effect = magnitude_df[
        (magnitude_df["score_range"] >= 0.01) & (magnitude_df["score_range"] < 0.03)
    ]["feature"].tolist()

    print(f"\n   💀 Negligible effect (range < 0.01): {len(negligible_effect)} features")
    for f in negligible_effect:
        print(f"      → {f}")
    print(f"   ⚠️  Weak effect (0.01 ≤ range < 0.03): {len(weak_effect)} features")

    # =========================================================================
    # 4. INTERACTION TERM ANALYSIS
    # =========================================================================
    print("\n" + "-" * 80)
    print("4️⃣  INTERACTION TERM ANALYSIS")
    print("-" * 80)

    if interaction_names_raw:
        interaction_df = (
            pd.DataFrame({
                "interaction": [" & ".join(pair) for pair in interaction_names_raw],
                "feat_1":      [pair[0] for pair in interaction_names_raw],
                "feat_2":      [pair[1] for pair in interaction_names_raw],
                "importance":  interaction_scores_raw,
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        total_main_imp = sum(main_scores)
        total_int_imp  = interaction_df["importance"].sum()
        int_share = total_int_imp / (total_main_imp + total_int_imp) * 100

        print(f"\n   📊 {len(interaction_df)} interaction term(s) detected")
        print(f"   Main effects total importance:  {total_main_imp:.4f}")
        print(f"   Interaction total importance:   {total_int_imp:.4f} "
              f"({int_share:.1f}% of total)")

        max_imp = interaction_df["importance"].max()
        for _, row in interaction_df.iterrows():
            bar = "█" * int(row["importance"] / max_imp * 20) if max_imp > 0 else ""
            print(f"   {row['interaction']:45s} {row['importance']:.4f}  {bar}")

        interacting_features: set = set()
        for pair in interaction_names_raw:
            interacting_features.update(pair)
        interacting_features = interacting_features & set(main_names)

        print(f"\n   🔗 Features in interactions: {sorted(interacting_features)}")
        print(f"      → Retain even if main effect seems small")
    else:
        interaction_df = pd.DataFrame()
        interacting_features = set()
        print("\n   ℹ️  No interaction terms (interactions=0 or none learned)")

    # =========================================================================
    # 5. REDUNDANCY CHECK
    # =========================================================================
    print("\n" + "-" * 80)
    print("5️⃣  REDUNDANCY CHECK — Linear features already handled upstream")
    print("   → Linear AND already in LR or RF → EBM is redundant for that feature")
    print("-" * 80)

    redundancy_data = []
    for _, row in linearity_df.iterrows():
        feat = row["feature"]
        is_interaction_term = " & " in feat
        in_lr = feat in lr_features if not is_interaction_term else False
        in_rf = feat in rf_features if not is_interaction_term else False
        is_linear = row["linearity_r2"] > 0.90
        is_negligible = row["score_range"] < 0.01
        upstream = in_lr or in_rf
        in_interaction = feat in interacting_features

        if is_negligible:
            verdict = "💀 NEGLIGIBLE EFFECT — safe to cut"
        elif is_interaction_term:
            verdict = "🔗 INTERACTION TERM — auto-learned by EBM"
        elif is_linear and upstream and not in_interaction:
            verdict = "🔄 REDUNDANT — linear + already upstream"
        elif is_linear and upstream and in_interaction:
            verdict = "⚠️  Linear + upstream BUT in interaction — keep"
        elif is_linear and not upstream:
            verdict = "📏 Linear but NOT upstream — EBM is only handler"
        else:
            verdict = "✅ NON-LINEAR — EBM adds unique value"

        redundancy_data.append({
            "feature":            feat,
            "r2":                 row["linearity_r2"],
            "score_range":        row["score_range"],
            "in_lr":              in_lr,
            "in_rf":              in_rf,
            "in_interaction":     in_interaction,
            "is_interaction_term": is_interaction_term,
            "verdict":            verdict,
        })

    redundancy_df = pd.DataFrame(redundancy_data)
    redundant_mask = redundancy_df["verdict"].str.contains("REDUNDANT|NEGLIGIBLE")

    if redundant_mask.any():
        print(f"\n   🎯 Features flagged for removal:")
        for _, row in redundancy_df[redundant_mask].iterrows():
            lr_flag = "LR✓" if row["in_lr"] else "LR✗"
            rf_flag = "RF✓" if row["in_rf"] else "RF✗"
            print(
                f"      {row['feature']:40s} R²={row['r2']:.3f}  "
                f"range={row['score_range']:.4f}  [{lr_flag} {rf_flag}]  {row['verdict']}"
            )

    print(f"\n   ✅ Features to keep ({(~redundant_mask).sum()}):")
    for _, row in (
        redundancy_df[~redundant_mask]
        .sort_values("score_range", ascending=False)
        .iterrows()
    ):
        lr_flag  = "LR✓" if row["in_lr"] else "LR✗"
        rf_flag  = "RF✓" if row["in_rf"] else "RF✗"
        int_flag = "INT✓" if row["in_interaction"] else ""
        ix_flag  = " [IX]" if row["is_interaction_term"] else ""
        print(
            f"      {row['feature']:40s} R²={row['r2']:.3f}  "
            f"range={row['score_range']:.4f}  [{lr_flag} {rf_flag}]  "
            f"{int_flag}{ix_flag}  {row['verdict']}"
        )

    # =========================================================================
    # 6. CORRELATION ANALYSIS
    # =========================================================================
    print("\n" + "-" * 80)
    print("6️⃣  CORRELATION ANALYSIS")
    print("-" * 80)

    corr_matrix = X.corr()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.7:
                feat_i = corr_matrix.columns[i]
                feat_j = corr_matrix.columns[j]
                imp_i = importance_df[importance_df["feature"] == feat_i]["imp_rank"].values
                imp_j = importance_df[importance_df["feature"] == feat_j]["imp_rank"].values
                rank_i = imp_i[0] if len(imp_i) > 0 else 999
                rank_j = imp_j[0] if len(imp_j) > 0 else 999
                keep = feat_i if rank_i < rank_j else feat_j
                drop = feat_j if rank_i < rank_j else feat_i
                high_corr_pairs.append({
                    "Feature 1":         feat_i,
                    "Feature 2":         feat_j,
                    "Correlation":        r,
                    "Keep (higher rank)": keep,
                    "Drop candidate":     drop,
                })

    if high_corr_pairs:
        corr_df = pd.DataFrame(high_corr_pairs).sort_values(
            "Correlation", key=abs, ascending=False
        )
        print(f"\n   📊 High Correlation Pairs (|r| > 0.7): {len(corr_df)} pair(s)")
        for _, row in corr_df.iterrows():
            flag = "🔴" if abs(row["Correlation"]) > 0.9 else "🟡"
            print(
                f"   {flag} {row['Feature 1']:35s} ↔ {row['Feature 2']:35s} "
                f"r={row['Correlation']:+.3f}"
            )
            print(f"      → Keep: {row['Keep (higher rank)']},  "
                  f"Drop candidate: {row['Drop candidate']}")
        corr_drop_candidates = list(set(corr_df["Drop candidate"].tolist()))
        print(f"\n   🎯 Correlation-based drop candidates: {corr_drop_candidates}")
    else:
        corr_df = pd.DataFrame()
        corr_drop_candidates = []
        print("\n   ✅ No high correlation pairs (|r| > 0.7)")

    # =========================================================================
    # 7. COMPOSITE RANKING & TIER ASSIGNMENT
    # =========================================================================
    print("\n" + "-" * 80)
    print("7️⃣  COMPOSITE RANKING & FEATURE TIERS")
    print("-" * 80)

    composite_df = importance_df[["feature", "importance", "imp_rank"]].copy()

    # Merge linearity and magnitude
    lin_merge = linearity_df[["feature", "linearity_r2", "score_range"]].copy()
    composite_df = composite_df.merge(lin_merge, on="feature", how="left")

    # Derive abs_r from raw sep_df point_biserial — categorical features get 0
    sep_working = sep_df[["feature", "point_biserial", "ks_stat"]].copy()
    sep_working["abs_r"] = sep_working["point_biserial"].abs().fillna(0.0)
    sep_merge = sep_working[["feature", "abs_r", "ks_stat"]].rename(
        columns={"abs_r": "class_sep_r", "ks_stat": "class_sep_ks"}
    )
    composite_df = composite_df.merge(sep_merge, on="feature", how="left")
    composite_df["class_sep_r"]  = composite_df["class_sep_r"].fillna(0.0)
    composite_df["class_sep_ks"] = composite_df["class_sep_ks"].fillna(0.0)

    # Interaction membership
    composite_df["in_interaction"] = composite_df["feature"].isin(interacting_features)
    composite_df["is_interaction_term"] = composite_df["feature"].str.contains(" & ")

    # Normalise
    _scaler = MinMaxScaler()
    score_cols = ["importance", "score_range", "class_sep_r", "class_sep_ks"]
    composite_df[["norm_imp", "norm_range", "norm_sep_r", "norm_sep_ks"]] = (
        _scaler.fit_transform(composite_df[score_cols].fillna(0))
    )

    # Non-linearity bonus: lower R² → more EBM-unique value
    composite_df["nonlin_bonus"] = 1 - composite_df["linearity_r2"].fillna(0.5)
    composite_df["norm_nonlin"] = MinMaxScaler().fit_transform(
        composite_df[["nonlin_bonus"]]
    )

    # Weighted composite:
    #   Importance    0.30 — how much EBM relies on it
    #   Effect range  0.20 — does it move predictions?
    #   Non-linearity 0.20 — does EBM add value over linear models?
    #   Class sep r   0.15 — point-biserial signal strength
    #   Class sep KS  0.15 — KS signal strength
    composite_df["composite_score"] = (
        0.30 * composite_df["norm_imp"]
        + 0.20 * composite_df["norm_range"]
        + 0.20 * composite_df["norm_nonlin"]
        + 0.15 * composite_df["norm_sep_r"]
        + 0.15 * composite_df["norm_sep_ks"]
    )

    # Interaction bonus
    composite_df.loc[composite_df["in_interaction"], "composite_score"] += 0.05

    composite_df = (
        composite_df
        .sort_values("composite_score", ascending=False)
        .reset_index(drop=True)
    )
    composite_df["composite_rank"] = range(1, len(composite_df) + 1)

    # Tier assignment
    def _assign_tier(rank: int) -> str:
        if rank <= 15:
            return "🟢 TIER 1 (keep)"
        elif rank <= 25:
            return "🟡 TIER 2 (likely keep)"
        elif rank <= 40:
            return "🟠 TIER 3 (test removal)"
        else:
            return "🔴 TIER 4 (safe to cut)"

    composite_df["tier"] = composite_df["composite_rank"].map(_assign_tier)

    print(
        f"\n   Composite = 0.30×Importance + 0.20×EffectRange + 0.20×NonLinearity "
        f"+ 0.15×ClassSep_r + 0.15×ClassSep_KS (+0.05 interaction bonus)\n"
    )

    for _, row in composite_df.iterrows():
        int_flag = " 🔗" if row["in_interaction"] else "   "
        ix_flag  = " [IX]" if row["is_interaction_term"] else ""
        print(
            f"   {int(row['composite_rank']):2d}. {row['feature']:40s} "
            f"Score={row['composite_score']:.3f}  "
            f"Imp=#{int(row['imp_rank']):2d}  "
            f"R²={row['linearity_r2']:.2f}  "
            f"Range={row['score_range']:.4f}{int_flag}{ix_flag}  {row['tier']}"
        )

    tier_counts = composite_df["tier"].value_counts().sort_index()
    print(f"\n   📊 Tier distribution:")
    for tier, count in tier_counts.items():
        print(f"      {tier}: {count} feature(s)")

    # Derive batch1 and safe_cut_list from tier system
    batch1 = composite_df[
        composite_df["tier"] == "🟠 TIER 3 (test removal)"
    ]["feature"].tolist()
    safe_cut_list = composite_df[
        composite_df["tier"] == "🔴 TIER 4 (safe to cut)"
    ]["feature"].tolist()

    print(f"\n   🟠 Batch 1 — test removal (TIER 3): {batch1}")
    print(f"   🔴 Safe cut (TIER 4):                {safe_cut_list}")

    # =========================================================================
    # 8. VISUALISATIONS
    # =========================================================================
    print("\n📊 Generating diagnostic plots...")

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # ── Plot 1: Cumulative importance curve ──
    ax = axes[0, 0]
    ax.plot(
        range(1, len(importance_df) + 1),
        importance_df["imp_cumulative_pct"].values,
        "b-o", markersize=3, linewidth=1.5,
    )
    for thresh, col, lbl in [(0.80, "green", "80%"), (0.90, "orange", "90%"), (0.95, "red", "95%")]:
        ax.axhline(y=thresh, color=col, linestyle="--", alpha=0.7, label=lbl)
    ax.axvline(x=n_for_80, color="green",  linestyle=":", alpha=0.5)
    ax.axvline(x=n_for_90, color="orange", linestyle=":", alpha=0.5)
    ax.axvline(x=n_for_95, color="red",    linestyle=":", alpha=0.5)
    ax.set_xlabel("Number of Features (ranked by importance)")
    ax.set_ylabel("Cumulative Importance %")
    ax.set_title(
        f"Cumulative Importance Curve\n"
        f"80%@{n_for_80}, 90%@{n_for_90}, 95%@{n_for_95} features"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Importance vs Linearity scatter ──
    ax = axes[0, 1]
    tier_color_map = {
        "🟢 TIER 1 (keep)":        "#2ecc71",
        "🟡 TIER 2 (likely keep)":  "#f1c40f",
        "🟠 TIER 3 (test removal)": "#e67e22",
        "🔴 TIER 4 (safe to cut)":  "#e74c3c",
    }
    scatter_df = composite_df.merge(
        linearity_df[["feature", "linearity_r2", "score_range"]],
        on="feature", suffixes=("", "_dup"),
    )
    colors = [tier_color_map.get(t, "#95a5a6") for t in scatter_df["tier"]]
    ax.scatter(
        scatter_df["linearity_r2"], scatter_df["importance"],
        c=colors, s=60, alpha=0.8, edgecolors="black", linewidth=0.5,
    )
    for _, row in scatter_df.head(10).iterrows():
        if " & " not in row["feature"]:
            ax.annotate(
                row["feature"], (row["linearity_r2"], row["importance"]),
                fontsize=6, alpha=0.8, rotation=10,
            )
    ax.axvline(x=0.90, color="red", linestyle="--", alpha=0.5, label="Linear threshold (R²=0.9)")
    ax.set_xlabel("Shape Function Linearity (R²)")
    ax.set_ylabel("Global Importance")
    ax.set_title("Importance vs Linearity\n(top-right = important but linear → may be redundant)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Composite score bar chart ──
    ax = axes[0, 2]
    bar_colors = [tier_color_map.get(t, "#95a5a6") for t in composite_df["tier"]]
    ax.barh(
        range(len(composite_df)), composite_df["composite_score"],
        color=bar_colors, edgecolor="black", linewidth=0.3,
    )
    ax.set_yticks(range(len(composite_df)))
    ax.set_yticklabels(composite_df["feature"], fontsize=6)
    ax.set_xlabel("Composite Score")
    ax.set_title("Feature Composite Ranking (by tier)")
    ax.invert_yaxis()

    # ── Plot 4: Effect magnitude ──
    ax = axes[1, 0]
    mag_sorted = magnitude_df.sort_values("score_range", ascending=True)
    mag_colors = [
        "#e74c3c" if sr < 0.01 else "#f1c40f" if sr < 0.03 else "#2ecc71"
        for sr in mag_sorted["score_range"]
    ]
    ax.barh(
        range(len(mag_sorted)), mag_sorted["score_range"],
        color=mag_colors, edgecolor="black", linewidth=0.3,
    )
    ax.set_yticks(range(len(mag_sorted)))
    ax.set_yticklabels(mag_sorted["feature"], fontsize=6)
    ax.axvline(x=0.01, color="red",    linestyle="--", alpha=0.7, label="Negligible (<0.01)")
    ax.axvline(x=0.03, color="orange", linestyle="--", alpha=0.7, label="Weak (<0.03)")
    ax.set_xlabel("Score Range (Effect Magnitude)")
    ax.set_title("Effect Magnitude per Feature")
    ax.legend(fontsize=7)

    # ── Plot 5: Correlation heatmap (top 20 main effects) ──
    ax = axes[1, 1]
    main_effect_ranked = composite_df[
        ~composite_df["is_interaction_term"]
    ]["feature"].tolist()
    top20_feats = [f for f in main_effect_ranked if f in X.columns][:20]
    top20_corr = X[top20_feats].corr()
    mask = np.triu(np.ones_like(top20_corr, dtype=bool), k=1)
    sns.heatmap(
        top20_corr, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, mask=mask, square=True, ax=ax,
        cbar_kws={"shrink": 0.8}, annot_kws={"size": 5},
    )
    ax.set_title("Correlation Matrix (Top 20 main effects)", fontsize=11)
    ax.tick_params(axis="both", labelsize=5)

    # ── Plot 6: Linearity R² distribution ──
    ax = axes[1, 2]
    r2_vals = linearity_df["linearity_r2"].dropna()
    ax.hist(r2_vals, bins=20, color="#3498db", edgecolor="black", alpha=0.7)
    ax.axvline(x=0.90, color="red",    linestyle="--", linewidth=2, label="Linear threshold (0.90)")
    ax.axvline(x=0.40, color="orange", linestyle="--", linewidth=2, label="Complex threshold (0.40)")
    ax.set_xlabel("Shape Function Linearity (R²)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Linearity Distribution\n"
        f"{len(highly_linear)} linear (R²>0.9), {len(complex_shapes)} complex (R²<0.4)"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =========================================================================
    # COPY-PASTE HELPER
    # =========================================================================
    if batch1 or safe_cut_list:
        all_cuts = batch1 + safe_cut_list
        reduced_features = [f for f in X.columns if f not in all_cuts]
        print(f"\n{'=' * 80}")
        print(f"📋 COPY-PASTE: EBM_FEATURES after Batch 1 removal ({len(reduced_features)} features)")
        print(f"{'=' * 80}")
        print(f"\nEBM_FEATURES = [")
        for f in reduced_features:
            tier_val = composite_df[composite_df["feature"] == f]["tier"].values
            rank_val = composite_df[composite_df["feature"] == f]["composite_rank"].values
            tier_str = tier_val[0] if len(tier_val) > 0 else "unknown"
            rank_str = int(rank_val[0]) if len(rank_val) > 0 else 0
            int_comment = "  # 🔗 in interaction" if f in interacting_features else ""
            print(f"    '{f}',  # #{rank_str} {tier_str}{int_comment}")
        print("]")

    print("\n" + "=" * 80)
    print("✅ EBM Diagnostics complete")
    print("=" * 80)

    return {
        "importance_df":   importance_df,
        "linearity_df":    linearity_df,
        "magnitude_df":    magnitude_df,
        "interaction_df":  interaction_df,
        "redundancy_df":   redundancy_df,
        "corr_df":         corr_df,
        "composite_df":    composite_df,
        "batch1":          batch1,
        "safe_cut_list":   safe_cut_list,
    }