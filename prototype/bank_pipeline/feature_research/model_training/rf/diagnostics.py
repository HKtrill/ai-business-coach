"""
model_training.rf.diagnostics
=============================
RF feature diagnostics for the binary feature space: Gini importance, CV
permutation importance, rank agreement, composite ranking, and cut batches.

Gini comes from rf_result's final fit (descriptive). Permutation importance is
out-of-fold: each fold refits the RF (and, with `feature_factory`, the feature
pipeline + binning) on its training rows and permutes on its validation rows.

Public API
----------
rf_diagnostics(rf_result, df, features, target_col, ...) → dict

Return dict keys
----------------
gini_df        DataFrame [feature, importance, gini_rank, gini_cumulative]
perm_df        DataFrame [feature, perm_importance, perm_std, perm_rank]
agreement_df   DataFrame — merged gini + perm ranks, rank_diff, avg_rank
spearman_rho   float
spearman_p     float
composite_df   DataFrame — composite scores and tiers
batch1         list — TIER 4 features (cut first)
batch2         list — TIER 3 features (test removal)
safe_cut_list  list — low on BOTH Gini (< 0.01) and permutation (< 0.001)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from feature_research.model_training.shared.folds import FeatureFactory, build_folds
from feature_research.model_training.shared.report import fmt_list, show_table, title
from feature_research.model_training.rf.trainer import RFResult, _fit, _require_binary


# ── Sub-computations ──────────────────────────────────────────────────────────

def _compute_gini(rf_result: RFResult, features: list[str]) -> pd.DataFrame:
    """Gini importance from rf_result's final fit; raises if `features` order differs from the fit."""
    clf = rf_result.pipe.named_steps["clf"]
    fitted = list(getattr(clf, "feature_names_in_", features))
    if fitted != list(features):
        raise ValueError(
            "rf_diagnostics: `features` does not match the order the RF was "
            f"fitted on — Gini importances would be mislabeled.\n"
            f"  fitted: {fitted}\n  given:  {list(features)}"
        )
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
    cv: BaseCrossValidator,
    feature_factory: Optional[FeatureFactory],
    n_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Out-of-fold permutation importance (ROC-AUC drop), averaged over folds.

    Each fold fits a fresh RF with rf_result.params on its training rows and
    permutes features on its validation rows. _fit() leaves the forest
    predicting single-threaded, so only permutation_importance runs in parallel.
    """
    folds = build_folds(X, y, features, cv, feature_factory, "perm", check=_require_binary)

    fold_means = []
    for fold in folds:
        pipe = _fit(rf_result.params, fold.X_tr, fold.y_tr, random_state)
        result = permutation_importance(
            pipe, fold.X_va, fold.y_va,
            n_repeats=n_repeats,
            scoring="roc_auc",
            random_state=random_state,
            n_jobs=-1,
        )
        fold_means.append(result.importances_mean)

    df = (
        pd.DataFrame({
            "feature": features,
            "perm_importance": np.mean(fold_means, axis=0),
            "perm_std": np.std(fold_means, axis=0),
        })
        .sort_values("perm_importance", ascending=False)
        .reset_index(drop=True)
    )
    df["perm_rank"] = range(1, len(df) + 1)
    return df


def _compute_rank_agreement(
    gini_df: pd.DataFrame,
    perm_df: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float]:
    """Merge Gini and permutation ranks; returns (agreement_df, Spearman rho, p-value)."""
    merged = (
        gini_df[["feature", "importance", "gini_rank"]]
        .merge(perm_df[["feature", "perm_importance", "perm_std", "perm_rank"]], on="feature")
    )
    merged["rank_diff"] = (merged["gini_rank"] - merged["perm_rank"]).abs()
    merged["avg_rank"]  = (merged["gini_rank"] + merged["perm_rank"]) / 2

    rho, p_val = spearmanr(merged["gini_rank"], merged["perm_rank"])
    return merged, float(rho), float(p_val)


def _compute_composite(agreement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite score = 0.40 × norm_gini + 0.60 × norm_perm (each min-max scaled
    to [0, 1]); permutation is weighted higher because Gini is biased for trees.

    Tiers by composite rank
    -----------------------
    TIER 1 : 1–10   keep
    TIER 2 : 11–20  likely keep
    TIER 3 : 21–30  test removal
    TIER 4 : 31+    safe to cut   (empty while the feature set has ≤ 30 columns)
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
    *,
    cv: Optional[BaseCrossValidator] = None,
    feature_factory: Optional[FeatureFactory] = None,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict:
    """
    RF-native feature diagnostics for the binary feature space.

    Parameters
    ----------
    rf_result       : train_rf() output (final fit + params).
    df              : Training rows including target_col.
                      With feature_factory: raw frame (df_train).
                      Without: df_rf_binary.
    features        : RF_FEATURES_BINARY, in the order rf_result was fit on.
    target_col      : TARGET_COL.
    top_n           : Points annotated in the scatter plot.
    show_plots      : False skips all plots.
    cv              : Permutation-importance splitter (pass the notebook's CV).
                      Default: shuffled StratifiedKFold(5, random_state).
    feature_factory : Same factory as train_rf; refit inside every fold.
    n_repeats       : Permutation repeats per fold.
    random_state    : Seeds the forests and the permutations.

    Returns
    -------
    dict — see module docstring.
    """
    if target_col not in df.columns:
        raise KeyError(f"target column {target_col!r} not in df")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    cv = cv if cv is not None else StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # ── Compute ───────────────────────────────────────────────────────────────
    gini_df = _compute_gini(rf_result, features)
    t0 = time.perf_counter()
    perm_df = _compute_permutation_cv(
        rf_result, X, y, features, cv, feature_factory, n_repeats, random_state
    )
    perm_s = time.perf_counter() - t0
    agreement_df, rho, p_val = _compute_rank_agreement(gini_df, perm_df)
    composite_df = _compute_composite(agreement_df)

    n_for_90      = int((gini_df["gini_cumulative"] < 0.90).sum()) + 1
    gini_dead     = gini_df.loc[gini_df["importance"] < 0.01, "feature"].tolist()
    perm_dead     = perm_df.loc[perm_df["perm_importance"] <= 0, "feature"].tolist()
    perm_marginal = perm_df.loc[perm_df["perm_importance"].between(0, 0.001, inclusive="neither"),
                                "feature"].tolist()

    gini_top10 = set(gini_df.head(10)["feature"])
    perm_top10 = set(perm_df.head(10)["feature"])
    top10_overlap = gini_top10 & perm_top10

    safe_cut_list = (
        agreement_df[(agreement_df["importance"] < 0.01) & (agreement_df["perm_importance"] < 0.001)]
        .sort_values("avg_rank", ascending=False)["feature"].tolist()
    )
    batch1 = composite_df.loc[composite_df["tier"].str.contains("TIER 4"), "feature"].tolist()
    batch2 = composite_df.loc[composite_df["tier"].str.contains("TIER 3"), "feature"].tolist()
    n_keep = int(composite_df["tier"].str.contains("TIER 1|TIER 2").sum())
    agreement = "strong" if rho > 0.8 else "moderate — some Gini bias possible" if rho > 0.5 else "weak — Gini likely biased"

    # ── Report ────────────────────────────────────────────────────────────────
    title(f"RF FEATURE DIAGNOSTICS — {len(features)} binary features")
    print(f"Gini from the final fit · permutation importance out-of-fold "
          f"({cv.get_n_splits()} folds × {n_repeats} repeats, {perm_s:.0f}s)")
    print("Score = 0.40 × norm Gini + 0.60 × norm permutation (AUC drop)")
    show_table(
        composite_df.merge(perm_df[["feature", "perm_std"]], on="feature")
        .set_index("composite_rank")
        .rename_axis("rank")
        .loc[:, ["feature", "importance", "gini_rank", "perm_importance", "perm_std",
                 "perm_rank", "composite_score", "tier"]]
        .rename(columns={"importance": "gini", "gini_rank": "gini #", "perm_importance": "perm",
                         "perm_std": "perm ±", "perm_rank": "perm #", "composite_score": "score"})
        .round({"gini": 4, "perm": 5, "perm ±": 5, "score": 3})
    )
    print(f"Agreement  Spearman ρ {rho:.3f} (p={p_val:.3g}, {agreement}) · top-10 overlap {len(top10_overlap)}/10")
    print(f"           Gini-only top-10: {fmt_list(sorted(gini_top10 - perm_top10))} · "
          f"perm-only top-10: {fmt_list(sorted(perm_top10 - gini_top10))}")
    print(f"Weak       Gini < 0.01: {len(gini_dead)} · perm ≤ 0: {len(perm_dead)} · "
          f"perm < 0.001: {len(perm_marginal)} · top {n_for_90} hold 90% of Gini")
    print(f"Cut        Batch 1 (TIER 4): {fmt_list(batch1)}")
    print(f"           Batch 2 (TIER 3): {fmt_list(batch2)}")
    print(f"           Low on both: {fmt_list(safe_cut_list)}")
    print(f"Next       keep ~{n_keep} (TIER 1–2); remove one batch at a time → re-run 15C → 15D")

    if show_plots:
        _plot_diagnostics(composite_df, gini_df, agreement_df, rho, len(features), top_n)

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


def _plot_diagnostics(
    composite_df: pd.DataFrame,
    gini_df: pd.DataFrame,
    agreement_df: pd.DataFrame,
    rho: float,
    n_features: int,
    top_n: int,
) -> None:
    """Gini-vs-permutation scatter, composite bars, and rank-agreement slope chart."""
    import matplotlib.pyplot as plt

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

    # Gini vs permutation scatter
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

    # Composite score bars
    ax = axes[1]
    bar_colors = [tier_color_map.get(t, "#95a5a6") for t in composite_df["tier"]]
    ax.barh(range(len(composite_df)), composite_df["composite_score"],
            color=bar_colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(composite_df)))
    ax.set_yticklabels(composite_df["feature"], fontsize=7)
    ax.set_xlabel("Composite Score  (0.40×Gini + 0.60×Perm)")
    ax.set_title("Feature Composite Ranking")
    ax.invert_yaxis()

    # Gini rank vs permutation rank (labels on moves > 10 ranks)
    ax = axes[2]
    ranks = agreement_df.set_index("feature")
    for _, row in composite_df.iterrows():
        color = tier_color_map.get(row["tier"], "#95a5a6")
        g_rank, p_rank = ranks.at[row["feature"], "gini_rank"], ranks.at[row["feature"], "perm_rank"]
        ax.plot([0, 1], [g_rank, p_rank], color=color, alpha=0.6, linewidth=1.5)
        ax.scatter([0], [g_rank], color=color, s=30, zorder=5)
        ax.scatter([1], [p_rank], color=color, s=30, zorder=5)
    for _, row in agreement_df[agreement_df["rank_diff"] > 10].iterrows():
        ax.annotate(row["feature"], (1.02, row["perm_rank"]), fontsize=6, alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Gini Rank", "Permutation Rank"])
    ax.set_ylabel("Rank (1 = best)")
    ax.set_title("Rank Agreement  (labelled: moved > 10 ranks)")
    ax.set_ylim(n_features + 1, 0)

    plt.tight_layout()
    plt.show()