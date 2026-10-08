"""
feature_research/model_training/ebm/ranking.py
===============================================
Feature-selection evidence built from the EBM term table: redundancy against
upstream stages, correlated pairs, and the weighted composite ranking with
tiers. Tiers are rank-based (relative), not significance tests.

Public API
----------
TIERS, TIER_COLORS, WEIGHTS, INTERACTION_BONUS
redundancy_table(terms, lr_features, rf_features, interacting) → DataFrame
correlated_pairs(X, terms, threshold)                         → DataFrame
composite_ranking(terms, sep_df, redundancy_df, interacting)  → DataFrame
tier_members(composite_df, tier)                              → list[str]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from feature_research.model_training.ebm.terms import LINEAR_R2, NEGLIGIBLE_RANGE

# (max composite rank, label, colour)
TIERS = (
    (15, "🟢 TIER 1 (keep)", "#2ecc71"),
    (25, "🟡 TIER 2 (likely keep)", "#f1c40f"),
    (40, "🟠 TIER 3 (test removal)", "#e67e22"),
    (np.inf, "🔴 TIER 4 (safe to cut)", "#e74c3c"),
)
TIER_COLORS = {label: color for _, label, color in TIERS}
TIER_3, TIER_4 = TIERS[2][1], TIERS[3][1]

# composite = Σ weight × min-max-normalised component (+ bonus if in an interaction)
WEIGHTS = {
    "norm_imp": 0.30,     # how much the EBM relies on it
    "norm_range": 0.20,   # does it move predictions?
    "norm_nonlin": 0.20,  # value over a linear model
    "norm_sep_r": 0.15,   # |point-biserial|
    "norm_sep_ks": 0.15,  # KS
}
INTERACTION_BONUS = 0.05


def _verdict(r2: float, score_range: float, is_ix: bool, upstream: bool, in_ix: bool, note: str) -> str:
    if score_range < NEGLIGIBLE_RANGE:
        return "💀 NEGLIGIBLE EFFECT — safe to cut"
    if is_ix:
        return "🔗 INTERACTION TERM"
    if np.isnan(r2):
        return f"➖ shape not assessed ({note})"
    if r2 > LINEAR_R2:
        if upstream and not in_ix:
            return "🔄 REDUNDANT — linear + already upstream"
        if upstream:
            return "⚠️ linear + upstream, but in an interaction — keep"
        return "📏 linear, not upstream — EBM is the only handler"
    return "✅ NON-LINEAR — EBM adds unique value"


def redundancy_table(
    terms: pd.DataFrame,
    lr_features: list[str],
    rf_features: list[str],
    interacting: set[str],
) -> pd.DataFrame:
    lr, rf = set(lr_features), set(rf_features)
    rows = []
    for t in terms.itertuples(index=False):
        is_ix = bool(t.is_interaction_term)
        in_lr = (not is_ix) and t.feature in lr
        in_rf = (not is_ix) and t.feature in rf
        in_ix = t.feature in interacting
        rows.append({
            "feature": t.feature,
            "r2": t.linearity_r2,
            "score_range": t.score_range,
            "in_lr": in_lr,
            "in_rf": in_rf,
            "in_interaction": in_ix,
            "is_interaction_term": is_ix,
            "verdict": _verdict(t.linearity_r2, t.score_range, is_ix, in_lr or in_rf, in_ix, t.note),
        })
    return pd.DataFrame(rows)


def correlated_pairs(X: pd.DataFrame, terms: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """|r| > threshold pairs; the lower-importance feature is the drop candidate."""
    cols = ["Feature 1", "Feature 2", "Correlation", "Keep (higher rank)", "Drop candidate"]
    corr = X.corr()
    rank = terms.set_index("feature")["imp_rank"]
    names = corr.columns
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = corr.iat[i, j]
            if abs(r) > threshold:
                a, b = names[i], names[j]
                keep, drop = (a, b) if rank.get(a, 999) < rank.get(b, 999) else (b, a)
                rows.append(dict(zip(cols, (a, b, float(r), keep, drop))))
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("Correlation", key=np.abs, ascending=False, kind="mergesort").reset_index(drop=True)


def _minmax(s: pd.Series) -> pd.Series:
    s = s.fillna(0.0).astype(float)
    span = s.max() - s.min()
    return (s - s.min()) / span if span > 0 else s * 0.0


def composite_ranking(
    terms: pd.DataFrame,
    sep_df: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    interacting: set[str],
) -> pd.DataFrame:
    sep = sep_df[["feature", "point_biserial", "ks_stat"]].copy()
    sep["class_sep_r"] = sep["point_biserial"].abs()
    sep = sep.rename(columns={"ks_stat": "class_sep_ks"})[["feature", "class_sep_r", "class_sep_ks"]]

    df = terms[["feature", "importance", "imp_rank", "linearity_r2", "score_range", "is_interaction_term"]].merge(
        sep, on="feature", how="left", validate="one_to_one",
    )
    df[["class_sep_r", "class_sep_ks"]] = df[["class_sep_r", "class_sep_ks"]].fillna(0.0)
    df["in_interaction"] = df["feature"].isin(interacting)

    df["norm_imp"] = _minmax(df["importance"])
    df["norm_range"] = _minmax(df["score_range"])
    df["norm_sep_r"] = _minmax(df["class_sep_r"])
    df["norm_sep_ks"] = _minmax(df["class_sep_ks"])
    df["nonlin_bonus"] = 1 - df["linearity_r2"].fillna(0.5)  # not assessed → neutral
    df["norm_nonlin"] = _minmax(df["nonlin_bonus"])

    df["composite_score"] = sum(w * df[c] for c, w in WEIGHTS.items())
    df.loc[df["in_interaction"], "composite_score"] += INTERACTION_BONUS

    df = df.merge(redundancy_df[["feature", "in_lr", "in_rf", "verdict"]], on="feature", how="left")
    df = df.sort_values("composite_score", ascending=False, kind="mergesort").reset_index(drop=True)
    df["composite_rank"] = np.arange(1, len(df) + 1)
    df["tier"] = df["composite_rank"].map(lambda r: next(label for cap, label, _ in TIERS if r <= cap))
    return df


def tier_members(composite_df: pd.DataFrame, tier: str) -> list[str]:
    return composite_df.loc[composite_df["tier"] == tier, "feature"].tolist()
