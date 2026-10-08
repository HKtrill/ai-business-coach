"""
feature_research/model_training/ebm/diagnostics.py
===================================================
EBM-native diagnostics. Interrogates a fitted EBMResult; nothing is refit.
This ranking is how the GLASS Stage 3 feature set was selected.

result.pipe is fit on all training rows, so everything here is in-sample
evidence for feature selection, not a performance estimate (OOF metrics live
in the trainer).

Sections (one compact block + one table + one figure)
--------
importance (main effects + interactions) · shape linearity (R² vs bin
midpoints) · effect size · interactions · upstream redundancy (LR/RF) ·
correlated pairs · composite ranking → tiers · copy-paste EBM_FEATURES

sep_df: raw compute_all_separations() output (feature, point_biserial,
ks_stat); |point_biserial| is derived here.

Return dict keys
----------------
importance_df, linearity_df, magnitude_df, interaction_df,
redundancy_df, corr_df, composite_df, batch1, safe_cut_list
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_research.model_training.ebm.plots import plot_diagnostics
from feature_research.model_training.ebm.ranking import (
    TIER_3,
    TIER_4,
    composite_ranking,
    correlated_pairs,
    redundancy_table,
    tier_members,
)
from feature_research.model_training.ebm.terms import (
    COMPLEX_R2,
    LINEAR_R2,
    NEGLIGIBLE_RANGE,
    WEAK_RANGE,
    interacting_features,
    split_tables,
    term_table,
)
from feature_research.model_training.ebm.trainer import EBMResult
from feature_research.model_training.shared.report import fmt_list, show_table, title

_SEP_COLUMNS = {"feature", "point_biserial", "ks_stat"}
_TABLE_COLUMNS = [
    "composite_rank", "feature", "tier", "composite_score", "imp_rank",
    "linearity_r2", "score_range", "class_sep_r", "class_sep_ks", "verdict",
]


def _check_inputs(result: EBMResult, X: pd.DataFrame, sep_df: pd.DataFrame) -> None:
    if list(X.columns) != list(result.features):
        raise ValueError("X columns must equal result.features (same names, same order)")
    if list(result.pipe.feature_names_in_) != list(result.features):
        raise ValueError("result.pipe was not fit on result.features")
    missing = _SEP_COLUMNS - set(sep_df.columns)
    if missing:
        raise KeyError(f"sep_df missing columns {sorted(missing)}")
    if not sep_df["feature"].is_unique:
        raise ValueError("sep_df has duplicate feature rows")


def _report(
    X: pd.DataFrame,
    terms: pd.DataFrame,
    redundancy_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    composite_df: pd.DataFrame,
    lr_features: list[str],
    rf_features: list[str],
    interacting: set[str],
    batch1: list[str],
    safe_cut: list[str],
) -> None:
    ix = terms["is_interaction_term"]
    main = terms[~ix]
    cum = terms["imp_cumulative_pct"]
    n_for = {p: min(int((cum < p).sum()) + 1, len(terms)) for p in (0.80, 0.90, 0.95)}
    ix_share = terms.loc[ix, "importance"].sum() / terms["importance"].sum()

    title("Stage 3 · EBM diagnostics (in-sample, all training rows)")
    print(f"Terms      {len(main)} main effects · {int(ix.sum())} interactions ({ix_share:.1%} of importance) · {len(X):,} rows")
    print(f"Importance 80% in top {n_for[0.80]} · 90% top {n_for[0.90]} · 95% top {n_for[0.95]} · "
          f"tail: {fmt_list(terms.loc[cum > 0.95, 'feature'])}")

    r2 = main["linearity_r2"]
    print(f"Shape R²   linear (>{LINEAR_R2}): {fmt_list(main.loc[r2 > LINEAR_R2, 'feature'])}")
    print(f"           complex (<{COMPLEX_R2}): {fmt_list(main.loc[r2 < COMPLEX_R2, 'feature'])}")
    if r2.isna().any():
        na = main[r2.isna()]
        print(f"           not assessed: {fmt_list(f'{f} ({n})' for f, n in zip(na['feature'], na['note']))}")

    rng = terms["score_range"]
    print(f"Effect     negligible (<{NEGLIGIBLE_RANGE}): {fmt_list(terms.loc[rng < NEGLIGIBLE_RANGE, 'feature'])}")
    print(f"           weak (<{WEAK_RANGE}): {fmt_list(terms.loc[(rng >= NEGLIGIBLE_RANGE) & (rng < WEAK_RANGE), 'feature'])}")

    ix_terms = terms[ix]
    print(f"Pairs      {fmt_list(f'{f} ({v:.4f})' for f, v in zip(ix_terms['feature'], ix_terms['importance']))}")
    if interacting:
        print(f"           members (kept regardless of main effect): {fmt_list(sorted(interacting))}")

    names = set(main["feature"])
    lr_overlap, rf_overlap = names & set(lr_features), names & set(rf_features)
    print(f"Upstream   shared with LR {len(lr_overlap)}/{len(lr_features)} · with RF {len(rf_overlap)}/{len(rf_features)}")
    if rf_features and not rf_overlap:
        print("           ⚠️ no RF overlap — pass RF source features, not bin columns")
    redundant = redundancy_df.loc[redundancy_df["verdict"].str.contains("REDUNDANT"), "feature"]
    print(f"Redundant  linear + upstream: {fmt_list(redundant)}")

    if corr_df.empty:
        print("Correlated none with |r| > 0.7")
    else:
        pairs = (f"{a} ↔ {b} {r:+.2f} (drop {d})" for a, b, r, d in
                 zip(corr_df["Feature 1"], corr_df["Feature 2"], corr_df["Correlation"], corr_df["Drop candidate"]))
        print(f"Correlated {fmt_list(pairs, limit=6)}")

    print("\nComposite = 0.30 importance + 0.20 range + 0.20 non-linearity + 0.15 |r| + 0.15 KS (+0.05 in interaction)")
    show_table(composite_df[_TABLE_COLUMNS].set_index("composite_rank").round(4))

    print(f"\nTier 3 (test removal): {fmt_list(batch1, limit=20)}")
    print(f"Tier 4 (safe to cut):  {fmt_list(safe_cut, limit=20)}")

    cuts = set(batch1) | set(safe_cut)
    if cuts:
        info = composite_df.set_index("feature")
        kept = [f for f in X.columns if f not in cuts]
        print(f"\n# EBM_FEATURES without Tier 3–4 ({len(kept)} features)")
        print("EBM_FEATURES = [")
        for f in kept:
            flag = "  🔗" if f in interacting else ""
            print(f"    {f!r},  # #{int(info.at[f, 'composite_rank'])} {info.at[f, 'tier']}{flag}")
        print("]")


def ebm_diagnostics(
    result: EBMResult,
    X: pd.DataFrame,
    sep_df: pd.DataFrame,
    lr_features: list[str],
    rf_features: list[str],
    plot: bool = True,
) -> dict[str, Any]:
    """
    result      EBMResult from train_ebm (pipe fit on all training rows)
    X           model input for that fit: df_engineered[EBM_FEATURES]
    sep_df      raw compute_all_separations() output
    lr_features LR_FEATURES (redundancy check)
    rf_features RF source features (redundancy check; bin column names never match)
    """
    _check_inputs(result, X, sep_df)

    terms = term_table(result.pipe)
    tables = split_tables(terms)
    interacting = interacting_features(terms)
    redundancy_df = redundancy_table(terms, lr_features, rf_features, interacting)
    corr_df = correlated_pairs(X, terms)
    composite_df = composite_ranking(terms, sep_df, redundancy_df, interacting)
    batch1 = tier_members(composite_df, TIER_3)
    safe_cut = tier_members(composite_df, TIER_4)

    _report(X, terms, redundancy_df, corr_df, composite_df,
            list(lr_features), list(rf_features), interacting, batch1, safe_cut)
    if plot:
        plot_diagnostics(terms, composite_df, X)

    return {
        **tables,
        "redundancy_df": redundancy_df,
        "corr_df": corr_df,
        "composite_df": composite_df,
        "batch1": batch1,
        "safe_cut_list": safe_cut,
    }
