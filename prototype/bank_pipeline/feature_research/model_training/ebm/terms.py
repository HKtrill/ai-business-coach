"""
feature_research/model_training/ebm/terms.py
=============================================
Reads a fitted EBM: one row per term (main effect or pairwise interaction)
with importance, shape linearity and effect size. No refitting.

Linearity
---------
R² of a straight-line fit of the shape scores against bin midpoints in
feature units (not bin index: bins are not evenly spaced). Not assessed
(NaN) for interactions, categorical shapes and shapes with ≤ 2 bins.

Public API
----------
LINEAR_R2, COMPLEX_R2, NEGLIGIBLE_RANGE, WEAK_RANGE   thresholds
term_table(model)          → DataFrame (one row per term, importance order)
split_tables(terms)        → importance_df, linearity_df, magnitude_df, interaction_df
interacting_features(terms) → set of features used in any interaction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

LINEAR_R2 = 0.90         # shape ≈ straight line → LR could carry it
COMPLEX_R2 = 0.40        # shape far from linear → EBM-specific signal
NEGLIGIBLE_RANGE = 0.01  # logit range below this barely moves predictions
WEAK_RANGE = 0.03


def _linearity(edges, scores: np.ndarray) -> tuple[float, str]:
    """R² of scores vs bin midpoints, plus a note when not assessed."""
    if scores.size and np.std(scores) < 1e-10:
        return 1.0, "constant"
    if scores.size <= 2:
        return np.nan, "≤2 bins"
    try:
        edges = np.asarray(edges, dtype=float)
    except (TypeError, ValueError):
        return np.nan, "categorical"
    if edges.ndim != 1 or edges.size != scores.size + 1:
        return np.nan, "categorical"
    x = (edges[:-1] + edges[1:]) / 2
    slope, intercept = np.polyfit(x, scores, 1)
    resid = scores - (slope * x + intercept)
    return float(max(0.0, 1.0 - resid.var() / scores.var())), ""


def term_table(model: ExplainableBoostingClassifier) -> pd.DataFrame:
    """One row per term: importance (mean |contribution|), shape stats, members."""
    expl = model.explain_global()
    feature_names = list(model.feature_names_in_)
    importances = np.asarray(model.term_importances(), dtype=float)

    rows = []
    for i, name in enumerate(model.term_names_):
        members = tuple(feature_names[j] for j in model.term_features_[i])
        data = expl.data(i)
        scores = np.asarray(data["scores"], dtype=float)
        is_ix = len(members) > 1
        r2, note = (np.nan, "interaction") if is_ix else _linearity(data.get("names"), scores.ravel())
        rows.append({
            "feature": name,
            "members": members,
            "is_interaction_term": is_ix,
            "importance": float(importances[i]),
            "linearity_r2": r2,
            "score_range": float(np.ptp(scores)),
            "score_std": float(np.std(scores)),
            "n_bins": int(scores.size),
            "note": note,
        })

    terms = (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    terms["imp_rank"] = np.arange(1, len(terms) + 1)
    terms["imp_cumulative"] = terms["importance"].cumsum()
    terms["imp_cumulative_pct"] = terms["imp_cumulative"] / terms["importance"].sum()
    return terms


def split_tables(terms: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The per-section tables returned by ebm_diagnostics (column names unchanged)."""
    importance_df = terms[["feature", "importance", "imp_rank", "imp_cumulative", "imp_cumulative_pct"]].copy()

    linearity_df = (
        terms[["feature", "linearity_r2", "score_range", "score_std", "n_bins", "note"]]
        .sort_values("linearity_r2", ascending=False, kind="mergesort", na_position="last")
        .reset_index(drop=True)
    )

    magnitude_df = (
        terms[["feature", "score_range"]]
        .sort_values("score_range", ascending=False, kind="mergesort")
        .reset_index(drop=True)
    )
    magnitude_df["mag_rank"] = np.arange(1, len(magnitude_df) + 1)

    ix = terms[terms["is_interaction_term"]]
    interaction_df = pd.DataFrame({
        "interaction": ix["feature"].to_numpy(),
        "feat_1": [m[0] for m in ix["members"]],
        "feat_2": [m[1] for m in ix["members"]],
        "importance": ix["importance"].to_numpy(),
    })

    return {
        "importance_df": importance_df,
        "linearity_df": linearity_df,
        "magnitude_df": magnitude_df,
        "interaction_df": interaction_df,
    }


def interacting_features(terms: pd.DataFrame) -> set[str]:
    return {f for m in terms.loc[terms["is_interaction_term"], "members"] for f in m}
