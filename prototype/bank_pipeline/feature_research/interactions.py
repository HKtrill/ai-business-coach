"""
feature_research/interactions.py
==================================
Mutual-information-based feature interaction discovery for the Glass Cascade
pipeline.

Strategy
--------
1. Rank all candidate features by their individual MI with the target.
2. Restrict the pair search to the top-30 features to keep runtime O(900)
   pairs worst-case rather than O(n²).
3. For each pair, discretise both features into a 5-bin quantile grid and
   encode the joint distribution as a single integer token
   ``bin_1 * 100 + bin_2``.
4. Compute ``MI(joint_token → y)`` and subtract ``MI(f1 → y) + MI(f2 → y)``
   to obtain an *interaction lift* — positive values indicate the pair carries
   supra-additive information not explained by either feature alone.

All MIs are Miller–Madow bias-corrected (see :func:`_mi`). The plug-in MI
estimate is inflated by roughly (cells − 1) / 2N, so without the correction a
joint token with many cells (e.g. day_of_week × month ≈ 50) gets a built-in
lift bonus over its parts regardless of any real interaction.

Provides
--------
- :func:`search_interactions_mi`      — full search; returns ranked DataFrame
- :func:`display_interaction_rankings` — console table + optional CSV persist
"""

from __future__ import annotations

from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

from feature_research.config import OUTPUT_DIR


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Number of quantile bins used when discretising a continuous feature for
#: MI estimation.  Kept small to avoid sparse joint cells.
_N_BINS: int = 5

#: Maximum number of candidate features ranked by individual MI before pair
#: enumeration begins.
_TOP_FEATURES_LIMIT: int = 30

#: Multiplier used in the joint-token hash ``bin_1 * _JOINT_HASH_SCALE + bin_2``.
#: Must exceed the maximum number of bins to avoid collisions.
_JOINT_HASH_SCALE: int = 100


# ---------------------------------------------------------------------------
# Core search
# ---------------------------------------------------------------------------

def search_interactions_mi(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    numeric_features: List[str],
    *,
    top_k: int = 25,
    max_pairs: int = 500,
) -> pd.DataFrame:
    """Search for feature interactions using mutual-information lift.

    Parameters
    ----------
    df:
        Fully preprocessed DataFrame.
    features:
        All candidate feature names (typically
        ``NUMERIC_FEATURES + CATEGORICAL_FEATURES``).
    target_col:
        Binary 0/1 target column name.
    numeric_features:
        Subset of *features* that are continuous — used to decide whether
        quantile discretisation is needed before computing MI.
    top_k:
        Number of top interactions returned in the output DataFrame.
    max_pairs:
        Hard cap on evaluated pairs.  When the number of valid pairs exceeds
        this value the first *max_pairs* are evaluated and the rest are
        skipped.  Keeps worst-case runtime predictable.

    Returns
    -------
    pd.DataFrame
        Up to *top_k* rows, one per interaction candidate, sorted by
        ``lift`` descending.  Columns:

        ``feature_1``, ``feature_2``
            The interacting pair.
        ``mi_joint``
            MI of the joint token with the target.
        ``mi_f1``, ``mi_f2``
            Individual feature MIs.
        ``mi_sum``
            Additive baseline ``mi_f1 + mi_f2``.
        ``lift``
            ``mi_joint - mi_sum``; positive ⟹ supra-additive interaction.
        ``lift_pct``
            Lift expressed as a percentage of ``mi_sum`` (NaN when
            ``mi_sum`` is 0, i.e. neither feature is informative alone).
    """
    numeric_set = set(numeric_features)
    y = df[target_col].values

    # ------------------------------------------------------------------
    # Step 1 — individual MIs
    # ------------------------------------------------------------------
    individual_mis: dict[str, float] = {}
    for feat in features:
        if df[feat].nunique() <= 1:
            continue  # constant feature — MI is always 0
        individual_mis[feat] = _mi(y, _discretise(df[feat], feat, numeric_set))

    top_feature_names = [
        feat
        for feat, _ in sorted(
            individual_mis.items(), key=lambda kv: kv[1], reverse=True
        )[:_TOP_FEATURES_LIMIT]
    ]

    # ------------------------------------------------------------------
    # Step 2 — enumerate pairs
    # ------------------------------------------------------------------
    pairs = list(combinations(top_feature_names, 2))
    n_possible = len(pairs)
    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]


    # ------------------------------------------------------------------
    # Step 3 — compute joint MI and lift per pair
    # ------------------------------------------------------------------
    results = []
    for idx, (f1, f2) in enumerate(pairs, start=1):
        f1_binned = _discretise(df[f1], f1, numeric_set)
        f2_binned = _discretise(df[f2], f2, numeric_set)

        joint = f1_binned * _JOINT_HASH_SCALE + f2_binned
        mi_joint = _mi(y, joint)

        mi_f1 = individual_mis[f1]
        mi_f2 = individual_mis[f2]
        mi_sum = mi_f1 + mi_f2

        # Lift is defined even when neither feature carries signal alone —
        # that is the purest interaction. Only the percentage needs mi_sum > 0.
        lift = mi_joint - mi_sum
        lift_pct = (lift / mi_sum * 100.0) if mi_sum > 0 else float("nan")

        results.append(
            {
                "feature_1": f1,
                "feature_2": f2,
                "mi_joint": mi_joint,
                "mi_f1": mi_f1,
                "mi_f2": mi_f2,
                "mi_sum": mi_sum,
                "lift": lift,
                "lift_pct": lift_pct,
            }
        )


    df_interactions = (
        pd.DataFrame(results)
        .sort_values("lift", ascending=False)
        .reset_index(drop=True)
    )
    df_interactions.attrs["summary"] = (
        f"{len(top_feature_names)} features, {len(pairs)} pairs evaluated"
        + (f" (capped from {n_possible})" if n_possible > len(pairs) else "")
        + f", {len(df):,} rows"
    )
    return df_interactions.head(top_k)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def display_interaction_rankings(
    df_interactions: pd.DataFrame,
    top_n: int = 25,
    save_csv: bool = True,
) -> None:
    """Print one compact interaction table and optionally persist it.

    Lift = MI(pair → y) − MI(f1 → y) − MI(f2 → y), Miller–Madow corrected.
    Positive lift means the pair carries information neither feature has
    alone. Writes the table to ``research_logs/interaction_rankings.csv``.
    """
    rows = df_interactions.head(top_n)
    summary = df_interactions.attrs.get("summary", "")

    print(f"\n{'=' * 78}")
    print(f"INTERACTION DISCOVERY — top {len(rows)} by MI lift")
    if summary:
        print(f"  {summary}")
    print(f"{'=' * 78}")
    print(f"  {'#':>2}  {'Pair':<36} {'Joint MI':>8} {'Sum MI':>8} {'Lift':>8} {'Lift %':>7}")
    print(f"  {'-' * 2}  {'-' * 36} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7}")

    for rank, (_, r) in enumerate(rows.iterrows(), start=1):
        pair = f"{r['feature_1']} × {r['feature_2']}"
        pct = "—" if pd.isna(r["lift_pct"]) else f"{r['lift_pct']:+.0f}%"
        print(f"  {rank:>2}  {pair:<36} {r['mi_joint']:>8.4f} {r['mi_sum']:>8.4f} "
              f"{r['lift']:>+8.4f} {pct:>7}")

    if save_csv:
        csv_path = OUTPUT_DIR / "interaction_rankings.csv"
        df_interactions.to_csv(csv_path, index=False)
        print(f"\n  Full table → {csv_path.name}")
    print(f"{'=' * 78}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mi(y: np.ndarray, x: np.ndarray) -> float:
    """Miller–Madow bias-corrected MI (nats) between binary y and discrete x.

    Subtracts (K_x − 1)(K_y − 1) / 2N, where K are the observed category
    counts, and clips at 0.
    """
    k_x = len(np.unique(x))
    k_y = len(np.unique(y))
    correction = (k_x - 1) * (k_y - 1) / (2.0 * len(x))
    return max(0.0, float(mutual_info_score(y, x)) - correction)


def _discretise(series: pd.Series, feat: str, numeric_set: set[str]) -> np.ndarray:
    """Return a 1-D integer array suitable for :func:`mutual_info_score`.

    Continuous features are quantile-binned into at most :data:`_N_BINS`
    buckets; tied quantile edges (common for campaign, previous, and the
    macro indicators) are merged instead of producing zero-width bins.
    Categorical features are passed through as-is (already integer-encoded).
    """
    if feat not in numeric_set:
        return series.values.astype(int)

    return quantile_bin_codes(series, min(_N_BINS, series.nunique()))


def quantile_bin_codes(series: pd.Series, n_bins: int) -> np.ndarray:
    """Integer bin codes from up to `n_bins` quantile bins, robust to ties.

    Interior quantiles are used as cut points with open outer bins, so a
    heavily tied value (e.g. previous == 0 for 86 % of rows) forms its own
    bin instead of every edge collapsing into a single bin.
    """
    x = series.to_numpy(dtype=float)
    interior = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.searchsorted(interior, x, side="left").astype(int)
