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
from sklearn.preprocessing import KBinsDiscretizer

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
            Lift expressed as a percentage of ``mi_sum``.
    """
    print("\n" + "=" * 80)
    print("🔍 INTERACTION DISCOVERY  (Mutual Information)")
    print("=" * 80)

    numeric_set = set(numeric_features)
    y = df[target_col].values

    # ------------------------------------------------------------------
    # Step 1 — individual MIs
    # ------------------------------------------------------------------
    print("\n📊 Computing individual MIs...")

    individual_mis: dict[str, float] = {}
    for feat in features:
        if df[feat].nunique() <= 1:
            continue  # constant feature — MI is always 0
        individual_mis[feat] = mutual_info_score(
            y, _discretise(df[feat], feat, numeric_set)
        )

    top_feature_names = [
        feat
        for feat, _ in sorted(
            individual_mis.items(), key=lambda kv: kv[1], reverse=True
        )[:_TOP_FEATURES_LIMIT]
    ]
    print(f"✅ Selected top {len(top_feature_names)} features for interaction search")

    # ------------------------------------------------------------------
    # Step 2 — enumerate pairs
    # ------------------------------------------------------------------
    pairs = list(combinations(top_feature_names, 2))
    if len(pairs) > max_pairs:
        print(f"⚠️  Limiting to {max_pairs} pairs (from {len(pairs)} possible)")
        pairs = pairs[:max_pairs]

    print(f"\n🔍 Evaluating {len(pairs)} pairs...")

    # ------------------------------------------------------------------
    # Step 3 — compute joint MI and lift per pair
    # ------------------------------------------------------------------
    results = []
    for idx, (f1, f2) in enumerate(pairs, start=1):
        if idx % 50 == 0:
            print(f"  Progress: {idx}/{len(pairs)}", end="\r")

        f1_binned = _discretise(df[f1], f1, numeric_set)
        f2_binned = _discretise(df[f2], f2, numeric_set)

        joint = f1_binned * _JOINT_HASH_SCALE + f2_binned
        mi_joint = mutual_info_score(y, joint)

        mi_f1 = individual_mis[f1]
        mi_f2 = individual_mis[f2]
        mi_sum = mi_f1 + mi_f2

        lift = mi_joint - mi_sum if mi_sum > 0 else 0.0
        lift_pct = (lift / mi_sum * 100.0) if mi_sum > 0 else 0.0

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

    print(f"\n✅ Evaluated {len(pairs)} pairs")

    df_interactions = (
        pd.DataFrame(results)
        .sort_values("lift", ascending=False)
        .reset_index(drop=True)
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
    """Print a formatted interaction ranking table and optionally save to CSV.

    Parameters
    ----------
    df_interactions:
        Output of :func:`search_interactions_mi`.
    top_n:
        Number of rows to display.
    save_csv:
        When ``True`` (default) writes the full table to
        ``research_logs/interaction_rankings.csv``.
    """
    print("\n" + "=" * 80)
    print(f"🏆 TOP {top_n} INTERACTION CANDIDATES")
    print("=" * 80)

    for rank, (_, row) in enumerate(df_interactions.head(top_n).iterrows(), start=1):
        print(f"\n{rank:2d}. {row['feature_1']}  ×  {row['feature_2']}")
        print(
            f"    Joint MI : {row['mi_joint']:.4f}  |  "
            f"Sum MI : {row['mi_sum']:.4f}  |  "
            f"Lift : {row['lift']:.4f}  ({row['lift_pct']:.1f}%)"
        )

    if save_csv:
        csv_path = OUTPUT_DIR / "interaction_rankings.csv"
        df_interactions.to_csv(csv_path, index=False)
        print(f"\n💾 Saved interactions to: {csv_path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _discretise(series: pd.Series, feat: str, numeric_set: set[str]) -> np.ndarray:
    """Return a 1-D integer array suitable for :func:`mutual_info_score`.

    Continuous features are quantile-binned into :data:`_N_BINS` buckets.
    Categorical features are passed through as-is (already integer-encoded).
    """
    if feat not in numeric_set:
        return series.values.astype(int)

    n_bins = min(_N_BINS, series.nunique())
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    return disc.fit_transform(series.values.reshape(-1, 1)).ravel().astype(int)