"""
feature_research/model_training/ebm/plots.py
=============================================
Six-panel EBM diagnostic figure. Display only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from feature_research.model_training.ebm.ranking import TIER_COLORS
from feature_research.model_training.ebm.terms import COMPLEX_R2, LINEAR_R2, NEGLIGIBLE_RANGE, WEAK_RANGE

_GREY = "#95a5a6"


def _n_for(cum_pct: pd.Series, p: float) -> int:
    return min(int((cum_pct < p).sum()) + 1, len(cum_pct))


def plot_diagnostics(terms: pd.DataFrame, composite_df: pd.DataFrame, X: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    tier_colors = [TIER_COLORS.get(t, _GREY) for t in composite_df["tier"]]

    # 1 · cumulative importance
    ax = axes[0, 0]
    cum = terms["imp_cumulative_pct"]
    ax.plot(np.arange(1, len(cum) + 1), cum, "b-o", markersize=3, linewidth=1.5)
    marks = []
    for p, color in ((0.80, "green"), (0.90, "orange"), (0.95, "red")):
        n = _n_for(cum, p)
        marks.append(f"{p:.0%}@{n}")
        ax.axhline(p, color=color, linestyle="--", alpha=0.7, label=f"{p:.0%}")
        ax.axvline(n, color=color, linestyle=":", alpha=0.5)
    ax.set(xlabel="Terms (ranked by importance)", ylabel="Cumulative importance",
           title="Cumulative importance\n" + ", ".join(marks))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2 · importance vs linearity (not-assessed shapes omitted)
    ax = axes[0, 1]
    ax.scatter(composite_df["linearity_r2"], composite_df["importance"],
               c=tier_colors, s=60, alpha=0.8, edgecolors="black", linewidth=0.5)
    for row in composite_df[~composite_df["is_interaction_term"]].head(10).itertuples():
        if not np.isnan(row.linearity_r2):
            ax.annotate(row.feature, (row.linearity_r2, row.importance), fontsize=6, alpha=0.8, rotation=10)
    ax.axvline(LINEAR_R2, color="red", linestyle="--", alpha=0.5, label=f"linear (R²={LINEAR_R2})")
    ax.set(xlabel="Shape linearity R² (vs bin midpoints)", ylabel="Importance",
           title="Importance vs linearity\n(right = linear → may be redundant)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3 · composite ranking
    ax = axes[0, 2]
    ax.barh(np.arange(len(composite_df)), composite_df["composite_score"],
            color=tier_colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(np.arange(len(composite_df)))
    ax.set_yticklabels(composite_df["feature"], fontsize=6)
    ax.set(xlabel="Composite score", title="Composite ranking (colour = tier)")
    ax.invert_yaxis()

    # 4 · effect magnitude
    ax = axes[1, 0]
    mag = terms.sort_values("score_range", kind="mergesort")
    colors = ["#e74c3c" if r < NEGLIGIBLE_RANGE else "#f1c40f" if r < WEAK_RANGE else "#2ecc71"
              for r in mag["score_range"]]
    ax.barh(np.arange(len(mag)), mag["score_range"], color=colors, edgecolor="black", linewidth=0.3)
    ax.set_yticks(np.arange(len(mag)))
    ax.set_yticklabels(mag["feature"], fontsize=6)
    ax.axvline(NEGLIGIBLE_RANGE, color="red", linestyle="--", alpha=0.7, label=f"negligible (<{NEGLIGIBLE_RANGE})")
    ax.axvline(WEAK_RANGE, color="orange", linestyle="--", alpha=0.7, label=f"weak (<{WEAK_RANGE})")
    ax.set(xlabel="Score range (logit)", title="Effect magnitude")
    ax.legend(fontsize=7)

    # 5 · correlation of the top 20 main effects
    ax = axes[1, 1]
    top = [f for f in composite_df.loc[~composite_df["is_interaction_term"], "feature"] if f in X.columns][:20]
    corr = X[top].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                mask=np.triu(np.ones_like(corr, dtype=bool), k=1), square=True, ax=ax,
                cbar_kws={"shrink": 0.8}, annot_kws={"size": 5})
    ax.set_title("Correlation (top 20 main effects)", fontsize=11)
    ax.tick_params(axis="both", labelsize=5)

    # 6 · linearity distribution
    ax = axes[1, 2]
    r2 = terms["linearity_r2"].dropna()
    ax.hist(r2, bins=20, color="#3498db", edgecolor="black", alpha=0.7)
    ax.axvline(LINEAR_R2, color="red", linestyle="--", linewidth=2, label=f"linear ({LINEAR_R2})")
    ax.axvline(COMPLEX_R2, color="orange", linestyle="--", linewidth=2, label=f"complex ({COMPLEX_R2})")
    ax.set(xlabel="Shape linearity R²", ylabel="Count",
           title=f"Linearity distribution\n{int((r2 > LINEAR_R2).sum())} linear, "
                 f"{int((r2 < COMPLEX_R2).sum())} complex, {int(terms['linearity_r2'].isna().sum())} not assessed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
