"""
feature_research/visualization.py
====================================
Matplotlib / Seaborn visualisations for feature-separation analysis.

Provides one plot function per feature type and a top-N orchestrator that
saves all figures to :data:`~feature_research.config.FIG_DIR`.

Provides
--------
- :func:`plot_numeric_feature`     — violin + histogram overlay + ECDF (3 panels)
- :func:`plot_categorical_feature` — target-rate with Wilson CI + class counts (2 panels)
- :func:`generate_all_plots`       — iterates ranked features and dispatches to above
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from feature_research.config import FIG_DIR
from feature_research.metrics import wilson_ci


# ---------------------------------------------------------------------------
# Numeric
# ---------------------------------------------------------------------------

def plot_numeric_feature(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Three-panel separation plot for a numeric feature.

    Panels
    ------
    1. **Violin** — distribution shape and spread by class.
    2. **Histogram overlay** — normalised density for direct comparison.
    3. **ECDF** — empirical CDF curves; vertical gap at any point equals the
       KS statistic.

    Parameters
    ----------
    df:
        Training rows (features + target).
    feature:
        Numeric column to visualise.
    target_col:
        Binary 0/1 target column name.
    save_path:
        If provided the figure is saved here.
    show:
        Display the figure inline (always true when *save_path* is None).
    """
    class_0 = df.loc[df[target_col] == 0, feature]
    class_1 = df.loc[df[target_col] == 1, feature]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(feature, fontsize=13, fontweight="bold")

    # Panel 1 — Violin -------------------------------------------------------
    axes[0].violinplot(
        [class_0, class_1],
        positions=[0, 1],
        showmeans=True,
        showmedians=True,
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["NOT_SUBSCRIBE", "SUBSCRIBE"])
    axes[0].set_ylabel(feature)
    axes[0].set_title("Distribution by Class")
    axes[0].grid(True, alpha=0.3)

    # Panel 2 — Histogram overlay --------------------------------------------
    axes[1].hist(class_0, bins=30, alpha=0.5, label="NOT_SUBSCRIBE", density=True, color="C0")
    axes[1].hist(class_1, bins=30, alpha=0.5, label="SUBSCRIBE",     density=True, color="C1")
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Density")
    axes[1].set_title("Histogram Overlay (Normalised)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3 — ECDF ---------------------------------------------------------
    c0_sorted = np.sort(class_0.values)
    c1_sorted = np.sort(class_1.values)
    ecdf_0 = np.arange(1, len(c0_sorted) + 1) / len(c0_sorted)
    ecdf_1 = np.arange(1, len(c1_sorted) + 1) / len(c1_sorted)

    axes[2].plot(c0_sorted, ecdf_0, label="NOT_SUBSCRIBE", alpha=0.7)
    axes[2].plot(c1_sorted, ecdf_1, label="SUBSCRIBE",     alpha=0.7)
    axes[2].set_xlabel(feature)
    axes[2].set_ylabel("ECDF")
    axes[2].set_title("Empirical CDF")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# Categorical
# ---------------------------------------------------------------------------

def plot_categorical_feature(
    df: pd.DataFrame,
    feature: str,
    target_col: str,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Two-panel separation plot for a categorical feature.

    Panels
    ------
    1. **Target rate** — bar chart of P(SUBSCRIBE) per category with Wilson
       score confidence intervals and the dataset-wide baseline as a red
       dashed reference line.
    2. **Class counts** — grouped bar chart showing absolute class frequencies
       per category.

    Parameters
    ----------
    df:
        Training rows (features + target).
    feature:
        Categorical column to visualise.
    target_col:
        Binary 0/1 target column name.
    save_path:
        If provided the figure is saved here.
    show:
        Display the figure inline (always true when *save_path* is None).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(feature, fontsize=13, fontweight="bold")

    # Aggregate per category -------------------------------------------------
    grouped = (
        df.groupby(feature)[target_col]
        .agg(successes="sum", trials="count", rate="mean")
        .sort_values("rate", ascending=False)
    )
    categories = grouped.index.tolist()
    target_rates = grouped["rate"].values
    x_pos = np.arange(len(categories))

    # Wilson CIs — clip error bars to be strictly non-negative
    cis = [wilson_ci(int(r.successes), int(r.trials)) for r in grouped.itertuples()]
    yerr_lower = np.maximum(0.0, target_rates - np.array([ci[0] for ci in cis]))
    yerr_upper = np.maximum(0.0, np.array([ci[1] for ci in cis]) - target_rates)

    # Panel 1 — Target rate with CI ------------------------------------------
    axes[0].bar(x_pos, target_rates, alpha=0.6, color="steelblue")
    axes[0].errorbar(
        x_pos, target_rates,
        yerr=[yerr_lower, yerr_upper],
        fmt="none", ecolor="black", capsize=5, alpha=0.7,
    )
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(categories, rotation=45, ha="right")
    axes[0].set_ylabel("P(SUBSCRIBE)")
    axes[0].set_title(f"Target Rate by {feature}")
    axes[0].axhline(
        df[target_col].mean(),
        color="red", linestyle="--", alpha=0.5, label="Overall rate",
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # Panel 2 — Class counts -------------------------------------------------
    crosstab = pd.crosstab(df[feature], df[target_col])
    crosstab.plot(kind="bar", stacked=False, ax=axes[1], alpha=0.7)
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Class Distribution by {feature}")
    axes[1].legend(["NOT_SUBSCRIBE", "SUBSCRIBE"])
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _save_or_show(fig, save_path, show)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_plots(
    df: pd.DataFrame,
    df_metrics: pd.DataFrame,
    target_col: str,
    top_n: int = 17,
    show: bool = False,
) -> None:
    """Save separation plots for the top-ranked features.

    Dispatches to :func:`plot_numeric_feature` or
    :func:`plot_categorical_feature` by the ``type`` column in *df_metrics*
    and saves each figure to :data:`~feature_research.config.FIG_DIR` with a
    zero-padded rank prefix (e.g. ``01_euribor3m.png``).

    Parameters
    ----------
    df:
        Training rows (features + target). Never pass test rows — the plots
        are split by class.
    df_metrics:
        Output of :func:`~feature_research.separation.compute_all_separations`.
    target_col:
        Binary 0/1 target column name.
    top_n:
        Number of top-ranked features to plot.
    show:
        Also display each figure inline (default False — files only).
    """
    ranked = df_metrics.head(top_n)
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        save_path = FIG_DIR / f"{rank:02d}_{row['feature']}.png"
        if row["type"] == "numeric":
            plot_numeric_feature(df, row["feature"], target_col, save_path, show=show)
        else:
            plot_categorical_feature(df, row["feature"], target_col, save_path, show=show)

    n_num = int((ranked["type"] == "numeric").sum())
    print(f"\n{'=' * 78}")
    print(f"SEPARATION PLOTS — top {len(ranked)} features "
          f"({n_num} numeric, {len(ranked) - n_num} categorical), {len(df):,} rows")
    print(f"{'=' * 78}")
    print(f"  Saved {len(ranked)} figures → {FIG_DIR}")
    print(f"  Numeric: violin · density overlay · ECDF   |   Categorical: rate ± Wilson CI · class counts")
    print(f"{'=' * 78}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_or_show(fig: plt.Figure, save_path: Optional[Path], show: bool = False) -> None:
    """Save *fig* to *save_path*; display it if requested (or if not saved)."""
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show or save_path is None:
        plt.show()
    else:
        plt.close(fig)