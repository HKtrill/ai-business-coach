"""
model_training.rf.lift_analysis
===============================
Decile lift of the continuous RF source features, used to find the bin
thresholds locked in binning.py. Descriptive only: no model is fit.

Run it on training rows (df_rf from Cell 10G). Lift for target-derived sources
(NSD, ECI, dow_month_encoded, …) is measured on encodings fit on those same
rows, so it reads higher than it would out-of-fold. Keep that in mind when
re-deriving thresholds at PR 39.

Importance analysis lives in rf_diagnostics (out-of-fold, on the binary space
the RF is actually trained on).

Public API
----------
compute_lift(df, features, target_col, n_deciles)  → dict[str, pd.DataFrame]
rf_lift_analysis(df, features, target_col, ...)    → dict  (orchestrator)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from feature_research.config import FIG_DIR
from feature_research.model_training.shared.report import in_notebook, title


# ── Public helpers ────────────────────────────────────────────────────────────

def compute_lift(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_deciles: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Conversion-rate lift by decile for each continuous feature.

    Features with fewer than `n_deciles` unique values are binned by value
    instead of quantile. A feature that cannot be binned is skipped with a
    printed warning.

    Returns
    -------
    dict: feature → DataFrame [bin, count, conversions, conv_rate, lift, feature],
    where lift = bin conversion rate / overall conversion rate.
    """
    overall_rate = df[target_col].mean()
    results: dict[str, pd.DataFrame] = {}

    for feat in features:
        tmp = pd.DataFrame({"val": df[feat], "y": df[target_col]})
        n_unique = tmp["val"].nunique()
        try:
            if n_unique < n_deciles:
                tmp["bin"] = tmp["val"]
            else:
                tmp["bin"] = pd.qcut(tmp["val"], q=n_deciles, duplicates="drop")
        except Exception as exc:
            print(f"   ⚠️  compute_lift: skipped '{feat}' ({type(exc).__name__}: {exc})")
            continue

        lift_df = (
            tmp.groupby("bin", observed=True)
            .agg(count=("y", "count"), conversions=("y", "sum"), conv_rate=("y", "mean"))
            .reset_index()
        )
        lift_df["lift"] = lift_df["conv_rate"] / overall_rate
        lift_df["feature"] = feat
        results[feat] = lift_df

    return results


# ── Private display / plot helpers ────────────────────────────────────────────

def _mark(lift: float) -> str:
    return "▲" if lift > 1.5 else "▼" if lift < 0.7 else " "


def _print_lift(lift_results: dict[str, pd.DataFrame]) -> None:
    """One line per feature: decile lifts low → high value (value bins for low-cardinality features)."""
    print("Decile lift, low → high value (▲ > 1.5×, ▼ < 0.7×)")
    rows = []
    for feat, ldf in lift_results.items():
        if isinstance(ldf["bin"].iloc[0], pd.Interval):
            rows.append((feat, " ".join(f"{l:5.2f}{_mark(l)}" for l in ldf["lift"])))
        else:
            cells = " · ".join(f"{b}: {l:.2f}{_mark(l)}".rstrip() for b, l in zip(ldf["bin"], ldf["lift"]))
            rows.append((f"{feat} (by value)", cells))
    width = max((len(label) for label, _ in rows), default=0) + 2
    for label, cells in rows:
        print(f"  {label:<{width}}{cells}")


def _plot_lift_curves(
    lift_results: dict[str, pd.DataFrame],
    overall_rate: float,
) -> None:
    """Grid of per-decile conversion-rate bars: saved to FIG_DIR/rf_lift_curves.png and shown in a notebook."""
    if not lift_results:
        return

    n_feats = len(lift_results)
    n_cols = 3
    n_rows = (n_feats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
    axes = axes.flatten() if n_feats > 1 else [axes]
    fig.suptitle(
        "Conversion Rate by Decile — Lift Analysis\n"
        "(Green=High Lift >1.5x, Red=Low Lift <0.7x)",
        fontsize=13, fontweight="bold",
    )

    for idx, (feat, ldf) in enumerate(lift_results.items()):
        ax = axes[idx]
        bars = ax.bar(range(len(ldf)), ldf["conv_rate"], color="steelblue", alpha=0.7)
        for bar, lift in zip(bars, ldf["lift"]):
            bar.set_color("darkgreen" if lift > 1.5 else "darkred" if lift < 0.7 else "steelblue")

        ax.axhline(overall_rate, color="red", linestyle="--", linewidth=2,
                   label=f"Overall ({overall_rate:.3f})")
        ax.set_xticks(range(len(ldf)))
        ax.set_xticklabels([str(b)[:15] for b in ldf["bin"]], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Conversion Rate")
        ax.set_title(
            f"{feat}\n(Max: {ldf['lift'].max():.2f}x, Min: {ldf['lift'].min():.2f}x)",
            fontweight="bold", fontsize=9,
        )
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    for idx in range(len(lift_results), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    save_path = FIG_DIR / "rf_lift_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Lift curves saved → {save_path}")
    if in_notebook():
        plt.show()
    else:
        plt.close(fig)


# ── Orchestrator ──────────────────────────────────────────────────────────────

def rf_lift_analysis(
    df: pd.DataFrame,
    features: list[str],
    target_col: str,
    n_deciles: int = 10,
) -> dict:
    """
    Decile lift for the continuous RF source features: one line per feature,
    plus the lift-curve figure (saved, and shown in a notebook).

    Parameters
    ----------
    df         : df_rf (features + target_col, training rows).
    features   : RF_FEATURES — the 8 continuous source features from 10G.
    target_col : TARGET_COL.
    n_deciles  : Quantile bins per feature.

    Returns
    -------
    dict: overall_rate, lift_results.
    """
    overall_rate = float(df[target_col].mean())
    lift_results = compute_lift(df, features, target_col, n_deciles)

    title(f"RF LIFT ANALYSIS — {len(features)} continuous sources · base rate {overall_rate:.3f}")
    _print_lift(lift_results)
    _plot_lift_curves(lift_results, overall_rate)

    return {
        "overall_rate": overall_rate,
        "lift_results": lift_results,
    }