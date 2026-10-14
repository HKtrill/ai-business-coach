"""
blackbox_pipeline.models.rf_router.reporting
=============================================
Training-OOF diagnostics and figures for Stage 2, moved out of the notebook.

Formerly notebook Cells 26 (threshold sweeps), 27 (OOF operating-point summary)
and 28 (Pass 1 vs Pass 2 agreement). Every function here reads TRAINING
out-of-fold arrays stored on a fitted router. None accepts test data.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

__all__ = [
    "HEADLINE_METRICS",
    "plot_threshold_sweeps",
    "feasible_tail",
    "binding_after",
    "train_oof_summary",
    "pass_agreement",
]

#: The rows the Stage 2 comparison tables print (former Cell 29 ``HEADLINE``).
HEADLINE_METRICS = [
    "coverage",
    "abstain_rate",
    "pass1_share",
    "pass1_npv",
    "pass1_subscribers_lost",
    "pass1_leakage_rate",
    "pass2_share",
    "pass2_precision",
    "pass2_recall",
    "abstain_base_rate",
    "resolved_accuracy",
    "resolved_f1",
]

_BLUE, _AMBER, _INK = "#185FA5", "#BA7517", "#3d3d3a"


def plot_threshold_sweeps(router, constraints, *, title: Optional[str] = None):
    """
    Two-panel figure of the training-OOF threshold sweeps (former Cell 26).

    Left: Pass 1 NPV and leakage against coverage. Right: Pass 2 precision and
    recall against the fraction of its population flagged. Dotted lines are the
    constraints; the dashed line is the chosen band. Returns the figure; the
    caller shows or saves it.
    """
    import matplotlib.pyplot as plt

    p1 = router.thresholds_.pass1
    p2 = router.thresholds_.pass2
    s1, s2 = p1.sweep, p2.sweep
    if s1 is None or s2 is None:
        raise ValueError("router was fitted without threshold sweeps")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        title or "Stage 2 RF router — threshold sweeps (training OOF)",
        fontsize=13, fontweight="bold",
    )

    def _chosen(ax, x, label):
        ax.axvline(x, color=_INK, ls="--", lw=1.5)
        ax.annotate(label, xy=(x, 0.5), xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=_INK, rotation=90, va="center")

    ax = axes[0]
    ax.plot(s1["coverage"], s1["npv"], color=_BLUE, lw=2, label="NPV  P(y=0 | routed)")
    ax.plot(s1["coverage"], s1["leakage_rate"], color=_AMBER, lw=2,
            label="leakage rate  subs routed / all subs")
    ax.axhline(constraints.min_precision_not_subscribe, color=_BLUE, ls=":", lw=1.2)
    ax.axhline(constraints.max_subscriber_leakage_rate, color=_AMBER, ls=":", lw=1.2)
    _chosen(ax, p1.band_fraction, f"chosen t1 = {router.thresholds_.t1:.3f}")
    ax.set(title="Pass 1 — route NOT_SUBSCRIBE",
           xlabel="Fraction of training rows routed", ylabel="Rate", ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(loc="center left", fontsize=9)

    ax = axes[1]
    ax.plot(s2["band_fraction"], s2["precision"], color=_BLUE, lw=2,
            label="precision  P(y=1 | flagged)")
    ax.plot(s2["band_fraction"], s2["recall"], color=_AMBER, lw=2,
            label=f"recall  ({constraints.recall_denominator} denominator)")
    ax.axhline(constraints.min_precision_subscribe, color=_BLUE, ls=":", lw=1.2)
    if constraints.min_recall_subscribe is not None:
        ax.axhline(constraints.min_recall_subscribe, color=_AMBER, ls=":", lw=1.2)
    _chosen(ax, p2.band_fraction, f"chosen t2 = {router.thresholds_.t2:.3f}")
    pop = "remainder" if getattr(router, "remainder_mask_", None) is not None else "training split"
    ax.set(title=f"Pass 2 — detect SUBSCRIBE (within the {pop})",
           xlabel=f"Fraction of {pop} flagged", ylabel="Rate", ylim=(0, 1.02))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    return fig


def feasible_tail(sweep: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """The ``n`` largest feasible bands — where the chosen band sits."""
    return sweep[sweep["feasible"]].tail(n)


def binding_after(sweep: Optional[pd.DataFrame], band_size: int) -> str:
    """
    Constraint(s) that stopped the band from growing past ``band_size``.

    Reads the first band in the sweep LARGER than the chosen one. Returns
    ``"none (whole population feasible)"`` when the chosen band is the largest
    band the score can realise — for Pass 2 that is the degeneracy signature.
    """
    if sweep is None:
        return "unknown (no sweep)"
    larger = sweep[sweep["band_size"] > band_size]
    if larger.empty:
        return "none (whole population feasible)"
    return str(larger.iloc[0]["binding"]) or "none"


def train_oof_summary(routers: Mapping[str, object]) -> pd.DataFrame:
    """
    Operating point of each fitted router on training OOF (former Cell 27).

    Columns are router names. Works for both the single-RF ablation and the
    two-pass router; ``remainder frac`` is NaN for the ablation.
    """
    out = {}
    for name, r in routers.items():
        th, b = r.thresholds_, r.bands_
        rem = (r.fit_report_.remainder or {}).get("fraction", float("nan"))
        out[name] = {
            "t1": th.t1,
            "t2": th.t2,
            "pass1 band frac": th.pass1.band_fraction,
            "pass1 NPV (OOF)": b.pass1_confidence,
            "pass1 leakage (OOF)": th.pass1.leakage_rate,
            "pass1 binding next": binding_after(th.pass1.sweep, th.pass1.band_size),
            "pass2 band frac": th.pass2.band_fraction,
            "pass2 precision (OOF)": b.pass2_confidence,
            "pass2 recall (OOF)": th.pass2.recall,
            "pass2 binding next": binding_after(th.pass2.sweep, th.pass2.band_size),
            "remainder frac": rem,
        }
    return pd.DataFrame(out)


def pass_agreement(router) -> dict:
    """
    How much the Pass 2 forest's ranking differs from Pass 1's on the remainder
    (former Cell 28).

    If Pass 2 flags nearly the same rows Pass 1's score would have flagged at
    the same rate, the remainder carries little information the Pass 1 forest
    lacked — and no tuning objective can manufacture it.
    """
    rem = router.remainder_mask_
    p1 = router.oof_pass1_proba_[rem]
    p2 = router.oof_pass2_proba_[rem]
    flagged = p2 > router.thresholds_.t2
    flag_rate = float(flagged.mean())
    by_p1 = p1 > np.quantile(p1, 1 - flag_rate)
    return {
        "spearman(p1, p2) on remainder": float(
            pd.Series(p1).corr(pd.Series(p2), method="spearman")
        ),
        "pass2 flagged by p2 > t2": int(flagged.sum()),
        "same rate by p1 rank": int(by_p1.sum()),
        "agreement": float((flagged == by_p1).mean()),
        "pass2 flag rate (of remainder)": flag_rate,
    }
