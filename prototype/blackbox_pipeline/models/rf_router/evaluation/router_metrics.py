"""
blackbox_pipeline.evaluation.router_metrics
============================================
Shared routing metrics for Stage 2.

One module, both arms. ``evaluate_router`` takes the ``(preds, conf, decisions)``
triple that ``GLASSBRWPipeline.predict`` and both RF routers emit, so the GLASS
and RF numbers are produced by identical code. Giving each arm its own metrics
helper is how comparisons end up arguing about denominators instead of models.

Every quantity here is SET-LEVEL, measured over the routed set as a whole.
GLASS's configured caps (``max_subscriber_leakage_rate`` and friends) are
per-rule and are not comparable to these; treat them as configuration and
compare only what this module measures.

Denominator conventions, matching ``glass_brw.rule_generator.rule_metrics``:

    pass1 NPV      = P(y = 0 | decisions == "pass1")
    leakage_rate   = subscribers routed by Pass 1 / ALL positives in the split
    pass2 precision= P(y = 1 | decisions == "pass2")
    pass2 recall   = positives flagged by Pass 2 / ALL positives in the split
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

__all__ = ["evaluate_router", "metrics_to_frame", "compare_routers"]

SUBSCRIBE = 1
NOT_SUBSCRIBE = 0
ABSTAIN = -1


def evaluate_router(
    preds: np.ndarray,
    conf: np.ndarray,
    decisions: np.ndarray,
    y_true,
    *,
    label: str = "router",
    base_rate: Optional[float] = None,
) -> dict:
    """
    Score one router's output.

    Parameters
    ----------
    preds, conf, decisions
        The Stage 2 triple. ``preds`` in {0, 1, -1}; ``decisions`` in
        {"pass1", "pass2", "uncertain"}.
    y_true
        Ground-truth labels for the SAME rows, in the same order.
    base_rate
        Training base rate, for context in the report. Optional.
    """
    preds = np.asarray(preds)
    conf = np.asarray(conf, dtype=float)
    decisions = np.asarray(decisions, dtype=object)
    y = np.asarray(y_true).astype(int)

    n = len(y)
    if not (len(preds) == len(conf) == len(decisions) == n):
        raise ValueError(
            "Length mismatch: "
            f"preds={len(preds)}, conf={len(conf)}, "
            f"decisions={len(decisions)}, y={n}"
        )
    if not np.isin(preds, (NOT_SUBSCRIBE, SUBSCRIBE, ABSTAIN)).all():
        raise ValueError("preds must be in {0, 1, -1}")
    if not np.isin(decisions, ("pass1", "pass2", "uncertain")).all():
        raise ValueError(
            "decisions must be in {'pass1', 'pass2', 'uncertain'}"
        )

    # Contract check: the two arrays must agree, or one of them is lying about
    # what the router did.
    if (preds[decisions == "pass1"] != NOT_SUBSCRIBE).any():
        raise ValueError("A 'pass1' decision does not carry prediction 0")
    if (preds[decisions == "pass2"] != SUBSCRIBE).any():
        raise ValueError("A 'pass2' decision does not carry prediction 1")
    if (preds[decisions == "uncertain"] != ABSTAIN).any():
        raise ValueError("An 'uncertain' decision does not carry prediction -1")

    total_pos = int(y.sum())
    total_neg = int(n - total_pos)

    m_p1 = decisions == "pass1"
    m_p2 = decisions == "pass2"
    m_ab = decisions == "uncertain"

    n_p1, n_p2, n_ab = int(m_p1.sum()), int(m_p2.sum()), int(m_ab.sum())
    resolved = ~m_ab
    n_res = int(resolved.sum())

    subs_routed = int(y[m_p1].sum())
    pos_flagged = int(y[m_p2].sum())

    out = {
        "label": label,
        "n_samples": n,
        "n_positives": total_pos,
        "n_negatives": total_neg,
        "split_base_rate": float(total_pos / n) if n else 0.0,
        "training_base_rate": None if base_rate is None else float(base_rate),

        # ---- headline -------------------------------------------------
        "coverage": float(n_res / n) if n else 0.0,
        "abstain_rate": float(n_ab / n) if n else 0.0,

        # ---- Pass 1: route NOT_SUBSCRIBE ------------------------------
        "pass1_n": n_p1,
        "pass1_share": float(n_p1 / n) if n else 0.0,
        "pass1_npv": float((y[m_p1] == 0).mean()) if n_p1 else float("nan"),
        "pass1_subscribers_lost": subs_routed,
        "pass1_leakage_rate": float(subs_routed / total_pos) if total_pos else 0.0,
        "pass1_negative_recall": (
            float((y[m_p1] == 0).sum() / total_neg) if total_neg else 0.0
        ),

        # ---- Pass 2: detect SUBSCRIBE ---------------------------------
        "pass2_n": n_p2,
        "pass2_share": float(n_p2 / n) if n else 0.0,
        "pass2_precision": float((y[m_p2] == 1).mean()) if n_p2 else float("nan"),
        "pass2_recall": float(pos_flagged / total_pos) if total_pos else 0.0,
        "pass2_positives_found": pos_flagged,

        # ---- abstain region -------------------------------------------
        "abstain_n": n_ab,
        "abstain_base_rate": float(y[m_ab].mean()) if n_ab else float("nan"),
        "abstain_positives": int(y[m_ab].sum()),

        # ---- resolved subset -------------------------------------------
        "resolved_n": n_res,
        "resolved_accuracy": (
            float((preds[resolved] == y[resolved]).mean()) if n_res else float("nan")
        ),
        "mean_confidence_resolved": (
            float(conf[resolved].mean()) if n_res else float("nan")
        ),
    }

    # F1 over the resolved subset, treating Pass 2 as the positive prediction.
    tp = pos_flagged
    fp = n_p2 - pos_flagged
    fn = int(y[resolved].sum()) - tp
    out["resolved_f1"] = (
        float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan")
    )
    return out


def metrics_to_frame(metrics: Mapping) -> pd.DataFrame:
    """One metrics dict as a two-column frame, for notebook display."""
    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )


def compare_routers(*metric_dicts: Mapping) -> pd.DataFrame:
    """
    Side-by-side table, one column per arm.

    Pass the GLASS arm and both RF arms through here so every number in the
    write-up comes from the same computation:

        compare_routers(
            evaluate_router(*glass.predict(X_test), y_test, label="GLASS"),
            evaluate_router(*single.predict(X_test), y_test, label="RF single"),
            evaluate_router(*router.predict(X_test), y_test, label="RF two-pass"),
        )
    """
    if not metric_dicts:
        raise ValueError("compare_routers needs at least one metrics dict")

    frames = {}
    for m in metric_dicts:
        name = m.get("label", f"arm_{len(frames)}")
        frames[name] = {k: v for k, v in m.items() if k != "label"}
    return pd.DataFrame(frames)