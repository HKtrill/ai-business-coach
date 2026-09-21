"""
blackbox_pipeline.models.rf_router.evaluation.router_metrics

Shared routing metrics for Stage 2.

One module, both arms. ``evaluate_router`` takes the ``(preds, conf, decisions)``
triple that ``GLASSBRWPipeline.predict`` and both RF routers emit, so the GLASS
and RF numbers are produced by identical code. Giving each arm its own metrics
helper is how comparisons end up arguing about denominators instead of models.

Every quantity here is SET-LEVEL, measured over the routed set as a whole.
GLASS's configured caps (``max_subscriber_leakage_rate`` and friends) are
per-rule and are not comparable to these; treat those as configuration and
compare only what this module measures.

Notes
-----
Denominator conventions, matching ``glass_brw.rule_generator.rule_metrics``::

    pass1 NPV       = P(y = 0 | decisions == "pass1")
    leakage_rate    = subscribers routed by Pass 1 / ALL positives in the split
    pass2 precision = P(y = 1 | decisions == "pass2")
    pass2 recall    = positives flagged by Pass 2 / ALL positives in the split

The decision codes come from ``..router.decisions`` rather than being redefined
here. They are one contract; a second copy in the module that validates against
them is how the two drift apart without any test failing.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import pandas as pd

from ..router.decisions import (
    ABSTAIN,
    NOT_SUBSCRIBE,
    PASS1,
    PASS2,
    SUBSCRIBE,
    UNCERTAIN,
)

__all__ = ["evaluate_router", "metrics_to_frame", "compare_routers"]


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
    preds, conf, decisions : numpy.ndarray
        The Stage 2 triple. ``preds`` in {0, 1, -1}; ``decisions`` in
        {"pass1", "pass2", "uncertain"}; ``conf`` in [0, 1].
    y_true : array-like
        Ground-truth labels for the SAME rows, in the same order.
    label : str, default 'router'
        Arm name. Becomes the column header in :func:`compare_routers`, so it
        must be unique across the arms being compared.
    base_rate : float, optional
        Training base rate, recorded for context. Not used in any calculation.

    Returns
    -------
    dict
        Split composition, headline coverage/abstention, per-pass statistics,
        the abstain region's profile, and accuracy plus F1 over the resolved
        subset.

    Raises
    ------
    ValueError
        If the four arrays differ in length, if ``preds`` or ``decisions`` hold
        unknown values, or if the two disagree — a ``"pass1"`` row that does not
        carry prediction 0 means one of the arrays is lying about what the
        router did.

    Notes
    -----
    ``resolved_f1`` treats Pass 2 as the positive prediction over the resolved
    subset only. Its false-negative term is therefore exactly the subscribers
    Pass 1 routed away; positives sitting in the abstain region are excluded
    from the F1 altogether. Read it as "how well did Pass 2 do on what the
    router was willing to decide", not as an overall F1.
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
    if not np.isin(decisions, (PASS1, PASS2, UNCERTAIN)).all():
        raise ValueError(
            f"decisions must be in {{{PASS1!r}, {PASS2!r}, {UNCERTAIN!r}}}"
        )

    # Contract check: the two arrays must agree, or one of them is lying about
    # what the router did.
    if (preds[decisions == PASS1] != NOT_SUBSCRIBE).any():
        raise ValueError(f"A {PASS1!r} decision does not carry prediction 0")
    if (preds[decisions == PASS2] != SUBSCRIBE).any():
        raise ValueError(f"A {PASS2!r} decision does not carry prediction 1")
    if (preds[decisions == UNCERTAIN] != ABSTAIN).any():
        raise ValueError(f"An {UNCERTAIN!r} decision does not carry prediction -1")

    total_pos = int(y.sum())
    total_neg = int(n - total_pos)

    m_p1 = decisions == PASS1
    m_p2 = decisions == PASS2
    m_ab = decisions == UNCERTAIN

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
    # fn reduces to the Pass 1 leak; abstained positives are out of scope here.
    tp = pos_flagged
    fp = n_p2 - pos_flagged
    fn = int(y[resolved].sum()) - tp
    out["resolved_f1"] = (
        float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan")
    )
    return out


def metrics_to_frame(metrics: Mapping) -> pd.DataFrame:
    """
    Render one metrics dict as a two-column frame for notebook display.

    Parameters
    ----------
    metrics : Mapping
        Output of :func:`evaluate_router`.

    Returns
    -------
    pandas.DataFrame
        Columns ``metric`` and ``value``, one row per entry, including
        ``label``.
    """
    return pd.DataFrame(
        {"metric": list(metrics.keys()), "value": list(metrics.values())}
    )


def compare_routers(*metric_dicts: Mapping) -> pd.DataFrame:
    """
    Side-by-side table, one column per arm.

    Parameters
    ----------
    *metric_dicts : Mapping
        One :func:`evaluate_router` output per arm. Each must carry a distinct
        ``label``.

    Returns
    -------
    pandas.DataFrame
        Metrics as rows, arms as columns. ``label`` becomes the column header
        rather than a row.

    Raises
    ------
    ValueError
        If no metrics dicts are given.

    Notes
    -----
    Columns are keyed on ``label``, so two arms sharing one silently collapse
    into a single column — the later dict wins and the earlier is lost without
    warning. Label every arm distinctly.

    Pass the GLASS arm and both RF arms through here so every number in the
    write-up comes from the same computation::

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
