"""
blackbox_pipeline.models.mlp.evaluation.metrics

Binary classification metrics at a chosen threshold.

Both helpers delegate the core numbers to ``glass_pipeline.lr.evaluation``, so
the black-box arm and the GLASS arm are always measured by the same code, then
add the two threshold-free/rate columns Stage 1 reports on top.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

__all__ = ["binary_metrics", "metrics_table"]


def binary_metrics(
    y_true,
    proba,
    threshold: float = 0.5,
    calibration_method: Optional[str] = None,
) -> pd.Series:
    """
    Score one set of predictions at one threshold.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels, 0/1.
    proba : array-like of shape (n_samples,)
        Predicted ``P(y = 1)``.
    threshold : float, default 0.5
        Cut applied as ``proba >= threshold``.
    calibration_method : str, optional
        Passed to the GLASS routine for its own bookkeeping.

    Returns
    -------
    pandas.Series
        The GLASS metrics, plus ``pr_auc`` (average precision) and
        ``pred_pos_rate`` (share of rows predicted positive).

    Notes
    -----
    The GLASS ``threshold`` and ``calibration`` columns are dropped: they echo
    the arguments rather than measuring anything, and they get in the way when
    several rows are stacked by :func:`metrics_table`.
    """
    from glass_pipeline.lr.evaluation import compute_metrics

    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)

    m = compute_metrics(y_true, pred, p, threshold, calibration_method)
    m = {k: v for k, v in m.items() if k not in ("threshold", "calibration")}
    m["pr_auc"] = float(average_precision_score(y_true, p))
    m["pred_pos_rate"] = float(pred.mean())
    return pd.Series(m)


def metrics_table(
    y_true,
    proba,
    label: str,
    thresholds: Sequence[float],
    calibration_method: Optional[str] = None,
) -> pd.DataFrame:
    """
    Stack :func:`binary_metrics` across several thresholds.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels, 0/1.
    proba : array-like of shape (n_samples,)
        Predicted ``P(y = 1)``.
    label : str
        Row-label prefix, e.g. ``"test"`` or ``"train OOF"``.
    thresholds : Sequence[float]
        Cuts to report, in the order they should appear.
    calibration_method : str, optional
        Passed through to :func:`binary_metrics`.

    Returns
    -------
    pandas.DataFrame
        One row per threshold, indexed ``"{label} @ {threshold:.2f}"``, one
        column per metric.

    Notes
    -----
    Row labels are rounded to two decimals and used as dict keys, so two
    thresholds that format identically collapse into a single row — 0.5 and
    0.499 both render as ``"x @ 0.50"`` and you get one row, not two, with no
    warning. Pass thresholds that differ at two decimals, or widen the format,
    if you need every one of them reported.
    """
    rows: Dict[str, pd.Series] = {}
    for t in thresholds:
        rows[f"{label} @ {t:.2f}"] = binary_metrics(
            y_true, proba, t, calibration_method
        )
    return pd.DataFrame(rows).T
