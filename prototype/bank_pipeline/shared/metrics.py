"""
shared.metrics
==============
The evaluation metrics both arms are reported and compared on.

Why it lives here
-----------------
Before PR 33 the "shared" metrics layer lived in
``blackbox_pipeline.models.mlp.evaluation`` and imported FROM GLASS, so GLASS
could not import it back. In practice the MLP reported two rows (0.50 and the
tuned threshold) plus ``pr_auc`` and ``pred_pos_rate``, while GLASS reported
one row and neither. Now both arms call these functions; ``lr.evaluation`` and
``mlp.evaluation.metrics`` re-export them.

Metric set
----------
``compute_metrics`` returns, at one operating point:

========================  ================================================
accuracy                  share of correct hard predictions
precision, recall         for the positive class
f1, f2                    F-beta at beta = 1 and 2 (F2 weights recall 4x)
roc_auc                   threshold-free ranking quality
pr_auc                    average precision — more informative than ROC-AUC
                          at an ~11% positive rate
brier                     mean squared error of the probabilities
ece                       expected calibration error, 10 equal-width bins
pred_pos_rate             share of rows predicted positive
threshold, calibration    echoed from the arguments, for bookkeeping
========================  ================================================

Behaviour changes vs pre-PR-33 code
-----------------------------------
* ``calculate_ece`` includes ``p == 0.0`` in the first bin. The old first bin
  was ``(0.0, 0.1]``, silently dropping exact zeros. Isotonic calibration
  emits exact zeros and sigmoid never does, so the old ECE was understated for
  isotonic runs only — biased along exactly the axis the arms may differ on.
* ``compute_metrics`` adds ``pr_auc`` and ``pred_pos_rate``.
* Arrays and Series are both accepted (the old code called ``y_test.values``
  and crashed on an ndarray). When labels and predictions are both Series,
  their indices must match.
* ``metrics_table`` widens row labels to three decimals rather than silently
  merging two thresholds that round to the same two-decimal label.

Public API
----------
calculate_ece, compute_metrics, binary_metrics, metrics_table
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

__all__ = ["calculate_ece", "compute_metrics", "binary_metrics", "metrics_table"]


def _check_aligned(y_true, *others) -> None:
    """
    Raise if ``y_true`` and any other argument are Series with different indices.

    A positional comparison of misaligned vectors produces plausible, wrong
    numbers, so this is checked whenever both sides carry an index. Arrays
    are passed through unchecked — they have nothing to check against.
    """
    if not isinstance(y_true, pd.Series):
        return
    for o in others:
        if isinstance(o, pd.Series) and not y_true.index.equals(o.index):
            raise ValueError(
                "labels and predictions are indexed differently — align them "
                "(e.g. proba = proba.loc[y.index]) before scoring"
            )


def calculate_ece(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Expected Calibration Error over equal-width probability bins.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels, 0/1.
    y_prob : array-like of shape (n_samples,)
        Predicted ``P(y = 1)``.
    n_bins : int, default 10
        Number of equal-width bins on [0, 1].

    Returns
    -------
    float
        ``sum_b (n_b / n) * |mean(y in b) - mean(p in b)|``. 0 is perfect.

    Notes
    -----
    Bin ``i`` is ``(edge_i, edge_{i+1}]`` except the first, which is closed on
    the left, ``[0, 1/n_bins]``, so predictions of exactly 0.0 are counted.
    Every prediction in [0, 1] lands in exactly one bin.

    Examples
    --------
    >>> calculate_ece([1, 0, 0, 0], [0.0, 0.0, 0.0, 0.9])
    0.475
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (p >= lo) & (p <= hi) if i == 0 else (p > lo) & (p <= hi)
        prop = in_bin.mean()
        if prop > 0:
            ece += abs(y[in_bin].mean() - p[in_bin].mean()) * prop
    return float(ece)


def compute_metrics(
    y_test,
    y_pred,
    y_proba,
    threshold: float,
    calibration_method: Optional[str],
) -> Dict:
    """
    The full metric set at one operating point.

    Parameters
    ----------
    y_test : array-like of shape (n_samples,)
        Binary labels, 0/1. Any split — the name is historical.
    y_pred : array-like of shape (n_samples,)
        Hard predictions, 0/1, already cut at ``threshold``.
    y_proba : array-like of shape (n_samples,)
        ``P(y = 1)``; used for ROC-AUC, PR-AUC, Brier and ECE.
    threshold : float
        The cut that produced ``y_pred``. Echoed, not applied.
    calibration_method : str or None
        E.g. ``'sigmoid'``, ``'isotonic'``, ``'none'``. Echoed.

    Returns
    -------
    dict
        Keys ``threshold``, ``calibration``, ``accuracy``, ``precision``,
        ``recall``, ``f1``, ``f2``, ``roc_auc``, ``pr_auc``, ``brier``,
        ``ece``, ``pred_pos_rate``.

    Raises
    ------
    ValueError
        If ``y_test`` is a Series and ``y_pred`` or ``y_proba`` is a Series
        with a different index.

    Notes
    -----
    Precision, recall and F-scores use ``zero_division=0``, so a threshold
    predicting no positives scores 0 rather than warning.

    See Also
    --------
    binary_metrics : applies the threshold for you.
    """
    _check_aligned(y_test, y_pred, y_proba)
    y = np.asarray(y_test).astype(int)
    pred = np.asarray(y_pred).astype(int)
    p = np.asarray(y_proba, dtype=float)
    return {
        "threshold":     threshold,
        "calibration":   calibration_method,
        "accuracy":      accuracy_score(y, pred),
        "precision":     precision_score(y, pred, zero_division=0),
        "recall":        recall_score(y, pred, zero_division=0),
        "f1":            f1_score(y, pred, zero_division=0),
        "f2":            fbeta_score(y, pred, beta=2, zero_division=0),
        "roc_auc":       roc_auc_score(y, p),
        "pr_auc":        average_precision_score(y, p),
        "brier":         brier_score_loss(y, p),
        "ece":           calculate_ece(y, p),
        "pred_pos_rate": float(pred.mean()),
    }


def binary_metrics(
    y_true,
    proba,
    threshold: float = 0.5,
    calibration_method: Optional[str] = None,
) -> pd.Series:
    """
    Score probabilities at one threshold.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels, 0/1.
    proba : array-like of shape (n_samples,)
        ``P(y = 1)``.
    threshold : float, default 0.5
        Cut applied as ``proba >= threshold``.
    calibration_method : str, optional
        Passed through to :func:`compute_metrics`.

    Returns
    -------
    pandas.Series
        The :func:`compute_metrics` values without the two echo keys
        (``threshold``, ``calibration``), which get in the way when rows are
        stacked by :func:`metrics_table`.

    Raises
    ------
    ValueError
        If both inputs are Series with different indices.
    """
    _check_aligned(y_true, proba)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    m = compute_metrics(np.asarray(y_true), pred, p, threshold, calibration_method)
    return pd.Series({k: v for k, v in m.items() if k not in ("threshold", "calibration")})


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
        ``P(y = 1)``. For the training split, pass OUT-OF-FOLD probabilities
        (``stage.proba_train_oof``) — never ``predict_proba(X_train)``.
    label : str
        Row-label prefix, e.g. ``"test"`` or ``"train OOF"``.
    thresholds : Sequence[float]
        Cuts to report, in display order.
    calibration_method : str, optional
        Passed through.

    Returns
    -------
    pandas.DataFrame
        One row per threshold, indexed ``"{label} @ {t:.2f}"``; one column per
        metric.

    Notes
    -----
    If two thresholds would share a two-decimal label (0.5 and 0.499), labels
    widen to three decimals for the whole table instead of one row silently
    overwriting the other.

    Examples
    --------
    >>> metrics_table(y_test, stage.predict_proba(X_test), "test",
    ...               thresholds=(0.5, stage.optimal_threshold))
    """
    fmt = ".2f"
    if len({f"{t:.2f}" for t in thresholds}) < len(set(thresholds)):
        fmt = ".3f"
    rows: Dict[str, pd.Series] = {}
    for t in thresholds:
        rows[f"{label} @ {t:{fmt}}"] = binary_metrics(y_true, proba, t, calibration_method)
    return pd.DataFrame(rows).T
