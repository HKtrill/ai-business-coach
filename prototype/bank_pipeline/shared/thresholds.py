"""
shared.thresholds
=================
The in-stage F-beta threshold sweep both arms use to pick an operating point.

What it does
------------
Given training labels and OUT-OF-FOLD probabilities, try every threshold on a
fixed grid, score F-beta at each, and return the best. Stage 1 uses F2
(recall weighted 4x precision) over ``np.arange(0.05, 0.50, 0.01)`` — 45
candidates — with ties going to the lowest threshold.

History
-------
Before PR 33 GLASS had its own loop inside ``_optimize_threshold_cv`` and the
MLP had ``sweep_f_beta`` / ``optimize_threshold_cv``. They agreed — same grid,
same F2, same tie rule — but as two implementations. There is now one;
``blackbox_pipeline.models.mlp.thresholds`` re-exports it. The MLP's wider
notebook sweep (``tune_threshold``, 181 points up to 0.95) was removed in
PR 33: it duplicated this sweep and was not part of the matched procedure.

Contamination
-------------
The threshold returned is an argmax over the labels of every row passed in.
When those rows are the training split and the threshold is then applied back
to them, train-side DECISIONS are fitted to those rows' labels even though the
probabilities are out-of-fold. ``StageOutput`` records this as
``threshold_source='in_stage_f2_cv'`` and refuses to hand out train-side
decisions without an explicit opt-in. Test-side decisions are unaffected.

Public API
----------
DEFAULT_GRID_SPEC
    ``(0.05, 0.50, 0.01)``.
sweep_f_beta
    Score every threshold; returns the full table.
optimize_threshold_cv
    Pick the best threshold from that table.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

__all__ = ["DEFAULT_GRID_SPEC", "sweep_f_beta", "optimize_threshold_cv"]

#: ``(start, stop, step)`` for ``np.arange`` — 45 points, 0.05 to 0.49.
#: The stop is exclusive, so 0.50 itself is never a candidate.
DEFAULT_GRID_SPEC: Tuple[float, float, float] = (0.05, 0.50, 0.01)


def sweep_f_beta(
    y_true,
    proba,
    beta: float = 2.0,
    grid: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Score F-beta at every threshold in ``grid``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels, 0/1.
    proba : array-like of shape (n_samples,)
        ``P(y = 1)`` for the same rows — out-of-fold when used for fitting.
    beta : float, default 2.0
        Recall weight. 2.0 is F2.
    grid : numpy.ndarray, optional
        Thresholds to try, in the order they should appear. Defaults to
        ``np.arange(*DEFAULT_GRID_SPEC)``.

    Returns
    -------
    pandas.DataFrame
        Columns ``threshold``, ``f_beta``, ``pred_pos_rate``; one row per
        threshold that predicts at least one positive. Thresholds predicting
        none are skipped, so the frame can be shorter than ``grid`` and can be
        empty (columns still present).

    Raises
    ------
    ValueError
        If ``y_true`` and ``proba`` are both Series with different indices.

    Examples
    --------
    >>> sweep = sweep_f_beta(y_train, stage.proba_train_oof)
    >>> sweep.loc[sweep["f_beta"].idxmax()]
    """
    if isinstance(y_true, pd.Series) and isinstance(proba, pd.Series):
        if not y_true.index.equals(proba.index):
            raise ValueError("y_true and proba are indexed differently")

    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    if grid is None:
        grid = np.arange(*DEFAULT_GRID_SPEC)

    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        if pred.sum() == 0:
            continue
        rows.append({
            "threshold": float(t),
            "f_beta": float(fbeta_score(y, pred, beta=beta, zero_division=0)),
            "pred_pos_rate": float(pred.mean()),
        })
    return pd.DataFrame(rows, columns=["threshold", "f_beta", "pred_pos_rate"])


def optimize_threshold_cv(
    y_true,
    proba_oof,
    beta: float = 2.0,
    grid_spec: Tuple[float, float, float] = DEFAULT_GRID_SPEC,
) -> Tuple[float, float, pd.DataFrame]:
    """
    Pick the F-beta-optimal threshold over ``np.arange(*grid_spec)``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary TRAINING labels.
    proba_oof : array-like of shape (n_samples,)
        Training OUT-OF-FOLD ``P(y = 1)``. Passing in-sample probabilities
        picks a threshold for a model that is sharper than it will be at test.
    beta : float, default 2.0
        Recall weight.
    grid_spec : tuple of (float, float, float), default DEFAULT_GRID_SPEC
        ``(start, stop, step)`` for ``np.arange``; ``stop`` is exclusive.

    Returns
    -------
    best_threshold : float
        Argmax of F-beta. 0.5 if no grid point predicted a positive.
    best_f_beta : float
        F-beta at that threshold, or 0.0 in the fallback case.
    sweep : pandas.DataFrame
        The full table from :func:`sweep_f_beta`.

    Notes
    -----
    Ties go to the LOWEST threshold: ``idxmax`` returns the first maximum and
    the sweep is ascending. This matches the original GLASS
    ``if f2 > best_f2`` loop exactly and favours recall, consistent with F2.

    The 0.5 fallback is degenerate — it means the calibrated probabilities
    never reached 0.05 — and should be investigated rather than accepted.

    Examples
    --------
    >>> t, f2, sweep = optimize_threshold_cv(y_train, stage.proba_train_oof)
    """
    start, stop, step = grid_spec
    sweep = sweep_f_beta(y_true, proba_oof, beta, np.arange(start, stop, step))

    if sweep.empty:
        return 0.5, 0.0, sweep

    best = sweep.loc[sweep["f_beta"].idxmax()]
    return float(best["threshold"]), float(best["f_beta"]), sweep
