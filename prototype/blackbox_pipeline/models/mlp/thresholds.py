"""
blackbox_pipeline.models.mlp.thresholds
========================================
Out-of-fold probabilities and the Stage 1 decision threshold.

Two sweeps live here, and they are deliberately different:

``sweep_f_beta`` / ``optimize_threshold_cv``
    The IN-STAGE sweep, run inside ``CalibratedStage1MLP.fit``. Grid 0.05–0.49 in
    0.01 steps, F2 — identical to the GLASS LR stage, so the two arms pick their
    thresholds the same way. Result lands on ``stage.optimal_threshold``.

``tune_threshold``
    The NOTEBOOK sweep (Cell 14/15). Wider grid, 0.05–0.95 in 0.005 steps.
    Finer, and able to go above 0.5, which the GLASS-matched sweep cannot.

They will not always agree, and that is fine — but only one can be the reported
operating point. Cell 15 overwrites ``STAGE1_OUTPUT["threshold"]`` with the
notebook sweep's answer, so that is the one in force downstream. Say which you
used when writing up, because a threshold from the wider grid is no longer
"the same procedure GLASS used".

Both take TRAINING out-of-fold probabilities. Neither accepts test data.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .calibration import assert_refittable

__all__ = [
    "oof_probabilities",
    "sweep_f_beta",
    "optimize_threshold_cv",
    "tune_threshold",
]


# ----------------------------------------------------------------------
def oof_probabilities(
    calibrated_model,
    X_scaled: np.ndarray,
    y,
    index: pd.Index,
    cv_folds: int,
    random_state: int,
    n_jobs: int = -1,
    strict: bool = True,
) -> Tuple[pd.Series, str]:
    """
    Out-of-fold calibrated ``P(y = 1)`` over the training split.

    Guarded by ``assert_refittable``: if the calibrator would survive cloning
    with its base model still fitted, these would silently be in-sample
    predictions. See ``calibration.py`` for why that is the one place Stage 1
    can leak without anything looking wrong.

    Returns ``(proba_oof, provenance)`` where provenance records what the guard
    found, for the fit report.
    """
    provenance = assert_refittable(calibrated_model, strict=strict)

    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )
    proba = cross_val_predict(
        calibrated_model, X_scaled, y,
        cv=cv, method="predict_proba", n_jobs=n_jobs,
    )[:, 1]

    if len(proba) != len(index):
        raise AssertionError(
            f"cross_val_predict returned {len(proba)} rows for {len(index)} "
            "training rows — fold assignment and index are misaligned."
        )

    return pd.Series(proba, index=index, name="stage1_proba_oof"), provenance


# ----------------------------------------------------------------------
def sweep_f_beta(
    y_true,
    proba,
    beta: float = 2.0,
    grid: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """F-beta at every threshold in ``grid``. Thresholds that predict nothing are skipped."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    if grid is None:
        grid = np.arange(0.05, 0.50, 0.01)

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
    return pd.DataFrame(rows)


def optimize_threshold_cv(
    y_true,
    proba_oof,
    beta: float = 2.0,
    grid_spec: Tuple[float, float, float] = (0.05, 0.50, 0.01),
) -> Tuple[float, float, pd.DataFrame]:
    """
    The GLASS-matched in-stage sweep.

    Returns ``(best_threshold, best_f_beta, sweep_frame)``. Falls back to 0.5
    when no threshold in the grid predicts a single positive.
    """
    start, stop, step = grid_spec
    sweep = sweep_f_beta(y_true, proba_oof, beta, np.arange(start, stop, step))

    if sweep.empty:
        return 0.5, 0.0, sweep

    best = sweep.loc[sweep["f_beta"].idxmax()]
    return float(best["threshold"]), float(best["f_beta"]), sweep


# ----------------------------------------------------------------------
def tune_threshold(
    y_true: pd.Series,
    proba: pd.Series,
    beta: float = 2.0,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, pd.DataFrame]:
    """
    The wider notebook sweep (Cell 14/15). Training (OOF) probabilities only.

    Kept signature-compatible with the notebook's original definition, including
    the ``f_beta`` column name, so Cell 15 runs unchanged.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    grid = (
        np.round(np.arange(0.05, 0.951, 0.005), 3)
        if grid is None
        else np.asarray(grid)
    )
    scores = np.array([
        fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0)
        for t in grid
    ])
    best = int(np.argmax(scores))
    return float(grid[best]), pd.DataFrame({"threshold": grid, "f_beta": scores})
