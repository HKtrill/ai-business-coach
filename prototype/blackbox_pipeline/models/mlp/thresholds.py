"""
blackbox_pipeline.models.mlp.thresholds

Out-of-fold probabilities and the Stage 1 decision threshold.

Two sweeps live here, and they are deliberately different.

:func:`sweep_f_beta` / :func:`optimize_threshold_cv`
    The IN-STAGE sweep, run inside ``CalibratedStage1MLP.fit``. 45 points,
    0.05–0.49 in 0.01 steps, F2 — identical to the GLASS LR stage, so the two
    arms pick their thresholds the same way. The result lands on
    ``stage.optimal_threshold``.

:func:`tune_threshold`
    The NOTEBOOK sweep (Cell 14/15). 181 points, 0.05–0.95 in 0.005 steps:
    finer, and able to go above 0.5, which the GLASS-matched sweep cannot.

Notes
-----
The two will not always agree, and that is fine — but only one can be the
reported operating point. Cell 15 overwrites ``STAGE1_OUTPUT["threshold"]`` with
the notebook sweep's answer, so that is the one in force downstream. Say which
you used when writing up: a threshold from the wider grid is no longer "the same
procedure GLASS used".

Both sweeps break ties toward the LOWEST threshold, which favours recall —
consistent with the intent of F2, but worth knowing when two operating points
score identically.

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

    Parameters
    ----------
    calibrated_model : estimator
        The calibrator from :func:`~.calibration.fit_stage1_calibration`.
    X_scaled : numpy.ndarray of shape (n_samples, n_features)
        Scaled TRAINING features.
    y : array-like of shape (n_samples,)
        Binary training labels.
    index : pandas.Index
        Training index, used to label the returned Series.
    cv_folds : int
        Folds for ``cross_val_predict``.
    random_state : int
        Seed for the fold split.
    n_jobs : int, default -1
        Parallelism for ``cross_val_predict``.
    strict : bool, default True
        Passed to :func:`~.calibration.assert_refittable`.

    Returns
    -------
    proba_oof : pandas.Series
        Out-of-fold ``P(y = 1)``, named ``stage1_proba_oof`` and indexed like
        ``index``.
    provenance : str
        What the leakage guard found, recorded for the fit report.

    Raises
    ------
    CalibrationLeakageError
        If the calibrator is prefit and ``strict`` is True.
    AssertionError
        If ``cross_val_predict`` returns a row count that does not match
        ``index``.

    Notes
    -----
    Guarded by :func:`~.calibration.assert_refittable`: if the calibrator would
    survive cloning with its base model still fitted, these would silently be
    in-sample predictions. See :mod:`~.calibration` for why that is the one
    place Stage 1 can leak without anything looking wrong.
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
    """
    Score F-beta at every threshold in ``grid``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels.
    proba : array-like of shape (n_samples,)
        Out-of-fold ``P(y = 1)``.
    beta : float, default 2.0
        Recall weight.
    grid : numpy.ndarray, optional
        Thresholds to try. Defaults to ``np.arange(0.05, 0.50, 0.01)``.

    Returns
    -------
    pandas.DataFrame
        Columns ``threshold``, ``f_beta``, ``pred_pos_rate``; one row per
        threshold. Thresholds that predict no positives are skipped, so the
        frame can be shorter than ``grid`` and can be empty.
    """
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

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary training labels.
    proba_oof : array-like of shape (n_samples,)
        Training out-of-fold ``P(y = 1)``.
    beta : float, default 2.0
        Recall weight.
    grid_spec : tuple of (float, float, float), default (0.05, 0.50, 0.01)
        ``(start, stop, step)`` handed to ``np.arange``.

    Returns
    -------
    best_threshold : float
        Argmax of F-beta, or 0.5 if no threshold predicted a positive.
    best_f_beta : float
        F-beta there, or 0.0 in the fallback case.
    sweep : pandas.DataFrame
        The full sweep from :func:`sweep_f_beta`.

    Notes
    -----
    Ties go to the lowest threshold: ``idxmax`` takes the first maximum and the
    sweep is in ascending threshold order. The 0.5 fallback is a degenerate
    case — every threshold in the grid predicting zero positives means the
    calibrated probabilities never reach 0.05, which is worth investigating
    rather than accepting.
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

    Parameters
    ----------
    y_true : pandas.Series
        Binary training labels.
    proba : pandas.Series
        Training out-of-fold ``P(y = 1)``.
    beta : float, default 2.0
        Recall weight.
    grid : numpy.ndarray, optional
        Thresholds to try. Defaults to 0.05–0.95 in 0.005 steps.

    Returns
    -------
    best_threshold : float
        Argmax of F-beta over ``grid``.
    sweep : pandas.DataFrame
        Columns ``threshold`` and ``f_beta``, one row per grid point. Unlike
        :func:`sweep_f_beta`, empty predictions are scored 0 rather than
        dropped, so the frame always matches ``grid``.

    Notes
    -----
    Signature-compatible with the notebook's original definition, including the
    ``f_beta`` column name, so Cell 15 runs unchanged.

    Ties go to the lowest threshold, since ``np.argmax`` takes the first
    maximum. There is no 0.5 fallback here: if no threshold predicts a positive
    the whole ``f_beta`` column is 0 and the first grid point, 0.05, is
    returned. Check ``sweep['f_beta'].max()`` before trusting the result.
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
