"""
blackbox_pipeline.models.mlp.thresholds

Out-of-fold probabilities and the Stage 1 decision threshold.

:func:`oof_probabilities`
    Out-of-fold calibrated ``P(y = 1)`` over the training split, guarded by the
    prefit leakage check. These probabilities are the Stage 4 train feed.

:func:`sweep_f_beta` / :func:`optimize_threshold_cv`
    The in-stage F2 sweep run inside ``CalibratedStage1MLP.fit``: 45 points,
    0.05–0.49 in 0.01 steps, ties to the lowest threshold. Re-exported from
    ``shared.thresholds`` — the SAME implementation the GLASS LR stage uses, so
    both arms pick their operating point by one procedure. The result lands on
    ``stage.optimal_threshold``.

Notes
-----
PR 33 removed ``tune_threshold`` — a second, wider notebook sweep (181 points,
0.05–0.95 in 0.005 steps) whose answer used to overwrite the in-stage
threshold. It duplicated the in-stage sweep, and using it meant the two arms
chose thresholds by different procedures (audit F7). The only extra information
it offered was whether the optimum lies above 0.49; on this data the optimum is
≈ 0.10.

Everything here takes TRAINING out-of-fold probabilities. Nothing accepts test
data.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from shared.thresholds import optimize_threshold_cv, sweep_f_beta

from .calibration import assert_refittable

__all__ = [
    "oof_probabilities",
    "sweep_f_beta",
    "optimize_threshold_cv",
]


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
    in-sample predictions. See :mod:`shared.calibration_guard` for why that is
    the one place Stage 1 can leak without anything looking wrong.

    The fold partition is ``StratifiedKFold(cv_folds, shuffle=True,
    random_state)`` — identical to the GLASS arm, which is what makes per-row
    comparison between the arms paired. ``shared.stage_io.fold_assignment``
    reconstructs it.
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
