"""
blackbox_pipeline.models.mlp.calibration

Probability calibration, delegated to GLASS, plus the prefit leakage guard.

Stage 1 must calibrate the same way the GLASS LR stage does, so this module
wraps ``glass_pipeline.lr.calibration.fit_calibration`` rather than
reimplementing it.

PR 33: the guard (``is_prefit_calibrator``, ``assert_refittable``,
``CalibrationLeakageError``) moved to ``shared.calibration_guard``
so GLASS runs the same check before its own ``cross_val_predict``. It is
re-exported here unchanged, so existing imports keep working. See that module
for why the check matters.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from shared.calibration_guard import (
    CalibrationLeakageError,
    assert_refittable,
    is_prefit_calibrator,
)

__all__ = [
    "fit_stage1_calibration",
    "is_prefit_calibrator",
    "assert_refittable",
    "CalibrationLeakageError",
]


def fit_stage1_calibration(
    model: Any,
    X_scaled: np.ndarray,
    y,
    method: str,
    cv_folds: int,
) -> Tuple[Any, str, Dict]:
    """
    Calibrate ``model`` using the GLASS routine.

    Returns
    -------
    calibrated_model : estimator
    chosen_method : str
        The method actually used — GLASS's winner under ``'auto'``.
    metrics : dict
    """
    from glass_pipeline.lr.calibration import fit_calibration

    calibrated, chosen, metrics = fit_calibration(
        model, X_scaled, y, method, cv_folds
    )
    return calibrated, chosen, metrics
