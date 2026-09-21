"""
blackbox_pipeline.models.mlp.calibration

Probability calibration, delegated to GLASS, plus the prefit leakage guard.

Stage 1 must calibrate the same way the GLASS LR stage does, so this module
wraps ``glass_pipeline.lr.calibration.fit_calibration`` rather than
reimplementing it. Isolating the dependency here means the rest of the package
imports nothing from GLASS, and a change in that API surfaces in one file.

Notes
-----
Why the prefit check matters. ``fit_calibration`` returns a fitted calibrator.
Stage 1 then computes out-of-fold probabilities by handing that object to
``cross_val_predict``, which CLONES it per fold. The clone is only honest if
cloning discards the fit:

* ``CalibratedClassifierCV(estimator=..., cv=<int>)`` — the clone refits the
  base estimator inside each fold. Out-of-fold, correct.
* ``CalibratedClassifierCV(estimator=<already fitted>, cv='prefit')``, or an
  estimator wrapped in ``sklearn.frozen.FrozenEstimator`` — the clone KEEPS the
  fitted base model. That model was fitted on the whole training split, so every
  "out-of-fold" probability comes from a model that saw the row. The numbers are
  in-sample, and a threshold chosen from them is tuned on memorised data.

:func:`assert_refittable` detects the second case and refuses. It is the single
leakage risk in the Stage 1 protocol, and it is invisible in the output — the
probabilities look fine, they are just optimistic.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

__all__ = [
    "fit_stage1_calibration",
    "is_prefit_calibrator",
    "assert_refittable",
    "CalibrationLeakageError",
]


class CalibrationLeakageError(RuntimeError):
    """Raised when a calibrator cannot produce honest out-of-fold predictions."""


# ----------------------------------------------------------------------
def fit_stage1_calibration(
    model: Any,
    X_scaled: np.ndarray,
    y,
    method: str,
    cv_folds: int,
) -> Tuple[Any, str, Dict]:
    """
    Calibrate ``model`` using the GLASS routine.

    Parameters
    ----------
    model : estimator
        Fitted ``Stage1MLPClassifier``.
    X_scaled : numpy.ndarray of shape (n_samples, n_features)
        Scaled TRAINING features.
    y : array-like of shape (n_samples,)
        Binary training labels.
    method : {'auto', 'sigmoid', 'isotonic'}
        Calibration family. ``'auto'`` lets GLASS choose.
    cv_folds : int
        Folds for the calibration fit.

    Returns
    -------
    calibrated_model : estimator
        The fitted calibrator.
    chosen_method : str
        The method actually used. Under ``'auto'`` this is GLASS's winner, and
        the caller should write it back so downstream reporting names the real
        method.
    metrics : dict
        GLASS's calibration diagnostics.
    """
    from glass_pipeline.lr.calibration import fit_calibration

    calibrated, chosen, metrics = fit_calibration(
        model, X_scaled, y, method, cv_folds
    )
    return calibrated, chosen, metrics


# ----------------------------------------------------------------------
def is_prefit_calibrator(calibrated: Any) -> Tuple[bool, str]:
    """
    Would cloning this estimator preserve an already-fitted base model?

    Parameters
    ----------
    calibrated : estimator
        The object ``cross_val_predict`` would be handed.

    Returns
    -------
    is_prefit : bool
        True if cloning keeps a fitted base model.
    reason : str
        What was detected, for the error message or the provenance string.

    Notes
    -----
    Conservative: anything it cannot positively identify as prefit is reported
    as safe, because a false alarm here would block a correct pipeline.
    """
    cv = getattr(calibrated, "cv", None)
    if isinstance(cv, str) and cv == "prefit":
        return True, "CalibratedClassifierCV(cv='prefit')"

    try:
        from sklearn.frozen import FrozenEstimator

        if isinstance(calibrated, FrozenEstimator):
            return True, "estimator is wrapped in sklearn.frozen.FrozenEstimator"
        inner = getattr(calibrated, "estimator", None)
        if inner is not None and isinstance(inner, FrozenEstimator):
            return True, "base estimator is wrapped in sklearn.frozen.FrozenEstimator"
    except ImportError:  # pragma: no cover - older sklearn
        pass

    return False, f"cv={cv!r}"


def assert_refittable(calibrated: Any, strict: bool = True) -> str:
    """
    Guard run before computing out-of-fold probabilities.

    Parameters
    ----------
    calibrated : estimator
        The calibrator to check.
    strict : bool, default True
        True raises on a prefit calibrator. False downgrades to a printed
        warning — use it only to reproduce an older run, and relabel the
        resulting metrics as in-sample rather than "train OOF".

    Returns
    -------
    str
        A short description of what was found, recorded on the stage as
        ``oof_provenance_`` and saved with the artifact.

    Raises
    ------
    CalibrationLeakageError
        If the calibrator is prefit and ``strict`` is True. The message explains
        the leak and how to fix it.
    """
    prefit, reason = is_prefit_calibrator(calibrated)

    if not prefit:
        return f"refittable ({reason})"

    message = (
        "Calibrator is PREFIT: " + reason + ".\n"
        "Cloning it keeps the base model that was fitted on the whole training\n"
        "split, so cross_val_predict would score every row with a model that\n"
        "trained on it. The resulting probabilities are in-sample, not\n"
        "out-of-fold, and a threshold tuned on them is tuned on memorised data.\n"
        "\n"
        "Fix: have glass_pipeline.lr.calibration.fit_calibration build the\n"
        "calibrator with an integer cv over an UNFITTED estimator, or compute\n"
        "the out-of-fold probabilities from an unfitted clone instead.\n"
        "Set Stage1MLPConfig(strict_oof=False) to proceed anyway — and relabel\n"
        "the metrics, because they will not be out-of-fold."
    )
    if strict:
        raise CalibrationLeakageError(message)

    print("⚠️  " + message.replace("\n", "\n    "))
    return f"PREFIT — out-of-fold probabilities are in-sample ({reason})"