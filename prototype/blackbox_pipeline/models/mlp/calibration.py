"""
blackbox_pipeline.models.mlp.calibration
=========================================
Probability calibration, delegated to GLASS.

Stage 1 must calibrate the same way the GLASS LR stage does, so this module is a
thin wrapper around ``glass_pipeline.lr.calibration.fit_calibration`` rather than
a reimplementation. Isolating the dependency here means the rest of the package
imports nothing from GLASS, and a change in that API surfaces in one file.

Why the prefit check matters
----------------------------
``fit_calibration`` returns a fitted calibrator. Stage 1 then computes
out-of-fold probabilities by handing that object to ``cross_val_predict``, which
CLONES it per fold. The clone is only honest if cloning discards the fit:

* ``CalibratedClassifierCV(estimator=..., cv=<int>)`` — the clone refits the
  base estimator inside each fold. Out-of-fold, correct.
* ``CalibratedClassifierCV(estimator=<already fitted>, cv="prefit")`` or an
  estimator wrapped in ``sklearn.frozen.FrozenEstimator`` — the clone KEEPS the
  fitted base model. That base model was fitted on the whole training split, so
  every "out-of-fold" probability comes from a model that saw the row. The
  numbers would be in-sample, and the decision threshold chosen from them would
  be tuned on data the model memorised.

``assert_refittable`` detects the second case and refuses. That is the single
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

    Returns ``(calibrated_model, chosen_method, metrics)``. When ``method`` is
    ``"auto"`` the returned method name is the winner GLASS picked, and the
    caller should write it back so downstream reporting names the real method.
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

    Returns ``(is_prefit, reason)``. Conservative: anything it cannot positively
    identify as refittable is reported as safe-by-default with a reason string,
    because a false alarm here would block a correct pipeline.
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
    Guard before computing out-of-fold probabilities.

    Parameters
    ----------
    strict
        True raises on a prefit calibrator. False downgrades to a printed
        warning — use it only to reproduce an older run, and relabel the
        resulting metrics as in-sample rather than "train OOF".

    Returns
    -------
    A short description of what was checked, for the fit report.
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
