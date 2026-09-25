"""
shared.calibration_guard
========================
The prefit leakage guard, run by BOTH arms before computing out-of-fold
probabilities.

Why it exists
-------------
Each Stage 1 arm produces its out-of-fold (OOF) training probabilities by
handing its fitted calibrator to ``sklearn.model_selection.cross_val_predict``.
``cross_val_predict`` CLONES that object once per fold and refits the clone on
the fold's training rows. The result is only out-of-fold if cloning throws the
original fit away:

* ``CalibratedClassifierCV(estimator=<unfitted>, cv=<int>)`` — the clone
  refits the base estimator inside each fold. Honest OOF. This is what
  ``lr.calibration.fit_calibration`` builds today.
* ``CalibratedClassifierCV(cv='prefit')``, or a base estimator wrapped in
  ``sklearn.frozen.FrozenEstimator`` — the clone KEEPS the base model that was
  fitted on the whole training split. Every "out-of-fold" probability then
  comes from a model that trained on that row.

The second case produces numbers that look entirely normal. Nothing downstream
can tell; the only symptom is a Stage 4 meta-learner that trusts an arm more on
train than it should at test. So the check runs before the numbers exist.

History
-------
Written for the black-box arm (``blackbox_pipeline.models.mlp.calibration``),
moved here in PR 33 so GLASS runs it too. Before that GLASS called
``cross_val_predict`` with no check, and was correct only because of how a
shared file happened to be written. The MLP module re-exports these names, so
existing imports still work.

Public API
----------
CalibrationLeakageError
    Raised on a prefit calibrator.
is_prefit_calibrator
    Detect the condition; returns ``(bool, reason)``.
assert_refittable
    Raise (or warn) on it; returns a provenance string to store.
"""

from __future__ import annotations

from typing import Any, Tuple

__all__ = ["CalibrationLeakageError", "is_prefit_calibrator", "assert_refittable"]


class CalibrationLeakageError(RuntimeError):
    """
    A calibrator cannot produce honest out-of-fold predictions.

    Raised by :func:`assert_refittable` when cloning the calibrator would keep
    a base model fitted on the full training split. The message explains the
    leak and how to fix it.
    """


def is_prefit_calibrator(calibrated: Any) -> Tuple[bool, str]:
    """
    Would cloning this estimator preserve an already-fitted base model?

    Parameters
    ----------
    calibrated : estimator
        The object that will be passed to ``cross_val_predict`` — normally the
        ``CalibratedClassifierCV`` returned by ``fit_calibration``.

    Returns
    -------
    is_prefit : bool
        True if a clone would keep a fitted base model.
    reason : str
        What was detected, e.g. ``"CalibratedClassifierCV(cv='prefit')"`` or,
        when safe, ``"cv=10"``. Used in the error message and the provenance
        string.

    Notes
    -----
    Detects three patterns: ``cv='prefit'``; the estimator itself wrapped in
    ``FrozenEstimator``; the ``estimator`` attribute wrapped in
    ``FrozenEstimator``. ``FrozenEstimator`` is only checked when the installed
    sklearn provides it (1.6+).

    Deliberately conservative: anything it cannot positively identify as
    prefit is reported as safe. A false alarm would block a correct pipeline,
    whereas a new prefit pattern would be caught in review of this function.

    Examples
    --------
    >>> from sklearn.calibration import CalibratedClassifierCV
    >>> from sklearn.linear_model import LogisticRegression
    >>> is_prefit_calibrator(CalibratedClassifierCV(LogisticRegression(), cv=10))
    (False, 'cv=10')
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
    except ImportError:  # pragma: no cover - sklearn < 1.6
        pass

    return False, f"cv={cv!r}"


def assert_refittable(calibrated: Any, strict: bool = True) -> str:
    """
    Guard run immediately before computing out-of-fold probabilities.

    Parameters
    ----------
    calibrated : estimator
        The object ``cross_val_predict`` will be handed.
    strict : bool, default True
        True raises :class:`CalibrationLeakageError` on a prefit calibrator.
        False prints a warning instead and lets the caller continue — use only
        to reproduce an older run, and relabel the resulting metrics as
        in-sample rather than "train OOF".

    Returns
    -------
    str
        Provenance, stored on the stage as ``oof_provenance_`` and in both the
        artifact and the ``StageOutput``. ``"refittable (cv=10)"`` when safe;
        a string beginning ``"PREFIT"`` when ``strict=False`` let a leak
        through.

    Raises
    ------
    CalibrationLeakageError
        If the calibrator is prefit and ``strict`` is True.

    See Also
    --------
    is_prefit_calibrator : the detection logic.

    Examples
    --------
    >>> provenance = assert_refittable(stage.calibrated_model)
    >>> provenance
    'refittable (cv=10)'
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
        "Fix: have fit_calibration build the calibrator with an integer cv over\n"
        "an UNFITTED estimator, or compute the out-of-fold probabilities from an\n"
        "unfitted clone instead. Pass strict=False to proceed anyway — and\n"
        "relabel the metrics, because they will not be out-of-fold."
    )
    if strict:
        raise CalibrationLeakageError(message)

    print("⚠️  " + message.replace("\n", "\n    "))
    return f"PREFIT — out-of-fold probabilities are in-sample ({reason})"
