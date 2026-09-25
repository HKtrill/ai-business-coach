"""
lr.calibration
==============
Probability calibration (Platt / isotonic). Shared by both arms — the MLP
calls ``fit_calibration`` through ``mlp.calibration.fit_stage1_calibration``.

Known limitation (audit F4, unchanged here)
-------------------------------------------
The family is selected by Brier on the SAME training rows the calibrators were
fitted on, which favours isotonic. The winner is recorded, and the request is
preserved separately on each stage, so the choice can at least be reported.

``calculate_ece`` now lives in ``shared.metrics`` (first bin fixed to include
p == 0.0) and is re-exported here for existing imports.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from shared.metrics import calculate_ece

__all__ = ["calculate_ece", "fit_calibration"]


def fit_calibration(model, X_scaled, y_train, method: str = "auto", cv_folds: int = 10):
    """
    Fit sigmoid and/or isotonic calibration, select best by Brier score.

    Returns
    -------
    calibrated_model, selected_method, calibration_metrics
    """
    y_proba_uncal = model.predict_proba(X_scaled)[:, 1]
    brier_uncal = brier_score_loss(y_train, y_proba_uncal)
    ece_uncal   = calculate_ece(np.asarray(y_train), y_proba_uncal)

    print(f"\n📊 Uncalibrated Metrics:")
    print(f"   Brier: {brier_uncal:.6f}")
    print(f"   ECE:   {ece_uncal:.6f}")

    brier_sigmoid = brier_isotonic = float("inf")
    ece_sigmoid   = ece_isotonic   = float("inf")
    cal_sigmoid   = cal_isotonic   = None

    if method in ("sigmoid", "auto"):
        print("\n   Fitting Platt Scaling (sigmoid)...")
        cal_sigmoid = CalibratedClassifierCV(
            estimator=model, method="sigmoid", cv=cv_folds, n_jobs=-1
        )
        cal_sigmoid.fit(X_scaled, y_train)
        p = cal_sigmoid.predict_proba(X_scaled)[:, 1]
        brier_sigmoid = brier_score_loss(y_train, p)
        ece_sigmoid   = calculate_ece(np.asarray(y_train), p)

    if method in ("isotonic", "auto"):
        print("   Fitting Isotonic Regression...")
        cal_isotonic = CalibratedClassifierCV(
            estimator=model, method="isotonic", cv=cv_folds, n_jobs=-1
        )
        cal_isotonic.fit(X_scaled, y_train)
        p = cal_isotonic.predict_proba(X_scaled)[:, 1]
        brier_isotonic = brier_score_loss(y_train, p)
        ece_isotonic   = calculate_ece(np.asarray(y_train), p)

    if method == "auto":
        if brier_sigmoid <= brier_isotonic:
            best_model, best_method = cal_sigmoid,   "sigmoid"
            best_brier, best_ece   = brier_sigmoid,  ece_sigmoid
        else:
            best_model, best_method = cal_isotonic,  "isotonic"
            best_brier, best_ece   = brier_isotonic, ece_isotonic
    elif method == "sigmoid":
        best_model, best_method = cal_sigmoid,  "sigmoid"
        best_brier, best_ece   = brier_sigmoid, ece_sigmoid
    else:
        best_model, best_method = cal_isotonic,  "isotonic"
        best_brier, best_ece   = brier_isotonic, ece_isotonic

    print(f"\n🏆 Selected Calibration: {best_method}")
    print(f"   Brier: {brier_uncal:.6f} → {best_brier:.6f}")
    print(f"   ECE:   {ece_uncal:.6f} → {best_ece:.6f}")

    cal_metrics = {
        "brier_uncalibrated": brier_uncal,
        "brier_calibrated":   best_brier,
        "ece_uncalibrated":   ece_uncal,
        "ece_calibrated":     best_ece,
    }
    return best_model, best_method, cal_metrics