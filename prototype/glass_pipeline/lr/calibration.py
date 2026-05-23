"""
lr.calibration
==============
Probability calibration (Platt / isotonic) and ECE calculation.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            ece += np.abs(y_true[in_bin].mean() - y_prob[in_bin].mean()) * prop_in_bin
    return ece


def fit_calibration(model, X_scaled, y_train, method: str = "auto", cv_folds: int = 10):
    """
    Fit sigmoid and/or isotonic calibration, select best by Brier score.

    Returns
    -------
    calibrated_model, selected_method, calibration_metrics
    """
    y_proba_uncal = model.predict_proba(X_scaled)[:, 1]
    brier_uncal = brier_score_loss(y_train, y_proba_uncal)
    ece_uncal   = calculate_ece(y_train.values, y_proba_uncal)

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
        ece_sigmoid   = calculate_ece(y_train.values, p)

    if method in ("isotonic", "auto"):
        print("   Fitting Isotonic Regression...")
        cal_isotonic = CalibratedClassifierCV(
            estimator=model, method="isotonic", cv=cv_folds, n_jobs=-1
        )
        cal_isotonic.fit(X_scaled, y_train)
        p = cal_isotonic.predict_proba(X_scaled)[:, 1]
        brier_isotonic = brier_score_loss(y_train, p)
        ece_isotonic   = calculate_ece(y_train.values, p)

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