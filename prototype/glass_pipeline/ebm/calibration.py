import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss


def calculate_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i + 1])
        if mask.any():
            ece += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()
    return ece


def calibrate_if_needed(
    model,
    X_train, y_train,
    X_test, y_test,
    ece_thresh=0.05
):
    """
    Calibrate EBM *using GLOBAL_SPLIT* and return
    TRAIN + TEST calibrated probabilities.
    """

    # raw probs
    train_raw = model.predict_proba(X_train)[:, 1]
    test_raw  = model.predict_proba(X_test)[:, 1]

    ece = calculate_ece(y_train, train_raw)

    if ece <= ece_thresh:
        return (
            None,
            train_raw,
            test_raw,
            ece
        )

    calibrated = CalibratedClassifierCV(
        model,
        method='isotonic',
        cv=3,
        n_jobs=-1
    )
    calibrated.fit(X_train, y_train)

    train_cal = calibrated.predict_proba(X_train)[:, 1]
    test_cal  = calibrated.predict_proba(X_test)[:, 1]

    return (
        calibrated,
        train_cal,
        test_cal,
        ece
    )

