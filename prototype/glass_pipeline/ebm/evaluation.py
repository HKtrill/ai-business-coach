"""
glass_pipeline.ebm.evaluation
==============================
Test-set evaluation for Stage 3 EBM.

Changes from prototype
----------------------
- find_optimal_threshold()  added — maximises F2 (β=2) over the probability
  range [0.10, 0.90]. Consistent with recall-biased tuning objective.
- evaluate_ebm()            now accepts an optional threshold override.
  If threshold=None, the optimal F2 threshold is found automatically.
- proba removed from returned metrics dict — it was an array living alongside
  scalars. Callers retrieve probabilities directly from model.predict_proba().
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_threshold(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    beta: float = 2.0,
    low: float = 0.10,
    high: float = 0.90,
    steps: int = 81,
) -> tuple[float, float]:
    """
    Find the probability threshold that maximises F-beta score.

    Searches linearly between [low, high] at `steps` candidate values.
    Consistent with the recall-biased F2 tuning objective.

    Parameters
    ----------
    y_true  : True binary labels.
    y_proba : Predicted probabilities for the positive class.
    beta    : F-score beta. Default 2 (recall weighted 4× over precision).
    low     : Lower bound of threshold search. Default 0.10.
    high    : Upper bound of threshold search. Default 0.90.
    steps   : Number of candidate thresholds. Default 81.

    Returns
    -------
    optimal_threshold : float
    best_f_beta       : float
    """
    thresholds = np.linspace(low, high, steps)
    scores = [
        fbeta_score(y_true, (y_proba >= t).astype(int), beta=beta, zero_division=0)
        for t in thresholds
    ]
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def evaluate_ebm(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float | None = None,
) -> dict:
    """
    Evaluate EBM on the test set at an optimal or specified threshold.

    Parameters
    ----------
    model     : Fitted ExplainableBoostingClassifier (or calibrated wrapper).
    X_test    : Test features.
    y_test    : True test labels.
    threshold : If None, find_optimal_threshold() is used to maximise F2.
                Pass an explicit float to override (e.g. 0.5 for comparison).

    Returns
    -------
    dict with scalar metrics only:
        Threshold, Accuracy, Precision, Recall, F1, F2, ROC-AUC.
    Probabilities are NOT included — retrieve via model.predict_proba() directly.
    """
    proba = model.predict_proba(X_test)[:, 1]

    if threshold is None:
        threshold, _ = find_optimal_threshold(np.asarray(y_test), proba)

    pred = (proba >= threshold).astype(int)

    return {
        'Threshold': threshold,
        'Accuracy':  accuracy_score(y_test, pred),
        'Precision': precision_score(y_test, pred, zero_division=0),
        'Recall':    recall_score(y_test, pred, zero_division=0),
        'F1':        f1_score(y_test, pred, zero_division=0),
        'F2':        fbeta_score(y_test, pred, beta=2, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_test, proba),
    }