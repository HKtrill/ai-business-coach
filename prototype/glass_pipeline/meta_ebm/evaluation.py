import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score
)


def compute_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'f2': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['roc_auc'] = 0.5

    return metrics


def evaluate_with_abstention(y_true, y_pred, y_prob):
    covered = y_pred != -1
    abstained = ~covered

    results = {
        'coverage': covered.mean(),
        'abstain_rate': abstained.mean(),
    }

    if covered.sum() > 0:
        results['metrics'] = compute_metrics(
            y_true[covered],
            y_pred[covered],
            y_prob[covered]
        )
    else:
        results['metrics'] = None

    return results
