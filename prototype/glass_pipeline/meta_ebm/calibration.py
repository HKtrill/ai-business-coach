import numpy as np
from sklearn.metrics import brier_score_loss

def compute_calibration(y_true, y_prob, n_bins=10):
    brier = brier_score_loss(y_true, y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bins[i]) & (y_prob <= bins[i+1])
        if mask.sum():
            ece += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()

    return {'brier': brier, 'ece': ece}
