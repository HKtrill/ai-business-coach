"""
glass_pipeline.meta_ebm.calibration
===================================
Brier + ECE diagnostics for Stage 4 trust weighting (train-side OOF only).

ECE's first bin is closed on the left, so probabilities of exactly 0.0 —
which isotonic calibration produces (Stage 3 reports ``n_zero_after``) — are
counted, matching ``shared.metrics.calculate_ece``'s convention.
"""

import numpy as np
from sklearn.metrics import brier_score_loss


def compute_calibration(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    brier = brier_score_loss(y_true, y_prob)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = (y_prob >= bins[i]) if i == 0 else (y_prob > bins[i])
        mask = lo & (y_prob <= bins[i + 1])
        if mask.sum():
            ece += abs(y_true[mask].mean() - y_prob[mask].mean()) * mask.mean()

    return {'brier': float(brier), 'ece': float(ece)}
