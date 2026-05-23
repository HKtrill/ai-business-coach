"""
glass_pipeline.meta_ebm.weighting
===================================
Ensemble weight computation for the three-model Glass Cascade arbiter:
LR (Stage 1) + EBM (Stage 3) + GLASS-BRW (Stage 2).

Weight strategy — Brier score + accuracy hybrid (alpha=0.5 default)
--------------------------------------------------------------------
Weights are derived from two complementary signals:

    1. Inverse Brier score  — rewards well-calibrated probability estimates.
       Lower Brier → higher weight. Robust to threshold choice.

    2. Accuracy             — rewards correct hard predictions on the
       training set using each model's optimised decision threshold.

The two components are normalised independently then blended:

    hybrid = alpha * inv_brier + (1 - alpha) * accuracy
    hybrid /= hybrid.sum()          # normalise to sum = 1.0

Design note
-----------
An earlier iteration used ECE + recall instead of Brier + accuracy.
That approach was abandoned because the recall component inflates LR's
weight artificially when its optimised threshold is very low (e.g. 0.10
after isotonic calibration compresses probabilities). Brier score is
threshold-agnostic and accuracy is computed at the optimised threshold,
so neither signal is distorted by threshold magnitude.

This strategy mirrors the original exploratory arbiter (feature_research/
model_training/meta_ebm/arbiter.py :: compute_weights) ported to the
three-model LR + EBM + GLASS-BRW ensemble.
"""

import numpy as np
from sklearn.metrics import accuracy_score


def compute_hybrid_weights(
    lr_cal:    dict,
    ebm_cal:   dict,
    glass_cal: dict,
    y_train:   np.ndarray,
    lr_prob:   np.ndarray,
    ebm_prob:  np.ndarray,
    glass_pred: np.ndarray,
    lr_thresh:  float,
    ebm_thresh: float,
    alpha: float = 0.5,
) -> dict:
    """
    Compute hybrid Brier-score + accuracy ensemble weights for the arbiter.

    Parameters
    ----------
    lr_cal, ebm_cal, glass_cal : calibration dicts — must contain 'brier' key.
                                  Produced by meta_ebm.calibration.compute_calibration.
    y_train     : ground-truth training labels (int array).
    lr_prob     : LR training-set probability array.
    ebm_prob    : EBM training-set probability array.
    glass_pred  : GLASS-BRW hard predictions on training set (int array, 0/1).
                  Pass (glass_decisions_train == 'pass2').astype(int).
    lr_thresh   : LR decision threshold (from artifact).
    ebm_thresh  : EBM decision threshold (from artifact).
    alpha       : blend coefficient. alpha=1.0 → pure Brier weighting;
                  alpha=0.0 → pure accuracy weighting. Default 0.5.

    Returns
    -------
    dict with keys:
        lr, ebm, glass  — normalised float weights summing to 1.0
        details         — sub-dict with raw brier and accuracy arrays
                          for inspection / logging.
    """
    lr_pred  = (lr_prob  >= lr_thresh).astype(int)
    ebm_pred = (ebm_prob >= ebm_thresh).astype(int)

    # ── Inverse Brier (threshold-agnostic calibration signal) ────────────────
    briers = np.array([lr_cal['brier'], ebm_cal['brier'], glass_cal['brier']])
    inv_b  = 1.0 / (briers + 1e-6)
    inv_b  /= inv_b.sum()

    # ── Accuracy (hard-prediction quality at optimised threshold) ────────────
    accs = np.array([
        accuracy_score(y_train, lr_pred),
        accuracy_score(y_train, ebm_pred),
        accuracy_score(y_train, glass_pred),
    ])
    accs /= accs.sum()

    # ── Blend + normalise ────────────────────────────────────────────────────
    hybrid  = alpha * inv_b + (1 - alpha) * accs
    hybrid /= hybrid.sum()

    return {
        'lr':    float(hybrid[0]),
        'ebm':   float(hybrid[1]),
        'glass': float(hybrid[2]),
        'details': {
            'brier':    briers.tolist(),
            'inv_brier': inv_b.tolist(),
            'accuracy': accs.tolist(),
        }
    }