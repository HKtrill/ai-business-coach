"""
glass_pipeline.meta_ebm.weighting
===================================
Ensemble weight computation for the three-model GLASS cascade arbiter:
GLASS LR (Stage 1) + GLASS Router (Stage 2) + GLASS EBM (Stage 3).

All inputs are TRAIN-side out-of-fold predictions.

Weight strategy — Brier score + accuracy hybrid (alpha=0.5 default)
--------------------------------------------------------------------
Weights are derived from two complementary signals:

    1. Inverse Brier score  — rewards well-calibrated probability estimates.
       Lower Brier → higher weight. Robust to threshold choice.

    2. Accuracy             — rewards correct hard predictions on the
       training set using each model's optimised decision threshold.
       The GLASS Router's accuracy is measured on the rows it covers
       (Pass 1 | Pass 2) — the only rows where it votes, and the same
       population its Brier score is computed on. Scoring it on all rows
       would count every abstention as a correct-or-wrong "0" vote.

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
three-model LR + GLASS Router + EBM ensemble.
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
    glass_covered_mask: np.ndarray = None,
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
    glass_pred  : GLASS Router hard predictions on training set (int array,
                  0/1): (glass_decisions_train == 'pass2').astype(int).
    lr_thresh   : LR decision threshold (Stage 4 recall-targeted).
    ebm_thresh  : EBM decision threshold (Stage 4 recall-targeted).
    alpha       : blend coefficient. alpha=1.0 → pure Brier weighting;
                  alpha=0.0 → pure accuracy weighting. Default 0.5.
    glass_covered_mask : rows the Router votes on (Pass 1 | Pass 2). When
                  given, GLASS accuracy is scored on those rows only. ``None``
                  reproduces the pre-audit behaviour (all rows; abstentions
                  counted as 0 votes) and is kept only for comparison.

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
    y_train    = np.asarray(y_train)
    glass_pred = np.asarray(glass_pred)
    if glass_covered_mask is None:
        g_y, g_pred, g_pop = y_train, glass_pred, "all_rows (pre-audit)"
    else:
        m = np.asarray(glass_covered_mask, dtype=bool)
        g_y, g_pred, g_pop = y_train[m], glass_pred[m], "covered_rows"
    accs_raw = np.array([
        accuracy_score(y_train, lr_pred),
        accuracy_score(y_train, ebm_pred),
        accuracy_score(g_y, g_pred),
    ])
    accs = accs_raw.copy()
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
            'accuracy_raw': accs_raw.tolist(),
            'glass_accuracy_population': g_pop,
            'alpha': float(alpha),
        }
    }