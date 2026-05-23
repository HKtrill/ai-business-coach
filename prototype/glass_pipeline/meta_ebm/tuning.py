"""
glass_pipeline.meta_ebm.tuning
================================
Arbiter threshold tuning for the Meta-EBM stage.

1D sweep over min_weighted_confidence — the sole abstention parameter.
Objective: best F2 on covered samples, subject to a minimum coverage floor.

Ported directly from the abstention sweep in feature_research:
    for mc in np.arange(0.03, 0.55, 0.02):
        abstain when max(conf1, conf0) < mc
        score = F2 on covered samples

Change from prototype
---------------------
scoring:      f1_score → fbeta_score(beta=2)
min_coverage: raw count < 50 → fraction < 0.50
sweep:        renamed tune_confidence_band → tune_arbiter_threshold,
              removed bands (confidence_band) parameter entirely
"""

import numpy as np
from sklearn.metrics import fbeta_score


def tune_arbiter_threshold(
    arbiter_fn,
    probs:        tuple,
    masks:        tuple,
    y_true:       np.ndarray,
    lr_thresh:    float,
    ebm_thresh:   float,
    weights:      dict,
    min_confs:    np.ndarray = None,
    min_coverage: float = 0.50,
) -> dict | None:
    """
    Sweep min_weighted_confidence to find the config with best F2.

    Parameters
    ----------
    arbiter_fn   : meta_arbiter callable
    probs        : (lr_prob_train, ebm_prob_train, glass_prob_train)
    masks        : (pass1_train, pass2_train)
    y_true       : training labels
    lr_thresh    : LR arbiter threshold (recall-targeted)
    ebm_thresh   : EBM arbiter threshold (recall-targeted)
    weights      : MODEL_WEIGHTS dict
    min_confs    : values to sweep — defaults to np.arange(0.03, 0.55, 0.02)
                   matching the original feature_research sweep
    min_coverage : minimum coverage fraction to be eligible (default 0.50)

    Returns
    -------
    dict with keys: min_weighted_confidence, train_f2, train_coverage
    or None if no config met the coverage floor.
    """
    if min_confs is None:
        min_confs = np.arange(0.03, 0.55, 0.02)

    best_score = 0.0
    best_cfg   = None

    for mc in min_confs:
        pred, _, _ = arbiter_fn(
            *probs, *masks,
            lr_thresh, ebm_thresh,
            weights,
            allow_abstain=True,
            min_weighted_confidence=float(mc),
        )

        covered = pred != -1
        if covered.mean() < min_coverage:
            continue

        score = fbeta_score(
            y_true[covered], pred[covered], beta=2, zero_division=0
        )

        if score > best_score:
            best_score = score
            best_cfg   = {
                'min_weighted_confidence': float(mc),
                'train_f2':               float(score),
                'train_coverage':         float(covered.mean()),
            }

    return best_cfg