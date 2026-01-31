import numpy as np
from sklearn.metrics import f1_score

def tune_confidence_band(
    arbiter_fn, probs, masks, y_true,
    lr_thresh, ebm_thresh, weights,
    bands, min_confs
):
    best = None
    best_score = 0

    for b in bands:
        for mc in min_confs:
            pred, _, _ = arbiter_fn(
                *probs, *masks,
                lr_thresh, ebm_thresh,
                weights,
                allow_abstain=True,
                confidence_band=b,
                min_weighted_confidence=mc
            )
            covered = pred != -1
            if covered.sum() < 50:
                continue

            score = f1_score(y_true[covered], pred[covered], zero_division=0)
            if score > best_score:
                best_score = score
                best = {'confidence_band': b, 'min_weighted_confidence': mc}

    return best
