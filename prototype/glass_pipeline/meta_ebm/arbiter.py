"""
glass_pipeline.meta_ebm.arbiter
================================
Weighted confidence arbiter for the three-model Glass Cascade ensemble:
LR (Stage 1) + EBM (Stage 3) + GLASS-BRW (Stage 2).

GLASS-BRW is a partial-coverage model — it only has opinions on samples
it routes to pass1 or pass2. The arbiter handles this via explicit masks.

Abstention strategy
-------------------
Abstain when the winning side's weighted confidence is below min_conf:

    if max(conf1, conf0) < min_weighted_confidence:
        abstain

Ported directly from feature_research/model_training/meta_ebm/arbiter.py.
"""

import numpy as np


def meta_arbiter(
    lr_prob:    np.ndarray,
    ebm_prob:   np.ndarray,
    glass_prob: np.ndarray,
    pass1_mask: np.ndarray,
    pass2_mask: np.ndarray,
    lr_thresh:  float,
    ebm_thresh: float,
    weights:    dict,
    allow_abstain:           bool  = True,
    min_weighted_confidence: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted confidence arbiter with optional abstention.

    LR and EBM vote on every sample.
    GLASS-BRW votes only where pass1_mask | pass2_mask is True.

    Abstains when the winning side's weighted confidence is below
    min_weighted_confidence.

    Parameters
    ----------
    lr_prob, ebm_prob, glass_prob : probability arrays (length n)
    pass1_mask  : GLASS-BRW routed to pass1 (covered, predicts 0)
    pass2_mask  : GLASS-BRW routed to pass2 (covered, predicts 1)
    lr_thresh   : LR decision threshold (recall-targeted)
    ebm_thresh  : EBM decision threshold (recall-targeted)
    weights     : dict(lr, ebm, glass) — hybrid weights
    allow_abstain           : enable abstention
    min_weighted_confidence : abstain when max(conf1, conf0) < this value

    Returns
    -------
    pred    : int array  (-1=abstain, 0=negative, 1=positive)
    prob    : float array (weighted ensemble probability)
    explain : str array  (per-sample decision explanation)
    """
    n       = len(lr_prob)
    pred    = np.zeros(n, dtype=int)
    prob    = np.zeros(n, dtype=float)
    explain = np.empty(n, dtype=object)

    for i in range(n):
        votes, confs, probs, ws = [], [], [], []

        lp = lr_prob[i]
        votes.append(int(lp >= lr_thresh))
        confs.append(abs(lp - lr_thresh))
        probs.append(lp)
        ws.append(weights['lr'])

        ep = ebm_prob[i]
        votes.append(int(ep >= ebm_thresh))
        confs.append(abs(ep - ebm_thresh))
        probs.append(ep)
        ws.append(weights['ebm'])

        if pass1_mask[i] or pass2_mask[i]:
            gp = glass_prob[i]
            votes.append(1 if pass2_mask[i] else 0)
            confs.append(abs(gp - 0.5))
            probs.append(gp)
            ws.append(weights['glass'])

        ws_arr  = np.array(ws)
        ws_arr  = ws_arr / ws_arr.sum()
        votes   = np.array(votes)
        confs   = np.array(confs)
        probs_a = np.array(probs)

        weighted_prob = float(np.sum(probs_a * ws_arr))
        conf1 = float(np.sum(confs * ws_arr * (votes == 1)))
        conf0 = float(np.sum(confs * ws_arr * (votes == 0)))

        prob[i] = weighted_prob

        if allow_abstain and max(conf1, conf0) < min_weighted_confidence:
            pred[i]    = -1
            explain[i] = "ABSTAIN: low weighted confidence"
            continue

        if conf1 > conf0:
            pred[i]    = 1
            explain[i] = "PREDICT 1: weighted confidence"
        else:
            pred[i]    = 0
            explain[i] = "PREDICT 0: weighted confidence"

    return pred, prob, explain