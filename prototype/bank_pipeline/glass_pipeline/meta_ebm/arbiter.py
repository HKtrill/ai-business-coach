"""
glass_pipeline.meta_ebm.arbiter
================================
Weighted confidence arbiter for the three-model GLASS cascade ensemble:
GLASS LR (Stage 1) + GLASS Router (Stage 2) + GLASS EBM (Stage 3).

The GLASS Router is a partial-coverage model — it only votes on rows it
routes to Pass 1 (NOT_SUBSCRIBE) or Pass 2 (SUBSCRIBE); everything else is
abstain/uncovered. The arbiter handles this via explicit masks.

Votes and confidence
--------------------
Every voter's confidence is the distance of its estimate from its own
decision point, in its vote's direction:

    LR / EBM   vote p >= t;             confidence |p - t|   (t: Stage 4 recall cut)
    Router     vote = route;            confidence: the matched rule's positive
               (Pass 2 -> 1, Pass 1 -> 0)  rate vs the base rate, directional and
                                         clipped at 0 (``router_evidence``, PR 36 F25)

Weights are renormalised over the voters of each row; ties go to 0.

Abstention strategy
-------------------
Abstain when the winning side's weighted confidence is below min_conf:

    if max(conf1, conf0) < min_weighted_confidence:
        abstain

Ported from feature_research/model_training/meta_ebm/arbiter.py; the router
confidence was |p - 0.5| until PR 36 F25.
"""

import numpy as np

from .router_evidence import directional_confidence, rule_positive_rate


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
    glass_base_rate=None,
    return_confidence: bool = False,
) -> tuple:
    """
    Weighted confidence arbiter with optional abstention.

    LR and EBM vote on every sample.
    The GLASS Router votes only where pass1_mask | pass2_mask is True; its
    vote is the route (Pass 2 → 1, Pass 1 → 0) and its confidence is the
    matched rule's positive rate measured against ``glass_base_rate`` in the
    vote's direction (``router_evidence``; zero when the rule is on the wrong
    side of the base rate).

    Abstains when the winning side's weighted confidence is below
    min_weighted_confidence.

    Parameters
    ----------
    lr_prob, ebm_prob, glass_prob : probability arrays (length n)
    pass1_mask  : GLASS Router Pass 1 (covered, NOT_SUBSCRIBE → votes 0)
    pass2_mask  : GLASS Router Pass 2 (covered, SUBSCRIBE → votes 1)
    lr_thresh   : LR decision threshold (recall-targeted)
    ebm_thresh  : EBM decision threshold (recall-targeted)
    weights     : dict(lr, ebm, glass) — hybrid weights
    allow_abstain           : enable abstention
    min_weighted_confidence : abstain when max(conf1, conf0) < this value
    glass_base_rate         : required. The router's training base rate per
                              row (scalar or array): for out-of-fold training
                              rows the base rate of each fold model's training
                              rows, for test rows the full training base rate
    return_confidence       : also return each row's max(conf1, conf0), the
                              value compared with min_weighted_confidence
                              (the abstention search reads it; PR 36 F30)

    Returns
    -------
    pred    : int array  (-1=abstain, 0=negative, 1=positive)
    prob    : float array (weighted ensemble probability)
    explain : str array  (per-sample decision explanation)
    confidence : float array, only with ``return_confidence=True``
    """
    n = len(lr_prob)
    # PR 36 F27: misaligned inputs used to be truncated or broadcast silently.
    lengths = {k: len(v) for k, v in (("lr_prob", lr_prob), ("ebm_prob", ebm_prob),
                                       ("glass_prob", glass_prob), ("pass1_mask", pass1_mask),
                                       ("pass2_mask", pass2_mask))}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"meta_arbiter: inputs differ in length {lengths}")
    if np.any(np.asarray(pass1_mask, dtype=bool) & np.asarray(pass2_mask, dtype=bool)):
        raise ValueError("meta_arbiter: a row is routed to both Pass 1 and Pass 2")
    missing = {"lr", "ebm", "glass"} - set(weights)
    if missing:
        raise KeyError(f"meta_arbiter: weights missing {sorted(missing)}")
    if glass_base_rate is None:
        raise ValueError("meta_arbiter: glass_base_rate is required (PR 36 F25)")
    glass_conf = directional_confidence(
        rule_positive_rate(glass_prob, pass1_mask, pass2_mask, glass_base_rate),
        pass1_mask, pass2_mask, glass_base_rate)

    pred    = np.zeros(n, dtype=int)
    prob    = np.zeros(n, dtype=float)
    explain = np.empty(n, dtype=object)
    side_conf = np.zeros(n, dtype=float)

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
            confs.append(glass_conf[i])
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
        side_conf[i] = max(conf1, conf0)

        if allow_abstain and side_conf[i] < min_weighted_confidence:
            pred[i]    = -1
            explain[i] = "ABSTAIN: low weighted confidence"
            continue

        if conf1 > conf0:
            pred[i]    = 1
            explain[i] = "PREDICT 1: weighted confidence"
        else:
            pred[i]    = 0
            explain[i] = "PREDICT 0: weighted confidence"

    if return_confidence:
        return pred, prob, explain, side_conf
    return pred, prob, explain