import numpy as np

def enhanced_meta_arbiter(
    lr_prob, ebm_prob, glass_prob,
    pass1_mask, pass2_mask,
    lr_thresh, ebm_thresh,
    weights,
    allow_abstain=True,
    confidence_band=0.10,
    min_weighted_confidence=0.15
):
    n = len(lr_prob)
    pred = np.zeros(n, dtype=int)
    prob = np.zeros(n)
    explain = []

    for i in range(n):
        votes, confs, probs, ws = [], [], [], []

        # LR
        lp = lr_prob[i]
        votes.append(int(lp >= lr_thresh))
        confs.append(abs(lp - lr_thresh))
        probs.append(lp)
        ws.append(weights['lr'])

        # EBM
        ep = ebm_prob[i]
        votes.append(int(ep >= ebm_thresh))
        confs.append(abs(ep - ebm_thresh))
        probs.append(ep)
        ws.append(weights['ebm'])

        # GLASS
        if pass1_mask[i] or pass2_mask[i]:
            gp = glass_prob[i]
            votes.append(1 if pass2_mask[i] else 0)
            confs.append(abs(gp - 0.5))
            probs.append(gp)
            ws.append(weights['glass'])

        ws = np.array(ws) / np.sum(ws)
        votes = np.array(votes)
        confs = np.array(confs)
        probs = np.array(probs)

        weighted_prob = np.sum(probs * ws)
        conf1 = np.sum(confs * ws * (votes == 1))
        conf0 = np.sum(confs * ws * (votes == 0))

        if allow_abstain and max(conf1, conf0) < min_weighted_confidence:
            pred[i] = -1
            prob[i] = weighted_prob
            explain.append("ABSTAIN: low confidence")
            continue

        if conf1 > conf0:
            pred[i] = 1
            explain.append("PREDICT 1: weighted confidence")
        else:
            pred[i] = 0
            explain.append("PREDICT 0: weighted confidence")

        prob[i] = weighted_prob

    return pred, prob, np.array(explain)
