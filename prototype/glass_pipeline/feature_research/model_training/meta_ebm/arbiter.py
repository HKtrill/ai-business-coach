"""
model_training/meta_ebm/arbiter.py
-------------------------------------
Pure arbiter helper functions for the meta-arbiter ensemble.
No I/O, no side effects — all functions are importable and
independently testable.

Public API
----------
find_recall_threshold(y_true, y_prob, target_recall) -> float
find_youden_threshold(y_true, y_prob) -> (thresh, tpr, fpr)
find_f2_threshold(y_true, y_prob)    -> (thresh, recall, prec, f2)
compute_calibration(y_true, y_prob)  -> (brier, ece)
compute_weights(y_true, probs, preds) -> dict(lr, rf, ebm)
arbiter(lr_p, rf_p, ebm_p, ...)     -> (pred, prob)
eval_arbiter(y_true, pred, prob)    -> metrics dict
sweep_abstention(...)               -> (best_metrics, best_conf)
run_experiment(...)                 -> (m_na, best_m, best_c, weights, prob_na)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    roc_curve,
)


# ══════════════════════════════════════════════════════════════════════════════
# THRESHOLD SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def find_recall_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_recall: float = 0.70,
) -> float:
    """Return decision threshold that hits target_recall on the provided set."""
    precs, recs, threshs = precision_recall_curve(y_true, y_prob)
    idx = np.argmin(np.abs(recs[:-1] - target_recall))
    return float(threshs[idx])


def find_youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float, float]:
    """
    Threshold at maximum Youden's J (TPR - FPR).

    Returns
    -------
    threshold, tpr, fpr
    """
    fpr, tpr, threshs = roc_curve(y_true, y_prob)
    opt = np.argmax(tpr - fpr)
    return float(threshs[opt]), float(tpr[opt]), float(fpr[opt])


def find_f2_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Threshold at maximum F2 score (recall-weighted).

    Returns
    -------
    threshold, recall, precision, f2
    """
    precs, recs, threshs = precision_recall_curve(y_true, y_prob)
    f2s = (5 * precs[:-1] * recs[:-1]) / (4 * precs[:-1] + recs[:-1] + 1e-10)
    best = np.argmax(f2s)
    return float(threshs[best]), float(recs[best]), float(precs[best]), float(f2s[best])


# ══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """
    Compute Brier score and Expected Calibration Error (ECE, 10 bins).

    Returns
    -------
    brier, ece
    """
    brier = brier_score_loss(y_true, y_prob)
    bins  = np.linspace(0, 1, 11)
    ece   = sum(
        abs(
            y_true[(y_prob > bins[i]) & (y_prob <= bins[i + 1])].mean()
            - y_prob[(y_prob > bins[i]) & (y_prob <= bins[i + 1])].mean()
        )
        * ((y_prob > bins[i]) & (y_prob <= bins[i + 1])).mean()
        for i in range(10)
        if ((y_prob > bins[i]) & (y_prob <= bins[i + 1])).sum() > 0
    )
    return float(brier), float(ece)


# ══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def compute_weights(
    y_true: np.ndarray,
    probs: list,
    preds: list,
) -> dict:
    """
    Hybrid Brier-score + accuracy ensemble weights for [LR, RF, EBM].

    Parameters
    ----------
    probs : [lr_p_train, rf_p_train, ebm_p_train]
    preds : [lr_pred_train, rf_pred_train, ebm_pred_train]

    Returns
    -------
    dict(lr=float, rf=float, ebm=float)  — sum to 1.0
    """
    briers = np.array([brier_score_loss(y_true, p) for p in probs])
    inv_b  = 1.0 / (briers + 1e-6)
    accs   = np.array([accuracy_score(y_true, p) for p in preds])
    hybrid = 0.5 * (inv_b / inv_b.sum()) + 0.5 * (accs / accs.sum())
    hybrid /= hybrid.sum()
    return {"lr": float(hybrid[0]), "rf": float(hybrid[1]), "ebm": float(hybrid[2])}


# ══════════════════════════════════════════════════════════════════════════════
# CORE ARBITER
# ══════════════════════════════════════════════════════════════════════════════

def arbiter(
    lr_p: np.ndarray,
    rf_p: np.ndarray,
    ebm_p: np.ndarray,
    lr_t: float,
    rf_t: float,
    ebm_t: float,
    weights: dict,
    abstain: bool = True,
    min_conf: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted confidence arbiter with optional abstention.

    Each sample's prediction is determined by comparing the
    weighted confidence of the positive vote vs the negative vote.
    When the winning side's confidence is below min_conf, the
    arbiter abstains (-1) rather than guessing.

    Parameters
    ----------
    lr_p, rf_p, ebm_p : test-set probability arrays
    lr_t, rf_t, ebm_t : per-model decision thresholds
    weights            : dict(lr, rf, ebm) from compute_weights()
    abstain            : enable abstention
    min_conf           : minimum weighted confidence to commit

    Returns
    -------
    pred : int array  (-1=abstain, 0=negative, 1=positive)
    prob : float array (weighted ensemble probability)
    """
    n   = len(lr_p)
    pred = np.zeros(n, dtype=int)
    prob = np.zeros(n, dtype=float)
    wa   = np.array([weights["lr"], weights["rf"], weights["ebm"]])
    wa  /= wa.sum()
    threshs = np.array([lr_t, rf_t, ebm_t])

    for i in range(n):
        ps    = np.array([lr_p[i], rf_p[i], ebm_p[i]])
        votes = (ps >= threshs).astype(int)
        confs = np.abs(ps - threshs)
        prob[i] = np.dot(ps, wa)
        c1 = np.sum(confs * wa * (votes == 1))
        c0 = np.sum(confs * wa * (votes == 0))
        if abstain and max(c1, c0) < min_conf:
            pred[i] = -1
        elif c1 > c0:
            pred[i] = 1
        # else: pred[i] = 0 (already initialised)

    return pred, prob


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def eval_arbiter(
    y_true: np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
) -> dict:
    """
    Evaluate arbiter predictions, skipping abstained samples (pred == -1).

    Returns
    -------
    dict: coverage, accuracy, recall, precision, f1, f2, auc
    """
    cov = pred != -1
    if cov.sum() == 0:
        return {k: 0.0 for k in ["coverage", "accuracy", "recall",
                                   "precision", "f1", "f2", "auc"]}
    yc, pc, prc = y_true[cov], pred[cov], prob[cov]
    return {
        "coverage":  float(cov.mean()),
        "accuracy":  float(accuracy_score(yc, pc)),
        "recall":    float(recall_score(yc, pc, zero_division=0)),
        "precision": float(precision_score(yc, pc, zero_division=0)),
        "f1":        float(f1_score(yc, pc, zero_division=0)),
        "f2":        float(fbeta_score(yc, pc, beta=2, zero_division=0)),
        "auc":       float(roc_auc_score(yc, prc)) if len(np.unique(yc)) > 1 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ABSTENTION SWEEP
# ══════════════════════════════════════════════════════════════════════════════

# Default score weights — tuned to balance recall and precision rather than
# rewarding coverage so heavily that min_conf always bottoms out at 0.03.
_DEFAULT_SWEEP_WEIGHTS = {"recall": 0.50, "precision": 0.30, "coverage": 0.20}


def sweep_abstention(
    lr_p: np.ndarray,
    rf_p: np.ndarray,
    ebm_p: np.ndarray,
    lr_t: float,
    rf_t: float,
    ebm_t: float,
    weights: dict,
    y_true: np.ndarray,
    mc_range: np.ndarray | None = None,
    min_coverage: float = 0.50,
    score_weights: dict | None = None,
    verbose: bool = True,
) -> tuple[dict | None, float | None]:
    """
    Sweep min_conf values and find the best abstention threshold.

    Parameters
    ----------
    mc_range      : array of min_conf values to try
    min_coverage  : hard floor — configs below this coverage are ignored
    score_weights : dict(recall, precision, coverage) controlling the
                    scoring objective. Defaults to 0.50/0.30/0.20.

    Returns
    -------
    best_metrics : dict (or None if no config passed min_coverage)
    best_conf    : float (or None)
    """
    if mc_range is None:
        mc_range = np.arange(0.03, 0.55, 0.02)
    sw = score_weights or _DEFAULT_SWEEP_WEIGHTS

    if verbose:
        print(
            f"\n   {'mc':>6} {'cover':>8} {'acc':>8} "
            f"{'recall':>8} {'prec':>8} {'f1':>8} {'f2':>8}"
        )
        print(f"   {'-' * 58}")

    best_score, best_conf, best_metrics = 0.0, None, None

    for mc in mc_range:
        pred, prob = arbiter(lr_p, rf_p, ebm_p, lr_t, rf_t, ebm_t, weights, True, mc)
        m = eval_arbiter(y_true, pred, prob)
        score = (
            sw["recall"]    * m["recall"]
            + sw["precision"] * m["precision"]
            + sw["coverage"]  * m["coverage"]
        )
        if verbose:
            print(
                f"   {mc:>6.2f} {m['coverage']:>8.1%} {m['accuracy']:>8.4f} "
                f"{m['recall']:>8.4f} {m['precision']:>8.4f} "
                f"{m['f1']:>8.4f} {m['f2']:>8.4f}"
            )
        if score > best_score and m["coverage"] >= min_coverage:
            best_score, best_conf, best_metrics = score, mc, m.copy()

    return best_metrics, best_conf


# ══════════════════════════════════════════════════════════════════════════════
# FULL EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(
    lr_ptr: np.ndarray,
    rf_ptr: np.ndarray,
    ebm_ptr: np.ndarray,
    lr_pte: np.ndarray,
    rf_pte: np.ndarray,
    ebm_pte: np.ndarray,
    lr_t: float,
    rf_t: float,
    ebm_t: float,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label: str = "",
    run_sweep: bool = True,
    sweep_score_weights: dict | None = None,
    verbose: bool = True,
) -> tuple:
    """
    Full arbiter experiment: calibration → weights → no-abstain → sweep.

    Returns
    -------
    m_na      : no-abstention metrics dict
    best_m    : best sweep metrics dict (None if run_sweep=False)
    best_c    : best min_conf (None if run_sweep=False)
    W         : ensemble weights dict
    prob_na   : ensemble probability array on test set
    """
    if verbose:
        print(f"\n{'─' * 80}")
        print(f"   ⚙️  CONFIG: {label}")
        print(f"   📐 Thresholds: LR={lr_t:.4f}, RF={rf_t:.4f}, EBM={ebm_t:.4f}")
        print(f"{'─' * 80}")

    # Calibration
    for name, prob in [("LR", lr_ptr), ("RF", rf_ptr), ("EBM", ebm_ptr)]:
        b, e = compute_calibration(y_train, prob)
        if verbose:
            print(f"   {name:<5} Brier={b:.4f}  ECE={e:.4f}")

    # Disagreement on train set
    lr_pt  = (lr_ptr  >= lr_t).astype(int)
    rf_pt  = (rf_ptr  >= rf_t).astype(int)
    ebm_pt = (ebm_ptr >= ebm_t).astype(int)

    if verbose:
        for la, lb, pa, pb in [
            ("LR", "RF", lr_pt, rf_pt),
            ("LR", "EBM", lr_pt, ebm_pt),
            ("RF", "EBM", rf_pt, ebm_pt),
        ]:
            dis = pa != pb
            if dis.sum():
                print(
                    f"   {la} vs {lb}: disagree {dis.mean():.1%}  →  "
                    f"{la} correct {(pa[dis] == y_train[dis]).mean():.1%},  "
                    f"{lb} correct {(pb[dis] == y_train[dis]).mean():.1%}"
                )

    # Weights
    W = compute_weights(y_train, [lr_ptr, rf_ptr, ebm_ptr], [lr_pt, rf_pt, ebm_pt])
    if verbose:
        print(f"   Weights: LR={W['lr']:.3f}, RF={W['rf']:.3f}, EBM={W['ebm']:.3f}")

    # No-abstention
    pred_na, prob_na = arbiter(
        lr_pte, rf_pte, ebm_pte, lr_t, rf_t, ebm_t, W, abstain=False
    )
    m_na = eval_arbiter(y_test, pred_na, prob_na)
    if verbose:
        print(f"\n   No-Abstention:")
        for k, v in m_na.items():
            print(f"      {k:<12} {v:.4f}")

    # Sweep
    best_m, best_c = None, None
    if run_sweep:
        if verbose:
            print(f"\n   Abstention sweep:")
        best_m, best_c = sweep_abstention(
            lr_pte, rf_pte, ebm_pte, lr_t, rf_t, ebm_t, W, y_test,
            score_weights=sweep_score_weights,
            verbose=verbose,
        )
        if best_m and verbose:
            print(f"\n   ✨ Best: min_conf={best_c:.2f}")
            for k, v in best_m.items():
                print(f"      {k:<12} {v:.4f}")

    return m_na, best_m, best_c, W, prob_na