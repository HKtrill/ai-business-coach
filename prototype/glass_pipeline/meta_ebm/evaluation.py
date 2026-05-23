"""
glass_pipeline.meta_ebm.evaluation
=====================================
Evaluating metrics and disagreement analysis for the three-model ensemble.

GLASS-BRW is partial-coverage — disagreement against GLASS is computed
only over the subset where GLASS has an opinion (pass1 | pass2).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
) -> dict:
    """Scalar classification metrics. ROC-AUC only if y_prob provided."""
    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'f2':        fbeta_score(y_true, y_pred, beta=2, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['roc_auc'] = 0.5
    return metrics


def evaluate_with_abstention(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Evaluate arbiter predictions, skipping abstained samples (pred == -1).

    Returns
    -------
    dict: coverage, abstain_rate, metrics (on covered samples only)
    """
    covered  = y_pred != -1
    results  = {
        'coverage':     float(covered.mean()),
        'abstain_rate': float((~covered).mean()),
    }
    if covered.sum() > 0:
        results['metrics'] = compute_metrics(
            y_true[covered], y_pred[covered], y_prob[covered]
        )
    else:
        results['metrics'] = None
    return results


def analyze_disagreements(
    y_true:             np.ndarray,
    lr_pred:            np.ndarray,
    ebm_pred:           np.ndarray,
    glass_pred:         np.ndarray,
    glass_covered_mask: np.ndarray,
) -> dict:
    """
    Compute pairwise disagreement rates and per-model accuracy on those zones.

    LR vs EBM is evaluated over all samples.
    LR vs GLASS and EBM vs GLASS are evaluated only where GLASS has coverage.

    Parameters
    ----------
    y_true             : true labels (train set)
    lr_pred            : LR binary predictions
    ebm_pred           : EBM binary predictions
    glass_pred         : GLASS-BRW binary predictions (0 for pass1, 1 for pass2)
    glass_covered_mask : boolean mask — True where GLASS voted (pass1 | pass2)

    Returns
    -------
    dict of float values, all keys prefixed by pair name.
    """
    report = {}

    # ------------------------------------------------------------------
    # LR vs EBM (full set)
    # ------------------------------------------------------------------
    dis = lr_pred != ebm_pred
    report['lr_ebm_disagree_rate'] = float(dis.mean())
    if dis.sum() > 0:
        report['lr_correct_on_lr_ebm_disagree']  = float(
            (lr_pred[dis]  == y_true[dis]).mean()
        )
        report['ebm_correct_on_lr_ebm_disagree'] = float(
            (ebm_pred[dis] == y_true[dis]).mean()
        )
    else:
        report['lr_correct_on_lr_ebm_disagree']  = float('nan')
        report['ebm_correct_on_lr_ebm_disagree'] = float('nan')

    # ------------------------------------------------------------------
    # GLASS comparisons — covered subset only
    # ------------------------------------------------------------------
    n_covered = glass_covered_mask.sum()
    report['glass_coverage_rate'] = float(glass_covered_mask.mean())

    if n_covered > 0:
        y_g    = y_true[glass_covered_mask]
        lr_g   = lr_pred[glass_covered_mask]
        ebm_g  = ebm_pred[glass_covered_mask]
        gl_g   = glass_pred[glass_covered_mask]

        # LR vs GLASS
        dis_lg = lr_g != gl_g
        report['lr_glass_disagree_rate'] = float(dis_lg.mean())
        if dis_lg.sum() > 0:
            report['lr_correct_on_lr_glass_disagree']    = float(
                (lr_g[dis_lg]  == y_g[dis_lg]).mean()
            )
            report['glass_correct_on_lr_glass_disagree'] = float(
                (gl_g[dis_lg]  == y_g[dis_lg]).mean()
            )
        else:
            report['lr_correct_on_lr_glass_disagree']    = float('nan')
            report['glass_correct_on_lr_glass_disagree'] = float('nan')

        # EBM vs GLASS
        dis_eg = ebm_g != gl_g
        report['ebm_glass_disagree_rate'] = float(dis_eg.mean())
        if dis_eg.sum() > 0:
            report['ebm_correct_on_ebm_glass_disagree']   = float(
                (ebm_g[dis_eg] == y_g[dis_eg]).mean()
            )
            report['glass_correct_on_ebm_glass_disagree'] = float(
                (gl_g[dis_eg]  == y_g[dis_eg]).mean()
            )
        else:
            report['ebm_correct_on_ebm_glass_disagree']   = float('nan')
            report['glass_correct_on_ebm_glass_disagree'] = float('nan')
    else:
        for k in [
            'lr_glass_disagree_rate', 'lr_correct_on_lr_glass_disagree',
            'glass_correct_on_lr_glass_disagree', 'ebm_glass_disagree_rate',
            'ebm_correct_on_ebm_glass_disagree', 'glass_correct_on_ebm_glass_disagree',
        ]:
            report[k] = float('nan')

    return report