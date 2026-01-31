import numpy as np
from sklearn.metrics import accuracy_score


def analyze_disagreements(
    y_true,
    lr_pred,
    ebm_pred,
    glass_pred,
    glass_covered_mask
):
    """
    Analyze disagreement patterns between LR, EBM, and GLASS.
    Returns interpretable diagnostics for XAI & paper reporting.
    """

    results = {}

    # ---------------------------------
    # LR vs EBM agreement
    # ---------------------------------
    lr_ebm_agree = (lr_pred == ebm_pred)
    results['lr_ebm_agreement_rate'] = lr_ebm_agree.mean()

    if lr_ebm_agree.any():
        results['lr_ebm_accuracy_when_agree'] = accuracy_score(
            y_true[lr_ebm_agree], lr_pred[lr_ebm_agree]
        )

    # ---------------------------------
    # LR+EBM vs GLASS (covered only)
    # ---------------------------------
    covered = glass_covered_mask & lr_ebm_agree

    if covered.any():
        lr_ebm_acc = accuracy_score(
            y_true[covered], lr_pred[covered]
        )
        glass_acc = accuracy_score(
            y_true[covered], glass_pred[covered]
        )

        results['lr_ebm_vs_glass'] = {
            'n_samples': int(covered.sum()),
            'lr_ebm_accuracy': lr_ebm_acc,
            'glass_accuracy': glass_acc
        }

    # ---------------------------------
    # Full disagreement (no majority)
    # ---------------------------------
    full_disagree = glass_covered_mask & (lr_pred != ebm_pred)
    results['lr_vs_ebm_disagreement_rate'] = full_disagree.mean()

    return results
