"""
glass_brw.model_outputs
=======================

Helpers for constructing GLASS-BRW prediction outputs and summary metrics.

These utilities sit between the fitted GLASS-BRW model and artifact saving.
They are intentionally kept outside core/ because they are pipeline/output
helpers, not rule or model primitives.
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import precision_score, recall_score


def build_glass_output(
    glass,
    X,
    abstain_value: int = -1,
) -> Dict[str, Any]:
    """
    Build standardized GLASS-BRW prediction output.

    Returns the output contract expected by ModelSaver:
        pred, confidence, decisions, covered, abstained
    """
    preds, conf, decisions = glass.predict(X)

    return {
        "pred": preds,
        "confidence": conf,
        "decisions": decisions,
        "covered": preds != abstain_value,
        "abstained": preds == abstain_value,
    }


def compute_glass_summary_metrics(
    test_out: Dict[str, Any],
    y_test,
    positive_label: int = 1,
) -> Dict[str, Any]:
    """
    Compute final GLASS-BRW summary metrics used by ModelSaver.print_final_summary.
    """
    test_pass1_mask = np.array([d == "pass1" for d in test_out["decisions"]])
    test_pass2_mask = np.array([d == "pass2" for d in test_out["decisions"]])

    test_pred = test_out["pred"]

    is_positive = (y_test == positive_label).values
    total_positives = is_positive.sum()

    pass1_blocked_positives = np.sum(test_pass1_mask & is_positive)
    pass2_detected_positives = np.sum(test_pass2_mask & is_positive)

    overall_recall = (
        pass2_detected_positives / total_positives
        if total_positives > 0
        else 0.0
    )

    covered = test_out["covered"]

    if covered.any():
        covered_precision = precision_score(
            y_test[covered],
            test_pred[covered],
            pos_label=positive_label,
            zero_division=0,
        )
        covered_recall = recall_score(
            y_test[covered],
            test_pred[covered],
            pos_label=positive_label,
            zero_division=0,
        )
    else:
        covered_precision = 0.0
        covered_recall = 0.0

    return {
        "pass1_blocked_subscribers": pass1_blocked_positives,
        "total_subscribers": total_positives,
        "overall_recall": overall_recall,
        "covered_precision": covered_precision,
        "covered_recall": covered_recall,
    }