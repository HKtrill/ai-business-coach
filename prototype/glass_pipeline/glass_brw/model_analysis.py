"""
glass_brw.model_analysis
========================

Reporting and analysis helpers for fitted GLASS-BRW models.

These functions summarize test-set decision flow, subscriber capture,
covered-sample performance, and selected rule quality.
"""

from typing import Any, Dict

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def compute_glass_analysis_metrics(
    glass,
    test_out: Dict[str, Any],
    y_test,
    positive_label: int = 1,
) -> Dict[str, Any]:
    """
    Compute detailed GLASS-BRW test-set analysis metrics.

    Parameters
    ----------
    glass : fitted GLASS-BRW model
        Model containing pass1_rules and pass2_rules.
    test_out : dict
        Standard GLASS-BRW output dict with:
            pred, confidence, decisions, covered, abstained
    y_test : array-like
        Test labels aligned to test_out.
    positive_label : int
        Positive class label.

    Returns
    -------
    metrics : dict
        Detailed decision-flow, subscriber, covered-performance,
        and rule-quality metrics.
    """
    y_test_arr = np.asarray(y_test)
    decisions = np.asarray(test_out["decisions"])
    preds = np.asarray(test_out["pred"])
    confidence = np.asarray(test_out["confidence"])
    covered_mask = np.asarray(test_out["covered"])

    n_test = len(y_test_arr)

    pass1_mask = decisions == "pass1"
    pass2_mask = decisions == "pass2"
    uncertain_mask = decisions == "uncertain"

    n_pass1 = int(pass1_mask.sum())
    n_pass2 = int(pass2_mask.sum())
    n_uncertain = int(uncertain_mask.sum())

    positive_mask = y_test_arr == positive_label
    total_positives = int(positive_mask.sum())

    blocked_positives = int((pass1_mask & positive_mask).sum())
    detected_positives = int((pass2_mask & positive_mask).sum())
    eligible_positives = total_positives - blocked_positives

    eligible_recall = (
        detected_positives / eligible_positives
        if eligible_positives > 0
        else 0.0
    )

    overall_recall = (
        detected_positives / total_positives
        if total_positives > 0
        else 0.0
    )

    if covered_mask.sum() > 0:
        y_covered = y_test_arr[covered_mask]
        pred_covered = preds[covered_mask]
        conf_covered = confidence[covered_mask]

        covered_precision = precision_score(
            y_covered,
            pred_covered,
            pos_label=positive_label,
            zero_division=0,
        )
        covered_recall = recall_score(
            y_covered,
            pred_covered,
            pos_label=positive_label,
            zero_division=0,
        )
        covered_f1 = f1_score(
            y_covered,
            pred_covered,
            pos_label=positive_label,
            zero_division=0,
        )
        covered_avg_confidence = float(conf_covered.mean())
    else:
        covered_precision = 0.0
        covered_recall = 0.0
        covered_f1 = 0.0
        covered_avg_confidence = 0.0

    pass1_precisions = [r.precision for r in glass.pass1_rules]
    pass2_recalls = [r.recall for r in glass.pass2_rules]
    pass2_precisions = [r.precision for r in glass.pass2_rules]

    return {
        # Decision flow
        "n_test": n_test,
        "n_pass1": n_pass1,
        "n_pass2": n_pass2,
        "n_uncertain": n_uncertain,
        "n_covered": int(covered_mask.sum()),
        "coverage_rate": float(covered_mask.mean()),

        # Positive/subscriber analysis
        "total_positives": total_positives,
        "blocked_positives": blocked_positives,
        "detected_positives": detected_positives,
        "eligible_positives": eligible_positives,
        "eligible_recall": eligible_recall,
        "overall_recall": overall_recall,

        # Covered performance
        "covered_precision": covered_precision,
        "covered_recall": covered_recall,
        "covered_f1": covered_f1,
        "covered_avg_confidence": covered_avg_confidence,

        # Rule quality
        "n_pass1_rules": len(glass.pass1_rules),
        "n_pass2_rules": len(glass.pass2_rules),
        "pass1_avg_precision": (
            float(np.mean(pass1_precisions)) if pass1_precisions else 0.0
        ),
        "pass2_avg_recall": (
            float(np.mean(pass2_recalls)) if pass2_recalls else 0.0
        ),
        "pass2_avg_precision": (
            float(np.mean(pass2_precisions)) if pass2_precisions else 0.0
        ),
    }


def print_glass_analysis_report(metrics: Dict[str, Any]) -> None:
    """
    Print detailed GLASS-BRW analysis report.
    """
    n_test = metrics["n_test"]
    total_positives = metrics["total_positives"]

    print("\n" + "=" * 80)
    print("📊 GLASS-BRW PERFORMANCE METRICS")
    print("=" * 80)

    print(f"\n🔀 Decision Flow:")
    print(
        f"   Pass 1 (NOT_SUBSCRIBE): "
        f"{metrics['n_pass1']:,} ({metrics['n_pass1'] / n_test:.1%})"
    )
    print(
        f"   Pass 2 (SUBSCRIBE):     "
        f"{metrics['n_pass2']:,} ({metrics['n_pass2'] / n_test:.1%})"
    )
    print(
        f"   Uncertain (abstain):    "
        f"{metrics['n_uncertain']:,} ({metrics['n_uncertain'] / n_test:.1%})"
    )
    print(
        f"   Total covered:          "
        f"{metrics['n_covered']:,} ({metrics['coverage_rate']:.1%})"
    )

    print(f"\n🎯 Subscriber Analysis:")
    print(f"   Total subscribers:        {metrics['total_positives']:,}")

    if total_positives > 0:
        print(
            f"   Blocked by Pass 1 (leak): "
            f"{metrics['blocked_positives']:,} "
            f"({metrics['blocked_positives'] / total_positives:.1%})"
        )
        print(
            f"   Eligible for Pass 2:      "
            f"{metrics['eligible_positives']:,} "
            f"({metrics['eligible_positives'] / total_positives:.1%})"
        )
        print(f"   Detected by Pass 2:       {metrics['detected_positives']:,}")
        print(f"   Eligible recall:          {metrics['eligible_recall']:.1%}")
        print(f"   Overall recall:           {metrics['overall_recall']:.1%}")
    else:
        print("   No positive-class samples found in test set.")

    print(f"\n📊 Performance on Covered Samples:")
    print(f"   Precision: {metrics['covered_precision']:.3f}")
    print(f"   Recall:    {metrics['covered_recall']:.3f}")
    print(f"   F1-Score:  {metrics['covered_f1']:.3f}")
    print(f"   Avg Conf:  {metrics['covered_avg_confidence']:.3f}")

    print(f"\n📋 Rule Summary:")
    print(f"   Pass 1 rules: {metrics['n_pass1_rules']}")
    print(f"   Pass 2 rules: {metrics['n_pass2_rules']}")

    if metrics["n_pass1_rules"] > 0:
        print(f"   Pass 1 avg precision: {metrics['pass1_avg_precision']:.3f}")

    if metrics["n_pass2_rules"] > 0:
        print(f"   Pass 2 avg recall:    {metrics['pass2_avg_recall']:.3f}")
        print(f"   Pass 2 avg precision: {metrics['pass2_avg_precision']:.3f}")