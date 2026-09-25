"""
lr.evaluation
=============
Metrics computation and 2x2 evaluation plot.

``compute_metrics`` now lives in ``shared.metrics`` so both arms run the same
code; it is re-exported here for existing imports. It now also returns
``pr_auc`` and ``pred_pos_rate``, and accepts arrays as well as Series.
"""
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_curve, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
)

from shared.metrics import compute_metrics

__all__ = ["compute_metrics", "plot_evaluation"]


def plot_evaluation(y_test, y_proba, y_pred, metrics: dict, optimal_threshold: float, calibration_method: str):
    """Render the 2x2 evaluation figure inline."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax1 = axes[0, 0]
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    ax1.plot(prob_pred, prob_true, "b-", marker="o", linewidth=2, markersize=8)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect")
    ax1.set_xlabel("Mean Predicted Probability")
    ax1.set_ylabel("Fraction of Positives")
    ax1.set_title(f"Calibration Curve ({calibration_method})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2 = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax2.plot(fpr, tpr, "b-", linewidth=2, label=f'AUC={metrics["roc_auc"]:.3f}')
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax3 = axes[1, 0]
    precisions, recalls, _ = precision_recall_curve(y_test, y_proba)
    ax3.plot(recalls, precisions, "b-", linewidth=2, label=f'PR-AUC={metrics["pr_auc"]:.3f}')
    ax3.scatter(
        [metrics["recall"]], [metrics["precision"]],
        color="red", s=150, zorder=5, marker="*",
        label=f"Selected (t={optimal_threshold:.3f})",
    )
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.set_title("Precision-Recall Curve")
    ax3.legend()
    ax3.grid(alpha=0.3)

    ax4 = axes[1, 1]
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"]).plot(
        ax=ax4, cmap="Blues", values_format="d"
    )
    ax4.set_title(f"Confusion Matrix (t={optimal_threshold:.3f})")

    plt.tight_layout()
    plt.show()
