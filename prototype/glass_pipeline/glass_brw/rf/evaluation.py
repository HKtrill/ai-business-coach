"""
glass_brw.rf.evaluation
========================
Evaluation plot for RF Stage 2.

Extracts the 3-panel diagnostic from Cell 12B:
    ROC curve | Precision-Recall curve | Confusion Matrix

Public API
----------
plot_rf_evaluation(rf_result, X_test, y_test, threshold) → None
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

from .rf_training import RFResult

def plot_rf_evaluation(
    rf_result:  RFResult,
    X_test:     pd.DataFrame,
    y_test:     pd.Series,
    threshold:  float = 0.5,
) -> None:
    """
    Render 3-panel RF evaluation figure inline.

    Parameters
    ----------
    rf_result  : RFResult from train_rf_stage()
    X_test     : 29-column binary test DataFrame
    y_test     : binary test labels
    threshold  : decision threshold for PR point and confusion matrix (default 0.5)
    """
    test_proba = rf_result.pipe.predict_proba(X_test)[:, 1]
    test_pred  = (test_proba >= threshold).astype(int)

    auc        = rf_result.metrics_cv["auc_mean"]
    prec_val   = precision_score(y_test, test_pred, zero_division=0)
    rec_val    = recall_score(y_test, test_pred)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── ROC ───────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # ── Precision-Recall ──────────────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(y_test, test_proba)
    axes[1].plot(rec[:-1], prec[:-1], lw=2)
    axes[1].scatter(
        [rec_val], [prec_val],
        color="red", s=200, zorder=5, marker="*",
        label=f"t={threshold:.3f}",
    )
    axes[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # ── Confusion Matrix ──────────────────────────────────────────────────
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, test_pred)
    ).plot(ax=axes[2], colorbar=False)
    axes[2].set_title(f"Confusion Matrix (t={threshold:.3f})")

    plt.suptitle(
        f"RF Stage 2 — {X_test.shape[1]} Binary Bins",
        fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.show()