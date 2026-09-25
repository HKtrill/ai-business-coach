# feature_research/model_training/lr
# ------------------------------------
# Canonical LR training + diagnostics for the Glass Cascade gate stage.
#
# Usage:
#   from feature_research.model_training.lr import train_lr, lr_diagnostics, LRResult

from feature_research.model_training.lr.trainer import (
    LRResult,
    evaluate_lr,
    train_lr,
    tune_lr,
)
from feature_research.model_training.lr.diagnostics import (
    lr_diagnostics,
    compute_vif,
    compute_correlation,
)

__all__ = [
    # training
    "train_lr",
    "tune_lr",
    "evaluate_lr",
    "LRResult",
    # diagnostics
    "lr_diagnostics",
    "compute_vif",
    "compute_correlation",
]