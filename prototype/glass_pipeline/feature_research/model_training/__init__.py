# model_training — per-stage training modules for the Glass Cascade
# Structure mirrors feature_research/: slim __init__, logic in subpackages.
#
#   model_training/
#     lr/        — Logistic Regression (gate stage, balanced)
#     rf/        — Random Forest / GLASS-BRW (recall-biased)
#     ebm/       — EBM (precision-biased, final filter)

from feature_research.model_training.lr import LRResult, evaluate_lr, train_lr, tune_lr

__all__ = ["train_lr", "tune_lr", "evaluate_lr", "LRResult"]