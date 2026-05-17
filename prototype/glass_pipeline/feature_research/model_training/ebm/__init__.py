"""
feature_research/model_training/ebm
=====================================
EBM (Explainable Boosting Machine) training and diagnostics.

Recall-biased topology: smooth shape functions + pairwise interactions
capped at 5 to preserve interpretability.

Optimises F2 score (β=2, recall weighted 4× over precision) to provide
directional diversity against the neutral-AUC LR and RF models in the
Glass Cascade.

Public API
----------
train_ebm       — Optuna-tuned recall-biased EBM trainer, returns EBMResult
EBMResult       — result dataclass (pipe, params, metrics, study, runtime_s)
ebm_diagnostics — shape function analysis, composite ranking, tier assignment
"""

from feature_research.model_training.ebm.trainer import EBMResult, train_ebm
from feature_research.model_training.ebm.diagnostics import ebm_diagnostics

__all__ = [
    "train_ebm",
    "EBMResult",
    "ebm_diagnostics",
]