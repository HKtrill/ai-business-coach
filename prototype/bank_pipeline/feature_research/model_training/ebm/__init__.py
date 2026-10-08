"""
feature_research/model_training/ebm
===================================
Stage 3 EBM (Explainable Boosting Machine): training and diagnostics.

Smooth shape functions + up to 5 pairwise interactions (cap kept for
interpretability). Balanced sample weights, as in LR/RF. Hyperparameters are
tuned on ½ AUC + ½ high-recall partial AUC (TPR ≥ 0.70) by default
(objective="auc" / "f2" available); the operating threshold is chosen
downstream on out-of-fold scores.

Modules
-------
model        search space, canonical params, fit/score helpers
trainer      train_ebm, EBMResult (protocol: per-fold features, OOF metrics)
terms        term table read from the fitted EBM
ranking      redundancy, correlated pairs, composite ranking, tiers
plots        six-panel diagnostic figure
diagnostics  ebm_diagnostics (terms → ranking → report → plots)

Public API
----------
train_ebm, EBMResult, ebm_diagnostics
"""

from feature_research.model_training.ebm.trainer import EBMResult, train_ebm
from feature_research.model_training.ebm.diagnostics import ebm_diagnostics

__all__ = ["train_ebm", "EBMResult", "ebm_diagnostics"]
