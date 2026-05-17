"""
model_training/meta_ebm/__init__.py
-------------------------------------
Meta-arbiter ensemble package: LR + RF + EBM with bitmask arbiter.
"""

from .trainer  import train_all_models, train_lr, train_rf, train_ebm
from .trainer  import LR_PARAMS, RF_PARAMS, EBM_PARAMS
from .arbiter  import (
    arbiter, eval_arbiter, compute_weights,
    sweep_abstention, run_experiment,
    find_recall_threshold, find_youden_threshold, find_f2_threshold,
    compute_calibration,
)
from .tracer   import run_bitmask_trace
from .analysis import run_missed_analysis

__all__ = [
    # trainer
    "train_all_models", "train_lr", "train_rf", "train_ebm",
    "LR_PARAMS", "RF_PARAMS", "EBM_PARAMS",
    # arbiter
    "arbiter", "eval_arbiter", "compute_weights",
    "sweep_abstention", "run_experiment",
    "find_recall_threshold", "find_youden_threshold", "find_f2_threshold",
    "compute_calibration",
    # tracer
    "run_bitmask_trace",
    # analysis
    "run_missed_analysis",
]