"""
lr
==
Logistic Regression stage package for the Glass Cascade pipeline.

Public API (notebook-facing)
----------------------------
    from lr import engineer_features          # feature engineering
    from lr import analyze_and_prune_features # correlation pruning
    from lr import train_lr_stage             # calibrated stage (production)
    from lr import train_baseline_lr_stage    # baseline stage (comparison)

Internal modules (not for direct notebook import)
-------------------------------------------------
    lr.base_stage      BaseLRStage (shared scaffolding)
    lr.calibration     fit_calibration, calculate_ece
    lr.evaluation      compute_metrics, plot_evaluation
    lr.tuning          run_optuna_tuning
    lr.artifacts       save_artifact
"""

from .feature_engineering import engineer_features, LRFeatureEngineer
from .lr_stage_calibrated import train_lr_stage, CalibratedLRStage
from .lr_stage import train_baseline_lr_stage, LRStage
from .correlation_analysis import analyze_and_prune_features, CorrelationAnalyzer

__all__ = [
    # Notebook entry points
    "engineer_features",
    "train_lr_stage",
    "train_baseline_lr_stage",
    "analyze_and_prune_features",

    # Classes
    "LRFeatureEngineer",
    "CalibratedLRStage",
    "LRStage",
    "CorrelationAnalyzer",
]