"""
glass_router.rf
===============
RF sub-package for Stage 2 (GLASS Router feature engineering + RF router).

Public API (notebook-facing)
----------------------------
    from glass_router.rf import analyze_rf_lift       # Cell 9
    from glass_router.rf import engineer_features      # Cell 10
    from glass_router.rf import train_rf_baseline      # Cell 11
    from glass_router.rf import train_rf_stage         # Cell 12
    from glass_router.rf import plot_rf_evaluation     # Cell 12B
    from glass_router.rf import save_rf_artifact       # Cell 12C

Internal modules (not for direct notebook import)
-------------------------------------------------
    glass_router.rf.binning             BINNING_STRATEGY, create_binary_features,
                                        validate_binary_features, RF_FEATURES_BINARY
    glass_router.rf.feature_engineering RFFeatureEngineer (internal DAG)
"""

from .lift_analysis import analyze_rf_lift, RFLiftAnalyzer
from .feature_engineering import engineer_features, RFFeatureEngineer
from .baseline import train_rf_baseline
from .rf_training import train_rf_stage, RFResult
from .evaluation import plot_rf_evaluation
from .artifacts import save_rf_artifact

__all__ = [
    # Notebook entry points
    "analyze_rf_lift",
    "engineer_features",
    "train_rf_baseline",
    "train_rf_stage",
    "plot_rf_evaluation",
    "save_rf_artifact",

    # Classes
    "RFLiftAnalyzer",
    "RFFeatureEngineer",
    "RFResult",
]