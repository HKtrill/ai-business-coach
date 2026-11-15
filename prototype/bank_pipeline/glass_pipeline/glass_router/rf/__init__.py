"""
glass_pipeline.glass_router.rf
==============================
The GLASS arm's Stage 2 random-forest tooling: lift analysis, the RF
baseline and RF stage of the GLASS notebook, and the RF feature ranker the
GLASS Router's rule selection uses. The GLASS Router itself is rule-based
(``glass_pipeline.glass_router``); the black-box RF Router is a separate
model (``blackbox_pipeline.models.rf_router``).

Public API (notebook-facing)
----------------------------
    from glass_pipeline.glass_router.rf import analyze_rf_lift       # Cell 9
    from glass_pipeline.glass_router.rf import engineer_features      # Cell 10
    from glass_pipeline.glass_router.rf import train_rf_baseline      # Cell 11
    from glass_pipeline.glass_router.rf import train_rf_stage         # Cell 12
    from glass_pipeline.glass_router.rf import plot_rf_evaluation     # Cell 12B
    from glass_pipeline.glass_router.rf import save_rf_artifact       # Cell 12C

The Stage 2 features and bins both routers read live in ``shared.stage2``:
    shared.stage2.binning              BINNING_STRATEGY, create_binary_features,
                                       validate_binary_features, RF_FEATURES_BINARY
    shared.stage2.feature_engineering  RFFeatureEngineer (internal DAG)
"""

from .lift_analysis import analyze_rf_lift, RFLiftAnalyzer
from shared.stage2.feature_engineering import engineer_features, RFFeatureEngineer
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