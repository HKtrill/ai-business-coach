# feature_research/model_training/rf
# ------------------------------------
# RF training, diagnostics, lift analysis and binary binning for the Glass Cascade.
#
# Usage:
#   from feature_research.model_training.rf import (
#       train_rf, RFResult, rf_diagnostics,
#       rf_lift_analysis, create_binary_features, validate_binary_features,
#       bin_features, BinaryFeaturePipeline, RF_FEATURES_BINARY,
#   )

from feature_research.model_training.rf.trainer import (
    RFResult,
    evaluate_rf,
    train_rf,
    tune_rf,
)
from feature_research.model_training.rf.diagnostics import (
    rf_diagnostics,
)
from feature_research.model_training.rf.lift_analysis import (
    compute_lift,
    rf_lift_analysis,
)
from feature_research.model_training.rf.binning import (
    BINNING_STRATEGY,
    RF_FEATURES_BINARY,
    BinaryFeaturePipeline,
    bin_features,
    bin_lift_table,
    create_binary_features,
    validate_binary_features,
)

__all__ = [
    # training
    "train_rf",
    "tune_rf",
    "evaluate_rf",
    "RFResult",
    # diagnostics
    "rf_diagnostics",
    # lift analysis
    "rf_lift_analysis",
    "compute_lift",
    # binning
    "bin_features",
    "bin_lift_table",
    "BinaryFeaturePipeline",
    "create_binary_features",
    "validate_binary_features",
    "RF_FEATURES_BINARY",
    "BINNING_STRATEGY",
]