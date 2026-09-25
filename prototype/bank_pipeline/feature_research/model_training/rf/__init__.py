# feature_research/model_training/rf
# ------------------------------------
# RF training, lift analysis, and binary binning for the Glass Cascade.
#
# Usage:
#   from feature_research.model_training.rf import (
#       train_rf, RFResult, rf_diagnostics,
#       rf_lift_analysis, create_binary_features,
#       validate_binary_features, RF_FEATURES_BINARY,
#   )

from feature_research.model_training.rf.trainer import (      # ← new
    train_rf,
    RFResult,
)
from feature_research.model_training.rf.diagnostics import (  # ← new
    rf_diagnostics,
)
from feature_research.model_training.rf.lift_analysis import (
    rf_lift_analysis,
    compute_lift,
    compute_gini_importance,
    compute_permutation_importance,
)
from feature_research.model_training.rf.binning import (
    create_binary_features,
    validate_binary_features,
    RF_FEATURES_BINARY,
    BINNING_STRATEGY,
)

__all__ = [
    # training
    "train_rf",
    "RFResult",
    # diagnostics
    "rf_diagnostics",
    # lift analysis
    "rf_lift_analysis",
    "compute_lift",
    "compute_gini_importance",
    "compute_permutation_importance",
    # binning
    "create_binary_features",
    "validate_binary_features",
    "RF_FEATURES_BINARY",
    "BINNING_STRATEGY",
]