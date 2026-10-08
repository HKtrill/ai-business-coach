"""
shared.stage2
=============

Shared Stage 2 feature infrastructure for the GLASS Router and
black-box RF Router.

Contains the common source-feature engineering and binary representation used
by both Stage 2 implementations. Router-specific training and selection logic
remains in the respective GLASS and black-box packages.
"""

from .feature_engineering import (
    RF_SOURCE_FEATURES,
    RFFeatureEngineer,
    engineer_features,
)
from .binning import (
    RF_FEATURES_BINARY,
)

__all__ = [
    "RF_FEATURES_BINARY",
    "RF_SOURCE_FEATURES",
    "RFFeatureEngineer",
    "engineer_features",
]