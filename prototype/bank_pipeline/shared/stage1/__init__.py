"""
shared.stage1
=============

Shared Stage 1 infrastructure for the GLASS Logistic Regression and
black-box MLP arms.

Contains the feature-engineering contract and calibration safety checks that
both Stage 1 implementations must use identically.
"""

from .calibration_guard import (
    CalibrationLeakageError,
    assert_refittable,
    is_prefit_calibrator,
)
from .feature_engineering import (
    LR_FEATURES,
    LRFeatureEngineer,
    engineer_features,
)

__all__ = [
    "CalibrationLeakageError",
    "LR_FEATURES",
    "LRFeatureEngineer",
    "assert_refittable",
    "engineer_features",
    "is_prefit_calibrator",
]