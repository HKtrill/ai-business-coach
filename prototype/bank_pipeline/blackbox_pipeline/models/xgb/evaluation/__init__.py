"""
blackbox_pipeline.models.xgb.evaluation
========================================
Stage 3 measurement.

``metrics``  the metric block, with ``to_glass_keys()`` for direct tabling
             against the EBM's ``evaluate_ebm`` output
``cv``       the k-fold reporting block mirroring the EBM's 10-fold summary
``compare``  EBM vs XGBoost at the same Stage 3 operating role
"""

from __future__ import annotations

from .compare import ArmScores, Stage3Comparison
from .cv import CVEvaluator, CVReport
from .metrics import Stage3Metrics

__all__ = [
    "Stage3Metrics",
    "CVEvaluator",
    "CVReport",
    "Stage3Comparison",
    "ArmScores",
]