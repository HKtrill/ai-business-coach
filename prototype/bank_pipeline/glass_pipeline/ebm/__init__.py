"""
glass_pipeline.ebm
==================
Stage 3 of the GLASS cascade — Explainable Boosting Machine.

Runs on ``shared.stage_runner`` (shared with the black-box XGBoost arm).
EBM-specific pieces:

``feature_engineering``  the validated research DAG + ``EBMFeaturePipeline``
``interactions``         the fixed pairwise interaction terms
``estimator``            ``EBMFactory`` + ``EBMSearchSpace``
``config``               ``EBMStage3Config``
``ebm_stage``            ``train_ebm_stage``
"""

from .config import EBMStage3Config
from .ebm_stage import make_ebm_factory, train_ebm_stage
from .estimator import EBMFactory, EBMSearchSpace
from .feature_engineering import EBM_FEATURES, EBMFeaturePipeline

__all__ = [
    "EBMStage3Config", "train_ebm_stage", "make_ebm_factory",
    "EBMFactory", "EBMSearchSpace", "EBM_FEATURES", "EBMFeaturePipeline",
]