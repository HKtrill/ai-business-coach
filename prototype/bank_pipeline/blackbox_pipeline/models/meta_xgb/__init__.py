"""
blackbox_pipeline.models.meta_xgb
=================================
Stage 4 of the black-box cascade — Meta-XGB, the learned counterpart of the
GLASS Meta-EBM weighted-confidence arbiter.

    MLP (Stage 1) ↔ GLASS LR · RF Router (Stage 2) ↔ GLASS Router ·
    XGBoost (Stage 3) ↔ GLASS EBM · Meta-XGB (Stage 4) ↔ GLASS Meta-EBM

    from blackbox_pipeline.models.meta_xgb import train_meta_xgb_stage
    meta_xgb_artifact, meta_xgb_path = train_meta_xgb_stage(
        GLOBAL_SPLIT, mlp_path=..., rf_path=..., xgb_path=..., meta_ebm_path=...)

Inputs are read and verified by ``shared.stage4`` — the same code the GLASS
Meta-EBM uses. The learner is tuned / cross-fitted / calibrated / thresholded
by the shared Stage 3 runner protocol.
"""

from .artifact import SCHEMA, MetaXGBArtifact, load_meta_xgb, save_meta_xgb
from .comparison import compare_with_meta_ebm, find_meta_ebm_artifact
from .config import (
    EXCLUDED_CHANNELS, FEATURE_CONTRACT, STAGE4_FEATURES, MetaXGBConfig, MetaXGBSearchSpace,
)
from .estimator import MetaXGBEstimator
from .inputs import build_feature_frames, load_blackbox_stage4_inputs
from .runner import MetaXGBRunner, train_meta_xgb_stage

__all__ = [
    "train_meta_xgb_stage", "MetaXGBRunner", "MetaXGBConfig", "MetaXGBSearchSpace",
    "MetaXGBEstimator", "MetaXGBArtifact", "SCHEMA", "save_meta_xgb", "load_meta_xgb",
    "load_blackbox_stage4_inputs", "build_feature_frames", "STAGE4_FEATURES",
    "FEATURE_CONTRACT", "EXCLUDED_CHANNELS", "compare_with_meta_ebm",
    "find_meta_ebm_artifact",
]
