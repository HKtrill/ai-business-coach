"""
blackbox_pipeline.models.xgb
=============================
Stage 3 black-box counterpart — XGBoost in place of the GLASS EBM.

The question
------------
What happens when the EBM's additive / interpretable constraint is removed and
nothing else changes? Both arms run the shared Stage 3 protocol
(``shared.stage3``): the same per-fold feature block, split, stratified folds
and seed, balanced sample weights, F2-at-0.5 tuning objective, Optuna budget,
sampler and pruner, 10-fold reporting block, OOF-gated isotonic calibration
and F2 threshold grid on out-of-fold training predictions. Only the estimator
and its search space are XGBoost's own (``shared_protocol``), so
``shared.stage3.assert_matched_protocol(EBM_STAGE3, XGB_STAGE3)`` compares like
with like.

Public interface (all from ``shared_protocol``)
-----------------------------------------------
``XGBStage3Config``  ``Stage3Config`` with the black-box XGBoost identity
``XGBSearchSpace``   the XGBoost-native Optuna space
``XGBEstimator``     builds and fits ``XGBClassifier`` for the shared runner
``train_xgb_stage``  ``Stage3Runner`` with that estimator → (artifact, path)

The artifact is a ``shared.stage3.Stage3Artifact``. The standalone
implementation that predated the shared protocol (its own config, folds,
weights, tuning, OOF, calibration, thresholds, evaluation, artifact and stage
modules) was removed in PR 36 Commit 4; nothing current imports or loads it.

Usage
-----
    from blackbox_pipeline.models.xgb import XGBStage3Config, train_xgb_stage
    from shared.stage3 import Stage3Block
    from shared.stage3.feature_engineering import EBM_FEATURES, EBMFeaturePipeline

    cfg = XGBStage3Config(expected_features=list(EBM_FEATURES))
    block = Stage3Block.from_global_split(GLOBAL_SPLIT, EBMFeaturePipeline, cfg,
                                          split_fingerprint=split_fp)
    artifact, path = train_xgb_stage(block=block, config=cfg)
    stage4_inputs = artifact.stage4_frame()      # OOF — safe to train on
"""

from __future__ import annotations

from .shared_protocol import XGBEstimator, XGBSearchSpace, XGBStage3Config, train_xgb_stage

__all__ = ["train_xgb_stage", "XGBStage3Config", "XGBSearchSpace", "XGBEstimator"]
