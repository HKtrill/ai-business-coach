"""
blackbox_pipeline.models.xgb
=============================
Stage 3 black-box counterpart — XGBoost in place of the GLASS EBM.

The question
------------
What happens when the EBM's additive / interpretable constraint is removed and
nothing else changes? So everything that could change is pinned to the EBM's
choice: the same Stage 3 population, the same engineered feature block, the
same target, the same split, the same stratified folds and seed, the same
balanced sample-weight vector, the same F2-at-0.5 tuning objective, the same
150-trial Optuna budget with the same sampler and pruner, the same 10-fold
reporting block, the same isotonic-gated-at-0.05 calibration family, and the
same F2 threshold grid.

``config.XGBStage3Config`` is where those values live, and
``config.XGBStage3Config.mirror_report()`` is the machine-readable statement of
what was matched and what could not be.

Documented departures
---------------------
1. The operating threshold is selected on out-of-fold TRAINING predictions.
   GLASS fits it on ``y_test``; PR 3 forbids that. Both values are recorded —
   the OOF one operates, the test-fitted one is kept as a labelled oracle.
2. The calibration gate reads out-of-fold ECE rather than in-sample ECE.
3. Out-of-fold Stage 3 predictions are produced for the training split, which
   GLASS does not currently do. Stage 4 trains on those; the in-sample column
   the EBM artifact stores is preserved separately for parity.
4. Median imputation is learned on train and reused on test.
5. The hyperparameter space is XGBoost-native — only ``learning_rate`` could
   be mirrored bound for bound.

Layout
------
``config``      every mirrored constant, as data
``features``    the EBM feature contract and data hygiene (no engineering)
``folds``       the one canonical stratified partition
``weights``     balanced sample weights, computed once
``estimator``   ``XGBClassifier`` construction
``tuning``      Optuna search
``oof``         leakage-safe out-of-fold predictions — Stage 4's input
``calibration`` OOF-gated, nested isotonic
``thresholds``  F2 grid on OOF predictions
``evaluation``  metric block, k-fold report, EBM-vs-XGBoost comparison
``artifacts``   the payload, and the OOF / refit separation
``stage``       the orchestrator

Usage
-----
    from blackbox_pipeline.models.xgb import XGBStage3Config, train_xgb_stage

    cfg = XGBStage3Config(expected_features=list(EBM_FEATURES))
    artifact, path = train_xgb_stage(
        GLOBAL_SPLIT, engineered=STAGE3_DATA, config=cfg
    )
    stage4_inputs = artifact.stage4_frame()      # OOF — safe to train on
"""

from __future__ import annotations

from .artifacts import Stage3Artifact, Stage3ArtifactStore
from .calibration import CalibrationReport, Stage3Calibrator, calculate_ece
from .config import DEFAULT_XGB_PARAMS, XGBSearchSpace, XGBStage3Config
from .estimator import XGBFactory
from .evaluation import (
    ArmScores,
    CVEvaluator,
    CVReport,
    Stage3Comparison,
    Stage3Metrics,
)
from .features import FeatureCleaner, Stage3FeatureContract
from .folds import FoldPlan
from .oof import OOFGenerator, OOFPredictions
from .stage import XGBStage3Pipeline, train_xgb_stage
from .thresholds import F2ThresholdSelector, ThresholdChoice
from .tuning import TuningResult, XGBTuner
from .weights import BalancedWeights

__all__ = [
    # entry points
    "train_xgb_stage",
    "XGBStage3Pipeline",
    # configuration
    "XGBStage3Config",
    "XGBSearchSpace",
    "DEFAULT_XGB_PARAMS",
    # components
    "Stage3FeatureContract",
    "FeatureCleaner",
    "FoldPlan",
    "BalancedWeights",
    "XGBFactory",
    "XGBTuner",
    "TuningResult",
    "OOFGenerator",
    "OOFPredictions",
    "Stage3Calibrator",
    "CalibrationReport",
    "calculate_ece",
    "F2ThresholdSelector",
    "ThresholdChoice",
    # evaluation
    "Stage3Metrics",
    "CVEvaluator",
    "CVReport",
    "Stage3Comparison",
    "ArmScores",
    # artifacts
    "Stage3Artifact",
    "Stage3ArtifactStore",
]