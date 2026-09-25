"""
shared.stage3
=============
Stage-3-specific shared infrastructure for the GLASS EBM and black-box
XGBoost arms. Each arm supplies only its estimator, search space and
``Stage3Config`` subclass; the protocol below is common to both, so the arms
are comparable by construction. Not a generic implementation for other
cascade stages.

Modules, in dependency order (no module imports one later in the list):

    config        Stage3Config, PROTOCOL_FIELDS
    data          FoldPlan, BalancedWeights, feature contract / cleaner, Stage3Block
    thresholding  F2ThresholdSelector, ThresholdChoice, decision_block
    calibration   Stage3Calibrator, CalibrationReport, calculate_ece
    estimator     Stage3Estimator protocol, CalibratedModel, Stage3ServingModel
    tuning        Stage3Tuner, TuningResult (Optuna lives here)
    evaluation    Stage3Metrics, CV report, OOF generation, metric suite
    artifact      Stage3Artifact, Stage3ArtifactStore, find_stage3_artifact
    protocol      protocol_diff, assert_matched_protocol, check_param_reuse
    comparison    ArmScores, Stage3Comparison
    runner        Stage3Runner (orchestration only)

Uses the existing shared modules rather than re-implementing them:
``shared.stage_io.fold_assignment`` / ``StageOutput``,
``shared.thresholds.sweep_f_beta``, ``shared.metrics.compute_metrics`` /
``calculate_ece``. Nothing here imports from glass_pipeline or
blackbox_pipeline.

Not loaded by ``import shared`` (it needs optuna); import it explicitly::

    from shared.stage3 import Stage3Block, Stage3Runner
    from shared.stage_runner import Stage3Block, Stage3Runner   # legacy path
"""

from .artifact import (
    STAGE4_FEATURE_SUFFIXES,
    TEST_SOURCES,
    Stage3Artifact,
    Stage3ArtifactStore,
    find_stage3_artifact,
)
from .calibration import (
    CalibrationReport,
    Stage3Calibrator,
    calculate_ece,
    calibration_diagnostics,
)
from .comparison import ArmScores, Stage3Comparison
from .config import FEATURE_FIT_SCOPES, PROTOCOL_FIELDS, Stage3Config
from .data import (
    BalancedWeights,
    FeatureCleaner,
    FoldFrames,
    FoldPlan,
    PassthroughPipeline,
    Stage3Block,
    Stage3FeatureContract,
)
from .estimator import (
    CalibratedModel,
    Stage3Estimator,
    Stage3ServingModel,
    positive_proba,
)
from .evaluation import (
    CVEvaluator,
    CVReport,
    OOFGenerator,
    OOFPredictions,
    Stage3Metrics,
    stage3_metric_suite,
)
from .protocol import (
    FAMILY_SPECIFIC,
    assert_matched_protocol,
    check_param_reuse,
    effective_tuning,
    protocol_diff,
)
from .runner import Stage3Runner
from .thresholding import F2ThresholdSelector, ThresholdChoice, decision_block
from .tuning import Stage3Tuner, TuningResult

__all__ = [
    # config
    "FEATURE_FIT_SCOPES", "PROTOCOL_FIELDS", "Stage3Config",
    # data
    "FoldPlan", "BalancedWeights", "Stage3FeatureContract", "FeatureCleaner",
    "PassthroughPipeline", "FoldFrames", "Stage3Block",
    # estimator
    "Stage3Estimator", "positive_proba", "CalibratedModel", "Stage3ServingModel",
    # tuning
    "TuningResult", "Stage3Tuner",
    # calibration
    "calculate_ece", "CalibrationReport", "Stage3Calibrator",
    "calibration_diagnostics",
    # thresholding
    "ThresholdChoice", "F2ThresholdSelector", "decision_block",
    # evaluation
    "Stage3Metrics", "CVReport", "CVEvaluator", "OOFPredictions", "OOFGenerator",
    "stage3_metric_suite",
    # artifact
    "STAGE4_FEATURE_SUFFIXES", "TEST_SOURCES", "Stage3Artifact",
    "Stage3ArtifactStore", "find_stage3_artifact",
    # protocol
    "FAMILY_SPECIFIC", "effective_tuning", "protocol_diff",
    "assert_matched_protocol", "check_param_reuse",
    # comparison
    "ArmScores", "Stage3Comparison",
    # runner
    "Stage3Runner",
]
