"""
blackbox_pipeline.models.mlp

Stage 1 black-box counterpart to the GLASS ``CalibratedLRStage``.

Provides the calibrated MLP pipeline, including feature definitions,
configuration, tuning, calibration, OOF threshold selection, persistence,
and stage orchestration.

Example
-------
>>> from blackbox_pipeline.models.mlp import (
...     STAGE1_FEATURES, CalibratedStage1MLP,
... )
>>> stage = CalibratedStage1MLP(
...     calibration_method="auto",
...     cv_folds=10,
...     n_trials=100,
...     random_state=42,
... ).fit(STAGE1["X_train"], STAGE1["y_train"])

Notes
-----
Shared evaluation metrics are defined in ``mlp.evaluation`` so the MLP and
GLASS LR are measured with identical code.

Threshold tuning remains part of the model package because it uses
training-only OOF predictions to produce the decision rule.
"""

from .artifacts import load_stage1_mlp, save_stage1_mlp
from .calibration import (
    CalibrationLeakageError,
    assert_refittable,
    fit_stage1_calibration,
    is_prefit_calibrator,
)
from .config import Stage1MLPConfig
from .estimator import Stage1MLPClassifier, oversample_positives
from .features import (
    STAGE1_FEATURES,
    assert_matches_glass,
    check_labels,
    select_features,
)
from .stage import CalibratedStage1MLP
from .thresholds import (
    oof_probabilities,
    optimize_threshold_cv,
    sweep_f_beta,
)
from .tuning import params_to_kwargs, sample_params, tune_stage1_mlp_auc

__all__ = [
    # input contract
    "STAGE1_FEATURES",
    "select_features",
    "check_labels",
    "assert_matches_glass",
    # config + estimator
    "Stage1MLPConfig",
    "Stage1MLPClassifier",
    "oversample_positives",
    # tuning
    "sample_params",
    "params_to_kwargs",
    "tune_stage1_mlp_auc",
    # calibration
    "fit_stage1_calibration",
    "is_prefit_calibrator",
    "assert_refittable",
    "CalibrationLeakageError",
    # thresholds (fitting, not evaluation — see module docstring)
    "oof_probabilities",
    "sweep_f_beta",
    "optimize_threshold_cv",
    "tune_threshold",
    # orchestration + persistence
    "CalibratedStage1MLP",
    "save_stage1_mlp",
    "load_stage1_mlp",
]
