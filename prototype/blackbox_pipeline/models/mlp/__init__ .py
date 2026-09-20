"""
blackbox_pipeline.models.mlp
=============================
Stage 1 — calibrated MLP.

Black-box counterpart to the GLASS ``CalibratedLRStage``, split into one module
per responsibility:

    features.py     the three-feature contract, shared with GLASS LR_FEATURES
    config.py       Stage1MLPConfig — explicit parameters and validation
    estimator.py    Stage1MLPClassifier — oversampling + early stopping
    tuning.py       Optuna ROC-AUC search (training only)
    calibration.py  GLASS fit_calibration wrapper + the prefit leakage guard
    thresholds.py   out-of-fold probabilities and the F-beta sweeps
    stage.py        CalibratedStage1MLP — orchestration
    artifacts.py    persistence
    stage1_mlp.py   back-compat shim; existing notebook cells import from here

Metrics live outside this package
---------------------------------
``binary_metrics`` and ``metrics_table`` moved to
``blackbox_pipeline.evaluation.classification_metrics``. They are SHARED
measurement: the Stage 1 MLP and the GLASS Stage 1 LR are compared on those
numbers, so both arms must compute them with the same code. A metrics helper
that lives inside one arm's package invites a second copy in the other, and
then the comparison starts arguing about denominators instead of models. Same
reasoning puts ``router_metrics`` there for Stage 2.

``tune_threshold`` deliberately stays here: it reads training out-of-fold
probabilities to PRODUCE a decision rule, which is fitting, not evaluation.

    from blackbox_pipeline.evaluation.classification_metrics import (
        binary_metrics, metrics_table,
    )

Typical use
-----------
    from blackbox_pipeline.models.mlp import (
        STAGE1_FEATURES, Stage1MLPConfig, CalibratedStage1MLP, save_stage1_mlp,
    )

    stage = CalibratedStage1MLP(
        calibration_method="auto", cv_folds=10, n_trials=100, random_state=42,
    ).fit(STAGE1["X_train"], STAGE1["y_train"])

    proba_test = stage.predict_proba(STAGE1["X_test"])
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
from .features import STAGE1_FEATURES, assert_matches_glass, select_features
from .stage import CalibratedStage1MLP
from .thresholds import (
    oof_probabilities,
    optimize_threshold_cv,
    sweep_f_beta,
    tune_threshold,
)
from .tuning import params_to_kwargs, sample_params, tune_stage1_mlp_auc

__all__ = [
    # features
    "STAGE1_FEATURES",
    "select_features",
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