"""
blackbox_pipeline.models.mlp.stage1_mlp
========================================
Back-compatibility shim.

The Stage 1 MLP is now a package: ``features``, ``config``, ``estimator``,
``tuning``, ``calibration``, ``thresholds``, ``metrics``, ``stage``,
``artifacts``. This module re-exports the public surface under the old name so
existing notebook cells keep working:

    from blackbox_pipeline.models.mlp.stage1_mlp import STAGE1_FEATURES, CalibratedStage1MLP
    import blackbox_pipeline.models.mlp.stage1_mlp as stage1_mlp
    stage1_mlp.tune_threshold(...)
    stage1_mlp.binary_metrics(...)

New code should import from ``blackbox_pipeline.models.mlp`` directly.

Note on ``importlib.reload``: reloading THIS module re-runs only these imports,
not the modules behind them. To pick up an edit to, say, ``thresholds.py``,
reload that module — or just restart the kernel.
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
from .evaluation.metrics import binary_metrics, metrics_table
from .stage import CalibratedStage1MLP
from .thresholds import (
    oof_probabilities,
    optimize_threshold_cv,
    sweep_f_beta,
    tune_threshold,
)
from .tuning import params_to_kwargs, sample_params, tune_stage1_mlp_auc

__all__ = [
    "STAGE1_FEATURES",
    "select_features",
    "assert_matches_glass",
    "Stage1MLPConfig",
    "Stage1MLPClassifier",
    "oversample_positives",
    "sample_params",
    "params_to_kwargs",
    "tune_stage1_mlp_auc",
    "fit_stage1_calibration",
    "is_prefit_calibrator",
    "assert_refittable",
    "CalibrationLeakageError",
    "oof_probabilities",
    "sweep_f_beta",
    "optimize_threshold_cv",
    "tune_threshold",
    "binary_metrics",
    "metrics_table",
    "CalibratedStage1MLP",
    "save_stage1_mlp",
    "load_stage1_mlp",
]
