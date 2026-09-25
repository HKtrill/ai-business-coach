"""
blackbox_pipeline.models.mlp.stage1_mlp

Back-compatibility shim.

Stage 1 is now a package — ``features``, ``config``, ``estimator``, ``tuning``,
``calibration``, ``thresholds``, ``evaluation``, ``stage``, ``artifacts``. This
module re-exports the public surface under the old name so existing notebook
cells keep working::

    from blackbox_pipeline.models.mlp.stage1_mlp import STAGE1_FEATURES, CalibratedStage1MLP
    import blackbox_pipeline.models.mlp.stage1_mlp as stage1_mlp
    stage1_mlp.binary_metrics(...)

New code should import from ``blackbox_pipeline.models.mlp`` directly, and take
the metrics from ``blackbox_pipeline.models.mlp.evaluation``.

Notes
-----
``importlib.reload`` on THIS module re-runs only these imports, not the modules
behind them. To pick up an edit to, say, ``thresholds.py``, reload that module —
or just restart the kernel.
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
from .evaluation.metrics import binary_metrics, metrics_table
from .stage import CalibratedStage1MLP
from .thresholds import (
    oof_probabilities,
    optimize_threshold_cv,
    sweep_f_beta,
)
from .tuning import params_to_kwargs, sample_params, tune_stage1_mlp_auc

__all__ = [
    "STAGE1_FEATURES",
    "select_features",
    "check_labels",
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
    "binary_metrics",
    "metrics_table",
    "CalibratedStage1MLP",
    "save_stage1_mlp",
    "load_stage1_mlp",
]
