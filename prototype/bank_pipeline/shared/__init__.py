"""
shared
======
Code that BOTH arms of the cascade must run identically.

If two arms are compared on a number, the number has to come from one
implementation. Two copies that happen to agree today drift tomorrow, and then
the comparison measures the copies rather than the models. Everything in this
package is therefore imported — never re-implemented — by the GLASS ``lr``
package and by ``blackbox_pipeline.models.mlp``.

Modules
-------
stage_io
    The stage-to-stage output contract. :class:`StageOutput` carries indexed
    out-of-fold train probabilities, test probabilities, fold ids, the
    operating point and how it was chosen. :func:`assert_comparable` checks two
    arms can be compared row for row.
calibration_guard
    :func:`assert_refittable` refuses a calibrator whose clones would keep a
    fitted base model, which would make "out-of-fold" probabilities in-sample.
thresholds
    :func:`optimize_threshold_cv` — the in-stage F-beta sweep both arms use to
    pick their operating point.
metrics
    :func:`compute_metrics`, :func:`binary_metrics`, :func:`metrics_table` and
    :func:`calculate_ece` — the evaluation both arms are reported on.
validation
    :class:`ValidationReport` — the audit ledger both notebooks record their
    validation checks in, so "all checks passed" means the same in each.

Dependency direction
--------------------
``blackbox_pipeline`` imports from here; nothing here imports from
``blackbox_pipeline`` or from ``lr``. GLASS is the library arm. Keep it that
way: a back-edge would create an import cycle and, worse, let one arm's
internals leak into the shared definition of a metric.

Import root
-----------
``bank_pipeline/`` is the import root: notebooks put it on ``sys.path`` (Cell 1)
and import every package under it by its top-level name::

    from shared import StageOutput, ValidationReport
    from glass_pipeline.lr.lr_stage_calibrated import train_calibrated_lr_stage
    from blackbox_pipeline.models.mlp import CalibratedStage1MLP
    from data.preprocessing import BankPreprocessor

Inside a package, use relative imports (``from .stage_io import ...``).

Use only this form. Importing the same module under two names (say ``shared``
and ``bank_pipeline.shared``) creates two distinct copies of every class, and
``isinstance`` checks between them fail. It also means ``bank_pipeline`` and
``churn_pipeline`` must not share one kernel: both define ``data`` and
``shared``.

Examples
--------
>>> from shared import StageOutput, assert_comparable
>>> glass = StageOutput.load("models/lr/stage1_glass_output.joblib")
>>> mlp = StageOutput.load("models/mlp/stage1_blackbox_output.joblib")
>>> assert_comparable(glass, mlp)
"""

from .calibration_guard import (
    CalibrationLeakageError,
    assert_refittable,
    is_prefit_calibrator,
)
from .metrics import (
    binary_metrics,
    calculate_ece,
    compute_metrics,
    metrics_table,
)
from .stage_io import (
    STAGE_OUTPUT_VERSION,
    THRESHOLD_SOURCES,
    StageOutput,
    StageOutputError,
    ThresholdContaminationError,
    assert_comparable,
    fold_assignment,
    split_fingerprint,
)
from .thresholds import (
    DEFAULT_GRID_SPEC,
    optimize_threshold_cv,
    sweep_f_beta,
)
from .validation import ValidationError, ValidationReport

__all__ = [
    # stage_io — the Stage 4 handoff contract
    "STAGE_OUTPUT_VERSION",
    "THRESHOLD_SOURCES",
    "StageOutput",
    "StageOutputError",
    "ThresholdContaminationError",
    "fold_assignment",
    "split_fingerprint",
    "assert_comparable",
    # calibration_guard — out-of-fold leakage check
    "CalibrationLeakageError",
    "is_prefit_calibrator",
    "assert_refittable",
    # thresholds — operating-point selection
    "DEFAULT_GRID_SPEC",
    "sweep_f_beta",
    "optimize_threshold_cv",
    # metrics — shared evaluation
    "calculate_ece",
    "compute_metrics",
    "binary_metrics",
    "metrics_table",
    # validation — shared audit ledger
    "ValidationReport",
    "ValidationError",
]
