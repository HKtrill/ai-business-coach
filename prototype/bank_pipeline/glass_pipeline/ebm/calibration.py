"""
glass_pipeline.ebm.calibration
==============================
Since PR 33 Stage 3 calibration is ``shared.stage_runner.Stage3Calibrator``,
shared with the XGBoost arm: gated on OUT-OF-FOLD ECE (> 0.05), fitted as a
fold-nested isotonic map on OOF probabilities, applied to test by a map fitted
on all OOF rows. The model itself is never refitted, so its balanced sample
weights are preserved.

``calculate_ece`` is re-exported; it is ``shared.metrics.calculate_ece`` (exact
zeros counted), the same ECE every stage reports.

``calibrate_if_needed`` is removed. It gated on in-sample train ECE and wrapped
the EBM in ``CalibratedClassifierCV(cv=3)``, which refitted three UNWEIGHTED
EBMs — so its calibrated scores came from a different model than the one
reported, and its calibrated train column was partly in-sample.
"""

from shared.stage_runner import (  # noqa: F401
    CalibrationReport,
    Stage3Calibrator,
    calculate_ece,
)


def calibrate_if_needed(*args, **kwargs):
    raise NotImplementedError(
        "calibrate_if_needed was removed in PR 33 (in-sample gate, unweighted "
        "CalibratedClassifierCV refit). Calibration now runs inside "
        "train_ebm_stage via shared.stage_runner.Stage3Calibrator."
    )


__all__ = ["calculate_ece", "Stage3Calibrator", "CalibrationReport",
           "calibrate_if_needed"]
