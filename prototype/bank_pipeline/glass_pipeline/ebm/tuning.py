"""
glass_pipeline.ebm.tuning
=========================
Since PR 33 the EBM Optuna search runs through ``shared.stage_runner`` — the
same loop the XGBoost arm uses (TPE seed 42, MedianPruner 20/2, 150 trials,
F2 at the 0.5 cut, balanced weights), on per-fold engineered features.

What stays EBM-specific lives in ``glass_pipeline.ebm.estimator``:
``EBMSearchSpace`` (the four searched parameters and their bounds) and
``EBMFactory`` (model construction with the validated interaction pairs).

``tune_ebm`` is removed: it cross-validated on features fitted to all of
train, kept no OOF predictions, and returned an in-sample refit. Use
``train_ebm_stage(block=..., config=...)``.
"""

from shared.stage_runner import Stage3Tuner, TuningResult  # noqa: F401

from .estimator import EBMFactory, EBMSearchSpace  # noqa: F401


def tune_ebm(*args, **kwargs):
    raise NotImplementedError(
        "tune_ebm was removed in PR 33 (feature-leaky CV, no OOF output). "
        "Use glass_pipeline.ebm.ebm_stage.train_ebm_stage(block=..., config=...)."
    )


__all__ = ["Stage3Tuner", "TuningResult", "EBMFactory", "EBMSearchSpace", "tune_ebm"]
