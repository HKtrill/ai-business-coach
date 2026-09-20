"""
blackbox_pipeline.models.mlp.config
====================================
Explicit configuration for the Stage 1 calibrated MLP.

Same discipline as the RF router's config: every knob is stated and validated,
so the notebook stays the single source of truth for an experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

__all__ = ["Stage1MLPConfig"]


@dataclass
class Stage1MLPConfig:
    """
    Parameters
    ----------
    calibration_method
        ``"auto"``, ``"sigmoid"`` or ``"isotonic"``. Passed through to the GLASS
        ``lr.calibration.fit_calibration``; ``"auto"`` lets it pick and the
        winner is written back to ``CalibratedStage1MLP.calibration_method``.
    cv_folds
        Folds for tuning, for calibration, and for the out-of-fold probabilities
        that the decision threshold is chosen from.
    n_trials
        Optuna trials. 100 matches the GLASS LR stage.
    random_state
        Seed for Optuna, every fold split, and every MLP fit.

    max_epochs, patience, val_fraction
        Early stopping inside each MLP fit. ``val_fraction`` of the fold's own
        rows are held out and ROC-AUC on them decides when to stop; the weights
        from the best epoch are restored.

    threshold_beta
        Beta for the F-beta threshold sweep. 2.0 weights recall over precision,
        matching GLASS.
    threshold_grid
        ``(start, stop, step)`` for the in-stage sweep. The GLASS stage sweeps
        0.05–0.49 in 0.01 steps, and that is reproduced here.
    n_jobs
        Parallelism for ``cross_val_predict``.

    strict_oof
        When True (default), the stage REFUSES to compute out-of-fold
        probabilities from a prefit calibrator. See ``calibration.py`` — a
        prefit wrapper survives cloning, so ``cross_val_predict`` would score
        every row with a model that trained on it and the resulting "train OOF"
        numbers would be in-sample. Set False only to reproduce an older run,
        and relabel the metrics if you do.
    """

    calibration_method: str = "auto"
    cv_folds: int = 10
    n_trials: int = 100
    random_state: int = 42

    max_epochs: int = 200
    patience: int = 15
    val_fraction: float = 0.15

    threshold_beta: float = 2.0
    threshold_grid: tuple = (0.05, 0.50, 0.01)

    n_jobs: int = -1
    verbose: bool = True

    strict_oof: bool = True

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        errors: list[str] = []

        if self.calibration_method not in ("auto", "sigmoid", "isotonic"):
            errors.append(
                "calibration_method must be 'auto', 'sigmoid' or 'isotonic', "
                f"got {self.calibration_method!r}"
            )
        if self.cv_folds < 2:
            errors.append("cv_folds must be >= 2")
        if self.n_trials < 1:
            errors.append("n_trials must be >= 1")
        if not 0.0 < self.val_fraction < 0.5:
            errors.append(
                f"val_fraction must be in (0, 0.5), got {self.val_fraction}"
            )
        if self.max_epochs < 1:
            errors.append("max_epochs must be >= 1")
        if self.patience < 1:
            errors.append("patience must be >= 1")
        if self.threshold_beta <= 0:
            errors.append("threshold_beta must be > 0")

        start, stop, step = self.threshold_grid
        if not 0.0 < start < stop <= 1.0 or step <= 0:
            errors.append(
                f"threshold_grid must be (start, stop, step) with "
                f"0 < start < stop <= 1 and step > 0, got {self.threshold_grid}"
            )

        if errors:
            raise ValueError(
                "Stage1MLPConfig validation errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    @property
    def estimator_fixed_kwargs(self) -> dict:
        """Kwargs shared by every Stage1MLPClassifier this config builds."""
        return {
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "val_fraction": self.val_fraction,
            "random_state": self.random_state,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Stage1MLPConfig":
        known = {f.name for f in fields(cls)}
        d = dict(d)
        if "threshold_grid" in d:
            d["threshold_grid"] = tuple(d["threshold_grid"])
        return cls(**{k: v for k, v in d.items() if k in known})

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Stage1MLPConfig(calibration='{self.calibration_method}', "
            f"folds={self.cv_folds}, trials={self.n_trials}, "
            f"seed={self.random_state}, strict_oof={self.strict_oof})"
        )
