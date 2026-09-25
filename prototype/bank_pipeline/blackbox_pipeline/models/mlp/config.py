"""
blackbox_pipeline.models.mlp.config

Configuration for the Stage 1 calibrated MLP.

Centralizes and validates tuning, calibration, early-stopping, threshold,
parallelism, and OOF-safety settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

__all__ = ["Stage1MLPConfig"]


@dataclass
class Stage1MLPConfig:
    """
    Stage 1 MLP configuration.

    Parameters
    ----------
    calibration_method
        Calibration method: ``"auto"``, ``"sigmoid"``, or ``"isotonic"``.
    cv_folds
        Cross-validation folds used throughout Stage 1.
    n_trials
        Number of Optuna trials.
    random_state
        Shared random seed.
    max_epochs, patience, val_fraction
        Early-stopping settings.
    threshold_beta
        Beta used for F-beta threshold selection.
    threshold_grid
        ``(start, stop, step)`` for threshold search.
    n_jobs
        Parallel jobs for OOF prediction.
    strict_oof
        Reject prefit calibrators that would invalidate OOF predictions.
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
        """Return estimator kwargs shared across Stage 1 MLP fits."""
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

