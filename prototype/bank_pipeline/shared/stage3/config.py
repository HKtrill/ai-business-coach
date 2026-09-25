"""
shared.stage3.config
====================
The Stage 3 protocol definition: ``Stage3Config`` plus the field list
(``PROTOCOL_FIELDS``) that must match across the GLASS EBM and black-box
XGBoost arms. Each arm subclasses ``Stage3Config`` to set its family-specific
identity and search space; everything else is shared.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar, Optional


FEATURE_FIT_SCOPES = ("per_fold", "global")


PROTOCOL_FIELDS: tuple[str, ...] = (
    "stage",
    "population",
    "random_state",
    "n_jobs",
    "n_tune_folds",
    "n_eval_folds",
    "stratify",
    "feature_fit_scope",
    "n_trials",
    "pruner_startup_trials",
    "pruner_warmup_steps",
    "beta",
    "tuning_decision_threshold",
    "threshold_low",
    "threshold_high",
    "threshold_steps",
    "reference_threshold",
    "nested_oof_threshold",
    "calibration_method",
    "ece_threshold",
    "ece_bins",
    "force_calibration",
    "class_weight",
    "clean_inf",
    "impute_missing",
    "expected_features",
)


@dataclass
class Stage3Config:
    """
    Shared Stage 3 protocol. Subclasses set the family-specific fields.

    Feature-fit scope
    -----------------
    ``feature_fit_scope="per_fold"`` (default) refits the target-dependent
    feature engineers inside every fold, so a validation row's engineered
    features never depend on its own label. This makes the OOF predictions
    Stage 4 trains on leakage-safe at the FEATURE level as well as the model
    level. ``"global"`` reproduces the pre-PR-33 behaviour (engineers fitted
    once on all of train) and exists only for before/after comparison.

    Population
    ----------
    ``population="full_train"``: Stage 3 trains on every training row. The rows
    Stage 3 actually SERVES in the cascade (those Stage 2 defers) are a subset;
    they are evaluated with ``Stage3Artifact.evaluate_served``.
    """

    # ---- experiment identity (family-specific; overridden) ---------------
    stage: str = "stage3"
    model_family: str = "unset"
    arm: str = "unset"                   # StageOutput arm: "glass" / "blackbox"
    counterpart: str = ""
    column_prefix: str = "stage3_"
    population: str = "full_train"

    # ---- reproducibility --------------------------------------------------
    random_state: int = 42
    n_jobs: int = 1

    # ---- cross-validation -------------------------------------------------
    # One 5-fold partition drives tuning, OOF, calibration and threshold
    # selection; a separate 10-fold partition drives the reporting block.
    n_tune_folds: int = 5
    n_eval_folds: int = 10
    stratify: bool = True
    feature_fit_scope: str = "per_fold"

    # ---- tuning -----------------------------------------------------------
    n_trials: int = 150
    pruner_startup_trials: int = 20
    pruner_warmup_steps: int = 2
    study_name: str = "stage3_recall_biased"
    search_space: Any = None            # family-specific dataclass

    # ---- objective --------------------------------------------------------
    # F2 on the 0.5 cut (what EBM ``model.predict`` scores).
    beta: float = 2.0
    tuning_decision_threshold: float = 0.5

    # ---- threshold selection ---------------------------------------------
    # F2 grid, scored on OUT-OF-FOLD training predictions — never on test.
    threshold_low: float = 0.10
    threshold_high: float = 0.90
    threshold_steps: int = 81
    reference_threshold: float = 0.5
    nested_oof_threshold: bool = True

    # ---- calibration ------------------------------------------------------
    # Isotonic, gated at OOF ECE > 0.05, fold-nested on the training side.
    calibration_method: str = "isotonic"
    ece_threshold: float = 0.05
    ece_bins: int = 10
    force_calibration: bool = False

    # ---- class imbalance --------------------------------------------------
    class_weight: str = "balanced"

    # ---- data hygiene -----------------------------------------------------
    clean_inf: bool = True
    impute_missing: bool = True

    # ---- artifacts --------------------------------------------------------
    artifact_dir: str = "models/stage3"
    artifact_stem: str = "stage3"
    timestamped: bool = True
    write_latest_pointer: bool = True

    # ---- reporting --------------------------------------------------------
    verbose: bool = True
    show_progress_bar: bool = True

    # ---- feature contract -------------------------------------------------
    expected_features: Optional[list[str]] = None

    # Subclasses set this so from_dict can rebuild the search space.
    _space_cls: ClassVar[Optional[type]] = None

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        errors: list[str] = []
        if self.n_tune_folds < 2:
            errors.append("n_tune_folds must be >= 2")
        if self.n_eval_folds < 2:
            errors.append("n_eval_folds must be >= 2")
        if self.n_trials < 1:
            errors.append("n_trials must be >= 1")
        if not 0.0 < self.threshold_low < self.threshold_high < 1.0:
            errors.append(
                "require 0 < threshold_low < threshold_high < 1, got "
                f"{self.threshold_low} / {self.threshold_high}"
            )
        if self.threshold_steps < 2:
            errors.append("threshold_steps must be >= 2")
        if self.beta <= 0:
            errors.append("beta must be > 0")
        if not 0.0 <= self.ece_threshold <= 1.0:
            errors.append("ece_threshold must be in [0, 1]")
        if self.ece_bins < 2:
            errors.append("ece_bins must be >= 2")
        if self.calibration_method not in ("isotonic", "sigmoid"):
            errors.append(
                f"calibration_method must be 'isotonic' or 'sigmoid', "
                f"got {self.calibration_method!r}"
            )
        if self.feature_fit_scope not in FEATURE_FIT_SCOPES:
            errors.append(
                f"feature_fit_scope must be one of {FEATURE_FIT_SCOPES}, "
                f"got {self.feature_fit_scope!r}"
            )
        if self.population != "full_train":
            errors.append(
                "population: only 'full_train' is implemented; the served "
                "subset is evaluated via Stage3Artifact.evaluate_served"
            )
        if errors:
            raise ValueError(
                f"{type(self).__name__} validation errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        space = out.get("search_space")
        out["search_space"] = space.to_dict() if hasattr(space, "to_dict") else space
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Stage3Config":
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in d.items() if k in known}
        space = payload.get("search_space")
        if isinstance(space, dict) and cls._space_cls is not None:
            payload["search_space"] = cls._space_cls(**space)
        return cls(**payload)

    def protocol(self) -> dict[str, Any]:
        """The fields that must match across arms, plus the search space."""
        out = {k: getattr(self, k) for k in PROTOCOL_FIELDS}
        out["search_space"] = (
            self.search_space.to_dict()
            if hasattr(self.search_space, "to_dict") else self.search_space
        )
        return out

    def mirror_report(self) -> dict[str, Any]:
        """Human-readable protocol statement embedded in every artifact."""
        return {
            "model_family": self.model_family,
            "counterpart": self.counterpart,
            "population": self.population,
            "random_state": self.random_state,
            "n_tune_folds": self.n_tune_folds,
            "n_eval_folds": self.n_eval_folds,
            "feature_fit_scope": self.feature_fit_scope,
            "n_trials": self.n_trials,
            "objective": f"F{self.beta:g} at decision threshold "
                         f"{self.tuning_decision_threshold}",
            "class_weight": self.class_weight,
            "calibration": f"{self.calibration_method}, gated at OOF ECE > "
                           f"{self.ece_threshold}, fold-nested on train",
            "threshold_rule": (
                f"F{self.beta:g}-maximising over "
                f"[{self.threshold_low}, {self.threshold_high}] in "
                f"{self.threshold_steps} steps, scored on OUT-OF-FOLD train "
                f"predictions"
                + ("; train-side decisions use fold-nested thresholds"
                   if self.nested_oof_threshold else "")
            ),
            "shared_implementation": "shared.stage_runner (both arms)",
            "family_specific": [
                "estimator",
                "hyperparameter search space",
            ],
        }
