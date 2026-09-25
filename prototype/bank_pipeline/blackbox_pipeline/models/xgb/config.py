"""
blackbox_pipeline.models.xgb.config
====================================
Stage 3 configuration, as data.

Every value here that has a GLASS Stage 3 counterpart carries the EBM's value.
The defaults ARE the mirroring contract, so a diff of this file against
``glass_pipeline.ebm.tuning`` / ``calibration`` / ``evaluation`` is the whole
argument that the two arms were run under matched conditions.

Where a value could not be mirrored literally, the docstring says so and names
the experimental purpose that was preserved instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

__all__ = ["XGBSearchSpace", "XGBStage3Config", "DEFAULT_XGB_PARAMS"]


# A plausible mid-space point, not a tuned result. Nothing in the package reads
# it; pass it as ``train_xgb_stage(..., params=DEFAULT_XGB_PARAMS)`` to skip
# Optuna (e.g. smoke tests). The notebook's "explicit" path passes its own dict.
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.02,
    "max_depth": 4,
    "min_child_weight": 5.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
}


@dataclass(frozen=True)
class XGBSearchSpace:
    """
    Bounded Optuna search space, recorded verbatim in the Stage 3 artifact.

    Mirroring notes
    ---------------
    ``learning_rate``
        Identical bounds and log scale to the EBM's ``learning_rate``.
    ``n_estimators``
        The EBM tunes ``max_rounds`` over [500, 5000] step 100. Boosted-tree
        rounds are far more expensive per round than EBM rounds at this data
        size, so the ceiling is lower. Purpose preserved: "how long may it
        boost", searched on the same step granularity.
    ``max_depth``
        THE experimental knob. The EBM is additive plus three declared pairwise
        interactions; depth > 1 lets XGBoost form interactions of any order
        without declaring them. The floor is 2 rather than 1 because the EBM
        does model pairwise terms — depth 1 would handicap XGBoost below the
        EBM's expressiveness rather than above it. Set ``max_depth_low=1`` to
        let the search rediscover a purely additive solution, which is a
        legitimate and interesting outcome.
    ``min_child_weight`` / ``reg_lambda`` / ``subsample`` / ``colsample_bytree``
        No EBM counterpart — the EBM's capacity control is bin count plus
        round count. These are the equivalent capacity controls for a tree
        ensemble, and leaving them fixed would mean comparing a tuned EBM
        against an untuned XGBoost.

    The space is 7-dimensional against the EBM's 4 at the same 150-trial
    budget, so XGBoost searches a larger space less densely. That asymmetry
    disadvantages XGBoost, which is the conservative direction for a
    "does removing the additivity constraint help?" question.
    """

    n_estimators_low: int = 200
    n_estimators_high: int = 2000
    n_estimators_step: int = 100

    learning_rate_low: float = 0.005      # identical to EBM
    learning_rate_high: float = 0.05      # identical to EBM
    learning_rate_log: bool = True        # identical to EBM

    max_depth_low: int = 2
    max_depth_high: int = 8

    min_child_weight_low: float = 1.0
    min_child_weight_high: float = 20.0
    min_child_weight_log: bool = True

    subsample_low: float = 0.6
    subsample_high: float = 1.0

    colsample_bytree_low: float = 0.6
    colsample_bytree_high: float = 1.0

    reg_lambda_low: float = 1e-3
    reg_lambda_high: float = 10.0
    reg_lambda_log: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> dict[str, str]:
        """Human-readable space, for the tuning printout and the artifact."""
        return {
            "n_estimators": f"int[{self.n_estimators_low}, {self.n_estimators_high}] "
                            f"step {self.n_estimators_step}",
            "learning_rate": f"float[{self.learning_rate_low}, {self.learning_rate_high}] "
                             f"{'log' if self.learning_rate_log else 'linear'}",
            "max_depth": f"int[{self.max_depth_low}, {self.max_depth_high}]",
            "min_child_weight": f"float[{self.min_child_weight_low}, "
                                f"{self.min_child_weight_high}] "
                                f"{'log' if self.min_child_weight_log else 'linear'}",
            "subsample": f"float[{self.subsample_low}, {self.subsample_high}]",
            "colsample_bytree": f"float[{self.colsample_bytree_low}, "
                                f"{self.colsample_bytree_high}]",
            "reg_lambda": f"float[{self.reg_lambda_low}, {self.reg_lambda_high}] "
                          f"{'log' if self.reg_lambda_log else 'linear'}",
        }


@dataclass
class XGBStage3Config:
    """
    Stage 3 XGBoost configuration.

    Attributes whose values are dictated by the EBM are marked MIRROR; changing
    one breaks the controlled comparison.
    """

    # ---- experiment identity ---------------------------------------------
    stage: str = "stage3"
    model_family: str = "xgboost"
    counterpart: str = "glass_pipeline.ebm"

    # ---- reproducibility --------------------------------------------------
    random_state: int = 42                 # MIRROR — EBM seeds everything with 42
    n_jobs: int = 1                        # MIRROR — EBM uses n_jobs=1 throughout

    # ---- cross-validation -------------------------------------------------
    # MIRROR — the EBM runs 5-fold stratified inside the Optuna objective and a
    # separate 10-fold stratified block for final reporting. The OOF folds used
    # for Stage 4, calibration and threshold selection are the SAME partition as
    # the tuning folds, so one canonical fold_id array covers all of them.
    n_tune_folds: int = 5
    n_eval_folds: int = 10
    stratify: bool = True

    # ---- tuning -----------------------------------------------------------
    n_trials: int = 150                    # MIRROR — EBM budget
    pruner_startup_trials: int = 20        # MIRROR
    pruner_warmup_steps: int = 2           # MIRROR
    study_name: str = "xgb_stage3_recall_biased"
    search_space: XGBSearchSpace = field(default_factory=XGBSearchSpace)

    # ---- objective --------------------------------------------------------
    # MIRROR — the EBM's Optuna objective is fbeta(beta=2) on model.predict(),
    # i.e. F2 at the default 0.5 cut, NOT F2 at a tuned threshold.
    beta: float = 2.0
    tuning_decision_threshold: float = 0.5

    # ---- threshold selection ---------------------------------------------
    # MIRROR — same grid as find_optimal_threshold(). The one deliberate
    # departure is the DATA the grid is scored on: out-of-fold training
    # predictions, never the test split (PR 3 requirement).
    threshold_low: float = 0.10
    threshold_high: float = 0.90
    threshold_steps: int = 81
    reference_threshold: float = 0.5       # fixed cut for the "test_at_half" block
    # Train-side (Stage 4) decision columns use per-fold thresholds chosen
    # without that fold's labels. False = one global OOF threshold for all
    # rows (pre-nesting behaviour). Test side always uses the global one.
    nested_oof_threshold: bool = True

    # ---- calibration ------------------------------------------------------
    # MIRROR — isotonic, gated at ECE > 0.05. The departure is that the gating
    # ECE is measured on out-of-fold probabilities rather than in-sample ones,
    # because an in-sample ECE is optimistic by construction and would skip
    # calibration that held-out data says is needed.
    calibration_method: str = "isotonic"
    ece_threshold: float = 0.05
    ece_bins: int = 10
    force_calibration: bool = False

    # ---- class imbalance --------------------------------------------------
    # MIRROR — sklearn compute_sample_weight("balanced"), computed once over the
    # full training split and sliced per fold. Passed as sample_weight, NOT as
    # scale_pos_weight: the weight vector is the same object the EBM receives,
    # so no translation step can drift.
    class_weight: str = "balanced"

    # ---- data hygiene -----------------------------------------------------
    # MIRROR — the EBM replaces inf with NaN then median-imputes inside
    # tune_ebm. XGBoost handles NaN natively, but imputing keeps the two arms'
    # training inputs identical. Medians are learned on train and reused on
    # test, which the EBM does not currently do. clean_inf=False with inf
    # present raises.
    clean_inf: bool = True
    impute_missing: bool = True

    # ---- artifacts --------------------------------------------------------
    artifact_dir: str = "models/xgb"
    artifact_stem: str = "xgb_stage3"
    timestamped: bool = True
    write_latest_pointer: bool = True

    # ---- reporting --------------------------------------------------------
    verbose: bool = True
    show_progress_bar: bool = True

    # ---- expected Stage 3 feature contract --------------------------------
    # When set, the stage asserts its input columns match exactly. The notebook
    # passes glass_pipeline.ebm.feature_engineering.EBM_FEATURES here, so a
    # drift in the GLASS feature DAG fails loudly instead of silently changing
    # what the comparison means.
    expected_features: Optional[list[str]] = None

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
        if errors:
            raise ValueError(
                "XGBStage3Config validation errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["search_space"] = self.search_space.to_dict()
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "XGBStage3Config":
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in d.items() if k in known}
        space = payload.get("search_space")
        if isinstance(space, dict):
            payload["search_space"] = XGBSearchSpace(**space)
        return cls(**payload)

    def mirror_report(self) -> dict[str, Any]:
        """The matched-conditions summary, embedded in every artifact."""
        return {
            "counterpart": self.counterpart,
            "random_state": self.random_state,
            "n_tune_folds": self.n_tune_folds,
            "n_eval_folds": self.n_eval_folds,
            "n_trials": self.n_trials,
            "objective": f"F{self.beta:g} at decision threshold "
                         f"{self.tuning_decision_threshold}",
            "class_weight": self.class_weight,
            "calibration": f"{self.calibration_method}, gated at OOF ECE > "
                           f"{self.ece_threshold}",
            "threshold_rule": (
                f"F{self.beta:g}-maximising over "
                f"[{self.threshold_low}, {self.threshold_high}] in "
                f"{self.threshold_steps} steps, scored on OUT-OF-FOLD train "
                f"predictions"
                + ("; train-side decisions use fold-nested thresholds"
                   if self.nested_oof_threshold else "")
            ),
            "documented_departures": [
                "threshold selected on OOF train predictions, not on the test "
                "split (PR 3 requirement; GLASS EBM currently fits its "
                "threshold on y_test)",
                "calibration gate measured on OOF probabilities rather than "
                "in-sample ones",
                "out-of-fold Stage 3 predictions produced for the training "
                "split (Stage 4 input); in-sample refit column kept for parity",
                "median imputation learned on train and reused on test",
                "search space is XGBoost-native; only learning_rate could be "
                "mirrored bound-for-bound",
            ],
        }