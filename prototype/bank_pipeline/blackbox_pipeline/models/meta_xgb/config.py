"""
blackbox_pipeline.models.meta_xgb.config
========================================
Meta-XGB protocol: the Stage 4 feature contract, the (small) search space and
``MetaXGBConfig`` — a ``Stage3Config`` subclass, so the meta-learner is tuned,
cross-fitted, calibrated and thresholded by the SAME shared runner as both
Stage 3 arms.

Feature contract (arm-neutral names; values from the black-box Stage 1–3)
------------------------------------------------------------------------
    stage1_proba     Stage 1 P(y=1)            MLP StageOutput (calibrated)
    stage2_proba     Stage 2 router P(y=1)     RF Router; MISSING (NaN) on rows
                                               the router abstains on
    stage2_pass1     1 if routed Pass 1 (NOT_SUBSCRIBE)
    stage2_pass2     1 if routed Pass 2 (SUBSCRIBE)
    stage2_abstain   1 if the router abstained ("uncertain")
    stage3_proba     Stage 3 P(y=1)            XGBoost Stage3Artifact, matched
                                               pair chosen by calibration["applied"]

These are exactly the channels the GLASS Meta-EBM receives (three
probabilities + the router's Pass 1 / Pass 2 masks). The router probability is
withheld where the router abstains because the Meta-EBM never reads it there;
the one-hot state lets the model tell an abstention from a real 0.5.
Nothing else enters: no raw or engineered features, no labels, no upstream
thresholds, margins or decisions, no router confidence, no test-derived value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Optional

try:                                    # current package layout
    from shared.stage3 import Stage3Config
except ImportError:                     # pragma: no cover - legacy monolith
    from shared.stage_runner import Stage3Config

STAGE4_FEATURES: tuple = (
    "stage1_proba",
    "stage2_proba",
    "stage2_pass1",
    "stage2_pass2",
    "stage2_abstain",
    "stage3_proba",
)

FEATURE_CONTRACT: dict = {
    "stage1_proba": "Stage 1 OOF P(y=1) (train) / full-train P(y=1) (test)",
    "stage2_proba": "Stage 2 router P(y=1); NaN where the router abstains",
    "stage2_pass1": "router route == Pass 1 (NOT_SUBSCRIBE)",
    "stage2_pass2": "router route == Pass 2 (SUBSCRIBE)",
    "stage2_abstain": "router abstained ('uncertain')",
    "stage3_proba": "Stage 3 OOF P(y=1) (train) / refit P(y=1) (test), matched space",
}

EXCLUDED_CHANNELS: tuple = (
    "raw Bank Marketing features", "Stage 1–3 engineered features", "labels / target statistics",
    "upstream thresholds, margins or decisions", "router confidence",
    "test-derived values", "oracle thresholds", "additional models",
)


@dataclass
class MetaXGBSearchSpace:
    """Deliberately small: 6 inputs need shallow, regularised trees."""

    n_estimators: tuple = (50, 600, 50)          # low, high, step
    learning_rate: tuple = (0.01, 0.20)          # log
    max_depth: tuple = (1, 4)
    min_child_weight: tuple = (1.0, 50.0)        # log
    subsample: tuple = (0.6, 1.0)
    reg_lambda: tuple = (0.1, 20.0)              # log
    reg_alpha: tuple = (1e-3, 5.0)               # log
    colsample_bytree: float = 1.0                # fixed — only 6 columns

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MetaXGBConfig(Stage3Config):
    """
    Shared Stage 3 protocol, re-pointed at Stage 4.

    Inherited unchanged from the shared protocol: seed 42, 5-fold stratified
    ``fold_assignment`` partition (tuning + meta OOF + calibration + threshold),
    10-fold reporting CV, TPE + MedianPruner, F2 objective at the 0.5 cut,
    balanced class weights, F2 threshold grid 0.10–0.90 × 81 on OOF, fold-nested
    train-side thresholds, isotonic calibration gated at OOF ECE > 0.05.

    Stage-4-specific: identity fields, the feature contract, a 50-trial budget
    (Meta-EBM has no search; see the report for why 50) and
    ``impute_missing=False`` so the router's abstain-row NaN stays missing.
    """

    stage: str = "stage4"
    model_family: str = "meta_xgb"
    arm: str = "blackbox"
    counterpart: str = "glass_pipeline.meta_ebm (Meta-EBM weighted-confidence arbiter)"
    column_prefix: str = "meta_xgb_"

    n_trials: int = 50
    pruner_startup_trials: int = 10
    study_name: str = "stage4_meta_xgb_f2"
    search_space: Any = field(default_factory=MetaXGBSearchSpace)

    impute_missing: bool = False
    expected_features: Optional[list] = field(default_factory=lambda: list(STAGE4_FEATURES))

    artifact_dir: str = "models/meta_xgb"
    artifact_stem: str = "meta_xgb_stage4"

    # Stage 4 operating choices
    operating_space: str = "raw"                 # decisions / abstention on raw meta p
    router_proba_on_abstain: str = "missing"     # "missing" (NaN) — see module docs

    _space_cls: ClassVar[Optional[type]] = MetaXGBSearchSpace

    def __post_init__(self) -> None:
        super().__post_init__()
        errors = []
        if list(self.expected_features or []) != list(STAGE4_FEATURES):
            errors.append("expected_features must equal STAGE4_FEATURES (fairness contract)")
        if self.operating_space not in ("raw", "calibrated"):
            errors.append("operating_space must be 'raw' or 'calibrated'")
        if self.router_proba_on_abstain != "missing":
            errors.append("router_proba_on_abstain: only 'missing' is implemented")
        if errors:
            raise ValueError("MetaXGBConfig: " + "; ".join(errors))
