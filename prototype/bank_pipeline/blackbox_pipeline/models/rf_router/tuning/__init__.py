"""
blackbox_pipeline.models.rf_router.tuning
==========================================
RF hyperparameter resolution and tuning for Stage 2.

Baseline path (former notebook Cells 20-23, behaviour unchanged)
----------------------------------------------------------------
``space``        the search space and the parameter normaliser — one source of truth
``legacy``       artifact loading, the ROC-AUC tuner, ``resolve_rf_params``

Objective-specialization harness (PR 35)
----------------------------------------
``budget``       ``TuningBudget`` shared by every arm; ``RouterFolds``
``stage``        Pass 1 stage reproduced outside the router -> the frozen remainder
``diagnostics``  Gate 0: exact Pass 2 degeneracy test; router read-out
``objectives``   what a trial measures; ``p1_coverage``, ``p2_recall``, ``auc``
``provenance``   contract version, data fingerprint, validated disk cache
``search``       ``tune``, headroom screen, noise floor, fresh-fold re-evaluation,
                 matched-control and cached-result checks
``ablation``     arms, coupling checks, decomposition, frozen decision rule

Everything in this package reads training data only.
"""

from __future__ import annotations

from .ablation import (
    ARM_A0,
    ARM_B,
    ARM_C,
    ARM_D,
    ARM_E,
    ArmSpec,
    DecisionRule,
    SpecializationPlan,
    assert_arm_coupling,
    decomposition_pairs,
    fit_arm,
    register_test_evaluation,
    run_specialization_tuning,
    specialization_verdict,
)
from .budget import RouterFolds, TuningBudget
from .diagnostics import (
    DegenerateObjectiveError,
    Pass2Degeneracy,
    diagnose_pass2_degeneracy,
    diagnose_router,
)
from .legacy import (
    cv_auc_legacy,
    load_rf_params_from_artifact,
    resolve_rf_params,
    tune_rf_train_only,
)
from .objectives import OBJECTIVES, evaluate_params
from .provenance import (
    TUNING_CONTRACT_VERSION,
    IncompatibleCacheError,
    cache_key,
    cached,
    data_fingerprint,
    frame_validator,
)
from .search import (
    HEADROOM_INSUFFICIENT,
    HEADROOM_LIMITED,
    HEADROOM_PROMISING,
    TuningResult,
    assert_matched_control,
    assert_same_budget,
    classify_headroom,
    fresh_fold_evaluation,
    headroom_probe,
    noise_floor,
    summarize_headroom,
    tune,
    tuning_result_validator,
)
from .space import REQUIRED_PARAM_KEYS, normalise_rf_params, suggest_rf_params
from .stage import Pass1Stage, mask_fingerprint, pass1_stage

__all__ = [
    # baseline path
    "REQUIRED_PARAM_KEYS",
    "normalise_rf_params",
    "suggest_rf_params",
    "cv_auc_legacy",
    "tune_rf_train_only",
    "load_rf_params_from_artifact",
    "resolve_rf_params",
    # specialization harness
    "TuningBudget",
    "RouterFolds",
    "Pass1Stage",
    "pass1_stage",
    "mask_fingerprint",
    "DegenerateObjectiveError",
    "Pass2Degeneracy",
    "diagnose_pass2_degeneracy",
    "diagnose_router",
    "OBJECTIVES",
    "evaluate_params",
    "TUNING_CONTRACT_VERSION",
    "IncompatibleCacheError",
    "data_fingerprint",
    "cache_key",
    "cached",
    "frame_validator",
    "TuningResult",
    "tune",
    "tuning_result_validator",
    "headroom_probe",
    "summarize_headroom",
    "classify_headroom",
    "HEADROOM_INSUFFICIENT",
    "HEADROOM_LIMITED",
    "HEADROOM_PROMISING",
    "noise_floor",
    "fresh_fold_evaluation",
    "assert_same_budget",
    "assert_matched_control",
    "ARM_A0",
    "ARM_B",
    "ARM_C",
    "ARM_D",
    "ARM_E",
    "ArmSpec",
    "SpecializationPlan",
    "DecisionRule",
    "run_specialization_tuning",
    "fit_arm",
    "assert_arm_coupling",
    "decomposition_pairs",
    "specialization_verdict",
    "register_test_evaluation",
]
