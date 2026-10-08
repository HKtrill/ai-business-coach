"""
feature_research/model_training/glass_arbiter
=============================================
GLASS Arbiter on the research stages (formerly meta_ebm): weighted-confidence
vote over the LR, RF and EBM out-of-fold scores, with abstention.

Same rules as the cascade's Stage 4 arbiter (glass_pipeline.meta_ebm), kept as
a separate implementation so the research and cascade code stay independent:
recall-target thresholds, Brier + accuracy hybrid weights, F2-tuned
min_weighted_confidence (coverage ≥ 0.50, fallback 0.07), the same metrics
and disagreement report.

Research protocol
-----------------
Inputs are lr_oof / rf_oof / ebm_oof on the training rows; no models are
retrained here and there is no second split. With no test set, the
configuration is cross-fitted (fit on K−1 folds, frozen, applied to the
held-out fold), so every metric and trace is out-of-fold.

Modules
-------
thresholds  recall-target (cascade rule) / Youden / F2 threshold rules
arbiter     vote rule, calibration, weights, metrics, disagreement, abstention tuning, fit/apply
crossfit    cross_fit_arbiter, ArbiterResult, compare_variants
tracer      agreement trace: correctness patterns, score states, unique targets, Venn
analysis    missed positives: hard floor, statistical profile, near misses, Excel
"""

from feature_research.model_training.glass_arbiter.analysis import run_missed_analysis
from feature_research.model_training.glass_arbiter.arbiter import (
    ABSTENTION_GRID,
    ABSTENTION_MIN_COVERAGE,
    MIN_CONF_FALLBACK,
    MODELS,
    ArbiterConfig,
    analyze_disagreements,
    apply_arbiter,
    compute_calibration,
    compute_hybrid_weights,
    compute_metrics,
    evaluate_with_abstention,
    fit_arbiter,
    tune_min_confidence,
)
from feature_research.model_training.glass_arbiter.crossfit import (
    ArbiterResult,
    compare_variants,
    cross_fit_arbiter,
)
from feature_research.model_training.glass_arbiter.thresholds import (
    find_f2_threshold,
    find_recall_threshold,
    find_youden_threshold,
)
from feature_research.model_training.glass_arbiter.tracer import run_bitmask_trace

__all__ = [
    "MODELS", "ABSTENTION_GRID", "ABSTENTION_MIN_COVERAGE", "MIN_CONF_FALLBACK",
    "ArbiterConfig", "fit_arbiter", "apply_arbiter", "tune_min_confidence",
    "compute_calibration", "compute_hybrid_weights", "compute_metrics",
    "evaluate_with_abstention", "analyze_disagreements",
    "ArbiterResult", "cross_fit_arbiter", "compare_variants",
    "find_recall_threshold", "find_youden_threshold", "find_f2_threshold",
    "run_bitmask_trace", "run_missed_analysis",
]
