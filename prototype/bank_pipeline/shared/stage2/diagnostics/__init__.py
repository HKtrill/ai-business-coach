"""
shared.stage2.diagnostics
=========================
Train-side Stage 2 router diagnostics (PR 37, Commit 1, Step 1C).

    schema            states, score spaces, provenance statuses, read-only views
    boundary          the ONLY artifact reader: allowlisted training-side fields,
                      mandatory checks, train reference (no test labels)
    operating_points  named upstream operating points -> flags; label-free
                      rate-matched cuts (strict and inclusive)
    metrics           decision-level router, router x upstream, router x router
                      and matched-rate diagnostics (Ratio with explicit
                      denominators); score-stream support gate
    equivalence       is the GLASS-arm RF oof_proba the RF Router's outer Pass 1
                      score? zero disagreements required
    trace             (Commit 2, Step 2A) both routers' three-state outer OOF
                      trace: checks T1-T7, routing-policy audit, state
                      distributions, routing rates, GLASS vs RF comparison
    suppression       (Commit 2, Step 2B) Pass 1 against the primary LR / EBM
                      flags and their union: intercepted false positives,
                      suppressed true positives, NPV and leakage
    contribution      (Commit 2, Step 2C) Pass 2 unique positives against the
                      frozen LR / EBM flags and against label-free rate-matched
                      LR / EBM flags on the common eligible population
    overlap           (Commit 2, Step 2D) router x upstream and GLASS x RF overlap,
                      signal availability, label-free score strata, score-
                      conditional states, observed vs expected positives,
                      threshold positions and arithmetic baselines
    baseline          (Commit 2, Step 2E) the frozen Pre-Correction Stage 2
                      Baseline package: export, verification and comparison of
                      the Steps 2A-2D results (imported explicitly, not
                      re-exported here)

Additive and diagnostic only: nothing here fits, tunes, refits or changes any
router, threshold or Stage 4 behaviour, and only ``baseline.write_baseline``
writes files (called by research_logs/baselines/stage2_baseline.py --generate).
Metric definitions live in ``metrics.METRIC_DEFINITIONS``.
"""

from .boundary import (
    FORBIDDEN_FIELD_TOKENS,
    TRANSFORM_ATOL,
    BoundaryCheckError,
    build_train_reference,
    extract_ebm_view,
    extract_lr_view,
    extract_rf_pass1_candidate,
    extract_router_view,
    extract_stage4_recall_thresholds,
    file_sha256,
    router_split_fingerprint,
    train_reference_from_global_split,
)
from .equivalence import (
    BOUNDARY_ATOL,
    EquivalenceReport,
    check_rf_pass1_score_equivalence,
    verified_rf_pass1_score,
)
from .metrics import (
    METRIC_DEFINITIONS,
    DiagnosticResult,
    Ratio,
    matched_rate_comparison,
    pass1_metrics,
    pass2_metrics,
    require_score_stream,
    router_metrics,
    router_router_table,
    router_upstream_metrics,
    state_counts,
)
from .operating_points import (
    OPERATING_POINTS,
    PRIMARY_OPERATING_POINTS,
    SECONDARY_OPERATING_POINTS,
    MatchedCut,
    OperatingPointSpec,
    UpstreamFlags,
    combine_flags,
    matched_rate_cut,
    resolve_operating_point,
)
from .schema import (
    ABSTAIN,
    ATTESTED,
    HISTORICAL_TEST_EXPOSURE,
    NOT_RECORDED,
    PASS_1_NOT_SUBSCRIBE,
    PASS_2_SUBSCRIBE,
    RECORDED_ONLY,
    ROUTER_STATES,
    SCORE_SPACES,
    SOURCE_DECISIONS,
    VALIDATED,
    ProvenanceItem,
    RFPass1ScoreCandidate,
    RouterDiagnosticError,
    RouterOOFView,
    ScoreStream,
    Stage4Thresholds,
    StoredDecision,
    ThresholdRecord,
    TrainReference,
    UnsupportedDiagnostic,
    UpstreamOOFView,
)
from .trace import (
    AUDIT_STATUSES,
    RouterTrace,
    TraceCheckError,
    build_router_trace,
    compare_state_distributions,
    routing_policy_audit,
    routing_rates,
    state_distribution,
)
from .suppression import (
    pass1_population,
    pass1_suppression,
    pass1_suppression_report,
    primary_reference_flags,
    reference_flag_summary,
)
from .contribution import (
    MatchedSelection,
    common_eligible_mask,
    common_eligible_population,
    matched_reference_selection,
    pass2_contribution,
    pass2_frozen_contribution,
    pass2_matched_contribution,
    pass2_population,
    pass2_positive_breakdown,
)
from .overlap import (
    ScoreStrata,
    boundary_diagnostics,
    crosstab_summary,
    group_observed_expected,
    key_groups,
    model_disagreement_baseline,
    overlap_notes,
    pass2_cross_router_overlap,
    router_crosstab,
    router_reference_agreement,
    router_reference_contingency,
    score_conditional_states,
    score_strata,
    signal_availability,
    state_standardization,
    unique_positive_chance_baselines,
    within_stratum_contrasts,
)

__all__ = [
    # schema
    "PASS_1_NOT_SUBSCRIBE", "PASS_2_SUBSCRIBE", "ABSTAIN", "ROUTER_STATES", "SOURCE_DECISIONS",
    "SCORE_SPACES", "VALIDATED", "ATTESTED", "RECORDED_ONLY", "NOT_RECORDED",
    "HISTORICAL_TEST_EXPOSURE", "RouterDiagnosticError", "UnsupportedDiagnostic",
    "ProvenanceItem", "TrainReference", "ScoreStream", "ThresholdRecord", "StoredDecision",
    "RouterOOFView", "UpstreamOOFView", "Stage4Thresholds", "RFPass1ScoreCandidate",
    # boundary
    "FORBIDDEN_FIELD_TOKENS", "TRANSFORM_ATOL", "BoundaryCheckError", "router_split_fingerprint",
    "build_train_reference", "train_reference_from_global_split", "extract_router_view",
    "extract_lr_view", "extract_ebm_view", "extract_stage4_recall_thresholds",
    "extract_rf_pass1_candidate", "file_sha256",
    # operating points
    "OperatingPointSpec", "OPERATING_POINTS", "PRIMARY_OPERATING_POINTS",
    "SECONDARY_OPERATING_POINTS", "UpstreamFlags", "resolve_operating_point", "combine_flags",
    "MatchedCut", "matched_rate_cut",
    # metrics
    "Ratio", "DiagnosticResult", "METRIC_DEFINITIONS", "state_counts", "pass1_metrics",
    "pass2_metrics", "router_metrics", "router_upstream_metrics", "router_router_table",
    "matched_rate_comparison", "require_score_stream",
    # equivalence
    "BOUNDARY_ATOL", "EquivalenceReport", "check_rf_pass1_score_equivalence",
    "verified_rf_pass1_score",
    # trace (Commit 2, Step 2A)
    "AUDIT_STATUSES", "RouterTrace", "TraceCheckError", "build_router_trace",
    "routing_policy_audit", "state_distribution", "routing_rates", "compare_state_distributions",
    # suppression (Commit 2, Step 2B)
    "pass1_population", "pass1_suppression", "primary_reference_flags", "reference_flag_summary",
    "pass1_suppression_report",
    # contribution (Commit 2, Step 2C)
    "MatchedSelection", "pass2_population", "common_eligible_mask", "common_eligible_population",
    "pass2_positive_breakdown", "pass2_contribution", "pass2_frozen_contribution",
    "matched_reference_selection", "pass2_matched_contribution",
    # overlap (Commit 2, Step 2D)
    "ScoreStrata", "router_reference_contingency", "router_reference_agreement",
    "router_crosstab", "crosstab_summary", "pass2_cross_router_overlap", "signal_availability",
    "score_strata", "score_conditional_states", "within_stratum_contrasts",
    "state_standardization", "key_groups", "group_observed_expected", "boundary_diagnostics",
    "unique_positive_chance_baselines", "model_disagreement_baseline", "overlap_notes",
]
