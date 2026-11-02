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

Additive and diagnostic only: nothing here fits, tunes, refits, writes files or
changes any router, threshold or Stage 4 behaviour. Metric definitions live in
``metrics.METRIC_DEFINITIONS``.
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
]
