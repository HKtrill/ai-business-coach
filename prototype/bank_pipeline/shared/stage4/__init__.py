"""
shared.stage4
=============
Stage 4 input contract shared by the GLASS Meta-EBM and the black-box
Meta-XGB: one set of artifact adapters and one set of split / index / fold /
OOF-honesty / router-semantics checks, so "inputs verified" means the same
thing in both arms.

    contract   StageStream, Stage4Inputs, build_stage4_inputs, split_reference
    adapters   stage_output_stream (Stage 1), router_stream (Stage 2),
               stage3_stream (Stage 3)
    protocol   constants both arms' abstention tuning mirror
    upstream   which arm filled an arbiter's Stage 1-3 slots: the declared
               2x2 combination, checked and recorded (PR 36 F29)
"""

from .contract import (
    FINGERPRINT_SCHEMES,
    Stage4InputError,
    Stage4Inputs,
    StageStream,
    SplitReference,
    build_stage4_inputs,
    check_proba,
    check_split_identity,
    fold_partition,
    index_alignment,
    not_identical,
    split_reference,
)
from .adapters import (
    ABSTAIN,
    PASS1,
    PASS2,
    normalize_router_decisions,
    router_stream,
    stage3_stream,
    stage_output_stream,
)
from .protocol import (
    ABSTENTION_GRID, ABSTENTION_MIN_COVERAGE, ABSTENTION_SEARCH, tune_min_confidence,
)
from .upstream import (
    UPSTREAM, UPSTREAM_ARMS, resolve_upstream, same_upstream, stream_fingerprint,
)

__all__ = [
    "FINGERPRINT_SCHEMES", "Stage4InputError", "Stage4Inputs", "StageStream",
    "SplitReference", "build_stage4_inputs", "check_proba", "check_split_identity", "fold_partition",
    "index_alignment", "not_identical", "split_reference",
    "ABSTAIN", "PASS1", "PASS2", "normalize_router_decisions",
    "router_stream", "stage3_stream", "stage_output_stream",
    "ABSTENTION_GRID", "ABSTENTION_MIN_COVERAGE", "ABSTENTION_SEARCH", "tune_min_confidence",
    "UPSTREAM", "UPSTREAM_ARMS", "resolve_upstream", "same_upstream", "stream_fingerprint",
]
