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
"""

from .contract import (
    FINGERPRINT_SCHEMES,
    Stage4InputError,
    Stage4Inputs,
    StageStream,
    SplitReference,
    build_stage4_inputs,
    check_proba,
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
from .protocol import ABSTENTION_GRID, ABSTENTION_MIN_COVERAGE, tune_min_confidence

__all__ = [
    "FINGERPRINT_SCHEMES", "Stage4InputError", "Stage4Inputs", "StageStream",
    "SplitReference", "build_stage4_inputs", "check_proba", "fold_partition",
    "index_alignment", "not_identical", "split_reference",
    "ABSTAIN", "PASS1", "PASS2", "normalize_router_decisions",
    "router_stream", "stage3_stream", "stage_output_stream",
    "ABSTENTION_GRID", "ABSTENTION_MIN_COVERAGE", "tune_min_confidence",
]
