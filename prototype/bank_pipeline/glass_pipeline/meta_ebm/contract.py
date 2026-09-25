"""
glass_pipeline.meta_ebm.contract
================================
Moved to ``shared.stage4.contract`` so the GLASS Meta-EBM and the black-box
Meta-XGB validate their inputs with one implementation. Re-exported here so
existing imports keep working.
"""

from shared.stage4.contract import (  # noqa: F401
    FINGERPRINT_SCHEMES,
    SplitReference,
    Stage4InputError,
    Stage4Inputs,
    StageStream,
    build_stage4_inputs,
    check_proba,
    fold_partition,
    index_alignment,
    not_identical,
    split_reference,
)
